use crate::compat::Mutex;
use crate::error::BackendError;
#[cfg(not(feature = "std"))]
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::bad_block::BadBlockTable;
use super::hardware::{FlashGeometry, FlashHardware};
use super::journal::FtlJournal;
use super::wear_leveling::EraseCountTable;

/// Sentinel value indicating an unmapped logical block.
const UNMAPPED: u32 = 0xFFFF_FFFF;

/// Trigger static wear leveling check every 256 erase operations.
///
/// Balances wear-leveling responsiveness against the overhead of scanning
/// the full erase-count table. 256 is a common industry default for
/// small-to-medium flash geometries (≤4 GiB).
const STATIC_WL_INTERVAL: u32 = 256;

/// Swap hot/cold blocks when the max-min erase count delta exceeds 100 cycles.
///
/// Prevents erase concentration on heavily-used blocks while avoiding
/// unnecessary block moves for small variations. Typical NAND endurance is
/// 3 000–100 000 P/E cycles; a 100-cycle threshold triggers early enough to
/// spread wear without excessive churn.
const STATIC_WL_THRESHOLD: u32 = 100;

/// Logical-to-physical block mapping table.
struct BlockMap {
    /// `forward[logical] = physical`. `UNMAPPED` if not allocated.
    forward: Vec<u32>,
    /// `reverse[physical] = logical`. `UNMAPPED` if free.
    reverse: Vec<u32>,
    /// Physical blocks not assigned to any logical block.
    free_list: Vec<u32>,
}

impl BlockMap {
    fn new(logical_blocks: u32, physical_blocks: u32) -> Self {
        Self {
            forward: vec![UNMAPPED; logical_blocks as usize],
            reverse: vec![UNMAPPED; physical_blocks as usize],
            free_list: Vec::new(),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_bytes(data: &[u8], physical_blocks: u32) -> Self {
        let logical_count = data.len() / 4;
        let mut forward = Vec::with_capacity(logical_count);
        for i in 0..logical_count {
            let off = i * 4;
            if off + 4 <= data.len() {
                forward.push(u32::from_le_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                ]));
            }
        }

        // Rebuild reverse map and free list
        let mut reverse = vec![UNMAPPED; physical_blocks as usize];
        for (logical, &physical) in forward.iter().enumerate() {
            if physical != UNMAPPED && (physical as usize) < reverse.len() {
                reverse[physical as usize] = logical as u32;
            }
        }

        let mut free_list = Vec::new();
        for phys in 0..physical_blocks {
            if reverse[phys as usize] == UNMAPPED {
                free_list.push(phys);
            }
        }

        Self {
            forward,
            reverse,
            free_list,
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.forward.len() * 4);
        for &p in &self.forward {
            out.extend_from_slice(&p.to_le_bytes());
        }
        out
    }

    #[inline]
    fn get(&self, logical_block: u32) -> Option<u32> {
        self.forward
            .get(logical_block as usize)
            .and_then(|&p| if p == UNMAPPED { None } else { Some(p) })
    }

    fn assign(&mut self, logical_block: u32, physical_block: u32) {
        if let Some(old_phys) = self.forward.get(logical_block as usize).copied()
            && old_phys != UNMAPPED
            && let Some(r) = self.reverse.get_mut(old_phys as usize)
        {
            *r = UNMAPPED;
        }

        if let Some(f) = self.forward.get_mut(logical_block as usize) {
            *f = physical_block;
        }
        if let Some(r) = self.reverse.get_mut(physical_block as usize) {
            *r = logical_block;
        }

        // Remove from free list via swap_remove for O(1) instead of O(n) retain
        if let Some(pos) = self.free_list.iter().position(|&b| b == physical_block) {
            self.free_list.swap_remove(pos);
        }
    }

    fn release(&mut self, logical_block: u32) {
        if let Some(&physical) = self.forward.get(logical_block as usize)
            && physical != UNMAPPED
        {
            if let Some(r) = self.reverse.get_mut(physical as usize) {
                *r = UNMAPPED;
            }
            self.free_list.push(physical);
        }
        if let Some(f) = self.forward.get_mut(logical_block as usize) {
            *f = UNMAPPED;
        }
    }

    fn free_blocks(&self) -> &[u32] {
        &self.free_list
    }
}

/// Internal mutable state of the FTL, protected by a Mutex.
struct FtlState<H: FlashHardware> {
    hw: H,
    geometry: FlashGeometry,
    block_map: BlockMap,
    erase_counts: EraseCountTable,
    bad_blocks: BadBlockTable,
    journal: FtlJournal,
    /// Current logical storage length in bytes.
    logical_len: u64,
    /// First physical block index used for data (after reserved region).
    data_region_start: u32,
    /// Erase operations since last static wear-level check.
    ops_since_static_wl: u32,
}

/// Flash Translation Layer providing logical-to-physical block mapping, wear
/// leveling, bad block management, and power-loss-safe metadata journaling.
pub(super) struct FlashTranslationLayer<H: FlashHardware> {
    state: Mutex<FtlState<H>>,
}

#[allow(clippy::cast_possible_truncation)]
impl<H: FlashHardware> FlashTranslationLayer<H> {
    /// Validate that flash geometry has no zero-valued fields that would cause
    /// division-by-zero or infinite loops.
    fn validate_geometry(geo: &FlashGeometry) -> core::result::Result<(), BackendError> {
        if geo.write_page_size == 0 || geo.erase_block_size == 0 || geo.total_blocks == 0 {
            #[cfg(feature = "std")]
            return Err(BackendError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid flash geometry: zero-valued field",
            )));
            #[cfg(not(feature = "std"))]
            return Err(BackendError::Message(String::from(
                "invalid flash geometry: zero-valued field",
            )));
        }
        Ok(())
    }

    /// Mount an existing FTL from flash, recovering state from the journal.
    ///
    /// If no valid journal is found, formats the device fresh.
    pub fn mount(hw: H) -> core::result::Result<Self, BackendError> {
        let geo = hw.geometry();
        Self::validate_geometry(&geo)?;
        let reserved = geo.reserved_blocks();
        let journal_blocks_per_slot = (reserved.saturating_sub(2)) / 2;

        if journal_blocks_per_slot == 0 || geo.total_blocks <= reserved {
            return Self::format(hw);
        }

        let slot_a_start = 0u32;

        let (journal, payload) = FtlJournal::mount(&hw, slot_a_start, journal_blocks_per_slot)?;

        let data_region_start = reserved;
        let data_physical_blocks = geo.total_blocks - data_region_start;

        match payload {
            Some(data) => {
                let (block_map, erase_counts, bad_blocks, logical_len) =
                    Self::deserialize_metadata(&data, data_physical_blocks, geo.total_blocks)?;

                Ok(Self {
                    state: Mutex::new(FtlState {
                        hw,
                        geometry: geo,
                        block_map,
                        erase_counts,
                        bad_blocks,
                        journal,
                        logical_len,
                        data_region_start,
                        ops_since_static_wl: 0,
                    }),
                })
            }
            None => Self::format_with_journal(hw, geo, journal, reserved),
        }
    }

    /// Format the flash device, erasing FTL metadata and creating a fresh mapping.
    pub fn format(hw: H) -> core::result::Result<Self, BackendError> {
        let geo = hw.geometry();
        Self::validate_geometry(&geo)?;
        let reserved = geo.reserved_blocks();
        let journal_blocks_per_slot = (reserved.saturating_sub(2)) / 2;

        if journal_blocks_per_slot == 0 {
            #[cfg(feature = "std")]
            return Err(BackendError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "flash device too small for FTL",
            )));
            #[cfg(not(feature = "std"))]
            return Err(BackendError::Message(String::from(
                "flash device too small for FTL",
            )));
        }

        // Erase journal slots
        for i in 0..(journal_blocks_per_slot * 2) {
            hw.erase_block(i)?;
        }

        let slot_a_start = 0u32;
        let (journal, _) = FtlJournal::mount(&hw, slot_a_start, journal_blocks_per_slot)?;

        Self::format_with_journal(hw, geo, journal, reserved)
    }

    fn format_with_journal(
        hw: H,
        geo: FlashGeometry,
        journal: FtlJournal,
        reserved: u32,
    ) -> core::result::Result<Self, BackendError> {
        let data_region_start = reserved;
        let data_physical_blocks = geo.total_blocks - data_region_start;

        let bad_blocks = BadBlockTable::scan(&hw)?;

        // Build free list from data region, excluding bad blocks
        let logical_block_count = geo.logical_block_count();
        let mut block_map = BlockMap::new(logical_block_count, data_physical_blocks);
        for phys in 0..data_physical_blocks {
            let global_phys = data_region_start + phys;
            if !bad_blocks.is_bad(global_phys) {
                block_map.free_list.push(phys);
            }
        }

        let erase_counts = EraseCountTable::new(geo.total_blocks);

        let mut state = FtlState {
            hw,
            geometry: geo,
            block_map,
            erase_counts,
            bad_blocks,
            journal,
            logical_len: 0,
            data_region_start,
            ops_since_static_wl: 0,
        };

        // Persist initial metadata
        let metadata = Self::serialize_metadata_inner(&state);
        let FtlState {
            ref hw,
            ref mut journal,
            ..
        } = state;
        journal.commit(hw, &metadata)?;

        Ok(Self {
            state: Mutex::new(state),
        })
    }

    /// Read `buf.len()` bytes starting at logical offset.
    pub fn read(&self, offset: u64, buf: &mut [u8]) -> core::result::Result<(), BackendError> {
        let state = self.state.lock();
        let ebs = u64::from(state.geometry.erase_block_size);
        debug_assert!(ebs > 0, "erase_block_size validated at construction");

        let mut remaining = buf.len();
        let mut buf_offset = 0usize;
        let mut current_offset = offset;

        while remaining > 0 {
            let logical_block = u32::try_from(current_offset / ebs).unwrap_or(u32::MAX);
            let offset_in_block = (current_offset % ebs) as usize;
            let chunk_len = remaining.min(ebs as usize - offset_in_block);

            match state.block_map.get(logical_block) {
                Some(phys_block) => {
                    let phys_offset = (u64::from(state.data_region_start) + u64::from(phys_block))
                        * ebs
                        + offset_in_block as u64;
                    state
                        .hw
                        .read(phys_offset, &mut buf[buf_offset..buf_offset + chunk_len])?;
                }
                None => {
                    // Unmapped logical block reads as zeros
                    buf[buf_offset..buf_offset + chunk_len].fill(0);
                }
            }

            buf_offset += chunk_len;
            current_offset += chunk_len as u64;
            remaining -= chunk_len;
        }

        Ok(())
    }

    /// Write `data` starting at logical offset.
    ///
    /// Uses copy-on-write: for each affected logical block, reads old data into
    /// a buffer, overlays the new data, allocates a fresh physical block, writes
    /// the combined data, and releases the old block.
    pub fn write(&self, offset: u64, data: &[u8]) -> core::result::Result<(), BackendError> {
        let mut state = self.state.lock();
        let ebs = state.geometry.erase_block_size;
        let ebs_u64 = u64::from(ebs);
        let wps = state.geometry.write_page_size as usize;
        debug_assert!(ebs > 0 && wps > 0, "geometry validated at construction");

        let mut remaining = data.len();
        let mut data_offset = 0usize;
        let mut current_offset = offset;

        while remaining > 0 {
            let logical_block = u32::try_from(current_offset / ebs_u64).unwrap_or(u32::MAX);
            let offset_in_block = (current_offset % ebs_u64) as usize;
            let chunk_len = remaining.min(ebs as usize - offset_in_block);

            // Prepare the full block buffer
            let mut block_buf = vec![0u8; ebs as usize];

            let old_phys = state.block_map.get(logical_block);

            // Read existing data if block is mapped
            if let Some(phys_block) = old_phys {
                let phys_offset =
                    (u64::from(state.data_region_start) + u64::from(phys_block)) * ebs_u64;
                state.hw.read(phys_offset, &mut block_buf)?;
            }

            // Overlay new data
            block_buf[offset_in_block..offset_in_block + chunk_len]
                .copy_from_slice(&data[data_offset..data_offset + chunk_len]);

            // Allocate fresh physical block (lowest erase count)
            let new_phys = Self::allocate_block(&mut state)?;

            // Erase the new block -- widen to u64 before adding to prevent u32 overflow
            let global_new = u64::from(state.data_region_start) + u64::from(new_phys);
            let global_new_u32 = (global_new & 0xFFFF_FFFF) as u32;
            state.hw.erase_block(global_new_u32)?;
            state.erase_counts.increment(global_new_u32);
            state.ops_since_static_wl += 1;

            // Write the full block in write-page-sized chunks
            let phys_base = global_new * ebs_u64;
            let mut page_offset = 0usize;
            while page_offset < ebs as usize {
                let write_len = wps.min(ebs as usize - page_offset);
                state.hw.write_page(
                    phys_base + page_offset as u64,
                    &block_buf[page_offset..page_offset + write_len],
                )?;
                page_offset += wps;
            }

            // Update mapping
            state.block_map.assign(logical_block, new_phys);

            // Release old physical block back to free list
            if let Some(old) = old_phys {
                state.block_map.free_list.push(old);
            }

            // Check for static wear leveling
            if state.ops_since_static_wl >= STATIC_WL_INTERVAL {
                Self::try_static_wear_level(&mut state)?;
                state.ops_since_static_wl = 0;
            }

            data_offset += chunk_len;
            current_offset += chunk_len as u64;
            remaining -= chunk_len;
        }

        Ok(())
    }

    /// Set the logical length of the storage.
    pub fn set_len(&self, len: u64) -> core::result::Result<(), BackendError> {
        let mut state = self.state.lock();
        let ebs = u64::from(state.geometry.erase_block_size);
        debug_assert!(ebs > 0, "erase_block_size validated at construction");
        let old_blocks = u32::try_from(state.logical_len.div_ceil(ebs)).unwrap_or(u32::MAX);
        let new_blocks = u32::try_from(len.div_ceil(ebs)).unwrap_or(u32::MAX);

        // If shrinking, release blocks beyond the new length
        if new_blocks < old_blocks {
            for logical in new_blocks..old_blocks {
                state.block_map.release(logical);
            }
        }

        // Extend the forward map if needed
        if new_blocks as usize > state.block_map.forward.len() {
            state
                .block_map
                .forward
                .resize(new_blocks as usize, UNMAPPED);
        }

        state.logical_len = len;
        Ok(())
    }

    /// Return current logical length.
    pub fn len(&self) -> core::result::Result<u64, BackendError> {
        Ok(self.state.lock().logical_len)
    }

    /// Flush: persist current FTL metadata to journal and sync hardware.
    pub fn sync(&self) -> core::result::Result<(), BackendError> {
        let mut state = self.state.lock();
        let metadata = Self::serialize_metadata_inner(&state);
        let FtlState {
            ref hw,
            ref mut journal,
            ..
        } = *state;
        journal.commit(hw, &metadata)?;
        hw.sync()
    }

    /// Shutdown: final metadata persist.
    pub fn close(&self) -> core::result::Result<(), BackendError> {
        self.sync()
    }

    /// Allocate a free physical block with the lowest erase count.
    fn allocate_block(state: &mut FtlState<H>) -> core::result::Result<u32, BackendError> {
        let free = state.block_map.free_blocks();
        if free.is_empty() {
            #[cfg(feature = "std")]
            return Err(BackendError::Io(std::io::Error::other(
                "flash device full: no free blocks",
            )));
            #[cfg(not(feature = "std"))]
            return Err(BackendError::Message(String::from(
                "flash device full: no free blocks",
            )));
        }

        let best = state.erase_counts.pick_lowest(free).unwrap_or(free[0]);

        // Remove from free list via swap_remove for O(1) instead of O(n) retain
        if let Some(pos) = state.block_map.free_list.iter().position(|&b| b == best) {
            state.block_map.free_list.swap_remove(pos);
        }
        Ok(best)
    }

    /// Attempt a static wear-leveling swap: move cold data from a low-wear block
    /// to a high-wear block to distribute erase cycles evenly.
    fn try_static_wear_level(state: &mut FtlState<H>) -> core::result::Result<(), BackendError> {
        let ebs = state.geometry.erase_block_size;
        let ebs_u64 = u64::from(ebs);
        let wps = state.geometry.write_page_size as usize;
        debug_assert!(ebs > 0 && wps > 0, "geometry validated at construction");

        // Collect in-use physical blocks (data-region-relative indices)
        let in_use: Vec<u32> = state
            .block_map
            .forward
            .iter()
            .filter_map(|&p| if p != UNMAPPED { Some(p) } else { None })
            .collect();

        let swap = state.erase_counts.check_static_swap(
            STATIC_WL_THRESHOLD,
            &in_use,
            state.block_map.free_blocks(),
        );

        if let Some((hot_phys, cold_phys)) = swap {
            // Widen to u64 before adding to prevent u32 overflow
            let hot_global = u64::from(state.data_region_start) + u64::from(hot_phys);
            let cold_global = u64::from(state.data_region_start) + u64::from(cold_phys);
            let cold_global_u32 = (cold_global & 0xFFFF_FFFF) as u32;

            // Read hot block data
            let mut buf = vec![0u8; ebs as usize];
            state.hw.read(hot_global * ebs_u64, &mut buf)?;

            // Erase cold block and write data there
            state.hw.erase_block(cold_global_u32)?;
            state.erase_counts.increment(cold_global_u32);

            let phys_base = cold_global * ebs_u64;
            let mut page_offset = 0usize;
            while page_offset < ebs as usize {
                let write_len = wps.min(ebs as usize - page_offset);
                state.hw.write_page(
                    phys_base + page_offset as u64,
                    &buf[page_offset..page_offset + write_len],
                )?;
                page_offset += wps;
            }

            // Find which logical block maps to `hot_phys` and remap to `cold_phys`
            if let Some(logical) = state
                .block_map
                .reverse
                .get(hot_phys as usize)
                .copied()
                .filter(|&l| l != UNMAPPED)
            {
                state.block_map.assign(logical, cold_phys);
                state.block_map.free_list.push(hot_phys);
            }
        }

        Ok(())
    }

    /// Serialize all FTL metadata into a single byte buffer for journal persistence.
    fn serialize_metadata_inner(state: &FtlState<H>) -> Vec<u8> {
        let map_bytes = state.block_map.to_bytes();
        let erase_bytes = state.erase_counts.to_bytes();
        let bbt_bytes = state.bad_blocks.to_bytes();

        let total = 8 + 4 + map_bytes.len() + 4 + erase_bytes.len() + 4 + bbt_bytes.len();
        let mut out = Vec::with_capacity(total);

        out.extend_from_slice(&state.logical_len.to_le_bytes());

        out.extend_from_slice(&(map_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&map_bytes);

        out.extend_from_slice(&(erase_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&erase_bytes);

        out.extend_from_slice(&(bbt_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&bbt_bytes);

        out
    }

    /// Deserialize FTL metadata from journal payload bytes.
    fn deserialize_metadata(
        data: &[u8],
        data_physical_blocks: u32,
        total_blocks: u32,
    ) -> core::result::Result<(BlockMap, EraseCountTable, BadBlockTable, u64), BackendError> {
        let err = || -> BackendError {
            #[cfg(feature = "std")]
            {
                BackendError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "corrupted FTL metadata",
                ))
            }
            #[cfg(not(feature = "std"))]
            {
                BackendError::Message(String::from("corrupted FTL metadata"))
            }
        };

        if data.len() < 8 {
            return Err(err());
        }

        let mut cursor = 0usize;

        // logical_len
        let logical_len =
            u64::from_le_bytes(data[cursor..cursor + 8].try_into().map_err(|_| err())?);
        cursor += 8;

        // block map
        if cursor + 4 > data.len() {
            return Err(err());
        }
        let map_len =
            u32::from_le_bytes(data[cursor..cursor + 4].try_into().map_err(|_| err())?) as usize;
        cursor += 4;
        if cursor + map_len > data.len() {
            return Err(err());
        }
        let block_map = BlockMap::from_bytes(&data[cursor..cursor + map_len], data_physical_blocks);
        cursor += map_len;

        // erase counts
        if cursor + 4 > data.len() {
            return Err(err());
        }
        let erase_len =
            u32::from_le_bytes(data[cursor..cursor + 4].try_into().map_err(|_| err())?) as usize;
        cursor += 4;
        if cursor + erase_len > data.len() {
            return Err(err());
        }
        let erase_counts = EraseCountTable::from_bytes(&data[cursor..cursor + erase_len]);
        cursor += erase_len;

        // bad block table
        if cursor + 4 > data.len() {
            return Err(err());
        }
        let bbt_len =
            u32::from_le_bytes(data[cursor..cursor + 4].try_into().map_err(|_| err())?) as usize;
        cursor += 4;
        if cursor + bbt_len > data.len() {
            return Err(err());
        }
        let bad_blocks = BadBlockTable::from_bytes(&data[cursor..cursor + bbt_len], total_blocks)?;

        Ok((block_map, erase_counts, bad_blocks, logical_len))
    }
}
