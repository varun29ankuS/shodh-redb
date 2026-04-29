use crate::error::BackendError;
use crate::tree_store::page_store::xxh3::hash128_with_seed;
#[cfg(not(feature = "std"))]
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::hardware::FlashHardware;

/// Magic bytes identifying a valid FTL journal entry.
const JOURNAL_MAGIC: [u8; 4] = [b'F', b'T', b'L', b'J'];

/// Size of the journal header: 4B magic + 8B sequence + 4B length = 16 bytes.
const JOURNAL_HEADER_SIZE: usize = 16;

/// Size of the journal checksum trailer: 16 bytes (xxh3-128).
const JOURNAL_CHECKSUM_SIZE: usize = 16;

/// Fixed seed for journal checksums.
const JOURNAL_CHECKSUM_SEED: u64 = 0xF7DE_ADBE_EFCA_FE00;

/// Double-buffered FTL metadata journal for power-loss safety.
///
/// Two contiguous regions on flash (slot A and slot B) alternate storing
/// the FTL metadata (block map, erase counts, bad block table). A monotonic
/// sequence number and xxh3-128 checksum determine which slot is current
/// after power-loss recovery.
pub(super) struct FtlJournal {
    /// First physical block index of slot A.
    slot_a_start: u32,
    /// Number of contiguous blocks in each slot.
    blocks_per_slot: u32,
    /// Whether the current active slot is A (true) or B (false).
    current_is_a: bool,
    /// Monotonically increasing sequence number.
    seq_no: u64,
    /// Erase block size (cached from geometry).
    erase_block_size: u32,
}

impl FtlJournal {
    /// Mount the journal by scanning both slots and recovering the latest valid state.
    ///
    /// Returns `(journal, Option<payload>)`. If no valid journal is found (fresh device),
    /// the payload is `None`.
    pub fn mount<H: FlashHardware>(
        hw: &H,
        slot_a_start: u32,
        blocks_per_slot: u32,
    ) -> core::result::Result<(Self, Option<Vec<u8>>), BackendError> {
        let geo = hw.geometry();
        let slot_b_start = slot_a_start + blocks_per_slot;

        let (seq_a, payload_a) =
            Self::try_read_slot(hw, slot_a_start, blocks_per_slot, geo.erase_block_size)?;
        let (seq_b, payload_b) =
            Self::try_read_slot(hw, slot_b_start, blocks_per_slot, geo.erase_block_size)?;

        let (current_is_a, seq_no, payload) = match (seq_a, seq_b) {
            (Some(sa), Some(sb)) => {
                if sa >= sb {
                    (true, sa, payload_a)
                } else {
                    (false, sb, payload_b)
                }
            }
            (Some(sa), None) => (true, sa, payload_a),
            (None, Some(sb)) => (false, sb, payload_b),
            (None, None) => (true, 0, None),
        };

        Ok((
            Self {
                slot_a_start,
                blocks_per_slot,
                current_is_a,
                seq_no,
                erase_block_size: geo.erase_block_size,
            },
            payload,
        ))
    }

    /// Write FTL metadata to the next journal slot (alternating A/B).
    ///
    /// The sequence is: erase target slot, write header + payload + checksum,
    /// then call `hw.sync()`.
    pub fn commit<H: FlashHardware>(
        &mut self,
        hw: &H,
        metadata: &[u8],
    ) -> core::result::Result<(), BackendError> {
        // Write to the *other* slot
        let target_is_a = !self.current_is_a;
        let target_start = if target_is_a {
            self.slot_a_start
        } else {
            self.slot_a_start + self.blocks_per_slot
        };

        self.seq_no += 1;
        let entry = Self::serialize_entry(self.seq_no, metadata);

        // Erase target slot blocks
        for i in 0..self.blocks_per_slot {
            hw.erase_block(target_start + i)?;
        }

        // Write the serialized entry across the slot's pages
        let ebs = u64::from(self.erase_block_size);
        let base_offset = u64::from(target_start) * ebs;
        let page_size = hw.geometry().write_page_size as usize;

        let mut offset = 0usize;
        while offset < entry.len() {
            let chunk_len = (entry.len() - offset).min(page_size);
            // Pad to full page size with 0xFF (erased state)
            let mut page_buf = vec![0xFFu8; page_size];
            page_buf[..chunk_len].copy_from_slice(&entry[offset..offset + chunk_len]);
            hw.write_page(base_offset + offset as u64, &page_buf)?;
            offset += page_size;
        }

        hw.sync()?;
        self.current_is_a = target_is_a;
        Ok(())
    }

    /// Serialize a journal entry: header + payload + checksum.
    #[allow(clippy::cast_possible_truncation)]
    fn serialize_entry(seq_no: u64, payload: &[u8]) -> Vec<u8> {
        let payload_len = payload.len() as u32;
        let total = JOURNAL_HEADER_SIZE + payload.len() + JOURNAL_CHECKSUM_SIZE;
        let mut entry = Vec::with_capacity(total);

        // Header
        entry.extend_from_slice(&JOURNAL_MAGIC);
        entry.extend_from_slice(&seq_no.to_le_bytes());
        entry.extend_from_slice(&payload_len.to_le_bytes());

        // Payload
        entry.extend_from_slice(payload);

        // Checksum over header + payload
        let checksum = hash128_with_seed(&entry, JOURNAL_CHECKSUM_SEED);
        entry.extend_from_slice(&checksum.to_le_bytes());

        entry
    }

    /// Try to read and validate a journal slot.
    ///
    /// Returns `(Some(seq_no), Some(payload))` if valid, `(None, None)` if corrupt/empty.
    fn try_read_slot<H: FlashHardware>(
        hw: &H,
        slot_start: u32,
        blocks_per_slot: u32,
        erase_block_size: u32,
    ) -> core::result::Result<(Option<u64>, Option<Vec<u8>>), BackendError> {
        let ebs = u64::from(erase_block_size);
        let base_offset = u64::from(slot_start) * ebs;
        let slot_capacity = u64::from(blocks_per_slot) * ebs;

        // Read the header first
        if slot_capacity < JOURNAL_HEADER_SIZE as u64 {
            return Ok((None, None));
        }

        let mut header = [0u8; JOURNAL_HEADER_SIZE];
        hw.read(base_offset, &mut header)?;

        // Check magic
        if header[0..4] != JOURNAL_MAGIC {
            return Ok((None, None));
        }

        let seq_no = u64::from_le_bytes([
            header[4], header[5], header[6], header[7], header[8], header[9], header[10],
            header[11],
        ]);

        let payload_len_raw = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
        let payload_len_u64 = u64::from(payload_len_raw);
        // Validate payload_len before usize conversion to prevent overflow on 32-bit targets.
        // Max valid payload = slot_capacity - header - checksum.
        let overhead = (JOURNAL_HEADER_SIZE + JOURNAL_CHECKSUM_SIZE) as u64;
        if payload_len_u64 > slot_capacity.saturating_sub(overhead) {
            return Ok((None, None));
        }
        // Safe: u32 always fits in usize (minimum 32-bit).
        let payload_len = payload_len_raw as usize;
        let total_len = JOURNAL_HEADER_SIZE + payload_len + JOURNAL_CHECKSUM_SIZE;

        // Read full entry
        let mut entry = vec![0u8; total_len];
        hw.read(base_offset, &mut entry)?;

        // Validate checksum
        let checksum_offset = total_len - JOURNAL_CHECKSUM_SIZE;
        let stored_bytes: [u8; 16] =
            entry[checksum_offset..total_len].try_into().map_err(|_| {
                #[cfg(feature = "std")]
                {
                    BackendError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "journal checksum read error",
                    ))
                }
                #[cfg(not(feature = "std"))]
                {
                    BackendError::Message(String::from("journal checksum read error"))
                }
            })?;
        let stored_checksum = u128::from_le_bytes(stored_bytes);
        let computed_checksum = hash128_with_seed(&entry[..checksum_offset], JOURNAL_CHECKSUM_SEED);

        if stored_checksum != computed_checksum {
            return Ok((None, None));
        }

        let payload = entry[JOURNAL_HEADER_SIZE..checksum_offset].to_vec();
        Ok((Some(seq_no), Some(payload)))
    }
}
