use crate::error::BackendError;
use core::fmt::Debug;

/// Describes the physical geometry of a flash device.
///
/// Users must provide accurate geometry for their specific flash chip.
/// Incorrect values (especially `erase_block_size` or `write_page_size`) will
/// cause data corruption.
#[derive(Debug, Clone, Copy)]
pub struct FlashGeometry {
    /// Size of one erase block in bytes. Typical: 4096 (NOR) or 131072 (NAND).
    pub erase_block_size: u32,
    /// Size of one write page in bytes.
    /// For NOR flash this typically equals `erase_block_size`.
    /// For NAND, typical values are 2048 or 4096.
    pub write_page_size: u32,
    /// Total number of erase blocks on the device.
    pub total_blocks: u32,
    /// Maximum erase cycles per block before wear-out.
    /// Typical: 100,000 (NOR) or 3,000 to 10,000 (NAND).
    pub max_erase_cycles: u32,
}

impl FlashGeometry {
    /// Total device capacity in bytes.
    #[inline]
    pub fn total_capacity(&self) -> u64 {
        u64::from(self.erase_block_size) * u64::from(self.total_blocks)
    }

    /// Number of write pages per erase block.
    #[inline]
    pub fn pages_per_block(&self) -> u32 {
        if self.write_page_size == 0 {
            return 0;
        }
        self.erase_block_size / self.write_page_size
    }

    /// Compute the number of physical blocks reserved for FTL metadata.
    ///
    /// This covers two journal slots (double-buffered) plus 2 spare blocks
    /// for garbage collection during block replacement.
    #[allow(clippy::cast_possible_truncation)]
    pub fn reserved_blocks(&self) -> u32 {
        if self.erase_block_size == 0 {
            return 0;
        }
        let metadata_size = 4u64 * u64::from(self.total_blocks) // block map
            + 4u64 * u64::from(self.total_blocks)               // erase counts
            + u64::from(self.total_blocks.div_ceil(8))           // bad block bitmap
            + 64; // journal headers + checksums
        let ebs = u64::from(self.erase_block_size);
        let blocks_per_slot = metadata_size.div_ceil(ebs) as u32;
        let journal_blocks = blocks_per_slot * 2; // two slots (A and B)
        let spare_blocks = 2u32; // for copy-on-write block replacement
        journal_blocks + spare_blocks
    }

    /// Minimum number of spare data-region blocks reserved for copy-on-write
    /// headroom. Without these, a fully-allocated device has no room to write
    /// a new physical block before releasing the old one.
    const COW_HEADROOM_BLOCKS: u32 = 2;

    /// Number of logical blocks available for data storage after reserving
    /// FTL overhead and COW headroom.
    ///
    /// The COW headroom ensures that even when all logical blocks are
    /// allocated, `allocate_block()` can still find free physical blocks for
    /// copy-on-write operations.
    pub fn logical_block_count(&self) -> u32 {
        self.total_blocks
            .saturating_sub(self.reserved_blocks())
            .saturating_sub(Self::COW_HEADROOM_BLOCKS)
    }
}

/// Hardware abstraction trait for raw flash devices.
///
/// Implementors provide the low-level read, write, and erase operations for
/// their specific flash chip (SPI NOR, QSPI NOR, eMMC, parallel NAND, etc.).
///
/// All operations are blocking. For async hardware (DMA-based SPI), the
/// implementation should block until the operation completes.
///
/// # Guarantees required from implementations
///
/// - `read` returns the data most recently written (or `0xFF` for erased regions).
/// - `write_page` only modifies bits from 1 to 0 within the target write page.
/// - `erase_block` sets all bytes in the block to `0xFF`.
/// - `is_bad_block` correctly reports factory-marked and runtime-detected bad blocks.
pub trait FlashHardware: 'static + Debug + Send + Sync {
    /// Read `buf.len()` bytes from the device starting at absolute byte `offset`.
    ///
    /// The implementation must handle crossing write-page boundaries internally.
    fn read(&self, offset: u64, buf: &mut [u8]) -> core::result::Result<(), BackendError>;

    /// Write `data` to the device starting at absolute byte `offset`.
    ///
    /// `offset` must be aligned to `write_page_size`.
    /// `data.len()` must not exceed `write_page_size`.
    /// The target region must have been erased first.
    fn write_page(&self, offset: u64, data: &[u8]) -> core::result::Result<(), BackendError>;

    /// Erase the block at the given block index (0-based).
    ///
    /// After erase, all bytes in the block read as `0xFF`.
    fn erase_block(&self, block_index: u32) -> core::result::Result<(), BackendError>;

    /// Check if a block is marked as bad (factory or runtime).
    ///
    /// For NOR flash that does not have bad blocks, always return `Ok(false)`.
    fn is_bad_block(&self, block_index: u32) -> core::result::Result<bool, BackendError>;

    /// Mark a block as bad (runtime detection after erase/write failure).
    ///
    /// For NOR flash, this can be a no-op returning `Ok(())`.
    fn mark_bad_block(&self, block_index: u32) -> core::result::Result<(), BackendError>;

    /// Return the geometry of this flash device.
    fn geometry(&self) -> FlashGeometry;

    /// Fence/barrier: ensure all previous writes are physically committed.
    ///
    /// For SPI flash, this waits until the device status register indicates
    /// write-complete. For memory-mapped flash, this may be a no-op.
    fn sync(&self) -> core::result::Result<(), BackendError>;
}
