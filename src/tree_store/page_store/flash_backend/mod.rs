mod bad_block;
mod ftl;
pub mod hardware;
mod journal;
mod wear_leveling;

pub use hardware::{FlashGeometry, FlashHardware};

use crate::StorageBackend;
use crate::error::BackendError;
use core::fmt::{Debug, Formatter};
use ftl::FlashTranslationLayer;

/// Storage backend for bare-metal flash devices.
///
/// Wraps a user-provided [`FlashHardware`] implementation with a full Flash
/// Translation Layer (FTL) providing:
///
/// - **Wear leveling** — both dynamic (allocate lowest-wear free blocks) and
///   static (periodic cold/hot data swap).
/// - **Bad block management** — scan on mount, runtime detection, transparent
///   remapping.
/// - **Power-loss safety** — double-buffered metadata journal with xxh3-128
///   checksums, matching shodh-redb's own double-buffered commit slot pattern.
/// - **Copy-on-write** — all writes go to fresh physical blocks; old blocks
///   are erased and returned to the free pool.
///
/// # Usage
///
/// ```rust,ignore
/// use shodh_redb::{Builder, FlashBackend, FlashGeometry, FlashHardware};
///
/// // 1. Implement FlashHardware for your specific flash chip
/// struct MySpiFlash { /* SPI peripheral, CS pin, etc. */ }
/// impl FlashHardware for MySpiFlash { /* ... */ }
///
/// // 2. Create the backend and open the database
/// let hw = MySpiFlash::new(/* ... */);
/// let backend = FlashBackend::mount(hw).expect("flash mount failed");
/// let db = Builder::new().create_with_backend(backend).expect("db open");
/// ```
///
/// # eMMC note
///
/// eMMC devices have their own internal FTL. For eMMC, consider implementing
/// [`StorageBackend`] directly (like [`InMemoryBackend`](crate::InMemoryBackend))
/// to avoid the overhead of a redundant translation layer.
pub struct FlashBackend<H: FlashHardware> {
    ftl: FlashTranslationLayer<H>,
}

impl<H: FlashHardware> Debug for FlashBackend<H> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FlashBackend").finish_non_exhaustive()
    }
}

impl<H: FlashHardware> FlashBackend<H> {
    /// Mount an existing database from flash, recovering FTL state from the journal.
    ///
    /// If no valid journal is found (fresh/erased device), the device is formatted
    /// automatically.
    pub fn mount(hw: H) -> core::result::Result<Self, BackendError> {
        let ftl = FlashTranslationLayer::mount(hw)?;
        Ok(Self { ftl })
    }

    /// Format the flash device and create a fresh FTL.
    ///
    /// **All existing data on the device is lost.**
    pub fn format(hw: H) -> core::result::Result<Self, BackendError> {
        let ftl = FlashTranslationLayer::format(hw)?;
        Ok(Self { ftl })
    }
}

impl<H: FlashHardware> StorageBackend for FlashBackend<H> {
    fn len(&self) -> core::result::Result<u64, BackendError> {
        self.ftl.len()
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> core::result::Result<(), BackendError> {
        self.ftl.read(offset, out)
    }

    fn set_len(&self, len: u64) -> core::result::Result<(), BackendError> {
        self.ftl.set_len(len)
    }

    fn sync_data(&self) -> core::result::Result<(), BackendError> {
        self.ftl.sync()
    }

    fn write(&self, offset: u64, data: &[u8]) -> core::result::Result<(), BackendError> {
        self.ftl.write(offset, data)
    }

    fn close(&self) -> core::result::Result<(), BackendError> {
        self.ftl.close()
    }
}
