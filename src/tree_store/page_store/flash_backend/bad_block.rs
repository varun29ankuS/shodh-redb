use crate::error::BackendError;
#[cfg(not(feature = "std"))]
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::hardware::FlashHardware;

/// Bitmap tracking bad blocks on the flash device.
///
/// One bit per physical block: bit `N` = 1 means block `N` is bad (factory-marked
/// or runtime-detected). The table is serialized as a packed byte vector and
/// persisted in the FTL journal.
pub(super) struct BadBlockTable {
    /// Packed bitmap: bit `N` in `bitmap[N / 8]` (LSB-first) = 1 means bad.
    bitmap: Vec<u8>,
    total_blocks: u32,
}

impl BadBlockTable {
    /// Scan the hardware and build an initial bad block table by querying each block.
    pub fn scan<H: FlashHardware>(hw: &H) -> core::result::Result<Self, BackendError> {
        let geo = hw.geometry();
        let byte_len = geo.total_blocks.div_ceil(8) as usize;
        let mut bitmap = vec![0u8; byte_len];

        for block in 0..geo.total_blocks {
            if hw.is_bad_block(block)? {
                let byte_idx = (block / 8) as usize;
                let bit_idx = block % 8;
                bitmap[byte_idx] |= 1 << bit_idx;
            }
        }

        Ok(Self {
            bitmap,
            total_blocks: geo.total_blocks,
        })
    }

    /// Deserialize a bad block table from raw bytes.
    ///
    /// Returns an error if `data` is too short for the given `total_blocks`.
    pub fn from_bytes(data: &[u8], total_blocks: u32) -> core::result::Result<Self, BackendError> {
        let expected = total_blocks.div_ceil(8) as usize;
        if data.len() < expected {
            #[cfg(feature = "std")]
            return Err(BackendError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad block table too short",
            )));
            #[cfg(not(feature = "std"))]
            return Err(BackendError::Message(String::from(
                "bad block table too short",
            )));
        }
        Ok(Self {
            bitmap: data[..expected].to_vec(),
            total_blocks,
        })
    }

    /// Serialize the table to bytes for journal persistence.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bitmap.clone()
    }

    /// Check if a physical block is marked as bad.
    #[inline]
    pub fn is_bad(&self, block_index: u32) -> bool {
        if block_index >= self.total_blocks {
            return true; // out-of-range blocks treated as bad
        }
        let byte_idx = (block_index / 8) as usize;
        let bit_idx = block_index % 8;
        (self.bitmap[byte_idx] >> bit_idx) & 1 == 1
    }

    /// Mark a physical block as bad at runtime.
    #[allow(dead_code)]
    pub fn mark_bad(&mut self, block_index: u32) {
        if block_index >= self.total_blocks {
            return;
        }
        let byte_idx = (block_index / 8) as usize;
        let bit_idx = block_index % 8;
        self.bitmap[byte_idx] |= 1 << bit_idx;
    }

    /// Number of usable (non-bad) blocks.
    #[allow(dead_code)]
    pub fn usable_block_count(&self) -> u32 {
        let mut count = 0u32;
        for block in 0..self.total_blocks {
            if !self.is_bad(block) {
                count += 1;
            }
        }
        count
    }
}
