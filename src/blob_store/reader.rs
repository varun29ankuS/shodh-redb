use crate::Result;
use alloc::vec::Vec;

/// A seekable reader for blob data stored in the database.
///
/// `BlobReader` implements [`std::io::Read`] and [`std::io::Seek`] (when the
/// `std` feature is enabled), allowing partial and streaming reads of blob data
/// without loading the entire blob into memory.
///
/// The blob data is assembled from B-tree chunks at construction time and held
/// in memory for the lifetime of the reader. For very large blobs, prefer using
/// [`read_blob_range`](crate::WriteTransaction::read_blob_range) to read
/// specific byte ranges without materializing the entire blob.
///
/// Range reads bypass checksum verification since the stored checksum covers
/// the entire blob. Use [`crate::WriteTransaction::get_blob`] or
/// [`crate::ReadTransaction::get_blob`] for full-blob reads with integrity
/// verification.
pub struct BlobReader {
    /// Pre-assembled blob data from B-tree chunks.
    data: Vec<u8>,
    /// Current read cursor position within the blob.
    position: u64,
}

impl BlobReader {
    /// Create a new `BlobReader` from pre-assembled chunk data.
    pub(crate) fn new(data: Vec<u8>) -> Self {
        Self { data, position: 0 }
    }

    /// Returns the total length of the blob in bytes.
    pub fn len(&self) -> u64 {
        self.data.len() as u64
    }

    /// Returns `true` if the blob has zero length.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the current read cursor position within the blob.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Returns the number of bytes remaining from the current position.
    pub fn remaining(&self) -> u64 {
        (self.data.len() as u64).saturating_sub(self.position)
    }

    /// Read a specific range of bytes from the blob.
    ///
    /// This is a convenience method that seeks to `offset` and reads `length` bytes.
    /// Returns the data as a `Vec<u8>`. The cursor position is advanced past the
    /// read range.
    pub fn read_range(&mut self, offset: u64, length: usize) -> Result<Vec<u8>> {
        if length == 0 {
            return Ok(Vec::new());
        }
        let blob_length = self.data.len() as u64;
        let end =
            offset
                .checked_add(length as u64)
                .ok_or(crate::StorageError::BlobRangeOutOfBounds {
                    blob_length,
                    requested_offset: offset,
                    requested_length: length as u64,
                })?;
        if end > blob_length {
            return Err(crate::StorageError::BlobRangeOutOfBounds {
                blob_length,
                requested_offset: offset,
                requested_length: length as u64,
            });
        }

        #[allow(clippy::cast_possible_truncation)]
        let start = offset as usize;
        #[allow(clippy::cast_possible_truncation)]
        let end = end as usize;
        self.position = end as u64;
        Ok(self.data[start..end].to_vec())
    }
}

#[cfg(feature = "std")]
impl std::io::Read for BlobReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let blob_length = self.data.len() as u64;
        if self.position >= blob_length {
            return Ok(0);
        }

        #[allow(clippy::cast_possible_truncation)]
        let remaining = (blob_length - self.position) as usize;
        let to_read = buf.len().min(remaining);
        if to_read == 0 {
            return Ok(0);
        }

        #[allow(clippy::cast_possible_truncation)]
        let start = self.position as usize;
        buf[..to_read].copy_from_slice(&self.data[start..start + to_read]);
        self.position += to_read as u64;
        Ok(to_read)
    }
}

#[cfg(feature = "std")]
impl std::io::Seek for BlobReader {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let blob_length = self.data.len() as u64;
        let new_pos = match pos {
            std::io::SeekFrom::Start(offset) => i64::try_from(offset).ok(),
            std::io::SeekFrom::End(offset) => i64::try_from(blob_length)
                .ok()
                .and_then(|len| len.checked_add(offset)),
            std::io::SeekFrom::Current(offset) => i64::try_from(self.position)
                .ok()
                .and_then(|pos| pos.checked_add(offset)),
        };

        match new_pos {
            Some(pos) if pos >= 0 => {
                #[allow(clippy::cast_sign_loss)]
                let unsigned = pos as u64;
                self.position = unsigned;
                Ok(self.position)
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid seek to a negative or overflowing position",
            )),
        }
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl core::fmt::Debug for BlobReader {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BlobReader")
            .field("blob_length", &self.data.len())
            .field("position", &self.position)
            .finish()
    }
}
