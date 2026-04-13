//! Transparent large-value indirection for `BfTree`.
//!
//! When `blob_threshold > 0`, values exceeding the threshold are split into
//! fixed-size chunks stored in the `__bf_value_chunks` system table.  A compact
//! 16-byte [`BlobHandle`] replaces the original value inline in the leaf page.
//!
//! | System table           | Key                                           | Value        |
//! |------------------------|-----------------------------------------------|--------------|
//! | `__bf_value_chunks`    | `blob_id` (4B BE) + `chunk_idx` (4B BE)       | chunk bytes  |
//! | `__bf_value_seq`       | `"next_id"`                                   | u32 LE       |
//!
//! The indirection is transparent to all `StorageWrite`/`StorageRead` consumers.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};

use super::database::{TableKind, encode_table_key};
use super::error::BfTreeError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic bytes identifying a [`BlobHandle`] vs raw inline data.
const BLOB_MAGIC: [u8; 4] = [0xB1, 0x0B, 0xDA, 0x7A];

/// Serialized size of a [`BlobHandle`].
pub(crate) const BLOB_HANDLE_SIZE: usize = 16;

/// System table that stores value chunks.
pub(crate) const VALUE_CHUNKS_TABLE: &str = "__bf_value_chunks";

/// System table that persists the next blob-id counter.
pub(crate) const VALUE_SEQ_TABLE: &str = "__bf_value_seq";

/// Key for the persisted blob-id counter.
pub(crate) const VALUE_SEQ_KEY: &[u8] = b"next_id";

/// Maximum chunk size for value data records.
///
/// Key overhead for `__bf_value_chunks`:
///   2 (`name_len`) + 20 (table name) + 1 (kind) + 8 (`blob_id` + `chunk_idx`) = 31 bytes.
/// `BfTree` default `max_record_size` is 1568.  Conservative: 1568 - 31 - 32 (framing) = 1505.
/// Use 1024 to match the existing `blob_store` chunk size and leave headroom.
pub(crate) const VALUE_CHUNK_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// BlobHandle
// ---------------------------------------------------------------------------

/// A 16-byte handle stored inline in the `BfTree` leaf page, pointing to
/// out-of-line chunked data in `__bf_value_chunks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlobHandle {
    pub(crate) chunk_count: u16,
    pub(crate) total_len: u32,
    pub(crate) blob_id: u32,
}

impl BlobHandle {
    /// Serialize to 16 bytes.
    pub(crate) fn encode(&self) -> [u8; BLOB_HANDLE_SIZE] {
        let mut buf = [0u8; BLOB_HANDLE_SIZE];
        buf[0..4].copy_from_slice(&BLOB_MAGIC);
        buf[4..6].copy_from_slice(&self.chunk_count.to_le_bytes());
        // buf[6..8] reserved
        buf[8..12].copy_from_slice(&self.total_len.to_le_bytes());
        buf[12..16].copy_from_slice(&self.blob_id.to_le_bytes());
        buf
    }

    /// Deserialize from exactly 16 bytes.  Returns `None` if magic mismatch.
    pub(crate) fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != BLOB_HANDLE_SIZE {
            return None;
        }
        if bytes[0..4] != BLOB_MAGIC {
            return None;
        }
        let chunk_count = u16::from_le_bytes([bytes[4], bytes[5]]);
        let total_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let blob_id = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        Some(Self {
            chunk_count,
            total_len,
            blob_id,
        })
    }
}

/// Returns `true` if `bytes` looks like an encoded [`BlobHandle`].
///
/// Checks length == 16 AND the 4-byte magic prefix.  False-positive rate ≈ 2⁻³².
#[inline]
pub(crate) fn is_blob_handle(bytes: &[u8]) -> bool {
    bytes.len() == BLOB_HANDLE_SIZE && bytes[0..4] == BLOB_MAGIC
}

// ---------------------------------------------------------------------------
// Chunk key encoding
// ---------------------------------------------------------------------------

/// Encode a chunk key for `__bf_value_chunks`:
/// `[table_prefix][blob_id (4B BE)][chunk_idx (4B BE)]`.
pub(crate) fn encode_chunk_key(blob_id: u32, chunk_idx: u32) -> Vec<u8> {
    let mut key_bytes = Vec::with_capacity(8);
    #[allow(clippy::big_endian_bytes)]
    key_bytes.extend_from_slice(&blob_id.to_be_bytes());
    #[allow(clippy::big_endian_bytes)]
    key_bytes.extend_from_slice(&chunk_idx.to_be_bytes());
    encode_table_key(VALUE_CHUNKS_TABLE, TableKind::Regular, &key_bytes)
}

// ---------------------------------------------------------------------------
// Chunk write / read / delete  (operate on WriteBuffer + BfTreeAdapter)
// ---------------------------------------------------------------------------

/// Split `data` into chunks and write each to `buffer`.
///
/// Returns the number of chunks written and the allocated `blob_id`.
pub(crate) fn write_chunks(
    buffer: &mut super::buffered_txn::WriteBuffer,
    blob_id_counter: &AtomicU32,
    data: &[u8],
) -> Result<BlobHandle, BfTreeError> {
    let blob_id = blob_id_counter.fetch_add(1, Ordering::Relaxed);
    let total_len = u32::try_from(data.len()).map_err(|_| {
        BfTreeError::InvalidKV(alloc::format!(
            "blob value too large: {} bytes exceeds u32::MAX",
            data.len()
        ))
    })?;
    let mut chunk_idx: u32 = 0;

    for chunk in data.chunks(VALUE_CHUNK_SIZE) {
        let key = encode_chunk_key(blob_id, chunk_idx);
        buffer
            .put(key, chunk.to_vec())
            .map_err(|e| BfTreeError::InvalidKV(alloc::format!("chunk buffer full: {e}")))?;
        chunk_idx += 1;
    }

    let chunk_count = u16::try_from(chunk_idx).map_err(|_| {
        BfTreeError::InvalidKV(alloc::format!(
            "blob requires {chunk_idx} chunks, exceeds u16::MAX"
        ))
    })?;

    Ok(BlobHandle {
        chunk_count,
        total_len,
        blob_id,
    })
}

/// Read all chunks for `blob_id` and reassemble the original value.
///
/// Tries `buffer` first (chunks may not be flushed yet), then falls back to
/// the `BfTree` adapter.
pub(crate) fn read_blob(
    handle: &BlobHandle,
    buffer: &super::buffered_txn::WriteBuffer,
    adapter: &super::adapter::BfTreeAdapter,
) -> Result<Vec<u8>, BfTreeError> {
    let mut out = Vec::with_capacity(handle.total_len as usize);
    let mut read_buf = alloc::vec![0u8; VALUE_CHUNK_SIZE + 64];

    for idx in 0..u32::from(handle.chunk_count) {
        let key = encode_chunk_key(handle.blob_id, idx);

        // Check write buffer first (may not be flushed yet).
        match buffer.get(&key) {
            super::buffered_txn::BufferLookup::Found(v) => {
                out.extend_from_slice(&v);
            }
            super::buffered_txn::BufferLookup::Tombstone => {
                return Err(BfTreeError::InvalidKV(alloc::format!(
                    "blob chunk {}/{} tombstoned for blob_id {}",
                    idx,
                    handle.chunk_count,
                    handle.blob_id
                )));
            }
            super::buffered_txn::BufferLookup::NotInBuffer => {
                // Read from BfTree.
                let n = adapter.read(&key, &mut read_buf)?;
                out.extend_from_slice(&read_buf[..n as usize]);
            }
        }
    }

    if out.len() != handle.total_len as usize {
        return Err(BfTreeError::InvalidKV(alloc::format!(
            "blob reassembly length mismatch: expected {}, got {}",
            handle.total_len,
            out.len()
        )));
    }

    Ok(out)
}

/// Read all chunks for `blob_id` from the `BfTree` adapter only (no buffer).
///
/// Used by read-only transactions where there is no write buffer.
pub(crate) fn read_blob_readonly(
    handle: &BlobHandle,
    adapter: &super::adapter::BfTreeAdapter,
) -> Result<Vec<u8>, BfTreeError> {
    let mut out = Vec::with_capacity(handle.total_len as usize);
    let mut read_buf = alloc::vec![0u8; VALUE_CHUNK_SIZE + 64];

    for idx in 0..u32::from(handle.chunk_count) {
        let key = encode_chunk_key(handle.blob_id, idx);
        let n = adapter.read(&key, &mut read_buf)?;
        out.extend_from_slice(&read_buf[..n as usize]);
    }

    if out.len() != handle.total_len as usize {
        return Err(BfTreeError::InvalidKV(alloc::format!(
            "blob reassembly length mismatch: expected {}, got {}",
            handle.total_len,
            out.len()
        )));
    }

    Ok(out)
}

/// Buffer tombstones for all chunks of a blob handle.
pub(crate) fn delete_chunks(
    buffer: &mut super::buffered_txn::WriteBuffer,
    handle: &BlobHandle,
) {
    for idx in 0..u32::from(handle.chunk_count) {
        let key = encode_chunk_key(handle.blob_id, idx);
        buffer.delete(key);
    }
}

/// Recover the next blob-id counter by reading the persisted value from
/// `__bf_value_seq`.  Returns 1 if no counter exists yet.
pub(crate) fn recover_blob_id(
    adapter: &super::adapter::BfTreeAdapter,
) -> u32 {
    let seq_key = encode_table_key(VALUE_SEQ_TABLE, TableKind::Regular, VALUE_SEQ_KEY);
    let mut buf = [0u8; 4];
    match adapter.read(&seq_key, &mut buf) {
        Ok(n) if n as usize >= 4 => u32::from_le_bytes(buf),
        _ => 1, // Start from 1 (0 is reserved as "no blob").
    }
}

/// Persist the current blob-id counter to `__bf_value_seq`.
pub(crate) fn persist_blob_id(
    buffer: &mut super::buffered_txn::WriteBuffer,
    next_id: u32,
) -> Result<(), BfTreeError> {
    let key = encode_table_key(VALUE_SEQ_TABLE, TableKind::Regular, VALUE_SEQ_KEY);
    buffer
        .put(key, next_id.to_le_bytes().to_vec())
        .map_err(|e| BfTreeError::InvalidKV(alloc::format!("persist blob id: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_handle_roundtrip() {
        let handle = BlobHandle {
            chunk_count: 42,
            total_len: 123_456,
            blob_id: 99,
        };
        let encoded = handle.encode();
        assert_eq!(encoded.len(), BLOB_HANDLE_SIZE);
        assert!(is_blob_handle(&encoded));

        let decoded = BlobHandle::decode(&encoded).unwrap();
        assert_eq!(decoded, handle);
    }

    #[test]
    fn is_blob_handle_rejects_short() {
        assert!(!is_blob_handle(&[0xB1, 0x0B, 0xDA, 0x7A]));
        assert!(!is_blob_handle(&[]));
        assert!(!is_blob_handle(&[0u8; 15]));
    }

    #[test]
    fn is_blob_handle_rejects_wrong_magic() {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
        assert!(!is_blob_handle(&buf));
    }

    #[test]
    fn chunk_key_ordering() {
        // Chunk keys should sort by blob_id first, then chunk_idx.
        let k0 = encode_chunk_key(1, 0);
        let k1 = encode_chunk_key(1, 1);
        let k2 = encode_chunk_key(2, 0);
        assert!(k0 < k1);
        assert!(k1 < k2);
    }
}
