//! Blob store for the Bf-Tree backend.
//!
//! Unlike the legacy blob store that uses a contiguous append-only region,
//! this implementation chunks blob data into `BfTree` KV records. Each chunk
//! is stored as a separate key-value pair, allowing the `BfTree`'s concurrent
//! B+tree to manage blob data alongside regular table data.
//!
//! # System Tables
//!
//! | Table | Key | Value |
//! |-------|-----|-------|
//! | `__bf_blob_meta` | BlobId (16B) | BlobMeta (137B) |
//! | `__bf_blob_data` | BlobId (16B) + chunk_idx (4B) | chunk bytes |
//! | `__bf_blob_dedup` | SHA-256 (32B) | DedupVal (40B) |
//! | `__bf_blob_temporal` | TemporalKey (32B) | empty |
//! | `__bf_blob_causal` | CausalEdgeKey (32B) | CausalEdge (80B) |
//! | `__bf_blob_tag` | TagKey (49B) | empty |
//! | `__bf_blob_ns` | NamespaceKey (80B) | empty |
//! | `__bf_blob_seq` | "next_seq" | u64 LE |

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::AtomicU64;
use std::collections::VecDeque;
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use crate::blob_store::types::{
    BlobId, BlobMeta, BlobRef, CausalEdge, CausalEdgeKey, ContentType, DedupVal, NamespaceKey,
    Sha256Key, StoreOptions, TagKey, TemporalKey,
};
use crate::temporal::HybridLogicalClock;

use super::adapter::BfTreeAdapter;
use super::buffered_txn::WriteBuffer;
use super::database::{TableKind, encode_table_key, table_prefix, table_prefix_end};
use super::error::BfTreeError;

// ---------------------------------------------------------------------------
// System table names
// ---------------------------------------------------------------------------

const BLOB_META_TABLE: &str = "__bf_blob_meta";
const BLOB_DATA_TABLE: &str = "__bf_blob_data";
const BLOB_DEDUP_TABLE: &str = "__bf_blob_dedup";
const BLOB_TEMPORAL_TABLE: &str = "__bf_blob_temporal";
const BLOB_CAUSAL_TABLE: &str = "__bf_blob_causal";
const BLOB_TAG_TABLE: &str = "__bf_blob_tag";
const BLOB_NS_TABLE: &str = "__bf_blob_ns";
const BLOB_SEQ_TABLE: &str = "__bf_blob_seq";

/// Sequence counter key within the `__bf_blob_seq` table.
const SEQ_KEY: &[u8] = b"next_seq";

/// Bytes reserved for content prefix hashing (FNV-1a-64 of first N bytes).
const PREFIX_HASH_LEN: usize = 4096;

/// Maximum chunk size for blob data records.
/// `BfTree`'s default `max_record_size` is 1568 (key + value total).
/// Key overhead for `__bf_blob_data`: 2 (`name_len`) + 14 (name) + 16 (`blob_id`) + 4 (`chunk_idx`) = 36 bytes.
/// Additional `BfTree` internal framing ~32 bytes. Conservative: 1568 - 36 - 32 = 1500.
const MAX_CHUNK_SIZE: usize = 1024;

/// Minimum blob size for dedup eligibility.
const DEDUP_MIN_SIZE: usize = 4096;

/// Maximum number of tags per blob.
const MAX_TAGS: usize = 8;

// ---------------------------------------------------------------------------
// Key encoding helpers
// ---------------------------------------------------------------------------

/// Encode a blob data chunk key: `[table_prefix][blob_id (16B)][chunk_idx (4B LE)]`.
fn encode_chunk_key(blob_id: BlobId, chunk_idx: u32) -> Vec<u8> {
    let mut key_bytes = Vec::with_capacity(BlobId::SERIALIZED_SIZE + 4);
    key_bytes.extend_from_slice(&blob_id.to_be_bytes());
    #[allow(clippy::big_endian_bytes)]
    key_bytes.extend_from_slice(&chunk_idx.to_be_bytes());
    encode_table_key(BLOB_DATA_TABLE, TableKind::Regular, &key_bytes)
}

/// Encode a blob metadata key.
fn encode_meta_key(blob_id: BlobId) -> Vec<u8> {
    encode_table_key(BLOB_META_TABLE, TableKind::Regular, &blob_id.to_be_bytes())
}

/// Encode a dedup index key.
fn encode_dedup_key(sha256: &Sha256Key) -> Vec<u8> {
    encode_table_key(BLOB_DEDUP_TABLE, TableKind::Regular, &sha256.0)
}

/// Encode a temporal index key.
fn encode_temporal_key(tk: &TemporalKey) -> Vec<u8> {
    encode_table_key(BLOB_TEMPORAL_TABLE, TableKind::Regular, &tk.to_be_bytes())
}

/// Encode a causal edge key.
fn encode_causal_key(cek: &CausalEdgeKey) -> Vec<u8> {
    encode_table_key(BLOB_CAUSAL_TABLE, TableKind::Regular, &cek.to_be_bytes())
}

/// Encode a tag index key.
fn encode_tag_key(tk: &TagKey) -> Vec<u8> {
    encode_table_key(BLOB_TAG_TABLE, TableKind::Regular, &tk.to_be_bytes())
}

/// Encode a namespace index key.
fn encode_ns_key(nk: &NamespaceKey) -> Vec<u8> {
    encode_table_key(BLOB_NS_TABLE, TableKind::Regular, &nk.to_be_bytes())
}

/// Encode the sequence counter key.
fn encode_seq_key() -> Vec<u8> {
    encode_table_key(BLOB_SEQ_TABLE, TableKind::Regular, SEQ_KEY)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Read a raw value from `BfTree` by encoded key. Returns `None` if not found.
fn read_raw(adapter: &BfTreeAdapter, encoded_key: &[u8]) -> Result<Option<Vec<u8>>, BfTreeError> {
    let max_val = adapter.inner().config().get_cb_max_record_size();
    let mut buf = vec![0u8; max_val];
    match adapter.read(encoded_key, &mut buf) {
        Ok(len) => Ok(Some(buf[..len as usize].to_vec())),
        Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Read a raw value checking the write buffer first, then `BfTree`.
fn read_raw_buffered(
    buffer: &Mutex<WriteBuffer>,
    adapter: &BfTreeAdapter,
    encoded_key: &[u8],
) -> Result<Option<Vec<u8>>, BfTreeError> {
    use super::buffered_txn::BufferLookup;
    let buf_guard = buffer.lock().unwrap_or_else(|e| e.into_inner());
    match buf_guard.get(encoded_key) {
        BufferLookup::Found(v) => Ok(Some(v)),
        BufferLookup::Tombstone => Ok(None),
        BufferLookup::NotInBuffer => {
            drop(buf_guard);
            read_raw(adapter, encoded_key)
        }
    }
}

/// Key-value pair type for scan results: `(key_bytes, value_bytes)`.
type KvPair = (Vec<u8>, Vec<u8>);

/// Scan a range, merging results from the write buffer and `BfTree`.
///
/// Returns `Vec<KvPair>` for all entries in `[start, end)`
/// where buffer entries override `BfTree` entries (tombstones suppress).
/// Keys are returned with the table prefix stripped (offset by `prefix_len`).
fn scan_range_buffered(
    buffer: &Mutex<WriteBuffer>,
    adapter: &BfTreeAdapter,
    start: &[u8],
    end: &[u8],
    prefix_len: usize,
) -> Result<Vec<KvPair>, BfTreeError> {
    use alloc::collections::BTreeMap;

    // Collect BfTree entries first.
    // Use 3x max_record for safety margin: key + value + alignment overhead.
    let max_record = adapter.inner().config().get_cb_max_record_size();
    let mut scan_buf = vec![0u8; max_record * 3];
    let mut combined: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();

    if let Ok(mut iter) = adapter.scan_range(start, end) {
        while let Some((key_len, val_len)) = iter.next(&mut scan_buf) {
            // Validate that key + value fits within the scan buffer.
            if key_len + val_len > scan_buf.len() {
                continue;
            }
            if key_len > prefix_len {
                let full_key = scan_buf[..key_len].to_vec();
                let val = scan_buf[key_len..key_len + val_len].to_vec();
                combined.insert(full_key, val);
            }
        }
    }

    // Overlay buffer entries (inserts override, tombstones remove).
    let buf_guard = buffer.lock().unwrap_or_else(|e| e.into_inner());
    for (key, value) in buf_guard.range(start, end) {
        match value {
            Some(v) => {
                combined.insert(key.clone(), v.clone());
            }
            None => {
                combined.remove(key);
            }
        }
    }
    drop(buf_guard);

    Ok(combined
        .into_iter()
        .filter(|(k, _)| k.len() > prefix_len)
        .map(|(k, v)| (k[prefix_len..].to_vec(), v))
        .collect())
}

/// Allocate the next blob sequence number, writing through the buffer.
///
/// The entire read-increment-write is performed under a single mutex acquisition
/// to prevent concurrent callers from observing the same sequence value.
fn next_sequence(buffer: &Mutex<WriteBuffer>, adapter: &BfTreeAdapter) -> Result<u64, BfTreeError> {
    use super::buffered_txn::BufferLookup;

    let seq_key = encode_seq_key();
    let mut buf_guard = buffer.lock().unwrap_or_else(|e| e.into_inner());

    // Read current value from buffer first, then BfTree.
    let current = match buf_guard.get(&seq_key) {
        BufferLookup::Found(v) => Some(v),
        BufferLookup::Tombstone => None,
        BufferLookup::NotInBuffer => {
            // Read from BfTree while still holding the buffer lock. This is safe
            // because BfTree reads are non-blocking and don't acquire the buffer mutex.
            read_raw(adapter, &seq_key)?
        }
    };

    let seq = match current {
        Some(bytes) if bytes.len() >= 8 => u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        _ => 1,
    };
    let next = seq.saturating_add(1);
    buf_guard.put(seq_key, next.to_le_bytes().to_vec())?;
    drop(buf_guard);
    Ok(seq)
}

/// Compute FNV-1a-64 of the given data for content prefix hashing.
///
/// Uses the standard FNV-1a algorithm (offset basis `0xcbf29ce484222325`,
/// prime `0x100000001b3`) which is deterministic and fast for short prefixes.
fn content_prefix_hash(prefix_data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for &b in prefix_data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    h
}

/// Derive a 128-bit integrity checksum from a SHA-256 digest.
///
/// Takes the first 16 bytes of the 32-byte SHA-256 digest and interprets them
/// as a little-endian u128. This provides full-content integrity coverage
/// (unlike the prefix-only FNV hash used for `content_prefix_hash`).
fn full_blob_checksum_from_sha256(sha256_digest: &[u8; 32]) -> u128 {
    let mut buf = [0u8; 16];
    buf.copy_from_slice(&sha256_digest[..16]);
    u128::from_le_bytes(buf)
}

/// Get current wall-clock nanoseconds.
fn wall_clock_ns() -> u64 {
    u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO)
            .as_nanos(),
    )
    .unwrap_or(u64::MAX)
}

// ---------------------------------------------------------------------------
// BfTreeBlobStore -- write-path blob operations (within a write transaction)
// ---------------------------------------------------------------------------

/// Blob store operations bound to a write transaction's buffer and adapter.
///
/// Provides streaming write, one-shot store, read, delete, and index queries
/// over `BfTree`'s chunked KV storage.
pub struct BfTreeBlobStore<'txn> {
    adapter: &'txn BfTreeAdapter,
    buffer: &'txn Mutex<WriteBuffer>,
    ops_count: &'txn AtomicU64,
}

impl<'txn> BfTreeBlobStore<'txn> {
    pub(crate) fn new(
        adapter: &'txn BfTreeAdapter,
        buffer: &'txn Mutex<WriteBuffer>,
        ops_count: &'txn AtomicU64,
    ) -> Self {
        Self {
            adapter,
            buffer,
            ops_count,
        }
    }

    /// Store a blob in one shot. Returns the assigned `BlobId`.
    ///
    /// This is the primary write API. For large blobs, use `begin_write()` for
    /// streaming writes.
    pub fn store(
        &self,
        data: &[u8],
        content_type: ContentType,
        label: &str,
        opts: StoreOptions,
    ) -> Result<BlobId, BfTreeError> {
        let mut writer = self.begin_write(content_type, label, opts)?;
        writer.write(data)?;
        writer.finish()
    }

    /// Begin a streaming blob write. Call `write()` one or more times, then `finish()`.
    pub fn begin_write(
        &self,
        content_type: ContentType,
        label: &str,
        opts: StoreOptions,
    ) -> Result<BfTreeBlobWriter<'_, 'txn>, BfTreeError> {
        let sequence = next_sequence(self.buffer, self.adapter)?;
        Ok(BfTreeBlobWriter {
            store: self,
            sequence,
            content_type,
            label: String::from(label),
            opts,
            chunks: Vec::new(),
            current_chunk: Vec::with_capacity(MAX_CHUNK_SIZE),
            prefix_buf: Vec::with_capacity(PREFIX_HASH_LEN),
            sha256_hasher: Sha256::new(),
            total_bytes: 0,
            finished: false,
        })
    }

    /// Read a blob by its ID. Returns the full data or `None` if not found.
    pub fn read(&self, blob_id: BlobId) -> Result<Option<Vec<u8>>, BfTreeError> {
        let meta = self.get_meta(blob_id)?;
        let Some(meta) = meta else {
            return Ok(None);
        };

        let total_len = usize::try_from(meta.blob_ref.length).unwrap_or(usize::MAX);
        let mut result = Vec::with_capacity(total_len);
        let num_chunks = total_len.div_ceil(MAX_CHUNK_SIZE);

        for chunk_idx in 0..num_chunks {
            let chunk_idx_u32 = u32::try_from(chunk_idx).unwrap_or(u32::MAX);
            let chunk_key = encode_chunk_key(blob_id, chunk_idx_u32);
            match read_raw_buffered(self.buffer, self.adapter, &chunk_key)? {
                Some(chunk_data) => result.extend_from_slice(&chunk_data),
                None => {
                    return Err(BfTreeError::Corruption(alloc::format!(
                        "missing chunk {chunk_idx} for blob {blob_id:?}"
                    )));
                }
            }
        }

        // Truncate to exact length (last chunk may be padded by BfTree).
        result.truncate(total_len);
        Ok(Some(result))
    }

    /// Read a range of bytes from a blob.
    pub fn read_range(
        &self,
        blob_id: BlobId,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, BfTreeError> {
        let meta = self.get_meta(blob_id)?.ok_or(BfTreeError::NotFound)?;
        let total_len = meta.blob_ref.length;

        if offset >= total_len {
            return Ok(Vec::new());
        }
        let actual_len = length.min(usize::try_from(total_len - offset).unwrap_or(usize::MAX));
        let mut result = Vec::with_capacity(actual_len);

        let chunk_size_u64 = MAX_CHUNK_SIZE as u64;
        let start_chunk = u32::try_from(offset / chunk_size_u64).unwrap_or(u32::MAX);
        let end_chunk = u32::try_from(
            (offset + u64::try_from(actual_len).unwrap_or(u64::MAX)).div_ceil(chunk_size_u64),
        )
        .unwrap_or(u32::MAX);

        let mut bytes_remaining = actual_len;
        let mut chunk_offset = usize::try_from(offset % chunk_size_u64).unwrap_or(0);

        for chunk_idx in start_chunk..end_chunk {
            let chunk_key = encode_chunk_key(blob_id, chunk_idx);
            let chunk_data =
                read_raw_buffered(self.buffer, self.adapter, &chunk_key)?.ok_or_else(|| {
                    BfTreeError::Corruption(alloc::format!(
                        "missing chunk {chunk_idx} for blob {blob_id:?}"
                    ))
                })?;

            let available = chunk_data.len().saturating_sub(chunk_offset);
            let to_copy = bytes_remaining.min(available);
            result.extend_from_slice(&chunk_data[chunk_offset..chunk_offset + to_copy]);
            bytes_remaining -= to_copy;
            chunk_offset = 0; // Only first chunk has an offset
        }

        Ok(result)
    }

    /// Get blob metadata by ID.
    pub fn get_meta(&self, blob_id: BlobId) -> Result<Option<BlobMeta>, BfTreeError> {
        let key = encode_meta_key(blob_id);
        match read_raw_buffered(self.buffer, self.adapter, &key)? {
            Some(bytes) if bytes.len() >= BlobMeta::SERIALIZED_SIZE => {
                let mut arr = [0u8; BlobMeta::SERIALIZED_SIZE];
                arr.copy_from_slice(&bytes[..BlobMeta::SERIALIZED_SIZE]);
                Ok(Some(BlobMeta::from_le_bytes(arr)))
            }
            _ => Ok(None),
        }
    }

    /// Delete a blob and all its index entries.
    ///
    /// Also decrements the dedup ref-count for blobs that are dedup-eligible.
    /// If the ref-count reaches zero, the dedup entry is removed.
    pub fn delete(&self, blob_id: BlobId) -> Result<bool, BfTreeError> {
        let Some(meta) = self.get_meta(blob_id)? else {
            return Ok(false);
        };

        // Delete data chunks. Zero-length blobs have 0 chunks (no data records).
        let total_len = usize::try_from(meta.blob_ref.length).unwrap_or(usize::MAX);
        let num_chunks = if total_len == 0 {
            0
        } else {
            total_len.div_ceil(MAX_CHUNK_SIZE)
        };

        // Use the SHA-256 hash stored in BlobMeta at write time for dedup lookup.
        // This avoids recomputing from chunks, which could silently produce a
        // wrong digest if any chunk is missing.
        let dedup_sha256 = if total_len >= DEDUP_MIN_SIZE && meta.sha256 != [0u8; 32] {
            Some(Sha256Key(meta.sha256))
        } else {
            None
        };

        {
            let mut buf = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
            for chunk_idx in 0..num_chunks {
                let chunk_key =
                    encode_chunk_key(blob_id, u32::try_from(chunk_idx).unwrap_or(u32::MAX));
                buf.delete(chunk_key);
            }

            // Delete metadata.
            buf.delete(encode_meta_key(blob_id));

            // Delete temporal index entry.
            let tk = TemporalKey {
                wall_clock_ns: meta.wall_clock_ns,
                hlc: HybridLogicalClock::from_raw(meta.hlc),
                blob_id,
            };
            buf.delete(encode_temporal_key(&tk));

            // Decrement dedup ref-count if applicable.
            if let Some(ref sha256) = dedup_sha256 {
                let dedup_key = encode_dedup_key(sha256);
                if let super::buffered_txn::BufferLookup::Found(bytes) = buf.get(&dedup_key) {
                    Self::apply_dedup_decrement(&mut buf, &dedup_key, &bytes)?;
                } else {
                    // Check BfTree directly (already inside buf lock, safe to read BfTree).
                    if let Ok(Some(bytes)) = read_raw(self.adapter, &dedup_key) {
                        Self::apply_dedup_decrement(&mut buf, &dedup_key, &bytes)?;
                    }
                }
            }

            // Delete tag entries -- scan tag table for this blob_id.
            self.delete_tags_for_blob(&mut buf, blob_id);

            // Delete namespace entries.
            self.delete_ns_for_blob(&mut buf, blob_id);

            // Delete causal edges where this blob is the parent.
            self.delete_causal_edges_for_parent(&mut buf, blob_id);
        }

        self.ops_count
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(true)
    }

    /// Decrement a dedup entry's ref-count, removing it entirely if it reaches zero.
    fn apply_dedup_decrement(
        buf: &mut WriteBuffer,
        dedup_key: &[u8],
        bytes: &[u8],
    ) -> Result<(), BfTreeError> {
        if bytes.len() >= DedupVal::SERIALIZED_SIZE + BlobId::SERIALIZED_SIZE {
            let mut arr = [0u8; DedupVal::SERIALIZED_SIZE];
            arr.copy_from_slice(&bytes[..DedupVal::SERIALIZED_SIZE]);
            let mut dedup = DedupVal::from_le_bytes(arr);

            if dedup.ref_count <= 1 {
                // Last reference -- remove the dedup entry entirely.
                buf.delete(dedup_key.to_vec());
            } else {
                dedup.ref_count -= 1;
                let blob_id_bytes = &bytes[DedupVal::SERIALIZED_SIZE
                    ..DedupVal::SERIALIZED_SIZE + BlobId::SERIALIZED_SIZE];
                let mut val =
                    Vec::with_capacity(DedupVal::SERIALIZED_SIZE + BlobId::SERIALIZED_SIZE);
                val.extend_from_slice(&dedup.to_le_bytes());
                val.extend_from_slice(blob_id_bytes);
                buf.put(dedup_key.to_vec(), val)?;
            }
        }
        Ok(())
    }

    /// List all blobs in temporal order (newest first).
    ///
    /// Buffer-aware: sees blobs stored in the current transaction before commit.
    pub fn list_temporal(&self, limit: usize) -> Result<Vec<(BlobId, BlobMeta)>, BfTreeError> {
        let prefix = table_prefix(BLOB_META_TABLE, TableKind::Regular);
        let prefix_end = table_prefix_end(BLOB_META_TABLE, TableKind::Regular);
        let prefix_len = prefix.len();

        let entries =
            scan_range_buffered(self.buffer, self.adapter, &prefix, &prefix_end, prefix_len)?;

        let mut results = Vec::new();
        for (key_bytes, val_bytes) in &entries {
            if results.len() >= limit {
                break;
            }
            if key_bytes.len() >= BlobId::SERIALIZED_SIZE
                && val_bytes.len() >= BlobMeta::SERIALIZED_SIZE
            {
                let blob_id =
                    BlobId::from_be_bytes(key_bytes[..BlobId::SERIALIZED_SIZE].try_into().unwrap());
                let mut meta_arr = [0u8; BlobMeta::SERIALIZED_SIZE];
                meta_arr.copy_from_slice(&val_bytes[..BlobMeta::SERIALIZED_SIZE]);
                let meta = BlobMeta::from_le_bytes(meta_arr);
                results.push((blob_id, meta));
            }
        }

        results.sort_by(|a, b| b.1.wall_clock_ns.cmp(&a.1.wall_clock_ns));
        if results.len() > limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    /// Query blobs by tag. Returns matching blob IDs.
    ///
    /// Buffer-aware: sees tags written in the current transaction before commit.
    pub fn query_by_tag(&self, tag: &str) -> Result<Vec<BlobId>, BfTreeError> {
        let start = TagKey::range_start(tag);
        let end = TagKey::range_end(tag);
        let start_encoded = encode_tag_key(&start);
        let end_encoded = encode_tag_key(&end);
        let prefix_len = table_prefix(BLOB_TAG_TABLE, TableKind::Regular).len();

        let entries = scan_range_buffered(
            self.buffer,
            self.adapter,
            &start_encoded,
            &end_encoded,
            prefix_len,
        )?;

        let mut results = Vec::new();
        for (key_bytes, _) in &entries {
            if key_bytes.len() >= TagKey::SERIALIZED_SIZE {
                let tk =
                    TagKey::from_be_bytes(key_bytes[..TagKey::SERIALIZED_SIZE].try_into().unwrap());
                results.push(tk.blob_id);
            }
        }
        Ok(results)
    }

    /// Query blobs by namespace. Returns matching blob IDs.
    ///
    /// Buffer-aware: sees namespace entries written in the current transaction.
    pub fn query_by_namespace(&self, namespace: &str) -> Result<Vec<BlobId>, BfTreeError> {
        let start = NamespaceKey::range_start(namespace);
        let end = NamespaceKey::range_end(namespace);
        let start_encoded = encode_ns_key(&start);
        let end_encoded = encode_ns_key(&end);
        let prefix_len = table_prefix(BLOB_NS_TABLE, TableKind::Regular).len();

        let entries = scan_range_buffered(
            self.buffer,
            self.adapter,
            &start_encoded,
            &end_encoded,
            prefix_len,
        )?;

        let mut results = Vec::new();
        for (key_bytes, _) in &entries {
            if key_bytes.len() >= NamespaceKey::SERIALIZED_SIZE {
                let nk = NamespaceKey::from_be_bytes(
                    key_bytes[..NamespaceKey::SERIALIZED_SIZE]
                        .try_into()
                        .unwrap(),
                );
                results.push(nk.blob_id);
            }
        }
        Ok(results)
    }

    /// Get children of a parent blob in the causal graph.
    ///
    /// Buffer-aware: sees causal edges written in the current transaction.
    pub fn causal_children(&self, parent: BlobId) -> Result<Vec<CausalEdge>, BfTreeError> {
        let start = CausalEdgeKey::new(parent, BlobId::MIN);
        let end = CausalEdgeKey::new(parent, BlobId::MAX);
        let start_encoded = encode_causal_key(&start);
        let end_encoded = encode_causal_key(&end);
        let prefix_len = table_prefix(BLOB_CAUSAL_TABLE, TableKind::Regular).len();

        let entries = scan_range_buffered(
            self.buffer,
            self.adapter,
            &start_encoded,
            &end_encoded,
            prefix_len,
        )?;

        let mut results = Vec::new();
        for (_, val_bytes) in &entries {
            if val_bytes.len() >= CausalEdge::SERIALIZED_SIZE {
                let mut arr = [0u8; CausalEdge::SERIALIZED_SIZE];
                arr.copy_from_slice(&val_bytes[..CausalEdge::SERIALIZED_SIZE]);
                results.push(CausalEdge::from_le_bytes(arr));
            }
        }
        Ok(results)
    }

    /// BFS traversal of the causal graph starting from `root`.
    ///
    /// Returns reachable blob IDs and their edges, bounded by both `max_depth`
    /// and `max_results` to prevent unbounded memory growth on large graphs.
    /// Traversal stops as soon as either limit is reached.
    pub fn causal_descendants(
        &self,
        root: BlobId,
        max_depth: usize,
        max_results: usize,
    ) -> Result<Vec<(BlobId, Option<CausalEdge>)>, BfTreeError> {
        let mut visited = alloc::collections::BTreeSet::new();
        let mut queue: VecDeque<(BlobId, usize)> = VecDeque::new();
        let mut result = Vec::new();

        visited.insert(root);
        queue.push_back((root, 0));
        result.push((root, None));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth || result.len() >= max_results {
                break;
            }
            let children = self.causal_children(current)?;
            for edge in children {
                if result.len() >= max_results {
                    break;
                }
                if visited.insert(edge.child) {
                    queue.push_back((edge.child, depth + 1));
                    result.push((edge.child, Some(edge)));
                }
            }
        }
        Ok(result)
    }

    /// Check SHA-256 dedup: returns existing `BlobId` if content already stored.
    ///
    /// This does NOT increment the ref-count. The caller is responsible for calling
    /// `increment_dedup_ref_count()` only after the dedup operation fully succeeds
    /// (e.g., after `write_dedup_chunk_refs()` completes).
    pub fn check_dedup(&self, sha256: &Sha256Key) -> Result<Option<BlobId>, BfTreeError> {
        let key = encode_dedup_key(sha256);
        match read_raw_buffered(self.buffer, self.adapter, &key)? {
            Some(bytes) if bytes.len() >= DedupVal::SERIALIZED_SIZE => {
                if bytes.len() >= DedupVal::SERIALIZED_SIZE + BlobId::SERIALIZED_SIZE {
                    let blob_id_bytes = &bytes[DedupVal::SERIALIZED_SIZE
                        ..DedupVal::SERIALIZED_SIZE + BlobId::SERIALIZED_SIZE];
                    let blob_id = BlobId::from_be_bytes(blob_id_bytes.try_into().unwrap());
                    Ok(Some(blob_id))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Increment the dedup ref-count for a given SHA-256 key.
    ///
    /// Called only after a dedup operation (chunk ref copy) has fully succeeded,
    /// ensuring the ref-count is never incremented for a failed or partial dedup.
    fn increment_dedup_ref_count(&self, sha256: &Sha256Key) -> Result<(), BfTreeError> {
        let key = encode_dedup_key(sha256);
        match read_raw_buffered(self.buffer, self.adapter, &key)? {
            Some(bytes) if bytes.len() >= DedupVal::SERIALIZED_SIZE + BlobId::SERIALIZED_SIZE => {
                let mut arr = [0u8; DedupVal::SERIALIZED_SIZE];
                arr.copy_from_slice(&bytes[..DedupVal::SERIALIZED_SIZE]);
                let mut dedup = DedupVal::from_le_bytes(arr);
                dedup.ref_count = dedup.ref_count.saturating_add(1);

                let blob_id_bytes = bytes[DedupVal::SERIALIZED_SIZE
                    ..DedupVal::SERIALIZED_SIZE + BlobId::SERIALIZED_SIZE]
                    .to_vec();

                let mut val =
                    Vec::with_capacity(DedupVal::SERIALIZED_SIZE + BlobId::SERIALIZED_SIZE);
                val.extend_from_slice(&dedup.to_le_bytes());
                val.extend_from_slice(&blob_id_bytes);
                self.buffer
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .put(key, val)?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn delete_tags_for_blob(&self, buf: &mut WriteBuffer, blob_id: BlobId) {
        let prefix = table_prefix(BLOB_TAG_TABLE, TableKind::Regular);
        let prefix_end = table_prefix_end(BLOB_TAG_TABLE, TableKind::Regular);
        let prefix_len = prefix.len();

        // Scan committed BfTree entries.
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut scan_buf = vec![0u8; max_record * 3];
        let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();

        if let Ok(mut iter) = self.adapter.scan_range(&prefix, &prefix_end) {
            while let Some((key_len, val_len)) = iter.next(&mut scan_buf) {
                if key_len + val_len > scan_buf.len() {
                    continue;
                }
                if key_len <= prefix_len {
                    continue;
                }
                let key_bytes = &scan_buf[prefix_len..key_len];
                if key_bytes.len() >= TagKey::SERIALIZED_SIZE {
                    let tk = TagKey::from_be_bytes(
                        key_bytes[..TagKey::SERIALIZED_SIZE].try_into().unwrap(),
                    );
                    if tk.blob_id == blob_id {
                        keys_to_delete.push(scan_buf[..key_len].to_vec());
                    }
                }
            }
        }

        // Also scan the write buffer for tag entries added in this transaction.
        for (key, value) in buf.range(&prefix, &prefix_end) {
            if value.is_none() {
                continue; // Already a tombstone.
            }
            if key.len() > prefix_len {
                let key_bytes = &key[prefix_len..];
                if key_bytes.len() >= TagKey::SERIALIZED_SIZE {
                    let tk = TagKey::from_be_bytes(
                        key_bytes[..TagKey::SERIALIZED_SIZE].try_into().unwrap(),
                    );
                    if tk.blob_id == blob_id && !keys_to_delete.contains(key) {
                        keys_to_delete.push(key.clone());
                    }
                }
            }
        }

        for key in keys_to_delete {
            buf.delete(key);
        }
    }

    fn delete_ns_for_blob(&self, buf: &mut WriteBuffer, blob_id: BlobId) {
        let prefix = table_prefix(BLOB_NS_TABLE, TableKind::Regular);
        let prefix_end = table_prefix_end(BLOB_NS_TABLE, TableKind::Regular);
        let prefix_len = prefix.len();

        // Scan committed BfTree entries.
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut scan_buf = vec![0u8; max_record * 3];
        let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();

        if let Ok(mut iter) = self.adapter.scan_range(&prefix, &prefix_end) {
            while let Some((key_len, val_len)) = iter.next(&mut scan_buf) {
                if key_len + val_len > scan_buf.len() {
                    continue;
                }
                if key_len <= prefix_len {
                    continue;
                }
                let key_bytes = &scan_buf[prefix_len..key_len];
                if key_bytes.len() >= NamespaceKey::SERIALIZED_SIZE {
                    let nk = NamespaceKey::from_be_bytes(
                        key_bytes[..NamespaceKey::SERIALIZED_SIZE]
                            .try_into()
                            .unwrap(),
                    );
                    if nk.blob_id == blob_id {
                        keys_to_delete.push(scan_buf[..key_len].to_vec());
                    }
                }
            }
        }

        // Also scan the write buffer for namespace entries added in this transaction.
        for (key, value) in buf.range(&prefix, &prefix_end) {
            if value.is_none() {
                continue; // Already a tombstone.
            }
            if key.len() > prefix_len {
                let key_bytes = &key[prefix_len..];
                if key_bytes.len() >= NamespaceKey::SERIALIZED_SIZE {
                    let nk = NamespaceKey::from_be_bytes(
                        key_bytes[..NamespaceKey::SERIALIZED_SIZE]
                            .try_into()
                            .unwrap(),
                    );
                    if nk.blob_id == blob_id && !keys_to_delete.contains(key) {
                        keys_to_delete.push(key.clone());
                    }
                }
            }
        }

        for key in keys_to_delete {
            buf.delete(key);
        }
    }

    fn delete_causal_edges_for_parent(&self, buf: &mut WriteBuffer, parent: BlobId) {
        let start = CausalEdgeKey::new(parent, BlobId::MIN);
        let end = CausalEdgeKey::new(parent, BlobId::MAX);
        let start_encoded = encode_causal_key(&start);
        let end_encoded = encode_causal_key(&end);
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut scan_buf = vec![0u8; max_record * 2];

        let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();

        // Scan committed BfTree entries.
        if let Ok(mut iter) = self.adapter.scan_range(&start_encoded, &end_encoded) {
            while let Some((key_len, _)) = iter.next(&mut scan_buf) {
                keys_to_delete.push(scan_buf[..key_len].to_vec());
            }
        }

        // Also scan the write buffer for causal edges added in this transaction.
        for (key, value) in buf.range(&start_encoded, &end_encoded) {
            if value.is_none() {
                continue; // Already a tombstone.
            }
            if !keys_to_delete.contains(key) {
                keys_to_delete.push(key.clone());
            }
        }

        for key in keys_to_delete {
            buf.delete(key);
        }
    }
}

// ---------------------------------------------------------------------------
// BfTreeBlobWriter -- streaming chunked writer
// ---------------------------------------------------------------------------

/// Streaming blob writer that chunks data into `BfTree` KV records.
///
/// Call `write()` one or more times to feed data, then `finish()` to commit
/// the blob and its metadata/indices.
pub struct BfTreeBlobWriter<'a, 'txn> {
    store: &'a BfTreeBlobStore<'txn>,
    sequence: u64,
    content_type: ContentType,
    label: String,
    opts: StoreOptions,
    /// Completed chunks waiting to be flushed.
    chunks: Vec<Vec<u8>>,
    /// Current chunk being accumulated.
    current_chunk: Vec<u8>,
    /// First `PREFIX_HASH_LEN` bytes for `content_prefix_hash`.
    prefix_buf: Vec<u8>,
    /// Incremental SHA-256 for dedup.
    sha256_hasher: Sha256,
    /// Total bytes written so far.
    total_bytes: u64,
    /// Whether `finish()` has been called.
    finished: bool,
}

impl BfTreeBlobWriter<'_, '_> {
    /// Write data to the blob. Can be called multiple times for streaming writes.
    pub fn write(&mut self, data: &[u8]) -> Result<(), BfTreeError> {
        if self.finished {
            return Err(BfTreeError::InvalidOperation("write after finish".into()));
        }

        // Feed prefix buffer.
        if self.prefix_buf.len() < PREFIX_HASH_LEN {
            let remaining = PREFIX_HASH_LEN - self.prefix_buf.len();
            let to_copy = data.len().min(remaining);
            self.prefix_buf.extend_from_slice(&data[..to_copy]);
        }

        // Feed SHA-256.
        self.sha256_hasher.update(data);

        // Chunk the data.
        let mut offset = 0;
        while offset < data.len() {
            let space = MAX_CHUNK_SIZE - self.current_chunk.len();
            let to_copy = (data.len() - offset).min(space);
            self.current_chunk
                .extend_from_slice(&data[offset..offset + to_copy]);
            offset += to_copy;

            if self.current_chunk.len() >= MAX_CHUNK_SIZE {
                let full_chunk =
                    core::mem::replace(&mut self.current_chunk, Vec::with_capacity(MAX_CHUNK_SIZE));
                self.chunks.push(full_chunk);
            }
        }

        self.total_bytes += data.len() as u64;
        Ok(())
    }

    /// Total bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.total_bytes
    }

    /// Finalize the blob: write chunks, metadata, and index entries.
    /// Returns the assigned `BlobId`.
    pub fn finish(mut self) -> Result<BlobId, BfTreeError> {
        self.finished = true;

        // Flush remaining chunk.
        if !self.current_chunk.is_empty() {
            let last_chunk = core::mem::take(&mut self.current_chunk);
            self.chunks.push(last_chunk);
        }

        // Compute hashes.
        let prefix_hash = content_prefix_hash(&self.prefix_buf);
        let blob_id = BlobId::new(self.sequence, prefix_hash);

        // Finalize SHA-256.
        let sha256_digest: [u8; 32] = self.sha256_hasher.clone().finalize().into();
        let sha256_key = Sha256Key(sha256_digest);

        // Derive full-content checksum from SHA-256 digest (covers all bytes, not just prefix).
        let checksum = full_blob_checksum_from_sha256(&sha256_digest);

        // Check dedup before writing chunks.
        if usize::try_from(self.total_bytes).unwrap_or(usize::MAX) >= DEDUP_MIN_SIZE
            && let Some(existing_id) = self.store.check_dedup(&sha256_key)?
        {
            // Content already exists -- create metadata pointing to existing data.
            // We still create a new BlobId with its own metadata for tracking,
            // but we can skip writing the actual data chunks.
            let wall_ns = wall_clock_ns();
            let hlc = HybridLogicalClock::now();
            let causal_parent = self.opts.causal_link.as_ref().map(|l| l.parent);

            let blob_ref = BlobRef {
                offset: 0, // Not used for BfTree backend
                length: self.total_bytes,
                checksum,
                ref_count: 1,
                content_type: self.content_type.as_byte(),
                compression: 0,
            };
            let meta = BlobMeta::with_sha256(
                blob_ref,
                wall_ns,
                hlc.to_raw(),
                causal_parent,
                &self.label,
                sha256_digest,
            );

            // Write metadata only (data is shared via dedup).
            let mut buf = self.store.buffer.lock().unwrap_or_else(|e| e.into_inner());
            buf.put(encode_meta_key(blob_id), meta.to_le_bytes().to_vec())?;

            // Copy chunk references from existing blob.
            drop(buf);
            self.write_dedup_chunk_refs(blob_id, existing_id)?;

            // Increment dedup ref-count only after chunk refs are successfully copied.
            self.store.increment_dedup_ref_count(&sha256_key)?;

            self.write_indices(blob_id, &meta, &sha256_key)?;

            self.store
                .ops_count
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            return Ok(blob_id);
        }

        // Write all chunks to the buffer.
        {
            let mut buf = self.store.buffer.lock().unwrap_or_else(|e| e.into_inner());
            for (idx, chunk) in self.chunks.iter().enumerate() {
                let chunk_key = encode_chunk_key(blob_id, u32::try_from(idx).unwrap_or(u32::MAX));
                buf.put(chunk_key, chunk.clone())?;
            }
        }

        // Write metadata.
        let wall_ns = wall_clock_ns();
        let hlc = HybridLogicalClock::now();
        let causal_parent = self.opts.causal_link.as_ref().map(|l| l.parent);

        let blob_ref = BlobRef {
            offset: 0,
            length: self.total_bytes,
            checksum,
            ref_count: 1,
            content_type: self.content_type.as_byte(),
            compression: 0,
        };
        let meta = BlobMeta::with_sha256(
            blob_ref,
            wall_ns,
            hlc.to_raw(),
            causal_parent,
            &self.label,
            sha256_digest,
        );

        {
            let mut buf = self.store.buffer.lock().unwrap_or_else(|e| e.into_inner());
            buf.put(encode_meta_key(blob_id), meta.to_le_bytes().to_vec())?;
        }

        // Write dedup entry.
        if usize::try_from(self.total_bytes).unwrap_or(usize::MAX) >= DEDUP_MIN_SIZE {
            let dedup_val = DedupVal {
                offset: 0,
                length: self.total_bytes,
                checksum,
                ref_count: 1,
            };
            let mut val = Vec::with_capacity(DedupVal::SERIALIZED_SIZE + BlobId::SERIALIZED_SIZE);
            val.extend_from_slice(&dedup_val.to_le_bytes());
            val.extend_from_slice(&blob_id.to_be_bytes());
            self.store
                .buffer
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .put(encode_dedup_key(&sha256_key), val)?;
        }

        // Write index entries.
        self.write_indices(blob_id, &meta, &sha256_key)?;

        self.store
            .ops_count
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(blob_id)
    }

    /// Copy chunk references from an existing deduped blob.
    ///
    /// Reads from the write buffer first (via `BufferLookup`), falling back to
    /// `BfTree` for committed data. The buffer lock is held for the entire
    /// read-then-write operation to prevent TOCTOU races where another thread
    /// could delete the source chunks between reading and writing.
    fn write_dedup_chunk_refs(
        &self,
        new_id: BlobId,
        existing_id: BlobId,
    ) -> Result<(), BfTreeError> {
        use super::buffered_txn::BufferLookup;

        let meta = self.store.get_meta(existing_id)?;
        if let Some(m) = meta {
            let total = usize::try_from(m.blob_ref.length).unwrap_or(usize::MAX);
            let num_chunks = total.div_ceil(MAX_CHUNK_SIZE);

            // Hold the buffer lock for the entire read + write cycle to
            // eliminate the TOCTOU gap between reading source chunks and
            // writing destination refs.
            let mut buf_guard = self.store.buffer.lock().unwrap_or_else(|e| e.into_inner());
            for idx in 0..num_chunks {
                let idx_u32 = u32::try_from(idx).unwrap_or(u32::MAX);
                let src_key = encode_chunk_key(existing_id, idx_u32);
                let chunk_bytes = match buf_guard.get(&src_key) {
                    BufferLookup::Found(v) => Some(v),
                    BufferLookup::Tombstone => None,
                    BufferLookup::NotInBuffer => {
                        // Read from BfTree while holding the lock. BfTree reads
                        // are lock-free internally so this does not deadlock.
                        read_raw(self.store.adapter, &src_key)?
                    }
                };
                if let Some(bytes) = chunk_bytes {
                    let dst_key = encode_chunk_key(new_id, idx_u32);
                    buf_guard.put(dst_key, bytes)?;
                }
            }
        }
        Ok(())
    }

    /// Write temporal, causal, tag, and namespace index entries.
    fn write_indices(
        &self,
        blob_id: BlobId,
        meta: &BlobMeta,
        _sha256_key: &Sha256Key,
    ) -> Result<(), BfTreeError> {
        let mut buf = self.store.buffer.lock().unwrap_or_else(|e| e.into_inner());

        // Temporal index.
        let tk = TemporalKey {
            wall_clock_ns: meta.wall_clock_ns,
            hlc: HybridLogicalClock::from_raw(meta.hlc),
            blob_id,
        };
        buf.put(encode_temporal_key(&tk), alloc::vec![0u8])?;

        // Causal edge.
        if let Some(ref link) = self.opts.causal_link {
            let edge = CausalEdge::new(blob_id, link.relation, &link.context);
            let edge_key = CausalEdgeKey::new(link.parent, blob_id);
            buf.put(encode_causal_key(&edge_key), edge.to_le_bytes().to_vec())?;
        }

        // Tags.
        let tag_count = self.opts.tags.len().min(MAX_TAGS);
        for tag in self.opts.tags.iter().take(tag_count) {
            let tk = TagKey::new(tag, blob_id);
            buf.put(encode_tag_key(&tk), alloc::vec![0u8])?;
        }

        // Namespace.
        if let Some(ref ns) = self.opts.namespace {
            let nk = NamespaceKey::new(ns, blob_id);
            buf.put(encode_ns_key(&nk), alloc::vec![0u8])?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BfTreeBlobReader -- read-only blob access
// ---------------------------------------------------------------------------

/// Read-only blob store bound to a read transaction's adapter.
pub struct BfTreeReadOnlyBlobStore<'txn> {
    adapter: &'txn BfTreeAdapter,
}

impl<'txn> BfTreeReadOnlyBlobStore<'txn> {
    pub(crate) fn new(adapter: &'txn BfTreeAdapter) -> Self {
        Self { adapter }
    }

    /// Read a full blob by its ID.
    pub fn read(&self, blob_id: BlobId) -> Result<Option<Vec<u8>>, BfTreeError> {
        let meta = self.get_meta(blob_id)?;
        let Some(meta) = meta else {
            return Ok(None);
        };

        let total_len = usize::try_from(meta.blob_ref.length).unwrap_or(usize::MAX);
        let mut result = Vec::with_capacity(total_len);
        let num_chunks = total_len.div_ceil(MAX_CHUNK_SIZE);

        for chunk_idx in 0..num_chunks {
            let chunk_idx_u32 = u32::try_from(chunk_idx).unwrap_or(u32::MAX);
            let chunk_key = encode_chunk_key(blob_id, chunk_idx_u32);
            match read_raw(self.adapter, &chunk_key)? {
                Some(chunk_data) => result.extend_from_slice(&chunk_data),
                None => {
                    return Err(BfTreeError::Corruption(alloc::format!(
                        "missing chunk {chunk_idx} for blob {blob_id:?}"
                    )));
                }
            }
        }

        result.truncate(total_len);
        Ok(Some(result))
    }

    /// Read a range from a blob.
    pub fn read_range(
        &self,
        blob_id: BlobId,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, BfTreeError> {
        let meta = self.get_meta(blob_id)?.ok_or(BfTreeError::NotFound)?;
        let total_len = meta.blob_ref.length;

        if offset >= total_len {
            return Ok(Vec::new());
        }
        let actual_len = length.min(usize::try_from(total_len - offset).unwrap_or(usize::MAX));
        let mut result = Vec::with_capacity(actual_len);

        let chunk_size_u64 = MAX_CHUNK_SIZE as u64;
        let start_chunk = u32::try_from(offset / chunk_size_u64).unwrap_or(u32::MAX);
        let end_chunk = u32::try_from(
            (offset + u64::try_from(actual_len).unwrap_or(u64::MAX)).div_ceil(chunk_size_u64),
        )
        .unwrap_or(u32::MAX);

        let mut bytes_remaining = actual_len;
        let mut chunk_offset = usize::try_from(offset % chunk_size_u64).unwrap_or(0);

        for chunk_idx in start_chunk..end_chunk {
            let chunk_key = encode_chunk_key(blob_id, chunk_idx);
            let chunk_data = read_raw(self.adapter, &chunk_key)?.ok_or_else(|| {
                BfTreeError::Corruption(alloc::format!(
                    "missing chunk {chunk_idx} for blob {blob_id:?}"
                ))
            })?;

            let available = chunk_data.len().saturating_sub(chunk_offset);
            let to_copy = bytes_remaining.min(available);
            result.extend_from_slice(&chunk_data[chunk_offset..chunk_offset + to_copy]);
            bytes_remaining -= to_copy;
            chunk_offset = 0;
        }

        Ok(result)
    }

    /// Get blob metadata by ID.
    pub fn get_meta(&self, blob_id: BlobId) -> Result<Option<BlobMeta>, BfTreeError> {
        let key = encode_meta_key(blob_id);
        match read_raw(self.adapter, &key)? {
            Some(bytes) if bytes.len() >= BlobMeta::SERIALIZED_SIZE => {
                let mut arr = [0u8; BlobMeta::SERIALIZED_SIZE];
                arr.copy_from_slice(&bytes[..BlobMeta::SERIALIZED_SIZE]);
                Ok(Some(BlobMeta::from_le_bytes(arr)))
            }
            _ => Ok(None),
        }
    }

    /// Query blobs by tag.
    pub fn query_by_tag(&self, tag: &str) -> Result<Vec<BlobId>, BfTreeError> {
        let start = TagKey::range_start(tag);
        let end = TagKey::range_end(tag);
        let start_encoded = encode_tag_key(&start);
        let end_encoded = encode_tag_key(&end);
        let prefix_len = table_prefix(BLOB_TAG_TABLE, TableKind::Regular).len();
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut scan_buf = vec![0u8; max_record * 2];
        let mut iter = self.adapter.scan_range(&start_encoded, &end_encoded)?;
        let mut results = Vec::new();

        while let Some((key_len, _)) = iter.next(&mut scan_buf) {
            if key_len <= prefix_len {
                continue;
            }
            let key_bytes = &scan_buf[prefix_len..key_len];
            if key_bytes.len() >= TagKey::SERIALIZED_SIZE {
                let tk =
                    TagKey::from_be_bytes(key_bytes[..TagKey::SERIALIZED_SIZE].try_into().unwrap());
                results.push(tk.blob_id);
            }
        }
        Ok(results)
    }

    /// Get causal children of a parent blob.
    pub fn causal_children(&self, parent: BlobId) -> Result<Vec<CausalEdge>, BfTreeError> {
        let start = CausalEdgeKey::new(parent, BlobId::MIN);
        let end = CausalEdgeKey::new(parent, BlobId::MAX);
        let start_encoded = encode_causal_key(&start);
        let end_encoded = encode_causal_key(&end);
        let prefix_len = table_prefix(BLOB_CAUSAL_TABLE, TableKind::Regular).len();
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut scan_buf = vec![0u8; max_record * 2];
        let mut iter = self.adapter.scan_range(&start_encoded, &end_encoded)?;
        let mut results = Vec::new();

        while let Some((key_len, val_len)) = iter.next(&mut scan_buf) {
            if key_len <= prefix_len {
                continue;
            }
            let val_bytes = &scan_buf[key_len..key_len + val_len];
            if val_bytes.len() >= CausalEdge::SERIALIZED_SIZE {
                let mut arr = [0u8; CausalEdge::SERIALIZED_SIZE];
                arr.copy_from_slice(&val_bytes[..CausalEdge::SERIALIZED_SIZE]);
                results.push(CausalEdge::from_le_bytes(arr));
            }
        }
        Ok(results)
    }

    /// Look up a blob by its sequence number.
    ///
    /// `BlobId` encodes the sequence in its first 8 bytes. We scan the meta table
    /// for blobs whose `BlobId::sequence()` matches, returning the first hit.
    pub fn blob_by_sequence(&self, seq: u64) -> Result<Option<(BlobId, BlobMeta)>, BfTreeError> {
        // BlobId(seq, counter) -- scan from BlobId::new(seq, 0) to BlobId::new(seq, u32::MAX)
        let start_id = BlobId::new(seq, 0);
        let end_id = BlobId::new(seq, u64::MAX);
        let start_key = encode_meta_key(start_id);
        let end_key = encode_meta_key(end_id);
        let prefix_len = table_prefix(BLOB_META_TABLE, TableKind::Regular).len();
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut scan_buf = vec![0u8; max_record * 2];
        let mut iter = self.adapter.scan_range(&start_key, &end_key)?;

        while let Some((key_len, val_len)) = iter.next(&mut scan_buf) {
            if key_len <= prefix_len {
                continue;
            }
            let key_bytes = &scan_buf[prefix_len..key_len];
            let val_bytes = &scan_buf[key_len..key_len + val_len];
            if key_bytes.len() >= BlobId::SERIALIZED_SIZE
                && val_bytes.len() >= BlobMeta::SERIALIZED_SIZE
            {
                let mut id_arr = [0u8; BlobId::SERIALIZED_SIZE];
                id_arr.copy_from_slice(&key_bytes[..BlobId::SERIALIZED_SIZE]);
                let blob_id = BlobId::from_be_bytes(id_arr);
                let mut meta_arr = [0u8; BlobMeta::SERIALIZED_SIZE];
                meta_arr.copy_from_slice(&val_bytes[..BlobMeta::SERIALIZED_SIZE]);
                return Ok(Some((blob_id, BlobMeta::from_le_bytes(meta_arr))));
            }
        }
        Ok(None)
    }

    /// Query blobs in a temporal range [`start_ns`, `end_ns`].
    pub fn blobs_in_time_range(
        &self,
        start_ns: u64,
        end_ns: u64,
    ) -> Result<Vec<(TemporalKey, BlobMeta)>, BfTreeError> {
        if start_ns > end_ns {
            return Ok(Vec::new());
        }
        let start_tk = TemporalKey::range_start(start_ns);
        let end_tk = TemporalKey::range_end(end_ns);
        let start_encoded = encode_temporal_key(&start_tk);
        let end_encoded = encode_temporal_key(&end_tk);
        let prefix_len = table_prefix(BLOB_TEMPORAL_TABLE, TableKind::Regular).len();
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut scan_buf = vec![0u8; max_record * 2];
        let mut iter = self.adapter.scan_range(&start_encoded, &end_encoded)?;
        let mut results = Vec::new();

        while let Some((key_len, _)) = iter.next(&mut scan_buf) {
            if key_len <= prefix_len {
                continue;
            }
            let key_bytes = &scan_buf[prefix_len..key_len];
            if key_bytes.len() >= TemporalKey::SERIALIZED_SIZE {
                let tk = TemporalKey::from_be_bytes(
                    key_bytes[..TemporalKey::SERIALIZED_SIZE]
                        .try_into()
                        .unwrap(),
                );
                if let Ok(Some(meta)) = self.get_meta(tk.blob_id) {
                    results.push((tk, meta));
                }
            }
        }
        Ok(results)
    }

    /// Query blobs in a namespace, returning (`BlobId`, `BlobMeta`) pairs.
    pub fn blobs_in_namespace(
        &self,
        namespace: &str,
    ) -> Result<Vec<(BlobId, BlobMeta)>, BfTreeError> {
        let start = NamespaceKey::range_start(namespace);
        let end = NamespaceKey::range_end(namespace);
        let start_encoded = encode_ns_key(&start);
        let end_encoded = encode_ns_key(&end);
        let prefix_len = table_prefix(BLOB_NS_TABLE, TableKind::Regular).len();
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut scan_buf = vec![0u8; max_record * 2];
        let mut iter = self.adapter.scan_range(&start_encoded, &end_encoded)?;
        let mut results = Vec::new();

        while let Some((key_len, _)) = iter.next(&mut scan_buf) {
            if key_len <= prefix_len {
                continue;
            }
            let key_bytes = &scan_buf[prefix_len..key_len];
            if key_bytes.len() >= NamespaceKey::SERIALIZED_SIZE {
                let nk = NamespaceKey::from_be_bytes(
                    key_bytes[..NamespaceKey::SERIALIZED_SIZE]
                        .try_into()
                        .unwrap(),
                );
                if let Ok(Some(meta)) = self.get_meta(nk.blob_id) {
                    results.push((nk.blob_id, meta));
                }
            }
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf_tree_store::config::BfTreeConfig;
    use crate::bf_tree_store::database::BfTreeDatabase;
    use crate::blob_store::types::{CausalLink, RelationType};

    fn test_db() -> BfTreeDatabase {
        BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap()
    }

    #[test]
    fn store_and_read_blob() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        let data = b"Hello, blob world!";
        let blob_id = blob_store
            .store(
                data,
                ContentType::OctetStream,
                "test-blob",
                StoreOptions::default(),
            )
            .unwrap();

        let read_data = blob_store.read(blob_id).unwrap().unwrap();
        assert_eq!(read_data, data);
        let _ = blob_store;
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let ro_store = rtxn.open_blob_store();
        let read_data = ro_store.read(blob_id).unwrap().unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn store_large_blob_multiple_chunks() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        #[allow(clippy::cast_possible_truncation)]
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 256) as u8).collect();
        let blob_id = blob_store
            .store(
                &data,
                ContentType::Embedding,
                "large-blob",
                StoreOptions::default(),
            )
            .unwrap();

        let read_data = blob_store.read(blob_id).unwrap().unwrap();
        assert_eq!(read_data.len(), data.len());
        assert_eq!(read_data, data);
        let _ = blob_store;
        wtxn.commit().unwrap();
    }

    #[test]
    fn streaming_write() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        let mut writer = blob_store
            .begin_write(ContentType::AudioWav, "streamed", StoreOptions::default())
            .unwrap();

        writer.write(b"chunk1-").unwrap();
        writer.write(b"chunk2-").unwrap();
        writer.write(b"chunk3").unwrap();
        assert_eq!(writer.bytes_written(), 20);

        let blob_id = writer.finish().unwrap();
        let read_data = blob_store.read(blob_id).unwrap().unwrap();
        assert_eq!(read_data, b"chunk1-chunk2-chunk3");
        let _ = blob_store;
        wtxn.commit().unwrap();
    }

    #[test]
    fn read_range() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        let data = b"0123456789abcdef";
        let blob_id = blob_store
            .store(
                data,
                ContentType::OctetStream,
                "range-test",
                StoreOptions::default(),
            )
            .unwrap();

        let range = blob_store.read_range(blob_id, 4, 8).unwrap();
        assert_eq!(range, b"456789ab");

        let range = blob_store.read_range(blob_id, 14, 100).unwrap();
        assert_eq!(range, b"ef");

        let range = blob_store.read_range(blob_id, 16, 10).unwrap();
        assert!(range.is_empty());
        let _ = blob_store;
        wtxn.commit().unwrap();
    }

    #[test]
    fn delete_blob() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        let blob_id = blob_store
            .store(
                b"delete-me",
                ContentType::OctetStream,
                "temp",
                StoreOptions::default(),
            )
            .unwrap();

        assert!(blob_store.read(blob_id).unwrap().is_some());
        let deleted = blob_store.delete(blob_id).unwrap();
        assert!(deleted);
        assert!(blob_store.read(blob_id).unwrap().is_none());
        assert!(!blob_store.delete(blob_id).unwrap());
        let _ = blob_store;
        wtxn.commit().unwrap();
    }

    #[test]
    fn blob_metadata() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        let blob_id = blob_store
            .store(
                b"meta-test-data",
                ContentType::ImagePng,
                "my-image",
                StoreOptions::default(),
            )
            .unwrap();

        let meta = blob_store.get_meta(blob_id).unwrap().unwrap();
        assert_eq!(meta.blob_ref.length, 14);
        assert_eq!(meta.blob_ref.content_type, ContentType::ImagePng.as_byte());
        assert_eq!(meta.label_str(), "my-image");
        assert!(meta.causal_parent.is_none());
        let _ = blob_store;
        wtxn.commit().unwrap();
    }

    #[test]
    fn tag_index() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        let opts = StoreOptions {
            tags: alloc::vec!["sensor".into(), "imu".into()],
            ..Default::default()
        };
        let blob_id = blob_store
            .store(b"sensor-data", ContentType::SensorImu, "imu-reading", opts)
            .unwrap();

        let _ = blob_store;
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let ro_store = rtxn.open_blob_store();
        let sensor_blobs = ro_store.query_by_tag("sensor").unwrap();
        assert!(sensor_blobs.contains(&blob_id));

        let imu_blobs = ro_store.query_by_tag("imu").unwrap();
        assert!(imu_blobs.contains(&blob_id));

        let empty = ro_store.query_by_tag("nonexistent").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn causal_graph() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        let parent_id = blob_store
            .store(
                b"parent",
                ContentType::OctetStream,
                "parent",
                StoreOptions::default(),
            )
            .unwrap();

        let opts = StoreOptions {
            causal_link: Some(CausalLink::new(
                parent_id,
                RelationType::Derived,
                "processed from parent",
            )),
            ..Default::default()
        };
        let child_id = blob_store
            .store(b"child", ContentType::OctetStream, "child", opts)
            .unwrap();

        let _ = blob_store;
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let ro_store = rtxn.open_blob_store();
        let children = ro_store.causal_children(parent_id).unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].child, child_id);
        assert_eq!(children[0].relation, RelationType::Derived);
        assert_eq!(children[0].context_str(), "processed from parent");
    }

    #[test]
    fn namespace_index() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        let opts = StoreOptions {
            namespace: Some("session-42".into()),
            ..Default::default()
        };
        let blob_id = blob_store
            .store(b"ns-data", ContentType::OctetStream, "ns-blob", opts)
            .unwrap();

        let _ = blob_store;
        wtxn.commit().unwrap();

        let wtxn2 = db.begin_write();
        let blob_store2 = wtxn2.open_blob_store();
        let ns_blobs = blob_store2.query_by_namespace("session-42").unwrap();
        assert!(ns_blobs.contains(&blob_id));

        let empty = blob_store2.query_by_namespace("other").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn list_temporal() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        let id1 = blob_store
            .store(
                b"first",
                ContentType::OctetStream,
                "b1",
                StoreOptions::default(),
            )
            .unwrap();
        let id2 = blob_store
            .store(
                b"second",
                ContentType::OctetStream,
                "b2",
                StoreOptions::default(),
            )
            .unwrap();

        let _ = blob_store;
        wtxn.commit().unwrap();

        let wtxn2 = db.begin_write();
        let blob_store2 = wtxn2.open_blob_store();
        let list = blob_store2.list_temporal(10).unwrap();
        assert_eq!(list.len(), 2);
        let ids: Vec<BlobId> = list.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn causal_descendants_bfs() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        let root = blob_store
            .store(
                b"root",
                ContentType::OctetStream,
                "root",
                StoreOptions::default(),
            )
            .unwrap();

        let child1 = blob_store
            .store(
                b"child1",
                ContentType::OctetStream,
                "c1",
                StoreOptions {
                    causal_link: Some(CausalLink::derived(root)),
                    ..Default::default()
                },
            )
            .unwrap();

        let _grandchild = blob_store
            .store(
                b"grandchild",
                ContentType::OctetStream,
                "gc",
                StoreOptions {
                    causal_link: Some(CausalLink::derived(child1)),
                    ..Default::default()
                },
            )
            .unwrap();

        let _ = blob_store;
        wtxn.commit().unwrap();

        let wtxn2 = db.begin_write();
        let blob_store2 = wtxn2.open_blob_store();
        let descendants = blob_store2.causal_descendants(root, 10, 1000).unwrap();
        assert_eq!(descendants.len(), 3);
        assert_eq!(descendants[0].0, root);
        assert!(descendants[0].1.is_none());
    }

    #[test]
    fn empty_blob() {
        let db = test_db();
        let wtxn = db.begin_write();
        let blob_store = wtxn.open_blob_store();

        let blob_id = blob_store
            .store(
                b"",
                ContentType::OctetStream,
                "empty",
                StoreOptions::default(),
            )
            .unwrap();

        let data = blob_store.read(blob_id).unwrap().unwrap();
        assert!(data.is_empty());

        let meta = blob_store.get_meta(blob_id).unwrap().unwrap();
        assert_eq!(meta.blob_ref.length, 0);
        let _ = blob_store;
        wtxn.commit().unwrap();
    }
}
