//! Contiguous cluster blob codec for IVF-PQ posting lists.
//!
//! Stores all vectors for a single IVF cluster as one contiguous byte blob,
//! enabling a single B-tree lookup per cluster instead of per-entry iteration.
//!
//! # Blob Layout
//!
//! ```text
//! Header (16 bytes):
//!   [0..4]   magic: 0x504F5354 ("POST") LE
//!   [4..8]   version: u32 LE = 1
//!   [8..12]  count: u32 LE = N (number of vectors)
//!   [12..14] pq_len: u16 LE = M (bytes per PQ code)
//!   [14..16] flags: u16 LE (bit 0 = has_raw_vectors)
//!
//! Vector IDs (N × 8 bytes):
//!   Sorted u64 LE array — enables binary search for upsert/delete
//!
//! PQ Codes (N × M bytes):
//!   Row-major contiguous block — hot path for ADC scan
//!
//! Raw Vectors (conditional, N × raw_dim bytes):
//!   Row-major f32 LE — only present if flags bit 0 set
//! ```

use alloc::vec::Vec;
use crate::error::StorageError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: u32 = 0x504F_5354; // "POST" in LE
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 16;
const FLAG_HAS_RAW_VECTORS: u16 = 1;

/// Entry with f32 raw vectors (for encoding from caller data).
type BlobEntryF32<'a> = (u64, &'a [u8], Option<&'a [f32]>);

/// Entry with pre-serialized raw vector bytes (for merge/remove from existing blobs).
type BlobEntryBytes<'a> = (u64, &'a [u8], Option<&'a [u8]>);

/// Owned entry for merge input (caller-provided new vectors).
type OwnedBlobEntry = (u64, Vec<u8>, Option<Vec<u8>>);

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Encode a cluster blob from entries sorted by `vector_id`.
///
/// Each entry is `(vector_id, pq_codes, optional_raw_vector)`.
/// `pq_len` is the expected PQ code length (`num_subvectors`).
///
/// If any entry has a raw vector, ALL entries must have one (the blob format
/// is uniform). The caller is responsible for enforcing this.
#[allow(clippy::cast_possible_truncation)]
pub fn encode_cluster_blob(
    entries: &[BlobEntryF32<'_>],
    pq_len: u16,
) -> Vec<u8> {
    let n = entries.len();
    let has_raw = !entries.is_empty() && entries[0].2.is_some();
    let raw_dim_bytes = if has_raw {
        entries[0].2.map_or(0, |r| r.len() * 4)
    } else {
        0
    };

    let total_size = HEADER_SIZE
        + n * 8                      // vector IDs
        + n * pq_len as usize       // PQ codes
        + if has_raw { n * raw_dim_bytes } else { 0 };

    let mut buf = Vec::with_capacity(total_size);

    // Header
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&pq_len.to_le_bytes());
    let flags: u16 = if has_raw { FLAG_HAS_RAW_VECTORS } else { 0 };
    buf.extend_from_slice(&flags.to_le_bytes());

    // Vector IDs (sorted)
    for &(vid, _, _) in entries {
        buf.extend_from_slice(&vid.to_le_bytes());
    }

    // PQ codes
    for &(_, pq, _) in entries {
        debug_assert_eq!(pq.len(), pq_len as usize);
        buf.extend_from_slice(pq);
    }

    // Raw vectors
    if has_raw {
        for &(_, _, raw_opt) in entries {
            if let Some(raw) = raw_opt {
                for &f in raw {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
        }
    }

    debug_assert_eq!(buf.len(), total_size);
    buf
}

/// Encode a cluster blob where raw vectors are already serialized as bytes.
///
/// Used by merge/remove helpers that work with blob-sourced raw data (already LE bytes).
#[allow(clippy::cast_possible_truncation)]
fn encode_cluster_blob_raw_bytes(
    entries: &[BlobEntryBytes<'_>],
    pq_len: u16,
) -> Vec<u8> {
    let n = entries.len();
    let has_raw = !entries.is_empty() && entries[0].2.is_some();
    let raw_vec_bytes = if has_raw {
        entries[0].2.map_or(0, |r| r.len())
    } else {
        0
    };

    let total_size = HEADER_SIZE
        + n * 8
        + n * pq_len as usize
        + if has_raw { n * raw_vec_bytes } else { 0 };

    let mut buf = Vec::with_capacity(total_size);

    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&pq_len.to_le_bytes());
    let flags: u16 = if has_raw { FLAG_HAS_RAW_VECTORS } else { 0 };
    buf.extend_from_slice(&flags.to_le_bytes());

    for &(vid, _, _) in entries {
        buf.extend_from_slice(&vid.to_le_bytes());
    }

    for &(_, pq, _) in entries {
        buf.extend_from_slice(pq);
    }

    if has_raw {
        for &(_, _, raw_opt) in entries {
            if let Some(raw) = raw_opt {
                buf.extend_from_slice(raw);
            }
        }
    }

    buf
}

// ---------------------------------------------------------------------------
// Zero-copy reader
// ---------------------------------------------------------------------------

/// Zero-copy view over a cluster blob.
///
/// All invariants are validated at construction time so that accessor methods
/// can use unchecked indexing in the hot path.
pub struct ClusterBlobRef<'a> {
    data: &'a [u8],
    count: u32,
    pq_len: u16,
    has_raw: bool,
    /// Offset where vector IDs begin.
    ids_offset: usize,
    /// Offset where PQ codes begin.
    pq_offset: usize,
    /// Offset where raw vectors begin (0 if none).
    raw_offset: usize,
    /// Byte length of one raw vector (dim * 4).
    raw_vec_bytes: usize,
}

impl<'a> ClusterBlobRef<'a> {
    /// Parse and validate a cluster blob.
    ///
    /// `expected_pq_len` must match the index's `num_subvectors`.
    /// `dim` is the vector dimensionality (needed to validate raw vector sizes).
    pub fn new(data: &'a [u8], expected_pq_len: u16, dim: usize) -> crate::Result<Self> {
        if data.len() < HEADER_SIZE {
            return Err(StorageError::Corrupted(alloc::format!(
                "cluster blob too small: {} < {HEADER_SIZE}",
                data.len()
            )));
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != MAGIC {
            return Err(StorageError::Corrupted(alloc::format!(
                "cluster blob bad magic: {magic:#010x}"
            )));
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != VERSION {
            return Err(StorageError::Corrupted(alloc::format!(
                "cluster blob version {version} != {VERSION}"
            )));
        }

        let count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let pq_len = u16::from_le_bytes([data[12], data[13]]);
        let flags = u16::from_le_bytes([data[14], data[15]]);
        let has_raw = flags & FLAG_HAS_RAW_VECTORS != 0;

        if pq_len != expected_pq_len {
            return Err(StorageError::Corrupted(alloc::format!(
                "cluster blob pq_len {pq_len} != expected {expected_pq_len}"
            )));
        }

        let n = count as usize;
        let ids_offset = HEADER_SIZE;
        let pq_offset = ids_offset + n * 8;
        let raw_offset = pq_offset + n * pq_len as usize;
        let raw_vec_bytes = if has_raw { dim * 4 } else { 0 };

        let expected_len = raw_offset + if has_raw { n * raw_vec_bytes } else { 0 };
        if data.len() < expected_len {
            return Err(StorageError::Corrupted(alloc::format!(
                "cluster blob truncated: {} < {expected_len} (count={count}, pq_len={pq_len}, has_raw={has_raw})",
                data.len()
            )));
        }

        Ok(Self {
            data,
            count,
            pq_len,
            has_raw,
            ids_offset,
            pq_offset,
            raw_offset,
            raw_vec_bytes,
        })
    }

    /// Number of vectors in this cluster.
    #[inline]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// PQ code length (`num_subvectors`).
    #[inline]
    pub fn pq_len(&self) -> u16 {
        self.pq_len
    }

    /// Whether the blob contains raw vectors.
    #[inline]
    pub fn has_raw_vectors(&self) -> bool {
        self.has_raw
    }

    /// Get the vector ID at position `i`.
    ///
    /// # Safety invariant
    /// `i < self.count` is the caller's responsibility. Validated at blob
    /// construction time via size checks.
    #[inline]
    pub fn vector_id(&self, i: u32) -> u64 {
        let offset = self.ids_offset + i as usize * 8;
        // SAFETY: blob size validated in new() to contain count * 8 bytes of IDs.
        debug_assert!(offset + 8 <= self.data.len());
        u64::from_le_bytes(unsafe {
            [
                *self.data.get_unchecked(offset),
                *self.data.get_unchecked(offset + 1),
                *self.data.get_unchecked(offset + 2),
                *self.data.get_unchecked(offset + 3),
                *self.data.get_unchecked(offset + 4),
                *self.data.get_unchecked(offset + 5),
                *self.data.get_unchecked(offset + 6),
                *self.data.get_unchecked(offset + 7),
            ]
        })
    }

    /// Get the PQ codes for vector at position `i`.
    #[inline]
    pub fn pq_codes(&self, i: u32) -> &[u8] {
        let start = self.pq_offset + i as usize * self.pq_len as usize;
        let end = start + self.pq_len as usize;
        // SAFETY: blob size validated in new().
        debug_assert!(end <= self.data.len());
        unsafe { self.data.get_unchecked(start..end) }
    }

    /// Get the entire PQ codes block (N × `pq_len` bytes).
    ///
    /// This is the hot-path accessor for ADC scanning — one contiguous slice.
    #[inline]
    pub fn pq_codes_block(&self) -> &[u8] {
        let end = self.pq_offset + self.count as usize * self.pq_len as usize;
        debug_assert!(end <= self.data.len());
        unsafe { self.data.get_unchecked(self.pq_offset..end) }
    }

    /// Get the raw vector at position `i` as a byte slice.
    ///
    /// Returns `None` if the blob has no raw vectors.
    #[inline]
    pub fn raw_vector_bytes(&self, i: u32) -> Option<&[u8]> {
        if !self.has_raw {
            return None;
        }
        let start = self.raw_offset + i as usize * self.raw_vec_bytes;
        let end = start + self.raw_vec_bytes;
        debug_assert!(end <= self.data.len());
        Some(unsafe { self.data.get_unchecked(start..end) })
    }

    /// Binary search for a vector ID. Returns the index if found.
    #[allow(clippy::cast_possible_truncation)]
    pub fn find_vector(&self, vid: u64) -> Option<u32> {
        let n = self.count as usize;
        if n == 0 {
            return None;
        }
        // Binary search over the sorted ID array.
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let mid_id = self.vector_id(mid as u32);
            match mid_id.cmp(&vid) {
                core::cmp::Ordering::Equal => return Some(mid as u32),
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }
}

impl core::fmt::Debug for ClusterBlobRef<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ClusterBlobRef")
            .field("count", &self.count)
            .field("pq_len", &self.pq_len)
            .field("has_raw", &self.has_raw)
            .field("blob_bytes", &self.data.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Merge helper for insert
// ---------------------------------------------------------------------------

/// Merge existing blob entries with new entries, producing a new sorted blob.
///
/// `existing` is the current cluster blob (or `None` for a new cluster).
/// `new_entries` are `(vector_id, pq_codes, optional_raw_vec_bytes)`, not necessarily sorted.
/// Raw vectors in `new_entries` must be pre-serialized as LE f32 bytes.
/// Duplicate `vector_ids` in `new_entries` replace existing entries (upsert).
///
/// Returns the encoded blob bytes.
pub fn merge_into_blob(
    existing: Option<&ClusterBlobRef<'_>>,
    new_entries: &mut [OwnedBlobEntry],
    pq_len: u16,
) -> Vec<u8> {
    // Sort new entries by vector_id for merge.
    new_entries.sort_unstable_by_key(|e| e.0);

    // Collect all entries: existing (minus any being replaced) + new.
    let mut merged: Vec<BlobEntryBytes<'_>> = Vec::new();

    if let Some(blob) = existing {
        let mut new_idx = 0;
        for i in 0..blob.count() {
            let vid = blob.vector_id(i);
            while new_idx < new_entries.len() && new_entries[new_idx].0 < vid {
                let e = &new_entries[new_idx];
                merged.push((e.0, &e.1, e.2.as_deref()));
                new_idx += 1;
            }
            if new_idx < new_entries.len() && new_entries[new_idx].0 == vid {
                // Replace existing with new.
                let e = &new_entries[new_idx];
                merged.push((e.0, &e.1, e.2.as_deref()));
                new_idx += 1;
            } else {
                // Keep existing entry (raw vector stays as bytes).
                let raw = if blob.has_raw_vectors() {
                    blob.raw_vector_bytes(i)
                } else {
                    None
                };
                merged.push((vid, blob.pq_codes(i), raw));
            }
        }
        for e in &new_entries[new_idx..] {
            merged.push((e.0, &e.1, e.2.as_deref()));
        }
    } else {
        for e in new_entries.iter() {
            merged.push((e.0, &e.1, e.2.as_deref()));
        }
    }

    encode_cluster_blob_raw_bytes(&merged, pq_len)
}

/// Remove a vector from a cluster blob. Returns the new blob bytes,
/// or `None` if the cluster is now empty or the vector was not found.
pub fn remove_from_blob(
    blob: &ClusterBlobRef<'_>,
    vector_id: u64,
    pq_len: u16,
) -> Option<Vec<u8>> {
    let idx = blob.find_vector(vector_id)?;
    let n = blob.count();
    if n == 1 {
        return None; // cluster becomes empty
    }

    let mut entries: Vec<BlobEntryBytes<'_>> = Vec::with_capacity((n - 1) as usize);
    for i in 0..n {
        if i == idx {
            continue;
        }
        let vid = blob.vector_id(i);
        let pq = blob.pq_codes(i);
        let raw = if blob.has_raw_vectors() {
            blob.raw_vector_bytes(i)
        } else {
            None
        };
        entries.push((vid, pq, raw));
    }

    Some(encode_cluster_blob_raw_bytes(&entries, pq_len))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_no_raw() {
        let entries: Vec<BlobEntryF32<'_>> = vec![
            (10, &[0, 1, 2, 3], None),
            (20, &[4, 5, 6, 7], None),
            (30, &[8, 9, 10, 11], None),
        ];
        let blob = encode_cluster_blob(&entries, 4);
        let view = ClusterBlobRef::new(&blob, 4, 0).unwrap();

        assert_eq!(view.count(), 3);
        assert_eq!(view.pq_len(), 4);
        assert!(!view.has_raw_vectors());

        assert_eq!(view.vector_id(0), 10);
        assert_eq!(view.vector_id(1), 20);
        assert_eq!(view.vector_id(2), 30);

        assert_eq!(view.pq_codes(0), &[0, 1, 2, 3]);
        assert_eq!(view.pq_codes(1), &[4, 5, 6, 7]);
        assert_eq!(view.pq_codes(2), &[8, 9, 10, 11]);

        assert!(view.raw_vector_bytes(0).is_none());
    }

    #[test]
    fn round_trip_with_raw() {
        let raw0: Vec<f32> = vec![1.0, 2.0];
        let raw1: Vec<f32> = vec![3.0, 4.0];
        let entries: Vec<BlobEntryF32<'_>> = vec![
            (5, &[10, 20], Some(&raw0)),
            (15, &[30, 40], Some(&raw1)),
        ];
        let blob = encode_cluster_blob(&entries, 2);
        let view = ClusterBlobRef::new(&blob, 2, 2).unwrap();

        assert_eq!(view.count(), 2);
        assert!(view.has_raw_vectors());

        assert_eq!(view.vector_id(0), 5);
        assert_eq!(view.pq_codes(0), &[10, 20]);

        let raw_bytes = view.raw_vector_bytes(0).unwrap();
        assert_eq!(raw_bytes.len(), 8); // 2 floats × 4 bytes
        let f0 = f32::from_le_bytes([raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]]);
        let f1 = f32::from_le_bytes([raw_bytes[4], raw_bytes[5], raw_bytes[6], raw_bytes[7]]);
        assert!((f0 - 1.0).abs() < f32::EPSILON);
        assert!((f1 - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn empty_blob() {
        let entries: Vec<BlobEntryF32<'_>> = vec![];
        let blob = encode_cluster_blob(&entries, 4);
        let view = ClusterBlobRef::new(&blob, 4, 0).unwrap();
        assert_eq!(view.count(), 0);
        assert!(view.find_vector(42).is_none());
        assert!(view.pq_codes_block().is_empty());
    }

    #[test]
    fn binary_search() {
        let entries: Vec<BlobEntryF32<'_>> = vec![
            (100, &[0], None),
            (200, &[1], None),
            (300, &[2], None),
            (400, &[3], None),
            (500, &[4], None),
        ];
        let blob = encode_cluster_blob(&entries, 1);
        let view = ClusterBlobRef::new(&blob, 1, 0).unwrap();

        assert_eq!(view.find_vector(100), Some(0));
        assert_eq!(view.find_vector(300), Some(2));
        assert_eq!(view.find_vector(500), Some(4));
        assert_eq!(view.find_vector(150), None);
        assert_eq!(view.find_vector(0), None);
        assert_eq!(view.find_vector(999), None);
    }

    #[test]
    fn pq_codes_block_contiguous() {
        let entries: Vec<BlobEntryF32<'_>> = vec![
            (1, &[10, 20, 30], None),
            (2, &[40, 50, 60], None),
        ];
        let blob = encode_cluster_blob(&entries, 3);
        let view = ClusterBlobRef::new(&blob, 3, 0).unwrap();

        let block = view.pq_codes_block();
        assert_eq!(block, &[10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn corrupted_magic() {
        let mut blob = encode_cluster_blob(&[], 4);
        blob[0] = 0xFF; // corrupt magic
        assert!(ClusterBlobRef::new(&blob, 4, 0).is_err());
    }

    #[test]
    fn corrupted_truncated() {
        let entries: Vec<BlobEntryF32<'_>> = vec![
            (1, &[0, 1], None),
        ];
        let blob = encode_cluster_blob(&entries, 2);
        // Truncate to lose the PQ codes.
        let truncated = &blob[..HEADER_SIZE + 8 - 1];
        assert!(ClusterBlobRef::new(truncated, 2, 0).is_err());
    }

    #[test]
    fn wrong_pq_len() {
        let entries: Vec<BlobEntryF32<'_>> = vec![
            (1, &[0, 1, 2, 3], None),
        ];
        let blob = encode_cluster_blob(&entries, 4);
        assert!(ClusterBlobRef::new(&blob, 8, 0).is_err()); // expects 8, blob has 4
    }

    #[test]
    fn merge_into_empty() {
        let mut new_entries: Vec<(u64, Vec<u8>, Option<Vec<u8>>)> = vec![
            (30, vec![3, 4], None),
            (10, vec![1, 2], None),
            (20, vec![2, 3], None),
        ];
        let blob = merge_into_blob(None, &mut new_entries, 2);
        let view = ClusterBlobRef::new(&blob, 2, 0).unwrap();
        assert_eq!(view.count(), 3);
        assert_eq!(view.vector_id(0), 10);
        assert_eq!(view.vector_id(1), 20);
        assert_eq!(view.vector_id(2), 30);
    }

    #[test]
    fn merge_upsert() {
        let entries: Vec<BlobEntryF32<'_>> = vec![
            (10, &[1, 1], None),
            (20, &[2, 2], None),
            (30, &[3, 3], None),
        ];
        let existing_blob = encode_cluster_blob(&entries, 2);
        let existing_ref = ClusterBlobRef::new(&existing_blob, 2, 0).unwrap();

        let mut new_entries: Vec<(u64, Vec<u8>, Option<Vec<u8>>)> = vec![
            (15, vec![9, 9], None),
            (20, vec![8, 8], None),
        ];
        let merged_blob = merge_into_blob(Some(&existing_ref), &mut new_entries, 2);
        let view = ClusterBlobRef::new(&merged_blob, 2, 0).unwrap();

        assert_eq!(view.count(), 4);
        assert_eq!(view.vector_id(0), 10);
        assert_eq!(view.vector_id(1), 15);
        assert_eq!(view.vector_id(2), 20);
        assert_eq!(view.vector_id(3), 30);
        assert_eq!(view.pq_codes(2), &[8, 8]);
        assert_eq!(view.pq_codes(0), &[1, 1]);
    }

    #[test]
    fn remove_vector() {
        let entries: Vec<BlobEntryF32<'_>> = vec![
            (10, &[1, 1], None),
            (20, &[2, 2], None),
            (30, &[3, 3], None),
        ];
        let blob = encode_cluster_blob(&entries, 2);
        let view = ClusterBlobRef::new(&blob, 2, 0).unwrap();

        let new_blob = remove_from_blob(&view, 20, 2).unwrap();
        let new_view = ClusterBlobRef::new(&new_blob, 2, 0).unwrap();
        assert_eq!(new_view.count(), 2);
        assert_eq!(new_view.vector_id(0), 10);
        assert_eq!(new_view.vector_id(1), 30);
    }

    #[test]
    fn remove_last_vector() {
        let entries: Vec<BlobEntryF32<'_>> = vec![
            (10, &[1, 1], None),
        ];
        let blob = encode_cluster_blob(&entries, 2);
        let view = ClusterBlobRef::new(&blob, 2, 0).unwrap();
        assert!(remove_from_blob(&view, 10, 2).is_none()); // empty -> None
    }

    #[test]
    fn remove_nonexistent() {
        let entries: Vec<BlobEntryF32<'_>> = vec![
            (10, &[1, 1], None),
        ];
        let blob = encode_cluster_blob(&entries, 2);
        let view = ClusterBlobRef::new(&blob, 2, 0).unwrap();
        assert!(remove_from_blob(&view, 999, 2).is_none());
    }
}
