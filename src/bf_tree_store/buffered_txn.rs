//! Write buffer overlay for `BfTree` transactions.
//!
//! Provides atomic commit/rollback semantics over `BfTree`'s immediate-write model.
//! All writes during a transaction are accumulated in a sorted `BTreeMap` buffer.
//! On commit, the buffer is flushed to `BfTree` atomically. On abort (or drop without
//! commit), the buffer is discarded -- providing true rollback.
//!
//! The `BufferedScanIter` merges buffered entries with `BfTree` scan results in sorted
//! order, implementing overlay semantics (buffer wins on key collision, tombstones
//! hide underlying entries).

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::storage_traits::OwnedKv;
use crate::types::{Key, Value};

use alloc::sync::Arc;

use super::adapter::BfTreeAdapter;
use super::database::{BfTreeTableScan, TableKind, table_prefix, table_prefix_end};
use super::error::BfTreeError;
use super::verification::{VerifyMode, should_verify, unwrap_value};

// ---------------------------------------------------------------------------
// BufferLookup -- result of checking the write buffer
// ---------------------------------------------------------------------------

/// Result of looking up a key in the write buffer.
pub(crate) enum BufferLookup {
    /// Key found with a value (insert or update).
    Found(Vec<u8>),
    /// Key found as a tombstone (pending delete).
    Tombstone,
    /// Key not in buffer -- caller should check `BfTree`.
    NotInBuffer,
}

// ---------------------------------------------------------------------------
// WriteBuffer -- sorted overlay of pending writes
// ---------------------------------------------------------------------------

/// Default maximum number of entries allowed in a single write buffer.
///
/// Prevents unbounded memory growth from adversarial or runaway write loops.
/// Each entry consumes at least one `BTreeMap` node plus the key/value
/// allocations, so 1 million entries already represents significant memory.
const DEFAULT_MAX_BUFFER_ENTRIES: usize = 1_000_000;

/// Sorted write buffer that accumulates mutations during a transaction.
///
/// Uses `BTreeMap` (not `HashMap`) to maintain sorted key order, which is
/// essential for the merge iterator that combines buffer entries with `BfTree`
/// scan results.
pub(crate) struct WriteBuffer {
    /// Key = full encoded table key (with namespace prefix).
    /// Value = `Some(bytes)` for insert/update, `None` for delete (tombstone).
    entries: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    /// Maximum number of entries permitted in this buffer. Writes that would
    /// exceed this limit are rejected with `BfTreeError::InvalidKV`.
    max_buffer_entries: usize,
}

impl WriteBuffer {
    /// Create an empty write buffer with the default entry limit.
    pub(crate) fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            max_buffer_entries: DEFAULT_MAX_BUFFER_ENTRIES,
        }
    }

    /// Create an empty write buffer with a custom entry limit.
    // Retained for downstream/test use; standard collection API.
    #[allow(dead_code)]
    pub(crate) fn with_max_entries(max_buffer_entries: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            max_buffer_entries,
        }
    }

    /// Buffer an insert or update.
    ///
    /// Returns an error if the buffer already contains `max_buffer_entries`
    /// entries and the key is not already present (i.e., this would be a net
    /// new entry, not an overwrite of an existing buffered key).
    pub(crate) fn put(&mut self, encoded_key: Vec<u8>, value: Vec<u8>) -> Result<(), BfTreeError> {
        if self.entries.len() >= self.max_buffer_entries && !self.entries.contains_key(&encoded_key)
        {
            return Err(BfTreeError::InvalidKV(alloc::format!(
                "write buffer full: {} entries (limit {})",
                self.entries.len(),
                self.max_buffer_entries
            )));
        }
        self.entries.insert(encoded_key, Some(value));
        Ok(())
    }

    /// Buffer a delete (tombstone).
    pub(crate) fn delete(&mut self, encoded_key: Vec<u8>) {
        self.entries.insert(encoded_key, None);
    }

    /// Look up a key in the buffer.
    pub(crate) fn get(&self, encoded_key: &[u8]) -> BufferLookup {
        match self.entries.get(encoded_key) {
            Some(Some(value)) => BufferLookup::Found(value.clone()),
            Some(None) => BufferLookup::Tombstone,
            None => BufferLookup::NotInBuffer,
        }
    }

    /// Get a sorted range of buffered entries within the given key bounds.
    ///
    /// Returns entries where `start <= key <= end` (inclusive both sides).
    /// Callers that need exclusive end semantics pass `prefix_end` (prefix + 1)
    /// as the upper bound, which naturally excludes keys past the prefix.
    /// For range scans with user-specified inclusive end, the exact encoded end
    /// key is passed and must be included.
    pub(crate) fn range(
        &self,
        start: &[u8],
        end: &[u8],
    ) -> alloc::collections::btree_map::Range<'_, Vec<u8>, Option<Vec<u8>>> {
        use core::ops::Bound;
        self.entries.range::<Vec<u8>, _>((
            Bound::Included(start.to_vec()),
            Bound::Included(end.to_vec()),
        ))
    }

    /// Get a sorted range of buffered entries with an exclusive end bound.
    ///
    /// Returns entries where `start <= key < end`. Use this when the upper
    /// bound is a computed "next prefix" that must not itself be included in
    /// the result set.
    pub(crate) fn range_excluded_end(
        &self,
        start: &[u8],
        end: &[u8],
    ) -> alloc::collections::btree_map::Range<'_, Vec<u8>, Option<Vec<u8>>> {
        use core::ops::Bound;
        self.entries.range::<Vec<u8>, _>((
            Bound::Included(start.to_vec()),
            Bound::Excluded(end.to_vec()),
        ))
    }

    /// Get all buffered entries whose key starts with the given prefix.
    ///
    /// This is safe for all-0xFF prefixes where `increment_prefix` overflows,
    /// because it filters by actual prefix match rather than relying on a
    /// computed upper bound.
    pub(crate) fn prefix_range(
        &self,
        prefix: &[u8],
    ) -> impl Iterator<Item = (&Vec<u8>, &Option<Vec<u8>>)> {
        use core::ops::Bound;
        let prefix_vec = prefix.to_vec();
        self.entries
            .range::<Vec<u8>, _>((Bound::Included(prefix_vec.clone()), Bound::Unbounded))
            .take_while(move |(k, _)| k.starts_with(&prefix_vec))
    }

    /// Number of buffered entries.
    // Standard collection API; retained for diagnostics and test use.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer is empty.
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Flush all buffered entries to `BfTree`.
    ///
    /// Pre-validates all insert entries against `BfTree`'s key/value size limits
    /// before writing anything. If validation fails, no entries are written.
    ///
    /// On partial write failure (adapter error mid-flush), compensating actions
    /// undo already-applied operations: inserts are rolled back via delete, and
    /// deletes are rolled back via re-insert of the previously-read value.
    pub(crate) fn flush(
        &mut self,
        adapter: &BfTreeAdapter,
        durability: super::config::DurabilityMode,
    ) -> Result<(), BfTreeError> {
        let max_record_size = adapter.inner().config().get_cb_max_record_size();
        let max_key_len = adapter.max_key_len();

        // Phase 1: Pre-validate all entries against BfTree's key and value size
        // limits before writing anything. Reject the entire flush upfront if any
        // entry would exceed limits, preventing partial writes and avoiding
        // panics during the flush phase.
        for (key, value) in &self.entries {
            // Reject empty keys for both inserts and tombstones. An empty key
            // is invalid regardless of the operation type.
            if key.is_empty() {
                return Err(BfTreeError::InvalidKV(alloc::string::String::from(
                    "key must not be empty",
                )));
            }
            if key.len() > max_key_len {
                return Err(BfTreeError::InvalidKV(alloc::format!(
                    "key size {} exceeds max {}",
                    key.len(),
                    max_key_len
                )));
            }
            if let Some(val) = value.as_ref().filter(|v| v.len() > max_record_size) {
                return Err(BfTreeError::InvalidKV(alloc::format!(
                    "value size {} exceeds max {}",
                    val.len(),
                    max_record_size
                )));
            }
        }

        // Phase 2: Separate inserts and deletes, apply in batch.
        //
        // BTreeMap iteration order is sorted by key -- exactly what
        // batch_insert_sorted_deferred_wal needs for leaf caching.
        let mut insert_pairs: Vec<(&[u8], &[u8])> = Vec::new();
        let mut delete_keys: Vec<&[u8]> = Vec::new();
        let mut delete_prev_values: Vec<(Vec<u8>, Option<Vec<u8>>)> = Vec::new();

        // Single reusable buffer for reading previous values before deletes.
        let mut delete_read_buf = vec![0u8; max_record_size];

        for (key, value) in &self.entries {
            if let Some(val) = value {
                insert_pairs.push((key.as_slice(), val.as_slice()));
            } else {
                // Snapshot the current value before deleting for rollback.
                let prev = match adapter.read(key, &mut delete_read_buf) {
                    Ok(len) => Some(delete_read_buf[..len as usize].to_vec()),
                    Err(_) => None,
                };
                delete_keys.push(key.as_slice());
                delete_prev_values.push((key.clone(), prev));
            }
        }

        // Batch insert: exploits key locality to skip redundant tree traversals.
        // For N sorted entries hitting ~K distinct leaves, this does K traversals
        // instead of N (typically 10-50x fewer for batch sizes 100-5000).
        if !insert_pairs.is_empty()
            && let Err(flush_err) = adapter.batch_insert_sorted_deferred_wal(&insert_pairs)
        {
            // Batch insert failed -- entries up to the failure point may have been
            // applied. Best-effort rollback: delete all intended insert keys.
            let flushed_inserts: Vec<Vec<u8>> =
                insert_pairs.iter().map(|(k, _)| k.to_vec()).collect();
            if let Err((rollback_failures, last_rollback_error)) =
                Self::compensate_rollback(adapter, &flushed_inserts, &[])
            {
                return Err(BfTreeError::PartialFlushRollbackFailed {
                    flush_error: alloc::format!("{flush_err}"),
                    rollback_failures,
                    last_rollback_error,
                });
            }
            return Err(flush_err);
        }

        // Batch delete.
        if !delete_keys.is_empty()
            && let Err(flush_err) = adapter.batch_delete_sorted_deferred_wal(&delete_keys)
        {
            // Rollback inserts + already-applied deletes.
            let flushed_inserts: Vec<Vec<u8>> =
                insert_pairs.iter().map(|(k, _)| k.to_vec()).collect();
            if let Err((rollback_failures, last_rollback_error)) =
                Self::compensate_rollback(adapter, &flushed_inserts, &delete_prev_values)
            {
                return Err(BfTreeError::PartialFlushRollbackFailed {
                    flush_error: alloc::format!("{flush_err}"),
                    rollback_failures,
                    last_rollback_error,
                });
            }
            return Err(flush_err);
        }

        // Flush WAL based on durability mode:
        // - Sync: fsync now (no data loss on crash)
        // - Periodic: WAL background thread fsyncs every wal_flush_interval_ms
        // - NoSync: no fsync (for benchmarks/ephemeral workloads)
        if durability == super::config::DurabilityMode::Sync {
            adapter.flush_wal().map_err(BfTreeError::from)?;
        }

        self.entries.clear();
        Ok(())
    }

    /// Compensating rollback: undo already-flushed entries on partial failure.
    ///
    /// Returns `Ok(())` if all compensating operations succeeded. Returns
    /// `Err((failure_count, last_error_message))` if any re-insert failed,
    /// so callers can surface a compound error indicating potential
    /// inconsistency.
    ///
    /// Note: undo-deletes (reverting a delete by re-inserting the old value)
    /// are the operations that can fail. Undo-inserts use `adapter.delete()`
    /// which is infallible in the `BfTree` API.
    fn compensate_rollback(
        adapter: &BfTreeAdapter,
        flushed_inserts: &[Vec<u8>],
        flushed_deletes: &[(Vec<u8>, Option<Vec<u8>>)],
    ) -> Result<(), (usize, alloc::string::String)> {
        let mut failure_count: usize = 0;
        let mut last_error = alloc::string::String::new();

        // Undo inserts by deleting the keys.
        for key in flushed_inserts {
            adapter.delete(key);
        }
        // Undo deletes by re-inserting the previous values.
        for (key, prev) in flushed_deletes {
            if let Some(val) = prev
                && let Err(e) = adapter.insert(key, val)
            {
                failure_count += 1;
                last_error = alloc::format!("{e}");
            }
        }

        if failure_count > 0 {
            Err((failure_count, last_error))
        } else {
            Ok(())
        }
    }

    /// Discard all buffered entries (rollback).
    pub(crate) fn discard(&mut self) {
        self.entries.clear();
    }

    /// Merge another write buffer into this one.
    ///
    /// Entries from `other` are applied on top of entries already in `self`.
    /// If both buffers contain the same key, `other`'s entry wins (last-writer
    /// wins), matching the semantics of sequential batch execution.
    ///
    /// The entry limit is checked against the post-merge size to prevent
    /// exceeding the maximum buffer capacity.
    pub(crate) fn merge_from(&mut self, other: WriteBuffer) -> Result<(), BfTreeError> {
        // Pre-check: will the merge exceed the entry limit? Count net new keys
        // (keys in `other` that are not already in `self`).
        let new_keys = other
            .entries
            .keys()
            .filter(|k| !self.entries.contains_key(*k))
            .count();
        let post_merge_len = self.entries.len() + new_keys;
        if post_merge_len > self.max_buffer_entries {
            return Err(BfTreeError::InvalidKV(alloc::format!(
                "merged buffer would have {} entries, exceeding limit of {}",
                post_merge_len,
                self.max_buffer_entries,
            )));
        }
        for (key, value) in other.entries {
            self.entries.insert(key, value);
        }
        Ok(())
    }

    /// Drain a table: tombstone all visible entries in `[prefix, prefix_end)`.
    ///
    /// - `BfTree` keys get tombstoned so they'll be deleted on flush.
    /// - Buffer-only inserts get replaced with tombstones.
    /// - Returns the count of visible entries that were drained.
    pub(crate) fn drain_table(
        &mut self,
        bftree_encoded_keys: &[Vec<u8>],
        prefix: &[u8],
        prefix_end: &[u8],
    ) -> u64 {
        use core::ops::Bound;

        let mut count = 0u64;

        // Step 1: Count and tombstone `BfTree` keys.
        for key in bftree_encoded_keys {
            match self.get(key) {
                BufferLookup::Tombstone => {} // already hidden
                _ => count += 1,
            }
            self.delete(key.clone());
        }

        // Step 2: Count and tombstone buffer-only inserts.
        // After step 1, all `BfTree` keys are tombstoned. Remaining inserts in range
        // are buffer-only entries.
        let buffer_only: Vec<Vec<u8>> = self
            .entries
            .range::<Vec<u8>, _>((
                Bound::Included(prefix.to_vec()),
                Bound::Excluded(prefix_end.to_vec()),
            ))
            .filter_map(|(k, v)| if v.is_some() { Some(k.clone()) } else { None })
            .collect();
        count += buffer_only.len() as u64;
        for key in buffer_only {
            self.entries.insert(key, None);
        }

        count
    }
}

// ---------------------------------------------------------------------------
// BufferedScanIter -- merge iterator over buffer + BfTree scan
// ---------------------------------------------------------------------------

/// Merge iterator that overlays write buffer entries onto a `BfTree` scan.
///
/// Produces entries in sorted key order. When a key exists in both the buffer
/// and the scan, the buffer entry wins (overlay semantics). Buffer tombstones
/// (`None` values) hide the corresponding scan entry.
pub struct BufferedScanIter<'a, K: Key + 'static, V: Value + 'static> {
    /// Buffered entries for the table's key range, collected into a vec for
    /// iteration. Each entry is `(encoded_key_without_prefix, Option<value>)`.
    buf_entries: Vec<(Vec<u8>, Option<Vec<u8>>)>,
    buf_idx: usize,

    /// `BfTree` scan iterator.
    scan: BfTreeTableScan<'a>,
    scan_buf: Vec<u8>,

    /// Current peeked scan entry: `(key_bytes, val_bytes)` with prefix stripped.
    scan_peek: Option<(Vec<u8>, Vec<u8>)>,
    scan_exhausted: bool,

    /// If set, skip entries whose user key matches this (exclusive start bound).
    exclude_start: Option<Vec<u8>>,
    /// If set, stop when user key matches this (exclusive end bound).
    exclude_end: Option<Vec<u8>>,

    /// Verification mode for checksum unwrapping of values.
    verify_mode: Arc<VerifyMode>,

    _key: PhantomData<K>,
    _val: PhantomData<V>,
}

impl<'a, K: Key + 'static, V: Value + 'static> BufferedScanIter<'a, K, V> {
    /// Create a new merge iterator.
    ///
    /// `buf_entries` must be sorted by key and have the table prefix stripped.
    /// `scan` is the `BfTree` range scan with prefix already set.
    pub(crate) fn new(
        buf_entries: Vec<(Vec<u8>, Option<Vec<u8>>)>,
        scan: BfTreeTableScan<'a>,
        max_record_size: usize,
        exclude_start: Option<Vec<u8>>,
        exclude_end: Option<Vec<u8>>,
        verify_mode: Arc<VerifyMode>,
    ) -> Self {
        Self {
            buf_entries,
            buf_idx: 0,
            scan,
            scan_buf: vec![0u8; max_record_size * 2],
            scan_peek: None,
            scan_exhausted: false,
            exclude_start,
            exclude_end,
            verify_mode,
            _key: PhantomData,
            _val: PhantomData,
        }
    }

    /// Advance the scan iterator and store the next entry in `scan_peek`.
    fn advance_scan(&mut self) {
        if self.scan_exhausted {
            return;
        }
        if let Some((key_bytes, val_bytes)) = self.scan.next(&mut self.scan_buf) {
            self.scan_peek = Some((key_bytes.to_vec(), val_bytes.to_vec()));
        } else {
            self.scan_peek = None;
            self.scan_exhausted = true;
        }
    }

    /// Peek at the current buffer entry (key bytes without prefix).
    fn buf_peek(&self) -> Option<(&[u8], &Option<Vec<u8>>)> {
        if self.buf_idx < self.buf_entries.len() {
            let (ref k, ref v) = self.buf_entries[self.buf_idx];
            Some((k.as_slice(), v))
        } else {
            None
        }
    }

    /// Advance the buffer index.
    fn advance_buf(&mut self) {
        self.buf_idx += 1;
    }
}

impl<K: Key + 'static, V: Value + 'static> Iterator for BufferedScanIter<'_, K, V> {
    type Item = crate::Result<(OwnedKv<K>, OwnedKv<V>)>;

    fn next(&mut self) -> Option<Self::Item> {
        // Initialize scan peek on first call.
        if self.scan_peek.is_none() && !self.scan_exhausted {
            self.advance_scan();
        }

        loop {
            let buf = self.buf_peek();
            let scan = self.scan_peek.as_ref();

            // Two-way merge: pick the entry with the smaller key.
            // On equal keys, buffer wins (overlay semantics).
            // Tombstones (None values) hide the entry.
            let entry: Option<(Vec<u8>, Vec<u8>)> = match (buf, scan) {
                (None, None) => return None,

                (Some((bk, bv)), None) => {
                    let key = bk.to_vec();
                    let val = bv.clone();
                    self.advance_buf();
                    val.map(|v| (key, v))
                }

                (None, Some((sk, sv))) => {
                    let key = sk.clone();
                    let val = sv.clone();
                    self.advance_scan();
                    Some((key, val))
                }

                (Some((bk, bv)), Some((sk, sv))) => {
                    use core::cmp::Ordering;
                    match bk.cmp(sk.as_slice()) {
                        Ordering::Less => {
                            let key = bk.to_vec();
                            let val = bv.clone();
                            self.advance_buf();
                            val.map(|v| (key, v))
                        }
                        Ordering::Equal => {
                            let key = bk.to_vec();
                            let val = bv.clone();
                            self.advance_buf();
                            self.advance_scan();
                            val.map(|v| (key, v))
                        }
                        Ordering::Greater => {
                            let key = sk.clone();
                            let val = sv.clone();
                            self.advance_scan();
                            Some((key, val))
                        }
                    }
                }
            };

            // Tombstone: entry is None, skip to next.
            let Some((key, val)) = entry else {
                continue;
            };

            // Exclusive start bound: skip the matching entry.
            // Only clear the filter when we see the excluded key itself or a
            // key that sorts after it (meaning we've passed the boundary).
            // Keys that sort before exclude_start pass through without clearing.
            if let Some(ref excl) = self.exclude_start {
                match key.as_slice().cmp(excl.as_slice()) {
                    core::cmp::Ordering::Equal => {
                        self.exclude_start = None;
                        continue;
                    }
                    core::cmp::Ordering::Greater => {
                        self.exclude_start = None;
                    }
                    core::cmp::Ordering::Less => {
                        // Key is before exclude_start; pass through, keep filter active.
                    }
                }
            }

            // Exclusive end bound: stop iteration.
            if self.exclude_end.as_ref().is_some_and(|excl| key == *excl) {
                return None;
            }

            let k = OwnedKv::new(key);
            let v = if self.verify_mode.is_enabled() {
                let verify = should_verify(self.verify_mode.as_ref());
                match unwrap_value(&val, verify) {
                    Ok(data) => OwnedKv::new(data.to_vec()),
                    Err(e) => return Some(Err(e.into())),
                }
            } else {
                OwnedKv::new(val)
            };
            return Some(Ok((k, v)));
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: collect buffer entries for a table's key range
// ---------------------------------------------------------------------------

/// Collect buffer entries that fall within a table's key range, stripping the
/// table prefix from keys.
///
/// Returns entries sorted by user key (prefix-stripped), suitable for
/// `BufferedScanIter`.
pub(crate) fn collect_buffer_entries_for_table(
    buffer: &WriteBuffer,
    table_name: &str,
    kind: TableKind,
    start_encoded: &[u8],
    end_encoded: &[u8],
) -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
    let prefix = table_prefix(table_name, kind);
    let prefix_len = prefix.len();

    buffer
        .range(start_encoded, end_encoded)
        .filter_map(|(key, val)| {
            if key.len() > prefix_len && key.starts_with(&prefix) {
                let user_key = key[prefix_len..].to_vec();
                Some((user_key, val.clone()))
            } else {
                None
            }
        })
        .collect()
}

/// Collect ALL buffer entries for a given table (full scan).
#[allow(dead_code)]
pub(crate) fn collect_all_buffer_entries_for_table(
    buffer: &WriteBuffer,
    table_name: &str,
    kind: TableKind,
) -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
    let prefix = table_prefix(table_name, kind);
    let prefix_end = table_prefix_end(table_name, kind);
    collect_buffer_entries_for_table(buffer, table_name, kind, &prefix, &prefix_end)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TableDefinition;
    use crate::bf_tree_store::config::{BfTreeConfig, DurabilityMode};
    use crate::bf_tree_store::database::{BfTreeDatabase, TableKind, encode_table_key};
    use crate::storage_traits::WriteTable;

    const ITEMS: TableDefinition<&str, u64> = TableDefinition::new("items");

    #[test]
    fn buffer_put_get() {
        let mut buf = WriteBuffer::new();
        let key = b"test_key".to_vec();

        // Initially not in buffer.
        assert!(matches!(buf.get(&key), BufferLookup::NotInBuffer));

        // Put and get.
        buf.put(key.clone(), b"value1".to_vec()).unwrap();
        match buf.get(&key) {
            BufferLookup::Found(v) => assert_eq!(v, b"value1"),
            _ => panic!("expected Found"),
        }

        // Overwrite.
        buf.put(key.clone(), b"value2".to_vec()).unwrap();
        match buf.get(&key) {
            BufferLookup::Found(v) => assert_eq!(v, b"value2"),
            _ => panic!("expected Found"),
        }
    }

    #[test]
    fn buffer_delete_tombstone() {
        let mut buf = WriteBuffer::new();
        let key = b"key".to_vec();

        buf.put(key.clone(), b"val".to_vec()).unwrap();
        buf.delete(key.clone());
        assert!(matches!(buf.get(&key), BufferLookup::Tombstone));
    }

    #[test]
    fn buffer_flush_applies_to_adapter() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let adapter = db.adapter();

        let mut buf = WriteBuffer::new();
        let key1 = encode_table_key("test", TableKind::Regular, b"k1");
        let key2 = encode_table_key("test", TableKind::Regular, b"k2");
        buf.put(key1.clone(), b"v1".to_vec()).unwrap();
        buf.put(key2.clone(), b"v2".to_vec()).unwrap();

        buf.flush(adapter, DurabilityMode::Sync).unwrap();

        // Verify in adapter.
        let max = adapter.inner().config().get_cb_max_record_size();
        let mut rbuf = vec![0u8; max];
        let len = adapter.read(&key1, &mut rbuf).unwrap();
        assert_eq!(&rbuf[..len as usize], b"v1");
    }

    #[test]
    fn buffer_discard_rollback() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let adapter = db.adapter();

        let mut buf = WriteBuffer::new();
        let key = encode_table_key("test", TableKind::Regular, b"k1");
        buf.put(key.clone(), b"val".to_vec()).unwrap();
        buf.discard();

        assert!(buf.is_empty());

        // Key should NOT be in adapter.
        let max = adapter.inner().config().get_cb_max_record_size();
        let mut rbuf = vec![0u8; max];
        assert!(adapter.read(&key, &mut rbuf).is_err());
    }

    #[test]
    fn buffer_flush_with_deletes() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let adapter = db.adapter();

        // Pre-populate BfTree.
        let key = encode_table_key("test", TableKind::Regular, b"existing");
        adapter.insert(&key, b"old_val").unwrap();

        // Buffer a delete for the existing key.
        let mut buf = WriteBuffer::new();
        buf.delete(key.clone());
        buf.flush(adapter, DurabilityMode::Sync).unwrap();

        // Key should be gone.
        let max = adapter.inner().config().get_cb_max_record_size();
        let mut rbuf = vec![0u8; max];
        assert!(adapter.read(&key, &mut rbuf).is_err());
    }

    #[test]
    fn buffered_write_txn_read_your_writes() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();

        // Insert via buffered write.
        WriteTable::st_insert(&mut table, &"hello", &42u64).unwrap();

        // Read back within same transaction -- should see buffered value.
        let val = WriteTable::st_get(&table, &"hello").unwrap();
        assert!(val.is_some());
        assert_eq!(val.unwrap().value(), 42u64);
    }

    #[test]
    fn buffered_write_txn_abort_on_drop() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        {
            let wtxn = db.begin_write();
            let mut table = wtxn.open_table(ITEMS).unwrap();
            WriteTable::st_insert(&mut table, &"temp", &99u64).unwrap();
            drop(table);
            // Drop without commit -- should rollback.
        }

        // Verify not visible via read transaction.
        let rtxn = db.begin_read();
        let mut ro = rtxn.open_table(ITEMS).unwrap();
        assert!(ro.get(&"temp").unwrap().is_none());
    }

    #[test]
    fn buffered_write_txn_commit_visible() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        WriteTable::st_insert(&mut table, &"committed", &77u64).unwrap();
        drop(table);
        wtxn.commit().unwrap();

        // Should be visible.
        let rtxn = db.begin_read();
        let mut ro = rtxn.open_table(ITEMS).unwrap();
        let val = ro.get(&"committed").unwrap().unwrap();
        assert_eq!(u64::from_le_bytes(val.as_slice().try_into().unwrap()), 77);
    }

    #[test]
    fn buffered_scan_merges_buffer_and_bftree() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        // Pre-populate BfTree with some data.
        {
            let wtxn = db.begin_write();
            let mut table = wtxn.open_table(ITEMS).unwrap();
            WriteTable::st_insert(&mut table, &"a", &1u64).unwrap();
            WriteTable::st_insert(&mut table, &"c", &3u64).unwrap();
            WriteTable::st_insert(&mut table, &"e", &5u64).unwrap();
            drop(table);
            wtxn.commit().unwrap();
        }

        // New transaction: add "b" and "d" in buffer, delete "c".
        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        WriteTable::st_insert(&mut table, &"b", &2u64).unwrap();
        WriteTable::st_insert(&mut table, &"d", &4u64).unwrap();
        WriteTable::st_remove(&mut table, &"c").unwrap();

        // Full range scan should merge correctly: a, b, d, e.
        let iter = WriteTable::st_range(&table, None, None, true, true).unwrap();
        let entries: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.value()).collect();
        assert_eq!(keys, vec!["a", "b", "d", "e"]);

        let vals: Vec<u64> = entries.iter().map(|(_, v)| v.value()).collect();
        assert_eq!(vals, vec![1, 2, 4, 5]);
    }

    #[test]
    fn buffered_overwrite_supersedes_bftree() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        // Pre-populate.
        {
            let wtxn = db.begin_write();
            let mut table = wtxn.open_table(ITEMS).unwrap();
            WriteTable::st_insert(&mut table, &"key", &100u64).unwrap();
            drop(table);
            wtxn.commit().unwrap();
        }

        // Overwrite in buffer.
        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        WriteTable::st_insert(&mut table, &"key", &200u64).unwrap();

        let val = WriteTable::st_get(&table, &"key").unwrap().unwrap();
        assert_eq!(val.value(), 200u64);

        // Scan should also see 200.
        let iter = WriteTable::st_range(&table, None, None, true, true).unwrap();
        let entries: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1.value(), 200u64);
    }

    #[test]
    fn buffered_delete_hides_bftree_entry() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        // Pre-populate.
        {
            let wtxn = db.begin_write();
            let mut table = wtxn.open_table(ITEMS).unwrap();
            WriteTable::st_insert(&mut table, &"visible", &1u64).unwrap();
            WriteTable::st_insert(&mut table, &"hidden", &2u64).unwrap();
            drop(table);
            wtxn.commit().unwrap();
        }

        // Delete "hidden" in buffer.
        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        WriteTable::st_remove(&mut table, &"hidden").unwrap();

        // Point lookup.
        assert!(WriteTable::st_get(&table, &"hidden").unwrap().is_none());

        // Scan should only show "visible".
        let iter = WriteTable::st_range(&table, None, None, true, true).unwrap();
        let entries: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(entries.len(), 1);
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.value()).collect();
        assert_eq!(keys, vec!["visible"]);
    }

    /// Exercises the compensating rollback path in `WriteBuffer::flush()`.
    ///
    /// Strategy: buffer two entries where the first (sorted by key) succeeds
    /// but the second fails on `BfTree`'s combined key+value size check. The
    /// pre-validation in `flush()` checks key and value lengths independently,
    /// but `BfTree` rejects records where `key.len() + value.len()` exceeds
    /// `cb_max_record_size`. After the second insert fails, the first insert
    /// must be rolled back (deleted from `BfTree`).
    #[test]
    fn flush_rollback_undoes_partial_writes() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let adapter = db.adapter();
        let max_record_size = adapter.inner().config().get_cb_max_record_size();
        let max_key_len = adapter.max_key_len();

        // key_a sorts before key_b so it gets flushed first.
        let key_a = encode_table_key("t", TableKind::Regular, b"aaa");
        let val_a = vec![1u8; 8];

        // key_b: large key + large value that individually pass pre-validation
        // but combined exceed cb_max_record_size.
        let raw_key_b = vec![b'b'; max_key_len - 4]; // -4 for encode_table_key overhead
        let key_b = encode_table_key("t", TableKind::Regular, &raw_key_b);
        assert!(key_b.len() <= max_key_len, "key_b must pass pre-validation");
        // value sized so key_b.len() + val_b.len() > max_record_size
        let val_b_len = max_record_size - key_b.len() + 1;
        assert!(
            val_b_len <= max_record_size,
            "val_b must pass pre-validation"
        );
        let val_b = vec![2u8; val_b_len];

        let mut buf = WriteBuffer::new();
        buf.put(key_a.clone(), val_a.clone()).unwrap();
        buf.put(key_b.clone(), val_b).unwrap();

        // Flush should fail because key_b's combined size exceeds max_record_size.
        let result = buf.flush(adapter, DurabilityMode::Sync);
        assert!(
            result.is_err(),
            "flush must fail on oversized combined record"
        );

        // key_a must NOT be in BfTree -- it was rolled back.
        let mut rbuf = vec![0u8; max_record_size];
        assert!(
            adapter.read(&key_a, &mut rbuf).is_err(),
            "key_a must be rolled back after partial flush failure"
        );
    }

    /// Rollback restores previously-deleted values when flush fails mid-way.
    ///
    /// Scenario: a pre-existing key is deleted in the buffer, then a later
    /// insert fails. The compensating rollback must re-insert the deleted
    /// key's original value.
    #[test]
    fn flush_rollback_restores_deleted_values() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let adapter = db.adapter();
        let max_record_size = adapter.inner().config().get_cb_max_record_size();
        let max_key_len = adapter.max_key_len();

        // Pre-populate BfTree with a key that will be deleted in the buffer.
        let key_a = encode_table_key("t", TableKind::Regular, b"aaa");
        let original_val = b"original_value";
        adapter.insert(&key_a, original_val).unwrap();

        // Buffer: delete key_a (tombstone), then insert oversized key_z.
        let raw_key_z = vec![b'z'; max_key_len - 4];
        let key_z = encode_table_key("t", TableKind::Regular, &raw_key_z);
        let val_z_len = max_record_size - key_z.len() + 1;
        let val_z = vec![3u8; val_z_len];

        let mut buf = WriteBuffer::new();
        buf.delete(key_a.clone());
        buf.put(key_z.clone(), val_z).unwrap();

        // Flush fails on key_z insert.
        let result = buf.flush(adapter, DurabilityMode::Sync);
        assert!(
            result.is_err(),
            "flush must fail on oversized combined record"
        );

        // key_a must be restored to its original value.
        let mut rbuf = vec![0u8; max_record_size];
        let len = adapter
            .read(&key_a, &mut rbuf)
            .expect("key_a must be restored after rollback");
        assert_eq!(
            &rbuf[..len as usize],
            original_val,
            "restored value must match original"
        );
    }
}
