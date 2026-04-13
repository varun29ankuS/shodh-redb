//! Bf-Tree backed table types providing familiar `open_table` -> `insert`/`get`/`remove` API.
//!
//! These types mirror the ergonomics of shodh-redb's `Table` and `ReadOnlyTable` but
//! are backed by the concurrent Bf-Tree engine. Unlike the legacy types:
//!
//! - **No `AccessGuard`**: Values are returned as owned `Vec<u8>` since Bf-Tree reads
//!   copy data into a caller-provided buffer (no zero-copy page references).
//! - **No `range()` returning iterator**: Scans use `BfTreeTableScan` from the database layer.
//! - **Concurrent writes**: Multiple `BfTreeTable` handles from different transactions
//!   can write to the same table simultaneously without blocking.

use crate::compat::Mutex;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::TableHandle;
use crate::cdc::types::{CdcEvent, ChangeOp};
use crate::sealed::Sealed;
use crate::storage_traits::OwnedKv;
use crate::types::{Key, Value};

use super::adapter::BfTreeAdapter;
use super::buffered_txn::{
    BufferLookup, BufferedScanIter, WriteBuffer, collect_buffer_entries_for_table,
};
use super::database::{
    BfTreeTableScan, TableKind, encode_ordered_table_key, encode_ordered_table_key_into,
    encode_table_key, table_prefix, table_prefix_end,
};
use super::error::BfTreeError;
use super::value_indirection::{self, BlobHandle, is_blob_handle};
use super::verification::{VerifyMode, should_verify, unwrap_value, wrap_value};

/// A writable table handle backed by Bf-Tree.
///
/// Obtained via [`BfTreeDatabaseWriteTxn::open_table()`]. Multiple handles
/// to the same table can coexist across threads -- writes are CAS-based.
///
/// # Example
///
/// ```ignore
/// let mut wtxn = db.begin_write();
/// let mut table = wtxn.open_table(MY_TABLE).unwrap();
/// table.insert(&"key", &42u64).unwrap();
/// let val = table.get(&"key").unwrap();
/// drop(table);
/// wtxn.commit().unwrap();
/// ```
pub struct BfTreeTable<'txn, K: Key + 'static, V: Value + 'static> {
    name: String,
    adapter: &'txn Arc<BfTreeAdapter>,
    ops_count: &'txn AtomicU64,
    cdc_log: Option<&'txn Mutex<Vec<CdcEvent>>>,
    buffer: &'txn Mutex<WriteBuffer>,
    verify_mode: &'txn Arc<VerifyMode>,
    /// Reusable buffer for bf-tree read output, avoiding per-read heap allocation.
    read_buf: Vec<u8>,
    /// Reusable buffer for encoded key, avoiding per-read heap allocation.
    key_buf: Vec<u8>,
    /// Shared counter for value-indirection blob IDs.
    blob_id_counter: &'txn Arc<AtomicU32>,
    /// Threshold in bytes above which values are chunked out-of-line.
    blob_threshold: usize,
    _key: PhantomData<K>,
    _val: PhantomData<V>,
}

impl<K: Key + 'static, V: Value + 'static> Sealed for BfTreeTable<'_, K, V> {}

impl<K: Key + 'static, V: Value + 'static> TableHandle for BfTreeTable<'_, K, V> {
    fn name(&self) -> &str {
        &self.name
    }
}

impl<'txn, K: Key + 'static, V: Value + 'static> BfTreeTable<'txn, K, V> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        name: &str,
        adapter: &'txn Arc<BfTreeAdapter>,
        ops_count: &'txn AtomicU64,
        cdc_log: Option<&'txn Mutex<Vec<CdcEvent>>>,
        buffer: &'txn Mutex<WriteBuffer>,
        verify_mode: &'txn Arc<VerifyMode>,
        blob_id_counter: &'txn Arc<AtomicU32>,
        blob_threshold: usize,
    ) -> Self {
        let max_record = adapter.inner().config().get_cb_max_record_size();
        let key_buf_size = adapter.max_key_len() + 64;
        Self {
            name: String::from(name),
            adapter,
            ops_count,
            cdc_log,
            buffer,
            verify_mode,
            read_buf: vec![0u8; max_record],
            key_buf: vec![0u8; key_buf_size],
            blob_id_counter,
            blob_threshold,
            _key: PhantomData,
            _val: PhantomData,
        }
    }

    /// Record a CDC event if CDC is enabled.
    fn record_cdc(&self, event: CdcEvent) {
        if let Some(log) = self.cdc_log {
            log.lock().push(event);
        }
    }

    /// Insert a key-value pair. Returns the previous value if the key existed.
    ///
    /// Writes are buffered -- they become visible within this transaction
    /// immediately (read-your-writes) but are only flushed to `BfTree` on commit.
    ///
    /// The buffer lock is held across the entire read-old + write-new sequence
    /// to prevent concurrent writers from interleaving and causing stale CDC
    /// events or lost updates.
    pub fn insert(
        &mut self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
    ) -> Result<Option<Vec<u8>>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let val_bytes = V::as_bytes(value);
        let encoded_key = encode_ordered_table_key::<K>(&self.name, TableKind::Regular, key);

        let checksumming = self.verify_mode.is_enabled();
        let store_bytes = if checksumming {
            wrap_value(val_bytes.as_ref())
        } else {
            val_bytes.as_ref().to_vec()
        };
        let blob_enabled = self.blob_threshold > 0;

        // Hold the buffer lock across the entire read-modify-write to prevent
        // TOCTOU races where a concurrent writer could interleave between the
        // old-value read and the new-value write, causing stale CDC events.
        let previous = {
            let mut buffer = self.buffer.lock();
            let prev_raw = match buffer.get(&encoded_key) {
                BufferLookup::Found(v) => Some(v),
                BufferLookup::Tombstone => None,
                BufferLookup::NotInBuffer => {
                    let max_val = self.adapter.inner().config().get_cb_max_record_size();
                    let mut buf = vec![0u8; max_val];
                    match self.adapter.read(&encoded_key, &mut buf) {
                        Ok(len) => Some(buf[..len as usize].to_vec()),
                        Err(BfTreeError::NotFound | BfTreeError::Deleted) => None,
                        Err(e) => return Err(e),
                    }
                }
            };

            // Resolve the previous value: reassemble blobs and clean up old chunks.
            let previous = match prev_raw {
                Some(raw) if blob_enabled && is_blob_handle(&raw) => {
                    let handle = BlobHandle::decode(&raw).ok_or_else(|| {
                        BfTreeError::InvalidKV(alloc::string::String::from(
                            "corrupt blob handle on insert overwrite",
                        ))
                    })?;
                    let reassembled =
                        value_indirection::read_blob(&handle, &buffer, self.adapter)?;
                    value_indirection::delete_chunks(&mut buffer, &handle);
                    if checksumming {
                        let verify = should_verify(self.verify_mode.as_ref());
                        Some(unwrap_value(&reassembled, verify)?.to_vec())
                    } else {
                        Some(reassembled)
                    }
                }
                Some(raw) if checksumming => {
                    let verify = should_verify(self.verify_mode.as_ref());
                    Some(unwrap_value(&raw, verify)?.to_vec())
                }
                other => other,
            };

            // Chunk the new value if above threshold.
            let final_bytes = if blob_enabled && store_bytes.len() > self.blob_threshold {
                let handle = value_indirection::write_chunks(
                    &mut buffer,
                    self.blob_id_counter,
                    &store_bytes,
                )?;
                handle.encode().to_vec()
            } else {
                store_bytes
            };

            buffer.put(encoded_key, final_bytes)?;
            previous
        };

        self.ops_count.fetch_add(1, Ordering::Relaxed);

        // Record CDC event if enabled.
        if self.cdc_log.is_some() {
            self.record_cdc(CdcEvent {
                table_name: self.name.clone(),
                op: if previous.is_some() {
                    ChangeOp::Update
                } else {
                    ChangeOp::Insert
                },
                key: key_bytes.as_ref().to_vec(),
                new_value: Some(val_bytes.as_ref().to_vec()),
                old_value: previous.clone(),
            });
        }

        Ok(previous)
    }

    /// Remove a key-value pair. Returns the removed value if the key existed.
    ///
    /// Writes a tombstone to the buffer. On commit, the tombstone deletes the
    /// entry from `BfTree`. On abort, the tombstone is discarded.
    ///
    /// The buffer lock is held across the entire read-old + tombstone-write
    /// sequence to prevent concurrent writers from interleaving and causing
    /// stale CDC events or a tombstone overwriting a concurrent insert.
    pub fn remove(&mut self, key: &K::SelfType<'_>) -> Result<Option<Vec<u8>>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_ordered_table_key::<K>(&self.name, TableKind::Regular, key);
        let checksumming = self.verify_mode.is_enabled();
        let blob_enabled = self.blob_threshold > 0;

        // Hold the buffer lock across the entire read + tombstone-write to
        // prevent TOCTOU races where a concurrent writer could insert between
        // the old-value read and the tombstone write.
        let previous = {
            let mut buffer = self.buffer.lock();
            let prev_raw = match buffer.get(&encoded_key) {
                BufferLookup::Found(v) => Some(v),
                BufferLookup::Tombstone => None,
                BufferLookup::NotInBuffer => {
                    let max_val = self.adapter.inner().config().get_cb_max_record_size();
                    let mut buf = vec![0u8; max_val];
                    match self.adapter.read(&encoded_key, &mut buf) {
                        Ok(len) => Some(buf[..len as usize].to_vec()),
                        Err(BfTreeError::NotFound | BfTreeError::Deleted) => None,
                        Err(e) => return Err(e),
                    }
                }
            };

            // Resolve previous value: reassemble blobs and clean up chunks.
            let previous = match prev_raw {
                Some(raw) if blob_enabled && is_blob_handle(&raw) => {
                    let handle = BlobHandle::decode(&raw).ok_or_else(|| {
                        BfTreeError::InvalidKV(alloc::string::String::from(
                            "corrupt blob handle on remove",
                        ))
                    })?;
                    let reassembled =
                        value_indirection::read_blob(&handle, &buffer, self.adapter)?;
                    value_indirection::delete_chunks(&mut buffer, &handle);
                    if checksumming {
                        let verify = should_verify(self.verify_mode.as_ref());
                        Some(unwrap_value(&reassembled, verify)?.to_vec())
                    } else {
                        Some(reassembled)
                    }
                }
                Some(raw) if checksumming => {
                    let verify = should_verify(self.verify_mode.as_ref());
                    Some(unwrap_value(&raw, verify)?.to_vec())
                }
                other => other,
            };

            if previous.is_some() {
                buffer.delete(encoded_key);
            }
            previous
        };

        if previous.is_some() {
            self.ops_count.fetch_add(1, Ordering::Relaxed);

            // Record CDC event if enabled.
            if self.cdc_log.is_some() {
                self.record_cdc(CdcEvent {
                    table_name: self.name.clone(),
                    op: ChangeOp::Delete,
                    key: key_bytes.as_ref().to_vec(),
                    new_value: None,
                    old_value: previous.clone(),
                });
            }
        }

        Ok(previous)
    }

    /// Read the value for a key.
    ///
    /// Checks the write buffer first (for read-your-writes), then falls through
    /// to `BfTree` if the key is not in the buffer. Uses pre-allocated buffers
    /// to avoid per-read heap allocations on the bf-tree fallthrough path.
    pub fn get(&mut self, key: &K::SelfType<'_>) -> Result<Option<Vec<u8>>, BfTreeError> {
        let enc_len = encode_ordered_table_key_into::<K>(
            &mut self.key_buf,
            &self.name,
            TableKind::Regular,
            key,
        );
        let encoded_key = &self.key_buf[..enc_len];
        let checksumming = self.verify_mode.is_enabled();
        let blob_enabled = self.blob_threshold > 0;

        let buffer = self.buffer.lock();
        match buffer.get(encoded_key) {
            BufferLookup::Found(v) => {
                let data = if blob_enabled && is_blob_handle(&v) {
                    let handle = BlobHandle::decode(&v).ok_or_else(|| {
                        BfTreeError::InvalidKV(alloc::string::String::from(
                            "corrupt blob handle in buffer",
                        ))
                    })?;
                    value_indirection::read_blob(&handle, &buffer, self.adapter)?
                } else {
                    v
                };
                drop(buffer);
                return if checksumming {
                    let verify = should_verify(self.verify_mode.as_ref());
                    Ok(Some(unwrap_value(&data, verify)?.to_vec()))
                } else {
                    Ok(Some(data))
                };
            }
            BufferLookup::Tombstone => return Ok(None),
            BufferLookup::NotInBuffer => {}
        }
        drop(buffer);

        match self.adapter.read(encoded_key, &mut self.read_buf) {
            Ok(len) => {
                let raw = &self.read_buf[..len as usize];
                if blob_enabled && is_blob_handle(raw) {
                    let handle = BlobHandle::decode(raw).ok_or_else(|| {
                        BfTreeError::InvalidKV(alloc::string::String::from(
                            "corrupt blob handle in BfTree",
                        ))
                    })?;
                    let data = value_indirection::read_blob_readonly(&handle, self.adapter)?;
                    if checksumming {
                        let verify = should_verify(self.verify_mode.as_ref());
                        Ok(Some(unwrap_value(&data, verify)?.to_vec()))
                    } else {
                        Ok(Some(data))
                    }
                } else if checksumming {
                    let verify = should_verify(self.verify_mode.as_ref());
                    Ok(Some(unwrap_value(raw, verify)?.to_vec()))
                } else {
                    Ok(Some(raw.to_vec()))
                }
            }
            Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Atomic read-modify-write merge operation within the write buffer.
    ///
    /// Reads the current value (buffer-aware), applies the merge operator, and
    /// writes the result back to the buffer. The merge is atomic within the
    /// transaction -- other transactions see either the old or the new value.
    ///
    /// The buffer lock is held across the entire read + merge + write sequence
    /// to prevent lost updates (e.g., counter 10 + 5 + 5 = 15 instead of 20).
    pub fn merge(
        &mut self,
        key: &K::SelfType<'_>,
        operand: &[u8],
        operator: &dyn crate::merge::MergeOperator,
    ) -> Result<(), BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_ordered_table_key::<K>(&self.name, TableKind::Regular, key);
        let checksumming = self.verify_mode.is_enabled();
        let blob_enabled = self.blob_threshold > 0;

        // Hold the buffer lock across the entire read-merge-write to prevent
        // TOCTOU races where a concurrent merge could read the same stale
        // value, causing a classic lost-update problem.
        let (old_value, new_value) = {
            let mut buffer = self.buffer.lock();
            let existing_raw = match buffer.get(&encoded_key) {
                BufferLookup::Found(v) => Some(v),
                BufferLookup::Tombstone => None,
                BufferLookup::NotInBuffer => {
                    let max_val = self.adapter.inner().config().get_cb_max_record_size();
                    let mut buf = vec![0u8; max_val];
                    match self.adapter.read(&encoded_key, &mut buf) {
                        Ok(len) => Some(buf[..len as usize].to_vec()),
                        Err(BfTreeError::NotFound | BfTreeError::Deleted) => None,
                        Err(e) => return Err(e),
                    }
                }
            };
            // Resolve blob handle and unwrap checksum before passing to merge operator.
            let existing = match existing_raw {
                Some(ref raw) if blob_enabled && is_blob_handle(raw) => {
                    let handle = BlobHandle::decode(raw).ok_or_else(|| {
                        BfTreeError::InvalidKV(alloc::string::String::from(
                            "corrupt blob handle on merge",
                        ))
                    })?;
                    let reassembled =
                        value_indirection::read_blob(&handle, &buffer, self.adapter)?;
                    value_indirection::delete_chunks(&mut buffer, &handle);
                    if checksumming {
                        let verify = should_verify(self.verify_mode.as_ref());
                        Some(unwrap_value(&reassembled, verify)?.to_vec())
                    } else {
                        Some(reassembled)
                    }
                }
                Some(ref raw) if checksumming => {
                    let verify = should_verify(self.verify_mode.as_ref());
                    Some(unwrap_value(raw, verify)?.to_vec())
                }
                other => other,
            };
            let merged = operator.merge(key_bytes.as_ref(), existing.as_deref(), operand);
            match merged {
                Some(ref new_val) => {
                    let store = if checksumming {
                        wrap_value(new_val)
                    } else {
                        new_val.clone()
                    };
                    // Chunk the merged result if above threshold.
                    let final_bytes =
                        if blob_enabled && store.len() > self.blob_threshold {
                            let handle = value_indirection::write_chunks(
                                &mut buffer,
                                self.blob_id_counter,
                                &store,
                            )?;
                            handle.encode().to_vec()
                        } else {
                            store
                        };
                    buffer.put(encoded_key, final_bytes)?;
                }
                None => buffer.delete(encoded_key),
            }
            (existing, merged)
        };

        self.ops_count.fetch_add(1, Ordering::Relaxed);

        // MERGE-03: Record CDC events for merge mutations, mirroring the
        // insert()/remove() logic so CDC consumers see all changes.
        if self.cdc_log.is_some() {
            let (op, cdc_new, cdc_old) = match (&old_value, &new_value) {
                (None, Some(nv)) => (ChangeOp::Insert, Some(nv.clone()), None),
                (Some(_), Some(nv)) => (ChangeOp::Update, Some(nv.clone()), old_value.clone()),
                (Some(_), None) => (ChangeOp::Delete, None, old_value.clone()),
                (None, None) => return Ok(()),
            };
            self.record_cdc(CdcEvent {
                table_name: self.name.clone(),
                op,
                key: key_bytes.as_ref().to_vec(),
                new_value: cdc_new,
                old_value: cdc_old,
            });
        }

        Ok(())
    }

    /// Check if a key exists in this table.
    ///
    /// Checks the write buffer first, falls through to `BfTree`.
    pub fn contains_key(&self, key: &K::SelfType<'_>) -> bool {
        let encoded_key = encode_ordered_table_key::<K>(&self.name, TableKind::Regular, key);

        let buffer = self.buffer.lock();
        match buffer.get(&encoded_key) {
            BufferLookup::Found(_) => return true,
            BufferLookup::Tombstone => return false,
            BufferLookup::NotInBuffer => {}
        }
        drop(buffer);

        self.adapter.contains_key(&encoded_key)
    }
}

/// A read-only table handle backed by Bf-Tree.
///
/// Obtained via [`BfTreeDatabaseReadTxn::open_table()`].
pub struct BfTreeReadOnlyTable<'txn, K: Key + 'static, V: Value + 'static> {
    name: String,
    adapter: &'txn Arc<BfTreeAdapter>,
    verify_mode: &'txn Arc<VerifyMode>,
    /// Reusable buffer for bf-tree read output, avoiding per-read heap allocation.
    read_buf: Vec<u8>,
    /// Reusable buffer for encoded key, avoiding per-read heap allocation.
    key_buf: Vec<u8>,
    /// Threshold for blob indirection (0 = disabled).
    blob_threshold: usize,
    _key: PhantomData<K>,
    _val: PhantomData<V>,
}

impl<K: Key + 'static, V: Value + 'static> Sealed for BfTreeReadOnlyTable<'_, K, V> {}

impl<K: Key + 'static, V: Value + 'static> TableHandle for BfTreeReadOnlyTable<'_, K, V> {
    fn name(&self) -> &str {
        &self.name
    }
}

impl<'txn, K: Key + 'static, V: Value + 'static> BfTreeReadOnlyTable<'txn, K, V> {
    pub(crate) fn new(
        name: &str,
        adapter: &'txn Arc<BfTreeAdapter>,
        verify_mode: &'txn Arc<VerifyMode>,
        blob_threshold: usize,
    ) -> Self {
        let max_record = adapter.inner().config().get_cb_max_record_size();
        let key_buf_size = adapter.max_key_len() + 64;
        Self {
            name: String::from(name),
            adapter,
            verify_mode,
            read_buf: vec![0u8; max_record],
            key_buf: vec![0u8; key_buf_size],
            blob_threshold,
            _key: PhantomData,
            _val: PhantomData,
        }
    }

    /// Read the value for a key.
    ///
    /// Uses pre-allocated buffers to avoid per-read heap allocations.
    pub fn get(&mut self, key: &K::SelfType<'_>) -> Result<Option<Vec<u8>>, BfTreeError> {
        let enc_len = encode_ordered_table_key_into::<K>(
            &mut self.key_buf,
            &self.name,
            TableKind::Regular,
            key,
        );
        let checksumming = self.verify_mode.is_enabled();
        let blob_enabled = self.blob_threshold > 0;
        match self
            .adapter
            .read(&self.key_buf[..enc_len], &mut self.read_buf)
        {
            Ok(len) => {
                let raw = &self.read_buf[..len as usize];
                if blob_enabled && is_blob_handle(raw) {
                    let handle = BlobHandle::decode(raw).ok_or_else(|| {
                        BfTreeError::InvalidKV(alloc::string::String::from(
                            "corrupt blob handle in readonly get",
                        ))
                    })?;
                    let data = value_indirection::read_blob_readonly(&handle, self.adapter)?;
                    if checksumming {
                        let verify = should_verify(self.verify_mode.as_ref());
                        Ok(Some(unwrap_value(&data, verify)?.to_vec()))
                    } else {
                        Ok(Some(data))
                    }
                } else if checksumming {
                    let verify = should_verify(self.verify_mode.as_ref());
                    Ok(Some(unwrap_value(raw, verify)?.to_vec()))
                } else {
                    Ok(Some(raw.to_vec()))
                }
            }
            Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Check if a key exists in this table.
    pub fn contains_key(&self, key: &K::SelfType<'_>) -> bool {
        let encoded_key = encode_ordered_table_key::<K>(&self.name, TableKind::Regular, key);
        self.adapter.contains_key(&encoded_key)
    }

    /// Scan all entries in this table.
    pub fn scan(&self) -> Result<BfTreeTableScan<'_>, BfTreeError> {
        let prefix = table_prefix(&self.name, TableKind::Regular);
        let prefix_end = table_prefix_end(&self.name, TableKind::Regular);
        let prefix_len = prefix.len();
        let iter = self.adapter.scan_range(&prefix, &prefix_end)?;
        Ok(BfTreeTableScan { iter, prefix_len })
    }
}

// ---------------------------------------------------------------------------
// BfTreeRangeIter -- Iterator adapter for storage trait range scans
// ---------------------------------------------------------------------------

/// Iterator over entries in a single Bf-Tree table, yielding typed `OwnedKv` pairs.
///
/// This wraps `BfTreeTableScan` (which uses a buffer-based `next(&mut buf)` API)
/// into a standard `Iterator` that allocates owned bytes for each entry.
///
/// Supports optional start/end key filtering for exclusive bounds, since
/// `crate::bf_tree::scan_with_end_key` uses inclusive bounds `[start, end]`.
pub struct BfTreeRangeIter<'a, K: Key + 'static, V: Value + 'static> {
    scan: BfTreeTableScan<'a>,
    buf: Vec<u8>,
    /// If set, skip entries whose raw (pre-strip) key matches this exactly (exclusive start).
    exclude_start: Option<Vec<u8>>,
    /// If set, skip entries whose raw (pre-strip) key matches this exactly (exclusive end).
    exclude_end: Option<Vec<u8>>,
    /// Verification mode for checksum unwrapping of values.
    verify_mode: Arc<VerifyMode>,
    _key: PhantomData<K>,
    _val: PhantomData<V>,
}

impl<'a, K: Key + 'static, V: Value + 'static> BfTreeRangeIter<'a, K, V> {
    fn new(
        scan: BfTreeTableScan<'a>,
        max_record_size: usize,
        exclude_start: Option<Vec<u8>>,
        exclude_end: Option<Vec<u8>>,
        verify_mode: Arc<VerifyMode>,
    ) -> Self {
        Self {
            scan,
            buf: vec![0u8; max_record_size * 2],
            exclude_start,
            exclude_end,
            verify_mode,
            _key: PhantomData,
            _val: PhantomData,
        }
    }
}

impl<K: Key + 'static, V: Value + 'static> Iterator for BfTreeRangeIter<'_, K, V> {
    type Item = crate::Result<(OwnedKv<K>, OwnedKv<V>)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (key_bytes, val_bytes) = self.scan.next(&mut self.buf)?;
            let key_ordered = key_bytes.to_vec();
            let val_owned = val_bytes.to_vec();

            // Check exclusion filters using byte-ordered key representation
            // (exclude values are already in ordered form).
            if let Some(ref excl) = self.exclude_start {
                match key_ordered.as_slice().cmp(excl.as_slice()) {
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
            if self
                .exclude_end
                .as_ref()
                .is_some_and(|excl| key_ordered == *excl)
            {
                return None; // Past the end boundary
            }

            // Reverse byte-order transformation before returning to caller.
            let mut key_user = key_ordered;
            K::from_byte_ordered_in_place(&mut key_user);
            let k = OwnedKv::new(key_user);
            let v = if self.verify_mode.is_enabled() {
                let verify = should_verify(self.verify_mode.as_ref());
                match unwrap_value(&val_owned, verify) {
                    Ok(data) => OwnedKv::new(data.to_vec()),
                    Err(e) => return Some(Err(e.into())),
                }
            } else {
                OwnedKv::new(val_owned)
            };
            return Some(Ok((k, v)));
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build a range scan from optional typed start/end keys
// ---------------------------------------------------------------------------

fn build_bf_range_scan<'a, K: Key + 'static, V: Value + 'static>(
    name: &str,
    adapter: &'a Arc<BfTreeAdapter>,
    start: Option<&K::SelfType<'_>>,
    end: Option<&K::SelfType<'_>>,
    start_inclusive: bool,
    end_inclusive: bool,
    verify_mode: Arc<VerifyMode>,
) -> crate::Result<BfTreeRangeIter<'a, K, V>> {
    // crate::bf_tree::scan_with_end_key uses inclusive bounds [start, end].
    // We handle exclusivity via iterator-level filtering on the user key.

    let (scan_start, exclude_start) = match start {
        Some(s) => {
            let mut s_bytes = K::as_bytes(s).as_ref().to_vec();
            K::to_byte_ordered_in_place(&mut s_bytes);
            let encoded = encode_table_key(name, TableKind::Regular, &s_bytes);
            if start_inclusive {
                (encoded, None)
            } else {
                // Exclude filter uses ordered bytes to match scan output.
                (encoded, Some(s_bytes))
            }
        }
        None => (table_prefix(name, TableKind::Regular), None),
    };

    let (scan_end, exclude_end) = match end {
        Some(e) => {
            let mut e_bytes = K::as_bytes(e).as_ref().to_vec();
            K::to_byte_ordered_in_place(&mut e_bytes);
            let encoded = encode_table_key(name, TableKind::Regular, &e_bytes);
            if end_inclusive {
                (encoded, None)
            } else {
                (encoded, Some(e_bytes))
            }
        }
        None => (table_prefix_end(name, TableKind::Regular), None),
    };

    let prefix_len = table_prefix(name, TableKind::Regular).len();
    let iter = adapter
        .scan_range(&scan_start, &scan_end)
        .map_err(crate::StorageError::from)?;
    let scan = BfTreeTableScan { iter, prefix_len };
    let max_record_size = adapter.inner().config().get_cb_max_record_size();
    Ok(BfTreeRangeIter::new(
        scan,
        max_record_size,
        exclude_start,
        exclude_end,
        verify_mode,
    ))
}

// ---------------------------------------------------------------------------
// Helper: build a buffered range scan (merge buffer + BfTree)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_buffered_range_scan<'a, K: Key + 'static, V: Value + 'static>(
    name: &str,
    adapter: &'a Arc<BfTreeAdapter>,
    buffer_mutex: &Mutex<WriteBuffer>,
    start: Option<&K::SelfType<'_>>,
    end: Option<&K::SelfType<'_>>,
    start_inclusive: bool,
    end_inclusive: bool,
    verify_mode: Arc<VerifyMode>,
) -> crate::Result<BufferedScanIter<'a, K, V>> {
    // BfTree scan uses inclusive bounds [start, end].
    // Exclusivity is handled at the iterator level via exclude_start/exclude_end.
    let (scan_start, exclude_start) = match start {
        Some(s) => {
            let mut s_bytes = K::as_bytes(s).as_ref().to_vec();
            K::to_byte_ordered_in_place(&mut s_bytes);
            let encoded = encode_table_key(name, TableKind::Regular, &s_bytes);
            if start_inclusive {
                (encoded, None)
            } else {
                (encoded, Some(s_bytes))
            }
        }
        None => (table_prefix(name, TableKind::Regular), None),
    };

    let (scan_end, exclude_end) = match end {
        Some(e) => {
            let mut e_bytes = K::as_bytes(e).as_ref().to_vec();
            K::to_byte_ordered_in_place(&mut e_bytes);
            let encoded = encode_table_key(name, TableKind::Regular, &e_bytes);
            if end_inclusive {
                (encoded, None)
            } else {
                (encoded, Some(e_bytes))
            }
        }
        None => (table_prefix_end(name, TableKind::Regular), None),
    };

    let prefix_len = table_prefix(name, TableKind::Regular).len();
    let iter = adapter
        .scan_range(&scan_start, &scan_end)
        .map_err(crate::StorageError::from)?;
    let scan = BfTreeTableScan { iter, prefix_len };
    let max_record_size = adapter.inner().config().get_cb_max_record_size();

    // Collect buffer entries for this range (prefix-stripped keys).
    let buf = buffer_mutex.lock();
    let buf_entries =
        collect_buffer_entries_for_table(&buf, name, TableKind::Regular, &scan_start, &scan_end);
    drop(buf);

    Ok(BufferedScanIter::new(
        buf_entries,
        scan,
        max_record_size,
        exclude_start,
        exclude_end,
        verify_mode,
    ))
}

// ---------------------------------------------------------------------------
// storage_traits::WriteTable for BfTreeTable
// ---------------------------------------------------------------------------

impl<K: Key + 'static, V: Value + 'static> crate::storage_traits::WriteTable<K, V>
    for BfTreeTable<'_, K, V>
{
    type RangeIter<'a>
        = BufferedScanIter<'a, K, V>
    where
        Self: 'a;

    fn st_get(&self, key: &K::SelfType<'_>) -> crate::Result<Option<OwnedKv<V>>> {
        // Trait requires `&self`; fall back to allocating read path.
        let encoded_key = encode_ordered_table_key::<K>(&self.name, TableKind::Regular, key);
        let checksumming = self.verify_mode.is_enabled();
        let blob_enabled = self.blob_threshold > 0;

        let buffer = self.buffer.lock();
        match buffer.get(&encoded_key) {
            BufferLookup::Found(v) => {
                let data = if blob_enabled && is_blob_handle(&v) {
                    let handle = BlobHandle::decode(&v).ok_or_else(|| {
                        BfTreeError::InvalidKV(alloc::string::String::from(
                            "corrupt blob handle in st_get buffer",
                        ))
                    })?;
                    value_indirection::read_blob(&handle, &buffer, self.adapter)?
                } else {
                    v
                };
                drop(buffer);
                return if checksumming {
                    let verify = should_verify(self.verify_mode.as_ref());
                    Ok(Some(OwnedKv::new(unwrap_value(&data, verify)?.to_vec())))
                } else {
                    Ok(Some(OwnedKv::new(data)))
                };
            }
            BufferLookup::Tombstone => return Ok(None),
            BufferLookup::NotInBuffer => {}
        }
        drop(buffer);

        let max_val = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.adapter.read(&encoded_key, &mut buf) {
            Ok(len) => {
                let raw = &buf[..len as usize];
                if blob_enabled && is_blob_handle(raw) {
                    let handle = BlobHandle::decode(raw).ok_or_else(|| {
                        BfTreeError::InvalidKV(alloc::string::String::from(
                            "corrupt blob handle in st_get BfTree",
                        ))
                    })?;
                    let data = value_indirection::read_blob_readonly(&handle, self.adapter)?;
                    if checksumming {
                        let verify = should_verify(self.verify_mode.as_ref());
                        Ok(Some(OwnedKv::new(unwrap_value(&data, verify)?.to_vec())))
                    } else {
                        Ok(Some(OwnedKv::new(data)))
                    }
                } else if checksumming {
                    let verify = should_verify(self.verify_mode.as_ref());
                    Ok(Some(OwnedKv::new(unwrap_value(raw, verify)?.to_vec())))
                } else {
                    Ok(Some(OwnedKv::new(raw.to_vec())))
                }
            }
            Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn st_insert(
        &mut self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
    ) -> crate::Result<Option<OwnedKv<V>>> {
        match self.insert(key, value) {
            Ok(Some(bytes)) => Ok(Some(OwnedKv::new(bytes))),
            Ok(None) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn st_remove(&mut self, key: &K::SelfType<'_>) -> crate::Result<Option<OwnedKv<V>>> {
        match self.remove(key) {
            Ok(Some(bytes)) => Ok(Some(OwnedKv::new(bytes))),
            Ok(None) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn st_range<'a>(
        &'a self,
        start: Option<&K::SelfType<'_>>,
        end: Option<&K::SelfType<'_>>,
        start_inclusive: bool,
        end_inclusive: bool,
    ) -> crate::Result<Self::RangeIter<'a>> {
        build_buffered_range_scan::<K, V>(
            &self.name,
            self.adapter,
            self.buffer,
            start,
            end,
            start_inclusive,
            end_inclusive,
            Arc::clone(self.verify_mode),
        )
    }

    fn st_drain_all(&mut self) -> crate::Result<u64> {
        let prefix = table_prefix(&self.name, TableKind::Regular);
        let prefix_end = table_prefix_end(&self.name, TableKind::Regular);
        let max_record_size = self.adapter.inner().config().get_cb_max_record_size();
        let mut total_count = 0u64;

        // Loop to handle TOCTOU: between scanning BfTree keys and acquiring
        // the buffer lock to tombstone them, concurrent commits may insert
        // new keys via CAS. Each iteration drains what it sees; the loop
        // terminates when a full pass finds no new entries to drain.
        loop {
            // Collect all BfTree keys for this table (scan must complete
            // before we can modify the buffer, since ScanIter holds internal
            // locks).
            let bftree_encoded_keys = {
                let mut buf = vec![0u8; max_record_size * 2];
                let mut keys: Vec<Vec<u8>> = Vec::new();
                let mut iter = self
                    .adapter
                    .scan_range(&prefix, &prefix_end)
                    .map_err(crate::StorageError::from)?;
                while let Ok(Some((key_len, _val_len))) = iter.next(&mut buf) {
                    keys.push(buf[..key_len].to_vec());
                }
                keys
            };

            let mut buffer = self.buffer.lock();
            let pass_count = buffer.drain_table(&bftree_encoded_keys, &prefix, &prefix_end);
            drop(buffer);

            total_count = total_count.saturating_add(pass_count);

            if pass_count == 0 {
                break;
            }
        }

        self.ops_count.fetch_add(total_count, Ordering::Relaxed);

        // Record a single CDC drain event summarizing the bulk delete.
        if self.cdc_log.is_some() && total_count > 0 {
            self.record_cdc(CdcEvent {
                table_name: self.name.clone(),
                op: ChangeOp::Delete,
                key: Vec::new(),
                new_value: None,
                old_value: None,
            });
        }

        Ok(total_count)
    }
}

// ---------------------------------------------------------------------------
// storage_traits::ReadTable for BfTreeTable (writable tables are also readable)
// ---------------------------------------------------------------------------

impl<K: Key + 'static, V: Value + 'static> crate::storage_traits::ReadTable<K, V>
    for BfTreeTable<'_, K, V>
{
    type RangeIter<'a>
        = BufferedScanIter<'a, K, V>
    where
        Self: 'a;

    fn st_get(&self, key: &K::SelfType<'_>) -> crate::Result<Option<OwnedKv<V>>> {
        crate::storage_traits::WriteTable::st_get(self, key)
    }

    fn st_range<'a>(
        &'a self,
        start: Option<&K::SelfType<'_>>,
        end: Option<&K::SelfType<'_>>,
        start_inclusive: bool,
        end_inclusive: bool,
    ) -> crate::Result<Self::RangeIter<'a>> {
        crate::storage_traits::WriteTable::st_range(
            self,
            start,
            end,
            start_inclusive,
            end_inclusive,
        )
    }
}

// ---------------------------------------------------------------------------
// storage_traits::ReadTable for BfTreeReadOnlyTable
// ---------------------------------------------------------------------------

impl<K: Key + 'static, V: Value + 'static> crate::storage_traits::ReadTable<K, V>
    for BfTreeReadOnlyTable<'_, K, V>
{
    type RangeIter<'a>
        = BfTreeRangeIter<'a, K, V>
    where
        Self: 'a;

    fn st_get(&self, key: &K::SelfType<'_>) -> crate::Result<Option<OwnedKv<V>>> {
        // Trait requires `&self`; fall back to allocating read path.
        let encoded_key = encode_ordered_table_key::<K>(&self.name, TableKind::Regular, key);
        let checksumming = self.verify_mode.is_enabled();
        let blob_enabled = self.blob_threshold > 0;
        let max_val = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.adapter.read(&encoded_key, &mut buf) {
            Ok(len) => {
                let raw = &buf[..len as usize];
                if blob_enabled && is_blob_handle(raw) {
                    let handle = BlobHandle::decode(raw).ok_or_else(|| {
                        BfTreeError::InvalidKV(alloc::string::String::from(
                            "corrupt blob handle in readonly st_get",
                        ))
                    })?;
                    let data = value_indirection::read_blob_readonly(&handle, self.adapter)?;
                    if checksumming {
                        let verify = should_verify(self.verify_mode.as_ref());
                        Ok(Some(OwnedKv::new(unwrap_value(&data, verify)?.to_vec())))
                    } else {
                        Ok(Some(OwnedKv::new(data)))
                    }
                } else if checksumming {
                    let verify = should_verify(self.verify_mode.as_ref());
                    Ok(Some(OwnedKv::new(unwrap_value(raw, verify)?.to_vec())))
                } else {
                    Ok(Some(OwnedKv::new(raw.to_vec())))
                }
            }
            Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn st_range<'a>(
        &'a self,
        start: Option<&K::SelfType<'_>>,
        end: Option<&K::SelfType<'_>>,
        start_inclusive: bool,
        end_inclusive: bool,
    ) -> crate::Result<Self::RangeIter<'a>> {
        build_bf_range_scan::<K, V>(
            &self.name,
            self.adapter,
            start,
            end,
            start_inclusive,
            end_inclusive,
            Arc::clone(self.verify_mode),
        )
    }
}

// ---------------------------------------------------------------------------
// storage_traits::StorageWrite for BfTreeDatabaseWriteTxn
// ---------------------------------------------------------------------------

impl crate::storage_traits::StorageWrite for super::database::BfTreeDatabaseWriteTxn {
    type Table<'txn, K: Key + 'static, V: Value + 'static>
        = BfTreeTable<'txn, K, V>
    where
        Self: 'txn;

    fn open_storage_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: crate::TableDefinition<K, V>,
    ) -> crate::Result<Self::Table<'_, K, V>> {
        // Bypass table name validation: internal modules (IVF-PQ) use "__" prefixed
        // tables legitimately. The public `open_table()` retains the check.
        Ok(BfTreeTable::new(
            definition.name(),
            &self.adapter,
            &self.ops_count,
            self.cdc_log.as_ref(),
            &self.buffer,
            &self.verify_mode,
            &self.next_value_blob_id,
            self.blob_threshold,
        ))
    }
}

// ---------------------------------------------------------------------------
// storage_traits::StorageRead for BfTreeDatabaseReadTxn
// ---------------------------------------------------------------------------

impl crate::storage_traits::StorageRead for super::database::BfTreeDatabaseReadTxn {
    type Table<'txn, K: Key + 'static, V: Value + 'static>
        = BfTreeReadOnlyTable<'txn, K, V>
    where
        Self: 'txn;

    fn open_storage_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: crate::TableDefinition<K, V>,
    ) -> crate::Result<Self::Table<'_, K, V>> {
        // Bypass table name validation: internal modules (IVF-PQ) use "__" prefixed
        // tables legitimately. The public `open_table()` retains the check.
        Ok(BfTreeReadOnlyTable::new(
            definition.name(),
            &self.adapter,
            &self.verify_mode,
            self.blob_threshold,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TableDefinition;
    use crate::bf_tree_store::config::BfTreeConfig;
    use crate::bf_tree_store::database::BfTreeDatabase;

    const ITEMS: TableDefinition<&str, u64> = TableDefinition::new("items");

    #[test]
    fn open_table_insert_get() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        let prev = table.insert(&"apple", &10u64).unwrap();
        assert!(prev.is_none());

        let val = table.get(&"apple").unwrap().unwrap();
        assert_eq!(u64::from_le_bytes(val.as_slice().try_into().unwrap()), 10);
        drop(table);
        wtxn.commit().unwrap();
    }

    #[test]
    fn open_table_insert_returns_previous() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        table.insert(&"key", &1u64).unwrap();
        let prev = table.insert(&"key", &2u64).unwrap();
        assert!(prev.is_some());
        let prev_val = u64::from_le_bytes(prev.unwrap().as_slice().try_into().unwrap());
        assert_eq!(prev_val, 1);
        drop(table);
        wtxn.commit().unwrap();
    }

    #[test]
    fn open_table_remove() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        table.insert(&"temp", &99u64).unwrap();
        let removed = table.remove(&"temp").unwrap();
        assert!(removed.is_some());
        assert!(table.get(&"temp").unwrap().is_none());

        // Remove non-existent key returns None.
        let removed2 = table.remove(&"nope").unwrap();
        assert!(removed2.is_none());
        drop(table);
        wtxn.commit().unwrap();
    }

    #[test]
    fn read_only_table() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        // Write some data.
        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        table.insert(&"x", &42u64).unwrap();
        table.insert(&"y", &43u64).unwrap();
        drop(table);
        wtxn.commit().unwrap();

        // Read via read transaction.
        let rtxn = db.begin_read();
        let mut ro_table = rtxn.open_table(ITEMS).unwrap();
        assert!(ro_table.contains_key(&"x"));
        assert!(!ro_table.contains_key(&"z"));
        let val = ro_table.get(&"x").unwrap().unwrap();
        assert_eq!(u64::from_le_bytes(val.as_slice().try_into().unwrap()), 42);
    }

    #[test]
    fn table_scan_via_handle() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        table.insert(&"a", &1u64).unwrap();
        table.insert(&"b", &2u64).unwrap();
        table.insert(&"c", &3u64).unwrap();
        drop(table);
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let ro_table = rtxn.open_table(ITEMS).unwrap();
        let mut scan = ro_table.scan().unwrap();
        let mut buf = vec![0u8; 4096];
        let mut count = 0;
        while scan.next(&mut buf).is_some() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    // -----------------------------------------------------------------------
    // storage_traits tests
    // -----------------------------------------------------------------------

    use crate::storage_traits::{ReadTable, StorageRead, StorageWrite, WriteTable};

    #[test]
    fn st_write_table_get_insert_remove() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_storage_table(ITEMS).unwrap();

        // Insert via trait
        let prev = WriteTable::st_insert(&mut table, &"key1", &100u64).unwrap();
        assert!(prev.is_none());

        // Get via trait
        let val = WriteTable::st_get(&table, &"key1").unwrap().unwrap();
        assert_eq!(val.value(), 100u64);

        // Overwrite
        let prev = WriteTable::st_insert(&mut table, &"key1", &200u64).unwrap();
        assert!(prev.is_some());
        assert_eq!(prev.unwrap().value(), 100u64);

        // Remove via trait
        let removed = WriteTable::st_remove(&mut table, &"key1").unwrap();
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().value(), 200u64);

        // Remove non-existent
        let removed2 = WriteTable::st_remove(&mut table, &"missing").unwrap();
        assert!(removed2.is_none());

        drop(table);
        wtxn.commit().unwrap();
    }

    #[test]
    fn st_range_scan_full() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_storage_table(ITEMS).unwrap();

        WriteTable::st_insert(&mut table, &"a", &1u64).unwrap();
        WriteTable::st_insert(&mut table, &"b", &2u64).unwrap();
        WriteTable::st_insert(&mut table, &"c", &3u64).unwrap();
        WriteTable::st_insert(&mut table, &"d", &4u64).unwrap();

        // Full range scan
        let iter = WriteTable::st_range(&table, None, None, true, true).unwrap();
        let entries: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(entries.len(), 4);

        // Verify ordering (Bf-Tree scans are sorted by key bytes)
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.value()).collect();
        assert_eq!(keys, vec!["a", "b", "c", "d"]);

        drop(table);
        wtxn.commit().unwrap();
    }

    #[test]
    fn st_range_scan_bounded() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_storage_table(ITEMS).unwrap();

        for (k, v) in [("a", 1u64), ("b", 2), ("c", 3), ("d", 4), ("e", 5)] {
            WriteTable::st_insert(&mut table, &k, &v).unwrap();
        }

        // Inclusive start "b", exclusive end "d"
        let s = "b";
        let e = "d";
        let s_ref: &str = s;
        let e_ref: &str = e;
        let iter = WriteTable::st_range(&table, Some(&s_ref), Some(&e_ref), true, false).unwrap();
        let entries: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.value()).collect();
        assert_eq!(keys, vec!["b", "c"]);

        // Inclusive both sides "b"..="d"
        let iter = WriteTable::st_range(&table, Some(&s_ref), Some(&e_ref), true, true).unwrap();
        let entries: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.value()).collect();
        assert_eq!(keys, vec!["b", "c", "d"]);

        drop(table);
        wtxn.commit().unwrap();
    }

    #[test]
    fn st_drain_all() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_storage_table(ITEMS).unwrap();

        WriteTable::st_insert(&mut table, &"x", &10u64).unwrap();
        WriteTable::st_insert(&mut table, &"y", &20u64).unwrap();
        WriteTable::st_insert(&mut table, &"z", &30u64).unwrap();

        let count = WriteTable::st_drain_all(&mut table).unwrap();
        assert_eq!(count, 3);

        // Verify all gone
        assert!(WriteTable::st_get(&table, &"x").unwrap().is_none());
        assert!(WriteTable::st_get(&table, &"y").unwrap().is_none());

        drop(table);
        wtxn.commit().unwrap();
    }

    #[test]
    fn st_read_table_via_read_txn() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        // Write data
        let wtxn = db.begin_write();
        let mut table = wtxn.open_storage_table(ITEMS).unwrap();
        WriteTable::st_insert(&mut table, &"r1", &100u64).unwrap();
        WriteTable::st_insert(&mut table, &"r2", &200u64).unwrap();
        drop(table);
        wtxn.commit().unwrap();

        // Read via StorageRead
        let rtxn = db.begin_read();
        let ro_table = StorageRead::open_storage_table(&rtxn, ITEMS).unwrap();

        let v1 = ReadTable::st_get(&ro_table, &"r1").unwrap().unwrap();
        assert_eq!(v1.value(), 100u64);

        let v2 = ReadTable::st_get(&ro_table, &"r2").unwrap().unwrap();
        assert_eq!(v2.value(), 200u64);

        assert!(ReadTable::st_get(&ro_table, &"missing").unwrap().is_none());

        // Range scan via ReadTable
        let iter = ReadTable::st_range(&ro_table, None, None, true, true).unwrap();
        let entries: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn st_storage_write_open_table() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();

        // Use StorageWrite trait to open table
        let mut table = StorageWrite::open_storage_table(&wtxn, ITEMS).unwrap();
        WriteTable::st_insert(&mut table, &"trait_key", &42u64).unwrap();
        let val = WriteTable::st_get(&table, &"trait_key").unwrap().unwrap();
        assert_eq!(val.value(), 42u64);

        drop(table);
        wtxn.commit().unwrap();
    }

    #[test]
    fn merge_increments_existing_value() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let op = crate::merge::NumericAdd;

        // Insert initial value.
        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        table.insert(&"counter", &10u64).unwrap();
        drop(table);
        wtxn.commit().unwrap();

        // Merge: add 5 to existing.
        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        table.merge(&"counter", &5u64.to_le_bytes(), &op).unwrap();
        drop(table);
        wtxn.commit().unwrap();

        // Verify result is 15.
        let rtxn = db.begin_read();
        let mut ro_table = rtxn.open_table(ITEMS).unwrap();
        let val = ro_table.get(&"counter").unwrap().unwrap();
        let result = u64::from_le_bytes(val.as_slice().try_into().unwrap());
        assert_eq!(result, 15);
    }

    #[test]
    fn merge_on_nonexistent_key_uses_operand() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let op = crate::merge::NumericAdd;

        // Merge on key that doesn't exist.
        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        table.merge(&"fresh", &42u64.to_le_bytes(), &op).unwrap();
        drop(table);
        wtxn.commit().unwrap();

        // Verify it was set to the operand value.
        let rtxn = db.begin_read();
        let mut ro_table = rtxn.open_table(ITEMS).unwrap();
        let val = ro_table.get(&"fresh").unwrap().unwrap();
        let result = u64::from_le_bytes(val.as_slice().try_into().unwrap());
        assert_eq!(result, 42);
    }

    #[test]
    fn merge_returning_none_deletes_key() {
        use alloc::vec::Vec;
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        // Operator that always returns None (= delete).
        let delete_op = crate::merge::merge_fn(
            |_key: &[u8], _existing: Option<&[u8]>, _operand: &[u8]| -> Option<Vec<u8>> { None },
        );

        // Insert initial value.
        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        table.insert(&"doomed", &99u64).unwrap();
        drop(table);
        wtxn.commit().unwrap();

        // Merge with delete-op.
        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(ITEMS).unwrap();
        table.merge(&"doomed", &[0u8], &delete_op).unwrap();
        drop(table);
        wtxn.commit().unwrap();

        // Key should be gone.
        let rtxn = db.begin_read();
        let mut ro_table = rtxn.open_table(ITEMS).unwrap();
        assert!(ro_table.get(&"doomed").unwrap().is_none());
    }

    // -------------------------------------------------------------------
    // Blob indirection tests
    // -------------------------------------------------------------------

    const BLOB_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("blobs");

    fn blob_config() -> BfTreeConfig {
        let mut cfg = BfTreeConfig::new_memory(8);
        cfg.blob_threshold = 64;
        cfg
    }

    #[test]
    fn blob_insert_read_roundtrip() {
        let db = BfTreeDatabase::create(blob_config()).unwrap();
        let big_val: Vec<u8> = (0u8..=255).cycle().take(10_000).collect();

        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(BLOB_TABLE).unwrap();
        let prev = table.insert(&"big", &big_val.as_slice()).unwrap();
        assert!(prev.is_none());

        // Read-your-writes within the same txn (from buffer).
        let got = table.get(&"big").unwrap().unwrap();
        assert_eq!(got, big_val);
        drop(table);
        wtxn.commit().unwrap();

        // Read from a fresh read txn (from BfTree).
        let rtxn = db.begin_read();
        let mut ro = rtxn.open_table(BLOB_TABLE).unwrap();
        let got2 = ro.get(&"big").unwrap().unwrap();
        assert_eq!(got2, big_val);
    }

    #[test]
    fn blob_overwrite_returns_previous() {
        let db = BfTreeDatabase::create(blob_config()).unwrap();
        let val1: Vec<u8> = vec![0xAA; 5_000];
        let val2: Vec<u8> = vec![0xBB; 8_000];

        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(BLOB_TABLE).unwrap();
        table.insert(&"k", &val1.as_slice()).unwrap();
        let prev = table.insert(&"k", &val2.as_slice()).unwrap();
        assert_eq!(prev.unwrap(), val1);

        let got = table.get(&"k").unwrap().unwrap();
        assert_eq!(got, val2);
        drop(table);
        wtxn.commit().unwrap();
    }

    #[test]
    fn blob_remove_returns_value_and_deletes() {
        let db = BfTreeDatabase::create(blob_config()).unwrap();
        let big: Vec<u8> = vec![0xCC; 3_000];

        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(BLOB_TABLE).unwrap();
        table.insert(&"del", &big.as_slice()).unwrap();
        let removed = table.remove(&"del").unwrap();
        assert_eq!(removed.unwrap(), big);
        assert!(table.get(&"del").unwrap().is_none());
        drop(table);
        wtxn.commit().unwrap();
    }

    #[test]
    fn blob_small_values_not_chunked() {
        let db = BfTreeDatabase::create(blob_config()).unwrap();
        let small: Vec<u8> = vec![0x01; 32];

        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(BLOB_TABLE).unwrap();
        table.insert(&"tiny", &small.as_slice()).unwrap();
        let got = table.get(&"tiny").unwrap().unwrap();
        assert_eq!(got, small);
        drop(table);
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let mut ro = rtxn.open_table(BLOB_TABLE).unwrap();
        assert_eq!(ro.get(&"tiny").unwrap().unwrap(), small);
    }

    #[test]
    fn blob_read_after_commit_readonly() {
        let db = BfTreeDatabase::create(blob_config()).unwrap();
        let val: Vec<u8> = vec![0xDD; 100_000];

        // Commit a 100KB blob.
        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(BLOB_TABLE).unwrap();
        table.insert(&"huge", &val.as_slice()).unwrap();
        drop(table);
        wtxn.commit().unwrap();

        // Read-only txn should reassemble correctly.
        let rtxn = db.begin_read();
        let mut ro = rtxn.open_table(BLOB_TABLE).unwrap();
        let got = ro.get(&"huge").unwrap().unwrap();
        assert_eq!(got.len(), 100_000);
        assert_eq!(got, val);
    }
}
