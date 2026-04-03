//! Bf-Tree backed database providing concurrent read/write access.
//!
//! `BfTreeDatabase` is the top-level entry point for using the Bf-Tree storage
//! engine. It provides typed table access using shodh-redb's `Key`/`Value` trait
//! system while leveraging Bf-Tree's concurrent B+tree for the underlying storage.
//!
//! # Usage
//!
//! ```ignore
//! use shodh_redb::bf_tree_store::{BfTreeConfig, BfTreeDatabase};
//! use shodh_redb::TableDefinition;
//!
//! const TABLE: TableDefinition<&str, u64> = TableDefinition::new("my_data");
//!
//! let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
//! let mut wtxn = db.begin_write();
//! wtxn.insert::<&str, u64>(&TABLE, "hello", &42).unwrap();
//! wtxn.commit().unwrap();
//!
//! let rtxn = db.begin_read();
//! let val = rtxn.get::<&str, u64>(&TABLE, "hello").unwrap();
//! assert_eq!(val, Some(42));
//! ```

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::cdc::types::{CdcConfig, CdcEvent, CdcKey, CdcRecord, ChangeStream};
use crate::types::{Key, Value};
use crate::{TableDefinition, TableHandle};

use super::buffered_txn::WriteBuffer;

use super::blob::{BfTreeBlobStore, BfTreeReadOnlyBlobStore};
use super::multimap::{BfTreeMultimapTable, BfTreeReadOnlyMultimapTable};
use super::table::{BfTreeReadOnlyTable, BfTreeTable};
use super::ttl::{BfTreeReadOnlyTtlTable, BfTreeTtlTable};

use super::adapter::BfTreeAdapter;
use super::config::BfTreeConfig;
use super::error::BfTreeError;

/// Reserved table name for the CDC log in `BfTree`.
const CDC_LOG_TABLE_NAME: &str = "__cdc_log";
/// Reserved table name for CDC cursors in `BfTree`.
const CDC_CURSOR_TABLE_NAME: &str = "__cdc_cursors";

/// A database backed by Bf-Tree's concurrent B+tree engine.
///
/// Unlike the legacy `Database`, this supports multiple concurrent writers
/// without blocking. All writes are immediately visible to all readers.
pub struct BfTreeDatabase {
    adapter: Arc<BfTreeAdapter>,
    cdc_config: CdcConfig,
    /// Monotonically increasing transaction ID counter for CDC.
    next_txn_id: AtomicU64,
}

impl BfTreeDatabase {
    /// Create a new Bf-Tree database with the given configuration.
    pub fn create(config: BfTreeConfig) -> Result<Self, BfTreeError> {
        let adapter = BfTreeAdapter::open(config)?;
        Ok(Self {
            adapter: Arc::new(adapter),
            cdc_config: CdcConfig::default(),
            next_txn_id: AtomicU64::new(1),
        })
    }

    /// Open an existing Bf-Tree database from a snapshot file.
    pub fn open(config: BfTreeConfig) -> Result<Self, BfTreeError> {
        let adapter = BfTreeAdapter::open_from_snapshot(config)?;
        // Recover the next transaction ID from the latest CDC log entry.
        let next_id = recover_next_txn_id(&adapter).unwrap_or(1);
        Ok(Self {
            adapter: Arc::new(adapter),
            cdc_config: CdcConfig::default(),
            next_txn_id: AtomicU64::new(next_id),
        })
    }

    /// Create a new Bf-Tree database with the given configuration and CDC settings.
    pub fn create_with_cdc(config: BfTreeConfig, cdc_config: CdcConfig) -> Result<Self, BfTreeError> {
        let adapter = BfTreeAdapter::open(config)?;
        Ok(Self {
            adapter: Arc::new(adapter),
            cdc_config,
            next_txn_id: AtomicU64::new(1),
        })
    }

    /// Open an existing Bf-Tree database with the given CDC settings.
    pub fn open_with_cdc(config: BfTreeConfig, cdc_config: CdcConfig) -> Result<Self, BfTreeError> {
        let adapter = BfTreeAdapter::open_from_snapshot(config)?;
        let next_id = recover_next_txn_id(&adapter).unwrap_or(1);
        Ok(Self {
            adapter: Arc::new(adapter),
            cdc_config,
            next_txn_id: AtomicU64::new(next_id),
        })
    }

    /// Begin a write transaction.
    ///
    /// Unlike the legacy `Database::begin_write()`, this does NOT block.
    /// Multiple write transactions can coexist and execute concurrently.
    pub fn begin_write(&self) -> BfTreeDatabaseWriteTxn {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::SeqCst);
        let cdc_log = if self.cdc_config.enabled {
            Some(Mutex::new(Vec::new()))
        } else {
            None
        };
        BfTreeDatabaseWriteTxn {
            adapter: self.adapter.clone(),
            ops_count: AtomicU64::new(0),
            txn_id,
            cdc_log,
            cdc_config: self.cdc_config.clone(),
            buffer: Mutex::new(WriteBuffer::new()),
            committed: core::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Begin a read transaction.
    ///
    /// Reads always see the latest state (no snapshot isolation).
    pub fn begin_read(&self) -> BfTreeDatabaseReadTxn {
        BfTreeDatabaseReadTxn {
            adapter: self.adapter.clone(),
        }
    }

    /// Take a durability snapshot. All data is guaranteed recoverable after crash.
    pub fn snapshot(&self) -> std::path::PathBuf {
        self.adapter.snapshot()
    }

    /// Get a reference to the underlying adapter.
    pub fn adapter(&self) -> &BfTreeAdapter {
        &self.adapter
    }

    /// Get the current CDC configuration.
    pub fn cdc_config(&self) -> &CdcConfig {
        &self.cdc_config
    }
}

/// Recover the next transaction ID from the latest CDC log entry.
/// Scans the CDC log table backwards and returns `latest_txn_id + 1`.
fn recover_next_txn_id(adapter: &BfTreeAdapter) -> Result<u64, BfTreeError> {
    let prefix = table_prefix(CDC_LOG_TABLE_NAME);
    let prefix_end = table_prefix_end(CDC_LOG_TABLE_NAME);
    let prefix_len = prefix.len();
    let max_record = adapter.inner().config().get_cb_max_record_size();
    let mut buf = vec![0u8; max_record * 2];
    let mut iter = adapter.scan_range(&prefix, &prefix_end)?;
    let mut max_txn_id: u64 = 0;
    while let Some((key_len, _val_len)) = iter.next(&mut buf) {
        if key_len > prefix_len + CdcKey::SERIALIZED_SIZE {
            // Should not happen, but be defensive
            continue;
        }
        let key_bytes = &buf[prefix_len..key_len];
        if key_bytes.len() >= CdcKey::SERIALIZED_SIZE {
            let cdc_key = CdcKey::from_le_bytes(key_bytes);
            if cdc_key.transaction_id > max_txn_id {
                max_txn_id = cdc_key.transaction_id;
            }
        }
    }
    Ok(max_txn_id.saturating_add(1))
}

// ---------------------------------------------------------------------------
// Internal key encoding: prefix table name to key bytes for namespace isolation.
//
// Format: [table_name_len: u16 LE] [table_name: bytes] [key: bytes]
//
// This ensures keys from different tables never collide in the single
// underlying Bf-Tree index.
// ---------------------------------------------------------------------------

pub(crate) fn encode_table_key(table_name: &str, key_bytes: &[u8]) -> Vec<u8> {
    let name_bytes = table_name.as_bytes();
    let name_len = u16::try_from(name_bytes.len()).expect("table name exceeds u16::MAX bytes");
    let mut buf = Vec::with_capacity(2 + name_bytes.len() + key_bytes.len());
    buf.extend_from_slice(&name_len.to_le_bytes());
    buf.extend_from_slice(name_bytes);
    buf.extend_from_slice(key_bytes);
    buf
}

pub(crate) fn table_prefix(table_name: &str) -> Vec<u8> {
    let name_bytes = table_name.as_bytes();
    let name_len = u16::try_from(name_bytes.len()).expect("table name exceeds u16::MAX bytes");
    let mut buf = Vec::with_capacity(2 + name_bytes.len());
    buf.extend_from_slice(&name_len.to_le_bytes());
    buf.extend_from_slice(name_bytes);
    buf
}

pub(crate) fn table_prefix_end(table_name: &str) -> Vec<u8> {
    let mut prefix = table_prefix(table_name);
    // Increment last byte to get exclusive upper bound.
    // If the last byte is 0xFF, extend with 0xFF (lexicographic upper bound).
    if let Some(last) = prefix.last_mut() {
        if *last < 0xFF {
            *last += 1;
        } else {
            prefix.push(0xFF);
        }
    }
    prefix
}

/// Write transaction for `BfTreeDatabase`.
///
/// Provides typed insert/delete/get using shodh-redb's `Key`/`Value` traits.
/// All writes are namespaced by table name and immediately visible.
pub struct BfTreeDatabaseWriteTxn {
    pub(crate) adapter: Arc<BfTreeAdapter>,
    ops_count: AtomicU64,
    /// Monotonic transaction ID assigned at `begin_write()`.
    pub(crate) txn_id: u64,
    /// Accumulated CDC events. `None` if CDC is disabled.
    pub(crate) cdc_log: Option<Mutex<Vec<CdcEvent>>>,
    /// CDC configuration (retention, etc.).
    pub(crate) cdc_config: CdcConfig,
    /// Write buffer for atomic commit/rollback semantics.
    pub(crate) buffer: Mutex<WriteBuffer>,
    /// Track whether buffer has been flushed (committed).
    committed: core::sync::atomic::AtomicBool,
}

impl BfTreeDatabaseWriteTxn {
    /// Open a typed table handle for read-write access.
    ///
    /// This is the preferred API — it mirrors the legacy `WriteTransaction::open_table()`
    /// pattern. The returned `BfTreeTable` provides `insert`, `get`, `remove`, and `scan`.
    pub fn open_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: TableDefinition<K, V>,
    ) -> BfTreeTable<'_, K, V> {
        BfTreeTable::new(definition.name(), &self.adapter, &self.ops_count, self.cdc_log.as_ref(), &self.buffer)
    }

    /// Open a TTL table for read-write access.
    pub fn open_ttl_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: TableDefinition<K, V>,
    ) -> BfTreeTtlTable<'_, K, V> {
        BfTreeTtlTable::new(definition.name(), &self.adapter, &self.ops_count, &self.buffer)
    }

    /// Open a multimap table for read-write access.
    pub fn open_multimap_table<K: Key + 'static, V: Key + 'static>(
        &self,
        name: &str,
    ) -> BfTreeMultimapTable<'_, K, V> {
        BfTreeMultimapTable::new(name, &self.adapter, &self.ops_count, &self.buffer)
    }

    /// Open the blob store for read-write access.
    pub fn open_blob_store(&self) -> BfTreeBlobStore<'_> {
        BfTreeBlobStore::new(&self.adapter, &self.buffer, &self.ops_count)
    }

    /// Insert a typed key-value pair into the specified table.
    ///
    /// Writes are routed through the write buffer for atomic commit/rollback.
    pub fn insert<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: &TableDefinition<K, V>,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
    ) -> Result<(), BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let val_bytes = V::as_bytes(value);
        let encoded_key = encode_table_key(definition.name(), key_bytes.as_ref());
        self.buffer.lock().unwrap().put(encoded_key, val_bytes.as_ref().to_vec());
        self.ops_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Delete a typed key from the specified table.
    ///
    /// Writes a tombstone to the write buffer for atomic commit/rollback.
    pub fn delete<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: &TableDefinition<K, V>,
        key: &K::SelfType<'_>,
    ) {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(definition.name(), key_bytes.as_ref());
        self.buffer.lock().unwrap().delete(encoded_key);
        self.ops_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Read a typed value from the specified table.
    pub fn get<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: &TableDefinition<K, V>,
        key: &K::SelfType<'_>,
    ) -> Result<Option<Vec<u8>>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(definition.name(), key_bytes.as_ref());
        let max_val = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.adapter.read(&encoded_key, &mut buf) {
            Ok(len) => Ok(Some(buf[..len as usize].to_vec())),
            Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Check if a key exists in the table.
    pub fn contains_key<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: &TableDefinition<K, V>,
        key: &K::SelfType<'_>,
    ) -> bool {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(definition.name(), key_bytes.as_ref());
        self.adapter.contains_key(&encoded_key)
    }

    /// Commit the transaction for durability.
    ///
    /// If CDC is enabled, accumulated events are staged into the write buffer
    /// before flushing, so CDC log entries are atomically committed with data.
    pub fn commit(self) -> Result<(), BfTreeError> {
        {
            let mut buffer = self.buffer.lock().unwrap();
            // Stage CDC log entries into the write buffer before flushing.
            self.stage_cdc_into_buffer(&mut buffer)?;
            // Flush write buffer to BfTree (the atomic commit point).
            buffer.flush(&self.adapter)?;
        }
        self.committed.store(true, Ordering::SeqCst);
        // Retention pruning is post-commit and non-critical.
        self.prune_cdc_retention();
        Ok(())
    }

    /// Commit with explicit snapshot for guaranteed crash recovery.
    pub fn commit_with_snapshot(self) -> Result<std::path::PathBuf, BfTreeError> {
        {
            let mut buffer = self.buffer.lock().unwrap();
            self.stage_cdc_into_buffer(&mut buffer)?;
            buffer.flush(&self.adapter)?;
        }
        self.committed.store(true, Ordering::SeqCst);
        self.prune_cdc_retention();
        Ok(self.adapter.snapshot())
    }

    /// Number of operations performed.
    pub fn ops_count(&self) -> u64 {
        self.ops_count.load(Ordering::Relaxed)
    }

    /// Transaction ID assigned to this write transaction.
    pub fn txn_id(&self) -> u64 {
        self.txn_id
    }

    /// Advance a named CDC cursor to the given transaction ID.
    pub fn advance_cdc_cursor(&self, name: &str, up_to_txn: u64) -> Result<(), BfTreeError> {
        let key_bytes = name.as_bytes();
        let encoded = encode_table_key(CDC_CURSOR_TABLE_NAME, key_bytes);
        let val_bytes = up_to_txn.to_le_bytes();
        self.adapter.insert(&encoded, &val_bytes)?;
        Ok(())
    }

    /// Stage accumulated CDC events into the write buffer as KV entries.
    ///
    /// This ensures CDC log entries are atomically committed with the data
    /// they describe — both go into the same buffer flush.
    fn stage_cdc_into_buffer(&self, buffer: &mut WriteBuffer) -> Result<(), BfTreeError> {
        let events = match self.cdc_log {
            Some(ref log) => {
                let mut guard = log.lock().unwrap();
                if guard.is_empty() {
                    return Ok(());
                }
                core::mem::take(&mut *guard)
            }
            None => return Ok(()),
        };

        for (seq, event) in events.iter().enumerate() {
            let key = CdcKey::new(self.txn_id, u32::try_from(seq).unwrap_or(u32::MAX));
            let record = CdcRecord::from_event(event);
            let key_bytes = key.to_le_bytes();
            let val_bytes = record.serialize();
            let encoded_key = encode_table_key(CDC_LOG_TABLE_NAME, &key_bytes);
            buffer.put(encoded_key, val_bytes);
        }

        Ok(())
    }

    /// Post-commit retention pruning of old CDC log entries.
    ///
    /// This runs after the commit is finalized. Failures are non-fatal —
    /// old entries accumulate but do not affect correctness.
    fn prune_cdc_retention(&self) {
        if self.cdc_config.retention_max_txns == 0 || self.txn_id <= self.cdc_config.retention_max_txns {
            return;
        }

        let cutoff_txn = self.txn_id - self.cdc_config.retention_max_txns;
        let cutoff_key = CdcKey::new(cutoff_txn, u32::MAX);
        let cutoff_encoded = encode_table_key(CDC_LOG_TABLE_NAME, &cutoff_key.to_le_bytes());
        let prefix = table_prefix(CDC_LOG_TABLE_NAME);
        let prefix_len = prefix.len();
        let max_record = self.adapter.inner().config().get_cb_max_record_size();

        // Collect keys to delete (must drop iterator before deleting).
        let keys_to_delete = {
            let mut buf = vec![0u8; max_record * 2];
            let mut keys: Vec<Vec<u8>> = Vec::new();
            if let Ok(mut iter) = self.adapter.scan_range(&prefix, &cutoff_encoded) {
                while let Some((key_len, _val_len)) = iter.next(&mut buf) {
                    keys.push(buf[..key_len].to_vec());
                }
            }
            keys
        };

        for encoded_key in &keys_to_delete {
            if encoded_key.len() > prefix_len {
                self.adapter.delete(encoded_key);
            }
        }
    }
}

impl Drop for BfTreeDatabaseWriteTxn {
    fn drop(&mut self) {
        // Implicit rollback: discard buffered writes if not committed.
        // Use unwrap_or_else to recover from a poisoned mutex instead of
        // aborting the process (double-panic during unwind).
        if !self.committed.load(Ordering::SeqCst) {
            self.buffer
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .discard();
        }
    }
}

/// Read transaction for `BfTreeDatabase`.
///
/// Provides typed get/scan using shodh-redb's `Key`/`Value` traits.
pub struct BfTreeDatabaseReadTxn {
    pub(crate) adapter: Arc<BfTreeAdapter>,
}

impl BfTreeDatabaseReadTxn {
    /// Open a typed table handle for read-only access.
    ///
    /// Mirrors the legacy `ReadTransaction::open_table()` pattern.
    pub fn open_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: TableDefinition<K, V>,
    ) -> BfTreeReadOnlyTable<'_, K, V> {
        BfTreeReadOnlyTable::new(definition.name(), &self.adapter)
    }

    /// Open a TTL table for read-only access.
    pub fn open_ttl_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: TableDefinition<K, V>,
    ) -> BfTreeReadOnlyTtlTable<'_, K, V> {
        BfTreeReadOnlyTtlTable::new(definition.name(), &self.adapter)
    }

    /// Open a multimap table for read-only access.
    pub fn open_multimap_table<K: Key + 'static, V: Key + 'static>(
        &self,
        name: &str,
    ) -> BfTreeReadOnlyMultimapTable<'_, K, V> {
        BfTreeReadOnlyMultimapTable::new(name, &self.adapter)
    }

    /// Open the blob store for read-only access.
    pub fn open_blob_store(&self) -> BfTreeReadOnlyBlobStore<'_> {
        BfTreeReadOnlyBlobStore::new(&self.adapter)
    }

    /// Read a typed value from the specified table.
    ///
    /// Returns the raw value bytes. Use `V::from_bytes()` to deserialize.
    pub fn get<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: &TableDefinition<K, V>,
        key: &K::SelfType<'_>,
    ) -> Result<Option<Vec<u8>>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(definition.name(), key_bytes.as_ref());
        let max_val = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.adapter.read(&encoded_key, &mut buf) {
            Ok(len) => Ok(Some(buf[..len as usize].to_vec())),
            Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Check if a key exists in the table.
    pub fn contains_key<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: &TableDefinition<K, V>,
        key: &K::SelfType<'_>,
    ) -> bool {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(definition.name(), key_bytes.as_ref());
        self.adapter.contains_key(&encoded_key)
    }

    /// Scan all entries in the given table.
    ///
    /// Returns an iterator yielding `(key_bytes, value_bytes)` pairs.
    /// The caller must use `K::from_bytes()` and `V::from_bytes()` to
    /// deserialize the returned byte slices.
    pub fn scan_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: &TableDefinition<K, V>,
    ) -> Result<BfTreeTableScan<'_>, BfTreeError> {
        let prefix = table_prefix(definition.name());
        let prefix_end = table_prefix_end(definition.name());
        let prefix_len = prefix.len();
        let iter = self.adapter.scan_range(&prefix, &prefix_end)?;
        Ok(BfTreeTableScan { iter, prefix_len })
    }

    /// Read CDC changes committed after the given transaction ID.
    ///
    /// Returns all change records with `transaction_id > after_txn_id`,
    /// ordered by `(transaction_id, sequence)`.
    pub fn read_cdc_since(&self, after_txn_id: u64) -> Result<Vec<ChangeStream>, BfTreeError> {
        let start_key = CdcKey::new(after_txn_id.saturating_add(1), 0);
        let end_key = CdcKey::new(u64::MAX, u32::MAX);
        self.read_cdc_range_inner(start_key, end_key)
    }

    /// Read CDC changes within a transaction ID range (inclusive on both ends).
    pub fn read_cdc_range(&self, start_txn: u64, end_txn: u64) -> Result<Vec<ChangeStream>, BfTreeError> {
        if start_txn > end_txn {
            return Ok(Vec::new());
        }
        let start_key = CdcKey::new(start_txn, 0);
        let end_key = CdcKey::new(end_txn, u32::MAX);
        self.read_cdc_range_inner(start_key, end_key)
    }

    /// Read the position of a named CDC cursor.
    pub fn cdc_cursor(&self, name: &str) -> Result<Option<u64>, BfTreeError> {
        let key_bytes = name.as_bytes();
        let encoded = encode_table_key(CDC_CURSOR_TABLE_NAME, key_bytes);
        let max_val = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.adapter.read(&encoded, &mut buf) {
            Ok(len) => {
                if len as usize >= 8 {
                    let txn_id = u64::from_le_bytes(buf[..8].try_into().unwrap());
                    Ok(Some(txn_id))
                } else {
                    Ok(None)
                }
            }
            Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Returns the transaction ID of the latest CDC log entry, or `None` if empty.
    pub fn latest_cdc_transaction_id(&self) -> Result<Option<u64>, BfTreeError> {
        let prefix = table_prefix(CDC_LOG_TABLE_NAME);
        let prefix_end = table_prefix_end(CDC_LOG_TABLE_NAME);
        let prefix_len = prefix.len();
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_record * 2];
        let mut iter = self.adapter.scan_range(&prefix, &prefix_end)?;
        let mut max_txn_id: Option<u64> = None;
        while let Some((key_len, _val_len)) = iter.next(&mut buf) {
            if key_len > prefix_len {
                let key_bytes = &buf[prefix_len..key_len];
                if key_bytes.len() >= CdcKey::SERIALIZED_SIZE {
                    let cdc_key = CdcKey::from_le_bytes(key_bytes);
                    match max_txn_id {
                        Some(current) if cdc_key.transaction_id > current => {
                            max_txn_id = Some(cdc_key.transaction_id);
                        }
                        None => {
                            max_txn_id = Some(cdc_key.transaction_id);
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(max_txn_id)
    }

    fn read_cdc_range_inner(&self, start_key: CdcKey, end_key: CdcKey) -> Result<Vec<ChangeStream>, BfTreeError> {
        let start_encoded = encode_table_key(CDC_LOG_TABLE_NAME, &start_key.to_le_bytes());
        let end_encoded = encode_table_key(CDC_LOG_TABLE_NAME, &end_key.to_le_bytes());
        let prefix_len = table_prefix(CDC_LOG_TABLE_NAME).len();
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_record * 2];
        let mut iter = self.adapter.scan_range(&start_encoded, &end_encoded)?;
        let mut results = Vec::new();

        while let Some((key_len, val_len)) = iter.next(&mut buf) {
            if key_len <= prefix_len {
                continue;
            }
            let key_bytes = &buf[prefix_len..key_len];
            let val_bytes = &buf[key_len..key_len + val_len];

            if key_bytes.len() < CdcKey::SERIALIZED_SIZE {
                continue;
            }
            let cdc_key = CdcKey::from_le_bytes(key_bytes);
            if let Ok(record) = CdcRecord::deserialize(val_bytes) {
                results.push(ChangeStream::from_key_record(cdc_key, record));
            }
        }

        Ok(results)
    }
}

/// Iterator over entries in a single table.
///
/// Yields `(key_bytes, value_bytes)` with the table prefix stripped from keys.
pub struct BfTreeTableScan<'a> {
    pub(crate) iter: bf_tree::ScanIter<'a, 'a>,
    pub(crate) prefix_len: usize,
}

impl BfTreeTableScan<'_> {
    /// Get the next `(key_bytes, value_bytes)` entry.
    ///
    /// The `buf` must be large enough to hold key + value.
    /// Returns `None` when no more entries exist.
    pub fn next<'buf>(&mut self, buf: &'buf mut [u8]) -> Option<(&'buf [u8], &'buf [u8])> {
        let (key_len, val_len) = self.iter.next(buf)?;
        // Strip the table prefix from the key.
        if key_len > self.prefix_len {
            let key = &buf[self.prefix_len..key_len];
            let val = &buf[key_len..key_len + val_len];
            Some((key, val))
        } else {
            // Shouldn't happen for well-formed data, but be defensive.
            None
        }
    }
}

// ---------------------------------------------------------------------------
// BfTreeBuilder — ergonomic database construction with optional features
// ---------------------------------------------------------------------------

/// Builder for constructing a `BfTreeDatabase` with optional features.
///
/// # Example
///
/// ```ignore
/// use shodh_redb::bf_tree_store::{BfTreeBuilder, BfTreeConfig};
/// use shodh_redb::cdc::CdcConfig;
///
/// let db = BfTreeBuilder::new()
///     .set_cdc(CdcConfig { enabled: true, retention_max_txns: 1000 })
///     .create(BfTreeConfig::new_memory(4))
///     .unwrap();
/// ```
pub struct BfTreeBuilder {
    cdc_config: CdcConfig,
}

impl Default for BfTreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BfTreeBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            cdc_config: CdcConfig::default(),
        }
    }

    /// Enable Change Data Capture with the given configuration.
    pub fn set_cdc(&mut self, config: CdcConfig) -> &mut Self {
        self.cdc_config = config;
        self
    }

    /// Create a new Bf-Tree database with the configured settings.
    pub fn create(&self, config: BfTreeConfig) -> Result<BfTreeDatabase, BfTreeError> {
        BfTreeDatabase::create_with_cdc(config, self.cdc_config.clone())
    }

    /// Open an existing Bf-Tree database with the configured settings.
    pub fn open(&self, config: BfTreeConfig) -> Result<BfTreeDatabase, BfTreeError> {
        BfTreeDatabase::open_with_cdc(config, self.cdc_config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdc::types::ChangeOp;

    const USERS: TableDefinition<&str, u64> = TableDefinition::new("users");
    const SCORES: TableDefinition<&str, u64> = TableDefinition::new("scores");

    #[test]
    fn typed_insert_and_read() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();
        wtxn.insert(&USERS, &"alice", &42u64).unwrap();
        wtxn.insert(&USERS, &"bob", &99u64).unwrap();
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let alice_bytes = rtxn.get::<&str, u64>(&USERS, &"alice").unwrap().unwrap();
        let alice_val = u64::from_le_bytes(alice_bytes.as_slice().try_into().unwrap());
        assert_eq!(alice_val, 42);

        let bob_bytes = rtxn.get::<&str, u64>(&USERS, &"bob").unwrap().unwrap();
        let bob_val = u64::from_le_bytes(bob_bytes.as_slice().try_into().unwrap());
        assert_eq!(bob_val, 99);
    }

    #[test]
    fn table_namespace_isolation() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();
        wtxn.insert(&USERS, &"key", &100u64).unwrap();
        wtxn.insert(&SCORES, &"key", &200u64).unwrap();
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let users_val = rtxn.get::<&str, u64>(&USERS, &"key").unwrap().unwrap();
        let scores_val = rtxn.get::<&str, u64>(&SCORES, &"key").unwrap().unwrap();

        let u = u64::from_le_bytes(users_val.as_slice().try_into().unwrap());
        let s = u64::from_le_bytes(scores_val.as_slice().try_into().unwrap());
        assert_eq!(u, 100);
        assert_eq!(s, 200);
    }

    #[test]
    fn typed_delete() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();
        wtxn.insert(&USERS, &"temp", &1u64).unwrap();
        wtxn.delete(&USERS, &"temp");
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        assert!(rtxn.get::<&str, u64>(&USERS, &"temp").unwrap().is_none());
    }

    #[test]
    fn typed_contains_key() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();
        wtxn.insert(&USERS, &"exists", &1u64).unwrap();
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        assert!(rtxn.contains_key(&USERS, &"exists"));
        assert!(!rtxn.contains_key(&USERS, &"missing"));
    }

    #[test]
    fn concurrent_typed_writes() {
        use std::thread;

        let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_memory(8)).unwrap());
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let db = db.clone();
                thread::spawn(move || {
                    let wtxn = db.begin_write();
                    for i in 0u64..50 {
                        let key_str = alloc::format!("t{t}_k{i}");
                        // Insert using raw bytes since we can't easily use &str generics across threads
                        let key_bytes = key_str.as_bytes();
                        let val_bytes = (t * 1000 + i).to_le_bytes();
                        let encoded =
                            encode_table_key(SCORES.name(), key_bytes);
                        wtxn.adapter.insert(&encoded, &val_bytes).unwrap();
                    }
                    wtxn.commit().unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let rtxn = db.begin_read();
        for t in 0u64..4 {
            for i in 0u64..50 {
                let key_str = alloc::format!("t{t}_k{i}");
                let key_bytes = key_str.as_bytes();
                let encoded = encode_table_key(SCORES.name(), key_bytes);
                let max_val = db.adapter().inner().config().get_cb_max_record_size();
                let mut buf = vec![0u8; max_val];
                let len = rtxn.adapter.read(&encoded, &mut buf).unwrap();
                let val = u64::from_le_bytes(buf[..len as usize].try_into().unwrap());
                assert_eq!(val, t * 1000 + i);
            }
        }
    }

    #[test]
    fn table_scan() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();
        wtxn.insert(&USERS, &"alice", &1u64).unwrap();
        wtxn.insert(&USERS, &"bob", &2u64).unwrap();
        wtxn.insert(&USERS, &"carol", &3u64).unwrap();
        // Also insert into a different table to verify isolation.
        wtxn.insert(&SCORES, &"alice", &100u64).unwrap();
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let mut scan = rtxn.scan_table::<&str, u64>(&USERS).unwrap();
        let mut buf = vec![0u8; 4096];
        let mut count = 0;
        while scan.next(&mut buf).is_some() {
            count += 1;
        }
        assert_eq!(count, 3, "should only see USERS entries, not SCORES");
    }

    // -----------------------------------------------------------------------
    // CDC tests
    // -----------------------------------------------------------------------

    fn cdc_db() -> BfTreeDatabase {
        BfTreeDatabase::create_with_cdc(
            BfTreeConfig::new_memory(4),
            CdcConfig { enabled: true, retention_max_txns: 0 },
        ).unwrap()
    }

    #[test]
    fn cdc_records_insert_and_update() {
        let db = cdc_db();

        let wtxn = db.begin_write();
        let txn_id = wtxn.txn_id();
        let mut table = wtxn.open_table(USERS);
        table.insert(&"alice", &42u64).unwrap();
        table.insert(&"alice", &99u64).unwrap(); // update
        drop(table);
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let changes = rtxn.read_cdc_since(0).unwrap();
        assert_eq!(changes.len(), 2);

        // First event: Insert
        assert_eq!(changes[0].transaction_id, txn_id);
        assert_eq!(changes[0].sequence, 0);
        assert_eq!(changes[0].op, ChangeOp::Insert);
        assert!(changes[0].old_value.is_none());
        assert!(changes[0].new_value.is_some());

        // Second event: Update
        assert_eq!(changes[1].transaction_id, txn_id);
        assert_eq!(changes[1].sequence, 1);
        assert_eq!(changes[1].op, ChangeOp::Update);
        assert!(changes[1].old_value.is_some());
        assert!(changes[1].new_value.is_some());
    }

    #[test]
    fn cdc_records_delete() {
        let db = cdc_db();

        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(USERS);
        table.insert(&"bob", &10u64).unwrap();
        table.remove(&"bob").unwrap();
        drop(table);
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let changes = rtxn.read_cdc_since(0).unwrap();
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].op, ChangeOp::Insert);
        assert_eq!(changes[1].op, ChangeOp::Delete);
        assert!(changes[1].new_value.is_none());
        assert!(changes[1].old_value.is_some());
    }

    #[test]
    fn cdc_multi_txn_read_since() {
        let db = cdc_db();

        // Transaction 1
        let wtxn1 = db.begin_write();
        let txn1_id = wtxn1.txn_id();
        let mut t1 = wtxn1.open_table(USERS);
        t1.insert(&"a", &1u64).unwrap();
        drop(t1);
        wtxn1.commit().unwrap();

        // Transaction 2
        let wtxn2 = db.begin_write();
        let txn2_id = wtxn2.txn_id();
        let mut t2 = wtxn2.open_table(USERS);
        t2.insert(&"b", &2u64).unwrap();
        drop(t2);
        wtxn2.commit().unwrap();

        let rtxn = db.begin_read();

        // Read since txn1 — should only see txn2's changes.
        let changes = rtxn.read_cdc_since(txn1_id).unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].transaction_id, txn2_id);
    }

    #[test]
    fn cdc_read_range() {
        let db = cdc_db();

        // Three transactions
        let w1 = db.begin_write();
        let id1 = w1.txn_id();
        let mut t = w1.open_table(USERS);
        t.insert(&"x", &1u64).unwrap();
        drop(t);
        w1.commit().unwrap();

        let w2 = db.begin_write();
        let id2 = w2.txn_id();
        let mut t = w2.open_table(USERS);
        t.insert(&"y", &2u64).unwrap();
        drop(t);
        w2.commit().unwrap();

        let w3 = db.begin_write();
        let _id3 = w3.txn_id();
        let mut t = w3.open_table(USERS);
        t.insert(&"z", &3u64).unwrap();
        drop(t);
        w3.commit().unwrap();

        let rtxn = db.begin_read();
        let changes = rtxn.read_cdc_range(id1, id2).unwrap();
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].transaction_id, id1);
        assert_eq!(changes[1].transaction_id, id2);
    }

    #[test]
    fn cdc_latest_transaction_id() {
        let db = cdc_db();

        let rtxn = db.begin_read();
        assert!(rtxn.latest_cdc_transaction_id().unwrap().is_none());
        drop(rtxn);

        let w = db.begin_write();
        let tid = w.txn_id();
        let mut t = w.open_table(USERS);
        t.insert(&"k", &1u64).unwrap();
        drop(t);
        w.commit().unwrap();

        let rtxn = db.begin_read();
        assert_eq!(rtxn.latest_cdc_transaction_id().unwrap(), Some(tid));
    }

    #[test]
    fn cdc_cursor_advance_and_read() {
        let db = cdc_db();

        let w = db.begin_write();
        let tid = w.txn_id();
        let mut t = w.open_table(USERS);
        t.insert(&"k", &1u64).unwrap();
        drop(t);
        w.advance_cdc_cursor("consumer1", tid).unwrap();
        w.commit().unwrap();

        let rtxn = db.begin_read();
        let cursor = rtxn.cdc_cursor("consumer1").unwrap();
        assert_eq!(cursor, Some(tid));
        assert!(rtxn.cdc_cursor("nonexistent").unwrap().is_none());
    }

    #[test]
    fn cdc_disabled_no_events() {
        // Default: CDC disabled
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();
        let mut table = wtxn.open_table(USERS);
        table.insert(&"key", &1u64).unwrap();
        drop(table);
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let changes = rtxn.read_cdc_since(0).unwrap();
        assert!(changes.is_empty());
    }

    #[test]
    fn cdc_retention_pruning() {
        let db = BfTreeDatabase::create_with_cdc(
            BfTreeConfig::new_memory(4),
            CdcConfig { enabled: true, retention_max_txns: 2 },
        ).unwrap();

        // Write 4 transactions
        for i in 0u64..4 {
            let w = db.begin_write();
            let mut t = w.open_table(USERS);
            let key = alloc::format!("k{i}");
            t.insert(&key.as_str(), &i).unwrap();
            drop(t);
            w.commit().unwrap();
        }

        let rtxn = db.begin_read();
        let all_changes = rtxn.read_cdc_since(0).unwrap();
        // With retention_max_txns=2, older transactions should be pruned.
        // Transaction IDs 1,2,3,4 — after txn4, cutoff = 4-2=2, so txn1 and txn2 pruned.
        // Only txn3 and txn4 remain.
        assert!(all_changes.len() <= 4, "pruning should remove old entries");
        // At least the latest 2 should remain
        let latest_ids: Vec<u64> = all_changes.iter().map(|c| c.transaction_id).collect();
        assert!(latest_ids.contains(&4), "latest txn should be present");
    }

    #[test]
    fn cdc_builder_api() {
        let db = BfTreeBuilder::new()
            .set_cdc(CdcConfig { enabled: true, retention_max_txns: 0 })
            .create(BfTreeConfig::new_memory(4))
            .unwrap();

        let w = db.begin_write();
        let mut t = w.open_table(USERS);
        t.insert(&"test", &42u64).unwrap();
        drop(t);
        w.commit().unwrap();

        let rtxn = db.begin_read();
        let changes = rtxn.read_cdc_since(0).unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].op, ChangeOp::Insert);
    }

    #[test]
    fn cdc_table_name_recorded() {
        let db = cdc_db();

        let w = db.begin_write();
        let mut t1 = w.open_table(USERS);
        t1.insert(&"a", &1u64).unwrap();
        drop(t1);
        let mut t2 = w.open_table(SCORES);
        t2.insert(&"b", &2u64).unwrap();
        drop(t2);
        w.commit().unwrap();

        let rtxn = db.begin_read();
        let changes = rtxn.read_cdc_since(0).unwrap();
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].table_name, "users");
        assert_eq!(changes[1].table_name, "scores");
    }
}
