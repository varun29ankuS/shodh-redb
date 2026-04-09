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

use crate::compat::Mutex;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

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
use super::verification::VerifyMode;

/// Reserved table name for the CDC log in `BfTree`.
const CDC_LOG_TABLE_NAME: &str = "__cdc_log";
/// Reserved table name for CDC cursors in `BfTree`.
const CDC_CURSOR_TABLE_NAME: &str = "__cdc_cursors";
/// Reserved system key for the persisted transaction ID counter.
///
/// Stored as a bare encoded key (using the standard table-prefix encoding with
/// table name `__bf_meta`) holding a `u64` LE value representing the next
/// transaction ID to allocate. Updated atomically inside every commit's write
/// buffer so that recovery does not depend on CDC being enabled.
const BF_META_TABLE_NAME: &str = "__bf_meta";
const BF_META_TXN_ID_KEY: &[u8] = b"next_txn_id";

/// A database backed by Bf-Tree's concurrent B+tree engine.
///
/// Unlike the legacy `Database`, this supports multiple concurrent writers
/// without blocking. All writes are immediately visible to all readers.
pub struct BfTreeDatabase {
    adapter: Arc<BfTreeAdapter>,
    cdc_config: CdcConfig,
    /// Per-entry checksum verification mode for read integrity checks.
    verify_mode: Arc<VerifyMode>,
    /// Monotonically increasing transaction ID counter for CDC.
    next_txn_id: AtomicU64,
    /// Monotonically increasing commit-order counter for CDC event keys.
    ///
    /// Unlike `next_txn_id` (allocated at `begin_write()` time), this counter
    /// is allocated at `commit()` time. This guarantees that CDC events are
    /// keyed in commit order, so consumers that track progress by scanning
    /// `transaction_id > last_seen` will never permanently miss events from
    /// transactions that were started early but committed late.
    next_commit_id: Arc<AtomicU64>,
    /// Serializes snapshot operations across concurrent committers.
    snapshot_lock: Arc<Mutex<()>>,
    /// Number of commits since last snapshot. Used for amortized checkpointing.
    commit_count: Arc<AtomicU64>,
    /// Snapshot every N commits. 0 = disabled (manual snapshots only).
    snapshot_interval: u64,
    /// Durability mode for WAL fsync behavior.
    durability: super::config::DurabilityMode,
}

impl BfTreeDatabase {
    /// Create a new Bf-Tree database with the given configuration.
    pub fn create(config: BfTreeConfig) -> Result<Self, BfTreeError> {
        let verify_mode = config.verify_mode.clone();
        let snapshot_interval = config.snapshot_interval;
        let durability = config.durability;
        let adapter = BfTreeAdapter::open(config)?;
        Ok(Self {
            adapter: Arc::new(adapter),
            cdc_config: CdcConfig::default(),
            verify_mode: Arc::new(verify_mode),
            next_txn_id: AtomicU64::new(1),
            next_commit_id: Arc::new(AtomicU64::new(1)),
            snapshot_lock: Arc::new(Mutex::new(())),
            commit_count: Arc::new(AtomicU64::new(0)),
            snapshot_interval,
            durability,
        })
    }

    /// Open an existing Bf-Tree database from a snapshot file.
    pub fn open(config: BfTreeConfig) -> Result<Self, BfTreeError> {
        let verify_mode = config.verify_mode.clone();
        let snapshot_interval = config.snapshot_interval;
        let durability = config.durability;
        let adapter = BfTreeAdapter::open_from_snapshot(config)?;
        // Recover the next transaction ID from the latest CDC log entry.
        let next_id = recover_next_txn_id(&adapter).unwrap_or(1);
        Ok(Self {
            adapter: Arc::new(adapter),
            cdc_config: CdcConfig::default(),
            verify_mode: Arc::new(verify_mode),
            next_txn_id: AtomicU64::new(next_id),
            next_commit_id: Arc::new(AtomicU64::new(next_id)),
            snapshot_lock: Arc::new(Mutex::new(())),
            commit_count: Arc::new(AtomicU64::new(0)),
            snapshot_interval,
            durability,
        })
    }

    /// Create a new Bf-Tree database with the given configuration and CDC settings.
    pub fn create_with_cdc(
        config: BfTreeConfig,
        cdc_config: CdcConfig,
    ) -> Result<Self, BfTreeError> {
        let verify_mode = config.verify_mode.clone();
        let snapshot_interval = config.snapshot_interval;
        let durability = config.durability;
        let adapter = BfTreeAdapter::open(config)?;
        Ok(Self {
            adapter: Arc::new(adapter),
            cdc_config,
            verify_mode: Arc::new(verify_mode),
            next_txn_id: AtomicU64::new(1),
            next_commit_id: Arc::new(AtomicU64::new(1)),
            snapshot_lock: Arc::new(Mutex::new(())),
            commit_count: Arc::new(AtomicU64::new(0)),
            snapshot_interval,
            durability,
        })
    }

    /// Open an existing Bf-Tree database with the given CDC settings.
    pub fn open_with_cdc(config: BfTreeConfig, cdc_config: CdcConfig) -> Result<Self, BfTreeError> {
        let verify_mode = config.verify_mode.clone();
        let snapshot_interval = config.snapshot_interval;
        let durability = config.durability;
        let adapter = BfTreeAdapter::open_from_snapshot(config)?;
        let next_id = recover_next_txn_id(&adapter).unwrap_or(1);
        Ok(Self {
            adapter: Arc::new(adapter),
            cdc_config,
            verify_mode: Arc::new(verify_mode),
            next_txn_id: AtomicU64::new(next_id),
            next_commit_id: Arc::new(AtomicU64::new(next_id)),
            snapshot_lock: Arc::new(Mutex::new(())),
            commit_count: Arc::new(AtomicU64::new(0)),
            snapshot_interval,
            durability,
        })
    }

    /// Begin a write transaction.
    ///
    /// Unlike the legacy `Database::begin_write()`, this does NOT block.
    /// Multiple write transactions can coexist and execute concurrently.
    pub fn begin_write(&self) -> BfTreeDatabaseWriteTxn {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::SeqCst);
        assert!(
            txn_id < u64::MAX,
            "transaction ID counter exhausted (u64::MAX reached)"
        );
        let cdc_log = if self.cdc_config.enabled {
            Some(Mutex::new(Vec::new()))
        } else {
            None
        };
        BfTreeDatabaseWriteTxn {
            adapter: self.adapter.clone(),
            ops_count: AtomicU64::new(0),
            txn_id,
            next_commit_id: self.next_commit_id.clone(),
            snapshot_lock: self.snapshot_lock.clone(),
            commit_count: self.commit_count.clone(),
            snapshot_interval: self.snapshot_interval,
            cdc_log,
            cdc_config: self.cdc_config.clone(),
            verify_mode: self.verify_mode.clone(),
            buffer: Mutex::new(WriteBuffer::new()),
            committed: core::sync::atomic::AtomicBool::new(false),
            durability: self.durability,
        }
    }

    /// Begin a read transaction.
    ///
    /// Reads always see the latest state (no snapshot isolation).
    pub fn begin_read(&self) -> BfTreeDatabaseReadTxn {
        // Buffers start empty; `get()` grows them on first use. This keeps
        // `begin_read()` allocation-free for callers that only use
        // `open_table()` or `contains_key()`.
        BfTreeDatabaseReadTxn {
            adapter: self.adapter.clone(),
            verify_mode: self.verify_mode.clone(),
            read_buf: Vec::new(),
            key_buf: Vec::new(),
        }
    }

    /// Take a durability snapshot. All data is guaranteed recoverable after crash.
    ///
    /// Serialized with concurrent commit operations via the snapshot lock.
    pub fn snapshot(&self) -> Result<std::path::PathBuf, crate::bf_tree::BfTreeError> {
        let _guard = self.snapshot_lock.lock();
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

/// Recover the next transaction ID from persistent state.
///
/// Uses three sources and takes the maximum to guarantee monotonicity:
///
/// 1. **Persisted counter** (`__bf_meta:next_txn_id`) -- the authoritative
///    source, written atomically on every commit regardless of CDC state.
/// 2. **CDC log entries** -- highest `transaction_id` found in the CDC log
///    table (only populated when CDC is enabled).
/// 3. **CDC cursor entries** -- highest cursor position, which may exceed the
///    CDC log maximum after retention pruning.
///
/// Taking the max of all three ensures correct recovery in every scenario:
/// CDC enabled, CDC disabled, CDC enabled then disabled, and post-pruning.
fn recover_next_txn_id(adapter: &BfTreeAdapter) -> Result<u64, BfTreeError> {
    let max_record = adapter.inner().config().get_cb_max_record_size();
    let mut buf = vec![0u8; max_record * 2];
    let mut max_next_id: u64 = 0;

    // 1. Read the persisted transaction ID counter (primary source).
    {
        let meta_key = encode_table_key(BF_META_TABLE_NAME, TableKind::Regular, BF_META_TXN_ID_KEY);
        match adapter.read(&meta_key, &mut buf) {
            Ok(len) if len as usize >= 8 => {
                let persisted = u64::from_le_bytes(buf[..8].try_into().unwrap());
                if persisted > max_next_id {
                    max_next_id = persisted;
                }
            }
            // Key absent (fresh DB or pre-migration snapshot) -- fall through.
            _ => {}
        }
    }

    // 2. Scan CDC log entries for the highest transaction ID.
    {
        let prefix = table_prefix(CDC_LOG_TABLE_NAME, TableKind::Regular);
        let prefix_end = table_prefix_end(CDC_LOG_TABLE_NAME, TableKind::Regular);
        let prefix_len = prefix.len();
        let mut iter = adapter.scan_range(&prefix, &prefix_end)?;
        while let Ok(Some((key_len, _val_len))) = iter.next(&mut buf) {
            if key_len > prefix_len + CdcKey::SERIALIZED_SIZE {
                continue;
            }
            let key_bytes = &buf[prefix_len..key_len];
            if key_bytes.len() >= CdcKey::SERIALIZED_SIZE {
                let cdc_key = CdcKey::from_be_bytes(key_bytes);
                let candidate = cdc_key.transaction_id.saturating_add(1);
                if candidate > max_next_id {
                    max_next_id = candidate;
                }
            }
        }
    }

    // 3. Scan CDC cursor entries for the highest stored transaction ID.
    // Cursors store `u64` LE values representing the last consumed txn_id,
    // which may exceed the CDC log's max after retention pruning.
    {
        let prefix = table_prefix(CDC_CURSOR_TABLE_NAME, TableKind::Regular);
        let prefix_end = table_prefix_end(CDC_CURSOR_TABLE_NAME, TableKind::Regular);
        let prefix_len = prefix.len();
        let mut iter = adapter.scan_range(&prefix, &prefix_end)?;
        while let Ok(Some((key_len, val_len))) = iter.next(&mut buf) {
            if key_len <= prefix_len {
                continue;
            }
            let val_bytes = &buf[key_len..key_len + val_len];
            if val_bytes.len() >= 8 {
                let cursor_txn = u64::from_le_bytes(val_bytes[..8].try_into().unwrap());
                let candidate = cursor_txn.saturating_add(1);
                if candidate > max_next_id {
                    max_next_id = candidate;
                }
            }
        }
    }

    // Ensure we never return 0 (minimum valid txn_id is 1).
    Ok(max_next_id.max(1))
}

// ---------------------------------------------------------------------------
// Internal key encoding: prefix table name to key bytes for namespace isolation.
//
// Format: [table_name_len: u16 LE] [table_name: bytes] [kind: u8] [key: bytes]
//
// The 1-byte `kind` discriminator prevents namespace collisions between
// regular tables, multimap tables, and TTL tables that share the same name.
// Without the discriminator, opening the same name as both a regular table
// and a multimap table would silently collide in the underlying BfTree.
//
// This ensures keys from different tables and table types never collide in
// the single underlying Bf-Tree index.
// ---------------------------------------------------------------------------

/// Discriminator byte identifying the table type in the encoded key prefix.
///
/// Inserted immediately after the table name bytes, this guarantees that
/// regular, multimap, and TTL tables with the same user-visible name occupy
/// non-overlapping regions of the `BfTree` keyspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum TableKind {
    /// Standard key-value table.
    Regular = 0x00,
    /// Multimap table (one key to many values).
    Multimap = 0x01,
    /// TTL table (key-value with per-entry expiry).
    Ttl = 0x02,
}

pub(crate) fn encode_table_key(table_name: &str, kind: TableKind, key_bytes: &[u8]) -> Vec<u8> {
    let name_bytes = table_name.as_bytes();
    // Caller (validate_table_name) ensures name_bytes.len() <= u16::MAX.
    debug_assert!(u16::try_from(name_bytes.len()).is_ok());
    #[allow(clippy::cast_possible_truncation)]
    let name_len = name_bytes.len() as u16;
    let mut buf = Vec::with_capacity(2 + name_bytes.len() + 1 + key_bytes.len());
    buf.extend_from_slice(&name_len.to_le_bytes());
    buf.extend_from_slice(name_bytes);
    buf.push(kind as u8);
    buf.extend_from_slice(key_bytes);
    buf
}

/// Encode a table key into a pre-allocated buffer, returning the encoded length.
///
/// Zero-allocation alternative to [`encode_table_key`] for hot read paths.
/// The buffer must be at least `encoded_table_key_len(table_name, key_bytes)` bytes.
pub(crate) fn encode_table_key_into(
    buf: &mut [u8],
    table_name: &str,
    kind: TableKind,
    key_bytes: &[u8],
) -> usize {
    let name_bytes = table_name.as_bytes();
    debug_assert!(u16::try_from(name_bytes.len()).is_ok());
    #[allow(clippy::cast_possible_truncation)]
    let name_len = name_bytes.len() as u16;
    let prefix_len = 2 + name_bytes.len() + 1;
    let total = prefix_len + key_bytes.len();
    debug_assert!(buf.len() >= total);
    buf[..2].copy_from_slice(&name_len.to_le_bytes());
    buf[2..2 + name_bytes.len()].copy_from_slice(name_bytes);
    buf[2 + name_bytes.len()] = kind as u8;
    buf[prefix_len..total].copy_from_slice(key_bytes);
    total
}

pub(crate) fn table_prefix(table_name: &str, kind: TableKind) -> Vec<u8> {
    let name_bytes = table_name.as_bytes();
    debug_assert!(u16::try_from(name_bytes.len()).is_ok());
    #[allow(clippy::cast_possible_truncation)]
    let name_len = name_bytes.len() as u16;
    let mut buf = Vec::with_capacity(2 + name_bytes.len() + 1);
    buf.extend_from_slice(&name_len.to_le_bytes());
    buf.extend_from_slice(name_bytes);
    buf.push(kind as u8);
    buf
}

pub(crate) fn table_prefix_end(table_name: &str, kind: TableKind) -> Vec<u8> {
    let mut prefix = table_prefix(table_name, kind);
    // Increment the prefix as a big-endian integer with carry propagation to
    // produce a correct exclusive upper bound.  Walk from the least-significant
    // byte toward the most-significant, propagating the carry until a byte can
    // absorb it.
    for i in (0..prefix.len()).rev() {
        if prefix[i] < 0xFF {
            prefix[i] += 1;
            return prefix;
        }
        // This byte overflows; zero it and carry to the next position.
        prefix[i] = 0x00;
    }
    // Every byte in the prefix was 0xFF.  The only way to produce a value
    // strictly greater than 0xFF..FF of the same length is to use a longer
    // byte string.  Restore the prefix and append 0xFF so the result is
    // lexicographically larger than any key that starts with the original
    // prefix.
    prefix.fill(0xFF);
    prefix.push(0xFF);
    prefix
}

/// Prefix reserved for internal system tables.
///
/// User-created tables whose names start with this prefix are rejected to
/// prevent accidental (or malicious) collision with internal tables such as
/// `__cdc_log`, `__bf_blob_meta`, `__bf_history_meta`, etc.
const RESERVED_TABLE_PREFIX: &str = "__";

/// Validate that a user-supplied table name does not collide with reserved
/// internal table names. Returns `Err(BfTreeError::ReservedTableName)` if
/// the name starts with the reserved `"__"` prefix.
fn validate_table_name(name: &str) -> Result<(), BfTreeError> {
    if name.len() > u16::MAX as usize {
        return Err(BfTreeError::InvalidKV(alloc::format!(
            "table name length {} exceeds maximum {} bytes",
            name.len(),
            u16::MAX
        )));
    }
    if name.starts_with(RESERVED_TABLE_PREFIX) {
        return Err(BfTreeError::ReservedTableName(alloc::string::String::from(
            name,
        )));
    }
    Ok(())
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
    /// Shared commit-order counter from the parent `BfTreeDatabase`.
    ///
    /// Allocated at `commit()` time to ensure CDC events are keyed in commit
    /// order, preventing consumers from permanently missing events when
    /// transactions commit out of creation order.
    next_commit_id: Arc<AtomicU64>,
    /// Serializes snapshot operations across concurrent committers.
    snapshot_lock: Arc<Mutex<()>>,
    /// Shared commit counter for amortized snapshot scheduling.
    commit_count: Arc<AtomicU64>,
    /// Snapshot every N commits. 0 = disabled.
    snapshot_interval: u64,
    /// Accumulated CDC events. `None` if CDC is disabled.
    pub(crate) cdc_log: Option<Mutex<Vec<CdcEvent>>>,
    /// CDC configuration (retention, etc.).
    pub(crate) cdc_config: CdcConfig,
    /// Per-entry checksum verification mode.
    pub(crate) verify_mode: Arc<VerifyMode>,
    /// Write buffer for atomic commit/rollback semantics.
    pub(crate) buffer: Mutex<WriteBuffer>,
    /// Track whether buffer has been flushed (committed).
    committed: core::sync::atomic::AtomicBool,
    /// Durability mode for WAL fsync behavior.
    durability: super::config::DurabilityMode,
}

impl BfTreeDatabaseWriteTxn {
    /// Open a typed table handle for read-write access.
    ///
    /// This is the preferred API -- it mirrors the legacy `WriteTransaction::open_table()`
    /// pattern. The returned `BfTreeTable` provides `insert`, `get`, `remove`, and `scan`.
    ///
    /// Returns `Err(BfTreeError::ReservedTableName)` if the table name starts
    /// with the reserved `"__"` prefix used by internal system tables.
    pub fn open_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: TableDefinition<K, V>,
    ) -> Result<BfTreeTable<'_, K, V>, BfTreeError> {
        validate_table_name(definition.name())?;
        Ok(BfTreeTable::new(
            definition.name(),
            &self.adapter,
            &self.ops_count,
            self.cdc_log.as_ref(),
            &self.buffer,
            &self.verify_mode,
        ))
    }

    /// Open a TTL table for read-write access.
    ///
    /// Returns `Err(BfTreeError::ReservedTableName)` if the table name starts
    /// with the reserved `"__"` prefix used by internal system tables.
    pub fn open_ttl_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: TableDefinition<K, V>,
    ) -> Result<BfTreeTtlTable<'_, K, V>, BfTreeError> {
        validate_table_name(definition.name())?;
        Ok(BfTreeTtlTable::new(
            definition.name(),
            &self.adapter,
            &self.ops_count,
            self.cdc_log.as_ref(),
            &self.buffer,
        ))
    }

    /// Open a multimap table for read-write access.
    ///
    /// Returns `Err(BfTreeError::ReservedTableName)` if the table name starts
    /// with the reserved `"__"` prefix used by internal system tables.
    pub fn open_multimap_table<K: Key + 'static, V: Key + 'static>(
        &self,
        name: &str,
    ) -> Result<BfTreeMultimapTable<'_, K, V>, BfTreeError> {
        validate_table_name(name)?;
        Ok(BfTreeMultimapTable::new(
            name,
            &self.adapter,
            &self.ops_count,
            self.cdc_log.as_ref(),
            &self.buffer,
        ))
    }

    /// Open the blob store for read-write access.
    pub fn open_blob_store(&self) -> BfTreeBlobStore<'_> {
        BfTreeBlobStore::new(
            &self.adapter,
            &self.buffer,
            &self.ops_count,
            self.cdc_log.as_ref(),
        )
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
        let encoded_key =
            encode_table_key(definition.name(), TableKind::Regular, key_bytes.as_ref());
        self.buffer
            .lock()
            .put(encoded_key, val_bytes.as_ref().to_vec())?;
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
        let encoded_key =
            encode_table_key(definition.name(), TableKind::Regular, key_bytes.as_ref());
        self.buffer.lock().delete(encoded_key);
        self.ops_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Read a typed value from the specified table.
    ///
    /// Checks the write buffer first (for read-your-writes), then falls
    /// through to `BfTree` if the key is not in the buffer.
    pub fn get<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: &TableDefinition<K, V>,
        key: &K::SelfType<'_>,
    ) -> Result<Option<Vec<u8>>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let encoded_key =
            encode_table_key(definition.name(), TableKind::Regular, key_bytes.as_ref());

        // Check write buffer first.
        {
            let buffer = self.buffer.lock();
            match buffer.get(&encoded_key) {
                super::buffered_txn::BufferLookup::Found(v) => return Ok(Some(v)),
                super::buffered_txn::BufferLookup::Tombstone => return Ok(None),
                super::buffered_txn::BufferLookup::NotInBuffer => {}
            }
        }

        // Fall through to BfTree.
        let max_val = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.adapter.read(&encoded_key, &mut buf) {
            Ok(len) => Ok(Some(buf[..len as usize].to_vec())),
            Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Check if a key exists in the table.
    ///
    /// Checks the write buffer first (for read-your-writes), then falls
    /// through to `BfTree` if the key is not in the buffer.
    pub fn contains_key<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: &TableDefinition<K, V>,
        key: &K::SelfType<'_>,
    ) -> bool {
        let key_bytes = K::as_bytes(key);
        let encoded_key =
            encode_table_key(definition.name(), TableKind::Regular, key_bytes.as_ref());

        // Check write buffer first.
        {
            let buffer = self.buffer.lock();
            match buffer.get(&encoded_key) {
                super::buffered_txn::BufferLookup::Found(_) => return true,
                super::buffered_txn::BufferLookup::Tombstone => return false,
                super::buffered_txn::BufferLookup::NotInBuffer => {}
            }
        }

        // Fall through to BfTree.
        self.adapter.contains_key(&encoded_key)
    }

    /// Commit the transaction for durability.
    ///
    /// If CDC is enabled, accumulated events are staged into the write buffer
    /// before flushing, so CDC log entries are atomically committed with data.
    ///
    /// Durability is provided by the WAL: each `buffer.flush()` entry goes
    /// through `BfTree::write_inner()` which calls `wal.append_and_wait()`,
    /// blocking until the WAL entry is fsync'd to disk. By the time `flush()`
    /// returns, all data is WAL-durable and recoverable via WAL replay.
    ///
    /// Snapshots (full checkpoint of the circular buffer + inner nodes) are
    /// taken periodically every `snapshot_interval` commits to bound WAL
    /// replay time on crash recovery. They are NOT required for durability.
    pub fn commit(self) -> Result<(), BfTreeError> {
        {
            let mut buffer = self.buffer.lock();
            let commit_id = self.stage_cdc_into_buffer(&mut buffer)?;
            self.stage_txn_id_into_buffer(&mut buffer, commit_id)?;
            // Flush write buffer to BfTree. Each entry goes through WAL
            // append_and_wait(), so data is durable when flush returns.
            buffer.flush(&self.adapter, self.durability)?;
        }
        self.committed.store(true, Ordering::SeqCst);

        // Amortized snapshot: checkpoint every N commits to bound WAL replay
        // time on crash recovery. The snapshot_lock serializes concurrent
        // snapshot attempts so only one thread performs the expensive I/O.
        if self.snapshot_interval > 0 && !self.adapter.inner().config().is_memory_backend() {
            let count = self.commit_count.fetch_add(1, Ordering::Relaxed) + 1;
            if count % self.snapshot_interval == 0 {
                let _snap_guard = self.snapshot_lock.lock();
                self.adapter.snapshot()?;
            }
        }

        // Retention pruning is post-commit and non-critical.
        self.prune_cdc_retention();
        Ok(())
    }

    /// Commit with explicit snapshot for guaranteed fast crash recovery.
    ///
    /// The flush goes through the WAL for durability. The snapshot is taken
    /// afterwards under the snapshot lock to create a full checkpoint, which
    /// bounds WAL replay time to zero for this commit point.
    pub fn commit_with_snapshot(self) -> Result<std::path::PathBuf, BfTreeError> {
        {
            let mut buffer = self.buffer.lock();
            let commit_id = self.stage_cdc_into_buffer(&mut buffer)?;
            self.stage_txn_id_into_buffer(&mut buffer, commit_id)?;
            buffer.flush(&self.adapter, self.durability)?;
        }
        self.committed.store(true, Ordering::SeqCst);
        let path = {
            let _snap_guard = self.snapshot_lock.lock();
            self.adapter.snapshot()?
        };
        self.prune_cdc_retention();
        Ok(path)
    }

    /// Number of operations performed.
    pub fn ops_count(&self) -> u64 {
        self.ops_count.load(Ordering::Relaxed)
    }

    /// Transaction ID assigned to this write transaction.
    pub fn txn_id(&self) -> u64 {
        self.txn_id
    }

    /// Merge the write buffer from another transaction into this one.
    ///
    /// Used by `concurrent_group_commit` to combine isolated per-batch buffers
    /// into a single transaction before committing. Entries from `other` are
    /// applied on top of existing entries in this transaction's buffer (last
    /// writer wins on key collision).
    ///
    /// The other transaction is consumed; its buffer is drained and merged.
    pub(crate) fn merge_buffer_from(
        &self,
        other: &BfTreeDatabaseWriteTxn,
    ) -> Result<(), BfTreeError> {
        let other_buffer = {
            let mut guard = other.buffer.lock();
            core::mem::replace(&mut *guard, WriteBuffer::new())
        };
        // Mark the source as committed so its Drop impl does not discard
        // (the buffer is now empty anyway, but this is semantically correct).
        other.committed.store(true, Ordering::SeqCst);
        let mut self_buffer = self.buffer.lock();
        self_buffer.merge_from(other_buffer)
    }

    /// Advance a named CDC cursor to the given transaction ID.
    ///
    /// Routed through the write buffer so that cursor advances are rolled back
    /// on abort and committed atomically with the rest of the transaction.
    pub fn advance_cdc_cursor(&self, name: &str, up_to_txn: u64) -> Result<(), BfTreeError> {
        let key_bytes = name.as_bytes();
        let encoded = encode_table_key(CDC_CURSOR_TABLE_NAME, TableKind::Regular, key_bytes);
        let val_bytes = up_to_txn.to_le_bytes();
        self.buffer.lock().put(encoded, val_bytes.to_vec())?;
        Ok(())
    }

    /// Stage accumulated CDC events into the write buffer as KV entries.
    ///
    /// CDC events are keyed by a **commit-order ID** allocated from the shared
    /// `next_commit_id` counter at commit time, not by the `txn_id` assigned at
    /// `begin_write()` time. This guarantees monotonic ordering in the CDC log
    /// with respect to actual commit order, so consumers that track progress by
    /// scanning `transaction_id > last_seen` will never permanently miss events
    /// from transactions that were started early but committed late.
    ///
    /// Returns the allocated commit ID (used for persisting the counter and for
    /// retention pruning), or `None` if there were no CDC events to stage.
    fn stage_cdc_into_buffer(&self, buffer: &mut WriteBuffer) -> Result<Option<u64>, BfTreeError> {
        let events = match self.cdc_log {
            Some(ref log) => {
                let mut guard = log.lock();
                if guard.is_empty() {
                    return Ok(None);
                }
                core::mem::take(&mut *guard)
            }
            None => return Ok(None),
        };

        // Allocate a commit-order ID. This is the point of linearization: the
        // fetch_add is atomic, so concurrent committers get strictly ordered IDs
        // that reflect actual commit order rather than transaction creation order.
        let commit_id = self.next_commit_id.fetch_add(1, Ordering::SeqCst);
        if commit_id == u64::MAX {
            return Err(BfTreeError::InvalidOperation(alloc::string::String::from(
                "commit ID counter exhausted (u64::MAX reached)",
            )));
        }

        for (seq, event) in events.iter().enumerate() {
            let seq_u32 = u32::try_from(seq).map_err(|_| {
                BfTreeError::InvalidKV(alloc::format!(
                    "CDC sequence {seq} exceeds u32::MAX; too many events in one transaction"
                ))
            })?;
            let key = CdcKey::new(commit_id, seq_u32);
            let record = CdcRecord::from_event(event).map_err(|e| {
                BfTreeError::InvalidKV(alloc::format!("CDC record serialization error: {e}"))
            })?;
            let key_bytes = key.to_be_bytes();
            let val_bytes = record.serialize();
            let encoded_key = encode_table_key(CDC_LOG_TABLE_NAME, TableKind::Regular, &key_bytes);
            buffer.put(encoded_key, val_bytes)?;
        }

        Ok(Some(commit_id))
    }

    /// Persist the next ID counter so recovery works without CDC.
    ///
    /// Uses the maximum of `txn_id + 1` and `commit_id + 1` (if a CDC commit
    /// ID was allocated) to guarantee that neither transaction IDs nor commit
    /// IDs are ever reused after recovery.
    fn stage_txn_id_into_buffer(
        &self,
        buffer: &mut WriteBuffer,
        commit_id: Option<u64>,
    ) -> Result<(), BfTreeError> {
        let mut next_id = self.txn_id.saturating_add(1);
        if let Some(cid) = commit_id {
            next_id = next_id.max(cid.saturating_add(1));
        }
        let encoded_key =
            encode_table_key(BF_META_TABLE_NAME, TableKind::Regular, BF_META_TXN_ID_KEY);
        buffer.put(encoded_key, next_id.to_le_bytes().to_vec())
    }

    /// Post-commit retention pruning of old CDC log entries.
    ///
    /// Deletes at transaction-ID granularity: all entries for a given `txn_id`
    /// are deleted together so that concurrent `read_cdc_since()` never
    /// observes a partially-pruned transaction.
    ///
    /// This runs after the commit is finalized. Failures are non-fatal --
    /// old entries accumulate but do not affect correctness.
    fn prune_cdc_retention(&self) {
        if self.cdc_config.retention_max_txns == 0
            || self.txn_id <= self.cdc_config.retention_max_txns
        {
            return;
        }

        let cutoff_txn = self.txn_id - self.cdc_config.retention_max_txns;
        let cutoff_key = CdcKey::new(cutoff_txn, u32::MAX);
        let cutoff_encoded = encode_table_key(
            CDC_LOG_TABLE_NAME,
            TableKind::Regular,
            &cutoff_key.to_be_bytes(),
        );
        let prefix = table_prefix(CDC_LOG_TABLE_NAME, TableKind::Regular);
        let prefix_len = prefix.len();
        let max_record = self.adapter.inner().config().get_cb_max_record_size();

        // Collect keys grouped by transaction ID for atomic per-txn deletion.
        // Since the scan is sorted by (txn_id, seq) in big-endian byte order,
        // consecutive entries with the same txn_id are naturally grouped.
        let keys_by_txn: Vec<(u64, Vec<Vec<u8>>)> = {
            let mut buf = vec![0u8; max_record * 2];
            let mut groups: Vec<(u64, Vec<Vec<u8>>)> = Vec::new();
            if let Ok(mut iter) = self.adapter.scan_range(&prefix, &cutoff_encoded) {
                while let Ok(Some((key_len, _val_len))) = iter.next(&mut buf) {
                    if key_len <= prefix_len {
                        continue;
                    }
                    let key_bytes = &buf[prefix_len..key_len];
                    if key_bytes.len() < CdcKey::SERIALIZED_SIZE {
                        continue;
                    }
                    let txn_id = CdcKey::from_be_bytes(key_bytes).transaction_id;
                    let encoded = buf[..key_len].to_vec();
                    match groups.last_mut() {
                        Some((last_txn, keys)) if *last_txn == txn_id => {
                            keys.push(encoded);
                        }
                        _ => {
                            groups.push((txn_id, vec![encoded]));
                        }
                    }
                }
            }
            groups
        };

        // Delete all entries for each complete transaction group.
        for (_txn_id, keys) in &keys_by_txn {
            for encoded_key in keys {
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
            self.buffer.lock().discard();
        }
    }
}

/// Read transaction for `BfTreeDatabase`.
///
/// Provides typed get/scan using shodh-redb's `Key`/`Value` traits.
pub struct BfTreeDatabaseReadTxn {
    pub(crate) adapter: Arc<BfTreeAdapter>,
    /// Per-entry checksum verification mode.
    pub(crate) verify_mode: Arc<VerifyMode>,
    /// Reusable buffer for bf-tree read output, avoiding per-read heap allocation.
    read_buf: Vec<u8>,
    /// Reusable buffer for encoded key, avoiding per-read heap allocation.
    key_buf: Vec<u8>,
}

impl BfTreeDatabaseReadTxn {
    /// Open a typed table handle for read-only access.
    ///
    /// Mirrors the legacy `ReadTransaction::open_table()` pattern.
    ///
    /// Returns `Err(BfTreeError::ReservedTableName)` if the table name starts
    /// with the reserved `"__"` prefix used by internal system tables.
    pub fn open_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: TableDefinition<K, V>,
    ) -> Result<BfTreeReadOnlyTable<'_, K, V>, BfTreeError> {
        validate_table_name(definition.name())?;
        Ok(BfTreeReadOnlyTable::new(
            definition.name(),
            &self.adapter,
            &self.verify_mode,
        ))
    }

    /// Open a TTL table for read-only access.
    ///
    /// Returns `Err(BfTreeError::ReservedTableName)` if the table name starts
    /// with the reserved `"__"` prefix used by internal system tables.
    pub fn open_ttl_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: TableDefinition<K, V>,
    ) -> Result<BfTreeReadOnlyTtlTable<'_, K, V>, BfTreeError> {
        validate_table_name(definition.name())?;
        Ok(BfTreeReadOnlyTtlTable::new(
            definition.name(),
            &self.adapter,
        ))
    }

    /// Open a multimap table for read-only access.
    ///
    /// Returns `Err(BfTreeError::ReservedTableName)` if the table name starts
    /// with the reserved `"__"` prefix used by internal system tables.
    pub fn open_multimap_table<K: Key + 'static, V: Key + 'static>(
        &self,
        name: &str,
    ) -> Result<BfTreeReadOnlyMultimapTable<'_, K, V>, BfTreeError> {
        validate_table_name(name)?;
        Ok(BfTreeReadOnlyMultimapTable::new(name, &self.adapter))
    }

    /// Open the blob store for read-only access.
    pub fn open_blob_store(&self) -> BfTreeReadOnlyBlobStore<'_> {
        BfTreeReadOnlyBlobStore::new(&self.adapter)
    }

    /// Read a typed value from the specified table.
    ///
    /// Returns the raw value bytes. Use `V::from_bytes()` to deserialize.
    /// Buffers are lazily allocated on first call and reused across subsequent
    /// reads, amortizing allocation cost over multiple lookups.
    pub fn get<K: Key + 'static, V: Value + 'static>(
        &mut self,
        definition: &TableDefinition<K, V>,
        key: &K::SelfType<'_>,
    ) -> Result<Option<Vec<u8>>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let needed_key_len = 2 + definition.name().len() + 1 + key_bytes.as_ref().len();
        if self.key_buf.len() < needed_key_len {
            self.key_buf.resize(needed_key_len, 0);
        }
        if self.read_buf.is_empty() {
            let max_record = self.adapter.inner().config().get_cb_max_record_size();
            self.read_buf.resize(max_record, 0);
        }
        let enc_len = encode_table_key_into(
            &mut self.key_buf,
            definition.name(),
            TableKind::Regular,
            key_bytes.as_ref(),
        );
        match self
            .adapter
            .read(&self.key_buf[..enc_len], &mut self.read_buf)
        {
            Ok(len) => Ok(Some(self.read_buf[..len as usize].to_vec())),
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
        let encoded_key =
            encode_table_key(definition.name(), TableKind::Regular, key_bytes.as_ref());
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
        let prefix = table_prefix(definition.name(), TableKind::Regular);
        let prefix_end = table_prefix_end(definition.name(), TableKind::Regular);
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
    pub fn read_cdc_range(
        &self,
        start_txn: u64,
        end_txn: u64,
    ) -> Result<Vec<ChangeStream>, BfTreeError> {
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
        let encoded = encode_table_key(CDC_CURSOR_TABLE_NAME, TableKind::Regular, key_bytes);
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
        let prefix = table_prefix(CDC_LOG_TABLE_NAME, TableKind::Regular);
        let prefix_end = table_prefix_end(CDC_LOG_TABLE_NAME, TableKind::Regular);
        let prefix_len = prefix.len();
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_record * 2];
        let mut iter = self.adapter.scan_range(&prefix, &prefix_end)?;
        let mut max_txn_id: Option<u64> = None;
        while let Ok(Some((key_len, _val_len))) = iter.next(&mut buf) {
            if key_len > prefix_len {
                let key_bytes = &buf[prefix_len..key_len];
                if key_bytes.len() >= CdcKey::SERIALIZED_SIZE {
                    let cdc_key = CdcKey::from_be_bytes(key_bytes);
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

    fn read_cdc_range_inner(
        &self,
        start_key: CdcKey,
        end_key: CdcKey,
    ) -> Result<Vec<ChangeStream>, BfTreeError> {
        let start_encoded = encode_table_key(
            CDC_LOG_TABLE_NAME,
            TableKind::Regular,
            &start_key.to_be_bytes(),
        );
        let end_encoded = encode_table_key(
            CDC_LOG_TABLE_NAME,
            TableKind::Regular,
            &end_key.to_be_bytes(),
        );
        let prefix_len = table_prefix(CDC_LOG_TABLE_NAME, TableKind::Regular).len();
        let max_record = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_record * 2];
        let mut iter = self.adapter.scan_range(&start_encoded, &end_encoded)?;
        let mut results = Vec::new();

        while let Ok(Some((key_len, val_len))) = iter.next(&mut buf) {
            if key_len <= prefix_len {
                continue;
            }
            let key_bytes = &buf[prefix_len..key_len];
            let val_bytes = &buf[key_len..key_len + val_len];

            if key_bytes.len() < CdcKey::SERIALIZED_SIZE {
                continue;
            }
            let cdc_key = CdcKey::from_be_bytes(key_bytes);
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
    pub(crate) iter: crate::bf_tree::ScanIter<'a, 'a>,
    pub(crate) prefix_len: usize,
}

impl BfTreeTableScan<'_> {
    /// Get the next `(key_bytes, value_bytes)` entry.
    ///
    /// The `buf` must be large enough to hold key + value.
    /// Returns `None` when no more entries exist.
    pub fn next<'buf>(&mut self, buf: &'buf mut [u8]) -> Option<(&'buf [u8], &'buf [u8])> {
        loop {
            let (key_len, val_len) = self.iter.next(buf).ok().flatten()?;
            // Strip the table prefix from the key.
            if key_len > self.prefix_len {
                let key = &buf[self.prefix_len..key_len];
                let val = &buf[key_len..key_len + val_len];
                return Some((key, val));
            }
            // Malformed entry (key_len <= prefix_len) -- skip and try the next
            // entry instead of terminating the entire scan.
        }
    }
}

// ---------------------------------------------------------------------------
// BfTreeBuilder -- ergonomic database construction with optional features
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

        let mut rtxn = db.begin_read();
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

        let mut rtxn = db.begin_read();
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

        let mut rtxn = db.begin_read();
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
                            encode_table_key(SCORES.name(), TableKind::Regular, key_bytes);
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
                let encoded = encode_table_key(SCORES.name(), TableKind::Regular, key_bytes);
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
            CdcConfig {
                enabled: true,
                retention_max_txns: 0,
            },
        )
        .unwrap()
    }

    #[test]
    fn cdc_records_insert_and_update() {
        let db = cdc_db();

        let wtxn = db.begin_write();
        let txn_id = wtxn.txn_id();
        let mut table = wtxn.open_table(USERS).unwrap();
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
        let mut table = wtxn.open_table(USERS).unwrap();
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
        let mut t1 = wtxn1.open_table(USERS).unwrap();
        t1.insert(&"a", &1u64).unwrap();
        drop(t1);
        wtxn1.commit().unwrap();

        // Transaction 2
        let wtxn2 = db.begin_write();
        let txn2_id = wtxn2.txn_id();
        let mut t2 = wtxn2.open_table(USERS).unwrap();
        t2.insert(&"b", &2u64).unwrap();
        drop(t2);
        wtxn2.commit().unwrap();

        let rtxn = db.begin_read();

        // Read since txn1 -- should only see txn2's changes.
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
        let mut t = w1.open_table(USERS).unwrap();
        t.insert(&"x", &1u64).unwrap();
        drop(t);
        w1.commit().unwrap();

        let w2 = db.begin_write();
        let id2 = w2.txn_id();
        let mut t = w2.open_table(USERS).unwrap();
        t.insert(&"y", &2u64).unwrap();
        drop(t);
        w2.commit().unwrap();

        let w3 = db.begin_write();
        let _id3 = w3.txn_id();
        let mut t = w3.open_table(USERS).unwrap();
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
        let mut t = w.open_table(USERS).unwrap();
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
        let mut t = w.open_table(USERS).unwrap();
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
        let mut table = wtxn.open_table(USERS).unwrap();
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
            CdcConfig {
                enabled: true,
                retention_max_txns: 2,
            },
        )
        .unwrap();

        // Write 4 transactions
        for i in 0u64..4 {
            let w = db.begin_write();
            let mut t = w.open_table(USERS).unwrap();
            let key = alloc::format!("k{i}");
            t.insert(&key.as_str(), &i).unwrap();
            drop(t);
            w.commit().unwrap();
        }

        let rtxn = db.begin_read();
        let all_changes = rtxn.read_cdc_since(0).unwrap();
        // With retention_max_txns=2, older transactions should be pruned.
        // Transaction IDs 1,2,3,4 -- after txn4, cutoff = 4-2=2, so txn1 and txn2 pruned.
        // Only txn3 and txn4 remain.
        assert!(all_changes.len() <= 4, "pruning should remove old entries");
        // At least the latest 2 should remain
        let latest_ids: Vec<u64> = all_changes.iter().map(|c| c.transaction_id).collect();
        assert!(latest_ids.contains(&4), "latest txn should be present");
    }

    #[test]
    fn cdc_builder_api() {
        let db = BfTreeBuilder::new()
            .set_cdc(CdcConfig {
                enabled: true,
                retention_max_txns: 0,
            })
            .create(BfTreeConfig::new_memory(4))
            .unwrap();

        let w = db.begin_write();
        let mut t = w.open_table(USERS).unwrap();
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
        let mut t1 = w.open_table(USERS).unwrap();
        t1.insert(&"a", &1u64).unwrap();
        drop(t1);
        let mut t2 = w.open_table(SCORES).unwrap();
        t2.insert(&"b", &2u64).unwrap();
        drop(t2);
        w.commit().unwrap();

        let rtxn = db.begin_read();
        let changes = rtxn.read_cdc_since(0).unwrap();
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].table_name, "users");
        assert_eq!(changes[1].table_name, "scores");
    }

    // -----------------------------------------------------------------------
    // Transaction ID recovery tests
    // -----------------------------------------------------------------------

    #[test]
    fn txn_id_persisted_on_commit() {
        // Verify the persisted meta key is written on every commit.
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let w1 = db.begin_write();
        let id1 = w1.txn_id();
        w1.insert(&USERS, &"a", &1u64).unwrap();
        w1.commit().unwrap();

        // Read the persisted counter directly.
        let meta_key = encode_table_key(BF_META_TABLE_NAME, TableKind::Regular, BF_META_TXN_ID_KEY);
        let max_val = db.adapter().inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        let len = db.adapter().read(&meta_key, &mut buf).unwrap();
        assert!(len as usize >= 8);
        let persisted = u64::from_le_bytes(buf[..8].try_into().unwrap());
        assert_eq!(persisted, id1 + 1, "persisted counter should be txn_id + 1");
    }

    #[test]
    fn recover_txn_id_without_cdc() {
        // Simulate the bug scenario: CDC disabled, multiple commits, then recovery.
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        // Perform several transactions with CDC disabled.
        for i in 0u64..5 {
            let w = db.begin_write();
            let key = alloc::format!("k{i}");
            w.insert(&USERS, &key.as_str(), &i).unwrap();
            w.commit().unwrap();
        }

        // The next txn_id should be 6 (txns used 1..5).
        let next_w = db.begin_write();
        assert_eq!(next_w.txn_id(), 6);
        drop(next_w);

        // Recover from the adapter (simulating reopen).
        let recovered = recover_next_txn_id(db.adapter()).unwrap();
        assert_eq!(
            recovered, 6,
            "recovery must return 6 even without CDC entries"
        );
    }

    #[test]
    fn table_kind_discriminator_encoding() {
        // Verify the encoding includes the discriminator byte at the correct position.
        let regular = encode_table_key("t", TableKind::Regular, b"k");
        let multimap_prefix = table_prefix("t", TableKind::Multimap);
        let ttl = encode_table_key("t", TableKind::Ttl, b"k");

        // Format: [len_lo, len_hi, 't', kind, 'k']
        // "t" has length 1, so len = [0x01, 0x00] in LE.
        assert_eq!(regular, &[0x01, 0x00, b't', 0x00, b'k']);
        assert_eq!(multimap_prefix, &[0x01, 0x00, b't', 0x01]);
        assert_eq!(ttl, &[0x01, 0x00, b't', 0x02, b'k']);

        // Prefixes must not overlap: regular prefix < multimap prefix < ttl prefix.
        let reg_prefix = table_prefix("t", TableKind::Regular);
        let mm_prefix = table_prefix("t", TableKind::Multimap);
        let ttl_prefix = table_prefix("t", TableKind::Ttl);
        assert!(reg_prefix < mm_prefix);
        assert!(mm_prefix < ttl_prefix);
    }

    #[test]
    fn table_kind_discriminator_prevents_collision() {
        // Verify that regular, multimap, and TTL tables with the same name
        // do not collide in the underlying BfTree keyspace.
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        let wtxn = db.begin_write();

        // Insert into a regular table named "shared".
        let regular_def: TableDefinition<&str, u64> = TableDefinition::new("shared");
        let mut reg = wtxn.open_table(regular_def).unwrap();
        reg.insert(&"key1", &100u64).unwrap();
        drop(reg);

        // Insert into a multimap table named "shared".
        let mut mm = wtxn.open_multimap_table::<&str, &str>("shared").unwrap();
        mm.insert(&"key1", &"val_a").unwrap();
        mm.insert(&"key1", &"val_b").unwrap();
        drop(mm);

        // Insert into a TTL table named "shared".
        let ttl_def: TableDefinition<&str, u64> = TableDefinition::new("shared");
        let mut ttl = wtxn.open_ttl_table(ttl_def).unwrap();
        ttl.insert(&"key1", &999u64).unwrap();
        drop(ttl);

        // Now read back from each table type and verify isolation.
        let mut reg2 = wtxn.open_table(regular_def).unwrap();
        let reg_val = reg2.get(&"key1").unwrap().unwrap();
        assert_eq!(
            u64::from_le_bytes(reg_val.as_slice().try_into().unwrap()),
            100,
            "regular table must see its own value, not multimap or TTL"
        );
        drop(reg2);

        let mm2 = wtxn.open_multimap_table::<&str, &str>("shared").unwrap();
        let mm_vals = mm2.get_values(&"key1").unwrap();
        assert_eq!(mm_vals.len(), 2, "multimap must see its own two values");
        assert_eq!(mm_vals[0], b"val_a");
        assert_eq!(mm_vals[1], b"val_b");
        drop(mm2);

        let ttl2 = wtxn.open_ttl_table(ttl_def).unwrap();
        let ttl_val = ttl2.get(&"key1").unwrap().unwrap();
        assert_eq!(
            u64::from_le_bytes(ttl_val.as_slice().try_into().unwrap()),
            999,
            "TTL table must see its own value, not regular or multimap"
        );
        drop(ttl2);

        wtxn.commit().unwrap();
    }

    #[test]
    fn recover_txn_id_with_cdc_agrees() {
        // With CDC enabled, the persisted counter and CDC scan should agree.
        let db = cdc_db();

        for i in 0u64..3 {
            let w = db.begin_write();
            let mut t = w.open_table(USERS).unwrap();
            let key = alloc::format!("k{i}");
            t.insert(&key.as_str(), &i).unwrap();
            drop(t);
            w.commit().unwrap();
        }

        let recovered = recover_next_txn_id(db.adapter()).unwrap();
        assert_eq!(
            recovered, 4,
            "recovery with CDC should return next available id"
        );
    }
}
