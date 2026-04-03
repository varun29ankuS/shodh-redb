//! TTL (Time-To-Live) table support for `BfTree`.
//!
//! Wraps standard `BfTree` table operations with an 8-byte expiry header prepended
//! to each value. Expired entries are transparent to callers — `get()` and range
//! scans return `None` for expired keys. Use `purge_expired()` to reclaim storage.
//!
//! # Value Encoding
//!
//! ```text
//! [u64 LE expires_at_ms][original value bytes]
//! ├── 8-byte header: milliseconds since UNIX epoch (0 = never expires)
//! └── Remaining bytes: the raw V serialized value
//! ```

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::types::{Key, Value};

use super::adapter::BfTreeAdapter;
use super::buffered_txn::{BufferLookup, WriteBuffer};
use super::database::{encode_table_key, table_prefix, table_prefix_end};
use super::error::BfTreeError;

const EXPIRY_HEADER_SIZE: usize = 8;

/// Returns the current wall-clock time as milliseconds since the UNIX epoch.
///
/// # Clock skew
///
/// This function uses `SystemTime`, which is subject to wall-clock adjustments
/// (NTP, manual changes, suspend/resume). Consequently:
/// - TTL expiry is best-effort and not monotonically precise.
/// - A backward clock jump may delay expiry; a forward jump may expire entries early.
/// - For most use cases this is acceptable. Applications requiring strict ordering
///   guarantees should use an external monotonic clock or logical timestamps.
#[allow(clippy::cast_possible_truncation)]
fn now_millis() -> u64 {
    // u128 → u64 truncation is safe: milliseconds since epoch will not
    // overflow u64 for ~584 million years.
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn encode_ttl_value(expires_at_ms: u64, value: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(EXPIRY_HEADER_SIZE + value.len());
    buf.extend_from_slice(&expires_at_ms.to_le_bytes());
    buf.extend_from_slice(value);
    buf
}

fn read_expiry(data: &[u8]) -> u64 {
    if data.len() < EXPIRY_HEADER_SIZE {
        return 0;
    }
    u64::from_le_bytes(data[..EXPIRY_HEADER_SIZE].try_into().unwrap())
}

fn strip_expiry(data: &[u8]) -> &[u8] {
    if data.len() > EXPIRY_HEADER_SIZE {
        &data[EXPIRY_HEADER_SIZE..]
    } else {
        &[]
    }
}

fn is_expired(expires_at_ms: u64) -> bool {
    expires_at_ms != 0 && expires_at_ms <= now_millis()
}

// ---------------------------------------------------------------------------
// Writable TTL table
// ---------------------------------------------------------------------------

/// A writable table with per-entry TTL, backed by `BfTree`.
///
/// Each value is stored with an 8-byte expiry header. Reads automatically
/// filter expired entries. Writes go through the transaction buffer for
/// atomic commit/rollback semantics.
pub struct BfTreeTtlTable<'txn, K: Key + 'static, V: Value + 'static> {
    name: String,
    adapter: &'txn Arc<BfTreeAdapter>,
    ops_count: &'txn AtomicU64,
    buffer: &'txn Mutex<WriteBuffer>,
    _key: PhantomData<K>,
    _val: PhantomData<V>,
}

impl<'txn, K: Key + 'static, V: Value + 'static> BfTreeTtlTable<'txn, K, V> {
    pub(crate) fn new(
        name: &str,
        adapter: &'txn Arc<BfTreeAdapter>,
        ops_count: &'txn AtomicU64,
        buffer: &'txn Mutex<WriteBuffer>,
    ) -> Self {
        Self {
            name: String::from(name),
            adapter,
            ops_count,
            buffer,
            _key: PhantomData,
            _val: PhantomData,
        }
    }

    /// Insert without expiry (never expires, expiry = 0).
    pub fn insert(
        &mut self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
    ) -> Result<Option<Vec<u8>>, BfTreeError> {
        self.insert_internal(key, value, 0)
    }

    /// Insert with a TTL duration from now.
    pub fn insert_with_ttl(
        &mut self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
        ttl: Duration,
    ) -> Result<Option<Vec<u8>>, BfTreeError> {
        #[allow(clippy::cast_possible_truncation)]
        // u128 → u64 truncation is safe: TTL durations will not exceed u64::MAX ms.
        let expires_at_ms = now_millis().saturating_add(ttl.as_millis() as u64);
        self.insert_internal(key, value, expires_at_ms)
    }

    /// Insert with an explicit expiry timestamp (ms since UNIX epoch, 0 = never).
    pub fn insert_with_expiry(
        &mut self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
        expires_at_ms: u64,
    ) -> Result<Option<Vec<u8>>, BfTreeError> {
        self.insert_internal(key, value, expires_at_ms)
    }

    fn insert_internal(
        &mut self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
        expires_at_ms: u64,
    ) -> Result<Option<Vec<u8>>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let val_bytes = V::as_bytes(value);
        let encoded_key = encode_table_key(&self.name, key_bytes.as_ref());
        let wrapped = encode_ttl_value(expires_at_ms, val_bytes.as_ref());

        let mut buffer = self.buffer.lock().unwrap();

        let previous_raw = self.read_raw_locked(&buffer, &encoded_key)?;
        buffer.put(encoded_key, wrapped);
        drop(buffer);
        self.ops_count.fetch_add(1, Ordering::Relaxed);

        Ok(Self::unwrap_if_not_expired(previous_raw))
    }

    /// Get a value, returning `None` if expired or absent.
    pub fn get(&self, key: &K::SelfType<'_>) -> Result<Option<Vec<u8>>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(&self.name, key_bytes.as_ref());

        let buffer = self.buffer.lock().unwrap();
        let raw = self.read_raw_locked(&buffer, &encoded_key)?;
        drop(buffer);

        Ok(Self::unwrap_if_not_expired(raw))
    }

    /// Get the expiry timestamp (ms since epoch) for a key, or `None` if absent.
    pub fn expires_at_ms(&self, key: &K::SelfType<'_>) -> Result<Option<u64>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(&self.name, key_bytes.as_ref());

        let buffer = self.buffer.lock().unwrap();
        let raw = self.read_raw_locked(&buffer, &encoded_key)?;
        drop(buffer);

        Ok(raw.map(|data| read_expiry(&data)))
    }

    /// Remove a key, returning the previous non-expired value if any.
    pub fn remove(&mut self, key: &K::SelfType<'_>) -> Result<Option<Vec<u8>>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(&self.name, key_bytes.as_ref());

        let mut buffer = self.buffer.lock().unwrap();
        let previous_raw = self.read_raw_locked(&buffer, &encoded_key)?;

        if previous_raw.is_some() {
            buffer.delete(encoded_key);
            drop(buffer);
            self.ops_count.fetch_add(1, Ordering::Relaxed);
        } else {
            drop(buffer);
        }

        Ok(Self::unwrap_if_not_expired(previous_raw))
    }

    /// Purge all expired entries. Returns the number of entries purged.
    ///
    /// # Consistency model
    ///
    /// TTL purge is best-effort with eventual consistency semantics. Phase 1
    /// scans `BfTree` without the buffer lock, so entries may be inserted or
    /// modified concurrently between the scan and the tombstoning in Phase 2.
    /// Phase 2 mitigates this by re-reading each entry through the buffer
    /// (which reflects the latest state) and re-checking the expiry timestamp
    /// before tombstoning, closing the TOCTOU window for entries that were
    /// updated or removed after the scan.
    pub fn purge_expired(&mut self) -> Result<u64, BfTreeError> {
        let prefix = table_prefix(&self.name);
        let prefix_end = table_prefix_end(&self.name);
        let max_record_size = self.adapter.inner().config().get_cb_max_record_size();

        // Phase 1: Collect all BfTree key candidates (scan must complete before
        // buffer mutation). We only collect keys here; the actual expiry check
        // happens in Phase 2 under the buffer lock.
        let bftree_keys: Vec<Vec<u8>> = {
            let mut buf = vec![0u8; max_record_size * 2];
            let mut result = Vec::new();
            let mut iter = self.adapter.scan_range(&prefix, &prefix_end)?;
            while let Some((key_len, _val_len)) = iter.next(&mut buf) {
                result.push(buf[..key_len].to_vec());
            }
            result
        };

        let mut buffer = self.buffer.lock().unwrap();
        let mut purged = 0u64;

        // Phase 2: Re-read each entry through the buffer-aware path and
        // re-check expiry before tombstoning. This closes the TOCTOU window:
        // if an entry was updated or deleted between Phase 1 and now, the
        // buffer read reflects the current state.
        for encoded_key in &bftree_keys {
            let raw = match buffer.get(encoded_key) {
                BufferLookup::Found(v) => v,
                BufferLookup::Tombstone => continue,
                BufferLookup::NotInBuffer => {
                    // Re-read from BfTree to get the current value.
                    let mut val_buf = vec![0u8; max_record_size];
                    match self.adapter.read(encoded_key, &mut val_buf) {
                        Ok(len) => val_buf[..len as usize].to_vec(),
                        Err(BfTreeError::NotFound | BfTreeError::Deleted) => continue,
                        Err(e) => return Err(e),
                    }
                }
            };
            if is_expired(read_expiry(&raw)) {
                buffer.delete(encoded_key.clone());
                purged += 1;
            }
        }

        // Phase 3: Check buffer-only inserts for expiry.
        let buf_only: Vec<(Vec<u8>, Vec<u8>)> = buffer
            .range(&prefix, &prefix_end)
            .filter_map(|(k, v)| {
                if let Some(val) = v {
                    // Only process entries not in the BfTree set (buffer-only).
                    if !bftree_keys.iter().any(|bk| bk == k) {
                        return Some((k.clone(), val.clone()));
                    }
                }
                None
            })
            .collect();

        for (encoded_key, raw_val) in buf_only {
            if is_expired(read_expiry(&raw_val)) {
                buffer.delete(encoded_key);
                purged += 1;
            }
        }

        drop(buffer);
        self.ops_count.fetch_add(purged, Ordering::Relaxed);
        Ok(purged)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Read raw (wrapped) value: buffer first, then `BfTree`. Returns `None` for
    /// tombstoned or absent keys.
    fn read_raw_locked(
        &self,
        buffer: &WriteBuffer,
        encoded_key: &[u8],
    ) -> Result<Option<Vec<u8>>, BfTreeError> {
        match buffer.get(encoded_key) {
            BufferLookup::Found(v) => Ok(Some(v)),
            BufferLookup::Tombstone => Ok(None),
            BufferLookup::NotInBuffer => {
                let max_val = self.adapter.inner().config().get_cb_max_record_size();
                let mut buf = vec![0u8; max_val];
                match self.adapter.read(encoded_key, &mut buf) {
                    Ok(len) => Ok(Some(buf[..len as usize].to_vec())),
                    Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// Strip expiry header and return value bytes, or `None` if expired.
    fn unwrap_if_not_expired(raw: Option<Vec<u8>>) -> Option<Vec<u8>> {
        raw.and_then(|data| {
            let exp = read_expiry(&data);
            if is_expired(exp) {
                None
            } else {
                Some(strip_expiry(&data).to_vec())
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Read-only TTL table
// ---------------------------------------------------------------------------

/// A read-only TTL table backed by `BfTree`.
///
/// Expired entries are automatically filtered on read.
pub struct BfTreeReadOnlyTtlTable<'txn, K: Key + 'static, V: Value + 'static> {
    name: String,
    adapter: &'txn Arc<BfTreeAdapter>,
    _key: PhantomData<K>,
    _val: PhantomData<V>,
}

impl<'txn, K: Key + 'static, V: Value + 'static> BfTreeReadOnlyTtlTable<'txn, K, V> {
    pub(crate) fn new(name: &str, adapter: &'txn Arc<BfTreeAdapter>) -> Self {
        Self {
            name: String::from(name),
            adapter,
            _key: PhantomData,
            _val: PhantomData,
        }
    }

    /// Get a value, returning `None` if expired or absent.
    pub fn get(&self, key: &K::SelfType<'_>) -> Result<Option<Vec<u8>>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(&self.name, key_bytes.as_ref());
        let max_val = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.adapter.read(&encoded_key, &mut buf) {
            Ok(len) => {
                let raw = &buf[..len as usize];
                let exp = read_expiry(raw);
                if is_expired(exp) {
                    Ok(None)
                } else {
                    Ok(Some(strip_expiry(raw).to_vec()))
                }
            }
            Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Get the expiry timestamp (ms since epoch) for a key.
    pub fn expires_at_ms(&self, key: &K::SelfType<'_>) -> Result<Option<u64>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(&self.name, key_bytes.as_ref());
        let max_val = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.adapter.read(&encoded_key, &mut buf) {
            Ok(len) => Ok(Some(read_expiry(&buf[..len as usize]))),
            Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TableDefinition;
    use crate::bf_tree_store::config::BfTreeConfig;
    use crate::bf_tree_store::database::BfTreeDatabase;

    const TTL_TABLE: TableDefinition<&str, u64> = TableDefinition::new("ttl_items");

    #[test]
    fn insert_and_get_no_expiry() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_ttl_table(TTL_TABLE);

        table.insert(&"key", &42u64).unwrap();
        let val = table.get(&"key").unwrap().unwrap();
        assert_eq!(u64::from_le_bytes(val.as_slice().try_into().unwrap()), 42);
    }

    #[test]
    fn expired_entry_returns_none() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_ttl_table(TTL_TABLE);

        // Insert with expiry in the past.
        table.insert_with_expiry(&"old", &99u64, 1).unwrap();
        // Should be expired.
        assert!(table.get(&"old").unwrap().is_none());
    }

    #[test]
    fn future_expiry_is_visible() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_ttl_table(TTL_TABLE);

        table
            .insert_with_ttl(&"future", &77u64, Duration::from_secs(3600))
            .unwrap();
        let val = table.get(&"future").unwrap().unwrap();
        assert_eq!(u64::from_le_bytes(val.as_slice().try_into().unwrap()), 77);
    }

    #[test]
    fn expires_at_ms_returns_timestamp() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_ttl_table(TTL_TABLE);

        table.insert(&"no_exp", &1u64).unwrap();
        assert_eq!(table.expires_at_ms(&"no_exp").unwrap(), Some(0));

        table
            .insert_with_expiry(&"with_exp", &2u64, 999_999)
            .unwrap();
        assert_eq!(table.expires_at_ms(&"with_exp").unwrap(), Some(999_999));
    }

    #[test]
    fn remove_returns_previous() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_ttl_table(TTL_TABLE);

        table.insert(&"k", &10u64).unwrap();
        let removed = table.remove(&"k").unwrap();
        assert!(removed.is_some());
        assert!(table.get(&"k").unwrap().is_none());
    }

    #[test]
    fn purge_expired_removes_stale() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_ttl_table(TTL_TABLE);

        table.insert(&"alive", &1u64).unwrap();
        table.insert_with_expiry(&"dead1", &2u64, 1).unwrap();
        table.insert_with_expiry(&"dead2", &3u64, 1).unwrap();

        let purged = table.purge_expired().unwrap();
        assert_eq!(purged, 2);

        // Alive entry still accessible.
        assert!(table.get(&"alive").unwrap().is_some());
    }

    #[test]
    fn ttl_survives_commit() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

        {
            let wtxn = db.begin_write();
            let mut table = wtxn.open_ttl_table(TTL_TABLE);
            table
                .insert_with_ttl(&"durable", &55u64, Duration::from_secs(3600))
                .unwrap();
            drop(table);
            wtxn.commit().unwrap();
        }

        // Read via read-only TTL table.
        let rtxn = db.begin_read();
        let ro = rtxn.open_ttl_table(TTL_TABLE);
        let val = ro.get(&"durable").unwrap().unwrap();
        assert_eq!(u64::from_le_bytes(val.as_slice().try_into().unwrap()), 55);
    }
}
