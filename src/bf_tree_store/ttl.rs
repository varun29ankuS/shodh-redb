//! TTL (Time-To-Live) table support for `BfTree`.
//!
//! Wraps standard `BfTree` table operations with a 9-byte header (1-byte magic tag +
//! 8-byte expiry timestamp) prepended to each value. Expired entries are transparent
//! to callers -- `get()` and range scans return `None` for expired keys. Use
//! `purge_expired()` to reclaim storage.
//!
//! # Value Encoding
//!
//! ```text
//! [0xE7][u64 LE expires_at_ms][original value bytes]
//! +-- 1-byte magic tag (TTL_MAGIC = 0xE7): distinguishes TTL values from plain values
//! +-- 8-byte header: milliseconds since UNIX epoch (0 = never expires)
//! +-- Remaining bytes: the raw V serialized value
//! ```

use crate::compat::Mutex;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cdc::types::{CdcEvent, ChangeOp};
use crate::types::{Key, Value};

use super::adapter::BfTreeAdapter;
use super::buffered_txn::{BufferLookup, WriteBuffer};
use super::database::{TableKind, encode_table_key, table_prefix, table_prefix_end};
use super::error::BfTreeError;

/// Magic byte prefix that distinguishes TTL-encoded values from plain values.
/// If the same table is opened as both a TTL table and a regular table, this
/// tag prevents misinterpreting the 8-byte expiry header as user data (or
/// vice versa).
const TTL_MAGIC: u8 = 0xE7;

/// Total header size: 1-byte magic tag + 8-byte expiry timestamp.
const EXPIRY_HEADER_SIZE: usize = 1 + 8;

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
    // u128 -> u64 truncation is safe: milliseconds since epoch will not
    // overflow u64 for ~584 million years.
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    // TTL-04: If the system clock is before the UNIX epoch (or returns 0),
    // all non-zero expiry timestamps would compare as `expires_at < 0` which
    // is always false for u64, making every entry immortal. Use a minimum
    // floor of 1 so that expiry comparisons remain meaningful.
    if ms == 0 { 1 } else { ms }
}

fn encode_ttl_value(expires_at_ms: u64, value: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(EXPIRY_HEADER_SIZE + value.len());
    buf.push(TTL_MAGIC);
    buf.extend_from_slice(&expires_at_ms.to_le_bytes());
    buf.extend_from_slice(value);
    buf
}

/// Read the expiry timestamp from a TTL-encoded value.
///
/// Returns `Ok(timestamp)` on success, or `Err(())` if the value is missing
/// the TTL magic byte or is too short to contain a valid header. The caller
/// must handle the error (e.g., return `BfTreeError::Corruption`).
fn read_expiry(data: &[u8]) -> Result<u64, ()> {
    // TTL-03 / TTL-06: Validate magic byte and minimum header size.
    if data.len() < EXPIRY_HEADER_SIZE || data[0] != TTL_MAGIC {
        return Err(());
    }
    let mut ts_buf = [0u8; 8];
    ts_buf.copy_from_slice(&data[1..EXPIRY_HEADER_SIZE]);
    Ok(u64::from_le_bytes(ts_buf))
}

fn strip_expiry(data: &[u8]) -> &[u8] {
    if data.len() > EXPIRY_HEADER_SIZE && data[0] == TTL_MAGIC {
        &data[EXPIRY_HEADER_SIZE..]
    } else {
        &[]
    }
}

/// Check if a TTL timestamp is expired relative to a given `now` value.
///
/// TTL-08: The `now` parameter should be captured once at method entry to
/// ensure consistent expiry decisions within a single operation.
fn is_expired_at(expires_at_ms: u64, now: u64) -> bool {
    // TTL-01: Use strict less-than so that an entry whose expiry equals the
    // current millisecond is still readable during that millisecond. The
    // previous `<=` caused entries with a TTL landing exactly on `now` to be
    // treated as expired the instant they were written.
    expires_at_ms != 0 && expires_at_ms < now
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
    cdc_log: Option<&'txn Mutex<Vec<CdcEvent>>>,
    buffer: &'txn Mutex<WriteBuffer>,
    _key: PhantomData<K>,
    _val: PhantomData<V>,
}

impl<'txn, K: Key + 'static, V: Value + 'static> BfTreeTtlTable<'txn, K, V> {
    pub(crate) fn new(
        name: &str,
        adapter: &'txn Arc<BfTreeAdapter>,
        ops_count: &'txn AtomicU64,
        cdc_log: Option<&'txn Mutex<Vec<CdcEvent>>>,
        buffer: &'txn Mutex<WriteBuffer>,
    ) -> Self {
        Self {
            name: String::from(name),
            adapter,
            ops_count,
            cdc_log,
            buffer,
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
        // Clamp the u128 millisecond value to u64::MAX before adding the current
        // timestamp. Without this, a very large Duration (e.g. Duration::MAX)
        // would silently truncate via `as u64`, wrapping the high bits to a small
        // value and causing immediate or near-immediate expiry.
        let ttl_ms_u128 = ttl.as_millis();
        let ttl_ms = if ttl_ms_u128 > u128::from(u64::MAX) {
            u64::MAX
        } else {
            #[allow(clippy::cast_possible_truncation)] // guarded by the u64::MAX check above
            {
                ttl_ms_u128 as u64
            }
        };
        let expires_at_ms = now_millis().saturating_add(ttl_ms);
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
        // TTL-08: Capture timestamp once for consistent expiry decisions.
        let now = now_millis();
        let key_bytes = K::as_bytes(key);
        let val_bytes = V::as_bytes(value);
        let encoded_key = encode_table_key(&self.name, TableKind::Ttl, key_bytes.as_ref());
        let wrapped = encode_ttl_value(expires_at_ms, val_bytes.as_ref());

        let mut buffer = self.buffer.lock();

        let previous_raw = self.read_raw_locked(&buffer, &encoded_key)?;
        buffer.put(encoded_key, wrapped)?;
        drop(buffer);
        self.ops_count.fetch_add(1, Ordering::Relaxed);

        // Record CDC event if enabled.
        if self.cdc_log.is_some() {
            let previous_value = Self::unwrap_if_not_expired(previous_raw.clone(), now);
            self.record_cdc(CdcEvent {
                table_name: self.name.clone(),
                op: if previous_value.is_some() {
                    ChangeOp::Update
                } else {
                    ChangeOp::Insert
                },
                key: key_bytes.as_ref().to_vec(),
                new_value: Some(val_bytes.as_ref().to_vec()),
                old_value: previous_value,
            });
        }

        Ok(Self::unwrap_if_not_expired(previous_raw, now))
    }

    /// Get a value, returning `None` if expired or absent.
    pub fn get(&self, key: &K::SelfType<'_>) -> Result<Option<Vec<u8>>, BfTreeError> {
        // TTL-08: Capture timestamp once for consistent expiry decisions.
        let now = now_millis();
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(&self.name, TableKind::Ttl, key_bytes.as_ref());

        let buffer = self.buffer.lock();
        let raw = self.read_raw_locked(&buffer, &encoded_key)?;
        drop(buffer);

        Ok(Self::unwrap_if_not_expired(raw, now))
    }

    /// Get the expiry timestamp (ms since epoch) for a key, or `None` if absent.
    pub fn expires_at_ms(&self, key: &K::SelfType<'_>) -> Result<Option<u64>, BfTreeError> {
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(&self.name, TableKind::Ttl, key_bytes.as_ref());

        let buffer = self.buffer.lock();
        let raw = self.read_raw_locked(&buffer, &encoded_key)?;
        drop(buffer);

        match raw {
            Some(data) => match read_expiry(&data) {
                Ok(ts) => Ok(Some(ts)),
                Err(()) => Err(BfTreeError::Corruption(alloc::format!(
                    "ttl table {:?}: value missing TTL header",
                    self.name
                ))),
            },
            None => Ok(None),
        }
    }

    /// Remove a key, returning the previous non-expired value if any.
    pub fn remove(&mut self, key: &K::SelfType<'_>) -> Result<Option<Vec<u8>>, BfTreeError> {
        // TTL-08: Capture timestamp once for consistent expiry decisions.
        let now = now_millis();
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(&self.name, TableKind::Ttl, key_bytes.as_ref());

        let mut buffer = self.buffer.lock();
        let previous_raw = self.read_raw_locked(&buffer, &encoded_key)?;

        let previous_value = Self::unwrap_if_not_expired(previous_raw, now);

        if previous_value.is_some() {
            buffer.delete(encoded_key);
            drop(buffer);
            self.ops_count.fetch_add(1, Ordering::Relaxed);

            // Record CDC event if enabled.
            self.record_cdc(CdcEvent {
                table_name: self.name.clone(),
                op: ChangeOp::Delete,
                key: key_bytes.as_ref().to_vec(),
                new_value: None,
                old_value: previous_value.clone(),
            });
        } else {
            drop(buffer);
        }

        Ok(previous_value)
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
        // TTL-08: Capture timestamp once for consistent expiry decisions.
        let now = now_millis();
        let prefix = table_prefix(&self.name, TableKind::Ttl);
        let prefix_end = table_prefix_end(&self.name, TableKind::Ttl);
        let prefix_len = prefix.len();
        let max_record_size = self.adapter.inner().config().get_cb_max_record_size();

        // Phase 1: Collect all BfTree key candidates (scan must complete before
        // buffer mutation). We only collect keys here; the actual expiry check
        // happens in Phase 2 under the buffer lock.
        let bftree_keys: Vec<Vec<u8>> = {
            let mut buf = vec![0u8; max_record_size * 2];
            let mut result = Vec::new();
            let mut iter = self.adapter.scan_range(&prefix, &prefix_end)?;
            while let Ok(Some((key_len, _val_len))) = iter.next(&mut buf) {
                result.push(buf[..key_len].to_vec());
            }
            result
        };

        let mut buffer = self.buffer.lock();
        let mut purged = 0u64;
        // Collect CDC events for purged entries to emit after releasing the
        // buffer lock. Each entry is (user_key_bytes, old_value_bytes).
        let mut purged_entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

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
            if let Ok(exp) = read_expiry(&raw)
                && is_expired_at(exp, now)
            {
                buffer.delete(encoded_key.clone());
                purged += 1;
                if self.cdc_log.is_some() {
                    let user_key = encoded_key.get(prefix_len..).unwrap_or(&[]).to_vec();
                    let old_value = strip_expiry(&raw).to_vec();
                    purged_entries.push((user_key, old_value));
                }
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
            if let Ok(exp) = read_expiry(&raw_val)
                && is_expired_at(exp, now)
            {
                if self.cdc_log.is_some() {
                    let user_key = encoded_key.get(prefix_len..).unwrap_or(&[]).to_vec();
                    let old_value = strip_expiry(&raw_val).to_vec();
                    purged_entries.push((user_key, old_value));
                }
                buffer.delete(encoded_key);
                purged += 1;
            }
        }

        drop(buffer);
        self.ops_count.fetch_add(purged, Ordering::Relaxed);

        // Record CDC delete events for all purged entries.
        for (user_key, old_value) in purged_entries {
            self.record_cdc(CdcEvent {
                table_name: self.name.clone(),
                op: ChangeOp::Delete,
                key: user_key,
                new_value: None,
                old_value: Some(old_value),
            });
        }

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

    /// Strip expiry header and return value bytes, or `None` if expired or
    /// if the value does not carry a valid TTL header (missing magic byte or
    /// sub-header-size).
    ///
    /// TTL-08: `now` is captured once at the top of the calling method.
    fn unwrap_if_not_expired(raw: Option<Vec<u8>>, now: u64) -> Option<Vec<u8>> {
        raw.and_then(|data| {
            let exp = read_expiry(&data).ok()?;
            if is_expired_at(exp, now) {
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
        // TTL-08: Capture timestamp once for consistent expiry decisions.
        let now = now_millis();
        let key_bytes = K::as_bytes(key);
        let encoded_key = encode_table_key(&self.name, TableKind::Ttl, key_bytes.as_ref());
        let max_val = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.adapter.read(&encoded_key, &mut buf) {
            Ok(len) => {
                let raw = &buf[..len as usize];
                let Ok(exp) = read_expiry(raw) else {
                    return Err(BfTreeError::Corruption(alloc::format!(
                        "ttl table {:?}: value missing TTL header",
                        self.name
                    )));
                };
                if is_expired_at(exp, now) {
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
        let encoded_key = encode_table_key(&self.name, TableKind::Ttl, key_bytes.as_ref());
        let max_val = self.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.adapter.read(&encoded_key, &mut buf) {
            Ok(len) => match read_expiry(&buf[..len as usize]) {
                Ok(ts) => Ok(Some(ts)),
                Err(()) => Err(BfTreeError::Corruption(alloc::format!(
                    "ttl table {:?}: value missing TTL header",
                    self.name
                ))),
            },
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
        let mut table = wtxn.open_ttl_table(TTL_TABLE).unwrap();

        table.insert(&"key", &42u64).unwrap();
        let val = table.get(&"key").unwrap().unwrap();
        assert_eq!(u64::from_le_bytes(val.as_slice().try_into().unwrap()), 42);
    }

    #[test]
    fn expired_entry_returns_none() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_ttl_table(TTL_TABLE).unwrap();

        // Insert with expiry in the past.
        table.insert_with_expiry(&"old", &99u64, 1).unwrap();
        // Should be expired.
        assert!(table.get(&"old").unwrap().is_none());
    }

    #[test]
    fn future_expiry_is_visible() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_ttl_table(TTL_TABLE).unwrap();

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
        let mut table = wtxn.open_ttl_table(TTL_TABLE).unwrap();

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
        let mut table = wtxn.open_ttl_table(TTL_TABLE).unwrap();

        table.insert(&"k", &10u64).unwrap();
        let removed = table.remove(&"k").unwrap();
        assert!(removed.is_some());
        assert!(table.get(&"k").unwrap().is_none());
    }

    #[test]
    fn purge_expired_removes_stale() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_ttl_table(TTL_TABLE).unwrap();

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
            let mut table = wtxn.open_ttl_table(TTL_TABLE).unwrap();
            table
                .insert_with_ttl(&"durable", &55u64, Duration::from_secs(3600))
                .unwrap();
            drop(table);
            wtxn.commit().unwrap();
        }

        // Read via read-only TTL table.
        let rtxn = db.begin_read();
        let ro = rtxn.open_ttl_table(TTL_TABLE).unwrap();
        let val = ro.get(&"durable").unwrap().unwrap();
        assert_eq!(u64::from_le_bytes(val.as_slice().try_into().unwrap()), 55);
    }

    #[test]
    fn huge_ttl_does_not_wrap_to_near_zero_expiry() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
        let wtxn = db.begin_write();
        let mut table = wtxn.open_ttl_table(TTL_TABLE).unwrap();

        // Duration::MAX has u128 milliseconds far exceeding u64::MAX.
        // Without the clamp fix, `as u64` truncation would wrap to a small
        // value, causing the entry to expire immediately.
        table
            .insert_with_ttl(&"huge", &123u64, Duration::MAX)
            .unwrap();

        // The entry must still be visible (not expired).
        let val = table.get(&"huge").unwrap();
        assert!(
            val.is_some(),
            "entry with Duration::MAX TTL must not expire immediately"
        );

        // The stored expiry must be u64::MAX (saturated).
        let expiry = table.expires_at_ms(&"huge").unwrap().unwrap();
        assert_eq!(expiry, u64::MAX);
    }
}
