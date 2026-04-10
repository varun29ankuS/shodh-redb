//! Multimap table support for `BfTree`.
//!
//! A multimap table maps each key to a set of values (one-to-many). It is
//! implemented via composite keys in the underlying `BfTree` KV store:
//!
//! ```text
//! Composite BfTree key:
//!   [table_name_len: u16 LE][table_name][0x01 kind][user_key_len: u32 LE][user_key_bytes][value_bytes]
//!
//! BfTree value: empty []
//! ```
//!
//! The value is encoded as part of the key, allowing efficient prefix scans
//! for all values of a given user key. Values within a key are inherently
//! sorted by their byte representation (`BfTree` lexicographic order).

use crate::compat::Mutex;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::cdc::types::{CdcEvent, ChangeOp};
use crate::types::Key;

use super::adapter::BfTreeAdapter;
use super::buffered_txn::{BufferLookup, WriteBuffer};
use super::database::TableKind;
use super::error::BfTreeError;

/// The discriminator byte for multimap tables, used in the key encoding to
/// prevent namespace collisions with regular and TTL tables.
const MULTIMAP_KIND: u8 = TableKind::Multimap as u8;

// ---------------------------------------------------------------------------
// Composite key encoding
// ---------------------------------------------------------------------------

/// Encode a full `BfTree` key for a multimap entry.
///
/// Format: `[table_name_len: u16 LE][table_name][0x01 kind][user_key_len: u32 LE][user_key][value_key]`
fn encode_multimap_key(
    table_name: &str,
    user_key: &[u8],
    value_key: &[u8],
) -> Result<Vec<u8>, BfTreeError> {
    let tbl = table_name.as_bytes();
    let tbl_len = u16::try_from(tbl.len()).map_err(|_| {
        BfTreeError::InvalidKV(alloc::format!(
            "multimap table name length {} exceeds u16::MAX",
            tbl.len()
        ))
    })?;
    let uk_len = u32::try_from(user_key.len()).map_err(|_| {
        BfTreeError::InvalidKV(alloc::format!(
            "multimap user key length {} exceeds u32::MAX",
            user_key.len()
        ))
    })?;

    let total = 2 + tbl.len() + 1 + 4 + user_key.len() + value_key.len();
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&tbl_len.to_le_bytes());
    buf.extend_from_slice(tbl);
    buf.push(MULTIMAP_KIND);
    buf.extend_from_slice(&uk_len.to_le_bytes());
    buf.extend_from_slice(user_key);
    buf.extend_from_slice(value_key);
    Ok(buf)
}

/// Compute the `BfTree` key prefix for all values of a user key.
///
/// Format: `[table_name_len: u16 LE][table_name][0x01 kind][user_key_len: u32 LE][user_key]`
fn multimap_key_prefix(table_name: &str, user_key: &[u8]) -> Result<Vec<u8>, BfTreeError> {
    let tbl = table_name.as_bytes();
    let tbl_len = u16::try_from(tbl.len()).map_err(|_| {
        BfTreeError::InvalidKV(alloc::format!(
            "multimap table name length {} exceeds u16::MAX",
            tbl.len()
        ))
    })?;
    let uk_len = u32::try_from(user_key.len()).map_err(|_| {
        BfTreeError::InvalidKV(alloc::format!(
            "multimap user key length {} exceeds u32::MAX",
            user_key.len()
        ))
    })?;

    let total = 2 + tbl.len() + 1 + 4 + user_key.len();
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&tbl_len.to_le_bytes());
    buf.extend_from_slice(tbl);
    buf.push(MULTIMAP_KIND);
    buf.extend_from_slice(&uk_len.to_le_bytes());
    buf.extend_from_slice(user_key);
    Ok(buf)
}

/// Compute the exclusive upper bound for a multimap key prefix.
///
/// Increments the prefix as a big-endian integer. Returns `None` if the
/// prefix is all 0xFF (overflow).
fn increment_prefix(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut result = prefix.to_vec();
    for i in (0..result.len()).rev() {
        if result[i] < 0xFF {
            result[i] += 1;
            return Some(result);
        }
        result[i] = 0x00;
    }
    None
}

/// Check whether a composite `BfTree` key belongs to the given multimap prefix.
///
/// Returns `true` when `composite_key` starts with exactly the bytes produced
/// by `multimap_key_prefix(table_name, user_key)`. This is used as a
/// post-filter for scan results when `increment_prefix` overflows (all-0xFF
/// prefix), ensuring we never leak entries from other user keys or tables.
fn key_matches_prefix(composite_key: &[u8], prefix: &[u8]) -> bool {
    composite_key.len() >= prefix.len() && composite_key[..prefix.len()] == *prefix
}

/// Compute the exclusive upper bound for scanning all values of a given user
/// key in a multimap table.
///
/// The prefix format is `[tbl_len: u16 LE][tbl][0x01 kind][uk_len: u32 LE][user_key]`.
/// When `increment_prefix` overflows (all-0xFF prefix), we return `None` to
/// indicate that callers must use a prefix-match filter on scan results
/// rather than relying on an upper bound key.
fn multimap_scan_end(table_name: &str, user_key: &[u8]) -> Result<Option<Vec<u8>>, BfTreeError> {
    let prefix = multimap_key_prefix(table_name, user_key)?;
    Ok(increment_prefix(&prefix))
}

/// Extract the value key portion from a composite key, given the known prefix length.
fn extract_value_key(composite_key: &[u8], prefix_len: usize) -> &[u8] {
    if composite_key.len() > prefix_len {
        &composite_key[prefix_len..]
    } else {
        &[]
    }
}

// ---------------------------------------------------------------------------
// Writable multimap table
// ---------------------------------------------------------------------------

/// A writable multimap table backed by `BfTree`.
///
/// Maps each key to a set of values. Values are stored as part of composite
/// `BfTree` keys with empty `BfTree` values, enabling efficient sorted iteration.
pub struct BfTreeMultimapTable<'txn, K: Key + 'static, V: Key + 'static> {
    name: String,
    adapter: &'txn Arc<BfTreeAdapter>,
    ops_count: &'txn AtomicU64,
    cdc_log: Option<&'txn Mutex<Vec<CdcEvent>>>,
    buffer: &'txn Mutex<WriteBuffer>,
    _key: PhantomData<K>,
    _val: PhantomData<V>,
}

impl<'txn, K: Key + 'static, V: Key + 'static> BfTreeMultimapTable<'txn, K, V> {
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

    /// Insert a (key, value) pair. Returns `true` if the pair already existed.
    pub fn insert(
        &mut self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
    ) -> Result<bool, BfTreeError> {
        let user_key = K::as_bytes(key);
        let mut user_key_ordered = user_key.as_ref().to_vec();
        K::to_byte_ordered_in_place(&mut user_key_ordered);
        let val_key = V::as_bytes(value);
        let mut val_key_ordered = val_key.as_ref().to_vec();
        V::to_byte_ordered_in_place(&mut val_key_ordered);
        let encoded = encode_multimap_key(&self.name, &user_key_ordered, &val_key_ordered)?;

        let mut buffer = self.buffer.lock();
        let already_exists = match buffer.get(&encoded) {
            BufferLookup::Found(_) => true,
            BufferLookup::Tombstone => false,
            BufferLookup::NotInBuffer => self.adapter.contains_key(&encoded),
        };

        // Store empty value -- the "value" is encoded in the composite key.
        buffer.put(encoded, alloc::vec![0u8])?;
        drop(buffer);
        self.ops_count.fetch_add(1, Ordering::Relaxed);

        if self.cdc_log.is_some() {
            // For multimap, the CDC key is the composite (user_key + value_key).
            let mut composite = user_key.as_ref().to_vec();
            composite.extend_from_slice(val_key.as_ref());
            self.record_cdc(CdcEvent {
                table_name: self.name.clone(),
                op: if already_exists {
                    ChangeOp::Update
                } else {
                    ChangeOp::Insert
                },
                key: composite,
                new_value: Some(val_key.as_ref().to_vec()),
                old_value: if already_exists {
                    Some(val_key.as_ref().to_vec())
                } else {
                    None
                },
            });
        }

        Ok(already_exists)
    }

    /// Remove a specific (key, value) pair. Returns `true` if the pair existed.
    pub fn remove(
        &mut self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
    ) -> Result<bool, BfTreeError> {
        let user_key = K::as_bytes(key);
        let mut user_key_ordered = user_key.as_ref().to_vec();
        K::to_byte_ordered_in_place(&mut user_key_ordered);
        let val_key = V::as_bytes(value);
        let mut val_key_ordered = val_key.as_ref().to_vec();
        V::to_byte_ordered_in_place(&mut val_key_ordered);
        let encoded = encode_multimap_key(&self.name, &user_key_ordered, &val_key_ordered)?;

        let mut buffer = self.buffer.lock();
        let existed = match buffer.get(&encoded) {
            BufferLookup::Found(_) => true,
            BufferLookup::Tombstone => false,
            BufferLookup::NotInBuffer => self.adapter.contains_key(&encoded),
        };

        if existed {
            buffer.delete(encoded);
            drop(buffer);
            self.ops_count.fetch_add(1, Ordering::Relaxed);

            if self.cdc_log.is_some() {
                let mut composite = user_key.as_ref().to_vec();
                composite.extend_from_slice(val_key.as_ref());
                self.record_cdc(CdcEvent {
                    table_name: self.name.clone(),
                    op: ChangeOp::Delete,
                    key: composite,
                    new_value: None,
                    old_value: Some(val_key.as_ref().to_vec()),
                });
            }
        } else {
            drop(buffer);
        }

        Ok(existed)
    }

    /// Remove all values for a given key. Returns the number of values removed.
    pub fn remove_all(&mut self, key: &K::SelfType<'_>) -> Result<u64, BfTreeError> {
        let user_key = K::as_bytes(key);
        let mut user_key_ordered = user_key.as_ref().to_vec();
        K::to_byte_ordered_in_place(&mut user_key_ordered);
        let prefix = multimap_key_prefix(&self.name, &user_key_ordered)?;
        let scan_end = multimap_scan_end(&self.name, &user_key_ordered)?;

        let max_record_size = self.adapter.inner().config().get_cb_max_record_size();

        // Collect all BfTree keys in this prefix range.
        // When scan_end is None (all-0xFF overflow), scan from prefix to the
        // end of the keyspace and post-filter with prefix matching.
        let bftree_keys: Vec<Vec<u8>> = {
            let mut buf = vec![0u8; max_record_size * 2];
            let mut keys = Vec::new();
            if let Some(ref end) = scan_end {
                let mut iter = self.adapter.scan_range(&prefix, end)?;
                while let Ok(Some((key_len, _val_len))) = iter.next(&mut buf) {
                    keys.push(buf[..key_len].to_vec());
                }
            } else {
                // Scan from prefix to end of keyspace, filtering by prefix.
                let max_end = {
                    let mut m = prefix.clone();
                    m.push(0xFF);
                    m
                };
                // Use a scan that goes past our prefix; filter results.
                let mut iter = self.adapter.scan_range(&prefix, &max_end)?;
                while let Ok(Some((key_len, _val_len))) = iter.next(&mut buf) {
                    let k = &buf[..key_len];
                    if key_matches_prefix(k, &prefix) {
                        keys.push(k.to_vec());
                    }
                }
            }
            keys
        };

        let mut buffer = self.buffer.lock();
        let mut count = 0u64;

        // Tombstone all BfTree entries.
        for encoded_key in &bftree_keys {
            if !matches!(buffer.get(encoded_key), BufferLookup::Tombstone) {
                count += 1;
            }
            buffer.delete(encoded_key.clone());
        }

        // Also tombstone buffer-only inserts in this range.
        // Use Excluded end bound to avoid including the boundary key itself.
        let buf_only: Vec<Vec<u8>> = match scan_end {
            Some(ref end) => buffer
                .range_excluded_end(&prefix, end)
                .filter_map(|(k, v)| {
                    if v.is_some() && !bftree_keys.iter().any(|bk| bk == k) {
                        Some(k.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            None => buffer
                .prefix_range(&prefix)
                .filter_map(|(k, v)| {
                    if v.is_some() && !bftree_keys.iter().any(|bk| bk == k) {
                        Some(k.clone())
                    } else {
                        None
                    }
                })
                .collect(),
        };
        count += buf_only.len() as u64;
        for k in buf_only {
            buffer.delete(k);
        }

        drop(buffer);
        self.ops_count.fetch_add(count, Ordering::Relaxed);

        if self.cdc_log.is_some() && count > 0 {
            self.record_cdc(CdcEvent {
                table_name: self.name.clone(),
                op: ChangeOp::Delete,
                key: user_key.as_ref().to_vec(),
                new_value: None,
                old_value: None,
            });
        }

        Ok(count)
    }

    /// Get all values for a given key, as a `Vec` of raw value bytes.
    ///
    /// Values are returned in sorted byte order.
    pub fn get_values(&self, key: &K::SelfType<'_>) -> Result<Vec<Vec<u8>>, BfTreeError> {
        let user_key = K::as_bytes(key);
        let mut user_key_ordered = user_key.as_ref().to_vec();
        K::to_byte_ordered_in_place(&mut user_key_ordered);
        let prefix = multimap_key_prefix(&self.name, &user_key_ordered)?;
        let scan_end = multimap_scan_end(&self.name, &user_key_ordered)?;
        let prefix_len = prefix.len();
        let max_record_size = self.adapter.inner().config().get_cb_max_record_size();

        // Collect BfTree entries.
        let bftree_entries: Vec<Vec<u8>> = {
            let mut buf = vec![0u8; max_record_size * 2];
            let mut keys = Vec::new();
            if let Some(ref end) = scan_end {
                let mut iter = self.adapter.scan_range(&prefix, end)?;
                while let Ok(Some((key_len, _val_len))) = iter.next(&mut buf) {
                    keys.push(buf[..key_len].to_vec());
                }
            } else {
                let max_end = {
                    let mut m = prefix.clone();
                    m.push(0xFF);
                    m
                };
                let mut iter = self.adapter.scan_range(&prefix, &max_end)?;
                while let Ok(Some((key_len, _val_len))) = iter.next(&mut buf) {
                    let k = &buf[..key_len];
                    if key_matches_prefix(k, &prefix) {
                        keys.push(k.to_vec());
                    }
                }
            }
            keys
        };

        let buffer = self.buffer.lock();
        let mut values = Vec::new();

        // Process BfTree entries, checking buffer for overrides.
        for encoded_key in &bftree_entries {
            match buffer.get(encoded_key) {
                BufferLookup::Tombstone => { /* hidden by tombstone, skip */ }
                BufferLookup::Found(_) | BufferLookup::NotInBuffer => {
                    let val_key = extract_value_key(encoded_key, prefix_len);
                    let mut val_decoded = val_key.to_vec();
                    V::from_byte_ordered_in_place(&mut val_decoded);
                    values.push(val_decoded);
                }
            }
        }

        // Add buffer-only inserts, using exclusive end bound to prevent
        // including entries at the exact boundary key.
        match scan_end {
            Some(ref end) => {
                for (k, v) in buffer.range_excluded_end(&prefix, end) {
                    if v.is_some() && !bftree_entries.iter().any(|bk| bk == k) {
                        let val_key = extract_value_key(k, prefix_len);
                        let mut val_decoded = val_key.to_vec();
                        V::from_byte_ordered_in_place(&mut val_decoded);
                        values.push(val_decoded);
                    }
                }
            }
            None => {
                for (k, v) in buffer.prefix_range(&prefix) {
                    if v.is_some() && !bftree_entries.iter().any(|bk| bk == k) {
                        let val_key = extract_value_key(k, prefix_len);
                        let mut val_decoded = val_key.to_vec();
                        V::from_byte_ordered_in_place(&mut val_decoded);
                        values.push(val_decoded);
                    }
                }
            }
        }

        // Sort for deterministic output (BfTree entries are sorted, but buffer
        // inserts may interleave).
        values.sort();
        drop(buffer);
        Ok(values)
    }

    /// Check if a specific (key, value) pair exists.
    pub fn contains(
        &self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
    ) -> Result<bool, BfTreeError> {
        let user_key = K::as_bytes(key);
        let mut user_key_ordered = user_key.as_ref().to_vec();
        K::to_byte_ordered_in_place(&mut user_key_ordered);
        let val_key = V::as_bytes(value);
        let mut val_key_ordered = val_key.as_ref().to_vec();
        V::to_byte_ordered_in_place(&mut val_key_ordered);
        let encoded = encode_multimap_key(&self.name, &user_key_ordered, &val_key_ordered)?;

        let buffer = self.buffer.lock();
        match buffer.get(&encoded) {
            BufferLookup::Found(_) => Ok(true),
            BufferLookup::Tombstone => Ok(false),
            BufferLookup::NotInBuffer => {
                drop(buffer);
                Ok(self.adapter.contains_key(&encoded))
            }
        }
    }

    /// Count values for a given key.
    pub fn count_values(&self, key: &K::SelfType<'_>) -> Result<u64, BfTreeError> {
        self.get_values(key).map(|v| v.len() as u64)
    }
}

// ---------------------------------------------------------------------------
// Read-only multimap table
// ---------------------------------------------------------------------------

/// A read-only multimap table backed by `BfTree`.
pub struct BfTreeReadOnlyMultimapTable<'txn, K: Key + 'static, V: Key + 'static> {
    name: String,
    adapter: &'txn Arc<BfTreeAdapter>,
    _key: PhantomData<K>,
    _val: PhantomData<V>,
}

impl<'txn, K: Key + 'static, V: Key + 'static> BfTreeReadOnlyMultimapTable<'txn, K, V> {
    pub(crate) fn new(name: &str, adapter: &'txn Arc<BfTreeAdapter>) -> Self {
        Self {
            name: String::from(name),
            adapter,
            _key: PhantomData,
            _val: PhantomData,
        }
    }

    /// Get all values for a given key, as a `Vec` of raw value bytes.
    pub fn get_values(&self, key: &K::SelfType<'_>) -> Result<Vec<Vec<u8>>, BfTreeError> {
        let user_key = K::as_bytes(key);
        let mut user_key_ordered = user_key.as_ref().to_vec();
        K::to_byte_ordered_in_place(&mut user_key_ordered);
        let prefix = multimap_key_prefix(&self.name, &user_key_ordered)?;
        let scan_end = multimap_scan_end(&self.name, &user_key_ordered)?;
        let prefix_len = prefix.len();
        let max_record_size = self.adapter.inner().config().get_cb_max_record_size();

        let mut buf = vec![0u8; max_record_size * 2];
        let mut values = Vec::new();
        if let Some(end) = scan_end {
            let mut iter = self.adapter.scan_range(&prefix, &end)?;
            while let Ok(Some((key_len, _val_len))) = iter.next(&mut buf) {
                let val_key = extract_value_key(&buf[..key_len], prefix_len);
                let mut val_decoded = val_key.to_vec();
                V::from_byte_ordered_in_place(&mut val_decoded);
                values.push(val_decoded);
            }
        } else {
            let max_end = {
                let mut m = prefix.clone();
                m.push(0xFF);
                m
            };
            let mut iter = self.adapter.scan_range(&prefix, &max_end)?;
            while let Ok(Some((key_len, _val_len))) = iter.next(&mut buf) {
                let k = &buf[..key_len];
                if key_matches_prefix(k, &prefix) {
                    let val_key = extract_value_key(k, prefix_len);
                    let mut val_decoded = val_key.to_vec();
                    V::from_byte_ordered_in_place(&mut val_decoded);
                    values.push(val_decoded);
                }
            }
        }
        Ok(values)
    }

    /// Check if a specific (key, value) pair exists.
    pub fn contains(
        &self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
    ) -> Result<bool, BfTreeError> {
        let user_key = K::as_bytes(key);
        let mut user_key_ordered = user_key.as_ref().to_vec();
        K::to_byte_ordered_in_place(&mut user_key_ordered);
        let val_key = V::as_bytes(value);
        let mut val_key_ordered = val_key.as_ref().to_vec();
        V::to_byte_ordered_in_place(&mut val_key_ordered);
        let encoded = encode_multimap_key(&self.name, &user_key_ordered, &val_key_ordered)?;
        Ok(self.adapter.contains_key(&encoded))
    }

    /// Count values for a given key.
    pub fn count_values(&self, key: &K::SelfType<'_>) -> Result<u64, BfTreeError> {
        self.get_values(key).map(|v| v.len() as u64)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::bf_tree_store::config::BfTreeConfig;
    use crate::bf_tree_store::database::BfTreeDatabase;

    fn make_db() -> BfTreeDatabase {
        BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap()
    }

    #[test]
    fn insert_and_get_values() {
        let db = make_db();
        let wtxn = db.begin_write();
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags").unwrap();

        assert!(!mm.insert(&"doc1", &"rust").unwrap());
        assert!(!mm.insert(&"doc1", &"systems").unwrap());
        assert!(!mm.insert(&"doc1", &"low-level").unwrap());

        let vals = mm.get_values(&"doc1").unwrap();
        assert_eq!(vals.len(), 3);
        // Values should be sorted.
        assert_eq!(vals[0], b"low-level");
        assert_eq!(vals[1], b"rust");
        assert_eq!(vals[2], b"systems");
    }

    #[test]
    fn insert_duplicate_returns_true() {
        let db = make_db();
        let wtxn = db.begin_write();
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags").unwrap();

        assert!(!mm.insert(&"k", &"v").unwrap());
        assert!(mm.insert(&"k", &"v").unwrap()); // duplicate
    }

    #[test]
    fn remove_specific_value() {
        let db = make_db();
        let wtxn = db.begin_write();
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags").unwrap();

        mm.insert(&"k", &"a").unwrap();
        mm.insert(&"k", &"b").unwrap();
        mm.insert(&"k", &"c").unwrap();

        assert!(mm.remove(&"k", &"b").unwrap());
        assert!(!mm.remove(&"k", &"nonexistent").unwrap());

        let vals = mm.get_values(&"k").unwrap();
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0], b"a");
        assert_eq!(vals[1], b"c");
    }

    #[test]
    fn remove_all_values() {
        let db = make_db();
        let wtxn = db.begin_write();
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags").unwrap();

        mm.insert(&"k", &"x").unwrap();
        mm.insert(&"k", &"y").unwrap();
        mm.insert(&"k", &"z").unwrap();

        let removed = mm.remove_all(&"k").unwrap();
        assert_eq!(removed, 3);

        assert!(mm.get_values(&"k").unwrap().is_empty());
    }

    #[test]
    fn key_isolation() {
        let db = make_db();
        let wtxn = db.begin_write();
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags").unwrap();

        mm.insert(&"alice", &"admin").unwrap();
        mm.insert(&"bob", &"user").unwrap();
        mm.insert(&"bob", &"editor").unwrap();

        assert_eq!(mm.get_values(&"alice").unwrap().len(), 1);
        assert_eq!(mm.get_values(&"bob").unwrap().len(), 2);
        assert!(mm.get_values(&"carol").unwrap().is_empty());
    }

    #[test]
    fn contains_check() {
        let db = make_db();
        let wtxn = db.begin_write();
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags").unwrap();

        mm.insert(&"k", &"val").unwrap();
        assert!(mm.contains(&"k", &"val").unwrap());
        assert!(!mm.contains(&"k", &"other").unwrap());
    }

    #[test]
    fn count_values_matches() {
        let db = make_db();
        let wtxn = db.begin_write();
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags").unwrap();

        mm.insert(&"k", &"a").unwrap();
        mm.insert(&"k", &"b").unwrap();
        assert_eq!(mm.count_values(&"k").unwrap(), 2);
    }

    #[test]
    fn survives_commit() {
        let db = make_db();

        {
            let wtxn = db.begin_write();
            let mut mm = wtxn.open_multimap_table::<&str, &str>("tags").unwrap();
            mm.insert(&"k", &"val1").unwrap();
            mm.insert(&"k", &"val2").unwrap();
            drop(mm);
            wtxn.commit().unwrap();
        }

        let rtxn = db.begin_read();
        let ro = rtxn.open_multimap_table::<&str, &str>("tags").unwrap();
        let vals = ro.get_values(&"k").unwrap();
        assert_eq!(vals.len(), 2);
        assert!(ro.contains(&"k", &"val1").unwrap());
        assert!(ro.contains(&"k", &"val2").unwrap());
    }

    #[test]
    fn rollback_discards_changes() {
        let db = make_db();

        {
            let wtxn = db.begin_write();
            let mut mm = wtxn.open_multimap_table::<&str, &str>("tags").unwrap();
            mm.insert(&"k", &"gone").unwrap();
            drop(mm);
            // Drop without commit -- rollback.
        }

        let rtxn = db.begin_read();
        let ro = rtxn.open_multimap_table::<&str, &str>("tags").unwrap();
        assert!(ro.get_values(&"k").unwrap().is_empty());
    }
}
