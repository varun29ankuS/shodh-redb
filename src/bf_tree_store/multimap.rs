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

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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
fn encode_multimap_key(table_name: &str, user_key: &[u8], value_key: &[u8]) -> Vec<u8> {
    let tbl = table_name.as_bytes();
    let tbl_len = u16::try_from(tbl.len()).expect("table name exceeds u16::MAX bytes");
    let uk_len = u32::try_from(user_key.len()).expect("user key exceeds u32::MAX bytes");

    let total = 2 + tbl.len() + 1 + 4 + user_key.len() + value_key.len();
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&tbl_len.to_le_bytes());
    buf.extend_from_slice(tbl);
    buf.push(MULTIMAP_KIND);
    buf.extend_from_slice(&uk_len.to_le_bytes());
    buf.extend_from_slice(user_key);
    buf.extend_from_slice(value_key);
    buf
}

/// Compute the `BfTree` key prefix for all values of a user key.
///
/// Format: `[table_name_len: u16 LE][table_name][0x01 kind][user_key_len: u32 LE][user_key]`
fn multimap_key_prefix(table_name: &str, user_key: &[u8]) -> Vec<u8> {
    let tbl = table_name.as_bytes();
    let tbl_len = u16::try_from(tbl.len()).expect("table name exceeds u16::MAX bytes");
    let uk_len = u32::try_from(user_key.len()).expect("user key exceeds u32::MAX bytes");

    let total = 2 + tbl.len() + 1 + 4 + user_key.len();
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&tbl_len.to_le_bytes());
    buf.extend_from_slice(tbl);
    buf.push(MULTIMAP_KIND);
    buf.extend_from_slice(&uk_len.to_le_bytes());
    buf.extend_from_slice(user_key);
    buf
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

/// Compute the exclusive upper bound for scanning all values of a given user
/// key in a multimap table.
///
/// The prefix format is `[tbl_len: u16 LE][tbl][0x01 kind][uk_len: u32 LE][user_key]`.
/// When `increment_prefix` overflows (all-0xFF prefix), we fall back to the
/// table-level prefix end. This is safe because all multimap entries for a
/// given table share the same table prefix with the multimap discriminator,
/// so using the table+kind boundary as the upper bound will never miss
/// entries and never include entries from a different table type.
fn multimap_scan_end(table_name: &str, user_key: &[u8]) -> Vec<u8> {
    let prefix = multimap_key_prefix(table_name, user_key);
    if let Some(end) = increment_prefix(&prefix) {
        return end;
    }
    // All bytes in the prefix are 0xFF. Use the table-level prefix end
    // (including the multimap kind discriminator) as the upper bound. This
    // is always correct because every multimap entry for this table starts
    // with the same [tbl_len][tbl][0x01] prefix.
    let tbl = table_name.as_bytes();
    let tbl_len = u16::try_from(tbl.len()).expect("table name exceeds u16::MAX bytes");
    let mut tbl_prefix = Vec::with_capacity(2 + tbl.len() + 1);
    tbl_prefix.extend_from_slice(&tbl_len.to_le_bytes());
    tbl_prefix.extend_from_slice(tbl);
    tbl_prefix.push(MULTIMAP_KIND);
    // Increment the table+kind prefix as a big-endian integer. If even that
    // overflows (table name is all 0xFF, length bytes are 0xFF, and kind is
    // 0xFF), append 0xFF to form a lexicographically larger bound.
    if let Some(end) = increment_prefix(&tbl_prefix) {
        end
    } else {
        tbl_prefix.push(0xFF);
        tbl_prefix
    }
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
    buffer: &'txn Mutex<WriteBuffer>,
    _key: PhantomData<K>,
    _val: PhantomData<V>,
}

impl<'txn, K: Key + 'static, V: Key + 'static> BfTreeMultimapTable<'txn, K, V> {
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

    /// Insert a (key, value) pair. Returns `true` if the pair already existed.
    pub fn insert(
        &mut self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
    ) -> Result<bool, BfTreeError> {
        let user_key = K::as_bytes(key);
        let val_key = V::as_bytes(value);
        let encoded = encode_multimap_key(&self.name, user_key.as_ref(), val_key.as_ref());

        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        let already_exists = match buffer.get(&encoded) {
            BufferLookup::Found(_) => true,
            BufferLookup::Tombstone => false,
            BufferLookup::NotInBuffer => self.adapter.contains_key(&encoded),
        };

        // Store empty value -- the "value" is encoded in the composite key.
        buffer.put(encoded, alloc::vec![0u8])?;
        drop(buffer);
        self.ops_count.fetch_add(1, Ordering::Relaxed);

        Ok(already_exists)
    }

    /// Remove a specific (key, value) pair. Returns `true` if the pair existed.
    pub fn remove(
        &mut self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
    ) -> Result<bool, BfTreeError> {
        let user_key = K::as_bytes(key);
        let val_key = V::as_bytes(value);
        let encoded = encode_multimap_key(&self.name, user_key.as_ref(), val_key.as_ref());

        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        let existed = match buffer.get(&encoded) {
            BufferLookup::Found(_) => true,
            BufferLookup::Tombstone => false,
            BufferLookup::NotInBuffer => self.adapter.contains_key(&encoded),
        };

        if existed {
            buffer.delete(encoded);
            drop(buffer);
            self.ops_count.fetch_add(1, Ordering::Relaxed);
        } else {
            drop(buffer);
        }

        Ok(existed)
    }

    /// Remove all values for a given key. Returns the number of values removed.
    pub fn remove_all(&mut self, key: &K::SelfType<'_>) -> Result<u64, BfTreeError> {
        let user_key = K::as_bytes(key);
        let prefix = multimap_key_prefix(&self.name, user_key.as_ref());
        let prefix_end = multimap_scan_end(&self.name, user_key.as_ref());

        let max_record_size = self.adapter.inner().config().get_cb_max_record_size();

        // Collect all BfTree keys in this prefix range.
        let bftree_keys: Vec<Vec<u8>> = {
            let mut buf = vec![0u8; max_record_size * 2];
            let mut keys = Vec::new();
            let mut iter = self.adapter.scan_range(&prefix, &prefix_end)?;
            while let Some((key_len, _val_len)) = iter.next(&mut buf) {
                keys.push(buf[..key_len].to_vec());
            }
            keys
        };

        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        let mut count = 0u64;

        // Tombstone all BfTree entries.
        for encoded_key in &bftree_keys {
            match buffer.get(encoded_key) {
                BufferLookup::Tombstone => {} // already hidden
                _ => count += 1,
            }
            buffer.delete(encoded_key.clone());
        }

        // Also tombstone buffer-only inserts in this range.
        let buf_only: Vec<Vec<u8>> = buffer
            .range(&prefix, &prefix_end)
            .filter_map(|(k, v)| {
                if v.is_some() && !bftree_keys.iter().any(|bk| bk == k) {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();
        count += buf_only.len() as u64;
        for k in buf_only {
            buffer.delete(k);
        }

        drop(buffer);
        self.ops_count.fetch_add(count, Ordering::Relaxed);
        Ok(count)
    }

    /// Get all values for a given key, as a `Vec` of raw value bytes.
    ///
    /// Values are returned in sorted byte order.
    pub fn get_values(&self, key: &K::SelfType<'_>) -> Result<Vec<Vec<u8>>, BfTreeError> {
        let user_key = K::as_bytes(key);
        let prefix = multimap_key_prefix(&self.name, user_key.as_ref());
        let prefix_end = multimap_scan_end(&self.name, user_key.as_ref());
        let prefix_len = prefix.len();
        let max_record_size = self.adapter.inner().config().get_cb_max_record_size();

        // Collect BfTree entries.
        let bftree_entries: Vec<Vec<u8>> = {
            let mut buf = vec![0u8; max_record_size * 2];
            let mut keys = Vec::new();
            let mut iter = self.adapter.scan_range(&prefix, &prefix_end)?;
            while let Some((key_len, _val_len)) = iter.next(&mut buf) {
                keys.push(buf[..key_len].to_vec());
            }
            keys
        };

        let buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        let mut values = Vec::new();

        // Process BfTree entries, checking buffer for overrides.
        for encoded_key in &bftree_entries {
            match buffer.get(encoded_key) {
                BufferLookup::Tombstone => { /* hidden by tombstone, skip */ }
                BufferLookup::Found(_) | BufferLookup::NotInBuffer => {
                    let val_key = extract_value_key(encoded_key, prefix_len);
                    values.push(val_key.to_vec());
                }
            }
        }

        // Add buffer-only inserts.
        for (k, v) in buffer.range(&prefix, &prefix_end) {
            if v.is_some() && !bftree_entries.iter().any(|bk| bk == k) {
                let val_key = extract_value_key(k, prefix_len);
                values.push(val_key.to_vec());
            }
        }

        // Sort for deterministic output (BfTree entries are sorted, but buffer
        // inserts may interleave).
        values.sort();
        drop(buffer);
        Ok(values)
    }

    /// Check if a specific (key, value) pair exists.
    pub fn contains(&self, key: &K::SelfType<'_>, value: &V::SelfType<'_>) -> bool {
        let user_key = K::as_bytes(key);
        let val_key = V::as_bytes(value);
        let encoded = encode_multimap_key(&self.name, user_key.as_ref(), val_key.as_ref());

        let buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        match buffer.get(&encoded) {
            BufferLookup::Found(_) => true,
            BufferLookup::Tombstone => false,
            BufferLookup::NotInBuffer => {
                drop(buffer);
                self.adapter.contains_key(&encoded)
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
        let prefix = multimap_key_prefix(&self.name, user_key.as_ref());
        let prefix_end = multimap_scan_end(&self.name, user_key.as_ref());
        let prefix_len = prefix.len();
        let max_record_size = self.adapter.inner().config().get_cb_max_record_size();

        let mut buf = vec![0u8; max_record_size * 2];
        let mut values = Vec::new();
        let mut iter = self.adapter.scan_range(&prefix, &prefix_end)?;
        while let Some((key_len, _val_len)) = iter.next(&mut buf) {
            let val_key = extract_value_key(&buf[..key_len], prefix_len);
            values.push(val_key.to_vec());
        }
        Ok(values)
    }

    /// Check if a specific (key, value) pair exists.
    pub fn contains(&self, key: &K::SelfType<'_>, value: &V::SelfType<'_>) -> bool {
        let user_key = K::as_bytes(key);
        let val_key = V::as_bytes(value);
        let encoded = encode_multimap_key(&self.name, user_key.as_ref(), val_key.as_ref());
        self.adapter.contains_key(&encoded)
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
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags");

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
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags");

        assert!(!mm.insert(&"k", &"v").unwrap());
        assert!(mm.insert(&"k", &"v").unwrap()); // duplicate
    }

    #[test]
    fn remove_specific_value() {
        let db = make_db();
        let wtxn = db.begin_write();
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags");

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
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags");

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
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags");

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
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags");

        mm.insert(&"k", &"val").unwrap();
        assert!(mm.contains(&"k", &"val"));
        assert!(!mm.contains(&"k", &"other"));
    }

    #[test]
    fn count_values_matches() {
        let db = make_db();
        let wtxn = db.begin_write();
        let mut mm = wtxn.open_multimap_table::<&str, &str>("tags");

        mm.insert(&"k", &"a").unwrap();
        mm.insert(&"k", &"b").unwrap();
        assert_eq!(mm.count_values(&"k").unwrap(), 2);
    }

    #[test]
    fn survives_commit() {
        let db = make_db();

        {
            let wtxn = db.begin_write();
            let mut mm = wtxn.open_multimap_table::<&str, &str>("tags");
            mm.insert(&"k", &"val1").unwrap();
            mm.insert(&"k", &"val2").unwrap();
            drop(mm);
            wtxn.commit().unwrap();
        }

        let rtxn = db.begin_read();
        let ro = rtxn.open_multimap_table::<&str, &str>("tags");
        let vals = ro.get_values(&"k").unwrap();
        assert_eq!(vals.len(), 2);
        assert!(ro.contains(&"k", &"val1"));
        assert!(ro.contains(&"k", &"val2"));
    }

    #[test]
    fn rollback_discards_changes() {
        let db = make_db();

        {
            let wtxn = db.begin_write();
            let mut mm = wtxn.open_multimap_table::<&str, &str>("tags");
            mm.insert(&"k", &"gone").unwrap();
            drop(mm);
            // Drop without commit -- rollback.
        }

        let rtxn = db.begin_read();
        let ro = rtxn.open_multimap_table::<&str, &str>("tags");
        assert!(ro.get_values(&"k").unwrap().is_empty());
    }
}
