//! History / time-travel snapshots for `BfTree`.
//!
//! Provides point-in-time snapshot management for the `BfTree` backend.
//! Each "history entry" records a snapshot taken after a committed transaction,
//! enabling time-travel reads by opening a historical snapshot as a read-only
//! database.
//!
//! # Storage
//!
//! History metadata is stored in the `__bf_history_meta` system table:
//! - Key: `snapshot_id` (u64 LE)
//! - Value: `HistoryEntry` serialized bytes

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::Ordering;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::config::BfTreeConfig;
use super::database::{
    encode_table_key, table_prefix, table_prefix_end, BfTreeDatabase,
};
use super::error::BfTreeError;

/// System table for history metadata.
const HISTORY_META_TABLE: &str = "__bf_history_meta";

/// Maximum snapshot path length in bytes.
const MAX_PATH_LEN: usize = 1024;

/// Size of a serialized `HistoryEntry` (fixed-width).
/// Layout: `txn_id`(8) + `timestamp_ns`(8) + `path_len`(2) + path(1024) = 1042 bytes.
const HISTORY_ENTRY_SIZE: usize = 8 + 8 + 2 + MAX_PATH_LEN;

/// A recorded history snapshot entry.
#[derive(Clone, Debug)]
pub struct HistoryEntry {
    /// Transaction ID at the time of snapshot.
    pub txn_id: u64,
    /// Wall-clock timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Filesystem path to the snapshot file.
    pub snapshot_path: String,
}

impl HistoryEntry {
    /// Serialize to fixed-width little-endian bytes.
    ///
    /// Returns an error if the snapshot path exceeds [`MAX_PATH_LEN`] bytes.
    pub fn to_le_bytes(&self) -> Result<[u8; HISTORY_ENTRY_SIZE], BfTreeError> {
        let path_bytes = self.snapshot_path.as_bytes();
        if path_bytes.len() > MAX_PATH_LEN {
            return Err(BfTreeError::InvalidOperation(alloc::format!(
                "snapshot path length {} exceeds maximum of {} bytes",
                path_bytes.len(),
                MAX_PATH_LEN,
            )));
        }
        let mut buf = [0u8; HISTORY_ENTRY_SIZE];
        buf[..8].copy_from_slice(&self.txn_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        #[allow(clippy::cast_possible_truncation)]
        let path_len_u16 = path_bytes.len() as u16;
        buf[16..18].copy_from_slice(&path_len_u16.to_le_bytes());
        buf[18..18 + path_bytes.len()].copy_from_slice(path_bytes);
        Ok(buf)
    }

    pub fn from_le_bytes(data: &[u8]) -> Self {
        let txn_id = u64::from_le_bytes(data[..8].try_into().unwrap());
        let timestamp_ns = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let path_len = u16::from_le_bytes(data[16..18].try_into().unwrap()) as usize;
        let path_len = path_len.min(MAX_PATH_LEN).min(data.len() - 18);
        let snapshot_path =
            core::str::from_utf8(&data[18..18 + path_len]).unwrap_or("").to_string();
        Self {
            txn_id,
            timestamp_ns,
            snapshot_path,
        }
    }
}

fn now_ns() -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO)
        .as_nanos() as u64;
    ns
}

use alloc::string::ToString;

/// History manager for `BfTree` — manages point-in-time snapshots.
pub struct BfTreeHistory {
    db: Arc<BfTreeDatabase>,
    next_snapshot_id: core::sync::atomic::AtomicU64,
}

impl BfTreeHistory {
    /// Create a new history manager for the given database.
    pub fn new(db: Arc<BfTreeDatabase>) -> Self {
        // Recover next snapshot ID from existing entries.
        let next_id = {
            let rtxn = db.begin_read();
            let prefix = table_prefix(HISTORY_META_TABLE);
            let prefix_end = table_prefix_end(HISTORY_META_TABLE);
            let prefix_len = prefix.len();
            let max_record = rtxn.adapter.inner().config().get_cb_max_record_size();
            let mut buf = vec![0u8; max_record * 2];
            let mut max_id: u64 = 0;
            if let Ok(mut iter) = rtxn.adapter.scan_range(&prefix, &prefix_end) {
                while let Some((key_len, _)) = iter.next(&mut buf) {
                    if key_len > prefix_len + 8 {
                        continue;
                    }
                    let key_bytes = &buf[prefix_len..key_len];
                    if key_bytes.len() >= 8 {
                        let id = u64::from_le_bytes(key_bytes[..8].try_into().unwrap());
                        if id > max_id {
                            max_id = id;
                        }
                    }
                }
            }
            max_id.saturating_add(1)
        };

        Self {
            db,
            next_snapshot_id: core::sync::atomic::AtomicU64::new(next_id),
        }
    }

    /// Commit the current state as a named history point.
    /// Takes a snapshot and records the entry in the history meta table.
    /// Returns the snapshot ID.
    pub fn commit_snapshot(&self) -> Result<(u64, PathBuf), BfTreeError> {
        let snapshot_id = self.next_snapshot_id.fetch_add(1, Ordering::SeqCst);
        let txn_id = {
            let rtxn = self.db.begin_read();
            rtxn.latest_cdc_transaction_id().unwrap_or(None).unwrap_or(0)
        };

        // Take the BfTree snapshot.
        let snapshot_path = self.db.snapshot();

        // Record the history entry.
        let entry = HistoryEntry {
            txn_id,
            timestamp_ns: now_ns(),
            snapshot_path: snapshot_path.to_string_lossy().to_string(),
        };

        let key = encode_table_key(HISTORY_META_TABLE, &snapshot_id.to_le_bytes());
        self.db.adapter().insert(&key, &entry.to_le_bytes()?)?;

        Ok((snapshot_id, snapshot_path))
    }

    /// List all history entries, newest first.
    pub fn list(&self) -> Result<Vec<(u64, HistoryEntry)>, BfTreeError> {
        let rtxn = self.db.begin_read();
        let prefix = table_prefix(HISTORY_META_TABLE);
        let prefix_end = table_prefix_end(HISTORY_META_TABLE);
        let prefix_len = prefix.len();
        let max_record = rtxn.adapter.inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_record * 2];
        let mut iter = rtxn.adapter.scan_range(&prefix, &prefix_end)?;
        let mut results = Vec::new();

        while let Some((key_len, val_len)) = iter.next(&mut buf) {
            if key_len <= prefix_len {
                continue;
            }
            let key_bytes = &buf[prefix_len..key_len];
            let val_bytes = &buf[key_len..key_len + val_len];

            if key_bytes.len() >= 8 && val_bytes.len() >= HISTORY_ENTRY_SIZE {
                let id = u64::from_le_bytes(key_bytes[..8].try_into().unwrap());
                let entry = HistoryEntry::from_le_bytes(val_bytes);
                results.push((id, entry));
            }
        }

        results.sort_by(|a, b| b.0.cmp(&a.0)); // newest first
        Ok(results)
    }

    /// Get a specific history entry by snapshot ID.
    pub fn get(&self, snapshot_id: u64) -> Result<Option<HistoryEntry>, BfTreeError> {
        let key = encode_table_key(HISTORY_META_TABLE, &snapshot_id.to_le_bytes());
        let max_val = self.db.adapter().inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.db.adapter().read(&key, &mut buf) {
            Ok(len) if (len as usize) >= HISTORY_ENTRY_SIZE => {
                Ok(Some(HistoryEntry::from_le_bytes(&buf[..len as usize])))
            }
            Ok(_) | Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Open a historical snapshot as a read-only database.
    /// The returned database is independent and can be read concurrently.
    pub fn open_historical(&self, snapshot_id: u64) -> Result<BfTreeDatabase, BfTreeError> {
        let entry = self.get(snapshot_id)?.ok_or(BfTreeError::NotFound)?;

        let path = PathBuf::from(&entry.snapshot_path);
        if !path.exists() {
            return Err(BfTreeError::Corruption(alloc::format!(
                "snapshot file not found: {}",
                entry.snapshot_path
            )));
        }

        BfTreeDatabase::open(BfTreeConfig::new_file(path, 4))
    }

    /// Prune old history entries, keeping only the most recent `keep` entries.
    /// Deletes the associated snapshot files from disk.
    pub fn prune(&self, keep: usize) -> Result<usize, BfTreeError> {
        let entries = self.list()?;
        if entries.len() <= keep {
            return Ok(0);
        }

        let to_remove = &entries[keep..];
        let mut removed = 0;

        for (id, entry) in to_remove {
            let path = Path::new(&entry.snapshot_path);
            if path.exists() && std::fs::remove_file(path).is_err() {
                // If we cannot remove the snapshot file, skip metadata deletion
                // so the entry remains and can be retried on a subsequent prune.
                continue;
            }
            // Delete the history entry from BfTree only after file removal succeeds
            // (or the file was already absent).
            let key = encode_table_key(HISTORY_META_TABLE, &id.to_le_bytes());
            self.db.adapter().delete(&key);
            removed += 1;
        }

        Ok(removed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf_tree_store::config::BfTreeConfig;
    use crate::TableDefinition;

    const DATA: TableDefinition<&str, u64> = TableDefinition::new("hist_data");

    #[test]
    fn history_entry_serialization() {
        let entry = HistoryEntry {
            txn_id: 42,
            timestamp_ns: 1234567890,
            snapshot_path: "/tmp/snap.bin".into(),
        };
        let bytes = entry.to_le_bytes().unwrap();
        let restored = HistoryEntry::from_le_bytes(&bytes);
        assert_eq!(restored.txn_id, 42);
        assert_eq!(restored.timestamp_ns, 1234567890);
        assert_eq!(restored.snapshot_path, "/tmp/snap.bin");
    }

    #[test]
    fn commit_and_list_snapshots() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("hist.bftree");
        let db = Arc::new(
            BfTreeDatabase::create(BfTreeConfig::new_file(&db_path, 4)).unwrap(),
        );

        let wtxn = db.begin_write();
        let mut t = wtxn.open_table(DATA);
        t.insert(&"key", &1u64).unwrap();
        drop(t);
        wtxn.commit().unwrap();

        let history = BfTreeHistory::new(db.clone());
        let (id1, _path1) = history.commit_snapshot().unwrap();
        let (id2, _path2) = history.commit_snapshot().unwrap();
        assert!(id2 > id1);

        let entries = history.list().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, id2);
        assert_eq!(entries[1].0, id1);
    }

    #[test]
    fn get_specific_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("hist2.bftree");
        let db = Arc::new(
            BfTreeDatabase::create(BfTreeConfig::new_file(&db_path, 4)).unwrap(),
        );
        let history = BfTreeHistory::new(db);
        let (id, _) = history.commit_snapshot().unwrap();

        let entry = history.get(id).unwrap().unwrap();
        assert_eq!(entry.txn_id, 0);
        assert!(!entry.snapshot_path.is_empty());

        assert!(history.get(999).unwrap().is_none());
    }

    #[test]
    fn prune_old_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("hist3.bftree");
        let db = Arc::new(
            BfTreeDatabase::create(BfTreeConfig::new_file(&db_path, 4)).unwrap(),
        );
        let history = BfTreeHistory::new(db);

        for _ in 0..5 {
            history.commit_snapshot().unwrap();
        }

        let removed = history.prune(2).unwrap();
        assert_eq!(removed, 3);

        let remaining = history.list().unwrap();
        assert_eq!(remaining.len(), 2);
    }
}
