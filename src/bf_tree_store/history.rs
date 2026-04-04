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
    BfTreeDatabase, TableKind, encode_table_key, table_prefix, table_prefix_end,
};
use super::error::BfTreeError;

/// System table for history metadata.
const HISTORY_META_TABLE: &str = "__bf_history_meta";

/// Dedicated key for persisting the high-water-mark snapshot ID.
/// Stored in the history meta table to prevent ID reuse after full prune.
const HISTORY_HWM_KEY: &[u8] = b"__hwm_snapshot_id";

/// Maximum snapshot path length in bytes.
const MAX_PATH_LEN: usize = 1024;

/// Size of a serialized `HistoryEntry` (fixed-width).
/// Layout: `txn_id`(8) + `timestamp_ns`(8) + `path_len`(2) + path(1024) + `status`(1) = 1043 bytes.
const HISTORY_ENTRY_SIZE: usize = 8 + 8 + 2 + MAX_PATH_LEN + 1;

/// Status byte for a history entry: snapshot creation is in progress.
const HISTORY_STATUS_PENDING: u8 = 0;
/// Status byte for a history entry: snapshot is complete and valid.
const HISTORY_STATUS_COMPLETE: u8 = 1;

/// A recorded history snapshot entry.
#[derive(Clone, Debug)]
pub struct HistoryEntry {
    /// Transaction ID at the time of snapshot.
    pub txn_id: u64,
    /// Wall-clock timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Filesystem path to the snapshot file.
    pub snapshot_path: String,
    /// Status: 0 = pending (snapshot creation in progress), 1 = complete.
    pub status: u8,
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
        buf[18 + MAX_PATH_LEN] = self.status;
        Ok(buf)
    }

    /// Deserialize from bytes. Returns `None` if the input is too short
    /// (less than the minimum required 18 bytes for the fixed header).
    pub fn from_le_bytes(data: &[u8]) -> Option<Self> {
        // Minimum: 8 (txn_id) + 8 (timestamp_ns) + 2 (path_len) = 18 bytes.
        if data.len() < 18 {
            return None;
        }
        let txn_id = u64::from_le_bytes(data[..8].try_into().unwrap());
        let timestamp_ns = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let path_len = u16::from_le_bytes(data[16..18].try_into().unwrap()) as usize;
        let available = data.len().saturating_sub(18);
        let path_len = path_len.min(MAX_PATH_LEN).min(available);
        let snapshot_path = core::str::from_utf8(&data[18..18 + path_len])
            .unwrap_or("")
            .to_string();
        // Status byte follows the path region at offset 18 + MAX_PATH_LEN.
        let status = if data.len() >= HISTORY_ENTRY_SIZE {
            data[18 + MAX_PATH_LEN]
        } else {
            // Legacy entries without a status byte are treated as complete.
            HISTORY_STATUS_COMPLETE
        };
        Some(Self {
            txn_id,
            timestamp_ns,
            snapshot_path,
            status,
        })
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

/// Validate that a snapshot path is safe to use for file operations.
///
/// Rejects paths containing path traversal components (`..`) to prevent a
/// crafted PENDING entry from causing `remove_file` to delete arbitrary
/// files on the filesystem.
fn validate_snapshot_path(snapshot_path: &str) -> Result<(), BfTreeError> {
    if snapshot_path.is_empty() {
        // Empty paths are allowed (e.g., legacy PENDING entries).
        // Recovery will skip file deletion for empty paths.
        return Ok(());
    }

    let path = Path::new(snapshot_path);

    // Reject any path component that is ".." to prevent directory traversal.
    for component in path.components() {
        if let std::path::Component::ParentDir = component {
            return Err(BfTreeError::InvalidOperation(alloc::format!(
                "snapshot path contains illegal '..' component: {snapshot_path}"
            )));
        }
    }

    Ok(())
}

/// History manager for `BfTree` -- manages point-in-time snapshots.
pub struct BfTreeHistory {
    db: Arc<BfTreeDatabase>,
    next_snapshot_id: core::sync::atomic::AtomicU64,
}

impl BfTreeHistory {
    /// Create a new history manager for the given database.
    ///
    /// On startup, recovers from incomplete snapshots by deleting any entries
    /// that are still in "pending" status (crash between metadata insert and
    /// snapshot completion).
    pub fn new(db: Arc<BfTreeDatabase>) -> Self {
        // Recover next snapshot ID and clean up pending entries.
        // The high-water-mark ID is checked first (prevents ID reuse after
        // full prune), then scan entries for the max ID as a fallback.
        let next_id = {
            let rtxn = db.begin_read();
            let prefix = table_prefix(HISTORY_META_TABLE, TableKind::Regular);
            let prefix_end = table_prefix_end(HISTORY_META_TABLE, TableKind::Regular);
            let prefix_len = prefix.len();
            let max_record = rtxn.adapter.inner().config().get_cb_max_record_size();
            let mut buf = vec![0u8; max_record * 2];
            let mut max_id: u64 = 0;
            let mut pending_keys: Vec<(Vec<u8>, String)> = Vec::new();

            // Read the persisted high-water-mark ID.
            let hwm_key =
                encode_table_key(HISTORY_META_TABLE, TableKind::Regular, HISTORY_HWM_KEY);
            let mut hwm_buf = [0u8; 8];
            if let Ok(len) = db.adapter().read(&hwm_key, &mut hwm_buf)
                && len as usize >= 8
            {
                max_id = u64::from_le_bytes(hwm_buf);
            }

            if let Ok(mut iter) = rtxn.adapter.scan_range(&prefix, &prefix_end) {
                while let Some((key_len, val_len)) = iter.next(&mut buf) {
                    if key_len > prefix_len + 8 {
                        continue;
                    }
                    let key_bytes = &buf[prefix_len..key_len];
                    if key_bytes.len() >= 8 {
                        let id = u64::from_le_bytes(key_bytes[..8].try_into().unwrap());
                        if id > max_id {
                            max_id = id;
                        }
                        // Check for pending entries that need cleanup.
                        let val_bytes = &buf[key_len..key_len + val_len];
                        if let Some(entry) = HistoryEntry::from_le_bytes(val_bytes)
                            .filter(|e| e.status == HISTORY_STATUS_PENDING)
                        {
                            let full_key = buf[..key_len].to_vec();
                            pending_keys.push((full_key, entry.snapshot_path));
                        }
                    }
                }
            }
            drop(rtxn);

            // Delete pending entries and their snapshot files.
            for (full_key, snap_path) in pending_keys {
                // Validate the path before attempting file deletion to prevent
                // path traversal attacks via crafted PENDING entries.
                if !snap_path.is_empty()
                    && validate_snapshot_path(&snap_path).is_ok()
                {
                    let path = Path::new(&snap_path);
                    if path.exists() {
                        let _ = std::fs::remove_file(path);
                    }
                }
                db.adapter().delete(&full_key);
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
    ///
    /// Uses a two-phase approach to prevent orphaned snapshot files on crash:
    ///
    /// 1. Take the `BfTree` snapshot to determine the file path.
    /// 2. Write a "pending" metadata entry with the real path.
    /// 3. Update the metadata entry to "complete".
    ///
    /// If we crash between phases 1 and 2, the snapshot file is orphaned but
    /// harmless (no metadata references it). If we crash between phases 2 and
    /// 3, recovery finds the PENDING entry with the real path and deletes the
    /// file.
    pub fn commit_snapshot(&self) -> Result<(u64, PathBuf), BfTreeError> {
        let snapshot_id = self.next_snapshot_id.fetch_add(1, Ordering::SeqCst);
        let txn_id = {
            let rtxn = self.db.begin_read();
            rtxn.latest_cdc_transaction_id()
                .unwrap_or(None)
                .unwrap_or(0)
        };

        let key = encode_table_key(
            HISTORY_META_TABLE,
            TableKind::Regular,
            &snapshot_id.to_le_bytes(),
        );

        // Persist the high-water-mark ID so that after a full prune + restart
        // the counter does not reset to 1 and cause ID collisions.
        let hwm_key =
            encode_table_key(HISTORY_META_TABLE, TableKind::Regular, HISTORY_HWM_KEY);
        self.db
            .adapter()
            .insert(&hwm_key, &snapshot_id.to_le_bytes())?;

        // Phase 1: Take the BfTree snapshot to determine the file path.
        let snapshot_path = self.db.snapshot();
        let snapshot_path_str = snapshot_path.to_string_lossy().to_string();
        let timestamp = now_ns();

        // Phase 2: Write a pending metadata entry with the real path.
        // If we crash after this but before phase 3, recovery will find the
        // pending entry and delete the snapshot file using the stored path.
        let pending_entry = HistoryEntry {
            txn_id,
            timestamp_ns: timestamp,
            snapshot_path: snapshot_path_str.clone(),
            status: HISTORY_STATUS_PENDING,
        };
        self.db
            .adapter()
            .insert(&key, &pending_entry.to_le_bytes()?)?;

        // Phase 3: Update the metadata entry to "complete".
        let complete_entry = HistoryEntry {
            txn_id,
            timestamp_ns: timestamp,
            snapshot_path: snapshot_path_str,
            status: HISTORY_STATUS_COMPLETE,
        };
        self.db
            .adapter()
            .insert(&key, &complete_entry.to_le_bytes()?)?;

        Ok((snapshot_id, snapshot_path))
    }

    /// List all completed history entries, newest first.
    /// Pending entries (incomplete snapshots) are excluded.
    pub fn list(&self) -> Result<Vec<(u64, HistoryEntry)>, BfTreeError> {
        let rtxn = self.db.begin_read();
        let prefix = table_prefix(HISTORY_META_TABLE, TableKind::Regular);
        let prefix_end = table_prefix_end(HISTORY_META_TABLE, TableKind::Regular);
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

            if key_bytes.len() >= 8 {
                let id = u64::from_le_bytes(key_bytes[..8].try_into().unwrap());
                if let Some(entry) = HistoryEntry::from_le_bytes(val_bytes)
                    .filter(|e| e.status == HISTORY_STATUS_COMPLETE)
                {
                    results.push((id, entry));
                }
            }
        }

        results.sort_by(|a, b| b.0.cmp(&a.0)); // newest first
        Ok(results)
    }

    /// Get a specific completed history entry by snapshot ID.
    /// Returns `None` if the entry does not exist or is still pending.
    pub fn get(&self, snapshot_id: u64) -> Result<Option<HistoryEntry>, BfTreeError> {
        let key = encode_table_key(
            HISTORY_META_TABLE,
            TableKind::Regular,
            &snapshot_id.to_le_bytes(),
        );
        let max_val = self.db.adapter().inner().config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        match self.db.adapter().read(&key, &mut buf) {
            Ok(len) => match HistoryEntry::from_le_bytes(&buf[..len as usize]) {
                Some(entry) if entry.status == HISTORY_STATUS_COMPLETE => Ok(Some(entry)),
                _ => Ok(None),
            },
            Err(BfTreeError::NotFound | BfTreeError::Deleted) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Open a historical snapshot as a read-only database.
    /// The returned database is independent and can be read concurrently.
    ///
    /// Validates the snapshot path to prevent path traversal attacks from
    /// crafted history entries.
    pub fn open_historical(&self, snapshot_id: u64) -> Result<BfTreeDatabase, BfTreeError> {
        let entry = self.get(snapshot_id)?.ok_or(BfTreeError::NotFound)?;

        // Validate the path before using it to prevent directory traversal.
        validate_snapshot_path(&entry.snapshot_path)?;

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
    ///
    /// Returns the number of entries removed, or the first error encountered
    /// during metadata deletion. File-removal failures cause the entry to be
    /// skipped (it can be retried on a subsequent prune).
    pub fn prune(&self, keep: usize) -> Result<usize, BfTreeError> {
        let entries = self.list()?;
        if entries.len() <= keep {
            return Ok(0);
        }

        let to_remove = &entries[keep..];
        let mut removed = 0;
        let mut first_error: Option<BfTreeError> = None;

        for (id, entry) in to_remove {
            // Validate the path to prevent directory traversal via crafted entries.
            if validate_snapshot_path(&entry.snapshot_path).is_err() {
                // Skip entries with invalid paths -- delete the metadata but
                // do not attempt file operations on potentially malicious paths.
                let key =
                    encode_table_key(HISTORY_META_TABLE, TableKind::Regular, &id.to_le_bytes());
                self.db.adapter().delete(&key);
                removed += 1;
                continue;
            }
            let path = Path::new(&entry.snapshot_path);
            if path.exists() && std::fs::remove_file(path).is_err() {
                // If we cannot remove the snapshot file, skip metadata
                // deletion so the entry remains for a subsequent prune.
                if first_error.is_none() {
                    first_error = Some(BfTreeError::InvalidOperation(alloc::format!(
                        "failed to remove snapshot file {}",
                        entry.snapshot_path
                    )));
                }
                continue;
            }
            // Delete the history entry from BfTree only after file removal
            // succeeds (or the file was already absent).
            let key = encode_table_key(HISTORY_META_TABLE, TableKind::Regular, &id.to_le_bytes());
            self.db.adapter().delete(&key);
            removed += 1;
        }

        // If some entries were pruned but others failed, return the count of
        // successful removals. Only propagate the error if nothing was removed.
        if let (0, Some(err)) = (removed, first_error) {
            return Err(err);
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
    use crate::TableDefinition;
    use crate::bf_tree_store::config::BfTreeConfig;

    const DATA: TableDefinition<&str, u64> = TableDefinition::new("hist_data");

    #[test]
    fn history_entry_serialization() {
        let entry = HistoryEntry {
            txn_id: 42,
            timestamp_ns: 1234567890,
            snapshot_path: "/tmp/snap.bin".into(),
            status: HISTORY_STATUS_COMPLETE,
        };
        let bytes = entry.to_le_bytes().unwrap();
        let restored = HistoryEntry::from_le_bytes(&bytes).unwrap();
        assert_eq!(restored.txn_id, 42);
        assert_eq!(restored.timestamp_ns, 1234567890);
        assert_eq!(restored.snapshot_path, "/tmp/snap.bin");
        assert_eq!(restored.status, HISTORY_STATUS_COMPLETE);
    }

    #[test]
    fn from_le_bytes_rejects_short_input() {
        assert!(HistoryEntry::from_le_bytes(&[0u8; 10]).is_none());
        assert!(HistoryEntry::from_le_bytes(&[]).is_none());
        // Exactly 18 bytes (minimum header) should succeed.
        assert!(HistoryEntry::from_le_bytes(&[0u8; 18]).is_some());
    }

    #[test]
    fn commit_and_list_snapshots() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("hist.bftree");
        let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_file(&db_path, 4)).unwrap());

        let wtxn = db.begin_write();
        let mut t = wtxn.open_table(DATA).unwrap();
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
        let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_file(&db_path, 4)).unwrap());
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
        let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_file(&db_path, 4)).unwrap());
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
