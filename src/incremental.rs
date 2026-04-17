//! Incremental backup and portable snapshot export/import.
//!
//! Computes a logical delta between two transaction snapshots using the `CoW`
//! B-tree checksum fast-path: if a table's root checksum is unchanged, its
//! entire tree is skipped with zero I/O. Changed tables are diffed at the
//! key/value level via `iter()`.
//!
//! Requires `Builder::set_history_retention()` greater than zero so that
//! historical snapshots are available for comparison.

use crate::db::Database;
use crate::transactions::ReadTransaction;
use crate::tree_store::{InternalTableDefinition, TableType, hash128_with_seed};
use crate::{ReadableDatabase, StorageError, TableDefinition, TableHandle, UntypedTableHandle};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
#[cfg(feature = "std")]
use sha2::{Digest, Sha256};
#[cfg(feature = "std")]
use std::time::{Duration, Instant};

const DELTA_MAGIC: [u8; 8] = *b"shdbdelt";
const DELTA_VERSION: u8 = 1;
const HEADER_SIZE: usize = 40;
const SHA256_SIZE: usize = 32;
const XXH3_SEED: u64 = 0x1337_CAFE_DEAD_BEEF;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A logical delta of all key/value changes between two transaction snapshots.
///
/// Produced by [`Database::export_incremental()`] and consumed by
/// [`Database::import_incremental()`]. Can be serialized to a portable byte
/// buffer with [`to_bytes()`](IncrementalSnapshot::to_bytes) and deserialized
/// with [`from_bytes()`](IncrementalSnapshot::from_bytes).
#[derive(Clone, Debug)]
pub struct IncrementalSnapshot {
    /// Transaction ID of the base (older) snapshot.
    pub base_txn: u64,
    /// Transaction ID of the current (newer) snapshot.
    pub current_txn: u64,
    tables: Vec<TableDelta>,
    dropped_tables: Vec<String>,
}

/// Summary of an incremental file backup operation.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct IncrementalBackupReport {
    pub base_txn: u64,
    pub current_txn: u64,
    pub tables_included: u64,
    pub tables_skipped: u64,
    pub entries_upserted: u64,
    pub entries_deleted: u64,
    pub bytes_written: u64,
    pub duration: Duration,
}

/// Summary of an incremental import operation.
#[derive(Clone, Debug)]
pub struct IncrementalImportReport {
    pub entries_upserted: u64,
    pub entries_deleted: u64,
    pub tables_created: u64,
    pub tables_dropped: u64,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TableDelta {
    name: String,
    upserts: Vec<(Vec<u8>, Vec<u8>)>,
    deletes: Vec<Vec<u8>>,
}

// ---------------------------------------------------------------------------
// Checksum fast-path helper
// ---------------------------------------------------------------------------

/// Sentinel for "table not found or not a Normal table".
const NO_CHECKSUM: (u128, u64) = (u128::MAX, u64::MAX);
/// Sentinel for "table exists but is empty (no root)".
const EMPTY_CHECKSUM: (u128, u64) = (0, 0);

fn table_checksum(txn: &ReadTransaction, name: &str) -> Result<(u128, u64), StorageError> {
    match txn.table_tree().get_table_untyped(name, TableType::Normal) {
        Ok(Some(InternalTableDefinition::Normal {
            table_root: Some(hdr),
            ..
        })) => Ok((hdr.checksum, hdr.length)),
        Ok(Some(InternalTableDefinition::Normal {
            table_root: None, ..
        })) => Ok(EMPTY_CHECKSUM),
        Ok(Some(InternalTableDefinition::Multimap { .. }) | None) => Ok(NO_CHECKSUM),
        Err(e) => Err(e.into_storage_error_or_corrupted("get table root header")),
    }
}

fn checksums_match(base: (u128, u64), current: (u128, u64)) -> bool {
    base != NO_CHECKSUM && current != NO_CHECKSUM && base == current
}

// ---------------------------------------------------------------------------
// Core diff engine
// ---------------------------------------------------------------------------

fn collect_table_names(txn: &ReadTransaction) -> Result<Vec<String>, StorageError> {
    let handles: Vec<UntypedTableHandle> = txn.list_tables()?.collect();
    let mut names: Vec<String> = handles.iter().map(|h| h.name().to_string()).collect();
    names.sort();
    Ok(names)
}

fn compute_table_delta(
    base_txn: &ReadTransaction,
    current_txn: &ReadTransaction,
    name: &str,
) -> Result<Option<TableDelta>, StorageError> {
    let base_cksum = table_checksum(base_txn, name)?;
    let current_cksum = table_checksum(current_txn, name)?;

    // Fast path: identical checksum means identical content
    if checksums_match(base_cksum, current_cksum) {
        return Ok(None);
    }

    // Collect current entries
    let mut current_map: alloc::collections::BTreeMap<Vec<u8>, Vec<u8>> =
        alloc::collections::BTreeMap::new();
    if let Ok(table) = current_txn.open_untyped_table(UntypedTableHandle::new(name.to_string())) {
        for entry in table.iter_raw()? {
            let entry = entry?;
            current_map.insert(entry.key().to_vec(), entry.value().to_vec());
        }
    }

    // Collect base entries
    let mut base_map: alloc::collections::BTreeMap<Vec<u8>, Vec<u8>> =
        alloc::collections::BTreeMap::new();
    if base_cksum != NO_CHECKSUM
        && let Ok(table) = base_txn.open_untyped_table(UntypedTableHandle::new(name.to_string()))
    {
        for entry in table.iter_raw()? {
            let entry = entry?;
            base_map.insert(entry.key().to_vec(), entry.value().to_vec());
        }
    }

    // Deletes: keys in base but not in current
    let deletes: Vec<Vec<u8>> = base_map
        .keys()
        .filter(|k| !current_map.contains_key(k.as_slice()))
        .cloned()
        .collect();

    // Upserts: entries in current that are new or have changed values
    let upserts: Vec<(Vec<u8>, Vec<u8>)> = current_map
        .into_iter()
        .filter(|(k, v)| base_map.get(k).is_none_or(|bv| bv != v))
        .collect();

    if upserts.is_empty() && deletes.is_empty() {
        return Ok(None);
    }

    Ok(Some(TableDelta {
        name: name.to_string(),
        upserts,
        deletes,
    }))
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

pub(crate) fn export_incremental(
    db: &Database,
    since_txn: u64,
) -> Result<IncrementalSnapshot, StorageError> {
    let current_txn = db.begin_read().map_err(|e| e.into_storage_error())?;
    let base_txn = db
        .begin_read_at(since_txn)
        .map_err(|e| e.into_storage_error())?;

    let current_txn_id = db
        .get_memory()
        .get_last_committed_transaction_id()?
        .raw_id();

    let current_names = collect_table_names(&current_txn)?;
    let base_names = collect_table_names(&base_txn)?;

    // Compute deltas for each table in current snapshot
    let mut tables = Vec::new();
    for name in &current_names {
        if let Some(delta) = compute_table_delta(&base_txn, &current_txn, name)? {
            tables.push(delta);
        }
    }

    // Find dropped tables (in base but not in current)
    let current_name_set: alloc::collections::BTreeSet<&str> =
        current_names.iter().map(|s| s.as_str()).collect();
    let dropped_tables: Vec<String> = base_names
        .into_iter()
        .filter(|n| !current_name_set.contains(n.as_str()))
        .collect();

    Ok(IncrementalSnapshot {
        base_txn: since_txn,
        current_txn: current_txn_id,
        tables,
        dropped_tables,
    })
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

pub(crate) fn import_incremental(
    db: &Database,
    snapshot: &IncrementalSnapshot,
) -> Result<IncrementalImportReport, StorageError> {
    let txn = db.begin_write().map_err(|e| e.into_storage_error())?;
    let mut report = IncrementalImportReport {
        entries_upserted: 0,
        entries_deleted: 0,
        tables_created: 0,
        tables_dropped: 0,
    };

    for delta in &snapshot.tables {
        // Check if this is a new table
        let table_exists = txn.list_tables()?.any(|h| h.name() == delta.name);
        if !table_exists {
            report.tables_created += 1;
        }

        {
            let def = TableDefinition::<&[u8], &[u8]>::new(&delta.name);
            let mut table = txn
                .open_table(def)
                .map_err(|e| e.into_storage_error_or_corrupted("open table during import"))?;
            for (key, value) in &delta.upserts {
                table.insert(key.as_slice(), value.as_slice())?;
                report.entries_upserted += 1;
            }
            for key in &delta.deletes {
                table.remove(key.as_slice())?;
                report.entries_deleted += 1;
            }
        }
    }

    for name in &snapshot.dropped_tables {
        let handle = UntypedTableHandle::new(name.clone());
        let deleted = txn
            .delete_table(handle)
            .map_err(|e| e.into_storage_error_or_corrupted("delete table during import"))?;
        if deleted {
            report.tables_dropped += 1;
        }
    }

    txn.commit().map_err(|e| e.into_storage_error())?;
    Ok(report)
}

// ---------------------------------------------------------------------------
// Snapshot inspection
// ---------------------------------------------------------------------------

impl IncrementalSnapshot {
    /// Number of tables that have at least one upsert or delete.
    pub fn tables_changed(&self) -> usize {
        self.tables.len()
    }

    /// Names of tables that were dropped between the base and current snapshots.
    pub fn dropped_table_names(&self) -> &[String] {
        &self.dropped_tables
    }

    /// Total number of upserted entries across all tables.
    pub fn total_upserts(&self) -> usize {
        self.tables.iter().map(|d| d.upserts.len()).sum()
    }

    /// Total number of deleted keys across all tables.
    pub fn total_deletes(&self) -> usize {
        self.tables.iter().map(|d| d.deletes.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Binary serialization
// ---------------------------------------------------------------------------

/// Verify that `payload[pos..pos+need]` is in bounds, returning a descriptive
/// error when the payload is truncated.
#[cfg(feature = "std")]
fn check_bounds(payload: &[u8], pos: usize, need: usize) -> Result<(), StorageError> {
    if pos.checked_add(need).is_none_or(|end| end > payload.len()) {
        return Err(StorageError::format_error(alloc::format!(
            "incremental snapshot truncated: need {} bytes at offset {}, have {}",
            need,
            pos,
            payload.len()
        )));
    }
    Ok(())
}

impl IncrementalSnapshot {
    /// Encode this snapshot into a portable byte buffer.
    ///
    /// The format includes per-entry xxh3-128 checksums and a SHA-256 footer
    /// for whole-buffer integrity verification.
    #[cfg(feature = "std")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&DELTA_MAGIC);
        buf.push(DELTA_VERSION);
        buf.extend_from_slice(&[0u8; 3]); // padding
        buf.extend_from_slice(&self.base_txn.to_le_bytes());
        buf.extend_from_slice(&self.current_txn.to_le_bytes());
        buf.extend_from_slice(&(self.tables.len() as u64).to_le_bytes());

        // Per-table sections
        for delta in &self.tables {
            let name_bytes = delta.name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&(delta.upserts.len() as u64).to_le_bytes());
            buf.extend_from_slice(&(delta.deletes.len() as u64).to_le_bytes());

            for (key, value) in &delta.upserts {
                buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
                buf.extend_from_slice(key);
                buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
                buf.extend_from_slice(value);
                let mut combined = Vec::with_capacity(key.len() + value.len());
                combined.extend_from_slice(key);
                combined.extend_from_slice(value);
                let checksum = hash128_with_seed(&combined, XXH3_SEED);
                buf.extend_from_slice(&checksum.to_le_bytes());
            }

            for key in &delta.deletes {
                buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
                buf.extend_from_slice(key);
            }
        }

        // Dropped tables section
        buf.extend_from_slice(&(self.dropped_tables.len() as u64).to_le_bytes());
        for name in &self.dropped_tables {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(name_bytes);
        }

        // SHA-256 footer
        let hash = Sha256::digest(&buf);
        buf.extend_from_slice(&hash);

        buf
    }

    /// Decode a snapshot from a byte buffer produced by [`to_bytes()`](Self::to_bytes).
    ///
    /// Verifies the SHA-256 footer and per-entry xxh3-128 checksums.
    #[cfg(feature = "std")]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_bytes(data: &[u8]) -> Result<Self, StorageError> {
        if data.len() < HEADER_SIZE + SHA256_SIZE {
            return Err(StorageError::format_error("incremental delta too short"));
        }

        // Verify SHA-256 footer
        let payload = &data[..data.len() - SHA256_SIZE];
        let stored_hash = &data[data.len() - SHA256_SIZE..];
        let computed_hash = Sha256::digest(payload);
        if computed_hash.as_slice() != stored_hash {
            return Err(StorageError::format_error(
                "incremental delta SHA-256 mismatch",
            ));
        }

        let mut pos = 0;

        // Header
        check_bounds(payload, pos, 8)?;
        let magic = &payload[pos..pos + 8];
        if magic != DELTA_MAGIC {
            return Err(StorageError::format_error("incremental delta bad magic"));
        }
        pos += 8;

        check_bounds(payload, pos, 4)?;
        let version = payload[pos];
        if version != DELTA_VERSION {
            return Err(StorageError::format_error(
                "incremental delta unsupported version",
            ));
        }
        pos += 1 + 3; // version + padding

        check_bounds(payload, pos, 8)?;
        let base_txn = u64::from_le_bytes(payload[pos..pos + 8].try_into().unwrap());
        pos += 8;
        check_bounds(payload, pos, 8)?;
        let current_txn = u64::from_le_bytes(payload[pos..pos + 8].try_into().unwrap());
        pos += 8;
        check_bounds(payload, pos, 8)?;
        let table_count = u64::from_le_bytes(payload[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let mut tables = Vec::new();
        for _ in 0..table_count {
            check_bounds(payload, pos, 2)?;
            let name_len = u16::from_le_bytes(payload[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            check_bounds(payload, pos, name_len)?;
            let name = core::str::from_utf8(&payload[pos..pos + name_len])
                .map_err(|_| StorageError::format_error("incremental delta invalid table name"))?;
            pos += name_len;

            check_bounds(payload, pos, 8)?;
            let upsert_count =
                u64::from_le_bytes(payload[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            check_bounds(payload, pos, 8)?;
            let delete_count =
                u64::from_le_bytes(payload[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;

            let mut upserts = Vec::with_capacity(upsert_count);
            for _ in 0..upsert_count {
                check_bounds(payload, pos, 4)?;
                let key_len =
                    u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                check_bounds(payload, pos, key_len)?;
                let key = payload[pos..pos + key_len].to_vec();
                pos += key_len;

                check_bounds(payload, pos, 4)?;
                let val_len =
                    u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                check_bounds(payload, pos, val_len)?;
                let value = payload[pos..pos + val_len].to_vec();
                pos += val_len;

                check_bounds(payload, pos, 16)?;
                let stored_checksum =
                    u128::from_le_bytes(payload[pos..pos + 16].try_into().unwrap());
                pos += 16;

                let mut combined = Vec::with_capacity(key_len + val_len);
                combined.extend_from_slice(&key);
                combined.extend_from_slice(&value);
                let computed_checksum = hash128_with_seed(&combined, XXH3_SEED);
                if stored_checksum != computed_checksum {
                    return Err(StorageError::format_error(
                        "incremental delta entry checksum mismatch",
                    ));
                }

                upserts.push((key, value));
            }

            let mut deletes = Vec::with_capacity(delete_count);
            for _ in 0..delete_count {
                check_bounds(payload, pos, 4)?;
                let key_len =
                    u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                check_bounds(payload, pos, key_len)?;
                let key = payload[pos..pos + key_len].to_vec();
                pos += key_len;
                deletes.push(key);
            }

            tables.push(TableDelta {
                name: name.to_string(),
                upserts,
                deletes,
            });
        }

        // Dropped tables
        check_bounds(payload, pos, 8)?;
        let dropped_count = u64::from_le_bytes(payload[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let mut dropped_tables = Vec::with_capacity(dropped_count);
        for _ in 0..dropped_count {
            check_bounds(payload, pos, 2)?;
            let name_len = u16::from_le_bytes(payload[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            check_bounds(payload, pos, name_len)?;
            let name = core::str::from_utf8(&payload[pos..pos + name_len]).map_err(|_| {
                StorageError::format_error("incremental delta invalid dropped table name")
            })?;
            pos += name_len;
            dropped_tables.push(name.to_string());
        }

        Ok(Self {
            base_txn,
            current_txn,
            tables,
            dropped_tables,
        })
    }
}

// ---------------------------------------------------------------------------
// File backup / apply
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
pub(crate) fn backup_incremental(
    db: &Database,
    dest: &std::path::Path,
    since_txn: u64,
) -> Result<IncrementalBackupReport, StorageError> {
    let start = Instant::now();
    let snapshot = export_incremental(db, since_txn)?;

    let mut tables_included = 0u64;
    let mut entries_upserted = 0u64;
    let mut entries_deleted = 0u64;
    for delta in &snapshot.tables {
        tables_included += 1;
        entries_upserted += delta.upserts.len() as u64;
        entries_deleted += delta.deletes.len() as u64;
    }

    // Count tables that were skipped (in current but not in delta)
    let skip_txn = db.begin_read().map_err(|e| e.into_storage_error())?;
    let total_tables = collect_table_names(&skip_txn)?.len() as u64;
    let tables_skipped =
        total_tables.saturating_sub(tables_included + snapshot.dropped_tables.len() as u64);

    let bytes = snapshot.to_bytes();
    let bytes_written = bytes.len() as u64;

    std::fs::write(dest, &bytes).map_err(|e| StorageError::Io(crate::BackendError::Io(e)))?;

    Ok(IncrementalBackupReport {
        base_txn: snapshot.base_txn,
        current_txn: snapshot.current_txn,
        tables_included,
        tables_skipped,
        entries_upserted,
        entries_deleted,
        bytes_written,
        duration: start.elapsed(),
    })
}

#[cfg(feature = "std")]
pub(crate) fn apply_incremental_backup(
    db: &Database,
    path: &std::path::Path,
) -> Result<IncrementalImportReport, StorageError> {
    let data = std::fs::read(path).map_err(|e| StorageError::Io(crate::BackendError::Io(e)))?;
    let snapshot = IncrementalSnapshot::from_bytes(&data)?;
    import_incremental(db, &snapshot)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Builder, Database, TableDefinition};

    const TABLE_A: TableDefinition<&str, u64> = TableDefinition::new("table_a");
    const TABLE_B: TableDefinition<&str, u64> = TableDefinition::new("table_b");

    fn create_db_with_history(retention: u64) -> (tempfile::NamedTempFile, Database) {
        let file = crate::create_tempfile();
        let db = Builder::new()
            .set_history_retention(retention)
            .create(file.path())
            .unwrap();
        (file, db)
    }

    fn get_txn_id(db: &Database) -> u64 {
        db.get_memory()
            .get_last_committed_transaction_id()
            .unwrap()
            .raw_id()
    }

    #[test]
    fn incremental_no_changes() {
        let (_file, db) = create_db_with_history(10);
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            t.insert("k1", &1u64).unwrap();
        }
        txn.commit().unwrap();
        let base_id = get_txn_id(&db);

        // No changes after this point -- just commit an empty txn to advance ID
        let txn = db.begin_write().unwrap();
        txn.commit().unwrap();

        let snapshot = export_incremental(&db, base_id).unwrap();
        assert!(snapshot.tables.is_empty());
        assert!(snapshot.dropped_tables.is_empty());
    }

    #[test]
    fn incremental_basic_upserts() {
        let (_file, db) = create_db_with_history(10);

        // Base: insert keys 1-10
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            for i in 0u64..10 {
                let key = alloc::format!("key_{i:03}");
                t.insert(key.as_str(), &i).unwrap();
            }
        }
        txn.commit().unwrap();
        let base_id = get_txn_id(&db);

        // Insert 10-19 and update key_000
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            for i in 10u64..20 {
                let key = alloc::format!("key_{i:03}");
                t.insert(key.as_str(), &i).unwrap();
            }
            t.insert("key_000", &999u64).unwrap();
        }
        txn.commit().unwrap();

        let snapshot = export_incremental(&db, base_id).unwrap();
        assert_eq!(snapshot.tables.len(), 1);
        let delta = &snapshot.tables[0];
        assert_eq!(delta.name, "table_a");
        // 10 new keys + 1 updated = 11 upserts
        assert_eq!(delta.upserts.len(), 11);
        assert!(delta.deletes.is_empty());
    }

    #[test]
    fn incremental_delete_tracking() {
        let (_file, db) = create_db_with_history(10);

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            for i in 0u64..10 {
                let key = alloc::format!("key_{i:03}");
                t.insert(key.as_str(), &i).unwrap();
            }
        }
        txn.commit().unwrap();
        let base_id = get_txn_id(&db);

        // Delete keys 0-4, insert 10-14
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            for i in 0u64..5 {
                let key = alloc::format!("key_{i:03}");
                t.remove(key.as_str()).unwrap();
            }
            for i in 10u64..15 {
                let key = alloc::format!("key_{i:03}");
                t.insert(key.as_str(), &i).unwrap();
            }
        }
        txn.commit().unwrap();

        let snapshot = export_incremental(&db, base_id).unwrap();
        assert_eq!(snapshot.tables.len(), 1);
        let delta = &snapshot.tables[0];
        assert_eq!(delta.upserts.len(), 5);
        assert_eq!(delta.deletes.len(), 5);
    }

    #[test]
    fn incremental_unchanged_table_skipped() {
        let (_file, db) = create_db_with_history(10);

        // Write to both tables
        let txn = db.begin_write().unwrap();
        {
            let mut a = txn.open_table(TABLE_A).unwrap();
            a.insert("a1", &1u64).unwrap();
            let mut b = txn.open_table(TABLE_B).unwrap();
            b.insert("b1", &1u64).unwrap();
        }
        txn.commit().unwrap();
        let base_id = get_txn_id(&db);

        // Only modify table_b
        let txn = db.begin_write().unwrap();
        {
            let mut b = txn.open_table(TABLE_B).unwrap();
            b.insert("b2", &2u64).unwrap();
        }
        txn.commit().unwrap();

        let snapshot = export_incremental(&db, base_id).unwrap();
        // Only table_b should appear in delta
        assert_eq!(snapshot.tables.len(), 1);
        assert_eq!(snapshot.tables[0].name, "table_b");
    }

    #[test]
    fn incremental_export_import_roundtrip() {
        let (_file_src, src_db) = create_db_with_history(10);

        // Base data
        let txn = src_db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            for i in 0u64..10 {
                let key = alloc::format!("key_{i:03}");
                t.insert(key.as_str(), &i).unwrap();
            }
        }
        txn.commit().unwrap();
        let base_id = get_txn_id(&src_db);

        // Changes
        let txn = src_db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            t.insert("key_000", &999u64).unwrap();
            t.insert("new_key", &42u64).unwrap();
            t.remove("key_001").unwrap();
        }
        txn.commit().unwrap();

        let snapshot = export_incremental(&src_db, base_id).unwrap();
        assert_eq!(snapshot.tables.len(), 1);
        let delta = &snapshot.tables[0];
        assert_eq!(delta.upserts.len(), 2); // key_000 updated + new_key
        assert_eq!(delta.deletes.len(), 1); // key_001

        // Import into a fresh DB (creates raw-typed tables)
        let file_dst = crate::create_tempfile();
        let dst_db = Database::create(file_dst.path()).unwrap();

        let report = import_incremental(&dst_db, &snapshot).unwrap();
        assert_eq!(report.entries_upserted, 2);
        assert_eq!(report.entries_deleted, 1); // delete operation is applied even on fresh DB
        assert_eq!(report.tables_created, 1);

        // Verify via untyped read
        let rtxn = dst_db.begin_read().unwrap();
        let t = rtxn
            .open_untyped_table(UntypedTableHandle::new("table_a".into()))
            .unwrap();
        let entries: Vec<_> = t
            .iter_raw()
            .unwrap()
            .collect::<core::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(entries.len(), 2); // only the upserts (key_000, new_key)
    }

    #[test]
    fn incremental_base_not_in_history() {
        let (_file, db) = create_db_with_history(0); // no history retention
        let txn = db.begin_write().unwrap();
        txn.commit().unwrap();

        let result = export_incremental(&db, 999);
        assert!(result.is_err());
    }

    #[test]
    fn incremental_snapshot_bytes_roundtrip() {
        let (_file, db) = create_db_with_history(10);

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            t.insert("hello", &42u64).unwrap();
        }
        txn.commit().unwrap();
        let base_id = get_txn_id(&db);

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            t.insert("world", &99u64).unwrap();
        }
        txn.commit().unwrap();

        let snapshot = export_incremental(&db, base_id).unwrap();
        let bytes = snapshot.to_bytes();
        let restored = IncrementalSnapshot::from_bytes(&bytes).unwrap();

        assert_eq!(restored.base_txn, snapshot.base_txn);
        assert_eq!(restored.current_txn, snapshot.current_txn);
        assert_eq!(restored.tables.len(), snapshot.tables.len());
        assert_eq!(
            restored.tables[0].upserts.len(),
            snapshot.tables[0].upserts.len()
        );
        assert_eq!(restored.dropped_tables.len(), snapshot.dropped_tables.len());
    }

    #[test]
    fn incremental_chain() {
        let (_file, db) = create_db_with_history(10);

        // txn1: insert 0-9
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            for i in 0u64..10 {
                let key = alloc::format!("key_{i:03}");
                t.insert(key.as_str(), &i).unwrap();
            }
        }
        txn.commit().unwrap();
        let txn1_id = get_txn_id(&db);

        // txn2: insert 10-19
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            for i in 10u64..20 {
                let key = alloc::format!("key_{i:03}");
                t.insert(key.as_str(), &i).unwrap();
            }
        }
        txn.commit().unwrap();
        let txn2_id = get_txn_id(&db);

        // txn3: insert 20-29
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            for i in 20u64..30 {
                let key = alloc::format!("key_{i:03}");
                t.insert(key.as_str(), &i).unwrap();
            }
        }
        txn.commit().unwrap();

        // delta1 captures changes from txn1 -> txn3 (keys 10-29)
        let delta1 = export_incremental(&db, txn1_id).unwrap();
        // delta2 captures changes from txn2 -> txn3 (keys 20-29)
        let delta2 = export_incremental(&db, txn2_id).unwrap();

        // Apply delta1 to a fresh DB -- should contain 20 upserts
        let file_dst = crate::create_tempfile();
        let dst_db = Database::create(file_dst.path()).unwrap();

        let r1 = import_incremental(&dst_db, &delta1).unwrap();
        assert_eq!(r1.entries_upserted, 20);

        // Apply delta2 (keys 20-29) on top -- already present, but re-applied
        let r2 = import_incremental(&dst_db, &delta2).unwrap();
        assert_eq!(r2.entries_upserted, 10);

        // Total unique entries: keys 10-29 = 20
        let rtxn = dst_db.begin_read().unwrap();
        let t = rtxn
            .open_untyped_table(UntypedTableHandle::new("table_a".into()))
            .unwrap();
        let count = t.iter_raw().unwrap().count();
        assert_eq!(count, 20);
    }

    #[test]
    fn incremental_file_backup_and_apply() {
        let (_file, db) = create_db_with_history(10);

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            t.insert("k1", &1u64).unwrap();
        }
        txn.commit().unwrap();
        let base_id = get_txn_id(&db);

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            t.insert("k2", &2u64).unwrap();
        }
        txn.commit().unwrap();

        let backup_file = crate::create_tempfile();
        let report = backup_incremental(&db, backup_file.path(), base_id).unwrap();
        assert_eq!(report.tables_included, 1);
        assert_eq!(report.entries_upserted, 1);

        // Apply to a fresh DB (no pre-seeding)
        let file_dst = crate::create_tempfile();
        let dst_db = Database::create(file_dst.path()).unwrap();

        let import_report = apply_incremental_backup(&dst_db, backup_file.path()).unwrap();
        assert_eq!(import_report.entries_upserted, 1);
        assert_eq!(import_report.tables_created, 1);

        // Verify via untyped read
        let rtxn = dst_db.begin_read().unwrap();
        let t = rtxn
            .open_untyped_table(UntypedTableHandle::new("table_a".into()))
            .unwrap();
        assert_eq!(t.iter_raw().unwrap().count(), 1);
    }

    #[test]
    fn incremental_file_integrity_check() {
        let (_file, db) = create_db_with_history(10);

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            t.insert("k1", &1u64).unwrap();
        }
        txn.commit().unwrap();
        let base_id = get_txn_id(&db);

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            t.insert("k2", &2u64).unwrap();
        }
        txn.commit().unwrap();

        let backup_file = crate::create_tempfile();
        backup_incremental(&db, backup_file.path(), base_id).unwrap();

        // Corrupt one byte in the middle
        let mut data = std::fs::read(backup_file.path()).unwrap();
        if data.len() > HEADER_SIZE + 10 {
            data[HEADER_SIZE + 5] ^= 0xFF;
            std::fs::write(backup_file.path(), &data).unwrap();
        }

        let dst_file = crate::create_tempfile();
        let dst_db = Database::create(dst_file.path()).unwrap();
        let result = apply_incremental_backup(&dst_db, backup_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn incremental_dropped_table() {
        let (_file, db) = create_db_with_history(10);

        // Create two tables
        let txn = db.begin_write().unwrap();
        {
            let mut ta = txn.open_table(TABLE_A).unwrap();
            ta.insert("a1", &1u64).unwrap();
            let mut tb = txn.open_table(TABLE_B).unwrap();
            tb.insert("b1", &10u64).unwrap();
        }
        txn.commit().unwrap();
        let base_id = get_txn_id(&db);

        // Drop TABLE_B
        let txn = db.begin_write().unwrap();
        txn.delete_table(TABLE_B).unwrap();
        txn.commit().unwrap();

        let snap = export_incremental(&db, base_id).unwrap();
        assert_eq!(snap.dropped_table_names().len(), 1);
        assert_eq!(snap.dropped_table_names()[0], "table_b");
        assert_eq!(snap.total_upserts(), 0);
        assert_eq!(snap.total_deletes(), 0);

        // Roundtrip through bytes
        let bytes = snap.to_bytes();
        let snap2 = IncrementalSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(snap2.dropped_table_names().len(), 1);
        assert_eq!(snap2.dropped_table_names()[0], "table_b");

        // Import into fresh DB that has TABLE_B pre-created via untyped path
        let dst_file = crate::create_tempfile();
        let dst_db = Database::create(dst_file.path()).unwrap();
        {
            let txn = dst_db.begin_write().unwrap();
            {
                let def = TableDefinition::<&[u8], &[u8]>::new("table_b");
                let mut t = txn.open_table(def).unwrap();
                t.insert(b"x".as_slice(), b"y".as_slice()).unwrap();
            }
            txn.commit().unwrap();
        }

        let report = import_incremental(&dst_db, &snap2).unwrap();
        assert_eq!(report.tables_dropped, 1);

        // Verify TABLE_B no longer exists
        let rtxn = dst_db.begin_read().unwrap();
        let result = rtxn.open_untyped_table(UntypedTableHandle::new("table_b".into()));
        assert!(result.is_err());
    }

    #[test]
    fn incremental_from_bytes_truncated_payload() {
        let (_file, db) = create_db_with_history(10);

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            t.insert("k1", &1u64).unwrap();
        }
        txn.commit().unwrap();
        let base_id = get_txn_id(&db);

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE_A).unwrap();
            t.insert("k2", &2u64).unwrap();
        }
        txn.commit().unwrap();

        let snapshot = export_incremental(&db, base_id).unwrap();
        let bytes = snapshot.to_bytes();

        // Truncate at various points within the payload (before the SHA-256
        // footer) and re-append a valid SHA-256 so the footer check passes but
        // the structural parse hits the bounds check.
        let payload_len = bytes.len() - SHA256_SIZE;
        let truncation_points = [
            HEADER_SIZE + 1,  // inside table section header
            HEADER_SIZE + 10, // inside upsert data
            HEADER_SIZE + 20, // deeper inside upsert data
            payload_len / 2,  // midpoint
        ];

        for &trunc in &truncation_points {
            if trunc >= payload_len {
                continue;
            }
            let truncated_payload = &bytes[..trunc];
            let hash = Sha256::digest(truncated_payload);
            let mut tampered = truncated_payload.to_vec();
            tampered.extend_from_slice(&hash);

            let result = IncrementalSnapshot::from_bytes(&tampered);
            assert!(
                result.is_err(),
                "expected error for truncation at byte {trunc}, got Ok"
            );
            let err_msg = alloc::format!("{}", result.unwrap_err());
            assert!(
                err_msg.contains("truncated"),
                "expected 'truncated' in error message for truncation at byte {trunc}, got: {err_msg}"
            );
        }
    }

    #[test]
    fn incremental_from_bytes_too_short() {
        // A payload shorter than HEADER_SIZE + SHA256_SIZE should fail immediately
        let result = IncrementalSnapshot::from_bytes(&[0u8; HEADER_SIZE + SHA256_SIZE - 1]);
        assert!(result.is_err());
    }
}
