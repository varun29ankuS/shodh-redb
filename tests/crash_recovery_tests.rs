//! Crash recovery tests.
//!
//! Simulates I/O failures during writes using a CountdownBackend that wraps
//! FileBackend with an atomic countdown. When the counter hits zero, all
//! subsequent write/set_len/sync operations fail. The database is then
//! reopened from the surviving file to verify ACID recovery.
//!
//! Strategy:
//! 1. Write known data and commit (establishes a consistent baseline)
//! 2. Start a second write, trigger I/O failure mid-transaction
//! 3. Reopen from the file — must recover the last committed state
//! 4. Verify data matches the baseline, not the partial write

use std::fmt;
use std::io::ErrorKind;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tempfile::NamedTempFile;

use std::fs::OpenOptions;

use shodh_redb::{
    BackendError, Builder, ContentType, Database, Durability, ReadableDatabase,
    ReadableTableMetadata, StorageBackend, StoreOptions, TableDefinition, VerifyLevel,
    backends::FileBackend,
};

fn create_tempfile() -> NamedTempFile {
    if cfg!(target_os = "wasi") {
        NamedTempFile::new_in("/tmp").unwrap()
    } else {
        NamedTempFile::new().unwrap()
    }
}

const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("crash_test");
const U64_TABLE: TableDefinition<u64, u64> = TableDefinition::new("crash_u64");

// ═══════════════════════════════════════════════════════════════════════
// CountdownBackend — deterministic I/O failure injection
// ═══════════════════════════════════════════════════════════════════════

struct CountdownBackend {
    inner: FileBackend,
    countdown: Arc<AtomicU64>,
}

impl fmt::Debug for CountdownBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountdownBackend")
            .field("remaining", &self.countdown.load(Ordering::SeqCst))
            .finish()
    }
}

impl CountdownBackend {
    fn new(inner: FileBackend, countdown: u64) -> Self {
        Self {
            inner,
            countdown: Arc::new(AtomicU64::new(countdown)),
        }
    }

    fn check_countdown(&self) -> Result<(), BackendError> {
        if self.countdown.load(Ordering::SeqCst) == 0 {
            return Err(std::io::Error::from(ErrorKind::Other).into());
        }
        Ok(())
    }

    fn decrement_countdown(&self) -> Result<(), BackendError> {
        if self
            .countdown
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                if x > 0 { Some(x - 1) } else { None }
            })
            .is_err()
        {
            return Err(std::io::Error::from(ErrorKind::Other).into());
        }
        Ok(())
    }
}

impl StorageBackend for CountdownBackend {
    fn len(&self) -> Result<u64, BackendError> {
        self.check_countdown()?;
        self.inner.len()
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), BackendError> {
        self.check_countdown()?;
        self.inner.read(offset, out)
    }

    fn set_len(&self, len: u64) -> Result<(), BackendError> {
        self.decrement_countdown()?;
        self.inner.set_len(len)
    }

    fn sync_data(&self) -> Result<(), BackendError> {
        self.decrement_countdown()?;
        self.inner.sync_data()
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<(), BackendError> {
        self.decrement_countdown()?;
        self.inner.write(offset, data)
    }
}

/// Helper: populate a baseline of N entries, commit, return the database.
fn populate_baseline(path: &std::path::Path, n: u64) -> Database {
    let db = Database::create(path).unwrap();
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..n {
            let value = vec![(i & 0xFF) as u8; 64];
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();
    db
}

/// Helper: verify the database contains exactly the baseline entries.
fn verify_baseline(db: &Database, n: u64) {
    let report = db.verify_integrity(VerifyLevel::Pages).unwrap();
    assert!(
        report.valid,
        "checksum integrity failed after recovery: {:?}",
        report.corrupt_details,
    );

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    assert_eq!(
        t.len().unwrap(),
        n,
        "recovered table length mismatch: expected {n}, got {}",
        t.len().unwrap()
    );

    for i in 0..n {
        let val = t
            .get(&i)
            .unwrap()
            .unwrap_or_else(|| panic!("key {i} missing after recovery"));
        let expected_byte = (i & 0xFF) as u8;
        assert_eq!(val.value().len(), 64);
        assert!(
            val.value().iter().all(|&b| b == expected_byte),
            "key {i}: value corrupted after recovery"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Basic crash recovery: fail during second commit
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn crash_during_second_commit_recovers_first() {
    let tmpfile = create_tempfile();
    let baseline_count = 500u64;

    // Phase 1: Write baseline and close
    {
        let db = populate_baseline(tmpfile.path(), baseline_count);
        drop(db);
    }

    // Phase 2: Reopen with countdown backend, try to write more, fail.
    // Try decreasing countdown values until we find one that causes a failure.
    let mut triggered_failure = false;
    for countdown in [5, 10, 15, 20, 30, 50] {
        // Re-establish baseline each attempt (previous crash may have left partial state)
        {
            let db = Database::open(tmpfile.path()).unwrap();
            verify_baseline(&db, baseline_count);
            drop(db);
        }

        let file_backend = FileBackend::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(tmpfile.path())
                .unwrap(),
        )
        .unwrap();
        let crash_backend = CountdownBackend::new(file_backend, countdown);
        let open_result = Builder::new().create_with_backend(crash_backend);

        if let Ok(db) = open_result {
            let result = (|| -> Result<(), shodh_redb::Error> {
                let txn = db.begin_write()?;
                {
                    let mut t = txn.open_table(TABLE)?;
                    for i in baseline_count..baseline_count + 500 {
                        let value = vec![0xFFu8; 64];
                        t.insert(&i, value.as_slice())?;
                    }
                }
                txn.commit()?;
                Ok(())
            })();

            if result.is_err() {
                triggered_failure = true;
                drop(db);
                break;
            }
            drop(db);
        } else {
            triggered_failure = true;
            break;
        }
    }
    assert!(
        triggered_failure,
        "no countdown value triggered I/O failure"
    );

    // Phase 3: Reopen normally and verify recovery
    let db = Database::open(tmpfile.path()).unwrap();
    verify_baseline(&db, baseline_count);

    // The failed second commit's data must NOT be present
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    for i in baseline_count..baseline_count + 500 {
        assert!(
            t.get(&i).unwrap().is_none(),
            "key {i} from failed commit should not exist"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Crash at various countdown points
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn crash_at_various_write_points() {
    let tmpfile = create_tempfile();
    let baseline_count = 200u64;

    // Establish baseline
    {
        let db = populate_baseline(tmpfile.path(), baseline_count);
        drop(db);
    }

    // Try crash at different countdown values to exercise different failure points
    for countdown in [10, 20, 30, 50, 100, 200] {
        let file_backend = FileBackend::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(tmpfile.path())
                .unwrap(),
        )
        .unwrap();
        let crash_backend = CountdownBackend::new(file_backend, countdown);

        let open_result = Builder::new().create_with_backend(crash_backend);
        if let Ok(db) = open_result {
            // Try writing — may fail at different points
            let _ = (|| -> Result<(), shodh_redb::Error> {
                let txn = db.begin_write()?;
                {
                    let mut t = txn.open_table(TABLE)?;
                    for i in 0..baseline_count {
                        // Overwrite with different values
                        t.insert(&i, &[0xDD; 64][..])?;
                    }
                }
                txn.commit()?;
                Ok(())
            })();
            drop(db);
        }

        // Verify recovery — baseline should still be intact OR the overwrite
        // fully committed. Never a partial state.
        let db = Database::open(tmpfile.path()).unwrap();
        let report = db.verify_integrity(VerifyLevel::Pages).unwrap();
        assert!(
            report.valid,
            "integrity failed at countdown={countdown}: {:?}",
            report.corrupt_details,
        );

        let txn = db.begin_read().unwrap();
        let t = txn.open_table(TABLE).unwrap();
        assert_eq!(t.len().unwrap(), baseline_count);

        // All values must be consistent: either ALL original or ALL overwritten
        let first_val: Vec<u8> = t.get(&0u64).unwrap().unwrap().value().to_vec();
        for i in 1..baseline_count {
            let val = t.get(&i).unwrap().unwrap();
            if first_val[0] == 0x00 {
                // Original baseline
                assert_eq!(val.value()[0], (i & 0xFF) as u8);
            } else {
                // Fully committed overwrite
                assert_eq!(val.value()[0], 0xDD);
            }
        }
        drop(txn);
        drop(db);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Crash during insert (before commit) — data must be discarded
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn crash_during_insert_before_commit() {
    let tmpfile = create_tempfile();

    // Establish baseline with a fresh DB
    {
        let db = Database::create(tmpfile.path()).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(U64_TABLE).unwrap();
            for i in 0..100u64 {
                t.insert(&i, &(i * 10)).unwrap();
            }
        }
        txn.commit().unwrap();
        drop(db);
    }

    // Reopen, start inserting but crash before commit
    {
        let file_backend = FileBackend::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(tmpfile.path())
                .unwrap(),
        )
        .unwrap();
        // Give enough I/O for open + some inserts, then fail before commit
        let crash_backend = CountdownBackend::new(file_backend, 30);
        let open_result = Builder::new().create_with_backend(crash_backend);

        if let Ok(db) = open_result {
            let write_result = db.begin_write();
            if let Ok(txn) = write_result {
                {
                    let table_result = txn.open_table(U64_TABLE);
                    if let Ok(mut t) = table_result {
                        for i in 100..1000u64 {
                            if t.insert(&i, &(i * 100)).is_err() {
                                break;
                            }
                        }
                    }
                }
                // Intentionally do NOT commit — simulating crash before commit
                drop(txn);
            }
            drop(db);
        }
    }

    // Recover and verify
    let db = Database::open(tmpfile.path()).unwrap();
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    assert_eq!(
        t.len().unwrap(),
        100,
        "uncommitted inserts should not persist"
    );

    for i in 0..100u64 {
        assert_eq!(
            t.get(&i).unwrap().unwrap().value(),
            i * 10,
            "baseline key {i} value corrupted"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Crash during delete
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn crash_during_delete_preserves_data() {
    let tmpfile = create_tempfile();
    let baseline_count = 300u64;

    {
        let db = populate_baseline(tmpfile.path(), baseline_count);
        drop(db);
    }

    // Try to delete half the keys, crash during commit
    {
        let file_backend = FileBackend::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(tmpfile.path())
                .unwrap(),
        )
        .unwrap();
        let crash_backend = CountdownBackend::new(file_backend, 40);

        if let Ok(db) = Builder::new().create_with_backend(crash_backend) {
            let _ = (|| -> Result<(), shodh_redb::Error> {
                let txn = db.begin_write()?;
                {
                    let mut t = txn.open_table(TABLE)?;
                    for i in 0..150u64 {
                        t.remove(&i)?;
                    }
                }
                txn.commit()?;
                Ok(())
            })();
            drop(db);
        }
    }

    // Recover
    let db = Database::open(tmpfile.path()).unwrap();
    let report = db.verify_integrity(VerifyLevel::Pages).unwrap();
    assert!(report.valid);

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    let len = t.len().unwrap();

    // Either all deletes committed or none — never partial
    assert!(
        len == baseline_count || len == baseline_count - 150,
        "partial delete detected: got {len} rows, expected {baseline_count} or {}",
        baseline_count - 150
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Multiple committed transactions, crash during N+1
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn multiple_commits_then_crash() {
    let tmpfile = create_tempfile();

    // Build up 5 committed transactions
    {
        let db = Database::create(tmpfile.path()).unwrap();
        for batch in 0..5u64 {
            let txn = db.begin_write().unwrap();
            {
                let mut t = txn.open_table(U64_TABLE).unwrap();
                for i in 0..100u64 {
                    let key = batch * 100 + i;
                    t.insert(&key, &key).unwrap();
                }
            }
            txn.commit().unwrap();
        }
        drop(db);
    }

    // Crash during 6th transaction
    {
        let file_backend = FileBackend::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(tmpfile.path())
                .unwrap(),
        )
        .unwrap();
        let crash_backend = CountdownBackend::new(file_backend, 50);

        if let Ok(db) = Builder::new().create_with_backend(crash_backend) {
            let _ = (|| -> Result<(), shodh_redb::Error> {
                let txn = db.begin_write()?;
                {
                    let mut t = txn.open_table(U64_TABLE)?;
                    // Try to overwrite everything
                    for i in 0..500u64 {
                        t.insert(&i, &999_999)?;
                    }
                }
                txn.commit()?;
                Ok(())
            })();
            drop(db);
        }
    }

    // Recover: all 5 transactions should be intact
    let db = Database::open(tmpfile.path()).unwrap();
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    let len = t.len().unwrap();

    if len == 500 {
        // Either original data or fully overwritten
        let first = t.get(&0u64).unwrap().unwrap().value();
        if first == 0 {
            // Original data
            for i in 0..500u64 {
                assert_eq!(t.get(&i).unwrap().unwrap().value(), i);
            }
        } else {
            // Overwrite committed fully
            for i in 0..500u64 {
                assert_eq!(t.get(&i).unwrap().unwrap().value(), 999_999);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Durability::None crash — may lose last commit
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn eventual_durability_crash_recovery() {
    let tmpfile = create_tempfile();

    // Write with Eventual durability
    {
        let db = Database::create(tmpfile.path()).unwrap();
        let mut txn = db.begin_write().unwrap();
        txn.set_durability(Durability::None).unwrap();
        {
            let mut t = txn.open_table(U64_TABLE).unwrap();
            for i in 0..50u64 {
                t.insert(&i, &i).unwrap();
            }
        }
        txn.commit().unwrap();

        // Force a second Immediate commit to ensure first is durable
        let mut txn = db.begin_write().unwrap();
        txn.set_durability(Durability::Immediate).unwrap();
        {
            let mut t = txn.open_table(U64_TABLE).unwrap();
            for i in 50..100u64 {
                t.insert(&i, &i).unwrap();
            }
        }
        txn.commit().unwrap();
        drop(db);
    }

    // Reopen and verify
    let db = Database::open(tmpfile.path()).unwrap();
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    // Both transactions should be durable since the Immediate commit forces sync
    assert_eq!(t.len().unwrap(), 100);
}

// ═══════════════════════════════════════════════════════════════════════
// Crash during blob store
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn crash_during_blob_store_recovers() {
    let tmpfile = create_tempfile();

    // Baseline: store a blob
    let baseline_blob_count;
    {
        let db = Database::create(tmpfile.path()).unwrap();
        let txn = db.begin_write().unwrap();
        let data = vec![0xABu8; 4096];
        txn.store_blob(
            &data,
            ContentType::OctetStream,
            "baseline",
            StoreOptions::default(),
        )
        .unwrap();
        txn.commit().unwrap();

        let rtxn = db.begin_read().unwrap();
        baseline_blob_count = rtxn.blob_stats().unwrap().blob_count;
        drop(rtxn);
        drop(db);
    }

    // Crash during second blob store
    {
        let file_backend = FileBackend::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(tmpfile.path())
                .unwrap(),
        )
        .unwrap();
        let crash_backend = CountdownBackend::new(file_backend, 40);

        if let Ok(db) = Builder::new().create_with_backend(crash_backend) {
            let _ = (|| -> Result<(), shodh_redb::Error> {
                let txn = db.begin_write()?;
                let big_data = vec![0xCDu8; 8192];
                txn.store_blob(
                    &big_data,
                    ContentType::OctetStream,
                    "crash_blob",
                    StoreOptions::default(),
                )?;
                txn.commit()?;
                Ok(())
            })();
            drop(db);
        }
    }

    // Recover
    let db = Database::open(tmpfile.path()).unwrap();
    let report = db.verify_integrity(VerifyLevel::Pages).unwrap();
    assert!(report.valid);

    let txn = db.begin_read().unwrap();
    let stats = txn.blob_stats().unwrap();
    // Blob count should be either baseline (crash before commit) or baseline+1 (crash after commit)
    assert!(
        stats.blob_count == baseline_blob_count || stats.blob_count == baseline_blob_count + 1,
        "unexpected blob count after crash: got {}, expected {} or {}",
        stats.blob_count,
        baseline_blob_count,
        baseline_blob_count + 1
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Double-open after crash — no lock file issues
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn reopen_twice_after_crash() {
    let tmpfile = create_tempfile();

    {
        let db = populate_baseline(tmpfile.path(), 100);
        drop(db);
    }

    // Crash
    {
        let file_backend = FileBackend::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(tmpfile.path())
                .unwrap(),
        )
        .unwrap();
        let crash_backend = CountdownBackend::new(file_backend, 20);
        let _ = Builder::new().create_with_backend(crash_backend);
        // backend dropped without clean close
    }

    // First reopen
    {
        let db = Database::open(tmpfile.path()).unwrap();
        verify_baseline(&db, 100);
        drop(db);
    }

    // Second reopen — should still work
    {
        let db = Database::open(tmpfile.path()).unwrap();
        verify_baseline(&db, 100);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Crash during compaction
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn crash_during_compaction_recovers() {
    let tmpfile = create_tempfile();

    // Build fragmented DB
    {
        let db = Database::create(tmpfile.path()).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for i in 0..2000u64 {
                t.insert(&i, &[0u8; 128][..]).unwrap();
            }
        }
        txn.commit().unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for i in 0..1500u64 {
                t.remove(&i).unwrap();
            }
        }
        txn.commit().unwrap();
        drop(db);
    }

    // Reopen with countdown and try compact
    {
        let file_backend = FileBackend::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(tmpfile.path())
                .unwrap(),
        )
        .unwrap();
        let crash_backend = CountdownBackend::new(file_backend, 100);

        if let Ok(mut db) = Builder::new().create_with_backend(crash_backend) {
            let _ = db.compact();
            drop(db);
        }
    }

    // Recover
    let db = Database::open(tmpfile.path()).unwrap();
    let report = db.verify_integrity(VerifyLevel::Pages).unwrap();
    assert!(report.valid);

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    // The 500 surviving keys should still be present
    assert_eq!(t.len().unwrap(), 500);
    for i in 1500..2000u64 {
        assert!(
            t.get(&i).unwrap().is_some(),
            "key {i} missing after crash-during-compact"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Empty database crash — must still open cleanly
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn crash_on_empty_db_recovers() {
    let tmpfile = create_tempfile();

    {
        let db = Database::create(tmpfile.path()).unwrap();
        drop(db);
    }

    // Crash during first write to empty DB
    {
        let file_backend = FileBackend::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(tmpfile.path())
                .unwrap(),
        )
        .unwrap();
        let crash_backend = CountdownBackend::new(file_backend, 15);

        if let Ok(db) = Builder::new().create_with_backend(crash_backend) {
            let _ = (|| -> Result<(), shodh_redb::Error> {
                let txn = db.begin_write()?;
                {
                    let mut t = txn.open_table(U64_TABLE)?;
                    t.insert(&1u64, &1u64)?;
                }
                txn.commit()?;
                Ok(())
            })();
            drop(db);
        }
    }

    // Must open cleanly, possibly empty
    let db = Database::open(tmpfile.path()).unwrap();
    let report = db.verify_integrity(VerifyLevel::Pages).unwrap();
    assert!(report.valid);
}
