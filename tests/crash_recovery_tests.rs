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
//! 3. Reopen from the file -- must recover the last committed state
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

// =======================================================================
// CountdownBackend -- deterministic I/O failure injection
// =======================================================================

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

// =======================================================================
// Basic crash recovery: fail during second commit
// =======================================================================

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

// =======================================================================
// Crash at various countdown points
// =======================================================================

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
            // Try writing -- may fail at different points
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

        // Verify recovery -- baseline should still be intact OR the overwrite
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

// =======================================================================
// Crash during insert (before commit) -- data must be discarded
// =======================================================================

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
                // Intentionally do NOT commit -- simulating crash before commit
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

// =======================================================================
// Crash during delete
// =======================================================================

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

    // Either all deletes committed or none -- never partial
    assert!(
        len == baseline_count || len == baseline_count - 150,
        "partial delete detected: got {len} rows, expected {baseline_count} or {}",
        baseline_count - 150
    );
}

// =======================================================================
// Multiple committed transactions, crash during N+1
// =======================================================================

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

// =======================================================================
// Durability::None crash -- may lose last commit
// =======================================================================

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

// =======================================================================
// Crash during blob store
// =======================================================================

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

// =======================================================================
// Double-open after crash -- no lock file issues
// =======================================================================

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

    // Second reopen -- should still work
    {
        let db = Database::open(tmpfile.path()).unwrap();
        verify_baseline(&db, 100);
    }
}

// =======================================================================
// Crash during compaction
// =======================================================================

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

// =======================================================================
// Empty database crash -- must still open cleanly
// =======================================================================

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

/// The layout fields in the database header (page_size, region_header_pages,
/// region_max_data_pages, ...) sit outside both commit slots, so the slot
/// checksums do not protect them. A corrupted value must produce a clean
/// error, not a panic: `Database::open` is the boundary where an untrusted
/// file is first parsed.
#[test]
fn corrupt_region_max_data_pages_does_not_panic() {
    // Offsets mirror the private constants in tree_store::page_store::header.
    const MAGICNUMBER_LEN: usize = 9;
    const GOD_BYTE_OFFSET: usize = MAGICNUMBER_LEN;
    const PAGE_SIZE_OFFSET: usize = GOD_BYTE_OFFSET + 1 + 2;
    const REGION_HEADER_PAGES_OFFSET: usize = PAGE_SIZE_OFFSET + 4;
    const REGION_MAX_DATA_PAGES_OFFSET: usize = REGION_HEADER_PAGES_OFFSET + 4;

    let tmpfile = create_tempfile();
    {
        let db = Database::create(tmpfile.path()).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(U64_TABLE).unwrap();
            t.insert(&1u64, &1u64).unwrap();
        }
        txn.commit().unwrap();
    }

    // Zero the region_max_data_pages field. Nothing checksums it.
    let mut bytes = std::fs::read(tmpfile.path()).unwrap();
    bytes[REGION_MAX_DATA_PAGES_OFFSET..REGION_MAX_DATA_PAGES_OFFSET + 4].fill(0);
    std::fs::write(tmpfile.path(), &bytes).unwrap();

    // Must be reported, not panicked on.
    assert!(
        Database::open(tmpfile.path()).is_err(),
        "opening a database with a zeroed region_max_data_pages must fail cleanly"
    );
}

/// Sibling of the above: a zeroed page_size makes the region size zero, which
/// `DatabaseLayout::recalculate` uses as a divisor.
#[test]
fn corrupt_page_size_does_not_panic() {
    const PAGE_SIZE_OFFSET: usize = 9 + 1 + 2;

    let tmpfile = create_tempfile();
    {
        let db = Database::create(tmpfile.path()).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(U64_TABLE).unwrap();
            t.insert(&1u64, &1u64).unwrap();
        }
        txn.commit().unwrap();
    }

    let mut bytes = std::fs::read(tmpfile.path()).unwrap();
    bytes[PAGE_SIZE_OFFSET..PAGE_SIZE_OFFSET + 4].fill(0);
    std::fs::write(tmpfile.path(), &bytes).unwrap();

    assert!(
        Database::open(tmpfile.path()).is_err(),
        "opening a database with a zeroed page_size must fail cleanly"
    );
}

/// A commit slot whose checksum fails must not be trusted, even when the god
/// byte says no recovery is needed. A crash cannot produce that combination
/// (a write transaction sets recovery_required durably first), so it means the
/// slot bytes rotted. The database must fall back to the intact slot and tell
/// the caller that a committed transaction was discarded.
#[test]
fn corrupt_commit_slot_falls_back_and_reports() {
    use std::sync::atomic::AtomicBool;

    const GOD_BYTE_OFFSET: usize = 9;
    const PRIMARY_BIT: u8 = 1;
    const TRANSACTION_0_OFFSET: usize = 64;
    const TRANSACTION_SIZE: usize = 128;
    const SLOT_CHECKSUM_IN_SLOT: usize = 112;
    const TRANSACTION_ID_IN_SLOT: usize = 104;

    #[derive(Default)]
    struct Rollbacks {
        fired: AtomicBool,
        discarded: AtomicU64,
        recovered: AtomicU64,
    }
    struct Watcher(Arc<Rollbacks>);
    impl shodh_redb::DatabaseObserver for Watcher {
        fn on_commit_slot_rollback(&self, discarded: u64, recovered: u64) {
            self.0.discarded.store(discarded, Ordering::SeqCst);
            self.0.recovered.store(recovered, Ordering::SeqCst);
            self.0.fired.store(true, Ordering::SeqCst);
        }
    }

    let tmpfile = create_tempfile();
    {
        let db = Database::create(tmpfile.path()).unwrap();
        for i in 0..3u64 {
            let txn = db.begin_write().unwrap();
            {
                let mut t = txn.open_table(U64_TABLE).unwrap();
                t.insert(&i, &i).unwrap();
            }
            txn.commit().unwrap();
        }
    }

    let mut bytes = std::fs::read(tmpfile.path()).unwrap();
    let primary = usize::from(bytes[GOD_BYTE_OFFSET] & PRIMARY_BIT != 0);
    let pslot = TRANSACTION_0_OFFSET + primary * TRANSACTION_SIZE;
    let sslot = TRANSACTION_0_OFFSET + (primary ^ 1) * TRANSACTION_SIZE;
    let read_txn_id = |b: &[u8], slot: usize| -> u64 {
        u64::from_le_bytes(
            b[slot + TRANSACTION_ID_IN_SLOT..slot + TRANSACTION_ID_IN_SLOT + 8]
                .try_into()
                .unwrap(),
        )
    };
    let primary_txn = read_txn_id(&bytes, pslot);
    let secondary_txn = read_txn_id(&bytes, sslot);
    assert!(
        primary_txn > secondary_txn,
        "expected the primary slot to hold the newer transaction"
    );

    // Break ONLY the slot's own checksum, so every data field stays valid.
    // This isolates the checksum check from the downstream page checksums.
    bytes[pslot + SLOT_CHECKSUM_IN_SLOT] ^= 0xFF;
    std::fs::write(tmpfile.path(), &bytes).unwrap();

    let seen = Arc::new(Rollbacks::default());
    let db = Builder::new()
        .set_observer(Watcher(seen.clone()))
        .open(tmpfile.path())
        .expect("must fall back to the intact commit slot");

    assert!(
        seen.fired.load(Ordering::SeqCst),
        "falling back past a committed transaction must be reported"
    );
    assert_eq!(seen.discarded.load(Ordering::SeqCst), primary_txn);
    assert_eq!(seen.recovered.load(Ordering::SeqCst), secondary_txn);

    let report = db.verify_integrity(VerifyLevel::Pages).unwrap();
    assert!(report.valid);
}

/// The five layout u32s in the database header sit outside both commit slots,
/// so nothing checksums them. Every value a corrupted field can take must
/// produce a result or a clean error -- never a panic, and never a wrapped
/// length that silently mis-describes the file.
#[test]
fn corrupt_layout_fields_never_panic() {
    const PAGE_SIZE_OFFSET: usize = 9 + 1 + 2;
    let fields = [
        ("page_size", PAGE_SIZE_OFFSET),
        ("region_header_pages", PAGE_SIZE_OFFSET + 4),
        ("region_max_data_pages", PAGE_SIZE_OFFSET + 8),
        ("full_regions", PAGE_SIZE_OFFSET + 12),
        ("trailing_region_data_pages", PAGE_SIZE_OFFSET + 16),
    ];
    let values = [0u32, 1, 2, u32::MAX, 0x8000_0000, 0x7FFF_FFFF];

    for (name, offset) in fields {
        for value in values {
            let tmpfile = create_tempfile();
            {
                let db = Database::create(tmpfile.path()).unwrap();
                let txn = db.begin_write().unwrap();
                {
                    let mut t = txn.open_table(U64_TABLE).unwrap();
                    t.insert(&1u64, &1u64).unwrap();
                }
                txn.commit().unwrap();
            }

            let mut bytes = std::fs::read(tmpfile.path()).unwrap();
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            std::fs::write(tmpfile.path(), &bytes).unwrap();

            // Only the outcome matters: Ok or Err, but no panic and no hang.
            if let Ok(db) = Database::open(tmpfile.path()) {
                // If it opened, the layout it chose must describe a file it can
                // actually walk.
                let report = db.verify_integrity(VerifyLevel::Pages);
                assert!(
                    report.is_ok(),
                    "{name}={value:#x}: opened but could not be verified: {report:?}"
                );
            }
        }
    }
}

/// The slot fallback must compose with `pick_primary_for_repair`, which runs
/// afterwards when recovery_required is also set. That function swaps to the
/// secondary when the primary is corrupt -- so unless repair_info is kept in
/// step with the fallback's swap, it swaps straight back to the damaged slot.
#[test]
fn corrupt_commit_slot_fallback_survives_repair_path() {
    const GOD_BYTE_OFFSET: usize = 9;
    const PRIMARY_BIT: u8 = 1;
    const RECOVERY_REQUIRED: u8 = 2;
    const TRANSACTION_0_OFFSET: usize = 64;
    const TRANSACTION_SIZE: usize = 128;
    const SLOT_CHECKSUM_IN_SLOT: usize = 112;

    let tmpfile = create_tempfile();
    {
        let db = Database::create(tmpfile.path()).unwrap();
        for i in 0..3u64 {
            let txn = db.begin_write().unwrap();
            {
                let mut t = txn.open_table(U64_TABLE).unwrap();
                t.insert(&i, &i).unwrap();
            }
            txn.commit().unwrap();
        }
    }

    let mut bytes = std::fs::read(tmpfile.path()).unwrap();
    let primary = usize::from(bytes[GOD_BYTE_OFFSET] & PRIMARY_BIT != 0);
    let pslot = TRANSACTION_0_OFFSET + primary * TRANSACTION_SIZE;
    // Damage the primary slot AND demand recovery, so both the fallback and
    // the repair path's slot selection run over the same header.
    bytes[pslot + SLOT_CHECKSUM_IN_SLOT] ^= 0xFF;
    bytes[GOD_BYTE_OFFSET] |= RECOVERY_REQUIRED;
    std::fs::write(tmpfile.path(), &bytes).unwrap();

    let db = Database::open(tmpfile.path())
        .expect("must recover using the intact commit slot, not the damaged one");
    let report = db.verify_integrity(VerifyLevel::Pages).unwrap();
    assert!(report.valid, "recovered database must verify: {report:?}");
}

/// `crash_at_various_write_points` samples six countdown values and reuses one
/// file, so each iteration crashes on the previous iteration's wreckage. Worse,
/// a full open+write+commit for that workload consumes fewer backend operations
/// than its smallest sampled value, so it injects no failure at all.
///
/// This sweeps every crash point from a pristine copy, asserting the ACID
/// guarantee directly: after a failure at any single I/O operation, the database
/// must reopen and verify. The workload is sized so the commit spans ~80
/// operations, and the sweep runs past the end of that sequence so both the
/// failing and the clean-commit cases are covered.
#[test]
fn crash_at_every_write_point_recovers() {
    const SWEEP: u64 = 100;
    // Guards against the sweep silently going vacuous if the I/O sequence ever
    // gets shorter than the range: without this, every case would commit
    // cleanly and the test would still pass while injecting nothing.
    const MIN_INJECTED: u64 = 40;

    let baseline = create_tempfile();
    {
        let db = populate_baseline(baseline.path(), 2000);
        drop(db);
    }
    let pristine = std::fs::read(baseline.path()).unwrap();

    let mut injected = 0u64;
    for countdown in 1..=SWEEP {
        let work = create_tempfile();
        std::fs::write(work.path(), &pristine).unwrap();

        {
            let file_backend = FileBackend::new(
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(work.path())
                    .unwrap(),
            )
            .unwrap();
            let crash_backend = CountdownBackend::new(file_backend, countdown);

            match Builder::new().create_with_backend(crash_backend) {
                Ok(db) => {
                    let result = (|| -> Result<(), shodh_redb::Error> {
                        let txn = db.begin_write()?;
                        {
                            let mut t = txn.open_table(TABLE)?;
                            for i in 0..2000u64 {
                                t.insert(&i, &[0xDD; 64][..])?;
                            }
                        }
                        txn.commit()?;
                        Ok(())
                    })();
                    if result.is_err() {
                        injected += 1;
                    }
                    drop(db);
                }
                Err(_) => injected += 1,
            }
        }

        let db = Database::open(work.path())
            .unwrap_or_else(|e| panic!("countdown {countdown}: reopen failed: {e:?}"));
        let report = db
            .verify_integrity(VerifyLevel::Pages)
            .unwrap_or_else(|e| panic!("countdown {countdown}: verify errored: {e:?}"));
        assert!(report.valid, "countdown {countdown}: {report:?}");
    }

    assert!(
        injected >= MIN_INJECTED,
        "only {injected} of {SWEEP} crash points injected a failure; the sweep          no longer covers the commit's I/O sequence"
    );
}

/// A page index must be addressable by `PageNumber`, so `region_max_data_pages`
/// can never legitimately reach `u32::MAX`. It must be rejected rather than
/// merely survived: `BuddyAllocator::new` sizes its bitmaps straight from this
/// value, so an unbounded one reserves roughly a gigabyte before anything
/// validates it. 64-bit absorbs that silently; 32-bit (wasm32/WASI) aborts.
#[test]
fn absurd_region_max_data_pages_is_rejected() {
    const REGION_MAX_DATA_PAGES_OFFSET: usize = 9 + 1 + 2 + 4 + 4;

    let tmpfile = create_tempfile();
    {
        let db = Database::create(tmpfile.path()).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(U64_TABLE).unwrap();
            t.insert(&1u64, &1u64).unwrap();
        }
        txn.commit().unwrap();
    }

    let mut bytes = std::fs::read(tmpfile.path()).unwrap();
    bytes[REGION_MAX_DATA_PAGES_OFFSET..REGION_MAX_DATA_PAGES_OFFSET + 4]
        .copy_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(tmpfile.path(), &bytes).unwrap();

    assert!(
        Database::open(tmpfile.path()).is_err(),
        "an unaddressable region_max_data_pages must be rejected, not allocated for"
    );
}

/// Regression for the first crash the `fuzz_db_image` target found, reduced to
/// a deterministic test so it runs in the normal suite rather than only under
/// the fuzzer (whose corpus is gitignored).
///
/// Zeroing a byte of the system root's `length` in the primary commit slot and
/// resealing that slot's checksum yields a header that parses as consistent
/// while describing a tree that claims to be empty but still holds entries.
/// Deleting through it during open computed `length - 1` and panicked with
/// "attempt to subtract with overflow".
/// Requires the `fuzzing` feature for `slot_checksum`; CI runs
/// `cargo test --all --all-features`, which includes it.
#[test]
#[cfg(feature = "fuzzing")]
fn resealed_slot_with_zero_length_does_not_underflow() {
    const GOD_BYTE_OFFSET: usize = 9;
    const PRIMARY_BIT: u8 = 1;
    const TRANSACTION_0_OFFSET: usize = 64;
    const TRANSACTION_SIZE: usize = 128;
    const SLOT_CHECKSUM_IN_SLOT: usize = 112;
    // system_root BtreeHeader spans 40..72 within a slot: page 40..48,
    // checksum 48..64, length 64..72.
    const SYSTEM_ROOT_LENGTH_IN_SLOT: usize = 64;

    let tmpfile = create_tempfile();
    {
        let db = Database::create(tmpfile.path()).unwrap();
        for i in 0..64u64 {
            let txn = db.begin_write().unwrap();
            {
                let mut t = txn.open_table(U64_TABLE).unwrap();
                t.insert(&i, &i).unwrap();
            }
            txn.commit().unwrap();
        }
    }

    let mut bytes = std::fs::read(tmpfile.path()).unwrap();
    let primary = usize::from(bytes[GOD_BYTE_OFFSET] & PRIMARY_BIT != 0);
    let slot = TRANSACTION_0_OFFSET + primary * TRANSACTION_SIZE;

    // Claim the system tree is empty.
    bytes[slot + SYSTEM_ROOT_LENGTH_IN_SLOT..slot + SYSTEM_ROOT_LENGTH_IN_SLOT + 8].fill(0);
    // Reseal, so the header is internally consistent and the field is trusted.
    let checksum = shodh_redb::fuzzing::slot_checksum(&bytes[slot..slot + SLOT_CHECKSUM_IN_SLOT]);
    bytes[slot + SLOT_CHECKSUM_IN_SLOT..slot + TRANSACTION_SIZE]
        .copy_from_slice(&checksum.to_le_bytes());
    std::fs::write(tmpfile.path(), &bytes).unwrap();

    // Any outcome is fine except a panic.
    if let Ok(db) = Database::open(tmpfile.path()) {
        let _ = db.verify_integrity(VerifyLevel::Pages);
    }
}
