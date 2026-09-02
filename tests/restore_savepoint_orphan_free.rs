//! Regression tests for the savepoint-restore orphan-free bug behind the
//! fuzz_redb len divergence (crash-f14b73608b1d426ffed4f4b2c53de2b374829ab0).
//!
//! Mechanism (established by instrumented replay of the reproducer):
//!
//! `WriteTransaction::restore_savepoint` snapshots `mem.drain_uncommitted()`
//! (every page allocated since the last commit) as "orphaned" and frees all
//! of them. That set is only safe to free when the transaction did nothing
//! but user-table writes before the restore. If `persistent_savepoint()` ran
//! earlier in the SAME transaction, the pages it allocated for live system
//! tree content (e.g. the new `next_savepoint_id` leaf) are in that set too.
//! Freeing them cancels their pending cache writes while the system tree
//! still references them, so the following durable commit persists a tree
//! pointing at a page whose bytes were never written to disk.
//!
//! Nothing notices in-process. But if the process crashes before a clean
//! shutdown, reopen runs repair, `verify_primary_checksums` fails on the
//! primary slot (leaf checksum mismatch on the never-written page), and
//! `do_repair` rolls back to the secondary slot -- silently resurrecting the
//! durable state from BEFORE the savepoint restore. That is how the fuzz
//! table ended up with an entry the reference model had already seen removed
//! by a durable, successfully committed restore.

#![cfg(not(target_os = "wasi"))]

use std::fmt;
use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tempfile::NamedTempFile;

use shodh_redb::{
    BackendError, Builder, Database, ReadableDatabase, ReadableTableMetadata, StorageBackend,
    TableDefinition, TableError, backends::FileBackend,
};

const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("restore_orphan_test");

// =======================================================================
// CountdownBackend -- deterministic I/O failure injection (same pattern as
// tests/crash_recovery_tests.rs). When the countdown hits zero every write,
// set_len, and sync fails, so dropping the Database cannot perform a clean
// shutdown and the on-disk recovery flag stays set. Reads keep working so
// nothing else is perturbed.
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

    fn decrement(&self) -> Result<(), BackendError> {
        if self
            .countdown
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| x.checked_sub(1))
            .is_err()
        {
            return Err(std::io::Error::from(ErrorKind::Other).into());
        }
        Ok(())
    }
}

impl StorageBackend for CountdownBackend {
    fn len(&self) -> Result<u64, BackendError> {
        self.inner.len()
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), BackendError> {
        self.inner.read(offset, out)
    }

    fn set_len(&self, len: u64) -> Result<(), BackendError> {
        self.decrement()?;
        self.inner.set_len(len)
    }

    fn sync_data(&self) -> Result<(), BackendError> {
        self.decrement()?;
        Ok(())
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<(), BackendError> {
        self.decrement()?;
        self.inner.write(offset, data)
    }
}

fn open_dup(file: &NamedTempFile) -> std::fs::File {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(file.path())
        .unwrap()
}

fn create_db(file: &NamedTempFile) -> (Database, Arc<AtomicU64>) {
    let backend = CountdownBackend::new(FileBackend::new(open_dup(file)).unwrap(), u64::MAX);
    let countdown = backend.countdown.clone();
    let db = Builder::new().create_with_backend(backend).unwrap();
    (db, countdown)
}

/// After crash recovery, the table must reflect the durably committed
/// restored state: the savepoint predates the table, so the table must be
/// absent (or at minimum must not contain the pre-restore entry).
fn assert_restored_state_survived(db: &Database) {
    let read = db.begin_read().unwrap();
    match read.open_table(TABLE) {
        Err(TableError::TableDoesNotExist(_)) => {
            // Correct: the restored savepoint predates the table.
        }
        Ok(table) => {
            // If the table somehow exists, it must at least be empty. A
            // resurrected pre-restore entry means the durable restore commit
            // was rolled back by repair.
            assert!(
                table.get(&1).unwrap().is_none(),
                "durable savepoint-restore commit was lost: pre-restore entry resurrected \
                 after crash recovery"
            );
            assert_eq!(
                table.len().unwrap(),
                0,
                "durable savepoint-restore commit was lost: pre-restore entries resurrected \
                 after crash recovery"
            );
        }
        Err(err) => panic!("unexpected error opening table: {err}"),
    }
}

/// The bug: `persistent_savepoint()` followed by `restore_savepoint()` in the
/// same transaction. The restore's orphan collection frees the system-tree
/// pages the savepoint creation just allocated, the durable commit persists a
/// system tree referencing a never-written page, and crash recovery rolls the
/// database back to the pre-restore slot.
#[test]
fn restore_after_persistent_savepoint_in_same_txn_survives_crash() {
    let file = NamedTempFile::new().unwrap();
    let (db, countdown) = create_db(&file);

    // Transaction 1: capture a persistent savepoint of the empty database.
    let txn = db.begin_write().unwrap();
    let sp_old = txn.persistent_savepoint().unwrap();
    txn.commit().unwrap();

    // Transaction 2: durably commit an entry the restore must undo. Several
    // commits with some bulk grow the file so the post-restore layout matches
    // the fuzz reproducer's shape (repair succeeds and silently rolls back,
    // instead of failing outright on the allocator rebuild).
    let txn = db.begin_write().unwrap();
    {
        let mut table = txn.open_table(TABLE).unwrap();
        table.insert(&1, b"payload".as_slice()).unwrap();
    }
    txn.commit().unwrap();
    for batch in 0u64..3 {
        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(TABLE).unwrap();
            for i in 0..50 {
                let key = 1000 + batch * 100 + i;
                table.insert(&key, vec![0xAB; 512].as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    // Transaction 3: create a persistent savepoint, then restore the old one,
    // in the SAME transaction. Then commit durably.
    let mut txn = db.begin_write().unwrap();
    let _sp_new = txn.persistent_savepoint().unwrap();
    let savepoint = txn.get_persistent_savepoint(sp_old).unwrap();
    txn.restore_savepoint(&savepoint).unwrap();
    drop(savepoint);
    txn.commit().unwrap();

    // The restore is durably committed: the table must be gone right now.
    let read = db.begin_read().unwrap();
    assert!(matches!(
        read.open_table(TABLE),
        Err(TableError::TableDoesNotExist(_))
    ));
    drop(read);

    // Simulate a crash: no further bytes reach the file, so dropping the
    // database cannot clean up and the recovery flag stays set on disk.
    countdown.store(0, Ordering::SeqCst);
    drop(db);

    // Reopen. Repair runs because the recovery flag is set. It must keep the
    // durably committed restore, not roll back to the pre-restore slot.
    let backend = CountdownBackend::new(FileBackend::new(open_dup(&file)).unwrap(), u64::MAX);
    let db = Builder::new().create_with_backend(backend).unwrap();
    assert_restored_state_survived(&db);
}

/// Control: the same sequence WITHOUT the same-transaction persistent
/// savepoint creation. This passes on current code and pins down that the
/// crash simulation itself is sound -- the resurrection above is caused
/// specifically by the persistent_savepoint + restore_savepoint combination.
#[test]
fn restore_alone_survives_crash() {
    let file = NamedTempFile::new().unwrap();
    let (db, countdown) = create_db(&file);

    let txn = db.begin_write().unwrap();
    let sp_old = txn.persistent_savepoint().unwrap();
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut table = txn.open_table(TABLE).unwrap();
        table.insert(&1, b"payload".as_slice()).unwrap();
    }
    txn.commit().unwrap();

    let mut txn = db.begin_write().unwrap();
    let savepoint = txn.get_persistent_savepoint(sp_old).unwrap();
    txn.restore_savepoint(&savepoint).unwrap();
    drop(savepoint);
    txn.commit().unwrap();

    countdown.store(0, Ordering::SeqCst);
    drop(db);

    let backend = CountdownBackend::new(FileBackend::new(open_dup(&file)).unwrap(), u64::MAX);
    let db = Builder::new().create_with_backend(backend).unwrap();
    assert_restored_state_survived(&db);
}
