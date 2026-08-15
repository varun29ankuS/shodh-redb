#![cfg(not(target_os = "wasi"))]
//! Regression test for the group-commit leadership leak.
//!
//! `GroupCommitter::finish_leader` deliberately *retains* leadership when it
//! returns work: batches that arrived while the leader was busy are handed
//! back, and `active_leader` stays `true` so no other thread starts leading
//! mid-flight. The caller must therefore keep processing until `finish_leader`
//! returns empty.
//!
//! `Database::run_group_commit` honoured that contract on its success path but
//! not on its `begin_write()` failure path, which did:
//!
//! ```ignore
//! let txn = match self.begin_write() {
//!     Err(e) => {
//!         for b in batches { /* send TransactionFailed */ }
//!         let _ = self.group_committer.finish_leader();   // <-- result discarded
//!         return;
//!     }
//! };
//! ```
//!
//! If any batch arrived between `drain_pending()` and `finish_leader()`, the
//! returned batches were dropped and `active_leader` stayed `true` forever.
//! From that point on every `submit_write_batch` enqueues as a follower and
//! blocks on `recv()` -- its sender is still alive inside `state.pending`, and
//! no thread can ever become leader again. The entire group-commit write path
//! deadlocks permanently.
//!
//! `begin_write()` fails whenever `check_io_errors()` reports a recorded I/O
//! error, so this is reachable after any transient disk failure -- precisely
//! the moment you want graceful degradation rather than a hang.

use std::fmt;
use std::io::ErrorKind;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;
use tempfile::NamedTempFile;

use shodh_redb::{
    BackendError, Builder, Database, StorageBackend, TableDefinition, WriteBatch,
    backends::FileBackend,
};

const TABLE: TableDefinition<u64, u64> = TableDefinition::new("group_commit_leadership");

struct CountdownBackend {
    inner: FileBackend,
    countdown: Arc<AtomicU64>,
}

impl fmt::Debug for CountdownBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountdownBackend").finish()
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
        self.fail_if_tripped()?;
        self.inner.set_len(len)
    }

    fn sync_data(&self) -> Result<(), BackendError> {
        self.fail_if_tripped()?;
        self.inner.sync_data()
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<(), BackendError> {
        self.fail_if_tripped()?;
        self.inner.write(offset, data)
    }
}

impl CountdownBackend {
    fn fail_if_tripped(&self) -> Result<(), BackendError> {
        if self.countdown.load(Ordering::SeqCst) == 0 {
            return Err(std::io::Error::from(ErrorKind::Other).into());
        }
        Ok(())
    }
}

/// After an I/O error has poisoned the database, every `submit_write_batch`
/// must return an error promptly. It must never block forever because the
/// group committer stranded its leadership flag.
#[test]
fn submit_write_batch_does_not_deadlock_after_io_failure() {
    let tmpfile = NamedTempFile::new().unwrap();
    let countdown = Arc::new(AtomicU64::new(u64::MAX));

    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(tmpfile.path())
        .unwrap();
    let backend = CountdownBackend {
        inner: FileBackend::new(file).unwrap(),
        countdown: countdown.clone(),
    };
    let db = Arc::new(Builder::new().create_with_backend(backend).unwrap());

    // Establish a baseline commit while I/O still works.
    {
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            t.insert(&0u64, &0u64).unwrap();
        }
        txn.commit().unwrap();
    }

    // Trip the backend, then force an I/O error to be recorded so that every
    // later `begin_write()` fails in `check_io_errors()`.
    countdown.store(0, Ordering::SeqCst);
    {
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            t.insert(&1u64, &1u64).unwrap();
        }
        let _ = txn.commit(); // expected to fail
    }

    // Sustained concurrent load. The leader's failure path is
    // drain_pending() -> begin_write() (fails on an atomic check) -> send
    // errors -> finish_leader(), so the window in which a batch can arrive and
    // be stranded is sub-microsecond. Overlapping threads submitting
    // continuously is what makes it reachable; round-based spawn/join is not.
    const THREADS: usize = 8;
    const PER_THREAD: usize = 400;
    let done = Arc::new(AtomicU64::new(0));

    for _ in 0..THREADS {
        let db = db.clone();
        let done = done.clone();
        // Detached: if leadership is stranded these threads block forever, and
        // the watchdog below is what reports it. Joining here would hang the
        // test binary instead of failing it.
        thread::spawn(move || {
            for _ in 0..PER_THREAD {
                let _ = db.submit_write_batch(WriteBatch::new(|txn| {
                    let mut t = txn.open_table(TABLE)?;
                    t.insert(&2u64, &2u64)?;
                    Ok(())
                }));
                done.fetch_add(1, Ordering::SeqCst);
            }
        });
    }

    let expected = (THREADS * PER_THREAD) as u64;
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut last = 0u64;
    let mut stalled_for = 0u32;
    while std::time::Instant::now() < deadline {
        let n = done.load(Ordering::SeqCst);
        if n >= expected {
            break;
        }
        if n == last {
            stalled_for += 1;
            // No progress across 3 seconds of polling with work outstanding
            // means every remaining submitter is blocked on recv() waiting for
            // a leader that can never exist.
            assert!(
                stalled_for < 30,
                "group commit deadlocked after {n}/{expected} submissions:                  leadership was never released, so no thread can lead again"
            );
        } else {
            stalled_for = 0;
            last = n;
        }
        thread::sleep(Duration::from_millis(100));
    }

    let completed = done.load(Ordering::SeqCst);
    assert_eq!(
        completed, expected,
        "only {completed}/{expected} submissions returned; the rest are blocked \
         on a group committer that never released leadership"
    );
}

/// Direct check of the protocol contract: `finish_leader` holds leadership
/// while it still has work to hand back, and only releases once it drains
/// empty. A caller that calls it once and returns therefore strands it.
#[test]
fn finish_leader_releases_only_after_draining_empty() {
    let tmpfile = NamedTempFile::new().unwrap();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    // A batch that always succeeds; we only care about scheduling here.
    let submit = |db: Arc<Database>| {
        thread::spawn(move || {
            db.submit_write_batch(WriteBatch::new(|txn| {
                let mut t = txn.open_table(TABLE)?;
                t.insert(&4u64, &4u64)?;
                Ok(())
            }))
        })
    };

    let mut handles = Vec::new();
    for _ in 0..8 {
        handles.push(submit(db.clone()));
    }
    for h in handles {
        let r = h.join().expect("thread panicked");
        assert!(r.is_ok(), "healthy group commit should succeed: {r:?}");
    }

    // Leadership must have been released, so a later batch can lead.
    let r = db.submit_write_batch(WriteBatch::new(|txn| {
        let mut t = txn.open_table(TABLE)?;
        t.insert(&5u64, &5u64)?;
        Ok(())
    }));
    assert!(
        r.is_ok(),
        "group commit stalled after a healthy round: {r:?}"
    );
}
