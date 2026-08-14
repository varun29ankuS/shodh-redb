#![cfg(not(target_os = "wasi"))]
//! Regression test for the `Database::backup` / `process_freed_pages` race.
//!
//! Background:
//!   `Database::backup` copies the **persisted file** byte-for-byte via
//!   `read_raw` -> `CachedFile::read_direct`, which bypasses the page cache.
//!   The file therefore contains only the last durable commit -- the primary
//!   slot, transaction id `D`. But the snapshot was pinned with
//!   `begin_read()`, which registers via `get_last_committed_transaction_id`
//!   and so honors `read_from_secondary`. After a non-durable commit that
//!   returns the secondary slot's id `N`, with `N > D`.
//!
//!   The safety invariant for any reader is `registered_id <= walked_tree_id`,
//!   because a durable commit's `process_freed_pages` frees DATA_FREED_TABLE /
//!   SYSTEM_FREED_TABLE entries for all txns `<= oldest_live_read_transaction`.
//!   With `registered = N` and `walked = D`, the invariant is violated and
//!   entries for `(D, N]` -- persisted pages still reachable from the tree at
//!   `D` -- become eligible for reclamation.
//!
//!   It takes two durable commits inside the copy window. The first is
//!   protected because `register_non_durable_commit` also holds a read hold on
//!   the durable ancestor `D`; that hold is only released by
//!   `clear_pending_non_durable_commits()`, which runs inside `durable_commit`
//!   after `mem.commit()`. The second durable commit then sees the backup's
//!   hold at `N` as the oldest, computes `free_until = N + 1`, and frees the
//!   `(D, N]` entries. The buddy allocator hands those pages straight back out,
//!   the commit fsyncs new content into the same file offsets, and the
//!   still-running copy loop reads the new bytes.
//!
//!   Result: `backup()` returns `Ok(())` and produces a file whose header names
//!   durable txn `D` but whose interior pages are a mix of `D` and a later
//!   commit -- interior B-tree nodes pointing at unrelated data. The failure is
//!   silent until someone tries to restore.
//!
//! This test exercises that exact pattern: a writer alternates non-durable and
//! durable commits with churn (insert + delete) so DATA_FREED_TABLE accumulates
//! entries spanning `(D, N]`, while a backup thread repeatedly copies the
//! database and verifies each copy with `verify_backup(VerifyLevel::Full)`.
//! Without the fix the copied file fails checksum or structural verification;
//! with the fix every backup must verify clean.

use shodh_redb::{Database, Durability, TableDefinition, VerifyLevel};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("backup_concurrent");

fn create_tempfile() -> tempfile::NamedTempFile {
    tempfile::NamedTempFile::new().unwrap()
}

#[test]
fn backup_stays_consistent_under_mixed_durability_commits() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    // Seed enough data that a full-file copy takes long enough for several
    // commits to land inside the copy window.
    {
        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(TABLE).unwrap();
            let value = vec![0xABu8; 1024];
            for i in 0..8_000u64 {
                table.insert(i, value.as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    let stop = Arc::new(AtomicBool::new(false));

    // Writer thread: alternate non-durable and durable commits with churn.
    // Each iteration:
    //   - a non-durable commit overwriting a band of keys, which sets
    //     `read_from_secondary = true` and advances last_committed to `N`
    //     while the primary slot stays at `D`,
    //   - a non-durable commit deleting part of that band, populating
    //     DATA_FREED_TABLE with entries for txns in `(D, N]` that list
    //     persisted pages reachable from the tree at `D`,
    //   - two durable commits, which is what it takes to drop the
    //     durable-ancestor hold and then free the `(D, N]` entries.
    let writer_db = db.clone();
    let writer_stop = stop.clone();
    let writer = thread::spawn(move || {
        let mut iter: u64 = 0;
        let large = vec![0xCDu8; 1024];
        while !writer_stop.load(Ordering::Relaxed) {
            let band = (iter % 16) * 256;

            // Non-durable overwrite -- keeps read_from_secondary set.
            {
                let mut txn = writer_db.begin_write().unwrap();
                txn.set_durability(Durability::None).unwrap();
                {
                    let mut table = txn.open_table(TABLE).unwrap();
                    for k in band..band + 256 {
                        table.insert(k, large.as_slice()).unwrap();
                    }
                }
                txn.commit().unwrap();
            }

            // Non-durable delete -- creates DATA_FREED_TABLE entries listing
            // persisted pages that are still reachable from the primary root.
            {
                let mut txn = writer_db.begin_write().unwrap();
                txn.set_durability(Durability::None).unwrap();
                {
                    let mut table = txn.open_table(TABLE).unwrap();
                    for k in band..band + 128 {
                        table.remove(&k).unwrap();
                    }
                }
                txn.commit().unwrap();
            }

            // Two durable commits: the first clears the pending non-durable
            // ancestor holds, the second frees the (D, N] entries.
            for step in 0..2u64 {
                let other_band = ((iter + 7 + step) % 16) * 256;
                let txn = writer_db.begin_write().unwrap();
                {
                    let mut table = txn.open_table(TABLE).unwrap();
                    for k in other_band..other_band + 64 {
                        table.insert(k, large.as_slice()).unwrap();
                    }
                }
                txn.commit().unwrap();
            }

            iter += 1;
        }
    });

    // Backup thread: copy the database, then verify the copy. Without the fix
    // the copy contains pages reclaimed and rewritten mid-copy, so full
    // verification reports corrupt pages or a broken structure.
    let backup_db = db.clone();
    let backup_stop = stop.clone();
    let backup = thread::spawn(move || -> Result<(), String> {
        let start = Instant::now();
        let mut runs: u64 = 0;
        while runs < 20 && start.elapsed() < Duration::from_secs(60) {
            let dest = create_tempfile();
            backup_db
                .backup(dest.path())
                .map_err(|e| format!("backup returned error: {e:?}"))?;

            let report = Database::verify_backup(dest.path(), VerifyLevel::Full)
                .map_err(|e| format!("verify_backup returned error on run {runs}: {e:?}"))?;
            if !report.valid {
                return Err(format!(
                    "backup produced a corrupt file on run {runs}: {} of {} pages corrupt, \
                     header_valid={}, structural_valid={:?}, details={:?}",
                    report.pages_corrupt,
                    report.pages_checked,
                    report.header_valid,
                    report.structural_valid,
                    report.corrupt_details
                ));
            }
            runs += 1;
        }
        backup_stop.store(true, Ordering::Relaxed);
        Ok(())
    });

    let backup_result = backup.join().expect("backup thread panicked");
    stop.store(true, Ordering::Relaxed);
    writer.join().expect("writer thread panicked");

    backup_result.expect("backup must produce a verifiable file under concurrent commits");
}
