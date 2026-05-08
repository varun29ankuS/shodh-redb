#![cfg(not(target_os = "wasi"))]
//! Regression test for the `begin_read_at` / retention-pruning race.
//!
//! Background:
//!   `Database::begin_read_at(T_history)` and `Database::begin_read_at_time`
//!   used to register the read guard via `allocate_read_transaction`, which
//!   pegs the hold to `last_committed_transaction_id` (T_guard) -- typically
//!   far newer than `T_history`. The historical reader then walks the
//!   snapshot's `user_root` (a tree-at-T_history) but only
//!   `history_hold[T_history]` registered at commit time pins those pages.
//!   Once retention pruning evicts T_history from the system history table
//!   it calls `deallocate_history_hold(T_history)`, the entry drops to zero,
//!   `oldest_live_read_transaction` advances past T_history, and a durable
//!   commit's `process_freed_pages` reclaims `DATA_FREED_TABLE` /
//!   `SYSTEM_FREED_TABLE` entries whose pages are still reachable from the
//!   reader's historical `user_root`. The reader then follows pointers into
//!   reused-mid-write pages and panics inside `EntryGuard::value_checked`
//!   with a reversed `value_range` -- the same id/root mismatch class as
//!   `verify_integrity` (#320) and the integrity scanner (#321).
//!
//! This test:
//!   1. opens a database with small `set_history_retention`
//!   2. captures `T_history` from an early commit
//!   3. opens a `begin_read_at(T_history)` reader and keeps it alive
//!   4. concurrently drives durable commits with churn that exceed the
//!      retention window, so retention pruning evicts `T_history` and
//!      `process_freed_pages` runs against the freed entries
//!   5. continually walks the reader's tables -- without the fix this
//!      panics within seconds; with the fix all reads remain valid

use shodh_redb::{Database, TableDefinition};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("history_eviction");

fn create_tempfile() -> tempfile::NamedTempFile {
    tempfile::NamedTempFile::new().unwrap()
}

#[test]
fn begin_read_at_survives_retention_eviction_and_durable_freeing() {
    let tmpfile = create_tempfile();
    // Small retention window so we can quickly prune past T_history.
    let db = Arc::new(
        Database::builder()
            .set_history_retention(3)
            .create(tmpfile.path())
            .unwrap(),
    );

    // Seed some keys so the historical user tree has structure to walk.
    {
        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(TABLE).unwrap();
            let value = vec![0xABu8; 256];
            for k in 0..2_000u64 {
                table.insert(k, value.as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    // Capture T_history (the snapshot we will read from).
    let history = db.transaction_history().unwrap();
    assert!(
        !history.is_empty(),
        "history_retention > 0 must record snapshots"
    );
    let t_history = history.last().unwrap().transaction_id;

    // Open the historical read transaction; this is the reader whose
    // `user_root` walk must remain safe across retention eviction.
    let historical_rtxn = db.begin_read_at(t_history).unwrap();

    let stop = Arc::new(AtomicBool::new(false));

    // Writer: hammer durable commits with insert + delete churn so
    // DATA_FREED_TABLE entries accumulate AND process_freed_pages runs
    // every commit. Many commits in a row also blow past the retention
    // window of 3, forcing pruning of T_history.
    let writer_db = db.clone();
    let writer_stop = stop.clone();
    let writer = thread::spawn(move || {
        let mut iter: u64 = 0;
        let large = vec![0xCDu8; 1024];
        while !writer_stop.load(Ordering::Relaxed) {
            let band = (iter % 16) * 256;

            // Durable insert/overwrite -- creates new history snapshot, may
            // prune the oldest one.
            {
                let txn = writer_db.begin_write().unwrap();
                {
                    let mut table = txn.open_table(TABLE).unwrap();
                    for k in band..band + 256 {
                        table.insert(k, large.as_slice()).unwrap();
                    }
                }
                txn.commit().unwrap();
            }

            // Durable delete -- DATA_FREED_TABLE entries that
            // process_freed_pages will reclaim on the next durable commit.
            {
                let txn = writer_db.begin_write().unwrap();
                {
                    let mut table = txn.open_table(TABLE).unwrap();
                    for k in band..band + 128 {
                        table.remove(&k).unwrap();
                    }
                }
                txn.commit().unwrap();
            }

            iter += 1;
        }
    });

    // Reader: walk the historical user tree continuously. Without the fix
    // this panics inside EntryGuard::value_checked once retention prunes
    // T_history and durable process_freed_pages reclaims a page reachable
    // from the snapshot's user_root.
    let start = Instant::now();
    let mut reads: u64 = 0;
    while reads < 200 && start.elapsed() < Duration::from_secs(15) {
        let table = historical_rtxn.open_table(TABLE).unwrap();
        // Walk a range covering all originally-seeded keys; the historical
        // snapshot must still see exactly the seed values.
        let mut count = 0u64;
        for entry in table.range::<u64>(..).unwrap() {
            let (key_guard, val_guard) = entry.unwrap();
            assert_eq!(val_guard.value().len(), 256, "historical value length");
            assert_eq!(val_guard.value()[0], 0xAB, "historical value byte");
            let _ = key_guard.value();
            count += 1;
        }
        assert_eq!(count, 2_000, "historical snapshot must see all seed rows");
        reads += 1;
    }

    stop.store(true, Ordering::Relaxed);
    writer.join().expect("writer thread panicked");

    // Confirm retention pruning actually evicted T_history during the
    // run. If it didn't, the test failed to exercise the race window.
    let post_history = db.transaction_history().unwrap();
    assert!(
        !post_history
            .iter()
            .any(|info| info.transaction_id == t_history),
        "test must drive retention pruning past T_history (history now: {:?})",
        post_history
            .iter()
            .map(|i| i.transaction_id)
            .collect::<Vec<_>>()
    );

    drop(historical_rtxn);
}
