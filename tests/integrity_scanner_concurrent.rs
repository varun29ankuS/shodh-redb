#![cfg(not(target_os = "wasi"))]
//! Regression test for the integrity scanner / `process_freed_pages` race.
//!
//! Background:
//!   `IntegrityScannerHandle` walks the **persisted** primary roots
//!   (`get_persisted_data_root` / `get_persisted_system_root`) on a background
//!   thread. Before the fix the scanner used `TransactionGuard::Verification`,
//!   a no-op guard that performed no `register_*` call. As a result
//!   `oldest_live_read_transaction` was unaffected by the scanner, and a
//!   concurrent durable commit's `process_freed_pages` was free to reclaim
//!   DATA_FREED_TABLE / SYSTEM_FREED_TABLE entries whose pages were still
//!   reachable from the primary tree being walked. Reused-mid-write pages
//!   produced reversed `value_range` panics inside `EntryGuard::value_checked`.
//!
//! This test exercises the same id/root mismatch class as
//! `verify_integrity_concurrent.rs`, but for the background scanner: a writer
//! alternates durable and non-durable commits with churn while the scanner
//! loops. Without the fix the scanner thread panics within seconds; with the
//! fix it must report no corruption for the entire run.

use shodh_redb::{
    Database, Durability, IntegrityScannerConfig, TableDefinition,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("scanner_concurrent");

fn create_tempfile() -> tempfile::NamedTempFile {
    tempfile::NamedTempFile::new().unwrap()
}

#[test]
fn integrity_scanner_races_with_mixed_durability_commits() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    // Seed the database so the scanner has structure to walk.
    {
        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(TABLE).unwrap();
            for i in 0..2_000u64 {
                let value = vec![0xABu8; 256];
                table.insert(i, value.as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    let corruption_detected = Arc::new(AtomicBool::new(false));
    let cycles_completed = Arc::new(AtomicU64::new(0));

    // Start the scanner with a short interval so it loops aggressively.
    let corruption_flag = corruption_detected.clone();
    let cycles_flag = cycles_completed.clone();
    let mut handle = db
        .start_integrity_scanner(IntegrityScannerConfig {
            scan_interval_secs: 0,
            on_cycle_complete: Some(Box::new(move |result| {
                if result.pages_corrupt > 0 {
                    corruption_flag.store(true, Ordering::Relaxed);
                }
                cycles_flag.fetch_add(1, Ordering::Relaxed);
            })),
        })
        .unwrap();

    let stop = Arc::new(AtomicBool::new(false));

    // Writer thread mirrors the verify_integrity_concurrent regression: a mix
    // of non-durable and durable commits with churn so DATA_FREED_TABLE and
    // SYSTEM_FREED_TABLE accumulate entries across both durability modes,
    // then a durable commit drives `process_freed_pages` while the scanner
    // is in the middle of walking the primary roots.
    let writer_db = db.clone();
    let writer_stop = stop.clone();
    let writer = thread::spawn(move || {
        let mut iter: u64 = 0;
        let large = vec![0xCDu8; 1024];
        while !writer_stop.load(Ordering::Relaxed) {
            let band = (iter % 16) * 256;

            // Non-durable insert/overwrite.
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

            // Non-durable delete (creates DATA_FREED_TABLE entries).
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

            // Durable commit on a different band -- triggers process_freed_pages
            // for accumulated DATA_FREED_TABLE entries.
            {
                let other_band = ((iter + 7) % 16) * 256;
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

    // Run for up to 15 seconds or until the scanner has completed enough
    // cycles to cover the race window many times. The bug reproduces in
    // well under one second of wall time without the fix.
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(15)
        && cycles_completed.load(Ordering::Relaxed) < 50
    {
        thread::sleep(Duration::from_millis(50));
    }

    stop.store(true, Ordering::Relaxed);
    writer.join().expect("writer thread panicked");

    // Drop the scanner handle to ensure the scanner thread is joined. If the
    // scanner thread panicked from a reversed value_range mid-walk, the
    // shutdown join will surface it via thread::join error -- but the panic
    // typically aborts the test binary first, so the surviving check is the
    // corruption flag.
    handle.shutdown();

    assert!(
        cycles_completed.load(Ordering::Relaxed) > 0,
        "scanner must complete at least one cycle"
    );
    assert!(
        !corruption_detected.load(Ordering::Relaxed),
        "integrity scanner must not report corruption under concurrent commits"
    );
}
