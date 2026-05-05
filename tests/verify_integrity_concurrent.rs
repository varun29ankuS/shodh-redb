//! Regression test for the verify_integrity / process_freed_pages race.
//!
//! Background:
//!   `Database::verify_integrity` reads the **persisted** roots
//!   (`get_persisted_data_root` / `get_persisted_system_root`, which always
//!   come from the primary slot) but registered its read hold via
//!   `register_read_transaction`, which uses `get_last_committed_transaction_id`.
//!   After a non-durable commit `read_from_secondary = true`, so the
//!   committed-id read returns the secondary slot's id `N` while the persisted
//!   root still reflects an older durable id `N-K`. A concurrent durable
//!   commit's `process_freed_pages` then frees DATA_FREED_TABLE /
//!   SYSTEM_FREED_TABLE entries for txns `N-K+1 ..= N` -- pages still
//!   reachable from the primary root being walked. The verification walk
//!   follows pointers into reclaimed pages and panics inside
//!   `EntryGuard::value_checked` (typically with a reversed `value_range`
//!   such as `3226..0` when the page has been reused mid-write).
//!
//! This test exercises that exact pattern: a writer alternates durable and
//! non-durable commits with churn (insert + delete) so DATA_FREED_TABLE and
//! SYSTEM_FREED_TABLE accumulate entries across both durability modes,
//! while a verifier loops on `verify_integrity(VerifyLevel::Full)`.
//! Without the fix this reproduces the panic in seconds; with the fix it
//! must run clean and never report corruption.

use shodh_redb::{
    Database, Durability, ReadableDatabase, ReadableTable, TableDefinition, VerifyLevel,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("verify_concurrent");

fn create_tempfile() -> tempfile::NamedTempFile {
    tempfile::NamedTempFile::new().unwrap()
}

#[test]
fn verify_integrity_races_with_mixed_durability_commits() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    // Seed the database so verification has structure to walk.
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

    let stop = Arc::new(AtomicBool::new(false));

    // Writer thread: alternate durable and non-durable commits with churn.
    // Each iteration:
    //   - one non-durable commit that inserts/overwrites a band of keys,
    //   - one non-durable commit that deletes part of that band,
    //   - one durable commit that overwrites a different band.
    // The deletes populate DATA_FREED_TABLE entries; the durable commit
    // triggers `process_freed_pages`; the non-durable commits keep
    // `read_from_secondary = true` so the verifier reads stale primary
    // roots while last_committed_id advances on the secondary slot.
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

    // Verifier thread: loop on verify_integrity(Full). Without the fix this
    // panics inside `EntryGuard::value_checked` with a reversed value_range
    // when a reachable page is reclaimed and reused mid-write.
    let verifier_db = db.clone();
    let verifier_stop = stop.clone();
    let verifier = thread::spawn(move || -> Result<(), String> {
        let start = Instant::now();
        let mut runs: u64 = 0;
        // Cap by iterations and wall-clock to keep CI cheap but still cover
        // the race window many times. The bug reproduces in well under
        // 1 second of wall time without the fix.
        while runs < 200 && start.elapsed() < Duration::from_secs(15) {
            let report = verifier_db
                .verify_integrity(VerifyLevel::Full)
                .map_err(|e| format!("verify_integrity returned error: {e:?}"))?;
            if !report.valid {
                return Err(format!(
                    "verify_integrity reported corruption: {} pages corrupt, structural_valid={:?}, details={:?}",
                    report.pages_corrupt, report.structural_valid, report.corrupt_details
                ));
            }
            runs += 1;
        }
        verifier_stop.store(true, Ordering::Relaxed);
        Ok(())
    });

    let verifier_result = verifier.join().expect("verifier thread panicked");
    stop.store(true, Ordering::Relaxed);
    writer.join().expect("writer thread panicked");

    verifier_result.expect("verify_integrity must remain valid under concurrent commits");
}
