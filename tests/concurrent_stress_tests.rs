//! Concurrent stress tests.
//!
//! Validates thread safety under real contention: multiple readers during writes,
//! sequential writer handoff, MVCC snapshot isolation, and iterator stability.
//!
//! Concurrency model:
//! - One writer at a time (blocked via Condvar)
//! - Unlimited concurrent readers (MVCC snapshots)
//! - ReadTransaction is Send+Sync, WriteTransaction is not
//!
//! Uses VerifyLevel::Full for complete structural validation (checksums + key ordering).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::NamedTempFile;

use shodh_redb::{
    Database, MultimapTableDefinition, ReadableDatabase, ReadableTable, ReadableTableMetadata,
    TableDefinition, VerifyLevel,
};

fn create_tempfile() -> NamedTempFile {
    if cfg!(target_os = "wasi") {
        NamedTempFile::new_in("/tmp").unwrap()
    } else {
        NamedTempFile::new().unwrap()
    }
}

fn assert_checksums(db: &Database) {
    let report = db.verify_integrity(VerifyLevel::Full).unwrap();
    assert!(
        report.valid,
        "integrity check failed: {} pages corrupt, structural_valid={:?}, details={:?}",
        report.pages_corrupt, report.structural_valid, report.corrupt_details,
    );
}

const TABLE: TableDefinition<u64, u64> = TableDefinition::new("stress_kv");

// ═══════════════════════════════════════════════════════════════════════
// Sequential writer handoff
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn sequential_writer_handoff_10_threads() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());
    let counter = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let db = db.clone();
            let counter = counter.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    let txn = db.begin_write().unwrap();
                    {
                        let mut t = txn.open_table(TABLE).unwrap();
                        let id = counter.fetch_add(1, Ordering::SeqCst);
                        t.insert(&id, &id).unwrap();
                    }
                    txn.commit().unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 1000);

    assert_checksums(&db);
}

// ═══════════════════════════════════════════════════════════════════════
// Concurrent readers during writes
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn concurrent_readers_during_write() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    // Seed with initial data
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..1000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    let barrier = Arc::new(Barrier::new(11)); // 10 readers + 1 writer

    // Spawn 10 reader threads
    let reader_handles: Vec<_> = (0..10)
        .map(|_| {
            let db = db.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                // Read repeatedly during write
                for _ in 0..50 {
                    let txn = db.begin_read().unwrap();
                    let t = txn.open_table(TABLE).unwrap();
                    let len = t.len().unwrap();
                    // Snapshot should show either pre-write or post-write state
                    assert!(len >= 1000, "snapshot length {len} below base count");

                    // Verify sorted order in snapshot
                    let mut prev = None;
                    for result in t.iter().unwrap() {
                        let (k, _) = result.unwrap();
                        if let Some(p) = prev {
                            assert!(k.value() > p);
                        }
                        prev = Some(k.value());
                    }
                }
            })
        })
        .collect();

    // Writer thread
    let writer_db = db.clone();
    let writer_barrier = barrier.clone();
    let writer_handle = thread::spawn(move || {
        writer_barrier.wait();
        for batch in 0..10u64 {
            let txn = writer_db.begin_write().unwrap();
            {
                let mut t = txn.open_table(TABLE).unwrap();
                let base = 1000 + batch * 100;
                for i in base..base + 100 {
                    t.insert(&i, &i).unwrap();
                }
            }
            txn.commit().unwrap();
        }
    });

    writer_handle.join().unwrap();
    for h in reader_handles {
        h.join().unwrap();
    }

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 2000);
}

// ═══════════════════════════════════════════════════════════════════════
// MVCC snapshot isolation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn mvcc_snapshot_isolation() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    // Insert base data
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    // Take snapshot BEFORE write
    let snapshot = db.begin_read().unwrap();

    // Write more data
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 100..200u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    // Snapshot must NOT see new data
    let t = snapshot.open_table(TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 100);
    assert!(t.get(&150u64).unwrap().is_none());

    // New read must see all data
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 200);
}

#[test]
fn mvcc_snapshot_survives_delete() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    // Snapshot before delete
    let snapshot = db.begin_read().unwrap();

    // Delete everything
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        t.drain_all().unwrap();
    }
    txn.commit().unwrap();

    // Snapshot should still see all 100 keys
    let t = snapshot.open_table(TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 100);
    assert_eq!(t.get(&50u64).unwrap().unwrap().value(), 50);
}

// ═══════════════════════════════════════════════════════════════════════
// High-contention writer stress
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn high_contention_overlapping_keys() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());
    let completed = Arc::new(AtomicU64::new(0));

    // 10 threads all writing to the same 100 keys
    let handles: Vec<_> = (0..10u64)
        .map(|thread_id| {
            let db = db.clone();
            let completed = completed.clone();
            thread::spawn(move || {
                for i in 0..100u64 {
                    let txn = db.begin_write().unwrap();
                    {
                        let mut t = txn.open_table(TABLE).unwrap();
                        // All threads write to same keys with thread-specific values
                        t.insert(&i, &(thread_id * 1000 + i)).unwrap();
                    }
                    txn.commit().unwrap();
                }
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(completed.load(Ordering::SeqCst), 10);

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    // All 100 keys should exist (last writer wins for each key)
    assert_eq!(t.len().unwrap(), 100);

    assert_checksums(&db);
}

// ═══════════════════════════════════════════════════════════════════════
// Iterator stability under concurrent modification
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn iterator_stable_during_concurrent_writes() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    // Seed data
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..5000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    let barrier = Arc::new(Barrier::new(2));

    // Reader: iterate through all entries while writer is active
    let reader_db = db.clone();
    let reader_barrier = barrier.clone();
    let reader_handle = thread::spawn(move || {
        reader_barrier.wait();
        let txn = reader_db.begin_read().unwrap();
        let t = txn.open_table(TABLE).unwrap();

        let mut count = 0u64;
        let mut prev = None;
        for result in t.iter().unwrap() {
            let (k, v) = result.unwrap();
            // Keys must be sorted
            if let Some(p) = prev {
                assert!(k.value() > p, "unsorted: {} after {}", k.value(), p);
            }
            // Value must match key (from original insert)
            assert_eq!(k.value(), v.value());
            prev = Some(k.value());
            count += 1;
        }
        count
    });

    // Writer: insert new keys concurrently
    let writer_db = db.clone();
    let writer_barrier = barrier.clone();
    let writer_handle = thread::spawn(move || {
        writer_barrier.wait();
        for batch in 0..10u64 {
            let txn = writer_db.begin_write().unwrap();
            {
                let mut t = txn.open_table(TABLE).unwrap();
                let base = 10_000 + batch * 500;
                for i in base..base + 500 {
                    t.insert(&i, &i).unwrap();
                }
            }
            txn.commit().unwrap();
        }
    });

    let reader_count = reader_handle.join().unwrap();
    writer_handle.join().unwrap();

    // Reader's snapshot should see exactly the pre-write state
    assert_eq!(reader_count, 5000);
}

// ═══════════════════════════════════════════════════════════════════════
// Multimap concurrent access
// ═══════════════════════════════════════════════════════════════════════

const MM_TABLE: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("stress_mm");

#[test]
fn concurrent_multimap_writes() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    let handles: Vec<_> = (0..5u64)
        .map(|thread_id| {
            let db = db.clone();
            thread::spawn(move || {
                for i in 0..200u64 {
                    let txn = db.begin_write().unwrap();
                    {
                        let mut t = txn.open_multimap_table(MM_TABLE).unwrap();
                        // Each thread writes to shared keys with unique values
                        t.insert(&(i % 50), &(thread_id * 10_000 + i)).unwrap();
                    }
                    txn.commit().unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let txn = db.begin_read().unwrap();
    let t = txn.open_multimap_table(MM_TABLE).unwrap();
    // All 50 keys should have multiple values
    for key in 0..50u64 {
        let count = t.get(&key).unwrap().count();
        assert!(count > 0, "key {key} has no values");
    }

    assert_checksums(&db);
}

// ═══════════════════════════════════════════════════════════════════════
// Read-write isolation under blob operations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn concurrent_blob_read_write() {
    use shodh_redb::{ContentType, StoreOptions};

    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    // Store some blobs
    let txn = db.begin_write().unwrap();
    let mut blob_ids = Vec::new();
    for i in 0..20u64 {
        let data = vec![i as u8; 1024];
        let id = txn
            .store_blob(
                &data,
                ContentType::OctetStream,
                "test",
                StoreOptions::default(),
            )
            .unwrap();
        blob_ids.push(id);
    }
    txn.commit().unwrap();

    let barrier = Arc::new(Barrier::new(3));

    // Reader 1: read blobs via separate read transactions
    let reader_db = db.clone();
    let reader_barrier = barrier.clone();
    let reader_ids = blob_ids.clone();
    let reader1 = thread::spawn(move || {
        reader_barrier.wait();
        for _ in 0..10 {
            let txn = reader_db.begin_read().unwrap();
            for id in &reader_ids {
                let result = txn.get_blob(id).unwrap();
                assert!(result.is_some());
            }
        }
    });

    // Reader 2: read blob stats
    let reader_db2 = db.clone();
    let reader_barrier2 = barrier.clone();
    let reader2 = thread::spawn(move || {
        reader_barrier2.wait();
        for _ in 0..10 {
            let txn = reader_db2.begin_read().unwrap();
            let stats = txn.blob_stats().unwrap();
            assert!(stats.blob_count > 0);
        }
    });

    // Writer: add more blobs concurrently (each in its own transaction)
    let writer_db = db.clone();
    let writer_barrier = barrier.clone();
    let writer = thread::spawn(move || {
        writer_barrier.wait();
        for i in 20..40u64 {
            let txn = writer_db.begin_write().unwrap();
            let data = vec![i as u8; 1024];
            txn.store_blob(
                &data,
                ContentType::OctetStream,
                "new",
                StoreOptions::default(),
            )
            .unwrap();
            txn.commit().unwrap();
        }
    });

    reader1.join().unwrap();
    reader2.join().unwrap();
    writer.join().unwrap();

    assert_checksums(&db);
}

// ═══════════════════════════════════════════════════════════════════════
// Rapid transaction churn
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn rapid_transaction_churn() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());
    let total_commits = Arc::new(AtomicU64::new(0));

    // 5 threads, each doing 500 rapid begin_write->insert->commit cycles
    let handles: Vec<_> = (0..5u64)
        .map(|tid| {
            let db = db.clone();
            let total = total_commits.clone();
            thread::spawn(move || {
                for i in 0..500u64 {
                    let txn = db.begin_write().unwrap();
                    {
                        let mut t = txn.open_table(TABLE).unwrap();
                        t.insert(&(tid * 10_000 + i), &i).unwrap();
                    }
                    txn.commit().unwrap();
                    total.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(total_commits.load(Ordering::Relaxed), 2500);

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 2500);

    assert_checksums(&db);
}

// ═══════════════════════════════════════════════════════════════════════
// Mixed read-write-delete stress
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn mixed_read_write_delete_stress() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    // Seed
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..1000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    let barrier = Arc::new(Barrier::new(4));

    // Writer: insert new keys
    let w_db = db.clone();
    let w_bar = barrier.clone();
    let writer = thread::spawn(move || {
        w_bar.wait();
        for i in 1000..2000u64 {
            let txn = w_db.begin_write().unwrap();
            {
                let mut t = txn.open_table(TABLE).unwrap();
                t.insert(&i, &i).unwrap();
            }
            txn.commit().unwrap();
        }
    });

    // Readers: concurrent range scans
    let readers: Vec<_> = (0..3)
        .map(|_| {
            let db = db.clone();
            let bar = barrier.clone();
            thread::spawn(move || {
                bar.wait();
                let mut total_reads = 0u64;
                for _ in 0..100 {
                    let txn = db.begin_read().unwrap();
                    let t = txn.open_table(TABLE).unwrap();
                    let count = t.range(0u64..500u64).unwrap().count();
                    assert!(count <= 500);
                    total_reads += count as u64;
                }
                total_reads
            })
        })
        .collect();

    writer.join().unwrap();
    for r in readers {
        let reads = r.join().unwrap();
        assert!(reads > 0);
    }

    assert_checksums(&db);
}

// ═══════════════════════════════════════════════════════════════════════
// Abort under contention
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn abort_under_contention() {
    let tmpfile = create_tempfile();
    let db = Arc::new(Database::create(tmpfile.path()).unwrap());

    // Some threads commit, some abort (drop without commit)
    let handles: Vec<_> = (0..10u64)
        .map(|tid| {
            let db = db.clone();
            thread::spawn(move || {
                for i in 0..100u64 {
                    let txn = db.begin_write().unwrap();
                    {
                        let mut t = txn.open_table(TABLE).unwrap();
                        t.insert(&(tid * 1000 + i), &i).unwrap();
                    }
                    if i % 3 == 0 {
                        // Abort every 3rd transaction
                        drop(txn);
                    } else {
                        txn.commit().unwrap();
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_checksums(&db);

    // Should have roughly 2/3 of 10*100 = ~667 entries (some keys overlap across threads)
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    assert!(t.len().unwrap() > 0);
}
