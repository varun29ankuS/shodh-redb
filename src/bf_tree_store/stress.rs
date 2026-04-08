//! Stress tests for Bf-Tree concurrent storage engine.
//!
//! These tests exercise heavy concurrent read/write workloads to validate
//! correctness under contention, verify table namespace isolation at scale,
//! and test file-backed persistence with snapshot/recovery.

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    use crate::bf_tree_store::config::BfTreeConfig;
    use crate::bf_tree_store::database::{BfTreeBuilder, BfTreeDatabase};
    use crate::cdc::types::{CdcConfig, ChangeOp};
    use crate::{TableDefinition, TableHandle};

    const TABLE_A: TableDefinition<&str, u64> = TableDefinition::new("table_a");
    const TABLE_B: TableDefinition<&str, u64> = TableDefinition::new("table_b");

    /// 8 writer threads, 4 reader threads, 1000 ops each, all concurrent.
    #[test]
    fn heavy_concurrent_read_write() {
        let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_memory(16)).unwrap());
        let barrier = Arc::new(Barrier::new(12)); // 8 writers + 4 readers
        let total_writes = Arc::new(AtomicU64::new(0));

        let mut handles: Vec<thread::JoinHandle<()>> = Vec::new();

        // 8 writer threads
        for t in 0..8u64 {
            let db = db.clone();
            let barrier = barrier.clone();
            let total_writes = total_writes.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                let wtxn = db.begin_write();
                for i in 0..1000u64 {
                    let key = alloc::format!("w{t}_k{i}");
                    let val = t * 10000 + i;
                    wtxn.insert(&TABLE_A, &key.as_str(), &val).unwrap();
                    total_writes.fetch_add(1, Ordering::Relaxed);
                }
                wtxn.commit().unwrap();
            }));
        }

        // 4 reader threads that continuously read while writers are active
        for _ in 0..4 {
            let db = db.clone();
            let barrier = barrier.clone();
            let total_writes = total_writes.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                let rtxn = db.begin_read();
                let mut found = 0u64;
                // Keep reading until most writes are done.
                while total_writes.load(Ordering::Relaxed) < 6000 {
                    // Try to read some keys -- they may or may not exist yet.
                    for t in 0..8u64 {
                        let key = alloc::format!("w{t}_k{found}");
                        if rtxn.contains_key(&TABLE_A, &key.as_str()) {
                            found += 1;
                        }
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Verify all 8000 writes landed.
        let mut rtxn = db.begin_read();
        for t in 0..8u64 {
            for i in 0..1000u64 {
                let key = alloc::format!("w{t}_k{i}");
                let val_bytes = rtxn
                    .get::<&str, u64>(&TABLE_A, &key.as_str())
                    .unwrap()
                    .unwrap();
                let val = u64::from_le_bytes(val_bytes.as_slice().try_into().unwrap());
                assert_eq!(val, t * 10000 + i);
            }
        }
    }

    /// Multiple tables with concurrent writers -- verify no cross-contamination.
    #[test]
    fn namespace_isolation_under_concurrency() {
        let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_memory(8)).unwrap());
        let barrier = Arc::new(Barrier::new(4));

        let mut handles = Vec::new();

        // 2 threads write to TABLE_A
        for t in 0..2u64 {
            let db = db.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                let wtxn = db.begin_write();
                for i in 0..500u64 {
                    let key = alloc::format!("a{t}_{i}");
                    wtxn.insert(&TABLE_A, &key.as_str(), &(i + 1000)).unwrap();
                }
                wtxn.commit().unwrap();
            }));
        }

        // 2 threads write to TABLE_B with same key patterns
        for t in 0..2u64 {
            let db = db.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                let wtxn = db.begin_write();
                for i in 0..500u64 {
                    let key = alloc::format!("a{t}_{i}"); // same key pattern!
                    wtxn.insert(&TABLE_B, &key.as_str(), &(i + 9000)).unwrap();
                }
                wtxn.commit().unwrap();
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Verify TABLE_A and TABLE_B have different values for the same keys.
        let mut rtxn = db.begin_read();
        for t in 0..2u64 {
            for i in 0..500u64 {
                let key = alloc::format!("a{t}_{i}");
                let a_bytes = rtxn
                    .get::<&str, u64>(&TABLE_A, &key.as_str())
                    .unwrap()
                    .unwrap();
                let b_bytes = rtxn
                    .get::<&str, u64>(&TABLE_B, &key.as_str())
                    .unwrap()
                    .unwrap();
                let a_val = u64::from_le_bytes(a_bytes.as_slice().try_into().unwrap());
                let b_val = u64::from_le_bytes(b_bytes.as_slice().try_into().unwrap());
                assert_eq!(a_val, i + 1000);
                assert_eq!(b_val, i + 9000);
            }
        }
    }

    /// File-backed snapshot and recovery.
    #[test]
    fn file_backed_snapshot_recovery() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.bftree");

        // Phase 1: Create, populate, snapshot.
        {
            let config = BfTreeConfig::new_file(&db_path, 4);
            let db = BfTreeDatabase::create(config).unwrap();

            let wtxn = db.begin_write();
            for i in 0..100u64 {
                let key = alloc::format!("key_{i}");
                wtxn.insert(&TABLE_A, &key.as_str(), &(i * 7)).unwrap();
            }
            wtxn.commit_with_snapshot().unwrap();
        }

        // Phase 2: Reopen from snapshot and verify.
        {
            let config = BfTreeConfig::new_file(&db_path, 4);
            let db = BfTreeDatabase::open(config).unwrap();

            let mut rtxn = db.begin_read();
            for i in 0..100u64 {
                let key = alloc::format!("key_{i}");
                let val_bytes = rtxn
                    .get::<&str, u64>(&TABLE_A, &key.as_str())
                    .unwrap()
                    .unwrap_or_else(|| panic!("key_{i} not found after recovery"));
                let val = u64::from_le_bytes(val_bytes.as_slice().try_into().unwrap());
                assert_eq!(val, i * 7, "key_{i} has wrong value after recovery");
            }
        }
    }

    /// Verify delete persistence across snapshot/recovery.
    #[test]
    fn delete_persists_across_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test_del.bftree");

        {
            let config = BfTreeConfig::new_file(&db_path, 4);
            let db = BfTreeDatabase::create(config).unwrap();

            let wtxn = db.begin_write();
            wtxn.insert(&TABLE_A, &"keep", &1u64).unwrap();
            wtxn.insert(&TABLE_A, &"remove", &2u64).unwrap();
            wtxn.delete(&TABLE_A, &"remove");
            wtxn.commit_with_snapshot().unwrap();
        }

        {
            let config = BfTreeConfig::new_file(&db_path, 4);
            let db = BfTreeDatabase::open(config).unwrap();

            let mut rtxn = db.begin_read();
            assert!(rtxn.get::<&str, u64>(&TABLE_A, &"keep").unwrap().is_some());
            assert!(
                rtxn.get::<&str, u64>(&TABLE_A, &"remove")
                    .unwrap()
                    .is_none(),
                "deleted key should not survive snapshot/recovery"
            );
        }
    }

    /// Large value stress -- write values near the max record size.
    #[test]
    fn large_values() {
        let db = BfTreeDatabase::create(BfTreeConfig::new_memory(8)).unwrap();

        let wtxn = db.begin_write();
        // Default max_record_size is 1568, so value must be < 1568 - key_overhead.
        // Key overhead: 2 (name_len) + 7 (table_a) + ~8 (key bytes) = ~17 bytes.
        let large_val = vec![0xABu8; 1400];
        for i in 0..50u64 {
            let key = alloc::format!("large_{i}");
            // Insert raw bytes via the adapter since our typed API serializes u64.
            let key_bytes = key.as_bytes();
            let encoded = super::super::database::encode_table_key(
                TABLE_A.name(),
                super::super::database::TableKind::Regular,
                key_bytes,
            );
            wtxn.adapter.insert(&encoded, &large_val).unwrap();
        }
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        for i in 0..50u64 {
            let key = alloc::format!("large_{i}");
            let key_bytes = key.as_bytes();
            let encoded = super::super::database::encode_table_key(
                TABLE_A.name(),
                super::super::database::TableKind::Regular,
                key_bytes,
            );
            let max_val = db.adapter().inner().config().get_cb_max_record_size();
            let mut buf = vec![0u8; max_val];
            let len = rtxn.adapter.read(&encoded, &mut buf).unwrap();
            assert_eq!(&buf[..len as usize], &large_val[..]);
        }
    }

    /// Rapid insert-overwrite cycles from multiple threads.
    #[test]
    fn overwrite_storm() {
        let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_memory(8)).unwrap());

        // Pre-populate 100 keys.
        {
            let wtxn = db.begin_write();
            for i in 0..100u64 {
                let key = alloc::format!("storm_{i}");
                wtxn.insert(&TABLE_A, &key.as_str(), &i).unwrap();
            }
            wtxn.commit().unwrap();
        }

        // 4 threads each overwrite all 100 keys with their thread ID.
        let barrier = Arc::new(Barrier::new(4));
        let handles: Vec<_> = (0..4u64)
            .map(|t| {
                let db = db.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    let wtxn = db.begin_write();
                    for i in 0..100u64 {
                        let key = alloc::format!("storm_{i}");
                        wtxn.insert(&TABLE_A, &key.as_str(), &(t * 1000 + i))
                            .unwrap();
                    }
                    wtxn.commit().unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Every key must have a valid value from one of the 4 threads.
        let mut rtxn = db.begin_read();
        for i in 0..100u64 {
            let key = alloc::format!("storm_{i}");
            let val_bytes = rtxn
                .get::<&str, u64>(&TABLE_A, &key.as_str())
                .unwrap()
                .unwrap();
            let val = u64::from_le_bytes(val_bytes.as_slice().try_into().unwrap());
            let thread_id = val / 1000;
            let key_id = val % 1000;
            assert!(thread_id < 4, "unexpected thread id {thread_id}");
            assert_eq!(key_id, i);
        }
    }

    /// Concurrent insert/remove on the same table within a single write txn
    /// via `thread::scope`. Exercises the TOCTOU fix (buffer lock held across
    /// read-modify-write) and verifies CDC event consistency.
    ///
    /// 4 threads share one `BfTreeDatabaseWriteTxn`, each opening its own
    /// `BfTreeTable` handle. They insert overlapping keys so the buffer mutex
    /// serializes their updates. After commit, CDC events must match the
    /// final persisted state.
    #[test]
    fn toctou_shared_txn_with_cdc() {
        const NUM_THREADS: usize = 4;
        const KEYS_PER_THREAD: u64 = 200;
        const OVERLAP: u64 = 50; // first 50 keys written by every thread

        let mut builder = BfTreeBuilder::new();
        builder.set_cdc(CdcConfig {
            enabled: true,
            retention_max_txns: 1000,
        });
        let db = builder.create(BfTreeConfig::new_memory(8)).unwrap();

        // Phase 1: Concurrent inserts within a single write txn.
        let wtxn = db.begin_write();
        let barrier = Barrier::new(NUM_THREADS);
        thread::scope(|s| {
            for t in 0..NUM_THREADS {
                let barrier = &barrier;
                let wtxn = &wtxn;
                s.spawn(move || {
                    barrier.wait();
                    let tid = t as u64;
                    let mut table = wtxn.open_table(TABLE_A).unwrap();
                    for i in 0..KEYS_PER_THREAD {
                        // First OVERLAP keys are written by all threads (contention).
                        // Remaining keys are thread-unique.
                        let key = if i < OVERLAP {
                            alloc::format!("shared_{i}")
                        } else {
                            alloc::format!("t{t}_{i}")
                        };
                        let val = tid * 10000 + i;
                        table.insert(&key.as_str(), &val).unwrap();
                    }
                });
            }
        });
        wtxn.commit().unwrap();

        // Phase 2: Verify persisted state.
        let mut rtxn = db.begin_read();

        // Shared keys: each must have a value from one of the threads.
        for i in 0..OVERLAP {
            let key = alloc::format!("shared_{i}");
            let val_bytes = rtxn
                .get::<&str, u64>(&TABLE_A, &key.as_str())
                .unwrap()
                .unwrap_or_else(|| panic!("shared_{i} missing"));
            let val = u64::from_le_bytes(val_bytes.as_slice().try_into().unwrap());
            let thread_id = val / 10000;
            let key_id = val % 10000;
            assert!(
                (thread_id as usize) < NUM_THREADS,
                "shared_{i} has invalid thread_id {thread_id}"
            );
            assert_eq!(key_id, i, "shared_{i} value index mismatch");
        }

        // Thread-unique keys must all exist.
        for t in 0..NUM_THREADS {
            for i in OVERLAP..KEYS_PER_THREAD {
                let key = alloc::format!("t{t}_{i}");
                assert!(
                    rtxn.get::<&str, u64>(&TABLE_A, &key.as_str())
                        .unwrap()
                        .is_some(),
                    "t{t}_{i} missing"
                );
            }
        }

        // Phase 3: Verify CDC events.
        let changes = rtxn.read_cdc_since(0).unwrap();
        assert!(
            !changes.is_empty(),
            "CDC should have recorded events for the committed txn"
        );

        // Build a map of the final CDC event per key (last-writer-wins).
        let mut final_ops: alloc::collections::BTreeMap<String, ChangeOp> =
            alloc::collections::BTreeMap::new();
        for c in &changes {
            if c.table_name == "table_a" {
                final_ops.insert(
                    alloc::string::String::from_utf8_lossy(&c.key).into_owned(),
                    c.op,
                );
            }
        }

        // Every shared key should have an Insert or Update CDC event.
        for i in 0..OVERLAP {
            let key = alloc::format!("shared_{i}");
            let op = final_ops
                .get(&key)
                .unwrap_or_else(|| panic!("no CDC event for shared_{i}"));
            assert!(
                matches!(op, ChangeOp::Insert | ChangeOp::Update),
                "shared_{i} CDC op should be Insert or Update, got {op:?}"
            );
        }
    }

    /// Concurrent insert + remove on overlapping keys within a single txn.
    /// Verifies that after commit, removed keys are gone and CDC records
    /// the Delete events.
    #[test]
    fn toctou_concurrent_insert_remove_cdc() {
        const KEYS: u64 = 100;

        let mut builder = BfTreeBuilder::new();
        builder.set_cdc(CdcConfig {
            enabled: true,
            retention_max_txns: 1000,
        });
        let db = builder.create(BfTreeConfig::new_memory(8)).unwrap();

        // Pre-populate keys so removes have something to act on.
        {
            let wtxn = db.begin_write();
            let mut table = wtxn.open_table(TABLE_A).unwrap();
            for i in 0..KEYS {
                let key = alloc::format!("k_{i}");
                table.insert(&key.as_str(), &i).unwrap();
            }
            let _ = table;
            wtxn.commit().unwrap();
        }

        // Phase 1: Two threads -- one inserts even keys, one removes odd keys.
        let wtxn = db.begin_write();
        let barrier = Barrier::new(2);
        thread::scope(|s| {
            let wtxn = &wtxn;
            let barrier = &barrier;

            // Thread A: overwrite even keys with new values.
            s.spawn(move || {
                barrier.wait();
                let mut table = wtxn.open_table(TABLE_A).unwrap();
                for i in (0..KEYS).step_by(2) {
                    let key = alloc::format!("k_{i}");
                    table.insert(&key.as_str(), &(i + 9000)).unwrap();
                }
            });

            // Thread B: remove odd keys.
            s.spawn(move || {
                barrier.wait();
                let mut table = wtxn.open_table(TABLE_A).unwrap();
                for i in (1..KEYS).step_by(2) {
                    let key = alloc::format!("k_{i}");
                    table.remove(&key.as_str()).unwrap();
                }
            });
        });
        wtxn.commit().unwrap();

        // Phase 2: Verify state.
        let mut rtxn = db.begin_read();
        for i in 0..KEYS {
            let key = alloc::format!("k_{i}");
            let result = rtxn.get::<&str, u64>(&TABLE_A, &key.as_str()).unwrap();
            if i % 2 == 0 {
                let val_bytes = result.unwrap_or_else(|| panic!("k_{i} should exist (even)"));
                let val = u64::from_le_bytes(val_bytes.as_slice().try_into().unwrap());
                assert_eq!(val, i + 9000, "k_{i} has wrong value");
            } else {
                assert!(result.is_none(), "k_{i} should be deleted (odd)");
            }
        }

        // Phase 3: Verify CDC has Delete events for odd keys.
        let changes = rtxn.read_cdc_since(0).unwrap();
        let delete_keys: alloc::collections::BTreeSet<String> = changes
            .iter()
            .filter(|c| c.table_name == "table_a" && matches!(c.op, ChangeOp::Delete))
            .map(|c| alloc::string::String::from_utf8_lossy(&c.key).into_owned())
            .collect();

        for i in (1..KEYS).step_by(2) {
            let key = alloc::format!("k_{i}");
            assert!(
                delete_keys.contains(&key),
                "CDC missing Delete event for {key}"
            );
        }
    }
}
