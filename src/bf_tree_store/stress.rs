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
    use crate::bf_tree_store::database::BfTreeDatabase;
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
        let rtxn = db.begin_read();
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
        let rtxn = db.begin_read();
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

            let rtxn = db.begin_read();
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

            let rtxn = db.begin_read();
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
            let encoded = super::super::database::encode_table_key(TABLE_A.name(), key_bytes);
            wtxn.adapter.insert(&encoded, &large_val).unwrap();
        }
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        for i in 0..50u64 {
            let key = alloc::format!("large_{i}");
            let key_bytes = key.as_bytes();
            let encoded = super::super::database::encode_table_key(TABLE_A.name(), key_bytes);
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
        let rtxn = db.begin_read();
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
}
