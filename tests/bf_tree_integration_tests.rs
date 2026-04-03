//! End-to-end integration tests for the BfTree storage engine.
//!
//! Exercises every feature built during the BfTree parity work:
//! - Basic CRUD + buffered transactions (commit/rollback)
//! - TTL tables with expiry
//! - Multimap tables
//! - Blob store (chunked storage, dedup, tags, namespaces, causal graph)
//! - Group commit (sequential + concurrent)
//! - History snapshots (file-backed)
//! - Read verification (checksums)
//! - Unified API (`BackendChoice`)
//! - CDC (change data capture)
//! - Concurrent multi-threaded access
//! - `StorageRead`/`StorageWrite` trait usage

#![cfg(feature = "bf_tree")]

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use shodh_redb::ReadableDatabase;
use shodh_redb::TableDefinition;
use shodh_redb::bf_tree_store::*;
use shodh_redb::blob_store::{CausalLink, ContentType, RelationType, StoreOptions};
use shodh_redb::cdc::CdcConfig;
use shodh_redb::storage_traits::{ReadTable, StorageRead, StorageWrite, WriteTable};

// ---------------------------------------------------------------------------
// Table definitions
// ---------------------------------------------------------------------------

const USERS: TableDefinition<&str, &str> = TableDefinition::new("users");
const COUNTERS: TableDefinition<&str, u64> = TableDefinition::new("counters");
const SCORES: TableDefinition<u64, u64> = TableDefinition::new("scores");

/// Helper: collect all entries from a BfTreeTableScan into a Vec<(Vec<u8>, Vec<u8>)>.
fn collect_scan(mut scan: BfTreeTableScan<'_>) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut buf = vec![0u8; 8192];
    let mut results = Vec::new();
    while let Some((k, v)) = scan.next(&mut buf) {
        results.push((k.to_vec(), v.to_vec()));
    }
    results
}

// ---------------------------------------------------------------------------
// 1. Basic CRUD + Buffered Transactions
// ---------------------------------------------------------------------------

#[test]
fn crud_insert_get_update_delete() {
    let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

    // Insert
    let wtxn = db.begin_write();
    let mut t = wtxn.open_table(USERS);
    t.insert(&"alice", &"engineer").unwrap();
    t.insert(&"bob", &"designer").unwrap();
    t.insert(&"carol", &"pm").unwrap();
    drop(t);
    wtxn.commit().unwrap();

    // Read back
    let rtxn = db.begin_read();
    let t = rtxn.open_table(USERS);
    assert_eq!(t.get(&"alice").unwrap().unwrap(), b"engineer");
    assert_eq!(t.get(&"bob").unwrap().unwrap(), b"designer");
    assert_eq!(t.get(&"carol").unwrap().unwrap(), b"pm");
    assert!(t.get(&"dave").unwrap().is_none());

    // Update
    let wtxn = db.begin_write();
    let mut t = wtxn.open_table(USERS);
    t.insert(&"alice", &"staff engineer").unwrap();
    drop(t);
    wtxn.commit().unwrap();

    let rtxn = db.begin_read();
    let t = rtxn.open_table(USERS);
    assert_eq!(t.get(&"alice").unwrap().unwrap(), b"staff engineer");

    // Delete
    let wtxn = db.begin_write();
    let mut t = wtxn.open_table(USERS);
    t.remove(&"bob").unwrap();
    drop(t);
    wtxn.commit().unwrap();

    let rtxn = db.begin_read();
    let t = rtxn.open_table(USERS);
    assert!(t.get(&"bob").unwrap().is_none());
    assert!(t.get(&"alice").unwrap().is_some());
}

#[test]
fn transaction_rollback_on_drop() {
    let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

    // Commit some data
    let wtxn = db.begin_write();
    let mut t = wtxn.open_table(COUNTERS);
    t.insert(&"x", &100u64).unwrap();
    drop(t);
    wtxn.commit().unwrap();

    // Start a write txn, mutate, but DROP without committing
    {
        let wtxn = db.begin_write();
        let mut t = wtxn.open_table(COUNTERS);
        t.insert(&"x", &999u64).unwrap();
        t.insert(&"y", &42u64).unwrap();
        drop(t);
        // wtxn dropped here — implicit rollback
    }

    // Original value should be intact, new key should not exist
    let rtxn = db.begin_read();
    let t = rtxn.open_table(COUNTERS);
    let val = t.get(&"x").unwrap().unwrap();
    assert_eq!(u64::from_le_bytes(val[..8].try_into().unwrap()), 100);
    assert!(t.get(&"y").unwrap().is_none());
}

#[test]
fn range_scan_ordered() {
    let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

    let wtxn = db.begin_write();
    let mut t = wtxn.open_table(SCORES);
    for i in 0u64..20 {
        t.insert(&i, &(i * 10)).unwrap();
    }
    drop(t);
    wtxn.commit().unwrap();

    let rtxn = db.begin_read();
    let t = rtxn.open_table(SCORES);
    let results = collect_scan(t.scan().unwrap());
    assert_eq!(results.len(), 20);

    // Keys should be in sorted lexicographic order (which for u64 LE is numeric order)
    for (i, (k, v)) in results.iter().enumerate() {
        let key = u64::from_le_bytes(k[..8].try_into().unwrap());
        let val = u64::from_le_bytes(v[..8].try_into().unwrap());
        assert_eq!(key, i as u64);
        assert_eq!(val, (i as u64) * 10);
    }
}

// ---------------------------------------------------------------------------
// 2. StorageRead / StorageWrite Trait Usage
// ---------------------------------------------------------------------------

#[test]
fn storage_traits_generic_code() {
    fn write_data<W: StorageWrite>(txn: &W) {
        let mut t = txn.open_storage_table(COUNTERS).unwrap();
        t.st_insert(&"alpha", &1u64).unwrap();
        t.st_insert(&"beta", &2u64).unwrap();
        t.st_insert(&"gamma", &3u64).unwrap();
    }

    fn read_data<R: StorageRead>(txn: &R) -> u64 {
        let t = txn.open_storage_table(COUNTERS).unwrap();
        let val = t.st_get(&"beta").unwrap().unwrap();
        val.value()
    }

    let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
    let wtxn = db.begin_write();
    write_data(&wtxn);
    wtxn.commit().unwrap();

    let rtxn = db.begin_read();
    let result = read_data(&rtxn);
    assert_eq!(result, 2);
}

// ---------------------------------------------------------------------------
// 3. TTL Tables
// ---------------------------------------------------------------------------

#[test]
fn ttl_table_expiry() {
    let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

    let wtxn = db.begin_write();
    let mut ttl = wtxn.open_ttl_table(USERS);

    ttl.insert_with_ttl(&"ephemeral", &"goes away", Duration::from_millis(100))
        .unwrap();
    ttl.insert_with_ttl(&"persistent", &"stays", Duration::from_secs(60))
        .unwrap();
    drop(ttl);
    wtxn.commit().unwrap();

    // Wait for expiry
    thread::sleep(Duration::from_millis(200));

    let rtxn = db.begin_read();
    let ttl = rtxn.open_ttl_table(USERS);
    assert!(
        ttl.get(&"ephemeral").unwrap().is_none(),
        "expired entry should be invisible"
    );
    assert!(
        ttl.get(&"persistent").unwrap().is_some(),
        "non-expired entry should be visible"
    );
}

// ---------------------------------------------------------------------------
// 4. Multimap Tables
// ---------------------------------------------------------------------------

#[test]
fn multimap_multiple_values_per_key() {
    let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

    let wtxn = db.begin_write();
    let mut mm = wtxn.open_multimap_table::<&str, &str>("tags");
    mm.insert(&"post:1", &"rust").unwrap();
    mm.insert(&"post:1", &"database").unwrap();
    mm.insert(&"post:1", &"embedded").unwrap();
    mm.insert(&"post:2", &"rust").unwrap();
    mm.insert(&"post:2", &"wasm").unwrap();
    drop(mm);
    wtxn.commit().unwrap();

    let rtxn = db.begin_read();
    let mm = rtxn.open_multimap_table::<&str, &str>("tags");

    let post1_tags = mm.get_values(&"post:1").unwrap();
    assert_eq!(post1_tags.len(), 3);

    let post2_tags = mm.get_values(&"post:2").unwrap();
    assert_eq!(post2_tags.len(), 2);

    // Remove one value
    let wtxn = db.begin_write();
    let mut mm = wtxn.open_multimap_table::<&str, &str>("tags");
    mm.remove(&"post:1", &"database").unwrap();
    drop(mm);
    wtxn.commit().unwrap();

    let rtxn = db.begin_read();
    let mm = rtxn.open_multimap_table::<&str, &str>("tags");
    let post1_tags = mm.get_values(&"post:1").unwrap();
    assert_eq!(post1_tags.len(), 2);
}

// ---------------------------------------------------------------------------
// 5. Blob Store
// ---------------------------------------------------------------------------

#[test]
fn blob_store_and_read() {
    let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();

    let payload = b"Hello from the BfTree blob store!";
    let blob_id;

    {
        let wtxn = db.begin_write();
        let bs = wtxn.open_blob_store();
        blob_id = bs
            .store(
                payload,
                ContentType::OctetStream,
                "greeting",
                StoreOptions::default(),
            )
            .unwrap();
        drop(bs);
        wtxn.commit().unwrap();
    }

    {
        let rtxn = db.begin_read();
        let bs = rtxn.open_blob_store();
        let data = bs.read(blob_id).unwrap().unwrap();
        assert_eq!(data, payload);

        let meta = bs.get_meta(blob_id).unwrap().unwrap();
        assert_eq!(meta.blob_ref.length, payload.len() as u64);
    }
}

#[test]
fn blob_large_chunked() {
    let db = BfTreeDatabase::create(BfTreeConfig::new_memory(8)).unwrap();

    // Create data larger than MAX_CHUNK_SIZE (1024 bytes)
    let payload: Vec<u8> = (0..5000u32).map(|i| (i % 256) as u8).collect();
    let blob_id;

    {
        let wtxn = db.begin_write();
        let bs = wtxn.open_blob_store();
        blob_id = bs
            .store(
                &payload,
                ContentType::Embedding,
                "vectors",
                StoreOptions::default(),
            )
            .unwrap();
        drop(bs);
        wtxn.commit().unwrap();
    }

    {
        let rtxn = db.begin_read();
        let bs = rtxn.open_blob_store();
        let data = bs.read(blob_id).unwrap().unwrap();
        assert_eq!(data.len(), 5000);
        assert_eq!(data, payload);

        // Partial read
        let chunk = bs.read_range(blob_id, 100, 50).unwrap();
        assert_eq!(chunk.len(), 50);
        assert_eq!(chunk, &payload[100..150]);
    }
}

#[test]
fn blob_tags_and_namespace() {
    let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
    let wtxn = db.begin_write();
    let bs = wtxn.open_blob_store();

    let opts_sensor = StoreOptions {
        namespace: Some("sensors".into()),
        tags: vec!["lidar".into(), "v2".into()],
        ..Default::default()
    };
    let id1 = bs
        .store(
            b"lidar-scan-001",
            ContentType::OctetStream,
            "scan1",
            opts_sensor,
        )
        .unwrap();

    let opts_camera = StoreOptions {
        namespace: Some("sensors".into()),
        tags: vec!["camera".into(), "v2".into()],
        ..Default::default()
    };
    let id2 = bs
        .store(
            b"image-frame-001",
            ContentType::OctetStream,
            "frame1",
            opts_camera,
        )
        .unwrap();

    let opts_other = StoreOptions {
        namespace: Some("logs".into()),
        tags: vec!["debug".into()],
        ..Default::default()
    };
    let _id3 = bs
        .store(b"log-entry", ContentType::OctetStream, "log1", opts_other)
        .unwrap();

    // Query WITHIN the same transaction (read-your-writes)
    let v2_blobs = bs.query_by_tag("v2").unwrap();
    assert_eq!(v2_blobs.len(), 2, "should see tags before commit");
    assert!(v2_blobs.contains(&id1));
    assert!(v2_blobs.contains(&id2));

    let lidar_blobs = bs.query_by_tag("lidar").unwrap();
    assert_eq!(lidar_blobs.len(), 1);
    assert_eq!(lidar_blobs[0], id1);

    let sensor_blobs = bs.query_by_namespace("sensors").unwrap();
    assert_eq!(
        sensor_blobs.len(),
        2,
        "should see namespace entries before commit"
    );

    drop(bs);
    wtxn.commit().unwrap();

    // Also verify after commit (data persisted to BfTree)
    let rtxn = db.begin_read();
    let bs = rtxn.open_blob_store();
    let v2_blobs = bs.query_by_tag("v2").unwrap();
    assert_eq!(v2_blobs.len(), 2, "should see tags after commit");
    let sensor_blobs = bs.blobs_in_namespace("sensors").unwrap();
    assert_eq!(sensor_blobs.len(), 2, "should see namespace after commit");
}

#[test]
fn blob_causal_graph() {
    let db = BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap();
    let wtxn = db.begin_write();
    let bs = wtxn.open_blob_store();

    // Parent blob
    let parent_id = bs
        .store(
            b"parent",
            ContentType::OctetStream,
            "root",
            StoreOptions::default(),
        )
        .unwrap();

    // Child blob with causal link
    let child_opts = StoreOptions {
        causal_link: Some(CausalLink {
            parent: parent_id,
            relation: RelationType::Derived,
            context: "processed output".into(),
        }),
        ..Default::default()
    };
    let child_id = bs
        .store(b"child", ContentType::OctetStream, "derived", child_opts)
        .unwrap();

    // Query WITHIN same transaction (read-your-writes for causal graph)
    let children = bs.causal_children(parent_id).unwrap();
    assert_eq!(children.len(), 1, "should see causal edges before commit");
    assert_eq!(children[0].child, child_id);
    assert_eq!(children[0].relation, RelationType::Derived);

    // Read blob content within same transaction
    let content = bs.read(parent_id).unwrap().unwrap();
    assert_eq!(content, b"parent");
    let content = bs.read(child_id).unwrap().unwrap();
    assert_eq!(content, b"child");

    drop(bs);
    wtxn.commit().unwrap();

    // Verify persistence after commit
    let rtxn = db.begin_read();
    let bs = rtxn.open_blob_store();
    let children = bs.causal_children(parent_id).unwrap();
    assert_eq!(
        children.len(),
        1,
        "causal edges should persist after commit"
    );
}

// ---------------------------------------------------------------------------
// 6. Group Commit
// ---------------------------------------------------------------------------

#[test]
fn group_commit_sequential() {
    let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap());
    let mut gc = GroupCommit::new(db.clone());

    gc.add(|txn| {
        let mut t = txn.open_table(COUNTERS);
        t.insert(&"batch_a", &10u64)?;
        Ok(())
    });
    gc.add(|txn| {
        let mut t = txn.open_table(COUNTERS);
        t.insert(&"batch_b", &20u64)?;
        Ok(())
    });
    gc.add(|txn| {
        let mut t = txn.open_table(COUNTERS);
        t.insert(&"batch_c", &30u64)?;
        Ok(())
    });

    let count = gc.execute().unwrap();
    assert_eq!(count, 3);

    let rtxn = db.begin_read();
    let t = rtxn.open_table(COUNTERS);
    assert!(t.get(&"batch_a").unwrap().is_some());
    assert!(t.get(&"batch_b").unwrap().is_some());
    assert!(t.get(&"batch_c").unwrap().is_some());
}

#[test]
fn group_commit_concurrent() {
    let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_memory(4)).unwrap());

    let batches: Vec<WriteBatchFn> = (0u64..8)
        .map(|i| {
            let batch: WriteBatchFn = Box::new(move |txn| {
                let mut t = txn.open_table(SCORES);
                t.insert(&(i * 100), &(i * 1000))?;
                Ok(())
            });
            batch
        })
        .collect();

    let count = concurrent_group_commit(db.clone(), batches).unwrap();
    assert_eq!(count, 8);

    let rtxn = db.begin_read();
    let t = rtxn.open_table(SCORES);
    for i in 0u64..8 {
        assert!(
            t.get(&(i * 100)).unwrap().is_some(),
            "key {} should exist after concurrent group commit",
            i * 100
        );
    }
}

// ---------------------------------------------------------------------------
// 7. History Snapshots (file-backed)
// ---------------------------------------------------------------------------

#[test]
fn history_snapshot_and_restore() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("history_test.bftree");
    let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_file(&db_path, 4)).unwrap());

    // Write V1 data
    let wtxn = db.begin_write();
    let mut t = wtxn.open_table(USERS);
    t.insert(&"alice", &"v1").unwrap();
    drop(t);
    wtxn.commit().unwrap();

    // Take snapshot
    let history = BfTreeHistory::new(db.clone());
    let (snap_id, _snap_path) = history.commit_snapshot().unwrap();

    // Open historical snapshot — should be a valid, readable database
    let historical = history.open_historical(snap_id).unwrap();
    let rtxn_hist = historical.begin_read();
    let t_hist = rtxn_hist.open_table(USERS);
    // The snapshot captured the state at snapshot time — alice should exist
    let alice_val = t_hist.get(&"alice").unwrap();
    assert!(alice_val.is_some(), "snapshot should contain alice");

    // List history entries
    let entries = history.list().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, snap_id);

    // Take a second snapshot after more writes
    let wtxn = db.begin_write();
    let mut t = wtxn.open_table(USERS);
    t.insert(&"bob", &"new").unwrap();
    drop(t);
    wtxn.commit().unwrap();

    let (snap_id2, _) = history.commit_snapshot().unwrap();
    assert!(snap_id2 > snap_id);

    let entries = history.list().unwrap();
    assert_eq!(entries.len(), 2);

    // Prune old snapshots, keep only latest
    let pruned = history.prune(1).unwrap();
    assert_eq!(pruned, 1);
    let entries = history.list().unwrap();
    assert_eq!(entries.len(), 1);
}

// ---------------------------------------------------------------------------
// 8. Read Verification (Checksums)
// ---------------------------------------------------------------------------

#[test]
fn verification_wrap_unwrap() {
    let original = b"important data that must not be corrupted";
    let wrapped = wrap_value(original);

    let data = unwrap_value(&wrapped, true).unwrap();
    assert_eq!(data, original);

    // Corrupt and detect
    let mut corrupted = wrapped.clone();
    corrupted[10] ^= 0xFF;
    assert!(unwrap_value(&corrupted, true).is_err());

    // Without verification, corruption passes
    assert!(unwrap_value(&corrupted, false).is_ok());
}

#[test]
fn verify_mode_sampling() {
    assert!(!should_verify(&VerifyMode::None));
    assert!(should_verify(&VerifyMode::Full));
    assert!(should_verify(&VerifyMode::Sampled(1.0)));
    assert!(!should_verify(&VerifyMode::Sampled(0.0)));
}

// ---------------------------------------------------------------------------
// 9. Unified API
// ---------------------------------------------------------------------------

#[test]
fn unified_api_bftree_backend() {
    let config = BfTreeConfig::new_memory(4);
    let udb = UnifiedDatabase::create(BackendChoice::BfTree(config), "").unwrap();
    assert!(udb.is_bf_tree());

    let db = udb.as_bf_tree().unwrap();
    let wtxn = db.begin_write();
    let mut t = wtxn.open_table(COUNTERS);
    t.insert(&"unified", &42u64).unwrap();
    drop(t);
    wtxn.commit().unwrap();

    let rtxn = db.begin_read();
    let t = rtxn.open_table(COUNTERS);
    let val = t.get(&"unified").unwrap().unwrap();
    assert_eq!(u64::from_le_bytes(val[..8].try_into().unwrap()), 42);
}

#[test]
fn unified_api_legacy_backend() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let udb = UnifiedDatabase::create(BackendChoice::Legacy, tmp.path()).unwrap();
    assert!(udb.is_legacy());

    let db = udb.as_legacy().unwrap();
    let wtxn = db.begin_write().unwrap();
    {
        let mut t = wtxn.open_table(COUNTERS).unwrap();
        t.insert("unified", &42u64).unwrap();
    }
    wtxn.commit().unwrap();

    let rtxn = db.begin_read().unwrap();
    let t = rtxn.open_table(COUNTERS).unwrap();
    assert_eq!(t.get("unified").unwrap().unwrap().value(), 42);
}

// ---------------------------------------------------------------------------
// 10. CDC (Change Data Capture)
// ---------------------------------------------------------------------------

#[test]
fn cdc_records_changes() {
    let config = BfTreeConfig::new_memory(4);
    let mut builder = BfTreeBuilder::new();
    builder.set_cdc(CdcConfig {
        enabled: true,
        retention_max_txns: 1000,
    });
    let db = builder.create(config).unwrap();

    let wtxn = db.begin_write();
    let mut t = wtxn.open_table(USERS);
    t.insert(&"alice", &"engineer").unwrap();
    t.insert(&"bob", &"designer").unwrap();
    drop(t);
    wtxn.commit().unwrap();

    let rtxn = db.begin_read();
    let changes = rtxn.read_cdc_since(0).unwrap();
    assert!(
        !changes.is_empty(),
        "CDC should have recorded the insert events"
    );
}

// ---------------------------------------------------------------------------
// 11. Concurrent Multi-threaded Access
// ---------------------------------------------------------------------------

#[test]
fn concurrent_readers_and_writers() {
    let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_memory(8)).unwrap());

    // Spawn 4 writer threads
    let mut write_handles = Vec::new();
    for thread_id in 0u64..4 {
        let db = db.clone();
        write_handles.push(thread::spawn(move || {
            for i in 0u64..50 {
                let wtxn = db.begin_write();
                let mut t = wtxn.open_table(SCORES);
                let key = thread_id * 1000 + i;
                t.insert(&key, &(key * 10)).unwrap();
                drop(t);
                wtxn.commit().unwrap();
            }
        }));
    }

    // Spawn reader that continuously reads during writes
    let db_r = db.clone();
    let reader = thread::spawn(move || {
        let mut reads = 0u64;
        for _ in 0..100 {
            let rtxn = db_r.begin_read();
            let t = rtxn.open_table(SCORES);
            let _ = collect_scan(t.scan().unwrap());
            reads += 1;
            thread::sleep(Duration::from_millis(1));
        }
        reads
    });

    for h in write_handles {
        h.join().unwrap();
    }
    let reads = reader.join().unwrap();
    assert!(reads > 0);

    // Verify all 200 writes landed
    let rtxn = db.begin_read();
    let t = rtxn.open_table(SCORES);
    let results = collect_scan(t.scan().unwrap());
    assert_eq!(results.len(), 200, "4 threads * 50 keys = 200");
}

// ---------------------------------------------------------------------------
// 12. Full Workflow: IoT Sensor Pipeline
// ---------------------------------------------------------------------------

#[test]
fn iot_sensor_pipeline() {
    let db = Arc::new(BfTreeDatabase::create(BfTreeConfig::new_memory(8)).unwrap());

    const READINGS: TableDefinition<u64, u64> = TableDefinition::new("readings");

    // Phase 1: Ingest sensor readings via group commit
    let mut gc = GroupCommit::new(db.clone());
    for sensor_id in 0u64..5 {
        gc.add(move |txn| {
            let mut t = txn.open_table(READINGS);
            for ts in 0u64..10 {
                let key = sensor_id * 10000 + ts;
                let value = sensor_id * 100 + ts;
                t.insert(&key, &value)?;
            }
            Ok(())
        });
    }
    let batches = gc.execute().unwrap();
    assert_eq!(batches, 5);

    // Phase 2: Store processed results as blobs with tags + causal links
    {
        let wtxn = db.begin_write();
        let bs = wtxn.open_blob_store();

        let parent_id = bs
            .store(
                b"aggregated sensor report v1",
                ContentType::OctetStream,
                "report-v1",
                StoreOptions {
                    namespace: Some("reports".into()),
                    tags: vec!["sensor".into(), "daily".into()],
                    ..Default::default()
                },
            )
            .unwrap();

        let _child_id = bs
            .store(
                b"filtered anomaly subset",
                ContentType::OctetStream,
                "anomalies-v1",
                StoreOptions {
                    namespace: Some("reports".into()),
                    tags: vec!["sensor".into(), "anomaly".into()],
                    causal_link: Some(CausalLink {
                        parent: parent_id,
                        relation: RelationType::Derived,
                        context: "anomaly detection".into(),
                    }),
                    ..Default::default()
                },
            )
            .unwrap();

        drop(bs);
        wtxn.commit().unwrap();

        // Query after commit
        let rtxn = db.begin_read();
        let bs = rtxn.open_blob_store();

        let sensor_reports = bs.query_by_tag("sensor").unwrap();
        assert_eq!(sensor_reports.len(), 2);

        let anomaly_reports = bs.query_by_tag("anomaly").unwrap();
        assert_eq!(anomaly_reports.len(), 1);
    }

    // Phase 3: TTL for ephemeral sensor data
    {
        let wtxn = db.begin_write();
        let mut ttl = wtxn.open_ttl_table(READINGS);

        ttl.insert_with_ttl(&99999u64, &12345u64, Duration::from_secs(3600))
            .unwrap();
        drop(ttl);
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let ttl = rtxn.open_ttl_table(READINGS);
        assert!(ttl.get(&99999u64).unwrap().is_some());
    }

    // Phase 4: Verify all sensor readings persisted
    {
        let rtxn = db.begin_read();
        let t = rtxn.open_table(READINGS);
        let all = collect_scan(t.scan().unwrap());
        assert!(
            all.len() >= 50,
            "should have at least 50 sensor readings, got {}",
            all.len()
        );
    }
}

// ---------------------------------------------------------------------------
// 13. Knowledge Graph: Multimap + Blob + Causal
// ---------------------------------------------------------------------------

#[test]
fn knowledge_graph_workflow() {
    let db = BfTreeDatabase::create(BfTreeConfig::new_memory(8)).unwrap();

    let wtxn = db.begin_write();
    let bs = wtxn.open_blob_store();

    let doc1 = bs
        .store(
            b"Rust is a systems programming language",
            ContentType::OctetStream,
            "doc1",
            StoreOptions {
                tags: vec!["rust".into(), "programming".into()],
                ..Default::default()
            },
        )
        .unwrap();

    let doc2 = bs
        .store(
            b"BfTree is a concurrent B+tree from MSR",
            ContentType::OctetStream,
            "doc2",
            StoreOptions {
                tags: vec!["bftree".into(), "database".into()],
                causal_link: Some(CausalLink {
                    parent: doc1,
                    relation: RelationType::Supports,
                    context: "related work".into(),
                }),
                ..Default::default()
            },
        )
        .unwrap();

    drop(bs);

    // Multimap: tag → doc sequence index
    let mut mm = wtxn.open_multimap_table::<&str, u64>("doc_tags");
    mm.insert(&"rust", &doc1.sequence).unwrap();
    mm.insert(&"programming", &doc1.sequence).unwrap();
    mm.insert(&"bftree", &doc2.sequence).unwrap();
    mm.insert(&"database", &doc2.sequence).unwrap();
    drop(mm);
    wtxn.commit().unwrap();

    // Query
    let rtxn = db.begin_read();
    let mm = rtxn.open_multimap_table::<&str, u64>("doc_tags");

    let rust_docs = mm.get_values(&"rust").unwrap();
    assert_eq!(rust_docs.len(), 1);

    let db_docs = mm.get_values(&"database").unwrap();
    assert_eq!(db_docs.len(), 1);

    // Read blob content
    let bs = rtxn.open_blob_store();
    let content = bs.read(doc1).unwrap().unwrap();
    assert_eq!(content, b"Rust is a systems programming language");

    // Causal graph
    let children = bs.causal_children(doc1).unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].child, doc2);
}
