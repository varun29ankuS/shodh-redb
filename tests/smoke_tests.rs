//! Smoke tests — fast happy-path validation of every major subsystem.
//! Each test opens a fresh DB, exercises one feature, and asserts correctness.
//! Target: ~100 tests pass in < 10 seconds.

use std::io::Write;
use std::time::Duration;

use shodh_redb::{
    BitwiseOr, BytesAppend, CdcConfig, ChangeOp, ContentType, Database, Durability, FloatAdd,
    MultimapTableDefinition, NumericAdd, NumericMax, NumericMin, ReadableDatabase,
    ReadableMultimapTable, ReadableTable, ReadableTableMetadata, SaturatingAdd, StoreOptions,
    TableDefinition, TtlTableDefinition, WriteBatch,
};
use tempfile::NamedTempFile;

fn create_tempfile() -> NamedTempFile {
    if cfg!(target_os = "wasi") {
        NamedTempFile::new_in("/tmp").unwrap()
    } else {
        NamedTempFile::new().unwrap()
    }
}

// ─── Core Operations ────────────────────────────────────────────────

const KV_TABLE: TableDefinition<&str, u64> = TableDefinition::new("smoke_kv");

#[test]
fn kv_crud() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Insert
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("alpha", &1u64).unwrap();
        t.insert("beta", &2u64).unwrap();
        t.insert("gamma", &3u64).unwrap();
    }
    txn.commit().unwrap();

    // Read + update
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        assert_eq!(t.get("alpha").unwrap().unwrap().value(), 1);
        t.insert("alpha", &10u64).unwrap();
    }
    txn.commit().unwrap();

    // Verify update + range scan
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(KV_TABLE).unwrap();
    assert_eq!(t.get("alpha").unwrap().unwrap().value(), 10);
    let count = t.range::<&str>(..).unwrap().count();
    assert_eq!(count, 3);

    // Remove
    drop(txn);
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.remove("beta").unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(KV_TABLE).unwrap();
    assert!(t.get("beta").unwrap().is_none());
    assert_eq!(t.len().unwrap(), 2);
}

const MM_TABLE: MultimapTableDefinition<&str, u64> = MultimapTableDefinition::new("smoke_mm");

#[test]
fn multimap_crud() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_multimap_table(MM_TABLE).unwrap();
        t.insert("tags", &1u64).unwrap();
        t.insert("tags", &2u64).unwrap();
        t.insert("tags", &3u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_multimap_table(MM_TABLE).unwrap();
    let values: Vec<u64> = t.get("tags").unwrap().map(|r| r.unwrap().value()).collect();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&1));
    assert!(values.contains(&2));
    assert!(values.contains(&3));
}

#[test]
fn transaction_commit_abort() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Committed write persists
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("committed", &1u64).unwrap();
    }
    txn.commit().unwrap();

    // Aborted write discarded
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("aborted", &2u64).unwrap();
    }
    txn.abort().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(KV_TABLE).unwrap();
    assert!(t.get("committed").unwrap().is_some());
    assert!(t.get("aborted").unwrap().is_none());
}

#[test]
fn savepoint_rollback() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // First txn: insert baseline data
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("before_sp", &1u64).unwrap();
    }
    txn.commit().unwrap();

    // Second txn: savepoint at start, write, rollback
    let mut txn = db.begin_write().unwrap();
    let sp = txn.ephemeral_savepoint().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("after_sp", &2u64).unwrap();
    }
    txn.restore_savepoint(&sp).unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(KV_TABLE).unwrap();
    assert!(t.get("before_sp").unwrap().is_some());
    assert!(t.get("after_sp").unwrap().is_none());
}

#[test]
fn durability_modes() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Durability::None — not persisted until followed by Immediate
    let mut txn = db.begin_write().unwrap();
    txn.set_durability(Durability::None).unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("none_dur", &1u64).unwrap();
    }
    txn.commit().unwrap();

    // Durability::Immediate (default)
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("imm_dur", &2u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(KV_TABLE).unwrap();
    assert!(t.get("imm_dur").unwrap().is_some());
}

// ─── Blob Store ─────────────────────────────────────────────────────

#[test]
fn blob_write_read() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let data = b"hello blob world";

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(
            data,
            ContentType::OctetStream,
            "smoke",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let (read_data, meta) = txn.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(read_data, data);
    assert_eq!(meta.blob_ref.length, data.len() as u64);
}

#[test]
fn blob_dedup() {
    let tmpfile = create_tempfile();
    let mut builder = Database::builder();
    builder.set_blob_dedup(true);
    let db = builder.create(tmpfile.path()).unwrap();
    let data = vec![42u8; 8192];

    let txn = db.begin_write().unwrap();
    let id1 = txn
        .store_blob(
            &data,
            ContentType::OctetStream,
            "dup1",
            StoreOptions::default(),
        )
        .unwrap();
    let id2 = txn
        .store_blob(
            &data,
            ContentType::OctetStream,
            "dup2",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    // Dedup: same hash, physical data stored once
    // BlobId seq may differ, but underlying content is deduplicated
    let txn = db.begin_read().unwrap();
    let (data1, _) = txn.get_blob(&id1).unwrap().unwrap();
    let (data2, _) = txn.get_blob(&id2).unwrap().unwrap();
    assert_eq!(data1, data2);
    assert_eq!(data1, data);
    // Stats should show dedup: blob_count may be 2 (logical) but live_bytes == one copy
    let stats = txn.blob_stats().unwrap();
    assert_eq!(stats.live_bytes, data.len() as u64);
}

#[test]
fn blob_streaming_write() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let chunk = vec![0xABu8; 4096];

    let txn = db.begin_write().unwrap();
    let mut writer = txn
        .blob_writer(ContentType::OctetStream, "stream", StoreOptions::default())
        .unwrap();
    writer.write_all(&chunk).unwrap();
    writer.write_all(&chunk).unwrap();
    let blob_id = writer.finish().unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let (data, _meta) = txn.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(data.len(), 8192);
    assert!(data.iter().all(|&b| b == 0xAB));
}

#[test]
fn blob_range_read() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let data: Vec<u8> = (0..=255).cycle().take(4096).collect();

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(
            &data,
            ContentType::OctetStream,
            "range",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let slice = txn.read_blob_range(&blob_id, 100, 50).unwrap().unwrap();
    assert_eq!(slice.len(), 50);
    assert_eq!(&slice[..], &data[100..150]);
}

#[test]
fn blob_delete_compact() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();
    let data = vec![0u8; 4096];

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(
            &data,
            ContentType::OctetStream,
            "del",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    assert!(txn.delete_blob(&blob_id).unwrap());
    txn.commit().unwrap();

    let _report = db.compact_blobs().unwrap();

    let txn = db.begin_read().unwrap();
    assert!(txn.get_blob(&blob_id).unwrap().is_none());
}

// ─── Vector Operations ──────────────────────────────────────────────

#[test]
fn vector_distance_metrics() {
    let a: &[f32] = &[1.0, 0.0, 0.0, 0.0];
    let b: &[f32] = &[0.0, 1.0, 0.0, 0.0];

    let dot = shodh_redb::dot_product(a, b);
    assert!((dot - 0.0).abs() < 1e-6);

    let l2 = shodh_redb::euclidean_distance_sq(a, b);
    assert!((l2 - 2.0).abs() < 1e-6);

    let cos = shodh_redb::cosine_distance(a, b);
    assert!((cos - 1.0).abs() < 1e-6); // orthogonal => cosine_distance = 1.0

    let man = shodh_redb::manhattan_distance(a, b);
    assert!((man - 2.0).abs() < 1e-6);
}

#[test]
fn vector_quantization() {
    use shodh_redb::quantize_binary;

    let v = [1.0f32, -1.0, 0.5, -0.3, 2.0, 0.0, -0.1, 0.9];
    let bq: Vec<u8> = quantize_binary(&v);
    // 8 floats -> 1 byte (8 bits)
    assert_eq!(bq.len(), 1);
}

#[test]
fn vector_nearest_k() {
    use shodh_redb::nearest_k;

    let vectors: Vec<(u64, Vec<f32>)> = (0..100)
        .map(|i| {
            let v = vec![i as f32; 4];
            (i, v)
        })
        .collect();

    let query = vec![50.0f32; 4];
    let results = nearest_k(
        vectors.iter().map(|(id, v)| (*id, v.clone())),
        &query,
        5,
        shodh_redb::euclidean_distance_sq,
    );

    assert_eq!(results.len(), 5);
    assert_eq!(results[0].key, 50); // closest is exact match
}

// ─── TTL ────────────────────────────────────────────────────────────

const TTL_TABLE: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("smoke_ttl");

#[test]
fn ttl_insert_expire() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_ttl_table(TTL_TABLE).unwrap();
        t.insert_with_ttl("ephemeral", &42u64, Duration::from_secs(3600))
            .unwrap();
        t.insert("permanent", &99u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_ttl_table(TTL_TABLE).unwrap();
    assert!(t.get("ephemeral").unwrap().is_some());
    assert!(t.get("permanent").unwrap().is_some());
}

// ─── CDC ────────────────────────────────────────────────────────────

#[test]
fn cdc_capture_changes() {
    let tmpfile = create_tempfile();
    let db = Database::builder()
        .set_cdc(CdcConfig {
            enabled: true,
            retention_max_txns: 0,
        })
        .create(tmpfile.path())
        .unwrap();

    // Insert
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("cdc_key", &1u64).unwrap();
    }
    txn.commit().unwrap();

    // Update
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("cdc_key", &2u64).unwrap();
    }
    txn.commit().unwrap();

    // Delete
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.remove("cdc_key").unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let changes = txn.read_cdc_since(0).unwrap();
    assert!(changes.len() >= 3); // insert + update + delete
}

// ─── Merge Operators ────────────────────────────────────────────────

const MERGE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("smoke_merge");

#[test]
fn merge_numeric_add() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(MERGE_TABLE).unwrap();
        t.merge("counter", &10u64.to_le_bytes(), &NumericAdd)
            .unwrap();
        t.merge("counter", &20u64.to_le_bytes(), &NumericAdd)
            .unwrap();
        t.merge("counter", &12u64.to_le_bytes(), &NumericAdd)
            .unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(MERGE_TABLE).unwrap();
    let val = t.get("counter").unwrap().unwrap();
    let bytes = val.value();
    let result = u64::from_le_bytes(bytes.try_into().unwrap());
    assert_eq!(result, 42);
}

#[test]
fn merge_all_operators() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(MERGE_TABLE).unwrap();

        // NumericMax
        t.merge("max", &10u64.to_le_bytes(), &NumericMax).unwrap();
        t.merge("max", &50u64.to_le_bytes(), &NumericMax).unwrap();
        t.merge("max", &30u64.to_le_bytes(), &NumericMax).unwrap();

        // NumericMin
        t.merge("min", &50u64.to_le_bytes(), &NumericMin).unwrap();
        t.merge("min", &10u64.to_le_bytes(), &NumericMin).unwrap();
        t.merge("min", &30u64.to_le_bytes(), &NumericMin).unwrap();

        // BytesAppend
        t.merge("log", b"hello", &BytesAppend).unwrap();
        t.merge("log", b" world", &BytesAppend).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(MERGE_TABLE).unwrap();

    let max_val = u64::from_le_bytes(t.get("max").unwrap().unwrap().value().try_into().unwrap());
    assert_eq!(max_val, 50);

    let min_val = u64::from_le_bytes(t.get("min").unwrap().unwrap().value().try_into().unwrap());
    assert_eq!(min_val, 10);

    let log_val = t.get("log").unwrap().unwrap();
    assert_eq!(log_val.value(), b"hello world");
}

// ─── Group Commit ───────────────────────────────────���───────────────

#[test]
fn group_commit_batch() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let batch = WriteBatch::new(|txn| {
        let mut t = txn.open_table(KV_TABLE)?;
        t.insert("batch_key", &999u64)?;
        Ok(())
    });
    db.submit_write_batch(batch).unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(KV_TABLE).unwrap();
    assert_eq!(t.get("batch_key").unwrap().unwrap().value(), 999);
}

// ─── Compaction ─────────────────────────────────────────────────────

#[test]
fn blob_compaction_policy() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Fresh DB — no dead space
    assert!(db.should_compact_blobs().unwrap().is_none());

    // Write + delete to create dead space
    let txn = db.begin_write().unwrap();
    let mut ids = Vec::new();
    for i in 0..100u8 {
        let data = vec![i; 4096];
        let id = txn
            .store_blob(
                &data,
                ContentType::OctetStream,
                "compact",
                StoreOptions::default(),
            )
            .unwrap();
        ids.push(id);
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    for id in &ids {
        txn.delete_blob(id).unwrap();
    }
    txn.commit().unwrap();

    // Now should_compact_blobs may return Some (depends on threshold)
    let _stats_opt = db.should_compact_blobs().unwrap();
    // Even if below threshold, at least blob_stats works
    let txn = db.begin_read().unwrap();
    let stats = txn.blob_stats().unwrap();
    assert_eq!(stats.blob_count, 0);
}

#[test]
fn blob_compaction_handle() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Write some blobs then delete them
    let txn = db.begin_write().unwrap();
    let mut ids = Vec::new();
    for i in 0..10u8 {
        let data = vec![i; 4096];
        let id = txn
            .store_blob(
                &data,
                ContentType::OctetStream,
                "handle",
                StoreOptions::default(),
            )
            .unwrap();
        ids.push(id);
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    for id in &ids {
        txn.delete_blob(id).unwrap();
    }
    txn.commit().unwrap();

    let mut handle = db.start_blob_compaction().unwrap();
    let report = handle.run().unwrap();
    // Compaction completed (may or may not have relocated blobs)
    let _ = report.blobs_relocated;
    let _ = report.bytes_reclaimed;
}

// ─── Builder / Config ───────────────────────────────────────────────

#[test]
fn builder_all_options() {
    let tmpfile = create_tempfile();
    let mut builder = Database::builder();
    builder.set_cache_size(64 * 1024 * 1024);
    builder.set_blob_dedup(true);
    builder.set_memory_budget(32 * 1024 * 1024);
    let db = builder.create(tmpfile.path()).unwrap();

    // Verify it works
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("builder_test", &1u64).unwrap();
    }
    txn.commit().unwrap();
}

#[test]
fn open_existing_db() {
    let tmpfile = create_tempfile();

    // Create and write
    {
        let db = Database::create(tmpfile.path()).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(KV_TABLE).unwrap();
            t.insert("persist", &42u64).unwrap();
        }
        txn.commit().unwrap();
    }

    // Reopen and verify
    {
        let db = Database::open(tmpfile.path()).unwrap();
        let txn = db.begin_read().unwrap();
        let t = txn.open_table(KV_TABLE).unwrap();
        assert_eq!(t.get("persist").unwrap().unwrap().value(), 42);
    }
}

// ─── Error Paths ────────────────────────────────────────────────────

#[test]
fn table_does_not_exist() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_read().unwrap();
    let result = txn.open_table(KV_TABLE);
    assert!(result.is_err());
}

const WRONG_TYPE_TABLE: TableDefinition<u64, u64> = TableDefinition::new("smoke_kv");

#[test]
fn type_mismatch() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Create table with (&str, u64)
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("x", &1u64).unwrap();
    }
    txn.commit().unwrap();

    // Try to open with (u64, u64) — same name, wrong types
    let txn = db.begin_write().unwrap();
    let result = txn.open_table(WRONG_TYPE_TABLE);
    assert!(result.is_err());
}

// ─── Extended KV Tests ──────────────────────────────────────────────

const U64_TABLE: TableDefinition<u64, u64> = TableDefinition::new("smoke_u64");
const BYTES_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("smoke_bytes");

#[test]
fn kv_integer_keys() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..50u64 {
            t.insert(&i, &(i * i)).unwrap();
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 50);
    assert_eq!(t.get(&7u64).unwrap().unwrap().value(), 49);
}

#[test]
fn kv_byte_keys() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(BYTES_TABLE).unwrap();
        t.insert(b"key1".as_slice(), b"val1".as_slice()).unwrap();
        t.insert(b"key2".as_slice(), b"val2".as_slice()).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(BYTES_TABLE).unwrap();
    assert_eq!(t.get(b"key1".as_slice()).unwrap().unwrap().value(), b"val1");
}

#[test]
fn kv_range_bounded() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    let count = t.range(10u64..20u64).unwrap().count();
    assert_eq!(count, 10);
}

#[test]
fn kv_range_inclusive() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    let count = t.range(3u64..=7u64).unwrap().count();
    assert_eq!(count, 5);
}

#[test]
fn kv_empty_table() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        txn.open_table(KV_TABLE).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(KV_TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 0);
    assert!(t.get("nonexistent").unwrap().is_none());
}

#[test]
fn kv_overwrite_value() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("k", &1u64).unwrap();
        t.insert("k", &2u64).unwrap();
        t.insert("k", &3u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(KV_TABLE).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 3);
    assert_eq!(t.len().unwrap(), 1);
}

#[test]
fn kv_remove_nonexistent() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        let removed = t.remove("ghost").unwrap();
        assert!(removed.is_none());
    }
    txn.commit().unwrap();
}

#[test]
fn kv_large_values() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let big_val = vec![0xFFu8; 64 * 1024]; // 64 KB value

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(BYTES_TABLE).unwrap();
        t.insert(b"big".as_slice(), big_val.as_slice()).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(BYTES_TABLE).unwrap();
    assert_eq!(
        t.get(b"big".as_slice()).unwrap().unwrap().value().len(),
        64 * 1024
    );
}

#[test]
fn kv_many_inserts() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..10_000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 10_000);
    assert_eq!(t.get(&9999u64).unwrap().unwrap().value(), 9999);
}

#[test]
fn kv_first_last() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        t.insert(&5u64, &50u64).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&9u64, &90u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    let (first_k, first_v) = t.first().unwrap().unwrap();
    assert_eq!(first_k.value(), 1);
    assert_eq!(first_v.value(), 10);
    let (last_k, last_v) = t.last().unwrap().unwrap();
    assert_eq!(last_k.value(), 9);
    assert_eq!(last_v.value(), 90);
}

// ─── Extended Multimap Tests ────────────────────────────────────────

const MM_U64: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("smoke_mm_u64");

#[test]
fn multimap_remove_value() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_multimap_table(MM_TABLE).unwrap();
        t.insert("colors", &1u64).unwrap();
        t.insert("colors", &2u64).unwrap();
        t.insert("colors", &3u64).unwrap();
        t.remove("colors", &2u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_multimap_table(MM_TABLE).unwrap();
    let values: Vec<u64> = t
        .get("colors")
        .unwrap()
        .map(|r| r.unwrap().value())
        .collect();
    assert_eq!(values.len(), 2);
    assert!(!values.contains(&2));
}

#[test]
fn multimap_multiple_keys() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_multimap_table(MM_U64).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &11u64).unwrap();
        t.insert(&2u64, &20u64).unwrap();
        t.insert(&2u64, &21u64).unwrap();
        t.insert(&2u64, &22u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_multimap_table(MM_U64).unwrap();
    assert_eq!(t.get(&1u64).unwrap().count(), 2);
    assert_eq!(t.get(&2u64).unwrap().count(), 3);
}

#[test]
fn multimap_empty_key() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        txn.open_multimap_table(MM_TABLE).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_multimap_table(MM_TABLE).unwrap();
    assert_eq!(t.get("nonexistent").unwrap().count(), 0);
}

// ─── Extended Transaction Tests ─────────────────────────────────────

#[test]
fn multiple_tables_one_txn() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t1 = txn.open_table(KV_TABLE).unwrap();
        t1.insert("a", &1u64).unwrap();
        drop(t1);
        let mut t2 = txn.open_table(U64_TABLE).unwrap();
        t2.insert(&1u64, &100u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    assert_eq!(
        txn.open_table(KV_TABLE)
            .unwrap()
            .get("a")
            .unwrap()
            .unwrap()
            .value(),
        1
    );
    assert_eq!(
        txn.open_table(U64_TABLE)
            .unwrap()
            .get(&1u64)
            .unwrap()
            .unwrap()
            .value(),
        100
    );
}

#[test]
fn sequential_transactions() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    for i in 0..10u64 {
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(U64_TABLE).unwrap();
            t.insert(&i, &(i * 10)).unwrap();
        }
        txn.commit().unwrap();
    }

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 10);
}

#[test]
fn read_txn_snapshot_isolation() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("snap", &1u64).unwrap();
    }
    txn.commit().unwrap();

    // Start read txn
    let rtxn = db.begin_read().unwrap();
    let t = rtxn.open_table(KV_TABLE).unwrap();
    assert_eq!(t.get("snap").unwrap().unwrap().value(), 1);

    // Write in a new txn while read is open
    let wtxn = db.begin_write().unwrap();
    {
        let mut t = wtxn.open_table(KV_TABLE).unwrap();
        t.insert("snap", &2u64).unwrap();
    }
    wtxn.commit().unwrap();

    // Read txn still sees old value (snapshot isolation)
    assert_eq!(t.get("snap").unwrap().unwrap().value(), 1);
    drop(t);
    drop(rtxn);

    // New read sees updated value
    let rtxn = db.begin_read().unwrap();
    let t = rtxn.open_table(KV_TABLE).unwrap();
    assert_eq!(t.get("snap").unwrap().unwrap().value(), 2);
}

#[test]
fn drop_write_txn_without_commit() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("dropped", &1u64).unwrap();
    }
    drop(txn); // implicit abort

    let txn = db.begin_read().unwrap();
    let result = txn.open_table(KV_TABLE);
    // Table may not exist since we never committed
    assert!(result.is_err() || result.unwrap().get("dropped").unwrap().is_none());
}

// ─── Extended Blob Tests ────────────────────────────────────────────

#[test]
fn blob_metadata() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(
            b"meta test",
            ContentType::Embedding,
            "my-label",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let meta = txn.get_blob_meta(&blob_id).unwrap().unwrap();
    assert_eq!(meta.blob_ref.content_type, ContentType::Embedding as u8);
    let label = std::str::from_utf8(&meta.label[..meta.label_len as usize]).unwrap();
    assert_eq!(label, "my-label");
}

#[test]
fn blob_multiple_content_types() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let types = [
        ContentType::OctetStream,
        ContentType::ImagePng,
        ContentType::ImageJpeg,
        ContentType::Embedding,
    ];

    let txn = db.begin_write().unwrap();
    let mut ids = Vec::new();
    for ct in &types {
        let id = txn
            .store_blob(b"data", *ct, "ct-test", StoreOptions::default())
            .unwrap();
        ids.push(id);
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    for (id, expected_ct) in ids.iter().zip(types.iter()) {
        let meta = txn.get_blob_meta(id).unwrap().unwrap();
        assert_eq!(meta.blob_ref.content_type, *expected_ct as u8);
    }
}

#[test]
fn blob_empty() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(
            b"",
            ContentType::OctetStream,
            "empty",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let (data, meta) = txn.get_blob(&blob_id).unwrap().unwrap();
    assert!(data.is_empty());
    assert_eq!(meta.blob_ref.length, 0);
}

#[test]
fn blob_delete_nonexistent() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(b"x", ContentType::OctetStream, "t", StoreOptions::default())
        .unwrap();
    txn.commit().unwrap();

    // Delete once
    let txn = db.begin_write().unwrap();
    assert!(txn.delete_blob(&blob_id).unwrap());
    txn.commit().unwrap();

    // Delete again
    let txn = db.begin_write().unwrap();
    assert!(!txn.delete_blob(&blob_id).unwrap());
    txn.commit().unwrap();
}

#[test]
fn blob_stats_fresh_db() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_read().unwrap();
    let stats = txn.blob_stats().unwrap();
    assert_eq!(stats.blob_count, 0);
    assert_eq!(stats.live_bytes, 0);
    assert_eq!(stats.dead_bytes, 0);
}

#[test]
fn blob_range_read_full() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let data = vec![42u8; 1024];

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(
            &data,
            ContentType::OctetStream,
            "full",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    // Read entire blob via range
    let txn = db.begin_read().unwrap();
    let slice = txn.read_blob_range(&blob_id, 0, 1024).unwrap().unwrap();
    assert_eq!(slice, data);
}

// ─── Extended Vector Tests ──────────────────────────────────────────

#[test]
fn vector_dot_product_parallel() {
    let a: &[f32] = &[1.0, 2.0, 3.0, 4.0];
    let b: &[f32] = &[5.0, 6.0, 7.0, 8.0];
    let dot = shodh_redb::dot_product(a, b);
    assert!((dot - 70.0).abs() < 1e-4); // 5+12+21+32 = 70
}

#[test]
fn vector_cosine_identical() {
    let a: &[f32] = &[1.0, 2.0, 3.0];
    let dist = shodh_redb::cosine_distance(a, a);
    assert!(dist.abs() < 1e-5); // identical vectors => distance 0
}

#[test]
fn vector_cosine_opposite() {
    let a: &[f32] = &[1.0, 0.0, 0.0];
    let b: &[f32] = &[-1.0, 0.0, 0.0];
    let dist = shodh_redb::cosine_distance(a, b);
    assert!((dist - 2.0).abs() < 1e-5); // opposite => distance 2
}

#[test]
fn vector_l2_norm() {
    let v: &[f32] = &[3.0, 4.0];
    let norm = shodh_redb::l2_norm(v);
    assert!((norm - 5.0).abs() < 1e-5);
}

#[test]
fn vector_l2_normalize() {
    let mut v = vec![3.0f32, 4.0];
    shodh_redb::l2_normalize(&mut v);
    assert!((v[0] - 0.6).abs() < 1e-5);
    assert!((v[1] - 0.8).abs() < 1e-5);
}

#[test]
fn vector_manhattan_zero() {
    let a: &[f32] = &[1.0, 2.0, 3.0];
    let dist = shodh_redb::manhattan_distance(a, a);
    assert!(dist.abs() < 1e-6);
}

#[test]
fn vector_hamming() {
    let a: &[u8] = &[0xFF, 0x00];
    let b: &[u8] = &[0x00, 0xFF];
    let dist = shodh_redb::hamming_distance(a, b);
    assert_eq!(dist, 16); // all 16 bits differ
}

#[test]
fn vector_quantize_binary_roundtrip() {
    let v: Vec<f32> = (0..32)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let bq = shodh_redb::quantize_binary(&v);
    assert_eq!(bq.len(), 4); // 32 floats -> 4 bytes
}

#[test]
fn vector_nearest_k_top1() {
    use shodh_redb::nearest_k;

    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (0, vec![0.0, 0.0]),
        (1, vec![1.0, 0.0]),
        (2, vec![0.0, 1.0]),
    ];

    let query = vec![0.9f32, 0.0];
    let results = nearest_k(
        vectors.iter().map(|(id, v)| (*id, v.clone())),
        &query,
        1,
        shodh_redb::euclidean_distance_sq,
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].key, 1);
}

#[test]
fn vector_write_read_f32_le() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let mut buf = vec![0u8; 16];
    shodh_redb::write_f32_le(&mut buf, &values);
    let read_back = shodh_redb::read_f32_le(&buf);
    assert_eq!(read_back, values);
}

// ─── Extended TTL Tests ─────────────────────────────────────────────

#[test]
fn ttl_without_expiry() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_ttl_table(TTL_TABLE).unwrap();
        t.insert("no_ttl", &100u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_ttl_table(TTL_TABLE).unwrap();
    assert_eq!(t.get("no_ttl").unwrap().unwrap().value(), 100);
}

#[test]
fn ttl_purge_expired_empty() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_ttl_table(TTL_TABLE).unwrap();
        t.insert("x", &1u64).unwrap();
        let purged = t.purge_expired().unwrap();
        assert_eq!(purged, 0);
    }
    txn.commit().unwrap();
}

// ─── Extended CDC Tests ─────────────────────────────────────────────

#[test]
fn cdc_disabled_by_default() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("cdc_off", &1u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let changes = txn.read_cdc_since(0).unwrap();
    assert!(changes.is_empty());
}

#[test]
fn cdc_change_ops() {
    let tmpfile = create_tempfile();
    let db = Database::builder()
        .set_cdc(CdcConfig {
            enabled: true,
            retention_max_txns: 0,
        })
        .create(tmpfile.path())
        .unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("op_test", &1u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("op_test", &2u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.remove("op_test").unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let changes = txn.read_cdc_since(0).unwrap();
    let ops: Vec<_> = changes.iter().map(|c| c.op).collect();
    assert!(ops.contains(&ChangeOp::Insert));
    assert!(ops.contains(&ChangeOp::Update));
    assert!(ops.contains(&ChangeOp::Delete));
}

// ─── Extended Merge Tests ───────────────────────────────────────────

#[test]
fn merge_saturating_add() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(MERGE_TABLE).unwrap();
        t.merge("sat", &(u64::MAX - 5).to_le_bytes(), &SaturatingAdd)
            .unwrap();
        t.merge("sat", &10u64.to_le_bytes(), &SaturatingAdd)
            .unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(MERGE_TABLE).unwrap();
    let val = u64::from_le_bytes(t.get("sat").unwrap().unwrap().value().try_into().unwrap());
    assert_eq!(val, u64::MAX);
}

#[test]
fn merge_float_add() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(MERGE_TABLE).unwrap();
        t.merge("f", &1.5f64.to_le_bytes(), &FloatAdd).unwrap();
        t.merge("f", &2.5f64.to_le_bytes(), &FloatAdd).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(MERGE_TABLE).unwrap();
    let val = f64::from_le_bytes(t.get("f").unwrap().unwrap().value().try_into().unwrap());
    assert!((val - 4.0).abs() < 1e-10);
}

#[test]
fn merge_bitwise_or() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(MERGE_TABLE).unwrap();
        t.merge("bits", &[0b1010u8], &BitwiseOr).unwrap();
        t.merge("bits", &[0b0101u8], &BitwiseOr).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(MERGE_TABLE).unwrap();
    assert_eq!(t.get("bits").unwrap().unwrap().value(), &[0b1111u8]);
}

#[test]
fn merge_on_missing_key() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(MERGE_TABLE).unwrap();
        // First merge on nonexistent key — should create it
        t.merge("fresh", &42u64.to_le_bytes(), &NumericAdd).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(MERGE_TABLE).unwrap();
    let val = u64::from_le_bytes(t.get("fresh").unwrap().unwrap().value().try_into().unwrap());
    assert_eq!(val, 42);
}

// ─── Extended Builder/Config Tests ──────────────────────────────────

#[test]
fn builder_small_cache() {
    let tmpfile = create_tempfile();
    let mut builder = Database::builder();
    builder.set_cache_size(1024 * 1024); // 1 MB
    let db = builder.create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("small_cache", &1u64).unwrap();
    }
    txn.commit().unwrap();
}

#[test]
fn multiple_databases() {
    let tmp1 = create_tempfile();
    let tmp2 = create_tempfile();

    let db1 = Database::create(tmp1.path()).unwrap();
    let db2 = Database::create(tmp2.path()).unwrap();

    let txn = db1.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("db1", &1u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db2.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("db2", &2u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db1.begin_read().unwrap();
    let t = txn.open_table(KV_TABLE).unwrap();
    assert!(t.get("db2").unwrap().is_none());

    let txn = db2.begin_read().unwrap();
    let t = txn.open_table(KV_TABLE).unwrap();
    assert!(t.get("db1").unwrap().is_none());
}

#[test]
fn reopen_after_multiple_commits() {
    let tmpfile = create_tempfile();

    {
        let db = Database::create(tmpfile.path()).unwrap();
        for i in 0..5u64 {
            let txn = db.begin_write().unwrap();
            {
                let mut t = txn.open_table(U64_TABLE).unwrap();
                t.insert(&i, &(i * 100)).unwrap();
            }
            txn.commit().unwrap();
        }
    }

    {
        let db = Database::open(tmpfile.path()).unwrap();
        let txn = db.begin_read().unwrap();
        let t = txn.open_table(U64_TABLE).unwrap();
        assert_eq!(t.len().unwrap(), 5);
        assert_eq!(t.get(&4u64).unwrap().unwrap().value(), 400);
    }
}

// ─── Extended Group Commit Tests ────────────────────────────────────

#[test]
fn group_commit_multiple_batches() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    for i in 0..5u64 {
        let batch = WriteBatch::new(move |txn| {
            let mut t = txn.open_table(U64_TABLE)?;
            t.insert(&i, &(i * 10))?;
            Ok(())
        });
        db.submit_write_batch(batch).unwrap();
    }

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 5);
}

// ─── Extended Savepoint Tests ───────────────────────────────────────

#[test]
fn persistent_savepoint() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("before_persist", &1u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    let sp_id = txn.persistent_savepoint().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("after_persist", &2u64).unwrap();
    }
    txn.commit().unwrap();

    // Savepoint ID should be positive
    assert!(sp_id > 0);
}

// ─── Extended Compaction Tests ──────────────────────────────────────

#[test]
fn compact_empty_db() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let report = db.compact_blobs().unwrap();
    assert!(report.was_noop);
}

#[test]
fn blob_stats_after_writes() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    for _ in 0..5 {
        txn.store_blob(
            &[0u8; 1024],
            ContentType::OctetStream,
            "stats",
            StoreOptions::default(),
        )
        .unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let stats = txn.blob_stats().unwrap();
    assert_eq!(stats.blob_count, 5);
    assert!(stats.live_bytes > 0);
    assert_eq!(stats.dead_bytes, 0);
}

// ─── Extended Error Path Tests ──────────────────────────────────────

#[test]
fn read_blob_nonexistent() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Store one blob to initialize blob region
    let txn = db.begin_write().unwrap();
    txn.store_blob(b"x", ContentType::OctetStream, "t", StoreOptions::default())
        .unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    // Use BlobId::MIN which won't match any real blob
    let fake_id = shodh_redb::BlobId::MIN;
    let result = txn.get_blob(&fake_id).unwrap();
    assert!(result.is_none());
}

#[test]
fn multimap_wrong_type() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Create as regular table
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("x", &1u64).unwrap();
    }
    txn.commit().unwrap();

    // Try to open same name as multimap
    const BAD_MM: MultimapTableDefinition<&str, u64> = MultimapTableDefinition::new("smoke_kv");
    let txn = db.begin_write().unwrap();
    let result = txn.open_multimap_table(BAD_MM);
    assert!(result.is_err());
}

#[test]
fn database_stats() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("stats", &1u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    let stats = txn.stats().unwrap();
    assert!(stats.tree_height() >= 1);
    assert!(stats.allocated_pages() > 0);
}

// ─── List / Delete Tables ───────────────────────────────────────────

#[test]
fn list_tables_write_txn() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("x", &1u64).unwrap();
        drop(t);
        let mut t2 = txn.open_table(U64_TABLE).unwrap();
        t2.insert(&1u64, &2u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    let tables: Vec<_> = txn.list_tables().unwrap().collect();
    assert!(tables.len() >= 2);
}

#[test]
fn list_tables_read_txn() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("x", &1u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let tables: Vec<_> = txn.list_tables().unwrap().collect();
    assert!(!tables.is_empty());
}

#[test]
fn delete_table() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("del", &1u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    assert!(txn.delete_table(KV_TABLE).unwrap());
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    assert!(txn.open_table(KV_TABLE).is_err());
}

#[test]
fn delete_table_nonexistent() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    const GHOST: TableDefinition<&str, &str> = TableDefinition::new("ghost_table");
    let txn = db.begin_write().unwrap();
    assert!(!txn.delete_table(GHOST).unwrap());
    txn.commit().unwrap();
}

// ─── Check Integrity / Compact ──────────────────────────────────────

#[test]
fn check_integrity() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV_TABLE).unwrap();
        t.insert("integrity", &1u64).unwrap();
    }
    txn.commit().unwrap();

    let ok = db.check_integrity().unwrap();
    assert!(ok);
}

#[test]
fn compact_db() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    // Write and delete to create reclaimable space
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..100u64 {
            t.remove(&i).unwrap();
        }
    }
    txn.commit().unwrap();

    let compacted = db.compact().unwrap();
    let _ = compacted; // may or may not reclaim depending on page layout
}

// ─── Range / Iterator Tests ─────────────────────────────────────────

#[test]
fn kv_reverse_range() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &(i * 10)).unwrap();
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    let vals: Vec<u64> = t
        .range(3u64..7u64)
        .unwrap()
        .rev()
        .map(|r| r.unwrap().1.value())
        .collect();
    assert_eq!(vals, vec![60, 50, 40, 30]);
}

#[test]
fn kv_iter_all() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..5u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    let count = t.iter().unwrap().count();
    assert_eq!(count, 5);
}

#[test]
fn kv_drain() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        let count = t.drain::<u64>(..).unwrap();
        assert_eq!(count, 10);
        assert_eq!(t.len().unwrap(), 0);
    }
    txn.commit().unwrap();
}

#[test]
fn kv_drain_range() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        let count = t.drain(3u64..7u64).unwrap();
        assert_eq!(count, 4);
        assert_eq!(t.len().unwrap(), 6);
    }
    txn.commit().unwrap();
}

// ─── CDC Range Query ────────────────────────────────────────────────

#[test]
fn cdc_range_query() {
    let tmpfile = create_tempfile();
    let db = Database::builder()
        .set_cdc(CdcConfig {
            enabled: true,
            retention_max_txns: 0,
        })
        .create(tmpfile.path())
        .unwrap();

    // Commit 3 separate transactions
    for i in 0..3u64 {
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(U64_TABLE).unwrap();
            t.insert(&i, &i).unwrap();
        }
        txn.commit().unwrap();
    }

    let txn = db.begin_read().unwrap();
    // All changes from txn 0 onwards
    let all = txn.read_cdc_since(0).unwrap();
    assert!(all.len() >= 3);
}

// ─── Blob Writer Patterns ───────────────────────────────────────────

#[test]
fn blob_writer_single_byte_chunks() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    let mut writer = txn
        .blob_writer(
            ContentType::OctetStream,
            "byte-by-byte",
            StoreOptions::default(),
        )
        .unwrap();
    for b in b"hello" {
        writer.write_all(&[*b]).unwrap();
    }
    let blob_id = writer.finish().unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let (data, _) = txn.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(data, b"hello");
}

#[test]
fn blob_large_streaming() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let chunk = vec![0xABu8; 4096];

    let txn = db.begin_write().unwrap();
    let mut writer = txn
        .blob_writer(ContentType::OctetStream, "large", StoreOptions::default())
        .unwrap();
    for _ in 0..100 {
        writer.write_all(&chunk).unwrap();
    }
    let blob_id = writer.finish().unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let (data, meta) = txn.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(data.len(), 4096 * 100);
    assert_eq!(meta.blob_ref.length, 4096 * 100);
}

// ─── Distance Metric Enum Tests ─────────────────────────────────────

#[test]
fn distance_metric_cosine() {
    use shodh_redb::DistanceMetric;
    let a: &[f32] = &[1.0, 0.0, 0.0];
    let b: &[f32] = &[0.0, 1.0, 0.0];
    let dist = DistanceMetric::Cosine.compute(a, b);
    assert!((dist - 1.0).abs() < 1e-5); // orthogonal vectors = distance 1
}

#[test]
fn distance_metric_euclidean_sq() {
    use shodh_redb::DistanceMetric;
    let a: &[f32] = &[0.0, 0.0];
    let b: &[f32] = &[3.0, 4.0];
    let dist = DistanceMetric::EuclideanSq.compute(a, b);
    assert!((dist - 25.0).abs() < 1e-4);
}

#[test]
fn distance_metric_dot_product() {
    use shodh_redb::DistanceMetric;
    let a: &[f32] = &[1.0, 2.0, 3.0];
    let b: &[f32] = &[4.0, 5.0, 6.0];
    let dist = DistanceMetric::DotProduct.compute(a, b);
    // DotProduct distance = negative dot product (for min-heap ordering)
    let dot = shodh_redb::dot_product(a, b);
    assert!((dist - (-dot)).abs() < 1e-4);
}

#[test]
fn distance_metric_manhattan() {
    use shodh_redb::DistanceMetric;
    let a: &[f32] = &[1.0, 2.0, 3.0];
    let b: &[f32] = &[4.0, 6.0, 3.0];
    let dist = DistanceMetric::Manhattan.compute(a, b);
    assert!((dist - 7.0).abs() < 1e-4); // |3| + |4| + |0| = 7
}

// ─── Scalar Quantization Tests ──────────────────────────────────────

#[test]
fn scalar_quantize_and_distance() {
    let v: [f32; 16] = core::array::from_fn(|i| i as f32 / 16.0);
    let sq = shodh_redb::quantize_scalar(&v);
    assert_eq!(sq.codes.len(), 16); // one u8 per dimension
}

#[test]
fn scalar_quantize_identity_distance() {
    let v: [f32; 32] = [0.5; 32];
    let sq = shodh_redb::quantize_scalar(&v);
    let dist = shodh_redb::sq_euclidean_distance_sq(&v, &sq);
    assert!(dist < 1e-4); // identical => near-zero distance
}

// ─── Multimap Drain ─────────────────────────────────────────────────

#[test]
fn multimap_iter_all() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_multimap_table(MM_U64).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &20u64).unwrap();
        t.insert(&2u64, &30u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_multimap_table(MM_U64).unwrap();
    let count: usize = t.iter().unwrap().map(|r| r.unwrap().1.count()).sum();
    assert_eq!(count, 3);
}

// ─── TTL Expired Filter ─────────────────────────────────────────────

#[test]
fn ttl_expired_not_visible() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_ttl_table(TTL_TABLE).unwrap();
        // Insert with a 1ms TTL
        t.insert_with_ttl("ephemeral", &1u64, Duration::from_millis(1))
            .unwrap();
        t.insert("permanent", &2u64).unwrap();
    }
    txn.commit().unwrap();

    // Wait for expiry
    std::thread::sleep(Duration::from_millis(50));

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_ttl_table(TTL_TABLE).unwrap();
        let purged = t.purge_expired().unwrap();
        assert_eq!(purged, 1);
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_ttl_table(TTL_TABLE).unwrap();
    assert!(t.get("ephemeral").unwrap().is_none());
    assert_eq!(t.get("permanent").unwrap().unwrap().value(), 2);
}

// ─── Durability Eventual ────────────────────────────────────────────

#[test]
fn durability_eventual_multiple_writes() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    for i in 0..10u64 {
        let mut txn = db.begin_write().unwrap();
        txn.set_durability(Durability::None).unwrap();
        {
            let mut t = txn.open_table(U64_TABLE).unwrap();
            t.insert(&i, &i).unwrap();
        }
        txn.commit().unwrap();
    }

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 10);
}

// ─── Blob SHA-256 Verification ──────────────────────────────────────

#[test]
fn blob_sha256_field_exists() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(
            b"sha256 test data",
            ContentType::OctetStream,
            "sha",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let meta = txn.get_blob_meta(&blob_id).unwrap().unwrap();
    // SHA-256 field exists and is 32 bytes (may be zeroed for non-streaming writes)
    assert_eq!(meta.sha256.len(), 32);
}

// ─── Blob Dedup with Verification ───────────────────────────────────

#[test]
fn blob_dedup_saves_space() {
    let tmpfile = create_tempfile();
    let mut builder = Database::builder();
    builder.set_blob_dedup(true);
    let db = builder.create(tmpfile.path()).unwrap();

    let data = vec![0xCDu8; 8192];

    let txn = db.begin_write().unwrap();
    let id1 = txn
        .store_blob(
            &data,
            ContentType::OctetStream,
            "a",
            StoreOptions::default(),
        )
        .unwrap();
    let id2 = txn
        .store_blob(
            &data,
            ContentType::OctetStream,
            "b",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    // Both blobs should be readable
    let txn = db.begin_read().unwrap();
    let (d1, _) = txn.get_blob(&id1).unwrap().unwrap();
    let (d2, _) = txn.get_blob(&id2).unwrap().unwrap();
    assert_eq!(d1, d2);
    assert_eq!(d1, data);

    // With dedup, live_bytes should be ~1 copy, not 2
    let stats = txn.blob_stats().unwrap();
    assert!(stats.live_bytes < data.len() as u64 * 2);
}

// ─── Multimap Range ─────────────────────────────────────────────────

#[test]
fn multimap_range() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_multimap_table(MM_U64).unwrap();
        for k in 0..5u64 {
            for v in 0..3u64 {
                t.insert(&k, &(k * 10 + v)).unwrap();
            }
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_multimap_table(MM_U64).unwrap();
    let range_count: usize = t
        .range(1u64..4u64)
        .unwrap()
        .map(|r| r.unwrap().1.count())
        .sum();
    assert_eq!(range_count, 9); // 3 keys x 3 values each
}
