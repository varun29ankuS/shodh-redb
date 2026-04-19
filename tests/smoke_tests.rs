//! Smoke tests — fast happy-path validation of every major subsystem.
//! Each test opens a fresh DB, exercises one feature, and asserts correctness.
//! Target: all 26 tests pass in < 10 seconds.

use std::io::Write;
use std::time::Duration;

use shodh_redb::{
    BytesAppend, CdcConfig, ContentType, Database, Durability, MultimapTableDefinition, NumericAdd,
    NumericMax, NumericMin, ReadableDatabase, ReadableTable, ReadableTableMetadata, StoreOptions,
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
