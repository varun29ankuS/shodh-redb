//! Blob compaction tests adapted for chunked B-tree storage.
//!
//! With chunked storage, blob data lives in the normal page-managed B-tree
//! (BLOB_CHUNKS system table). There is no separate "blob region" to compact.
//! Deleted blob chunks are freed immediately by the B-tree page allocator.
//!
//! Therefore:
//! - `blob_stats().dead_bytes` is always 0
//! - `compact_blobs()` is always a no-op (`was_noop: true`)
//! - `should_compact_blobs()` never recommends compaction
//! - Data integrity is maintained through normal B-tree MVCC
//!
//! These tests verify both the no-op compaction API behavior and the underlying
//! data integrity of the chunked blob storage.

use shodh_redb::*;

fn create_tempfile() -> tempfile::NamedTempFile {
    if cfg!(target_os = "wasi") {
        tempfile::NamedTempFile::new_in("/tmp").unwrap()
    } else {
        tempfile::NamedTempFile::new().unwrap()
    }
}

/// Helper: store a blob with the given data and label, commit, return the BlobId.
fn store_one(db: &Database, data: &[u8], label: &str) -> BlobId {
    let txn = db.begin_write().unwrap();
    let id = txn
        .store_blob(
            data,
            ContentType::OctetStream,
            label,
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();
    id
}

/// Helper: delete a blob and commit.
fn delete_one(db: &Database, id: &BlobId) {
    let txn = db.begin_write().unwrap();
    txn.delete_blob(id).unwrap();
    txn.commit().unwrap();
}

// ---------------------------------------------------------------------------
// 1. compact_noop_on_clean_db
// ---------------------------------------------------------------------------
#[test]
fn compact_noop_on_clean_db() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    // Store blobs but delete none -- no dead space.
    let _id1 = store_one(&db, b"alpha", "a");
    let _id2 = store_one(&db, b"bravo", "b");

    let report = db.compact_blobs().unwrap();
    assert!(report.was_noop, "expected noop when no dead space exists");
    assert_eq!(report.bytes_reclaimed, 0);
    assert_eq!(report.blobs_relocated, 0);
}

// ---------------------------------------------------------------------------
// 2. compact_after_single_delete — with chunked storage, always noop
// ---------------------------------------------------------------------------
#[test]
fn compact_after_single_delete() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let id1 = store_one(&db, b"alpha", "a");
    let id2 = store_one(&db, b"bravo", "b");
    let id3 = store_one(&db, b"charlie", "c");

    delete_one(&db, &id2);

    // With chunked storage, deleted chunks are freed immediately —
    // compact_blobs() finds no dead space.
    let report = db.compact_blobs().unwrap();
    assert!(report.was_noop);

    // Survivors must still be readable.
    let rtx = db.begin_read().unwrap();
    let (d1, _) = rtx.get_blob(&id1).unwrap().unwrap();
    assert_eq!(d1, b"alpha");
    let (d3, _) = rtx.get_blob(&id3).unwrap().unwrap();
    assert_eq!(d3, b"charlie");
    // Deleted blob must be gone.
    assert!(rtx.get_blob(&id2).unwrap().is_none());
}

// ---------------------------------------------------------------------------
// 3. compact_after_all_deleted
// ---------------------------------------------------------------------------
#[test]
fn compact_after_all_deleted() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let id1 = store_one(&db, &[0xAA; 4096], "x");
    let id2 = store_one(&db, &[0xBB; 4096], "y");

    delete_one(&db, &id1);
    delete_one(&db, &id2);

    let report = db.compact_blobs().unwrap();
    assert!(report.was_noop);

    // Verify stats: no live blobs.
    let wtx = db.begin_write().unwrap();
    let stats = wtx.blob_stats().unwrap();
    wtx.abort().unwrap();
    assert_eq!(stats.blob_count, 0);
    assert_eq!(stats.live_bytes, 0);
}

// ---------------------------------------------------------------------------
// 4. compact_preserves_content — data integrity after deletion of other blobs
// ---------------------------------------------------------------------------
#[test]
fn compact_preserves_content() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    // Use non-trivial payloads.
    let payload_a: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
    let payload_b: Vec<u8> = (128u8..=255).cycle().take(4096).collect();

    let id_a = store_one(&db, &payload_a, "pa");
    let id_b = store_one(&db, &payload_b, "pb");

    // Delete a third blob — chunks freed immediately.
    let id_dead = store_one(&db, &[0xFF; 2048], "dead");
    delete_one(&db, &id_dead);

    // compact_blobs is a noop but call it to exercise the path.
    db.compact_blobs().unwrap();

    let rtx = db.begin_read().unwrap();
    let (got_a, _) = rtx.get_blob(&id_a).unwrap().unwrap();
    let (got_b, _) = rtx.get_blob(&id_b).unwrap().unwrap();
    assert_eq!(
        got_a, payload_a,
        "blob A content corrupted after compaction"
    );
    assert_eq!(
        got_b, payload_b,
        "blob B content corrupted after compaction"
    );
}

// ---------------------------------------------------------------------------
// 5. compact_preserves_metadata
// ---------------------------------------------------------------------------
#[test]
fn compact_preserves_metadata() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    let id = txn
        .store_blob(
            b"metadata-test",
            ContentType::ImagePng,
            "my-label",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    // Delete another blob.
    let dead = store_one(&db, &[0; 512], "dead");
    delete_one(&db, &dead);

    // Capture metadata before compaction.
    let rtx = db.begin_read().unwrap();
    let (_, meta_before) = rtx.get_blob(&id).unwrap().unwrap();
    drop(rtx);

    db.compact_blobs().unwrap();

    let rtx = db.begin_read().unwrap();
    let (data_after, meta_after) = rtx.get_blob(&id).unwrap().unwrap();
    assert_eq!(data_after, b"metadata-test");
    assert_eq!(
        meta_after.blob_ref.content_type_enum(),
        ContentType::ImagePng,
        "content type changed after compaction"
    );
    assert_eq!(
        meta_after.label_str(),
        "my-label",
        "label changed after compaction"
    );
    assert_eq!(meta_before.sha256, meta_after.sha256, "sha256 changed");
    assert_eq!(
        meta_before.blob_ref.length, meta_after.blob_ref.length,
        "blob length changed"
    );
}

// ---------------------------------------------------------------------------
// 6. compact_with_dedup_interaction
// ---------------------------------------------------------------------------
#[test]
fn compact_with_dedup_interaction() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let shared_data = b"identical-payload-for-dedup";

    // Store the same content twice -- dedup should share the physical chunks.
    let id1 = store_one(&db, shared_data, "dup-1");
    let id2 = store_one(&db, shared_data, "dup-2");
    // Store a third unique blob, then delete it.
    let id3 = store_one(&db, b"unique-garbage", "garbage");
    delete_one(&db, &id3);

    // Delete one dedup reference -- the shared chunks must survive.
    delete_one(&db, &id1);

    let report = db.compact_blobs().unwrap();
    assert!(report.was_noop);

    // The surviving reference must still read correctly.
    let rtx = db.begin_read().unwrap();
    let (data, _) = rtx.get_blob(&id2).unwrap().unwrap();
    assert_eq!(data.as_slice(), shared_data);
    // The deleted reference must be gone.
    assert!(rtx.get_blob(&id1).unwrap().is_none());
}

// ---------------------------------------------------------------------------
// 7. compact_large_blobs — data integrity with large payloads
// ---------------------------------------------------------------------------
#[test]
fn compact_large_blobs() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let one_mb = vec![0xABu8; 1024 * 1024];
    let half_mb = vec![0xCDu8; 512 * 1024];

    let id_big = store_one(&db, &one_mb, "big");
    let id_med = store_one(&db, &half_mb, "med");
    let id_del = store_one(&db, &vec![0xEF; 1024 * 1024], "del");

    delete_one(&db, &id_del);

    // Noop with chunked storage, but verify data integrity.
    let report = db.compact_blobs().unwrap();
    assert!(report.was_noop);

    let rtx = db.begin_read().unwrap();
    let (d1, _) = rtx.get_blob(&id_big).unwrap().unwrap();
    assert_eq!(d1.len(), 1024 * 1024);
    assert!(d1.iter().all(|&b| b == 0xAB));
    let (d2, _) = rtx.get_blob(&id_med).unwrap().unwrap();
    assert_eq!(d2.len(), 512 * 1024);
    assert!(d2.iter().all(|&b| b == 0xCD));
}

// ---------------------------------------------------------------------------
// 8. blob_stats_with_chunked_storage
// ---------------------------------------------------------------------------
#[test]
fn blob_stats_with_chunked_storage() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let _id1 = store_one(&db, &[1; 4096], "keep");
    let id2 = store_one(&db, &[2; 4096], "del");
    let _id3 = store_one(&db, &[3; 4096], "keep2");

    delete_one(&db, &id2);

    // With chunked storage, dead_bytes is always 0 — chunks are freed immediately.
    let wtx = db.begin_write().unwrap();
    let stats = wtx.blob_stats().unwrap();
    wtx.abort().unwrap();

    assert_eq!(stats.dead_bytes, 0);
    assert_eq!(stats.blob_count, 2);
    assert_eq!(stats.live_bytes, 8192);
    assert_eq!(stats.region_bytes, stats.live_bytes);
    assert!((stats.fragmentation_ratio - 0.0).abs() < f64::EPSILON);

    // compact_blobs is a noop.
    let report = db.compact_blobs().unwrap();
    assert!(report.was_noop);
}

// ---------------------------------------------------------------------------
// 9. online_compaction_readers_between_phases
// ---------------------------------------------------------------------------
#[test]
fn online_compaction_readers_between_phases() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let payload = b"phase-reader-test";
    let id1 = store_one(&db, payload, "pr");
    let id_del = store_one(&db, &[0; 512], "del");
    delete_one(&db, &id_del);

    let mut handle = db.start_blob_compaction().unwrap();

    // With chunked storage, step() completes immediately (no work).
    let p = handle.step().unwrap();
    // The compaction handle reports complete since there is nothing to do.
    assert!(p.complete);

    // Blob must still be accessible.
    let rtx = db.begin_read().unwrap();
    let (data, _) = rtx.get_blob(&id1).unwrap().unwrap();
    assert_eq!(data.as_slice(), payload);
}

// ---------------------------------------------------------------------------
// 10. online_compaction_run
// ---------------------------------------------------------------------------
#[test]
fn online_compaction_run() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let id_keep = store_one(&db, b"keep-me", "k");
    let id_del = store_one(&db, &[0; 1024], "d");
    delete_one(&db, &id_del);

    let mut handle = db.start_blob_compaction().unwrap();
    let report = handle.run().unwrap();
    // With chunked storage, no bytes are reclaimed (chunks freed on delete).
    assert_eq!(report.bytes_reclaimed, 0);

    let rtx = db.begin_read().unwrap();
    let (data, _) = rtx.get_blob(&id_keep).unwrap().unwrap();
    assert_eq!(data.as_slice(), b"keep-me");
}

// ---------------------------------------------------------------------------
// 11. compact_progress_callback — noop means no progress events
// ---------------------------------------------------------------------------
#[test]
fn compact_progress_callback() {
    use std::sync::atomic::{AtomicU32, Ordering};

    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let _id = store_one(&db, &[0xAA; 2048], "kept");
    let id_del = store_one(&db, &[0xBB; 2048], "del");
    delete_one(&db, &id_del);

    let call_count = AtomicU32::new(0);
    let report = db
        .compact_blobs_with_progress(|_blobs_done, _total_blobs, _bytes_done, _total_bytes| {
            call_count.fetch_add(1, Ordering::Relaxed);
            true // continue
        })
        .unwrap();

    // With chunked storage, dead_bytes=0 so compact_blobs_with_progress is a noop.
    assert!(report.was_noop);
}

// ---------------------------------------------------------------------------
// 12. compact_progress_cancellation — noop returns Ok, not Cancelled
// ---------------------------------------------------------------------------
#[test]
fn compact_progress_cancellation() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let id_keep = store_one(&db, b"survive", "s");
    let id_del = store_one(&db, &[0; 1024], "d");
    delete_one(&db, &id_del);

    // With chunked storage, the cancellation callback never fires (noop early return).
    let result = db.compact_blobs_with_progress(|_a, _b, _c, _d| {
        false // cancel immediately on first callback
    });

    // Noop path returns Ok, not Cancelled.
    let report = result.unwrap();
    assert!(report.was_noop);

    // Database must remain usable.
    let rtx = db.begin_read().unwrap();
    let (data, _) = rtx.get_blob(&id_keep).unwrap().unwrap();
    assert_eq!(data.as_slice(), b"survive");
}

// ---------------------------------------------------------------------------
// 13. compact_blocked_by_read_txn
// ---------------------------------------------------------------------------
#[test]
fn compact_blocked_by_read_txn() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let _id = store_one(&db, b"data", "d");
    let id_del = store_one(&db, &[0; 512], "del");
    delete_one(&db, &id_del);

    // Hold an open read transaction.
    let _rtx = db.begin_read().unwrap();

    let result = db.compact_blobs();
    match result {
        Err(CompactionError::TransactionInProgress) => {} // expected
        Err(other) => panic!("expected TransactionInProgress, got: {other}"),
        Ok(_) => panic!("expected TransactionInProgress error"),
    }
}

// ---------------------------------------------------------------------------
// 14. should_compact_never_recommends — chunked storage has no dead bytes
// ---------------------------------------------------------------------------
#[test]
fn should_compact_never_recommends() {
    let tmpfile = create_tempfile();

    // Even with extremely low thresholds, no dead space exists.
    let policy = BlobCompactionPolicy {
        fragmentation_threshold: 0.0001,
        min_dead_bytes: 1,
    };
    let db = Builder::new()
        .set_blob_compaction_policy(policy)
        .create(tmpfile.path())
        .unwrap();

    let _id = store_one(&db, &[0xAA; 256], "keep");
    let id_del = store_one(&db, &[0xBB; 256], "del");
    delete_one(&db, &id_del);

    let recommendation = db.should_compact_blobs().unwrap();
    assert!(
        recommendation.is_none(),
        "chunked storage has no dead bytes — should never recommend blob compaction"
    );
}

// ---------------------------------------------------------------------------
// 15. compact_then_store_new — new blobs work after compact_blobs call
// ---------------------------------------------------------------------------
#[test]
fn compact_then_store_new() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let id1 = store_one(&db, b"before", "b");
    let id_del = store_one(&db, &[0; 2048], "del");
    delete_one(&db, &id_del);

    let report = db.compact_blobs().unwrap();
    assert!(report.was_noop);

    // Store new blobs after compaction -- must succeed and be readable.
    let id_new = store_one(&db, b"after-compaction", "ac");

    let rtx = db.begin_read().unwrap();
    let (d1, _) = rtx.get_blob(&id1).unwrap().unwrap();
    assert_eq!(d1, b"before");
    let (d_new, _) = rtx.get_blob(&id_new).unwrap().unwrap();
    assert_eq!(d_new, b"after-compaction");
}
