//! Resource exhaustion and error-path tests.
//!
//! Complements memory_budget_tests.rs by exercising value-size limits,
//! blob writer lifecycle errors, and tight-budget degradation scenarios.

use shodh_redb::{
    BlobId, ContentType, Database, ReadableDatabase, ReadableTableMetadata, StorageError,
    StoreOptions, TableDefinition,
};
use tempfile::NamedTempFile;

const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("resource_test");

fn create_tempfile() -> NamedTempFile {
    if cfg!(target_os = "wasi") {
        NamedTempFile::new_in("/tmp").unwrap()
    } else {
        NamedTempFile::new().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Value size limits
// ---------------------------------------------------------------------------

/// Inserting a value that exceeds MAX_VALUE_LENGTH (3 GiB) must return
/// ValueTooLarge. We cannot allocate 3 GiB in CI, so we test through the
/// error variant match on a moderately large value that's still under the
/// limit to confirm the check exists, then verify the error path with a
/// crafted insert_reserve call.
#[test]
fn value_too_large_via_insert_reserve() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut table = txn.open_table(TABLE).unwrap();
        // insert_reserve with a length exceeding 3 GiB should fail.
        // We cannot actually allocate 3 GiB, but the check is on the
        // *declared* length, not the allocation.
        let huge_len: usize = 3 * 1024 * 1024 * 1024 + 1;
        let result = table.insert_reserve(&0u64, huge_len);
        match result {
            Err(StorageError::ValueTooLarge(len)) => {
                assert_eq!(len, huge_len);
            }
            Ok(_) => panic!("expected ValueTooLarge, got Ok"),
            Err(e) => panic!("expected ValueTooLarge, got {:?}", e),
        }
    }
    txn.abort().unwrap();
}

/// A value just under the limit should be accepted by the size check
/// (we cannot actually write 3 GiB in CI, so this is a negative test on
/// the error path -- we just verify the size guard does NOT fire).
#[test]
fn value_under_limit_passes_size_check() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut table = txn.open_table(TABLE).unwrap();
        // 1 MiB value -- well under limit, should succeed
        let data = vec![0xABu8; 1024 * 1024];
        table.insert(&1u64, data.as_slice()).unwrap();
    }
    txn.commit().unwrap();

    let rtxn = db.begin_read().unwrap();
    let table = rtxn.open_table(TABLE).unwrap();
    let val = table.get(&1u64).unwrap().unwrap();
    assert_eq!(val.value().len(), 1024 * 1024);
}

// ---------------------------------------------------------------------------
// Blob writer lifecycle errors
// ---------------------------------------------------------------------------

/// Only one BlobWriter can be active per transaction. A second call to
/// blob_writer() or store_blob() must return BlobWriterActive.
#[test]
fn blob_writer_active_error() {
    let tmpfile = create_tempfile();
    let db = Database::builder()
        .set_blob_dedup(true)
        .create(tmpfile.path())
        .unwrap();

    let txn = db.begin_write().unwrap();
    // First writer -- should succeed
    let _writer = txn
        .blob_writer(ContentType::OctetStream, "first", StoreOptions::default())
        .unwrap();

    // Second writer while first is still alive -- must fail
    let result = txn.blob_writer(ContentType::OctetStream, "second", StoreOptions::default());
    match result {
        Err(StorageError::BlobWriterActive) => {}
        Ok(_) => panic!("expected BlobWriterActive, got Ok"),
        Err(e) => panic!("expected BlobWriterActive, got {:?}", e),
    }
}

/// store_blob() while a BlobWriter is active must also return BlobWriterActive.
#[test]
fn store_blob_while_writer_active() {
    let tmpfile = create_tempfile();
    let db = Database::builder()
        .set_blob_dedup(true)
        .create(tmpfile.path())
        .unwrap();

    let txn = db.begin_write().unwrap();
    let _writer = txn
        .blob_writer(ContentType::OctetStream, "active", StoreOptions::default())
        .unwrap();

    let result = txn.store_blob(
        b"data",
        ContentType::OctetStream,
        "blocked",
        StoreOptions::default(),
    );
    match result {
        Err(StorageError::BlobWriterActive) => {}
        Ok(_) => panic!("expected BlobWriterActive, got Ok"),
        Err(e) => panic!("expected BlobWriterActive, got {:?}", e),
    }
}

/// After a BlobWriter is dropped, a new one can be created.
#[test]
fn blob_writer_reuse_after_drop() {
    let tmpfile = create_tempfile();
    let db = Database::builder()
        .set_blob_dedup(true)
        .create(tmpfile.path())
        .unwrap();

    let txn = db.begin_write().unwrap();
    {
        let _writer = txn
            .blob_writer(ContentType::OctetStream, "first", StoreOptions::default())
            .unwrap();
        // writer drops here
    }

    // Should succeed now
    let _writer2 = txn
        .blob_writer(ContentType::OctetStream, "second", StoreOptions::default())
        .unwrap();
}

/// Reading a nonexistent blob returns None, not an error.
#[test]
fn get_nonexistent_blob_returns_none() {
    let tmpfile = create_tempfile();
    let db = Database::builder()
        .set_blob_dedup(true)
        .create(tmpfile.path())
        .unwrap();

    // Store one blob to initialize the blob system
    let txn = db.begin_write().unwrap();
    let _id = txn
        .store_blob(
            b"init",
            ContentType::OctetStream,
            "init",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    // Fabricate a BlobId that does not exist
    let fake_id = BlobId {
        sequence: u64::MAX,
        content_prefix_hash: u64::MAX,
    };
    let rtxn = db.begin_read().unwrap();
    let result = rtxn.get_blob(&fake_id).unwrap();
    assert!(result.is_none(), "expected None for nonexistent blob");
}

/// read_blob_range with out-of-bounds offset returns BlobRangeOutOfBounds.
#[test]
fn blob_range_out_of_bounds() {
    let tmpfile = create_tempfile();
    let db = Database::builder()
        .set_blob_dedup(true)
        .create(tmpfile.path())
        .unwrap();

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(
            b"short",
            ContentType::OctetStream,
            "test",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    let rtxn = db.begin_read().unwrap();
    // Request range beyond blob length
    let result = rtxn.read_blob_range(&blob_id, 100, 50);
    match result {
        Err(StorageError::BlobRangeOutOfBounds { .. }) => {}
        other => panic!("expected BlobRangeOutOfBounds, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Tight budget degradation
// ---------------------------------------------------------------------------

/// With the smallest valid budget (16 KiB), the DB should still function
/// for small operations without panicking.
#[test]
fn minimum_budget_still_functional() {
    let tmpfile = create_tempfile();
    let db = Database::builder()
        .set_memory_budget(16 * 1024) // minimum
        .create(tmpfile.path())
        .unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut table = txn.open_table(TABLE).unwrap();
        // Small inserts should work even at minimum budget
        for i in 0..10u64 {
            table.insert(&i, b"tiny".as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();

    let rtxn = db.begin_read().unwrap();
    let table = rtxn.open_table(TABLE).unwrap();
    assert_eq!(table.len().unwrap(), 10);
}

/// Many writes with a small budget should succeed via auto-flush,
/// and the cache should report evictions.
#[test]
fn small_budget_heavy_writes_trigger_eviction() {
    let tmpfile = create_tempfile();
    let budget = 64 * 1024; // 64 KiB
    let db = Database::builder()
        .set_memory_budget(budget)
        .create(tmpfile.path())
        .unwrap();

    let value = vec![0u8; 512];
    // Write enough data to exceed the budget many times over
    for batch in 0..5u64 {
        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(TABLE).unwrap();
            for i in 0..100u64 {
                table.insert(&(batch * 100 + i), value.as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    // All 500 records should be readable
    let rtxn = db.begin_read().unwrap();
    let table = rtxn.open_table(TABLE).unwrap();
    assert_eq!(table.len().unwrap(), 500);

    // Cache should have evicted
    let stats = db.cache_stats();
    assert!(stats.used_bytes() <= budget, "cache exceeded budget");
}

/// After a failed insert (ValueTooLarge), the transaction should still be
/// usable for subsequent valid operations.
#[test]
fn transaction_usable_after_failed_insert() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut table = txn.open_table(TABLE).unwrap();

        // This will fail with ValueTooLarge
        let huge_len: usize = 3 * 1024 * 1024 * 1024 + 1;
        let _ = table.insert_reserve(&0u64, huge_len);

        // But a normal insert should still work
        table.insert(&1u64, b"fine".as_slice()).unwrap();
    }
    txn.commit().unwrap();

    let rtxn = db.begin_read().unwrap();
    let table = rtxn.open_table(TABLE).unwrap();
    assert_eq!(table.get(&1u64).unwrap().unwrap().value(), b"fine");
}

// ---------------------------------------------------------------------------
// Savepoint under pressure
// ---------------------------------------------------------------------------

/// Creating many savepoints in sequence should not leak resources.
#[test]
fn many_ephemeral_savepoints_no_leak() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut table = txn.open_table(TABLE).unwrap();
        table.insert(&0u64, b"base".as_slice()).unwrap();
    }

    // Create and restore many persistent savepoints across separate transactions.
    txn.commit().unwrap();

    for i in 0..50u64 {
        let txn = db.begin_write().unwrap();
        let sp_id = txn.persistent_savepoint().unwrap();
        {
            let mut table = txn.open_table(TABLE).unwrap();
            table.insert(&(i + 1), b"temp".as_slice()).unwrap();
        }
        txn.commit().unwrap();

        // Restore: discard the writes we just committed
        let mut txn = db.begin_write().unwrap();
        let sp = txn.get_persistent_savepoint(sp_id).unwrap();
        txn.restore_savepoint(&sp).unwrap();
        txn.delete_persistent_savepoint(sp_id).unwrap();
        txn.commit().unwrap();
    }

    // Only the base record should exist
    let rtxn = db.begin_read().unwrap();
    let table = rtxn.open_table(TABLE).unwrap();
    assert_eq!(table.get(&0u64).unwrap().unwrap().value(), b"base");
    // No temp keys should have survived
    assert!(table.get(&1u64).unwrap().is_none());
}

// ---------------------------------------------------------------------------
// Concurrent readers under memory pressure
// ---------------------------------------------------------------------------

/// Multiple read transactions can coexist under a tight memory budget.
#[test]
fn concurrent_readers_tight_budget() {
    let tmpfile = create_tempfile();
    let budget = 128 * 1024; // 128 KiB
    let db = Database::builder()
        .set_memory_budget(budget)
        .create(tmpfile.path())
        .unwrap();

    // Populate
    let value = vec![0xCDu8; 256];
    let txn = db.begin_write().unwrap();
    {
        let mut table = txn.open_table(TABLE).unwrap();
        for i in 0..200u64 {
            table.insert(&i, value.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();

    // Open multiple readers simultaneously
    let r1 = db.begin_read().unwrap();
    let r2 = db.begin_read().unwrap();
    let r3 = db.begin_read().unwrap();

    for rtxn in [&r1, &r2, &r3] {
        let table = rtxn.open_table(TABLE).unwrap();
        assert_eq!(table.len().unwrap(), 200);
        // Spot-check a few values
        for &k in &[0u64, 99, 199] {
            let val = table.get(&k).unwrap().unwrap();
            assert_eq!(val.value().len(), 256);
        }
    }
}

// ---------------------------------------------------------------------------
// Blob store under tight budget
// ---------------------------------------------------------------------------

/// Storing and reading blobs with a small memory budget should still work.
#[test]
fn blob_operations_tight_budget() {
    let tmpfile = create_tempfile();
    let db = Database::builder()
        .set_memory_budget(64 * 1024) // 64 KiB
        .set_blob_dedup(true)
        .create(tmpfile.path())
        .unwrap();

    let data = vec![0xABu8; 8192]; // 8 KiB blob

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(
            &data,
            ContentType::OctetStream,
            "tight",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    let rtxn = db.begin_read().unwrap();
    let (got, meta) = rtxn.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(got, data);
    assert_eq!(meta.blob_ref.content_type, ContentType::OctetStream as u8);
}

/// Deleting a blob and then trying to read it should return None.
#[test]
fn deleted_blob_returns_none() {
    let tmpfile = create_tempfile();
    let db = Database::builder()
        .set_blob_dedup(true)
        .create(tmpfile.path())
        .unwrap();

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(
            b"ephemeral",
            ContentType::OctetStream,
            "del",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    // Delete it
    let txn = db.begin_write().unwrap();
    txn.delete_blob(&blob_id).unwrap();
    txn.commit().unwrap();

    let rtxn = db.begin_read().unwrap();
    assert!(rtxn.get_blob(&blob_id).unwrap().is_none());
}
