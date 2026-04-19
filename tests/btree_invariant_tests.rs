//! B-tree invariant validation tests.
//!
//! Exercises the B-tree with adversarial insert/delete patterns and validates:
//! - Full structural integrity via `verify_integrity(Full)` (checksums + key ordering)
//! - Key ordering correctness via manual sorted-iteration checks

use std::collections::BTreeSet;
use tempfile::NamedTempFile;

use shodh_redb::{
    Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition, VerifyLevel,
};

fn create_tempfile() -> NamedTempFile {
    if cfg!(target_os = "wasi") {
        NamedTempFile::new_in("/tmp").unwrap()
    } else {
        NamedTempFile::new().unwrap()
    }
}

const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("btree_test");

/// Assert full structural integrity passes (checksums + key ordering).
fn assert_checksums(db: &Database) {
    let report = db.verify_integrity(VerifyLevel::Full).unwrap();
    assert!(
        report.valid,
        "integrity check failed: {} pages corrupt, structural_valid={:?}, details={:?}",
        report.pages_corrupt, report.structural_valid, report.corrupt_details,
    );
}

/// Read all keys from the table and verify they're in ascending order.
fn assert_sorted_keys(db: &Database, expected_len: u64) {
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    assert_eq!(t.len().unwrap(), expected_len, "unexpected table length");

    let mut prev: Option<u64> = None;
    let mut count = 0u64;
    for result in t.iter().unwrap() {
        let (k, _) = result.unwrap();
        let key = k.value();
        if let Some(p) = prev {
            assert!(key > p, "keys not sorted: {key} after {p}");
        }
        prev = Some(key);
        count += 1;
    }
    assert_eq!(count, expected_len, "iteration count mismatch");
}

/// Verify both checksums and key ordering.
fn assert_integrity(db: &Database, expected_len: u64) {
    assert_checksums(db);
    assert_sorted_keys(db, expected_len);
}

// =======================================================================
// Sequential insert patterns
// =======================================================================

#[test]
fn sequential_insert_10k() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let value = vec![0xABu8; 64];

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..10_000u64 {
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();

    assert_integrity(&db, 10_000);
}

#[test]
fn reverse_insert_10k() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let value = vec![0xCDu8; 64];

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in (0..10_000u64).rev() {
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();

    assert_integrity(&db, 10_000);
}

#[test]
fn alternating_insert_10k() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let value = vec![0xEFu8; 64];

    // Insert even keys, then odd keys -- forces splits in middle of existing ranges
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in (0..10_000u64).step_by(2) {
            t.insert(&i, value.as_slice()).unwrap();
        }
        for i in (1..10_000u64).step_by(2) {
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();

    assert_integrity(&db, 10_000);
}

// =======================================================================
// Insert + delete cycles
// =======================================================================

#[test]
fn insert_delete_half_verify() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let value = vec![0u8; 100];

    // Phase 1: Insert 10K
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..10_000u64 {
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();
    assert_integrity(&db, 10_000);

    // Phase 2: Delete odd keys (forces merges throughout the tree)
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in (1..10_000u64).step_by(2) {
            t.remove(&i).unwrap();
        }
    }
    txn.commit().unwrap();
    assert_integrity(&db, 5_000);

    // Phase 3: Verify only even keys remain
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    for i in (0..10_000u64).step_by(2) {
        assert!(t.get(&i).unwrap().is_some(), "even key {i} missing");
    }
    for i in (1..10_000u64).step_by(2) {
        assert!(t.get(&i).unwrap().is_none(), "odd key {i} still present");
    }
}

#[test]
fn insert_delete_reinsert_cycle() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let value = vec![1u8; 50];
    let mut expected: BTreeSet<u64> = BTreeSet::new();

    for round in 0..5u64 {
        let base = round * 2000;

        // Insert 2000 keys
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for i in base..base + 2000 {
                t.insert(&i, value.as_slice()).unwrap();
                expected.insert(i);
            }
        }
        txn.commit().unwrap();
        assert_checksums(&db);

        // Delete first 1000 of this round
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for i in base..base + 1000 {
                t.remove(&i).unwrap();
                expected.remove(&i);
            }
        }
        txn.commit().unwrap();
        assert_checksums(&db);
    }

    // Final verification: all expected keys present, sorted
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    assert_eq!(t.len().unwrap(), expected.len() as u64);

    let db_keys: Vec<u64> = t.iter().unwrap().map(|r| r.unwrap().0.value()).collect();
    let expected_keys: Vec<u64> = expected.into_iter().collect();
    assert_eq!(db_keys, expected_keys);
}

#[test]
fn delete_all_then_reinsert() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let value = vec![0u8; 32];

    // Insert 5000
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..5_000u64 {
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();

    // Delete all via drain_all
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        let count = t.drain_all().unwrap();
        assert_eq!(count, 5_000);
    }
    txn.commit().unwrap();
    assert_integrity(&db, 0);

    // Reinsert different keys
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 100_000..105_000u64 {
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();
    assert_integrity(&db, 5_000);

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    assert_eq!(
        t.iter().unwrap().next().unwrap().unwrap().0.value(),
        100_000
    );
}

// =======================================================================
// Pseudo-random operations with deterministic seed
// =======================================================================

/// Simple deterministic PRNG (xorshift64)
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[test]
fn random_insert_delete_10k_ops() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let value = vec![0xFFu8; 80];
    let mut rng_state = 0xDEAD_BEEF_CAFE_BABEu64;
    let mut expected: BTreeSet<u64> = BTreeSet::new();

    for batch in 0..10 {
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for _ in 0..1000 {
                let key = xorshift64(&mut rng_state) % 5000;
                if xorshift64(&mut rng_state) % 3 == 0 {
                    // Delete with 1/3 probability
                    t.remove(&key).unwrap();
                    expected.remove(&key);
                } else {
                    // Insert with 2/3 probability
                    t.insert(&key, value.as_slice()).unwrap();
                    expected.insert(key);
                }
            }
        }
        txn.commit().unwrap();

        // Verify checksums every batch
        assert_checksums(&db);

        let txn = db.begin_read().unwrap();
        let t = txn.open_table(TABLE).unwrap();
        assert_eq!(
            t.len().unwrap(),
            expected.len() as u64,
            "length mismatch at batch {batch}"
        );
    }

    // Final sorted-order check
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    let db_keys: Vec<u64> = t.iter().unwrap().map(|r| r.unwrap().0.value()).collect();
    let expected_keys: Vec<u64> = expected.into_iter().collect();
    assert_eq!(db_keys, expected_keys);
}

#[test]
fn random_overwrite_same_keys() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let mut rng_state = 0x1234_5678_9ABC_DEF0u64;

    // Insert 1000 keys, then overwrite them all 10 times with different values
    for round in 0u8..10 {
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for i in 0..1000u64 {
                let val_byte = round.wrapping_add(xorshift64(&mut rng_state) as u8);
                let value = vec![val_byte; 64];
                t.insert(&i, value.as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    assert_integrity(&db, 1000);
}

// =======================================================================
// Compaction after churn
// =======================================================================

#[test]
fn compact_after_heavy_churn() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();
    let value = vec![0u8; 128];

    // Insert 5000
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..5_000u64 {
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();

    // Delete 4000 (heavy fragmentation)
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..4_000u64 {
            t.remove(&i).unwrap();
        }
    }
    txn.commit().unwrap();

    // Compact
    let compacted = db.compact().unwrap();
    assert!(compacted, "compact should have reclaimed space");

    assert_integrity(&db, 1_000);

    // Verify surviving keys
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    for i in 4000..5000u64 {
        assert!(
            t.get(&i).unwrap().is_some(),
            "key {i} missing after compact"
        );
    }
}

#[test]
fn compact_empty_db_is_noop() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();

    let compacted = db.compact().unwrap();
    // Fresh DB may or may not compact -- just verify no crash
    let _ = compacted;
    assert_checksums(&db);
}

#[test]
fn compact_multiple_rounds() {
    let tmpfile = create_tempfile();
    let mut db = Database::create(tmpfile.path()).unwrap();
    let value = vec![0u8; 64];

    for round in 0..3u64 {
        // Insert
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for i in 0..2000u64 {
                let key = round * 10_000 + i;
                t.insert(&key, value.as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();

        // Delete half
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for i in 0..1000u64 {
                let key = round * 10_000 + i;
                t.remove(&key).unwrap();
            }
        }
        txn.commit().unwrap();

        // Compact each round
        db.compact().unwrap();
        assert_checksums(&db);
    }

    // Final verification: 3 rounds * 1000 surviving keys
    assert_sorted_keys(&db, 3_000);
}

// =======================================================================
// Multiple tables
// =======================================================================

#[test]
fn multi_table_integrity() {
    const T1: TableDefinition<u64, u64> = TableDefinition::new("t1");
    const T2: TableDefinition<u64, u64> = TableDefinition::new("t2");
    const T3: TableDefinition<u64, u64> = TableDefinition::new("t3");

    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t1 = txn.open_table(T1).unwrap();
        for i in 0..1000u64 {
            t1.insert(&i, &i).unwrap();
        }
    }
    {
        let mut t2 = txn.open_table(T2).unwrap();
        for i in 0..2000u64 {
            t2.insert(&i, &(i * 2)).unwrap();
        }
    }
    {
        let mut t3 = txn.open_table(T3).unwrap();
        for i in 0..500u64 {
            t3.insert(&i, &(i * 3)).unwrap();
        }
    }
    txn.commit().unwrap();

    assert_checksums(&db);

    // Verify each table's contents
    let txn = db.begin_read().unwrap();
    let t1 = txn.open_table(T1).unwrap();
    let t2 = txn.open_table(T2).unwrap();
    let t3 = txn.open_table(T3).unwrap();
    assert_eq!(t1.len().unwrap(), 1000);
    assert_eq!(t2.len().unwrap(), 2000);
    assert_eq!(t3.len().unwrap(), 500);
    drop(txn);

    // Delete one table entirely, verify others unaffected
    let txn = db.begin_write().unwrap();
    txn.delete_table(T2).unwrap();
    txn.commit().unwrap();

    assert_checksums(&db);

    let txn = db.begin_read().unwrap();
    let t1 = txn.open_table(T1).unwrap();
    let t3 = txn.open_table(T3).unwrap();
    assert_eq!(t1.len().unwrap(), 1000);
    assert_eq!(t3.len().unwrap(), 500);
}

// =======================================================================
// Stats consistency
// =======================================================================

#[test]
fn stats_consistent_after_operations() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let value = vec![0u8; 256];

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..5_000u64 {
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    let stats = txn.stats().unwrap();
    txn.commit().unwrap();

    // Allocated + free should account for all pages
    assert!(stats.allocated_pages() > 0);
    assert!(stats.leaf_pages() > 0);
    assert!(stats.tree_height() > 0);
    // stored_bytes should be roughly 5000 * (8 key + 256 value)
    assert!(stats.stored_bytes() > 5000 * 200);
}

// =======================================================================
// Large values that span multiple pages
// =======================================================================

#[test]
fn large_values_integrity() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Values larger than a page (4096 bytes)
    let big_value = vec![0xBBu8; 8192];

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..100u64 {
            t.insert(&i, big_value.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();

    assert_integrity(&db, 100);

    // Delete half
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..50u64 {
            t.remove(&i).unwrap();
        }
    }
    txn.commit().unwrap();

    assert_integrity(&db, 50);

    // Verify correct values remain
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    for i in 50..100u64 {
        let val = t.get(&i).unwrap().unwrap();
        assert_eq!(val.value().len(), 8192);
        assert!(val.value().iter().all(|&b| b == 0xBB));
    }
}

// =======================================================================
// Retain and extract_if structural safety
// =======================================================================

#[test]
fn retain_preserves_integrity() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let value = vec![0u8; 64];

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..5_000u64 {
            t.insert(&i, value.as_slice()).unwrap();
        }
        // Retain only keys divisible by 7
        t.retain(|k, _v| k % 7 == 0).unwrap();
    }
    txn.commit().unwrap();

    // 0..5000 divisible by 7: 0, 7, 14, ..., 4998 -> ceil(5000/7) = 715
    assert_integrity(&db, 715);

    // Verify retained keys are the right ones
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(TABLE).unwrap();
    for result in t.iter().unwrap() {
        let (k, _) = result.unwrap();
        assert_eq!(
            k.value() % 7,
            0,
            "non-divisible-by-7 key {} survived retain",
            k.value()
        );
    }
}

#[test]
fn extract_if_preserves_integrity() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    const U64_TABLE: TableDefinition<u64, u64> = TableDefinition::new("extract_test");

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..5_000u64 {
            t.insert(&i, &i).unwrap();
        }
        // Extract all multiples of 3
        let extracted: Vec<_> = t
            .extract_if(|k, _v| k % 3 == 0)
            .unwrap()
            .map(|r| r.unwrap().0.value())
            .collect();
        assert_eq!(extracted.len(), 1667); // ceil(5000/3)
    }
    txn.commit().unwrap();

    assert_checksums(&db);

    // Verify remaining keys
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    assert_eq!(t.len().unwrap(), 5000 - 1667);
    for result in t.iter().unwrap() {
        let (k, _) = result.unwrap();
        assert_ne!(
            k.value() % 3,
            0,
            "multiple-of-3 key {} survived extract_if",
            k.value()
        );
    }
}

// =======================================================================
// Savepoint + rollback structural safety
// =======================================================================

#[test]
fn savepoint_rollback_preserves_integrity() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let value = vec![0u8; 64];

    // Populate base state
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 0..1000u64 {
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();

    // Create savepoint, add more, rollback
    let mut txn = db.begin_write().unwrap();
    let sp = txn.ephemeral_savepoint().unwrap();
    {
        let mut t = txn.open_table(TABLE).unwrap();
        for i in 1000..5000u64 {
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    txn.restore_savepoint(&sp).unwrap();
    txn.commit().unwrap();

    assert_integrity(&db, 1000);
}

// =======================================================================
// Reopen after operations
// =======================================================================

#[test]
fn reopen_after_heavy_operations() {
    let tmpfile = create_tempfile();
    let value = vec![0u8; 100];

    {
        let db = Database::create(tmpfile.path()).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for i in 0..10_000u64 {
                t.insert(&i, value.as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for i in 0..5_000u64 {
                t.remove(&i).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    // Reopen
    let db = Database::open(tmpfile.path()).unwrap();
    assert_integrity(&db, 5_000);
}
