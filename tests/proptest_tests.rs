//! Property-based tests using proptest for shodh-redb.
//!
//! These tests generate random inputs and verify invariants that must hold
//! for ALL inputs, not just hand-crafted examples.

use proptest::prelude::*;
use shodh_redb::*;
use tempfile::NamedTempFile;

fn create_tempfile() -> NamedTempFile {
    if cfg!(target_os = "wasi") {
        NamedTempFile::new_in("/tmp").unwrap()
    } else {
        NamedTempFile::new().unwrap()
    }
}

const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("prop");
const STR_TABLE: TableDefinition<&str, &str> = TableDefinition::new("str_prop");
const MM_TABLE: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_prop");

// ---------------------------------------------------------------------------
// 1. Insert-then-get roundtrip: any key/value inserted must be retrievable
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn insert_get_roundtrip(key in 0u64..10_000, value in prop::collection::vec(any::<u8>(), 0..1024)) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            t.insert(&key, value.as_slice()).unwrap();
        }
        txn.commit().unwrap();

        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(TABLE).unwrap();
        let got = t.get(&key).unwrap().unwrap();
        prop_assert_eq!(got.value(), value.as_slice());
    }
}

// ---------------------------------------------------------------------------
// 2. Remove-then-get: removed keys must return None
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn remove_then_get(key in 0u64..10_000) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            t.insert(&key, b"data".as_slice()).unwrap();
        }
        txn.commit().unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            t.remove(&key).unwrap();
        }
        txn.commit().unwrap();

        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(TABLE).unwrap();
        prop_assert!(t.get(&key).unwrap().is_none());
    }
}

// ---------------------------------------------------------------------------
// 3. Sorted iteration: keys from iter() must be in ascending order
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn iter_sorted_order(keys in prop::collection::vec(0u64..100_000, 1..500)) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for &k in &keys {
                t.insert(&k, b"v".as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();

        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(TABLE).unwrap();
        let db_keys: Vec<u64> = t.iter().unwrap().map(|r| r.unwrap().0.value()).collect();

        for w in db_keys.windows(2) {
            prop_assert!(w[0] < w[1], "keys not sorted: {} >= {}", w[0], w[1]);
        }

        // Unique key count matches
        let mut unique: Vec<u64> = keys.clone();
        unique.sort();
        unique.dedup();
        prop_assert_eq!(db_keys.len(), unique.len());
    }
}

// ---------------------------------------------------------------------------
// 4. Range scan correctness: range results match filter over full scan
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn range_scan_matches_filter(
        keys in prop::collection::vec(0u64..10_000, 10..200),
        lo in 0u64..5_000,
        hi in 5_000u64..10_000
    ) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for &k in &keys {
                t.insert(&k, b"v".as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();

        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(TABLE).unwrap();

        let range_keys: Vec<u64> = t.range(lo..=hi).unwrap().map(|r| r.unwrap().0.value()).collect();
        let filter_keys: Vec<u64> = t
            .iter()
            .unwrap()
            .map(|r| r.unwrap().0.value())
            .filter(|k| *k >= lo && *k <= hi)
            .collect();

        prop_assert_eq!(range_keys, filter_keys);
    }
}

// ---------------------------------------------------------------------------
// 5. Abort discards all writes
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn abort_discards_writes(
        committed_key in 0u64..5_000,
        aborted_key in 5_000u64..10_000
    ) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        // Committed write
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            t.insert(&committed_key, b"committed".as_slice()).unwrap();
        }
        txn.commit().unwrap();

        // Aborted write
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            t.insert(&aborted_key, b"aborted".as_slice()).unwrap();
        }
        txn.abort().unwrap();

        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(TABLE).unwrap();
        prop_assert!(t.get(&committed_key).unwrap().is_some());
        prop_assert!(t.get(&aborted_key).unwrap().is_none());
    }
}

// ---------------------------------------------------------------------------
// 6. Overwrite preserves latest value
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn overwrite_keeps_latest(
        key in 0u64..10_000,
        v1 in prop::collection::vec(any::<u8>(), 1..512),
        v2 in prop::collection::vec(any::<u8>(), 1..512)
    ) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            t.insert(&key, v1.as_slice()).unwrap();
            t.insert(&key, v2.as_slice()).unwrap();
        }
        txn.commit().unwrap();

        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(TABLE).unwrap();
        let got = t.get(&key).unwrap().unwrap();
        prop_assert_eq!(got.value(), v2.as_slice());
    }
}

// ---------------------------------------------------------------------------
// 7. Table length matches distinct inserted keys
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn table_len_matches_distinct_keys(keys in prop::collection::vec(0u64..5_000, 1..500)) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for &k in &keys {
                t.insert(&k, b"v".as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();

        let mut unique = keys.clone();
        unique.sort();
        unique.dedup();

        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(TABLE).unwrap();
        prop_assert_eq!(t.len().unwrap() as usize, unique.len());
    }
}

// ---------------------------------------------------------------------------
// 8. Multimap: multiple values per key, all retrievable
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn multimap_values_all_retrievable(
        key in 0u64..1_000,
        values in prop::collection::vec(0u64..10_000, 1..50)
    ) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_multimap_table(MM_TABLE).unwrap();
            for &v in &values {
                t.insert(&key, &v).unwrap();
            }
        }
        txn.commit().unwrap();

        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_multimap_table(MM_TABLE).unwrap();
        let db_vals: Vec<u64> = t
            .get(&key)
            .unwrap()
            .map(|r| r.unwrap().value())
            .collect();

        let mut unique_vals: Vec<u64> = values.clone();
        unique_vals.sort();
        unique_vals.dedup();

        // Multimap stores unique values per key, sorted
        prop_assert_eq!(db_vals, unique_vals);
    }
}

// ---------------------------------------------------------------------------
// 9. String key roundtrip: arbitrary UTF-8 strings
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn string_key_roundtrip(
        key in "[a-zA-Z0-9_]{1,100}",
        value in "[a-zA-Z0-9 ]{0,200}"
    ) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(STR_TABLE).unwrap();
            t.insert(key.as_str(), value.as_str()).unwrap();
        }
        txn.commit().unwrap();

        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(STR_TABLE).unwrap();
        let got = t.get(key.as_str()).unwrap().unwrap();
        prop_assert_eq!(got.value(), value.as_str());
    }
}

// ---------------------------------------------------------------------------
// 10. Savepoint rollback restores exact prior state
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn savepoint_rollback_restores_state(
        pre_keys in prop::collection::vec(0u64..5_000, 1..50),
        post_keys in prop::collection::vec(5_000u64..10_000, 1..50)
    ) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        // Insert pre_keys
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for &k in &pre_keys {
                t.insert(&k, b"pre".as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();

        // Savepoint, insert post_keys, rollback
        let mut txn = db.begin_write().unwrap();
        let sp = txn.ephemeral_savepoint().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for &k in &post_keys {
                t.insert(&k, b"post".as_slice()).unwrap();
            }
        }
        txn.restore_savepoint(&sp).unwrap();
        txn.commit().unwrap();

        // post_keys should not exist
        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(TABLE).unwrap();
        for &k in &post_keys {
            if !pre_keys.contains(&k) {
                prop_assert!(t.get(&k).unwrap().is_none(), "post_key {} should not exist after rollback", k);
            }
        }
        // pre_keys should exist
        for &k in &pre_keys {
            prop_assert!(t.get(&k).unwrap().is_some(), "pre_key {} should survive rollback", k);
        }
    }
}

// ---------------------------------------------------------------------------
// 11. Reopen persistence: data survives close + reopen
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn reopen_persistence(keys in prop::collection::vec(0u64..10_000, 1..100)) {
        let tmpfile = create_tempfile();

        {
            let db = Database::create(tmpfile.path()).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let mut t = txn.open_table(TABLE).unwrap();
                for &k in &keys {
                    t.insert(&k, b"persist".as_slice()).unwrap();
                }
            }
            txn.commit().unwrap();
        }

        let db = Database::open(tmpfile.path()).unwrap();
        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(TABLE).unwrap();

        let mut unique = keys.clone();
        unique.sort();
        unique.dedup();
        prop_assert_eq!(t.len().unwrap() as usize, unique.len());

        for &k in &unique {
            let got = t.get(&k).unwrap().unwrap();
            prop_assert_eq!(got.value(), b"persist".as_slice());
        }
    }
}

// ---------------------------------------------------------------------------
// 12. Random insert/delete sequence: len == expected count
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn random_insert_delete_len(ops in prop::collection::vec(
        (0u64..1_000, prop::bool::ANY), 10..500
    )) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        let mut expected = std::collections::BTreeSet::new();

        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for &(key, insert) in &ops {
                if insert {
                    t.insert(&key, b"v".as_slice()).unwrap();
                    expected.insert(key);
                } else {
                    t.remove(&key).unwrap();
                    expected.remove(&key);
                }
            }
        }
        txn.commit().unwrap();

        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(TABLE).unwrap();
        prop_assert_eq!(t.len().unwrap() as usize, expected.len());

        let db_keys: Vec<u64> = t.iter().unwrap().map(|r| r.unwrap().0.value()).collect();
        let expected_keys: Vec<u64> = expected.into_iter().collect();
        prop_assert_eq!(db_keys, expected_keys);
    }
}

// ---------------------------------------------------------------------------
// 13. Distance metric symmetry: d(a,b) == d(b,a)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn distance_metric_symmetry(
        dim in 2usize..64,
        seed_a in any::<u64>(),
        seed_b in any::<u64>()
    ) {
        let a = make_vec(dim, seed_a);
        let b = make_vec(dim, seed_b);

        let dot_ab = dot_product(&a, &b);
        let dot_ba = dot_product(&b, &a);
        prop_assert!((dot_ab - dot_ba).abs() < 1e-5 * dot_ab.abs().max(1.0),
            "dot_product not symmetric: {} vs {}", dot_ab, dot_ba);

        let euc_ab = euclidean_distance_sq(&a, &b);
        let euc_ba = euclidean_distance_sq(&b, &a);
        prop_assert!((euc_ab - euc_ba).abs() < 1e-5 * euc_ab.abs().max(1.0),
            "euclidean not symmetric: {} vs {}", euc_ab, euc_ba);

        let man_ab = manhattan_distance(&a, &b);
        let man_ba = manhattan_distance(&b, &a);
        prop_assert!((man_ab - man_ba).abs() < 1e-5 * man_ab.abs().max(1.0),
            "manhattan not symmetric: {} vs {}", man_ab, man_ba);

        let cos_ab = cosine_similarity(&a, &b);
        let cos_ba = cosine_similarity(&b, &a);
        prop_assert!((cos_ab - cos_ba).abs() < 1e-5,
            "cosine not symmetric: {} vs {}", cos_ab, cos_ba);
    }
}

// ---------------------------------------------------------------------------
// 14. Euclidean distance non-negativity and identity
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn euclidean_non_negative_identity(
        dim in 2usize..128,
        seed in any::<u64>()
    ) {
        let v = make_vec(dim, seed);
        let self_dist = euclidean_distance_sq(&v, &v);
        prop_assert!(self_dist >= 0.0 || self_dist.is_nan(), "self-distance must be >= 0");
        prop_assert!(self_dist.abs() < 1e-4, "self-distance should be ~0, got {}", self_dist);
    }
}

// ---------------------------------------------------------------------------
// 15. Blob store roundtrip: any byte sequence stored and retrieved intact
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn blob_store_roundtrip(data in prop::collection::vec(any::<u8>(), 0..4096)) {
        let tmpfile = create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();

        let blob_id = {
            let txn = db.begin_write().unwrap();
            let id = txn
                .store_blob(&data, ContentType::OctetStream, "test", StoreOptions::default())
                .unwrap();
            txn.commit().unwrap();
            id
        };

        let rtxn = db.begin_read().unwrap();
        let (got, _meta) = rtxn.get_blob(&blob_id).unwrap().unwrap();
        prop_assert_eq!(got, data);
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn make_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = seed;
    for _ in 0..dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0);
    }
    v
}
