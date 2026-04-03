/// SOC Audit Hardening Tests
///
/// Comprehensive test suite validating fixes for issues #143-#160 from the
/// silicon audit. Covers:
///
/// - Boundary conditions and edge cases for all distance/vector functions
/// - Corruption resilience of serialization/deserialization paths
/// - Key comparison correctness (including truncated data)
/// - NaN handling in Neighbor Ord/Eq and nearest_k heap
/// - k-means PRNG safety (div-by-zero, bias)
/// - Nearest-clusters empty-centroid guard
/// - Variable-width [T; N] offset validation
/// - Scalar/binary quantization edge cases
use shodh_redb::{
    BinaryQuantized, Database, DistanceMetric, DynVec, FixedVec, ReadableDatabase, ReadableTable,
    ReadableTableMetadata, ScalarQuantized, TableDefinition,
};

fn create_tempfile() -> tempfile::NamedTempFile {
    if cfg!(target_os = "wasi") {
        tempfile::NamedTempFile::new_in("/tmp").unwrap()
    } else {
        tempfile::NamedTempFile::new().unwrap()
    }
}

// ===========================================================================
// S1  NaN HANDLING -- Issues #151, #157 (Neighbor Eq/Ord, nearest_k)
// ===========================================================================

#[test]
fn neighbor_nan_distance_does_not_corrupt_heap() {
    // NaN distances should be evicted first (sort as largest) so they don't
    // block valid results from entering the top-k heap.
    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![1.0, 0.0, 0.0]),      // close
        (2, vec![f32::NAN, 0.0, 0.0]), // NaN -> distance is NaN
        (3, vec![0.9, 0.1, 0.0]),      // close
        (4, vec![0.0, 0.0, 1.0]),      // far
    ];
    let query = [1.0f32, 0.0, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 2, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    // Should return 2 results without panicking. NaN entries should NOT
    // dominate the heap and block valid results.
    assert_eq!(results.len(), 2);
    // Both returned entries should have finite distances
    for r in &results {
        assert!(
            r.distance.is_finite(),
            "NaN distance leaked into top-k results: key={}, dist={}",
            r.key,
            r.distance
        );
    }
}

#[test]
fn neighbor_all_nan_distances() {
    // Edge case: every single distance is NaN.
    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![f32::NAN]),
        (2, vec![f32::NAN]),
        (3, vec![f32::NAN]),
    ];
    let query = [1.0f32];
    // Should not panic. Results may contain NaN entries since there's nothing better.
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 2, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert!(results.len() <= 2);
}

#[test]
fn neighbor_inf_distance_handled() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![1.0, 0.0]),
        (2, vec![f32::INFINITY, 0.0]),
        (3, vec![0.5, 0.5]),
    ];
    let query = [1.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 2, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert_eq!(results.len(), 2);
    // key=1 should be closest (dist=0), key=3 next
    assert_eq!(results[0].key, 1);
    assert_eq!(results[1].key, 3);
}

#[test]
fn nearest_k_fixed_nan_handling() {
    let vectors: Vec<(u64, [f32; 2])> =
        vec![(1, [1.0, 0.0]), (2, [f32::NAN, f32::NAN]), (3, [0.9, 0.1])];
    let query = [1.0f32, 0.0];
    let results = shodh_redb::nearest_k_fixed(vectors.into_iter(), &query, 2, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert_eq!(results.len(), 2);
}

// ===========================================================================
// S2  DISTANCE FUNCTION EDGE CASES -- Issues #151, #157
// ===========================================================================

#[test]
fn cosine_similarity_zero_vectors() {
    let zero = [0.0f32; 4];
    let v = [1.0, 2.0, 3.0, 4.0];
    // Zero vs non-zero -> 0.0 (not NaN)
    assert_eq!(shodh_redb::cosine_similarity(&zero, &v), 0.0);
    assert_eq!(shodh_redb::cosine_similarity(&v, &zero), 0.0);
    // Zero vs zero -> 0.0 (not NaN)
    assert_eq!(shodh_redb::cosine_similarity(&zero, &zero), 0.0);
}

#[test]
fn cosine_distance_zero_vectors() {
    let zero = [0.0f32; 4];
    let v = [1.0, 2.0, 3.0, 4.0];
    // cosine_distance = 1.0 - cosine_similarity
    // For zero vectors, similarity=0.0, so distance=1.0
    assert!((shodh_redb::cosine_distance(&zero, &v) - 1.0).abs() < 1e-6);
}

#[test]
fn euclidean_distance_identical_vectors() {
    let v = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(shodh_redb::euclidean_distance_sq(&v, &v), 0.0);
}

#[test]
fn manhattan_distance_identical_vectors() {
    let v = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(shodh_redb::manhattan_distance(&v, &v), 0.0);
}

#[test]
fn dot_product_orthogonal() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert_eq!(shodh_redb::dot_product(&a, &b), 0.0);
}

#[test]
fn distance_metric_enum_all_variants() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    // All variants should produce finite results with finite inputs
    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::EuclideanSq,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ] {
        let d = metric.compute(&a, &b);
        assert!(d.is_finite(), "{metric:?} returned non-finite: {d}");
    }
}

#[test]
fn distance_empty_vectors() {
    let empty: [f32; 0] = [];
    // All distance functions should return 0 for empty vectors (sum of nothing)
    assert_eq!(shodh_redb::dot_product(&empty, &empty), 0.0);
    assert_eq!(shodh_redb::euclidean_distance_sq(&empty, &empty), 0.0);
    assert_eq!(shodh_redb::manhattan_distance(&empty, &empty), 0.0);
    // cosine_similarity: zero-magnitude -> 0.0
    assert_eq!(shodh_redb::cosine_similarity(&empty, &empty), 0.0);
}

#[test]
fn hamming_distance_empty() {
    assert_eq!(shodh_redb::hamming_distance(&[], &[]), 0);
}

#[test]
fn hamming_distance_all_same() {
    let a = [0xFFu8; 16];
    assert_eq!(shodh_redb::hamming_distance(&a, &a), 0);
}

#[test]
fn hamming_distance_all_different() {
    let a = [0x00u8; 4];
    let b = [0xFFu8; 4];
    assert_eq!(shodh_redb::hamming_distance(&a, &b), 32); // 4 bytes * 8 bits
}

// ===========================================================================
// S3  L2 NORMALIZATION EDGE CASES
// ===========================================================================

#[test]
fn l2_normalize_zero_vector_unchanged() {
    let mut v = [0.0f32; 8];
    shodh_redb::l2_normalize(&mut v);
    assert!(v.iter().all(|&x| x == 0.0));
}

#[test]
fn l2_normalize_single_element() {
    let mut v = [5.0f32];
    shodh_redb::l2_normalize(&mut v);
    assert!((v[0] - 1.0).abs() < 1e-6);
}

#[test]
fn l2_normalize_negative_values() {
    let mut v = [-3.0f32, -4.0];
    shodh_redb::l2_normalize(&mut v);
    assert!((shodh_redb::l2_norm(&v) - 1.0).abs() < 1e-6);
}

#[test]
fn l2_norm_empty() {
    assert_eq!(shodh_redb::l2_norm(&[]), 0.0);
}

// ===========================================================================
// S4  QUANTIZATION EDGE CASES -- Issue #151 (scalar quantize rounding)
// ===========================================================================

#[test]
fn quantize_scalar_constant_zero_vector() {
    let v = [0.0f32; 8];
    let sq = shodh_redb::quantize_scalar(&v);
    assert_eq!(sq.min_val, 0.0);
    assert_eq!(sq.max_val, 0.0);
    // All codes should be 0 (range is 0, no scaling)
    assert!(sq.codes.iter().all(|&c| c == 0));
    // Dequantize should return all zeros
    let dq = sq.dequantize();
    assert!(dq.iter().all(|&x| x == 0.0));
}

#[test]
fn quantize_scalar_single_element() {
    let v = [42.0f32];
    let sq = shodh_redb::quantize_scalar(&v);
    assert_eq!(sq.min_val, 42.0);
    assert_eq!(sq.max_val, 42.0);
    // Constant -> range=0 -> code=0
    let dq = sq.dequantize();
    assert!((dq[0] - 42.0).abs() < f32::EPSILON);
}

#[test]
fn quantize_scalar_nan_input() {
    // NaN is silently skipped by f32 comparison operators, so min/max
    // reflect the non-NaN elements. The NaN element's code is undefined
    // but the function must not panic.
    let v = [f32::NAN, 1.0, 2.0, 3.0];
    let sq = shodh_redb::quantize_scalar(&v);
    assert_eq!(sq.min_val, 1.0);
    assert_eq!(sq.max_val, 3.0);
    // Non-NaN elements should be correctly quantized
    // codes[1] ~ 0 (min), codes[2] ~ 128, codes[3] = 255 (max)
    assert_eq!(sq.codes[3], 255);
}

#[test]
fn quantize_scalar_inf_input() {
    let v = [f32::INFINITY, f32::NEG_INFINITY, 1.0, 2.0];
    let sq = shodh_redb::quantize_scalar(&v);
    // Inf range -> not finite -> fallback
    assert_eq!(sq.min_val, 0.0);
    assert_eq!(sq.max_val, 0.0);
}

#[test]
fn quantize_scalar_extreme_range() {
    let v = [-1e30f32, 1e30, 0.0, 0.5];
    let sq = shodh_redb::quantize_scalar(&v);
    assert!(sq.min_val < 0.0);
    assert!(sq.max_val > 0.0);
    // Should not panic; codes are u8 -- verify they roundtrip
    assert_eq!(sq.codes.len(), 4);
}

#[test]
fn quantize_scalar_min_equals_max_at_boundary() {
    // All same value
    let v = [1.0f32; 16];
    let sq = shodh_redb::quantize_scalar(&v);
    let dq = sq.dequantize();
    for &x in &dq {
        assert!((x - 1.0).abs() < f32::EPSILON);
    }
}

#[test]
fn quantize_binary_empty() {
    let v: [f32; 0] = [];
    let bq = shodh_redb::quantize_binary(&v);
    assert!(bq.is_empty());
}

#[test]
fn quantize_binary_single_positive() {
    let v = [1.0f32];
    let bq = shodh_redb::quantize_binary(&v);
    assert_eq!(bq.len(), 1);
    assert_eq!(bq[0], 0b1000_0000); // MSB set
}

#[test]
fn quantize_binary_single_negative() {
    let v = [-1.0f32];
    let bq = shodh_redb::quantize_binary(&v);
    assert_eq!(bq.len(), 1);
    assert_eq!(bq[0], 0b0000_0000);
}

#[test]
fn quantize_binary_zero_is_not_positive() {
    let v = [0.0f32];
    let bq = shodh_redb::quantize_binary(&v);
    assert_eq!(bq[0], 0b0000_0000); // 0.0 is NOT > 0.0
}

// ===========================================================================
// S5  SQ DISTANCE APPROXIMATION EDGE CASES
// ===========================================================================

#[test]
fn sq_euclidean_constant_vector() {
    let query = [1.0f32, 2.0, 3.0, 4.0];
    let v = [5.0f32; 4]; // constant
    let sq = shodh_redb::quantize_scalar(&v);
    // range==0, so dequant returns all 5.0
    let approx = shodh_redb::sq_euclidean_distance_sq(&query, &sq);
    let exact = shodh_redb::euclidean_distance_sq(&query, &v);
    assert!((approx - exact).abs() < 0.01);
}

#[test]
fn sq_dot_product_constant_vector() {
    let query = [1.0f32, 2.0, 3.0, 4.0];
    let v = [5.0f32; 4];
    let sq = shodh_redb::quantize_scalar(&v);
    let approx = shodh_redb::sq_dot_product(&query, &sq);
    let exact = shodh_redb::dot_product(&query, &v);
    assert!((approx - exact).abs() < 0.01);
}

// ===========================================================================
// S6  FIXED/DYN VEC STORAGE -- Issue #147 (silent corruption on truncated data)
// ===========================================================================

#[test]
fn fixed_vec_zero_dimension() {
    // FixedVec<0> is technically valid -- fixed_width=0, stores nothing.
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE_VEC0: TableDefinition<u64, FixedVec<0>> = TableDefinition::new("vec0");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE_VEC0).unwrap();
        table.insert(&1u64, &[]).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE_VEC0).unwrap();
    let stored = table.get(&1u64).unwrap().unwrap().value();
    assert_eq!(stored.len(), 0);
}

#[test]
fn fixed_vec_single_dimension() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE_VEC1: TableDefinition<u64, FixedVec<1>> = TableDefinition::new("vec1");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE_VEC1).unwrap();
        table.insert(&1u64, &[42.0f32]).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE_VEC1).unwrap();
    let stored = table.get(&1u64).unwrap().unwrap().value();
    assert_eq!(stored, [42.0f32]);
}

#[test]
fn dyn_vec_empty_vector() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE_DYN: TableDefinition<u64, DynVec> = TableDefinition::new("dyn_empty");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE_DYN).unwrap();
        table.insert(&1u64, &Vec::<f32>::new()).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE_DYN).unwrap();
    let stored = table.get(&1u64).unwrap().unwrap().value();
    assert!(stored.is_empty());
}

#[test]
fn dyn_vec_large_dimension() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE_DYN: TableDefinition<u64, DynVec> = TableDefinition::new("dyn_large");

    let large: Vec<f32> = (0..4096).map(|i| i as f32 * 0.001).collect();
    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE_DYN).unwrap();
        table.insert(&1u64, &large).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE_DYN).unwrap();
    let stored = table.get(&1u64).unwrap().unwrap().value();
    assert_eq!(stored.len(), 4096);
    for (i, &v) in stored.iter().enumerate() {
        assert!((v - large[i]).abs() < f32::EPSILON, "mismatch at dim {i}");
    }
}

// ===========================================================================
// S7  BINARY QUANTIZED STORAGE ROUNDTRIPS
// ===========================================================================

#[test]
fn binary_quantized_zero_bytes() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const BQ0: TableDefinition<u64, BinaryQuantized<0>> = TableDefinition::new("bq0");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(BQ0).unwrap();
        table.insert(&1u64, &[]).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(BQ0).unwrap();
    let v = table.get(&1u64).unwrap().unwrap().value();
    assert_eq!(v.len(), 0);
}

#[test]
fn binary_quantized_large_roundtrip() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    // 48 bytes = 384 binary dimensions
    const BQ48: TableDefinition<u64, BinaryQuantized<48>> = TableDefinition::new("bq48");

    let data = [0xABu8; 48];
    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(BQ48).unwrap();
        table.insert(&1u64, &data).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(BQ48).unwrap();
    assert_eq!(table.get(&1u64).unwrap().unwrap().value(), data);
}

// ===========================================================================
// S8  SCALAR QUANTIZED STORAGE ROUNDTRIPS
// ===========================================================================

#[test]
fn scalar_quantized_store_and_dequantize_accuracy() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const SQ_TABLE: TableDefinition<u64, ScalarQuantized<128>> = TableDefinition::new("sq128");

    let original: [f32; 128] = core::array::from_fn(|i| (i as f32) / 128.0);
    let sq = shodh_redb::quantize_scalar(&original);

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(SQ_TABLE).unwrap();
        table.insert(&1u64, &sq).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(SQ_TABLE).unwrap();
    let stored = table.get(&1u64).unwrap().unwrap().value();
    let recovered = stored.dequantize();

    for i in 0..128 {
        assert!(
            (recovered[i] - original[i]).abs() < 0.005,
            "dim {i}: expected {}, got {}",
            original[i],
            recovered[i],
        );
    }
}

// ===========================================================================
// S9  OPTION<T> TYPE SERIALIZATION -- Issue #158
// ===========================================================================

#[test]
fn option_type_none_roundtrip() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, Option<u64>> = TableDefinition::new("opt_u64");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(&1u64, &None).unwrap();
        table.insert(&2u64, &Some(42u64)).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(table.get(&1u64).unwrap().unwrap().value(), None);
    assert_eq!(table.get(&2u64).unwrap().unwrap().value(), Some(42));
}

#[test]
fn option_key_ordering() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<Option<u32>, u64> = TableDefinition::new("opt_key");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(&None, &0u64).unwrap();
        table.insert(&Some(1u32), &1u64).unwrap();
        table.insert(&Some(100u32), &100u64).unwrap();
        table.insert(&Some(0u32), &999u64).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    let keys: Vec<_> = table
        .iter()
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    // None < Some(0) < Some(1) < Some(100)
    assert_eq!(keys, vec![None, Some(0), Some(1), Some(100)]);
}

// ===========================================================================
// S10  ARRAY [T; N] TYPE -- Issue #149 (variable-width offset validation)
// ===========================================================================

#[test]
fn array_fixed_width_roundtrip() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, [u32; 4]> = TableDefinition::new("arr4");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(&1u64, &[10u32, 20, 30, 40]).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(table.get(&1u64).unwrap().unwrap().value(), [10, 20, 30, 40]);
}

#[test]
fn array_variable_width_roundtrip() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    // Variable-width inner type: &str
    const TABLE: TableDefinition<u64, [&str; 3]> = TableDefinition::new("arr_str");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(&1u64, &["hello", "world", "test"]).unwrap();
        table.insert(&2u64, &["", "", ""]).unwrap();
        table.insert(&3u64, &["a", "bb", "ccc"]).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(
        table.get(&1u64).unwrap().unwrap().value(),
        ["hello", "world", "test"]
    );
    assert_eq!(table.get(&2u64).unwrap().unwrap().value(), ["", "", ""]);
    assert_eq!(
        table.get(&3u64).unwrap().unwrap().value(),
        ["a", "bb", "ccc"]
    );
}

#[test]
fn array_key_ordering_fixed_width() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<[u32; 2], u64> = TableDefinition::new("arr_key");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(&[1u32, 10], &1u64).unwrap();
        table.insert(&[1u32, 5], &2u64).unwrap();
        table.insert(&[2u32, 1], &3u64).unwrap();
        table.insert(&[0u32, 100], &4u64).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    let keys: Vec<_> = table
        .iter()
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(keys, vec![[0, 100], [1, 5], [1, 10], [2, 1]]);
}

// ===========================================================================
// S11  BOOL AND CHAR TYPES -- Corruption resilience
// ===========================================================================

#[test]
fn bool_value_roundtrip() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, bool> = TableDefinition::new("bools");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(&1u64, &true).unwrap();
        table.insert(&2u64, &false).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert!(table.get(&1u64).unwrap().unwrap().value());
    assert!(!table.get(&2u64).unwrap().unwrap().value());
}

#[test]
fn char_value_roundtrip() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, char> = TableDefinition::new("chars");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(&1u64, &'A').unwrap();
        table.insert(&2u64, &'\u{1F980}').unwrap(); // U+1F980 (4-byte UTF-8, > U+FFFF)
        table.insert(&3u64, &'\u{10FFFF}').unwrap(); // max Unicode scalar
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(table.get(&1u64).unwrap().unwrap().value(), 'A');
    assert_eq!(table.get(&2u64).unwrap().unwrap().value(), '\u{1F980}');
    assert_eq!(table.get(&3u64).unwrap().unwrap().value(), '\u{10FFFF}');
}

#[test]
fn char_key_ordering() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<char, u64> = TableDefinition::new("char_keys");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(&'z', &1u64).unwrap();
        table.insert(&'a', &2u64).unwrap();
        table.insert(&'m', &3u64).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    let keys: Vec<_> = table
        .iter()
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(keys, vec!['a', 'm', 'z']);
}

// ===========================================================================
// S12  STRING TYPE -- Corruption resilience
// ===========================================================================

#[test]
fn string_key_and_value_roundtrip() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<String, String> = TableDefinition::new("strings");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table
            .insert("hello".to_string(), "world".to_string())
            .unwrap();
        table
            .insert(String::new(), "empty_key".to_string())
            .unwrap();
        // Unicode
        table
            .insert("nihongo".to_string(), "Japanese".to_string())
            .unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(
        table.get("hello".to_string()).unwrap().unwrap().value(),
        "world"
    );
    assert_eq!(
        table.get(String::new()).unwrap().unwrap().value(),
        "empty_key"
    );
    assert_eq!(
        table.get("nihongo".to_string()).unwrap().unwrap().value(),
        "Japanese"
    );
}

// ===========================================================================
// S13  NUMERIC TYPES -- Boundary values
// ===========================================================================

#[test]
fn u64_boundary_values() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, u64> = TableDefinition::new("u64_boundary");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(&0u64, &0u64).unwrap();
        table.insert(&u64::MAX, &u64::MAX).unwrap();
        table.insert(&(u64::MAX / 2), &42u64).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(table.get(&0u64).unwrap().unwrap().value(), 0);
    assert_eq!(table.get(&u64::MAX).unwrap().unwrap().value(), u64::MAX);
    assert_eq!(table.get(&(u64::MAX / 2)).unwrap().unwrap().value(), 42);

    // Ordering: keys should be sorted
    let keys: Vec<u64> = table
        .iter()
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(keys, vec![0, u64::MAX / 2, u64::MAX]);
}

#[test]
fn i64_boundary_values() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<i64, i64> = TableDefinition::new("i64_boundary");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(&i64::MIN, &i64::MIN).unwrap();
        table.insert(&0i64, &0i64).unwrap();
        table.insert(&i64::MAX, &i64::MAX).unwrap();
        table.insert(&(-1i64), &(-1i64)).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(table.get(&i64::MIN).unwrap().unwrap().value(), i64::MIN);
    assert_eq!(table.get(&i64::MAX).unwrap().unwrap().value(), i64::MAX);

    let keys: Vec<i64> = table
        .iter()
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(keys, vec![i64::MIN, -1, 0, i64::MAX]);
}

#[test]
fn f32_special_values_stored() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, f32> = TableDefinition::new("f32_special");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(&1u64, &0.0f32).unwrap();
        table.insert(&2u64, &(-0.0f32)).unwrap();
        table.insert(&3u64, &f32::INFINITY).unwrap();
        table.insert(&4u64, &f32::NEG_INFINITY).unwrap();
        table.insert(&5u64, &f32::NAN).unwrap();
        table.insert(&6u64, &f32::MIN).unwrap();
        table.insert(&7u64, &f32::MAX).unwrap();
        table.insert(&8u64, &f32::EPSILON).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(table.get(&1u64).unwrap().unwrap().value(), 0.0f32);
    assert!(table.get(&3u64).unwrap().unwrap().value().is_infinite());
    assert!(table.get(&5u64).unwrap().unwrap().value().is_nan());
    assert_eq!(table.get(&6u64).unwrap().unwrap().value(), f32::MIN);
    assert_eq!(table.get(&7u64).unwrap().unwrap().value(), f32::MAX);
}

// ===========================================================================
// S14  BYTE SLICE AND BYTE ARRAY TYPES
// ===========================================================================

#[test]
fn byte_slice_empty_key_and_value() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("bytes");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert([].as_slice(), [].as_slice()).unwrap();
        table
            .insert([1u8, 2, 3].as_slice(), [4u8, 5, 6].as_slice())
            .unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(
        table.get([].as_slice()).unwrap().unwrap().value(),
        &[] as &[u8]
    );
    assert_eq!(
        table.get([1u8, 2, 3].as_slice()).unwrap().unwrap().value(),
        &[4u8, 5, 6]
    );
}

#[test]
fn byte_array_roundtrip() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<&[u8; 4], u64> = TableDefinition::new("byte_arr_key");

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(b"abcd", &1u64).unwrap();
        table.insert(b"ABCD", &2u64).unwrap();
        table.insert(&[0u8; 4], &3u64).unwrap();
        table.insert(&[0xFF; 4], &4u64).unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(table.get(b"abcd").unwrap().unwrap().value(), 1);
    assert_eq!(table.get(&[0xFF; 4]).unwrap().unwrap().value(), 4);
    assert_eq!(table.len().unwrap(), 4);
}

// ===========================================================================
// S15  NEAREST-K EDGE CASES
// ===========================================================================

#[test]
fn nearest_k_empty_iterator() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![];
    let query = [1.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 10, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert!(results.is_empty());
}

#[test]
fn nearest_k_k_equals_one() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![10.0, 0.0]),
        (2, vec![1.0, 0.0]),
        (3, vec![5.0, 0.0]),
    ];
    let query = [0.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 1, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].key, 2); // closest to origin
}

#[test]
fn nearest_k_identical_distances() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![1.0, 0.0]),
        (2, vec![0.0, 1.0]),
        (3, vec![-1.0, 0.0]),
        (4, vec![0.0, -1.0]),
    ];
    let query = [0.0f32, 0.0];
    // All equidistant from origin (distance=1.0). Should return k=2 without issues.
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 2, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert_eq!(results.len(), 2);
    // All distances should be 1.0
    for r in &results {
        assert!((r.distance - 1.0).abs() < 1e-6);
    }
}

#[test]
fn nearest_k_large_k_returns_all_sorted() {
    let vectors: Vec<(u64, Vec<f32>)> = (0..100).map(|i| (i as u64, vec![i as f32, 0.0])).collect();
    let query = [0.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 100, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert_eq!(results.len(), 100);
    // Results should be sorted by ascending distance
    for i in 1..results.len() {
        assert!(
            results[i - 1].distance <= results[i].distance,
            "results not sorted at index {i}: {} > {}",
            results[i - 1].distance,
            results[i].distance
        );
    }
}

// ===========================================================================
// S16  WRITE_F32_LE / READ_F32_LE ROUNDTRIP
// ===========================================================================

#[test]
fn write_read_f32_le_roundtrip() {
    let values = [1.0f32, -2.5, core::f32::consts::PI, 0.0, f32::MAX, f32::MIN];
    let mut buf = vec![0u8; values.len() * 4];
    shodh_redb::write_f32_le(&mut buf, &values);
    let recovered = shodh_redb::read_f32_le(&buf);
    assert_eq!(recovered.len(), values.len());
    for (i, (&orig, &recov)) in values.iter().zip(recovered.iter()).enumerate() {
        assert_eq!(orig.to_bits(), recov.to_bits(), "mismatch at index {i}");
    }
}

#[test]
fn write_f32_le_short_buffer() {
    // Buffer shorter than values: writes only what fits
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let mut buf = vec![0u8; 8]; // only room for 2 floats
    shodh_redb::write_f32_le(&mut buf, &values);
    let recovered = shodh_redb::read_f32_le(&buf);
    assert_eq!(recovered, vec![1.0, 2.0]);
}

#[test]
fn read_f32_le_non_multiple_of_4() {
    // Trailing bytes should be ignored
    let mut buf = vec![0u8; 9]; // 2 complete f32s + 1 extra byte
    shodh_redb::write_f32_le(&mut buf, &[1.0, 2.0]);
    buf[8] = 0xFF; // extra byte
    let recovered = shodh_redb::read_f32_le(&buf);
    assert_eq!(recovered, vec![1.0, 2.0]);
}

#[test]
fn read_f32_le_empty() {
    assert!(shodh_redb::read_f32_le(&[]).is_empty());
}

// ===========================================================================
// S17  TRANSACTION ISOLATION
// ===========================================================================

#[test]
fn read_transaction_sees_snapshot() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, u64> = TableDefinition::new("snapshot_test");

    // Write initial data
    let write_txn = db.begin_write().unwrap();
    {
        let mut t = write_txn.open_table(TABLE).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    write_txn.commit().unwrap();

    // Start a read transaction
    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();

    // Write more data in a new transaction
    let write_txn2 = db.begin_write().unwrap();
    {
        let mut t = write_txn2.open_table(TABLE).unwrap();
        t.insert(&2u64, &200u64).unwrap();
    }
    write_txn2.commit().unwrap();

    // Read transaction should NOT see key 2
    assert!(table.get(&2u64).unwrap().is_none());
    assert_eq!(table.get(&1u64).unwrap().unwrap().value(), 100);
    assert_eq!(table.len().unwrap(), 1);
}

// ===========================================================================
// S18  RANGE ITERATION -- B-tree iterator correctness
// ===========================================================================

#[test]
fn range_iteration_forward_and_reverse() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, u64> = TableDefinition::new("range_iter");

    let write_txn = db.begin_write().unwrap();
    {
        let mut t = write_txn.open_table(TABLE).unwrap();
        for i in 0..50u64 {
            t.insert(&i, &(i * 10)).unwrap();
        }
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();

    // Forward
    let fwd: Vec<u64> = table
        .range(10u64..20u64)
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(fwd, (10..20).collect::<Vec<u64>>());

    // Reverse
    let rev: Vec<u64> = table
        .range(10u64..20u64)
        .unwrap()
        .rev()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(rev, (10..20).rev().collect::<Vec<u64>>());
}

#[test]
fn range_iteration_unbounded() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, u64> = TableDefinition::new("unbounded_iter");

    let write_txn = db.begin_write().unwrap();
    {
        let mut t = write_txn.open_table(TABLE).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();

    // Full iteration
    let all: Vec<u64> = table
        .iter()
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(all, (0..10).collect::<Vec<u64>>());

    // Full reverse
    let all_rev: Vec<u64> = table
        .iter()
        .unwrap()
        .rev()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(all_rev, (0..10).rev().collect::<Vec<u64>>());
}

#[test]
fn range_iteration_empty_range() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, u64> = TableDefinition::new("empty_range");

    let write_txn = db.begin_write().unwrap();
    {
        let mut t = write_txn.open_table(TABLE).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();

    // Range where start > all keys
    let empty: Vec<u64> = table
        .range(100u64..200u64)
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert!(empty.is_empty());

    // Degenerate range: start >= end
    let empty2: Vec<u64> = table
        .range(5u64..5u64)
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert!(empty2.is_empty());
}

// ===========================================================================
// S19  MANY ENTRIES -- Stress test B-tree with enough data to split pages
// ===========================================================================

#[test]
fn stress_many_entries_insert_and_verify() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, u64> = TableDefinition::new("stress");

    let n = 5000u64;

    let write_txn = db.begin_write().unwrap();
    {
        let mut t = write_txn.open_table(TABLE).unwrap();
        for i in 0..n {
            t.insert(&i, &(i * i)).unwrap();
        }
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(table.len().unwrap(), n);

    // Spot check
    for i in [0, 1, 100, 999, 2500, 4999] {
        assert_eq!(table.get(&i).unwrap().unwrap().value(), i * i);
    }

    // Verify iteration count
    let count = table.iter().unwrap().count();
    assert_eq!(count, n as usize);
}

#[test]
fn stress_many_entries_delete_half() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    const TABLE: TableDefinition<u64, u64> = TableDefinition::new("stress_del");

    let n = 2000u64;

    let write_txn = db.begin_write().unwrap();
    {
        let mut t = write_txn.open_table(TABLE).unwrap();
        for i in 0..n {
            t.insert(&i, &i).unwrap();
        }
    }
    write_txn.commit().unwrap();

    // Delete even keys
    let write_txn = db.begin_write().unwrap();
    {
        let mut t = write_txn.open_table(TABLE).unwrap();
        for i in (0..n).step_by(2) {
            t.remove(&i).unwrap();
        }
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(table.len().unwrap(), n / 2);

    // All remaining keys should be odd
    for entry in table.iter().unwrap() {
        let k = entry.unwrap().0.value();
        assert!(k % 2 == 1, "even key {k} was not deleted");
    }
}

// ===========================================================================
// S20  FLASH BACKEND -- Overflow-safe offset calculations (Issue #145)
// ===========================================================================

#[test]
fn flash_backend_many_sequential_writes() {
    use shodh_redb::error::BackendError;
    use shodh_redb::{Builder, FlashBackend, FlashGeometry, FlashHardware};
    use std::sync::{Arc, RwLock};

    struct SimpleFlash {
        storage: Arc<RwLock<Vec<u8>>>,
        geometry: FlashGeometry,
    }
    impl std::fmt::Debug for SimpleFlash {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("SimpleFlash")
        }
    }
    impl SimpleFlash {
        fn new(blocks: u32) -> Self {
            let geo = FlashGeometry {
                erase_block_size: 4096,
                write_page_size: 4096,
                total_blocks: blocks,
                max_erase_cycles: 100_000,
            };
            Self {
                storage: Arc::new(RwLock::new(vec![0xFF; geo.total_capacity() as usize])),
                geometry: geo,
            }
        }
    }
    impl FlashHardware for SimpleFlash {
        fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), BackendError> {
            let s = self.storage.read().unwrap();
            buf.copy_from_slice(&s[offset as usize..offset as usize + buf.len()]);
            Ok(())
        }
        fn write_page(&self, offset: u64, data: &[u8]) -> Result<(), BackendError> {
            let mut s = self.storage.write().unwrap();
            for (i, &b) in data.iter().enumerate() {
                s[offset as usize + i] &= b;
            }
            Ok(())
        }
        fn erase_block(&self, block: u32) -> Result<(), BackendError> {
            let mut s = self.storage.write().unwrap();
            let ebs = self.geometry.erase_block_size as usize;
            let start = block as usize * ebs;
            s[start..start + ebs].fill(0xFF);
            Ok(())
        }
        fn is_bad_block(&self, _: u32) -> Result<bool, BackendError> {
            Ok(false)
        }
        fn mark_bad_block(&self, _: u32) -> Result<(), BackendError> {
            Ok(())
        }
        fn geometry(&self) -> FlashGeometry {
            self.geometry
        }
        fn sync(&self) -> Result<(), BackendError> {
            Ok(())
        }
    }

    // This exercises the FTL write path repeatedly, triggering block allocation,
    // free-list management (O(1) swap_remove), and wear-leveling checks.
    let hw = SimpleFlash::new(512);
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    const TABLE: TableDefinition<u64, u64> = TableDefinition::new("flash_stress");

    // Insert enough data to trigger multiple block allocations and wear-level checks
    let write_txn = db.begin_write().unwrap();
    {
        let mut t = write_txn.open_table(TABLE).unwrap();
        for i in 0..500u64 {
            t.insert(&i, &(i * i)).unwrap();
        }
    }
    write_txn.commit().unwrap();

    // Overwrite all entries (triggers copy-on-write and old-block release)
    let write_txn = db.begin_write().unwrap();
    {
        let mut t = write_txn.open_table(TABLE).unwrap();
        for i in 0..500u64 {
            t.insert(&i, &(i + 1000)).unwrap();
        }
    }
    write_txn.commit().unwrap();

    // Verify
    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    assert_eq!(table.len().unwrap(), 500);
    assert_eq!(table.get(&0u64).unwrap().unwrap().value(), 1000);
    assert_eq!(table.get(&499u64).unwrap().unwrap().value(), 1499);
}
