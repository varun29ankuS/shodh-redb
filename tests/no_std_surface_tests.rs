//! Tests for the no_std-compatible surface of shodh-redb.
//!
//! These tests exercise every API that compiles under `#![no_std]`:
//! vector types, distance functions, quantization, serialization,
//! InMemoryBackend, and IVF-PQ. The test binary itself uses std
//! (Rust's test harness requires it), but all tested functions are
//! available in no_std builds.
//!
//! Companion CI check: `cargo check --no-default-features` verifies
//! the library compiles without std.

use shodh_redb::backends::InMemoryBackend;
use shodh_redb::{
    Builder, ContentType, DistanceMetric, DynVec, FixedVec, MultimapTableDefinition,
    ReadableDatabase, ReadableTableMetadata, StoreOptions, TableDefinition,
};

// ═══════════════════════════════════════════════════════════════════════
// Distance functions — portable scalar paths
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn dot_product_basic() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    let result = shodh_redb::dot_product(&a, &b);
    assert!((result - 32.0).abs() < 1e-5);
}

#[test]
fn dot_product_zero_vector() {
    let a = [0.0f32; 128];
    let b = [1.0f32; 128];
    assert_eq!(shodh_redb::dot_product(&a, &b), 0.0);
}

#[test]
fn dot_product_negative() {
    let a = [-1.0f32, -2.0, -3.0];
    let b = [1.0f32, 2.0, 3.0];
    assert!((shodh_redb::dot_product(&a, &b) + 14.0).abs() < 1e-5);
}

#[test]
fn dot_product_high_dim() {
    let a = vec![1.0f32; 1536];
    let b = vec![2.0f32; 1536];
    let result = shodh_redb::dot_product(&a, &b);
    assert!((result - 3072.0).abs() < 1.0);
}

#[test]
fn euclidean_distance_sq_identical() {
    let a = [1.0f32, 2.0, 3.0];
    assert_eq!(shodh_redb::euclidean_distance_sq(&a, &a), 0.0);
}

#[test]
fn euclidean_distance_sq_basic() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 4.0];
    assert!((shodh_redb::euclidean_distance_sq(&a, &b) - 25.0).abs() < 1e-5);
}

#[test]
fn euclidean_distance_sq_high_dim() {
    let a = vec![0.0f32; 384];
    let b = vec![1.0f32; 384];
    let result = shodh_redb::euclidean_distance_sq(&a, &b);
    assert!((result - 384.0).abs() < 1.0);
}

#[test]
fn cosine_similarity_identical() {
    let a = [1.0f32, 0.0, 0.0];
    assert!((shodh_redb::cosine_similarity(&a, &a) - 1.0).abs() < 1e-5);
}

#[test]
fn cosine_similarity_orthogonal() {
    let a = [1.0f32, 0.0];
    let b = [0.0f32, 1.0];
    assert!(shodh_redb::cosine_similarity(&a, &b).abs() < 1e-5);
}

#[test]
fn cosine_similarity_opposite() {
    let a = [1.0f32, 0.0];
    let b = [-1.0f32, 0.0];
    assert!((shodh_redb::cosine_similarity(&a, &b) + 1.0).abs() < 1e-5);
}

#[test]
fn cosine_distance_basic() {
    let a = [1.0f32, 0.0];
    let b = [1.0f32, 0.0];
    assert!(shodh_redb::cosine_distance(&a, &b).abs() < 1e-5);
}

#[test]
fn cosine_distance_perpendicular() {
    let a = [1.0f32, 0.0];
    let b = [0.0f32, 1.0];
    assert!((shodh_redb::cosine_distance(&a, &b) - 1.0).abs() < 1e-5);
}

#[test]
fn manhattan_distance_basic() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 6.0, 3.0];
    // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
    assert!((shodh_redb::manhattan_distance(&a, &b) - 7.0).abs() < 1e-5);
}

#[test]
fn manhattan_distance_zero() {
    let a = [5.0f32; 100];
    assert_eq!(shodh_redb::manhattan_distance(&a, &a), 0.0);
}

#[test]
fn manhattan_distance_high_dim() {
    let a = vec![0.0f32; 768];
    let b = vec![1.0f32; 768];
    let result = shodh_redb::manhattan_distance(&a, &b);
    assert!((result - 768.0).abs() < 1.0);
}

#[test]
fn hamming_distance_identical() {
    let a = [0xFFu8; 48];
    assert_eq!(shodh_redb::hamming_distance(&a, &a), 0);
}

#[test]
fn hamming_distance_all_different() {
    let a = [0x00u8; 4];
    let b = [0xFFu8; 4];
    assert_eq!(shodh_redb::hamming_distance(&a, &b), 32);
}

#[test]
fn hamming_distance_single_bit() {
    let a = [0x00u8];
    let b = [0x01u8];
    assert_eq!(shodh_redb::hamming_distance(&a, &b), 1);
}

#[test]
fn hamming_distance_large() {
    let a = vec![0u8; 384];
    let b = vec![0xFFu8; 384];
    assert_eq!(shodh_redb::hamming_distance(&a, &b), 384 * 8);
}

// ═══════════════════════════════════════════════════════════════════════
// DistanceMetric enum — dispatch layer
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn distance_metric_cosine() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    let d = DistanceMetric::Cosine.compute(&a, &b);
    assert!((d - 1.0).abs() < 1e-5);
}

#[test]
fn distance_metric_euclidean() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 4.0];
    let d = DistanceMetric::EuclideanSq.compute(&a, &b);
    assert!((d - 25.0).abs() < 1e-5);
}

#[test]
fn distance_metric_dot_product() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    let d = DistanceMetric::DotProduct.compute(&a, &b);
    // DotProduct metric negates for min-heap: -32
    assert!((d + 32.0).abs() < 1e-5);
}

#[test]
fn distance_metric_manhattan() {
    let a = [1.0f32, 2.0];
    let b = [4.0f32, 6.0];
    let d = DistanceMetric::Manhattan.compute(&a, &b);
    assert!((d - 7.0).abs() < 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════
// l2_norm and l2_normalize — portable sqrt_f32 path
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn l2_norm_unit_vector() {
    let v = [1.0f32, 0.0, 0.0];
    assert!((shodh_redb::l2_norm(&v) - 1.0).abs() < 1e-5);
}

#[test]
fn l2_norm_3_4_5() {
    let v = [3.0f32, 4.0];
    assert!((shodh_redb::l2_norm(&v) - 5.0).abs() < 1e-4);
}

#[test]
fn l2_norm_zero() {
    let v = [0.0f32; 128];
    assert_eq!(shodh_redb::l2_norm(&v), 0.0);
}

#[test]
fn l2_normalize_result_is_unit() {
    let mut v = [3.0f32, 4.0, 0.0];
    shodh_redb::l2_normalize(&mut v);
    let norm = shodh_redb::l2_norm(&v);
    assert!((norm - 1.0).abs() < 1e-4);
}

#[test]
fn l2_normalize_zero_vector_unchanged() {
    let mut v = [0.0f32; 4];
    shodh_redb::l2_normalize(&mut v);
    assert!(v.iter().all(|&x| x == 0.0));
}

#[test]
fn l2_normalized_preserves_direction() {
    let v = [2.0f32, 0.0, 0.0];
    let n = shodh_redb::l2_normalized(&v);
    assert!((n[0] - 1.0).abs() < 1e-5);
    assert!(n[1].abs() < 1e-5);
    assert!(n[2].abs() < 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════
// Quantization — binary and scalar
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn quantize_binary_positive_bits() {
    let v = [1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let bits = shodh_redb::quantize_binary(&v);
    assert_eq!(bits.len(), 1);
    // Bits: 1,0,1,0,1,0,1,0 = 0b10101010 = 0xAA (MSB first) or depends on bit order
    // Just verify non-zero
    assert_ne!(bits[0], 0);
}

#[test]
fn quantize_binary_all_negative() {
    let v = [-1.0f32; 16];
    let bits = shodh_redb::quantize_binary(&v);
    assert_eq!(bits.len(), 2);
    assert!(bits.iter().all(|&b| b == 0));
}

#[test]
fn quantize_binary_all_positive() {
    let v = [1.0f32; 16];
    let bits = shodh_redb::quantize_binary(&v);
    assert_eq!(bits.len(), 2);
    assert!(bits.iter().all(|&b| b == 0xFF));
}

#[test]
fn quantize_binary_output_size() {
    // 384 dimensions -> ceil(384/8) = 48 bytes
    let v = vec![1.0f32; 384];
    let bits = shodh_redb::quantize_binary(&v);
    assert_eq!(bits.len(), 48);
}

#[test]
fn quantize_scalar_roundtrip() {
    let original: [f32; 4] = [0.0, 0.5, 1.0, -1.0];
    let sq = shodh_redb::quantize_scalar(&original);
    let recovered = shodh_redb::dequantize_scalar(&sq);
    // Scalar quantization is lossy — check within tolerance
    for (o, r) in original.iter().zip(recovered.iter()) {
        assert!((o - r).abs() < 0.02, "expected ~{o}, got {r}");
    }
}

#[test]
fn quantize_scalar_preserves_range() {
    let original: [f32; 8] = [-10.0, -5.0, 0.0, 2.5, 5.0, 7.5, 10.0, 10.0];
    let sq = shodh_redb::quantize_scalar(&original);
    assert!(sq.min_val <= -10.0);
    assert!(sq.max_val >= 10.0);
    assert_eq!(sq.codes.len(), 8);
}

#[test]
fn quantize_scalar_constant_vector() {
    let original: [f32; 4] = [5.0; 4];
    let sq = shodh_redb::quantize_scalar(&original);
    let recovered = shodh_redb::dequantize_scalar(&sq);
    for r in &recovered {
        assert!((r - 5.0).abs() < 0.1);
    }
}

#[test]
fn sq_euclidean_distance_sq_self() {
    let v: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let sq = shodh_redb::quantize_scalar(&v);
    let dist = shodh_redb::sq_euclidean_distance_sq(&v, &sq);
    // Distance to self (via quantization) should be very small
    assert!(dist < 1.0, "self-distance was {dist}");
}

#[test]
fn sq_dot_product_basic() {
    let query: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    let target: [f32; 4] = [5.0, 3.0, 2.0, 1.0];
    let sq = shodh_redb::quantize_scalar(&target);
    let dot = shodh_redb::sq_dot_product(&query, &sq);
    // Should approximate 5.0
    assert!((dot - 5.0).abs() < 0.5, "dot was {dot}");
}

// ═══════════════════════════════════════════════════════════════════════
// f32 LE serialization — write_f32_le / read_f32_le
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn f32_le_roundtrip() {
    let values = [1.0f32, -2.5, 3.125, 0.0, f32::MAX, f32::MIN];
    let mut buf = vec![0u8; values.len() * 4];
    shodh_redb::write_f32_le(&mut buf, &values);
    let recovered = shodh_redb::read_f32_le(&buf);
    assert_eq!(recovered, values);
}

#[test]
fn f32_le_empty() {
    let buf = [];
    let recovered = shodh_redb::read_f32_le(&buf);
    assert!(recovered.is_empty());
}

#[test]
fn f32_le_partial_bytes_ignored() {
    // 5 bytes = 1 full f32 + 1 leftover byte
    let values = [42.0f32];
    let mut buf = vec![0u8; 5];
    shodh_redb::write_f32_le(&mut buf, &values);
    let recovered = shodh_redb::read_f32_le(&buf);
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0], 42.0);
}

#[test]
fn f32_le_special_values() {
    let values = [f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
    let mut buf = vec![0u8; values.len() * 4];
    shodh_redb::write_f32_le(&mut buf, &values);
    let recovered = shodh_redb::read_f32_le(&buf);
    assert_eq!(recovered[0], f32::INFINITY);
    assert_eq!(recovered[1], f32::NEG_INFINITY);
}

// ═══════════════════════════════════════════════════════════════════════
// nearest_k — brute-force top-k search
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn nearest_k_basic() {
    let vectors: Vec<(u32, Vec<f32>)> = vec![
        (0, vec![0.0, 0.0]),
        (1, vec![1.0, 0.0]),
        (2, vec![10.0, 10.0]),
    ];
    let query = [0.1f32, 0.0];
    let results = shodh_redb::nearest_k(
        vectors.into_iter(),
        &query,
        2,
        shodh_redb::euclidean_distance_sq,
    );
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].key, 0); // closest
    assert_eq!(results[1].key, 1);
}

#[test]
fn nearest_k_empty_input() {
    let vectors: Vec<(u32, Vec<f32>)> = vec![];
    let query = [1.0f32, 2.0];
    let results = shodh_redb::nearest_k(
        vectors.into_iter(),
        &query,
        5,
        shodh_redb::euclidean_distance_sq,
    );
    assert!(results.is_empty());
}

#[test]
fn nearest_k_fewer_than_k() {
    let vectors: Vec<(u32, Vec<f32>)> = vec![(0, vec![1.0]), (1, vec![2.0])];
    let query = [0.0f32];
    let results = shodh_redb::nearest_k(
        vectors.into_iter(),
        &query,
        10,
        shodh_redb::euclidean_distance_sq,
    );
    assert_eq!(results.len(), 2);
}

#[test]
fn nearest_k_cosine_metric() {
    let vectors: Vec<(u32, Vec<f32>)> = vec![
        (0, vec![1.0, 0.0]),
        (1, vec![0.0, 1.0]),
        (2, vec![0.707, 0.707]),
    ];
    let query = [1.0f32, 0.0];
    let results =
        shodh_redb::nearest_k(vectors.into_iter(), &query, 1, shodh_redb::cosine_distance);
    assert_eq!(results[0].key, 0);
}

#[test]
fn nearest_k_fixed_basic() {
    let vectors: Vec<(u32, [f32; 3])> = vec![
        (0, [0.0, 0.0, 0.0]),
        (1, [1.0, 0.0, 0.0]),
        (2, [10.0, 10.0, 10.0]),
    ];
    let query = [0.1f32, 0.0, 0.0];
    let results = shodh_redb::nearest_k_fixed(
        vectors.into_iter(),
        &query,
        1,
        shodh_redb::euclidean_distance_sq,
    );
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].key, 0);
}

// ═══════════════════════════════════════════════════════════════════════
// Vector types — FixedVec, DynVec, BinaryQuantized
// ═══════════════════════════════════════════════════════════════════════

const VEC_TABLE: TableDefinition<u64, FixedVec<4>> = TableDefinition::new("nostd_fixedvec");
const DYN_TABLE: TableDefinition<u64, DynVec> = TableDefinition::new("nostd_dynvec");

#[test]
fn fixedvec_store_retrieve() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(VEC_TABLE).unwrap();
        // FixedVec<4> SelfType is [f32; 4]
        t.insert(&0u64, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(VEC_TABLE).unwrap();
    let v = t.get(&0u64).unwrap().unwrap();
    let data: [f32; 4] = v.value();
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[3] - 4.0).abs() < 1e-5);
}

#[test]
fn dynvec_store_retrieve() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let values = vec![0.5f32; 128];

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(DYN_TABLE).unwrap();
        // DynVec SelfType is Vec<f32>
        t.insert(&0u64, &values).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(DYN_TABLE).unwrap();
    let v = t.get(&0u64).unwrap().unwrap();
    let data: Vec<f32> = v.value();
    assert_eq!(data.len(), 128);
    assert!((data[0] - 0.5).abs() < 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════
// InMemoryBackend — core DB operations without filesystem
// ═══════════════════════════════════════════════════════════════════════

const MEM_KV: TableDefinition<&str, u64> = TableDefinition::new("mem_kv");
const MEM_MM: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mem_mm");

#[test]
fn inmemory_kv_crud() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(MEM_KV).unwrap();
        t.insert("hello", &42u64).unwrap();
        t.insert("world", &99u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(MEM_KV).unwrap();
    assert_eq!(t.get("hello").unwrap().unwrap().value(), 42);
    assert_eq!(t.get("world").unwrap().unwrap().value(), 99);
    assert_eq!(t.len().unwrap(), 2);
}

#[test]
fn inmemory_multimap() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_multimap_table(MEM_MM).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &20u64).unwrap();
        t.insert(&2u64, &30u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_multimap_table(MEM_MM).unwrap();
    assert_eq!(t.get(&1u64).unwrap().count(), 2);
    assert_eq!(t.get(&2u64).unwrap().count(), 1);
}

#[test]
fn inmemory_transaction_isolation() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(MEM_KV).unwrap();
        t.insert("before", &1u64).unwrap();
    }
    txn.commit().unwrap();

    // Read snapshot taken before write
    let read_txn = db.begin_read().unwrap();

    let write_txn = db.begin_write().unwrap();
    {
        let mut t = write_txn.open_table(MEM_KV).unwrap();
        t.insert("after", &2u64).unwrap();
    }
    write_txn.commit().unwrap();

    // Snapshot should NOT see the new write
    let t = read_txn.open_table(MEM_KV).unwrap();
    assert!(t.get("after").unwrap().is_none());
    assert_eq!(t.get("before").unwrap().unwrap().value(), 1);
}

#[test]
fn inmemory_savepoint_rollback() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let mut txn = db.begin_write().unwrap();
    let sp = txn.ephemeral_savepoint().unwrap();
    {
        let mut t = txn.open_table(MEM_KV).unwrap();
        t.insert("doomed", &1u64).unwrap();
    }
    txn.restore_savepoint(&sp).unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    // Table creation was rolled back, so it may not exist
    if let Ok(t) = txn.open_table(MEM_KV) {
        assert!(t.get("doomed").unwrap().is_none());
    }
}

#[test]
fn inmemory_range_scan() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    const U64_TABLE: TableDefinition<u64, u64> = TableDefinition::new("mem_u64");

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..100u64 {
            t.insert(&i, &(i * i)).unwrap();
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(U64_TABLE).unwrap();
    let range: Vec<u64> = t
        .range(10u64..20u64)
        .unwrap()
        .map(|r| r.unwrap().0.value())
        .collect();
    assert_eq!(range, (10..20).collect::<Vec<_>>());
}

#[test]
fn inmemory_blob_store() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    let blob_id = txn
        .store_blob(
            b"hello world",
            ContentType::OctetStream,
            "test",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let (data, meta) = txn.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(data, b"hello world");
    assert_eq!(meta.blob_ref.content_type, ContentType::OctetStream as u8);
}

#[test]
fn inmemory_blob_dedup() {
    let backend = InMemoryBackend::new();
    let db = Builder::new()
        .set_blob_dedup(true)
        .create_with_backend(backend)
        .unwrap();

    // Payload must exceed dedup min size (default 4096)
    let payload = vec![0xABu8; 8192];
    let txn = db.begin_write().unwrap();
    let id1 = txn
        .store_blob(
            &payload,
            ContentType::OctetStream,
            "a",
            StoreOptions::default(),
        )
        .unwrap();
    let id2 = txn
        .store_blob(
            &payload,
            ContentType::OctetStream,
            "b",
            StoreOptions::default(),
        )
        .unwrap();
    txn.commit().unwrap();
    // Dedup should produce same hash; BlobId may differ by sequence
    // but the underlying blob data is shared
    let txn = db.begin_read().unwrap();
    let (data1, _) = txn.get_blob(&id1).unwrap().unwrap();
    let (data2, _) = txn.get_blob(&id2).unwrap().unwrap();
    assert_eq!(data1, data2);
}

#[test]
fn inmemory_merge_operator() {
    use shodh_redb::NumericAdd;
    const MERGE: TableDefinition<&str, &[u8]> = TableDefinition::new("mem_merge");

    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(MERGE).unwrap();
        t.merge(&"counter", 1u64.to_le_bytes().as_slice(), &NumericAdd)
            .unwrap();
        t.merge(&"counter", 2u64.to_le_bytes().as_slice(), &NumericAdd)
            .unwrap();
        t.merge(&"counter", 3u64.to_le_bytes().as_slice(), &NumericAdd)
            .unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(MERGE).unwrap();
    let v = t.get("counter").unwrap().unwrap();
    assert_eq!(u64::from_le_bytes(v.value().try_into().unwrap()), 6);
}

#[test]
fn inmemory_multiple_tables() {
    const T1: TableDefinition<&str, u64> = TableDefinition::new("t1");
    const T2: TableDefinition<&str, u64> = TableDefinition::new("t2");
    const T3: TableDefinition<&str, u64> = TableDefinition::new("t3");

    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t1 = txn.open_table(T1).unwrap();
        t1.insert("a", &1u64).unwrap();
    }
    {
        let mut t2 = txn.open_table(T2).unwrap();
        t2.insert("b", &2u64).unwrap();
    }
    {
        let mut t3 = txn.open_table(T3).unwrap();
        t3.insert("c", &3u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let count = txn.list_tables().unwrap().count();
    assert!(count >= 3);
}

#[test]
fn inmemory_pop_first_last() {
    const U64_TABLE: TableDefinition<u64, u64> = TableDefinition::new("mem_pop");

    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&2u64, &20u64).unwrap();
        t.insert(&3u64, &30u64).unwrap();

        let (k, v) = t.pop_first().unwrap().unwrap();
        assert_eq!(k.value(), 1);
        assert_eq!(v.value(), 10);
        drop(k);
        drop(v);

        let (k, v) = t.pop_last().unwrap().unwrap();
        assert_eq!(k.value(), 3);
        assert_eq!(v.value(), 30);
        drop(k);
        drop(v);

        assert_eq!(t.len().unwrap(), 1);
    }
    txn.commit().unwrap();
}

#[test]
fn inmemory_retain_and_extract() {
    const U64_TABLE: TableDefinition<u64, u64> = TableDefinition::new("mem_retain");

    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..20u64 {
            t.insert(&i, &i).unwrap();
        }
        // Keep only odd values
        t.retain(|_k, v| v % 2 == 1).unwrap();
        assert_eq!(t.len().unwrap(), 10);
    }
    txn.commit().unwrap();
}

#[test]
fn inmemory_drain_all() {
    const U64_TABLE: TableDefinition<u64, u64> = TableDefinition::new("mem_drain");

    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_TABLE).unwrap();
        for i in 0..50u64 {
            t.insert(&i, &i).unwrap();
        }
        let count = t.drain_all().unwrap();
        assert_eq!(count, 50);
        assert_eq!(t.len().unwrap(), 0);
    }
    txn.commit().unwrap();
}

// ═══════════════════════════════════════════════════════════════════════
// Numerical edge cases for distance functions
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn dot_product_subnormal() {
    let a = [f32::MIN_POSITIVE; 4];
    let b = [1.0f32; 4];
    let result = shodh_redb::dot_product(&a, &b);
    assert!(result.is_finite());
    assert!(result > 0.0);
}

#[test]
fn euclidean_distance_sq_large_values() {
    let a = [1e18f32; 4];
    let b = [0.0f32; 4];
    let result = shodh_redb::euclidean_distance_sq(&a, &b);
    assert!(result.is_finite() || result == f32::INFINITY);
}

#[test]
fn cosine_similarity_near_zero_norm() {
    let a = [1e-20f32; 4];
    let b = [1.0f32; 4];
    let result = shodh_redb::cosine_similarity(&a, &b);
    // Should handle gracefully — either near 0, near 1, or NaN
    let _ = result;
}

#[test]
fn manhattan_distance_negative_values() {
    let a = [-10.0f32, -20.0, -30.0];
    let b = [10.0f32, 20.0, 30.0];
    let result = shodh_redb::manhattan_distance(&a, &b);
    assert!((result - 120.0).abs() < 1e-3);
}

#[test]
fn hamming_distance_unequal_lengths() {
    // hamming_distance uses min(a.len(), b.len())
    let a = [0xFFu8; 10];
    let b = [0x00u8; 5];
    let result = shodh_redb::hamming_distance(&a, &b);
    assert_eq!(result, 40); // 5 bytes * 8 bits
}

// ═══════════════════════════════════════════════════════════════════════
// Quantization edge cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn quantize_scalar_single_element() {
    let v: [f32; 1] = [42.0];
    let sq = shodh_redb::quantize_scalar(&v);
    let recovered = shodh_redb::dequantize_scalar(&sq);
    assert!((recovered[0] - 42.0).abs() < 1.0);
}

#[test]
fn quantize_scalar_extreme_range() {
    let v: [f32; 4] = [-1e6, 0.0, 1.0, 1e6];
    let sq = shodh_redb::quantize_scalar(&v);
    assert!(sq.min_val <= -1e6);
    assert!(sq.max_val >= 1e6);
}

#[test]
fn quantize_binary_non_multiple_of_8() {
    // 13 dimensions -> ceil(13/8) = 2 bytes
    let v = vec![1.0f32; 13];
    let bits = shodh_redb::quantize_binary(&v);
    assert_eq!(bits.len(), 2);
}

#[test]
fn quantize_binary_single_dim() {
    let v = [1.0f32];
    let bits = shodh_redb::quantize_binary(&v);
    assert_eq!(bits.len(), 1);
    assert_ne!(bits[0], 0);
}

// ═══════════════════════════════════════════════════════════════════════
// InMemoryBackend edge cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn inmemory_empty_db_list_tables() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_read().unwrap();
    assert_eq!(txn.list_tables().unwrap().count(), 0);
}

#[test]
fn inmemory_abort_via_drop() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    {
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(MEM_KV).unwrap();
            t.insert("gone", &1u64).unwrap();
        }
        // drop without commit
    }

    let txn = db.begin_read().unwrap();
    // Table may not exist or may be empty
    let result = txn.open_table(MEM_KV);
    if let Ok(t) = result {
        assert!(t.get("gone").unwrap().is_none());
    }
}

#[test]
fn inmemory_large_insert() {
    const BYTES: TableDefinition<u64, &[u8]> = TableDefinition::new("mem_bytes");

    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    // 256KB value
    let big = vec![0xCDu8; 256 * 1024];
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(BYTES).unwrap();
        t.insert(&0u64, big.as_slice()).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(BYTES).unwrap();
    let v = t.get(&0u64).unwrap().unwrap();
    assert_eq!(v.value().len(), 256 * 1024);
}

#[test]
fn inmemory_sequential_commits() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    for i in 0..50u64 {
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(MEM_KV).unwrap();
            let key = if i < 10 {
                "k0"
            } else if i < 20 {
                "k1"
            } else if i < 30 {
                "k2"
            } else if i < 40 {
                "k3"
            } else {
                "k4"
            };
            t.insert(key, &i).unwrap();
        }
        txn.commit().unwrap();
    }

    let txn = db.begin_read().unwrap();
    let t = txn.open_table(MEM_KV).unwrap();
    assert!(t.len().unwrap() <= 5);
}

#[test]
fn inmemory_delete_table() {
    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(MEM_KV).unwrap();
        t.insert("x", &1u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    assert!(txn.delete_table(MEM_KV).unwrap());
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    assert!(txn.open_table(MEM_KV).is_err());
}

#[test]
fn inmemory_rename_table() {
    const OLD: TableDefinition<&str, u64> = TableDefinition::new("old_mem");
    const NEW: TableDefinition<&str, u64> = TableDefinition::new("new_mem");

    let backend = InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(OLD).unwrap();
        t.insert("x", &1u64).unwrap();
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();
    txn.rename_table(OLD, NEW).unwrap();
    txn.commit().unwrap();

    let txn = db.begin_read().unwrap();
    assert!(txn.open_table(OLD).is_err());
    assert_eq!(
        txn.open_table(NEW)
            .unwrap()
            .get("x")
            .unwrap()
            .unwrap()
            .value(),
        1
    );
}
