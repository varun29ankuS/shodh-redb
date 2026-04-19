// SIMD/scalar parity validation tests.
//
// The scalar fallback functions are private, so we validate parity indirectly:
// mathematical properties and known-value computations that would fail if the
// SIMD fast path diverged from the scalar reference. We also sweep dimensions
// that exercise exact SIMD chunks, tail-only paths, and chunk+tail remainders.

use shodh_redb::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random vector generator (LCG, same sequence on all
/// platforms regardless of SIMD availability).
fn make_vector(dim: usize, seed: u64) -> Vec<f32> {
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

fn assert_close(a: f32, b: f32, tol: f32) {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!(
        diff < tol * scale,
        "expected ~{b}, got {a}, diff={diff}, rel_tol={tol}"
    );
}

/// Naive dot product computed entirely in f64 to serve as a ground-truth
/// reference for tolerance checks.
fn naive_dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum()
}

/// Naive squared Euclidean distance in f64.
fn naive_euclidean_sq_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x as f64) - (y as f64);
            d * d
        })
        .sum()
}

/// Naive Manhattan distance in f64.
fn naive_manhattan_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x as f64) - (y as f64)).abs())
        .sum()
}

/// Naive cosine similarity in f64.
fn naive_cosine_sim_f64(a: &[f32], b: &[f32]) -> f64 {
    let dot = naive_dot_f64(a, b);
    let norm_a = naive_dot_f64(a, a).sqrt();
    let norm_b = naive_dot_f64(b, b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Key dimensions that exercise SIMD boundaries.
/// Pure tail: 1, 7. Exact chunks: 8, 16, 32, 64, 128, 256.
/// Chunk+tail: 15, 31, 33, 63, 255. Real embedding sizes: 384, 768.
const KEY_DIMS: &[usize] = &[1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 128, 255, 256, 384, 768];

// ---------------------------------------------------------------------------
// Dot product
// ---------------------------------------------------------------------------

#[test]
fn dot_product_known_values() {
    // [1, 2, 3] . [4, 5, 6] = 4 + 10 + 18 = 32
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];
    assert_close(dot_product(&a, &b), 32.0, 1e-6);

    // [1, 0, 0] . [0, 1, 0] = 0
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0];
    assert_close(dot_product(&a, &b), 0.0, 1e-6);

    // Single element: [3.5] . [2.0] = 7.0
    let a = vec![3.5f32];
    let b = vec![2.0f32];
    assert_close(dot_product(&a, &b), 7.0, 1e-6);
}

#[test]
fn dot_product_orthogonal_zero() {
    // Construct orthogonal vectors at various dimensions by interleaving zeros.
    for &dim in &[2, 8, 16, 32, 64] {
        let mut a = vec![0.0f32; dim];
        let mut b = vec![0.0f32; dim];
        for i in 0..dim {
            if i % 2 == 0 {
                a[i] = (i as f32) + 1.0;
            } else {
                b[i] = (i as f32) + 1.0;
            }
        }
        assert_close(dot_product(&a, &b), 0.0, 1e-6);
    }
}

#[test]
fn dot_product_self_equals_l2_norm_sq() {
    for &dim in KEY_DIMS {
        let v = make_vector(dim, 42);
        let dp = dot_product(&v, &v);
        let norm = l2_norm(&v);
        assert_close(dp, norm * norm, 1e-5);
    }
}

// ---------------------------------------------------------------------------
// Euclidean distance (squared)
// ---------------------------------------------------------------------------

#[test]
fn euclidean_self_distance_zero() {
    for &dim in KEY_DIMS {
        let v = make_vector(dim, 99);
        let d = euclidean_distance_sq(&v, &v);
        assert!(
            d.abs() < 1e-6,
            "dim={dim}: euclidean_distance_sq(v, v) = {d}, expected 0.0"
        );
    }
}

#[test]
fn euclidean_known_values() {
    // [1,0] vs [0,1]: squared distance = (1-0)^2 + (0-1)^2 = 2.0
    let a = vec![1.0f32, 0.0];
    let b = vec![0.0f32, 1.0];
    assert_close(euclidean_distance_sq(&a, &b), 2.0, 1e-6);

    // [0,0,0] vs [3,4,0]: squared distance = 9 + 16 = 25.0
    let a = vec![0.0f32, 0.0, 0.0];
    let b = vec![3.0f32, 4.0, 0.0];
    assert_close(euclidean_distance_sq(&a, &b), 25.0, 1e-6);
}

#[test]
fn euclidean_triangle_inequality() {
    // Triangle inequality: sqrt(d(a,c)) <= sqrt(d(a,b)) + sqrt(d(b,c))
    for &dim in &[8, 33, 128, 384] {
        let a = make_vector(dim, 1);
        let b = make_vector(dim, 2);
        let c = make_vector(dim, 3);

        let ab = euclidean_distance_sq(&a, &b).sqrt();
        let bc = euclidean_distance_sq(&b, &c).sqrt();
        let ac = euclidean_distance_sq(&a, &c).sqrt();

        assert!(
            ac <= ab + bc + 1e-4,
            "dim={dim}: triangle inequality violated: {ac} > {ab} + {bc}"
        );
    }
}

// ---------------------------------------------------------------------------
// Cosine similarity / distance
// ---------------------------------------------------------------------------

#[test]
fn cosine_identical_vectors_one() {
    for &dim in KEY_DIMS {
        let v = make_vector(dim, 77);
        let sim = cosine_similarity(&v, &v);
        assert_close(sim, 1.0, 1e-5);
    }
}

#[test]
fn cosine_opposite_vectors() {
    for &dim in KEY_DIMS {
        let v = make_vector(dim, 55);
        let neg_v: Vec<f32> = v.iter().map(|&x| -x).collect();
        let sim = cosine_similarity(&v, &neg_v);
        assert_close(sim, -1.0, 1e-5);
    }
}

#[test]
fn cosine_orthogonal_zero() {
    // Construct orthogonal vectors by placing non-zero values in
    // non-overlapping positions.
    for &dim in &[2, 8, 16, 32, 64, 128] {
        let mut a = vec![0.0f32; dim];
        let mut b = vec![0.0f32; dim];
        for i in 0..dim {
            if i % 2 == 0 {
                a[i] = 1.0;
            } else {
                b[i] = 1.0;
            }
        }
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-5,
            "dim={dim}: cosine_similarity of orthogonal vectors = {sim}, expected ~0"
        );
    }
}

#[test]
fn cosine_distance_complement() {
    for &dim in KEY_DIMS {
        let a = make_vector(dim, 10);
        let b = make_vector(dim, 20);
        let sim = cosine_similarity(&a, &b);
        let dist = cosine_distance(&a, &b);
        assert_close(dist, 1.0 - sim, 1e-5);
    }
}

// ---------------------------------------------------------------------------
// Manhattan distance
// ---------------------------------------------------------------------------

#[test]
fn manhattan_known_values() {
    // [1, 2, 3] vs [4, 6, 3] -> |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 6.0, 3.0];
    assert_close(manhattan_distance(&a, &b), 7.0, 1e-6);

    // Single element: [5.0] vs [-3.0] -> 8.0
    let a = vec![5.0f32];
    let b = vec![-3.0f32];
    assert_close(manhattan_distance(&a, &b), 8.0, 1e-6);
}

#[test]
fn manhattan_self_zero() {
    for &dim in KEY_DIMS {
        let v = make_vector(dim, 88);
        let d = manhattan_distance(&v, &v);
        assert!(
            d.abs() < 1e-6,
            "dim={dim}: manhattan_distance(v, v) = {d}, expected 0.0"
        );
    }
}

// ---------------------------------------------------------------------------
// Hamming distance
// ---------------------------------------------------------------------------

#[test]
fn hamming_identical_zero() {
    for len in [1, 7, 8, 15, 16, 31, 32, 33, 64, 128, 256] {
        let v: Vec<u8> = (0..len).map(|i| (i & 0xFF) as u8).collect();
        let d = hamming_distance(&v, &v);
        assert_eq!(d, 0, "len={len}: hamming_distance(v, v) = {d}, expected 0");
    }
}

#[test]
fn hamming_all_different() {
    // Complementary bytes: every bit differs -> 8 bits per byte.
    for len in [1, 8, 16, 32, 64] {
        let a: Vec<u8> = vec![0x55; len]; // 01010101
        let b: Vec<u8> = vec![0xAA; len]; // 10101010
        let expected = 8 * len as u32;
        let d = hamming_distance(&a, &b);
        assert_eq!(
            d, expected,
            "len={len}: hamming_distance = {d}, expected {expected}"
        );
    }
}

#[test]
fn hamming_known_values() {
    // Single byte: 0b0000_0000 vs 0b0000_0001 -> 1 bit
    assert_eq!(hamming_distance(&[0x00], &[0x01]), 1);

    // Single byte: 0xFF vs 0x00 -> 8 bits
    assert_eq!(hamming_distance(&[0xFF], &[0x00]), 8);

    // Two bytes: [0xFF, 0x00] vs [0x00, 0xFF] -> 16 bits
    assert_eq!(hamming_distance(&[0xFF, 0x00], &[0x00, 0xFF]), 16);

    // Three bytes with specific bit patterns:
    // 0b1010_1010 vs 0b1111_0000 -> bits differ at positions 0,1,2,3,5,7 = 6
    assert_eq!(hamming_distance(&[0xAA], &[0xF0]), 4);
}

// ---------------------------------------------------------------------------
// Cross-dimension consistency against naive f64 reference
// ---------------------------------------------------------------------------

#[test]
fn all_metrics_consistent_across_dimensions() {
    let tol = 1e-4;
    for &dim in KEY_DIMS {
        let a = make_vector(dim, 100);
        let b = make_vector(dim, 200);

        // Dot product
        let dp = dot_product(&a, &b);
        let dp_ref = naive_dot_f64(&a, &b) as f32;
        assert_close(dp, dp_ref, tol);

        // Euclidean squared
        let esq = euclidean_distance_sq(&a, &b);
        let esq_ref = naive_euclidean_sq_f64(&a, &b) as f32;
        assert_close(esq, esq_ref, tol);

        // Manhattan
        let man = manhattan_distance(&a, &b);
        let man_ref = naive_manhattan_f64(&a, &b) as f32;
        assert_close(man, man_ref, tol);

        // Cosine similarity
        let cs = cosine_similarity(&a, &b);
        let cs_ref = naive_cosine_sim_f64(&a, &b) as f32;
        assert_close(cs, cs_ref, tol);

        // Cosine distance
        let cd = cosine_distance(&a, &b);
        let cd_ref = 1.0 - cs_ref;
        assert_close(cd, cd_ref, tol);
    }
}

// ---------------------------------------------------------------------------
// l2_normalize
// ---------------------------------------------------------------------------

#[test]
fn l2_normalize_unit_length() {
    for &dim in KEY_DIMS {
        let v = make_vector(dim, 333);

        // In-place variant
        let mut v_mut = v.clone();
        l2_normalize(&mut v_mut);
        assert_close(l2_norm(&v_mut), 1.0, 1e-5);

        // Allocating variant
        let v_normed = l2_normalized(&v);
        assert_close(l2_norm(&v_normed), 1.0, 1e-5);

        // Direction preserved: cosine similarity with original should be 1.0
        let sim = cosine_similarity(&v, &v_normed);
        assert_close(sim, 1.0, 1e-5);
    }
}

// ---------------------------------------------------------------------------
// DistanceMetric enum dispatch
// ---------------------------------------------------------------------------

#[test]
fn distance_metric_enum_dispatch() {
    for &dim in &[1, 8, 33, 128, 384] {
        let a = make_vector(dim, 500);
        let b = make_vector(dim, 600);

        // Cosine: DistanceMetric::compute == cosine_distance
        let enum_val = DistanceMetric::Cosine.compute(&a, &b);
        let direct_val = cosine_distance(&a, &b);
        assert_close(enum_val, direct_val, 1e-6);

        // EuclideanSq: DistanceMetric::compute == euclidean_distance_sq
        let enum_val = DistanceMetric::EuclideanSq.compute(&a, &b);
        let direct_val = euclidean_distance_sq(&a, &b);
        assert_close(enum_val, direct_val, 1e-6);

        // DotProduct: DistanceMetric::compute == -dot_product (negated)
        let enum_val = DistanceMetric::DotProduct.compute(&a, &b);
        let direct_val = -dot_product(&a, &b);
        assert_close(enum_val, direct_val, 1e-6);

        // Manhattan: DistanceMetric::compute == manhattan_distance
        let enum_val = DistanceMetric::Manhattan.compute(&a, &b);
        let direct_val = manhattan_distance(&a, &b);
        assert_close(enum_val, direct_val, 1e-6);
    }
}
