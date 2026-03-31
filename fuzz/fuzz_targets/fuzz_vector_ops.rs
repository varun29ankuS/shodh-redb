#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use redb::{
    cosine_distance, cosine_similarity, dot_product, euclidean_distance_sq, hamming_distance,
    l2_norm, l2_normalize, l2_normalized, manhattan_distance, nearest_k, nearest_k_fixed,
    read_f32_le, write_f32_le, DistanceMetric,
};

/// Convert a slice of u32 to f32 via `from_bits`.
/// This naturally generates NaN, Inf, subnormals, and signed zeros.
fn bits_to_f32(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn bits_to_f32_array<const N: usize>(bits: &[u32]) -> [f32; N] {
    let mut arr = [0.0f32; N];
    for (i, val) in arr.iter_mut().enumerate() {
        *val = bits.get(i).map_or(0.0, |&b| f32::from_bits(b));
    }
    arr
}

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    /// Test all distance metrics between two vectors.
    DistanceMetrics {
        a_bits: Vec<u32>,
        b_bits: Vec<u32>,
    },
    /// Test hamming distance with arbitrary byte slices.
    HammingDistance {
        a: Vec<u8>,
        b: Vec<u8>,
    },
    /// Test l2_norm, l2_normalize, l2_normalized.
    L2Norm {
        bits: Vec<u32>,
    },
    /// Test nearest_k with dynamic vectors.
    NearestK {
        query_bits: Vec<u32>,
        vectors_bits: Vec<Vec<u32>>,
        k: u8,
    },
    /// Test nearest_k_fixed with 8-dim vectors.
    NearestKFixed {
        query_bits: [u32; 8],
        vectors_bits: Vec<[u32; 8]>,
        k: u8,
    },
    /// Test write_f32_le / read_f32_le roundtrip.
    F32LeRoundtrip {
        bits: Vec<u32>,
    },
    /// Test DistanceMetric::compute method.
    MetricCompute {
        metric_idx: u8,
        a_bits: Vec<u32>,
        b_bits: Vec<u32>,
    },
}

fuzz_target!(|op: FuzzOp| {
    match op {
        FuzzOp::DistanceMetrics { a_bits, b_bits } => {
            let a = bits_to_f32(&a_bits);
            let b = bits_to_f32(&b_bits);
            let min_len = a.len().min(b.len());
            if min_len == 0 {
                return;
            }
            let a = &a[..min_len];
            let b = &b[..min_len];

            // These must not panic.
            let _ = dot_product(a, b);
            let _ = euclidean_distance_sq(a, b);
            let _ = cosine_similarity(a, b);
            let _ = cosine_distance(a, b);
            let _ = manhattan_distance(a, b);

            // Self-distance properties (only for finite vectors).
            if a.iter().all(|x| x.is_finite()) {
                let euc = euclidean_distance_sq(a, a);
                assert!(
                    euc >= 0.0 || euc.is_nan(),
                    "euc self-distance must be >= 0: {euc}"
                );
            }
        }

        FuzzOp::HammingDistance { a, b } => {
            let min_len = a.len().min(b.len());
            if min_len == 0 {
                return;
            }
            let dist = hamming_distance(&a[..min_len], &b[..min_len]);
            // Hamming distance bounded by 8 * byte_count.
            assert!(
                dist <= (min_len as u32) * 8,
                "hamming distance {dist} exceeds max {}",
                min_len * 8
            );
            // Self-distance is zero.
            assert_eq!(hamming_distance(&a[..min_len], &a[..min_len]), 0);
        }

        FuzzOp::L2Norm { bits } => {
            let v = bits_to_f32(&bits);
            if v.is_empty() {
                return;
            }
            let norm = l2_norm(&v);
            assert!(norm >= 0.0 || norm.is_nan(), "l2_norm must be >= 0: {norm}");

            // l2_normalized must not panic.
            let normed = l2_normalized(&v);
            assert_eq!(normed.len(), v.len());

            // l2_normalize in-place must not panic.
            let mut v_mut = v.clone();
            l2_normalize(&mut v_mut);
            assert_eq!(v_mut.len(), v.len());

            // Normalized vector should have norm ~1.0 if input is finite non-zero.
            if v.iter().all(|x| x.is_finite()) && norm > 1e-10 {
                let normed_norm = l2_norm(&normed);
                assert!(
                    (normed_norm - 1.0).abs() < 0.01 || normed_norm.is_nan(),
                    "normalized vector norm should be ~1.0, got {normed_norm}"
                );
            }
        }

        FuzzOp::NearestK {
            query_bits,
            vectors_bits,
            k,
        } => {
            let query = bits_to_f32(&query_bits);
            if query.is_empty() {
                return;
            }
            let dim = query.len();
            let k = k as usize;

            let vectors: Vec<(u64, Vec<f32>)> = vectors_bits
                .iter()
                .enumerate()
                .map(|(i, bits)| {
                    let mut v = bits_to_f32(bits);
                    v.resize(dim, 0.0);
                    (i as u64, v)
                })
                .collect();

            let n = vectors.len();
            let results =
                nearest_k(vectors.into_iter(), &query, k, |a, b| {
                    euclidean_distance_sq(a, b)
                });

            // Length invariant: results <= min(k, n).
            assert!(
                results.len() <= k.min(n),
                "nearest_k returned {} results, expected <= {}",
                results.len(),
                k.min(n)
            );

            // Sorted ascending by distance (using total_cmp for NaN safety).
            for w in results.windows(2) {
                assert!(
                    w[0].distance.total_cmp(&w[1].distance).is_le(),
                    "results not sorted: {} > {}",
                    w[0].distance,
                    w[1].distance
                );
            }
        }

        FuzzOp::NearestKFixed {
            query_bits,
            vectors_bits,
            k,
        } => {
            let query: [f32; 8] = bits_to_f32_array(&query_bits);
            let k = k as usize;

            let vectors: Vec<(u64, [f32; 8])> = vectors_bits
                .iter()
                .enumerate()
                .map(|(i, bits)| (i as u64, bits_to_f32_array(bits)))
                .collect();

            let n = vectors.len();
            let results = nearest_k_fixed(vectors.into_iter(), &query, k, |a, b| {
                euclidean_distance_sq(a, b)
            });

            assert!(results.len() <= k.min(n));
            for w in results.windows(2) {
                assert!(w[0].distance.total_cmp(&w[1].distance).is_le());
            }
        }

        FuzzOp::F32LeRoundtrip { bits } => {
            let values = bits_to_f32(&bits);
            if values.is_empty() {
                return;
            }
            let byte_len = values.len() * 4;
            let mut buf = vec![0u8; byte_len];
            write_f32_le(&mut buf, &values);
            let restored = read_f32_le(&buf);
            assert_eq!(restored.len(), values.len());
            for (orig, rest) in values.iter().zip(restored.iter()) {
                assert_eq!(
                    orig.to_bits(),
                    rest.to_bits(),
                    "f32 LE roundtrip failed: {orig} != {rest}"
                );
            }
        }

        FuzzOp::MetricCompute {
            metric_idx,
            a_bits,
            b_bits,
        } => {
            let metric = match metric_idx % 4 {
                0 => DistanceMetric::Cosine,
                1 => DistanceMetric::EuclideanSq,
                2 => DistanceMetric::DotProduct,
                _ => DistanceMetric::Manhattan,
            };
            let a = bits_to_f32(&a_bits);
            let b = bits_to_f32(&b_bits);
            let min_len = a.len().min(b.len());
            if min_len == 0 {
                return;
            }
            // Must not panic.
            let _ = metric.compute(&a[..min_len], &b[..min_len]);
        }
    }
});
