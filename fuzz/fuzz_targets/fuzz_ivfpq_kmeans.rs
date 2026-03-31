#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use redb::ivfpq::kmeans::{assign_nearest, kmeans, nearest_clusters};
use redb::probe_select::DiversityConfig;
use redb::DistanceMetric;

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    /// Which operation to run.
    op: FuzzKmeansOp,
}

#[derive(Arbitrary, Debug)]
enum FuzzKmeansOp {
    /// Run kmeans with small parameters.
    Kmeans {
        /// 0..3 maps to dim 2/4/8/16.
        dim_sel: u8,
        /// Number of vectors (clamped to 1..32).
        num_vectors: u8,
        /// Number of clusters (clamped to 1..8).
        k: u8,
        /// Max iterations (clamped to 1..5).
        max_iter: u8,
        /// 0..3 maps to metric.
        metric_sel: u8,
        /// Raw f32 data via u32 bits.
        data_bits: Vec<u32>,
    },
    /// Run assign_nearest with fuzz centroids.
    AssignNearest {
        dim_sel: u8,
        num_clusters: u8,
        metric_sel: u8,
        vector_bits: Vec<u32>,
        centroid_bits: Vec<u32>,
    },
    /// Run nearest_clusters with fuzz centroids.
    NearestClusters {
        dim_sel: u8,
        num_clusters: u8,
        nprobe: u8,
        metric_sel: u8,
        diversity_lambda: u8,
        query_bits: Vec<u32>,
        centroid_bits: Vec<u32>,
    },
    /// Edge case: empty input.
    KmeansEmpty { k: u8 },
    /// Edge case: single vector.
    KmeansSingle { dim_sel: u8, bits: Vec<u32> },
}

fn select_dim(sel: u8) -> usize {
    match sel % 4 {
        0 => 2,
        1 => 4,
        2 => 8,
        _ => 16,
    }
}

fn select_metric(sel: u8) -> DistanceMetric {
    match sel % 4 {
        0 => DistanceMetric::Cosine,
        1 => DistanceMetric::EuclideanSq,
        2 => DistanceMetric::DotProduct,
        _ => DistanceMetric::Manhattan,
    }
}

fn bits_to_f32(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fuzz_target!(|input: FuzzInput| {
    match input.op {
        FuzzKmeansOp::Kmeans {
            dim_sel,
            num_vectors,
            k,
            max_iter,
            metric_sel,
            data_bits,
        } => {
            let dim = select_dim(dim_sel);
            let num_vectors = (num_vectors as usize).clamp(1, 32);
            let k = (k as usize).clamp(1, 8);
            let max_iter = (max_iter as usize).clamp(1, 5);
            let metric = select_metric(metric_sel);

            // Build flat vector data of exactly num_vectors * dim floats.
            let raw = bits_to_f32(&data_bits);
            let needed = num_vectors * dim;
            let mut flat = vec![0.0f32; needed];
            let copy_len = raw.len().min(needed);
            flat[..copy_len].copy_from_slice(&raw[..copy_len]);

            // Replace NaN/Inf with 0.0 to avoid degenerate clustering.
            for v in &mut flat {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }

            let centroids = kmeans(&flat, dim, k, max_iter, metric);

            // Centroid count: k * dim (or less if k was clamped to n).
            let actual_k = centroids.len() / dim;
            assert!(
                actual_k <= k,
                "kmeans returned more clusters ({}) than requested ({})",
                actual_k,
                k
            );
            assert_eq!(
                centroids.len() % dim,
                0,
                "centroids length {} not multiple of dim {}",
                centroids.len(),
                dim
            );
        }

        FuzzKmeansOp::AssignNearest {
            dim_sel,
            num_clusters,
            metric_sel,
            vector_bits,
            centroid_bits,
        } => {
            let dim = select_dim(dim_sel);
            let num_clusters = (num_clusters as usize).clamp(1, 16);
            let metric = select_metric(metric_sel);

            let mut vector = bits_to_f32(&vector_bits);
            vector.resize(dim, 0.0);
            for v in &mut vector {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }

            let needed = num_clusters * dim;
            let raw = bits_to_f32(&centroid_bits);
            let mut centroids = vec![0.0f32; needed];
            let copy_len = raw.len().min(needed);
            centroids[..copy_len].copy_from_slice(&raw[..copy_len]);
            for v in &mut centroids {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }

            let (cluster_idx, dist) = assign_nearest(&vector, &centroids, dim, num_clusters, metric);

            assert!(
                (cluster_idx as usize) < num_clusters,
                "assign_nearest returned index {} >= num_clusters {}",
                cluster_idx,
                num_clusters
            );
            // Distance should be finite for finite inputs.
            assert!(
                dist.is_finite(),
                "assign_nearest returned non-finite distance: {}",
                dist
            );
        }

        FuzzKmeansOp::NearestClusters {
            dim_sel,
            num_clusters,
            nprobe,
            metric_sel,
            diversity_lambda,
            query_bits,
            centroid_bits,
        } => {
            let dim = select_dim(dim_sel);
            let num_clusters = (num_clusters as usize).clamp(1, 16);
            let nprobe = (nprobe as usize).clamp(1, num_clusters);
            let metric = select_metric(metric_sel);

            let mut query = bits_to_f32(&query_bits);
            query.resize(dim, 0.0);
            for v in &mut query {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }

            let needed = num_clusters * dim;
            let raw = bits_to_f32(&centroid_bits);
            let mut centroids = vec![0.0f32; needed];
            let copy_len = raw.len().min(needed);
            centroids[..copy_len].copy_from_slice(&raw[..copy_len]);
            for v in &mut centroids {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }

            let diversity = DiversityConfig {
                lambda: if diversity_lambda == 0 {
                    0.0
                } else {
                    f32::from(diversity_lambda) / 255.0
                },
            };

            let results = nearest_clusters(
                &query,
                &centroids,
                dim,
                num_clusters,
                nprobe,
                metric,
                diversity,
            );

            // Result length bounded by nprobe.
            assert!(
                results.len() <= nprobe,
                "nearest_clusters returned {} results, expected <= {}",
                results.len(),
                nprobe
            );

            // All cluster indices must be valid.
            for &(idx, _) in &results {
                assert!(
                    (idx as usize) < num_clusters,
                    "nearest_clusters returned invalid index {}",
                    idx
                );
            }
        }

        FuzzKmeansOp::KmeansEmpty { k } => {
            let centroids = kmeans(&[], 4, k as usize, 3, DistanceMetric::EuclideanSq);
            assert!(centroids.is_empty());

            let centroids = kmeans(&[1.0, 2.0, 3.0, 4.0], 4, 0, 3, DistanceMetric::EuclideanSq);
            assert!(centroids.is_empty());

            let centroids = kmeans(&[], 0, k as usize, 3, DistanceMetric::EuclideanSq);
            assert!(centroids.is_empty());
        }

        FuzzKmeansOp::KmeansSingle { dim_sel, bits } => {
            let dim = select_dim(dim_sel);
            let mut vector = bits_to_f32(&bits);
            vector.resize(dim, 0.0);
            for v in &mut vector {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
            let centroids = kmeans(&vector, dim, 1, 3, DistanceMetric::EuclideanSq);
            assert_eq!(centroids.len(), dim);
        }
    }
});
