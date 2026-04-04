//! Diversity-aware probe selection for multi-probe vector search.
//!
//! Provides a Greedy MMR (Maximal Marginal Relevance) selection algorithm
//! that balances proximity to the query with spatial diversity across
//! the selected probe set. Used by both IVF-PQ and fractal indices.

use alloc::vec;
use alloc::vec::Vec;

use crate::vector_ops::DistanceMetric;

/// Configuration for diversity-aware probe selection.
///
/// When `lambda` is 0.0 (default), selection is identical to pure
/// distance-based top-nprobe -- zero overhead. Increasing `lambda`
/// toward 1.0 trades some proximity for spatial diversity.
#[derive(Debug, Clone, Copy)]
pub struct DiversityConfig {
    /// Diversity weight in [0.0, 1.0].
    /// - 0.0: pure distance ranking (default, zero overhead)
    /// - 0.15-0.3: mild diversity, good for moderately skewed data
    /// - 0.5+: aggressive diversity, for highly clustered distributions
    pub lambda: f32,
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self { lambda: 0.0 }
    }
}

impl DiversityConfig {
    /// Returns `true` if diversity selection is active.
    #[inline]
    pub fn enabled(&self) -> bool {
        self.lambda > 0.0
    }
}

/// Select `nprobe` probes from a pre-sorted candidate list, balancing
/// proximity to the query with inter-probe diversity via Greedy MMR.
///
/// # Arguments
/// - `candidates`: `(id, distance_to_query)` pairs, sorted ascending by distance
/// - `candidate_centroids`: flat f32 array, `candidates.len() * dim` floats,
///   ordered 1:1 with `candidates`
/// - `dim`: vector dimensionality
/// - `nprobe`: number of probes to select
/// - `diversity`: diversity configuration
/// - `metric`: distance metric for inter-centroid distances
///
/// # Complexity
/// `O(nprobe * shortlist_size * dim)` where `shortlist_size = min(2 * nprobe, candidates.len())`.
/// When `lambda == 0.0`, returns immediately with a simple slice copy.
pub fn select_diverse_probes(
    candidates: &[(u32, f32)],
    candidate_centroids: &[f32],
    dim: usize,
    nprobe: usize,
    diversity: DiversityConfig,
    metric: DistanceMetric,
) -> Vec<(u32, f32)> {
    let n = nprobe.min(candidates.len());
    if n == 0 {
        return Vec::new();
    }

    // Fast path: no diversity or not enough candidates to be selective
    if !diversity.enabled() || n >= candidates.len() {
        return candidates[..n].to_vec();
    }

    if candidate_centroids.len() != candidates.len() * dim {
        // Length mismatch -- cannot safely index into centroid data.
        // Return the top-n by distance as a fallback rather than panicking.
        return candidates[..n].to_vec();
    }

    // Shortlist: top 2*nprobe by distance (already sorted)
    let shortlist_len = candidates.len().min(nprobe.saturating_mul(2));
    let shortlist = &candidates[..shortlist_len];
    let shortlist_centroids = &candidate_centroids[..shortlist_len * dim];

    let lambda = diversity.lambda.clamp(0.0, 1.0);

    // Normalize query distances to [0, 1]
    let d_min = shortlist[0].1;
    let d_max = shortlist[shortlist_len - 1].1;
    let d_range = if (d_max - d_min).abs() > f32::EPSILON {
        d_max - d_min
    } else {
        1.0
    };

    let mut selected: Vec<usize> = Vec::with_capacity(nprobe);
    let mut result: Vec<(u32, f32)> = Vec::with_capacity(nprobe);
    let mut is_selected = vec![false; shortlist_len];

    // Always select the closest candidate first
    selected.push(0);
    is_selected[0] = true;
    result.push(shortlist[0]);

    // Scratch buffer for per-candidate min-inter-distances
    let mut min_inter_dists = vec![0.0f32; shortlist_len];

    // Greedy MMR selection for remaining slots
    for _ in 1..nprobe {
        if selected.len() >= shortlist_len {
            break;
        }

        // Pass 1: compute min inter-centroid distance for each unselected
        // candidate and find the min/max for normalization.
        let mut inter_min: f32 = f32::INFINITY;
        let mut inter_max: f32 = f32::NEG_INFINITY;
        for j in 0..shortlist_len {
            if is_selected[j] {
                continue;
            }
            let centroid_j = &shortlist_centroids[j * dim..(j + 1) * dim];
            let mut min_d = f32::INFINITY;
            for &s in &selected {
                let centroid_s = &shortlist_centroids[s * dim..(s + 1) * dim];
                let d = metric.compute(centroid_j, centroid_s);
                if d < min_d {
                    min_d = d;
                }
            }
            min_inter_dists[j] = min_d;
            if min_d < inter_min {
                inter_min = min_d;
            }
            if min_d > inter_max {
                inter_max = min_d;
            }
        }

        // Shift-and-scale normalization: maps to [0, 1] regardless of sign.
        // Works correctly for all metrics including DotProduct (negative distances).
        let inter_range = if (inter_max - inter_min).abs() > f32::EPSILON {
            inter_max - inter_min
        } else {
            1.0
        };

        // Pass 2: score candidates with consistent normalization
        let mut best_idx = usize::MAX;
        let mut best_score = f32::INFINITY;
        for j in 0..shortlist_len {
            if is_selected[j] {
                continue;
            }
            let relevance = (shortlist[j].1 - d_min) / d_range;
            let diversity_score = (min_inter_dists[j] - inter_min) / inter_range;
            // MMR: lower = better (relevance low = close, diversity high = spread)
            let score = (1.0 - lambda) * relevance - lambda * diversity_score;
            if score < best_score {
                best_score = score;
                best_idx = j;
            }
        }

        if best_idx == usize::MAX {
            break;
        }

        selected.push(best_idx);
        is_selected[best_idx] = true;
        result.push(shortlist[best_idx]);
    }

    // Sort by original distance for consistent downstream processing
    result.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn make_candidates_and_centroids(
        positions: &[(f32, f32)],
        query: (f32, f32),
    ) -> (Vec<(u32, f32)>, Vec<f32>) {
        let dim = 2;
        let mut candidates: Vec<(u32, f32)> = positions
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| {
                let dist = (x - query.0).powi(2) + (y - query.1).powi(2);
                #[allow(clippy::cast_possible_truncation)]
                (i as u32, dist)
            })
            .collect();
        candidates
            .sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        // Build centroids in sorted order
        let mut centroids = Vec::with_capacity(candidates.len() * dim);
        for &(id, _) in &candidates {
            let (x, y) = positions[id as usize];
            centroids.push(x);
            centroids.push(y);
        }

        (candidates, centroids)
    }

    #[test]
    fn lambda_zero_identical_to_greedy() {
        let positions = vec![(1.0, 0.0), (1.1, 0.0), (1.2, 0.0), (0.0, 5.0), (0.0, 5.1)];
        let query = (0.0, 0.0);
        let (candidates, centroids) = make_candidates_and_centroids(&positions, query);

        let config = DiversityConfig { lambda: 0.0 };
        let result = select_diverse_probes(
            &candidates,
            &centroids,
            2,
            3,
            config,
            DistanceMetric::EuclideanSq,
        );

        // Should be the 3 closest by distance
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, candidates[0].0);
        assert_eq!(result[1].0, candidates[1].0);
        assert_eq!(result[2].0, candidates[2].0);
    }

    #[test]
    fn diversity_selects_from_both_clusters() {
        // Two tight groups: group A near (1,0), group B near (0,10)
        let positions = vec![
            (1.0, 0.0),
            (1.05, 0.0),
            (1.1, 0.0),
            (1.15, 0.0),
            (1.2, 0.0),
            (0.0, 10.0),
            (0.0, 10.1),
            (0.0, 10.2),
        ];
        let query = (0.0, 0.0);
        let (candidates, centroids) = make_candidates_and_centroids(&positions, query);

        // With lambda=0.0, nprobe=3: all from group A
        let greedy = select_diverse_probes(
            &candidates,
            &centroids,
            2,
            3,
            DiversityConfig { lambda: 0.0 },
            DistanceMetric::EuclideanSq,
        );
        let greedy_ids: Vec<u32> = greedy.iter().map(|x| x.0).collect();
        // All 3 should be from group A (ids correspond to positions 0-4)
        assert!(greedy_ids.iter().all(|&id| id <= 4));

        // With lambda=0.5, nprobe=3: should pick at least one from group B
        let diverse = select_diverse_probes(
            &candidates,
            &centroids,
            2,
            3,
            DiversityConfig { lambda: 0.5 },
            DistanceMetric::EuclideanSq,
        );
        let diverse_ids: Vec<u32> = diverse.iter().map(|x| x.0).collect();
        let has_group_b = diverse_ids.iter().any(|&id| id >= 5);
        assert!(
            has_group_b,
            "diversity should select from group B, got IDs: {diverse_ids:?}"
        );
    }

    #[test]
    fn single_probe_always_closest() {
        let positions = vec![(5.0, 0.0), (0.0, 10.0), (3.0, 3.0)];
        let query = (0.0, 0.0);
        let (candidates, centroids) = make_candidates_and_centroids(&positions, query);

        for lambda in [0.0, 0.5, 1.0] {
            let result = select_diverse_probes(
                &candidates,
                &centroids,
                2,
                1,
                DiversityConfig { lambda },
                DistanceMetric::EuclideanSq,
            );
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].0, candidates[0].0);
        }
    }

    #[test]
    fn nprobe_exceeds_candidates() {
        let positions = vec![(1.0, 0.0), (0.0, 1.0)];
        let query = (0.0, 0.0);
        let (candidates, centroids) = make_candidates_and_centroids(&positions, query);

        let result = select_diverse_probes(
            &candidates,
            &centroids,
            2,
            10,
            DiversityConfig { lambda: 0.5 },
            DistanceMetric::EuclideanSq,
        );
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn deterministic() {
        let positions = vec![(1.0, 0.0), (2.0, 0.0), (0.0, 3.0), (3.0, 3.0), (5.0, 5.0)];
        let query = (0.0, 0.0);
        let (candidates, centroids) = make_candidates_and_centroids(&positions, query);

        let config = DiversityConfig { lambda: 0.3 };
        let r1 = select_diverse_probes(
            &candidates,
            &centroids,
            2,
            3,
            config,
            DistanceMetric::EuclideanSq,
        );
        let r2 = select_diverse_probes(
            &candidates,
            &centroids,
            2,
            3,
            config,
            DistanceMetric::EuclideanSq,
        );
        assert_eq!(r1, r2);
    }

    #[test]
    fn empty_candidates() {
        let result = select_diverse_probes(
            &[],
            &[],
            2,
            5,
            DiversityConfig { lambda: 0.5 },
            DistanceMetric::EuclideanSq,
        );
        assert!(result.is_empty());
    }

    #[test]
    fn dot_product_diversity_selects_spread() {
        // DotProduct metric returns -dot_product(a,b), so distances can be negative.
        // Group A: unit-ish vectors near (1,0) -- all very similar to each other
        // Group B: unit-ish vectors near (0,1) -- orthogonal to group A
        // Query: (0.9, 0.1) normalized-ish -- closest to group A
        let dim = 2;
        let centroids_raw: Vec<(f32, f32)> = vec![
            (0.99, 0.0), // A0
            (0.98, 0.0), // A1
            (0.97, 0.0), // A2
            (0.96, 0.0), // A3
            (0.0, 0.99), // B0
            (0.0, 0.98), // B1
        ];
        let query = (0.9, 0.1);

        // Compute DotProduct distances (= -dot(q, c))
        let mut candidates: Vec<(u32, f32)> = centroids_raw
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| {
                let dist = -(query.0 * x + query.1 * y);
                #[allow(clippy::cast_possible_truncation)]
                (i as u32, dist)
            })
            .collect();
        candidates
            .sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        let mut centroids_flat = Vec::with_capacity(candidates.len() * dim);
        for &(id, _) in &candidates {
            let (x, y) = centroids_raw[id as usize];
            centroids_flat.push(x);
            centroids_flat.push(y);
        }

        // Without diversity: all from group A (closest by dot product)
        let greedy = select_diverse_probes(
            &candidates,
            &centroids_flat,
            dim,
            3,
            DiversityConfig { lambda: 0.0 },
            DistanceMetric::DotProduct,
        );
        let greedy_ids: Vec<u32> = greedy.iter().map(|x| x.0).collect();
        assert!(
            greedy_ids.iter().all(|&id| id <= 3),
            "greedy should pick all from group A, got {greedy_ids:?}"
        );

        // With diversity: should include at least one from group B
        let diverse = select_diverse_probes(
            &candidates,
            &centroids_flat,
            dim,
            3,
            DiversityConfig { lambda: 0.5 },
            DistanceMetric::DotProduct,
        );
        let diverse_ids: Vec<u32> = diverse.iter().map(|x| x.0).collect();
        let has_group_b = diverse_ids.iter().any(|&id| id >= 4);
        assert!(
            has_group_b,
            "diversity should select from group B with DotProduct, got IDs: {diverse_ids:?}"
        );
    }

    #[test]
    fn cosine_diversity_selects_spread() {
        // Cosine distance is in [0, 2]. Test that diversity works with this range.
        let dim = 2;
        // Group A: vectors pointing right, Group B: vectors pointing up
        let centroids_raw: Vec<(f32, f32)> = vec![
            (1.0, 0.1),
            (1.0, 0.15),
            (1.0, 0.2),
            (1.0, 0.25),
            (0.1, 1.0),
            (0.15, 1.0),
        ];
        let query = (1.0, 0.05);

        let mut candidates: Vec<(u32, f32)> = centroids_raw
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| {
                let dist = DistanceMetric::Cosine.compute(&[query.0, query.1], &[x, y]);
                #[allow(clippy::cast_possible_truncation)]
                (i as u32, dist)
            })
            .collect();
        candidates
            .sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        let mut centroids_flat = Vec::with_capacity(candidates.len() * dim);
        for &(id, _) in &candidates {
            let (x, y) = centroids_raw[id as usize];
            centroids_flat.push(x);
            centroids_flat.push(y);
        }

        let diverse = select_diverse_probes(
            &candidates,
            &centroids_flat,
            dim,
            3,
            DiversityConfig { lambda: 0.5 },
            DistanceMetric::Cosine,
        );
        let diverse_ids: Vec<u32> = diverse.iter().map(|x| x.0).collect();
        let has_group_b = diverse_ids.iter().any(|&id| id >= 4);
        assert!(
            has_group_b,
            "diversity should select from group B with Cosine, got IDs: {diverse_ids:?}"
        );
    }
}
