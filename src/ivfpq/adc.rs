use alloc::vec::Vec;

use crate::vector_ops::{DistanceMetric, dot_product, euclidean_distance_sq, manhattan_distance};

use super::pq::Codebooks;

// ---------------------------------------------------------------------------
// ADC -- Asymmetric Distance Computation
// ---------------------------------------------------------------------------

/// Precomputed lookup table for fast approximate distance computation.
///
/// For each sub-quantizer `m` (`0..num_subvectors`) and each codeword `k`
/// (0..256), stores the distance from the query sub-vector to that codeword's
/// centroid. Total storage: `num_subvectors x 256` f32 values.
///
/// At scan time the approximate distance to a PQ-encoded vector is the sum
/// of `num_subvectors` table lookups -- extremely fast.
pub struct AdcTable {
    /// Flat array: `distances[m * 256 + k]`.
    distances: Vec<f32>,
    num_subvectors: usize,
}

impl AdcTable {
    /// Build the ADC lookup table from a query vector and codebooks.
    ///
    /// For Cosine metric the query should already be L2-normalized before
    /// calling this method (normalization is the caller's responsibility at
    /// the index level, matching the normalisation applied at insert time).
    pub fn build(query: &[f32], codebooks: &Codebooks, metric: DistanceMetric) -> Self {
        let m = codebooks.num_subvectors;
        let sub_dim = codebooks.sub_dim;
        let required_len = m.saturating_mul(sub_dim);
        if query.len() < required_len {
            return Self {
                distances: Vec::new(),
                num_subvectors: 0,
            };
        }
        let mut distances = Vec::with_capacity(m * 256);

        for sub_idx in 0..m {
            let q_sub = &query[sub_idx * sub_dim..(sub_idx + 1) * sub_dim];
            for k in 0..256 {
                let centroid = codebooks.centroid(sub_idx, k);
                let d = subvector_distance(q_sub, centroid, metric);
                distances.push(d);
            }
        }

        Self {
            distances,
            num_subvectors: m,
        }
    }

    /// Compute the approximate distance from the query to a PQ-encoded vector.
    ///
    /// `pq_codes` must have length `num_subvectors`. Each byte is a codebook
    /// index. The result is the sum of precomputed sub-vector distances.
    #[inline]
    #[cfg(test)]
    pub fn approximate_distance(&self, pq_codes: &[u8]) -> f32 {
        let len = pq_codes.len().min(self.num_subvectors);
        let mut dist = 0.0f32;
        for (m, &code) in pq_codes[..len].iter().enumerate() {
            let idx = m * 256 + code as usize;
            if let Some(&d) = self.distances.get(idx) {
                dist += d;
            }
        }
        dist
    }

    /// Access the raw f32 distances (for `IntAdcTable` construction).
    pub(crate) fn distances(&self) -> &[f32] {
        &self.distances
    }

    /// Number of sub-quantizers.
    pub(crate) fn num_subvectors(&self) -> usize {
        self.num_subvectors
    }
}

// ---------------------------------------------------------------------------
// IntAdcTable -- integer-quantized ADC for chip-friendly computation
// ---------------------------------------------------------------------------

/// Quantized u16 ADC table for integer-only distance accumulation.
///
/// Constructed from a float ADC table by affine-mapping all per-entry
/// distances into `[0, 65535]`. The monotonic mapping preserves ranking
/// order within a single cluster, so the u32 accumulation loop uses no
/// floating-point operations.
///
/// After accumulation, `to_f32()` converts the u32 sum back to an
/// approximate f32 distance for cross-cluster comparison in the heap.
pub struct IntAdcTable {
    /// Flat array: `distances[m * 256 + code]` is a u16 distance.
    distances: Vec<u16>,
    num_subvectors: usize,
    /// Per-entry scale: `(max - min) / 65535.0`.
    scale: f32,
    /// Per-entry offset: the global minimum f32 distance across all entries.
    offset: f32,
}

impl IntAdcTable {
    /// Build a quantized integer ADC table from a query vector and codebooks.
    pub fn build(query: &[f32], codebooks: &Codebooks, metric: DistanceMetric) -> Self {
        let float_adc = AdcTable::build(query, codebooks, metric);
        Self::from_float(&float_adc)
    }

    /// Quantize an existing float ADC table to u16.
    fn from_float(adc: &AdcTable) -> Self {
        let src = adc.distances();
        let n_sub = adc.num_subvectors();
        if src.is_empty() {
            return Self {
                distances: Vec::new(),
                num_subvectors: 0,
                scale: 0.0,
                offset: 0.0,
            };
        }

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &d in src {
            if d < min_val {
                min_val = d;
            }
            if d > max_val {
                max_val = d;
            }
        }

        let range = max_val - min_val;
        let inv_range = if range < 1e-30 { 0.0 } else { 65535.0 / range };

        let distances: Vec<u16> = src
            .iter()
            .map(|&d| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    ((d - min_val) * inv_range + 0.5) as u16
                }
            })
            .collect();

        let per_entry_scale = if range < 1e-30 { 0.0 } else { range / 65535.0 };

        Self {
            distances,
            num_subvectors: n_sub,
            scale: per_entry_scale,
            offset: min_val,
        }
    }

    /// Compute the approximate distance as a u32 sum of u16 lookups.
    ///
    /// The inner loop is pure integer: u16 loads + u32 accumulation.
    /// No floating-point operations. Uses unchecked indexing for maximum
    /// throughput on the hot scan path (~40K calls per search).
    #[inline]
    pub(crate) fn approximate_distance(&self, pq_codes: &[u8]) -> u32 {
        debug_assert!(pq_codes.len() >= self.num_subvectors);
        debug_assert!(self.distances.len() >= self.num_subvectors * 256);
        let mut dist = 0u32;
        for m in 0..self.num_subvectors {
            // SAFETY: `pq_codes` length validated by `ClusterBlobRef` (>= num_subvectors).
            // `self.distances` has exactly `num_subvectors * 256` entries (invariant of build()).
            // `code` is u8 so `m * 256 + code` <= `num_subvectors * 256 - 1`.
            unsafe {
                let code = *pq_codes.get_unchecked(m);
                dist += u32::from(*self.distances.get_unchecked(m * 256 + code as usize));
            }
        }
        dist
    }

    /// Convert an accumulated u32 distance back to approximate f32.
    ///
    /// The u32 sum is the sum of `num_subvectors` quantized u16 entries.
    /// Each entry was quantized as `(f32_val - offset) / scale`, so:
    /// `f32_total = u32_sum * scale + num_subvectors * offset`.
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn to_f32(&self, dist_u32: u32) -> f32 {
        dist_u32 as f32 * self.scale + self.num_subvectors as f32 * self.offset
    }
}

impl core::fmt::Debug for IntAdcTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IntAdcTable")
            .field("num_subvectors", &self.num_subvectors)
            .field("table_entries", &self.distances.len())
            .field("scale", &self.scale)
            .field("offset", &self.offset)
            .finish()
    }
}

impl core::fmt::Debug for AdcTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AdcTable")
            .field("num_subvectors", &self.num_subvectors)
            .field("table_entries", &self.distances.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Sub-vector distance computation per metric
// ---------------------------------------------------------------------------

/// Compute the distance between a query sub-vector and a codebook centroid.
///
/// The distance function used depends on the metric:
/// - `EuclideanSq`: squared Euclidean (sums of squared diffs -- additive over
///   sub-vectors, so the total PQ distance is the sum of sub-vector distances).
/// - `DotProduct`: negative dot product (additive: total = sum of sub-products).
/// - `Cosine`: same as `DotProduct` (query is L2-normalised at index level,
///   stored vectors are L2-normalised at insert time).
/// - `Manhattan`: L1 distance (additive over sub-vectors).
#[inline]
fn subvector_distance(query_sub: &[f32], centroid: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::EuclideanSq => euclidean_distance_sq(query_sub, centroid),
        DistanceMetric::DotProduct | DistanceMetric::Cosine => -dot_product(query_sub, centroid),
        DistanceMetric::Manhattan => manhattan_distance(query_sub, centroid),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ivfpq::pq::train_codebooks;

    #[test]
    fn adc_matches_exact_distance() {
        // 8-dim, 2 sub-vectors. Train codebooks on simple data.
        #[rustfmt::skip]
        let training: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0,  0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,  1.0, 0.0, 0.0, 0.0,
        ];
        let codebooks = train_codebooks(&training, 8, 2, 25, DistanceMetric::EuclideanSq).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let adc = AdcTable::build(&query, &codebooks, DistanceMetric::EuclideanSq);

        // Encode the first training vector (same as query).
        let codes = codebooks.encode(&training[0..8]);
        let approx_dist = adc.approximate_distance(&codes);

        // Exact distance to self should be 0 or near-zero via PQ approximation.
        assert!(
            approx_dist < 0.5,
            "expected near-zero approx distance for self, got {approx_dist}"
        );
    }

    #[test]
    fn adc_ordering_preserved() {
        // Verify that ADC distances preserve relative ordering for simple cases.
        #[rustfmt::skip]
        let training: Vec<f32> = vec![
            0.0, 0.0,  0.0, 0.0,
            1.0, 1.0,  1.0, 1.0,
            5.0, 5.0,  5.0, 5.0,
            10.0, 10.0, 10.0, 10.0,
        ];
        let codebooks = train_codebooks(&training, 4, 2, 25, DistanceMetric::EuclideanSq).unwrap();
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let adc = AdcTable::build(&query, &codebooks, DistanceMetric::EuclideanSq);

        let codes_near = codebooks.encode(&[1.0, 1.0, 1.0, 1.0]);
        let codes_far = codebooks.encode(&[10.0, 10.0, 10.0, 10.0]);

        let d_near = adc.approximate_distance(&codes_near);
        let d_far = adc.approximate_distance(&codes_far);

        assert!(
            d_near < d_far,
            "ordering violated: near={d_near}, far={d_far}"
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn int_adc_ranking_matches_float() {
        // Train on varied data so ADC table has non-trivial distances.
        let mut training = Vec::with_capacity(20 * 8);
        for i in 0..20_u16 {
            for d in 0..8_u16 {
                training.push(f32::from(i) * 0.5 + f32::from(d) * 0.1);
            }
        }
        let codebooks = train_codebooks(&training, 8, 2, 25, DistanceMetric::EuclideanSq).unwrap();
        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let float_adc = AdcTable::build(&query, &codebooks, DistanceMetric::EuclideanSq);
        let int_adc = IntAdcTable::build(&query, &codebooks, DistanceMetric::EuclideanSq);

        // Encode all training vectors and rank by both methods.
        let mut float_dists: Vec<(usize, f32)> = (0..20)
            .map(|i| {
                let codes = codebooks.encode(&training[i * 8..(i + 1) * 8]);
                (i, float_adc.approximate_distance(&codes))
            })
            .collect();
        let mut int_dists: Vec<(usize, u32)> = (0..20)
            .map(|i| {
                let codes = codebooks.encode(&training[i * 8..(i + 1) * 8]);
                (i, int_adc.approximate_distance(&codes))
            })
            .collect();

        float_dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        int_dists.sort_by_key(|e| e.1);

        let float_order: Vec<usize> = float_dists.iter().map(|e| e.0).collect();
        let int_order: Vec<usize> = int_dists.iter().map(|e| e.0).collect();

        // Rankings should be identical (monotonic quantization).
        assert_eq!(
            float_order, int_order,
            "integer ADC ranking diverged from float ADC"
        );
    }

    #[test]
    fn int_adc_to_f32_accuracy() {
        let mut training = Vec::with_capacity(10 * 8);
        for i in 0..10_u16 {
            for d in 0..8_u16 {
                training.push(f32::from(i) * 1.5 + f32::from(d) * 0.3);
            }
        }
        let codebooks = train_codebooks(&training, 8, 2, 25, DistanceMetric::EuclideanSq).unwrap();
        let query = vec![2.0, 3.0, 1.0, 4.0, 0.5, 2.5, 3.5, 1.5];

        let float_adc = AdcTable::build(&query, &codebooks, DistanceMetric::EuclideanSq);
        let int_adc = IntAdcTable::build(&query, &codebooks, DistanceMetric::EuclideanSq);

        for i in 0..10 {
            let codes = codebooks.encode(&training[i * 8..(i + 1) * 8]);
            let f_dist = float_adc.approximate_distance(&codes);
            let i_dist = int_adc.approximate_distance(&codes);
            let recovered = int_adc.to_f32(i_dist);

            // Tolerance: num_subvectors * scale / 2 (rounding error per entry).
            let tol = 2.0 * int_adc.scale + 1e-6;
            assert!(
                (recovered - f_dist).abs() < tol,
                "to_f32 inaccurate for vec {i}: float={f_dist}, recovered={recovered}, tol={tol}"
            );
        }
    }

    #[test]
    fn int_adc_empty_table() {
        let codebooks = Codebooks {
            data: Vec::new(),
            num_subvectors: 0,
            sub_dim: 0,
        };
        let int_adc = IntAdcTable::build(&[], &codebooks, DistanceMetric::EuclideanSq);
        assert_eq!(int_adc.approximate_distance(&[]), 0);
        assert!(int_adc.to_f32(0).abs() < f32::EPSILON);
    }
}
