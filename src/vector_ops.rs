use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{self, Debug};

use crate::vector::SQVec;

#[cfg(all(target_arch = "x86_64", feature = "std"))]
mod simd_x86;

/// Portable `f32::sqrt` that works in `no_std` on wasm32.
///
/// On targets with `std`, delegates to the hardware/libm-backed `f32::sqrt()`.
/// On `no_std` (e.g. wasm32 without wasi), uses a bit-level Newton's method
/// implementation that converges in a fixed number of iterations.
#[inline]
fn sqrt_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        if x < 0.0 || x.is_nan() {
            return f32::NAN;
        }
        if x == 0.0 || x.is_infinite() {
            return x;
        }
        // Initial estimate via bit manipulation (fast inverse sqrt trick variant)
        let bits = x.to_bits();
        let guess_bits = (bits >> 1) + 0x1FC0_0000;
        let mut guess = f32::from_bits(guess_bits);
        // Newton-Raphson iterations (5 iterations for full f32 precision)
        guess = 0.5 * (guess + x / guess);
        guess = 0.5 * (guess + x / guess);
        guess = 0.5 * (guess + x / guess);
        guess = 0.5 * (guess + x / guess);
        guess = 0.5 * (guess + x / guess);
        guess
    }
}

// ---------------------------------------------------------------------------
// Distance metric enum
// ---------------------------------------------------------------------------

/// Specifies the distance metric for vector similarity search.
///
/// Lower distance values indicate more similar vectors for all metrics.
///
/// # Usage
///
/// ```rust,ignore
/// use shodh_redb::{DistanceMetric, FixedVec, TableDefinition, ReadableTable};
///
/// let query = [1.0f32, 0.0, 0.0];
/// let metric = DistanceMetric::Cosine;
///
/// // Scan and rank vectors
/// for entry in table.iter()? {
///     let (key, guard) = entry?;
///     let vec = guard.value();
///     let dist = metric.compute(&query, &vec);
///     println!("{}: {}", key.value(), dist);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceMetric {
    /// Cosine distance: `1.0 - cosine_similarity(a, b)`. Range `[0.0, 2.0]`.
    Cosine,
    /// Squared Euclidean distance: `sum((a_i - b_i)^2)`. Range `[0.0, inf)`.
    EuclideanSq,
    /// Dot product distance: `-dot_product(a, b)`. Negate so lower = more similar.
    /// Use with L2-normalized vectors for equivalent cosine ranking without sqrt.
    DotProduct,
    /// Manhattan (L1) distance: `sum(|a_i - b_i|)`. Range `[0.0, inf)`.
    Manhattan,
}

impl DistanceMetric {
    /// Computes the distance between two f32 vectors using this metric.
    ///
    /// Lower values indicate more similar vectors for all metrics.
    ///
    /// Returns [`f32::MAX`] when the vectors have mismatched dimensions
    /// (truncated or corrupted data) to prevent garbage results from being
    /// promoted to the top of nearest-neighbor heaps. Also returns
    /// [`f32::MAX`] when the computed distance is NaN (e.g. due to NaN
    /// elements in the input vectors) to avoid silent NaN propagation
    /// through search results.
    #[inline]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        let d = match self {
            Self::Cosine => cosine_distance(a, b),
            Self::EuclideanSq => euclidean_distance_sq(a, b),
            Self::DotProduct => -dot_product(a, b),
            Self::Manhattan => manhattan_distance(a, b),
        };
        if d.is_nan() { f32::MAX } else { d }
    }
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cosine => f.write_str("cosine"),
            Self::EuclideanSq => f.write_str("euclidean_sq"),
            Self::DotProduct => f.write_str("dot_product"),
            Self::Manhattan => f.write_str("manhattan"),
        }
    }
}

// ---------------------------------------------------------------------------
// Core distance functions
// ---------------------------------------------------------------------------

/// Computes the dot product of two f32 slices.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`. Callers that need graceful mismatch
/// handling should use [`DistanceMetric::compute`] instead.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_product: dimension mismatch");
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected; slices have equal length (asserted above).
            return unsafe { simd_x86::dot_product_avx2(a, b) };
        }
    }
    dot_product_scalar(a, b)
}

#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes the squared Euclidean distance between two f32 slices.
///
/// Returns the sum of squared element-wise differences. Take the square root
/// for the actual Euclidean distance, but the squared form is sufficient for
/// nearest-neighbor comparisons and avoids the sqrt cost.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn euclidean_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "euclidean_distance_sq: dimension mismatch"
    );
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected; slices have equal length (asserted above).
            return unsafe { simd_x86::euclidean_distance_sq_avx2(a, b) };
        }
    }
    euclidean_distance_sq_scalar(a, b)
}

#[inline]
fn euclidean_distance_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Computes the cosine similarity between two f32 slices.
///
/// Returns a value in `[-1.0, 1.0]` where 1.0 means identical direction,
/// 0.0 means orthogonal, and -1.0 means opposite direction.
///
/// Returns 0.0 if either vector has zero magnitude.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine_similarity: dimension mismatch");
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected; slices have equal length (asserted above).
            return unsafe { simd_x86::cosine_similarity_avx2(a, b) };
        }
    }
    cosine_similarity_scalar(a, b)
}

#[inline]
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = sqrt_f32(norm_a) * sqrt_f32(norm_b);
    if denom == 0.0 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Computes the cosine distance between two f32 slices.
///
/// Defined as `1.0 - cosine_similarity(a, b)`, returning a value in `[0.0, 2.0]`
/// where 0.0 means identical direction and 2.0 means opposite direction.
///
/// Returns 1.0 if either vector has zero magnitude.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Computes the Manhattan (L1) distance between two f32 slices.
///
/// Returns the sum of absolute element-wise differences.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "manhattan_distance: dimension mismatch");
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected; slices have equal length (asserted above).
            return unsafe { simd_x86::manhattan_distance_avx2(a, b) };
        }
    }
    manhattan_distance_scalar(a, b)
}

#[inline]
fn manhattan_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Computes the Hamming distance between two byte slices interpreted as binary vectors.
///
/// Counts the number of bits that differ between `a` and `b`. Useful for binary
/// embeddings (e.g., Cohere binary, Matryoshka quantized vectors).
///
/// If lengths differ, computes over the shorter length.
#[inline]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len().min(b.len());
    let a = &a[..len];
    let b = &b[..len];
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected; slices are trimmed to equal length above.
            return unsafe { simd_x86::hamming_distance_avx2(a, b) };
        }
    }
    hamming_distance_scalar(a, b)
}

#[inline]
fn hamming_distance_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Computes the L2 (Euclidean) norm of a vector.
///
/// Returns `sqrt(sum(x_i^2))`.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    sqrt_f32(v.iter().map(|x| x * x).sum::<f32>())
}

/// Normalizes a vector to unit length (L2 norm = 1.0) in place.
///
/// After normalization, `dot_product(v, v) ~= 1.0` and `cosine_similarity`
/// reduces to a simple `dot_product`, which is significantly faster.
///
/// If the vector has zero magnitude, it is left unchanged. For vectors with
/// extremely large elements (where the raw norm overflows to infinity), the
/// vector is first scaled down by the maximum absolute element value before
/// computing the norm to avoid producing a zero vector.
#[inline]
pub fn l2_normalize(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm.is_finite() && norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    } else if !norm.is_finite() {
        // Norm overflowed to Inf. Scale down by max absolute value first,
        // then normalize the scaled vector to get correct unit direction.
        let max_abs = v.iter().fold(0.0f32, |acc, &x| {
            let a = x.abs();
            if a > acc { a } else { acc }
        });
        if max_abs == 0.0 || !max_abs.is_finite() {
            return;
        }
        let inv_max = 1.0 / max_abs;
        for x in v.iter_mut() {
            *x *= inv_max;
        }
        let scaled_norm = l2_norm(v);
        if scaled_norm.is_finite() && scaled_norm > 0.0 {
            let inv = 1.0 / scaled_norm;
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
    }
}

/// Returns a new L2-normalized copy of the input vector.
///
/// If the input has zero magnitude, returns a zero vector of the same length.
#[inline]
pub fn l2_normalized(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    l2_normalize(&mut out);
    out
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/// Converts an f32 vector to a binary quantized representation.
///
/// Each f32 dimension is mapped to a single bit: 1 if positive, 0 otherwise.
/// The result is packed into bytes (MSB-first within each byte), with the
/// output length equal to `ceil(input.len() / 8)`.
///
/// This gives 32x compression over f32 storage. Use
/// [`hamming_distance`] to compare binary vectors.
///
/// # Example
///
/// ```rust,ignore
/// let v = [1.0f32, -0.5, 0.3, -0.1, 0.0, 0.7, -0.2, 0.9];
/// let bq = shodh_redb::quantize_binary(&v);
/// // bit pattern: [1,0,1,0, 0,1,0,1] = 0b10100101 = 0xA5
/// assert_eq!(bq, vec![0xA5]);
/// ```
pub fn quantize_binary(v: &[f32]) -> Vec<u8> {
    let byte_count = v.len().div_ceil(8);
    let mut result = vec![0u8; byte_count];
    for (i, &val) in v.iter().enumerate() {
        if val > 0.0 {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8); // MSB-first
            result[byte_idx] |= 1 << bit_idx;
        }
    }
    result
}

/// Scalar-quantizes an f32 vector to u8 codes with min/max scale factors.
///
/// Maps each f32 value linearly to the `[0, 255]` range based on the vector's
/// min and max values. Returns an [`SQVec`] containing the scale factors and codes.
///
/// This gives approximately 4x compression over f32 storage with bounded
/// quantization error of `(max - min) / 510` per dimension.
///
/// # Example
///
/// ```rust,ignore
/// let v = [0.0f32, 0.5, 1.0, 0.25];
/// let sq: SQVec<4> = shodh_redb::quantize_scalar(&v);
/// assert_eq!(sq.min_val, 0.0);
/// assert_eq!(sq.max_val, 1.0);
/// assert_eq!(sq.codes[2], 255); // max maps to 255
/// ```
pub fn quantize_scalar<const N: usize>(v: &[f32; N]) -> SQVec<N> {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &x in v {
        if x < min_val {
            min_val = x;
        }
        if x > max_val {
            max_val = x;
        }
    }

    let mut codes = [0u8; N];
    let range = max_val - min_val;
    if !range.is_finite() {
        // Input contained NaN or Inf -- quantization is meaningless.
        // Return zero codes with clamped min/max so dequantize produces 0.0.
        return SQVec {
            min_val: 0.0,
            max_val: 0.0,
            codes,
        };
    }
    if range > 0.0 {
        let inv_range = 255.0 / range;
        for (i, &x) in v.iter().enumerate() {
            // Quantize to [0, 255]: value is guaranteed non-negative and <= 255.5
            // because x is clamped within [min_val, max_val].
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let q = ((x - min_val) * inv_range + 0.5) as u8;
            codes[i] = q;
        }
    }
    // If range == 0 all codes stay 0, and dequantize returns min_val for all

    SQVec {
        min_val,
        max_val,
        codes,
    }
}

/// Dequantizes an [`SQVec`] back to an array of f32 values.
///
/// This is a convenience wrapper around [`SQVec::dequantize`].
#[inline]
pub fn dequantize_scalar<const N: usize>(sq: &SQVec<N>) -> [f32; N] {
    sq.dequantize()
}

/// Computes approximate squared Euclidean distance between an f32 query and
/// a scalar-quantized vector, dequantizing on the fly.
///
/// This avoids materializing the full f32 vector when doing distance
/// comparisons during search, reducing memory bandwidth.
///
/// Returns [`f32::MAX`] if the scale factors are non-finite or the result
/// is NaN, preventing silent corruption of search rankings.
#[inline]
pub fn sq_euclidean_distance_sq<const N: usize>(query: &[f32; N], sq: &SQVec<N>) -> f32 {
    let range = sq.max_val - sq.min_val;
    if !range.is_finite() {
        return f32::MAX;
    }
    if range == 0.0 {
        // All codes dequantize to min_val -- compute exact per-dimension distances.
        let d: f32 = query
            .iter()
            .map(|&q| {
                let diff = q - sq.min_val;
                diff * diff
            })
            .sum();
        return if d.is_nan() { f32::MAX } else { d };
    }
    let scale = range / 255.0;
    let mut sum = 0.0f32;
    for (i, &q) in query.iter().enumerate() {
        let dequant = sq.min_val + f32::from(sq.codes[i]) * scale;
        let diff = q - dequant;
        sum += diff * diff;
    }
    if sum.is_nan() { f32::MAX } else { sum }
}

/// Computes approximate dot product between an f32 query and a scalar-quantized
/// vector, dequantizing on the fly.
///
/// Returns `0.0` if the scale factors are non-finite. Returns the raw dot
/// product otherwise (callers negate for distance ranking via
/// [`DistanceMetric::DotProduct`]).
#[inline]
pub fn sq_dot_product<const N: usize>(query: &[f32; N], sq: &SQVec<N>) -> f32 {
    let range = sq.max_val - sq.min_val;
    if !range.is_finite() {
        return 0.0;
    }
    if range == 0.0 {
        let d = query.iter().sum::<f32>() * sq.min_val;
        return if d.is_nan() { 0.0 } else { d };
    }
    let scale = range / 255.0;
    let mut sum = 0.0f32;
    for (i, &q) in query.iter().enumerate() {
        let dequant = sq.min_val + f32::from(sq.codes[i]) * scale;
        sum += q * dequant;
    }
    if sum.is_nan() { 0.0 } else { sum }
}

// ---------------------------------------------------------------------------
// Top-K scan
// ---------------------------------------------------------------------------

/// A scored result from a nearest-neighbor search.
#[derive(Debug, Clone)]
pub struct Neighbor<K> {
    /// The key of the matching row.
    pub key: K,
    /// The distance from the query vector (lower = more similar).
    pub distance: f32,
}

impl<K> PartialEq for Neighbor<K> {
    fn eq(&self, other: &Self) -> bool {
        // Treat NaN as equal to NaN for heap consistency (IEEE NaN != NaN breaks Eq).
        self.distance.to_bits() == other.distance.to_bits()
    }
}

impl<K> Eq for Neighbor<K> {}

impl<K> PartialOrd for Neighbor<K> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<K> Ord for Neighbor<K> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // BinaryHeap is a max-heap; we want the *largest* distance at the top
        // so we can efficiently evict it. NaN sorts as greater-than everything
        // so NaN entries sit at the heap root and are evicted first.
        self.distance.total_cmp(&other.distance)
    }
}

/// Brute-force top-k nearest neighbor scan over an iterator of `(key, vector)` pairs.
///
/// Returns up to `k` nearest neighbors sorted by ascending distance (closest first).
/// The distance function should return lower values for more similar vectors.
///
/// This is the fundamental building block for vector search. Higher-level index
/// structures (IVF, HNSW) use this for scanning candidate shortlists.
///
/// # Example
///
/// ```rust,ignore
/// use shodh_redb::{nearest_k, DistanceMetric, FixedVec, ReadableTable};
///
/// let query = [1.0f32, 0.0, 0.0, 0.0];
/// let metric = DistanceMetric::Cosine;
///
/// let results = nearest_k(
///     table.iter()?.map(|r| {
///         let (k, v) = r.unwrap();
///         (k.value(), v.value())
///     }),
///     &query,
///     10,
///     |a, b| metric.compute(a, b),
/// );
///
/// for neighbor in &results {
///     println!("key={}, distance={}", neighbor.key, neighbor.distance);
/// }
/// ```
pub fn nearest_k<K, I, F>(iter: I, query: &[f32], k: usize, distance_fn: F) -> Vec<Neighbor<K>>
where
    I: Iterator<Item = (K, Vec<f32>)>,
    F: Fn(&[f32], &[f32]) -> f32,
{
    if k == 0 {
        return Vec::new();
    }

    // Max-heap of size k: the root is the worst (largest distance) candidate.
    // When we find something better, we pop the worst and push the new one.
    let mut heap: BinaryHeap<Neighbor<K>> = BinaryHeap::with_capacity(k + 1);

    for (key, vec) in iter {
        let dist = distance_fn(query, &vec);
        if heap.len() < k {
            heap.push(Neighbor {
                key,
                distance: dist,
            });
        } else if heap
            .peek()
            .is_some_and(|worst| dist.total_cmp(&worst.distance).is_lt())
        {
            heap.pop();
            heap.push(Neighbor {
                key,
                distance: dist,
            });
        }
    }

    let mut results: Vec<Neighbor<K>> = heap.into_vec();
    results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    results
}

/// Brute-force top-k scan with a fixed-size array query (zero-copy variant).
///
/// Same as [`nearest_k`] but takes `[f32; N]` vectors from the iterator,
/// avoiding the `Vec<f32>` allocation overhead for fixed-dimension tables.
pub fn nearest_k_fixed<K, I, F, const N: usize>(
    iter: I,
    query: &[f32; N],
    k: usize,
    distance_fn: F,
) -> Vec<Neighbor<K>>
where
    I: Iterator<Item = (K, [f32; N])>,
    F: Fn(&[f32], &[f32]) -> f32,
{
    if k == 0 {
        return Vec::new();
    }

    let mut heap: BinaryHeap<Neighbor<K>> = BinaryHeap::with_capacity(k + 1);

    for (key, vec) in iter {
        let dist = distance_fn(query.as_slice(), vec.as_slice());
        if heap.len() < k {
            heap.push(Neighbor {
                key,
                distance: dist,
            });
        } else if heap
            .peek()
            .is_some_and(|worst| dist.total_cmp(&worst.distance).is_lt())
        {
            heap.pop();
            heap.push(Neighbor {
                key,
                distance: dist,
            });
        }
    }

    let mut results: Vec<Neighbor<K>> = heap.into_vec();
    results.sort_by(|a, b| a.distance.total_cmp(&b.distance));

    results
}

// ---------------------------------------------------------------------------
// LE byte helpers
// ---------------------------------------------------------------------------

/// Writes a slice of f32 values as little-endian bytes into a destination buffer.
///
/// Useful for populating `insert_reserve` buffers when using `FixedVec<N>`.
///
/// If the buffer is smaller than `values.len() * 4`, writes only as many
/// complete f32 values as fit. If `values` is shorter, only those values
/// are written.
#[inline]
pub fn write_f32_le(dest: &mut [u8], values: &[f32]) {
    let count = (dest.len() / 4).min(values.len());
    #[cfg(target_endian = "little")]
    {
        let byte_len = count * 4;
        // SAFETY: On LE targets, f32 memory layout matches LE bytes.
        // `count` ensures we don't read past `values` or write past `dest`.
        unsafe {
            core::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                dest.as_mut_ptr(),
                byte_len,
            );
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        for (i, val) in values.iter().enumerate().take(count) {
            let start = i * 4;
            dest[start..start + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
}

/// Reads little-endian f32 values from a byte slice.
///
/// If `src.len()` is not a multiple of 4, trailing bytes are ignored.
#[inline]
pub fn read_f32_le(src: &[u8]) -> Vec<f32> {
    let usable = src.len() - (src.len() % 4);
    let count = usable / 4;
    #[cfg(target_endian = "little")]
    {
        let mut result = vec![0.0f32; count];
        // SAFETY: On LE targets, f32 byte representation matches memory layout.
        // `usable` = count * 4, both buffers are valid for that length.
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), result.as_mut_ptr().cast::<u8>(), usable);
        }
        result
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * 4;
            let bytes: [u8; 4] = src[start..start + 4].try_into().unwrap();
            result.push(f32::from_le_bytes(bytes));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dimensions that exercise all tail-loop edge cases:
    /// 1 (pure tail), 7 (tail=7), 8 (exact chunk), 15, 16, 31, 32, 128, 384, 768
    const DIMS: &[usize] = &[1, 7, 8, 15, 16, 31, 32, 128, 384, 768];

    fn make_vecs(dim: usize) -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.2 + 1.0).collect();
        (a, b)
    }

    fn assert_close(actual: f32, expected: f32, tol: f32, label: &str, dim: usize) {
        let diff = (actual - expected).abs();
        let scale = expected.abs().max(1.0);
        assert!(
            diff < tol * scale,
            "{label} dim={dim}: expected={expected}, actual={actual}, diff={diff}"
        );
    }

    #[test]
    fn dot_product_matches_scalar() {
        for &dim in DIMS {
            let (a, b) = make_vecs(dim);
            let scalar = dot_product_scalar(&a, &b);
            let result = dot_product(&a, &b);
            assert_close(result, scalar, 1e-5, "dot_product", dim);
        }
    }

    #[test]
    fn euclidean_distance_sq_matches_scalar() {
        for &dim in DIMS {
            let (a, b) = make_vecs(dim);
            let scalar = euclidean_distance_sq_scalar(&a, &b);
            let result = euclidean_distance_sq(&a, &b);
            assert_close(result, scalar, 1e-5, "euclidean_distance_sq", dim);
        }
    }

    #[test]
    fn cosine_similarity_matches_scalar() {
        for &dim in DIMS {
            let (a, b) = make_vecs(dim);
            let scalar = cosine_similarity_scalar(&a, &b);
            let result = cosine_similarity(&a, &b);
            assert_close(result, scalar, 1e-5, "cosine_similarity", dim);
        }
    }

    #[test]
    fn manhattan_distance_matches_scalar() {
        for &dim in DIMS {
            let (a, b) = make_vecs(dim);
            let scalar = manhattan_distance_scalar(&a, &b);
            let result = manhattan_distance(&a, &b);
            assert_close(result, scalar, 1e-5, "manhattan_distance", dim);
        }
    }

    #[test]
    fn hamming_distance_matches_scalar() {
        for dim in [1usize, 7, 8, 15, 16, 31, 32, 64, 128, 256] {
            let a: Vec<u8> = (0..dim).map(|i| (i * 37 + 13) as u8).collect();
            let b: Vec<u8> = (0..dim).map(|i| (i * 53 + 7) as u8).collect();
            let scalar = hamming_distance_scalar(&a, &b);
            let result = hamming_distance(&a, &b);
            assert_eq!(
                result, scalar,
                "hamming_distance dim={dim}: scalar={scalar}, simd={result}"
            );
        }
    }

    #[test]
    fn dot_product_zero_vectors() {
        let a = vec![0.0f32; 128];
        let b = vec![0.0f32; 128];
        assert_eq!(dot_product(&a, &b), 0.0);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0f32; 32];
        let b = vec![1.0f32; 32];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_similarity_identical() {
        let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let result = cosine_similarity(&a, &a);
        assert!(
            (result - 1.0).abs() < 1e-6,
            "identical vectors: sim={result}"
        );
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let result = cosine_similarity(&a, &b);
        assert!(
            (result - (-1.0)).abs() < 1e-6,
            "opposite vectors: sim={result}"
        );
    }

    #[test]
    fn hamming_distance_known_pattern() {
        // 0xFF ^ 0x00 = 0xFF -> 8 bits per byte
        let a = vec![0xFF_u8; 32];
        let b = vec![0x00_u8; 32];
        assert_eq!(hamming_distance(&a, &b), 32 * 8);
    }

    #[test]
    fn hamming_distance_identical() {
        let a = vec![0xAB_u8; 64];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn euclidean_distance_sq_identical() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        assert_eq!(euclidean_distance_sq(&a, &a), 0.0);
    }

    #[test]
    fn manhattan_distance_identical() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        assert_eq!(manhattan_distance(&a, &a), 0.0);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn dot_product_dimension_mismatch_panics() {
        let a = vec![1.0f32; 10];
        let b = vec![1.0f32; 11];
        dot_product(&a, &b);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn euclidean_dimension_mismatch_panics() {
        let a = vec![1.0f32; 10];
        let b = vec![1.0f32; 11];
        euclidean_distance_sq(&a, &b);
    }

    #[test]
    fn distance_metric_nan_returns_max() {
        let a = [1.0f32, f32::NAN, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let d = DistanceMetric::EuclideanSq.compute(&a, &b);
        assert_eq!(d, f32::MAX);
    }

    #[test]
    fn distance_metric_mismatch_returns_max() {
        let a = [1.0f32, 2.0];
        let b = [1.0f32, 2.0, 3.0];
        let d = DistanceMetric::Cosine.compute(&a, &b);
        assert_eq!(d, f32::MAX);
    }

    #[test]
    fn write_read_f32_le_roundtrip() {
        let values: Vec<f32> = (0..100).map(|i| (i as f32) * 0.123 - 6.0).collect();
        let mut buf = vec![0u8; values.len() * 4];
        write_f32_le(&mut buf, &values);
        let decoded = read_f32_le(&buf);
        assert_eq!(decoded, values);
    }
}
