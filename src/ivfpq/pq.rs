use alloc::vec::Vec;

use crate::vector_ops::{DistanceMetric, euclidean_distance_sq};

use super::kmeans;

// ---------------------------------------------------------------------------
// Codebooks -- PQ codebook storage and encode/decode
// ---------------------------------------------------------------------------

/// Product Quantization codebooks.
///
/// Contains `num_subvectors` codebooks, each with 256 centroids of dimension
/// `sub_dim`. Stored as a single flat `Vec<f32>`:
///
/// ```text
/// codebooks[m][k] starts at offset (m * 256 + k) * sub_dim
/// ```
///
/// Total size: `num_subvectors * 256 * sub_dim` floats.
#[derive(Clone)]
pub struct Codebooks {
    /// Flat storage of all codebook centroids.
    pub data: Vec<f32>,
    /// Number of sub-quantizers (= number of subvector positions).
    pub num_subvectors: usize,
    /// Dimension of each sub-vector (= dim / `num_subvectors`).
    pub sub_dim: usize,
}

impl Codebooks {
    /// Returns the centroid for sub-quantizer `m`, codeword `k`.
    ///
    /// Returns an empty slice if indices are out of bounds (corrupted data).
    #[inline]
    pub fn centroid(&self, m: usize, k: usize) -> &[f32] {
        let Some(start) = (m.checked_mul(256))
            .and_then(|v| v.checked_add(k))
            .and_then(|v| v.checked_mul(self.sub_dim))
        else {
            return &[];
        };
        let end = match start.checked_add(self.sub_dim) {
            Some(e) if e <= self.data.len() => e,
            _ => return &[],
        };
        &self.data[start..end]
    }

    /// Encode a full vector into PQ codes.
    ///
    /// The vector must have length `num_subvectors * sub_dim`. Returns a
    /// `Vec<u8>` of length `num_subvectors`, where each byte is the index
    /// of the nearest codebook centroid for that sub-vector position.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let required_len = self.num_subvectors.saturating_mul(self.sub_dim);
        if vector.len() < required_len {
            return Vec::new();
        }
        let mut codes = Vec::with_capacity(self.num_subvectors);
        for m in 0..self.num_subvectors {
            let sub = &vector[m * self.sub_dim..(m + 1) * self.sub_dim];
            let mut best_k = 0u8;
            let mut best_dist = f32::INFINITY;
            for k in 0..256usize {
                let centroid = self.centroid(m, k);
                let d = euclidean_distance_sq(sub, centroid);
                if d < best_dist {
                    best_dist = d;
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        best_k = k as u8;
                    }
                }
            }
            codes.push(best_k);
        }
        codes
    }

    /// Decode PQ codes back to an approximate vector.
    ///
    /// Reconstructs the vector by concatenating the codebook centroids
    /// indicated by each code byte.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.num_subvectors * self.sub_dim);
        for (m, &code) in codes.iter().enumerate() {
            vector.extend_from_slice(self.centroid(m, code as usize));
        }
        vector
    }

    /// Total number of f32 elements in the codebook data.
    pub fn data_len(&self) -> usize {
        self.num_subvectors * 256 * self.sub_dim
    }

    /// Serialize all codebooks to bytes (f32 little-endian).
    ///
    /// Each codebook `m` is serialized as `256 * sub_dim * 4` bytes.
    pub fn serialize_codebook(&self, m: usize) -> Vec<u8> {
        let start = m * 256 * self.sub_dim;
        let end = start + 256 * self.sub_dim;
        let floats = &self.data[start..end];
        let mut bytes = Vec::with_capacity(floats.len() * 4);
        for &f in floats {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        bytes
    }

    /// Deserialize a single codebook from bytes.
    ///
    /// If the byte length does not match the expected `256 * sub_dim * 4`,
    /// returns as many floats as possible (truncated or padded with zeros).
    pub fn deserialize_codebook(bytes: &[u8], sub_dim: usize) -> Vec<f32> {
        let expected = 256 * sub_dim;
        let num_floats = bytes.len() / 4;
        let mut floats = Vec::with_capacity(num_floats.max(expected));
        for chunk in bytes.chunks_exact(4) {
            if let Ok(b) = chunk.try_into() {
                floats.push(f32::from_le_bytes(b));
            }
        }
        // Pad to expected size if data was truncated
        floats.resize(expected, 0.0);
        floats
    }
}

impl core::fmt::Debug for Codebooks {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Codebooks")
            .field("num_subvectors", &self.num_subvectors)
            .field("sub_dim", &self.sub_dim)
            .field("data_len", &self.data.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Sub-vector k-means training (parallel under std, sequential under no_std)
// ---------------------------------------------------------------------------

/// Train one sub-quantizer: extract sub-vectors for position `m`, run k-means,
/// return a flat `Vec<f32>` of exactly `256 * sub_dim` floats (padded if k < 256).
fn train_one_subvector(
    flat_vectors: &[f32],
    dim: usize,
    n: usize,
    m: usize,
    sub_dim: usize,
    k: usize,
    max_iter: usize,
) -> Vec<f32> {
    let mut sub_flat = Vec::with_capacity(n * sub_dim);
    for i in 0..n {
        let start = i * dim + m * sub_dim;
        sub_flat.extend_from_slice(&flat_vectors[start..start + sub_dim]);
    }
    let centroids = kmeans::kmeans(&sub_flat, sub_dim, k, max_iter, DistanceMetric::EuclideanSq);
    let mut result = centroids;
    if k < 256 {
        result.resize(256 * sub_dim, 0.0);
    }
    result
}

/// Parallel sub-vector training using `std::thread::scope`.
#[cfg(feature = "std")]
fn train_subvectors(
    flat_vectors: &[f32],
    dim: usize,
    n: usize,
    num_subvectors: usize,
    sub_dim: usize,
    k: usize,
    max_iter: usize,
) -> crate::Result<Vec<f32>> {
    let mut results: Vec<Vec<f32>> = Vec::with_capacity(num_subvectors);

    std::thread::scope(|s| {
        let handles: Vec<_> = (0..num_subvectors)
            .map(|m| {
                s.spawn(move || train_one_subvector(flat_vectors, dim, n, m, sub_dim, k, max_iter))
            })
            .collect();
        for handle in handles {
            results.push(handle.join().map_err(|_| {
                crate::StorageError::Internal(
                    "IVF-PQ subvector training thread panicked".into(),
                )
            })?);
        }
        Ok::<(), crate::StorageError>(())
    })
    .map_err(|_| {
        crate::StorageError::Internal("IVF-PQ training scope panicked".into())
    })?;

    let mut all_data = Vec::with_capacity(num_subvectors * 256 * sub_dim);
    for chunk in results {
        all_data.extend_from_slice(&chunk);
    }
    Ok(all_data)
}

/// Sequential sub-vector training for no_std environments.
#[cfg(not(feature = "std"))]
fn train_subvectors(
    flat_vectors: &[f32],
    dim: usize,
    n: usize,
    num_subvectors: usize,
    sub_dim: usize,
    k: usize,
    max_iter: usize,
) -> crate::Result<Vec<f32>> {
    let mut all_data = Vec::with_capacity(num_subvectors * 256 * sub_dim);
    for m in 0..num_subvectors {
        let chunk = train_one_subvector(flat_vectors, dim, n, m, sub_dim, k, max_iter);
        all_data.extend_from_slice(&chunk);
    }
    Ok(all_data)
}

// ---------------------------------------------------------------------------
// PQ training
// ---------------------------------------------------------------------------

/// Train PQ codebooks from a set of training vectors.
///
/// `flat_vectors` contains `n` vectors of dimension `dim`, stored contiguously.
/// `num_subvectors` sub-quantizers are trained, each with 256 codewords.
///
/// For each sub-vector position, k-means with k=256 is run on the corresponding
/// sub-vector slices from all training vectors.
pub fn train_codebooks(
    flat_vectors: &[f32],
    dim: usize,
    num_subvectors: usize,
    max_iter: usize,
    _metric: DistanceMetric,
) -> Result<Codebooks, crate::StorageError> {
    if dim == 0 || num_subvectors == 0 || dim % num_subvectors != 0 {
        return Err(crate::StorageError::invalid_index_config(
            "PQ training: dim must be non-zero and divisible by num_subvectors",
        ));
    }
    let n = flat_vectors.len() / dim;
    let sub_dim = dim / num_subvectors;

    let k = 256usize.min(n); // Can't have more codewords than training vectors.

    let all_data = train_subvectors(flat_vectors, dim, n, num_subvectors, sub_dim, k, max_iter)?;

    Ok(Codebooks {
        data: all_data,
        num_subvectors,
        sub_dim,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        // 8-dim vectors, 2 sub-vectors of 4 dims each, 4 training vectors.
        #[rustfmt::skip]
        let training: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0,  0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,  1.0, 0.0, 0.0, 0.0,
        ];

        let codebooks = train_codebooks(&training, 8, 2, 25, DistanceMetric::EuclideanSq).unwrap();
        assert_eq!(codebooks.num_subvectors, 2);
        assert_eq!(codebooks.sub_dim, 4);

        // Encode one of the training vectors -- should reconstruct closely.
        let original = &training[0..8];
        let codes = codebooks.encode(original);
        assert_eq!(codes.len(), 2);

        let reconstructed = codebooks.decode(&codes);
        assert_eq!(reconstructed.len(), 8);

        // The reconstruction should be close to the original.
        let error: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            error < 1.0,
            "reconstruction error too high: {error}, original={original:?}, reconstructed={reconstructed:?}"
        );
    }

    #[test]
    fn codebook_serialize_roundtrip() {
        let codebooks = Codebooks {
            data: [1.0, 2.0, 3.0, 4.0].repeat(256), // 1 subvec, sub_dim=4, 256 codewords
            num_subvectors: 1,
            sub_dim: 4,
        };
        let bytes = codebooks.serialize_codebook(0);
        let floats = Codebooks::deserialize_codebook(&bytes, 4);
        assert_eq!(floats.len(), 256 * 4);
        assert!((floats[0] - 1.0).abs() < 1e-6);
    }
}
