#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use redb::ivfpq::pq::{train_codebooks, Codebooks};
use redb::DistanceMetric;

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    /// Deserialize arbitrary bytes as a codebook — must not panic.
    DeserializeRaw { data: Vec<u8>, sub_dim_sel: u8 },
    /// Construct Codebooks, encode a vector, decode codes — roundtrip.
    EncodeDecodeRoundtrip {
        /// Number of sub-vectors: 1..4.
        num_sub_sel: u8,
        /// Sub-dimension: 2/4/8.
        sub_dim_sel: u8,
        /// Raw f32 bits for codebook centroids.
        centroid_data: Vec<u8>,
        /// Raw f32 bits for test vector.
        vector_data: Vec<u8>,
    },
    /// serialize_codebook → deserialize_codebook roundtrip.
    SerializeRoundtrip {
        num_sub_sel: u8,
        sub_dim_sel: u8,
        centroid_data: Vec<u8>,
    },
    /// Train codebooks on small data (≤16 vectors, small dims).
    Train {
        /// Number of vectors: clamped to 2..16.
        num_vectors: u8,
        /// Dimension selector: 0..3 → 2/4/6/8.
        dim_sel: u8,
        /// Number of sub-vectors selector: 1 or 2.
        num_sub_sel: u8,
        /// Raw f32 bits for training vectors.
        vector_data: Vec<u8>,
    },
}

fn select_sub_dim(sel: u8) -> usize {
    match sel % 3 {
        0 => 2,
        1 => 4,
        _ => 8,
    }
}

fn select_dim(sel: u8) -> usize {
    match sel % 4 {
        0 => 2,
        1 => 4,
        2 => 6,
        _ => 8,
    }
}

/// Build a Codebooks struct from arbitrary byte data.
fn build_codebooks(num_subvectors: usize, sub_dim: usize, centroid_data: &[u8]) -> Codebooks {
    let total_floats = num_subvectors * 256 * sub_dim;
    let mut data = Vec::with_capacity(total_floats);
    for i in 0..total_floats {
        let byte_offset = (i * 4) % centroid_data.len().max(1);
        let bytes: [u8; 4] = [
            centroid_data.get(byte_offset).copied().unwrap_or(0),
            centroid_data.get(byte_offset + 1).copied().unwrap_or(0),
            centroid_data.get(byte_offset + 2).copied().unwrap_or(0),
            centroid_data.get(byte_offset + 3).copied().unwrap_or(0),
        ];
        let val = f32::from_le_bytes(bytes);
        // Replace NaN/Inf with 0.0 to keep centroids usable.
        if val.is_finite() {
            data.push(val);
        } else {
            data.push(0.0);
        }
    }
    Codebooks {
        data,
        num_subvectors,
        sub_dim,
    }
}

/// Build a vector of f32 from arbitrary byte data.
fn build_vector(len: usize, raw: &[u8]) -> Vec<f32> {
    let mut vec = Vec::with_capacity(len);
    for i in 0..len {
        let byte_offset = (i * 4) % raw.len().max(1);
        let bytes: [u8; 4] = [
            raw.get(byte_offset).copied().unwrap_or(0),
            raw.get(byte_offset + 1).copied().unwrap_or(0),
            raw.get(byte_offset + 2).copied().unwrap_or(0),
            raw.get(byte_offset + 3).copied().unwrap_or(0),
        ];
        let val = f32::from_le_bytes(bytes);
        if val.is_finite() {
            vec.push(val);
        } else {
            vec.push(0.0);
        }
    }
    vec
}

fuzz_target!(|op: FuzzOp| {
    match op {
        FuzzOp::DeserializeRaw {
            data,
            sub_dim_sel,
        } => {
            let sub_dim = select_sub_dim(sub_dim_sel).max(1);
            // Must not panic on any input.
            let result = Codebooks::deserialize_codebook(&data, sub_dim);
            // Result should always be 256 * sub_dim floats.
            assert_eq!(result.len(), 256 * sub_dim);
        }

        FuzzOp::EncodeDecodeRoundtrip {
            num_sub_sel,
            sub_dim_sel,
            centroid_data,
            vector_data,
        } => {
            let num_subvectors = ((num_sub_sel % 4) as usize).max(1);
            let sub_dim = select_sub_dim(sub_dim_sel);
            let dim = num_subvectors * sub_dim;

            if centroid_data.is_empty() {
                return;
            }

            let codebooks = build_codebooks(num_subvectors, sub_dim, &centroid_data);
            assert_eq!(codebooks.data.len(), codebooks.data_len());

            // Build a test vector of the right dimension.
            let vector = if vector_data.is_empty() {
                vec![0.0f32; dim]
            } else {
                build_vector(dim, &vector_data)
            };

            // Encode must produce num_subvectors codes.
            let codes = codebooks.encode(&vector);
            assert_eq!(codes.len(), num_subvectors);

            // Decode must produce a vector of the right dimension.
            let decoded = codebooks.decode(&codes);
            assert_eq!(decoded.len(), dim);

            // Each decoded value must be finite.
            for val in &decoded {
                assert!(val.is_finite());
            }
        }

        FuzzOp::SerializeRoundtrip {
            num_sub_sel,
            sub_dim_sel,
            centroid_data,
        } => {
            let num_subvectors = ((num_sub_sel % 4) as usize).max(1);
            let sub_dim = select_sub_dim(sub_dim_sel);

            if centroid_data.is_empty() {
                return;
            }

            let codebooks = build_codebooks(num_subvectors, sub_dim, &centroid_data);

            for m in 0..num_subvectors {
                let serialized = codebooks.serialize_codebook(m);
                // Should be 256 * sub_dim * 4 bytes (each f32 = 4 bytes).
                assert_eq!(serialized.len(), 256 * sub_dim * 4);

                let deserialized = Codebooks::deserialize_codebook(&serialized, sub_dim);
                assert_eq!(deserialized.len(), 256 * sub_dim);

                // Verify roundtrip: deserialized centroids must match originals.
                for k in 0..256 {
                    let original = codebooks.centroid(m, k);
                    let restored = &deserialized[k * sub_dim..(k + 1) * sub_dim];
                    assert_eq!(original.len(), restored.len());
                    for (a, b) in original.iter().zip(restored.iter()) {
                        assert!((a - b).abs() < f32::EPSILON);
                    }
                }
            }
        }

        FuzzOp::Train {
            num_vectors,
            dim_sel,
            num_sub_sel,
            vector_data,
        } => {
            let dim = select_dim(dim_sel);
            let num_subvectors = if num_sub_sel % 2 == 0 { 1 } else { 2 };

            // dim must be divisible by num_subvectors.
            if dim % num_subvectors != 0 {
                return;
            }

            let n = (num_vectors as usize).clamp(2, 16);
            let total_floats = n * dim;

            if vector_data.is_empty() {
                return;
            }

            let flat_vectors = build_vector(total_floats, &vector_data);
            let result = train_codebooks(
                &flat_vectors,
                dim,
                num_subvectors,
                5, // Small max_iter for fuzzing speed.
                DistanceMetric::EuclideanSq,
            );

            if let Ok(codebooks) = result {
                assert_eq!(codebooks.num_subvectors, num_subvectors);
                assert_eq!(codebooks.sub_dim, dim / num_subvectors);
                assert_eq!(codebooks.data.len(), codebooks.data_len());

                // Encode one of the training vectors and verify output shape.
                let test_vec = &flat_vectors[..dim];
                let codes = codebooks.encode(test_vec);
                assert_eq!(codes.len(), num_subvectors);

                let decoded = codebooks.decode(&codes);
                assert_eq!(decoded.len(), dim);
            }
        }
    }
});
