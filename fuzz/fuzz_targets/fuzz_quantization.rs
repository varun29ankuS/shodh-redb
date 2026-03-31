#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use redb::{
    dequantize_scalar, hamming_distance, quantize_binary, quantize_scalar, read_f32_le,
    sq_dot_product, sq_euclidean_distance_sq, write_f32_le, BinaryQuantized, DynVec, FixedVec,
    SQVec, ScalarQuantized, Value,
};

fn bits_to_f32_array<const N: usize>(bits: &[u32]) -> [f32; N] {
    let mut arr = [0.0f32; N];
    for (i, val) in arr.iter_mut().enumerate() {
        *val = bits.get(i).map_or(0.0, |&b| f32::from_bits(b));
    }
    arr
}

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    /// Scalar quantize/dequantize roundtrip (8-dim).
    ScalarRoundtrip8 { bits: [u32; 8] },
    /// Scalar quantize/dequantize roundtrip (4-dim).
    ScalarRoundtrip4 { bits: [u32; 4] },
    /// Binary quantize + hamming self-distance.
    BinaryQuantize { bits: Vec<u32> },
    /// SQ euclidean distance approximation.
    SqEuclidean {
        query_bits: [u32; 8],
        target_bits: [u32; 8],
    },
    /// SQ dot product approximation.
    SqDotProduct {
        query_bits: [u32; 8],
        target_bits: [u32; 8],
    },
    /// FixedVec from_bytes with arbitrary data.
    FixedVecFromBytes { data: Vec<u8> },
    /// DynVec from_bytes with arbitrary data.
    DynVecFromBytes { data: Vec<u8> },
    /// ScalarQuantized from_bytes with arbitrary data.
    ScalarQuantizedFromBytes { data: Vec<u8> },
    /// BinaryQuantized from_bytes with arbitrary data.
    BinaryQuantizedFromBytes { data: Vec<u8> },
    /// write_f32_le into undersized buffer.
    F32LeUndersize {
        bits: Vec<u32>,
        buf_len: u8,
    },
}

fuzz_target!(|op: FuzzOp| {
    match op {
        FuzzOp::ScalarRoundtrip8 { bits } => {
            let v: [f32; 8] = bits_to_f32_array(&bits);
            let sq = quantize_scalar(&v);
            let restored = dequantize_scalar(&sq);

            // For all-finite inputs with nonzero range, bounded error holds.
            if v.iter().all(|x| x.is_finite()) {
                let range = sq.max_val - sq.min_val;
                if range > 0.0 {
                    let max_error = range / 510.0;
                    for (i, (&orig, &rest)) in v.iter().zip(restored.iter()).enumerate() {
                        let err = (orig - rest).abs();
                        assert!(
                            err <= max_error + 1e-6,
                            "dim {i}: error {err} exceeds bound {max_error}"
                        );
                    }
                }
            }
        }

        FuzzOp::ScalarRoundtrip4 { bits } => {
            let v: [f32; 4] = bits_to_f32_array(&bits);
            let sq = quantize_scalar(&v);
            let _ = dequantize_scalar(&sq);
            // Must not panic.
        }

        FuzzOp::BinaryQuantize { bits } => {
            let v: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
            if v.is_empty() {
                return;
            }
            let bq = quantize_binary(&v);
            let expected_len = v.len().div_ceil(8);
            assert_eq!(bq.len(), expected_len);

            // Self hamming distance is zero.
            assert_eq!(hamming_distance(&bq, &bq), 0);

            // Count of set bits should match count of positive values (ignoring NaN).
            let positive_count: u32 = v.iter().filter(|&&x| x > 0.0).count() as u32;
            let bit_count: u32 = bq.iter().map(|&b| b.count_ones()).sum();
            assert_eq!(
                bit_count, positive_count,
                "bit count {bit_count} != positive count {positive_count}"
            );
        }

        FuzzOp::SqEuclidean {
            query_bits,
            target_bits,
        } => {
            let query: [f32; 8] = bits_to_f32_array(&query_bits);
            let target: [f32; 8] = bits_to_f32_array(&target_bits);
            let sq = quantize_scalar(&target);
            let dist = sq_euclidean_distance_sq(&query, &sq);
            // Must not panic. Distance can be NaN for NaN inputs.
            if query.iter().all(|x| x.is_finite()) && target.iter().all(|x| x.is_finite()) {
                assert!(
                    dist >= 0.0 || dist.is_nan(),
                    "sq euclidean distance must be >= 0: {dist}"
                );
            }
        }

        FuzzOp::SqDotProduct {
            query_bits,
            target_bits,
        } => {
            let query: [f32; 8] = bits_to_f32_array(&query_bits);
            let target: [f32; 8] = bits_to_f32_array(&target_bits);
            let sq = quantize_scalar(&target);
            // Must not panic.
            let _ = sq_dot_product(&query, &sq);
        }

        FuzzOp::FixedVecFromBytes { data } => {
            // Must not panic for any input.
            let _: [f32; 8] = FixedVec::<8>::from_bytes(&data);
            let _: [f32; 1] = FixedVec::<1>::from_bytes(&data);
            let _: [f32; 16] = FixedVec::<16>::from_bytes(&data);
        }

        FuzzOp::DynVecFromBytes { data } => {
            let v: Vec<f32> = DynVec::from_bytes(&data);
            // Dimension should be data.len() / 4 (truncated).
            assert_eq!(v.len(), data.len() / 4);
        }

        FuzzOp::ScalarQuantizedFromBytes { data } => {
            // Must not panic for any input.
            let _: SQVec<8> = ScalarQuantized::<8>::from_bytes(&data);
            let _: SQVec<1> = ScalarQuantized::<1>::from_bytes(&data);
            let _: SQVec<16> = ScalarQuantized::<16>::from_bytes(&data);
        }

        FuzzOp::BinaryQuantizedFromBytes { data } => {
            // Must not panic for any input.
            let _: [u8; 8] = BinaryQuantized::<8>::from_bytes(&data);
            let _: [u8; 1] = BinaryQuantized::<1>::from_bytes(&data);
            let _: [u8; 48] = BinaryQuantized::<48>::from_bytes(&data);
        }

        FuzzOp::F32LeUndersize { bits, buf_len } => {
            let values: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
            let buf_len = buf_len as usize;
            let mut buf = vec![0u8; buf_len];
            // Must not panic even if buffer is too small.
            write_f32_le(&mut buf, &values);
            let restored = read_f32_le(&buf);
            // Restored count should be min(buf_len/4, values.len()).
            let expected = (buf_len / 4).min(values.len());
            assert_eq!(restored.len(), buf_len / 4);
            for (orig, rest) in values.iter().zip(restored.iter()).take(expected) {
                assert_eq!(orig.to_bits(), rest.to_bits());
            }
        }
    }
});
