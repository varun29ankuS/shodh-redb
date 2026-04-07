//! AVX2 SIMD implementations of distance functions for `x86_64`.
//!
//! Each function is gated with `#[target_feature(enable = "avx2")]` and is
//! `unsafe` -- callers must verify AVX2 support via `is_x86_feature_detected!`
//! before calling.
//!
//! All functions process 8 f32 elements per iteration (256-bit lanes) with a
//! scalar tail loop for remainders.

use core::arch::x86_64::{
    __m256, __m256i, _mm_add_ss, _mm_cvtss_f32, _mm256_add_epi8, _mm256_add_epi64, _mm256_add_ps,
    _mm256_and_si256, _mm256_andnot_ps, _mm256_castps256_ps128, _mm256_castsi256_ps,
    _mm256_extractf128_ps, _mm256_hadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_mul_ps,
    _mm256_sad_epu8, _mm256_set1_epi8, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setzero_ps,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_sub_ps, _mm256_xor_si256,
};

/// Horizontal sum of an `__m256` register (8 f32 lanes to single f32).
///
/// Uses two rounds of `hadd` to reduce within 128-bit lanes, then adds
/// the low and high 128-bit halves.
#[inline]
unsafe fn hsum_avx(v: __m256) -> f32 {
    // SAFETY: All intrinsics operate on the provided register. Caller guarantees
    // AVX2 is available via #[target_feature] on the enclosing function.
    unsafe {
        let sum1 = _mm256_hadd_ps(v, v);
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        let hi = _mm256_extractf128_ps(sum2, 1);
        let lo = _mm256_castps256_ps128(sum2);
        let total = _mm_add_ss(lo, hi);
        _mm_cvtss_f32(total)
    }
}

/// Dot product of two f32 slices using AVX2.
///
/// Processes 8 elements per iteration. Scalar tail for remainder.
///
/// # Safety
///
/// Caller must ensure AVX2 is supported on the current CPU.
/// Slices must have equal length.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut acc = _mm256_setzero_ps();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            // SAFETY: offset + 8 <= chunks * 8 <= len, so reads are in bounds.
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
        }

        let mut sum = hsum_avx(acc);

        let tail_start = chunks * 8;
        for i in 0..remainder {
            sum += a[tail_start + i] * b[tail_start + i];
        }

        sum
    }
}

/// Squared Euclidean distance using AVX2.
///
/// # Safety
///
/// Caller must ensure AVX2 is supported. Slices must have equal length.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn euclidean_distance_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut acc = _mm256_setzero_ps();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            // SAFETY: offset + 8 <= len, reads are in bounds.
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            let diff = _mm256_sub_ps(va, vb);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
        }

        let mut sum = hsum_avx(acc);

        let tail_start = chunks * 8;
        for i in 0..remainder {
            let d = a[tail_start + i] - b[tail_start + i];
            sum += d * d;
        }

        sum
    }
}

/// Cosine similarity using AVX2.
///
/// Maintains three accumulators (dot, `norm_a`, `norm_b`) simultaneously.
///
/// # Safety
///
/// Caller must ensure AVX2 is supported. Slices must have equal length.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut dot_acc = _mm256_setzero_ps();
        let mut norm_a_acc = _mm256_setzero_ps();
        let mut norm_b_acc = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            // SAFETY: offset + 8 <= len, reads are in bounds.
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            dot_acc = _mm256_add_ps(dot_acc, _mm256_mul_ps(va, vb));
            norm_a_acc = _mm256_add_ps(norm_a_acc, _mm256_mul_ps(va, va));
            norm_b_acc = _mm256_add_ps(norm_b_acc, _mm256_mul_ps(vb, vb));
        }

        let mut dot = hsum_avx(dot_acc);
        let mut norm_a = hsum_avx(norm_a_acc);
        let mut norm_b = hsum_avx(norm_b_acc);

        let tail_start = chunks * 8;
        for i in 0..remainder {
            let x = a[tail_start + i];
            let y = b[tail_start + i];
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            (dot / denom).clamp(-1.0, 1.0)
        }
    }
}

/// Manhattan (L1) distance using AVX2.
///
/// Uses bitwise AND-NOT to compute absolute value (clear sign bit).
///
/// # Safety
///
/// Caller must ensure AVX2 is supported. Slices must have equal length.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut acc = _mm256_setzero_ps();
        // Sign bit mask: 0x80000000 in each lane. ANDNOT with this clears sign bit = abs.
        let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            // SAFETY: offset + 8 <= len, reads are in bounds.
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            let diff = _mm256_sub_ps(va, vb);
            // abs(diff) = andnot(sign_mask, diff) -- clears the sign bit
            let abs_diff = _mm256_andnot_ps(sign_mask, diff);
            acc = _mm256_add_ps(acc, abs_diff);
        }

        let mut sum = hsum_avx(acc);

        let tail_start = chunks * 8;
        for i in 0..remainder {
            sum += (a[tail_start + i] - b[tail_start + i]).abs();
        }

        sum
    }
}

/// Hamming distance between byte slices using AVX2.
///
/// Uses Mula's SSSE3/AVX2 vectorized popcount method: split each byte into
/// nibbles, use a lookup table (`pshufb`) for the popcount of each nibble,
/// then accumulate with SAD against zero.
///
/// Processes 32 bytes per iteration (256-bit). This is the biggest SIMD win
/// -- roughly 16-32x throughput improvement over scalar byte-at-a-time.
///
/// # Safety
///
/// Caller must ensure AVX2 is supported. Slices must have equal length.
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_possible_truncation, clippy::cast_ptr_alignment)]
pub(crate) unsafe fn hamming_distance_avx2(a: &[u8], b: &[u8]) -> u32 {
    unsafe {
        let len = a.len();
        let chunks = len / 32;
        let remainder = len % 32;

        // Lookup table for popcount of a 4-bit nibble (0..15).
        // Repeated in both 128-bit lanes of the 256-bit register.
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, // low lane
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, // high lane
        );
        let low_nibble_mask = _mm256_set1_epi8(0x0F);

        let mut total_acc = _mm256_setzero_si256();

        for i in 0..chunks {
            // SAFETY: i < chunks, so i*32 + 32 <= len. Using loadu which handles unaligned.
            let a_ptr = a.as_ptr().add(i * 32).cast::<__m256i>();
            let b_ptr = b.as_ptr().add(i * 32).cast::<__m256i>();
            let va = _mm256_loadu_si256(a_ptr);
            let vb = _mm256_loadu_si256(b_ptr);

            // XOR to get differing bits
            let xor = _mm256_xor_si256(va, vb);

            // Split into low and high nibbles
            let lo = _mm256_and_si256(xor, low_nibble_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(xor, 4), low_nibble_mask);

            // Lookup popcount for each nibble
            let popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
            let popcnt_hi = _mm256_shuffle_epi8(lookup, hi);

            // Sum nibble popcounts per byte, then accumulate with SAD
            let byte_popcnt = _mm256_add_epi8(popcnt_lo, popcnt_hi);
            // SAD sums groups of 8 bytes against zero, producing u64 partial sums
            let sad = _mm256_sad_epu8(byte_popcnt, _mm256_setzero_si256());
            total_acc = _mm256_add_epi64(total_acc, sad);
        }

        // Horizontal sum of the 4 x u64 lanes in total_acc.
        // SAFETY: __m256i and [u64; 4] have identical size (32 bytes) and the
        // register value is valid for any bit pattern as u64.
        let lanes: [u64; 4] = core::mem::transmute(total_acc);
        let mut total: u32 = 0;
        for &lane in &lanes {
            total += lane as u32;
        }

        // Scalar tail
        let tail_start = chunks * 32;
        for i in 0..remainder {
            total += (a[tail_start + i] ^ b[tail_start + i]).count_ones();
        }

        total
    }
}
