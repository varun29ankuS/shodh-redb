use core::sync::atomic::{AtomicU64, Ordering};

/// Lock-free xorshift64 PRNG for sampling read verification.
///
/// Uses `AtomicU64` with relaxed ordering -- we only need statistical
/// uniformity, not cryptographic security or strict ordering.
pub(crate) struct SamplingRng {
    state: AtomicU64,
}

impl SamplingRng {
    /// Create a new PRNG with the given seed. Seed must be non-zero.
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: AtomicU64::new(if seed == 0 {
                0x517c_c1b7_2722_0a95
            } else {
                seed
            }),
        }
    }

    /// Advance the PRNG and return true with approximately `rate` probability.
    /// `rate` must be in `[0.0, 1.0]`.
    pub(crate) fn should_verify(&self, rate: f32) -> bool {
        if rate <= 0.0 {
            return false;
        }
        if rate >= 1.0 {
            return true;
        }
        // xorshift64: load, compute, store. Relaxed is fine -- occasional
        // repeated values just slightly skew the distribution.
        let mut s = self.state.load(Ordering::Relaxed);
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state.store(s, Ordering::Relaxed);
        // Map to [0, 1) and compare against rate.
        // Using >> 11 gives 53 bits of mantissa precision, matching f64.
        #[allow(clippy::cast_precision_loss)]
        let uniform = (s >> 11) as f64 / ((1u64 << 53) as f64);
        uniform < f64::from(rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_never_fires_at_zero_rate() {
        let rng = SamplingRng::new(42);
        for _ in 0..10_000 {
            assert!(!rng.should_verify(0.0));
        }
    }

    #[test]
    fn rng_always_fires_at_full_rate() {
        let rng = SamplingRng::new(42);
        for _ in 0..10_000 {
            assert!(rng.should_verify(1.0));
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn rng_roughly_correct_rate() {
        let rng = SamplingRng::new(123_456_789);
        let n: u32 = 100_000;
        let rate = 0.10_f32;
        let mut hits: u32 = 0;
        for _ in 0..n {
            if rng.should_verify(rate) {
                hits += 1;
            }
        }
        let actual = f64::from(hits) / f64::from(n);
        // Allow +/-2% tolerance
        assert!(
            (actual - f64::from(rate)).abs() < 0.02,
            "expected ~{rate}, got {actual}"
        );
    }

    #[test]
    fn zero_seed_uses_default() {
        let rng = SamplingRng::new(0);
        // Just ensure it doesn't panic or get stuck
        let _ = rng.should_verify(0.5);
        let _ = rng.should_verify(0.5);
    }
}
