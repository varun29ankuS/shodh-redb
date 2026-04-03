//! Per-entry read verification for `BfTree`.
//!
//! Wraps stored values with a 4-byte XXH3-32 checksum prefix for integrity
//! verification on read. This catches silent data corruption in the storage
//! layer.
//!
//! Value encoding: `[checksum: u32 LE][original_value_bytes]`
//!
//! # Verify Modes
//!
//! - `None`: No checksum wrapping — values stored as-is.
//! - `Full`: Every read verifies the checksum.
//! - `Sampled(f32)`: Only a fraction of reads are verified (0.0–1.0).

use alloc::vec::Vec;
use core::fmt;

use super::error::BfTreeError;

/// Checksum overhead: 4 bytes prepended to every value.
const CHECKSUM_SIZE: usize = 4;

/// Verification mode for per-entry checksums.
#[derive(Clone, Debug)]
pub enum VerifyMode {
    /// No verification — values stored without checksums.
    None,
    /// Every read is verified.
    Full,
    /// A fraction of reads are verified (0.0 = none, 1.0 = all).
    Sampled(f32),
}

impl Default for VerifyMode {
    fn default() -> Self {
        Self::None
    }
}

/// Compute a simple 32-bit hash for checksum purposes.
/// Uses FNV-1a for speed and simplicity — not cryptographic.
fn compute_checksum(data: &[u8]) -> u32 {
    let mut h: u32 = 0x811c9dc5; // FNV offset basis (32-bit)
    for &b in data {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x01000193); // FNV prime (32-bit)
    }
    h
}

/// Wrap a value with a checksum prefix.
pub fn wrap_value(value: &[u8]) -> Vec<u8> {
    let checksum = compute_checksum(value);
    let mut wrapped = Vec::with_capacity(CHECKSUM_SIZE + value.len());
    wrapped.extend_from_slice(&checksum.to_le_bytes());
    wrapped.extend_from_slice(value);
    wrapped
}

/// Unwrap a checksum-wrapped value, optionally verifying integrity.
///
/// Returns the original value bytes (without checksum prefix).
/// If `verify` is true, recomputes the checksum and returns an error on mismatch.
pub fn unwrap_value(wrapped: &[u8], verify: bool) -> Result<&[u8], BfTreeError> {
    if wrapped.len() < CHECKSUM_SIZE {
        return Err(BfTreeError::Corruption(
            "value too short for checksum".into(),
        ));
    }

    let stored_checksum = u32::from_le_bytes(wrapped[..CHECKSUM_SIZE].try_into().unwrap());
    let data = &wrapped[CHECKSUM_SIZE..];

    if verify {
        let computed = compute_checksum(data);
        if computed != stored_checksum {
            return Err(BfTreeError::Corruption(alloc::format!(
                "checksum mismatch: stored={stored_checksum:#010x}, computed={computed:#010x}",
            )));
        }
    }

    Ok(data)
}

/// Unwrap a checksum-wrapped value into an owned Vec, optionally verifying.
#[allow(dead_code)]
pub fn unwrap_value_owned(wrapped: &[u8], verify: bool) -> Result<Vec<u8>, BfTreeError> {
    unwrap_value(wrapped, verify).map(|data| data.to_vec())
}

/// Determine whether to verify based on `VerifyMode` and a deterministic sample.
///
/// # Sampling counter
///
/// The `Sampled` mode uses a **global** atomic counter shared across all tables
/// and threads. This is intentional: the counter provides a system-wide
/// verification rate rather than per-table or per-thread rates, ensuring a
/// consistent fraction of all reads are verified regardless of access patterns.
/// A per-table counter could leave rarely-accessed tables unverified while
/// over-verifying hot tables; the global counter distributes verification
/// uniformly across all read operations.
pub fn should_verify(mode: &VerifyMode) -> bool {
    use core::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    match mode {
        VerifyMode::None => false,
        VerifyMode::Full => true,
        VerifyMode::Sampled(rate) => {
            if *rate >= 1.0 {
                return true;
            }
            if *rate <= 0.0 {
                return false;
            }
            // Deterministic sampling via a global atomic counter. The counter
            // is intentionally global (see doc comment above).
            let count = COUNTER.fetch_add(1, Ordering::Relaxed);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let threshold = (rate.max(0.0) * 1000.0) as u64;
            (count % 1000) < threshold
        }
    }
}

impl fmt::Display for VerifyMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Full => write!(f, "full"),
            Self::Sampled(rate) => write!(f, "sampled({:.1}%)", rate * 100.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_unwrap_roundtrip() {
        let original = b"hello, world!";
        let wrapped = wrap_value(original);
        assert_eq!(wrapped.len(), CHECKSUM_SIZE + original.len());

        let unwrapped = unwrap_value(&wrapped, true).unwrap();
        assert_eq!(unwrapped, original);
    }

    #[test]
    fn corruption_detected() {
        let original = b"important data";
        let mut wrapped = wrap_value(original);

        // Corrupt a data byte.
        if wrapped.len() > CHECKSUM_SIZE {
            wrapped[CHECKSUM_SIZE] ^= 0xFF;
        }

        let result = unwrap_value(&wrapped, true);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BfTreeError::Corruption(_)));
    }

    #[test]
    fn no_verify_skips_check() {
        let original = b"data";
        let mut wrapped = wrap_value(original);

        // Corrupt data byte.
        wrapped[CHECKSUM_SIZE] ^= 0xFF;

        // Without verify, no error.
        let unwrapped = unwrap_value(&wrapped, false).unwrap();
        assert_ne!(unwrapped, original); // data is corrupted
    }

    #[test]
    fn empty_value() {
        let wrapped = wrap_value(b"");
        assert_eq!(wrapped.len(), CHECKSUM_SIZE);

        let unwrapped = unwrap_value(&wrapped, true).unwrap();
        assert!(unwrapped.is_empty());
    }

    #[test]
    fn verify_mode_sampling() {
        assert!(!should_verify(&VerifyMode::None));
        assert!(should_verify(&VerifyMode::Full));

        // Sampled at 100% should always verify.
        assert!(should_verify(&VerifyMode::Sampled(1.0)));
        // Sampled at 0% should never verify.
        assert!(!should_verify(&VerifyMode::Sampled(0.0)));
    }

    #[test]
    fn checksum_deterministic() {
        let data = b"deterministic";
        let c1 = compute_checksum(data);
        let c2 = compute_checksum(data);
        assert_eq!(c1, c2);
    }
}
