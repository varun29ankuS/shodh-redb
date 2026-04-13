use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{self, Debug};

/// Trait for atomic read-modify-write merge operations on raw byte values.
///
/// Merge operators work at the raw byte level, enabling atomic updates without
/// full transaction boilerplate. The caller is responsible for correct serialization
/// of operand bytes to match the value type stored in the table.
///
/// # Return value
///
/// - `Some(bytes)` -- the merged value to store
/// - `None` -- delete the key
pub trait MergeOperator: Send + Sync {
    /// Merge `operand` into the `existing` value (if present), returning the new value.
    ///
    /// `key` is provided for context (e.g., per-key merge logic).
    /// `existing` is `None` when the key does not yet exist in the table.
    /// Return `None` to delete the key.
    fn merge(&self, key: &[u8], existing: Option<&[u8]>, operand: &[u8]) -> Option<Vec<u8>>;
}

/// Adds two little-endian encoded **integer** values using wrapping arithmetic.
///
/// Supports 1, 2, 4, and 8-byte widths (u8/i8 through u64/i64).
/// If the key does not exist, the operand is used as the initial value.
///
/// If `existing` and `operand` have different byte widths, the existing value
/// is preserved unchanged (no panic).
///
/// **Note:** This operator performs integer wrapping addition on the raw bytes.
/// It is not suitable for floating-point values (f32/f64). For float addition,
/// implement a custom [`MergeOperator`] or use [`FloatAdd`].
#[derive(Clone, Copy)]
pub struct NumericAdd;

impl Debug for NumericAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NumericAdd")
    }
}

impl MergeOperator for NumericAdd {
    fn merge(&self, _key: &[u8], existing: Option<&[u8]>, operand: &[u8]) -> Option<Vec<u8>> {
        let Some(existing) = existing else {
            return Some(operand.to_vec());
        };
        if existing.len() != operand.len() {
            return Some(existing.to_vec());
        }
        let result = match operand.len() {
            1 => {
                let a = existing[0];
                let b = operand[0];
                vec![a.wrapping_add(b)]
            }
            2 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u16::from_le_bytes(a_bytes);
                let b = u16::from_le_bytes(b_bytes);
                a.wrapping_add(b).to_le_bytes().to_vec()
            }
            4 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u32::from_le_bytes(a_bytes);
                let b = u32::from_le_bytes(b_bytes);
                a.wrapping_add(b).to_le_bytes().to_vec()
            }
            8 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u64::from_le_bytes(a_bytes);
                let b = u64::from_le_bytes(b_bytes);
                a.wrapping_add(b).to_le_bytes().to_vec()
            }
            _ => return Some(existing.to_vec()),
        };
        Some(result)
    }
}

/// Adds two little-endian encoded **integer** values using saturating arithmetic.
///
/// Like [`NumericAdd`], supports 1, 2, 4, and 8-byte widths (u8/i8 through u64/i64).
/// If the key does not exist, the operand is used as the initial value.
///
/// Unlike `NumericAdd`, this operator clamps at the type's maximum value instead
/// of wrapping. For example, `u64::MAX + 1` yields `u64::MAX` (not 0).
///
/// If `existing` and `operand` have different byte widths, the existing value
/// is preserved unchanged (no panic).
#[derive(Clone, Copy)]
pub struct SaturatingAdd;

impl Debug for SaturatingAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SaturatingAdd")
    }
}

impl MergeOperator for SaturatingAdd {
    fn merge(&self, _key: &[u8], existing: Option<&[u8]>, operand: &[u8]) -> Option<Vec<u8>> {
        let Some(existing) = existing else {
            return Some(operand.to_vec());
        };
        if existing.len() != operand.len() {
            return Some(existing.to_vec());
        }
        let result = match operand.len() {
            1 => {
                let a = existing[0];
                let b = operand[0];
                vec![a.saturating_add(b)]
            }
            2 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u16::from_le_bytes(a_bytes);
                let b = u16::from_le_bytes(b_bytes);
                a.saturating_add(b).to_le_bytes().to_vec()
            }
            4 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u32::from_le_bytes(a_bytes);
                let b = u32::from_le_bytes(b_bytes);
                a.saturating_add(b).to_le_bytes().to_vec()
            }
            8 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u64::from_le_bytes(a_bytes);
                let b = u64::from_le_bytes(b_bytes);
                a.saturating_add(b).to_le_bytes().to_vec()
            }
            _ => return Some(existing.to_vec()),
        };
        Some(result)
    }
}

/// Adds two little-endian encoded floating-point values.
///
/// Supports 4-byte (f32) and 8-byte (f64) widths.
/// If the key does not exist, the operand is used as the initial value.
///
/// If `existing` and `operand` have different byte widths, or the width is
/// not 4 or 8, the existing value is preserved unchanged (no panic).
///
/// If either operand is NaN or infinite, the result follows standard IEEE 754
/// arithmetic rules.
#[derive(Clone, Copy)]
pub struct FloatAdd;

impl Debug for FloatAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FloatAdd")
    }
}

impl MergeOperator for FloatAdd {
    fn merge(&self, _key: &[u8], existing: Option<&[u8]>, operand: &[u8]) -> Option<Vec<u8>> {
        let Some(existing) = existing else {
            return Some(operand.to_vec());
        };
        if existing.len() != operand.len() {
            return Some(existing.to_vec());
        }
        let result = match operand.len() {
            4 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = f32::from_le_bytes(a_bytes);
                let b = f32::from_le_bytes(b_bytes);
                (a + b).to_le_bytes().to_vec()
            }
            8 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = f64::from_le_bytes(a_bytes);
                let b = f64::from_le_bytes(b_bytes);
                (a + b).to_le_bytes().to_vec()
            }
            _ => return Some(existing.to_vec()),
        };
        Some(result)
    }
}

/// Keeps the maximum of two little-endian encoded unsigned numeric values.
///
/// Supports 1, 2, 4, and 8-byte widths. Comparison is unsigned.
/// If the key does not exist, the operand is used as the initial value.
///
/// If `existing` and `operand` have different byte widths, the existing value
/// is preserved unchanged (no panic).
#[derive(Clone, Copy)]
pub struct NumericMax;

impl Debug for NumericMax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NumericMax")
    }
}

impl MergeOperator for NumericMax {
    fn merge(&self, _key: &[u8], existing: Option<&[u8]>, operand: &[u8]) -> Option<Vec<u8>> {
        let Some(existing) = existing else {
            return Some(operand.to_vec());
        };
        if existing.len() != operand.len() {
            return Some(existing.to_vec());
        }
        let use_operand = match operand.len() {
            1 => operand[0] > existing[0],
            2 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u16::from_le_bytes(a_bytes);
                let b = u16::from_le_bytes(b_bytes);
                b > a
            }
            4 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u32::from_le_bytes(a_bytes);
                let b = u32::from_le_bytes(b_bytes);
                b > a
            }
            8 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u64::from_le_bytes(a_bytes);
                let b = u64::from_le_bytes(b_bytes);
                b > a
            }
            _ => return Some(existing.to_vec()),
        };
        if use_operand {
            Some(operand.to_vec())
        } else {
            Some(existing.to_vec())
        }
    }
}

/// Keeps the minimum of two little-endian encoded unsigned numeric values.
///
/// Supports 1, 2, 4, and 8-byte widths. Comparison is unsigned.
/// If the key does not exist, the operand is used as the initial value.
///
/// If `existing` and `operand` have different byte widths, the existing value
/// is preserved unchanged (no panic).
#[derive(Clone, Copy)]
pub struct NumericMin;

impl Debug for NumericMin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NumericMin")
    }
}

impl MergeOperator for NumericMin {
    fn merge(&self, _key: &[u8], existing: Option<&[u8]>, operand: &[u8]) -> Option<Vec<u8>> {
        let Some(existing) = existing else {
            return Some(operand.to_vec());
        };
        if existing.len() != operand.len() {
            return Some(existing.to_vec());
        }
        let use_operand = match operand.len() {
            1 => operand[0] < existing[0],
            2 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u16::from_le_bytes(a_bytes);
                let b = u16::from_le_bytes(b_bytes);
                b < a
            }
            4 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u32::from_le_bytes(a_bytes);
                let b = u32::from_le_bytes(b_bytes);
                b < a
            }
            8 => {
                let (Ok(a_bytes), Ok(b_bytes)) = (existing.try_into(), operand.try_into()) else {
                    return Some(existing.to_vec());
                };
                let a = u64::from_le_bytes(a_bytes);
                let b = u64::from_le_bytes(b_bytes);
                b < a
            }
            _ => return Some(existing.to_vec()),
        };
        if use_operand {
            Some(operand.to_vec())
        } else {
            Some(existing.to_vec())
        }
    }
}

/// Bitwise OR of fixed-width byte slices.
///
/// Both existing and operand must have the same length.
/// If the key does not exist, the operand is used as the initial value.
///
/// If `existing` and `operand` have different lengths, the existing value
/// is preserved unchanged (no panic).
#[derive(Clone, Copy)]
pub struct BitwiseOr;

impl Debug for BitwiseOr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BitwiseOr")
    }
}

impl MergeOperator for BitwiseOr {
    fn merge(&self, _key: &[u8], existing: Option<&[u8]>, operand: &[u8]) -> Option<Vec<u8>> {
        let Some(existing) = existing else {
            return Some(operand.to_vec());
        };
        if existing.len() != operand.len() {
            return Some(existing.to_vec());
        }
        let result: Vec<u8> = existing
            .iter()
            .zip(operand.iter())
            .map(|(a, b)| a | b)
            .collect();
        Some(result)
    }
}

/// Appends operand bytes to the existing value.
///
/// If the key does not exist, the operand is used as the initial value.
#[derive(Clone, Copy)]
pub struct BytesAppend;

impl Debug for BytesAppend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BytesAppend")
    }
}

impl MergeOperator for BytesAppend {
    fn merge(&self, _key: &[u8], existing: Option<&[u8]>, operand: &[u8]) -> Option<Vec<u8>> {
        let Some(existing) = existing else {
            return Some(operand.to_vec());
        };
        let mut result = Vec::with_capacity(existing.len() + operand.len());
        result.extend_from_slice(existing);
        result.extend_from_slice(operand);
        Some(result)
    }
}

/// A merge operator backed by a closure.
///
/// Created via [`merge_fn()`].
pub struct FnMergeOperator<F>
where
    F: Fn(&[u8], Option<&[u8]>, &[u8]) -> Option<Vec<u8>> + Send + Sync,
{
    f: F,
}

impl<F> Debug for FnMergeOperator<F>
where
    F: Fn(&[u8], Option<&[u8]>, &[u8]) -> Option<Vec<u8>> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FnMergeOperator")
    }
}

impl<F> MergeOperator for FnMergeOperator<F>
where
    F: Fn(&[u8], Option<&[u8]>, &[u8]) -> Option<Vec<u8>> + Send + Sync,
{
    fn merge(&self, key: &[u8], existing: Option<&[u8]>, operand: &[u8]) -> Option<Vec<u8>> {
        (self.f)(key, existing, operand)
    }
}

/// Creates a [`MergeOperator`] from a closure.
///
/// # Example
///
/// ```rust,ignore
/// use shodh_redb::merge_fn;
///
/// let op = merge_fn(|_key, existing, operand| {
///     // Custom merge: multiply existing by operand
///     let a = existing.map_or(1u64, |b| u64::from_le_bytes(b.try_into().unwrap()));
///     let b = u64::from_le_bytes(operand.try_into().unwrap());
///     Some((a * b).to_le_bytes().to_vec())
/// });
/// ```
pub fn merge_fn<F>(f: F) -> FnMergeOperator<F>
where
    F: Fn(&[u8], Option<&[u8]>, &[u8]) -> Option<Vec<u8>> + Send + Sync,
{
    FnMergeOperator { f }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &[u8] = b"test_key";

    // -----------------------------------------------------------------------
    // NumericAdd
    // -----------------------------------------------------------------------

    #[test]
    fn numeric_add_no_existing() {
        let op = NumericAdd;
        let operand = 42u64.to_le_bytes();
        let result = op.merge(KEY, None, &operand).unwrap();
        assert_eq!(result, operand);
    }

    #[test]
    fn numeric_add_u8() {
        let op = NumericAdd;
        let result = op.merge(KEY, Some(&[200]), &[100]).unwrap();
        assert_eq!(result, vec![44]); // 200 + 100 = 300, wraps to 44
    }

    #[test]
    fn numeric_add_u16() {
        let op = NumericAdd;
        let a = 1000u16.to_le_bytes();
        let b = 2000u16.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(u16::from_le_bytes(result.try_into().unwrap()), 3000);
    }

    #[test]
    fn numeric_add_u32() {
        let op = NumericAdd;
        let a = 100_000u32.to_le_bytes();
        let b = 200_000u32.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(u32::from_le_bytes(result.try_into().unwrap()), 300_000);
    }

    #[test]
    fn numeric_add_u64() {
        let op = NumericAdd;
        let a = 1_000_000u64.to_le_bytes();
        let b = 2_000_000u64.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(u64::from_le_bytes(result.try_into().unwrap()), 3_000_000);
    }

    #[test]
    fn numeric_add_wrapping_u64() {
        let op = NumericAdd;
        let a = u64::MAX.to_le_bytes();
        let b = 1u64.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(u64::from_le_bytes(result.try_into().unwrap()), 0);
    }

    #[test]
    fn numeric_add_mismatched_widths() {
        let op = NumericAdd;
        let a = 42u32.to_le_bytes();
        let b = 1u64.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(result, a, "mismatched widths should preserve existing");
    }

    #[test]
    fn numeric_add_unsupported_width() {
        let op = NumericAdd;
        let a = [1u8; 3];
        let b = [2u8; 3];
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(result, a, "3-byte width should preserve existing");
    }

    // -----------------------------------------------------------------------
    // SaturatingAdd
    // -----------------------------------------------------------------------

    #[test]
    fn saturating_add_no_existing() {
        let op = SaturatingAdd;
        let operand = 42u64.to_le_bytes();
        let result = op.merge(KEY, None, &operand).unwrap();
        assert_eq!(result, operand);
    }

    #[test]
    fn saturating_add_u8_no_overflow() {
        let op = SaturatingAdd;
        let result = op.merge(KEY, Some(&[100]), &[50]).unwrap();
        assert_eq!(result, vec![150]);
    }

    #[test]
    fn saturating_add_u8_saturates() {
        let op = SaturatingAdd;
        let result = op.merge(KEY, Some(&[200]), &[100]).unwrap();
        assert_eq!(result, vec![255]); // saturates at u8::MAX
    }

    #[test]
    fn saturating_add_u64_saturates() {
        let op = SaturatingAdd;
        let a = u64::MAX.to_le_bytes();
        let b = 1u64.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(
            u64::from_le_bytes(result.try_into().unwrap()),
            u64::MAX
        );
    }

    // -----------------------------------------------------------------------
    // FloatAdd
    // -----------------------------------------------------------------------

    #[test]
    fn float_add_no_existing() {
        let op = FloatAdd;
        let operand = 3.14f32.to_le_bytes();
        let result = op.merge(KEY, None, &operand).unwrap();
        assert_eq!(result, operand);
    }

    #[test]
    fn float_add_f32() {
        let op = FloatAdd;
        let a = 1.5f32.to_le_bytes();
        let b = 2.5f32.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        let sum = f32::from_le_bytes(result.try_into().unwrap());
        assert!((sum - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn float_add_f64() {
        let op = FloatAdd;
        let a = 1.5f64.to_le_bytes();
        let b = 2.5f64.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        let sum = f64::from_le_bytes(result.try_into().unwrap());
        assert!((sum - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn float_add_nan_propagates() {
        let op = FloatAdd;
        let a = f32::NAN.to_le_bytes();
        let b = 1.0f32.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        let sum = f32::from_le_bytes(result.try_into().unwrap());
        assert!(sum.is_nan());
    }

    #[test]
    fn float_add_unsupported_width() {
        let op = FloatAdd;
        let a = [1u8; 3];
        let b = [2u8; 3];
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(result, a);
    }

    // -----------------------------------------------------------------------
    // NumericMax
    // -----------------------------------------------------------------------

    #[test]
    fn numeric_max_no_existing() {
        let op = NumericMax;
        let operand = 42u64.to_le_bytes();
        let result = op.merge(KEY, None, &operand).unwrap();
        assert_eq!(result, operand);
    }

    #[test]
    fn numeric_max_u64_picks_larger() {
        let op = NumericMax;
        let a = 100u64.to_le_bytes();
        let b = 200u64.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(u64::from_le_bytes(result.try_into().unwrap()), 200);
    }

    #[test]
    fn numeric_max_u64_keeps_existing() {
        let op = NumericMax;
        let a = 200u64.to_le_bytes();
        let b = 100u64.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(u64::from_le_bytes(result.try_into().unwrap()), 200);
    }

    #[test]
    fn numeric_max_u8() {
        let op = NumericMax;
        let result = op.merge(KEY, Some(&[10]), &[20]).unwrap();
        assert_eq!(result, vec![20]);
    }

    // -----------------------------------------------------------------------
    // NumericMin
    // -----------------------------------------------------------------------

    #[test]
    fn numeric_min_no_existing() {
        let op = NumericMin;
        let operand = 42u64.to_le_bytes();
        let result = op.merge(KEY, None, &operand).unwrap();
        assert_eq!(result, operand);
    }

    #[test]
    fn numeric_min_u64_picks_smaller() {
        let op = NumericMin;
        let a = 200u64.to_le_bytes();
        let b = 100u64.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(u64::from_le_bytes(result.try_into().unwrap()), 100);
    }

    #[test]
    fn numeric_min_u64_keeps_existing() {
        let op = NumericMin;
        let a = 100u64.to_le_bytes();
        let b = 200u64.to_le_bytes();
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(u64::from_le_bytes(result.try_into().unwrap()), 100);
    }

    // -----------------------------------------------------------------------
    // BitwiseOr
    // -----------------------------------------------------------------------

    #[test]
    fn bitwise_or_no_existing() {
        let op = BitwiseOr;
        let operand = vec![0b1010_1010, 0b0101_0101];
        let result = op.merge(KEY, None, &operand).unwrap();
        assert_eq!(result, operand);
    }

    #[test]
    fn bitwise_or_combines() {
        let op = BitwiseOr;
        let a = [0b1010_0000, 0b0000_1111];
        let b = [0b0101_0000, 0b1111_0000];
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(result, vec![0b1111_0000, 0b1111_1111]);
    }

    #[test]
    fn bitwise_or_mismatched_lengths() {
        let op = BitwiseOr;
        let a = [0xFF, 0xFF];
        let b = [0x00];
        let result = op.merge(KEY, Some(&a), &b).unwrap();
        assert_eq!(result, a, "mismatched lengths should preserve existing");
    }

    // -----------------------------------------------------------------------
    // BytesAppend
    // -----------------------------------------------------------------------

    #[test]
    fn bytes_append_no_existing() {
        let op = BytesAppend;
        let operand = b"hello";
        let result = op.merge(KEY, None, operand).unwrap();
        assert_eq!(result, b"hello");
    }

    #[test]
    fn bytes_append_concatenates() {
        let op = BytesAppend;
        let result = op.merge(KEY, Some(b"hello "), b"world").unwrap();
        assert_eq!(result, b"hello world");
    }

    #[test]
    fn bytes_append_empty_operand() {
        let op = BytesAppend;
        let result = op.merge(KEY, Some(b"existing"), b"").unwrap();
        assert_eq!(result, b"existing");
    }

    #[test]
    fn bytes_append_empty_existing() {
        let op = BytesAppend;
        let result = op.merge(KEY, Some(b""), b"new").unwrap();
        assert_eq!(result, b"new");
    }

    // -----------------------------------------------------------------------
    // merge_fn
    // -----------------------------------------------------------------------

    #[test]
    fn merge_fn_custom_delete() {
        let op = merge_fn(|_key, _existing, operand| {
            if operand == b"DELETE" {
                None
            } else {
                Some(operand.to_vec())
            }
        });
        assert!(op.merge(KEY, Some(b"val"), b"DELETE").is_none());
        assert_eq!(
            op.merge(KEY, Some(b"val"), b"keep").unwrap(),
            b"keep"
        );
    }

    #[test]
    fn merge_fn_key_aware() {
        let op = merge_fn(|key, existing, operand| {
            if key == b"counter" {
                NumericAdd.merge(key, existing, operand)
            } else {
                Some(operand.to_vec())
            }
        });
        let a = 10u64.to_le_bytes();
        let b = 20u64.to_le_bytes();
        let result = op.merge(b"counter", Some(&a), &b).unwrap();
        assert_eq!(u64::from_le_bytes(result.try_into().unwrap()), 30);

        let result = op.merge(b"other", Some(b"old"), b"new").unwrap();
        assert_eq!(result, b"new");
    }
}
