#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use redb::merge::{
    BitwiseOr, BytesAppend, FloatAdd, MergeOperator, NumericAdd, NumericMax, NumericMin,
    SaturatingAdd,
};

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    /// Test NumericAdd with arbitrary data.
    NumericAdd {
        key: Vec<u8>,
        existing: Vec<u8>,
        operand: Vec<u8>,
    },
    /// Test SaturatingAdd with arbitrary data.
    SaturatingAdd {
        key: Vec<u8>,
        existing: Vec<u8>,
        operand: Vec<u8>,
    },
    /// Test FloatAdd with arbitrary data.
    FloatAdd {
        key: Vec<u8>,
        existing: Vec<u8>,
        operand: Vec<u8>,
    },
    /// Test NumericMax with arbitrary data.
    NumericMax {
        key: Vec<u8>,
        existing: Vec<u8>,
        operand: Vec<u8>,
    },
    /// Test NumericMin with arbitrary data.
    NumericMin {
        key: Vec<u8>,
        existing: Vec<u8>,
        operand: Vec<u8>,
    },
    /// Test BitwiseOr with arbitrary data.
    BitwiseOr {
        key: Vec<u8>,
        existing: Vec<u8>,
        operand: Vec<u8>,
    },
    /// Test BytesAppend with arbitrary data.
    BytesAppend {
        key: Vec<u8>,
        existing: Vec<u8>,
        operand: Vec<u8>,
    },
    /// Test all operators with no existing value.
    NoExisting {
        operator_sel: u8,
        key: Vec<u8>,
        operand: Vec<u8>,
    },
}

fuzz_target!(|op: FuzzOp| {
    match op {
        FuzzOp::NumericAdd {
            key,
            existing,
            operand,
        } => {
            let result = NumericAdd.merge(&key, Some(&existing), &operand);
            if existing.len() == operand.len() && matches!(existing.len(), 1 | 2 | 4 | 8) {
                // Valid widths: must produce a result.
                assert!(result.is_some());
                assert_eq!(result.unwrap().len(), existing.len());
            }
        }

        FuzzOp::SaturatingAdd {
            key,
            existing,
            operand,
        } => {
            let result = SaturatingAdd.merge(&key, Some(&existing), &operand);
            if existing.len() == operand.len() && matches!(existing.len(), 1 | 2 | 4 | 8) {
                assert!(result.is_some());
                assert_eq!(result.unwrap().len(), existing.len());
            }
        }

        FuzzOp::FloatAdd {
            key,
            existing,
            operand,
        } => {
            let result = FloatAdd.merge(&key, Some(&existing), &operand);
            if existing.len() == operand.len() && matches!(existing.len(), 4 | 8) {
                assert!(result.is_some());
                assert_eq!(result.unwrap().len(), existing.len());
            }
        }

        FuzzOp::NumericMax {
            key,
            existing,
            operand,
        } => {
            let result = NumericMax.merge(&key, Some(&existing), &operand);
            if existing.len() == operand.len() && matches!(existing.len(), 1 | 2 | 4 | 8) {
                assert!(result.is_some());
                assert_eq!(result.unwrap().len(), existing.len());
            }
        }

        FuzzOp::NumericMin {
            key,
            existing,
            operand,
        } => {
            let result = NumericMin.merge(&key, Some(&existing), &operand);
            if existing.len() == operand.len() && matches!(existing.len(), 1 | 2 | 4 | 8) {
                assert!(result.is_some());
                assert_eq!(result.unwrap().len(), existing.len());
            }
        }

        FuzzOp::BitwiseOr {
            key,
            existing,
            operand,
        } => {
            let result = BitwiseOr.merge(&key, Some(&existing), &operand);
            assert!(result.is_some());
            let result = result.unwrap();
            assert_eq!(result.len(), existing.len().max(operand.len()));
        }

        FuzzOp::BytesAppend {
            key,
            existing,
            operand,
        } => {
            let result = BytesAppend.merge(&key, Some(&existing), &operand);
            assert!(result.is_some());
            let result = result.unwrap();
            assert_eq!(result.len(), existing.len() + operand.len());
            // Verify prefix is existing, suffix is operand.
            assert_eq!(&result[..existing.len()], &existing[..]);
            assert_eq!(&result[existing.len()..], &operand[..]);
        }

        FuzzOp::NoExisting {
            operator_sel,
            key,
            operand,
        } => {
            // With no existing value, all operators should return Some(operand).
            let result = match operator_sel % 7 {
                0 => NumericAdd.merge(&key, None, &operand),
                1 => SaturatingAdd.merge(&key, None, &operand),
                2 => FloatAdd.merge(&key, None, &operand),
                3 => NumericMax.merge(&key, None, &operand),
                4 => NumericMin.merge(&key, None, &operand),
                5 => BitwiseOr.merge(&key, None, &operand),
                _ => BytesAppend.merge(&key, None, &operand),
            };
            assert!(result.is_some());
            assert_eq!(result.unwrap(), operand);
        }
    }
});
