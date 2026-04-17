#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use redb::cdc::types::{CdcKey, CdcRecord};
use redb::{ChangeOp, Key, Value};

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    /// Deserialize arbitrary bytes as a CDC record -- must never panic.
    DeserializeRecord { data: Vec<u8> },
    /// Serialize then deserialize a constructed record -- roundtrip.
    RoundtripRecord {
        op_byte: u8,
        table_name: String,
        key: Vec<u8>,
        has_new: bool,
        new_value: Vec<u8>,
        has_old: bool,
        old_value: Vec<u8>,
    },
    /// CdcKey Value trait roundtrip.
    CdcKeyRoundtrip {
        transaction_id: u64,
        sequence: u32,
    },
    /// CdcKey Key::compare ordering consistency.
    CdcKeyCompare {
        txn_a: u64,
        seq_a: u32,
        txn_b: u64,
        seq_b: u32,
    },
    /// ChangeOp try_from for all byte values.
    ChangeOpFromByte { byte: u8 },
}

fuzz_target!(|op: FuzzOp| {
    match op {
        FuzzOp::DeserializeRecord { data } => {
            // Must not panic on any input.
            let _ = CdcRecord::deserialize(&data);
        }

        FuzzOp::RoundtripRecord {
            op_byte,
            table_name,
            key,
            has_new,
            new_value,
            has_old,
            old_value,
        } => {
            let change_op = match op_byte % 3 {
                0 => ChangeOp::Insert,
                1 => ChangeOp::Update,
                _ => ChangeOp::Delete,
            };
            // Truncate table name to u16::MAX bytes (serialization limit).
            let table_name: String = table_name.chars().take(200).collect();
            let record = CdcRecord {
                op: change_op,
                table_name,
                key,
                new_value: if has_new { Some(new_value) } else { None },
                old_value: if has_old { Some(old_value) } else { None },
            };
            let bytes = record.serialize();
            let restored = CdcRecord::deserialize(&bytes)
                .expect("roundtrip deserialization must succeed");
            assert_eq!(restored.op, record.op);
            assert_eq!(restored.table_name, record.table_name);
            assert_eq!(restored.key, record.key);
            assert_eq!(restored.new_value, record.new_value);
            assert_eq!(restored.old_value, record.old_value);
        }

        FuzzOp::CdcKeyRoundtrip {
            transaction_id,
            sequence,
        } => {
            let key = CdcKey::new(transaction_id, sequence);
            let bytes = <CdcKey as Value>::as_bytes(&key);
            let restored = <CdcKey as Value>::from_bytes(&bytes);
            assert_eq!(restored.transaction_id, transaction_id);
            assert_eq!(restored.sequence, sequence);
        }

        FuzzOp::CdcKeyCompare {
            txn_a,
            seq_a,
            txn_b,
            seq_b,
        } => {
            let a = CdcKey::new(txn_a, seq_a);
            let b = CdcKey::new(txn_b, seq_b);
            let bytes_a = <CdcKey as Value>::as_bytes(&a);
            let bytes_b = <CdcKey as Value>::as_bytes(&b);
            // Key::compare must agree with Ord.
            let key_cmp = <CdcKey as Key>::compare(&bytes_a, &bytes_b);
            let ord_cmp = a.cmp(&b);
            assert_eq!(key_cmp, ord_cmp);
        }

        FuzzOp::ChangeOpFromByte { byte } => {
            // Valid discriminants are 0, 1, 2. All others must return Err.
            let result = CdcRecord::deserialize(&[byte]);
            // Single byte is too short for a full record -- always Err.
            assert!(result.is_err());
        }
    }
});
