#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use redb::IncrementalSnapshot;

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    /// Deserialize arbitrary bytes — must never panic.
    DeserializeRaw { data: Vec<u8> },
    /// If deserialization succeeds, roundtrip through to_bytes → from_bytes.
    DeserializeAndRoundtrip { data: Vec<u8> },
}

fuzz_target!(|op: FuzzOp| {
    match op {
        FuzzOp::DeserializeRaw { data } => {
            // Must not panic on any input.
            let _ = IncrementalSnapshot::from_bytes(&data);
        }

        FuzzOp::DeserializeAndRoundtrip { data } => {
            // If arbitrary bytes happen to parse, verify roundtrip.
            if let Ok(snapshot) = IncrementalSnapshot::from_bytes(&data) {
                let bytes = snapshot.to_bytes();
                let restored = IncrementalSnapshot::from_bytes(&bytes)
                    .expect("roundtrip deserialization must succeed");
                assert_eq!(restored.base_txn, snapshot.base_txn);
                assert_eq!(restored.current_txn, snapshot.current_txn);
                assert_eq!(restored.tables_changed(), snapshot.tables_changed());
                assert_eq!(restored.total_upserts(), snapshot.total_upserts());
                assert_eq!(restored.total_deletes(), snapshot.total_deletes());
                assert_eq!(
                    restored.dropped_table_names(),
                    snapshot.dropped_table_names()
                );
            }
        }
    }
});
