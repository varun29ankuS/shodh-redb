#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use redb::blob_store::types::{BlobId, BlobRef, ContentType};
use redb::{Key, Value};

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    /// BlobId Value::from_bytes with arbitrary (possibly truncated) data.
    BlobIdFromBytes { data: Vec<u8> },
    /// BlobId roundtrip: construct -> to_be_bytes -> from_be_bytes.
    BlobIdRoundtrip { sequence: u64, hash: u64 },
    /// BlobId Key::compare ordering consistency.
    BlobIdCompare {
        seq_a: u64,
        hash_a: u64,
        seq_b: u64,
        hash_b: u64,
    },
    /// BlobRef roundtrip.
    BlobRefRoundtrip {
        offset: u64,
        length: u64,
        checksum: u128,
        ref_count: u32,
        content_type: u8,
        compression: u8,
    },
    /// ContentType from_byte for all values.
    ContentTypeFromByte { byte: u8 },
    /// BlobId from_bytes with exact size.
    BlobIdFromBytesExact { sequence: u64, hash: u64 },
}

fuzz_target!(|op: FuzzOp| {
    match op {
        FuzzOp::BlobIdFromBytes { data } => {
            // Must not panic for any input -- uses try_into_padded.
            let id = <BlobId as Value>::from_bytes(&data);
            let _ = id.sequence;
            let _ = id.content_prefix_hash;
        }

        FuzzOp::BlobIdRoundtrip { sequence, hash } => {
            let id = BlobId::new(sequence, hash);
            let bytes = id.to_be_bytes();
            let restored = BlobId::from_be_bytes(bytes);
            assert_eq!(restored.sequence, id.sequence);
            assert_eq!(restored.content_prefix_hash, id.content_prefix_hash);

            // Value trait roundtrip.
            let val_bytes = <BlobId as Value>::as_bytes(&id);
            let val_restored = <BlobId as Value>::from_bytes(&val_bytes);
            assert_eq!(val_restored.sequence, id.sequence);
            assert_eq!(val_restored.content_prefix_hash, id.content_prefix_hash);
        }

        FuzzOp::BlobIdCompare {
            seq_a,
            hash_a,
            seq_b,
            hash_b,
        } => {
            let a = BlobId::new(seq_a, hash_a);
            let b = BlobId::new(seq_b, hash_b);

            let bytes_a = <BlobId as Value>::as_bytes(&a);
            let bytes_b = <BlobId as Value>::as_bytes(&b);

            // Key::compare must be consistent with Ord.
            let key_cmp = <BlobId as Key>::compare(&bytes_a, &bytes_b);
            let ord_cmp = a.cmp(&b);
            assert_eq!(
                key_cmp, ord_cmp,
                "Key::compare disagrees with Ord: {key_cmp:?} vs {ord_cmp:?} for {a:?} vs {b:?}"
            );
        }

        FuzzOp::BlobRefRoundtrip {
            offset,
            length,
            checksum,
            ref_count,
            content_type,
            compression,
        } => {
            let blob_ref = BlobRef {
                offset,
                length,
                checksum,
                ref_count,
                content_type,
                compression,
            };
            let bytes = blob_ref.to_le_bytes();
            let restored = BlobRef::from_le_bytes(bytes);
            assert_eq!(restored.offset, blob_ref.offset);
            assert_eq!(restored.length, blob_ref.length);
            assert_eq!(restored.checksum, blob_ref.checksum);
            assert_eq!(restored.ref_count, blob_ref.ref_count);
            assert_eq!(restored.content_type, blob_ref.content_type);
            assert_eq!(restored.compression, blob_ref.compression);
        }

        FuzzOp::ContentTypeFromByte { byte } => {
            let ct = ContentType::from_byte(byte);
            // Roundtrip for known values.
            if byte <= 9 {
                assert_eq!(ct.as_byte(), byte);
            } else {
                // Unknown values map to OctetStream (0).
                assert_eq!(ct.as_byte(), 0);
            }
        }

        FuzzOp::BlobIdFromBytesExact { sequence, hash } => {
            let id = BlobId::new(sequence, hash);
            let bytes = <BlobId as Value>::as_bytes(&id);
            let restored = <BlobId as Value>::from_bytes(&bytes);
            assert_eq!(restored.sequence, sequence);
            assert_eq!(restored.content_prefix_hash, hash);
        }
    }
});
