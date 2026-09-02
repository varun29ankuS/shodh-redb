//! With compression enabled, a value stored uncompressed carries a one-byte
//! flags envelope. `get()` strips it; the iterator path did not, so `iter()`
//! returned a different value than `get()` for the same key, and TTL entries
//! never expired because their expiry header was read shifted by one byte.
#![cfg(feature = "compression_lz4")]

use shodh_redb::{
    CompressionConfig, Database, ReadableDatabase, ReadableTable, TableDefinition,
    TtlTableDefinition,
};
use std::time::Duration;

const KV: TableDefinition<&str, String> = TableDefinition::new("kv");
const TTL: TtlTableDefinition<&str, String> = TtlTableDefinition::new("ttl");

fn compressed_db(file: &tempfile::NamedTempFile) -> Database {
    Database::builder()
        .set_compression(CompressionConfig::Lz4)
        .create(file.path())
        .unwrap()
}

#[test]
fn iter_returns_the_same_value_as_get_for_an_uncompressed_value() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let db = compressed_db(&file);

    // Under MIN_VALUE_COMPRESS_SIZE, so this is stored uncompressed behind the
    // flags envelope -- the case the two read paths disagreed about.
    let short = "hello".to_string();
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(KV).unwrap();
        t.insert("short", &short).unwrap();
    }
    txn.commit().unwrap();

    let rtx = db.begin_read().unwrap();
    let t = rtx.open_table(KV).unwrap();

    assert_eq!(t.get("short").unwrap().unwrap().value(), short, "get()");

    let mut seen = 0;
    for item in t.iter().unwrap() {
        let (k, v) = item.unwrap();
        assert_eq!(k.value(), "short");
        assert_eq!(
            v.value(),
            short,
            "iter() must not surface the compression flags byte"
        );
        seen += 1;
    }
    assert_eq!(seen, 1);

    // range() shares the same guard.
    for item in t.range("a".."z").unwrap() {
        let (_, v) = item.unwrap();
        assert_eq!(v.value(), short, "range() must agree with get()");
    }
}

#[test]
fn an_uncompressed_ttl_entry_actually_expires() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let db = compressed_db(&file);

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_ttl_table(TTL).unwrap();
        t.insert_with_ttl("k", &"v".to_string(), Duration::from_millis(200))
            .unwrap();
    }
    txn.commit().unwrap();

    std::thread::sleep(Duration::from_millis(600));

    // The expiry header sits directly after the flags byte. Read one byte off,
    // it decodes as expiry << 8 -- a timestamp far in the future -- so the
    // entry looked permanently alive and purge_expired removed nothing.
    let txn = db.begin_write().unwrap();
    let purged = {
        let mut t = txn.open_ttl_table(TTL).unwrap();
        t.purge_expired().unwrap()
    };
    txn.commit().unwrap();

    assert_eq!(
        purged, 1,
        "an expired entry stored uncompressed must be purged, not treated as alive"
    );
}
