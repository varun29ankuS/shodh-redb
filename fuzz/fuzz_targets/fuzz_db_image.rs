#![no_main]

//! Fuzz `Database::open` and a first read against a crafted database image.
//!
//! A fuzzer that only flips random bytes mostly proves that the checksums
//! reject garbage -- it never reaches the code behind them. This target starts
//! from a real database image, applies mutations, and then *rewrites the commit
//! slot checksum*, so the header is internally consistent and the fields it
//! changed are actually read.
//!
//! The invariant under test is narrow and absolute: opening any byte string
//! must return `Ok` or `Err`, never panic and never abort. Corruption is
//! expected; panicking on it is not.

use libfuzzer_sys::fuzz_target;
use redb::backends::FileBackend;
use redb::fuzzing::slot_checksum;
use redb::{
    Builder, Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, StorageBackend,
    TableDefinition,
};
use std::fs::OpenOptions;
use std::sync::OnceLock;

const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("fuzz");

// Header layout, mirroring the private constants in page_store::header.
const TRANSACTION_0_OFFSET: usize = 64;
const TRANSACTION_SIZE: usize = 128;
const SLOT_CHECKSUM_IN_SLOT: usize = 112;
const DB_HEADER_SIZE: usize = 320;

/// A real database image, built once and reused as the mutation base.
fn pristine() -> &'static Vec<u8> {
    static IMAGE: OnceLock<Vec<u8>> = OnceLock::new();
    IMAGE.get_or_init(|| {
        let file = tempfile::NamedTempFile::new().unwrap();
        {
            let db = Database::create(file.path()).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let mut t = txn.open_table(TABLE).unwrap();
                // Enough entries to force a branch level, so the image has
                // interior pages and not just a single leaf.
                for i in 0..256u64 {
                    t.insert(&i, [i as u8; 64].as_slice()).unwrap();
                }
            }
            txn.commit().unwrap();
        }
        std::fs::read(file.path()).unwrap()
    })
}

/// Rewrite both commit slots' checksums so the header parses as consistent.
fn reseal_slots(image: &mut [u8]) {
    if image.len() < DB_HEADER_SIZE {
        return;
    }
    for slot in 0..2 {
        let base = TRANSACTION_0_OFFSET + slot * TRANSACTION_SIZE;
        let checksum = slot_checksum(&image[base..base + SLOT_CHECKSUM_IN_SLOT]);
        image[base + SLOT_CHECKSUM_IN_SLOT..base + TRANSACTION_SIZE]
            .copy_from_slice(&checksum.to_le_bytes());
    }
}

fuzz_target!(|data: &[u8]| {
    // Need at least a mutation count plus one (offset, byte) pair.
    if data.len() < 4 {
        return;
    }

    let mut image = pristine().clone();

    // Interpret the input as a series of (u16 offset, u8 value) edits. Biasing
    // offsets into the header keeps the fuzzer on the metadata that decides how
    // the rest of the file is interpreted, rather than scattering bytes across
    // page payloads that the page checksums will reject anyway.
    let header_biased = data[0] & 1 == 0;
    let mut i = 1;
    while i + 2 < data.len() {
        let raw = u16::from_le_bytes([data[i], data[i + 1]]) as usize;
        let value = data[i + 2];
        i += 3;

        let offset = if header_biased {
            raw % DB_HEADER_SIZE
        } else {
            raw % image.len()
        };
        image[offset] = value;
    }

    // Make the header self-consistent, so mutations to slot fields survive
    // validation and are actually acted on.
    if data[0] & 2 == 0 {
        reseal_slots(&mut image);
    }

    let file = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    if std::fs::write(file.path(), &image).is_err() {
        return;
    }

    let backend = match OpenOptions::new()
        .read(true)
        .write(true)
        .open(file.path())
        .map_err(drop)
        .and_then(|f| FileBackend::new(f).map_err(drop))
    {
        Ok(b) => b,
        Err(()) => return,
    };

    // The whole point: any outcome except a panic is acceptable.
    let Ok(db) = Builder::new().create_with_backend(backend) else {
        return;
    };

    let Ok(txn) = db.begin_read() else {
        return;
    };
    let Ok(table) = txn.open_table(TABLE) else {
        return;
    };

    // Walk the tree, since that is where an inconsistent-but-sealed header
    // sends us. Errors are fine; panics are not.
    if let Ok(iter) = table.iter() {
        for entry in iter.take(512) {
            if entry.is_err() {
                break;
            }
        }
    }
    let _ = table.get(&0u64);
    let _ = table.len();
});

// Keep the backend trait in scope: FileBackend must satisfy it for
// create_with_backend, and naming it here makes that dependency explicit.
const _: fn() = || {
    fn assert_backend<T: StorageBackend>() {}
    assert_backend::<FileBackend>();
};
