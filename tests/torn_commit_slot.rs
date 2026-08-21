//! A torn commit slot must be recoverable, not fatal.
//!
//! The two-slot commit design exists so that a crash partway through writing
//! one slot leaves the other intact. `pick_primary_for_repair` recovers from
//! whichever slot verified. That only works if parsing a torn slot reports it
//! as corrupted; returning an error instead failed the whole open, which is
//! data loss on a file whose data is entirely intact.
//!
//! The version byte is the first byte of a slot, so a torn write routinely
//! leaves a version this build does not recognise.
#![cfg(not(target_os = "wasi"))]

use shodh_redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use std::io::{Read, Seek, SeekFrom, Write};

const TABLE: TableDefinition<u64, u64> = TableDefinition::new("t");
const GOD_BYTE_OFFSET: u64 = 9;
const PRIMARY_BIT: u8 = 1;
const TRANSACTION_0_OFFSET: u64 = 64;
const TRANSACTION_1_OFFSET: u64 = 192;
const TRANSACTION_SIZE: usize = 128;

fn seeded_database() -> (tempfile::NamedTempFile, std::path::PathBuf) {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    {
        let db = Database::create(&path).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(TABLE).unwrap();
            for k in 0..500u64 {
                t.insert(k, k * 2).unwrap();
            }
        }
        txn.commit().unwrap();
    }
    (tmp, path)
}

fn slot_offsets(path: &std::path::Path) -> (u64, u64) {
    let mut f = std::fs::File::open(path).unwrap();
    let mut god = [0u8; 1];
    f.seek(SeekFrom::Start(GOD_BYTE_OFFSET)).unwrap();
    f.read_exact(&mut god).unwrap();
    if god[0] & PRIMARY_BIT == 0 {
        (TRANSACTION_0_OFFSET, TRANSACTION_1_OFFSET)
    } else {
        (TRANSACTION_1_OFFSET, TRANSACTION_0_OFFSET)
    }
}

fn overwrite(path: &std::path::Path, at: u64, bytes: &[u8]) {
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    f.seek(SeekFrom::Start(at)).unwrap();
    f.write_all(bytes).unwrap();
    f.sync_all().unwrap();
}

#[test]
fn a_torn_secondary_slot_does_not_prevent_opening() {
    let (_tmp, path) = seeded_database();
    let (_primary, secondary) = slot_offsets(&path);
    overwrite(&path, secondary, &[0xFFu8; TRANSACTION_SIZE]);

    let db = Database::open(&path).expect("a torn secondary slot must not fail the open");
    let rtxn = db.begin_read().unwrap();
    let t = rtxn.open_table(TABLE).unwrap();
    assert_eq!(t.get(&499).unwrap().unwrap().value(), 998);
    assert_eq!(t.iter().unwrap().count(), 500);
}

/// The primary is the slot the god byte points at, but a torn primary is just
/// as survivable: recovery swaps to the secondary.
#[test]
fn a_torn_primary_slot_does_not_prevent_opening() {
    let (_tmp, path) = seeded_database();
    let (primary, _secondary) = slot_offsets(&path);
    overwrite(&path, primary, &[0xFFu8; TRANSACTION_SIZE]);

    let db = Database::open(&path).expect("a torn primary slot must not fail the open");
    assert!(db.begin_read().is_ok());
}

/// Both slots unreadable is a genuinely unopenable file, and it must still say
/// so rather than being quietly tolerated.
#[test]
fn two_torn_slots_are_still_reported() {
    let (_tmp, path) = seeded_database();
    overwrite(&path, TRANSACTION_0_OFFSET, &[0xFFu8; TRANSACTION_SIZE]);
    overwrite(&path, TRANSACTION_1_OFFSET, &[0xFFu8; TRANSACTION_SIZE]);

    assert!(
        Database::open(&path).is_err(),
        "a file with no readable commit slot must not open"
    );
}
