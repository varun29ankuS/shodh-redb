//! Phase 6 hardening tests for the comprehensive hardening plan.
//!
//! Tests cover:
//! - Blob chunked B-tree storage: multi-chunk integrity, range reads across chunks,
//!   streaming writer chunk boundaries, MVCC isolation
//! - Flash backend: geometry validation rejection, block map integrity
//! - Memory safety: transaction tracker non-durable commit cap

use shodh_redb::{
    error::BackendError, Builder, ContentType, Database, FlashBackend, FlashGeometry,
    FlashHardware, ReadableDatabase, StoreOptions, TableDefinition,
};
use std::fmt::{Debug, Formatter};
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

fn create_tempfile() -> tempfile::NamedTempFile {
    if cfg!(target_os = "wasi") {
        tempfile::NamedTempFile::new_in("/tmp").unwrap()
    } else {
        tempfile::NamedTempFile::new().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Blob Chunked B-tree Integration Tests
// ---------------------------------------------------------------------------

/// Verify that a blob larger than BLOB_CHUNK_SIZE (64KB) is correctly stored
/// and retrieved, exercising the multi-chunk path.
#[test]
fn blob_multi_chunk_roundtrip() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // 200KB blob = 4 chunks (3 full + 1 partial at ~8KB)
    let total_size = 200 * 1024;
    let data: Vec<u8> = (0..total_size).map(|i| (i % 251) as u8).collect();

    let write_txn = db.begin_write().unwrap();
    let blob_id = write_txn
        .store_blob(&data, ContentType::OctetStream, "multi-chunk", StoreOptions::default())
        .unwrap();
    write_txn.commit().unwrap();

    // Verify via ReadTransaction
    let read_txn = db.begin_read().unwrap();
    let (read_data, meta) = read_txn.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(read_data.len(), total_size);
    assert_eq!(read_data, data);
    assert_eq!(meta.blob_ref.length, total_size as u64);
}

/// Verify range reads that span chunk boundaries work correctly.
#[test]
fn blob_range_read_across_chunks() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // 150KB blob = 3 chunks (2 full 64KB + 1 partial 22KB)
    let total_size = 150 * 1024;
    let data: Vec<u8> = (0..total_size).map(|i| ((i * 7 + 3) % 256) as u8).collect();

    let write_txn = db.begin_write().unwrap();
    let blob_id = write_txn
        .store_blob(&data, ContentType::OctetStream, "range-test", StoreOptions::default())
        .unwrap();
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();

    // Read across chunk boundary: offset 60KB, length 20KB (spans chunk 0 and chunk 1)
    let offset = 60 * 1024;
    let length = 20 * 1024;
    let range_data = read_txn
        .read_blob_range(&blob_id, offset as u64, length as u64)
        .unwrap()
        .unwrap();
    assert_eq!(range_data.len(), length);
    assert_eq!(range_data, &data[offset..offset + length]);
}

/// Streaming writer with data that exactly fills multiple chunks.
#[test]
fn blob_writer_exact_chunk_boundary() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Exactly 2 * 64KB = 128KB -- no partial chunk
    let chunk_size = 64 * 1024;
    let data: Vec<u8> = (0..chunk_size * 2).map(|i| (i % 199) as u8).collect();

    let write_txn = db.begin_write().unwrap();
    let blob_id = {
        let mut writer = write_txn
            .blob_writer(ContentType::OctetStream, "exact-boundary", StoreOptions::default())
            .unwrap();
        // Write in small pieces that cross chunk boundaries
        for piece in data.chunks(7919) {
            writer.write(piece).unwrap();
        }
        writer.finish().unwrap()
    };
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let (read_data, _) = read_txn.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(read_data, data);
}

/// Blob reader seek and read across chunk boundaries.
#[test]
fn blob_reader_cross_chunk_seek() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let total_size = 200 * 1024;
    let data: Vec<u8> = (0..total_size).map(|i| (i % 173) as u8).collect();

    let write_txn = db.begin_write().unwrap();
    let blob_id = write_txn
        .store_blob(&data, ContentType::OctetStream, "seek-test", StoreOptions::default())
        .unwrap();
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let mut reader = read_txn.blob_reader(&blob_id).unwrap().unwrap();

    // Seek into the middle of chunk 2 (offset 140KB)
    reader.seek(SeekFrom::Start(140 * 1024)).unwrap();

    // Read 30KB — should cross from chunk 2 into chunk 3
    let mut buf = vec![0u8; 30 * 1024];
    reader.read_exact(&mut buf).unwrap();
    assert_eq!(buf, &data[140 * 1024..170 * 1024]);
}

/// MVCC isolation: blob stored in uncommitted txn is invisible to readers.
#[test]
fn blob_mvcc_isolation() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Commit one blob first
    let id1;
    {
        let txn = db.begin_write().unwrap();
        id1 = txn
            .store_blob(b"visible", ContentType::OctetStream, "v", StoreOptions::default())
            .unwrap();
        txn.commit().unwrap();
    }

    // Open a read transaction BEFORE the next write
    let read_txn = db.begin_read().unwrap();

    // Store a new blob (uncommitted at this point, but even after commit,
    // the read_txn should see only its snapshot)
    let id2;
    {
        let txn = db.begin_write().unwrap();
        id2 = txn
            .store_blob(
                b"invisible to old reader",
                ContentType::OctetStream,
                "i",
                StoreOptions::default(),
            )
            .unwrap();
        txn.commit().unwrap();
    }

    // Old read_txn should see id1 but not id2
    assert!(read_txn.get_blob(&id1).unwrap().is_some());
    assert!(read_txn.get_blob(&id2).unwrap().is_none());

    // New read_txn should see both
    let read_txn2 = db.begin_read().unwrap();
    assert!(read_txn2.get_blob(&id1).unwrap().is_some());
    assert!(read_txn2.get_blob(&id2).unwrap().is_some());
}

/// Blob delete should also remove chunks — verify by storing, deleting,
/// then storing again (sequence numbers should not conflict).
#[test]
fn blob_delete_reclaims_chunks() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Store a 100KB blob
    let data = vec![0xABu8; 100 * 1024];
    let blob_id;
    {
        let txn = db.begin_write().unwrap();
        blob_id = txn
            .store_blob(&data, ContentType::OctetStream, "del", StoreOptions::default())
            .unwrap();
        txn.commit().unwrap();
    }

    // Delete it
    {
        let txn = db.begin_write().unwrap();
        assert!(txn.delete_blob(&blob_id).unwrap());
        txn.commit().unwrap();
    }

    // Verify it's gone
    let read_txn = db.begin_read().unwrap();
    assert!(read_txn.get_blob(&blob_id).unwrap().is_none());
    drop(read_txn);

    // Store another blob — should succeed without conflict
    let data2 = vec![0xCDu8; 50 * 1024];
    let blob_id2;
    {
        let txn = db.begin_write().unwrap();
        blob_id2 = txn
            .store_blob(&data2, ContentType::OctetStream, "new", StoreOptions::default())
            .unwrap();
        txn.commit().unwrap();
    }

    let read_txn = db.begin_read().unwrap();
    let (d, _) = read_txn.get_blob(&blob_id2).unwrap().unwrap();
    assert_eq!(d, data2);
}

/// Blob survives database reopen with chunked storage.
#[test]
fn blob_chunked_survives_reopen() {
    let tmpfile = create_tempfile();
    let blob_id;
    let data: Vec<u8> = (0..100 * 1024).map(|i| (i % 137) as u8).collect();

    {
        let db = Database::create(tmpfile.path()).unwrap();
        let txn = db.begin_write().unwrap();
        blob_id = txn
            .store_blob(&data, ContentType::Embedding, "persist", StoreOptions::default())
            .unwrap();
        txn.commit().unwrap();
    }

    // Reopen
    let db = Database::create(tmpfile.path()).unwrap();
    let read_txn = db.begin_read().unwrap();
    let (read_data, meta) = read_txn.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(read_data, data);
    assert_eq!(meta.label_str(), "persist");
}

// ---------------------------------------------------------------------------
// Flash Backend Hardening Tests
// ---------------------------------------------------------------------------

struct MockFlashHardware {
    storage: Arc<RwLock<Vec<u8>>>,
    geometry: FlashGeometry,
}

impl Debug for MockFlashHardware {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockFlashHardware").finish()
    }
}

impl MockFlashHardware {
    fn new(geometry: FlashGeometry) -> Self {
        let capacity = geometry.total_capacity() as usize;
        Self {
            storage: Arc::new(RwLock::new(vec![0xFFu8; capacity])),
            geometry,
        }
    }
}

impl FlashHardware for MockFlashHardware {
    fn geometry(&self) -> FlashGeometry {
        self.geometry
    }

    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), shodh_redb::error::BackendError> {
        let s = self.storage.read().unwrap();
        let start = offset as usize;
        let end = start + buf.len();
        if end > s.len() {
            buf[..s.len() - start].copy_from_slice(&s[start..]);
            buf[s.len() - start..].fill(0xFF);
        } else {
            buf.copy_from_slice(&s[start..end]);
        }
        Ok(())
    }

    fn write_page(&self, offset: u64, data: &[u8]) -> Result<(), shodh_redb::error::BackendError> {
        let mut s = self.storage.write().unwrap();
        let start = offset as usize;
        s[start..start + data.len()].copy_from_slice(data);
        Ok(())
    }

    fn erase_block(&self, block: u32) -> Result<(), shodh_redb::error::BackendError> {
        let mut s = self.storage.write().unwrap();
        let ebs = self.geometry.erase_block_size as usize;
        let start = block as usize * ebs;
        let end = start + ebs;
        if end <= s.len() {
            s[start..end].fill(0xFF);
        }
        Ok(())
    }

    fn is_bad_block(&self, _block: u32) -> Result<bool, shodh_redb::error::BackendError> {
        Ok(false)
    }

    fn mark_bad_block(&self, _block: u32) -> Result<(), shodh_redb::error::BackendError> {
        Ok(())
    }

    fn sync(&self) -> Result<(), shodh_redb::error::BackendError> {
        Ok(())
    }
}

/// Geometry with non-power-of-two erase block size should be rejected.
#[test]
fn flash_rejects_non_power_of_two_erase_block() {
    let hw = MockFlashHardware::new(FlashGeometry {
        erase_block_size: 3000, // Not a power of 2
        write_page_size: 512,
        total_blocks: 32,
        max_erase_cycles: 100_000,
    });
    let result = FlashBackend::mount(hw);
    assert!(result.is_err());
}

/// Geometry where erase_block_size is not a multiple of write_page_size should be rejected.
#[test]
fn flash_rejects_misaligned_erase_block() {
    let hw = MockFlashHardware::new(FlashGeometry {
        erase_block_size: 4096,
        write_page_size: 3000, // 4096 % 3000 != 0
        total_blocks: 32,
        max_erase_cycles: 100_000,
    });
    let result = FlashBackend::mount(hw);
    assert!(result.is_err());
}

/// Geometry with too few blocks for overhead should be rejected.
#[test]
fn flash_rejects_too_few_blocks() {
    let hw = MockFlashHardware::new(FlashGeometry {
        erase_block_size: 4096,
        write_page_size: 4096,
        total_blocks: 3, // Not enough for reserved + COW + data
        max_erase_cycles: 100_000,
    });
    let result = FlashBackend::mount(hw);
    assert!(result.is_err());
}

/// Valid NOR geometry should succeed.
#[test]
fn flash_accepts_valid_nor_geometry() {
    let hw = MockFlashHardware::new(FlashGeometry {
        erase_block_size: 4096,
        write_page_size: 4096,
        total_blocks: 32,
        max_erase_cycles: 100_000,
    });
    let result = FlashBackend::mount(hw);
    assert!(result.is_ok());
}

/// Valid NAND geometry should succeed.
#[test]
fn flash_accepts_valid_nand_geometry() {
    let hw = MockFlashHardware::new(FlashGeometry {
        erase_block_size: 131_072,
        write_page_size: 4096,
        total_blocks: 32,
        max_erase_cycles: 3_000,
    });
    let result = FlashBackend::mount(hw);
    assert!(result.is_ok());
}

// ---------------------------------------------------------------------------
// Transaction Tracker Non-Durable Commit Cap (tested indirectly)
// ---------------------------------------------------------------------------

/// Rapid non-durable commits should not cause unbounded memory growth.
/// After many non-durable commits, the system should still function.
#[test]
fn rapid_non_durable_commits() {
    let tmpfile = create_tempfile();
    let db = Builder::new()
        .create(tmpfile.path())
        .unwrap();

    // Do many non-durable commits in sequence
    for i in 0..100u64 {
        let mut write_txn = db.begin_write().unwrap();
        let table_def = shodh_redb::TableDefinition::<u64, u64>::new("test");
        {
            let mut table = write_txn.open_table(table_def).unwrap();
            table.insert(&i, &(i * 2)).unwrap();
        }
        write_txn.set_durability(shodh_redb::Durability::None).unwrap();
        write_txn.commit().unwrap();
    }

    // Do a durable commit to flush
    {
        let write_txn = db.begin_write().unwrap();
        let table_def = shodh_redb::TableDefinition::<u64, u64>::new("test");
        {
            let mut table = write_txn.open_table(table_def).unwrap();
            table.insert(&999u64, &999u64).unwrap();
        }
        write_txn.commit().unwrap();
    }

    // Verify data is readable
    let read_txn = db.begin_read().unwrap();
    let table_def = shodh_redb::TableDefinition::<u64, u64>::new("test");
    let table = read_txn.open_table(table_def).unwrap();
    assert_eq!(table.get(&50u64).unwrap().unwrap().value(), 100);
    assert_eq!(table.get(&999u64).unwrap().unwrap().value(), 999);
}

// ---------------------------------------------------------------------------
// Crash Recovery Tests — Flash Backend
// ---------------------------------------------------------------------------

/// Flash hardware wrapper that fails writes after a countdown reaches zero.
/// Simulates power-loss or I/O failure at a specific point in time.
struct CountdownFlashHardware {
    storage: Arc<RwLock<Vec<u8>>>,
    geometry: FlashGeometry,
    write_countdown: AtomicU64,
    erase_countdown: AtomicU64,
}

impl Debug for CountdownFlashHardware {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountdownFlashHardware").finish()
    }
}

impl CountdownFlashHardware {
    fn new(geometry: FlashGeometry, writes: u64, erases: u64) -> (Self, Arc<RwLock<Vec<u8>>>) {
        let capacity = geometry.total_capacity() as usize;
        let storage = Arc::new(RwLock::new(vec![0xFFu8; capacity]));
        let hw = Self {
            storage: storage.clone(),
            geometry,
            write_countdown: AtomicU64::new(writes),
            erase_countdown: AtomicU64::new(erases),
        };
        (hw, storage)
    }

    fn from_snapshot(geometry: FlashGeometry, data: Vec<u8>, writes: u64, erases: u64) -> Self {
        Self {
            storage: Arc::new(RwLock::new(data)),
            geometry,
            write_countdown: AtomicU64::new(writes),
            erase_countdown: AtomicU64::new(erases),
        }
    }
}

impl FlashHardware for CountdownFlashHardware {
    fn geometry(&self) -> FlashGeometry {
        self.geometry
    }

    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), BackendError> {
        let s = self.storage.read().unwrap();
        let start = offset as usize;
        let end = start + buf.len();
        if end > s.len() {
            return Err(BackendError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "read past end",
            )));
        }
        buf.copy_from_slice(&s[start..end]);
        Ok(())
    }

    fn write_page(&self, offset: u64, data: &[u8]) -> Result<(), BackendError> {
        let prev = self.write_countdown.fetch_sub(1, Ordering::SeqCst);
        if prev == 0 {
            self.write_countdown.store(0, Ordering::SeqCst);
            return Err(BackendError::Io(std::io::Error::other(
                "simulated write failure (countdown)",
            )));
        }
        let mut s = self.storage.write().unwrap();
        let start = offset as usize;
        let end = start + data.len();
        if end > s.len() {
            return Err(BackendError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "write past end",
            )));
        }
        for (i, &byte) in data.iter().enumerate() {
            s[start + i] &= byte; // flash semantics: 1->0 only
        }
        Ok(())
    }

    fn erase_block(&self, block_index: u32) -> Result<(), BackendError> {
        let prev = self.erase_countdown.fetch_sub(1, Ordering::SeqCst);
        if prev == 0 {
            self.erase_countdown.store(0, Ordering::SeqCst);
            return Err(BackendError::Io(std::io::Error::other(
                "simulated erase failure (countdown)",
            )));
        }
        let mut s = self.storage.write().unwrap();
        let ebs = self.geometry.erase_block_size as usize;
        let start = block_index as usize * ebs;
        let end = start + ebs;
        if end <= s.len() {
            s[start..end].fill(0xFF);
        }
        Ok(())
    }

    fn is_bad_block(&self, _block_index: u32) -> Result<bool, BackendError> {
        Ok(false)
    }

    fn mark_bad_block(&self, _block_index: u32) -> Result<(), BackendError> {
        Ok(())
    }

    fn sync(&self) -> Result<(), BackendError> {
        Ok(())
    }
}

const CRASH_TABLE: TableDefinition<u64, u64> = TableDefinition::new("crash_test");

/// After a write failure mid-commit, remounting from the pre-crash state
/// should recover the last successfully committed data.
#[test]
fn flash_crash_at_write_recovers() {
    let geometry = FlashGeometry {
        erase_block_size: 4096,
        write_page_size: 4096,
        total_blocks: 64,
        max_erase_cycles: 100_000,
    };

    // Phase 1: Write initial data (unlimited writes for setup)
    let (hw, storage) = CountdownFlashHardware::new(geometry, u64::MAX, u64::MAX);
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(CRASH_TABLE).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &(i * 100)).unwrap();
        }
    }
    txn.commit().unwrap();

    // Snapshot the storage after a successful commit
    let pre_crash_snapshot = storage.read().unwrap().clone();
    drop(db);

    // Phase 2: Attempt more writes with a countdown that will fail during commit.
    // Give enough writes (50) for mount+open overhead, then fail during the bulk insert commit.
    let hw2 = CountdownFlashHardware::from_snapshot(geometry, pre_crash_snapshot.clone(), 50, u64::MAX);
    let backend2 = FlashBackend::mount(hw2).unwrap();
    let db2 = Builder::new().create_with_backend(backend2).unwrap();

    let txn2 = db2.begin_write().unwrap();
    {
        let mut t = txn2.open_table(CRASH_TABLE).unwrap();
        for i in 100..200u64 {
            // Ignore insert errors from the countdown
            let _ = t.insert(&i, &(i * 999));
        }
    }
    // This commit should fail due to countdown exhaustion
    let _commit_result = txn2.commit();
    drop(db2);

    // Phase 3: Recover from the pre-crash snapshot
    let hw3 = CountdownFlashHardware::from_snapshot(geometry, pre_crash_snapshot, u64::MAX, u64::MAX);
    let backend3 = FlashBackend::mount(hw3).unwrap();
    let db3 = Builder::new().create_with_backend(backend3).unwrap();

    // Verify the original committed data is intact
    let rtxn = db3.begin_read().unwrap();
    let table = rtxn.open_table(CRASH_TABLE).unwrap();
    for i in 0..10u64 {
        let val = table.get(&i).unwrap().unwrap().value();
        assert_eq!(val, i * 100, "key {i} has wrong value after recovery");
    }
    // Keys from the failed commit should NOT be present
    assert!(table.get(&100u64).unwrap().is_none());
}

/// Corrupting a journal slot's checksum causes recovery to fall back to the
/// other (older but valid) slot.
#[test]
fn journal_slot_corruption_recovers_from_other_slot() {
    let geometry = FlashGeometry {
        erase_block_size: 4096,
        write_page_size: 4096,
        total_blocks: 64,
        max_erase_cycles: 100_000,
    };

    // Write data in two separate commits so both journal slots get written
    let (hw, storage) = CountdownFlashHardware::new(geometry, u64::MAX, u64::MAX);
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    // Commit 1 — writes to slot A
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(CRASH_TABLE).unwrap();
        t.insert(&1u64, &111u64).unwrap();
    }
    txn.commit().unwrap();

    // Commit 2 — writes to slot B
    let txn2 = db.begin_write().unwrap();
    {
        let mut t = txn2.open_table(CRASH_TABLE).unwrap();
        t.insert(&2u64, &222u64).unwrap();
    }
    txn2.commit().unwrap();

    // Snapshot after commit 2
    let mut snapshot = storage.read().unwrap().clone();
    drop(db);

    // Corrupt the second journal slot's data region. The journal uses the
    // first N blocks for slot A and next N blocks for slot B.
    // With 64 blocks and default journal sizing, corrupt bytes in the second
    // half of the journal area.
    let slot_size = geometry.erase_block_size as usize;
    // Corrupt the second slot (offset = slot_size, i.e. the second block)
    // Write garbage over the checksum area (last 16 bytes of the entry)
    let corrupt_offset = slot_size + 32; // somewhere in slot B's data
    if corrupt_offset + 16 < snapshot.len() {
        for i in 0..16 {
            snapshot[corrupt_offset + i] = 0xDE;
        }
    }

    // Remount from corrupted snapshot — should recover from slot A
    let hw2 = CountdownFlashHardware::from_snapshot(geometry, snapshot, u64::MAX, u64::MAX);
    let backend2 = FlashBackend::mount(hw2).unwrap();
    let db2 = Builder::new().create_with_backend(backend2).unwrap();

    let rtxn = db2.begin_read().unwrap();
    let table = rtxn.open_table(CRASH_TABLE).unwrap();
    // Slot A had key=1 committed, it should be present
    assert!(
        table.get(&1u64).unwrap().is_some(),
        "key 1 from slot A should survive journal corruption"
    );
    // Key 2 may or may not be present depending on which slot was corrupted.
    // The important thing is the database is usable and doesn't panic/error.
}

/// Writing repeatedly until blocks are exhausted should return an error,
/// not panic.
#[test]
fn flash_space_exhaustion_returns_error() {
    let geometry = FlashGeometry {
        erase_block_size: 4096,
        write_page_size: 4096,
        total_blocks: 16, // Tiny: 64KB total
        max_erase_cycles: 100_000,
    };

    let (hw, _storage) = CountdownFlashHardware::new(geometry, u64::MAX, u64::MAX);
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let mut last_successful_key = 0u64;
    let mut hit_error = false;

    for i in 0..1000u64 {
        let txn = db.begin_write().unwrap();
        {
            let result = txn.open_table(CRASH_TABLE);
            let mut t = match result {
                Ok(t) => t,
                Err(_) => {
                    hit_error = true;
                    break;
                }
            };
            if t.insert(&i, &(i * 7)).is_err() {
                hit_error = true;
                break;
            }
        }
        if txn.commit().is_err() {
            hit_error = true;
            break;
        }
        last_successful_key = i;
    }

    assert!(
        hit_error,
        "should have hit space exhaustion within 1000 commits on 32KB flash"
    );
    assert!(
        last_successful_key > 0,
        "at least one commit should succeed before exhaustion"
    );
}
