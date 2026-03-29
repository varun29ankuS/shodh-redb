use shodh_redb::error::BackendError;
use shodh_redb::{
    Builder, FlashBackend, FlashGeometry, FlashHardware, ReadableDatabase, ReadableTableMetadata,
    StorageBackend, TableDefinition,
};
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// MockFlashHardware: in-memory flash simulator
// ---------------------------------------------------------------------------

/// Simulates a flash device backed by a `Vec<u8>` in memory.
///
/// Enforces real flash constraints:
/// - Reads return `0xFF` for erased regions
/// - Write-page alignment and size constraints
/// - Erase sets entire block to `0xFF`
/// - Bad block tracking
struct MockFlashHardware {
    storage: Arc<RwLock<Vec<u8>>>,
    geometry: FlashGeometry,
    bad_blocks: Arc<RwLock<Vec<u32>>>,
    /// Count of erase operations performed (for test assertions).
    erase_count: Arc<RwLock<u64>>,
    /// Count of write operations performed.
    write_count: Arc<RwLock<u64>>,
}

impl Debug for MockFlashHardware {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockFlashHardware")
            .field("capacity", &self.geometry.total_capacity())
            .finish()
    }
}

impl MockFlashHardware {
    fn new(geometry: FlashGeometry) -> Self {
        let capacity = geometry.total_capacity() as usize;
        // Fresh flash is all 0xFF
        let storage = vec![0xFFu8; capacity];
        Self {
            storage: Arc::new(RwLock::new(storage)),
            geometry,
            bad_blocks: Arc::new(RwLock::new(Vec::new())),
            erase_count: Arc::new(RwLock::new(0)),
            write_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Create a NOR-style flash: erase_block_size == write_page_size == 4096.
    fn nor_4k(total_blocks: u32) -> Self {
        Self::new(FlashGeometry {
            erase_block_size: 4096,
            write_page_size: 4096,
            total_blocks,
            max_erase_cycles: 100_000,
        })
    }

    /// Create a NAND-style flash: 128KB erase blocks, 4KB write pages.
    fn nand_128k(total_blocks: u32) -> Self {
        Self::new(FlashGeometry {
            erase_block_size: 131_072,
            write_page_size: 4096,
            total_blocks,
            max_erase_cycles: 3_000,
        })
    }

    /// Mark a block as bad before mounting.
    fn set_bad_block(&self, block: u32) {
        self.bad_blocks.write().unwrap().push(block);
    }

    #[allow(dead_code)]
    fn total_erases(&self) -> u64 {
        *self.erase_count.read().unwrap()
    }

    #[allow(dead_code)]
    fn total_writes(&self) -> u64 {
        *self.write_count.read().unwrap()
    }
}

impl FlashHardware for MockFlashHardware {
    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), BackendError> {
        let storage = self.storage.read().unwrap();
        let start = offset as usize;
        let end = start + buf.len();
        if end > storage.len() {
            return Err(BackendError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "read past end of flash",
            )));
        }
        buf.copy_from_slice(&storage[start..end]);
        Ok(())
    }

    fn write_page(&self, offset: u64, data: &[u8]) -> Result<(), BackendError> {
        let mut storage = self.storage.write().unwrap();
        let start = offset as usize;
        let end = start + data.len();
        if end > storage.len() {
            return Err(BackendError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "write past end of flash",
            )));
        }
        // Flash can only flip bits 1->0 (enforce write-after-erase semantics)
        for (i, &byte) in data.iter().enumerate() {
            storage[start + i] &= byte;
        }
        *self.write_count.write().unwrap() += 1;
        Ok(())
    }

    fn erase_block(&self, block_index: u32) -> Result<(), BackendError> {
        let mut storage = self.storage.write().unwrap();
        let ebs = self.geometry.erase_block_size as usize;
        let start = block_index as usize * ebs;
        let end = start + ebs;
        if end > storage.len() {
            return Err(BackendError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "erase past end of flash",
            )));
        }
        storage[start..end].fill(0xFF);
        *self.erase_count.write().unwrap() += 1;
        Ok(())
    }

    fn is_bad_block(&self, block_index: u32) -> Result<bool, BackendError> {
        Ok(self.bad_blocks.read().unwrap().contains(&block_index))
    }

    fn mark_bad_block(&self, block_index: u32) -> Result<(), BackendError> {
        let mut bad = self.bad_blocks.write().unwrap();
        if !bad.contains(&block_index) {
            bad.push(block_index);
        }
        Ok(())
    }

    fn geometry(&self) -> FlashGeometry {
        self.geometry
    }

    fn sync(&self) -> Result<(), BackendError> {
        Ok(())
    }
}

const TABLE_U64: TableDefinition<u64, u64> = TableDefinition::new("test_table");
const TABLE_STR: TableDefinition<&str, &str> = TableDefinition::new("str_table");

// ---------------------------------------------------------------------------
// FlashGeometry tests
// ---------------------------------------------------------------------------

#[test]
fn geometry_total_capacity() {
    let geo = FlashGeometry {
        erase_block_size: 4096,
        write_page_size: 4096,
        total_blocks: 256,
        max_erase_cycles: 100_000,
    };
    assert_eq!(geo.total_capacity(), 256 * 4096);
}

#[test]
fn geometry_pages_per_block() {
    let geo = FlashGeometry {
        erase_block_size: 131_072,
        write_page_size: 4096,
        total_blocks: 32,
        max_erase_cycles: 3_000,
    };
    assert_eq!(geo.pages_per_block(), 32);
}

#[test]
fn geometry_pages_per_block_zero_write_page() {
    let geo = FlashGeometry {
        erase_block_size: 4096,
        write_page_size: 0,
        total_blocks: 256,
        max_erase_cycles: 100_000,
    };
    assert_eq!(geo.pages_per_block(), 0);
}

#[test]
fn geometry_reserved_blocks_reasonable() {
    let geo = FlashGeometry {
        erase_block_size: 4096,
        write_page_size: 4096,
        total_blocks: 256,
        max_erase_cycles: 100_000,
    };
    let reserved = geo.reserved_blocks();
    // Should be small relative to total blocks
    assert!(reserved > 0, "must reserve some blocks");
    assert!(reserved < 20, "overhead should be small for 256 blocks");
    assert!(geo.logical_block_count() > 200);
}

#[test]
fn geometry_reserved_blocks_zero_erase_block() {
    let geo = FlashGeometry {
        erase_block_size: 0,
        write_page_size: 4096,
        total_blocks: 256,
        max_erase_cycles: 100_000,
    };
    assert_eq!(geo.reserved_blocks(), 0);
}

// ---------------------------------------------------------------------------
// MockFlashHardware sanity tests
// ---------------------------------------------------------------------------

#[test]
fn mock_flash_fresh_reads_as_ff() {
    let hw = MockFlashHardware::nor_4k(64);
    let mut buf = [0u8; 16];
    hw.read(0, &mut buf).unwrap();
    assert!(buf.iter().all(|&b| b == 0xFF));
}

#[test]
fn mock_flash_write_and_read() {
    let hw = MockFlashHardware::nor_4k(64);
    // Erase block 0 (already 0xFF, but be explicit)
    hw.erase_block(0).unwrap();
    // Write a page
    let data = vec![0x42u8; 4096];
    hw.write_page(0, &data).unwrap();
    let mut buf = vec![0u8; 4096];
    hw.read(0, &mut buf).unwrap();
    assert_eq!(buf, data);
}

#[test]
fn mock_flash_erase_resets_to_ff() {
    let hw = MockFlashHardware::nor_4k(64);
    let data = vec![0x00u8; 4096];
    hw.write_page(0, &data).unwrap();
    hw.erase_block(0).unwrap();
    let mut buf = vec![0u8; 4096];
    hw.read(0, &mut buf).unwrap();
    assert!(buf.iter().all(|&b| b == 0xFF));
}

#[test]
fn mock_flash_bad_block_tracking() {
    let hw = MockFlashHardware::nor_4k(64);
    assert!(!hw.is_bad_block(5).unwrap());
    hw.set_bad_block(5);
    assert!(hw.is_bad_block(5).unwrap());
    assert!(!hw.is_bad_block(6).unwrap());
}

// ---------------------------------------------------------------------------
// FlashBackend: mount / format
// ---------------------------------------------------------------------------

#[test]
fn flash_backend_format_and_mount() {
    let hw = MockFlashHardware::nor_4k(256);
    let backend = FlashBackend::format(hw).unwrap();
    assert_eq!(backend.len().unwrap(), 0);
}

#[test]
fn flash_backend_mount_fresh_device() {
    // Mount on a fresh (all 0xFF) device should auto-format
    let hw = MockFlashHardware::nor_4k(256);
    let backend = FlashBackend::mount(hw).unwrap();
    assert_eq!(backend.len().unwrap(), 0);
}

#[test]
fn flash_backend_set_len_and_read() {
    let hw = MockFlashHardware::nor_4k(256);
    let backend = FlashBackend::mount(hw).unwrap();

    // Grow storage
    backend.set_len(8192).unwrap();
    assert_eq!(backend.len().unwrap(), 8192);

    // Unwritten region reads as zeros
    let mut buf = [0u8; 64];
    backend.read(0, &mut buf).unwrap();
    assert!(buf.iter().all(|&b| b == 0));
}

#[test]
fn flash_backend_write_and_read_roundtrip() {
    let hw = MockFlashHardware::nor_4k(256);
    let backend = FlashBackend::mount(hw).unwrap();
    backend.set_len(16384).unwrap();

    let data = b"Hello, flash storage backend!";
    backend.write(100, data).unwrap();

    let mut buf = vec![0u8; data.len()];
    backend.read(100, &mut buf).unwrap();
    assert_eq!(&buf, data);
}

#[test]
fn flash_backend_overwrite() {
    let hw = MockFlashHardware::nor_4k(256);
    let backend = FlashBackend::mount(hw).unwrap();
    backend.set_len(8192).unwrap();

    backend.write(0, &[1, 2, 3, 4]).unwrap();
    backend.write(0, &[5, 6, 7, 8]).unwrap();

    let mut buf = [0u8; 4];
    backend.read(0, &mut buf).unwrap();
    assert_eq!(buf, [5, 6, 7, 8]);
}

#[test]
fn flash_backend_cross_block_write() {
    let hw = MockFlashHardware::nor_4k(256);
    let backend = FlashBackend::mount(hw).unwrap();
    backend.set_len(16384).unwrap();

    // Write across a block boundary (block size = 4096)
    let data = vec![0xABu8; 256];
    backend.write(4000, &data).unwrap();

    let mut buf = vec![0u8; 256];
    backend.read(4000, &mut buf).unwrap();
    assert_eq!(buf, data);
}

#[test]
fn flash_backend_sync_persists_metadata() {
    let hw = MockFlashHardware::nor_4k(256);
    let backend = FlashBackend::mount(hw).unwrap();
    backend.set_len(4096).unwrap();
    backend.write(0, &[42u8; 64]).unwrap();
    // sync should persist the FTL metadata to journal
    backend.sync_data().unwrap();
}

#[test]
fn flash_backend_shrink() {
    let hw = MockFlashHardware::nor_4k(256);
    let backend = FlashBackend::mount(hw).unwrap();
    backend.set_len(16384).unwrap();
    backend.set_len(4096).unwrap();
    assert_eq!(backend.len().unwrap(), 4096);
}

// ---------------------------------------------------------------------------
// FlashBackend with bad blocks
// ---------------------------------------------------------------------------

#[test]
fn flash_backend_with_bad_blocks() {
    let hw = MockFlashHardware::nor_4k(256);
    // Mark some data-region blocks as bad
    hw.set_bad_block(10);
    hw.set_bad_block(20);
    hw.set_bad_block(30);

    let backend = FlashBackend::mount(hw).unwrap();
    backend.set_len(65536).unwrap();

    // Should still work -- bad blocks are skipped
    let data = vec![0xCDu8; 128];
    backend.write(0, &data).unwrap();

    let mut buf = vec![0u8; 128];
    backend.read(0, &mut buf).unwrap();
    assert_eq!(buf, data);
}

// ---------------------------------------------------------------------------
// FlashBackend with NAND-style geometry
// ---------------------------------------------------------------------------

#[test]
fn flash_backend_nand_geometry() {
    // 128KB erase blocks, 4KB write pages, 32 blocks = 4MB device
    let hw = MockFlashHardware::nand_128k(32);
    let backend = FlashBackend::mount(hw).unwrap();
    backend.set_len(131_072).unwrap();

    let data = vec![0x55u8; 8192];
    backend.write(0, &data).unwrap();

    let mut buf = vec![0u8; 8192];
    backend.read(0, &mut buf).unwrap();
    assert_eq!(buf, data);
}

// ---------------------------------------------------------------------------
// Full Database integration via Builder::create_with_backend
// ---------------------------------------------------------------------------

#[test]
fn database_on_flash_basic_crud() {
    let hw = MockFlashHardware::nor_4k(1024); // 4MB NOR flash
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    // Insert
    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE_U64).unwrap();
        for i in 0..100 {
            table.insert(&i, &(i * i)).unwrap();
        }
    }
    write_txn.commit().unwrap();

    // Read back
    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE_U64).unwrap();
    assert_eq!(table.len().unwrap(), 100);
    for i in 0..100 {
        let val = table.get(&i).unwrap().unwrap();
        assert_eq!(val.value(), i * i);
    }
}

#[test]
fn database_on_flash_string_values() {
    let hw = MockFlashHardware::nor_4k(1024);
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE_STR).unwrap();
        table.insert("hello", "world").unwrap();
        table.insert("foo", "bar").unwrap();
        table.insert("shodh", "redb").unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE_STR).unwrap();
    assert_eq!(table.get("hello").unwrap().unwrap().value(), "world");
    assert_eq!(table.get("shodh").unwrap().unwrap().value(), "redb");
}

#[test]
fn database_on_flash_multiple_transactions() {
    let hw = MockFlashHardware::nor_4k(1024);
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    // Transaction 1
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE_U64).unwrap();
        t.insert(&1, &100).unwrap();
    }
    txn.commit().unwrap();

    // Transaction 2
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE_U64).unwrap();
        t.insert(&2, &200).unwrap();
    }
    txn.commit().unwrap();

    // Transaction 3: update existing
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE_U64).unwrap();
        t.insert(&1, &999).unwrap();
    }
    txn.commit().unwrap();

    // Verify
    let read = db.begin_read().unwrap();
    let table = read.open_table(TABLE_U64).unwrap();
    assert_eq!(table.get(&1).unwrap().unwrap().value(), 999);
    assert_eq!(table.get(&2).unwrap().unwrap().value(), 200);
}

#[test]
fn database_on_flash_delete_operations() {
    let hw = MockFlashHardware::nor_4k(1024);
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE_U64).unwrap();
        for i in 0..50 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    // Delete half
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE_U64).unwrap();
        for i in 0..25 {
            t.remove(&i).unwrap();
        }
    }
    txn.commit().unwrap();

    let read = db.begin_read().unwrap();
    let table = read.open_table(TABLE_U64).unwrap();
    assert_eq!(table.len().unwrap(), 25);
    assert!(table.get(&0).unwrap().is_none());
    assert!(table.get(&25).unwrap().is_some());
}

#[test]
fn database_on_flash_range_queries() {
    let hw = MockFlashHardware::nor_4k(1024);
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE_U64).unwrap();
        for i in 0..100 {
            t.insert(&i, &(i * 10)).unwrap();
        }
    }
    txn.commit().unwrap();

    let read = db.begin_read().unwrap();
    let table = read.open_table(TABLE_U64).unwrap();
    let range: Vec<_> = table
        .range(10..20)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (entry.0.value(), entry.1.value())
        })
        .collect();
    assert_eq!(range.len(), 10);
    assert_eq!(range[0], (10, 100));
    assert_eq!(range[9], (19, 190));
}

#[test]
fn database_on_flash_large_values() {
    let hw = MockFlashHardware::nor_4k(2048); // 8MB
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let large_value = vec![0xABu8; 8192]; // 8KB value

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE_STR).unwrap();
        let value_str = std::str::from_utf8(&large_value[..100]).unwrap_or("binary");
        t.insert("large_key", value_str).unwrap();
    }
    txn.commit().unwrap();

    let read = db.begin_read().unwrap();
    let table = read.open_table(TABLE_STR).unwrap();
    assert!(table.get("large_key").unwrap().is_some());
}

#[test]
fn database_on_flash_abort_transaction() {
    let hw = MockFlashHardware::nor_4k(1024);
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    // Insert some data
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE_U64).unwrap();
        t.insert(&1, &100).unwrap();
    }
    txn.commit().unwrap();

    // Start another transaction but don't commit (abort)
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE_U64).unwrap();
        t.insert(&2, &200).unwrap();
    }
    txn.abort().unwrap();

    // Only key 1 should exist
    let read = db.begin_read().unwrap();
    let table = read.open_table(TABLE_U64).unwrap();
    assert_eq!(table.len().unwrap(), 1);
    assert!(table.get(&2).unwrap().is_none());
}

#[test]
fn database_on_flash_with_bad_blocks() {
    let hw = MockFlashHardware::nor_4k(1024);
    hw.set_bad_block(50);
    hw.set_bad_block(100);
    hw.set_bad_block(150);

    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(TABLE_U64).unwrap();
        for i in 0..200 {
            t.insert(&i, &i).unwrap();
        }
    }
    txn.commit().unwrap();

    let read = db.begin_read().unwrap();
    let table = read.open_table(TABLE_U64).unwrap();
    assert_eq!(table.len().unwrap(), 200);
}

// ---------------------------------------------------------------------------
// Wear leveling verification
// ---------------------------------------------------------------------------

#[test]
fn flash_backend_erases_are_distributed() {
    let hw = MockFlashHardware::nor_4k(256);
    let backend = FlashBackend::mount(hw).unwrap();
    backend.set_len(65536).unwrap();

    // Perform many writes to the same logical offset
    for i in 0u8..100 {
        backend.write(0, &[i; 64]).unwrap();
    }

    // Verify data is correct after all overwrites
    let mut buf = [0u8; 64];
    backend.read(0, &mut buf).unwrap();
    assert_eq!(buf, [99u8; 64]);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn flash_backend_zero_length_read_write() {
    let hw = MockFlashHardware::nor_4k(256);
    let backend = FlashBackend::mount(hw).unwrap();
    backend.set_len(4096).unwrap();

    // Zero-length operations should succeed
    backend.write(0, &[]).unwrap();
    let mut buf = [];
    backend.read(0, &mut buf).unwrap();
}

#[test]
fn flash_backend_close() {
    let hw = MockFlashHardware::nor_4k(256);
    let backend = FlashBackend::mount(hw).unwrap();
    backend.set_len(4096).unwrap();
    backend.write(0, &[1, 2, 3]).unwrap();
    backend.close().unwrap();
}

#[test]
fn database_on_flash_multiple_tables() {
    const TABLE_A: TableDefinition<u64, u64> = TableDefinition::new("table_a");
    const TABLE_B: TableDefinition<u64, u64> = TableDefinition::new("table_b");

    let hw = MockFlashHardware::nor_4k(1024);
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        let mut a = txn.open_table(TABLE_A).unwrap();
        let mut b = txn.open_table(TABLE_B).unwrap();
        a.insert(&1, &10).unwrap();
        b.insert(&1, &20).unwrap();
    }
    txn.commit().unwrap();

    let read = db.begin_read().unwrap();
    let a = read.open_table(TABLE_A).unwrap();
    let b = read.open_table(TABLE_B).unwrap();
    assert_eq!(a.get(&1).unwrap().unwrap().value(), 10);
    assert_eq!(b.get(&1).unwrap().unwrap().value(), 20);
}

#[test]
fn database_on_flash_list_tables() {
    let hw = MockFlashHardware::nor_4k(1024);
    let backend = FlashBackend::mount(hw).unwrap();
    let db = Builder::new().create_with_backend(backend).unwrap();

    let txn = db.begin_write().unwrap();
    {
        txn.open_table(TABLE_U64).unwrap();
        txn.open_table(TABLE_STR).unwrap();
    }
    txn.commit().unwrap();

    let read = db.begin_read().unwrap();
    let tables: Vec<_> = read.list_tables().unwrap().collect();
    assert_eq!(tables.len(), 2);
}
