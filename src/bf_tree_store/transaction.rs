//! Transaction bridge for Bf-Tree storage engine.
//!
//! Maps shodh-redb's transaction semantics onto Bf-Tree's concurrent model:
//!
//! - `BfTreeWriteTxn`: Multiple concurrent writers (no exclusive lock). Writes
//!   are immediately visible to all readers. `commit()` triggers a WAL flush
//!   for durability. `snapshot()` creates a crash-recovery checkpoint.
//!
//! - `BfTreeReadTxn`: Lock-free reads against the current state. No snapshot
//!   isolation -- reads always see the latest committed data.
//!
//! # Semantic Differences from Legacy Model
//!
//! | Property           | Legacy B-tree        | Bf-Tree              |
//! |--------------------|----------------------|----------------------|
//! | Writer concurrency | Single (exclusive)   | Multiple (CAS-based) |
//! | Read isolation     | Snapshot at txn start| Latest state always  |
//! | Commit cost        | Full page flush      | WAL append + flush   |
//! | Abort support      | Discard dirty pages  | No-op (writes immediate) |
//!
//! # Important
//!
//! Because Bf-Tree writes are immediate (not buffered), there is no true
//! `abort()` -- once `insert()` is called, the data is visible. For use cases
//! requiring rollback, the caller must implement application-level compensation
//! (e.g., delete the inserted keys).

use alloc::sync::Arc;

use super::adapter::BfTreeAdapter;
use super::error::BfTreeError;

/// A write handle into the Bf-Tree.
///
/// Unlike the legacy `WriteTransaction`, this does NOT acquire an exclusive lock.
/// Multiple `BfTreeWriteTxn` instances can coexist and write concurrently.
///
/// Writes are immediately visible. `commit()` ensures durability via WAL flush.
pub struct BfTreeWriteTxn {
    adapter: Arc<BfTreeAdapter>,
    /// Track number of operations for metrics/diagnostics.
    ops_count: u64,
    /// Whether commit has been called.
    committed: bool,
}

impl BfTreeWriteTxn {
    /// Create a new write transaction handle.
    ///
    /// This is cheap -- no locks acquired, no state copied.
    #[allow(dead_code)] // wired into Database layer in subsequent commit
    pub(crate) fn new(adapter: Arc<BfTreeAdapter>) -> Self {
        Self {
            adapter,
            ops_count: 0,
            committed: false,
        }
    }

    /// Insert a key-value pair. Immediately visible to all readers.
    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), BfTreeError> {
        self.adapter.insert(key, value)?;
        self.ops_count += 1;
        Ok(())
    }

    /// Delete a key. Immediately visible to all readers.
    pub fn delete(&mut self, key: &[u8]) {
        self.adapter.delete(key);
        self.ops_count += 1;
    }

    /// Read a value within this transaction context.
    pub fn read(&self, key: &[u8], out_buffer: &mut [u8]) -> Result<u32, BfTreeError> {
        self.adapter.read(key, out_buffer)
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &[u8]) -> bool {
        self.adapter.contains_key(key)
    }

    /// Commit this transaction for durability.
    ///
    /// For file-backed databases, this method takes an explicit snapshot
    /// (fsync'd to disk) before returning, ensuring all writes from this
    /// transaction are recoverable after a crash. The snapshot is necessary
    /// because the bf-tree WAL background thread flushes asynchronously on
    /// a timer; without the snapshot, a crash between commit and the next WAL
    /// flush could lose data.
    ///
    /// For in-memory databases, this is a no-op (data is not persisted).
    pub fn commit(mut self) -> Result<(), BfTreeError> {
        self.committed = true;
        // Ensure durability for non-memory backends by forcing a snapshot.
        if !self.adapter.inner().config().is_memory_backend() {
            self.adapter.snapshot()?;
        }
        Ok(())
    }

    /// Commit with an explicit durability checkpoint (snapshot).
    ///
    /// More expensive than `commit()` but guarantees all data is recoverable
    /// even without WAL replay.
    pub fn commit_with_snapshot(mut self) -> Result<std::path::PathBuf, BfTreeError> {
        self.committed = true;
        Ok(self.adapter.snapshot()?)
    }

    /// Number of insert/delete operations performed in this transaction.
    pub fn ops_count(&self) -> u64 {
        self.ops_count
    }

    /// Get a reference to the underlying adapter for advanced operations.
    pub fn adapter(&self) -> &BfTreeAdapter {
        &self.adapter
    }
}

impl Drop for BfTreeWriteTxn {
    fn drop(&mut self) {
        if !self.committed && self.ops_count > 0 {
            // Writes are already applied -- there's no rollback in Bf-Tree.
            // Log a warning in debug builds.
            #[cfg(debug_assertions)]
            {
                eprintln!(
                    "bf-tree: BfTreeWriteTxn dropped without commit ({} ops applied but not durability-flushed)",
                    self.ops_count
                );
            }
        }
    }
}

/// A read handle into the Bf-Tree.
///
/// Unlike the legacy `ReadTransaction`, this does NOT capture a snapshot.
/// Reads always see the latest state (no isolation from concurrent writers).
pub struct BfTreeReadTxn {
    adapter: Arc<BfTreeAdapter>,
}

impl BfTreeReadTxn {
    /// Create a new read transaction handle.
    #[allow(dead_code)] // wired into Database layer in subsequent commit
    pub(crate) fn new(adapter: Arc<BfTreeAdapter>) -> Self {
        Self { adapter }
    }

    /// Read the value for a key into the provided buffer.
    ///
    /// Returns the number of bytes written on success.
    pub fn read(&self, key: &[u8], out_buffer: &mut [u8]) -> Result<u32, BfTreeError> {
        self.adapter.read(key, out_buffer)
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &[u8]) -> bool {
        self.adapter.contains_key(key)
    }

    /// Scan keys starting from `start_key`, returning up to `count` entries.
    pub fn scan_from(
        &self,
        start_key: &[u8],
        count: usize,
    ) -> Result<crate::bf_tree::ScanIter<'_, '_>, BfTreeError> {
        self.adapter.scan_from(start_key, count)
    }

    /// Scan keys in range `[start_key, end_key)`.
    pub fn scan_range(
        &self,
        start_key: &[u8],
        end_key: &[u8],
    ) -> Result<crate::bf_tree::ScanIter<'_, '_>, BfTreeError> {
        self.adapter.scan_range(start_key, end_key)
    }

    /// Get a reference to the underlying adapter for advanced operations.
    pub fn adapter(&self) -> &BfTreeAdapter {
        &self.adapter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf_tree_store::config::BfTreeConfig;

    #[test]
    fn write_txn_basic() {
        let config = BfTreeConfig::new_memory(4);
        let adapter = Arc::new(BfTreeAdapter::open(config).unwrap());

        let mut txn = BfTreeWriteTxn::new(adapter.clone());
        txn.insert(b"key1", b"val1").unwrap();
        txn.insert(b"key2", b"val2").unwrap();
        assert_eq!(txn.ops_count(), 2);
        txn.commit().unwrap();

        // Read back via read txn.
        let rtxn = BfTreeReadTxn::new(adapter);
        let mut buf = [0u8; 64];
        let len = rtxn.read(b"key1", &mut buf).unwrap();
        assert_eq!(&buf[..len as usize], b"val1");
    }

    #[test]
    fn concurrent_write_txns() {
        use std::thread;

        let config = BfTreeConfig::new_memory(8);
        let adapter = Arc::new(BfTreeAdapter::open(config).unwrap());

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let adapter = adapter.clone();
                thread::spawn(move || {
                    let mut txn = BfTreeWriteTxn::new(adapter);
                    for i in 0..50 {
                        let key = format!("t{t}_k{i}");
                        let val = format!("t{t}_v{i}");
                        txn.insert(key.as_bytes(), val.as_bytes()).unwrap();
                    }
                    txn.commit().unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Verify all writes via read txn.
        let rtxn = BfTreeReadTxn::new(adapter);
        let mut buf = [0u8; 256];
        for t in 0..4 {
            for i in 0..50 {
                let key = format!("t{t}_k{i}");
                let expected = format!("t{t}_v{i}");
                let len = rtxn.read(key.as_bytes(), &mut buf).unwrap();
                assert_eq!(&buf[..len as usize], expected.as_bytes());
            }
        }
    }

    #[test]
    fn write_visible_to_concurrent_read() {
        let config = BfTreeConfig::new_memory(4);
        let adapter = Arc::new(BfTreeAdapter::open(config).unwrap());

        // Writer inserts data.
        let mut wtxn = BfTreeWriteTxn::new(adapter.clone());
        wtxn.insert(b"visible", b"yes").unwrap();

        // Reader sees it immediately (before commit).
        let rtxn = BfTreeReadTxn::new(adapter.clone());
        let mut buf = [0u8; 64];
        let len = rtxn.read(b"visible", &mut buf).unwrap();
        assert_eq!(&buf[..len as usize], b"yes");

        wtxn.commit().unwrap();
    }

    #[test]
    fn delete_visible_immediately() {
        let config = BfTreeConfig::new_memory(4);
        let adapter = Arc::new(BfTreeAdapter::open(config).unwrap());

        let mut wtxn = BfTreeWriteTxn::new(adapter.clone());
        wtxn.insert(b"gone", b"soon").unwrap();
        wtxn.delete(b"gone");

        let rtxn = BfTreeReadTxn::new(adapter);
        let mut buf = [0u8; 64];
        let result = rtxn.read(b"gone", &mut buf);
        assert!(matches!(result, Err(BfTreeError::Deleted)));

        wtxn.commit().unwrap();
    }

    #[test]
    fn scan_via_read_txn() {
        let config = BfTreeConfig::new_memory(4);
        let adapter = Arc::new(BfTreeAdapter::open(config).unwrap());

        let mut wtxn = BfTreeWriteTxn::new(adapter.clone());
        wtxn.insert(b"aaa", b"1").unwrap();
        wtxn.insert(b"bbb", b"2").unwrap();
        wtxn.insert(b"ccc", b"3").unwrap();
        wtxn.commit().unwrap();

        let rtxn = BfTreeReadTxn::new(adapter);
        let mut iter = rtxn.scan_from(b"aaa", 10).unwrap();
        let mut buf = [0u8; 256];
        let mut count = 0;
        while matches!(iter.next(&mut buf), Ok(Some(_))) {
            count += 1;
        }
        assert_eq!(count, 3);
    }
}
