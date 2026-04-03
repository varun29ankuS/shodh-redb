//! Core adapter bridging bf-tree's concurrent B+tree to shodh-redb's storage layer.

use std::sync::Arc;

use bf_tree::{BfTree, LeafInsertResult, LeafReadResult, ScanIter, ScanReturnField};

use super::config::BfTreeConfig;
use super::error::BfTreeError;

/// Thread-safe concurrent storage adapter backed by Bf-Tree.
///
/// Unlike the legacy single-writer B-tree, `BfTreeAdapter` allows multiple threads
/// to concurrently insert, read, and delete keys without external synchronization.
///
/// # Concurrency Model
///
/// - **Reads**: Lock-free via optimistic versioned locks on inner nodes. Multiple
///   readers proceed concurrently without blocking.
/// - **Writes**: Use CAS-based state machine on leaf mini-pages. Multiple writers
///   proceed concurrently; contention is resolved via retry (backoff).
/// - **Scans**: Snapshot-consistent range iteration.
///
/// # Durability
///
/// When WAL is enabled, writes are durable after the background flush thread
/// persists the log. Call [`snapshot`](Self::snapshot) for explicit durability
/// checkpoints with crash-recovery support.
pub struct BfTreeAdapter {
    inner: BfTree,
}

impl BfTreeAdapter {
    /// Create a new Bf-Tree storage engine with the given configuration.
    pub fn open(config: BfTreeConfig) -> Result<Self, BfTreeError> {
        let bf_config = config.into_bf_config()?;
        let inner = BfTree::with_config(bf_config, None)?;
        Ok(Self { inner })
    }

    /// Open a Bf-Tree from an existing snapshot file for crash recovery.
    pub fn open_from_snapshot(config: BfTreeConfig) -> Result<Self, BfTreeError> {
        let bf_config = config.into_bf_config()?;
        let inner = BfTree::new_from_snapshot(bf_config, None)?;
        Ok(Self { inner })
    }

    /// Insert or update a key-value pair.
    ///
    /// This operation is thread-safe and can be called concurrently from
    /// multiple threads. Internally uses CAS-based mini-page operations.
    ///
    /// Returns `Ok(())` on success, or `Err` if the key/value exceeds size limits.
    pub fn insert(&self, key: &[u8], value: &[u8]) -> Result<(), BfTreeError> {
        match self.inner.insert(key, value) {
            LeafInsertResult::Success => Ok(()),
            LeafInsertResult::InvalidKV(msg) => Err(BfTreeError::InvalidKV(msg)),
        }
    }

    /// Read the value for a key into the provided buffer.
    ///
    /// Returns the number of bytes written to `out_buffer` on success.
    /// The buffer must be large enough to hold the value.
    ///
    /// This operation is lock-free and can be called concurrently.
    pub fn read(&self, key: &[u8], out_buffer: &mut [u8]) -> Result<u32, BfTreeError> {
        match self.inner.read(key, out_buffer) {
            LeafReadResult::Found(len) => Ok(len),
            LeafReadResult::NotFound => Err(BfTreeError::NotFound),
            LeafReadResult::Deleted => Err(BfTreeError::Deleted),
            LeafReadResult::InvalidKey => Err(BfTreeError::InvalidKey),
        }
    }

    /// Delete a key from the index.
    ///
    /// Inserts a tombstone marker. The space is reclaimed during eviction.
    pub fn delete(&self, key: &[u8]) {
        self.inner.delete(key);
    }

    /// Check if a key exists (without reading the value).
    ///
    /// Allocates a temporary buffer sized to the configured max record size.
    /// For hot-path existence checks, prefer caching the result.
    pub fn contains_key(&self, key: &[u8]) -> bool {
        let max_val = self.inner.config().get_cb_max_record_size();
        let mut buf = vec![0u8; max_val];
        matches!(self.inner.read(key, &mut buf), LeafReadResult::Found(_))
    }

    /// Scan keys starting from `start_key`, returning up to `count` entries.
    ///
    /// Returns an iterator that yields `(key_len, value_len)` pairs.
    /// The caller provides a buffer to `ScanIter::next()` to receive key+value data.
    pub fn scan_from(
        &self,
        start_key: &[u8],
        count: usize,
    ) -> Result<ScanIter<'_, '_>, BfTreeError> {
        self.inner
            .scan_with_count(start_key, count, ScanReturnField::KeyAndValue)
            .map_err(BfTreeError::from)
    }

    /// Scan keys in range `[start_key, end_key)`.
    pub fn scan_range(
        &self,
        start_key: &[u8],
        end_key: &[u8],
    ) -> Result<ScanIter<'_, '_>, BfTreeError> {
        self.inner
            .scan_with_end_key(start_key, end_key, ScanReturnField::KeyAndValue)
            .map_err(BfTreeError::from)
    }

    /// Take a durability checkpoint. All data written before this call is
    /// guaranteed to be recoverable after a crash.
    ///
    /// This is a stop-the-world operation that flushes the circular buffer
    /// to disk and writes snapshot metadata.
    pub fn snapshot(&self) -> std::path::PathBuf {
        self.inner.snapshot()
    }

    /// Get a reference to the underlying Bf-Tree for advanced operations.
    pub fn inner(&self) -> &BfTree {
        &self.inner
    }

    /// Get the current circular buffer utilization metrics.
    pub fn buffer_metrics(&self) -> bf_tree::circular_buffer::CircularBufferMetrics {
        self.inner.get_buffer_metrics()
    }
}

/// Wrap the adapter in an Arc for sharing across threads.
impl BfTreeAdapter {
    /// Create a thread-safe shared handle.
    pub fn into_shared(self) -> Arc<Self> {
        Arc::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_read() {
        let config = BfTreeConfig::new_memory(4);
        let adapter = BfTreeAdapter::open(config).unwrap();

        adapter.insert(b"hello", b"world").unwrap();

        let mut buf = [0u8; 64];
        let len = adapter.read(b"hello", &mut buf).unwrap();
        assert_eq!(&buf[..len as usize], b"world");
    }

    #[test]
    fn read_not_found() {
        let config = BfTreeConfig::new_memory(4);
        let adapter = BfTreeAdapter::open(config).unwrap();

        let mut buf = [0u8; 64];
        let result = adapter.read(b"missing", &mut buf);
        assert!(matches!(result, Err(BfTreeError::NotFound)));
    }

    #[test]
    fn insert_delete_read() {
        let config = BfTreeConfig::new_memory(4);
        let adapter = BfTreeAdapter::open(config).unwrap();

        adapter.insert(b"key1", b"val1").unwrap();
        adapter.delete(b"key1");

        let mut buf = [0u8; 64];
        let result = adapter.read(b"key1", &mut buf);
        assert!(matches!(result, Err(BfTreeError::Deleted)));
    }

    #[test]
    fn contains_key() {
        let config = BfTreeConfig::new_memory(4);
        let adapter = BfTreeAdapter::open(config).unwrap();

        assert!(!adapter.contains_key(b"key"));
        adapter.insert(b"key", b"val").unwrap();
        assert!(adapter.contains_key(b"key"));
    }

    #[test]
    fn concurrent_writes() {
        use std::sync::Arc;
        use std::thread;

        let config = BfTreeConfig::new_memory(8);
        let adapter = Arc::new(BfTreeAdapter::open(config).unwrap());

        let num_threads = 4;
        let writes_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let adapter = adapter.clone();
                thread::spawn(move || {
                    for i in 0..writes_per_thread {
                        let key = format!("t{t}_k{i}");
                        let val = format!("t{t}_v{i}");
                        adapter.insert(key.as_bytes(), val.as_bytes()).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Verify all writes are visible.
        let mut buf = [0u8; 256];
        for t in 0..num_threads {
            for i in 0..writes_per_thread {
                let key = format!("t{t}_k{i}");
                let expected = format!("t{t}_v{i}");
                let len = adapter.read(key.as_bytes(), &mut buf).unwrap();
                assert_eq!(&buf[..len as usize], expected.as_bytes());
            }
        }
    }

    #[test]
    fn scan_basic() {
        let config = BfTreeConfig::new_memory(4);
        let adapter = BfTreeAdapter::open(config).unwrap();

        adapter.insert(b"aaa", b"1").unwrap();
        adapter.insert(b"bbb", b"2").unwrap();
        adapter.insert(b"ccc", b"3").unwrap();

        let mut iter = adapter.scan_from(b"aaa", 10).unwrap();
        let mut buf = [0u8; 256];
        let mut count = 0;
        while iter.next(&mut buf).is_some() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn overwrite_value() {
        let config = BfTreeConfig::new_memory(4);
        let adapter = BfTreeAdapter::open(config).unwrap();

        adapter.insert(b"key", b"v1").unwrap();
        adapter.insert(b"key", b"v2_longer").unwrap();

        let mut buf = [0u8; 64];
        let len = adapter.read(b"key", &mut buf).unwrap();
        assert_eq!(&buf[..len as usize], b"v2_longer");
    }
}
