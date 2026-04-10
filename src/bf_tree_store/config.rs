//! Configuration bridge between shodh-redb and bf-tree.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::bf_tree::{Config, StorageBackend, WalConfig};

use super::BfTreeError;
use super::verification::VerifyMode;

/// Configuration for the Bf-Tree storage engine.
///
/// Wraps `crate::bf_tree::Config` with shodh-redb-specific defaults and validation.
pub struct BfTreeConfig {
    /// Path to the data file. Use `:memory:` for in-memory mode.
    pub file_path: PathBuf,
    /// Size of the circular buffer in bytes. Must be a power of two.
    /// Default: 32 MiB. For embedded targets, consider 4-8 MiB.
    pub circular_buffer_size: usize,
    /// Maximum key length in bytes (including table prefix overhead). Default: 256.
    /// Must not exceed 2020 (bf-tree hard limit).
    /// Note: table namespace encoding adds `2 + table_name.len()` bytes to each key.
    pub max_key_len: usize,
    /// Minimum record size (key + value) in bytes. Default: 4.
    pub min_record_size: usize,
    /// Maximum record size (key + value) in bytes. Default: 1568.
    pub max_record_size: usize,
    /// Leaf page size in bytes. Default: 4096. Must be multiple of 4096 for disk mode.
    pub leaf_page_size: usize,
    /// Enable write-ahead log for durability. Default: true.
    pub enable_wal: bool,
    /// WAL flush interval in milliseconds. Default: 1.
    pub wal_flush_interval_ms: u64,
    /// Storage backend selection.
    pub backend: BfTreeBackend,
    /// Per-entry checksum verification mode.
    ///
    /// When set to `Full` or `Sampled`, values are wrapped with a 4-byte
    /// FNV-1a checksum on write and verified on read. This adds 4 bytes of
    /// overhead per value and a small CPU cost per read. Default: `None`.
    pub verify_mode: VerifyMode,
    /// Number of commits between automatic snapshots. Default: 100.
    ///
    /// Snapshots are expensive (full circular buffer drain + fsync) but speed
    /// up crash recovery by bounding WAL replay length. Between snapshots,
    /// durability is provided by the WAL (`append_and_wait` blocks until
    /// fsync). Set to 0 to disable automatic snapshots entirely (you must
    /// call `BfTreeDatabase::snapshot()` manually, or accept longer recovery).
    pub snapshot_interval: u64,
    /// Durability mode controlling when WAL data is fsynced. Default: `Sync`.
    pub durability: DurabilityMode,
    /// Maximum cumulative bytes (key + value) a single write transaction may
    /// write before further inserts are rejected with `BfTreeError::TransactionTooLarge`.
    /// `None` means no limit. Default: `None`.
    pub max_transaction_bytes: Option<usize>,
}

/// Controls when WAL data is fsynced to disk.
///
/// Determines the durability-performance trade-off for committed data.
/// All modes still write WAL entries for crash recovery -- the difference
/// is when those entries are fsynced to stable storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityMode {
    /// Every commit fsyncs the WAL. No data loss on crash. (Default)
    Sync,
    /// WAL writes go to OS page cache. The WAL background thread fsyncs
    /// every `wal_flush_interval_ms`. Up to one flush interval of committed
    /// data may be lost on crash. Matches `RocksDB`'s default behavior.
    Periodic,
    /// No fsync at all. For benchmarks and ephemeral workloads only.
    /// All committed data is lost on crash.
    NoSync,
}

/// Storage backend for Bf-Tree.
#[derive(Debug, Clone)]
pub enum BfTreeBackend {
    /// In-memory storage (no persistence).
    Memory,
    /// Standard file I/O (cross-platform).
    Std,
    /// Linux `io_uring` in blocking mode (Linux only, better throughput).
    #[cfg(target_os = "linux")]
    IoUringBlocking,
    /// Linux `io_uring` in polling mode (Linux only, lowest latency).
    #[cfg(target_os = "linux")]
    IoUringPolling,
}

impl Default for BfTreeConfig {
    fn default() -> Self {
        Self {
            file_path: PathBuf::from(":memory:"),
            circular_buffer_size: 32 * 1024 * 1024, // 32 MiB
            max_key_len: 256,
            min_record_size: 4,
            max_record_size: 1568,
            leaf_page_size: 4096,
            enable_wal: true,
            wal_flush_interval_ms: 1,
            backend: BfTreeBackend::Memory,
            verify_mode: VerifyMode::None,
            snapshot_interval: 100,
            durability: DurabilityMode::Sync,
            max_transaction_bytes: None,
        }
    }
}

impl BfTreeBackend {
    /// Returns `true` if this backend persists data to a file.
    fn is_file_backed(&self) -> bool {
        match self {
            BfTreeBackend::Memory => false,
            BfTreeBackend::Std => true,
            #[cfg(target_os = "linux")]
            BfTreeBackend::IoUringBlocking => true,
            #[cfg(target_os = "linux")]
            BfTreeBackend::IoUringPolling => true,
        }
    }
}

impl BfTreeConfig {
    /// Create a configuration for file-backed storage.
    pub fn new_file(path: impl AsRef<Path>, buffer_size_mib: usize) -> Self {
        Self {
            file_path: path.as_ref().to_path_buf(),
            circular_buffer_size: buffer_size_mib * 1024 * 1024,
            backend: BfTreeBackend::Std,
            ..Self::default()
        }
    }

    /// Create a configuration for in-memory storage.
    pub fn new_memory(buffer_size_mib: usize) -> Self {
        Self {
            file_path: PathBuf::from(":memory:"),
            circular_buffer_size: buffer_size_mib * 1024 * 1024,
            enable_wal: false,
            backend: BfTreeBackend::Memory,
            ..Self::default()
        }
    }

    /// Create a configuration optimized for embedded targets.
    /// Uses smaller buffer (4 MiB) and conservative record sizes.
    pub fn new_embedded(path: impl AsRef<Path>) -> Self {
        Self {
            file_path: path.as_ref().to_path_buf(),
            circular_buffer_size: 4 * 1024 * 1024, // 4 MiB
            max_key_len: 16,
            min_record_size: 4,
            max_record_size: 512,
            leaf_page_size: 4096,
            enable_wal: true,
            wal_flush_interval_ms: 10, // less frequent flush for embedded
            backend: BfTreeBackend::Std,
            verify_mode: VerifyMode::None,
            snapshot_interval: 100,
            durability: DurabilityMode::Sync,
            max_transaction_bytes: None,
        }
    }

    /// Convert to bf-tree's native `Config`.
    ///
    /// Returns `Err(BfTreeError::InvalidConfig)` if WAL is disabled on a
    /// file-backed backend. Without WAL, committed data is only durable after
    /// an explicit `snapshot()` call; any crash between `commit()` and
    /// `snapshot()` silently loses data. This configuration is rejected to
    /// prevent silent data loss.
    pub(crate) fn into_bf_config(self) -> Result<Config, BfTreeError> {
        if self.backend.is_file_backed() && !self.enable_wal {
            return Err(BfTreeError::InvalidConfig(String::from(
                "WAL must be enabled for file-backed storage backends; \
                 disabling WAL on a file backend causes silent data loss on crash. \
                 Use enable_wal: true, or switch to BfTreeBackend::Memory for \
                 non-durable in-memory usage",
            )));
        }

        let mut config = Config::new(&self.file_path, self.circular_buffer_size);

        let storage_backend = match self.backend {
            BfTreeBackend::Memory => StorageBackend::Memory,
            BfTreeBackend::Std => StorageBackend::Std,
            #[cfg(target_os = "linux")]
            BfTreeBackend::IoUringBlocking => StorageBackend::IoUringBlocking,
            #[cfg(target_os = "linux")]
            BfTreeBackend::IoUringPolling => StorageBackend::IoUringPolling,
        };
        config.storage_backend(storage_backend);
        config.cb_max_key_len(self.max_key_len);
        config.cb_min_record_size(self.min_record_size);
        config.cb_max_record_size(self.max_record_size);
        config.leaf_page_size(self.leaf_page_size);

        if self.enable_wal {
            let wal_path = if self.file_path.to_str() == Some(":memory:") {
                PathBuf::from(":memory:")
            } else {
                let parent = self.file_path.parent().unwrap_or(Path::new("."));
                let stem = self
                    .file_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_str()
                    .unwrap_or("data");
                parent.join(format!("{stem}.wal"))
            };
            let mut wal_config = WalConfig::new(wal_path);
            wal_config.flush_interval(std::time::Duration::from_millis(self.wal_flush_interval_ms));
            config.enable_write_ahead_log(Arc::new(wal_config));
        }

        config.validate()?;
        Ok(config)
    }
}
