//! Database observability infrastructure.
//!
//! The [`DatabaseObserver`] trait provides push-based event callbacks for
//! monitoring database operations. Implement this trait and register it via
//! [`Builder::set_observer`](crate::Builder::set_observer) to receive
//! notifications about transaction lifecycle, compaction, training, and more.
//!
//! All methods have default no-op implementations. Override only what you need.
//!
//! The [`DbMetrics`] struct (behind the `metrics` feature) provides atomic
//! counters for pull-based monitoring.

use alloc::sync::Arc;

use crate::db::CompactionProgress;
use crate::ivfpq::index::TrainProgress;

/// Information about a committed write transaction.
#[derive(Debug, Clone)]
pub struct CommitInfo {
    /// Unique transaction ID.
    pub transaction_id: u64,
    /// Number of dirty pages written in this transaction.
    pub dirty_page_count: u64,
    /// Whether two-phase commit was used.
    pub two_phase: bool,
    /// Time spent in the commit path (including fsync).
    #[cfg(feature = "std")]
    pub commit_duration: core::time::Duration,
}

/// Events emitted by the database for observability.
///
/// All methods have default no-op implementations. Override only what you need.
/// The observer is called synchronously on the committing/reading thread.
/// Implementations MUST NOT block, panic, or perform I/O that could fail.
pub trait DatabaseObserver: Send + Sync + 'static {
    /// A write transaction was committed successfully.
    fn on_write_commit(&self, _info: &CommitInfo) {}

    /// A write transaction was aborted (explicit abort or drop without commit).
    fn on_write_abort(&self, _transaction_id: u64) {}

    /// A read transaction was opened.
    fn on_read_begin(&self, _transaction_id: u64) {}

    /// A read transaction was dropped.
    fn on_read_end(&self, _transaction_id: u64) {}

    /// A page compaction step completed.
    fn on_compaction_step(&self, _progress: &CompactionProgress) {}

    /// Page compaction finished (all pages relocated).
    fn on_compaction_complete(&self, _total_pages_relocated: u64) {}

    /// IVF-PQ training progress update.
    fn on_train_progress(&self, _index_name: &str, _progress: &TrainProgress) {}

    /// A blob was written to the store.
    fn on_blob_write(&self, _blob_id: u64, _size_bytes: u64) {}

    /// A blob was deduplicated (existing content reused, bytes saved).
    fn on_blob_dedup(&self, _blob_id: u64, _saved_bytes: u64) {}

    /// A page checksum verification failed.
    fn on_checksum_failure(&self, _page_number: u64) {}
}

/// Zero-sized no-op observer. Used as the default when no observer is registered.
/// The compiler devirtualizes calls to this, making it truly zero-cost.
pub(crate) struct NoopObserver;

impl DatabaseObserver for NoopObserver {}

/// Returns the default observer (no-op).
pub(crate) fn default_observer() -> Arc<dyn DatabaseObserver> {
    Arc::new(NoopObserver)
}

/// Atomic counters for database-wide metrics.
///
/// Accessible via [`Database::metrics()`](crate::Database::metrics) when the
/// `metrics` feature is enabled. All counters are monotonically increasing
/// and use relaxed ordering for minimal overhead.
#[cfg(feature = "metrics")]
pub struct DbMetrics {
    /// Number of write transactions committed.
    pub(crate) write_txn_committed: portable_atomic::AtomicU64,
    /// Number of write transactions aborted.
    pub(crate) write_txn_aborted: portable_atomic::AtomicU64,
    /// Number of read transactions opened.
    pub(crate) read_txn_opened: portable_atomic::AtomicU64,
    /// Number of read transactions closed.
    pub(crate) read_txn_closed: portable_atomic::AtomicU64,
    /// Total pages allocated across all transactions.
    pub(crate) pages_allocated: portable_atomic::AtomicU64,
    /// Total pages freed across all transactions.
    pub(crate) pages_freed: portable_atomic::AtomicU64,
    /// Logical bytes written (user data).
    pub(crate) bytes_written_logical: portable_atomic::AtomicU64,
    /// Physical bytes written (actual I/O). Divide by logical for write amplification.
    pub(crate) bytes_written_physical: portable_atomic::AtomicU64,
    /// Number of blob write operations.
    pub(crate) blob_writes: portable_atomic::AtomicU64,
    /// Number of blob dedup hits (content already existed).
    pub(crate) blob_dedup_hits: portable_atomic::AtomicU64,
    /// Number of vector search operations.
    pub(crate) vector_searches: portable_atomic::AtomicU64,
    /// Total pages relocated by compaction.
    pub(crate) compaction_pages_relocated: portable_atomic::AtomicU64,
}

#[cfg(feature = "metrics")]
impl DbMetrics {
    pub(crate) fn new() -> Self {
        Self {
            write_txn_committed: portable_atomic::AtomicU64::new(0),
            write_txn_aborted: portable_atomic::AtomicU64::new(0),
            read_txn_opened: portable_atomic::AtomicU64::new(0),
            read_txn_closed: portable_atomic::AtomicU64::new(0),
            pages_allocated: portable_atomic::AtomicU64::new(0),
            pages_freed: portable_atomic::AtomicU64::new(0),
            bytes_written_logical: portable_atomic::AtomicU64::new(0),
            bytes_written_physical: portable_atomic::AtomicU64::new(0),
            blob_writes: portable_atomic::AtomicU64::new(0),
            blob_dedup_hits: portable_atomic::AtomicU64::new(0),
            vector_searches: portable_atomic::AtomicU64::new(0),
            compaction_pages_relocated: portable_atomic::AtomicU64::new(0),
        }
    }

    /// Number of write transactions committed.
    pub fn write_txn_committed(&self) -> u64 {
        self.write_txn_committed
            .load(portable_atomic::Ordering::Relaxed)
    }

    /// Number of write transactions aborted.
    pub fn write_txn_aborted(&self) -> u64 {
        self.write_txn_aborted
            .load(portable_atomic::Ordering::Relaxed)
    }

    /// Number of read transactions opened.
    pub fn read_txn_opened(&self) -> u64 {
        self.read_txn_opened
            .load(portable_atomic::Ordering::Relaxed)
    }

    /// Number of read transactions closed.
    pub fn read_txn_closed(&self) -> u64 {
        self.read_txn_closed
            .load(portable_atomic::Ordering::Relaxed)
    }

    /// Total pages allocated.
    pub fn pages_allocated(&self) -> u64 {
        self.pages_allocated
            .load(portable_atomic::Ordering::Relaxed)
    }

    /// Total pages freed.
    pub fn pages_freed(&self) -> u64 {
        self.pages_freed.load(portable_atomic::Ordering::Relaxed)
    }

    /// Logical bytes written (user data).
    pub fn bytes_written_logical(&self) -> u64 {
        self.bytes_written_logical
            .load(portable_atomic::Ordering::Relaxed)
    }

    /// Physical bytes written to storage.
    pub fn bytes_written_physical(&self) -> u64 {
        self.bytes_written_physical
            .load(portable_atomic::Ordering::Relaxed)
    }

    /// Number of blob write operations.
    pub fn blob_writes(&self) -> u64 {
        self.blob_writes.load(portable_atomic::Ordering::Relaxed)
    }

    /// Number of blob dedup hits.
    pub fn blob_dedup_hits(&self) -> u64 {
        self.blob_dedup_hits
            .load(portable_atomic::Ordering::Relaxed)
    }

    /// Number of vector search operations.
    pub fn vector_searches(&self) -> u64 {
        self.vector_searches
            .load(portable_atomic::Ordering::Relaxed)
    }

    /// Total pages relocated by compaction.
    pub fn compaction_pages_relocated(&self) -> u64 {
        self.compaction_pages_relocated
            .load(portable_atomic::Ordering::Relaxed)
    }
}

#[cfg(feature = "metrics")]
impl core::fmt::Debug for DbMetrics {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DbMetrics")
            .field("write_txn_committed", &self.write_txn_committed())
            .field("write_txn_aborted", &self.write_txn_aborted())
            .field("read_txn_opened", &self.read_txn_opened())
            .field("read_txn_closed", &self.read_txn_closed())
            .field("pages_allocated", &self.pages_allocated())
            .field("pages_freed", &self.pages_freed())
            .field("bytes_written_logical", &self.bytes_written_logical())
            .field("bytes_written_physical", &self.bytes_written_physical())
            .field("blob_writes", &self.blob_writes())
            .field("blob_dedup_hits", &self.blob_dedup_hits())
            .field("vector_searches", &self.vector_searches())
            .field(
                "compaction_pages_relocated",
                &self.compaction_pages_relocated(),
            )
            .finish()
    }
}
