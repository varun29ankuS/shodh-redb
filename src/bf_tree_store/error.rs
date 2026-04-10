//! Error types for the Bf-Tree storage bridge.

use alloc::string::String;
use core::fmt;

/// Errors that can occur during Bf-Tree operations.
#[derive(Debug)]
pub enum BfTreeError {
    /// The key or value exceeds Bf-Tree's maximum size limits.
    /// Max key: 2020 bytes, max value: 16332 bytes.
    InvalidKV(String),
    /// The requested key was not found.
    NotFound,
    /// The key was found but has been marked as deleted (tombstone).
    Deleted,
    /// The key exceeds the configured maximum key length.
    InvalidKey,
    /// Configuration validation failed.
    Config(crate::bf_tree::ConfigError),
    /// Scan operation failed.
    Scan(crate::bf_tree::ScanIterError),
    /// Data corruption detected (e.g., missing blob chunks).
    Corruption(String),
    /// Invalid operation (e.g., write after finish).
    InvalidOperation(String),
    /// The table name uses a reserved prefix (`"__"`) that is reserved for
    /// internal system tables (CDC log, blob metadata, etc.).
    ReservedTableName(String),
    /// Invalid configuration (e.g., WAL disabled on a file-backed backend).
    InvalidConfig(String),
    /// A write transaction exceeded its configured `max_transaction_bytes` limit.
    TransactionTooLarge {
        /// Cumulative bytes already written in this transaction.
        written: usize,
        /// The configured byte limit.
        limit: usize,
    },
    /// A flush partially failed and the compensating rollback also failed.
    ///
    /// The database may be in an inconsistent state: some entries from the
    /// write buffer were applied but could not be undone. The first field
    /// describes the original flush error; the second describes which
    /// rollback operations failed (number of failures and last error seen).
    PartialFlushRollbackFailed {
        /// Description of the original error that caused the flush to abort.
        flush_error: String,
        /// Number of compensating operations that failed during rollback.
        rollback_failures: usize,
        /// Description of the last rollback error encountered.
        last_rollback_error: String,
    },
}

impl fmt::Display for BfTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKV(msg) => write!(f, "invalid key/value: {msg}"),
            Self::NotFound => write!(f, "key not found"),
            Self::Deleted => write!(f, "key deleted"),
            Self::InvalidKey => write!(f, "invalid key"),
            Self::Config(e) => write!(f, "bf-tree config error: {e:?}"),
            Self::Scan(e) => write!(f, "bf-tree scan error: {e:?}"),
            Self::Corruption(msg) => write!(f, "data corruption: {msg}"),
            Self::InvalidOperation(msg) => write!(f, "invalid operation: {msg}"),
            Self::ReservedTableName(name) => write!(
                f,
                "table name \"{name}\" uses the reserved \"__\" prefix; \
                 names starting with \"__\" are reserved for internal system tables"
            ),
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            Self::TransactionTooLarge { written, limit } => write!(
                f,
                "transaction size limit exceeded: {written} bytes written, limit is {limit} bytes"
            ),
            Self::PartialFlushRollbackFailed {
                flush_error,
                rollback_failures,
                last_rollback_error,
            } => write!(
                f,
                "partial flush rollback failed: original error: {flush_error}; \
                 rollback had {rollback_failures} failure(s), last: {last_rollback_error}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BfTreeError {}

impl From<crate::bf_tree::ConfigError> for BfTreeError {
    fn from(e: crate::bf_tree::ConfigError) -> Self {
        Self::Config(e)
    }
}

impl From<crate::bf_tree::BfTreeError> for BfTreeError {
    fn from(e: crate::bf_tree::BfTreeError) -> Self {
        match e {
            crate::bf_tree::BfTreeError::Config(ce) => Self::Config(ce),
            crate::bf_tree::BfTreeError::Io(io) => {
                Self::Corruption(alloc::format!("bf-tree I/O error: {io}"))
            }
        }
    }
}

impl From<crate::bf_tree::ScanIterError> for BfTreeError {
    fn from(e: crate::bf_tree::ScanIterError) -> Self {
        Self::Scan(e)
    }
}

#[cfg(feature = "std")]
impl From<BfTreeError> for crate::StorageError {
    fn from(e: BfTreeError) -> Self {
        match e {
            // NotFound/Deleted should only escape the bf_tree_store layer when
            // internal lookups fail unexpectedly (e.g., missing blob metadata or
            // snapshot entries that should exist). This indicates data inconsistency.
            BfTreeError::NotFound => crate::StorageError::Corrupted(alloc::string::String::from(
                "bf-tree: unexpected missing entry (internal lookup failed)",
            )),
            BfTreeError::Deleted => crate::StorageError::Corrupted(alloc::string::String::from(
                "bf-tree: unexpected tombstone encountered (internal lookup failed)",
            )),
            // InvalidKV covers both value-too-large and other validation errors
            // (empty key, CDC serialization). Map to Corrupted with the original
            // message to preserve diagnostic context.
            BfTreeError::InvalidKV(msg) => {
                crate::StorageError::Corrupted(alloc::format!("bf-tree: invalid key/value: {msg}"))
            }
            // InvalidKey from LeafReadResult means the B-tree node contains a key
            // that violates size constraints -- this is data-level inconsistency.
            BfTreeError::InvalidKey => crate::StorageError::Corrupted(alloc::string::String::from(
                "bf-tree: key exceeds maximum length (data inconsistency)",
            )),
            // Config errors are startup/initialization failures, not data corruption.
            BfTreeError::Config(e) => {
                crate::StorageError::Io(crate::BackendError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    alloc::format!("bf-tree config error: {e:?}"),
                )))
            }
            // Scan errors originate from the storage/IO layer during iteration.
            BfTreeError::Scan(e) => crate::StorageError::Io(crate::BackendError::Io(
                std::io::Error::other(alloc::format!("bf-tree scan error: {e:?}")),
            )),
            BfTreeError::Corruption(msg) => crate::StorageError::Corrupted(msg),
            BfTreeError::InvalidOperation(msg) => {
                crate::StorageError::Corrupted(alloc::format!("invalid operation: {msg}"))
            }
            BfTreeError::ReservedTableName(name) => {
                crate::StorageError::Io(crate::BackendError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    alloc::format!("table name \"{name}\" uses the reserved \"__\" prefix"),
                )))
            }
            BfTreeError::InvalidConfig(msg) => {
                crate::StorageError::Io(crate::BackendError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    alloc::format!("bf-tree invalid config: {msg}"),
                )))
            }
            BfTreeError::TransactionTooLarge { written, limit } => {
                crate::StorageError::Io(crate::BackendError::Io(std::io::Error::other(
                    alloc::format!(
                        "bf-tree transaction size limit exceeded: {written} bytes written, limit {limit}"
                    ),
                )))
            }
            BfTreeError::PartialFlushRollbackFailed {
                flush_error,
                rollback_failures,
                last_rollback_error,
            } => crate::StorageError::Corrupted(alloc::format!(
                "bf-tree: partial flush with failed rollback ({rollback_failures} \
                 rollback failure(s)): original: {flush_error}; last rollback error: \
                 {last_rollback_error}"
            )),
        }
    }
}

#[cfg(not(feature = "std"))]
impl From<BfTreeError> for crate::StorageError {
    fn from(e: BfTreeError) -> Self {
        match e {
            BfTreeError::NotFound => crate::StorageError::Corrupted(alloc::string::String::from(
                "bf-tree: unexpected missing entry (internal lookup failed)",
            )),
            BfTreeError::Deleted => crate::StorageError::Corrupted(alloc::string::String::from(
                "bf-tree: unexpected tombstone encountered (internal lookup failed)",
            )),
            BfTreeError::InvalidKV(msg) => {
                crate::StorageError::Corrupted(alloc::format!("bf-tree: invalid key/value: {msg}"))
            }
            BfTreeError::InvalidKey => crate::StorageError::Corrupted(alloc::string::String::from(
                "bf-tree: key exceeds maximum length (data inconsistency)",
            )),
            BfTreeError::Config(e) => {
                crate::StorageError::Corrupted(alloc::format!("bf-tree config error: {e:?}"))
            }
            BfTreeError::Scan(e) => {
                crate::StorageError::Corrupted(alloc::format!("bf-tree scan error: {e:?}"))
            }
            BfTreeError::Corruption(msg) => crate::StorageError::Corrupted(msg),
            BfTreeError::InvalidOperation(msg) => {
                crate::StorageError::Corrupted(alloc::format!("invalid operation: {msg}"))
            }
            BfTreeError::ReservedTableName(name) => crate::StorageError::Corrupted(alloc::format!(
                "table name \"{name}\" uses the reserved \"__\" prefix"
            )),
            BfTreeError::InvalidConfig(msg) => {
                crate::StorageError::Corrupted(alloc::format!("bf-tree invalid config: {msg}"))
            }
            BfTreeError::TransactionTooLarge { written, limit } => {
                crate::StorageError::Corrupted(alloc::format!(
                    "bf-tree transaction size limit exceeded: {written} bytes written, limit {limit}"
                ))
            }
            BfTreeError::PartialFlushRollbackFailed {
                flush_error,
                rollback_failures,
                last_rollback_error,
            } => crate::StorageError::Corrupted(alloc::format!(
                "bf-tree: partial flush with failed rollback ({rollback_failures} \
                 rollback failure(s)): original: {flush_error}; last rollback error: \
                 {last_rollback_error}"
            )),
        }
    }
}
