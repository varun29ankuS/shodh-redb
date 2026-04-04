//! Error types for the Bf-Tree storage bridge.

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
    Config(bf_tree::ConfigError),
    /// Scan operation failed.
    Scan(bf_tree::ScanIterError),
    /// Data corruption detected (e.g., missing blob chunks).
    Corruption(String),
    /// Invalid operation (e.g., write after finish).
    InvalidOperation(String),
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
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BfTreeError {}

impl From<bf_tree::ConfigError> for BfTreeError {
    fn from(e: bf_tree::ConfigError) -> Self {
        Self::Config(e)
    }
}

impl From<bf_tree::ScanIterError> for BfTreeError {
    fn from(e: bf_tree::ScanIterError) -> Self {
        Self::Scan(e)
    }
}

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
        }
    }
}
