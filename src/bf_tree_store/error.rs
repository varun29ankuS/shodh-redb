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
            BfTreeError::NotFound | BfTreeError::Deleted => {
                crate::StorageError::Corrupted(alloc::format!("bf-tree: {e}"))
            }
            BfTreeError::InvalidKV(msg) => crate::StorageError::ValueTooLarge(msg.len()),
            BfTreeError::InvalidKey => {
                crate::StorageError::Corrupted(alloc::string::String::from("bf-tree: invalid key"))
            }
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
        }
    }
}
