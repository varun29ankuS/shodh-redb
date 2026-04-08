//! Unified backend selection for shodh-redb.
//!
//! Provides `BackendChoice` for choosing between the legacy B-tree and `BfTree`
//! storage engines at database creation time. Higher-level code that needs to
//! be backend-agnostic should use the `StorageRead`/`StorageWrite` traits from
//! `storage_traits.rs`, which both backends implement.
//!
//! # Design Rationale
//!
//! The legacy B-tree and `BfTree` engines have fundamentally different
//! concurrency models, transaction semantics, and feature sets:
//!
//! | Feature | Legacy B-tree | `BfTree` |
//! |---------|---------------|--------|
//! | Writers | Single-writer | Multi-writer |
//! | Isolation | MVCC snapshots | Read-latest |
//! | Savepoints | Yes | No (buffered rollback) |
//! | Durability | Double-buffered pages | WAL + snapshots |
//! | Max value | ~3 GiB | ~16 KiB |
//!
//! Because of these differences, a full type-erased wrapper (enum or trait object)
//! would either:
//! - Expose only the lowest common denominator (losing features), or
//! - Require every method to handle both variants (massive enum dispatch)
//!
//! Instead, this module provides:
//! 1. `BackendChoice` -- declarative backend selection
//! 2. `UnifiedDatabase` -- lightweight enum for database lifecycle
//! 3. Generic code should use `StorageRead`/`StorageWrite` traits directly

use std::path::Path;

use super::config::BfTreeConfig;
use super::database::BfTreeDatabase;
use super::error::BfTreeError;

/// Backend selection for database creation.
///
/// # Example
///
/// ```rust,ignore
/// use shodh_redb::bf_tree_store::{BackendChoice, BfTreeConfig, UnifiedDatabase};
///
/// // Legacy (default)
/// let db = UnifiedDatabase::create(BackendChoice::Legacy, "data.redb")?;
///
/// // BfTree with file-backed storage
/// let cfg = BfTreeConfig::new_file("data.bftree", 32);
/// let db = UnifiedDatabase::create(BackendChoice::BfTree(cfg), "data.bftree")?;
/// ```
pub enum BackendChoice {
    /// Legacy single-writer B-tree engine (default, full feature set).
    Legacy,
    /// `BfTree` concurrent B+tree engine (multi-writer, lock-free reads).
    BfTree(BfTreeConfig),
}

impl Default for BackendChoice {
    fn default() -> Self {
        Self::Legacy
    }
}

/// A database handle that wraps either backend.
///
/// For type-safe access to backend-specific features, match on the enum
/// variant. For generic code, extract a transaction and use `StorageRead`
/// or `StorageWrite` traits.
pub enum UnifiedDatabase {
    /// Legacy B-tree database.
    Legacy(crate::Database),
    /// `BfTree` concurrent database.
    BfTree(BfTreeDatabase),
}

impl UnifiedDatabase {
    /// Create a new database with the specified backend.
    ///
    /// For `BackendChoice::Legacy`, `path` is the database file path.
    /// For `BackendChoice::BfTree`, `path` is ignored if the `BfTreeConfig`
    /// already specifies a file path; otherwise it is used.
    #[cfg(feature = "std")]
    pub fn create(backend: BackendChoice, path: impl AsRef<Path>) -> Result<Self, UnifiedError> {
        match backend {
            BackendChoice::Legacy => {
                let db = crate::Database::create(path)?;
                Ok(Self::Legacy(db))
            }
            BackendChoice::BfTree(config) => {
                let db = BfTreeDatabase::create(config)?;
                Ok(Self::BfTree(db))
            }
        }
    }

    /// Open an existing database.
    ///
    /// For `BackendChoice::Legacy`, opens the file at `path`.
    /// For `BackendChoice::BfTree`, opens using the config's file path.
    #[cfg(feature = "std")]
    pub fn open(backend: BackendChoice, path: impl AsRef<Path>) -> Result<Self, UnifiedError> {
        match backend {
            BackendChoice::Legacy => {
                let db = crate::Database::open(path)?;
                Ok(Self::Legacy(db))
            }
            BackendChoice::BfTree(config) => {
                let db = BfTreeDatabase::open(config)?;
                Ok(Self::BfTree(db))
            }
        }
    }

    /// Returns `true` if this is a legacy B-tree database.
    pub fn is_legacy(&self) -> bool {
        matches!(self, Self::Legacy(_))
    }

    /// Returns `true` if this is a `BfTree` database.
    pub fn is_bf_tree(&self) -> bool {
        matches!(self, Self::BfTree(_))
    }

    /// Get a reference to the legacy database, if this is one.
    pub fn as_legacy(&self) -> Option<&crate::Database> {
        match self {
            Self::Legacy(db) => Some(db),
            Self::BfTree(_) => None,
        }
    }

    /// Get a reference to the `BfTree` database, if this is one.
    pub fn as_bf_tree(&self) -> Option<&BfTreeDatabase> {
        match self {
            Self::BfTree(db) => Some(db),
            Self::Legacy(_) => None,
        }
    }

    /// Consume and return the legacy database, if this is one.
    pub fn into_legacy(self) -> Option<crate::Database> {
        match self {
            Self::Legacy(db) => Some(db),
            Self::BfTree(_) => None,
        }
    }

    /// Consume and return the `BfTree` database, if this is one.
    pub fn into_bf_tree(self) -> Option<BfTreeDatabase> {
        match self {
            Self::BfTree(db) => Some(db),
            Self::Legacy(_) => None,
        }
    }
}

/// Error type that wraps both backend error types.
#[derive(Debug)]
pub enum UnifiedError {
    /// Legacy B-tree error.
    Legacy(crate::DatabaseError),
    /// `BfTree` error.
    BfTree(BfTreeError),
}

impl core::fmt::Display for UnifiedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Legacy(e) => write!(f, "legacy: {e}"),
            Self::BfTree(e) => write!(f, "bftree: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for UnifiedError {}

impl From<crate::DatabaseError> for UnifiedError {
    fn from(e: crate::DatabaseError) -> Self {
        Self::Legacy(e)
    }
}

impl From<BfTreeError> for UnifiedError {
    fn from(e: BfTreeError) -> Self {
        Self::BfTree(e)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TableDefinition;
    use crate::db::ReadableDatabase;

    const TABLE: TableDefinition<&str, u64> = TableDefinition::new("unified_test");

    #[test]
    fn unified_bftree_create_and_use() {
        let config = BfTreeConfig::new_memory(4);
        let udb = UnifiedDatabase::create(BackendChoice::BfTree(config), "").unwrap();
        assert!(udb.is_bf_tree());
        assert!(!udb.is_legacy());

        let db = udb.as_bf_tree().unwrap();
        let wtxn = db.begin_write();
        let mut t = wtxn.open_table(TABLE).unwrap();
        t.insert(&"hello", &42u64).unwrap();
        drop(t);
        wtxn.commit().unwrap();

        let rtxn = db.begin_read();
        let mut t = rtxn.open_table(TABLE).unwrap();
        let val = t.get(&"hello").unwrap().unwrap();
        assert_eq!(u64::from_le_bytes(val[..8].try_into().unwrap()), 42);
    }

    #[test]
    fn unified_legacy_create_and_use() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let udb = UnifiedDatabase::create(BackendChoice::Legacy, tmp.path()).unwrap();
        assert!(udb.is_legacy());

        let db = udb.as_legacy().unwrap();
        let wtxn = db.begin_write().unwrap();
        {
            let mut t = wtxn.open_table(TABLE).unwrap();
            t.insert("hello", &42u64).unwrap();
        }
        wtxn.commit().unwrap();

        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(TABLE).unwrap();
        assert_eq!(t.get("hello").unwrap().unwrap().value(), 42);
    }

    #[test]
    fn backend_choice_default_is_legacy() {
        assert!(matches!(BackendChoice::default(), BackendChoice::Legacy));
    }

    #[test]
    fn into_conversions() {
        let config = BfTreeConfig::new_memory(4);
        let udb = UnifiedDatabase::create(BackendChoice::BfTree(config), "").unwrap();
        assert!(udb.into_legacy().is_none());

        let config2 = BfTreeConfig::new_memory(4);
        let udb2 = UnifiedDatabase::create(BackendChoice::BfTree(config2), "").unwrap();
        assert!(udb2.into_bf_tree().is_some());
    }
}
