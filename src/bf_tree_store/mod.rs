//! Bf-Tree concurrent storage engine bridge.
//!
//! Wraps Microsoft Research's `bf-tree` crate to provide concurrent read/write
//! access to shodh-redb's storage layer. This replaces the single-writer B-tree
//! core with a lock-free, mini-page-based index that supports multiple concurrent
//! writers and readers.
//!
//! # Architecture
//!
//! ```text
//! shodh-redb Table layer (typed Key/Value)
//!        |
//!        v
//! BfTreeAdapter (this module)
//!        |
//!        v
//! bf-tree::BfTree (concurrent B+tree with mini-pages)
//!        |
//!        v
//! VfsImpl (StdVfs / MemoryVfs / IoUringVfs)
//!        |
//!        v
//! Disk / Flash / Memory
//! ```
//!
//! # References
//!
//! - Hao & Chandramouli, "Bf-Tree: A Modern Read-Write-Optimized Concurrent
//!   Larger-Than-Memory Range Index", PVLDB Vol 17, 2024.

mod adapter;
mod blob;
mod buffered_txn;
mod config;
mod database;
mod error;
mod group_commit;
mod history;
mod multimap;
mod stress;
mod table;
mod transaction;
mod ttl;
mod unified;
mod verification;

pub use adapter::BfTreeAdapter;
pub use blob::{BfTreeBlobStore, BfTreeBlobWriter, BfTreeReadOnlyBlobStore};
pub use config::{BfTreeBackend, BfTreeConfig};
pub use database::{
    BfTreeBuilder, BfTreeDatabase, BfTreeDatabaseReadTxn, BfTreeDatabaseWriteTxn, BfTreeTableScan,
};
pub use error::BfTreeError;
pub use group_commit::{GroupCommit, WriteBatchFn, concurrent_group_commit};
pub use history::{BfTreeHistory, HistoryEntry};
pub use multimap::{BfTreeMultimapTable, BfTreeReadOnlyMultimapTable};
pub use table::{BfTreeReadOnlyTable, BfTreeTable};
pub use transaction::{BfTreeReadTxn, BfTreeWriteTxn};
pub use ttl::{BfTreeReadOnlyTtlTable, BfTreeTtlTable};
pub use unified::{BackendChoice, UnifiedDatabase, UnifiedError};
pub use verification::{VerifyMode, should_verify, unwrap_value, wrap_value};
