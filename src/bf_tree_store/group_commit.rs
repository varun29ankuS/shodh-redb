//! Group commit for `BfTree` transactions.
//!
//! Batches multiple write closures into a single durable commit point.
//! All closures share a single write transaction, ensuring atomicity:
//! either all batches succeed and are committed together, or none are.

use alloc::vec::Vec;
use std::sync::Arc;

use super::database::{BfTreeDatabase, BfTreeDatabaseWriteTxn};
use super::error::BfTreeError;

/// A write batch — a closure that operates on a shared write transaction.
pub type WriteBatchFn = Box<dyn FnOnce(&BfTreeDatabaseWriteTxn) -> Result<(), BfTreeError> + Send>;

/// Group commit coordinator.
///
/// Collects multiple write batches and executes them atomically:
/// 1. A single write transaction is created.
/// 2. Each batch runs against the shared transaction.
/// 3. A single `commit()` at the end makes all writes atomic — either all
///    succeed or none are persisted.
pub struct GroupCommit {
    db: Arc<BfTreeDatabase>,
    batches: Vec<WriteBatchFn>,
}

impl GroupCommit {
    /// Create a new group commit coordinator.
    pub fn new(db: Arc<BfTreeDatabase>) -> Self {
        Self {
            db,
            batches: Vec::new(),
        }
    }

    /// Add a write batch to the group.
    pub fn add<F>(&mut self, batch: F)
    where
        F: FnOnce(&BfTreeDatabaseWriteTxn) -> Result<(), BfTreeError> + Send + 'static,
    {
        self.batches.push(Box::new(batch));
    }

    /// Execute all batches atomically and commit.
    ///
    /// All batches share a single write transaction. If any batch fails, the
    /// transaction is rolled back (dropped without commit) and no writes are
    /// persisted. Returns the number of batches executed on success.
    pub fn execute(self) -> Result<usize, BfTreeError> {
        let count = self.batches.len();
        let wtxn = self.db.begin_write();
        for batch in self.batches {
            batch(&wtxn)?;
        }
        wtxn.commit()?;
        Ok(count)
    }

    /// Execute all batches atomically with explicit snapshot for durability.
    ///
    /// All batches share a single write transaction. A snapshot is taken after
    /// commit for crash recovery. Only use with file-backed databases.
    pub fn execute_with_snapshot(self) -> Result<(usize, std::path::PathBuf), BfTreeError> {
        let count = self.batches.len();
        let wtxn = self.db.begin_write();
        for batch in self.batches {
            batch(&wtxn)?;
        }
        wtxn.commit()?;
        let path = self.db.snapshot();
        Ok((count, path))
    }
}

/// Execute a group of write batches concurrently, then commit atomically.
///
/// All batches share a single write transaction protected by a mutex. Each
/// batch runs in its own thread, acquiring the mutex to interact with the
/// transaction. A single `commit()` at the end ensures atomicity.
pub fn concurrent_group_commit(
    db: Arc<BfTreeDatabase>,
    batches: Vec<WriteBatchFn>,
) -> Result<usize, BfTreeError> {
    use std::thread;

    let count = batches.len();
    let wtxn = Arc::new(db.begin_write());
    let mut handles = Vec::with_capacity(count);

    for batch in batches {
        let wtxn = wtxn.clone();
        handles.push(thread::spawn(move || -> Result<(), BfTreeError> {
            batch(&wtxn)?;
            Ok(())
        }));
    }

    // Join all threads and collect errors.
    for handle in handles {
        handle
            .join()
            .map_err(|_| BfTreeError::InvalidOperation("batch thread panicked".into()))??;
    }

    // All batches completed successfully — commit atomically.
    // unwrap Arc: all thread handles have been joined so we hold the only reference.
    let wtxn = Arc::into_inner(wtxn).ok_or_else(|| {
        BfTreeError::InvalidOperation("outstanding references to write transaction".into())
    })?;
    wtxn.commit()?;

    Ok(count)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TableDefinition;
    use crate::bf_tree_store::config::BfTreeConfig;

    const DATA: TableDefinition<&str, u64> = TableDefinition::new("data");

    fn test_db() -> Arc<BfTreeDatabase> {
        Arc::new(BfTreeDatabase::create(BfTreeConfig::new_memory(8)).unwrap())
    }

    #[test]
    fn sequential_group_commit() {
        let db = test_db();
        let mut gc = GroupCommit::new(db.clone());

        gc.add(|txn| {
            let mut t = txn.open_table(DATA);
            t.insert(&"a", &1u64)?;
            Ok(())
        });
        gc.add(|txn| {
            let mut t = txn.open_table(DATA);
            t.insert(&"b", &2u64)?;
            Ok(())
        });
        gc.add(|txn| {
            let mut t = txn.open_table(DATA);
            t.insert(&"c", &3u64)?;
            Ok(())
        });

        let count = gc.execute().unwrap();
        assert_eq!(count, 3);

        let rtxn = db.begin_read();
        let t = rtxn.open_table(DATA);
        assert!(t.get(&"a").unwrap().is_some());
        assert!(t.get(&"b").unwrap().is_some());
        assert!(t.get(&"c").unwrap().is_some());
    }

    #[test]
    fn concurrent_group_commit_test() {
        let db = test_db();

        let batches: Vec<WriteBatchFn> = (0u64..4)
            .map(|i| {
                let batch: WriteBatchFn = Box::new(move |txn| {
                    let mut t = txn.open_table(DATA);
                    let key = alloc::format!("key_{i}");
                    t.insert(&key.as_str(), &(i * 10))?;
                    Ok(())
                });
                batch
            })
            .collect();

        let count = concurrent_group_commit(db.clone(), batches).unwrap();
        assert_eq!(count, 4);

        let rtxn = db.begin_read();
        let t = rtxn.open_table(DATA);
        for i in 0u64..4 {
            let key = alloc::format!("key_{i}");
            assert!(t.get(&key.as_str()).unwrap().is_some());
        }
    }

    #[test]
    fn empty_group_commit() {
        let db = test_db();
        let gc = GroupCommit::new(db);
        assert_eq!(gc.execute().unwrap(), 0);
    }
}
