use crate::Error;
use crate::error::StorageError;
use crate::transactions::WriteTransaction;
use std::fmt::{Display, Formatter};
use std::sync::mpsc;
use std::sync::Mutex;

/// Error from a group commit operation.
#[derive(Debug)]
#[non_exhaustive]
pub enum GroupCommitError {
    /// This batch's operations caused an error.
    BatchFailed(Error),
    /// This batch was rolled back because another batch in the group failed.
    /// The caller may retry by resubmitting.
    PeerFailed,
    /// The write transaction could not be acquired.
    TransactionFailed(StorageError),
    /// The commit itself failed (fsync error, etc.).
    CommitFailed(StorageError),
    /// The database is shutting down.
    Shutdown,
    /// An internal mutex was poisoned (a thread panicked while holding the lock).
    LockPoisoned,
}

impl Display for GroupCommitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BatchFailed(e) => write!(f, "Batch operation failed: {e}"),
            Self::PeerFailed => write!(f, "Rolled back: another batch in the group failed"),
            Self::TransactionFailed(e) => write!(f, "Transaction acquisition failed: {e}"),
            Self::CommitFailed(e) => write!(f, "Commit failed: {e}"),
            Self::Shutdown => write!(f, "Database is shutting down"),
            Self::LockPoisoned => write!(f, "Internal mutex poisoned"),
        }
    }
}

impl std::error::Error for GroupCommitError {}

type BatchFn =
    Box<dyn FnOnce(&WriteTransaction) -> std::result::Result<(), Error> + Send + 'static>;

/// A batch of write operations submitted to group commit.
///
/// Create a batch from a closure that receives a `&WriteTransaction` and performs
/// mutations (open tables, insert, remove, etc.). The group committer manages the
/// transaction lifecycle -- do not call `commit()` or `abort()` within the closure.
///
/// # Example
///
/// ```ignore
/// use shodh_redb::{TableDefinition, WriteBatch};
///
/// const TABLE: TableDefinition<&str, u64> = TableDefinition::new("my_data");
///
/// let batch = WriteBatch::new(|txn| {
///     let mut table = txn.open_table(TABLE)?;
///     table.insert("key", &42)?;
///     Ok(())
/// });
/// db.submit_write_batch(batch)?;
/// ```
pub struct WriteBatch {
    operations: BatchFn,
}

impl WriteBatch {
    /// Create a batch from a closure that receives a shared `&WriteTransaction`.
    ///
    /// The closure should open tables, insert/remove entries, etc.
    /// Do not call `commit()` or `abort()` -- the group committer manages the lifecycle.
    pub fn new<F>(f: F) -> Self
    where
        F: FnOnce(&WriteTransaction) -> std::result::Result<(), Error> + Send + 'static,
    {
        Self {
            operations: Box::new(f),
        }
    }

    pub(crate) fn apply(self, txn: &WriteTransaction) -> std::result::Result<(), Error> {
        (self.operations)(txn)
    }
}

pub(crate) struct PendingBatch {
    pub batch: WriteBatch,
    pub result_tx: mpsc::SyncSender<Result<(), GroupCommitError>>,
}

struct GroupCommitState {
    pending: Vec<PendingBatch>,
    active_leader: bool,
    shutdown: bool,
}

pub(crate) struct GroupCommitter {
    state: Mutex<GroupCommitState>,
}

impl GroupCommitter {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(GroupCommitState {
                pending: Vec::new(),
                active_leader: false,
                shutdown: false,
            }),
        }
    }

    /// Enqueue a batch and determine whether this thread should become the leader.
    /// Returns `(should_lead, result_rx)`.
    pub fn enqueue(
        &self,
        batch: WriteBatch,
    ) -> Result<(bool, mpsc::Receiver<Result<(), GroupCommitError>>), GroupCommitError> {
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        let mut state = self
            .state
            .lock()
            .map_err(|_| GroupCommitError::LockPoisoned)?;
        if state.shutdown {
            return Err(GroupCommitError::Shutdown);
        }
        let should_lead = !state.active_leader;
        if should_lead {
            state.active_leader = true;
        }
        state.pending.push(PendingBatch { batch, result_tx });
        Ok((should_lead, result_rx))
    }

    /// Drain all pending batches. Called by the leader.
    pub fn drain_pending(&self) -> Result<Vec<PendingBatch>, GroupCommitError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| GroupCommitError::LockPoisoned)?;
        Ok(std::mem::take(&mut state.pending))
    }

    /// Atomically relinquish leadership, draining any batches that arrived
    /// while the leader was processing the previous round.
    ///
    /// Returns `Ok(batches)` — if non-empty the caller must process them
    /// before calling `finish_leader` again, preventing orphaned batches.
    pub fn finish_leader(&self) -> Result<Vec<PendingBatch>, GroupCommitError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| GroupCommitError::LockPoisoned)?;
        let remaining = std::mem::take(&mut state.pending);
        if remaining.is_empty() {
            state.active_leader = false;
        }
        // If remaining is non-empty we keep active_leader = true so no
        // other thread tries to become leader while we process the leftovers.
        Ok(remaining)
    }

    /// Shut down the group committer, failing all pending batches.
    pub fn shutdown(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.shutdown = true;
            let pending = std::mem::take(&mut state.pending);
            drop(state);
            for p in pending {
                let _ = p.result_tx.send(Err(GroupCommitError::Shutdown));
            }
        }
    }
}
