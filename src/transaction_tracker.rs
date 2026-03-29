use crate::compat::HashMap;
#[cfg(not(feature = "std"))]
use crate::compat::Mutex;
use crate::error::StorageError;
use crate::tree_store::TransactionalMemory;
use crate::{Key, Result, Savepoint, TypeName, Value};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use core::cmp::Ordering;
use core::mem;
use core::mem::size_of;
#[cfg(feature = "logging")]
use log::debug;
#[cfg(feature = "std")]
use std::sync::{Condvar, Mutex};

#[derive(Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(crate) struct TransactionId(u64);

impl TransactionId {
    pub(crate) fn new(value: u64) -> TransactionId {
        Self(value)
    }

    pub(crate) fn raw_id(self) -> u64 {
        self.0
    }

    pub(crate) fn next(self) -> Result<TransactionId> {
        let value = self.0.checked_add(1).ok_or_else(|| {
            StorageError::Corrupted(format!("TransactionId overflow at {}", self.0))
        })?;
        Ok(TransactionId(value))
    }

    pub(crate) fn increment(&mut self) -> Result<TransactionId> {
        let next = self.next()?;
        *self = next;
        Ok(next)
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug)]
pub(crate) struct SavepointId(pub u64);

impl SavepointId {
    pub(crate) fn next(self) -> Result<SavepointId> {
        let value = self.0.checked_add(1).ok_or_else(|| {
            StorageError::Corrupted(format!("SavepointId overflow at {}", self.0))
        })?;
        Ok(SavepointId(value))
    }
}

impl Value for SavepointId {
    type SelfType<'a> = SavepointId;
    type AsBytes<'a> = [u8; size_of::<u64>()];

    fn fixed_width() -> Option<usize> {
        Some(size_of::<u64>())
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        if data.len() < size_of::<u64>() {
            return SavepointId(0);
        }
        SavepointId(u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]))
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        value.0.to_le_bytes()
    }

    fn type_name() -> TypeName {
        TypeName::internal("redb::SavepointId")
    }
}

impl Key for SavepointId {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        Self::from_bytes(data1).0.cmp(&Self::from_bytes(data2).0)
    }
}

struct State {
    next_savepoint_id: SavepointId,
    // reference count of read transactions per transaction id
    live_read_transactions: BTreeMap<TransactionId, u64>,
    next_transaction_id: TransactionId,
    live_write_transaction: Option<TransactionId>,
    valid_savepoints: BTreeMap<SavepointId, TransactionId>,
    // Non-durable commits that are still in-memory, and waiting for a durable commit to get flushed
    // We need to make sure that the freed-table does not get processed for these, since they are not durable yet
    // Therefore, we hold a read transaction on their nearest durable ancestor
    //
    // Maps non-durable transaction id -> durable ancestor
    pending_non_durable_commits: HashMap<TransactionId, TransactionId>,
    // Non-durable commits which have NOT been processed in the freed table
    unprocessed_freed_non_durable_commits: BTreeSet<TransactionId>,
}

pub(crate) struct TransactionTracker {
    state: Mutex<State>,
    #[cfg(feature = "std")]
    live_write_transaction_available: Condvar,
}

impl TransactionTracker {
    pub(crate) fn new(next_transaction_id: TransactionId) -> Self {
        Self {
            state: Mutex::new(State {
                next_savepoint_id: SavepointId(0),
                live_read_transactions: Default::default(),
                next_transaction_id,
                live_write_transaction: None,
                valid_savepoints: Default::default(),
                pending_non_durable_commits: Default::default(),
                unprocessed_freed_non_durable_commits: Default::default(),
            }),
            #[cfg(feature = "std")]
            live_write_transaction_available: Condvar::new(),
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn start_write_transaction(&self) -> Result<TransactionId> {
        let mut state = self.state.lock()?;
        while state.live_write_transaction.is_some() {
            state = self.live_write_transaction_available.wait(state)?;
        }
        if state.live_write_transaction.is_some() {
            return Err(StorageError::Corrupted(
                "Write transaction still active after condvar wait".into(),
            ));
        }
        let transaction_id = state.next_transaction_id.increment()?;
        #[cfg(feature = "logging")]
        debug!("Beginning write transaction id={transaction_id:?}");
        state.live_write_transaction = Some(transaction_id);

        Ok(transaction_id)
    }

    #[cfg(not(feature = "std"))]
    pub(crate) fn start_write_transaction(&self) -> Result<TransactionId> {
        const MAX_SPIN_RETRIES: u32 = 1000;
        let mut retries: u32 = 0;
        let mut backoff: u32 = 1;

        loop {
            let mut state = self.state.lock();
            if state.live_write_transaction.is_none() {
                let transaction_id = state.next_transaction_id.increment()?;
                #[cfg(feature = "logging")]
                debug!("Beginning write transaction id={transaction_id:?}");
                state.live_write_transaction = Some(transaction_id);
                return Ok(transaction_id);
            }
            drop(state);

            retries += 1;
            if retries >= MAX_SPIN_RETRIES {
                return Err(StorageError::Corrupted(
                    "Timed out waiting for write transaction lock after 1000 spin iterations"
                        .into(),
                ));
            }

            for _ in 0..backoff {
                core::hint::spin_loop();
            }
            backoff = backoff.saturating_mul(2).min(64);
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn end_write_transaction(&self, id: TransactionId) -> Result {
        let mut state = self.state.lock()?;
        match state.live_write_transaction {
            Some(active_id) if active_id == id => {
                state.live_write_transaction = None;
                self.live_write_transaction_available.notify_one();
                Ok(())
            }
            Some(active_id) => Err(StorageError::Corrupted(format!(
                "end_write_transaction called with id {id:?}, but active transaction is {active_id:?}"
            ))),
            None => Err(StorageError::Corrupted(format!(
                "end_write_transaction called with id {id:?}, but no write transaction is active"
            ))),
        }
    }

    #[cfg(not(feature = "std"))]
    pub(crate) fn end_write_transaction(&self, id: TransactionId) -> Result {
        let mut state = self.state.lock();
        match state.live_write_transaction {
            Some(active_id) if active_id == id => {
                state.live_write_transaction = None;
                Ok(())
            }
            Some(active_id) => Err(StorageError::Corrupted(format!(
                "end_write_transaction called with id {id:?}, but active transaction is {active_id:?}"
            ))),
            None => Err(StorageError::Corrupted(format!(
                "end_write_transaction called with id {id:?}, but no write transaction is active"
            ))),
        }
    }

    pub(crate) fn clear_pending_non_durable_commits(&self) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        let ids = mem::take(&mut state.pending_non_durable_commits);
        for (_, durable_ancestor) in ids {
            if let Some(ref_count) = state.live_read_transactions.get_mut(&durable_ancestor) {
                *ref_count -= 1;
                if *ref_count == 0 {
                    state.live_read_transactions.remove(&durable_ancestor);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn is_unprocessed_non_durable_commit(&self, id: TransactionId) -> Result<bool> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(state.unprocessed_freed_non_durable_commits.contains(&id))
    }

    pub(crate) fn mark_unprocessed_non_durable_commit(&self, id: TransactionId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        state.unprocessed_freed_non_durable_commits.remove(&id);
        Ok(())
    }

    pub(crate) fn oldest_unprocessed_non_durable_commit(&self) -> Result<Option<TransactionId>> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(state
            .unprocessed_freed_non_durable_commits
            .iter()
            .next()
            .copied())
    }

    pub(crate) fn register_non_durable_commit(
        &self,
        id: TransactionId,
        durable_ancestor: TransactionId,
    ) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        state
            .live_read_transactions
            .entry(durable_ancestor)
            .and_modify(|x| *x += 1)
            .or_insert(1);
        state
            .pending_non_durable_commits
            .insert(id, durable_ancestor);
        state.unprocessed_freed_non_durable_commits.insert(id);
        Ok(())
    }

    pub(crate) fn restore_savepoint_counter_state(&self, next_savepoint: SavepointId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        if !state.valid_savepoints.is_empty() {
            return Err(StorageError::Corrupted(
                "restore_savepoint_counter_state called with active savepoints".into(),
            ));
        }
        state.next_savepoint_id = next_savepoint;
        Ok(())
    }

    pub(crate) fn register_persistent_savepoint(&self, savepoint: &Savepoint) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        state
            .live_read_transactions
            .entry(savepoint.get_transaction_id())
            .and_modify(|x| *x += 1)
            .or_insert(1);
        state
            .valid_savepoints
            .insert(savepoint.get_id(), savepoint.get_transaction_id());
        Ok(())
    }

    pub(crate) fn register_read_transaction(
        &self,
        mem: &TransactionalMemory,
    ) -> Result<TransactionId> {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        let id = mem.get_last_committed_transaction_id()?;
        state
            .live_read_transactions
            .entry(id)
            .and_modify(|x| *x += 1)
            .or_insert(1);

        Ok(id)
    }

    pub(crate) fn deallocate_read_transaction(&self, id: TransactionId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        if let Some(ref_count) = state.live_read_transactions.get_mut(&id) {
            *ref_count -= 1;
            if *ref_count == 0 {
                state.live_read_transactions.remove(&id);
            }
        }
        Ok(())
    }

    pub(crate) fn any_savepoint_exists(&self) -> Result<bool> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(!state.valid_savepoints.is_empty())
    }

    pub(crate) fn allocate_savepoint(&self, transaction_id: TransactionId) -> Result<SavepointId> {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        let id = state.next_savepoint_id.next()?;
        state.next_savepoint_id = id;
        state.valid_savepoints.insert(id, transaction_id);
        Ok(id)
    }

    // Deallocates the given savepoint and its matching reference count on the transaction
    pub(crate) fn deallocate_savepoint(
        &self,
        savepoint: SavepointId,
        transaction: TransactionId,
    ) -> Result {
        {
            #[cfg(feature = "std")]
            let mut state = self.state.lock()?;
            #[cfg(not(feature = "std"))]
            let mut state = self.state.lock();
            state.valid_savepoints.remove(&savepoint);
        }
        self.deallocate_read_transaction(transaction)
    }

    pub(crate) fn is_valid_savepoint(&self, id: SavepointId) -> Result<bool> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(state.valid_savepoints.contains_key(&id))
    }

    pub(crate) fn invalidate_savepoints_after(&self, id: SavepointId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        state.valid_savepoints.retain(|x, _| *x <= id);
        Ok(())
    }

    pub(crate) fn oldest_savepoint(&self) -> Result<Option<(SavepointId, TransactionId)>> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(state
            .valid_savepoints
            .first_key_value()
            .map(|x| (*x.0, *x.1)))
    }

    pub(crate) fn oldest_live_read_transaction(&self) -> Result<Option<TransactionId>> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(state.live_read_transactions.keys().next().copied())
    }

    pub(crate) fn register_history_hold(&self, transaction_id: TransactionId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        state
            .live_read_transactions
            .entry(transaction_id)
            .and_modify(|x| *x += 1)
            .or_insert(1);
        Ok(())
    }

    pub(crate) fn deallocate_history_hold(&self, transaction_id: TransactionId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        if let Some(ref_count) = state.live_read_transactions.get_mut(&transaction_id) {
            *ref_count -= 1;
            if *ref_count == 0 {
                state.live_read_transactions.remove(&transaction_id);
            }
        }
        Ok(())
    }

    // Returns the transaction id of the oldest non-durable transaction which has not been processed
    // for freeing, which has live read transactions
    pub(crate) fn oldest_live_read_nondurable_transaction(&self) -> Result<Option<TransactionId>> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        for id in state.live_read_transactions.keys() {
            if state.pending_non_durable_commits.contains_key(id) {
                return Ok(Some(*id));
            }
        }
        Ok(None)
    }
}
