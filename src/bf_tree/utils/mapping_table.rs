// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use alloc::vec::Vec;

use core::{cell::UnsafeCell, mem::MaybeUninit};

use crate::bf_tree::sync::{Mutex, MutexGuard};

// we support at most 128 batches, which is 128M pages.
const MAX_BATCHES: usize = 128;

struct RecordBatch<T> {
    data: Vec<MaybeUninit<T>>,
}

impl<T> RecordBatch<T> {
    fn new(record_per_batch: usize) -> Self {
        let mut data = Vec::with_capacity(record_per_batch);

        data.extend((0..record_per_batch).map(|_| MaybeUninit::uninit()));

        Self { data }
    }

    /// # Safety
    /// (1) The record must be initialized.
    /// (2) The record must be smaller than `RECORD_PER_BATCH`.
    unsafe fn get_record(&self, id: usize) -> &T {
        let record = unsafe { self.data.get_unchecked(id) };
        unsafe { record.assume_init_ref() }
    }
}

struct States {
    next_id: u64,
    current_initialized_batch: usize,
}

/// A mapping table that allows you to insert a record and later retrieve it by id.
/// Insertion and retrieval only takes exactly one (not O(1)!) memory access.
pub struct MappingTable<T> {
    states: Mutex<States>,
    batches: UnsafeCell<Vec<MaybeUninit<RecordBatch<T>>>>,
    record_per_batch: usize,
}

impl<T> Default for MappingTable<T> {
    fn default() -> Self {
        Self::new(DEFAULT_RECORD_PER_BATCH)
    }
}

impl<T> Drop for MappingTable<T> {
    fn drop(&mut self) {
        let mut states = self.states.lock();
        let initialized_records = states.next_id - 1;
        for i in 0..initialized_records {
            let batch_id = self.get_batch_id(i);
            let record_id = self.get_record_id(i);
            // SAFETY: batch_id is derived from a valid record id < next_id, so the
            // batch is initialized. Mutex is held via `states`.
            let batch = unsafe { self.get_batch_mut(batch_id, &mut states) };
            // SAFETY: record_id is within [0, record_per_batch) by modular arithmetic.
            let record = unsafe { batch.data.get_unchecked_mut(record_id) };
            // SAFETY: All records with id < next_id were previously written via `set`,
            // so the MaybeUninit is initialized and safe to drop in place.
            unsafe {
                record.as_mut_ptr().drop_in_place();
            }
        }
        let batch_cnt = states.current_initialized_batch + 1;
        for i in 0..batch_cnt {
            // SAFETY: We hold &mut self (Drop), so no concurrent access to the UnsafeCell.
            let record_batch_vec = unsafe { &mut *self.batches.get() };
            // SAFETY: i < batch_cnt <= MAX_BATCHES, within the pre-allocated Vec capacity.
            let batch = unsafe { record_batch_vec.get_unchecked_mut(i) };
            // SAFETY: Batches 0..=current_initialized_batch were initialized in `new` or `set`.
            unsafe {
                batch.as_mut_ptr().drop_in_place();
            }
        }
    }
}

const DEFAULT_RECORD_PER_BATCH: usize = 1024 * 1024;

impl<T> MappingTable<T> {
    pub fn new(record_per_batch: usize) -> Self {
        let mut batches = Vec::with_capacity(MAX_BATCHES);
        for i in 0..MAX_BATCHES {
            if i == 0 {
                batches.push(MaybeUninit::new(RecordBatch::new(record_per_batch)));
            } else {
                batches.push(MaybeUninit::uninit());
            }
        }

        Self {
            states: Mutex::new(States {
                next_id: 0,
                current_initialized_batch: 0,
            }),
            batches: UnsafeCell::new(batches),
            record_per_batch,
        }
    }

    pub(crate) fn new_from_iter(mapping: impl Iterator<Item = (u64, T)>) -> Self {
        let mt = Self::default();
        let mut states = mt.states.lock();

        for (id, val) in mapping {
            states.next_id = id + 1;
            mt.set(id, val, &mut states);
        }
        drop(states);
        mt
    }

    /// # Safety
    /// (1) The batch must be initialized.
    /// (2) `batch_id` must be less than `MAX_BATCHES`.
    unsafe fn get_batch(&self, batch_id: usize) -> &RecordBatch<T> {
        debug_assert!(
            batch_id < MAX_BATCHES,
            "batch_id {batch_id} >= MAX_BATCHES {MAX_BATCHES}"
        );
        // SAFETY: Callers ensure no mutable aliases exist; the UnsafeCell access is
        // safe because get_batch only produces a shared reference to the inner Vec.
        let record_batch_vec = unsafe { &mut *self.batches.get() };
        // SAFETY: batch_id < MAX_BATCHES and Vec was pre-allocated to exactly MAX_BATCHES.
        let batch = unsafe { record_batch_vec.get_unchecked_mut(batch_id) };
        // SAFETY: Caller's precondition (1) guarantees this batch has been initialized.
        let batch = unsafe { batch.assume_init_ref() };
        batch
    }

    /// # Safety
    /// (1) The batch must be initialized.
    /// (2) The batch must be smaller than `MAX_BATCHES`.
    /// (3) Mutex lock must be held.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_batch_mut(
        &self,
        batch_id: usize,
        _lock: &mut MutexGuard<'_, States>,
    ) -> &mut RecordBatch<T> {
        // SAFETY: Mutex lock is held (enforced by `_lock` parameter), preventing
        // concurrent mutable access to the UnsafeCell contents.
        let record_batch_vec = unsafe { &mut *self.batches.get() };
        // SAFETY: batch_id < MAX_BATCHES per caller precondition (2).
        let batch = unsafe { record_batch_vec.get_unchecked_mut(batch_id) };
        // SAFETY: Caller precondition (1) guarantees this batch is initialized.
        let batch = unsafe { batch.assume_init_mut() };
        batch
    }

    fn get_batch_id(&self, id: u64) -> usize {
        (id / self.record_per_batch as u64) as usize
    }

    fn get_record_id(&self, id: u64) -> usize {
        (id % self.record_per_batch as u64) as usize
    }

    /// Peek next id, used for iterating all entries.
    pub(crate) fn peek_next_id(&self) -> u64 {
        let states = self.states.lock();
        states.next_id
    }

    /// Get the record by id.
    /// The id must be returned by `insert`.
    pub fn get(&self, id: u64) -> &T {
        let batch_id = self.get_batch_id(id);
        let record_id = self.get_record_id(id);

        // SAFETY: id was returned by `insert`, so batch_id <= current_initialized_batch
        // and the batch is initialized. batch_id < MAX_BATCHES by construction.
        let batch = unsafe { self.get_batch(batch_id) };
        // SAFETY: record_id < record_per_batch and the record at this id was written
        // during `insert`, so it is initialized.
        let record = unsafe { batch.get_record(record_id) };
        record
    }

    fn set(&self, id: u64, val: T, states: &mut MutexGuard<States>) {
        let batch_id = self.get_batch_id(id);
        let record_id = self.get_record_id(id);

        if batch_id > states.current_initialized_batch {
            if batch_id >= MAX_BATCHES {
                panic!("Reached max batches!");
            }
            // SAFETY: Mutex is held (via `states`), so no concurrent UnsafeCell access.
            let batches = unsafe { &mut *self.batches.get() };
            // SAFETY: batch_id < MAX_BATCHES (checked above), within pre-allocated capacity.
            let batch = unsafe { batches.get_unchecked_mut(batch_id) };
            // SAFETY: Writing into an uninit slot; no previous value needs dropping.
            unsafe {
                batch
                    .as_mut_ptr()
                    .write(RecordBatch::new(self.record_per_batch));
            }
            states.current_initialized_batch = batch_id;
        }

        // SAFETY: batch_id is initialized (either batch 0 from `new`, or just
        // initialized above). Mutex is held via `states`.
        let batch = unsafe { self.get_batch_mut(batch_id, states) };
        // SAFETY: record_id < record_per_batch by modular arithmetic.
        let record = unsafe { batch.data.get_unchecked_mut(record_id) };
        // SAFETY: Writing a new value into a MaybeUninit slot. Each id is written
        // exactly once (next_id is monotonically incremented), so no double-init.
        unsafe {
            record.as_mut_ptr().write(val);
        }
    }

    /// Insert a record into the mapping table.
    /// Returns the id of the record and the reference to the record.
    ///
    /// The id can be used to retrieve the record later using `get`.
    pub fn insert(&self, val: T) -> (u64, &T) {
        let mut states = self.states.lock();

        let page_id = states.next_id;
        states.next_id += 1;

        self.set(page_id, val, &mut states);

        (page_id, self.get(page_id))
    }
}
