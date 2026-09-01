use crate::compat::Arc;
use crate::compat::{Mutex, RwLock};
use crate::tree_store::page_store::base::PageHint;
use crate::tree_store::page_store::lru_cache::LRUCache;
use crate::{CacheStats, DatabaseError, Result, StorageBackend, StorageError};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Index;
use core::slice::SliceIndex;
use core::sync::atomic::Ordering;
#[cfg(feature = "cache_metrics")]
use portable_atomic::AtomicU64;
use portable_atomic::{AtomicBool, AtomicUsize};

pub(super) struct WritablePage {
    buffer: Arc<Mutex<LRUWriteCache>>,
    offset: u64,
    /// Generation of the write-cache entry this buffer was taken from. Offsets
    /// get reused, so identity needs more than the offset alone.
    generation: u64,
    data: Arc<[u8]>,
}

impl WritablePage {
    pub(super) fn mem(&self) -> &[u8] {
        &self.data
    }

    pub(super) fn mem_mut(&mut self) -> core::result::Result<&mut [u8], StorageError> {
        Arc::get_mut(&mut self.data).ok_or(StorageError::Internal(String::from(
            "WritablePage::mem_mut() called while other Arc references exist",
        )))
    }
}

impl Drop for WritablePage {
    fn drop(&mut self) {
        self.buffer
            .lock()
            .return_value(self.offset, self.generation, self.data.clone());
    }
}

impl<I: SliceIndex<[u8]>> Index<I> for WritablePage {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.mem().index(index)
    }
}

/// What [`LRUWriteCache::remove`] found at a key. The three cases have to stay
/// distinguishable: "present but checked out" needs different byte accounting
/// from "absent", and collapsing the two is the bug this enum replaced.
enum RemovedEntry {
    /// Present, holding its buffer.
    Buffered(Arc<[u8]>),
    /// Present, but `take_value` has handed the buffer to a live
    /// `WritablePage`. Carries the entry length, because there is no buffer
    /// left to measure and the caller still has to take those bytes off the
    /// write budget.
    ///
    /// Making the caller supply that length is what went wrong before:
    /// `cancel_pending_write` was handed a `len` it ignored, so the bytes were
    /// never returned and the budget drifted upward for the life of the
    /// database. The length is the entry own fact, so the entry reports it.
    CheckedOut(usize),
    /// No entry at this key.
    Absent,
}

/// One write-buffer entry, stamped with the generation it was created in.
///
/// The stamp lets a returning `WritablePage` tell its own entry from a
/// different entry that happens to sit at the same offset. Offsets are reused:
/// a page can be freed mid-write, which removes its entry while the page is
/// still alive, and the next allocation inserts a fresh entry at that offset.
/// Without the stamp the stale page returned its buffer into the new entry
/// slot, and the new page own return then found the slot occupied:
///
/// ```text
/// LRUWriteCache::return_value() slot was not empty for key 5120
/// ```
///
/// Silently dropping one of the two buffers would be worse than the panic:
/// whichever arrives second loses, so the entry can be left holding the
/// cancelled page stale bytes.
#[derive(Default)]
struct WriteCacheEntry {
    generation: u64,
    /// Length of the buffer, kept on the entry rather than derived from
    /// `value`, because a checked-out entry has no buffer to measure. Without
    /// it `cancel_pending_write` could not account for the bytes it was
    /// dropping and had to be handed a length by its caller.
    len: usize,
    value: Option<Arc<[u8]>>,
}

#[derive(Default)]
struct LRUWriteCache {
    cache: LRUCache<WriteCacheEntry>,
    next_generation: u64,
}

impl LRUWriteCache {
    fn new() -> Self {
        Self {
            cache: Default::default(),
            next_generation: 0,
        }
    }

    /// Insert `value` and return the generation stamp that `return_value` will
    /// require in order to accept the buffer back.
    fn insert(&mut self, key: u64, value: Arc<[u8]>) -> u64 {
        let generation = self.next_generation;
        self.next_generation += 1;
        let len = value.len();
        let prev = self.cache.insert(
            key,
            WriteCacheEntry {
                generation,
                len,
                value: Some(value),
            },
        );
        debug_assert!(
            prev.is_none(),
            "LRUWriteCache::insert() duplicate key {key}"
        );
        generation
    }

    fn get(&self, key: u64) -> Option<&Arc<[u8]>> {
        self.cache.get(key).and_then(|x| x.value.as_ref())
    }

    /// Remove `key`, reporting whether it was present and whether its buffer
    /// was checked out.
    ///
    /// This used to collapse [`RemovedEntry::CheckedOut`] into "absent" behind
    /// a `debug_assert!(value.is_some())`, which made the caller silently skip
    /// its byte accounting and fired under fuzzing. The state is reachable:
    /// `cancel_pending_write` runs from the page free and rollback paths, and
    /// a page can be freed while a `WritablePage` for it is still alive.
    fn remove(&mut self, key: u64) -> RemovedEntry {
        match self.cache.remove(key) {
            Some(WriteCacheEntry {
                value: Some(buffer),
                ..
            }) => RemovedEntry::Buffered(buffer),
            Some(WriteCacheEntry {
                value: None, len, ..
            }) => RemovedEntry::CheckedOut(len),
            None => RemovedEntry::Absent,
        }
    }

    /// Return a checked-out buffer, but only to the entry it came from.
    ///
    /// Called from `Drop`, which cannot propagate errors, so every rejected
    /// case just drops `value`. There are two, and both are expected:
    ///
    /// - No entry. `cancel_pending_write` removed it while this page held the
    ///   buffer, which is what happens when a page is freed mid-write. Its
    ///   bytes already came off `write_buffer_bytes` there.
    /// - Generation mismatch. The entry at this offset was removed and a
    ///   different one inserted in its place while this page was alive. That
    ///   newer entry owns the offset now, and writing a cancelled page bytes
    ///   into it would be silent corruption rather than a lost update.
    fn return_value(&mut self, key: u64, generation: u64, value: Arc<[u8]>) {
        if let Some(entry) = self.cache.get_mut(key) {
            if entry.generation != generation {
                return;
            }
            let prev = entry.value.replace(value);
            debug_assert!(
                prev.is_none(),
                "LRUWriteCache::return_value() slot was not empty for key {key}"
            );
        }
    }

    fn take_value(&mut self, key: u64) -> Option<(u64, Arc<[u8]>)> {
        let entry = self.cache.get_mut(key)?;
        let generation = entry.generation;
        entry.value.take().map(|value| (generation, value))
    }

    fn pop_lowest_priority(&mut self) -> Option<(u64, Arc<[u8]>)> {
        for _ in 0..self.cache.len() {
            if let Some((k, mut entry)) = self.cache.pop_lowest_priority() {
                if let Some(value) = entry.value.take() {
                    return Some((k, value));
                }

                // Value is borrowed by take_value(). We can't evict it, so put
                // it back -- with its original generation, or the live page
                // holding that buffer could not return it.
                self.cache.insert(k, entry);
            } else {
                break;
            }
        }
        None
    }

    fn clear(&mut self) {
        self.cache.clear();
    }
}

#[derive(Debug)]
struct CheckedBackend {
    file: Box<dyn StorageBackend>,
    io_failed: AtomicBool,
    closed: AtomicBool,
}

impl CheckedBackend {
    fn new(file: Box<dyn StorageBackend>) -> Self {
        Self {
            file,
            io_failed: AtomicBool::new(false),
            closed: AtomicBool::new(false),
        }
    }

    fn check_failure(&self) -> Result<()> {
        if self.io_failed.load(Ordering::Acquire) {
            if self.closed.load(Ordering::Acquire) {
                Err(StorageError::DatabaseClosed)
            } else {
                Err(StorageError::PreviousIo)
            }
        } else {
            Ok(())
        }
    }

    fn close(&self) -> Result {
        self.closed.store(true, Ordering::Release);
        self.io_failed.store(true, Ordering::Release);
        self.file.close()?;

        Ok(())
    }

    fn len(&self) -> Result<u64> {
        self.check_failure()?;
        let result = self.file.len();
        if result.is_err() {
            self.io_failed.store(true, Ordering::Release);
        }
        result.map_err(StorageError::from)
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        self.check_failure()?;
        let result = self.file.read(offset, out);
        if result.is_err() {
            self.io_failed.store(true, Ordering::Release);
        }
        result.map_err(StorageError::from)
    }

    fn set_len(&self, len: u64) -> Result<()> {
        self.check_failure()?;
        let result = self.file.set_len(len);
        if result.is_err() {
            self.io_failed.store(true, Ordering::Release);
        }
        result.map_err(StorageError::from)
    }

    fn sync_data(&self) -> Result<()> {
        self.check_failure()?;
        let result = self.file.sync_data();
        if result.is_err() {
            self.io_failed.store(true, Ordering::Release);
        }
        result.map_err(StorageError::from)
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        self.check_failure()?;
        let result = self.file.write(offset, data);
        if result.is_err() {
            self.io_failed.store(true, Ordering::Release);
        }
        result.map_err(StorageError::from)
    }
}

pub(super) struct PagedCachedFile {
    file: CheckedBackend,
    page_size: u64,
    max_read_cache_bytes: usize,
    read_cache_bytes: AtomicUsize,
    max_write_buffer_bytes: usize,
    write_buffer_bytes: AtomicUsize,
    memory_budget: Option<usize>,
    #[cfg(feature = "cache_metrics")]
    reads_total: AtomicU64,
    #[cfg(feature = "cache_metrics")]
    reads_hits: AtomicU64,
    #[cfg(feature = "cache_metrics")]
    writes_total: AtomicU64,
    #[cfg(feature = "cache_metrics")]
    writes_hits: AtomicU64,
    #[cfg(feature = "cache_metrics")]
    evictions: AtomicU64,
    read_cache: Vec<RwLock<LRUCache<Arc<[u8]>>>>,
    // Design: write buffer lives on PagedCachedFile so that cache state
    // survives across transaction boundaries for non-durable commits.
    write_buffer: Arc<Mutex<LRUWriteCache>>,
}

impl PagedCachedFile {
    pub(super) fn new(
        file: Box<dyn StorageBackend>,
        page_size: u64,
        max_read_cache_bytes: usize,
        max_write_buffer_bytes: usize,
        memory_budget: Option<usize>,
    ) -> Result<Self, DatabaseError> {
        let read_cache = (0..Self::lock_stripes())
            .map(|_| RwLock::new(LRUCache::new()))
            .collect();

        Ok(Self {
            file: CheckedBackend::new(file),
            page_size,
            max_read_cache_bytes,
            read_cache_bytes: AtomicUsize::new(0),
            max_write_buffer_bytes,
            write_buffer_bytes: AtomicUsize::new(0),
            memory_budget,
            #[cfg(feature = "cache_metrics")]
            reads_total: Default::default(),
            #[cfg(feature = "cache_metrics")]
            reads_hits: Default::default(),
            #[cfg(feature = "cache_metrics")]
            writes_total: Default::default(),
            #[cfg(feature = "cache_metrics")]
            writes_hits: Default::default(),
            #[cfg(feature = "cache_metrics")]
            evictions: Default::default(),
            read_cache,
            write_buffer: Arc::new(Mutex::new(LRUWriteCache::new())),
        })
    }

    pub(crate) fn cache_stats(&self) -> CacheStats {
        let read_bytes = self.read_cache_bytes.load(Ordering::Acquire);
        let write_bytes = self.write_buffer_bytes.load(Ordering::Acquire);
        let used_bytes = read_bytes + write_bytes;
        let budget_bytes = self.memory_budget;

        #[cfg(not(feature = "cache_metrics"))]
        {
            CacheStats {
                evictions: 0,
                read_hits: 0,
                read_misses: 0,
                write_hits: 0,
                write_misses: 0,
                used_bytes,
                budget_bytes,
            }
        }

        #[cfg(feature = "cache_metrics")]
        {
            let read_hits = self.reads_hits.load(Ordering::Acquire);
            let read_total = self.reads_total.load(Ordering::Acquire);
            let write_hits = self.writes_hits.load(Ordering::Acquire);
            let write_total = self.writes_total.load(Ordering::Acquire);
            CacheStats {
                evictions: self.evictions.load(Ordering::Acquire),
                read_hits,
                read_misses: read_total - read_hits,
                write_hits,
                write_misses: write_total - write_hits,
                used_bytes,
                budget_bytes,
            }
        }
    }

    /// Returns the total cache memory usage (read cache + write buffer).
    #[inline]
    fn total_cache_bytes(&self) -> usize {
        self.read_cache_bytes.load(Ordering::Acquire)
            + self.write_buffer_bytes.load(Ordering::Acquire)
    }

    /// Returns whether a memory budget is configured and currently exceeded.
    #[inline]
    fn is_over_budget(&self) -> bool {
        self.memory_budget
            .is_some_and(|budget| self.total_cache_bytes() > budget)
    }

    /// Evicts entries from the read cache across all stripes until `bytes_to_free`
    /// bytes have been freed or all stripes are exhausted.
    ///
    /// Returns the number of bytes actually freed.
    fn evict_read_cache_global(&self, bytes_to_free: usize) -> usize {
        let mut freed = 0usize;
        #[allow(clippy::cast_possible_truncation)] // lock_stripes() == 131, always fits in usize
        let num_stripes: usize = Self::lock_stripes() as usize;
        for stripe in 0..num_stripes {
            if freed >= bytes_to_free {
                break;
            }
            let mut lock = self.read_cache[stripe].write();
            while freed < bytes_to_free {
                if let Some((_, v)) = lock.pop_lowest_priority() {
                    freed += v.len();
                    #[cfg(feature = "cache_metrics")]
                    {
                        self.evictions.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    break;
                }
            }
        }
        if freed > 0 {
            self.read_cache_bytes.fetch_sub(freed, Ordering::AcqRel);
        }
        freed
    }

    pub(crate) fn close(&self) -> Result {
        self.file.close()
    }

    pub(crate) fn check_io_errors(&self) -> Result {
        self.file.check_failure()
    }

    pub(crate) fn raw_file_len(&self) -> Result<u64> {
        self.file.len()
    }

    const fn lock_stripes() -> u64 {
        131
    }

    /// Returns `(offset % lock_stripes())` as a `usize` cache-slot index.
    /// Safe because `lock_stripes()` is 131, so the result always fits in a `usize`.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn cache_slot(offset: u64) -> usize {
        (offset % Self::lock_stripes()) as usize
    }

    fn flush_write_buffer(&self) -> Result {
        let mut write_buffer = self.write_buffer.lock();

        for (offset, buffer) in write_buffer.cache.iter() {
            let raw = buffer.value.as_ref().ok_or_else(|| {
                StorageError::Internal(String::from(
                    "flush_write_buffer: write cache entry has no data",
                ))
            })?;
            self.file.write(*offset, raw)?;
        }
        for (offset, buffer) in write_buffer.cache.iter_mut() {
            let buffer = buffer.value.take().ok_or_else(|| {
                StorageError::Internal(String::from(
                    "flush_write_buffer: write cache entry has no data during promotion",
                ))
            })?;
            let cache_size = self
                .read_cache_bytes
                .fetch_add(buffer.len(), Ordering::AcqRel);

            if cache_size + buffer.len() <= self.max_read_cache_bytes {
                let cache_slot: usize = Self::cache_slot(*offset);
                let mut lock = self.read_cache[cache_slot].write();
                if let Some(replaced) = lock.insert(*offset, buffer) {
                    // A race could cause us to replace an existing buffer
                    self.read_cache_bytes
                        .fetch_sub(replaced.len(), Ordering::AcqRel);
                }
            } else {
                self.read_cache_bytes
                    .fetch_sub(buffer.len(), Ordering::AcqRel);
                break;
            }
        }
        self.write_buffer_bytes.store(0, Ordering::Release);
        write_buffer.clear();

        // If we have a memory budget and are over it after promoting write buffer
        // entries to the read cache, perform cross-stripe eviction.
        if let Some(budget) = self.memory_budget {
            let total = self.total_cache_bytes();
            if total > budget {
                self.evict_read_cache_global(total - budget);
            }
        }

        Ok(())
    }

    // Caller should invalidate all cached pages that are no longer valid
    pub(super) fn resize(&self, len: u64) -> Result {
        // Design: full read-cache invalidation on flush. Fine-grained tracking
        // of written pages would add per-page bookkeeping for marginal benefit.
        self.invalidate_cache_all();

        self.file.set_len(len)
    }

    pub(super) fn flush(&self) -> Result {
        self.flush_write_buffer()?;

        self.file.sync_data()
    }

    // Make writes visible to readers, but does not guarantee any durability
    pub(super) fn write_barrier(&self) -> Result {
        // Design: non-durable commits still flush to disk to avoid data loss if
        // the process crashes between a non-durable and subsequent durable commit.
        // Skipping the flush would require dirty-page tracking across commits.
        self.flush_write_buffer()
    }

    /// Write directly to the file, bypassing the page cache.
    /// Used for blob region writes which are not page-aligned.
    pub(super) fn write_direct(&self, offset: u64, data: &[u8]) -> Result {
        self.file.write(offset, data)
    }

    /// Ensure the file is at least `len` bytes, extending with zeros if needed.
    pub(super) fn ensure_len(&self, len: u64) -> Result {
        let current = self.file.len()?;
        if len > current {
            self.file.set_len(len)?;
        }
        Ok(())
    }

    // Read directly from the file, ignoring any cached data
    pub(super) fn read_direct(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut buffer = vec![0; len];
        self.file.read(offset, &mut buffer)?;
        Ok(buffer)
    }

    // Read with caching. Caller must not read overlapping ranges without first calling invalidate_cache().
    // Doing so will not cause UB, but is a logic error.
    pub(super) fn read(&self, offset: u64, len: usize, hint: PageHint) -> Result<Arc<[u8]>> {
        debug_assert_eq!(0, offset % self.page_size);
        #[cfg(feature = "cache_metrics")]
        self.reads_total.fetch_add(1, Ordering::AcqRel);

        if !matches!(hint, PageHint::Clean) {
            let lock = self.write_buffer.lock();
            if let Some(cached) = lock.get(offset) {
                #[cfg(feature = "cache_metrics")]
                self.reads_hits.fetch_add(1, Ordering::Release);
                #[cfg(not(fuzzing))]
                debug_assert_eq!(cached.len(), len);
                return Ok(cached.clone());
            }
        }

        let cache_slot: usize = Self::cache_slot(offset);
        {
            let read_lock = self.read_cache[cache_slot].read();
            if let Some(cached) = read_lock.get(offset) {
                #[cfg(feature = "cache_metrics")]
                self.reads_hits.fetch_add(1, Ordering::Release);
                #[cfg(not(fuzzing))]
                debug_assert_eq!(cached.len(), len);
                return Ok(cached.clone());
            }
        }

        // Cache miss -- read from disk
        let buffer: Arc<[u8]> = self.read_direct(offset, len)?.into();

        // If we have a memory budget and total usage already exceeds it,
        // skip caching entirely to prevent further memory growth.
        if self.is_over_budget() {
            self.evict_read_cache_global(buffer.len());
            return Ok(buffer);
        }

        let mut write_lock = self.read_cache[cache_slot].write();
        let cache_size = self
            .read_cache_bytes
            .fetch_add(buffer.len(), Ordering::AcqRel);
        let cache_size = if let Some(replaced) = write_lock.insert(offset, buffer.clone()) {
            // A race could cause us to replace an existing buffer
            self.read_cache_bytes
                .fetch_sub(replaced.len(), Ordering::AcqRel)
        } else {
            cache_size
        };
        let mut removed = 0;
        if cache_size + buffer.len() > self.max_read_cache_bytes {
            while removed < buffer.len() {
                if let Some((_, v)) = write_lock.pop_lowest_priority() {
                    #[cfg(feature = "cache_metrics")]
                    {
                        self.evictions.fetch_add(1, Ordering::Relaxed);
                    }
                    removed += v.len();
                } else {
                    break;
                }
            }
        }
        if removed > 0 {
            self.read_cache_bytes.fetch_sub(removed, Ordering::AcqRel);
        }

        // After per-stripe eviction, check if we need cross-stripe eviction
        // to bring total usage below the memory budget.
        if let Some(budget) = self.memory_budget {
            let total = self.total_cache_bytes();
            if total > budget {
                drop(write_lock);
                self.evict_read_cache_global(total - budget);
            }
        }

        Ok(buffer)
    }

    // Discard pending writes to the given range
    /// `_len` is the caller's idea of the page length. It is no longer used:
    /// the entry reports what it was charged, which is the only figure that
    /// can balance the insert. Kept in the signature because callers derive it
    /// anyway and removing it would churn `page_manager` for nothing.
    pub(super) fn cancel_pending_write(&self, offset: u64, _len: usize) {
        debug_assert_eq!(
            0,
            offset % self.page_size,
            "cancel_pending_write: offset not page-aligned"
        );
        match self.write_buffer.lock().remove(offset) {
            // Account with the buffer's own length.
            RemovedEntry::Buffered(removed) => {
                self.write_buffer_bytes
                    .fetch_sub(removed.len(), Ordering::Release);
            }
            // Present but checked out to a live `WritablePage`. Subtract what
            // the entry was actually charged at insert.
            //
            // The caller's length is deliberately not used, and asserting the
            // two agree was wrong: `fuzz_redb` produced "caller says 1024
            // bytes, entry holds 4096" immediately. The caller derives its
            // length from the page number's CURRENT order, while the entry
            // holds what was charged when it was inserted, and an offset
            // reused at a different order makes those differ legitimately --
            // the same reuse behind every other defect in this file.
            //
            // Only the entry's figure can balance the insert, so only the
            // entry's figure is used.
            RemovedEntry::CheckedOut(entry_len) => {
                self.write_buffer_bytes
                    .fetch_sub(entry_len, Ordering::Release);
            }
            RemovedEntry::Absent => {}
        }
    }

    // Invalidate any caching of the given range. After this call overlapping reads of the range are allowed
    //
    // NOTE: Invalidating a cached region in subsections is permitted, as long as all subsections are invalidated
    pub(super) fn invalidate_cache(&self, offset: u64, len: usize) {
        let cache_slot: usize = Self::cache_slot(offset);
        let mut lock = self.read_cache[cache_slot].write();
        if let Some(removed) = lock.remove(offset) {
            #[cfg(not(fuzzing))]
            debug_assert_eq!(
                len,
                removed.len(),
                "invalidate_cache: length mismatch for offset {offset}"
            );
            self.read_cache_bytes
                .fetch_sub(removed.len(), Ordering::AcqRel);
        }
    }

    pub(super) fn invalidate_cache_all(&self) {
        for cache_slot in 0..self.read_cache.len() {
            let mut lock = self.read_cache[cache_slot].write();
            while let Some((_, removed)) = lock.pop_lowest_priority() {
                self.read_cache_bytes
                    .fetch_sub(removed.len(), Ordering::AcqRel);
            }
        }
    }

    /// Return `data` when it is the sole reference to its allocation, otherwise
    /// a fresh copy of it.
    ///
    /// `WritablePage::mem_mut` hands out `&mut [u8]` from inside the `Arc`,
    /// which is sound only while no other reference exists. Both caches give
    /// readers `Arc` clones -- `read` returns `cached.clone()` from the write
    /// buffer and from the read cache alike -- so removing an entry from its
    /// cache drops the cache's own reference and nothing else. A `PageImpl`
    /// obtained before that removal still holds its clone, and the bytes it
    /// points at are about to be mutated underneath it.
    ///
    /// The check is not racy. `get_mut` succeeds only when this is the sole
    /// reference, and by the time it runs the entry has already been removed
    /// from both caches under their locks, so no new clone can be handed out
    /// while we decide.
    ///
    /// Upstream redb has the same shape here -- `Arc::get_mut(..).unwrap()` --
    /// and panics outright on this state, so copying closes an inherited hole
    /// rather than papering over a fork regression.
    fn uniquely_owned(mut data: Arc<[u8]>) -> Arc<[u8]> {
        if Arc::get_mut(&mut data).is_some() {
            data
        } else {
            data.to_vec().into()
        }
    }

    // If overwrite is true, the page is initialized to zero
    // cache_policy takes the existing data as an argument and returns the priority. The priority should be stable and not change after WritablePage is dropped
    pub(super) fn write(&self, offset: u64, len: usize, overwrite: bool) -> Result<WritablePage> {
        if offset % self.page_size != 0 {
            return Err(StorageError::Internal(String::from(
                "write: offset not page-aligned",
            )));
        }
        let mut lock = self.write_buffer.lock();

        // Performance: skipping the read-cache lookup for known-dirty pages would
        // save one hash probe per write. Marginal gain; deferred.
        let cache_slot: usize = Self::cache_slot(offset);
        let existing = {
            let mut lock = self.read_cache[cache_slot].write();
            if let Some(removed) = lock.remove(offset) {
                // The entry has left the cache either way, so its bytes come off
                // the accounting either way.
                self.read_cache_bytes
                    .fetch_sub(removed.len(), Ordering::AcqRel);
                if len == removed.len() {
                    Some(removed)
                } else {
                    // Stale, not corrupt. The read cache is keyed by offset
                    // alone, so a page freed at one order and reallocated at
                    // another reuses the offset with a different length. The
                    // free paths do invalidate (`page_manager::free` and the
                    // rollback loop), but both derive the length from
                    // `address_range(..)?`, which fails on corrupted input, and
                    // savepoint restore and repair replace allocator state
                    // wholesale rather than freeing page by page.
                    //
                    // `invalidate_cache` already treats exactly this state as
                    // benign -- it drops the entry regardless, behind a
                    // `debug_assert` that is gated `not(fuzzing)`. This branch
                    // used to call the same state a logic bug and return
                    // `Internal`, so the two paths disagreed about whether a
                    // stale-length entry was a defect. Agreeing with the one
                    // that tolerates it is the safe direction: the cached bytes
                    // describe a different extent and are not valid for this
                    // request under any interpretation, so discarding them and
                    // falling through to a fresh buffer (or a `read_direct` at
                    // the requested length) is what the caller needs.
                    //
                    // This does not weaken any allocator guard. A genuine double
                    // allocation is caught by the free-list and page-count
                    // validation on load, and by the `open_dirty_pages`
                    // checks -- not by a cache-coherence comparison here.
                    None
                }
            } else {
                None
            }
        };

        // A buffered page whose length does not match the request is stale for
        // the same reason a cached one is: the write buffer is keyed by offset,
        // and a page freed at one order and reallocated at another reuses the
        // offset at a different length.
        //
        // Reusing it was the original bug -- the caller got a page shorter than
        // it asked for (`assertion failed: mem.mem().len() >= allocation_size`,
        // and in release a silently truncated write). Reporting
        // `Internal` instead was the first correction, and it was still wrong:
        // the state is reachable without any logic error, because the free
        // paths that would have dropped this entry derive their length from
        // `address_range(..)?`, which fails on corrupted input, and savepoint
        // restore and repair replace allocator state wholesale.
        //
        // Discarding is safe, and the length mismatch is itself the proof. A
        // page's length is fixed by its allocation order, so a different length
        // at the same offset means the allocation that owned those buffered
        // bytes is gone. There is no live write to lose -- which is the point
        // I got wrong when this check was left alone while the read-cache one
        // was fixed.
        let buffered = match lock.take_value(offset) {
            Some((generation, removed)) if removed.len() == len => Some((generation, removed)),
            Some((_, removed)) => {
                // Drop the whole entry, not just its value: `take_value` leaves
                // the slot in place, and the fresh path below inserts at this
                // same offset, which would trip the duplicate-key assert.
                lock.remove(offset);
                self.write_buffer_bytes
                    .fetch_sub(removed.len(), Ordering::AcqRel);
                None
            }
            None => None,
        };

        let (generation, data) = if let Some((generation, removed)) = buffered {
            #[cfg(feature = "cache_metrics")]
            self.writes_hits.fetch_add(1, Ordering::AcqRel);
            (generation, Self::uniquely_owned(removed))
        } else {
            let previous = self.write_buffer_bytes.fetch_add(len, Ordering::AcqRel);
            // Compute how many bytes to evict: at least the overage beyond the limit.
            let overage = (previous + len).saturating_sub(self.max_write_buffer_bytes);
            if overage > 0 {
                let mut removed_bytes = 0;
                while removed_bytes < overage {
                    if let Some((evict_offset, buffer)) = lock.pop_lowest_priority() {
                        let removed_len = buffer.len();
                        let result = self.file.write(evict_offset, &buffer);
                        if result.is_err() {
                            lock.insert(evict_offset, buffer);
                        }
                        result?;
                        self.write_buffer_bytes
                            .fetch_sub(removed_len, Ordering::Release);
                        #[cfg(feature = "cache_metrics")]
                        {
                            self.evictions.fetch_add(1, Ordering::Relaxed);
                        }
                        removed_bytes += removed_len;
                    } else {
                        break;
                    }
                }
            }
            // Under a memory budget, also evict from the read cache to keep
            // total usage bounded during write-heavy transactions.
            if let Some(budget) = self.memory_budget {
                let total = self.total_cache_bytes();
                if total > budget {
                    self.evict_read_cache_global(total - budget);
                }
            }
            let result = if overwrite {
                // A cached entry holds the page's PREVIOUS bytes. Preferring it
                // over the zeroed buffer, as this did, handed stale contents to
                // a caller that had asked for a blank page -- the zeroing branch
                // was reachable only on a cache miss. `overwrite` is passed when
                // the header is rewritten and when the allocator hands out a
                // fresh page, so the stale bytes were another page's data.
                #[cfg(feature = "cache_metrics")]
                self.writes_hits.fetch_add(1, Ordering::AcqRel);
                drop(existing);
                vec![0; len].into()
            } else if let Some(data) = existing {
                #[cfg(feature = "cache_metrics")]
                self.writes_hits.fetch_add(1, Ordering::AcqRel);
                Self::uniquely_owned(data)
            } else {
                self.read_direct(offset, len)?.into()
            };
            lock.insert(offset, result);
            lock.take_value(offset).ok_or_else(|| {
                StorageError::Internal(String::from(
                    "write: take_value failed immediately after insert",
                ))
            })?
        };
        #[cfg(feature = "cache_metrics")]
        self.writes_total.fetch_add(1, Ordering::AcqRel);
        Ok(WritablePage {
            buffer: self.write_buffer.clone(),
            offset,
            generation,
            data,
        })
    }
}

#[cfg(test)]
mod test {
    use crate::StorageBackend;
    use crate::backends::InMemoryBackend;
    use crate::tree_store::PageHint;
    use crate::tree_store::page_store::cached_file::PagedCachedFile;
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use core::sync::atomic::Ordering;

    #[test]
    fn cache_leak() {
        let backend = InMemoryBackend::new();
        backend.set_len(1024).unwrap();
        let cached_file = PagedCachedFile::new(Box::new(backend), 128, 1024, 128, None).unwrap();
        let cached_file = Arc::new(cached_file);

        let t1 = {
            let cached_file = cached_file.clone();
            std::thread::spawn(move || {
                for _ in 0..1000 {
                    cached_file.read(0, 128, PageHint::None).unwrap();
                    cached_file.invalidate_cache(0, 128);
                }
            })
        };
        let t2 = {
            let cached_file = cached_file.clone();
            std::thread::spawn(move || {
                for _ in 0..1000 {
                    cached_file.read(0, 128, PageHint::None).unwrap();
                    cached_file.invalidate_cache(0, 128);
                }
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();
        cached_file.invalidate_cache(0, 128);
        assert_eq!(cached_file.read_cache_bytes.load(Ordering::Acquire), 0);
    }

    /// `write` removes the page from the read cache and used to reuse that
    /// allocation as the writable buffer. Removing it drops only the cache's
    /// reference, so a reader that already holds a clone -- a live `PageImpl`
    /// -- keeps the allocation shared, and `mem_mut` then refuses to hand out
    /// `&mut [u8]`.
    ///
    /// `fuzz_redb` reached this as
    /// `internal error: WritablePage::mem_mut() called while other Arc
    /// references exist`. It is not a fork regression: upstream redb unwraps
    /// the same `Arc::get_mut` and panics on the identical state.
    #[test]
    fn write_does_not_reuse_a_buffer_a_reader_still_holds() {
        let backend = InMemoryBackend::new();
        backend.set_len(1024).unwrap();
        let cached_file = PagedCachedFile::new(Box::new(backend), 128, 1024, 128, None).unwrap();

        // Populate the read cache and keep the reader's clone alive.
        let reader = cached_file.read(0, 128, PageHint::None).unwrap();

        let mut page = cached_file.write(0, 128, false).unwrap();
        page.mem_mut()
            .expect("the writable buffer must not be shared with a live reader");

        // The reader must still see its own bytes, not the writer's scratch.
        assert_eq!(reader.len(), 128);
        drop(reader);
    }

    /// `page_manager::free` and the rollback loop call `cancel_pending_write`,
    /// and a page can be freed while a `WritablePage` for it is still alive.
    /// That left the write cache entry removed while the buffer was checked
    /// out, which broke two invariants at once: `remove` hit
    /// `debug_assert!(value.is_some())` (`fuzz_db_image` panicked here once
    /// #365 removed the OOM that had been stopping it earlier), and the later
    /// `return_value` from `Drop` hit `key not found`.
    ///
    /// In release, where both asserts are absent, the failure was quieter and
    /// worse: the bytes were never returned, so `write_buffer_bytes` drifted
    /// upward permanently and shrank the effective write budget.
    #[test]
    fn cancel_pending_write_while_the_page_is_checked_out() {
        let backend = InMemoryBackend::new();
        backend.set_len(1024).unwrap();
        let cached_file = PagedCachedFile::new(Box::new(backend), 128, 1024, 512, None).unwrap();

        let before = cached_file.write_buffer_bytes.load(Ordering::Acquire);

        // Check the buffer out of the write cache.
        let page = cached_file.write(0, 128, true).unwrap();
        assert!(
            cached_file.write_buffer_bytes.load(Ordering::Acquire) > before,
            "the write should have been charged to the budget"
        );

        // Free the page while it is still held.
        cached_file.cancel_pending_write(0, 128);

        // Must not panic: the slot this page wants to return to is gone.
        drop(page);

        assert_eq!(
            cached_file.write_buffer_bytes.load(Ordering::Acquire),
            before,
            "cancelling a checked-out write must return its bytes to the budget"
        );
    }

    /// A page freed mid-write leaves a live `WritablePage` whose entry is gone.
    /// The next allocation reuses the offset and inserts a fresh entry, and the
    /// stale page then returned its buffer into that new entry, panicking with
    /// `LRUWriteCache::return_value() slot was not empty for key 5120`,
    /// which the 2026-09-01 nightly hit on `fuzz_redb`. The panic is the mild
    /// version. Dropping one of the two buffers instead, which is the obvious
    /// way to silence it, leaves whichever arrived second discarded -- so the
    /// entry can end up holding the cancelled page stale bytes and the live
    /// page loses its write. The generation stamp settles which buffer belongs
    /// to the entry instead of guessing from arrival order.
    #[test]
    fn a_cancelled_page_does_not_return_its_buffer_to_a_reused_offset() {
        let backend = InMemoryBackend::new();
        backend.set_len(4096).unwrap();
        let cached_file = PagedCachedFile::new(Box::new(backend), 1024, 4096, 4096, None).unwrap();

        // A checks out the buffer at this offset.
        let mut stale = cached_file.write(1024, 1024, true).unwrap();
        stale.mem_mut().unwrap().fill(0xAA);

        // The page is freed while A is still alive: its entry is removed.
        cached_file.cancel_pending_write(1024, 1024);

        // The offset is reallocated, so B gets a brand new entry there.
        let mut fresh = cached_file.write(1024, 1024, true).unwrap();
        fresh.mem_mut().unwrap().fill(0xBB);

        // A returns first and must not claim B entry.
        drop(stale);
        // B returns into its own entry, which must still be empty.
        drop(fresh);

        // The write buffer must hold B bytes, not A.
        let seen = cached_file.read(1024, 1024, PageHint::None).unwrap();
        assert!(
            seen.iter().all(|&b| b == 0xBB),
            "the reused offset must hold the live page bytes, not the cancelled page"
        );
    }

    /// The write buffer is keyed by offset too, so the same stale-length case
    /// arises there: a page buffered at one order, freed, and reallocated at
    /// another is requested at the same offset with a different length.
    ///
    /// `fuzz_redb` hit this as
    /// `Internal("write: write-buffer inconsistency at offset 5120, buffered
    /// 1024 bytes but 4096 requested")`.
    ///
    /// Discarding is safe and the mismatch proves it: a page's length is fixed
    /// by its allocation order, so a different length at the same offset means
    /// the allocation owning those buffered bytes is gone.
    #[test]
    fn write_discards_a_buffered_page_of_the_wrong_length() {
        let backend = InMemoryBackend::new();
        backend.set_len(16384).unwrap();
        let cached_file = PagedCachedFile::new(Box::new(backend), 1024, 8192, 8192, None).unwrap();

        // Buffer a page at one order and let it settle back into the buffer.
        {
            let mut small = cached_file.write(4096, 1024, true).unwrap();
            small.mem_mut().unwrap().fill(0xEE);
        }
        let charged = cached_file.write_buffer_bytes.load(Ordering::Acquire);
        assert!(charged >= 1024, "the buffered write should be charged");

        // Same offset, larger order: the small entry is stale.
        let mut big = cached_file
            .write(4096, 4096, true)
            .expect("a stale-length buffered page must not be fatal");
        let mem = big.mem_mut().unwrap();
        assert_eq!(mem.len(), 4096, "must match the requested length");
        assert!(
            mem.iter().all(|&b| b == 0),
            "overwrite must not surface the discarded page's bytes"
        );
        drop(big);

        // The discarded entry's bytes must not stay charged to the budget.
        assert!(
            cached_file.write_buffer_bytes.load(Ordering::Acquire) <= 4096,
            "stale buffered bytes were left on the budget"
        );
    }

    /// The read cache is keyed by offset alone, so a page freed at one order
    /// and reallocated at another reuses the offset with a different length.
    /// `write` used to call that a logic bug and return
    /// `Internal("write: cache inconsistency, length mismatch for cached page")`,
    /// which `fuzz_redb` hit once the shared-Arc defect in #376 was fixed.
    ///
    /// It is stale, not corrupt: `invalidate_cache` already drops exactly this
    /// state behind a `not(fuzzing)` debug assert. The cached bytes describe a
    /// different extent, so discarding them and reading fresh is what the
    /// caller needs.
    #[test]
    fn write_discards_a_cached_page_of_the_wrong_length() {
        let backend = InMemoryBackend::new();
        backend.set_len(4096).unwrap();
        backend.write(0, &[0xCD; 512]).unwrap();
        let cached_file = PagedCachedFile::new(Box::new(backend), 128, 4096, 512, None).unwrap();

        // Cache the offset at one length, then drop the reader's clone so this
        // isolates the length mismatch rather than the uniqueness check.
        let cached = cached_file.read(0, 128, PageHint::None).unwrap();
        assert_eq!(cached.len(), 128);
        drop(cached);

        // Same offset, larger order. Must succeed and be the requested size.
        let mut page = cached_file
            .write(0, 256, false)
            .expect("a stale-length cache entry must not be fatal");
        assert_eq!(
            page.mem_mut().unwrap().len(),
            256,
            "the page handed back must match the requested length"
        );
    }

    /// `overwrite` promises a zeroed page, but the cached-entry branch was
    /// tested first and won whenever the page happened to be in the read
    /// cache, returning the page's PREVIOUS bytes. The zeroing branch was
    /// reachable only on a cache miss.
    ///
    /// This matters because `overwrite` is what the allocator passes when it
    /// hands out a fresh page, so the stale contents belonged to whatever
    /// occupied that page before it was freed.
    #[test]
    fn overwrite_zeroes_a_page_that_is_still_in_the_read_cache() {
        let backend = InMemoryBackend::new();
        backend.set_len(1024).unwrap();
        backend.write(0, &[0xAB; 128]).unwrap();
        let cached_file = PagedCachedFile::new(Box::new(backend), 128, 1024, 128, None).unwrap();

        // Pull the page into the read cache so the `existing` branch is live,
        // then drop the clone so uniqueness is not what is under test here.
        let cached = cached_file.read(0, 128, PageHint::None).unwrap();
        assert_eq!(cached[0], 0xAB, "backend fixture did not take");
        drop(cached);

        let mut page = cached_file.write(0, 128, true).unwrap();
        let mem = page.mem_mut().unwrap();
        assert!(
            mem.iter().all(|&b| b == 0),
            "overwrite must hand back a zeroed page, not the cached contents"
        );
    }
}
