#[cfg(debug_assertions)]
use crate::compat::{HashMap, HashSet};
use crate::compat::Mutex;
use crate::db::{ReadVerification, ReadVerificationAction, ReadVerificationCallback};
use crate::transaction_tracker::TransactionId;
use crate::transactions::{AllocatorStateKey, AllocatorStateTree, AllocatorStateTreeMut};
use crate::tree_store::btree_base::{BtreeHeader, Checksum};
use crate::tree_store::page_store::base::{MAX_PAGE_INDEX, PageHint};
use crate::tree_store::page_store::buddy_allocator::BuddyAllocator;
use crate::tree_store::page_store::cached_file::PagedCachedFile;
use crate::tree_store::page_store::compression::CompressionConfig;
use crate::tree_store::page_store::fast_hash::PageNumberHashSet;
use crate::tree_store::page_store::header::{
    DB_HEADER_SIZE, DatabaseHeader, HeaderRepairInfo, MAGICNUMBER, MIRROR_MAGIC,
};
use crate::tree_store::page_store::layout::DatabaseLayout;
use crate::tree_store::page_store::region::{Allocators, RegionTracker};
use crate::tree_store::page_store::{PageImpl, PageMut, hash128_with_seed};
use crate::tree_store::read_verify::SamplingRng;
use crate::tree_store::{Page, PageNumber, PageTrackerPolicy};
use crate::{CacheStats, StorageBackend};
use crate::{DatabaseError, Result, StorageError};
use alloc::boxed::Box;
#[cfg(feature = "std")]
use alloc::format;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::{max, min};
use core::sync::atomic::{AtomicBool, Ordering};

// The region header is optional in the v3 file format
// It's an artifact of the v2 file format, so we initialize new databases without headers to save space
const NO_HEADER: u32 = 0;

// Regions have a maximum size of 4GiB. A `4GiB - overhead` value is the largest that can be represented,
// because the leaf node format uses 32bit offsets
const MAX_USABLE_REGION_SPACE: u64 = 4 * 1024 * 1024 * 1024;
// Retained as an upper bound for page order validation.
pub(crate) const MAX_MAX_PAGE_ORDER: u8 = 20;
pub(super) const MIN_USABLE_PAGES: u32 = 10;
const MIN_DESIRED_USABLE_BYTES: u64 = 1024 * 1024;

pub(super) const INITIAL_REGIONS: u32 = 1000; // Enough for a 4TiB database

// Original file format. No lengths stored with btrees
pub(crate) const FILE_FORMAT_VERSION1: u8 = 1;
// New file format. All btrees have a separate length stored in their header for constant time access
pub(crate) const FILE_FORMAT_VERSION2: u8 = 2;
// New file format:
// * Allocator state is stored in a system table, instead of in the region headers
// * Freed tree split into two system tables: one for the data tables, and one for the system tables
//   It is no longer stored in a separate tree
// * New "allocated pages table" which tracks the pages allocated, in the data tree, by a transaction.
//   This is a system table. It is only written when a savepoint exists
// * New persistent savepoint format
pub(crate) const FILE_FORMAT_VERSION3: u8 = 3;
// New file format:
// * Adds value-level compression (LZ4 / zstd)
// * Compression algorithm stored in database header
pub(crate) const FILE_FORMAT_VERSION4: u8 = 4;
// New file format:
// * Adds blob store with temporal indexing
// * Blob region metadata stored in commit slot _UNUSED2 area (4 x u64)
pub(crate) const FILE_FORMAT_VERSION5: u8 = 5;

#[derive(Copy, Clone)]
pub(crate) enum ShrinkPolicy {
    // Try to shrink the file by the default amount
    Default,
    // Try to shrink the file by the maximum amount
    Maximum,
    // Do not try to shrink the file
    Never,
}

#[allow(clippy::cast_possible_truncation)]
fn ceil_log2(x: usize) -> u8 {
    if x.is_power_of_two() {
        // trailing_zeros() returns at most usize::BITS-1 which always fits in u8
        x.trailing_zeros() as u8
    } else {
        x.next_power_of_two().trailing_zeros() as u8
    }
}

pub(crate) fn xxh3_checksum(data: &[u8]) -> Checksum {
    hash128_with_seed(data, 0)
}

struct InMemoryState {
    header: DatabaseHeader,
    // Design: initialized to a sentinel and overwritten during Database::open().
    // Making this Option would propagate unwrap/expect through all read paths.
    allocators: Allocators,
}

impl InMemoryState {
    fn new(header: DatabaseHeader) -> Result<Self> {
        let allocators = Allocators::new(header.layout())?;
        Ok(Self { header, allocators })
    }

    fn get_region(&self, region: u32) -> &BuddyAllocator {
        &self.allocators.region_allocators[region as usize]
    }

    fn get_region_mut(&mut self, region: u32) -> &mut BuddyAllocator {
        &mut self.allocators.region_allocators[region as usize]
    }

    fn get_region_tracker_mut(&mut self) -> &mut RegionTracker {
        &mut self.allocators.region_tracker
    }
}

/// Blob region commit state, applied atomically during commit.
#[derive(Clone, Default)]
pub(crate) struct BlobCommitState {
    pub(crate) region_offset: u64,
    pub(crate) region_length: u64,
    pub(crate) next_sequence: u64,
    pub(crate) hlc_state: u64,
}

pub(crate) struct TransactionalMemory {
    // Pages allocated since the last commit
    // Design: kept in TransactionalMemory for access from both read and write
    // paths during crash recovery.
    allocated_since_commit: Mutex<PageNumberHashSet>,
    unpersisted: Mutex<PageNumberHashSet>,
    // True if the allocator state was corrupted when the file was opened
    // Design: lightweight guard retained independent of CheckedBackend, which
    // may not be enabled in all configurations.
    needs_recovery: AtomicBool,
    storage: PagedCachedFile,
    state: Mutex<InMemoryState>,
    // The number of PageMut which are outstanding
    #[cfg(debug_assertions)]
    open_dirty_pages: Arc<Mutex<HashSet<PageNumber>>>,
    // Reference counts of PageImpls that are outstanding.
    // Debug-only: used to catch use-after-free in development. In release builds,
    // freeing a page with active readers is safe because PageImpl holds Arc<[u8]>.
    #[cfg(debug_assertions)]
    read_page_ref_counts: Arc<Mutex<HashMap<PageNumber, u64>>>,
    // Set of all allocated pages for debugging assertions
    #[cfg(debug_assertions)]
    allocated_pages: Arc<Mutex<PageNumberHashSet>>,
    // Indicates that a non-durable commit has been made, so reads should be served from the secondary meta page
    read_from_secondary: AtomicBool,
    page_size: u32,
    // We store these separately from the layout because they're static, and accessed on the get_page()
    // code path where there is no locking
    region_size: u64,
    region_header_with_padding_size: u64,
    compression: CompressionConfig,
    // Pending blob region state, applied to the commit slot during commit
    pending_blob_state: Mutex<BlobCommitState>,
    // Size of the EOF mirror header appended after data. Non-zero after a
    // successful durable commit writes the mirror. Used by file_len() to
    // exclude the mirror so that new blob regions are placed correctly.
    eof_mirror_size: portable_atomic::AtomicU64,
    // System-tree pages whose freeing was deferred because concurrent readers
    // were traversing the old snapshot. Drained at the start of the next
    // non-durable commit (or durable commit).
    pub(crate) deferred_nondurable_frees: Mutex<Vec<PageNumber>>,
    // Persisted system-tree pages whose freeing was deferred because a reader
    // registered between the pre-commit check and post-commit freeing (TOCTOU).
    // Drained into system_freed_pages at the start of the next durable commit
    // so they enter the standard SYSTEM_FREED_TABLE lifecycle.
    pub(crate) deferred_system_tree_frees: Mutex<Vec<PageNumber>>,
    // Read integrity verification
    read_verification: ReadVerification,
    sampling_rng: SamplingRng,
    read_verification_callback: Option<Arc<ReadVerificationCallback>>,
}

impl TransactionalMemory {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        file: Box<dyn StorageBackend>,
        // Allow initializing a new database in an empty file
        allow_initialize: bool,
        page_size: usize,
        requested_region_size: Option<u64>,
        read_cache_size_bytes: usize,
        write_cache_size_bytes: usize,
        read_only: bool,
        compression: CompressionConfig,
        memory_budget: Option<usize>,
        read_verification: ReadVerification,
        read_verification_callback: Option<Arc<ReadVerificationCallback>>,
    ) -> Result<Self, DatabaseError> {
        if !page_size.is_power_of_two() || page_size < DB_HEADER_SIZE {
            return Err(StorageError::invalid_config(
                "page_size must be a power of two and at least DB_HEADER_SIZE",
            )
            .into());
        }

        let region_size = requested_region_size.unwrap_or(MAX_USABLE_REGION_SPACE);
        let region_size = min(
            region_size,
            (u64::from(MAX_PAGE_INDEX) + 1) * page_size as u64,
        );
        if !region_size.is_power_of_two() {
            return Err(StorageError::invalid_config("region_size must be a power of two").into());
        }

        let storage = PagedCachedFile::new(
            file,
            page_size as u64,
            read_cache_size_bytes,
            write_cache_size_bytes,
            memory_budget,
        )?;

        let initial_storage_len = storage.raw_file_len()?;

        let magic_number: [u8; MAGICNUMBER.len()] =
            if initial_storage_len >= MAGICNUMBER.len() as u64 {
                storage
                    .read_direct(0, MAGICNUMBER.len())?
                    .try_into()
                    .map_err(|_| {
                        StorageError::format_error(
                            "Failed to read magic number: unexpected byte length",
                        )
                    })?
            } else {
                [0; MAGICNUMBER.len()]
            };

        if initial_storage_len > 0 {
            // File already exists check that the magic number matches
            if magic_number != MAGICNUMBER {
                return Err(StorageError::format_error("Invalid database file").into());
            }
        } else {
            // File is empty, check that we're allowed to initialize a new database (i.e. the caller is Database::create() and not open())
            if !allow_initialize {
                return Err(StorageError::format_error("Invalid database file").into());
            }
        }

        if magic_number != MAGICNUMBER {
            let region_tracker_required_bytes =
                RegionTracker::new(INITIAL_REGIONS, MAX_MAX_PAGE_ORDER + 1)
                    .to_vec()?
                    .len();

            // Make sure that there is enough room to allocate the region tracker into a page
            let size: u64 = max(
                MIN_DESIRED_USABLE_BYTES,
                page_size as u64 * u64::from(MIN_USABLE_PAGES),
            );
            let tracker_space =
                (page_size * region_tracker_required_bytes.div_ceil(page_size)) as u64;
            let starting_size = size + tracker_space;

            let page_size_u64 = u64::try_from(page_size)
                .map_err(|_| StorageError::invalid_config("page_size exceeds u64 range"))?;
            let page_capacity = (region_size / page_size_u64)
                .try_into()
                .map_err(|_| StorageError::invalid_config("page_capacity exceeds u32 range"))?;
            let layout = DatabaseLayout::calculate(
                starting_size,
                page_capacity,
                NO_HEADER,
                page_size.try_into().map_err(|_| {
                    StorageError::invalid_config("page_size exceeds target integer range")
                })?,
            );

            {
                let file_len = storage.raw_file_len()?;

                if file_len < layout.len() {
                    storage.resize(layout.len())?;
                }
            }

            let mut header = DatabaseHeader::new(layout, TransactionId::new(0), compression);

            header.recovery_required = false;
            header.two_phase_commit = true;
            storage
                .write(0, DB_HEADER_SIZE, true)?
                .mem_mut()?
                .copy_from_slice(&header.to_bytes(false));

            storage.flush()?;
            // Write the magic number only after the data structure is initialized and written to disk
            // to ensure that it's crash safe
            storage
                .write(0, DB_HEADER_SIZE, true)?
                .mem_mut()?
                .copy_from_slice(&header.to_bytes(true));
            storage.flush()?;
        }
        let header_bytes = storage.read_direct(0, DB_HEADER_SIZE)?;
        let (mut header, mut repair_info) = DatabaseHeader::from_bytes(&header_bytes)?;

        // If both commit slots are corrupted, attempt recovery from the EOF mirror
        if repair_info.primary_corrupted
            && repair_info.secondary_corrupted
            && let Some((mirror_header, mirror_repair)) = Self::try_load_mirror(&storage)?
        {
            header = mirror_header;
            repair_info = mirror_repair;
            // Restore the primary header from the mirror so future opens don't
            // need the mirror. Only possible when the file is writable.
            if !read_only {
                storage
                    .write(0, DB_HEADER_SIZE, true)?
                    .mem_mut()?
                    .copy_from_slice(&header.to_bytes(true));
                storage.flush()?;
            }
        }

        // For existing databases, the on-disk compression config takes precedence.
        let compression = header.compression;

        if header.page_size() as usize != page_size {
            return Err(StorageError::invalid_config(
                "Database page_size does not match requested page_size",
            )
            .into());
        }
        // The blob region (if any) is appended after the B-tree region.
        // blob_region_offset marks where the B-tree region ends and blobs begin.
        // Exclude the EOF mirror (if present) from the B-tree file length.
        let blob_region_offset = header.primary_slot().blob_region_offset;
        let btree_file_len = Self::effective_btree_file_len(&storage, blob_region_offset)?;
        if btree_file_len < header.layout().len() {
            return Err(StorageError::format_error(
                "B-tree file length is less than the database layout length",
            )
            .into());
        }
        let needs_recovery = header.recovery_required || header.layout().len() != btree_file_len;
        if needs_recovery {
            if read_only {
                return Err(DatabaseError::RepairAborted);
            }
            let layout = header.layout();
            let region_max_pages = layout.full_region_layout().num_pages();
            let region_header_pages = layout.full_region_layout().get_header_pages();
            header.set_layout(DatabaseLayout::recalculate(
                btree_file_len,
                region_header_pages,
                region_max_pages,
                page_size.try_into().unwrap(),
            ));
            header.pick_primary_for_repair(repair_info)?;
            if repair_info.invalid_magic_number {
                return Err(
                    StorageError::format_error("Invalid magic number during recovery").into(),
                );
            }
            storage
                .write(0, DB_HEADER_SIZE, true)?
                .mem_mut()?
                .copy_from_slice(&header.to_bytes(true));
            storage.flush()?;
        }

        let layout = header.layout();
        if layout.len() != btree_file_len {
            return Err(StorageError::format_error(
                "Database layout length does not match B-tree file length",
            )
            .into());
        }
        let region_size = layout.full_region_layout().len();
        let region_header_size = layout.full_region_layout().data_section().start;

        // Detect whether an EOF mirror currently exists so file_len() can exclude it
        let initial_mirror_size = {
            let raw = storage.raw_file_len()?;
            if raw >= 2 * DB_HEADER_SIZE as u64
                && Self::has_mirror_at(&storage, raw - DB_HEADER_SIZE as u64)?
            {
                DB_HEADER_SIZE as u64
            } else {
                0
            }
        };

        let state = InMemoryState::new(header)?;

        debug_assert!(page_size >= DB_HEADER_SIZE);

        Ok(Self {
            allocated_since_commit: Mutex::new(Default::default()),
            unpersisted: Mutex::new(Default::default()),
            needs_recovery: AtomicBool::new(needs_recovery),
            storage,
            state: Mutex::new(state),
            #[cfg(debug_assertions)]
            open_dirty_pages: Arc::new(Mutex::new(HashSet::new())),
            #[cfg(debug_assertions)]
            read_page_ref_counts: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(debug_assertions)]
            allocated_pages: Arc::new(Mutex::new(Default::default())),
            read_from_secondary: AtomicBool::new(false),
            page_size: page_size.try_into().unwrap(),
            region_size,
            region_header_with_padding_size: region_header_size,
            compression,
            pending_blob_state: Mutex::new(BlobCommitState::default()),
            eof_mirror_size: portable_atomic::AtomicU64::new(initial_mirror_size),
            deferred_nondurable_frees: Mutex::new(Vec::new()),
            deferred_system_tree_frees: Mutex::new(Vec::new()),
            read_verification,
            sampling_rng: SamplingRng::new(0xDEAD_BEEF_CAFE_1337),
            read_verification_callback,
        })
    }

    /// Creates a `TransactionalMemory` for read-only verification.
    ///
    /// Unlike `new()`, this method never writes to the storage backend.
    /// When the file needs recovery (backup files always do), the header is
    /// recalculated in memory only.
    ///
    /// Returns `(Self, header_valid)` where `header_valid` is `true` when the
    /// primary commit slot checksum was intact (no slot swap needed).
    #[cfg(feature = "std")]
    pub(crate) fn new_for_verify(
        storage: Box<dyn StorageBackend>,
        page_size: usize,
        region_size: Option<u64>,
        compression: CompressionConfig,
    ) -> core::result::Result<(Self, bool), DatabaseError> {
        let _region_size = region_size.unwrap_or(MAX_USABLE_REGION_SPACE);
        #[allow(clippy::cast_possible_truncation)]
        let storage = PagedCachedFile::new(storage, page_size as u64, 0, 0, None)?;

        let initial_storage_len = storage.raw_file_len()?;
        if initial_storage_len < DB_HEADER_SIZE as u64 {
            return Err(StorageError::format_error("Invalid database file").into());
        }

        let magic_number: [u8; MAGICNUMBER.len()] = storage
            .read_direct(0, MAGICNUMBER.len())?
            .try_into()
            .unwrap();

        if magic_number != MAGICNUMBER {
            return Err(StorageError::format_error("Invalid database file").into());
        }

        let header_bytes = storage.read_direct(0, DB_HEADER_SIZE)?;
        let (mut header, mut repair_info) = DatabaseHeader::from_bytes(&header_bytes)?;

        // If both commit slots are corrupted, attempt recovery from the EOF mirror
        if repair_info.primary_corrupted
            && repair_info.secondary_corrupted
            && let Some((mirror_header, mirror_repair)) = Self::try_load_mirror(&storage)?
        {
            header = mirror_header;
            repair_info = mirror_repair;
        }

        let mut header_valid = !repair_info.primary_corrupted;

        // Exclude EOF mirror from the effective file length used for layout calculations
        let blob_region_offset = header.primary_slot().blob_region_offset;
        let effective_file_len = Self::effective_btree_file_len(&storage, blob_region_offset)?;
        let needs_recovery =
            header.recovery_required || header.layout().len() != effective_file_len;
        if needs_recovery {
            // Recalculate layout in memory -- never write to the file
            let layout = header.layout();
            let region_max_pages = layout.full_region_layout().num_pages();
            let region_header_pages = layout.full_region_layout().get_header_pages();
            header.set_layout(DatabaseLayout::recalculate(
                effective_file_len,
                region_header_pages,
                region_max_pages,
                page_size.try_into().unwrap(),
            ));
            // pick_primary_for_repair can return Err for both-slots-corrupted
            match header.pick_primary_for_repair(repair_info) {
                Ok(primary_was_valid) => {
                    header_valid = primary_was_valid;
                }
                Err(e) => return Err(e.into()),
            }
        }

        let layout = header.layout();
        let file_len = storage.raw_file_len()?;
        if file_len < layout.len() {
            return Err(StorageError::format_error(format!(
                "File too short: {file_len} bytes, expected at least {} bytes",
                layout.len()
            ))
            .into());
        }
        let actual_region_size = layout.full_region_layout().len();
        let region_header_size = layout.full_region_layout().data_section().start;
        let state = InMemoryState::new(header)?;

        let mem = Self {
            allocated_since_commit: Mutex::new(Default::default()),
            unpersisted: Mutex::new(Default::default()),
            needs_recovery: AtomicBool::new(needs_recovery),
            storage,
            state: Mutex::new(state),
            #[cfg(debug_assertions)]
            open_dirty_pages: Arc::new(Mutex::new(HashSet::new())),
            #[cfg(debug_assertions)]
            read_page_ref_counts: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(debug_assertions)]
            allocated_pages: Arc::new(Mutex::new(Default::default())),
            read_from_secondary: AtomicBool::new(false),
            page_size: page_size.try_into().unwrap(),
            region_size: actual_region_size,
            region_header_with_padding_size: region_header_size,
            compression,
            pending_blob_state: Mutex::new(BlobCommitState::default()),
            eof_mirror_size: portable_atomic::AtomicU64::new(0),
            deferred_nondurable_frees: Mutex::new(Vec::new()),
            deferred_system_tree_frees: Mutex::new(Vec::new()),
            read_verification: ReadVerification::None,
            sampling_rng: SamplingRng::new(0xDEAD_BEEF_CAFE_1337),
            read_verification_callback: None,
        };

        Ok((mem, header_valid))
    }

    pub(crate) fn compression(&self) -> CompressionConfig {
        self.compression
    }

    /// Returns `true` if the next page read should be verified.
    ///
    /// Cost: None -> 0 (branch on enum discriminant), Sampled -> xorshift64,
    /// Full -> 0 (always true).
    pub(crate) fn should_verify_read(&self) -> bool {
        match self.read_verification {
            ReadVerification::None => false,
            ReadVerification::Sampled { rate } => self.sampling_rng.should_verify(rate),
            ReadVerification::Full => true,
        }
    }

    /// Handle a read verification failure. Returns `Ok(())` if the callback
    /// chose `Continue`, otherwise returns a `StorageError::Corrupted`.
    pub(crate) fn on_verification_failure(&self, page_number: PageNumber) -> Result {
        let page_num_raw =
            u64::from(page_number.page_index) | (u64::from(page_number.region) << 32);
        if let Some(ref cb) = self.read_verification_callback {
            match cb(page_num_raw) {
                ReadVerificationAction::ReturnError => Err(StorageError::page_corrupted(
                    page_number,
                    "read verification checksum mismatch",
                )),
                ReadVerificationAction::Continue => Ok(()),
            }
        } else {
            Err(StorageError::page_corrupted(
                page_number,
                "read verification checksum mismatch",
            ))
        }
    }

    /// Get the blob state for a write transaction.
    ///
    /// Returns the pending (in-transaction) state if any blob writes have occurred
    /// in the current transaction, otherwise falls back to the committed state.
    /// Must only be called from `WriteTransaction` methods.
    #[allow(dead_code)]
    pub(crate) fn get_blob_state(&self) -> BlobCommitState {
        let pending = self.pending_blob_state.lock().clone();
        if pending.region_offset != 0 || pending.next_sequence != 0 {
            return pending;
        }
        self.get_committed_blob_state()
    }

    /// Get the committed blob state from the header slot.
    ///
    /// Safe for concurrent readers -- only returns durably committed values.
    #[allow(dead_code)]
    pub(crate) fn get_committed_blob_state(&self) -> BlobCommitState {
        let state = self.state.lock();
        let slot = if self.read_from_secondary.load(Ordering::Acquire) {
            state.header.secondary_slot()
        } else {
            state.header.primary_slot()
        };
        BlobCommitState {
            region_offset: slot.blob_region_offset,
            region_length: slot.blob_region_length,
            next_sequence: slot.blob_next_sequence,
            hlc_state: slot.blob_hlc_state,
        }
    }

    /// Set pending blob state to be committed in the next transaction.
    #[allow(dead_code)]
    pub(crate) fn set_pending_blob_state(&self, state: BlobCommitState) {
        *self.pending_blob_state.lock() = state;
    }

    /// Write blob data directly to the file (bypasses page cache).
    #[allow(dead_code)]
    pub(crate) fn blob_write(&self, file_offset: u64, data: &[u8]) -> Result {
        self.storage.ensure_len(file_offset + data.len() as u64)?;
        self.storage.write_direct(file_offset, data)
    }

    /// Read blob data directly from the file (bypasses page cache).
    #[allow(dead_code)]
    pub(crate) fn blob_read(&self, file_offset: u64, length: usize) -> Result<Vec<u8>> {
        self.storage.read_direct(file_offset, length)
    }

    /// Get the current data-end file length, excluding the EOF mirror header.
    /// Used for initializing the blob region offset.
    #[allow(dead_code)]
    pub(crate) fn file_len(&self) -> Result<u64> {
        let raw = self.storage.raw_file_len()?;
        let mirror = self.eof_mirror_size.load(Ordering::Acquire);
        Ok(raw.saturating_sub(mirror))
    }

    /// Truncate the file to the given length.
    ///
    /// Used by blob compaction to reclaim space after the blob region shrinks.
    /// The caller must ensure `len` is at least `layout().len()` (the B-tree
    /// region size) and covers the committed blob region.
    #[allow(dead_code)]
    pub(crate) fn truncate_to(&self, len: u64) -> Result {
        // Truncation destroys the EOF mirror; the next commit will rewrite it.
        self.eof_mirror_size.store(0, Ordering::Release);
        self.storage.resize(len)
    }

    pub(crate) fn cache_stats(&self) -> CacheStats {
        self.storage.cache_stats()
    }

    pub(crate) fn check_io_errors(&self) -> Result {
        self.storage.check_io_errors()
    }

    #[cfg(debug_assertions)]
    pub(crate) fn mark_debug_allocated_page(&self, page: PageNumber) {
        // Idempotent: during crash recovery, corrupted B-trees may reference
        // the same page from multiple trees. The allocation-time assertion in
        // allocate_non_contiguous() catches real duplicate-allocation bugs.
        self.allocated_pages.lock().insert(page);
    }

    #[cfg(feature = "std")]
    #[cfg(debug_assertions)]
    pub(crate) fn all_allocated_pages(&self) -> Vec<PageNumber> {
        self.allocated_pages.lock().iter().copied().collect()
    }

    #[cfg(feature = "std")]
    #[cfg(debug_assertions)]
    pub(crate) fn debug_check_allocator_consistency(&self) {
        let state = self.state.lock();
        let mut region_pages = vec![vec![]; state.allocators.region_allocators.len()];
        for p in self.allocated_pages.lock().iter() {
            region_pages[p.region as usize].push(*p);
        }
        for (i, allocator) in state.allocators.region_allocators.iter().enumerate() {
            allocator.check_allocated_pages(i.try_into().unwrap(), &region_pages[i]);
        }
    }

    pub(crate) fn clear_read_cache(&self) {
        self.storage.invalidate_cache_all();
    }

    pub(crate) fn clear_cache_and_reload(&mut self) -> Result<bool, DatabaseError> {
        if !self.allocated_since_commit.lock().is_empty() {
            return Err(StorageError::Internal(alloc::string::String::from(
                "Cannot reload: uncommitted page allocations still pending",
            ))
            .into());
        }

        self.storage.flush()?;
        self.storage.invalidate_cache_all();

        let header_bytes = self.storage.read_direct(0, DB_HEADER_SIZE)?;
        let (mut header, repair_info) = DatabaseHeader::from_bytes(&header_bytes)?;
        // Note: recovery_required is typically true here because this is called
        // from check_integrity() after the database is already open and a write
        // transaction has begun (which sets recovery_required).
        let mut was_clean = true;
        if header.recovery_required {
            if !header.pick_primary_for_repair(repair_info)? {
                was_clean = false;
            }
            if repair_info.invalid_magic_number {
                return Err(StorageError::format_error("Invalid magic number").into());
            }
            // Recheck the layout against the actual file length in case it changed
            let blob_region_offset = header.primary_slot().blob_region_offset;
            let btree_file_len = Self::effective_btree_file_len(&self.storage, blob_region_offset)?;
            if header.layout().len() != btree_file_len {
                let layout = header.layout();
                let region_max_pages = layout.full_region_layout().num_pages();
                let region_header_pages = layout.full_region_layout().get_header_pages();
                header.set_layout(DatabaseLayout::recalculate(
                    btree_file_len,
                    region_header_pages,
                    region_max_pages,
                    self.page_size,
                ));
            }
            self.storage
                .write(0, DB_HEADER_SIZE, true)?
                .mem_mut()?
                .copy_from_slice(&header.to_bytes(true));
            self.storage.flush()?;
        }

        self.needs_recovery
            .store(header.recovery_required, Ordering::Release);
        self.state.lock().header = header;

        Ok(was_clean)
    }

    pub(crate) fn begin_writable(&self) -> Result {
        let mut state = self.state.lock();
        if state.header.recovery_required {
            return Err(StorageError::RecoveryRequired);
        }
        state.header.recovery_required = true;
        self.write_header(&state.header)?;
        self.storage.flush()
    }

    pub(crate) fn used_two_phase_commit(&self) -> bool {
        self.state.lock().header.two_phase_commit
    }

    pub(crate) fn allocator_hash(&self) -> u128 {
        self.state.lock().allocators.xxh3_hash()
    }

    /// Returns true if a storage failure has been detected and the database
    /// needs recovery before further operations. This is set when I/O errors
    /// occur during commit, rollback, or file growth, indicating that the
    /// on-disk state may be inconsistent.
    pub(crate) fn storage_failure(&self) -> bool {
        self.needs_recovery.load(Ordering::Acquire)
    }

    /// Mark the database as needing recovery. Called when an I/O error occurs
    /// during a critical operation (commit, rollback, resize) and the on-disk
    /// state may be inconsistent. Once set, all subsequent write transactions
    /// will fail with `RecoveryRequired` until the database is repaired.
    fn mark_needs_recovery(&self) {
        self.needs_recovery.store(true, Ordering::Release);
    }

    pub(crate) fn repair_primary_corrupted(&self) {
        let mut state = self.state.lock();
        state.header.swap_primary_slot();
    }

    pub(crate) fn begin_repair(&self) -> Result<()> {
        let mut state = self.state.lock();
        state.allocators = Allocators::new(state.header.layout())?;
        #[cfg(debug_assertions)]
        self.allocated_pages.lock().clear();

        Ok(())
    }

    pub(crate) fn mark_page_allocated(&self, page_number: PageNumber) -> Result {
        let mut state = self.state.lock();
        let region_index = page_number.region;
        let allocator = state.get_region_mut(region_index);
        allocator.record_alloc(page_number.page_index, page_number.page_order)?;
        #[cfg(debug_assertions)]
        // Idempotent: corrupted on-disk data may reference the same page twice.
        self.allocated_pages.lock().insert(page_number);
        Ok(())
    }

    fn write_header(&self, header: &DatabaseHeader) -> Result {
        self.storage
            .write(0, DB_HEADER_SIZE, true)?
            .mem_mut()?
            .copy_from_slice(&header.to_bytes(true));

        Ok(())
    }

    /// Check whether a valid EOF mirror header exists at the given file offset.
    fn has_mirror_at(storage: &PagedCachedFile, offset: u64) -> Result<bool> {
        let magic_bytes = storage.read_direct(offset, MIRROR_MAGIC.len())?;
        Ok(magic_bytes[..] == MIRROR_MAGIC[..])
    }

    /// Try to load a mirror header from the end of the file.
    /// Returns `Some((header, repair_info))` if a parseable mirror with at least
    /// one valid commit slot was found.
    fn try_load_mirror(
        storage: &PagedCachedFile,
    ) -> core::result::Result<Option<(DatabaseHeader, HeaderRepairInfo)>, DatabaseError> {
        let file_len = storage.raw_file_len()?;
        if file_len < 2 * DB_HEADER_SIZE as u64 {
            return Ok(None);
        }
        let mirror_offset = file_len - DB_HEADER_SIZE as u64;
        if !Self::has_mirror_at(storage, mirror_offset)? {
            return Ok(None);
        }
        let mut mirror_bytes = storage.read_direct(mirror_offset, DB_HEADER_SIZE)?;
        // Restore standard magic number so DatabaseHeader::from_bytes can parse it
        mirror_bytes[..MAGICNUMBER.len()].copy_from_slice(&MAGICNUMBER);
        match DatabaseHeader::from_bytes(&mirror_bytes) {
            Ok((header, repair_info)) => {
                if repair_info.primary_corrupted && repair_info.secondary_corrupted {
                    Ok(None)
                } else {
                    Ok(Some((header, repair_info)))
                }
            }
            Err(_) => Ok(None),
        }
    }

    /// Compute the effective B-tree file length, excluding any EOF mirror.
    /// When no blob region is present, the raw file length may include a trailing
    /// mirror header that must not be counted as part of the B-tree layout.
    fn effective_btree_file_len(storage: &PagedCachedFile, blob_region_offset: u64) -> Result<u64> {
        if blob_region_offset > 0 {
            return Ok(blob_region_offset);
        }
        let raw_len = storage.raw_file_len()?;
        if raw_len >= 2 * DB_HEADER_SIZE as u64 {
            let mirror_start = raw_len - DB_HEADER_SIZE as u64;
            if Self::has_mirror_at(storage, mirror_start)? {
                return Ok(mirror_start);
            }
        }
        Ok(raw_len)
    }

    /// Write a redundant copy of the database header at the end of the data region.
    ///
    /// Called during the commit path, before the final header swap and flush.
    /// If this write fails, the commit fails cleanly (the old commit slot is still
    /// primary), so errors propagate normally through the commit.
    fn write_mirror_header(&self, header: &DatabaseHeader, data_end: u64) -> Result {
        let required_len = data_end + DB_HEADER_SIZE as u64;
        let current_len = self.storage.raw_file_len()?;
        if current_len < required_len {
            self.storage.resize(required_len)?;
        }
        let mut mirror_bytes = header.to_bytes(true);
        mirror_bytes[..MIRROR_MAGIC.len()].copy_from_slice(&MIRROR_MAGIC);
        self.storage.write_direct(data_end, &mirror_bytes)?;
        // No separate flush -- the caller's flush persists the mirror data.
        self.eof_mirror_size
            .store(DB_HEADER_SIZE as u64, Ordering::Release);
        Ok(())
    }

    pub(crate) fn end_repair(&self) -> Result<()> {
        let mut state = self.state.lock();
        state.header.recovery_required = false;
        self.write_header(&state.header)?;
        let result = self.storage.flush();
        self.needs_recovery.store(false, Ordering::Release);

        result
    }

    pub(crate) fn reserve_allocator_state(
        &self,
        tree: &mut AllocatorStateTreeMut,
        transaction_id: TransactionId,
    ) -> Result<u32> {
        let state = self.state.lock();
        let layout = state.header.layout();
        let num_regions = layout.num_regions();
        let region_tracker_len = state.allocators.region_tracker.to_vec()?.len();
        let region_lens: Vec<usize> = state
            .allocators
            .region_allocators
            .iter()
            .map(|x| x.to_vec().map(|v| v.len()))
            .collect::<Result<Vec<_>>>()?;
        drop(state);

        for i in 0..num_regions {
            let region_bytes_len = region_lens[i as usize];
            tree.insert(
                &AllocatorStateKey::Region(i),
                &vec![0; region_bytes_len].as_ref(),
            )?;
        }

        tree.insert(
            &AllocatorStateKey::RegionTracker,
            &vec![0; region_tracker_len].as_ref(),
        )?;

        tree.insert(
            &AllocatorStateKey::TransactionId,
            &transaction_id.raw_id().to_le_bytes().as_ref(),
        )?;

        Ok(num_regions)
    }

    // Returns true on success, or false if the number of regions has changed
    pub(crate) fn try_save_allocator_state(
        &self,
        tree: &mut AllocatorStateTreeMut,
        num_regions: u32,
    ) -> Result<bool> {
        // Has the number of regions changed since reserve_allocator_state() was called?
        let state = self.state.lock();
        if num_regions != state.header.layout().num_regions() {
            return Ok(false);
        }

        for i in 0..num_regions {
            let region_bytes = state.allocators.region_allocators[i as usize].to_vec()?;
            if tree
                .get(&AllocatorStateKey::Region(i))?
                .unwrap()
                .value()
                .len()
                < region_bytes.len()
            {
                // The allocator state grew too much since we reserved space
                return Ok(false);
            }
            tree.insert_inplace(&AllocatorStateKey::Region(i), &region_bytes.as_ref())?;
        }

        let region_tracker_bytes = state.allocators.region_tracker.to_vec()?;
        if tree
            .get(&AllocatorStateKey::RegionTracker)?
            .unwrap()
            .value()
            .len()
            < region_tracker_bytes.len()
        {
            // The allocator state grew too much since we reserved space
            return Ok(false);
        }
        tree.insert_inplace(
            &AllocatorStateKey::RegionTracker,
            &region_tracker_bytes.as_ref(),
        )?;

        Ok(true)
    }

    // Returns true if the allocator state table is up to date, or false if it's stale
    pub(crate) fn is_valid_allocator_state(&self, tree: &AllocatorStateTree) -> Result<bool> {
        // See if this is stale allocator state left over from a previous transaction. That won't
        // happen during normal operation, since WriteTransaction::commit() always updates the
        // allocator state table before calling TransactionalMemory::commit(), but there are also
        // a few places where TransactionalMemory::commit() is called directly without using a
        // WriteTransaction. When that happens, any existing allocator state table will be left
        // in place but is no longer valid. (And even if there were no such calls today, it would
        // be an easy mistake to make! So it's good that we check.)
        let Some(value) = tree.get(&AllocatorStateKey::TransactionId)? else {
            return Ok(false);
        };
        let transaction_id = TransactionId::new(u64::from_le_bytes(
            value.value().try_into().map_err(|_| {
                StorageError::Corrupted("allocator state: invalid transaction ID length".into())
            })?,
        ));

        Ok(transaction_id == self.get_last_committed_transaction_id()?)
    }

    pub(crate) fn load_allocator_state(&self, tree: &AllocatorStateTree) -> Result {
        if !self.is_valid_allocator_state(tree)? {
            return Err(StorageError::RecoveryRequired);
        }

        // Load the allocator state
        let mut region_allocators = vec![];
        for region in
            tree.range(&(AllocatorStateKey::Region(0)..=AllocatorStateKey::Region(u32::MAX)))?
        {
            region_allocators.push(BuddyAllocator::from_bytes(region?.value())?);
        }

        let region_tracker = RegionTracker::from_bytes(
            tree.get(&AllocatorStateKey::RegionTracker)?
                .ok_or_else(|| {
                    StorageError::Corrupted("Missing RegionTracker entry in allocator state".into())
                })?
                .value(),
        )?;

        let mut state = self.state.lock();
        state.allocators = Allocators {
            region_tracker,
            region_allocators,
        };

        // Resize the allocators to match the current file size
        let layout = state.header.layout();
        state.allocators.resize_to(layout)?;
        drop(state);

        self.state.lock().header.recovery_required = false;
        self.needs_recovery.store(false, Ordering::Release);

        Ok(())
    }

    #[cfg_attr(not(debug_assertions), expect(unused_variables))]
    #[cfg_attr(not(debug_assertions), allow(clippy::unused_self))]
    pub(crate) fn is_allocated(&self, page: PageNumber) -> bool {
        #[cfg(debug_assertions)]
        {
            let allocated = self.allocated_pages.lock();
            allocated.contains(&page)
        }
        #[cfg(not(debug_assertions))]
        {
            unreachable!()
        }
    }

    // Commit all outstanding changes and make them visible as the primary
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit(
        &self,
        data_root: Option<BtreeHeader>,
        system_root: Option<BtreeHeader>,
        transaction_id: TransactionId,
        two_phase: bool,
        shrink_policy: ShrinkPolicy,
    ) -> Result {
        let result = self.commit_inner(
            data_root,
            system_root,
            transaction_id,
            two_phase,
            shrink_policy,
        );
        if result.is_err() {
            self.mark_needs_recovery();
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_inner(
        &self,
        data_root: Option<BtreeHeader>,
        system_root: Option<BtreeHeader>,
        transaction_id: TransactionId,
        two_phase: bool,
        shrink_policy: ShrinkPolicy,
    ) -> Result {
        // All mutable pages must be dropped, this ensures that when a transaction completes
        // no more writes can happen to the pages it allocated. Thus it is safe to make them visible
        // to future read transactions
        #[cfg(all(debug_assertions, not(fuzzing)))]
        debug_assert!(self.open_dirty_pages.lock().is_empty());
        if self.needs_recovery.load(Ordering::Acquire) {
            return Err(StorageError::RecoveryRequired);
        }

        let mut state = self.state.lock();
        // Trim surplus file space, before finalizing the commit
        let shrunk = if !matches!(shrink_policy, ShrinkPolicy::Never) {
            Self::try_shrink(&mut state, matches!(shrink_policy, ShrinkPolicy::Maximum))?
        } else {
            false
        };
        // Copy the header so that we can release the state lock, while we flush the file
        let mut header = state.header.clone();
        drop(state);

        let old_transaction_id = header.secondary_slot().transaction_id;
        let secondary = header.secondary_slot_mut();
        secondary.transaction_id = transaction_id;
        secondary.user_root = data_root;
        secondary.system_root = system_root;

        // Apply blob region state: use pending if set, otherwise carry forward committed
        let blob_state = self.get_blob_state();
        secondary.blob_region_offset = blob_state.region_offset;
        secondary.blob_region_length = blob_state.region_length;
        secondary.blob_next_sequence = blob_state.next_sequence;
        secondary.blob_hlc_state = blob_state.hlc_state;

        // Upgrade to V5 when blob store is in use
        if blob_state.region_offset > 0 && secondary.version < FILE_FORMAT_VERSION5 {
            secondary.version = FILE_FORMAT_VERSION5;
        }

        self.write_header(&header)?;

        // Use 2-phase commit, if checksums are disabled
        if two_phase {
            self.storage.flush()?;
        }

        // Make our new commit the primary, and record whether it was a 2-phase commit.
        // These two bits need to be written atomically
        header.swap_primary_slot();
        header.two_phase_commit = two_phase;

        // Compute the data region end for the mirror and shrink logic
        let btree_len = header.layout().len();
        let blob_end = blob_state
            .region_offset
            .saturating_add(blob_state.region_length);
        let data_end = btree_len.max(blob_end);

        // Write the new header to disk
        self.write_header(&header)?;
        // Write redundant header mirror at the end of the data region.
        // This is part of the commit I/O sequence: if it fails, the commit fails
        // cleanly before the flush, so the old primary slot remains valid.
        self.write_mirror_header(&header, data_end)?;
        self.storage.flush()?;

        if shrunk {
            // When a blob region exists past the B-tree layout, the file must be
            // at least large enough to hold both. Never truncate into the blob region.
            // The mirror occupies space beyond data_end, so resize to include it.
            let target_len = data_end + DB_HEADER_SIZE as u64;
            let result = self.storage.resize(target_len);
            if result.is_err() {
                self.mark_needs_recovery();
                return result;
            }
        }

        let mut allocated_since_commit = self.allocated_since_commit.lock();
        allocated_since_commit.clear();
        allocated_since_commit.shrink_to_fit();
        let mut unpersisted = self.unpersisted.lock();
        unpersisted.clear();
        unpersisted.shrink_to_fit();

        let mut state = self.state.lock();
        if state.header.secondary_slot().transaction_id != old_transaction_id {
            return Err(StorageError::Internal(alloc::string::String::from(
                "Secondary slot transaction_id changed unexpectedly during commit",
            )));
        }
        state.header = header;
        self.read_from_secondary.store(false, Ordering::Release);
        // Hold lock until read_from_secondary is set to false, so that the new primary state is read.
        // Hold lock until read_from_secondary is false so readers see the new primary state.
        drop(state);

        // Reset pending blob state so the next transaction starts from committed header
        *self.pending_blob_state.lock() = BlobCommitState::default();

        Ok(())
    }

    // Make changes visible, without a durability guarantee.
    //
    // Crash safety: pages moved into `unpersisted` are not tracked on disk.
    // On crash, they would leak because the in-memory set is lost and the
    // on-disk header was never flushed. This is safe because `begin_writable`
    // durably sets `recovery_required = true` before any writes. On next open,
    // the recovery process (`begin_repair` / `end_repair`) rebuilds the
    // allocator from scratch by walking only reachable pages from the primary
    // header, so non-durable pages are implicitly reclaimed.
    pub(crate) fn non_durable_commit(
        &self,
        data_root: Option<BtreeHeader>,
        system_root: Option<BtreeHeader>,
        transaction_id: TransactionId,
    ) -> Result {
        // All mutable pages must be dropped, this ensures that when a transaction completes
        // no more writes can happen to the pages it allocated. Thus it is safe to make them visible
        // to future read transactions
        #[cfg(all(debug_assertions, not(fuzzing)))]
        debug_assert!(self.open_dirty_pages.lock().is_empty());
        if self.needs_recovery.load(Ordering::Acquire) {
            return Err(StorageError::RecoveryRequired);
        }

        // Verify that recovery_required is set on disk. This is the invariant
        // that guarantees non-durable pages are reclaimed after a crash.
        // Without this flag, a crash would leave pages in `unpersisted`
        // allocated but unreferenced, permanently leaking disk space.
        if !self.state.lock().header.recovery_required {
            return Err(StorageError::Internal(alloc::string::String::from(
                "non_durable_commit requires recovery_required flag to be set \
                 for crash-safe page reclamation",
            )));
        }

        let mut unpersisted = self.unpersisted.lock();
        let mut allocated_since_commit = self.allocated_since_commit.lock();
        unpersisted.extend(allocated_since_commit.drain());
        allocated_since_commit.shrink_to_fit();
        self.storage.write_barrier()?;

        // Read blob state before locking `state` to avoid deadlock
        // (get_blob_state may lock `state` internally via get_committed_blob_state)
        let blob_state = self.get_blob_state();

        let mut state = self.state.lock();
        let secondary = state.header.secondary_slot_mut();
        secondary.transaction_id = transaction_id;
        secondary.user_root = data_root;
        secondary.system_root = system_root;
        secondary.blob_region_offset = blob_state.region_offset;
        secondary.blob_region_length = blob_state.region_length;
        secondary.blob_next_sequence = blob_state.next_sequence;
        secondary.blob_hlc_state = blob_state.hlc_state;

        // Upgrade to V5 when blob store is in use
        if blob_state.region_offset > 0 && secondary.version < FILE_FORMAT_VERSION5 {
            secondary.version = FILE_FORMAT_VERSION5;
        }

        // Signal readers to use the secondary slot until the next durable commit
        // promotes it to primary.
        self.read_from_secondary.store(true, Ordering::Release);

        Ok(())
    }

    pub(crate) fn rollback_uncommitted_writes(&self) -> Result {
        let result = self.rollback_uncommitted_writes_inner();
        if result.is_err() {
            self.mark_needs_recovery();
        }
        result
    }

    fn rollback_uncommitted_writes_inner(&self) -> Result {
        #[cfg(all(debug_assertions, not(fuzzing)))]
        {
            let dirty_pages = self.open_dirty_pages.lock();
            debug_assert!(
                dirty_pages.is_empty(),
                "Dirty pages outstanding: {dirty_pages:?}"
            );
        }
        if self.needs_recovery.load(Ordering::Acquire) {
            return Err(StorageError::RecoveryRequired);
        }
        let mut state = self.state.lock();
        let mut guard = self.allocated_since_commit.lock();
        for page_number in guard.iter() {
            let region_index = page_number.region;
            state
                .get_region_tracker_mut()
                .mark_free(page_number.page_order, region_index)?;
            state
                .get_region_mut(region_index)
                .free(page_number.page_index, page_number.page_order)?;
            #[cfg(debug_assertions)]
            // Tolerate missing entries: corrupted data may cause inconsistent tracking.
            self.allocated_pages.lock().remove(page_number);

            let address = page_number.address_range(
                self.page_size.into(),
                self.region_size,
                self.region_header_with_padding_size,
                self.page_size,
            );
            let len: usize = (address.end - address.start).try_into().unwrap();
            self.storage.invalidate_cache(address.start, len);
            self.storage.cancel_pending_write(address.start, len);
        }
        guard.clear();
        guard.shrink_to_fit();

        // Reset pending blob state so aborted writes don't leak sequence numbers
        // or region length to the next transaction
        *self.pending_blob_state.lock() = BlobCommitState::default();

        Ok(())
    }

    // Design: default hint is acceptable for cold paths. Hot paths already
    // provide explicit hints.
    pub(crate) fn get_page(&self, page_number: PageNumber) -> Result<PageImpl> {
        self.get_page_extended(page_number, PageHint::None)
    }

    pub(crate) fn get_page_extended(
        &self,
        page_number: PageNumber,
        hint: PageHint,
    ) -> Result<PageImpl> {
        let range = page_number.address_range(
            self.page_size.into(),
            self.region_size,
            self.region_header_with_padding_size,
            self.page_size,
        );
        let len: usize = (range.end - range.start).try_into().unwrap();
        let mem = self.storage.read(range.start, len, hint)?;

        // In single-writer mode, we should not read a page that we already have
        // opened for writing. However, concurrent verification may legitimately read
        // pages that a writer has marked dirty (the read returns the pre-CoW copy).
        // This check remains as a debug hint but is not an invariant violation.
        #[cfg(debug_assertions)]
        {
            let dirty_pages = self.open_dirty_pages.lock();
            if dirty_pages.contains(&page_number) {
                // This can happen during concurrent verify_integrity + writer.
                // The read is safe because CoW ensures the underlying data is stable.
            }
        }

        #[cfg(debug_assertions)]
        {
            *(self
                .read_page_ref_counts
                .lock()
                .entry(page_number)
                .or_default()) += 1;
        }

        Ok(PageImpl {
            mem,
            page_number,
            #[cfg(debug_assertions)]
            open_pages: self.read_page_ref_counts.clone(),
        })
    }

    // NOTE: the caller must ensure that the read cache has been invalidated or stale reads my occur
    pub(crate) fn get_page_mut(&self, page_number: PageNumber) -> Result<PageMut> {
        #[cfg(debug_assertions)]
        {
            // read_page_ref_counts not checked: a recycled page number can have
            // stale readers that hold Arc<[u8]> copies of the old data.
            debug_assert!(!self.open_dirty_pages.lock().contains(&page_number));
        }

        let address_range = page_number.address_range(
            self.page_size.into(),
            self.region_size,
            self.region_header_with_padding_size,
            self.page_size,
        );
        let len: usize = (address_range.end - address_range.start)
            .try_into()
            .unwrap();
        let mem = self.storage.write(address_range.start, len, false)?;

        #[cfg(debug_assertions)]
        {
            debug_assert!(self.open_dirty_pages.lock().insert(page_number));
        }

        Ok(PageMut {
            mem,
            page_number,
            #[cfg(debug_assertions)]
            open_pages: self.open_dirty_pages.clone(),
        })
    }

    pub(crate) fn get_version(&self) -> u8 {
        let state = self.state.lock();
        if self.read_from_secondary.load(Ordering::Acquire) {
            state.header.secondary_slot().version
        } else {
            state.header.primary_slot().version
        }
    }

    pub(crate) fn get_data_root(&self) -> Option<BtreeHeader> {
        let state = self.state.lock();
        if self.read_from_secondary.load(Ordering::Acquire) {
            state.header.secondary_slot().user_root
        } else {
            state.header.primary_slot().user_root
        }
    }

    pub(crate) fn get_system_root(&self) -> Option<BtreeHeader> {
        let state = self.state.lock();
        if self.read_from_secondary.load(Ordering::Acquire) {
            state.header.secondary_slot().system_root
        } else {
            state.header.primary_slot().system_root
        }
    }

    /// Returns the data root from the last **durable** commit (primary slot).
    /// Unlike `get_data_root()`, this ignores non-durable commits. Use this
    /// when verifying integrity to avoid racing with non-durable page freeing.
    #[cfg(feature = "std")]
    pub(crate) fn get_persisted_data_root(&self) -> Option<BtreeHeader> {
        let state = self.state.lock();
        state.header.primary_slot().user_root
    }

    /// Returns the system root from the last **durable** commit (primary slot).
    #[cfg(feature = "std")]
    pub(crate) fn get_persisted_system_root(&self) -> Option<BtreeHeader> {
        let state = self.state.lock();
        state.header.primary_slot().system_root
    }

    pub(crate) fn get_last_committed_transaction_id(&self) -> Result<TransactionId> {
        let state = self.state.lock();
        if self.read_from_secondary.load(Ordering::Acquire) {
            Ok(state.header.secondary_slot().transaction_id)
        } else {
            Ok(state.header.primary_slot().transaction_id)
        }
    }

    pub(crate) fn get_last_durable_transaction_id(&self) -> Result<TransactionId> {
        let state = self.state.lock();
        Ok(state.header.primary_slot().transaction_id)
    }

    pub(crate) fn free(&self, page: PageNumber, allocated: &mut PageTrackerPolicy) -> Result {
        self.allocated_since_commit.lock().remove(&page);
        self.free_helper(page, allocated)
    }

    /// Attempt to free a page. Returns `false` if the page has active read
    /// references (concurrent readers hold `PageImpl` handles), in which case
    /// the page is NOT freed and the caller should defer the free to a later
    /// commit.
    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    pub(crate) fn try_free(
        &self,
        page: PageNumber,
        allocated: &mut PageTrackerPolicy,
    ) -> Result<bool> {
        self.allocated_since_commit.lock().remove(&page);
        if self.read_page_ref_counts.lock().contains_key(&page) {
            return Ok(false);
        }
        self.free_helper(page, allocated)?;
        Ok(true)
    }

    /// Free the page if it is in the unpersisted set. Returns true if freed.
    ///
    /// NOTE: This does not check `read_page_ref_counts`. Freeing a page that has
    /// active readers is safe because `PageImpl` holds an `Arc<[u8]>` copy of the
    /// data -- the allocator reclamation doesn't invalidate existing readers.
    pub(crate) fn free_if_unpersisted(
        &self,
        page: PageNumber,
        allocated: &mut PageTrackerPolicy,
    ) -> Result<bool> {
        if self.unpersisted.lock().remove(&page) {
            self.free_helper(page, allocated)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn free_helper(&self, page: PageNumber, allocated: &mut PageTrackerPolicy) -> Result {
        #[cfg(debug_assertions)]
        {
            // Idempotent: during crash recovery on corrupted data, a page may
            // appear in multiple B-tree paths and be freed more than once.
            // The remove may return false -- that is tolerated.
            self.allocated_pages.lock().remove(&page);
            // open_dirty_pages is always consistent (not derived from on-disk data).
            // During fuzzing, simulated IO errors can cause inconsistent state.
            #[cfg(not(fuzzing))]
            debug_assert!(!self.open_dirty_pages.lock().contains(&page));
        }
        allocated.remove(page);
        let mut state = self.state.lock();
        let region_index = page.region;
        // Free in the regional allocator
        state
            .get_region_mut(region_index)
            .free(page.page_index, page.page_order)?;
        // Ensure that the region is marked as having free space
        state
            .get_region_tracker_mut()
            .mark_free(page.page_order, region_index)?;

        let address_range = page.address_range(
            self.page_size.into(),
            self.region_size,
            self.region_header_with_padding_size,
            self.page_size,
        );
        let len: usize = (address_range.end - address_range.start)
            .try_into()
            .unwrap();
        self.storage.invalidate_cache(address_range.start, len);
        self.storage.cancel_pending_write(address_range.start, len);
        Ok(())
    }

    // Frees the page if it was allocated since the last commit. Returns true, if the page was freed
    pub(crate) fn free_if_uncommitted(
        &self,
        page: PageNumber,
        allocated: &mut PageTrackerPolicy,
    ) -> Result<bool> {
        if self.allocated_since_commit.lock().remove(&page) {
            self.free_helper(page, allocated)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // Page has not been committed
    pub(crate) fn uncommitted(&self, page: PageNumber) -> bool {
        self.allocated_since_commit.lock().contains(&page)
    }

    /// Drain all pages from the uncommitted set and return them.
    ///
    /// Used by `restore_savepoint` to reclaim pages allocated during a
    /// rolled-back write that are now unreachable from any tree root.
    pub(crate) fn drain_uncommitted(&self) -> Vec<PageNumber> {
        self.allocated_since_commit.lock().drain().collect()
    }

    pub(crate) fn unpersisted(&self, page: PageNumber) -> bool {
        self.unpersisted.lock().contains(&page)
    }

    pub(crate) fn allocate_helper(
        &self,
        allocation_size: usize,
        lowest: bool,
        transactional: bool,
    ) -> Result<PageMut> {
        let required_pages = allocation_size.div_ceil(self.get_page_size());
        let required_order = ceil_log2(required_pages);

        let mut state = self.state.lock();

        let page_number = if let Some(page_number) =
            Self::allocate_helper_retry(&mut state, required_order, lowest)?
        {
            page_number
        } else {
            self.grow(&mut state, required_order)?;
            Self::allocate_helper_retry(&mut state, required_order, lowest)?.unwrap()
        };

        #[cfg(all(debug_assertions, not(fuzzing)))]
        {
            debug_assert!(self.allocated_pages.lock().insert(page_number));
            debug_assert!(
                !self.read_page_ref_counts.lock().contains_key(&page_number),
                "Allocated a page that is still referenced! {page_number:?}"
            );
            debug_assert!(!self.open_dirty_pages.lock().contains(&page_number));
        }

        if transactional {
            self.allocated_since_commit.lock().insert(page_number);
        }

        let address_range = page_number.address_range(
            self.page_size.into(),
            self.region_size,
            self.region_header_with_padding_size,
            self.page_size,
        );
        let len: usize = (address_range.end - address_range.start)
            .try_into()
            .unwrap();

        #[allow(unused_mut)]
        let mut mem = self.storage.write(address_range.start, len, true)?;
        debug_assert!(mem.mem().len() >= allocation_size);

        #[cfg(debug_assertions)]
        {
            debug_assert!(self.open_dirty_pages.lock().insert(page_number));

            // Poison the memory in debug mode to help detect uninitialized reads
            mem.mem_mut()?.fill(0xFF);
        }

        Ok(PageMut {
            mem,
            page_number,
            #[cfg(debug_assertions)]
            open_pages: self.open_dirty_pages.clone(),
        })
    }

    fn allocate_helper_retry(
        state: &mut InMemoryState,
        required_order: u8,
        lowest: bool,
    ) -> Result<Option<PageNumber>> {
        loop {
            let Some(candidate_region) = state.get_region_tracker_mut().find_free(required_order)
            else {
                return Ok(None);
            };
            let region = state.get_region_mut(candidate_region);
            let r = if lowest {
                region.alloc_lowest(required_order)?
            } else {
                region.alloc(required_order)?
            };
            if let Some(page) = r {
                return Ok(Some(PageNumber::new(
                    candidate_region,
                    page,
                    required_order,
                )));
            }
            // Mark the region, if it's full
            state
                .get_region_tracker_mut()
                .mark_full(required_order, candidate_region)?;
        }
    }

    fn try_shrink(state: &mut InMemoryState, force: bool) -> Result<bool> {
        let layout = state.header.layout();
        let last_region_index = layout.num_regions() - 1;
        let last_allocator = state.get_region(last_region_index);
        let trailing_free = last_allocator.trailing_free_pages()?;
        let last_allocator_len = last_allocator.len();
        if trailing_free == 0 {
            return Ok(false);
        }
        if trailing_free < last_allocator_len / 2 && !force {
            return Ok(false);
        }
        let reduce_by = if layout.num_regions() > 1 && trailing_free == last_allocator_len {
            trailing_free
        } else if force {
            // Do not shrink the database to zero size
            min(last_allocator_len - 1, trailing_free)
        } else {
            trailing_free / 2
        };

        let mut new_layout = layout;
        new_layout.reduce_last_region(reduce_by);
        state.allocators.resize_to(new_layout)?;
        if new_layout.len() > layout.len() {
            return Err(StorageError::Internal(alloc::string::String::from(
                "Shrink produced a layout larger than the original",
            )));
        }
        state.header.set_layout(new_layout);

        Ok(true)
    }

    fn grow(&self, state: &mut InMemoryState, required_order_allocation: u8) -> Result<()> {
        let layout = state.header.layout();
        let required_growth =
            2u64.pow(required_order_allocation.into()) * u64::from(state.header.page_size());
        let max_region_size = u64::from(state.header.layout().full_region_layout().num_pages())
            * u64::from(state.header.page_size());
        let next_desired_size = if layout.num_full_regions() > 0 {
            if let Some(trailing) = layout.trailing_region_layout() {
                if 2 * required_growth < max_region_size - trailing.usable_bytes() {
                    // Fill out the trailing region
                    layout.usable_bytes() + (max_region_size - trailing.usable_bytes())
                } else {
                    // Fill out trailing & Grow by 1 region
                    layout.usable_bytes() + 2 * max_region_size - trailing.usable_bytes()
                }
            } else {
                // Grow by 1 region
                layout.usable_bytes() + max_region_size
            }
        } else {
            max(
                layout.usable_bytes() * 2,
                layout.usable_bytes() + required_growth * 2,
            )
        };

        let new_layout = DatabaseLayout::calculate(
            next_desired_size,
            state.header.layout().full_region_layout().num_pages(),
            state
                .header
                .layout()
                .full_region_layout()
                .get_header_pages(),
            self.page_size,
        );
        if new_layout.len() < layout.len() {
            return Err(StorageError::Internal(alloc::string::String::from(
                "Grow produced a layout smaller than the original",
            )));
        }

        // Determine the effective blob boundary. The pending state (current
        // transaction) takes priority; if empty, fall back to the committed
        // header. We must not grow the B-tree layout into either.
        let mut pending_blob = self.pending_blob_state.lock();
        let (blob_offset, blob_len) = if pending_blob.region_offset > 0 {
            (pending_blob.region_offset, pending_blob.region_length)
        } else {
            let slot = if self.read_from_secondary.load(Ordering::Acquire) {
                state.header.secondary_slot()
            } else {
                state.header.primary_slot()
            };
            (slot.blob_region_offset, slot.blob_region_length)
        };

        // If the new B-tree layout would overlap the blob region, relocate
        // the blob data past the new layout boundary.
        if blob_offset > 0 && new_layout.len() > blob_offset {
            let new_blob_offset = new_layout.len();
            if blob_len > 0 {
                #[allow(clippy::cast_possible_truncation)]
                let old_data = self.storage.read_direct(blob_offset, blob_len as usize)?;
                self.storage.ensure_len(new_blob_offset + blob_len)?;
                self.storage.write_direct(new_blob_offset, &old_data)?;
            }
            // Update pending state so commit_inner writes the new offset.
            // Carry forward all other fields from the effective state.
            if pending_blob.region_offset > 0 {
                pending_blob.region_offset = new_blob_offset;
            } else {
                let slot = if self.read_from_secondary.load(Ordering::Acquire) {
                    state.header.secondary_slot()
                } else {
                    state.header.primary_slot()
                };
                *pending_blob = BlobCommitState {
                    region_offset: new_blob_offset,
                    region_length: slot.blob_region_length,
                    next_sequence: slot.blob_next_sequence,
                    hlc_state: slot.blob_hlc_state,
                };
            }
        }
        let file_target = if pending_blob.region_offset > 0 {
            new_layout
                .len()
                .max(pending_blob.region_offset + pending_blob.region_length)
        } else if blob_offset > 0 {
            // Committed blob not relocated (layout fits before it)
            new_layout.len().max(blob_offset + blob_len)
        } else {
            new_layout.len()
        };
        drop(pending_blob);

        // Growing the file overwrites the EOF mirror; the next commit will rewrite it.
        self.eof_mirror_size.store(0, Ordering::Release);
        let result = self.storage.resize(file_target);
        if result.is_err() {
            self.mark_needs_recovery();
            return result;
        }

        state.allocators.resize_to(new_layout)?;
        state.header.set_layout(new_layout);
        Ok(())
    }

    pub(crate) fn allocate(
        &self,
        allocation_size: usize,
        allocated: &mut PageTrackerPolicy,
    ) -> Result<PageMut> {
        let result = self.allocate_helper(allocation_size, false, true);
        if let Ok(ref page) = result {
            allocated.insert(page.get_page_number());
        }
        result
    }

    pub(crate) fn allocate_lowest(&self, allocation_size: usize) -> Result<PageMut> {
        self.allocate_helper(allocation_size, true, true)
    }

    pub(crate) fn count_allocated_pages(&self) -> Result<u64> {
        let state = self.state.lock();
        let mut count = 0u64;
        for i in 0..state.header.layout().num_regions() {
            count += u64::from(state.get_region(i).count_allocated_pages());
        }

        Ok(count)
    }

    pub(crate) fn count_free_pages(&self) -> Result<u64> {
        let state = self.state.lock();
        let mut count = 0u64;
        for i in 0..state.header.layout().num_regions() {
            count += u64::from(state.get_region(i).count_free_pages());
        }

        Ok(count)
    }

    pub(crate) fn trailing_free_pages(&self) -> Result<u64> {
        let state = self.state.lock();
        let layout = state.header.layout();
        if layout.num_regions() == 0 {
            return Ok(0);
        }
        let last_region = layout.num_regions() - 1;
        Ok(u64::from(
            state.get_region(last_region).trailing_free_pages()?,
        ))
    }

    pub(crate) fn get_page_size(&self) -> usize {
        self.page_size.try_into().unwrap()
    }

    /// Flush all pending writes to the underlying storage
    #[cfg(feature = "std")]
    pub(crate) fn flush_data(&self) -> Result {
        self.storage.flush()
    }

    /// Read raw bytes from the underlying storage, bypassing the cache
    #[cfg(feature = "std")]
    pub(crate) fn read_raw(&self, offset: u64, buf: &mut [u8]) -> Result {
        let data = self.storage.read_direct(offset, buf.len())?;
        buf.copy_from_slice(&data);
        Ok(())
    }

    /// Get the raw file length of the underlying storage
    #[cfg(feature = "std")]
    pub(crate) fn raw_len(&self) -> Result<u64> {
        self.storage.raw_file_len()
    }

    pub(crate) fn close(&self) -> Result {
        let is_panicking = {
            #[cfg(feature = "std")]
            {
                std::thread::panicking()
            }
            #[cfg(not(feature = "std"))]
            {
                false
            }
        };
        if !self.needs_recovery.load(Ordering::Acquire) && !is_panicking {
            let mut state = self.state.lock();
            if self.storage.flush().is_ok() {
                state.header.recovery_required = false;
                self.write_header(&state.header)?;
                self.storage.flush()?;
            }
        }

        self.storage.close()?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::tree_store::page_store::page_manager::INITIAL_REGIONS;
    use crate::{Database, TableDefinition};

    // Test that the region tracker expansion code works, by adding more data than fits into the initial max regions
    #[test]
    fn out_of_regions() {
        let tmpfile = crate::create_tempfile();
        let table_definition: TableDefinition<u32, &[u8]> = TableDefinition::new("x");
        let page_size = 1024;
        let big_value = vec![0u8; 5 * page_size];

        let db = Database::builder()
            .set_region_size((8 * page_size).try_into().unwrap())
            .set_page_size(page_size)
            .create(tmpfile.path())
            .unwrap();

        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(table_definition).unwrap();
            for i in 0..=INITIAL_REGIONS {
                table.insert(&i, big_value.as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
        drop(db);

        let mut db = Database::builder()
            .set_region_size((8 * page_size).try_into().unwrap())
            .set_page_size(page_size)
            .open(tmpfile.path())
            .unwrap();
        assert!(db.check_integrity().unwrap());
    }

    // Make sure the database remains consistent after a panic
    #[test]
    #[cfg(panic = "unwind")]
    fn panic() {
        let tmpfile = crate::create_tempfile();
        let table_definition: TableDefinition<u32, &[u8]> = TableDefinition::new("x");

        let _ = std::panic::catch_unwind(|| {
            let db = Database::create(&tmpfile).unwrap();
            let txn = db.begin_write().unwrap();
            txn.open_table(table_definition).unwrap();
            panic!();
        });

        let mut db = Database::open(tmpfile).unwrap();
        assert!(db.check_integrity().unwrap());
    }
}
