// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[cfg(feature = "tracing")]
use crate::bf_tree::nodes::PageID;
use crate::bf_tree::{
    circular_buffer::TombstoneHandle,
    error::{IoErrorKind, TreeError},
    fs::{VfsImpl, buffer_alloc, buffer_dealloc},
    nodes::{
        DISK_PAGE_SIZE,
        leaf_node::{
            GetScanRecordByPosResult, LeafNode, LeafReadResult as LeafResult, MiniPageNextLevel,
            OpType,
        },
    },
    range_scan::{ScanError, ScanPosition, ScanReturnField},
    storage::{LeafStorage, PageLocation},
    sync::atomic::AtomicUsize,
    tree::key_value_physical_size,
    utils::stats::LeafStats,
    utils::{
        inner_lock::ReadGuard,
        rw_lock::{RwLockReadGuard, RwLockWriteGuard},
    },
};
#[allow(unused_imports)]
use crate::{check_parent, counter, histogram, info};
use alloc::{alloc::Layout, boxed::Box, vec::Vec};
use core::ops::{Deref, DerefMut};
use core::panic;

struct TmpBuffer {
    is_dirty: bool,
    size: usize, // Equals to leaf base page size
    ptr: *mut u8,
}

impl TmpBuffer {
    fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, DISK_PAGE_SIZE).unwrap();
        let buffer = buffer_alloc(layout);
        Self {
            is_dirty: false,
            size,
            ptr: buffer,
        }
    }

    fn from_leaf_node(leaf: *mut LeafNode, size: usize) -> Self {
        let mut buffer = Self::new(size);
        // SAFETY: The caller provides a valid `leaf` pointer to a LeafNode allocation.
        // We only read the `node_size` field from the header, which is within bounds.
        assert!(unsafe { &*leaf }.meta.node_size as usize == size);

        // SAFETY: `leaf` points to a valid LeafNode of `size` bytes (just asserted above).
        // `buffer.ptr` was freshly allocated with the same `size`. The two allocations
        // do not overlap, so copy_nonoverlapping is sound.
        unsafe {
            core::ptr::copy_nonoverlapping(leaf as *const u8, buffer.ptr, size);
        }
        buffer.is_dirty = true;

        buffer
    }

    fn as_u8_slice_mut(&mut self) -> &mut [u8] {
        // SAFETY: `self.ptr` was allocated via `buffer_alloc` with `self.size` bytes and
        // DISK_PAGE_SIZE alignment. We hold `&mut self`, so no aliasing references exist.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    fn as_u8_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` was allocated via `buffer_alloc` with `self.size` bytes.
        // The buffer remains valid for the lifetime of `&self`.
        unsafe { core::slice::from_raw_parts(self.ptr, self.size) }
    }

    fn as_leaf_node(&self) -> &LeafNode {
        // SAFETY: `self.ptr` was allocated with at least `size_of::<LeafNode>()` bytes and
        // DISK_PAGE_SIZE alignment (which exceeds LeafNode alignment). The buffer contents
        // were initialized as a valid LeafNode (via copy or init_node_with_fence).
        unsafe { &*(self.ptr as *const LeafNode) }
    }

    fn as_leaf_node_mut(&mut self) -> &mut LeafNode {
        self.is_dirty = true;
        // SAFETY: Same alignment and size guarantees as `as_leaf_node`. We hold `&mut self`,
        // ensuring exclusive access to the underlying buffer.
        unsafe { &mut *(self.ptr as *mut LeafNode) }
    }
}

impl Drop for TmpBuffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, DISK_PAGE_SIZE).unwrap();
        buffer_dealloc(self.ptr, layout);
    }
}

pub(crate) trait LeafOperations {
    /// Returns the base page from the already-populated temporary buffer.
    ///
    /// # Panics
    ///
    /// Panics if `tmp_buffer` has not been populated by a prior `load_base_page` call.
    /// Callers must ensure the buffer is loaded before invoking this method.
    fn load_base_page_from_buffer(&self) -> &LeafNode;

    fn get_page_location(&self) -> &PageLocation;

    fn load_base_page(&mut self, offset: usize) -> Result<&LeafNode, TreeError>;

    fn load_cache_page(&self, ptr: *mut LeafNode) -> &LeafNode {
        // SAFETY: `ptr` comes from a PageLocation::Mini or PageLocation::Full entry,
        // which holds a pointer to a live CircularBuffer allocation containing a valid
        // LeafNode. The allocation outlives the returned reference due to lock-guarded access.
        unsafe { &*ptr }
    }

    fn scan_record_by_pos_with_bound(
        &self,
        pos: &ScanPosition,
        out_buffer: &mut [u8],
        return_field: ScanReturnField,
        end_key: &Option<Vec<u8>>,
    ) -> GetScanRecordByPosResult {
        match pos {
            ScanPosition::Base(pos) => {
                let base = self.load_base_page_from_buffer();
                base.get_record_by_pos_with_bound(*pos, out_buffer, return_field, end_key)
            }
            ScanPosition::Full(pos) => {
                let page_ptr = match self.get_page_location() {
                    PageLocation::Base(_) | PageLocation::Mini(_) => unreachable!(),
                    PageLocation::Full(ptr) => *ptr,
                    PageLocation::Null => {
                        debug_assert!(false, "scan_value_by_pos on Null page");
                        return GetScanRecordByPosResult::EndOfLeaf;
                    }
                };
                let full = self.load_cache_page(page_ptr);
                full.get_record_by_pos_with_bound(*pos, out_buffer, return_field, end_key)
            }
        }
    }

    fn get_right_sibling(&mut self) -> Result<Vec<u8>, TreeError> {
        let page_loc = self.get_page_location();
        match page_loc {
            PageLocation::Base(offset) => {
                let base_ref = self.load_base_page(*offset)?;
                Ok(base_ref.get_high_fence_key())
            }
            PageLocation::Full(ptr) => {
                let page_ref = self.load_cache_page(*ptr);
                Ok(page_ref.get_high_fence_key())
            }
            PageLocation::Mini(ptr) => {
                let mini = self.load_cache_page(*ptr);
                let offset = mini.next_level.as_offset();
                let base_ref = self.load_base_page(offset)?;
                Ok(base_ref.get_high_fence_key())
            }
            PageLocation::Null => Err(TreeError::IoError(IoErrorKind::Corruption)),
        }
    }

    fn cache_page_about_to_evict(&self, storage: &LeafStorage) -> bool {
        match self.get_page_location() {
            PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                let mini_page = self.load_cache_page(*ptr);
                storage.mini_page_copy_on_access(mini_page)
            }
            PageLocation::Base(_) => false,
            PageLocation::Null => {
                debug_assert!(false, "cache_page_about_to_evict on Null page");
                false
            }
        }
    }

    fn read(
        &mut self,
        key: &[u8],
        out_buffer: &mut [u8],
        mini_page_binary_search: bool,
        cache_only: bool,
    ) -> Result<ReadResult, TreeError> {
        let page_loc = self.get_page_location();

        let next_level = match page_loc {
            PageLocation::Base(offset) => {
                let base_ref = self.load_base_page(*offset)?;
                let out = base_ref.read_by_key(key, out_buffer);
                return Ok(ReadResult::Base(out));
            }
            PageLocation::Full(ptr) => {
                counter!(FullPageRead);
                let base_ref = self.load_cache_page(*ptr);
                let out = base_ref.read_by_key(key, out_buffer);
                histogram!(HitMiniPage, base_ref.meta.node_size as u64);
                return Ok(ReadResult::Full(out));
            }
            PageLocation::Mini(ptr) => {
                counter!(MiniPageRead);
                let mini_page = self.load_cache_page(*ptr);
                let out = mini_page.read_by_key_inner(key, out_buffer, mini_page_binary_search);
                histogram!(HitMiniPage, mini_page.meta.node_size as u64);
                match out {
                    LeafResult::Found(_) | LeafResult::Deleted | LeafResult::InvalidKey => {
                        return Ok(ReadResult::Mini(out));
                    }
                    LeafResult::NotFound => {
                        // In cache only mode, we return not found
                        if cache_only {
                            return Ok(ReadResult::None);
                        }

                        // fall through
                        counter!(MiniPageReadMiss);
                        mini_page.next_level
                    }
                }
            }
            PageLocation::Null => {
                return Ok(ReadResult::None);
            }
        };

        let base_ref = self.load_base_page(next_level.as_offset())?;

        let out = base_ref.read_by_key(key, out_buffer);
        Ok(ReadResult::Base(out))
    }

    /// Returns the pos of the key when scanning, i.e., the keys[pos] >= key.
    ///
    /// If we hit a page with mini page, we need to merge it, consolidate it, then scan.
    fn get_scan_position(&mut self, key: &[u8]) -> Result<ScanPosition, ScanError> {
        let page_loc = self.get_page_location();

        match page_loc {
            PageLocation::Base(offset) => {
                let base_ref = self.load_base_page(*offset).map_err(|e| match e {
                    TreeError::IoError(io) => ScanError::Io(io),
                    _ => ScanError::Io(IoErrorKind::Corruption),
                })?;
                let pos = base_ref.lower_bound(key);
                Ok(ScanPosition::Base(pos as u32))
            }
            PageLocation::Full(ptr) => {
                let page_ref = self.load_cache_page(*ptr);
                let pos = page_ref.lower_bound(key);
                Ok(ScanPosition::Full(pos as u32))
            }
            PageLocation::Mini(_ptr) => Err(ScanError::NeedMergeMiniPage),
            PageLocation::Null => Err(ScanError::Io(IoErrorKind::Corruption)),
        }
    }
}

/// CRC-32 checksum size in bytes, stored at the end of each disk page.
const CRC32_SIZE: usize = 4;

/// Page table read locked entry.
pub(crate) struct LeafEntrySLocked<'a> {
    raw_guard: RwLockReadGuard<'a, PageLocation>,
    #[cfg(feature = "tracing")]
    pid: PageID,
    file_handle: &'a dyn VfsImpl,
    tmp_buffer_size: usize,
    tmp_buffer: Option<TmpBuffer>,
    verify_checksums: bool,
    cow_boundary: &'a AtomicUsize,
}

impl LeafOperations for LeafEntrySLocked<'_> {
    fn load_base_page_from_buffer(&self) -> &LeafNode {
        let l = self
            .tmp_buffer
            .as_ref()
            .expect("load_base_page_from_buffer called before load_base_page populated the buffer")
            .as_leaf_node();
        debug_assert!(l.is_base_page());
        l
    }

    fn get_page_location(&self) -> &PageLocation {
        self.raw_guard.deref()
    }

    fn load_base_page(&mut self, offset: usize) -> Result<&LeafNode, TreeError> {
        match self.tmp_buffer {
            Some(ref buffer) => Ok(buffer.as_leaf_node()),
            None => {
                let mut buffer = TmpBuffer::new(self.tmp_buffer_size);
                let slice = buffer.as_u8_slice_mut();
                self.file_handle
                    .read(offset, slice)
                    .map_err(|_| TreeError::IoError(IoErrorKind::VfsRead { offset }))?;

                if self.verify_checksums && slice.len() >= CRC32_SIZE {
                    let data_end = slice.len() - CRC32_SIZE;
                    let stored = u32::from_le_bytes(
                        slice[data_end..data_end + CRC32_SIZE].try_into().unwrap(),
                    );
                    // stored == 0 means the page was written before checksums
                    // were enabled (alloc_zeroed produces trailing zeros).
                    if stored != 0 {
                        let computed = crate::bf_tree::utils::crc32::crc32(&slice[..data_end]);
                        if stored != computed {
                            return Err(TreeError::IoError(IoErrorKind::ChecksumMismatch {
                                offset,
                            }));
                        }
                    }
                }

                self.tmp_buffer = Some(buffer);
                self.load_base_page(offset)
            }
        }
    }
}

impl<'a> LeafEntrySLocked<'a> {
    pub(crate) fn new(
        guard: RwLockReadGuard<'a, PageLocation>,
        #[cfg(feature = "tracing")] pid: PageID,
        file_handle: &'a dyn VfsImpl,
        leaf_page_size: usize,
        verify_checksums: bool,
        cow_boundary: &'a AtomicUsize,
    ) -> Self {
        Self {
            raw_guard: guard,
            #[cfg(feature = "tracing")]
            pid,
            file_handle,
            tmp_buffer_size: leaf_page_size,
            tmp_buffer: None,
            verify_checksums,
            cow_boundary,
        }
    }

    /// Upgrade the reader lock to writer lock.
    pub(crate) fn try_upgrade(mut self) -> Result<LeafEntryXLocked<'a>, LeafEntrySLocked<'a>> {
        match self.raw_guard.try_upgrade() {
            Ok(g) => {
                let x = LeafEntryXLocked {
                    raw_guard: g,
                    #[cfg(feature = "tracing")]
                    pid: self.pid,
                    file_handle: self.file_handle,
                    tmp_buffer_size: self.tmp_buffer_size,
                    tmp_buffer: self.tmp_buffer.take(),
                    verify_checksums: self.verify_checksums,
                    cow_boundary: self.cow_boundary,
                };
                // here we don't need to explicitly forget self.
                return Ok(x);
            }
            Err(e) => {
                self.raw_guard = e;
            }
        };
        Err(self)
    }
}

pub(crate) enum ReadResult {
    Mini(LeafResult),
    Full(LeafResult),
    Base(LeafResult),
    None,
}

/// Page table write locked entry.
pub(crate) struct LeafEntryXLocked<'a> {
    raw_guard: RwLockWriteGuard<'a, PageLocation>,
    #[cfg(feature = "tracing")]
    pid: PageID,
    file_handle: &'a dyn VfsImpl,
    tmp_buffer_size: usize,
    tmp_buffer: Option<TmpBuffer>,
    verify_checksums: bool,
    /// COW boundary: offsets below this value belong to a snapshot and must
    /// not be overwritten. The Drop impl redirects such writes to new offsets.
    cow_boundary: &'a AtomicUsize,
}

impl Drop for LeafEntryXLocked<'_> {
    fn drop(&mut self) {
        // Determine the disk offset first (requires borrowing self.raw_guard and
        // potentially self for load_cache_page_mut), then mutate the buffer for
        // the checksum, then write. This two-phase approach avoids a simultaneous
        // mutable borrow of self.tmp_buffer and immutable borrow of self.
        let dirty = self.tmp_buffer.as_ref().is_some_and(|b| b.is_dirty);

        if !dirty {
            return;
        }

        let boundary = self
            .cow_boundary
            .load(core::sync::atomic::Ordering::Acquire);

        let write_offset = match self.raw_guard.deref() {
            PageLocation::Base(offset) => {
                let offset = *offset;
                if boundary > 0 && offset < boundary {
                    // COW: base page offset is in the snapshot region.
                    // Allocate a new offset and update the page table entry.
                    let new_offset = self.file_handle.alloc_offset(self.tmp_buffer_size);
                    *self.raw_guard.deref_mut() = PageLocation::Base(new_offset);
                    new_offset
                } else {
                    offset
                }
            }
            PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                let ptr = *ptr;
                let mini_page = self.load_cache_page_mut(ptr);
                let offset = mini_page.next_level.as_offset();
                if boundary > 0 && offset < boundary {
                    // COW: the mini/full page's base offset is in the snapshot
                    // region. Allocate a new offset and update next_level.
                    // We must NOT change the PageLocation variant (Mini/Full)
                    // because the circular buffer owns that entry.
                    let new_offset = self.file_handle.alloc_offset(self.tmp_buffer_size);
                    mini_page.next_level =
                        crate::bf_tree::nodes::leaf_node::MiniPageNextLevel::new(new_offset);
                    new_offset
                } else {
                    offset
                }
            }
            PageLocation::Null => {
                debug_assert!(false, "Dropping a tmp buffer of a Null page");
                return;
            }
        };

        if let Some(ref mut b) = self.tmp_buffer {
            // Compute and store CRC-32 checksum in the last 4 bytes of the page.
            if self.verify_checksums {
                let slice = b.as_u8_slice_mut();
                if slice.len() >= CRC32_SIZE {
                    let data_end = slice.len() - CRC32_SIZE;
                    let checksum = crate::bf_tree::utils::crc32::crc32(&slice[..data_end]);
                    slice[data_end..data_end + CRC32_SIZE].copy_from_slice(&checksum.to_le_bytes());
                }
            }

            let slice = b.as_u8_slice();
            // Best-effort page write during Drop. Durability is guaranteed by
            // the WAL; this write is an optimization to avoid re-reading the WAL
            // on recovery. If it fails, the next snapshot or WAL replay will
            // reconstruct this page. We cannot propagate errors from Drop.
            let _ = self.file_handle.write(write_offset, slice);
        }
    }
}

impl LeafOperations for LeafEntryXLocked<'_> {
    fn load_base_page_from_buffer(&self) -> &LeafNode {
        let l = self
            .tmp_buffer
            .as_ref()
            .expect("load_base_page_from_buffer called before load_base_page populated the buffer")
            .as_leaf_node();
        debug_assert!(l.is_base_page());
        l
    }

    fn get_page_location(&self) -> &PageLocation {
        self.raw_guard.deref()
    }

    fn load_base_page(&mut self, offset: usize) -> Result<&LeafNode, TreeError> {
        match self.tmp_buffer {
            Some(ref mut buffer) => return Ok(buffer.as_leaf_node()),
            None => {
                let mut buffer = TmpBuffer::new(self.tmp_buffer_size);
                let slice = buffer.as_u8_slice_mut();
                self.file_handle
                    .read(offset, slice)
                    .map_err(|_| TreeError::IoError(IoErrorKind::VfsRead { offset }))?;

                if self.verify_checksums && slice.len() >= CRC32_SIZE {
                    let data_end = slice.len() - CRC32_SIZE;
                    let stored = u32::from_le_bytes(
                        slice[data_end..data_end + CRC32_SIZE].try_into().unwrap(),
                    );
                    if stored != 0 {
                        let computed = crate::bf_tree::utils::crc32::crc32(&slice[..data_end]);
                        if stored != computed {
                            return Err(TreeError::IoError(IoErrorKind::ChecksumMismatch {
                                offset,
                            }));
                        }
                    }
                }

                self.tmp_buffer = Some(buffer);

                if let Some(ref b) = self.tmp_buffer {
                    return Ok(b.as_leaf_node());
                }
            }
        };

        unreachable!();
    }
}

impl<'a> LeafEntryXLocked<'a> {
    pub(crate) fn new(
        raw_guard: RwLockWriteGuard<'a, PageLocation>,
        #[cfg(feature = "tracing")] page_id: PageID,
        file_handle: &'a dyn VfsImpl,
        tmp_buffer_size: usize,
        verify_checksums: bool,
        cow_boundary: &'a AtomicUsize,
    ) -> Self {
        Self {
            raw_guard,
            file_handle,
            #[cfg(feature = "tracing")]
            pid: page_id,
            tmp_buffer_size,
            tmp_buffer: None,
            verify_checksums,
            cow_boundary,
        }
    }

    pub(crate) fn with_buffer(
        raw_guard: RwLockWriteGuard<'a, PageLocation>,
        file_handle: &'a dyn VfsImpl,
        #[cfg(feature = "tracing")] page_id: PageID,
        tmp_buffer_size: usize,
        leaf_buffer: *mut LeafNode,
        verify_checksums: bool,
        cow_boundary: &'a AtomicUsize,
    ) -> Self {
        Self {
            raw_guard,
            file_handle,
            #[cfg(feature = "tracing")]
            pid: page_id,
            tmp_buffer_size,
            tmp_buffer: Some(TmpBuffer::from_leaf_node(leaf_buffer, tmp_buffer_size)),
            verify_checksums,
            cow_boundary,
        }
    }

    /// Attempt to downgrade an exclusive lock to a shared lock.
    ///
    /// This is not currently supported because the underlying `RwLock`
    /// does not provide an atomic downgrade path. Returns `Err(self)` so
    /// the caller retains the exclusive guard and can retry with a fresh
    /// read acquisition.
    #[allow(dead_code)]
    pub(crate) fn downgrade(self) -> Result<LeafEntrySLocked<'a>, Self> {
        Err(self)
    }

    pub(crate) fn dealloc_self(&mut self, storage: &LeafStorage, cache_only: bool) {
        let page_loc = self.raw_guard.deref().clone();
        match page_loc {
            PageLocation::Base(offset) => {
                if let Ok(page) = self.load_base_page(offset) {
                    debug_assert!(page.next_level.is_null());
                }
                self.file_handle.dealloc_offset(offset);
            }
            PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                let leaf_node = self.load_cache_page_mut(ptr);
                let h = match storage.begin_dealloc_mini_page(leaf_node) {
                    Ok(h) => h,
                    Err(_) => {
                        debug_assert!(false, "begin_dealloc_mini_page failed in dealloc path");
                        return;
                    }
                };
                let base_page = leaf_node.next_level;
                storage.finish_dealloc_mini_page(h);

                // In cache-only mode, there is no base page allocated
                if !cache_only {
                    self.file_handle.dealloc_offset(base_page.as_offset());
                }
            }
            PageLocation::Null => {}
        }
        // we need to drop buffer here, o.w., the buffer will be write to offset, which is no longer accessible.
        let buffer = self.tmp_buffer.take();
        drop(buffer);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn insert(
        &mut self,
        key: &[u8],
        value: &[u8],
        parent: Option<ReadGuard<'_>>,
        op_type: OpType,
        storage: &LeafStorage,
        write_load_full_page: &bool,
        cache_only: &bool,
        mini_page_size_classes: &[usize],
    ) -> Result<(), TreeError> {
        let page_loc = self.raw_guard.deref().clone();
        match page_loc {
            PageLocation::Base(offset) => {
                if *cache_only {
                    return Err(TreeError::IoError(IoErrorKind::CacheOnlyViolation));
                }

                // Root leaf node does not have a corresponding mini-page
                if parent.is_none() {
                    let success = self.load_base_page_mut()?.insert(
                        key,
                        value,
                        op_type,
                        storage.config.max_fence_len,
                    );
                    if !success {
                        self.load_base_page_mut()?.set_split_flag();
                        return Err(TreeError::NeedRestart);
                    } else {
                        return Ok(());
                    }
                }

                let mini_page_size = LeafNode::get_chain_size_hint(
                    key.len(),
                    value.len(),
                    mini_page_size_classes,
                    *cache_only,
                )
                .map_err(TreeError::IoError)?;
                let mini_page_guard = storage.alloc_mini_page(mini_page_size)?;

                LeafNode::initialize_mini_page(
                    &mini_page_guard,
                    mini_page_size,
                    MiniPageNextLevel::new(offset),
                    false,
                );

                let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
                let mini_loc = PageLocation::Mini(new_mini_ptr);
                self.create_cache_page_loc(mini_loc);

                let mini_page_ref = self.load_cache_page_mut(new_mini_ptr);
                let insert_success = mini_page_ref.insert(key, value, op_type, 0);
                if !insert_success {
                    return Err(TreeError::IoError(IoErrorKind::InvariantViolation));
                }
                counter!(InsertCreatedMiniPage);

                info!(
                    base_id = self.pid.raw(),
                    "finished creating mini page for base"
                );
                Ok(())
            }
            PageLocation::Full(ptr) => {
                if *cache_only {
                    return Err(TreeError::IoError(IoErrorKind::CacheOnlyViolation));
                }

                histogram!(HitMiniPage, storage.config.leaf_page_size as u64);
                // SAFETY: `ptr` comes from PageLocation::Full, a valid pointer to a
                // LeafNode in the circular buffer. We hold an exclusive write lock on
                // this page table entry, ensuring no concurrent mutable access.
                let mini_page = unsafe { &mut *ptr };
                let insert_success =
                    mini_page.insert(key, value, op_type, storage.config.max_fence_len);
                if insert_success {
                    return Ok(());
                }

                self.merge_full_page_and_dealloc(mini_page, storage)?;

                counter!(InsertMergeFullPage);

                Err(TreeError::NeedRestart)
            }

            PageLocation::Mini(ptr) => {
                // Root leaf node does not have a corresponding mini-page
                // In cache-only mode, root leaf page is a mini-page
                // Unlike other leaf nodes, root leaf node's split is done through
                // detecting split_flag upon tree traversal
                if *cache_only && parent.is_none() {
                    let root_page = self.load_cache_page_mut(ptr);
                    let mut success = root_page.insert(key, value, op_type, 0);

                    // Consolidate first, and then try insert again
                    // We cannot split a root mini-page with a single key
                    if !success {
                        root_page.consolidate_inner(OpType::Insert, None, true, *cache_only, None);
                        success = root_page.insert(key, value, op_type, 0);
                        if !success {
                            // There are at least two distinctive INSERT keys in the root, safe for splitting.
                            root_page.set_split_flag();
                            return Err(TreeError::NeedRestart);
                        } else {
                            return Ok(());
                        }
                    }
                }

                let mini_page = self.load_cache_page_mut(ptr);
                if !(*cache_only) {
                    debug_assert!(!mini_page.next_level.is_null());
                } else {
                    assert!(mini_page.next_level.is_null());
                }
                histogram!(HitMiniPage, mini_page.meta.node_size as u64);
                let insert_success = mini_page.insert(key, value, op_type, 0);

                if insert_success {
                    counter!(InsertToMiniPageSuccess);
                    return Ok(());
                }

                let kv_size = key_value_physical_size(key, value);
                let new_size =
                    mini_page.new_size_if_upgrade(kv_size, mini_page_size_classes, *cache_only);

                match new_size {
                    Some(s) => {
                        info!(
                            from = mini_page.meta.node_size,
                            to = s,
                            "upgrading mini page size"
                        );

                        let h = storage.begin_dealloc_mini_page(mini_page)?;

                        let mini_page_guard = storage.alloc_mini_page(s)?;
                        LeafNode::initialize_mini_page(
                            &mini_page_guard,
                            s,
                            mini_page.next_level,
                            *cache_only,
                        );
                        mini_page.copy_initialize_to(
                            mini_page_guard.as_ptr() as *mut LeafNode,
                            s,
                            true,
                        );

                        let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
                        let new_mini_loc = PageLocation::Mini(new_mini_ptr);
                        self.create_cache_page_loc(new_mini_loc);

                        let rt = self
                            .load_cache_page_mut(new_mini_ptr)
                            .insert(key, value, op_type, 0);
                        drop(mini_page_guard);

                        storage.finish_dealloc_mini_page(h);

                        counter!(InsertMiniPageUpgraded);
                        assert!(rt);
                        Ok(())
                    }
                    None => {
                        info!(
                            to_insert = kv_size,
                            pid = self.pid.raw(),
                            "mini page merge because too large"
                        );

                        // In cache-only mode, when a mini-page grows out of the leaf page size it is split into two mini-pages directly.
                        // To avoid creating empty mini-pages during consolidation after split, we do consolidation first to see if we could
                        // avoid page splitting.
                        if *cache_only {
                            let cur_mini_page = self.load_cache_page_mut(ptr);

                            // Only split when the current mini-page has reached the max leaf page size
                            assert!(
                                cur_mini_page.meta.node_size as usize
                                    == storage.config.leaf_page_size
                            );

                            // Consolidate the existing node first while skipping the record with the same key as k, if exists
                            cur_mini_page.consolidate_skip_key(key);

                            // Insert the k/v pair
                            let insert_success = cur_mini_page.insert(key, value, op_type, 0);

                            if insert_success {
                                counter!(InsertToMiniPageSuccess);
                                return Ok(());
                            }

                            // This is guaranteed to be true because:
                            // Assume it is not true, then there is only one key in the current mini-page.
                            // Since the required new size is greater than the leaf page size, the current leaf
                            // page size must be greater than or equal to half of the leaf page size as the max record size
                            // is less than half of the leaf page size. However, if that's true then the above insert must
                            // have succeeded which is a contradiction.
                            debug_assert!(cur_mini_page.meta.meta_count_without_fence() > 0);

                            // Upon reaching here, it is guaranteed that all records are INSERT and there
                            // are at two records with distinctive keys (including the new k/v pair) s.t. upon split and consolidation,
                            // there is at least one record per page.
                            let record_size = (key.len() + value.len()) as u16;
                            let insert_split_key = cur_mini_page
                                .get_cache_only_insert_split_key(key, &record_size)
                                .map_err(TreeError::IoError)?;

                            // Obtain version lock on self's parent
                            let self_parent = parent
                                .ok_or(TreeError::IoError(IoErrorKind::InvariantViolation))?;
                            self_parent.check_version()?;
                            let mut x_parent = self_parent.upgrade().map_err(|(_, e)| e)?;

                            if x_parent.as_ref().have_space_for(&insert_split_key) {
                                // Create a new mini-page of the same size as the current root node
                                let mini_page_guard =
                                    storage.alloc_mini_page(storage.config.leaf_page_size)?;
                                LeafNode::initialize_mini_page(
                                    &mini_page_guard,
                                    storage.config.leaf_page_size,
                                    MiniPageNextLevel::new_null(),
                                    *cache_only,
                                );
                                let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
                                let mini_loc = PageLocation::Mini(new_mini_ptr);

                                // Insert the new page into mapping table
                                let (sibling_id, _mini_lock) =
                                    storage.mapping_table().insert_mini_page_mapping(mini_loc);

                                // Split current page using the splitting key
                                let cur_page_loc = self.get_page_location().clone();
                                match cur_page_loc {
                                    PageLocation::Mini(_) => {
                                        // SAFETY: `new_mini_ptr` was just obtained from a freshly
                                        // allocated mini page guard. We hold the write lock on the
                                        // page entry, so no other thread accesses this pointer.
                                        let sibling_page = unsafe { &mut *new_mini_ptr };
                                        cur_mini_page.split_with_key(
                                            sibling_page,
                                            &insert_split_key,
                                            true,
                                        );
                                        x_parent.as_mut().insert(&insert_split_key, sibling_id);
                                        {
                                            // We are safe to drop parent here because both sibling nodes are already locked.
                                            // Holding parent lock slows down other operations.
                                            drop(x_parent);
                                        }

                                        // Directly insert the new record into its correponding mini-page
                                        let cmp = key.cmp(&insert_split_key);
                                        let ok = match cmp {
                                            core::cmp::Ordering::Greater
                                            | core::cmp::Ordering::Equal => {
                                                sibling_page.insert(key, value, op_type, 0)
                                            }
                                            core::cmp::Ordering::Less => {
                                                cur_mini_page.insert(key, value, op_type, 0)
                                            }
                                        };
                                        if !ok {
                                            return Err(TreeError::IoError(
                                                IoErrorKind::InvariantViolation,
                                            ));
                                        }

                                        debug_assert!(
                                            cur_mini_page.meta.meta_count_with_fence() > 0
                                        );

                                        debug_assert!(
                                            sibling_page.meta.meta_count_with_fence() > 0
                                        );
                                        return Ok(());
                                    }
                                    _ => {
                                        return Err(TreeError::IoError(
                                            IoErrorKind::CacheOnlyViolation,
                                        ));
                                    }
                                }
                            } else {
                                x_parent.as_mut().meta.set_split_flag();
                                counter!(MergeFailedDueToParentFull); // [TODO] a different counter
                                return Err(TreeError::NeedRestart);
                            }
                        } else {
                            // we are already too large, we need to do whole page cache.
                            // Whole page cache is different from mini page because it is a gap cache, rather than record cache.
                            // it caches the entire gap.
                            let base_offset = mini_page.next_level;
                            self.merge_mini_page_and_dealloc(
                                mini_page,
                                storage,
                                parent
                                    .ok_or(TreeError::IoError(IoErrorKind::InvariantViolation))?,
                            )?;

                            info!(pid = self.pid.raw(), "old mini page deallocated");
                            if *write_load_full_page {
                                // now we merged the mini page, we need to create a full page cache because it seems that this page is very hot.
                                let base_page_ref = self.load_base_page(base_offset.as_offset())?;

                                // Avoid bringing back empty full page
                                if base_page_ref.meta.meta_count_without_fence() != 0 {
                                    let full_page_loc =
                                        upgrade_to_full_page(storage, base_page_ref, base_offset)?;

                                    self.create_cache_page_loc(full_page_loc);

                                    info!(
                                        pid = self.pid.raw(),
                                        "created a whole page cache mini page"
                                    );

                                    counter!(UpgradeToFullPage);
                                }
                            }
                        }
                        Err(TreeError::NeedRestart)
                    }
                }
            }
            PageLocation::Null => Err(TreeError::IoError(IoErrorKind::NullPage)),
        }
    }

    pub(crate) fn load_cache_page_mut<'b>(&self, ptr: *mut LeafNode) -> &'b mut LeafNode {
        // SAFETY: `ptr` comes from a PageLocation::Mini or PageLocation::Full entry pointing
        // to a valid LeafNode in the circular buffer. The caller holds a write lock on this
        // page table entry, ensuring exclusive mutable access. The lifetime 'b is bounded
        // by the caller's lock scope.
        unsafe { &mut *ptr }
    }

    pub(crate) fn get_split_flag(&mut self) -> Result<bool, TreeError> {
        let page_loc = self.raw_guard.deref();
        match page_loc {
            PageLocation::Base(offset) => {
                let base_ref = self.load_base_page(*offset)?;
                Ok(base_ref.get_split_flag())
            }
            PageLocation::Full(ptr) | PageLocation::Mini(ptr) => {
                let base_ref = self.load_cache_page_mut(*ptr);
                Ok(base_ref.get_split_flag())
            }
            PageLocation::Null => Ok(false), // This happens in the rare case in cache-only mode where the leaf node to insert a page in is evicted.
        }
    }

    /// Calling this function will set base page to be dirty
    pub(crate) fn load_base_page_mut(&mut self) -> Result<&mut LeafNode, TreeError> {
        let page_loc = self.raw_guard.deref().clone();

        match self.tmp_buffer {
            Some(ref mut buffer) => return Ok(buffer.as_leaf_node_mut()),
            None => {
                let offset = match page_loc {
                    PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                        let page = self.load_cache_page_mut(ptr);
                        page.next_level.as_offset()
                    }
                    PageLocation::Base(offset) => offset,
                    PageLocation::Null => {
                        return Err(TreeError::IoError(IoErrorKind::Corruption));
                    }
                };

                let mut buffer = TmpBuffer::new(self.tmp_buffer_size);
                let slice = buffer.as_u8_slice_mut();
                self.file_handle
                    .read(offset, slice)
                    .map_err(|_| TreeError::IoError(IoErrorKind::VfsRead { offset }))?;

                self.tmp_buffer = Some(buffer);

                if let Some(ref mut b) = self.tmp_buffer {
                    return Ok(b.as_leaf_node_mut());
                }
            }
        };
        unreachable!()
    }

    pub(crate) fn merge_mini_page_and_dealloc(
        &mut self,
        mini_page: &mut LeafNode,
        storage: &LeafStorage,
        parent: ReadGuard<'_>,
    ) -> Result<(), TreeError> {
        let h = storage.begin_dealloc_mini_page(mini_page)?;
        self.try_merge_mini_page(&h, parent, storage)?;
        self.change_to_base_loc();
        storage.finish_dealloc_mini_page(h);
        Ok(())
    }

    pub(crate) fn merge_full_page_and_dealloc(
        &mut self,
        mini_page: &mut LeafNode,
        storage: &LeafStorage,
    ) -> Result<(), TreeError> {
        let h = storage.begin_dealloc_mini_page(mini_page)?;
        self.merge_full_page(&h);
        storage.finish_dealloc_mini_page(h);
        Ok(())
    }

    /// TODO: we want to shrink the cache page size if possible.
    pub(crate) fn move_cache_page_to_tail(
        &mut self,
        storage: &LeafStorage,
    ) -> Result<(), TreeError> {
        let page_loc = self.raw_guard.deref();
        let new_loc = match page_loc {
            PageLocation::Mini(ptr) => {
                let mini_page = self.load_cache_page_mut(*ptr);
                let h: TombstoneHandle = storage.begin_dealloc_mini_page(mini_page)?;
                let new_page =
                    storage.move_mini_page_to_tail(h, mini_page.meta.node_size as usize)?;

                counter!(MoveMiniPageToTail);
                PageLocation::Mini(new_page.as_ptr() as *mut LeafNode)
            }
            PageLocation::Full(ptr) => {
                let mini_page = self.load_cache_page_mut(*ptr);
                debug_assert!(mini_page.meta.node_size as usize == storage.config.leaf_page_size);
                let h = storage.begin_dealloc_mini_page(mini_page)?;
                let new_page =
                    storage.move_full_page_to_tail(h, mini_page.meta.node_size as usize)?;

                counter!(MoveFullPageToTail);
                PageLocation::Full(new_page.as_ptr() as *mut LeafNode)
            }
            PageLocation::Base(_) => unreachable!(),
            PageLocation::Null => {
                return Err(TreeError::IoError(IoErrorKind::NullPage));
            }
        };
        *self.raw_guard.deref_mut() = new_loc;
        Ok(())
    }

    /// Flush a full page into its corresponding base page
    pub(crate) fn merge_full_page(&mut self, mini_page_handle: &TombstoneHandle) {
        let mini_page = self.load_cache_page_mut(mini_page_handle.as_ptr() as *mut LeafNode);
        debug_assert!(mini_page.meta.node_size as usize == self.tmp_buffer_size);

        if !mini_page.need_actually_merge_to_disk() {
            self.change_to_base_loc();
            return;
        }

        mini_page.convert_cache_records_to_insert();

        self.change_to_base_loc();
        let buffer = self.tmp_buffer.take();
        match buffer {
            Some(mut b) => {
                debug_assert!(b.is_dirty);

                // SAFETY: `mini_page_handle.as_ptr()` points to a valid full page of
                // `node_size` bytes in the circular buffer. `b.ptr` is a separately
                // allocated TmpBuffer of `tmp_buffer_size` (asserted equal to node_size).
                // The two buffers do not overlap.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        mini_page_handle.as_ptr(),
                        b.ptr,
                        mini_page.meta.node_size as usize,
                    );
                }
                let base = b.as_leaf_node_mut();
                base.next_level = MiniPageNextLevel::new_null();

                self.tmp_buffer = Some(b);
            }
            None => {
                let mut buffer = TmpBuffer::new(self.tmp_buffer_size);
                // SAFETY: `mini_page_handle.as_ptr()` points to a valid full page of
                // `node_size` bytes. `buffer.ptr` is a freshly allocated TmpBuffer of
                // `tmp_buffer_size` bytes. The two allocations do not overlap.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        mini_page_handle.as_ptr(),
                        buffer.ptr,
                        mini_page.meta.node_size as usize,
                    );
                }

                let base = buffer.as_leaf_node_mut();
                base.next_level = MiniPageNextLevel::new_null();
                self.tmp_buffer = Some(buffer);
            }
        }
    }

    /// On success, returns the old mini page pointer.
    /// It's caller's responsibility to free the old mini page.
    /// It's caller's responsibility to acquire tombstone handle.
    /// It's caller's responsibility to update the mapping table to point to the new mini page, if any.
    pub(crate) fn try_merge_mini_page(
        &mut self,
        mini_page_handle: &TombstoneHandle,
        parent: ReadGuard,
        storage: &LeafStorage,
    ) -> Result<MergeResult, TreeError> {
        parent.check_version()?;
        let mini_page = self.load_cache_page_mut(mini_page_handle.as_ptr() as *mut LeafNode);

        if !mini_page.need_actually_merge_to_disk() {
            return Ok(MergeResult::NoSplit);
        }

        let base_ref = self.load_base_page_mut()?;

        // If base page has only one record, consolidate it first
        if base_ref.meta.meta_count_without_fence() == 1 {
            base_ref.consolidate_inner(OpType::Insert, None, true, false, None);
        }

        if base_ref.merge_mini_page(mini_page, storage.config.max_fence_len) {
            return Ok(MergeResult::NoSplit);
        }

        info!("mini page merge causing base page to split");
        counter!(MergeTriggerSplit);

        let mut x_parent = parent.upgrade().map_err(|(_, e)| e)?;

        // If there is only one distinctive key, then the merge should have succeeded before.
        // Choose a splitting key based on records in both mini-page and the correponding base page
        let merge_split_key = base_ref
            .get_merge_split_key(mini_page)
            .map_err(TreeError::IoError)?;

        if x_parent.as_ref().have_space_for(&merge_split_key) {
            let (sibling_node_id, mut sibling_node) = storage.alloc_base_page_and_lock();
            let sibling_node_ref = sibling_node.load_base_page_mut()?;
            base_ref.split_with_key(sibling_node_ref, &merge_split_key, false);
            x_parent.as_mut().insert(&merge_split_key, sibling_node_id);
            {
                // We are safe to drop parent here because both sibling nodes are already locked.
                // Holding parent lock slows down other operations.
                drop(x_parent);
            }

            // TODO: we have a design problem here:
            // what if the splitted pages are still too large that it need to be split again to absorb the mini page?
            // Right now, we keep splitting the base page in multiple iterations until the new record fits.
            // However, this could incur unnecessary mini page merges compared to doing the multiple page splitting in one run
            for kv_meta in mini_page.meta_iter() {
                let op_type = kv_meta.op_type();
                if !op_type.is_dirty() {
                    continue;
                }

                let key_to_merge = mini_page.get_full_key(kv_meta);
                let value = mini_page.get_value(kv_meta);

                let cmp = key_to_merge.cmp(&merge_split_key);
                match cmp {
                    core::cmp::Ordering::Greater | core::cmp::Ordering::Equal => {
                        let ok = sibling_node_ref.insert(
                            &key_to_merge,
                            value,
                            op_type,
                            storage.config.max_fence_len,
                        );

                        if !ok {
                            return Err(TreeError::IoError(IoErrorKind::InvariantViolation));
                        }
                    }
                    core::cmp::Ordering::Less => {
                        let ok = base_ref.insert(
                            &key_to_merge,
                            value,
                            op_type,
                            storage.config.max_fence_len,
                        );
                        if !ok {
                            return Err(TreeError::IoError(IoErrorKind::InvariantViolation));
                        }
                    }
                }
            }

            Ok(MergeResult::MergeAndSplit)
        } else {
            x_parent.as_mut().meta.set_split_flag();
            counter!(MergeFailedDueToParentFull);
            Err(TreeError::NeedRestart)
        }
    }

    pub(crate) fn get_stats(&mut self) -> Result<LeafStats, TreeError> {
        let page_loc = self.raw_guard.deref();
        match page_loc {
            PageLocation::Mini(ptr) => {
                let mini_page = self.load_cache_page_mut(*ptr);
                let mut mini_stats = mini_page.get_stats();
                let next_level = mini_page.next_level;

                let base_ref = self.load_base_page(next_level.as_offset())?;
                let stats = base_ref.get_stats();
                mini_stats.base_node = Some(Box::new(stats));
                Ok(mini_stats)
            }
            PageLocation::Full(ptr) => {
                let base_ref = self.load_cache_page_mut(*ptr);
                Ok(base_ref.get_stats())
            }
            PageLocation::Base(offset) => {
                let mut buffer = TmpBuffer::new(self.tmp_buffer_size);
                let slice = buffer.as_u8_slice_mut();
                self.file_handle
                    .read(*offset, slice)
                    .map_err(|_| TreeError::IoError(IoErrorKind::VfsRead { offset: *offset }))?;
                let base_ref = buffer.as_leaf_node();
                Ok(base_ref.get_stats())
            }
            PageLocation::Null => Err(TreeError::IoError(IoErrorKind::Corruption)),
        }
    }

    pub(crate) fn create_cache_page_loc(&mut self, cache_page_loc: PageLocation) {
        // (1) we don't need to manually flush buffer here, it will auto evict when the lock drops.
        // (2) It's possible that we are already mini page, due to mini page grow.
        let _old_loc = core::mem::replace(self.raw_guard.deref_mut(), cache_page_loc);
    }

    pub(crate) fn change_to_base_loc(&mut self) {
        let old_loc = self.raw_guard.deref().clone();
        match old_loc {
            PageLocation::Base(_) => {
                debug_assert!(false, "change_to_base_loc called on already-base page");
                // Already base -- no-op.
            }
            PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                let mini_page = self.load_cache_page_mut(ptr);
                let offset = mini_page.next_level.as_offset();
                let base_loc = PageLocation::Base(offset);

                let _old_loc = core::mem::replace(self.raw_guard.deref_mut(), base_loc);
                // we don't need to manually flush buffer here, it will auto evict when the lock drops.
            }
            PageLocation::Null => {
                debug_assert!(false, "change_to_base_loc on Null page");
            }
        }
    }

    // Change the mapping entry to null
    pub(crate) fn change_to_null_loc(&mut self) {
        let _old_loc = core::mem::replace(self.raw_guard.deref_mut(), PageLocation::Null);
    }

    pub(crate) fn get_disk_offset(&self) -> Result<u64, TreeError> {
        let page_loc = self.raw_guard.deref();
        match page_loc {
            PageLocation::Base(offset) => Ok(*offset as u64),
            PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                let mini_page = self.load_cache_page(*ptr);
                Ok(mini_page.next_level.as_offset() as u64)
            }
            PageLocation::Null => Err(TreeError::IoError(IoErrorKind::NullPage)),
        }
    }

    pub(crate) fn update_lsn(&mut self, lsn: u64) -> Result<(), TreeError> {
        let page_loc = self.raw_guard.deref();
        match page_loc {
            PageLocation::Base(_offset) => {
                let base_ref = self.load_base_page_mut()?;
                base_ref.lsn = lsn;
            }
            PageLocation::Full(ptr) | PageLocation::Mini(ptr) => {
                let page_ref = self.load_cache_page_mut(*ptr);
                page_ref.lsn = lsn;
            }
            PageLocation::Null => return Err(TreeError::IoError(IoErrorKind::Corruption)),
        }
        Ok(())
    }
}

pub(crate) enum MergeResult {
    NoSplit,
    MergeAndSplit,
}

pub(crate) fn upgrade_to_full_page(
    storage: &LeafStorage,
    base_page: &LeafNode,
    base_page_offset: MiniPageNextLevel,
) -> Result<PageLocation, TreeError> {
    let full_page = storage.alloc_mini_page(storage.config.leaf_page_size)?;

    assert_eq!(
        base_page.meta.node_size as usize,
        storage.config.leaf_page_size
    );
    // SAFETY: `base_page` is a valid LeafNode reference of `leaf_page_size` bytes (asserted
    // above). `full_page.as_ptr()` is a freshly allocated circular buffer region of the same
    // size. The source (base_page on disk/buffer) and dest (circular buffer) do not overlap.
    unsafe {
        core::ptr::copy_nonoverlapping(
            base_page as *const LeafNode as *const u8,
            full_page.as_ptr(),
            storage.config.leaf_page_size,
        );
    }
    let full_page_ptr = full_page.as_ptr() as *mut LeafNode;
    // SAFETY: `full_page_ptr` points to the circular buffer region just initialized with a
    // valid LeafNode copy above. The allocation is exclusively owned via `full_page` guard,
    // so creating a mutable reference is sound.
    let full_page_ref = unsafe { &mut *full_page_ptr };
    full_page_ref.covert_insert_records_to_cache();
    full_page_ref.next_level = base_page_offset;

    Ok(PageLocation::Full(full_page.as_ptr() as *mut LeafNode))
}
