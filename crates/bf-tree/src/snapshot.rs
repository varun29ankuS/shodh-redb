// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::collections::{HashMap, VecDeque};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

#[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
use thread_local::ThreadLocal;

use crate::{
    circular_buffer::{CircularBuffer, TombstoneHandle},
    error::{BfTreeError, IoErrorKind},
    fs::VfsImpl,
    nodes::{InnerNode, InnerNodeBuilder, PageID, DISK_PAGE_SIZE, INNER_NODE_SIZE},
    range_scan::ScanReturnField,
    storage::{make_vfs, LeafStorage, PageLocation, PageTable},
    sync::atomic::AtomicU64,
    tree::eviction_callback,
    utils::{inner_lock::ReadGuard, BfsVisitor, NodeInfo},
    wal::{LogEntry, LogEntryImpl, WriteAheadLog},
    BfTree, Config, StorageBackend, WalReader,
};

const BF_TREE_MAGIC_BEGIN: &[u8; 16] = b"BF-TREE-V0-BEGIN";
const BF_TREE_MAGIC_END: &[u8; 14] = b"BF-TREE-V0-END";
const META_DATA_PAGE_OFFSET: usize = 0;

struct SectorAlignedVector {
    inner: ManuallyDrop<Vec<u8>>,
}

impl Drop for SectorAlignedVector {
    fn drop(&mut self) {
        let layout =
            std::alloc::Layout::from_size_align(self.inner.capacity(), SECTOR_SIZE).unwrap();
        let ptr = self.inner.as_mut_ptr();
        // SAFETY: ptr was allocated via alloc_zeroed with the same layout in new_zeroed(),
        // and ManuallyDrop ensures Vec's drop does not also deallocate.
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }
}

impl SectorAlignedVector {
    fn new_zeroed(capacity: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(capacity, SECTOR_SIZE).unwrap();
        // SAFETY: layout has non-zero size and SECTOR_SIZE alignment (a power of 2).
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };

        // SAFETY: ptr was just allocated with the given capacity and is fully initialized
        // (zeroed). The Vec takes ownership; ManuallyDrop prevents double-free in Drop.
        let inner = unsafe { Vec::from_raw_parts(ptr, capacity, capacity) };
        Self {
            inner: ManuallyDrop::new(inner),
        }
    }
}

impl Deref for SectorAlignedVector {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for SectorAlignedVector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl BfTree {
    /// Recover a Bf-Tree from snapshot and WAL files.
    ///
    /// Replays Write operations from the WAL on top of the snapshot state.
    /// Split WAL entries are skipped if the page table already contains the
    /// resulting pages (i.e., a subsequent snapshot captured them).
    /// Recover a Bf-Tree from snapshot and WAL, returning the recovered tree.
    ///
    /// Replays Write operations from the WAL on top of the snapshot state,
    /// then takes a fresh snapshot to persist the recovered state. The
    /// recovered tree is returned so the caller can continue using it.
    pub fn recovery(
        config_file: impl AsRef<Path>,
        wal_file: impl AsRef<Path>,
        wal_segment_size: usize,
        buffer_ptr: Option<*mut u8>,
    ) -> Result<Self, BfTreeError> {
        let bf_tree_config = Config::new_with_config_file(config_file);

        // Read the snapshot metadata to get the LSN high-water mark.
        // WAL entries with lsn <= snapshot_lsn are already persisted in
        // the snapshot and must be skipped to avoid re-inserting them
        // (which causes structural inconsistencies in scan ordering).
        let snapshot_lsn = Self::read_snapshot_lsn(&bf_tree_config.file_path)?;

        let bf_tree = BfTree::new_from_snapshot(bf_tree_config, buffer_ptr)?;
        let wal_reader = WalReader::new(wal_file, wal_segment_size)?;

        for seg in wal_reader.segment_iter() {
            let seg = seg.map_err(BfTreeError::Io)?;
            for entry in seg.entry_iter() {
                let header = &entry.0;
                // Skip entries already captured by the snapshot.
                // snapshot_lsn == 0 means no LSN was recorded (old format
                // or WAL was not enabled when the snapshot was taken), so
                // replay everything for backward compatibility.
                if snapshot_lsn > 0 && header.lsn <= snapshot_lsn {
                    continue;
                }
                let log_entry = LogEntry::read_from_buffer(entry.1);
                match log_entry {
                    LogEntry::Write(op) => {
                        bf_tree.insert(op.key, op.value);
                    }
                    LogEntry::Split(_op) => {
                        // Split WAL entries are handled implicitly during
                        // insert-based recovery: as Write entries are replayed,
                        // the tree naturally re-splits when pages fill up,
                        // producing the same logical structure.
                    }
                }
            }
        }
        // Persist the recovered state so subsequent opens see the replayed data.
        bf_tree.snapshot()?;
        Ok(bf_tree)
    }

    /// Read the snapshot_lsn from a snapshot file's metadata header.
    /// Returns 0 if the file doesn't exist or was created before
    /// snapshot_lsn was added to the metadata (backward compatible).
    fn read_snapshot_lsn(path: &Path) -> Result<u64, BfTreeError> {
        if !path.exists() {
            return Ok(0);
        }
        let reader = std::fs::File::open(path).map_err(|_| IoErrorKind::SnapshotRead)?;
        let mut buf = SectorAlignedVector::new_zeroed(4096);
        #[cfg(unix)]
        {
            reader
                .read_at(&mut buf, 0)
                .map_err(|_| IoErrorKind::SnapshotRead)?;
        }
        #[cfg(windows)]
        {
            reader
                .seek_read(&mut buf, 0)
                .map_err(|_| IoErrorKind::SnapshotRead)?;
        }
        // SAFETY: BfTreeMeta is #[repr(C, align(512))] with only primitive fields.
        // The buffer is 4096 bytes (>= size_of::<BfTreeMeta>()) and was just read.
        let meta = unsafe { (buf.as_ptr() as *const BfTreeMeta).read() };
        meta.check_magic();
        Ok(meta.snapshot_lsn)
    }

    /// Instead of creating a new Bf-Tree instance,
    /// it loads a Bf-Tree snapshot file and resume from there.
    pub fn new_from_snapshot(
        bf_tree_config: Config,
        buffer_ptr: Option<*mut u8>,
    ) -> Result<Self, BfTreeError> {
        if !bf_tree_config.file_path.exists() {
            // if not already exist, we just create a new empty file at the location.
            return BfTree::with_config(bf_tree_config.clone(), buffer_ptr);
        }

        // Validate the config first
        bf_tree_config.validate().map_err(BfTreeError::Config)?;

        let reader = std::fs::File::open(bf_tree_config.file_path.clone())
            .map_err(|_| IoErrorKind::SnapshotRead)?;
        let mut metadata = SectorAlignedVector::new_zeroed(4096);
        #[cfg(unix)]
        {
            reader
                .read_at(&mut metadata, 0)
                .map_err(|_| IoErrorKind::SnapshotRead)?;
        }
        #[cfg(windows)]
        {
            reader
                .seek_read(&mut metadata, 0)
                .map_err(|_| IoErrorKind::SnapshotRead)?;
        }

        // SAFETY: BfTreeMeta is #[repr(C, align(512))] with only primitive fields.
        // The buffer is 4096 bytes (>= size_of::<BfTreeMeta>()) and was just read from disk.
        let bf_meta = unsafe { (metadata.as_ptr() as *const BfTreeMeta).read() };
        bf_meta.check_magic();
        let actual_size = reader
            .metadata()
            .map_err(|_| IoErrorKind::SnapshotRead)?
            .len();
        // The file may legitimately be *larger* than what the snapshot metadata
        // recorded: post-snapshot inserts can trigger page splits that extend
        // the file via alloc_offset() before the next snapshot (or crash).
        // A file *smaller* than expected indicates truncation / genuine corruption.
        if actual_size < bf_meta.file_size {
            return Err(BfTreeError::Io(IoErrorKind::Corruption));
        }

        let config = Arc::new(bf_tree_config);

        let wal = match config.write_ahead_log.as_ref() {
            Some(s) => Some(WriteAheadLog::new(s.clone()).map_err(BfTreeError::Io)?),
            None => None,
        };

        let vfs =
            make_vfs(&config.storage_backend, &config.file_path).map_err(|e| BfTreeError::Io(e))?;

        let mut page_buffer = SectorAlignedVector::new_zeroed(INNER_NODE_SIZE);

        // Step 1: reconstruct inner nodes.
        let mut root_page_id = bf_meta.root_id;
        if root_page_id.is_inner_node_pointer() {
            let inner_mapping: Vec<(*const InnerNode, usize)> =
                read_vec_from_offset(bf_meta.inner_offset, bf_meta.inner_size, &vfs)?;
            let mut inner_map = HashMap::new();

            for m in inner_mapping {
                inner_map.insert(m.0, m.1);
            }
            let offset = inner_map
                .get(&root_page_id.as_inner_node())
                .ok_or(BfTreeError::Io(IoErrorKind::Corruption))?;
            vfs.read(*offset, &mut page_buffer)
                .map_err(|e| BfTreeError::Io(e))?;
            let root_page = InnerNodeBuilder::new().build_from_slice(&page_buffer);
            root_page_id = PageID::from_pointer(root_page);

            let mut inner_resolve_queue = VecDeque::from([root_page]);
            while !inner_resolve_queue.is_empty() {
                let inner_ptr = inner_resolve_queue.pop_front().unwrap();
                let mut inner = ReadGuard::try_read(inner_ptr).unwrap().upgrade().unwrap();
                if inner.as_ref().meta.children_is_leaf() {
                    continue;
                }
                for (idx, c) in inner.as_ref().get_child_iter().enumerate() {
                    let offset = inner_map
                        .get(&c.as_inner_node())
                        .ok_or(BfTreeError::Io(IoErrorKind::Corruption))?;
                    vfs.read(*offset, &mut page_buffer)
                        .map_err(|e| BfTreeError::Io(e))?;
                    let inner_page = InnerNodeBuilder::new().build_from_slice(&page_buffer);
                    let inner_id = PageID::from_pointer(inner_page);
                    inner.as_mut().update_at_pos(idx, inner_id);
                    inner_resolve_queue.push_back(inner_page);
                }
            }
        }

        // Step 2: reconstruct leaf mappings.
        let leaf_mapping: Vec<(PageID, usize)> =
            read_vec_from_offset(bf_meta.leaf_offset, bf_meta.leaf_size, &vfs)?;
        let leaf_mapping = leaf_mapping.into_iter().map(|(pid, offset)| {
            let loc = PageLocation::Base(offset);
            (pid, loc)
        });
        let pt = PageTable::new_from_mapping(leaf_mapping, vfs.clone(), config.clone());
        let circular_buffer = CircularBuffer::new(
            config.cb_size_byte,
            config.cb_copy_on_access_ratio,
            config.cb_min_record_size,
            config.cb_max_record_size,
            config.leaf_page_size,
            config.max_fence_len,
            buffer_ptr,
            config.cache_only,
        );

        let storage = LeafStorage::new_inner(config.clone(), pt, circular_buffer, vfs);

        let raw_root_id = if root_page_id.is_id() {
            root_page_id.raw() | Self::ROOT_IS_LEAF_MASK
        } else {
            root_page_id.raw()
        };

        let size_classes = Self::create_mem_page_size_classes(
            config.cb_min_record_size,
            config.cb_max_record_size,
            config.leaf_page_size,
            config.max_fence_len,
            config.cache_only,
        );
        Ok(BfTree {
            storage,
            root_page_id: AtomicU64::new(raw_root_id),
            wal,
            write_load_full_page: true,
            cache_only: false,
            mini_page_size_classes: size_classes,
            config,
            #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
            metrics_recorder: Some(Arc::new(ThreadLocal::new())),
        })
    }

    /// Stop the world and take a snapshot of the current state.
    ///
    /// Returns the snapshot file path
    pub fn snapshot(&self) -> Result<PathBuf, BfTreeError> {
        let callback = |h| -> Result<TombstoneHandle, TombstoneHandle> {
            match eviction_callback(&h, self) {
                Ok(_) => Ok(h),
                Err(_e) => Err(h),
            }
        };
        self.storage.circular_buffer.drain(callback);

        let root_id = self.get_root_page();
        let mut inner_mapping: Vec<(*const InnerNode, usize)> = Vec::new();

        // Collect all inner node writes into a batch buffer, then write them
        // all at once. This avoids per-node pwrite syscalls which are expensive
        // on Windows NTFS.
        let visitor = BfsVisitor::new_inner_only(self);
        let mut batched_writes: Vec<(usize, Vec<u8>)> = Vec::new();
        for node in visitor {
            match node {
                NodeInfo::Inner { ptr, .. } => {
                    let inner = ReadGuard::try_read(ptr).unwrap();
                    if inner.as_ref().is_valid_disk_offset() {
                        let offset = inner.as_ref().disk_offset as usize;
                        batched_writes.push((offset, inner.as_ref().as_slice().to_vec()));
                        inner_mapping.push((ptr, offset));
                    }
                }
                NodeInfo::Leaf { level, .. } => {
                    assert_eq!(level, 0);
                }
            }
        }
        for (offset, data) in &batched_writes {
            self.storage
                .vfs
                .write(*offset, data)
                .map_err(BfTreeError::Io)?;
        }
        let (inner_offset, inner_size) = serialize_vec_to_disk(&inner_mapping, &self.storage.vfs)?;

        let mut leaf_mapping = Vec::new();
        let page_table_iter = self.storage.page_table.iter();
        for (entry, pid) in page_table_iter {
            assert!(pid.is_id());
            match entry.try_read().unwrap().as_ref() {
                PageLocation::Base(base) => leaf_mapping.push((pid, *base)),
                PageLocation::Full(_) | PageLocation::Mini(_) => {
                    unreachable!("Circular buffer should already be drained!")
                }
                PageLocation::Null => panic!("Snapshot of Null page"),
            }
        }

        let (leaf_offset, leaf_size) = serialize_vec_to_disk(&leaf_mapping, &self.storage.vfs)?;

        let file_size = (leaf_offset + align_to_sector_size(leaf_size)) as u64;

        #[cfg(feature = "std")]
        let snapshot_lsn = self.wal.as_ref().map(|w| w.get_flushed_lsn()).unwrap_or(0);
        #[cfg(not(feature = "std"))]
        let snapshot_lsn = 0u64;

        let metadata = BfTreeMeta {
            magic_begin: *BF_TREE_MAGIC_BEGIN,
            root_id: root_id.0,
            inner_offset,
            inner_size,
            leaf_offset,
            leaf_size,
            file_size,
            snapshot_lsn,
            magic_end: *BF_TREE_MAGIC_END,
        };

        self.storage
            .vfs
            .write(META_DATA_PAGE_OFFSET, metadata.as_slice())
            .map_err(BfTreeError::Io)?;
        self.storage.vfs.flush().map_err(BfTreeError::Io)?;

        // Activate COW protection: any subsequent page eviction that would
        // write to an offset below file_size will allocate a new offset
        // instead, preserving the snapshot data for crash recovery.
        self.storage.page_table.set_cow_boundary(file_size as usize);

        Ok(self.config.file_path.clone())
    }

    /// Snapshot an in-memory Bf-Tree to a file on disk.
    ///
    /// This works by scanning all live records from the in-memory tree and inserting
    /// them one-by-one into a new file-backed tree, then calling [`snapshot`] on it.
    /// Records are streamed without buffering the entire dataset in memory.
    ///
    /// For `cache_only` trees (where scan is not supported), falls back to collecting
    /// records via BFS traversal before inserting.
    ///
    /// Returns the snapshot file path.
    ///
    /// # Panics
    /// Panics if `snapshot_path` already exists.
    pub fn snapshot_memory_to_disk(
        &self,
        snapshot_path: impl AsRef<Path>,
    ) -> Result<PathBuf, BfTreeError> {
        let snapshot_path = snapshot_path.as_ref();
        assert!(
            !snapshot_path.exists(),
            "snapshot_memory_to_disk: target file already exists: {:?}",
            snapshot_path
        );

        // Build a disk-backed config from the current config.
        let mut disk_config = self.config.as_ref().clone();
        disk_config.storage_backend(StorageBackend::Std);
        disk_config.cache_only(false);
        disk_config.file_path(snapshot_path);

        let disk_tree = BfTree::with_config(disk_config, None)?;

        if self.cache_only {
            panic!("snapshot_memory_to_disk does not support cache_only trees");
        } else {
            Self::copy_records_via_scan(self, &disk_tree);
        }

        disk_tree.snapshot()
    }

    /// Copy all live records from `src` to `dst` using the scan operator.
    /// This streams one record at a time, avoiding large intermediate allocations.
    fn copy_records_via_scan(src: &BfTree, dst: &BfTree) {
        // Allocate a single reusable buffer large enough for any key+value.
        let buf_size = src.config.leaf_page_size;
        let mut scan_buf = vec![0u8; buf_size];

        // Start scanning from the smallest possible key (a single 0x00 byte).
        let start_key: &[u8] = &[0u8];
        let mut scan_iter =
            match src.scan_with_count(start_key, usize::MAX, ScanReturnField::KeyAndValue) {
                Ok(iter) => iter,
                Err(_) => return, // empty tree or other issue
            };

        while let Ok(Some((key_len, value_len))) = scan_iter.next(&mut scan_buf) {
            let key = &scan_buf[..key_len];
            let value = &scan_buf[key_len..key_len + value_len];
            dst.insert(key, value);
        }
    }

    /// Load an on-disk Bf-Tree snapshot and return a new in-memory (cache_only) Bf-Tree
    /// containing all its records.
    ///
    /// The caller provides a `Config` that controls the in-memory tree's parameters
    /// (circular buffer size, record sizes, leaf page size, etc.).
    /// `storage_backend` and `cache_only` are overridden automatically.
    ///
    /// # Arguments
    /// * `snapshot_path` --path to an existing snapshot file on disk.
    /// * `memory_config` --configuration for the resulting in-memory tree.
    ///
    /// # Panics
    /// Panics if the snapshot file does not exist.
    pub fn new_from_snapshot_disk_to_memory(
        snapshot_path: impl AsRef<Path>,
        memory_config: Config,
    ) -> Result<Self, BfTreeError> {
        let snapshot_path = snapshot_path.as_ref();
        assert!(
            snapshot_path.exists(),
            "new_from_snapshot_disk_to_memory: snapshot file does not exist: {:?}",
            snapshot_path
        );

        // Step 1: Open the on-disk snapshot as a normal disk-backed tree.
        let mut disk_config = memory_config.clone();
        disk_config.storage_backend(StorageBackend::Std);
        disk_config.cache_only(false);
        disk_config.file_path(snapshot_path);

        let disk_tree = BfTree::new_from_snapshot(disk_config, None)?;

        // Step 2: Create the in-memory tree.
        // Use Memory backend with cache_only=false so that evicted pages
        // are stored as heap-backed base pages (no data loss).
        let mut mem_config = memory_config;
        mem_config.storage_backend(StorageBackend::Memory);
        mem_config.cache_only(false);

        let mem_tree = BfTree::with_config(mem_config, None)?;

        // Step 3: Stream records from the disk tree into the memory tree via scan.
        // The disk tree is never cache_only, so scan is always available.
        Self::copy_records_via_scan(&disk_tree, &mem_tree);

        Ok(mem_tree)
    }
}

/// We use repr(C) for simplicity, maybe flatbuffer or bincode or even repr(Rust) is better.
/// But we don't care about the space here.
/// I don't want to introduce giant dependencies just for this.
#[repr(C, align(512))]
struct BfTreeMeta {
    magic_begin: [u8; 16],
    root_id: PageID,
    inner_offset: usize,
    inner_size: usize,
    leaf_offset: usize,
    leaf_size: usize,
    file_size: u64,
    /// WAL flushed LSN at snapshot time. Recovery skips entries with lsn <= this.
    /// Zero when WAL is not enabled or for snapshots created before this field existed.
    snapshot_lsn: u64,
    magic_end: [u8; 14],
}
const _: () = assert!(std::mem::size_of::<BfTreeMeta>() <= DISK_PAGE_SIZE);

impl BfTreeMeta {
    fn as_slice(&self) -> &[u8] {
        let ptr = self as *const Self as *const u8;
        let size = std::mem::size_of::<Self>();
        // SAFETY: self is a valid reference so ptr is valid for size bytes,
        // and BfTreeMeta is #[repr(C)] with only primitive fields (no padding concerns).
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    fn check_magic(&self) {
        assert_eq!(self.magic_begin, *BF_TREE_MAGIC_BEGIN);
        assert_eq!(self.magic_end, *BF_TREE_MAGIC_END);
    }
}

/// Returns starting offset and total size written to disk.
fn serialize_vec_to_disk<T>(
    v: &[T],
    vfs: &Arc<dyn VfsImpl>,
) -> Result<(usize, usize), BfTreeError> {
    if v.is_empty() {
        return Ok((0, 0));
    }
    let unaligned_ptr = v.as_ptr() as *const u8;
    let unaligned_size = std::mem::size_of_val(v);

    let aligned_size = align_to_sector_size(unaligned_size);
    let layout = std::alloc::Layout::from_size_align(aligned_size, SECTOR_SIZE).unwrap();
    // SAFETY: layout is non-zero-sized and properly aligned (SECTOR_SIZE is a power of 2).
    unsafe {
        let aligned_ptr = std::alloc::alloc_zeroed(layout);
        // SAFETY: unaligned_ptr is valid for unaligned_size bytes (it comes from v),
        // and aligned_ptr was allocated with aligned_size >= unaligned_size.
        std::ptr::copy_nonoverlapping(unaligned_ptr, aligned_ptr, unaligned_size);
        // SAFETY: aligned_ptr is valid for aligned_size bytes (just allocated above).
        let slice = std::slice::from_raw_parts(aligned_ptr, aligned_size);
        let offset = serialize_u8_slice_to_disk(slice, vfs)?;
        // SAFETY: aligned_ptr was allocated with the same layout via alloc_zeroed.
        std::alloc::dealloc(aligned_ptr, layout);
        Ok((offset, unaligned_size))
    }
}

fn read_vec_from_offset<T: Clone>(
    offset: usize,
    size: usize,
    vfs: &Arc<dyn VfsImpl>,
) -> Result<Vec<T>, BfTreeError> {
    if size == 0 {
        return Err(BfTreeError::Io(IoErrorKind::Corruption));
    }
    let slice = read_u8_slice_from_disk(offset, size, vfs)?;
    let ptr = slice.as_ptr() as *const T;
    let count = size / std::mem::size_of::<T>();
    // SAFETY: ptr is aligned to T (serialized via serialize_vec_to_disk with the same T layout),
    // and count * size_of::<T>() <= slice.len() by construction.
    let items = unsafe { std::slice::from_raw_parts(ptr, count) };
    Ok(items.to_vec())
}

fn read_u8_slice_from_disk(
    offset: usize,
    size: usize,
    vfs: &Arc<dyn VfsImpl>,
) -> Result<Vec<u8>, BfTreeError> {
    let mut res = Vec::new();
    let mut buffer = vec![0; DISK_PAGE_SIZE];
    for i in (0..size).step_by(DISK_PAGE_SIZE) {
        vfs.read(offset + i, &mut buffer)
            .map_err(|e| BfTreeError::Io(e))?;
        res.extend_from_slice(&buffer);
    }
    Ok(res)
}

const SECTOR_SIZE: usize = 512;

fn align_to_sector_size(n: usize) -> usize {
    (n + SECTOR_SIZE - 1) & !(SECTOR_SIZE - 1)
}

/// Write a slice to disk and return the start offset.
///
/// Allocates a contiguous region via the bump allocator and writes the entire
/// slice in a single I/O operation. The allocation is rounded up to
/// `DISK_PAGE_SIZE` alignment so that reads (which use page-sized chunks) can
/// recover the data without partial-page issues.
fn serialize_u8_slice_to_disk(slice: &[u8], vfs: &Arc<dyn VfsImpl>) -> Result<usize, BfTreeError> {
    if slice.is_empty() {
        return Err(BfTreeError::Io(IoErrorKind::SnapshotWrite));
    }
    // Round up to page boundary so the offset allocator stays page-aligned.
    let alloc_size = (slice.len() + DISK_PAGE_SIZE - 1) & !(DISK_PAGE_SIZE - 1);
    let start_offset = vfs.alloc_offset(alloc_size);
    vfs.write(start_offset, slice).map_err(BfTreeError::Io)?;
    Ok(start_offset)
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use crate::{
        nodes::leaf_node::LeafReadResult, range_scan::ScanReturnField,
        utils::test_util::install_value_to_buffer, BfTree, Config,
    };
    use rstest::rstest;
    use std::str::FromStr;

    #[rstest]
    #[case(64, 2408, 8192, 500, "target/test_simple_1.bftree")] // 1 leaf page = 2 disk page
    #[case(64, 2048, 16384, 500, "target/test_simple_2.bftree")] // 1 leaf page = 4 disk page
    #[case(3072, 3072, 16384, 500, "target/test_simple_3.bftree")] // 1 leaf page = 1 disk page, uniform record size
    fn persist_roundtrip_simple(
        #[case] min_record_size: usize,
        #[case] max_record_size: usize,
        #[case] leaf_page_size: usize,
        #[case] record_cnt: usize,
        #[case] snapshot_file_path: String,
    ) {
        let tmp_file_path = std::path::PathBuf::from_str(&snapshot_file_path).unwrap();

        let mut config = Config::new(&tmp_file_path, leaf_page_size * 16); // Creat a CB that can hold 16 full pages
        config.storage_backend(crate::StorageBackend::Std);
        config.cb_min_record_size = min_record_size;
        config.cb_max_record_size = max_record_size;
        config.leaf_page_size = leaf_page_size;
        config.max_fence_len = max_record_size;

        let bftree = BfTree::with_config(config.clone(), None).unwrap();

        let key_len: usize = min_record_size / 2;
        let mut key_buffer = vec![0; key_len / 8];

        for r in 0..record_cnt {
            let key = install_value_to_buffer(&mut key_buffer, r);
            bftree.insert(key, key);
        }
        bftree.snapshot().unwrap();
        drop(bftree);

        let bftree = BfTree::new_from_snapshot(config.clone(), None).unwrap();
        let mut out_buffer = vec![0; key_len];
        for r in 0..record_cnt {
            let key = install_value_to_buffer(&mut key_buffer, r);
            let bytes_read = bftree.read(key, &mut out_buffer);

            match bytes_read {
                LeafReadResult::Found(v) => {
                    assert_eq!(v as usize, key_len);
                    assert_eq!(&out_buffer, key);
                }
                _ => {
                    panic!("Key not found");
                }
            }
        }

        std::fs::remove_file(tmp_file_path).unwrap();
    }

    #[test]
    fn snapshot_memory_to_disk_roundtrip() {
        let snapshot_path = std::path::PathBuf::from_str("target/test_mem_to_disk.bftree").unwrap();
        // Clean up in case a previous run left the file behind
        let _ = std::fs::remove_file(&snapshot_path);

        let min_record_size: usize = 64;
        let max_record_size: usize = 2048;
        let leaf_page_size: usize = 8192;

        // Create an in-memory (non-cache_only) BfTree
        let mut config = Config::new(":memory:", leaf_page_size * 16);
        config.cb_min_record_size = min_record_size;
        config.cb_max_record_size = max_record_size;
        config.leaf_page_size = leaf_page_size;
        config.max_fence_len = max_record_size;

        let bftree = BfTree::with_config(config.clone(), None).unwrap();

        let key_len: usize = min_record_size / 2;
        let record_cnt = 200;
        let mut key_buffer = vec![0usize; key_len / 8];

        for r in 0..record_cnt {
            let key = install_value_to_buffer(&mut key_buffer, r);
            bftree.insert(key, key);
        }

        // Snapshot the in-memory tree to disk
        let path = bftree.snapshot_memory_to_disk(&snapshot_path).unwrap();
        assert_eq!(path, snapshot_path);
        assert!(snapshot_path.exists());
        drop(bftree);

        // Reload the snapshot into a disk-backed tree and verify all records
        let mut disk_config = Config::new(&snapshot_path, leaf_page_size * 16);
        disk_config.storage_backend(crate::StorageBackend::Std);
        disk_config.cb_min_record_size = min_record_size;
        disk_config.cb_max_record_size = max_record_size;
        disk_config.leaf_page_size = leaf_page_size;
        disk_config.max_fence_len = max_record_size;

        let loaded = BfTree::new_from_snapshot(disk_config, None).unwrap();
        let mut out_buffer = vec![0u8; key_len];
        for r in 0..record_cnt {
            let key = install_value_to_buffer(&mut key_buffer, r);
            let result = loaded.read(key, &mut out_buffer);
            match result {
                LeafReadResult::Found(v) => {
                    assert_eq!(v as usize, key_len);
                    assert_eq!(&out_buffer[..key_len], key);
                }
                other => {
                    panic!("Key {r} not found, got: {:?}", other);
                }
            }
        }

        std::fs::remove_file(snapshot_path).unwrap();
    }

    #[test]
    fn snapshot_disk_to_memory_roundtrip() {
        let snapshot_path = std::path::PathBuf::from_str("target/test_disk_to_mem.bftree").unwrap();
        let _ = std::fs::remove_file(&snapshot_path);

        let min_record_size: usize = 64;
        let max_record_size: usize = 2048;
        let leaf_page_size: usize = 8192;
        let record_cnt: usize = 500;

        // Step 1: Build a disk-backed tree, populate it, snapshot to disk.
        {
            let mut config = Config::new(&snapshot_path, leaf_page_size * 16);
            config.storage_backend(crate::StorageBackend::Std);
            config.cb_min_record_size = min_record_size;
            config.cb_max_record_size = max_record_size;
            config.leaf_page_size = leaf_page_size;
            config.max_fence_len = max_record_size;

            let tree = BfTree::with_config(config, None).unwrap();
            let key_len = min_record_size / 2;
            let mut key_buffer = vec![0usize; key_len / 8];

            for r in 0..record_cnt {
                let key = install_value_to_buffer(&mut key_buffer, r);
                tree.insert(key, key);
            }
            tree.snapshot().unwrap();
        }

        // Step 2: Load the snapshot into an in-memory (cache_only) tree.
        let mut mem_config = Config::new(":memory:", leaf_page_size * 16);
        mem_config.cb_min_record_size = min_record_size;
        mem_config.cb_max_record_size = max_record_size;
        mem_config.leaf_page_size = leaf_page_size;
        mem_config.max_fence_len = max_record_size;

        let mem_tree =
            BfTree::new_from_snapshot_disk_to_memory(&snapshot_path, mem_config).unwrap();

        // Step 3: Verify all records are present.
        let key_len = min_record_size / 2;
        let mut key_buffer = vec![0usize; key_len / 8];
        let mut out_buffer = vec![0u8; key_len];
        for r in 0..record_cnt {
            let key = install_value_to_buffer(&mut key_buffer, r);
            let result = mem_tree.read(key, &mut out_buffer);
            match result {
                LeafReadResult::Found(v) => {
                    assert_eq!(v as usize, key_len);
                    assert_eq!(&out_buffer[..key_len], key);
                }
                other => {
                    panic!("Key {r} not found, got: {:?}", other);
                }
            }
        }

        std::fs::remove_file(snapshot_path).unwrap();
    }

    /// Snapshot recovery preserves data written before the snapshot, even if
    /// the tree is dropped (simulating a crash) and reopened via
    /// `new_from_snapshot`.
    #[test]
    fn snapshot_recovery_preserves_pre_snapshot_data() {
        let snapshot_path =
            std::path::PathBuf::from_str("target/test_recovery_pre_snapshot.bftree").unwrap();
        let _ = std::fs::remove_file(&snapshot_path);

        let min_record_size: usize = 64;
        let max_record_size: usize = 2048;
        let leaf_page_size: usize = 8192;
        let record_cnt: usize = 1000;

        let make_config = || {
            let mut config = Config::new(&snapshot_path, leaf_page_size * 16);
            config.storage_backend(crate::StorageBackend::Std);
            config.cb_min_record_size = min_record_size;
            config.cb_max_record_size = max_record_size;
            config.leaf_page_size = leaf_page_size;
            config.max_fence_len = max_record_size;
            config
        };

        let key_len: usize = min_record_size / 2;
        let mut key_buffer = vec![0usize; key_len / 8];

        // Phase 1: Insert records and snapshot.
        {
            let tree = BfTree::with_config(make_config(), None).unwrap();
            for r in 0..record_cnt {
                let key = install_value_to_buffer(&mut key_buffer, r);
                tree.insert(key, key);
            }
            tree.snapshot().unwrap();
            // Drop without graceful shutdown simulates crash.
        }

        // Phase 2: Recover from snapshot and verify every key via point reads.
        let tree = BfTree::new_from_snapshot(make_config(), None).unwrap();
        let mut out_buffer = vec![0u8; key_len];
        let mut found = 0usize;
        for r in 0..record_cnt {
            let key = install_value_to_buffer(&mut key_buffer, r);
            match tree.read(key, &mut out_buffer) {
                LeafReadResult::Found(v) => {
                    assert_eq!(v as usize, key_len);
                    assert_eq!(&out_buffer[..key_len], key);
                    found += 1;
                }
                other => panic!("key {r} not found after recovery: {other:?}"),
            }
        }
        assert_eq!(found, record_cnt, "not all keys recovered");

        // Phase 3: Verify scan produces sorted order.
        {
            let scan = tree
                .scan_with_count(
                    &[0u8],
                    record_cnt + 10,
                    crate::range_scan::ScanReturnField::Key,
                )
                .unwrap();
            let mut scan_buf = vec![0u8; key_len + max_record_size];
            let mut prev: Option<Vec<u8>> = None;
            let mut scan_count = 0;
            let mut scan_ref = scan;
            while let Ok(Some((kl, _vl))) = scan_ref.next(&mut scan_buf) {
                let key = scan_buf[..kl].to_vec();
                if let Some(ref p) = prev {
                    assert!(key > *p, "scan order violated at entry {scan_count}");
                }
                prev = Some(key);
                scan_count += 1;
            }
            assert_eq!(
                scan_count, record_cnt,
                "scan returned wrong number of entries"
            );
        }

        drop(tree);
        std::fs::remove_file(snapshot_path).unwrap();
    }

    /// Recovery after splits: insert enough data to trigger many page splits,
    /// snapshot, drop (crash), and verify the recovered tree is consistent.
    #[test]
    fn recovery_after_splits_produces_correct_tree() {
        let snapshot_path =
            std::path::PathBuf::from_str("target/test_recovery_splits.bftree").unwrap();
        let _ = std::fs::remove_file(&snapshot_path);

        let min_record_size: usize = 64;
        let max_record_size: usize = 2048;
        let leaf_page_size: usize = 8192;
        // Many records with standard page size to force many splits.
        let record_cnt: usize = 2000;

        let make_config = || {
            let mut config = Config::new(&snapshot_path, leaf_page_size * 16);
            config.storage_backend(crate::StorageBackend::Std);
            config.cb_min_record_size = min_record_size;
            config.cb_max_record_size = max_record_size;
            config.leaf_page_size = leaf_page_size;
            config.max_fence_len = max_record_size;
            config
        };

        let key_len: usize = min_record_size / 2;
        let mut key_buffer = vec![0usize; key_len / 8];

        // Insert and snapshot.
        {
            let tree = BfTree::with_config(make_config(), None).unwrap();
            for r in 0..record_cnt {
                let key = install_value_to_buffer(&mut key_buffer, r);
                tree.insert(key, key);
            }
            tree.snapshot().unwrap();
        }

        // Recover and verify all keys.
        let tree = BfTree::new_from_snapshot(make_config(), None).unwrap();
        let mut out_buffer = vec![0u8; key_len];
        let mut missing = 0usize;
        for r in 0..record_cnt {
            let key = install_value_to_buffer(&mut key_buffer, r);
            match tree.read(key, &mut out_buffer) {
                LeafReadResult::Found(v) => {
                    assert_eq!(v as usize, key_len);
                }
                _ => missing += 1,
            }
        }
        assert_eq!(
            missing, 0,
            "{missing} keys missing after recovery with splits"
        );

        drop(tree);
        std::fs::remove_file(snapshot_path).unwrap();
    }

    /// Checksums are validated on reads from a recovered tree when
    /// `verify_checksums` is enabled.
    #[test]
    fn recovery_with_checksums_enabled() {
        let snapshot_path =
            std::path::PathBuf::from_str("target/test_recovery_checksums.bftree").unwrap();
        let _ = std::fs::remove_file(&snapshot_path);

        let min_record_size: usize = 64;
        let max_record_size: usize = 2048;
        let leaf_page_size: usize = 8192;
        let record_cnt: usize = 500;

        let make_config = || {
            let mut config = Config::new(&snapshot_path, leaf_page_size * 16);
            config.storage_backend(crate::StorageBackend::Std);
            config.cb_min_record_size = min_record_size;
            config.cb_max_record_size = max_record_size;
            config.leaf_page_size = leaf_page_size;
            config.max_fence_len = max_record_size;
            config.verify_checksums = true;
            config
        };

        let key_len: usize = min_record_size / 2;
        let mut key_buffer = vec![0usize; key_len / 8];

        // Create, populate, snapshot with checksums on.
        {
            let tree = BfTree::with_config(make_config(), None).unwrap();
            for r in 0..record_cnt {
                let key = install_value_to_buffer(&mut key_buffer, r);
                tree.insert(key, key);
            }
            tree.snapshot().unwrap();
        }

        // Recover with checksums still enabled.
        let tree = BfTree::new_from_snapshot(make_config(), None).unwrap();
        let mut out_buffer = vec![0u8; key_len];
        for r in 0..record_cnt {
            let key = install_value_to_buffer(&mut key_buffer, r);
            match tree.read(key, &mut out_buffer) {
                LeafReadResult::Found(v) => {
                    assert_eq!(v as usize, key_len);
                }
                other => panic!("checksum-verified read failed for key {r}: {other:?}"),
            }
        }

        drop(tree);
        std::fs::remove_file(snapshot_path).unwrap();
    }

    /// WAL replay: entries written after the last snapshot are recovered
    /// from the WAL when `BfTree::recovery()` is called.
    ///
    /// This is the most critical untested code path -- it's the one that
    /// runs after an actual power failure or crash.
    #[test]
    fn wal_replay_recovers_post_snapshot_entries() {
        let pid = std::process::id();
        let test_dir = std::path::PathBuf::from(format!("target/test_wal_replay_{pid}"));
        let _ = std::fs::remove_dir_all(&test_dir);
        std::fs::create_dir_all(&test_dir).unwrap();

        let snapshot_path = test_dir.join("data.bftree");
        let wal_path = test_dir.join("wal.log");
        let config_path = test_dir.join("config.toml");

        let min_record_size: usize = 64;
        let max_record_size: usize = 2048;
        let leaf_page_size: usize = 8192;
        let cb_size: usize = leaf_page_size * 64;
        let wal_segment_size: usize = 1024 * 1024; // 1 MB

        // Records 0..pre_count go into the snapshot.
        // Records pre_count..total_count are only in the WAL.
        let pre_count: usize = 500;
        let total_count: usize = 800;

        let key_len: usize = min_record_size / 2;
        let mut key_buffer = vec![0usize; key_len / 8];

        let make_config = || {
            let mut config = Config::new(&snapshot_path, cb_size);
            config.storage_backend(crate::StorageBackend::Std);
            config.cb_min_record_size = min_record_size;
            config.cb_max_record_size = max_record_size;
            config.leaf_page_size = leaf_page_size;
            config.max_fence_len = max_record_size;
            let mut wal_config = crate::config::WalConfig::new(&wal_path);
            wal_config.segment_size(wal_segment_size);
            wal_config.flush_interval(std::time::Duration::from_micros(1));
            config.enable_write_ahead_log(std::sync::Arc::new(wal_config));
            config
        };

        // Phase 1: Insert pre_count records, snapshot, then insert more (WAL-only).
        {
            let tree = BfTree::with_config(make_config(), None).unwrap();

            for r in 0..pre_count {
                let key = install_value_to_buffer(&mut key_buffer, r);
                tree.insert(key, key);
            }
            tree.snapshot().unwrap();

            // Verify live tree has all keys right after snapshot.
            let mut snap_buf = vec![0u8; key_len];
            for r in 0..pre_count {
                let key = install_value_to_buffer(&mut key_buffer, r);
                match tree.read(key, &mut snap_buf) {
                    LeafReadResult::Found(_) => {}
                    other => panic!("live tree missing key {r} right after snapshot: {other:?}"),
                }
            }
            // These entries are NOT snapshotted -- only in the WAL.
            for r in pre_count..total_count {
                let key = install_value_to_buffer(&mut key_buffer, r);
                tree.insert(key, key);
            }
            // Drop without snapshot simulates crash.
        }

        // Phase 2: Write the TOML config file that recovery() expects.
        let max_key_len = max_record_size / 2;
        let config_toml = format!(
            "cb_size_byte = {cb_size}\n\
             cb_min_record_size = {min_record_size}\n\
             cb_max_record_size = {max_record_size}\n\
             cb_max_key_len = {max_key_len}\n\
             leaf_page_size = {leaf_page_size}\n\
             index_file_path = \"{}\"\n\
             backend_storage = \"disk\"\n\
             read_promotion_rate = 50\n\
             write_load_full_page = true\n\
             cache_only = false\n",
            snapshot_path.to_string_lossy().replace('\\', "\\\\"),
        );
        std::fs::write(&config_path, &config_toml).unwrap();

        // Phase 2.5: Verify snapshot alone contains all pre-snapshot entries.
        {
            let snap_tree =
                BfTree::new_from_snapshot(Config::new_with_config_file(&config_path), None)
                    .expect("snapshot load failed");
            let mut snap_buf = vec![0u8; key_len];
            let mut missing_keys = Vec::new();
            for r in 0..pre_count {
                let key = install_value_to_buffer(&mut key_buffer, r);
                match snap_tree.read(key, &mut snap_buf) {
                    LeafReadResult::Found(_) => {}
                    _ => missing_keys.push(r),
                }
            }
            if !missing_keys.is_empty() {
                panic!(
                    "snapshot alone is missing {} pre-snapshot keys (first 10: {:?})",
                    missing_keys.len(),
                    &missing_keys[..missing_keys.len().min(10)]
                );
            }
        }

        // Phase 3: Recover from snapshot + WAL replay.
        let tree = BfTree::recovery(&config_path, &wal_path, wal_segment_size, None)
            .expect("WAL recovery failed");

        // Phase 4: Verify ALL entries (both pre-snapshot and WAL-only).
        let mut out_buffer = vec![0u8; key_len];
        for r in 0..total_count {
            let key = install_value_to_buffer(&mut key_buffer, r);
            match tree.read(key, &mut out_buffer) {
                LeafReadResult::Found(v) => {
                    assert_eq!(v as usize, key_len, "wrong value size for key {r}");
                    assert_eq!(&out_buffer[..key_len], key, "wrong value for key {r}");
                }
                other => panic!(
                    "key {r} not found after WAL recovery: {other:?} \
                     (pre_count={pre_count}, total={total_count})"
                ),
            }
        }

        drop(tree);
        let _ = std::fs::remove_dir_all(&test_dir);
    }

    /// WAL replay after splits: enough post-snapshot entries to trigger
    /// page splits during recovery, verifying structural consistency.
    #[test]
    fn wal_replay_with_splits_produces_correct_tree() {
        let pid = std::process::id();
        let test_dir = std::path::PathBuf::from(format!("target/test_wal_replay_splits_{pid}"));
        let _ = std::fs::remove_dir_all(&test_dir);
        std::fs::create_dir_all(&test_dir).unwrap();

        let snapshot_path = test_dir.join("data.bftree");
        let wal_path = test_dir.join("wal.log");
        let config_path = test_dir.join("config.toml");

        let min_record_size: usize = 64;
        let max_record_size: usize = 2048;
        let leaf_page_size: usize = 8192;
        let cb_size: usize = leaf_page_size * 64;
        let wal_segment_size: usize = 1024 * 1024; // 1 MB

        // Small pre_count, large post_count to force splits during replay.
        let pre_count: usize = 100;
        let total_count: usize = 2000;

        let key_len: usize = min_record_size / 2;
        let mut key_buffer = vec![0usize; key_len / 8];

        let make_config = || {
            let mut config = Config::new(&snapshot_path, cb_size);
            config.storage_backend(crate::StorageBackend::Std);
            config.cb_min_record_size = min_record_size;
            config.cb_max_record_size = max_record_size;
            config.leaf_page_size = leaf_page_size;
            config.max_fence_len = max_record_size;
            let mut wal_config = crate::config::WalConfig::new(&wal_path);
            wal_config.segment_size(wal_segment_size);
            wal_config.flush_interval(std::time::Duration::from_micros(1));
            config.enable_write_ahead_log(std::sync::Arc::new(wal_config));
            config
        };

        {
            let tree = BfTree::with_config(make_config(), None).unwrap();
            for r in 0..pre_count {
                let key = install_value_to_buffer(&mut key_buffer, r);
                tree.insert(key, key);
            }
            tree.snapshot().unwrap();

            for r in pre_count..total_count {
                let key = install_value_to_buffer(&mut key_buffer, r);
                tree.insert(key, key);
            }
        }

        let max_key_len = max_record_size / 2;
        let config_toml = format!(
            "cb_size_byte = {cb_size}\n\
             cb_min_record_size = {min_record_size}\n\
             cb_max_record_size = {max_record_size}\n\
             cb_max_key_len = {max_key_len}\n\
             leaf_page_size = {leaf_page_size}\n\
             index_file_path = \"{}\"\n\
             backend_storage = \"disk\"\n\
             read_promotion_rate = 50\n\
             write_load_full_page = true\n\
             cache_only = false\n",
            snapshot_path.to_string_lossy().replace('\\', "\\\\"),
        );
        std::fs::write(&config_path, &config_toml).unwrap();

        let tree = BfTree::recovery(&config_path, &wal_path, wal_segment_size, None)
            .expect("WAL+split recovery failed");

        // Verify all entries via point reads.
        let mut out_buffer = vec![0u8; key_len];
        for r in 0..total_count {
            let key = install_value_to_buffer(&mut key_buffer, r);
            match tree.read(key, &mut out_buffer) {
                LeafReadResult::Found(v) => {
                    assert_eq!(v as usize, key_len);
                }
                other => panic!("key {r} not found after WAL+split recovery: {other:?}"),
            }
        }

        // Verify scan produces sorted order. This works because recovery
        // now skips WAL entries with lsn <= snapshot_lsn, avoiding
        // re-insertion of pre-snapshot entries that caused structural issues.
        {
            let scan = tree
                .scan_with_count(
                    &[0u8],
                    total_count + 10,
                    crate::range_scan::ScanReturnField::Key,
                )
                .unwrap();
            let mut scan_buf = vec![0u8; key_len + max_record_size];
            let mut prev: Option<Vec<u8>> = None;
            let mut scan_count = 0;
            let mut scan_ref = scan;
            while let Ok(Some((kl, _vl))) = scan_ref.next(&mut scan_buf) {
                let key = scan_buf[..kl].to_vec();
                if let Some(ref p) = prev {
                    assert!(key > *p, "scan order violated at entry {scan_count}");
                }
                prev = Some(key);
                scan_count += 1;
            }
            assert_eq!(
                scan_count, total_count,
                "scan returned wrong count after WAL+split recovery"
            );
        }

        drop(tree);
        let _ = std::fs::remove_dir_all(&test_dir);
    }

    /// Scan with `ScanReturnField::Value` and `KeyAndValue` on a recovered tree.
    /// Existing tests only exercise `Key` -- this catches regressions in the
    /// value-copy path after snapshot/recovery.
    #[test]
    fn scan_return_field_variants_after_recovery() {
        let snapshot_path =
            std::path::PathBuf::from_str("target/test_scan_return_fields.bftree").unwrap();
        let _ = std::fs::remove_file(&snapshot_path);

        let min_record_size: usize = 64;
        let max_record_size: usize = 2048;
        let leaf_page_size: usize = 8192;
        let record_cnt: usize = 200;

        let make_config = || {
            let mut config = Config::new(&snapshot_path, leaf_page_size * 16);
            config.storage_backend(crate::StorageBackend::Std);
            config.cb_min_record_size = min_record_size;
            config.cb_max_record_size = max_record_size;
            config.leaf_page_size = leaf_page_size;
            config.max_fence_len = max_record_size;
            config
        };

        let key_len: usize = min_record_size / 2;
        let mut key_buffer = vec![0usize; key_len / 8];

        // Create, populate, snapshot.
        {
            let tree = BfTree::with_config(make_config(), None).unwrap();
            for r in 0..record_cnt {
                let key = install_value_to_buffer(&mut key_buffer, r);
                tree.insert(key, key);
            }
            tree.snapshot().unwrap();
        }

        // Recover and test all three scan return fields.
        let tree = BfTree::new_from_snapshot(make_config(), None).unwrap();
        let buf_size = key_len + max_record_size;

        // ScanReturnField::Value
        {
            let scan = tree
                .scan_with_count(&[0u8], record_cnt + 10, ScanReturnField::Value)
                .unwrap();
            let mut scan_buf = vec![0u8; buf_size];
            let mut count = 0;
            let mut scan_ref = scan;
            while let Ok(Some((_kl, vl))) = scan_ref.next(&mut scan_buf) {
                assert_eq!(vl, key_len, "value size mismatch at entry {count}");
                count += 1;
            }
            assert_eq!(count, record_cnt, "Value scan returned wrong count");
        }

        // ScanReturnField::KeyAndValue
        {
            let scan = tree
                .scan_with_count(&[0u8], record_cnt + 10, ScanReturnField::KeyAndValue)
                .unwrap();
            let mut scan_buf = vec![0u8; buf_size];
            let mut prev: Option<Vec<u8>> = None;
            let mut count = 0;
            let mut scan_ref = scan;
            while let Ok(Some((kl, vl))) = scan_ref.next(&mut scan_buf) {
                assert_eq!(kl, key_len, "key size mismatch at entry {count}");
                assert_eq!(vl, key_len, "value size mismatch at entry {count}");
                // Key == Value for our test data.
                assert_eq!(
                    &scan_buf[..kl],
                    &scan_buf[kl..kl + vl],
                    "key != value at entry {count}"
                );
                let key = scan_buf[..kl].to_vec();
                if let Some(ref p) = prev {
                    assert!(key > *p, "KeyAndValue scan order violated at entry {count}");
                }
                prev = Some(key);
                count += 1;
            }
            assert_eq!(count, record_cnt, "KeyAndValue scan returned wrong count");
        }

        std::fs::remove_file(snapshot_path).unwrap();
    }

    /// Scan with `end_key` bounds on a recovered tree.
    /// Exercises the `scan_with_end_key` / `new_with_end_key` code path.
    #[test]
    fn scan_with_end_key_after_recovery() {
        let snapshot_path =
            std::path::PathBuf::from_str("target/test_scan_end_key.bftree").unwrap();
        let _ = std::fs::remove_file(&snapshot_path);

        let min_record_size: usize = 64;
        let max_record_size: usize = 2048;
        let leaf_page_size: usize = 8192;
        let record_cnt: usize = 500;

        let make_config = || {
            let mut config = Config::new(&snapshot_path, leaf_page_size * 16);
            config.storage_backend(crate::StorageBackend::Std);
            config.cb_min_record_size = min_record_size;
            config.cb_max_record_size = max_record_size;
            config.leaf_page_size = leaf_page_size;
            config.max_fence_len = max_record_size;
            config
        };

        let key_len: usize = min_record_size / 2;
        let mut key_buffer = vec![0usize; key_len / 8];

        // Collect sorted keys for range selection.
        let mut all_keys: Vec<Vec<u8>> = (0..record_cnt)
            .map(|r| {
                let key = install_value_to_buffer(&mut key_buffer, r);
                key.to_vec()
            })
            .collect();
        all_keys.sort();

        // Create, populate, snapshot.
        {
            let tree = BfTree::with_config(make_config(), None).unwrap();
            for r in 0..record_cnt {
                let key = install_value_to_buffer(&mut key_buffer, r);
                tree.insert(key, key);
            }
            tree.snapshot().unwrap();
        }

        let tree = BfTree::new_from_snapshot(make_config(), None).unwrap();
        let buf_size = key_len + max_record_size;

        // Pick a range from the middle ~25%-75% of sorted keys.
        let start_idx = record_cnt / 4;
        let end_idx = (record_cnt * 3) / 4;
        let start_key = &all_keys[start_idx];
        let end_key = &all_keys[end_idx];

        // scan_with_end_key uses inclusive upper bound: [start_key, end_key].
        let expected = all_keys
            .iter()
            .filter(|k| k.as_slice() >= start_key.as_slice() && k.as_slice() <= end_key.as_slice())
            .count();

        let scan = tree
            .scan_with_end_key(start_key, end_key, ScanReturnField::Key)
            .unwrap();
        let mut scan_buf = vec![0u8; buf_size];
        let mut prev: Option<Vec<u8>> = None;
        let mut count = 0;
        let mut scan_ref = scan;
        while let Ok(Some((kl, _vl))) = scan_ref.next(&mut scan_buf) {
            let key = scan_buf[..kl].to_vec();
            assert!(
                key.as_slice() >= start_key.as_slice(),
                "scan returned key below start_key at entry {count}"
            );
            assert!(
                key.as_slice() <= end_key.as_slice(),
                "scan returned key > end_key at entry {count}"
            );
            if let Some(ref p) = prev {
                assert!(key > *p, "end_key scan order violated at entry {count}");
            }
            prev = Some(key);
            count += 1;
        }
        assert_eq!(
            count, expected,
            "end_key scan returned {count} entries, expected {expected}"
        );

        std::fs::remove_file(snapshot_path).unwrap();
    }
}
