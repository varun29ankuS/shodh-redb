// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::path::Path;
use std::sync::Arc;

mod operations;

use crate::bf_tree::config::WalConfig;
use crate::bf_tree::error::{IoErrorKind, TreeError};
use crate::bf_tree::fs::VfsImpl;
use crate::bf_tree::storage::make_vfs;
use crate::bf_tree::sync::atomic::AtomicBool;
use std::sync::{Condvar, Mutex};

pub(crate) use operations::{LogEntry, SplitOp, WriteOp};

const BLOCK_SIZE: usize = 512;

pub(crate) trait LogEntryImpl<'a> {
    fn log_size(&self) -> usize;
    fn write_to_buffer(&self, buffer: &mut [u8]);
    fn read_from_buffer(buffer: &'a [u8]) -> Self;
}

/// Ptr aligned to block size, so that it can be directly write to storage device
struct RawBuffer {
    buffer_size: usize,
    ptr: *mut u8,
}

impl RawBuffer {
    fn new(buffer_size: usize) -> RawBuffer {
        let layout = std::alloc::Layout::from_size_align(buffer_size, BLOCK_SIZE).unwrap();
        // SAFETY: layout has non-zero size (buffer_size) and BLOCK_SIZE alignment (power of 2).
        let ptr = unsafe { std::alloc::alloc(layout) };
        RawBuffer { ptr, buffer_size }
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: self.ptr was allocated with buffer_size bytes in new() and remains
        // valid for the lifetime of RawBuffer. The slice borrows &self preventing mutation.
        unsafe { std::slice::from_raw_parts(self.ptr, self.buffer_size) }
    }

    /// Return a slice covering only the first `len` bytes of the buffer.
    ///
    /// # Safety
    /// Caller must ensure `len <= self.buffer_size`.
    unsafe fn as_slice_len(&self, len: usize) -> &[u8] {
        debug_assert!(len <= self.buffer_size);
        // SAFETY: self.ptr valid for buffer_size bytes; len <= buffer_size by caller contract.
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }

    /// # Safety
    /// Caller must ensure `offset + size <= buffer_size` and that no other references
    /// overlap the returned slice for the duration of the borrow.
    unsafe fn as_mut_slice_at_exact(&mut self, offset: usize, size: usize) -> &mut [u8] {
        debug_assert!(
            offset
                .checked_add(size)
                .is_some_and(|end| end <= self.buffer_size),
            "as_mut_slice_at_exact: offset({offset}) + size({size}) overflows or exceeds buffer_size({})",
            self.buffer_size,
        );
        // SAFETY: Precondition guaranteed by caller; ptr is valid for buffer_size bytes.
        // The debug_assert above catches violations in debug builds.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), size) }
    }
}

// SAFETY: RawBuffer owns its heap allocation exclusively; the raw pointer does not
// reference thread-local or non-Send data, so it is safe to transfer across threads.
unsafe impl Send for RawBuffer {}
// SAFETY: RawBuffer is only mutated through &mut self, so shared references (&self)
// cannot cause data races. Concurrent reads of the buffer are safe.
unsafe impl Sync for RawBuffer {}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.buffer_size, BLOCK_SIZE).unwrap();
        // SAFETY: self.ptr was allocated with this exact layout in new() and has not
        // been deallocated (Drop runs exactly once).
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

struct WriteAheadLogInner {
    buffer: RawBuffer,
    file_handle: Arc<dyn VfsImpl>,
    buffer_cursor: usize,
    file_offset: usize,
    next_lsn: u64,
    flushed_lsn: u64,
    need_flush: bool,
    /// Sticky I/O error from the last flush attempt. Once set, all subsequent
    /// operations that require durability will fail until the WAL is reset.
    /// This prevents silent data loss when the storage layer fails.
    last_io_error: Option<IoErrorKind>,
}

impl WriteAheadLogInner {
    fn flush(&mut self) {
        if self.buffer_cursor == 0 {
            // nothing to flush
            return;
        }

        self.clear_next_header();
        // Write only the used portion of the buffer, not the entire segment.
        // The segment may be up to 1 GiB; writing unused zero-fill is wasteful
        // and catastrophically slow on Windows.
        // SAFETY: buffer_cursor <= buffer_size guaranteed by alloc_buffer's debug_assert.
        let used_slice = unsafe { self.buffer.as_slice_len(self.buffer_cursor) };
        if let Err(_e) = self.file_handle.write(self.file_offset, used_slice) {
            self.last_io_error = Some(IoErrorKind::VfsWrite {
                offset: self.file_offset,
            });
            return;
        }
        // NOTE: fsync is required after write to guarantee WAL durability on crash.
        if let Err(_e) = self.file_handle.flush() {
            self.last_io_error = Some(IoErrorKind::WalFlush);
            return;
        }

        // Always advance -- append-only WAL. Old in-place rewrite caused
        // degenerate read-modify-write cycles on NTFS when rewriting the same
        // file offset repeatedly for small flushes.
        self.file_offset += self.buffer_cursor;
        self.buffer_cursor = 0;

        self.flushed_lsn = self.next_lsn - 1;
        self.need_flush = false;
    }

    /// Check for and return any sticky I/O error from a previous flush.
    fn check_io_error(&self) -> Result<(), TreeError> {
        if let Some(ref err) = self.last_io_error {
            Err(TreeError::IoError(err.clone()))
        } else {
            Ok(())
        }
    }

    fn clear_next_header(&mut self) {
        if self.buffer_cursor + 8 <= self.buffer.buffer_size {
            // SAFETY: the guard `buffer_cursor + 8 <= buffer_size` ensures the range is in bounds.
            let slice = unsafe { self.buffer.as_mut_slice_at_exact(self.buffer_cursor, 8) };
            slice.copy_from_slice(&[0u8; 8]);
        }
    }

    unsafe fn alloc_buffer(&mut self, size: usize) -> &mut [u8] {
        debug_assert!(
            self.buffer_cursor + size <= self.buffer.buffer_size,
            "buffer overflow"
        );
        let cursor = self.buffer_cursor;
        self.buffer_cursor += size;
        // SAFETY: debug_assert above verifies cursor + size <= buffer_size. The &mut self
        // borrow ensures no other references to the buffer exist concurrently.
        unsafe { self.buffer.as_mut_slice_at_exact(cursor, size) }
    }

    fn alloc_lsn(&mut self) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        lsn
    }
}

pub(crate) struct WriteAheadLog {
    inner: Mutex<WriteAheadLogInner>,
    flushed_cond: Condvar,    // for workers that waiting for flush
    need_flush_cond: Condvar, // for background job
    background_job_running: AtomicBool,
    config: Arc<WalConfig>,
}

impl WriteAheadLog {
    /// Create a new wal instance, and start a background thread to flush wal buffer.
    pub(crate) fn new(
        config: Arc<WalConfig>,
    ) -> Result<Arc<Self>, crate::bf_tree::error::IoErrorKind> {
        let vfs = make_vfs(&config.storage_backend, &config.file_path)?;
        let wal = WriteAheadLog {
            inner: Mutex::new(WriteAheadLogInner {
                buffer: RawBuffer::new(config.segment_size),
                file_handle: vfs,
                buffer_cursor: 0,
                file_offset: 0,
                next_lsn: 0,
                flushed_lsn: 0,
                need_flush: false,
                last_io_error: None,
            }),
            flushed_cond: Condvar::new(),
            need_flush_cond: Condvar::new(),
            background_job_running: AtomicBool::new(true),
            config,
        };

        let wal = Arc::new(wal);
        WriteAheadLog::start_flush_job(wal.clone());
        Ok(wal)
    }

    fn start_flush_job(wal: Arc<Self>) {
        let h = crate::bf_tree::sync::thread::spawn(move || wal.background_flush_job());
        drop(h); // detach the thread
    }

    /// Returns the LSN up to which all entries have been durably flushed.
    /// Used by snapshot to record the WAL high-water mark.
    pub(crate) fn get_flushed_lsn(&self) -> u64 {
        match self.inner.lock() {
            Ok(inner) => inner.flushed_lsn,
            Err(_) => 0,
        }
    }

    pub(crate) fn stop_background_job(&self) {
        self.background_job_running
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.need_flush_cond.notify_all();
    }

    pub(crate) fn background_flush_job(&self) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_poisoned) => return, // prior panic poisoned the mutex; exit cleanly
        };

        let flush_interval = self.config.flush_interval;
        let mut last_flush = std::time::Instant::now();
        loop {
            let v = match self.need_flush_cond.wait_timeout(inner, flush_interval) {
                Ok(v) => v,
                Err(_poisoned) => return, // mutex poisoned; exit cleanly
            };

            inner = v.0;

            if !self
                .background_job_running
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                // stop the background job, gracefully shutdown.
                break;
            }

            if inner.need_flush || last_flush.elapsed() > flush_interval {
                inner.flush();
                last_flush = std::time::Instant::now();
                self.flushed_cond.notify_all();
            }
        }
    }

    #[must_use = "The returned flushed lsn must be write to page meta"]
    pub(crate) fn append_and_wait<'a>(
        &self,
        log_entry: &impl LogEntryImpl<'a>,
        page_offset: u64,
    ) -> Result<u64, TreeError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| TreeError::IoError(IoErrorKind::WalAppend))?;

        // Fail fast if a previous flush encountered an I/O error.
        inner.check_io_error()?;

        // log header + wal size
        let required_bytes = std::mem::size_of::<LogHeader>() + log_entry.log_size();
        let remaining = inner.buffer.buffer_size - inner.buffer_cursor;
        if required_bytes > remaining {
            // Buffer full -- flush directly (caller-side) then retry.
            inner.flush();
            inner.check_io_error()?;
            self.flushed_cond.notify_all();
            drop(inner);
            return self.append_and_wait(log_entry, page_offset);
        }

        let lsn = inner.alloc_lsn();
        let header = LogHeader::new(lsn, page_offset, required_bytes);
        // SAFETY: `alloc_buffer` returns a mutable slice within the pre-allocated WAL buffer.
        // The debug_assert inside guarantees `buffer_cursor + required_bytes <= buffer_size`.
        let buffer = unsafe { inner.alloc_buffer(required_bytes) };
        buffer[0..LogHeader::size()].copy_from_slice(header.as_slice());
        log_entry.write_to_buffer(&mut buffer[LogHeader::size()..]);

        // Caller-side flush: write+fsync directly instead of waiting for
        // background thread. This eliminates condvar round-trip latency.
        inner.flush();
        inner.check_io_error()?;
        self.flushed_cond.notify_all();

        Ok(lsn)
    }

    /// Append a log entry to the WAL buffer without waiting for fsync.
    ///
    /// Returns the assigned LSN. The caller must call `wait_for_lsn` later
    /// to ensure durability. This enables batch-append patterns where many
    /// entries are buffered before a single fsync.
    #[must_use = "The returned LSN must be passed to wait_for_lsn for durability"]
    pub(crate) fn append_no_wait<'a>(
        &self,
        log_entry: &impl LogEntryImpl<'a>,
        page_offset: u64,
    ) -> Result<u64, TreeError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| TreeError::IoError(IoErrorKind::WalAppend))?;

        // Fail fast if a previous flush encountered an I/O error.
        inner.check_io_error()?;

        let required_bytes = std::mem::size_of::<LogHeader>() + log_entry.log_size();
        let remaining = inner.buffer.buffer_size - inner.buffer_cursor;
        if required_bytes > remaining {
            // Buffer full -- flush directly (caller-side) then retry.
            inner.flush();
            inner.check_io_error()?;
            self.flushed_cond.notify_all();
            drop(inner);
            return self.append_no_wait(log_entry, page_offset);
        }

        let lsn = inner.alloc_lsn();
        let header = LogHeader::new(lsn, page_offset, required_bytes);
        // SAFETY: `alloc_buffer` returns a mutable slice within the pre-allocated WAL buffer.
        // The debug_assert inside guarantees `buffer_cursor + required_bytes <= buffer_size`.
        let buffer = unsafe { inner.alloc_buffer(required_bytes) };
        buffer[0..LogHeader::size()].copy_from_slice(header.as_slice());
        log_entry.write_to_buffer(&mut buffer[LogHeader::size()..]);

        Ok(lsn)
    }

    /// Flush all buffered WAL entries and block until fsync completes.
    ///
    /// The calling thread performs the write+fsync directly while holding the
    /// mutex, eliminating the condvar round-trip to the background thread.
    /// This naturally provides group commit: if multiple threads call this
    /// concurrently, the first to acquire the lock flushes all pending entries
    /// in a single write+fsync, and subsequent callers see their LSN already
    /// flushed.
    ///
    /// Returns the flushed LSN. No-op if the buffer is empty.
    pub(crate) fn flush_and_wait(&self) -> Result<u64, TreeError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| TreeError::IoError(IoErrorKind::WalFlush))?;

        // Fail fast if a previous flush encountered an I/O error.
        inner.check_io_error()?;

        let target_lsn = inner.next_lsn.saturating_sub(1);
        if target_lsn == 0 || inner.flushed_lsn >= target_lsn {
            return Ok(inner.flushed_lsn);
        }

        // Caller-side flush: do write+fsync directly instead of signaling the
        // background thread and waiting for it to wake up. This eliminates
        // ~5ms of condvar round-trip latency per commit.
        inner.flush();
        inner.check_io_error()?;
        self.flushed_cond.notify_all();

        Ok(inner.flushed_lsn)
    }

    /// Block until the given LSN has been durably flushed to disk.
    ///
    /// The calling thread performs the flush directly (caller-side group commit).
    pub(crate) fn wait_for_lsn(&self, lsn: u64) -> Result<(), TreeError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| TreeError::IoError(IoErrorKind::WalFlush))?;

        // Fail fast if a previous flush encountered an I/O error.
        inner.check_io_error()?;

        if inner.flushed_lsn >= lsn {
            return Ok(());
        }

        // Caller-side flush: write+fsync directly.
        inner.flush();
        inner.check_io_error()?;
        self.flushed_cond.notify_all();

        Ok(())
    }
}

/// Read the write-ahead-log file produced by Bf-Tree.
///
/// Allows users to iterate over the log entries in the file and decide what to do with them.
///
///
/// Example
/// ```ignore
/// let reader = WalReader::new(&file, 4096);
/// for segment in reader.segment_iter() {
///     let seg_iter = segment.iter();
///     for (header, buffer) in seg_iter {
///         ...
///     }
/// }
/// ```
pub struct WalReader {
    log_file: std::fs::File,
    segment_size: usize,
    file_size: usize,
}

impl WalReader {
    /// Create a new WalReader instance.
    ///
    /// The `segment_size` should be the same as the one used to create the WriteAheadLog instance.
    ///
    /// Returns an error if the WAL file cannot be opened or its metadata cannot be read.
    pub fn new(
        path: impl AsRef<Path>,
        segment_size: usize,
    ) -> Result<Self, crate::bf_tree::error::BfTreeError> {
        let log_file = std::fs::OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|_| crate::bf_tree::error::IoErrorKind::SnapshotRead)?;
        let file_size = log_file
            .metadata()
            .map_err(|_| crate::bf_tree::error::IoErrorKind::SnapshotRead)?
            .len() as usize;
        Ok(WalReader {
            log_file,
            segment_size,
            file_size,
        })
    }

    /// Iterate through all the segments in the wal file.
    ///
    /// Each segment contains multiple log entries,
    /// you can iterate through the log entries in each segment using the `iter` method on `WalSegment`.
    pub fn segment_iter(&self) -> WalSegmentIter<'_> {
        WalSegmentIter {
            reader: self,
            cursor: 0,
        }
    }
}

pub struct WalSegmentIter<'a> {
    reader: &'a WalReader,
    cursor: u64,
}

impl Iterator for WalSegmentIter<'_> {
    type Item = Result<WalSegment, crate::bf_tree::error::IoErrorKind>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor as usize >= self.reader.file_size {
            return None;
        }

        let mut buffer = vec![0u8; self.reader.segment_size];
        let page_offset = self.cursor;
        // The last segment may be shorter than segment_size because the WAL
        // writer only flushes the used portion of the buffer (buffer_cursor
        // bytes) to avoid writing megabytes of unused zero-fill on every fsync.
        let remaining = self.reader.file_size - page_offset as usize;
        let read_len = remaining.min(self.reader.segment_size);

        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            if self
                .reader
                .log_file
                .read_exact_at(&mut buffer[..read_len], page_offset)
                .is_err()
            {
                return Some(Err(crate::bf_tree::error::IoErrorKind::SnapshotRead));
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            match self
                .reader
                .log_file
                .seek_read(&mut buffer[..read_len], page_offset)
            {
                Ok(bytes_read) if bytes_read == read_len => {}
                _ => return Some(Err(crate::bf_tree::error::IoErrorKind::SnapshotRead)),
            }
        }

        self.cursor += self.reader.segment_size as u64;

        Some(Ok(WalSegment { data: buffer }))
    }
}

pub struct WalSegment {
    data: Vec<u8>,
}

impl WalSegment {
    /// Iterate through all the log entries in the segment.
    pub fn entry_iter(&self) -> WalEntryIter<'_> {
        WalEntryIter {
            segment: self,
            cur_offset: 0,
        }
    }
}

pub struct WalEntryIter<'a> {
    segment: &'a WalSegment,
    cur_offset: u64,
}

impl<'a> Iterator for WalEntryIter<'a> {
    type Item = (LogHeader, &'a [u8]);
    fn next(&mut self) -> Option<Self::Item> {
        if (self.cur_offset as usize + LogHeader::size()) >= self.segment.data.len() {
            return None;
        }

        let header = LogHeader::from_slice(&self.segment.data[self.cur_offset as usize..]);

        if header.log_len == 0 {
            return None;
        }

        let data_start = self.cur_offset as usize + LogHeader::size();
        let data_end = data_start + header.log_len - LogHeader::size();
        let data = &self.segment.data[data_start..data_end];
        self.cur_offset += header.log_len as u64;
        Some((header, data))
    }
}

/// The header of a log entry in the wal file.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct LogHeader {
    pub log_len: usize,
    pub lsn: u64,
    pub page_offset: u64,
}

impl LogHeader {
    fn new(lsn: u64, page_offset: u64, log_len: usize) -> Self {
        LogHeader {
            log_len,
            lsn,
            page_offset,
        }
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: LogHeader is #[repr(C)] with only primitive fields, self is a valid
        // reference, and the slice length equals size_of::<LogHeader>() (24 bytes).
        unsafe {
            std::slice::from_raw_parts(self as *const _ as *const u8, std::mem::size_of::<Self>())
        }
    }

    fn from_slice(buffer: &[u8]) -> Self {
        let log_len = usize::from_le_bytes(buffer[0..8].try_into().unwrap());
        let lsn = u64::from_le_bytes(buffer[8..16].try_into().unwrap());
        let page_offset = u64::from_le_bytes(buffer[16..24].try_into().unwrap());
        Self::new(lsn, page_offset, log_len)
    }

    const fn size() -> usize {
        std::mem::size_of::<Self>()
    }
}

const _: () = assert!(LogHeader::size() == 24);

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use std::time::Duration;

    use crate::bf_tree::utils;

    use super::*;

    struct TestLogEntry {
        val: usize,
    }

    impl TestLogEntry {
        fn new(val: usize) -> Self {
            TestLogEntry { val }
        }
    }

    impl LogEntryImpl<'_> for TestLogEntry {
        fn log_size(&self) -> usize {
            8
        }

        #[allow(clippy::disallowed_methods)] // test-only: usize serialization for test entries
        fn write_to_buffer(&self, buffer: &mut [u8]) {
            buffer.copy_from_slice(&self.val.to_le_bytes());
        }

        #[allow(clippy::disallowed_methods)] // test-only: usize deserialization for test entries
        fn read_from_buffer(buffer: &[u8]) -> Self {
            let val = usize::from_le_bytes(buffer.try_into().unwrap());
            TestLogEntry { val }
        }
    }

    fn make_test_wal(name: &str, segment_size: usize) -> Arc<WriteAheadLog> {
        let tmp_dir = std::env::temp_dir();
        let tmp_file = tmp_dir.join(name);
        let mut wal_config = WalConfig::new(&tmp_file);
        wal_config.segment_size(segment_size);
        wal_config.flush_interval(Duration::from_micros(1));
        WriteAheadLog::new(Arc::new(wal_config)).unwrap()
    }

    #[test]
    fn simple_wal() {
        const TEST_SEGMENT_SIZE: usize = 4096;
        let pid = std::process::id();
        let tid = utils::thread_id_to_u64(std::thread::current().id());
        let wal = make_test_wal(
            &format!("wal_simple_test_{}_{}.log", pid, tid),
            TEST_SEGMENT_SIZE,
        );
        let tmp_file = wal.config.file_path.clone();

        let log_entry_cnt = 4096;

        for i in 0..log_entry_cnt {
            let log = TestLogEntry::new(i);
            let lsn = wal.append_and_wait(&log, log.val as u64).unwrap();
            assert_eq!(lsn, i as u64);
        }

        wal.stop_background_job();
        drop(wal);

        let reader = WalReader::new(&tmp_file, TEST_SEGMENT_SIZE).unwrap();
        let mut cnt = 0;
        for segment in reader.segment_iter() {
            let segment = segment.unwrap();
            let seg_iter = segment.entry_iter();
            for (header, data) in seg_iter {
                let val = TestLogEntry::read_from_buffer(data);
                assert_eq!(
                    header.log_len,
                    TestLogEntry::new(0).log_size() + LogHeader::size()
                );
                assert_eq!(header.lsn, cnt as u64);
                assert_eq!(header.page_offset, cnt as u64);
                assert_eq!(val.val, cnt);
                cnt += 1;
            }
        }
        assert_eq!(cnt, log_entry_cnt);
        std::fs::remove_file(tmp_file).unwrap();
    }

    #[test]
    fn multi_thread_wal() {
        const TEST_SEGMENT_SIZE: usize = 4096;
        let pid = std::process::id();
        let tid = utils::thread_id_to_u64(std::thread::current().id());
        let wal = make_test_wal(
            &format!("wal_multi_thread_test_{}_{}.log", pid, tid),
            TEST_SEGMENT_SIZE,
        );
        let tmp_file = wal.config.file_path.clone();

        let log_entry_cnt = 4096;
        let thread_cnt = 4;

        let join_handles = (0..thread_cnt)
            .map(|_| {
                let wal_t = wal.clone();
                crate::bf_tree::sync::thread::spawn(move || {
                    for i in 0..log_entry_cnt {
                        let log = TestLogEntry::new(i);
                        let _lsn = wal_t.append_and_wait(&log, log.val as u64).unwrap();
                    }
                })
            })
            .collect::<Vec<_>>();

        for h in join_handles.into_iter() {
            h.join().unwrap();
        }

        wal.stop_background_job();
        drop(wal);

        let reader = WalReader::new(&tmp_file, TEST_SEGMENT_SIZE).unwrap();
        let mut cnt = 0;
        for segment in reader.segment_iter() {
            let segment = segment.unwrap();
            let seg_iter = segment.entry_iter();
            for (header, data) in seg_iter {
                let val = TestLogEntry::read_from_buffer(data);
                assert_eq!(
                    header.log_len,
                    TestLogEntry::new(0).log_size() + LogHeader::size()
                );
                assert_eq!(val.val, header.page_offset as usize);
                cnt += 1;
            }
        }
        assert_eq!(cnt, log_entry_cnt * thread_cnt);
        std::fs::remove_file(tmp_file).unwrap();
    }

    #[test]
    fn split_op_serialization_roundtrip() {
        use crate::bf_tree::wal::operations::{LogEntry, SplitOp};

        let split_key = b"split_separator_key";
        let original = SplitOp {
            source_page_id: 42,
            new_page_id: 99,
            split_key: split_key.as_ref(),
        };

        let entry = LogEntry::Split(original);
        let size = entry.log_size();
        let mut buffer = vec![0u8; size];
        entry.write_to_buffer(&mut buffer);

        let recovered = LogEntry::read_from_buffer(&buffer);
        match recovered {
            LogEntry::Split(op) => {
                assert_eq!(op.source_page_id, 42);
                assert_eq!(op.new_page_id, 99);
                assert_eq!(op.split_key, split_key.as_ref());
            }
            _ => panic!("Expected Split entry"),
        }
    }

    /// As of https://github.com/awslabs/shuttle/issues/74
    /// Shuttle can not properly handle wait_timeout, so we can't really test this with shuttle.
    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_wal_concurrent_op() {
        use std::{path::PathBuf, str::FromStr};

        tracing_subscriber::fmt()
            .with_ansi(true)
            .with_thread_names(false)
            .with_target(false)
            .init();
        let mut config = shuttle::Config::default();
        config.max_steps = shuttle::MaxSteps::None;
        config.failure_persistence =
            shuttle::FailurePersistence::File(Some(PathBuf::from_str("target").unwrap()));

        let mut runner = shuttle::PortfolioRunner::new(true, config);

        let available_cores = std::thread::available_parallelism().unwrap().get().min(4);

        for _i in 0..available_cores {
            runner.add(shuttle::scheduler::PctScheduler::new(10, 4_000));
        }

        runner.run(multi_thread_wal);
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_wal_replay() {
        tracing_subscriber::fmt()
            .with_ansi(true)
            .with_thread_names(false)
            .with_target(false)
            .init();

        shuttle::replay_from_file(multi_thread_wal, "target/schedule003.txt");
    }
}
