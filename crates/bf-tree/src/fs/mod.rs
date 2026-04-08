// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

mod memory_vfs;

#[cfg(feature = "std")]
mod std_vfs;

#[cfg(feature = "std")]
mod write_through_vfs;

#[cfg(all(feature = "std", target_os = "linux"))]
mod std_direct_vfs;

use core::sync::atomic::Ordering;

#[cfg(all(feature = "std", target_os = "linux"))]
pub(crate) use std_direct_vfs::StdDirectVfs;

#[cfg(all(feature = "std", target_os = "linux"))]
mod io_uring_vfs;
#[cfg(all(feature = "std", target_os = "linux"))]
pub(crate) use io_uring_vfs::IoUringVfs;

pub(crate) use memory_vfs::MemoryVfs;
#[cfg(feature = "std")]
pub(crate) use std_vfs::StdVfs;
#[cfg(feature = "std")]
pub(crate) use write_through_vfs::WriteThroughVfs;

use crate::error::IoErrorKind;
use crate::nodes::DISK_PAGE_SIZE;

/// Similar to `std::io::Write` and `std::io::Read`, but without &mut self, i.e., no locking.
/// All I/O methods return `Result` so callers can propagate errors instead of panicking.
pub(crate) trait VfsImpl: Send + Sync {
    fn read(&self, offset: usize, buf: &mut [u8]) -> Result<(), IoErrorKind>;

    fn write(&self, offset: usize, buf: &[u8]) -> Result<(), IoErrorKind>;

    /// Allocate a new page returns the physical offset of the page.
    /// The size of the page is a multiple of DISK_PAGE_SIZE
    fn alloc_offset(&self, size: usize) -> usize;

    /// When we no longer need a page, we let fs know so it can be reused.
    fn dealloc_offset(&self, offset: usize);

    /// Flush the data to disk, similar to fsync on Linux.
    fn flush(&self) -> Result<(), IoErrorKind>;
}

pub(crate) fn buffer_alloc(layout: alloc::alloc::Layout) -> *mut u8 {
    // SAFETY: layout is non-zero-sized and properly aligned by the caller.
    unsafe { alloc::alloc::alloc(layout) }
}

pub(crate) fn buffer_dealloc(ptr: *mut u8, layout: alloc::alloc::Layout) {
    // SAFETY: ptr was allocated with the same layout via buffer_alloc.
    unsafe { alloc::alloc::dealloc(ptr, layout) }
}

/// A simple page allocator for disk.
pub(crate) struct OffsetAlloc {
    next_available_offset: crate::sync::atomic::AtomicUsize,
}

impl OffsetAlloc {
    pub(crate) fn new_with(mut offset: usize) -> Self {
        if offset < DISK_PAGE_SIZE {
            // the file was empty, we start from second page
            offset = DISK_PAGE_SIZE;
        }
        Self {
            next_available_offset: crate::sync::atomic::AtomicUsize::new(offset),
        }
    }

    pub(crate) fn alloc(&self, size: usize) -> usize {
        self.next_available_offset.fetch_add(size, Ordering::AcqRel)
    }

    pub(crate) fn dealloc_offset(&self, _offset: usize) {
        // We don't need to do anything here.
    }
}
