// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Write-through VFS: every write is durable without an explicit flush/fsync.
//!
//! On Windows this uses `FILE_FLAG_WRITE_THROUGH` which bypasses the OS page
//! cache write-back and pushes data directly to the disk controller. On Linux
//! this uses `O_DSYNC` which guarantees each `write()` is durable when it
//! returns (data + enough metadata for retrieval).
//!
//! The `flush()` method is a cheap no-op because durability is guaranteed
//! per-write. This eliminates the ~5ms `FlushFileBuffers()`/`fdatasync()`
//! cost per WAL commit on NTFS.

use std::path::PathBuf;

use crate::counter;
use crate::error::IoErrorKind;

use super::{OffsetAlloc, VfsImpl};

pub(crate) struct WriteThroughVfs {
    file: std::fs::File,
    offset_alloc: OffsetAlloc,
    _path: PathBuf,
}

impl WriteThroughVfs {
    pub(crate) fn open(path: impl AsRef<std::path::Path>) -> Result<Self, IoErrorKind> {
        let path = path.as_ref().to_path_buf();
        let parent = path.parent().ok_or(IoErrorKind::VfsRead { offset: 0 })?;
        _ = std::fs::create_dir_all(parent);

        let file = Self::open_write_through(&path)?;

        let offset = file
            .metadata()
            .map_err(|_| IoErrorKind::VfsRead { offset: 0 })?
            .len();
        Ok(Self {
            file,
            offset_alloc: OffsetAlloc::new_with(offset as usize),
            _path: path,
        })
    }

    #[cfg(windows)]
    fn open_write_through(path: &std::path::Path) -> Result<std::fs::File, IoErrorKind> {
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_WRITE_THROUGH = 0x80000000
        // This flag causes every write to go directly to the disk controller,
        // bypassing the OS write-back cache. No FlushFileBuffers() needed.
        const FILE_FLAG_WRITE_THROUGH: u32 = 0x8000_0000;
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(FILE_FLAG_WRITE_THROUGH)
            .open(path)
            .map_err(|_| IoErrorKind::VfsRead { offset: 0 })
    }

    #[cfg(unix)]
    fn open_write_through(path: &std::path::Path) -> Result<std::fs::File, IoErrorKind> {
        use std::os::unix::fs::OpenOptionsExt;
        // O_DSYNC: each write() is durable when it returns (data + metadata
        // sufficient to locate the data). No fsync()/fdatasync() needed.
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(libc::O_DSYNC)
            .open(path)
            .map_err(|_| IoErrorKind::VfsRead { offset: 0 })
    }
}

impl VfsImpl for WriteThroughVfs {
    fn alloc_offset(&self, size: usize) -> usize {
        self.offset_alloc.alloc(size)
    }

    fn dealloc_offset(&self, offset: usize) {
        self.offset_alloc.dealloc_offset(offset)
    }

    #[cfg(unix)]
    fn read(&self, offset: usize, buf: &mut [u8]) -> Result<(), IoErrorKind> {
        use std::os::unix::fs::FileExt;
        counter!(IOReadRequest);
        self.file
            .read_at(buf, offset as u64)
            .map_err(|_| IoErrorKind::VfsRead { offset })?;
        Ok(())
    }

    #[cfg(windows)]
    fn read(&self, offset: usize, buf: &mut [u8]) -> Result<(), IoErrorKind> {
        use std::os::windows::fs::FileExt;
        counter!(IOReadRequest);
        self.file
            .seek_read(buf, offset as u64)
            .map_err(|_| IoErrorKind::VfsRead { offset })?;
        Ok(())
    }

    fn flush(&self) -> Result<(), IoErrorKind> {
        // No-op: FILE_FLAG_WRITE_THROUGH / O_DSYNC guarantees every write is
        // durable when it returns. No explicit fsync needed.
        Ok(())
    }

    #[cfg(unix)]
    fn write(&self, offset: usize, buf: &[u8]) -> Result<(), IoErrorKind> {
        use std::os::unix::fs::FileExt;
        counter!(IOWriteRequest);
        self.file
            .write_at(buf, offset as u64)
            .map_err(|_| IoErrorKind::VfsWrite { offset })?;
        Ok(())
    }

    #[cfg(windows)]
    fn write(&self, offset: usize, buf: &[u8]) -> Result<(), IoErrorKind> {
        use std::os::windows::fs::FileExt;
        counter!(IOWriteRequest);
        self.file
            .seek_write(buf, offset as u64)
            .map_err(|_| IoErrorKind::VfsWrite { offset })?;
        Ok(())
    }
}
