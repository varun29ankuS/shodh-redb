// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{
    fs::{File, OpenOptions},
    path::PathBuf,
};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

use crate::bf_tree::error::IoErrorKind;
#[allow(unused_imports)]
use crate::{check_parent, counter, histogram, info};

use super::{OffsetAlloc, VfsImpl};

pub(crate) struct StdVfs {
    file: File,
    offset_alloc: OffsetAlloc,
    _path: PathBuf,
}

impl StdVfs {
    pub(crate) fn open(path: impl AsRef<std::path::Path>) -> Result<Self, IoErrorKind> {
        let path = path.as_ref().to_path_buf();
        let parent = path.parent().ok_or(IoErrorKind::VfsRead { offset: 0 })?;
        _ = std::fs::create_dir_all(parent);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|_| IoErrorKind::VfsRead { offset: 0 })?;
        let offset = file
            .metadata()
            .map_err(|_| IoErrorKind::VfsRead { offset: 0 })?
            .len();
        Ok(Self {
            file,
            offset_alloc: OffsetAlloc::new_with(offset as usize),
            _path: path.to_path_buf(),
        })
    }
}

impl VfsImpl for StdVfs {
    fn alloc_offset(&self, size: usize) -> usize {
        self.offset_alloc.alloc(size)
    }

    fn dealloc_offset(&self, offset: usize) {
        self.offset_alloc.dealloc_offset(offset)
    }

    #[cfg(unix)]
    fn read(&self, offset: usize, buf: &mut [u8]) -> Result<(), IoErrorKind> {
        counter!(IOReadRequest);
        self.file
            .read_at(buf, offset as u64)
            .map_err(|_| IoErrorKind::VfsRead { offset })?;
        Ok(())
    }

    #[cfg(windows)]
    fn read(&self, offset: usize, buf: &mut [u8]) -> Result<(), IoErrorKind> {
        counter!(IOReadRequest);
        self.file
            .seek_read(buf, offset as u64)
            .map_err(|_| IoErrorKind::VfsRead { offset })?;
        Ok(())
    }

    fn flush(&self) -> Result<(), IoErrorKind> {
        // sync_data() skips flushing file metadata (size, timestamps) which is
        // significantly faster on Windows NTFS. For WAL and snapshot durability,
        // only the data content needs to be persisted -- metadata correctness
        // is not required for crash recovery.
        self.file.sync_data().map_err(|_| IoErrorKind::VfsFlush)
    }

    #[cfg(unix)]
    fn write(&self, offset: usize, buf: &[u8]) -> Result<(), IoErrorKind> {
        counter!(IOWriteRequest);
        self.file
            .write_at(buf, offset as u64)
            .map_err(|_| IoErrorKind::VfsWrite { offset })?;
        Ok(())
    }

    #[cfg(windows)]
    fn write(&self, offset: usize, buf: &[u8]) -> Result<(), IoErrorKind> {
        counter!(IOWriteRequest);
        self.file
            .seek_write(buf, offset as u64)
            .map_err(|_| IoErrorKind::VfsWrite { offset })?;
        Ok(())
    }
}
