// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{
    ffi::CString,
    fs::File,
    io::Write,
    os::{
        fd::FromRawFd,
        unix::{ffi::OsStrExt, fs::FileExt},
    },
    path::PathBuf,
};

use crate::bf_tree::error::IoErrorKind;
#[allow(unused_imports)]
use crate::{check_parent, counter, histogram, info};

use super::{OffsetAlloc, VfsImpl};

pub(crate) struct StdDirectVfs {
    file: File,
    offset_alloc: OffsetAlloc,
    _path: PathBuf,
}

impl StdDirectVfs {
    pub(crate) fn open(path: impl AsRef<std::path::Path>) -> Result<Self, IoErrorKind> {
        let path = path.as_ref();

        let parent = path.parent().ok_or(IoErrorKind::VfsRead { offset: 0 })?;
        _ = std::fs::create_dir_all(parent);

        let path_cstr = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| IoErrorKind::VfsRead { offset: 0 })?;
        // SAFETY: path_cstr is a valid null-terminated C string. O_CREAT|O_RDWR opens or
        // creates the file. The mode bits grant owner read/write.
        let raw_fd = unsafe {
            libc::open(
                path_cstr.as_ptr(),
                libc::O_DIRECT | libc::O_RDWR | libc::O_CREAT,
                libc::S_IRUSR | libc::S_IWUSR,
            )
        };
        if raw_fd < 0 {
            return Err(IoErrorKind::VfsRead { offset: 0 });
        }

        // SAFETY: raw_fd is a valid file descriptor returned by libc::open above.
        let mut file = unsafe { File::from_raw_fd(raw_fd) };
        file.flush().map_err(|_| IoErrorKind::VfsFlush)?;
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

impl VfsImpl for StdDirectVfs {
    fn alloc_offset(&self, size: usize) -> usize {
        self.offset_alloc.alloc(size)
    }

    fn dealloc_offset(&self, offset: usize) {
        self.offset_alloc.dealloc_offset(offset)
    }

    fn read(&self, offset: usize, buf: &mut [u8]) -> Result<(), IoErrorKind> {
        counter!(IOReadRequest);
        self.file
            .read_at(buf, offset as u64)
            .map_err(|_| IoErrorKind::VfsRead { offset })?;
        Ok(())
    }

    fn flush(&self) -> Result<(), IoErrorKind> {
        self.file.sync_all().map_err(|_| IoErrorKind::VfsFlush)
    }

    fn write(&self, offset: usize, buf: &[u8]) -> Result<(), IoErrorKind> {
        counter!(IOWriteRequest);
        self.file
            .write_at(buf, offset as u64)
            .map_err(|_| IoErrorKind::VfsWrite { offset })?;
        Ok(())
    }
}
