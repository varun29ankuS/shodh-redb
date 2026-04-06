// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::error::IoErrorKind;
use crate::nodes::LeafNode;

use super::VfsImpl;

pub(crate) struct MemoryVfs {}

impl MemoryVfs {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl VfsImpl for MemoryVfs {
    fn read(&self, offset: usize, buf: &mut [u8]) -> Result<(), IoErrorKind> {
        // SAFETY: offset is a valid pointer cast from a previously allocated LeafNode page,
        // and buf.len() is bounded by the allocation size enforced by alloc_offset.
        let buf_to_read = unsafe { core::slice::from_raw_parts(offset as *const u8, buf.len()) };
        buf.copy_from_slice(buf_to_read);
        Ok(())
    }

    fn write(&self, offset: usize, buf: &[u8]) -> Result<(), IoErrorKind> {
        // SAFETY: offset is a valid pointer cast from a previously allocated LeafNode page,
        // and buf.len() is bounded by the allocation size enforced by alloc_offset.
        let buf_to_write = unsafe { core::slice::from_raw_parts_mut(offset as *mut u8, buf.len()) };
        buf_to_write.copy_from_slice(buf);
        Ok(())
    }

    fn flush(&self) -> Result<(), IoErrorKind> {
        Ok(())
    }

    fn alloc_offset(&self, size: usize) -> usize {
        LeafNode::make_base_page(size) as usize
    }

    fn dealloc_offset(&self, offset: usize) {
        LeafNode::free_base_page(offset as *mut LeafNode);
    }
}
