// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::bf_tree::sync::atomic::Ordering;
use core::cell::UnsafeCell;

use crate::bf_tree::{error::TreeError, nodes::InnerNode};

pub(crate) fn is_locked(version: u16) -> bool {
    (version & 0b10) == 0b10
}

#[derive(Debug)]
pub(crate) struct ReadGuard<'a> {
    version: u16,
    node: &'a UnsafeCell<InnerNode>,
}

impl<'a> ReadGuard<'a> {
    pub(crate) fn new(v: u16, node: &'a InnerNode) -> Self {
        Self {
            version: v,
            // SAFETY: InnerNode and UnsafeCell<InnerNode> have identical layout (#[repr(transparent)]
            // for UnsafeCell). The caller guarantees node is valid for lifetime 'a.
            node: unsafe { &*(node as *const InnerNode as *const UnsafeCell<InnerNode>) },
        }
    }
    pub(crate) fn try_read(ptr: *const InnerNode) -> Result<ReadGuard<'a>, TreeError> {
        // SAFETY: The caller guarantees ptr is a valid, aligned pointer to a live InnerNode
        // with lifetime 'a. We create only a shared reference for reading version_lock.
        let node = unsafe { &*ptr };
        let v = node.version_lock.load(Ordering::Acquire);
        if is_locked(v) {
            Err(TreeError::Locked)
        } else {
            Ok(Self::new(v, node))
        }
    }

    pub(crate) fn check_version(&self) -> Result<u16, TreeError> {
        let v = self.as_ref().version_lock.load(Ordering::Acquire);

        if v == self.version {
            Ok(v)
        } else {
            Err(TreeError::Locked)
        }
    }

    pub(crate) fn as_ref(&self) -> &InnerNode {
        // SAFETY: ReadGuard holds a shared (optimistic) lock on the node. The UnsafeCell::get()
        // returns a raw pointer to the inner InnerNode; creating a shared reference is safe
        // because the version check ensures no concurrent writer has modified the node.
        unsafe { &*self.node.get() }
    }

    pub(crate) fn upgrade(self) -> Result<WriteGuard<'a>, (Self, TreeError)> {
        let new_version = self.version + 0b10;
        match self.as_ref().version_lock.compare_exchange_weak(
            self.version,
            new_version,
            Ordering::Release,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(WriteGuard { node: self.node }),
            Err(_v) => Err((self, TreeError::Locked)),
        }
    }
}

#[derive(Debug)]
pub struct WriteGuard<'a> {
    pub(crate) node: &'a UnsafeCell<InnerNode>,
}

impl<'a> WriteGuard<'a> {
    pub(crate) fn as_ref(&self) -> &'a InnerNode {
        // SAFETY: WriteGuard has exclusive ownership of the node (version_lock is held in
        // locked state). UnsafeCell::get() yields a valid pointer; no other thread can hold
        // a read or write lock simultaneously.
        unsafe { &*self.node.get() }
    }

    pub(crate) fn as_mut(&mut self) -> &'a mut InnerNode {
        // SAFETY: WriteGuard has exclusive ownership (&mut self ensures single access).
        // The version_lock is in locked state preventing concurrent readers from validating
        // their version checks. Creating a mutable reference is sound.
        unsafe { &mut *self.node.get() }
    }

    #[allow(dead_code)]
    pub(crate) fn mark_obsolete(&mut self) {
        self.as_mut()
            .version_lock
            .fetch_add(0b01, Ordering::Release);
    }

    pub(crate) fn downgrade(self) -> ReadGuard<'a> {
        let new_v = self
            .as_ref()
            .version_lock
            .fetch_add(0b10, Ordering::Release)
            + 0b10;
        let n = self.node.get();
        // SAFETY: n was obtained from UnsafeCell::get() which returns a valid pointer.
        // The version_lock has just been updated (unlocked), so creating a shared reference
        // for the new ReadGuard is safe. mem::forget(self) prevents the Drop impl from
        // double-incrementing the version.
        let rt = ReadGuard::new(new_v, unsafe { &*n });
        core::mem::forget(self);
        rt
    }
}

impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        self.as_mut()
            .version_lock
            .fetch_add(0b10, Ordering::Release);
    }
}
