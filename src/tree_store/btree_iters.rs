use crate::compat::Mutex;
use crate::tree_store::btree_base::{
    BRANCH, BranchAccessor, Checksum, DEFERRED, LEAF, LeafAccessor, branch_checksum, leaf_checksum,
};
use crate::tree_store::btree_iters::RangeIterState::{Internal, Leaf};
use crate::tree_store::btree_mutator::MutateHelper;
use crate::tree_store::page_store::compression::{CompressionConfig, decompress_value};
use crate::tree_store::page_store::{Page, PageImpl, TransactionalMemory};
use crate::tree_store::{BtreeHeader, PageNumber, PageTrackerPolicy};
use crate::types::{Key, Value};
use crate::{Result, StorageError};
use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;
use core::marker::PhantomData;
use core::ops::Bound::{Excluded, Included, Unbounded};
use core::ops::{Range, RangeBounds};

#[derive(Debug, Clone)]
pub enum RangeIterState {
    Leaf {
        page: PageImpl,
        fixed_key_size: Option<usize>,
        fixed_value_size: Option<usize>,
        entry: usize,
        parent: Option<Box<RangeIterState>>,
    },
    Internal {
        page: PageImpl,
        fixed_key_size: Option<usize>,
        fixed_value_size: Option<usize>,
        child: usize,
        parent: Option<Box<RangeIterState>>,
    },
}

impl RangeIterState {
    fn page_number(&self) -> PageNumber {
        match self {
            Leaf { page, .. } | Internal { page, .. } => page.get_page_number(),
        }
    }

    fn next(self, reverse: bool, manager: &TransactionalMemory) -> Result<Option<RangeIterState>> {
        match self {
            Leaf {
                page,
                fixed_key_size,
                fixed_value_size,
                entry,
                parent,
            } => {
                let accessor = LeafAccessor::new(page.memory(), fixed_key_size, fixed_value_size)?;
                let num_pairs = accessor.num_pairs();
                let next_entry = if reverse {
                    entry.checked_sub(1)
                } else {
                    let next = entry + 1;
                    if next < num_pairs { Some(next) } else { None }
                };
                if let Some(next_entry) = next_entry {
                    Ok(Some(Leaf {
                        page,
                        fixed_key_size,
                        fixed_value_size,
                        entry: next_entry,
                        parent,
                    }))
                } else {
                    Ok(parent.map(|x| *x))
                }
            }
            Internal {
                page,
                fixed_key_size,
                fixed_value_size,
                child,
                mut parent,
            } => {
                let accessor = BranchAccessor::new(&page, fixed_key_size)?;
                let child_page_num = accessor.child_page(child).ok_or_else(|| {
                    StorageError::invalid_child_pointer(page.get_page_number(), child)
                })?;
                let child_page = manager.get_page(child_page_num)?;
                // Inline read verification for the child page
                if manager.should_verify_read()
                    && let Some(expected) = accessor.child_checksum(child)
                    && expected != DEFERRED
                {
                    let computed = match child_page.memory()[0] {
                        LEAF => leaf_checksum(&child_page, fixed_key_size, fixed_value_size)?,
                        BRANCH => branch_checksum(&child_page, fixed_key_size)?,
                        _ => {
                            return Err(StorageError::page_corrupted(
                                child_page_num,
                                "Read verification: unknown page type in iterator",
                            ));
                        }
                    };
                    if computed != expected {
                        manager.on_verification_failure(child_page_num)?;
                    }
                }
                let next_child = if reverse {
                    child.checked_sub(1)
                } else {
                    let next = child + 1;
                    if next < accessor.count_children() {
                        Some(next)
                    } else {
                        None
                    }
                };
                if let Some(next_child) = next_child {
                    parent = Some(Box::new(Internal {
                        page,
                        fixed_key_size,
                        fixed_value_size,
                        child: next_child,
                        parent,
                    }));
                }
                match child_page.memory()[0] {
                    LEAF => {
                        let child_accessor = LeafAccessor::new(
                            child_page.memory(),
                            fixed_key_size,
                            fixed_value_size,
                        )?;
                        let entry = if reverse {
                            child_accessor.num_pairs().saturating_sub(1)
                        } else {
                            0
                        };
                        Ok(Some(Leaf {
                            page: child_page,
                            fixed_key_size,
                            fixed_value_size,
                            entry,
                            parent,
                        }))
                    }
                    BRANCH => {
                        let child_accessor = BranchAccessor::new(&child_page, fixed_key_size)?;
                        let child = if reverse {
                            child_accessor.count_children().saturating_sub(1)
                        } else {
                            0
                        };
                        Ok(Some(Internal {
                            page: child_page,
                            fixed_key_size,
                            fixed_value_size,
                            child,
                            parent,
                        }))
                    }
                    x => Err(StorageError::invalid_page_type(
                        child_page.get_page_number(),
                        x,
                    )),
                }
            }
        }
    }

    fn get_entry<K: Key, V: Value>(
        &self,
        compression_enabled: bool,
    ) -> Result<Option<EntryGuard<K, V>>> {
        match self {
            Leaf {
                page,
                fixed_key_size,
                fixed_value_size,
                entry,
                ..
            } => {
                if let Some((key, value)) =
                    LeafAccessor::new(page.memory(), *fixed_key_size, *fixed_value_size)?
                        .entry_ranges(*entry)
                {
                    Ok(Some(EntryGuard::new(
                        page.clone(),
                        key,
                        value,
                        compression_enabled,
                    )?))
                } else {
                    Ok(None)
                }
            }
            Internal { .. } => Ok(None),
        }
    }
}

pub(crate) struct EntryGuard<K: Key, V: Value> {
    page: PageImpl,
    key_range: Range<usize>,
    value_range: Range<usize>,
    decompressed_value: Option<Vec<u8>>,
    _key_type: PhantomData<K>,
    _value_type: PhantomData<V>,
}

impl<K: Key, V: Value> EntryGuard<K, V> {
    fn new(
        page: PageImpl,
        key_range: Range<usize>,
        value_range: Range<usize>,
        compression_enabled: bool,
    ) -> Result<Self> {
        let decompressed_value = if compression_enabled {
            let raw = &page.memory()[value_range.clone()];
            match decompress_value(raw) {
                Ok(Cow::Owned(decompressed)) => Some(decompressed),
                Ok(Cow::Borrowed(_)) => None,
                Err(_) => {
                    return Err(StorageError::Corrupted(String::from(
                        "value decompression failed: compressed data is corrupt",
                    )));
                }
            }
        } else {
            None
        };
        Ok(Self {
            page,
            key_range,
            value_range,
            decompressed_value,
            _key_type: Default::default(),
            _value_type: Default::default(),
        })
    }

    pub(crate) fn key_data(&self) -> Vec<u8> {
        self.page.memory()[self.key_range.clone()].to_vec()
    }

    pub(crate) fn value_data(&self) -> &[u8] {
        if let Some(ref decompressed) = self.decompressed_value {
            decompressed
        } else {
            &self.page.memory()[self.value_range.clone()]
        }
    }

    pub(crate) fn key(&self) -> K::SelfType<'_> {
        K::from_bytes(&self.page.memory()[self.key_range.clone()])
    }

    pub(crate) fn value(&self) -> V::SelfType<'_> {
        if let Some(ref decompressed) = self.decompressed_value {
            V::from_bytes(decompressed)
        } else {
            V::from_bytes(&self.page.memory()[self.value_range.clone()])
        }
    }

    /// Access the stored value, catching panics from corrupt data.
    ///
    /// On `std` builds, wraps `from_bytes` in `catch_unwind`. On `no_std`,
    /// calls `from_bytes` directly (panics on corrupt data).
    #[cfg(feature = "std")]
    pub(crate) fn value_checked(
        &self,
    ) -> core::result::Result<V::SelfType<'_>, crate::StorageError> {
        let bytes: &[u8] = if let Some(ref decompressed) = self.decompressed_value {
            decompressed
        } else {
            &self.page.memory()[self.value_range.clone()]
        };
        let ptr = bytes.as_ptr();
        let len = bytes.len();
        // SAFETY: catch_unwind requires UnwindSafe. The byte slice has no
        // interior mutability, so AssertUnwindSafe is sound.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let slice = unsafe { core::slice::from_raw_parts(ptr, len) };
            V::from_bytes(slice)
        }))
        .map_err(|_| {
            crate::StorageError::Corrupted(alloc::string::String::from(
                "panic in Value::from_bytes (corrupted metadata)",
            ))
        })
    }

    /// `no_std` fallback: calls `from_bytes` directly (no panic protection).
    #[cfg(not(feature = "std"))]
    pub(crate) fn value_checked(
        &self,
    ) -> core::result::Result<V::SelfType<'_>, crate::StorageError> {
        Ok(self.value())
    }

    pub(crate) fn into_raw(self) -> (PageImpl, Range<usize>, Range<usize>, Option<Vec<u8>>) {
        (
            self.page,
            self.key_range,
            self.value_range,
            self.decompressed_value,
        )
    }
}

pub(crate) struct AllPageNumbersBtreeIter {
    next: Option<RangeIterState>,
    manager: Arc<TransactionalMemory>,
}

impl AllPageNumbersBtreeIter {
    pub(crate) fn new(
        root: PageNumber,
        fixed_key_size: Option<usize>,
        fixed_value_size: Option<usize>,
        manager: Arc<TransactionalMemory>,
    ) -> Result<Self> {
        let root_page = manager.get_page(root)?;
        let page_type = root_page.memory()[0];
        let start = match page_type {
            LEAF => Leaf {
                page: root_page,
                fixed_key_size,
                fixed_value_size,
                entry: 0,
                parent: None,
            },
            BRANCH => Internal {
                page: root_page,
                fixed_key_size,
                fixed_value_size,
                child: 0,
                parent: None,
            },
            x => {
                return Err(StorageError::invalid_page_type(root, x));
            }
        };
        Ok(Self {
            next: Some(start),
            manager,
        })
    }
}

impl Iterator for AllPageNumbersBtreeIter {
    type Item = Result<PageNumber>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let state = self.next.take()?;
            let value = state.page_number();
            // Only return each page number once
            let once = match state {
                Leaf { entry, .. } => entry == 0,
                Internal { child, .. } => child == 0,
            };
            match state.next(false, &self.manager) {
                Ok(next) => {
                    self.next = next;
                }
                Err(err) => {
                    return Some(Err(err));
                }
            }
            if once {
                return Some(Ok(value));
            }
        }
    }
}

pub(crate) struct BtreeExtractIf<
    'a,
    K: Key + 'static,
    V: Value + 'static,
    F: for<'f> FnMut(K::SelfType<'f>, V::SelfType<'f>) -> bool,
> {
    root: &'a mut Option<BtreeHeader>,
    inner: BtreeRangeIter<K, V>,
    predicate: F,
    free_on_drop: Vec<PageNumber>,
    master_free_list: Arc<Mutex<Vec<PageNumber>>>,
    allocated: Arc<Mutex<PageTrackerPolicy>>,
    mem: Arc<TransactionalMemory>,
    compression: CompressionConfig,
}

impl<'a, K: Key, V: Value, F: for<'f> FnMut(K::SelfType<'f>, V::SelfType<'f>) -> bool>
    BtreeExtractIf<'a, K, V, F>
{
    pub(crate) fn new(
        root: &'a mut Option<BtreeHeader>,
        inner: BtreeRangeIter<K, V>,
        predicate: F,
        master_free_list: Arc<Mutex<Vec<PageNumber>>>,
        allocated: Arc<Mutex<PageTrackerPolicy>>,
        mem: Arc<TransactionalMemory>,
        compression: CompressionConfig,
    ) -> Self {
        Self {
            root,
            inner,
            predicate,
            free_on_drop: vec![],
            master_free_list,
            allocated,
            mem,
            compression,
        }
    }
}

impl<K: Key, V: Value, F: for<'f> FnMut(K::SelfType<'f>, V::SelfType<'f>) -> bool> Iterator
    for BtreeExtractIf<'_, K, V, F>
{
    type Item = Result<EntryGuard<K, V>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut item = self.inner.next();
        while let Some(Ok(ref entry)) = item {
            if (self.predicate)(entry.key(), entry.value()) {
                let deleted_key = entry.key_data();
                let mut operation: MutateHelper<'_, '_, K, V> = MutateHelper::new_do_not_modify(
                    self.root,
                    self.mem.clone(),
                    &mut self.free_on_drop,
                    self.allocated.clone(),
                    self.compression,
                );
                match operation.delete(&K::from_bytes(&deleted_key)) {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        return Some(Err(crate::StorageError::Corrupted(
                            "extract_if: key existed during iteration but delete returned None"
                                .into(),
                        )));
                    }
                    Err(x) => {
                        return Some(Err(x));
                    }
                }
                break;
            }
            item = self.inner.next();
        }
        item
    }
}

impl<K: Key, V: Value, F: for<'f> FnMut(K::SelfType<'f>, V::SelfType<'f>) -> bool>
    DoubleEndedIterator for BtreeExtractIf<'_, K, V, F>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let mut item = self.inner.next_back();
        while let Some(Ok(ref entry)) = item {
            if (self.predicate)(entry.key(), entry.value()) {
                let deleted_key = entry.key_data();
                let mut operation: MutateHelper<'_, '_, K, V> = MutateHelper::new_do_not_modify(
                    self.root,
                    self.mem.clone(),
                    &mut self.free_on_drop,
                    self.allocated.clone(),
                    self.compression,
                );
                match operation.delete(&K::from_bytes(&deleted_key)) {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        return Some(Err(crate::StorageError::Corrupted(
                            "extract_if: key existed during iteration but delete returned None"
                                .into(),
                        )));
                    }
                    Err(x) => {
                        return Some(Err(x));
                    }
                }
                break;
            }
            item = self.inner.next_back();
        }
        item
    }
}

impl<K: Key, V: Value, F: for<'f> FnMut(K::SelfType<'f>, V::SelfType<'f>) -> bool> Drop
    for BtreeExtractIf<'_, K, V, F>
{
    fn drop(&mut self) {
        self.inner.close();
        let mut master_free_list = self.master_free_list.lock();
        let mut allocated = self.allocated.lock();
        for page in self.free_on_drop.drain(..) {
            match self.mem.free_if_uncommitted(page, &mut allocated) {
                Ok(true) => {}
                Ok(false) | Err(_) => master_free_list.push(page),
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct BtreeRangeIter<K: Key + 'static, V: Value + 'static> {
    left: Option<RangeIterState>, // Exclusive. The previous element returned
    right: Option<RangeIterState>, // Exclusive. The previous element returned
    include_left: bool,           // left is inclusive, instead of exclusive
    include_right: bool,          // right is inclusive, instead of exclusive
    compression_enabled: bool,
    manager: Arc<TransactionalMemory>,
    _key_type: PhantomData<K>,
    _value_type: PhantomData<V>,
}

fn range_is_empty<'a, K: Key + 'static, KR: Borrow<K::SelfType<'a>>, T: RangeBounds<KR>>(
    range: &T,
) -> bool {
    match (range.start_bound(), range.end_bound()) {
        (Unbounded, _) | (_, Unbounded) => false,
        (Included(start), Excluded(end)) | (Excluded(start), Included(end) | Excluded(end)) => {
            let start_tmp = K::as_bytes(start.borrow());
            let start_value = start_tmp.as_ref();
            let end_tmp = K::as_bytes(end.borrow());
            let end_value = end_tmp.as_ref();
            K::compare(start_value, end_value).is_ge()
        }
        (Included(start), Included(end)) => {
            let start_tmp = K::as_bytes(start.borrow());
            let start_value = start_tmp.as_ref();
            let end_tmp = K::as_bytes(end.borrow());
            let end_value = end_tmp.as_ref();
            K::compare(start_value, end_value).is_gt()
        }
    }
}

impl<K: Key + 'static, V: Value + 'static> BtreeRangeIter<K, V> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<'a, T: RangeBounds<KR>, KR: Borrow<K::SelfType<'a>>>(
        query_range: &'_ T,
        table_root: Option<PageNumber>,
        root_checksum: Option<Checksum>,
        fixed_value_size: Option<usize>,
        manager: Arc<TransactionalMemory>,
        compression_enabled: bool,
    ) -> Result<Self> {
        if range_is_empty::<K, KR, T>(query_range) {
            return Ok(Self {
                left: None,
                right: None,
                include_left: false,
                include_right: false,
                compression_enabled,
                manager,
                _key_type: Default::default(),
                _value_type: Default::default(),
            });
        }
        if let Some(root) = table_root {
            let (include_left, left) = match query_range.start_bound() {
                Included(k) => find_iter_left::<K, V>(
                    manager.get_page(root)?,
                    None,
                    K::as_bytes(k.borrow()).as_ref(),
                    true,
                    fixed_value_size,
                    &manager,
                    root_checksum,
                )?,
                Excluded(k) => find_iter_left::<K, V>(
                    manager.get_page(root)?,
                    None,
                    K::as_bytes(k.borrow()).as_ref(),
                    false,
                    fixed_value_size,
                    &manager,
                    root_checksum,
                )?,
                Unbounded => {
                    let state = find_iter_unbounded::<K, V>(
                        manager.get_page(root)?,
                        None,
                        false,
                        fixed_value_size,
                        &manager,
                        root_checksum,
                    )?;
                    (true, state)
                }
            };
            let (include_right, right) = match query_range.end_bound() {
                Included(k) => find_iter_right::<K, V>(
                    manager.get_page(root)?,
                    None,
                    K::as_bytes(k.borrow()).as_ref(),
                    true,
                    fixed_value_size,
                    &manager,
                    root_checksum,
                )?,
                Excluded(k) => find_iter_right::<K, V>(
                    manager.get_page(root)?,
                    None,
                    K::as_bytes(k.borrow()).as_ref(),
                    false,
                    fixed_value_size,
                    &manager,
                    root_checksum,
                )?,
                Unbounded => {
                    let state = find_iter_unbounded::<K, V>(
                        manager.get_page(root)?,
                        None,
                        true,
                        fixed_value_size,
                        &manager,
                        root_checksum,
                    )?;
                    (true, state)
                }
            };
            Ok(Self {
                left,
                right,
                include_left,
                include_right,
                compression_enabled,
                manager,
                _key_type: Default::default(),
                _value_type: Default::default(),
            })
        } else {
            Ok(Self {
                left: None,
                right: None,
                include_left: false,
                include_right: false,
                compression_enabled,
                manager,
                _key_type: Default::default(),
                _value_type: Default::default(),
            })
        }
    }

    fn close(&mut self) {
        self.left = None;
        self.right = None;
    }
}

impl<K: Key, V: Value> Iterator for BtreeRangeIter<K, V> {
    type Item = Result<EntryGuard<K, V>>;

    fn next(&mut self) -> Option<Self::Item> {
        if let (
            Some(Leaf {
                page: left_page,
                entry: left_entry,
                ..
            }),
            Some(Leaf {
                page: right_page,
                entry: right_entry,
                ..
            }),
        ) = (&self.left, &self.right)
            && left_page.get_page_number() == right_page.get_page_number()
            && (left_entry > right_entry
                || (left_entry == right_entry && (!self.include_left || !self.include_right)))
        {
            return None;
        }

        loop {
            if !self.include_left {
                match self.left.take()?.next(false, &self.manager) {
                    Ok(left) => {
                        self.left = left;
                    }
                    Err(err) => {
                        return Some(Err(err));
                    }
                }
            }
            // Return None if the next state is None
            self.left.as_ref()?;

            if let (
                Some(Leaf {
                    page: left_page,
                    entry: left_entry,
                    ..
                }),
                Some(Leaf {
                    page: right_page,
                    entry: right_entry,
                    ..
                }),
            ) = (&self.left, &self.right)
                && left_page.get_page_number() == right_page.get_page_number()
                && (left_entry > right_entry || (left_entry == right_entry && !self.include_right))
            {
                return None;
            }

            self.include_left = false;
            let ce = self.compression_enabled;
            if let Some(state) = self.left.as_ref() {
                match state.get_entry::<K, V>(ce) {
                    Ok(Some(entry)) => return Some(Ok(entry)),
                    Ok(None) => {}
                    Err(err) => return Some(Err(err)),
                }
            }
        }
    }
}

impl<K: Key, V: Value> DoubleEndedIterator for BtreeRangeIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let (
            Some(Leaf {
                page: left_page,
                entry: left_entry,
                ..
            }),
            Some(Leaf {
                page: right_page,
                entry: right_entry,
                ..
            }),
        ) = (&self.left, &self.right)
            && left_page.get_page_number() == right_page.get_page_number()
            && (left_entry > right_entry
                || (left_entry == right_entry && (!self.include_left || !self.include_right)))
        {
            return None;
        }

        loop {
            if !self.include_right {
                match self.right.take()?.next(true, &self.manager) {
                    Ok(right) => {
                        self.right = right;
                    }
                    Err(err) => {
                        return Some(Err(err));
                    }
                }
            }
            // Return None if the next state is None
            self.right.as_ref()?;

            if let (
                Some(Leaf {
                    page: left_page,
                    entry: left_entry,
                    ..
                }),
                Some(Leaf {
                    page: right_page,
                    entry: right_entry,
                    ..
                }),
            ) = (&self.left, &self.right)
                && left_page.get_page_number() == right_page.get_page_number()
                && (left_entry > right_entry || (left_entry == right_entry && !self.include_left))
            {
                return None;
            }

            self.include_right = false;
            let ce = self.compression_enabled;
            if let Some(state) = self.right.as_ref() {
                match state.get_entry::<K, V>(ce) {
                    Ok(Some(entry)) => return Some(Ok(entry)),
                    Ok(None) => {}
                    Err(err) => return Some(Err(err)),
                }
            }
        }
    }
}

/// Verify a page's checksum during iterator initialization, if verification is enabled.
fn maybe_verify_iter_page(
    page: &PageImpl,
    expected: Option<Checksum>,
    fixed_key_size: Option<usize>,
    fixed_value_size: Option<usize>,
    manager: &TransactionalMemory,
) -> Result {
    if let Some(expected) = expected
        && expected != DEFERRED
        && manager.should_verify_read()
    {
        let computed = match page.memory()[0] {
            LEAF => leaf_checksum(page, fixed_key_size, fixed_value_size)?,
            BRANCH => branch_checksum(page, fixed_key_size)?,
            _ => {
                return Err(StorageError::page_corrupted(
                    page.get_page_number(),
                    "Read verification: unknown page type in iterator init",
                ));
            }
        };
        if computed != expected {
            manager.on_verification_failure(page.get_page_number())?;
        }
    }
    Ok(())
}

fn find_iter_unbounded<K: Key, V: Value>(
    page: PageImpl,
    mut parent: Option<Box<RangeIterState>>,
    reverse: bool,
    fixed_value_size: Option<usize>,
    manager: &TransactionalMemory,
    expected_checksum: Option<Checksum>,
) -> Result<Option<RangeIterState>> {
    maybe_verify_iter_page(
        &page,
        expected_checksum,
        K::fixed_width(),
        fixed_value_size,
        manager,
    )?;
    let page_type = page.memory()[0];
    match page_type {
        LEAF => {
            let accessor = LeafAccessor::new(page.memory(), K::fixed_width(), fixed_value_size)?;
            let entry = if reverse {
                accessor.num_pairs().saturating_sub(1)
            } else {
                0
            };
            Ok(Some(Leaf {
                page,
                fixed_key_size: K::fixed_width(),
                fixed_value_size,
                entry,
                parent,
            }))
        }
        BRANCH => {
            let accessor = BranchAccessor::new(&page, K::fixed_width())?;
            let child_index = if reverse {
                accessor.count_children().saturating_sub(1)
            } else {
                0
            };
            let child_page_number = accessor.child_page(child_index).ok_or_else(|| {
                StorageError::invalid_child_pointer(page.get_page_number(), child_index)
            })?;
            let child_checksum = accessor.child_checksum(child_index);
            let child_page = manager.get_page(child_page_number)?;
            let next_child = if reverse {
                child_index.checked_sub(1)
            } else {
                let next = child_index + 1;
                if next < accessor.count_children() {
                    Some(next)
                } else {
                    None
                }
            };
            if let Some(sibling) = next_child {
                parent = Some(Box::new(Internal {
                    page,
                    fixed_key_size: K::fixed_width(),
                    fixed_value_size,
                    child: sibling,
                    parent,
                }));
            }
            find_iter_unbounded::<K, V>(
                child_page,
                parent,
                reverse,
                fixed_value_size,
                manager,
                child_checksum,
            )
        }
        x => Err(StorageError::invalid_page_type(page.get_page_number(), x)),
    }
}

// Returns a bool indicating whether the first entry pointed to by the state is included in the
// queried range
fn find_iter_left<K: Key, V: Value>(
    page: PageImpl,
    mut parent: Option<Box<RangeIterState>>,
    query: &[u8],
    include_query: bool,
    fixed_value_size: Option<usize>,
    manager: &TransactionalMemory,
    expected_checksum: Option<Checksum>,
) -> Result<(bool, Option<RangeIterState>)> {
    maybe_verify_iter_page(
        &page,
        expected_checksum,
        K::fixed_width(),
        fixed_value_size,
        manager,
    )?;
    let page_type = page.memory()[0];
    match page_type {
        LEAF => {
            let accessor = LeafAccessor::new(page.memory(), K::fixed_width(), fixed_value_size)?;
            if accessor.num_pairs() == 0 {
                return Ok((false, None));
            }
            let (mut position, found) = accessor.position::<K>(query);
            let include = if position < accessor.num_pairs() {
                include_query || !found
            } else {
                // Back up to the last valid position (safe: num_pairs > 0 checked above)
                position -= 1;
                // and exclude it
                false
            };
            let result = Leaf {
                page,
                fixed_key_size: K::fixed_width(),
                fixed_value_size,
                entry: position,
                parent,
            };
            Ok((include, Some(result)))
        }
        BRANCH => {
            let accessor = BranchAccessor::new(&page, K::fixed_width())?;
            let (child_index, child_page_number) = accessor.child_for_key::<K>(query)?;
            let child_checksum = accessor.child_checksum(child_index);
            let child_page = manager.get_page(child_page_number)?;
            if child_index + 1 < accessor.count_children() {
                parent = Some(Box::new(Internal {
                    page,
                    fixed_key_size: K::fixed_width(),
                    fixed_value_size,
                    child: child_index + 1,
                    parent,
                }));
            }
            find_iter_left::<K, V>(
                child_page,
                parent,
                query,
                include_query,
                fixed_value_size,
                manager,
                child_checksum,
            )
        }
        x => Err(StorageError::invalid_page_type(page.get_page_number(), x)),
    }
}

fn find_iter_right<K: Key, V: Value>(
    page: PageImpl,
    mut parent: Option<Box<RangeIterState>>,
    query: &[u8],
    include_query: bool,
    fixed_value_size: Option<usize>,
    manager: &TransactionalMemory,
    expected_checksum: Option<Checksum>,
) -> Result<(bool, Option<RangeIterState>)> {
    maybe_verify_iter_page(
        &page,
        expected_checksum,
        K::fixed_width(),
        fixed_value_size,
        manager,
    )?;
    let page_type = page.memory()[0];
    match page_type {
        LEAF => {
            let accessor = LeafAccessor::new(page.memory(), K::fixed_width(), fixed_value_size)?;
            if accessor.num_pairs() == 0 {
                return Ok((false, None));
            }
            let (mut position, found) = accessor.position::<K>(query);
            let include = if position < accessor.num_pairs() {
                include_query && found
            } else {
                // Back up to the last valid position (safe: num_pairs > 0 checked above)
                position -= 1;
                // and include it
                true
            };
            let result = Leaf {
                page,
                fixed_key_size: K::fixed_width(),
                fixed_value_size,
                entry: position,
                parent,
            };
            Ok((include, Some(result)))
        }
        BRANCH => {
            let accessor = BranchAccessor::new(&page, K::fixed_width())?;
            let (child_index, child_page_number) = accessor.child_for_key::<K>(query)?;
            let child_checksum = accessor.child_checksum(child_index);
            let child_page = manager.get_page(child_page_number)?;
            if child_index > 0 && accessor.child_page(child_index - 1).is_some() {
                parent = Some(Box::new(Internal {
                    page,
                    fixed_key_size: K::fixed_width(),
                    fixed_value_size,
                    child: child_index - 1,
                    parent,
                }));
            }
            find_iter_right::<K, V>(
                child_page,
                parent,
                query,
                include_query,
                fixed_value_size,
                manager,
                child_checksum,
            )
        }
        x => Err(StorageError::invalid_page_type(page.get_page_number(), x)),
    }
}

// --- Raw (untyped) iteration ---
// Walks all leaf entries yielding raw key/value byte slices without type information.
// Used by ReadOnlyUntypedTable for CLI dump and debugging tools.

fn find_iter_unbounded_raw(
    page: PageImpl,
    mut parent: Option<Box<RangeIterState>>,
    fixed_key_size: Option<usize>,
    fixed_value_size: Option<usize>,
    manager: &TransactionalMemory,
    expected_checksum: Option<Checksum>,
) -> Result<Option<RangeIterState>> {
    maybe_verify_iter_page(
        &page,
        expected_checksum,
        fixed_key_size,
        fixed_value_size,
        manager,
    )?;
    let page_type = page.memory()[0];
    match page_type {
        LEAF => Ok(Some(Leaf {
            page,
            fixed_key_size,
            fixed_value_size,
            entry: 0,
            parent,
        })),
        BRANCH => {
            let accessor = BranchAccessor::new(&page, fixed_key_size)?;
            let child_page_number = accessor
                .child_page(0)
                .ok_or_else(|| StorageError::invalid_child_pointer(page.get_page_number(), 0))?;
            let child_checksum = accessor.child_checksum(0);
            let child_page = manager.get_page(child_page_number)?;
            if 1 < accessor.count_children() {
                parent = Some(Box::new(Internal {
                    page,
                    fixed_key_size,
                    fixed_value_size,
                    child: 1,
                    parent,
                }));
            }
            find_iter_unbounded_raw(
                child_page,
                parent,
                fixed_key_size,
                fixed_value_size,
                manager,
                child_checksum,
            )
        }
        x => Err(StorageError::invalid_page_type(page.get_page_number(), x)),
    }
}

/// Entry from raw (untyped) table iteration, holding references to key and value bytes.
pub struct RawEntryGuard {
    page: PageImpl,
    key_range: Range<usize>,
    value_range: Range<usize>,
}

impl RawEntryGuard {
    /// Raw key bytes as stored in the database.
    pub fn key(&self) -> &[u8] {
        &self.page.memory()[self.key_range.clone()]
    }

    /// Raw value bytes as stored in the database.
    pub fn value(&self) -> &[u8] {
        &self.page.memory()[self.value_range.clone()]
    }
}

/// Iterator over all entries in a table as raw key/value byte slices.
/// Does not require knowing the key or value types at compile time.
pub struct RawEntryIter {
    state: Option<RangeIterState>,
    include_current: bool,
    manager: Arc<TransactionalMemory>,
}

impl RawEntryIter {
    pub(crate) fn new(
        root: Option<PageNumber>,
        fixed_key_size: Option<usize>,
        fixed_value_size: Option<usize>,
        manager: Arc<TransactionalMemory>,
    ) -> Result<Self> {
        let state = if let Some(root_page) = root {
            let page = manager.get_page(root_page)?;
            find_iter_unbounded_raw(page, None, fixed_key_size, fixed_value_size, &manager, None)?
        } else {
            None
        };
        Ok(Self {
            state,
            include_current: true,
            manager,
        })
    }
}

impl Iterator for RawEntryIter {
    type Item = Result<RawEntryGuard>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.include_current {
                match self.state.take()?.next(false, &self.manager) {
                    Ok(s) => self.state = s,
                    Err(e) => return Some(Err(e)),
                }
            }
            self.include_current = false;

            let state = self.state.as_ref()?;
            match state {
                Leaf {
                    page,
                    fixed_key_size,
                    fixed_value_size,
                    entry,
                    ..
                } => {
                    let accessor = match LeafAccessor::new(
                        page.memory(),
                        *fixed_key_size,
                        *fixed_value_size,
                    ) {
                        Ok(a) => a,
                        Err(e) => return Some(Err(e)),
                    };
                    if let Some((key_range, value_range)) = accessor.entry_ranges(*entry) {
                        return Some(Ok(RawEntryGuard {
                            page: page.clone(),
                            key_range,
                            value_range,
                        }));
                    }
                }
                Internal { .. } => {}
            }
        }
    }
}
