//! Backend-agnostic storage traits for shodh-redb.
//!
//! These traits abstract the storage engine so that higher-level modules
//! (`IvfPq`, CDC) can work with any backend -- legacy B-tree
//! or Bf-Tree -- without code duplication.
//!
//! # Trait Hierarchy
//!
//! ```text
//! StorageWrite              StorageRead
//!     |                         |
//!     v                         v
//! open_table() -->         open_table() -->
//!     WriteTable<K,V>        ReadTable<K,V>
//!       | get()                | get()
//!       | insert()             | range()
//!       | remove()             |
//!       | range()              v
//!       | drain_all()      OwnedKv (key, value bytes)
//!       v
//!   OwnedKv (key, value bytes)
//! ```
//!
//! # Design Notes
//!
//! - All returned values are **owned** (`Vec<u8>`) -- no zero-copy page references.
//!   This is the common denominator between B-tree (page-backed) and Bf-Tree
//!   (copy-to-buffer). The `OwnedKv` wrapper provides `.value()` extraction
//!   matching the `AccessGuard` ergonomics.
//!
//! - Error type is `StorageError` throughout, matching existing index code that
//!   converts `TableError -> StorageError` via helper functions.
//!
//! - Range iteration yields `Result<(OwnedKv<K>, OwnedKv<V>)>` entries.

use alloc::string::String;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::RangeBounds;

use crate::TableDefinition;
use crate::types::{Key, Value};

// ---------------------------------------------------------------------------
// OwnedKv -- owned-bytes wrapper with .value() extraction
// ---------------------------------------------------------------------------

/// Owned key-value guard wrapping raw bytes.
///
/// Provides the `.value()` method that `AccessGuard` exposes, but backed by
/// an owned `Vec<u8>` instead of a page reference. This enables the same
/// `guard.value()` pattern used throughout the `IvfPq` index code.
pub struct OwnedKv<T: Value + 'static> {
    data: Vec<u8>,
    _type: PhantomData<T>,
}

impl<T: Value + 'static> OwnedKv<T> {
    /// Create a new owned guard from raw bytes.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            _type: PhantomData,
        }
    }

    /// Deserialize the stored bytes into the value's `SelfType`.
    ///
    /// This mirrors `AccessGuard::value()`.
    pub fn value(&self) -> T::SelfType<'_> {
        T::from_bytes(&self.data)
    }

    /// Get the raw bytes.
    pub fn raw_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume the guard and return the raw bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}

impl<T: Value + 'static> core::fmt::Debug for OwnedKv<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OwnedKv")
            .field("len", &self.data.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// WriteTable -- write-capable table operations
// ---------------------------------------------------------------------------

/// A writable table handle that can get, insert, remove, range-scan, and drain.
///
/// Both `Table<'txn, K, V>` (legacy) and `BfTreeTable<'txn, K, V>` implement
/// this trait, allowing `IvfPq` code to be generic over the backend.
pub trait WriteTable<K: Key + 'static, V: Value + 'static> {
    /// Range iterator type returned by `range()`.
    type RangeIter<'a>: Iterator<Item = crate::Result<(OwnedKv<K>, OwnedKv<V>)>>
    where
        Self: 'a;

    /// Read the value for a key.
    ///
    /// Returns `Ok(Some(guard))` if found, `Ok(None)` if not found.
    fn st_get(&self, key: &K::SelfType<'_>) -> crate::Result<Option<OwnedKv<V>>>;

    /// Insert a key-value pair. Returns the previous value if the key existed.
    fn st_insert(
        &mut self,
        key: &K::SelfType<'_>,
        value: &V::SelfType<'_>,
    ) -> crate::Result<Option<OwnedKv<V>>>;

    /// Remove a key. Returns the removed value if the key existed.
    fn st_remove(&mut self, key: &K::SelfType<'_>) -> crate::Result<Option<OwnedKv<V>>>;

    /// Range scan. Returns an iterator yielding `(key_guard, value_guard)` pairs.
    fn st_range<'a>(
        &'a self,
        start: Option<&K::SelfType<'_>>,
        end: Option<&K::SelfType<'_>>,
        start_inclusive: bool,
        end_inclusive: bool,
    ) -> crate::Result<Self::RangeIter<'a>>;

    /// Remove all entries. Returns the number of entries removed.
    fn st_drain_all(&mut self) -> crate::Result<u64>;
}

// ---------------------------------------------------------------------------
// ReadTable -- read-only table operations
// ---------------------------------------------------------------------------

/// A read-only table handle.
///
/// Both `ReadOnlyTable<K, V>` (legacy) and `BfTreeReadOnlyTable<K, V>` implement
/// this trait.
pub trait ReadTable<K: Key + 'static, V: Value + 'static> {
    /// Range iterator type.
    type RangeIter<'a>: Iterator<Item = crate::Result<(OwnedKv<K>, OwnedKv<V>)>>
    where
        Self: 'a;

    /// Read the value for a key.
    fn st_get(&self, key: &K::SelfType<'_>) -> crate::Result<Option<OwnedKv<V>>>;

    /// Range scan.
    fn st_range<'a>(
        &'a self,
        start: Option<&K::SelfType<'_>>,
        end: Option<&K::SelfType<'_>>,
        start_inclusive: bool,
        end_inclusive: bool,
    ) -> crate::Result<Self::RangeIter<'a>>;
}

// Note: No blanket `impl<T: WriteTable> ReadTable for T` -- Rust's orphan rules
// would prevent downstream crates from adding `WriteTable` impls, causing E0119
// with explicit `ReadTable` impls for read-only types. Instead, both `WriteTable`
// and `ReadTable` are implemented explicitly for each table type.

// ---------------------------------------------------------------------------
// StorageWrite -- write transaction that opens typed tables
// ---------------------------------------------------------------------------

/// A write transaction that can open typed table handles.
///
/// Both `WriteTransaction` (legacy) and `BfTreeDatabaseWriteTxn` implement this.
///
/// The `'txn` lifetime on the associated type allows the returned table handle
/// to borrow from the transaction (e.g., `Table<'txn, K, V>` holds a `&'txn WriteTransaction`).
pub trait StorageWrite {
    /// The writable table type returned by `open_storage_table`.
    ///
    /// The lifetime `'txn` ties the table handle to `&'txn self`, ensuring the
    /// table cannot outlive the transaction.
    type Table<'txn, K: Key + 'static, V: Value + 'static>: WriteTable<K, V>
    where
        Self: 'txn;

    /// Open a typed table for reading and writing.
    ///
    /// If the table does not exist, it is created.
    fn open_storage_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: TableDefinition<K, V>,
    ) -> crate::Result<Self::Table<'_, K, V>>;
}

// ---------------------------------------------------------------------------
// StorageRead -- read transaction that opens typed tables
// ---------------------------------------------------------------------------

/// A read transaction that can open typed read-only table handles.
///
/// Both `ReadTransaction` (legacy) and `BfTreeDatabaseReadTxn` implement this.
pub trait StorageRead {
    /// The read-only table type returned by `open_storage_table`.
    ///
    /// The lifetime `'txn` ties the table handle to `&'txn self`.
    type Table<'txn, K: Key + 'static, V: Value + 'static>: ReadTable<K, V>
    where
        Self: 'txn;

    /// Open a typed table for reading.
    fn open_storage_table<K: Key + 'static, V: Value + 'static>(
        &self,
        definition: TableDefinition<K, V>,
    ) -> crate::Result<Self::Table<'_, K, V>>;
}

// ---------------------------------------------------------------------------
// Range helper -- convert RangeBounds to the (start, end, inclusive) tuple
// ---------------------------------------------------------------------------

/// Decompose a `RangeBounds<K::SelfType<'_>>` into start/end byte boundaries.
///
/// Returns `(start_bytes, end_bytes, start_inclusive, end_inclusive)`.
/// `None` means unbounded on that side.
pub fn range_to_bounds<K: Key + 'static>(
    range: &impl RangeBounds<Vec<u8>>,
) -> (Option<Vec<u8>>, Option<Vec<u8>>, bool, bool) {
    use core::ops::Bound;

    let start = match range.start_bound() {
        Bound::Included(k) => (Some(k.clone()), true),
        Bound::Excluded(k) => (Some(k.clone()), false),
        Bound::Unbounded => (None, true),
    };

    let end = match range.end_bound() {
        Bound::Included(k) => (Some(k.clone()), true),
        Bound::Excluded(k) => (Some(k.clone()), false),
        Bound::Unbounded => (None, true),
    };

    (start.0, end.0, start.1, end.1)
}

/// Encode a key value to bytes for range bound operations.
pub fn key_to_bytes<K: Key + 'static>(key: &K::SelfType<'_>) -> Vec<u8> {
    K::as_bytes(key).as_ref().to_vec()
}

/// Helper to construct a typed range query using standard `RangeBounds`.
///
/// Converts Rust range syntax (`start..end`, `start..=end`, `..`, etc.)
/// into the `(start, end, start_inclusive, end_inclusive)` tuple that
/// `WriteTable::st_range` and `ReadTable::st_range` expect.
pub struct TypedRange {
    pub start: Option<Vec<u8>>,
    pub end: Option<Vec<u8>>,
    pub start_inclusive: bool,
    pub end_inclusive: bool,
}

impl TypedRange {
    /// Create from a `RangeBounds` of key bytes.
    pub fn from_byte_bounds(range: &impl RangeBounds<Vec<u8>>) -> Self {
        let (start, end, si, ei) = range_to_bounds::<()>(range);
        Self {
            start,
            end,
            start_inclusive: si,
            end_inclusive: ei,
        }
    }

    /// Create from typed key bounds. Encodes keys to bytes.
    pub fn from_key_pair<K: Key + 'static>(
        start: Option<&K::SelfType<'_>>,
        end: Option<&K::SelfType<'_>>,
        start_inclusive: bool,
        end_inclusive: bool,
    ) -> Self {
        Self {
            start: start.map(|k| key_to_bytes::<K>(k)),
            end: end.map(|k| key_to_bytes::<K>(k)),
            start_inclusive,
            end_inclusive,
        }
    }
}

// ---------------------------------------------------------------------------
// Table name type
// ---------------------------------------------------------------------------

/// Represents a table name for dynamic table creation.
///
/// Used when index code needs to construct table names at runtime
/// (e.g., `__ivfpq:myindex:postings`).
#[derive(Debug, Clone)]
pub struct DynTableName(pub String);

impl DynTableName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
