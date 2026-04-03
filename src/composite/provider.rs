//! Backend-agnostic trait for blob queries used by `CompositeQuery`.
//!
//! `BlobQueryProvider` abstracts the blob-related query operations that
//! `CompositeQuery` needs. Both the legacy `ReadTransaction` and the
//! `BfTree` `BfTreeReadOnlyBlobStore` implement this trait, enabling
//! composite queries to work with either storage backend.

use alloc::vec::Vec;

use crate::blob_store::types::{BlobId, BlobMeta, CausalEdge, TemporalKey};

/// Provides blob query operations needed by `CompositeQuery`.
///
/// This trait decouples the composite query engine from the storage backend,
/// allowing it to work with both legacy B-tree and `BfTree` blob stores.
pub trait BlobQueryProvider {
    /// Error type returned by blob operations.
    type Error: Into<crate::StorageError>;

    /// Get blob metadata by ID.
    fn get_blob_meta(&self, blob_id: &BlobId) -> Result<Option<BlobMeta>, Self::Error>;

    /// Look up a blob by its sequence number.
    ///
    /// Returns the `(BlobId, BlobMeta)` pair for the blob with the given sequence,
    /// or `None` if no blob matches.
    fn blob_by_sequence(&self, seq: u64) -> Result<Option<(BlobId, BlobMeta)>, Self::Error>;

    /// Query blobs in a temporal range `[start_ns, end_ns]`.
    fn blobs_in_time_range(
        &self,
        start_ns: u64,
        end_ns: u64,
    ) -> Result<Vec<(TemporalKey, BlobMeta)>, Self::Error>;

    /// Query blobs in a namespace, returning `(BlobId, BlobMeta)` pairs.
    fn blobs_in_namespace(
        &self,
        namespace: &str,
    ) -> Result<Vec<(BlobId, BlobMeta)>, Self::Error>;

    /// Query blobs that have the given tag.
    fn blobs_by_tag(&self, tag: &str) -> Result<Vec<BlobId>, Self::Error>;

    /// Get the causal children of a parent blob.
    fn causal_children(&self, blob_id: &BlobId) -> Result<Vec<CausalEdge>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Legacy ReadTransaction implementation
// ---------------------------------------------------------------------------

impl BlobQueryProvider for crate::transactions::ReadTransaction {
    type Error = crate::StorageError;

    fn get_blob_meta(&self, blob_id: &BlobId) -> Result<Option<BlobMeta>, Self::Error> {
        self.get_blob_meta(blob_id)
    }

    fn blob_by_sequence(&self, seq: u64) -> Result<Option<(BlobId, BlobMeta)>, Self::Error> {
        self.blob_by_sequence(seq)
    }

    fn blobs_in_time_range(
        &self,
        start_ns: u64,
        end_ns: u64,
    ) -> Result<Vec<(TemporalKey, BlobMeta)>, Self::Error> {
        self.blobs_in_time_range(start_ns, end_ns)
    }

    fn blobs_in_namespace(
        &self,
        namespace: &str,
    ) -> Result<Vec<(BlobId, BlobMeta)>, Self::Error> {
        self.blobs_in_namespace(namespace)
    }

    fn blobs_by_tag(&self, tag: &str) -> Result<Vec<BlobId>, Self::Error> {
        self.blobs_by_tag(tag)
    }

    fn causal_children(&self, blob_id: &BlobId) -> Result<Vec<CausalEdge>, Self::Error> {
        self.causal_children(blob_id)
    }
}

// ---------------------------------------------------------------------------
// BfTree ReadOnlyBlobStore implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "bf_tree")]
impl BlobQueryProvider for crate::bf_tree_store::BfTreeReadOnlyBlobStore<'_> {
    type Error = crate::bf_tree_store::BfTreeError;

    fn get_blob_meta(&self, blob_id: &BlobId) -> Result<Option<BlobMeta>, Self::Error> {
        self.get_meta(*blob_id)
    }

    fn blob_by_sequence(&self, seq: u64) -> Result<Option<(BlobId, BlobMeta)>, Self::Error> {
        self.blob_by_sequence(seq)
    }

    fn blobs_in_time_range(
        &self,
        start_ns: u64,
        end_ns: u64,
    ) -> Result<Vec<(TemporalKey, BlobMeta)>, Self::Error> {
        self.blobs_in_time_range(start_ns, end_ns)
    }

    fn blobs_in_namespace(
        &self,
        namespace: &str,
    ) -> Result<Vec<(BlobId, BlobMeta)>, Self::Error> {
        self.blobs_in_namespace(namespace)
    }

    fn blobs_by_tag(&self, tag: &str) -> Result<Vec<BlobId>, Self::Error> {
        self.query_by_tag(tag)
    }

    fn causal_children(&self, blob_id: &BlobId) -> Result<Vec<CausalEdge>, Self::Error> {
        self.causal_children(*blob_id)
    }
}
