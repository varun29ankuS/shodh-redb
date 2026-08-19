//! IVF-PQ vector index -- disk-first approximate nearest neighbor search.
//!
//! This module implements an Inverted File Index with Product Quantization
//! (IVF-PQ) built on shodh-redb's B-tree storage. It replaces brute-force
//! `nearest_k` scans with O(sqrt(n)) approximate search that scales to
//! 100M+ vectors.
//!
//! # Architecture
//!
//! - **IVF**: Partitions the vector space into clusters via k-means. At query
//!   time, only the `nprobe` nearest clusters are scanned.
//! - **PQ**: Compresses each vector from `dim * 4` bytes to `num_subvectors`
//!   bytes (one byte per sub-quantizer), a rate of `8 * num_subvectors / dim`
//!   bits per dimension. Trained codebooks map each sub-vector to its nearest
//!   centroid index. That rate is the main recall dial; `IndexConfig`'s
//!   `num_subvectors` documentation carries measured figures.
//! - **ADC**: At search time, a precomputed distance lookup table enables
//!   approximate distance computation via `num_subvectors` table lookups per
//!   candidate.
//!
//! All index data is stored in regular B-tree tables with prefixed names,
//! fully ACID and crash-safe.
//!
//! # Example
//!
//! ```rust,ignore
//! use shodh_redb::{Database, DistanceMetric, IvfPqIndexDefinition, SearchParams};
//!
//! // 48 sub-vectors over 384 dimensions is 1 bit per dimension -- 48 bytes per
//! // vector against 1536 raw. At that rate the codes need re-ranking to reach
//! // high recall, which is what `with_raw_vectors` buys. For roughly 90%
//! // recall from the codes alone, use `dim / 2` sub-vectors (192 here) and
//! // drop `with_raw_vectors`. See `IndexConfig::num_subvectors`.
//! const INDEX: IvfPqIndexDefinition = IvfPqIndexDefinition::new(
//!     "embeddings", 384, 256, 48, DistanceMetric::EuclideanSq,
//! ).with_raw_vectors();
//!
//! let db = Database::create("vectors.redb")?;
//!
//! // Train + insert
//! let write_txn = db.begin_write()?;
//! let mut idx = write_txn.open_ivfpq_index(&INDEX)?;
//! idx.train(training_data.into_iter(), 25)?;
//! idx.insert(1, &embedding)?;
//! write_txn.commit()?;
//!
//! // Search
//! let read_txn = db.begin_read()?;
//! let idx = read_txn.open_ivfpq_index(&INDEX)?;
//! let results = idx.search(&read_txn, &query, &SearchParams::top_k(10))?;
//! ```

pub(crate) mod adc;
#[cfg(fuzzing)]
pub mod cluster_blob;
#[cfg(not(fuzzing))]
#[allow(dead_code)] // Wired into index.rs in upcoming commit.
pub(crate) mod cluster_blob;
pub mod config;
pub(crate) mod index;
#[cfg(fuzzing)]
pub mod kmeans;
#[cfg(not(fuzzing))]
pub(crate) mod kmeans;
pub mod metadata;
#[cfg(fuzzing)]
pub mod pq;
#[cfg(not(fuzzing))]
pub(crate) mod pq;
#[cfg(fuzzing)]
pub mod types;
#[cfg(not(fuzzing))]
pub(crate) mod types;

pub use config::{IndexConfig, IvfPqIndexDefinition, SearchParams};
pub use index::{IvfPqIndex, ReadOnlyIvfPqIndex};
pub use metadata::{MetadataFilter, MetadataMap, MetadataValue};

pub use pq::Codebooks;
