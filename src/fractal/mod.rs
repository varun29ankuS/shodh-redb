//! Adaptive fractal vector index -- self-organizing hierarchical clustering.
//!
//! This module implements a fractal vector index that replaces static IVF-PQ
//! centroids with a self-organizing cluster tree. Clusters maintain running
//! centroids, split when population exceeds a threshold, and merge when
//! population drops. Write-buffered cascading amortizes structural changes.
//!
//! # Architecture
//!
//! - **Hierarchical cluster tree**: Internal nodes route queries; leaf nodes
//!   own posting lists with PQ-compressed vectors.
//! - **Write buffers**: New vectors are buffered at leaf clusters. When the
//!   buffer fills, vectors are PQ-encoded and moved to the posting list.
//! - **Incremental centroids**: Welford's online algorithm with f64 sum
//!   accumulators for precision across millions of updates.
//! - **Global PQ codebooks**: Trained once, shared across all clusters.
//! - **Temporal awareness**: Each cluster tracks its temporal range.
//!   Search can skip clusters outside a query's time window.
//!
//! All data is stored in regular B-tree tables, fully ACID and crash-safe.
//!
//! # Example
//!
//! ```rust,ignore
//! use shodh_redb::{Database, DistanceMetric, FractalIndexDefinition, FractalSearchParams};
//!
//! const INDEX: FractalIndexDefinition = FractalIndexDefinition::new(
//!     "memory", 384, 48, DistanceMetric::Cosine,
//! ).with_raw_vectors();
//!
//! let db = Database::create("memory.redb")?;
//!
//! // Train codebooks + insert
//! let write_txn = db.begin_write()?;
//! let mut idx = write_txn.open_fractal_index(&INDEX)?;
//! idx.train_codebooks(training_data.into_iter(), 25)?;
//! idx.insert(1, &embedding)?;
//! write_txn.commit()?;
//!
//! // Search
//! let read_txn = db.begin_read()?;
//! let idx = read_txn.open_fractal_index(&INDEX)?;
//! let results = idx.search(&read_txn, &query, &FractalSearchParams::top_k(10))?;
//! ```

pub(crate) mod cluster;
pub mod config;
pub(crate) mod index;
pub(crate) mod search;
#[cfg(fuzzing)]
pub mod types;
#[cfg(not(fuzzing))]
pub(crate) mod types;

pub use config::{FractalIndexConfig, FractalIndexDefinition, FractalSearchParams};
pub use index::{FractalIndex, ReadOnlyFractalIndex};
