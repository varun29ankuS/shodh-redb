use crate::StorageError;
use crate::blob_store::types::BlobId;
use crate::compat::HashMap;
use crate::fractal::{FractalSearchParams, ReadOnlyFractalIndex};
use crate::ivfpq::{ReadOnlyIvfPqIndex, SearchParams};
use crate::transactions::ReadTransaction;
use alloc::collections::BinaryHeap;
use alloc::string::ToString;
use alloc::vec::Vec;

use super::provider::BlobQueryProvider;
use super::scoring::{causal_bfs, normalize_causal, normalize_semantic, normalize_temporal};
use super::types::{
    CandidateEntry, HeapEntry, ScoredBlob, SignalScores, SignalWeights, heap_to_sorted,
};

/// Builder for composite multi-signal queries.
///
/// Fuses vector similarity, temporal recency, and causal proximity into a single
/// ranked result set using weighted linear combination. Supports optional
/// pre-filtering by namespace and tags.
///
/// The type parameter `P` is the blob query provider -- typically a
/// `ReadTransaction` (legacy B-tree) or `BfTreeReadOnlyBlobStore` (`BfTree`).
///
/// # Example
///
/// ```rust,ignore
/// let results = CompositeQuery::new(&txn)
///     .semantic(&index, &query_vec, 0.6)
///     .temporal(0.3)
///     .time_range(start_ns, end_ns)
///     .causal(root_id, 0.1)
///     .namespace("sensor-data")
///     .tag("lidar")
///     .top_k(10)
///     .execute()?;
/// ```
pub struct CompositeQuery<'a, P: BlobQueryProvider = ReadTransaction> {
    provider: &'a P,

    /// Legacy transaction reference -- needed for vector index search which
    /// takes `&ReadTransaction` directly. `None` when using `BfTree` backend.
    legacy_txn: Option<&'a ReadTransaction>,

    // Semantic signal (IVF-PQ or Fractal)
    vector_index: Option<&'a ReadOnlyIvfPqIndex>,
    fractal_index: Option<&'a ReadOnlyFractalIndex>,
    query_vector: Option<&'a [f32]>,
    search_params: Option<SearchParams>,
    fractal_search_params: Option<FractalSearchParams>,
    semantic_candidates: Option<usize>,

    // Temporal signal
    time_range: Option<(u64, u64)>,

    // Causal signal
    causal_root: Option<BlobId>,
    causal_max_hops: usize,

    // Filtering
    namespace: Option<&'a str>,
    tags: Vec<&'a str>,

    // Weights and limits
    weights: SignalWeights,
    top_k: usize,
}

/// Convenience constructor for legacy `ReadTransaction`.
impl<'a> CompositeQuery<'a, ReadTransaction> {
    /// Start building a composite query on the given legacy read transaction.
    pub fn new(txn: &'a ReadTransaction) -> Self {
        Self {
            provider: txn,
            legacy_txn: Some(txn),
            vector_index: None,
            fractal_index: None,
            query_vector: None,
            search_params: None,
            fractal_search_params: None,
            semantic_candidates: None,
            time_range: None,
            causal_root: None,
            causal_max_hops: 32,
            namespace: None,
            tags: Vec::new(),
            weights: SignalWeights::default(),
            top_k: 10,
        }
    }
}

impl<'a, P: BlobQueryProvider> CompositeQuery<'a, P> {
    /// Start building a composite query on any `BlobQueryProvider`.
    ///
    /// For the legacy backend, prefer [`CompositeQuery::new()`] which
    /// automatically sets up vector index search support.
    pub fn with_provider(provider: &'a P) -> Self {
        Self {
            provider,
            legacy_txn: None,
            vector_index: None,
            fractal_index: None,
            query_vector: None,
            search_params: None,
            fractal_search_params: None,
            semantic_candidates: None,
            time_range: None,
            causal_root: None,
            causal_max_hops: 32,
            namespace: None,
            tags: Vec::new(),
            weights: SignalWeights::default(),
            top_k: 10,
        }
    }

    /// Attach a legacy `ReadTransaction` for vector index search.
    ///
    /// Required when using `BfTree` provider with IVF-PQ or Fractal vector
    /// indices, since those indices currently require a `ReadTransaction`.
    #[must_use]
    pub fn with_legacy_txn(mut self, txn: &'a ReadTransaction) -> Self {
        self.legacy_txn = Some(txn);
        self
    }

    /// Set the semantic similarity signal with the given weight.
    ///
    /// Requires a pre-opened IVF-PQ index and a query vector.
    /// Weight of 0.0 disables the signal.
    #[must_use]
    pub fn semantic(
        mut self,
        index: &'a ReadOnlyIvfPqIndex,
        query: &'a [f32],
        weight: f32,
    ) -> Self {
        self.vector_index = Some(index);
        self.query_vector = Some(query);
        self.weights.semantic = weight.max(0.0);
        self
    }

    /// Override IVF-PQ search parameters (nprobe, rerank, etc.).
    #[must_use]
    pub fn search_params(mut self, params: SearchParams) -> Self {
        self.search_params = Some(params);
        self
    }

    /// Override how many vector candidates to fetch. Default: `top_k * 5`.
    #[must_use]
    pub fn semantic_candidates(mut self, n: usize) -> Self {
        self.semantic_candidates = Some(n);
        self
    }

    /// Set the semantic signal using a fractal vector index.
    ///
    /// Alternative to [`semantic()`](Self::semantic) -- uses the adaptive
    /// fractal index instead of IVF-PQ. If both are set, fractal takes priority.
    #[must_use]
    pub fn semantic_fractal(
        mut self,
        index: &'a ReadOnlyFractalIndex,
        query: &'a [f32],
        weight: f32,
    ) -> Self {
        self.fractal_index = Some(index);
        self.query_vector = Some(query);
        self.weights.semantic = weight.max(0.0);
        self
    }

    /// Override fractal search parameters.
    #[must_use]
    pub fn fractal_search_params(mut self, params: FractalSearchParams) -> Self {
        self.fractal_search_params = Some(params);
        self
    }

    /// Set the temporal recency signal with the given weight.
    ///
    /// More recent blobs score higher. Use `time_range()` to restrict the
    /// temporal window; without it, all candidates are scored by recency.
    #[must_use]
    pub fn temporal(mut self, weight: f32) -> Self {
        self.weights.temporal = weight.max(0.0);
        self
    }

    /// Restrict temporal candidate generation to a wall-clock range (nanoseconds).
    #[must_use]
    pub fn time_range(mut self, start_ns: u64, end_ns: u64) -> Self {
        self.time_range = Some((start_ns, end_ns));
        self
    }

    /// Set the causal proximity signal with the given weight.
    ///
    /// Blobs closer to the root in the causal graph score higher.
    #[must_use]
    pub fn causal(mut self, root: BlobId, weight: f32) -> Self {
        self.causal_root = Some(root);
        self.weights.causal = weight.max(0.0);
        self
    }

    /// Override max hops for causal BFS. Default: 32.
    #[must_use]
    pub fn causal_max_hops(mut self, max_hops: usize) -> Self {
        self.causal_max_hops = max_hops;
        self
    }

    /// Pre-filter: only consider blobs in this namespace.
    #[must_use]
    pub fn namespace(mut self, ns: &'a str) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Pre-filter: only consider blobs with this tag.
    ///
    /// Multiple calls use AND semantics: the blob must have ALL specified tags.
    #[must_use]
    pub fn tag(mut self, tag: &'a str) -> Self {
        self.tags.push(tag);
        self
    }

    /// Number of results to return.
    #[must_use]
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Execute the composite query and return fused results, sorted by score descending.
    pub fn execute(self) -> crate::Result<Vec<ScoredBlob>> {
        self.validate()?;

        let (w_sem, w_tmp, w_cau) = self.weights.normalized_f64();
        let sem_active = w_sem > 0.0;
        let tmp_active = w_tmp > 0.0;
        let cau_active = w_cau > 0.0;

        // Phase 1: Collect candidates
        let mut candidates: HashMap<BlobId, CandidateEntry> = HashMap::new();

        // 1a: Semantic candidates (fractal index takes priority over IVF-PQ)
        if sem_active {
            let query = self.query_vector.ok_or_else(|| {
                StorageError::Corrupted(
                    "semantic search enabled but no query vector provided".to_string(),
                )
            })?;
            let k = self.semantic_candidates.unwrap_or(self.top_k.max(1) * 5);

            // Vector index search requires a legacy ReadTransaction.
            let txn = self.legacy_txn.ok_or_else(|| {
                StorageError::Corrupted(
                    "semantic search requires a ReadTransaction (use with_legacy_txn)".to_string(),
                )
            })?;

            let results = if let Some(fi) = self.fractal_index {
                let params = self.fractal_search_params.unwrap_or(FractalSearchParams {
                    nprobe: 8,
                    candidates: k * 2,
                    k,
                    rerank: true,
                    min_hlc: 0,
                    diversity: crate::probe_select::DiversityConfig { lambda: 0.0 },
                });
                fi.search(txn, query, &params)?
            } else {
                let index = self.vector_index.ok_or_else(|| {
                    StorageError::Corrupted(
                        "semantic search enabled but no vector index provided".to_string(),
                    )
                })?;
                let params = self.search_params.unwrap_or(SearchParams {
                    nprobe: 16,
                    candidates: k * 2,
                    k,
                    rerank: true,
                    diversity: crate::probe_select::DiversityConfig { lambda: 0.0 },
                });
                index.search(txn, query, &params)?
            };

            for neighbor in &results {
                if let Some((id, meta)) = self
                    .provider
                    .blob_by_sequence(neighbor.key)
                    .map_err(Into::into)?
                {
                    let entry = candidates.entry(id).or_insert_with(|| CandidateEntry {
                        blob_id: id,
                        meta: None,
                        raw_distance: None,
                        wall_clock_ns: None,
                        causal_hops: None,
                    });
                    entry.raw_distance = Some(neighbor.distance);
                    entry.meta = Some(meta.clone());
                    entry.wall_clock_ns = Some(meta.wall_clock_ns);
                }
            }
        }

        // 1b: Temporal candidates
        if tmp_active && let Some((start, end)) = self.time_range {
            let temporal_results = self
                .provider
                .blobs_in_time_range(start, end)
                .map_err(Into::into)?;
            for (tkey, meta) in &temporal_results {
                let id = tkey.blob_id;
                let entry = candidates.entry(id).or_insert_with(|| CandidateEntry {
                    blob_id: id,
                    meta: None,
                    raw_distance: None,
                    wall_clock_ns: None,
                    causal_hops: None,
                });
                entry.wall_clock_ns = Some(meta.wall_clock_ns);
                if entry.meta.is_none() {
                    entry.meta = Some(meta.clone());
                }
            }
        }
        // If no time_range, temporal scores computed from metadata of existing candidates

        // 1c: Causal candidates
        let causal_distances = if cau_active {
            if let Some(ref root) = self.causal_root {
                let distances = causal_bfs(self.provider, root, self.causal_max_hops)?;
                for (id, hops) in &distances {
                    let entry = candidates.entry(*id).or_insert_with(|| CandidateEntry {
                        blob_id: *id,
                        meta: None,
                        raw_distance: None,
                        wall_clock_ns: None,
                        causal_hops: None,
                    });
                    entry.causal_hops = Some(*hops);
                }
                distances
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

        // Fill in missing metadata and timestamps
        for entry in candidates.values_mut() {
            if entry.meta.is_none() {
                if let Some(meta) = self
                    .provider
                    .get_blob_meta(&entry.blob_id)
                    .map_err(Into::into)?
                {
                    entry.wall_clock_ns = Some(meta.wall_clock_ns);
                    entry.meta = Some(meta);
                }
            } else if entry.wall_clock_ns.is_none() {
                entry.wall_clock_ns = entry.meta.as_ref().map(|m| m.wall_clock_ns);
            }
        }

        // Remove candidates without metadata (deleted/corrupt blobs)
        candidates.retain(|_, e| e.meta.is_some());

        // Phase 2: Pre-filtering
        if let Some(ns) = self.namespace {
            let ns_blobs: crate::compat::HashSet<BlobId> = self
                .provider
                .blobs_in_namespace(ns)
                .map_err(Into::into)?
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            candidates.retain(|id, _| ns_blobs.contains(id));
        }

        for tag in &self.tags {
            let tag_blobs: crate::compat::HashSet<BlobId> = self
                .provider
                .blobs_by_tag(tag)
                .map_err(Into::into)?
                .into_iter()
                .collect();
            candidates.retain(|id, _| tag_blobs.contains(id));
        }

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 3: Normalize and score
        let sem_scores = if sem_active {
            let pairs: Vec<(BlobId, f32)> = candidates
                .values()
                .filter_map(|e| e.raw_distance.map(|d| (e.blob_id, d)))
                .collect();
            normalize_semantic(&pairs)
        } else {
            HashMap::new()
        };

        let tmp_scores = if tmp_active {
            let pairs: Vec<(BlobId, u64)> = candidates
                .values()
                .filter_map(|e| e.wall_clock_ns.map(|t| (e.blob_id, t)))
                .collect();
            normalize_temporal(&pairs)
        } else {
            HashMap::new()
        };

        let cau_scores = if cau_active {
            normalize_causal(&causal_distances)
        } else {
            HashMap::new()
        };

        // Phase 4: Top-K selection
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(self.top_k + 1);

        for entry in candidates.values() {
            // Skip candidates without metadata -- they cannot be scored
            let Some(meta) = entry.meta.clone() else {
                continue;
            };

            let sem = if sem_active {
                Some(*sem_scores.get(&entry.blob_id).unwrap_or(&0.0))
            } else {
                None
            };
            let tmp = if tmp_active {
                Some(*tmp_scores.get(&entry.blob_id).unwrap_or(&0.0))
            } else {
                None
            };
            let cau = if cau_active {
                Some(*cau_scores.get(&entry.blob_id).unwrap_or(&0.0))
            } else {
                None
            };

            let score = w_sem * sem.unwrap_or(0.0)
                + w_tmp * tmp.unwrap_or(0.0)
                + w_cau * cau.unwrap_or(0.0);

            let he = HeapEntry {
                score,
                blob_id: entry.blob_id,
                meta,
                signals: SignalScores {
                    semantic: sem,
                    temporal: tmp,
                    causal: cau,
                },
            };

            heap.push(he);
            if heap.len() > self.top_k {
                heap.pop(); // evict lowest score
            }
        }

        Ok(heap_to_sorted(heap))
    }

    fn validate(&self) -> crate::Result<()> {
        if !self.weights.any_active() {
            return Err(StorageError::Corrupted(
                "CompositeQuery: at least one signal must have weight > 0".to_string(),
            ));
        }
        if self.weights.semantic > 0.0 && self.query_vector.is_none() {
            return Err(StorageError::Corrupted(
                "CompositeQuery: semantic signal requires a query vector".to_string(),
            ));
        }
        if self.weights.semantic > 0.0
            && self.vector_index.is_none()
            && self.fractal_index.is_none()
        {
            return Err(StorageError::Corrupted(
                "CompositeQuery: semantic signal requires an index (IVF-PQ or fractal)".to_string(),
            ));
        }
        if self.weights.semantic > 0.0 && self.legacy_txn.is_none() {
            return Err(StorageError::Corrupted(
                "CompositeQuery: semantic signal requires a ReadTransaction (use with_legacy_txn)"
                    .to_string(),
            ));
        }
        if self.weights.causal > 0.0 && self.causal_root.is_none() {
            return Err(StorageError::Corrupted(
                "CompositeQuery: causal signal requires a root blob_id".to_string(),
            ));
        }
        if self.top_k == 0 {
            return Err(StorageError::Corrupted(
                "CompositeQuery: top_k must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}
