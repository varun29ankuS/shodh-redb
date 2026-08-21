use crate::probe_select::DiversityConfig;
use crate::vector_ops::DistanceMetric;
use core::fmt;

use super::metadata::MetadataFilter;

// ---------------------------------------------------------------------------
// IndexConfig -- persisted index configuration
// ---------------------------------------------------------------------------

/// Persisted configuration for an IVF-PQ index. Stored as a single row in the
/// metadata table and loaded at index-open time.
#[derive(Clone, PartialEq)]
pub struct IndexConfig {
    /// Vector dimensionality (e.g. 384, 768, 1536).
    pub dim: u32,
    /// Number of IVF clusters (centroids). Typical: 256-4096.
    pub num_clusters: u32,
    /// Number of PQ sub-vectors. `dim` must be divisible by this.
    /// Each sub-vector is `dim / num_subvectors` floats.
    ///
    /// This is the compression/recall dial, and the useful way to read it is
    /// **bits per dimension**: each sub-vector becomes one byte, so the rate is
    /// `8 * num_subvectors / dim`. Measured recall@10 at dim 128, 256 clusters,
    /// nprobe at the default, no re-ranking:
    ///
    /// | `num_subvectors` | bits/dim | bytes/vector | recall@10 |
    /// |---|---|---|---|
    /// | `dim / 8` | 1 | `dim / 8` | low -- needs re-ranking |
    /// | `dim / 4` | 2 | `dim / 4` | ~0.74 |
    /// | `dim / 2` | 4 | `dim / 2` | ~0.90 |
    ///
    /// So `num_subvectors = dim / 2` is the setting that reaches roughly 90%
    /// recall on its own. Below that, plan on `store_raw_vectors` and
    /// re-ranking to make up the difference -- which costs `4 * dim` bytes per
    /// vector, far more than the codes themselves, so it is a real trade rather
    /// than a free upgrade.
    ///
    /// Recall also depends on the data: these figures come from a synthetic
    /// corpus with a deliberately flat spectrum, which is the hard case for
    /// product quantization. Real embeddings, whose variance concentrates in a
    /// few directions, do better at the same bit rate.
    pub num_subvectors: u32,
    /// Codewords per sub-quantizer. Always 256 (u8 codes).
    pub num_codewords: u16,
    /// Distance metric used for training, encoding, and search.
    pub metric: DistanceMetric,
    /// Whether to store full-precision vectors for re-ranking.
    pub store_raw_vectors: bool,
    /// Default number of clusters to probe at search time.
    pub default_nprobe: u32,
    /// Training state: 0 = untrained, 1 = trained.
    pub(crate) state: u8,
    /// Total number of vectors currently in the index.
    pub num_vectors: u64,
    /// Storage format version. 0 = legacy per-entry posting lists,
    /// 1 = contiguous cluster blobs.
    pub format_version: u8,
}

/// Training state: index has not been trained yet.
pub const STATE_UNTRAINED: u8 = 0;
/// Training state: index is trained and ready for inserts/queries.
pub const STATE_TRAINED: u8 = 1;

/// Format version: legacy per-entry posting lists (deprecated).
pub const FORMAT_V0_LEGACY: u8 = 0;
/// Format version: contiguous cluster blobs.
pub const FORMAT_V1_BLOBS: u8 = 1;

impl IndexConfig {
    /// Returns the training state (0 = untrained, 1 = trained).
    pub fn state(&self) -> u8 {
        self.state
    }

    /// Returns the dimensionality of each PQ sub-vector.
    pub fn sub_dim(&self) -> usize {
        if self.num_subvectors == 0 {
            return 0;
        }
        self.dim as usize / self.num_subvectors as usize
    }

    /// Returns the byte discriminant for the distance metric.
    pub fn metric_byte(&self) -> u8 {
        match self.metric {
            DistanceMetric::Cosine => 0,
            DistanceMetric::EuclideanSq => 1,
            DistanceMetric::DotProduct => 2,
            DistanceMetric::Manhattan => 3,
        }
    }
}

impl fmt::Debug for IndexConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("IndexConfig");
        s.field("dim", &self.dim)
            .field("num_clusters", &self.num_clusters)
            .field("num_subvectors", &self.num_subvectors)
            .field("num_codewords", &self.num_codewords)
            .field("metric", &self.metric)
            .field("store_raw", &self.store_raw_vectors)
            .field("nprobe", &self.default_nprobe)
            .field("state", &self.state)
            .field("num_vectors", &self.num_vectors)
            .field("format_version", &self.format_version)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SearchParams -- per-query search configuration
// ---------------------------------------------------------------------------

/// Parameters for a single IVF-PQ approximate nearest neighbor query.
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Number of IVF clusters to probe. Higher = more accurate, slower.
    /// Must be >= 1. Clamped to `num_clusters` at search time.
    pub nprobe: u32,
    /// Number of PQ-distance candidates to shortlist before re-ranking.
    /// Only meaningful when `rerank` is true. Defaults to `k * 10`.
    pub candidates: usize,
    /// Number of final results to return.
    pub k: usize,
    /// If true and the index stores raw vectors, re-rank the top
    /// `candidates` with exact distances for higher recall.
    pub rerank: bool,
    /// Diversity-aware probe selection. Default: disabled (lambda=0.0).
    pub diversity: DiversityConfig,
    /// Optional metadata filter. When set, candidates whose metadata does not
    /// match this predicate are excluded before entering the top-K heap.
    pub filter: Option<MetadataFilter>,
}

impl SearchParams {
    /// Create search params for a top-k query with default settings.
    pub fn top_k(k: usize) -> Self {
        Self {
            nprobe: 10,
            candidates: k.saturating_mul(10).max(100),
            k,
            rerank: true,
            diversity: DiversityConfig { lambda: 0.0 },
            filter: None,
        }
    }

    /// Enable diversity-aware probe selection.
    /// `lambda` in [0.0, 1.0]: 0.0 = pure distance (default), higher = more diversity.
    #[must_use]
    pub fn with_diversity(mut self, lambda: f32) -> Self {
        self.diversity = DiversityConfig {
            lambda: lambda.clamp(0.0, 1.0),
        };
        self
    }

    /// Set a metadata filter. Candidates whose metadata does not match are
    /// excluded from results.
    #[must_use]
    pub fn with_filter(mut self, filter: MetadataFilter) -> Self {
        self.filter = Some(filter);
        self
    }
}

// ---------------------------------------------------------------------------
// IvfPqIndexDefinition -- user-facing index declaration
// ---------------------------------------------------------------------------

/// Definition for an IVF-PQ vector index.
///
/// Analogous to [`crate::TableDefinition`] -- a compile-time description of an
/// index that is passed to `open_ivfpq_index()` to create or open it.
///
/// # Example
///
/// ```rust,ignore
/// use shodh_redb::{DistanceMetric, IvfPqIndexDefinition};
///
/// // 96 sub-vectors over 768 dimensions is 1 bit per dimension: 96 bytes per
/// // vector, a 32x compression. At that rate the codes alone do not carry
/// // enough information for high recall, so this configuration stores raw
/// // vectors and re-ranks -- which is what `with_raw_vectors` is paying for.
/// const COMPACT: IvfPqIndexDefinition = IvfPqIndexDefinition::new(
///     "embeddings", 768, 256, 96, DistanceMetric::EuclideanSq,
/// ).with_raw_vectors();
///
/// // 384 sub-vectors is 4 bits per dimension: 384 bytes per vector, 8x
/// // compression, and roughly 90% recall without re-ranking. Larger codes,
/// // but no `4 * dim` bytes of raw vectors and no second pass at query time.
/// const STANDALONE: IvfPqIndexDefinition = IvfPqIndexDefinition::new(
///     "embeddings", 768, 256, 384, DistanceMetric::EuclideanSq,
/// );
/// ```
///
/// `nprobe` defaults to `sqrt(num_clusters)`, which is where recall stops
/// improving in measurement; set it explicitly with `with_nprobe` only to trade
/// query time against recall deliberately.
pub struct IvfPqIndexDefinition {
    name: &'static str,
    dim: u32,
    num_clusters: u32,
    num_subvectors: u32,
    metric: DistanceMetric,
    store_raw_vectors: bool,
    default_nprobe: u32,
}

/// Default number of clusters to probe, derived from the partition size.
///
/// This was a flat `10` regardless of `num_clusters`, which means very
/// different things at different scales: 16% of a 64-cluster index, but 0.24%
/// of a 4096-cluster one. The larger the index, the more the fixed default
/// under-probed it -- precisely where recall matters most.
///
/// `sqrt(num_clusters)` is the standard heuristic and it matches measurement.
/// On a 256-cluster index at dim 128, recall@10 rises steeply to nprobe 16 and
/// then flattens: 4 -> 0.410, 16 -> 0.575, 32 -> 0.580. The knee is at 16,
/// which is exactly `sqrt(256)`. Probing beyond it costs query time for
/// hundredths of a point of recall.
///
/// Callers who want a different trade-off set it explicitly with
/// [`IvfPqIndexDefinition::with_nprobe`] or per-query via [`SearchParams`].
#[must_use]
pub const fn default_nprobe_for(num_clusters: u32) -> u32 {
    if num_clusters == 0 {
        return 1;
    }
    // Integer sqrt by bit-halving; `u32::isqrt` is not const-stable on the MSRV.
    let mut root = 0u32;
    let mut bit = 1u32 << 30;
    let mut n = num_clusters;
    while bit > n {
        bit >>= 2;
    }
    while bit != 0 {
        if n >= root + bit {
            n -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    if root == 0 { 1 } else { root }
}

impl IvfPqIndexDefinition {
    /// Create a new IVF-PQ index definition.
    ///
    /// `dim` must be divisible by `num_subvectors`.
    pub const fn new(
        name: &'static str,
        dim: u32,
        num_clusters: u32,
        num_subvectors: u32,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            name,
            dim,
            num_clusters,
            num_subvectors,
            metric,
            store_raw_vectors: false,
            default_nprobe: default_nprobe_for(num_clusters),
        }
    }

    /// Enable storage of full-precision vectors for re-ranking.
    #[must_use]
    pub const fn with_raw_vectors(mut self) -> Self {
        self.store_raw_vectors = true;
        self
    }

    /// Set the default number of clusters to probe at search time.
    #[must_use]
    pub const fn with_nprobe(mut self, nprobe: u32) -> Self {
        self.default_nprobe = nprobe;
        self
    }

    /// Returns the index name.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the requested number of IVF clusters.
    pub const fn num_clusters(&self) -> u32 {
        self.num_clusters
    }

    /// Convert to a full [`IndexConfig`] (with state=untrained, `num_vectors`=0).
    pub fn to_config(&self) -> IndexConfig {
        IndexConfig {
            dim: self.dim,
            num_clusters: self.num_clusters,
            num_subvectors: self.num_subvectors,
            num_codewords: 256,
            metric: self.metric,
            store_raw_vectors: self.store_raw_vectors,
            default_nprobe: self.default_nprobe,
            state: STATE_UNTRAINED,
            num_vectors: 0,
            format_version: FORMAT_V1_BLOBS,
        }
    }
}

impl fmt::Debug for IvfPqIndexDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IvfPqIndexDefinition({:?}, dim={}, clusters={}, subvecs={}, {:?}",
            self.name, self.dim, self.num_clusters, self.num_subvectors, self.metric,
        )?;
        write!(f, ")")
    }
}

#[cfg(test)]
mod default_nprobe_tests {
    use super::{IvfPqIndexDefinition, default_nprobe_for};
    use crate::vector_ops::DistanceMetric;

    /// The measured knee. On a 256-cluster index at dim 128, recall@10 goes
    /// 4 -> 0.410, 16 -> 0.575, 32 -> 0.580: it flattens at 16, and
    /// `sqrt(256) == 16`.
    #[test]
    fn matches_the_measured_knee_at_256_clusters() {
        assert_eq!(default_nprobe_for(256), 16);
    }

    /// The old flat default of 10 meant wildly different coverage at different
    /// scales. These are the cases it got worst: a large partition barely
    /// probed at all.
    #[test]
    fn scales_with_the_partition() {
        assert_eq!(default_nprobe_for(64), 8);
        assert_eq!(default_nprobe_for(1024), 32);
        assert_eq!(default_nprobe_for(4096), 64);
        assert_eq!(default_nprobe_for(16384), 128);
    }

    /// Non-square inputs floor, which is the conservative direction: probing
    /// one cluster fewer costs a little recall, one more costs query time.
    #[test]
    fn non_squares_floor() {
        assert_eq!(default_nprobe_for(255), 15);
        assert_eq!(default_nprobe_for(257), 16);
        assert_eq!(default_nprobe_for(100), 10);
        assert_eq!(default_nprobe_for(99), 9);
    }

    /// Never zero: a zero nprobe would probe nothing and return no results.
    /// `num_clusters == 0` is rejected by `validate_config`, but this must not
    /// depend on that.
    #[test]
    fn never_returns_zero() {
        assert_eq!(default_nprobe_for(0), 1);
        assert_eq!(default_nprobe_for(1), 1);
        assert_eq!(default_nprobe_for(2), 1);
        assert_eq!(default_nprobe_for(3), 1);
        assert_eq!(default_nprobe_for(4), 2);
    }

    /// The largest input must not overflow the bit-halving loop.
    #[test]
    fn handles_the_maximum() {
        assert_eq!(default_nprobe_for(u32::MAX), 65535);
    }

    /// A definition picks the derived value up, and an explicit `with_nprobe`
    /// still overrides it.
    #[test]
    fn definition_uses_the_derived_default() {
        const DERIVED: IvfPqIndexDefinition =
            IvfPqIndexDefinition::new("t", 128, 256, 32, DistanceMetric::EuclideanSq);
        const EXPLICIT: IvfPqIndexDefinition =
            IvfPqIndexDefinition::new("t", 128, 256, 32, DistanceMetric::EuclideanSq)
                .with_nprobe(3);

        assert_eq!(DERIVED.to_config().default_nprobe, 16);
        assert_eq!(EXPLICIT.to_config().default_nprobe, 3);
    }
}
