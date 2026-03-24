use crate::probe_select::DiversityConfig;
use crate::vector_ops::DistanceMetric;
use core::fmt;

// ---------------------------------------------------------------------------
// FractalIndexConfig -- persisted index configuration (96 bytes)
// ---------------------------------------------------------------------------

/// Training / operational states.
pub const STATE_NEW: u8 = 0;
pub const STATE_CODEBOOKS_TRAINED: u8 = 1;
pub const STATE_OPERATIONAL: u8 = 2;

/// Sentinel value for "no parent" in `ClusterMeta.parent_id`.
pub const NO_PARENT: u32 = u32::MAX;

/// Flag bit: cluster is the root of the tree.
pub const FLAG_IS_ROOT: u8 = 0x01;

/// On-disk size of [`FractalIndexConfig`].
pub const FRACTAL_CONFIG_SIZE: usize = 96;

/// Persisted configuration for a fractal vector index.
#[derive(Clone, PartialEq)]
pub struct FractalIndexConfig {
    pub dim: u32,
    pub num_subvectors: u32,
    pub num_codewords: u16,
    pub metric: DistanceMetric,
    pub store_raw_vectors: bool,
    pub default_nprobe: u32,
    pub(crate) state: u8,
    pub root_cluster_id: u32,
    pub next_cluster_id: u32,
    pub max_leaf_population: u32,
    pub min_leaf_population: u32,
    pub max_buffer_size: u32,
    pub max_children: u32,
    pub max_depth: u32,
    pub num_vectors: u64,
    pub num_clusters: u64,
    pub variance_split_factor: f32,
}

impl FractalIndexConfig {
    pub fn state(&self) -> u8 {
        self.state
    }

    pub fn sub_dim(&self) -> usize {
        self.dim as usize / self.num_subvectors as usize
    }

    pub fn metric_byte(&self) -> u8 {
        match self.metric {
            DistanceMetric::Cosine => 0,
            DistanceMetric::EuclideanSq => 1,
            DistanceMetric::DotProduct => 2,
            DistanceMetric::Manhattan => 3,
        }
    }

    /// Allocate a new cluster ID, advancing the counter.
    pub fn alloc_cluster_id(&mut self) -> u32 {
        let id = self.next_cluster_id;
        self.next_cluster_id += 1;
        self.num_clusters += 1;
        id
    }
}

impl fmt::Debug for FractalIndexConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FractalIndexConfig")
            .field("dim", &self.dim)
            .field("num_subvectors", &self.num_subvectors)
            .field("metric", &self.metric)
            .field("store_raw", &self.store_raw_vectors)
            .field("nprobe", &self.default_nprobe)
            .field("state", &self.state)
            .field("root", &self.root_cluster_id)
            .field("num_vectors", &self.num_vectors)
            .field("num_clusters", &self.num_clusters)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Encode / Decode
// ---------------------------------------------------------------------------

pub fn encode_fractal_config(cfg: &FractalIndexConfig) -> [u8; FRACTAL_CONFIG_SIZE] {
    let mut buf = [0u8; FRACTAL_CONFIG_SIZE];
    buf[0..4].copy_from_slice(&cfg.dim.to_le_bytes());
    buf[4..8].copy_from_slice(&cfg.num_subvectors.to_le_bytes());
    buf[8..10].copy_from_slice(&cfg.num_codewords.to_le_bytes());
    buf[10] = cfg.metric_byte();
    buf[11] = u8::from(cfg.store_raw_vectors);
    buf[12..16].copy_from_slice(&cfg.default_nprobe.to_le_bytes());
    buf[16] = cfg.state;
    // 17..20 padding
    buf[20..24].copy_from_slice(&cfg.root_cluster_id.to_le_bytes());
    buf[24..28].copy_from_slice(&cfg.next_cluster_id.to_le_bytes());
    buf[28..32].copy_from_slice(&cfg.max_leaf_population.to_le_bytes());
    buf[32..36].copy_from_slice(&cfg.min_leaf_population.to_le_bytes());
    buf[36..40].copy_from_slice(&cfg.max_buffer_size.to_le_bytes());
    buf[40..44].copy_from_slice(&cfg.max_children.to_le_bytes());
    buf[44..48].copy_from_slice(&cfg.max_depth.to_le_bytes());
    buf[48..56].copy_from_slice(&cfg.num_vectors.to_le_bytes());
    buf[56..64].copy_from_slice(&cfg.num_clusters.to_le_bytes());
    buf[64..68].copy_from_slice(&cfg.variance_split_factor.to_bits().to_le_bytes());
    // 68..96 reserved
    buf
}

pub fn decode_fractal_config(data: &[u8]) -> FractalIndexConfig {
    let dim = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let num_subvectors = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let num_codewords = u16::from_le_bytes(data[8..10].try_into().unwrap());
    let metric = match data[10] {
        0 => DistanceMetric::Cosine,
        2 => DistanceMetric::DotProduct,
        3 => DistanceMetric::Manhattan,
        _ => DistanceMetric::EuclideanSq,
    };
    let store_raw_vectors = data[11] != 0;
    let default_nprobe = u32::from_le_bytes(data[12..16].try_into().unwrap());
    let state = data[16];
    let root_cluster_id = u32::from_le_bytes(data[20..24].try_into().unwrap());
    let next_cluster_id = u32::from_le_bytes(data[24..28].try_into().unwrap());
    let max_leaf_population = u32::from_le_bytes(data[28..32].try_into().unwrap());
    let min_leaf_population = u32::from_le_bytes(data[32..36].try_into().unwrap());
    let max_buffer_size = u32::from_le_bytes(data[36..40].try_into().unwrap());
    let max_children = u32::from_le_bytes(data[40..44].try_into().unwrap());
    let max_depth = u32::from_le_bytes(data[44..48].try_into().unwrap());
    let num_vectors = u64::from_le_bytes(data[48..56].try_into().unwrap());
    let num_clusters = u64::from_le_bytes(data[56..64].try_into().unwrap());
    let variance_split_factor =
        f32::from_bits(u32::from_le_bytes(data[64..68].try_into().unwrap()));

    FractalIndexConfig {
        dim,
        num_subvectors,
        num_codewords,
        metric,
        store_raw_vectors,
        default_nprobe,
        state,
        root_cluster_id,
        next_cluster_id,
        max_leaf_population,
        min_leaf_population,
        max_buffer_size,
        max_children,
        max_depth,
        num_vectors,
        num_clusters,
        variance_split_factor,
    }
}

// ---------------------------------------------------------------------------
// FractalSearchParams
// ---------------------------------------------------------------------------

/// Parameters for a single fractal index search query.
#[derive(Debug, Clone)]
pub struct FractalSearchParams {
    /// Number of children to probe at each internal level.
    pub nprobe: u32,
    /// Number of PQ-distance candidates to shortlist before re-ranking.
    pub candidates: usize,
    /// Number of final results to return.
    pub k: usize,
    /// Re-rank top candidates with exact distances from raw vectors.
    pub rerank: bool,
    /// Optional: only consider clusters with data newer than this HLC value.
    /// Set to 0 to disable temporal filtering.
    pub min_hlc: u64,
    /// Diversity-aware probe selection. Default: disabled (lambda=0.0).
    pub diversity: DiversityConfig,
}

impl FractalSearchParams {
    pub fn top_k(k: usize) -> Self {
        Self {
            nprobe: 8,
            candidates: k.saturating_mul(10).max(100),
            k,
            rerank: true,
            min_hlc: 0,
            diversity: DiversityConfig { lambda: 0.0 },
        }
    }

    /// Set the per-level probe count.
    #[must_use]
    pub const fn with_nprobe(mut self, nprobe: u32) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Only search clusters containing data newer than this HLC timestamp.
    #[must_use]
    pub const fn with_min_hlc(mut self, min_hlc: u64) -> Self {
        self.min_hlc = min_hlc;
        self
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
}

// ---------------------------------------------------------------------------
// FractalIndexDefinition -- user-facing index declaration
// ---------------------------------------------------------------------------

/// Definition for a fractal vector index.
///
/// Analogous to [`crate::TableDefinition`] -- a compile-time description of an
/// index that is passed to `open_fractal_index()` to create or open it.
pub struct FractalIndexDefinition {
    name: &'static str,
    dim: u32,
    num_subvectors: u32,
    metric: DistanceMetric,
    store_raw_vectors: bool,
    default_nprobe: u32,
    max_leaf_population: u32,
    min_leaf_population: u32,
    max_buffer_size: u32,
    max_children: u32,
    max_depth: u32,
    variance_split_factor: f32,
}

impl FractalIndexDefinition {
    /// Create a new fractal index definition.
    ///
    /// `dim` must be divisible by `num_subvectors`.
    pub const fn new(
        name: &'static str,
        dim: u32,
        num_subvectors: u32,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            name,
            dim,
            num_subvectors,
            metric,
            store_raw_vectors: false,
            default_nprobe: 8,
            max_leaf_population: 1000,
            min_leaf_population: 50,
            max_buffer_size: 128,
            max_children: 64,
            max_depth: 8,
            // f32 can't be used in const context easily, so we store the bits
            // and decode later. 2.0f32.to_bits() = 0x4000_0000
            variance_split_factor: 2.0,
        }
    }

    /// Enable storage of full-precision vectors for re-ranking.
    #[must_use]
    pub const fn with_raw_vectors(mut self) -> Self {
        self.store_raw_vectors = true;
        self
    }

    /// Set the default per-level probe count for search.
    #[must_use]
    pub const fn with_nprobe(mut self, nprobe: u32) -> Self {
        self.default_nprobe = nprobe;
        self
    }

    /// Set the maximum leaf cluster population before split.
    #[must_use]
    pub const fn with_max_leaf_population(mut self, max: u32) -> Self {
        self.max_leaf_population = max;
        self
    }

    /// Set the minimum leaf cluster population before merge.
    #[must_use]
    pub const fn with_min_leaf_population(mut self, min: u32) -> Self {
        self.min_leaf_population = min;
        self
    }

    /// Set the write buffer size per cluster (cascade threshold).
    #[must_use]
    pub const fn with_max_buffer_size(mut self, size: u32) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Returns the index name.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Convert to a full [`FractalIndexConfig`] with default state.
    pub fn to_config(&self) -> FractalIndexConfig {
        FractalIndexConfig {
            dim: self.dim,
            num_subvectors: self.num_subvectors,
            num_codewords: 256,
            metric: self.metric,
            store_raw_vectors: self.store_raw_vectors,
            default_nprobe: self.default_nprobe,
            state: STATE_NEW,
            root_cluster_id: 0,
            next_cluster_id: 0,
            max_leaf_population: self.max_leaf_population,
            min_leaf_population: self.min_leaf_population,
            max_buffer_size: self.max_buffer_size,
            max_children: self.max_children,
            max_depth: self.max_depth,
            num_vectors: 0,
            num_clusters: 0,
            variance_split_factor: self.variance_split_factor,
        }
    }
}

impl fmt::Debug for FractalIndexDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FractalIndexDefinition({:?}, dim={}, subvecs={}, {:?})",
            self.name, self.dim, self.num_subvectors, self.metric,
        )
    }
}
