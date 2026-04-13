use crate::types::{Key, TypeName, Value};
use core::cmp::Ordering;
use core::fmt;

// ---------------------------------------------------------------------------
// PostingKey -- composite key for the IVF posting list table
// ---------------------------------------------------------------------------

/// Composite key `(cluster_id, vector_id)` for the IVF posting list.
///
/// Serialized as **big-endian** so that B-tree byte ordering matches the
/// logical `(cluster_id, vector_id)` ordering. This makes range scans over a
/// single cluster a contiguous B-tree region.
///
/// Fixed width: 12 bytes (4 + 8).
#[allow(dead_code)] // Retained for potential migration tooling.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PostingKey {
    pub cluster_id: u32,
    pub vector_id: u64,
}

#[allow(dead_code)]
impl PostingKey {
    pub const SERIALIZED_SIZE: usize = 12;

    pub const fn new(cluster_id: u32, vector_id: u64) -> Self {
        Self {
            cluster_id,
            vector_id,
        }
    }

    /// Returns the first possible key for the given cluster (inclusive lower bound).
    pub const fn cluster_start(cluster_id: u32) -> Self {
        Self {
            cluster_id,
            vector_id: 0,
        }
    }

    /// Returns the last possible key for the given cluster (inclusive upper bound).
    pub const fn cluster_end(cluster_id: u32) -> Self {
        Self {
            cluster_id,
            vector_id: u64::MAX,
        }
    }

    #[allow(clippy::big_endian_bytes)]
    pub fn to_be_bytes(self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut buf = [0u8; Self::SERIALIZED_SIZE];
        buf[..4].copy_from_slice(&self.cluster_id.to_be_bytes());
        buf[4..12].copy_from_slice(&self.vector_id.to_be_bytes());
        buf
    }

    #[allow(clippy::big_endian_bytes)]
    pub fn from_be_bytes(data: &[u8]) -> Self {
        debug_assert!(
            data.len() >= Self::SERIALIZED_SIZE,
            "PostingKey::from_be_bytes: truncated data ({} < {})",
            data.len(),
            Self::SERIALIZED_SIZE,
        );
        if data.len() < Self::SERIALIZED_SIZE {
            return Self {
                cluster_id: 0,
                vector_id: 0,
            };
        }
        let cluster_id = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let vector_id = u64::from_be_bytes([
            data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11],
        ]);
        Self {
            cluster_id,
            vector_id,
        }
    }
}

impl PartialOrd for PostingKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PostingKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cluster_id
            .cmp(&other.cluster_id)
            .then(self.vector_id.cmp(&other.vector_id))
    }
}

impl fmt::Debug for PostingKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PostingKey(cluster={}, vec={})",
            self.cluster_id, self.vector_id
        )
    }
}

impl Value for PostingKey {
    type SelfType<'a>
        = PostingKey
    where
        Self: 'a;
    type AsBytes<'a>
        = [u8; PostingKey::SERIALIZED_SIZE]
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        Some(Self::SERIALIZED_SIZE)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        Self::from_be_bytes(data)
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        value.to_be_bytes()
    }

    fn type_name() -> TypeName {
        TypeName::internal("shodh_redb::ivfpq::PostingKey")
    }
}

impl Key for PostingKey {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        // Big-endian serialization means raw byte comparison is correct.
        let len = Self::SERIALIZED_SIZE.min(data1.len()).min(data2.len());
        data1[..len]
            .cmp(&data2[..len])
            .then(data1.len().cmp(&data2.len()))
    }
}

// ---------------------------------------------------------------------------
// AssignmentValue -- u32 cluster_id stored as a value in the assignments table
// ---------------------------------------------------------------------------

/// Thin wrapper so we can use `u32` cluster IDs as values in the assignments
/// table without conflicting with the built-in `u32` Value impl (which uses
/// little-endian). We also use little-endian here for consistency as a value.
pub struct AssignmentValue;

impl fmt::Debug for AssignmentValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AssignmentValue")
    }
}

impl Value for AssignmentValue {
    type SelfType<'a>
        = u32
    where
        Self: 'a;
    type AsBytes<'a>
        = [u8; 4]
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        Some(4)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        debug_assert!(
            data.len() >= 4,
            "AssignmentValue::from_bytes: truncated data ({} < 4)",
            data.len(),
        );
        if data.len() < 4 {
            return 0;
        }
        u32::from_le_bytes([data[0], data[1], data[2], data[3]])
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        value.to_le_bytes()
    }

    fn type_name() -> TypeName {
        TypeName::internal("shodh_redb::ivfpq::AssignmentValue")
    }
}

// ---------------------------------------------------------------------------
// IndexConfigValue -- fixed-width serialisation of IndexConfig for the meta table
// ---------------------------------------------------------------------------

/// Serialised index configuration stored in the metadata table.
///
/// Layout (48 bytes, all little-endian):
/// ```text
/// [dim:4][num_clusters:4][num_subvectors:4][num_codewords:2]
/// [metric:1][store_raw:1][default_nprobe:4][state:1][_pad:3]
/// [num_vectors:8][_reserved:16]
/// ```
pub const INDEX_CONFIG_SIZE: usize = 48;

/// Encode an [`super::config::IndexConfig`] into a fixed-width byte array.
pub fn encode_index_config(cfg: &super::config::IndexConfig) -> [u8; INDEX_CONFIG_SIZE] {
    let mut buf = [0u8; INDEX_CONFIG_SIZE];
    buf[0..4].copy_from_slice(&cfg.dim.to_le_bytes());
    buf[4..8].copy_from_slice(&cfg.num_clusters.to_le_bytes());
    buf[8..12].copy_from_slice(&cfg.num_subvectors.to_le_bytes());
    buf[12..14].copy_from_slice(&cfg.num_codewords.to_le_bytes());
    buf[14] = cfg.metric_byte();
    buf[15] = u8::from(cfg.store_raw_vectors);
    buf[16..20].copy_from_slice(&cfg.default_nprobe.to_le_bytes());
    buf[20] = cfg.state;
    // bytes 21..24 padding
    buf[24..32].copy_from_slice(&cfg.num_vectors.to_le_bytes());
    buf[32] = cfg.format_version;
    // bytes 33..48 reserved
    buf
}

fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    if offset + 2 > data.len() {
        return 0;
    }
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    if offset + 4 > data.len() {
        return 0;
    }
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    if offset + 8 > data.len() {
        return 0;
    }
    u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

fn read_byte(data: &[u8], offset: usize) -> u8 {
    if offset >= data.len() {
        return 0;
    }
    data[offset]
}

/// Decode an [`super::config::IndexConfig`] from a fixed-width byte slice.
pub fn decode_index_config(data: &[u8]) -> super::config::IndexConfig {
    use super::config::IndexConfig;
    use crate::vector_ops::DistanceMetric;

    let dim = read_u32_le(data, 0);
    let num_clusters = read_u32_le(data, 4);
    let num_subvectors = read_u32_le(data, 8);
    let num_codewords = read_u16_le(data, 12);
    let metric = match read_byte(data, 14) {
        0 => DistanceMetric::Cosine,
        2 => DistanceMetric::DotProduct,
        3 => DistanceMetric::Manhattan,
        _ => DistanceMetric::EuclideanSq,
    };
    let store_raw_vectors = read_byte(data, 15) != 0;
    let default_nprobe = read_u32_le(data, 16);
    let state = read_byte(data, 20);
    let num_vectors = read_u64_le(data, 24);
    let format_version = read_byte(data, 32);
    // bytes 33..48 reserved

    IndexConfig {
        dim,
        num_clusters,
        num_subvectors,
        num_codewords,
        metric,
        store_raw_vectors,
        default_nprobe,
        state,
        num_vectors,
        format_version,
    }
}
