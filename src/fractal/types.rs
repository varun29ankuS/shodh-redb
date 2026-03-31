use crate::types::{Key, TypeName, Value};
use core::cmp::Ordering;
use core::fmt;

use super::config::{FLAG_IS_ROOT, NO_PARENT};

// ---------------------------------------------------------------------------
// ClusterMeta -- per-cluster metadata (128 bytes fixed)
// ---------------------------------------------------------------------------

/// Metadata for a single cluster node in the fractal tree.
///
/// Layout (128 bytes, all little-endian):
/// ```text
/// [cluster_id:4][parent_id:4][level:1][flags:1][num_children:2]
/// [population:4][buffer_count:4][_pad:4]
/// [sum_variance:8(f64)][oldest_hlc:8][newest_hlc:8]
/// [oldest_wall_ns:8][newest_wall_ns:8][_reserved:64]
/// ```
pub const CLUSTER_META_SIZE: usize = 128;

#[derive(Clone)]
pub struct ClusterMeta {
    data: [u8; CLUSTER_META_SIZE],
}

#[allow(dead_code)]
impl ClusterMeta {
    pub fn new(cluster_id: u32, parent_id: u32, level: u8, is_root: bool) -> Self {
        let mut data = [0u8; CLUSTER_META_SIZE];
        data[0..4].copy_from_slice(&cluster_id.to_le_bytes());
        data[4..8].copy_from_slice(&parent_id.to_le_bytes());
        data[8] = level;
        data[9] = if is_root { FLAG_IS_ROOT } else { 0 };
        Self { data }
    }

    pub fn cluster_id(&self) -> u32 {
        u32::from_le_bytes(self.data[0..4].try_into().unwrap())
    }

    pub fn parent_id(&self) -> u32 {
        u32::from_le_bytes(self.data[4..8].try_into().unwrap())
    }

    pub fn level(&self) -> u8 {
        self.data[8]
    }

    pub fn set_level(&mut self, level: u8) {
        self.data[8] = level;
    }

    pub fn flags(&self) -> u8 {
        self.data[9]
    }

    pub fn is_root(&self) -> bool {
        self.data[9] & FLAG_IS_ROOT != 0
    }

    pub fn is_leaf(&self) -> bool {
        self.level() == 0
    }

    pub fn num_children(&self) -> u16 {
        u16::from_le_bytes(self.data[10..12].try_into().unwrap())
    }

    pub fn set_num_children(&mut self, n: u16) {
        self.data[10..12].copy_from_slice(&n.to_le_bytes());
    }

    pub fn population(&self) -> u32 {
        u32::from_le_bytes(self.data[12..16].try_into().unwrap())
    }

    pub fn set_population(&mut self, n: u32) {
        self.data[12..16].copy_from_slice(&n.to_le_bytes());
    }

    pub fn buffer_count(&self) -> u32 {
        u32::from_le_bytes(self.data[16..20].try_into().unwrap())
    }

    pub fn set_buffer_count(&mut self, n: u32) {
        self.data[16..20].copy_from_slice(&n.to_le_bytes());
    }

    pub fn sum_variance(&self) -> f64 {
        f64::from_le_bytes(self.data[24..32].try_into().unwrap())
    }

    pub fn set_sum_variance(&mut self, v: f64) {
        self.data[24..32].copy_from_slice(&v.to_le_bytes());
    }

    pub fn oldest_hlc(&self) -> u64 {
        u64::from_le_bytes(self.data[32..40].try_into().unwrap())
    }

    pub fn set_oldest_hlc(&mut self, v: u64) {
        self.data[32..40].copy_from_slice(&v.to_le_bytes());
    }

    pub fn newest_hlc(&self) -> u64 {
        u64::from_le_bytes(self.data[40..48].try_into().unwrap())
    }

    pub fn set_newest_hlc(&mut self, v: u64) {
        self.data[40..48].copy_from_slice(&v.to_le_bytes());
    }

    pub fn oldest_wall_ns(&self) -> u64 {
        u64::from_le_bytes(self.data[48..56].try_into().unwrap())
    }

    pub fn set_oldest_wall_ns(&mut self, v: u64) {
        self.data[48..56].copy_from_slice(&v.to_le_bytes());
    }

    pub fn newest_wall_ns(&self) -> u64 {
        u64::from_le_bytes(self.data[56..64].try_into().unwrap())
    }

    pub fn set_newest_wall_ns(&mut self, v: u64) {
        self.data[56..64].copy_from_slice(&v.to_le_bytes());
    }

    pub fn set_parent_id(&mut self, parent_id: u32) {
        self.data[4..8].copy_from_slice(&parent_id.to_le_bytes());
    }

    pub fn set_flags(&mut self, flags: u8) {
        self.data[9] = flags;
    }

    /// Check if this cluster has no parent (is the root).
    pub fn has_no_parent(&self) -> bool {
        self.parent_id() == NO_PARENT
    }

    /// Returns the raw byte representation.
    pub fn as_bytes(&self) -> &[u8; CLUSTER_META_SIZE] {
        &self.data
    }

    /// Construct from raw bytes.
    ///
    /// If `data` is shorter than `CLUSTER_META_SIZE`, missing bytes are zero-filled.
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut buf = [0u8; CLUSTER_META_SIZE];
        let copy_len = data.len().min(CLUSTER_META_SIZE);
        buf[..copy_len].copy_from_slice(&data[..copy_len]);
        Self { data: buf }
    }
}

impl fmt::Debug for ClusterMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClusterMeta")
            .field("id", &self.cluster_id())
            .field("parent", &self.parent_id())
            .field("level", &self.level())
            .field("children", &self.num_children())
            .field("pop", &self.population())
            .field("buf", &self.buffer_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// HierarchyKey -- composite key for parent-child cluster relationships
// ---------------------------------------------------------------------------

/// Composite key `(parent_id, child_id)` for the cluster hierarchy table.
///
/// Serialized as **big-endian** so that range scans over a parent's children
/// are contiguous in the B-tree.
///
/// Fixed width: 8 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct HierarchyKey {
    pub parent_id: u32,
    pub child_id: u32,
}

impl HierarchyKey {
    pub const SERIALIZED_SIZE: usize = 8;

    pub const fn new(parent_id: u32, child_id: u32) -> Self {
        Self {
            parent_id,
            child_id,
        }
    }

    /// First possible key for the given parent (inclusive lower bound).
    pub const fn children_start(parent_id: u32) -> Self {
        Self {
            parent_id,
            child_id: 0,
        }
    }

    /// Last possible key for the given parent (inclusive upper bound).
    pub const fn children_end(parent_id: u32) -> Self {
        Self {
            parent_id,
            child_id: u32::MAX,
        }
    }

    #[allow(clippy::big_endian_bytes)]
    pub fn to_be_bytes(self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut buf = [0u8; Self::SERIALIZED_SIZE];
        buf[..4].copy_from_slice(&self.parent_id.to_be_bytes());
        buf[4..8].copy_from_slice(&self.child_id.to_be_bytes());
        buf
    }

    #[allow(clippy::big_endian_bytes)]
    pub fn from_be_bytes(data: &[u8]) -> Self {
        debug_assert!(
            data.len() >= Self::SERIALIZED_SIZE,
            "HierarchyKey::from_be_bytes: truncated data ({} < {})",
            data.len(),
            Self::SERIALIZED_SIZE,
        );
        if data.len() < Self::SERIALIZED_SIZE {
            return Self {
                parent_id: 0,
                child_id: 0,
            };
        }
        let parent_id = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let child_id = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        Self {
            parent_id,
            child_id,
        }
    }
}

impl PartialOrd for HierarchyKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HierarchyKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.parent_id
            .cmp(&other.parent_id)
            .then(self.child_id.cmp(&other.child_id))
    }
}

impl fmt::Debug for HierarchyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HierarchyKey(parent={}, child={})",
            self.parent_id, self.child_id
        )
    }
}

impl Value for HierarchyKey {
    type SelfType<'a>
        = HierarchyKey
    where
        Self: 'a;
    type AsBytes<'a>
        = [u8; HierarchyKey::SERIALIZED_SIZE]
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
        TypeName::internal("shodh_redb::fractal::HierarchyKey")
    }
}

impl Key for HierarchyKey {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        // Big-endian serialization means raw byte comparison is correct.
        let len = Self::SERIALIZED_SIZE.min(data1.len()).min(data2.len());
        data1[..len]
            .cmp(&data2[..len])
            .then(data1.len().cmp(&data2.len()))
    }
}

// ---------------------------------------------------------------------------
// ClusterMetaValue -- Value impl for ClusterMeta
// ---------------------------------------------------------------------------

/// Marker type for `ClusterMeta` as a B-tree value.
#[derive(Debug)]
pub struct ClusterMetaValue;

impl Value for ClusterMetaValue {
    type SelfType<'a>
        = ClusterMeta
    where
        Self: 'a;
    type AsBytes<'a>
        = [u8; CLUSTER_META_SIZE]
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        Some(CLUSTER_META_SIZE)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        ClusterMeta::from_bytes(data)
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        *value.as_bytes()
    }

    fn type_name() -> TypeName {
        TypeName::internal("shodh_redb::fractal::ClusterMeta")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_meta_roundtrip() {
        let mut meta = ClusterMeta::new(42, NO_PARENT, 0, true);
        meta.set_population(500);
        meta.set_buffer_count(64);
        meta.set_sum_variance(1.5);
        meta.set_oldest_hlc(100);
        meta.set_newest_hlc(200);
        meta.set_oldest_wall_ns(1000);
        meta.set_newest_wall_ns(2000);
        meta.set_num_children(3);

        let bytes = meta.as_bytes();
        let restored = ClusterMeta::from_bytes(bytes);

        assert_eq!(restored.cluster_id(), 42);
        assert_eq!(restored.parent_id(), NO_PARENT);
        assert_eq!(restored.level(), 0);
        assert!(restored.is_root());
        assert!(restored.is_leaf());
        assert_eq!(restored.num_children(), 3);
        assert_eq!(restored.population(), 500);
        assert_eq!(restored.buffer_count(), 64);
        assert!((restored.sum_variance() - 1.5).abs() < f64::EPSILON);
        assert_eq!(restored.oldest_hlc(), 100);
        assert_eq!(restored.newest_hlc(), 200);
        assert_eq!(restored.oldest_wall_ns(), 1000);
        assert_eq!(restored.newest_wall_ns(), 2000);
    }

    #[test]
    fn hierarchy_key_ordering() {
        let a = HierarchyKey::new(1, 10);
        let b = HierarchyKey::new(1, 20);
        let c = HierarchyKey::new(2, 5);

        assert!(a < b);
        assert!(b < c);

        // Byte comparison must match logical comparison
        let ab = a.to_be_bytes();
        let bb = b.to_be_bytes();
        let cb = c.to_be_bytes();
        assert!(ab < bb);
        assert!(bb < cb);
    }

    #[test]
    fn hierarchy_key_roundtrip() {
        let key = HierarchyKey::new(0xDEAD_BEEF, 0xCAFE_BABE);
        let bytes = key.to_be_bytes();
        let restored = HierarchyKey::from_be_bytes(&bytes);
        assert_eq!(restored.parent_id, key.parent_id);
        assert_eq!(restored.child_id, key.child_id);
    }
}
