//! Per-vector metadata storage and filtering for vector search indexes.
//!
//! Metadata is an opaque key-value map attached to each vector. At search time,
//! a [`MetadataFilter`] predicate can exclude candidates before they enter the
//! top-K heap, avoiding wasted re-ranking and distance computation.
//!
//! # Wire format (`no_std` compatible, no serde dependency)
//!
//! ```text
//! [version: u8][num_fields: u16 LE]
//!   [key_len: u16 LE][key: [u8; key_len]][val_type: u8][value: ...]
//!   ...
//! ```
//!
//! Value types:
//! - `0` = String: `[len: u16 LE][utf8 bytes]`
//! - `1` = U64: `[8 bytes LE]`
//! - `2` = F64: `[8 bytes LE]`
//! - `3` = Bool: `[1 byte: 0 or 1]`
//! - `4` = Bytes: `[len: u16 LE][raw bytes]`

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// Current wire format version.
const FORMAT_VERSION: u8 = 1;

// -------------------------------------------------------------------------
// MetadataValue
// -------------------------------------------------------------------------

/// A single metadata field value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    String(String),
    U64(u64),
    F64(f64),
    Bool(bool),
    Bytes(Vec<u8>),
}

impl MetadataValue {
    fn type_tag(&self) -> u8 {
        match self {
            Self::String(_) => 0,
            Self::U64(_) => 1,
            Self::F64(_) => 2,
            Self::Bool(_) => 3,
            Self::Bytes(_) => 4,
        }
    }

    #[allow(clippy::cast_possible_truncation)] // Wire format uses u16 lengths.
    fn encode_value(&self, out: &mut Vec<u8>) {
        match self {
            Self::String(s) => {
                let bytes = s.as_bytes();
                out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
                out.extend_from_slice(bytes);
            }
            Self::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
            Self::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
            Self::Bool(b) => out.push(u8::from(*b)),
            Self::Bytes(b) => {
                out.extend_from_slice(&(b.len() as u16).to_le_bytes());
                out.extend_from_slice(b);
            }
        }
    }

    fn decode_value(tag: u8, buf: &[u8], pos: &mut usize) -> Option<Self> {
        match tag {
            0 => {
                let len = read_u16(buf, pos)? as usize;
                let end = pos.checked_add(len)?;
                if end > buf.len() {
                    return None;
                }
                let s = core::str::from_utf8(&buf[*pos..end]).ok()?;
                *pos = end;
                Some(Self::String(String::from(s)))
            }
            1 => {
                let end = pos.checked_add(8)?;
                if end > buf.len() {
                    return None;
                }
                let v = u64::from_le_bytes(buf[*pos..end].try_into().ok()?);
                *pos = end;
                Some(Self::U64(v))
            }
            2 => {
                let end = pos.checked_add(8)?;
                if end > buf.len() {
                    return None;
                }
                let v = f64::from_le_bytes(buf[*pos..end].try_into().ok()?);
                *pos = end;
                Some(Self::F64(v))
            }
            3 => {
                if *pos >= buf.len() {
                    return None;
                }
                let v = buf[*pos] != 0;
                *pos += 1;
                Some(Self::Bool(v))
            }
            4 => {
                let len = read_u16(buf, pos)? as usize;
                let end = pos.checked_add(len)?;
                if end > buf.len() {
                    return None;
                }
                let b = buf[*pos..end].to_vec();
                *pos = end;
                Some(Self::Bytes(b))
            }
            _ => None,
        }
    }

    /// Compare numerically. Returns None if types are incompatible.
    #[allow(clippy::cast_precision_loss)] // Intentional: u64→f64 for cross-type comparison.
    fn partial_cmp_numeric(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match (self, other) {
            (Self::U64(a), Self::U64(b)) => Some(a.cmp(b)),
            (Self::F64(a), Self::F64(b)) => a.partial_cmp(b),
            (Self::U64(a), Self::F64(b)) => (*a as f64).partial_cmp(b),
            (Self::F64(a), Self::U64(b)) => a.partial_cmp(&(*b as f64)),
            (Self::String(a), Self::String(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }
}

// -------------------------------------------------------------------------
// MetadataMap -- a set of key-value fields
// -------------------------------------------------------------------------

/// A map of metadata fields for a single vector.
#[derive(Debug, Clone, Default)]
pub struct MetadataMap {
    fields: Vec<(String, MetadataValue)>,
}

impl MetadataMap {
    /// Create an empty metadata map.
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// Insert a field. If the key already exists, its value is replaced.
    pub fn insert(&mut self, key: impl Into<String>, value: MetadataValue) {
        let key = key.into();
        if let Some(entry) = self.fields.iter_mut().find(|(k, _)| k == &key) {
            entry.1 = value;
        } else {
            self.fields.push((key, value));
        }
    }

    /// Get a field by key.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Serialize to wire format.
    #[allow(clippy::cast_possible_truncation)] // Wire format uses u16 lengths.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(FORMAT_VERSION);
        out.extend_from_slice(&(self.fields.len() as u16).to_le_bytes());
        for (key, val) in &self.fields {
            let kb = key.as_bytes();
            out.extend_from_slice(&(kb.len() as u16).to_le_bytes());
            out.extend_from_slice(kb);
            out.push(val.type_tag());
            val.encode_value(&mut out);
        }
        out
    }

    /// Deserialize from wire format. Returns `None` on malformed input.
    pub fn decode(buf: &[u8]) -> Option<Self> {
        let mut pos = 0;
        if buf.is_empty() {
            return None;
        }
        let version = buf[pos];
        pos += 1;
        if version != FORMAT_VERSION {
            return None;
        }
        let num_fields = read_u16(buf, &mut pos)? as usize;
        let mut fields = Vec::with_capacity(num_fields);
        for _ in 0..num_fields {
            let key_len = read_u16(buf, &mut pos)? as usize;
            let key_end = pos.checked_add(key_len)?;
            if key_end > buf.len() {
                return None;
            }
            let key = core::str::from_utf8(&buf[pos..key_end]).ok()?;
            pos = key_end;
            if pos >= buf.len() {
                return None;
            }
            let tag = buf[pos];
            pos += 1;
            let val = MetadataValue::decode_value(tag, buf, &mut pos)?;
            fields.push((String::from(key), val));
        }
        Some(Self { fields })
    }
}

// -------------------------------------------------------------------------
// MetadataFilter -- predicate tree
// -------------------------------------------------------------------------

/// A composable filter predicate for per-vector metadata.
///
/// Evaluated during search to exclude candidates before they enter the top-K
/// heap. Supports equality, comparison, and boolean composition.
///
/// # Example
///
/// ```rust,ignore
/// use shodh_redb::ivfpq::metadata::{MetadataFilter, MetadataValue};
///
/// // category == "sports" AND price < 100.0
/// let filter = MetadataFilter::And(
///     Box::new(MetadataFilter::Eq("category".into(), MetadataValue::String("sports".into()))),
///     Box::new(MetadataFilter::Lt("price".into(), MetadataValue::F64(100.0))),
/// );
/// ```
#[derive(Debug, Clone)]
pub enum MetadataFilter {
    /// Field equals value.
    Eq(String, MetadataValue),
    /// Field does not equal value.
    Ne(String, MetadataValue),
    /// Field less than value (numeric/string comparison).
    Lt(String, MetadataValue),
    /// Field less than or equal to value.
    Le(String, MetadataValue),
    /// Field greater than value.
    Gt(String, MetadataValue),
    /// Field greater than or equal to value.
    Ge(String, MetadataValue),
    /// Both sub-filters must pass.
    And(Box<MetadataFilter>, Box<MetadataFilter>),
    /// At least one sub-filter must pass.
    Or(Box<MetadataFilter>, Box<MetadataFilter>),
}

impl MetadataFilter {
    /// Evaluate this filter against a metadata map.
    ///
    /// Returns `true` if the metadata passes the filter. Missing fields cause
    /// the filter to fail (fail-closed: if metadata is absent, the candidate is
    /// excluded).
    pub fn matches(&self, meta: &MetadataMap) -> bool {
        match self {
            Self::Eq(key, expected) => meta.get(key).is_some_and(|v| v == expected),
            Self::Ne(key, expected) => meta.get(key).is_some_and(|v| v != expected),
            Self::Lt(key, threshold) => meta
                .get(key)
                .and_then(|v| v.partial_cmp_numeric(threshold))
                .is_some_and(|ord| ord == core::cmp::Ordering::Less),
            Self::Le(key, threshold) => meta
                .get(key)
                .and_then(|v| v.partial_cmp_numeric(threshold))
                .is_some_and(|ord| ord != core::cmp::Ordering::Greater),
            Self::Gt(key, threshold) => meta
                .get(key)
                .and_then(|v| v.partial_cmp_numeric(threshold))
                .is_some_and(|ord| ord == core::cmp::Ordering::Greater),
            Self::Ge(key, threshold) => meta
                .get(key)
                .and_then(|v| v.partial_cmp_numeric(threshold))
                .is_some_and(|ord| ord != core::cmp::Ordering::Less),
            Self::And(a, b) => a.matches(meta) && b.matches(meta),
            Self::Or(a, b) => a.matches(meta) || b.matches(meta),
        }
    }
}

/// Evaluate a filter against raw metadata bytes from the metadata table.
///
/// Returns `false` (fail-closed) if the metadata cannot be decoded or is
/// absent. This is the function called in the hot search loop.
pub fn passes_filter(metadata_bytes: &[u8], filter: &MetadataFilter) -> bool {
    match MetadataMap::decode(metadata_bytes) {
        Some(map) => filter.matches(&map),
        None => false,
    }
}

/// Generate the metadata table name for an IVF-PQ index.
pub fn ivfpq_metadata_table_name(index_name: &str) -> String {
    alloc::format!("__ivfpq:{index_name}:vector_meta")
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

fn read_u16(buf: &[u8], pos: &mut usize) -> Option<u16> {
    let end = pos.checked_add(2)?;
    if end > buf.len() {
        return None;
    }
    let v = u16::from_le_bytes(buf[*pos..end].try_into().ok()?);
    *pos = end;
    Some(v)
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_map_roundtrip() {
        let mut map = MetadataMap::new();
        map.insert("category", MetadataValue::String("sports".into()));
        map.insert("price", MetadataValue::F64(29.99));
        map.insert("count", MetadataValue::U64(42));
        map.insert("featured", MetadataValue::Bool(true));
        map.insert("raw", MetadataValue::Bytes(vec![0xDE, 0xAD]));

        let encoded = map.encode();
        let decoded = MetadataMap::decode(&encoded).expect("decode should succeed");

        assert_eq!(
            decoded.get("category"),
            Some(&MetadataValue::String("sports".into()))
        );
        assert_eq!(decoded.get("price"), Some(&MetadataValue::F64(29.99)));
        assert_eq!(decoded.get("count"), Some(&MetadataValue::U64(42)));
        assert_eq!(decoded.get("featured"), Some(&MetadataValue::Bool(true)));
        assert_eq!(
            decoded.get("raw"),
            Some(&MetadataValue::Bytes(vec![0xDE, 0xAD]))
        );
    }

    #[test]
    fn metadata_map_replace_existing_key() {
        let mut map = MetadataMap::new();
        map.insert("x", MetadataValue::U64(1));
        map.insert("x", MetadataValue::U64(2));
        assert_eq!(map.get("x"), Some(&MetadataValue::U64(2)));
        let encoded = map.encode();
        let decoded = MetadataMap::decode(&encoded).unwrap();
        assert_eq!(decoded.get("x"), Some(&MetadataValue::U64(2)));
    }

    #[test]
    fn decode_empty_returns_none() {
        assert!(MetadataMap::decode(&[]).is_none());
    }

    #[test]
    fn decode_wrong_version_returns_none() {
        assert!(MetadataMap::decode(&[99, 0, 0]).is_none());
    }

    #[test]
    fn filter_eq() {
        let mut map = MetadataMap::new();
        map.insert("color", MetadataValue::String("red".into()));

        let filter = MetadataFilter::Eq("color".into(), MetadataValue::String("red".into()));
        assert!(filter.matches(&map));

        let filter = MetadataFilter::Eq("color".into(), MetadataValue::String("blue".into()));
        assert!(!filter.matches(&map));
    }

    #[test]
    fn filter_ne() {
        let mut map = MetadataMap::new();
        map.insert("status", MetadataValue::U64(1));

        let filter = MetadataFilter::Ne("status".into(), MetadataValue::U64(0));
        assert!(filter.matches(&map));

        let filter = MetadataFilter::Ne("status".into(), MetadataValue::U64(1));
        assert!(!filter.matches(&map));
    }

    #[test]
    fn filter_numeric_comparisons() {
        let mut map = MetadataMap::new();
        map.insert("price", MetadataValue::F64(50.0));

        assert!(MetadataFilter::Lt("price".into(), MetadataValue::F64(100.0)).matches(&map));
        assert!(!MetadataFilter::Lt("price".into(), MetadataValue::F64(50.0)).matches(&map));
        assert!(MetadataFilter::Le("price".into(), MetadataValue::F64(50.0)).matches(&map));
        assert!(MetadataFilter::Gt("price".into(), MetadataValue::F64(10.0)).matches(&map));
        assert!(!MetadataFilter::Gt("price".into(), MetadataValue::F64(50.0)).matches(&map));
        assert!(MetadataFilter::Ge("price".into(), MetadataValue::F64(50.0)).matches(&map));
    }

    #[test]
    fn filter_cross_type_numeric() {
        let mut map = MetadataMap::new();
        map.insert("count", MetadataValue::U64(100));

        // U64 field vs F64 threshold
        assert!(MetadataFilter::Lt("count".into(), MetadataValue::F64(100.5)).matches(&map));
        assert!(!MetadataFilter::Lt("count".into(), MetadataValue::F64(99.5)).matches(&map));
    }

    #[test]
    fn filter_missing_field_fails_closed() {
        let map = MetadataMap::new();

        let filter = MetadataFilter::Eq("missing".into(), MetadataValue::U64(0));
        assert!(!filter.matches(&map));

        let filter = MetadataFilter::Lt("missing".into(), MetadataValue::F64(100.0));
        assert!(!filter.matches(&map));
    }

    #[test]
    fn filter_and_or() {
        let mut map = MetadataMap::new();
        map.insert("category", MetadataValue::String("sports".into()));
        map.insert("price", MetadataValue::F64(29.99));

        let filter = MetadataFilter::And(
            Box::new(MetadataFilter::Eq(
                "category".into(),
                MetadataValue::String("sports".into()),
            )),
            Box::new(MetadataFilter::Lt(
                "price".into(),
                MetadataValue::F64(100.0),
            )),
        );
        assert!(filter.matches(&map));

        let filter = MetadataFilter::And(
            Box::new(MetadataFilter::Eq(
                "category".into(),
                MetadataValue::String("music".into()),
            )),
            Box::new(MetadataFilter::Lt(
                "price".into(),
                MetadataValue::F64(100.0),
            )),
        );
        assert!(!filter.matches(&map));

        let filter = MetadataFilter::Or(
            Box::new(MetadataFilter::Eq(
                "category".into(),
                MetadataValue::String("music".into()),
            )),
            Box::new(MetadataFilter::Lt(
                "price".into(),
                MetadataValue::F64(100.0),
            )),
        );
        assert!(filter.matches(&map));
    }

    #[test]
    fn passes_filter_on_raw_bytes() {
        let mut map = MetadataMap::new();
        map.insert("tag", MetadataValue::String("ok".into()));
        let encoded = map.encode();

        let filter = MetadataFilter::Eq("tag".into(), MetadataValue::String("ok".into()));
        assert!(passes_filter(&encoded, &filter));

        let filter = MetadataFilter::Eq("tag".into(), MetadataValue::String("bad".into()));
        assert!(!passes_filter(&encoded, &filter));
    }

    #[test]
    fn passes_filter_corrupted_bytes_fail_closed() {
        let filter = MetadataFilter::Eq("x".into(), MetadataValue::U64(0));
        assert!(!passes_filter(&[0xFF, 0xFF], &filter));
        assert!(!passes_filter(&[], &filter));
    }

    #[test]
    fn filter_bool_equality() {
        let mut map = MetadataMap::new();
        map.insert("active", MetadataValue::Bool(true));

        assert!(MetadataFilter::Eq("active".into(), MetadataValue::Bool(true)).matches(&map));
        assert!(!MetadataFilter::Eq("active".into(), MetadataValue::Bool(false)).matches(&map));
    }

    #[test]
    fn table_name_helpers() {
        assert_eq!(
            ivfpq_metadata_table_name("embeddings"),
            "__ivfpq:embeddings:vector_meta"
        );
    }
}
