/// SOC Compliance Test Suite
///
/// Comprehensive tests covering type system, merge operators, multimap,
/// transactions, distance function properties, HLC, builder config,
/// blob store types, TTL tables, causal edges, and more.
use shodh_redb::{
    BlobId, Builder, CausalLink, CdcConfig, ChangeOp, ContentType, Database, DatabaseStats,
    Durability, MultimapTableDefinition, MultimapTableHandle, ReadVerification, ReadableDatabase,
    ReadableMultimapTable, ReadableTable, ReadableTableMetadata, RelationType, StoreOptions,
    TableDefinition, TableHandle, TtlTableDefinition, VerifyLevel,
};
use std::sync::{Arc, Barrier};
use std::time::Duration;

fn tmpfile() -> tempfile::NamedTempFile {
    if cfg!(target_os = "wasi") {
        tempfile::NamedTempFile::new_in("/tmp").unwrap()
    } else {
        tempfile::NamedTempFile::new().unwrap()
    }
}

// ===========================================================================
// S1 TYPE SYSTEM -- u8
// ===========================================================================

#[test]
fn u8_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u8, u8> = TableDefinition::new("u8rt");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&0u8, &0u8).unwrap();
        t.insert(&255u8, &255u8).unwrap();
        t.insert(&128u8, &128u8).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&0u8).unwrap().unwrap().value(), 0u8);
    assert_eq!(t.get(&255u8).unwrap().unwrap().value(), 255u8);
}

#[test]
fn u8_key_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u8, u8> = TableDefinition::new("u8ord");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&255u8, &1u8).unwrap();
        t.insert(&0u8, &2u8).unwrap();
        t.insert(&128u8, &3u8).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<u8> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![0, 128, 255]);
}

// ===========================================================================
// S1 TYPE SYSTEM -- u16
// ===========================================================================

#[test]
fn u16_roundtrip_and_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u16, u16> = TableDefinition::new("u16");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&0u16, &0u16).unwrap();
        t.insert(&u16::MAX, &u16::MAX).unwrap();
        t.insert(&32768u16, &32768u16).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<u16> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![0, 32768, u16::MAX]);
}

// ===========================================================================
// S1 TYPE SYSTEM -- u32
// ===========================================================================

#[test]
fn u32_roundtrip_and_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u32, u32> = TableDefinition::new("u32");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&0u32, &0u32).unwrap();
        t.insert(&u32::MAX, &u32::MAX).unwrap();
        t.insert(&1u32, &1u32).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<u32> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![0, 1, u32::MAX]);
}

// ===========================================================================
// S1 TYPE SYSTEM -- u128
// ===========================================================================

#[test]
fn u128_roundtrip_and_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u128, u128> = TableDefinition::new("u128");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&0u128, &0u128).unwrap();
        t.insert(&u128::MAX, &u128::MAX).unwrap();
        t.insert(&(u128::MAX / 2), &42u128).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&u128::MAX).unwrap().unwrap().value(), u128::MAX);
    let keys: Vec<u128> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![0, u128::MAX / 2, u128::MAX]);
}

// ===========================================================================
// S1 TYPE SYSTEM -- i8
// ===========================================================================

#[test]
fn i8_roundtrip_and_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<i8, i8> = TableDefinition::new("i8");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&i8::MIN, &i8::MIN).unwrap();
        t.insert(&0i8, &0i8).unwrap();
        t.insert(&i8::MAX, &i8::MAX).unwrap();
        t.insert(&(-1i8), &(-1i8)).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<i8> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![i8::MIN, -1, 0, i8::MAX]);
}

// ===========================================================================
// S1 TYPE SYSTEM -- i16
// ===========================================================================

#[test]
fn i16_roundtrip_and_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<i16, i16> = TableDefinition::new("i16");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&i16::MIN, &i16::MIN).unwrap();
        t.insert(&0i16, &0i16).unwrap();
        t.insert(&i16::MAX, &i16::MAX).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<i16> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![i16::MIN, 0, i16::MAX]);
}

// ===========================================================================
// S1 TYPE SYSTEM -- i32
// ===========================================================================

#[test]
fn i32_roundtrip_and_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<i32, i32> = TableDefinition::new("i32");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&i32::MIN, &i32::MIN).unwrap();
        t.insert(&0i32, &0i32).unwrap();
        t.insert(&i32::MAX, &i32::MAX).unwrap();
        t.insert(&(-1i32), &(-1i32)).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<i32> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![i32::MIN, -1, 0, i32::MAX]);
}

// ===========================================================================
// S1 TYPE SYSTEM -- i128
// ===========================================================================

#[test]
fn i128_roundtrip_and_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<i128, i128> = TableDefinition::new("i128");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&i128::MIN, &i128::MIN).unwrap();
        t.insert(&0i128, &0i128).unwrap();
        t.insert(&i128::MAX, &i128::MAX).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<i128> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![i128::MIN, 0, i128::MAX]);
}

// ===========================================================================
// S1 TYPE SYSTEM -- f32
// ===========================================================================

#[test]
fn f32_value_roundtrip_specials() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, f32> = TableDefinition::new("f32sp");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &0.0f32).unwrap();
        t.insert(&2u64, &(-0.0f32)).unwrap();
        t.insert(&3u64, &f32::INFINITY).unwrap();
        t.insert(&4u64, &f32::NEG_INFINITY).unwrap();
        t.insert(&5u64, &f32::NAN).unwrap();
        t.insert(&6u64, &f32::MIN_POSITIVE).unwrap();
        t.insert(&7u64, &f32::EPSILON).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 0.0f32);
    assert!(t.get(&3u64).unwrap().unwrap().value().is_infinite());
    assert!(t.get(&5u64).unwrap().unwrap().value().is_nan());
    assert_eq!(t.get(&6u64).unwrap().unwrap().value(), f32::MIN_POSITIVE);
}

// ===========================================================================
// S1 TYPE SYSTEM -- f64
// ===========================================================================

#[test]
fn f64_value_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, f64> = TableDefinition::new("f64rt");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &0.0f64).unwrap();
        t.insert(&2u64, &f64::MAX).unwrap();
        t.insert(&3u64, &f64::MIN).unwrap();
        t.insert(&4u64, &f64::INFINITY).unwrap();
        t.insert(&5u64, &f64::NAN).unwrap();
        t.insert(&6u64, &core::f64::consts::PI).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&2u64).unwrap().unwrap().value(), f64::MAX);
    assert!(t.get(&5u64).unwrap().unwrap().value().is_nan());
    assert!((t.get(&6u64).unwrap().unwrap().value() - core::f64::consts::PI).abs() < f64::EPSILON);
}

// ===========================================================================
// S1 TYPE SYSTEM -- &str edge cases
// ===========================================================================

#[test]
fn str_empty_key_and_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, &str> = TableDefinition::new("str_empty");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("", "").unwrap();
        t.insert("a", "b").unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("").unwrap().unwrap().value(), "");
    assert_eq!(t.get("a").unwrap().unwrap().value(), "b");
}

#[test]
fn str_key_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("str_ord");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("zebra", &1u64).unwrap();
        t.insert("apple", &2u64).unwrap();
        t.insert("mango", &3u64).unwrap();
        t.insert("", &4u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<String> = t
        .iter()
        .unwrap()
        .map(|e| e.unwrap().0.value().to_string())
        .collect();
    assert_eq!(keys, vec!["", "apple", "mango", "zebra"]);
}

#[test]
fn str_long_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, &str> = TableDefinition::new("str_long");
    let long = "x".repeat(10_000);
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, long.as_str()).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), long.as_str());
}

// ===========================================================================
// S1 TYPE SYSTEM -- unit type
// ===========================================================================

#[test]
fn unit_type_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, ()> = TableDefinition::new("unit");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &()).unwrap();
        t.insert(&2u64, &()).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 2);
}

// ===========================================================================
// S1 TYPE SYSTEM -- tuples
// ===========================================================================

#[test]
fn tuple_u32_u32_key_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<(u32, u32), u64> = TableDefinition::new("tup_u32");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&(2u32, 1u32), &1u64).unwrap();
        t.insert(&(1u32, 100u32), &2u64).unwrap();
        t.insert(&(1u32, 1u32), &3u64).unwrap();
        t.insert(&(0u32, 0u32), &4u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<(u32, u32)> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![(0, 0), (1, 1), (1, 100), (2, 1)]);
}

#[test]
fn tuple_u64_bool_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<(u64, bool), &str> = TableDefinition::new("tup_bool");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&(1u64, true), "yes").unwrap();
        t.insert(&(1u64, false), "no").unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&(1u64, true)).unwrap().unwrap().value(), "yes");
    assert_eq!(t.get(&(1u64, false)).unwrap().unwrap().value(), "no");
}

// ===========================================================================
// S1 TYPE SYSTEM -- Option<T> extended
// ===========================================================================

#[test]
fn option_u8_none_some_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, Option<u8>> = TableDefinition::new("opt_u8");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &None).unwrap();
        t.insert(&2u64, &Some(42u8)).unwrap();
        t.insert(&3u64, &Some(0u8)).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), None);
    assert_eq!(t.get(&2u64).unwrap().unwrap().value(), Some(42));
    assert_eq!(t.get(&3u64).unwrap().unwrap().value(), Some(0));
}

#[test]
fn option_key_none_sorts_first() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<Option<u64>, u64> = TableDefinition::new("opt_key");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&Some(5u64), &1u64).unwrap();
        t.insert(&None, &2u64).unwrap();
        t.insert(&Some(1u64), &3u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<Option<u64>> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![None, Some(1), Some(5)]);
}

// ===========================================================================
// S1 TYPE SYSTEM -- bool
// ===========================================================================

#[test]
fn bool_key_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<bool, u64> = TableDefinition::new("bool_key");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&true, &1u64).unwrap();
        t.insert(&false, &2u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<bool> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![false, true]);
}

// ===========================================================================
// S1 TYPE SYSTEM -- [T; N] extended
// ===========================================================================

#[test]
fn array_0_element() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, [u32; 0]> = TableDefinition::new("arr0");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &[]).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), [] as [u32; 0]);
}

#[test]
fn array_1_element() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, [u64; 1]> = TableDefinition::new("arr1");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &[u64::MAX]).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), [u64::MAX]);
}

#[test]
fn array_3_element_key_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<[u8; 3], u64> = TableDefinition::new("arr3key");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&[2u8, 0, 0], &1u64).unwrap();
        t.insert(&[1u8, 0, 0], &2u64).unwrap();
        t.insert(&[1u8, 1, 0], &3u64).unwrap();
        t.insert(&[0u8, 0, 0], &4u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<[u8; 3]> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]]);
}

// ===========================================================================
// S2 MERGE OPERATORS
// ===========================================================================

#[test]
fn merge_numeric_add_u64() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_add");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &10u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &5u64.to_le_bytes(), &shodh_redb::NumericAdd)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 15);
}

#[test]
fn merge_numeric_add_new_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_add_new");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &42u64.to_le_bytes(), &shodh_redb::NumericAdd)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 42);
}

#[test]
fn merge_numeric_max() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_max");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &50u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &30u64.to_le_bytes(), &shodh_redb::NumericMax)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 50);
}

#[test]
fn merge_numeric_max_replaces_when_larger() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_max2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &30u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &50u64.to_le_bytes(), &shodh_redb::NumericMax)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 50);
}

#[test]
fn merge_numeric_min() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_min");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &50u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &30u64.to_le_bytes(), &shodh_redb::NumericMin)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 30);
}

#[test]
fn merge_numeric_min_keeps_when_smaller() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_min2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &10u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &30u64.to_le_bytes(), &shodh_redb::NumericMin)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 10);
}

#[test]
fn merge_sequential_adds() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_seq");
    for i in 0..10u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.merge("k", &i.to_le_bytes(), &shodh_redb::NumericAdd)
                .unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    // 0+1+2+...+9 = 45
    assert_eq!(t.get("k").unwrap().unwrap().value(), 45);
}

// ===========================================================================
// S3 MULTIMAP OPERATIONS
// ===========================================================================

#[test]
fn multimap_insert_multiple_values() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_multi");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &20u64).unwrap();
        t.insert(&1u64, &30u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let vals: Vec<u64> = t.get(&1u64).unwrap().map(|v| v.unwrap().value()).collect();
    assert_eq!(vals.len(), 3);
}

#[test]
fn multimap_remove_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_rm");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &20u64).unwrap();
        t.insert(&1u64, &30u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.remove(&1u64, &20u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let vals: Vec<u64> = t.get(&1u64).unwrap().map(|v| v.unwrap().value()).collect();
    assert_eq!(vals.len(), 2);
}

#[test]
fn multimap_empty_key_returns_empty() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_empty");
    let w = db.begin_write().unwrap();
    {
        w.open_multimap_table(T).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let vals: Vec<u64> = t.get(&1u64).unwrap().map(|v| v.unwrap().value()).collect();
    assert!(vals.is_empty());
}

#[test]
fn multimap_multiple_keys() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, &str> = MultimapTableDefinition::new("mm_keys");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, "a").unwrap();
        t.insert(&1u64, "b").unwrap();
        t.insert(&2u64, "c").unwrap();
        t.insert(&2u64, "d").unwrap();
        t.insert(&2u64, "e").unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let v1: Vec<String> = t
        .get(&1u64)
        .unwrap()
        .map(|v| v.unwrap().value().to_string())
        .collect();
    let v2: Vec<String> = t
        .get(&2u64)
        .unwrap()
        .map(|v| v.unwrap().value().to_string())
        .collect();
    assert_eq!(v1.len(), 2);
    assert_eq!(v2.len(), 3);
}

#[test]
fn multimap_range_query() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_range");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        for k in 0..10u64 {
            t.insert(&k, &(k * 10)).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let count = t.range(3u64..7u64).unwrap().count();
    assert_eq!(count, 4);
}

// ===========================================================================
// S4 TRANSACTION SEMANTICS
// ===========================================================================

#[test]
fn txn_rollback_on_drop() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("rollback");
    // Commit initial data
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    w.commit().unwrap();
    // Start write txn but drop without commit
    {
        let w = db.begin_write().unwrap();
        let mut t = w.open_table(T).unwrap();
        t.insert(&2u64, &200u64).unwrap();
        t.insert(&1u64, &999u64).unwrap();
        drop(t);
        // w is dropped here without commit
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 100); // not 999
    assert!(t.get(&2u64).unwrap().is_none()); // not inserted
}

#[test]
fn txn_read_after_write_same_txn() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("raw");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
        // Read within same write transaction
        assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 100);
    }
    w.commit().unwrap();
}

#[test]
fn txn_delete_and_reinsert() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("del_reins");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    w.commit().unwrap();

    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.remove(&1u64).unwrap();
        assert!(t.get(&1u64).unwrap().is_none());
        t.insert(&1u64, &200u64).unwrap();
        assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 200);
    }
    w.commit().unwrap();

    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 200);
}

#[test]
fn txn_empty_commit() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    w.commit().unwrap();
    // Should not panic
}

#[test]
fn txn_multi_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T1: TableDefinition<u64, u64> = TableDefinition::new("t1");
    const T2: TableDefinition<u64, &str> = TableDefinition::new("t2");
    let w = db.begin_write().unwrap();
    {
        let mut t1 = w.open_table(T1).unwrap();
        let mut t2 = w.open_table(T2).unwrap();
        t1.insert(&1u64, &100u64).unwrap();
        t2.insert(&1u64, "hello").unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t1 = r.open_table(T1).unwrap();
    let t2 = r.open_table(T2).unwrap();
    assert_eq!(t1.get(&1u64).unwrap().unwrap().value(), 100);
    assert_eq!(t2.get(&1u64).unwrap().unwrap().value(), "hello");
}

#[test]
fn txn_sequential_commits() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("seq");
    for i in 0..20u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.insert(&i, &(i * i)).unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 20);
    assert_eq!(t.get(&19u64).unwrap().unwrap().value(), 361);
}

#[test]
fn txn_durability_none() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("dur_none");
    let mut w = db.begin_write().unwrap();
    w.set_durability(Durability::None).unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 1);
}

#[test]
fn txn_durability_immediate() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("dur_imm");
    let mut w = db.begin_write().unwrap();
    w.set_durability(Durability::Immediate).unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 1);
}

#[test]
fn txn_table_stats() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stats");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 100);
}

#[test]
fn txn_overwrite_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("overwrite");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
        t.insert(&1u64, &200u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 200);
    assert_eq!(t.len().unwrap(), 1);
}

// ===========================================================================
// S5 DISTANCE FUNCTION PROPERTIES
// ===========================================================================

#[test]
fn euclidean_symmetry() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    assert_eq!(
        shodh_redb::euclidean_distance_sq(&a, &b),
        shodh_redb::euclidean_distance_sq(&b, &a)
    );
}

#[test]
fn manhattan_symmetry() {
    let a = [1.0f32, -2.0, 3.0];
    let b = [-4.0f32, 5.0, -6.0];
    assert_eq!(
        shodh_redb::manhattan_distance(&a, &b),
        shodh_redb::manhattan_distance(&b, &a)
    );
}

#[test]
fn cosine_symmetry() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    assert_eq!(
        shodh_redb::cosine_similarity(&a, &b),
        shodh_redb::cosine_similarity(&b, &a)
    );
}

#[test]
fn dot_product_symmetry() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    assert_eq!(
        shodh_redb::dot_product(&a, &b),
        shodh_redb::dot_product(&b, &a)
    );
}

#[test]
fn hamming_symmetry() {
    let a = [0xAAu8, 0x55];
    let b = [0x55u8, 0xAA];
    assert_eq!(
        shodh_redb::hamming_distance(&a, &b),
        shodh_redb::hamming_distance(&b, &a)
    );
}

#[test]
fn euclidean_identity() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(shodh_redb::euclidean_distance_sq(&a, &a), 0.0);
}

#[test]
fn manhattan_identity() {
    let a = [1.0f32, -2.0, 3.0];
    assert_eq!(shodh_redb::manhattan_distance(&a, &a), 0.0);
}

#[test]
fn cosine_identity() {
    let a = [1.0f32, 2.0, 3.0];
    let sim = shodh_redb::cosine_similarity(&a, &a);
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn dot_product_orthogonal_is_zero() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert_eq!(shodh_redb::dot_product(&a, &b), 0.0);
}

#[test]
fn euclidean_triangle_inequality() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 0.0];
    let c = [0.0f32, 4.0];
    let ab = shodh_redb::euclidean_distance_sq(&a, &b).sqrt();
    let bc = shodh_redb::euclidean_distance_sq(&b, &c).sqrt();
    let ac = shodh_redb::euclidean_distance_sq(&a, &c).sqrt();
    assert!(ac <= ab + bc + 1e-6);
}

#[test]
fn manhattan_triangle_inequality() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 0.0];
    let c = [0.0f32, 4.0];
    let ab = shodh_redb::manhattan_distance(&a, &b);
    let bc = shodh_redb::manhattan_distance(&b, &c);
    let ac = shodh_redb::manhattan_distance(&a, &c);
    assert!(ac <= ab + bc + 1e-6);
}

#[test]
fn cosine_normalized_vectors() {
    let mut a = [3.0f32, 4.0];
    shodh_redb::l2_normalize(&mut a);
    let mut b = [1.0f32, 0.0];
    shodh_redb::l2_normalize(&mut b);
    let sim = shodh_redb::cosine_similarity(&a, &b);
    // cos(angle) between (3,4) and (1,0) = 3/5 = 0.6
    assert!((sim - 0.6).abs() < 1e-5);
}

#[test]
fn dot_product_parallel_vectors() {
    let a = [2.0f32, 0.0];
    let b = [3.0f32, 0.0];
    assert_eq!(shodh_redb::dot_product(&a, &b), 6.0);
}

#[test]
fn manhattan_vs_euclidean_relationship() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    let l1 = shodh_redb::manhattan_distance(&a, &b);
    let l2 = shodh_redb::euclidean_distance_sq(&a, &b).sqrt();
    // L1 >= L2 always
    assert!(l1 >= l2 - 1e-6);
}

#[test]
fn hamming_all_zeros() {
    let a = [0x00u8; 16];
    let b = [0x00u8; 16];
    assert_eq!(shodh_redb::hamming_distance(&a, &b), 0);
}

#[test]
fn hamming_all_ones() {
    let a = [0xFFu8; 16];
    let b = [0xFFu8; 16];
    assert_eq!(shodh_redb::hamming_distance(&a, &b), 0);
}

#[test]
fn hamming_alternating_bits() {
    let a = [0xAAu8]; // 10101010
    let b = [0x55u8]; // 01010101
    assert_eq!(shodh_redb::hamming_distance(&a, &b), 8); // all bits differ
}

#[test]
fn distance_large_dimension() {
    let a: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
    let b: Vec<f32> = (0..384).map(|i| 1.0 - i as f32 / 384.0).collect();
    // Should not panic
    let _ = shodh_redb::euclidean_distance_sq(&a, &b);
    let _ = shodh_redb::cosine_similarity(&a, &b);
    let _ = shodh_redb::manhattan_distance(&a, &b);
    let _ = shodh_redb::dot_product(&a, &b);
}

#[test]
fn distance_single_dimension() {
    let a = [5.0f32];
    let b = [3.0f32];
    assert_eq!(shodh_redb::euclidean_distance_sq(&a, &b), 4.0);
    assert_eq!(shodh_redb::manhattan_distance(&a, &b), 2.0);
    assert_eq!(shodh_redb::dot_product(&a, &b), 15.0);
}

#[test]
fn distance_negative_values() {
    let a = [-1.0f32, -2.0, -3.0];
    let b = [-4.0f32, -5.0, -6.0];
    let d = shodh_redb::euclidean_distance_sq(&a, &b);
    assert!(d > 0.0);
    assert_eq!(d, 27.0); // (3^2 + 3^2 + 3^2) = 27
}

#[test]
fn cosine_opposite_vectors() {
    let a = [1.0f32, 0.0];
    let b = [-1.0f32, 0.0];
    let sim = shodh_redb::cosine_similarity(&a, &b);
    assert!((sim - (-1.0)).abs() < 1e-6);
}

#[test]
fn distance_metric_enum_compute() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    let d1 = shodh_redb::DistanceMetric::EuclideanSq.compute(&a, &b);
    let d2 = shodh_redb::euclidean_distance_sq(&a, &b);
    assert_eq!(d1, d2);
}

#[test]
fn distance_metric_cosine_compute() {
    let a = [1.0f32, 0.0];
    let b = [0.0f32, 1.0];
    let d = shodh_redb::DistanceMetric::Cosine.compute(&a, &b);
    assert!((d - 1.0).abs() < 1e-6); // cosine distance = 1 - cos(90) = 1
}

#[test]
fn distance_metric_manhattan_compute() {
    let a = [1.0f32, 2.0];
    let b = [4.0f32, 6.0];
    let d = shodh_redb::DistanceMetric::Manhattan.compute(&a, &b);
    assert_eq!(d, 7.0);
}

#[test]
fn distance_metric_dot_compute() {
    let a = [1.0f32, 2.0];
    let b = [3.0f32, 4.0];
    let d = shodh_redb::DistanceMetric::DotProduct.compute(&a, &b);
    // dot product distance = negative dot product for max-heap usage
    let dp = shodh_redb::dot_product(&a, &b);
    assert_eq!(d, -dp);
}

// ===========================================================================
// S6 QUANTIZATION EXTENDED
// ===========================================================================

#[test]
fn quantize_binary_all_positive() {
    let v = [1.0f32, 0.5, 0.001, 100.0, f32::MIN_POSITIVE];
    let bq = shodh_redb::quantize_binary(&v);
    // MSB-first packing: 5 positive values -> bits 7,6,5,4,3 set = 0xF8
    assert_eq!(bq[0] & 0xF8, 0xF8);
}

#[test]
fn quantize_binary_all_negative() {
    let v = [-1.0f32, -0.5, -0.001, -100.0];
    let bq = shodh_redb::quantize_binary(&v);
    // MSB-first packing: 4 negative values -> bits 7,6,5,4 all zero = upper nibble 0
    assert_eq!(bq[0] & 0xF0, 0x00);
}

#[test]
fn quantize_scalar_128_dims() {
    let v: [f32; 128] = core::array::from_fn(|i| i as f32 / 128.0);
    let sq = shodh_redb::quantize_scalar(&v);
    let dq = sq.dequantize();
    for i in 0..128 {
        assert!(
            (dq[i] - v[i]).abs() < 0.01,
            "dim {i}: expected {}, got {}",
            v[i],
            dq[i]
        );
    }
}

#[test]
fn quantize_scalar_negative_range() {
    let v = [-10.0f32, -5.0, 0.0, 5.0, 10.0];
    let sq = shodh_redb::quantize_scalar(&v);
    assert_eq!(sq.min_val, -10.0);
    assert_eq!(sq.max_val, 10.0);
    let dq = sq.dequantize();
    for i in 0..5 {
        assert!((dq[i] - v[i]).abs() < 0.1);
    }
}

#[test]
fn dequantize_scalar_roundtrip() {
    let v: [f32; 16] = core::array::from_fn(|i| (i as f32 - 8.0) * 0.5);
    let sq = shodh_redb::quantize_scalar(&v);
    let dq = shodh_redb::dequantize_scalar(&sq);
    for i in 0..16 {
        assert!((dq[i] - v[i]).abs() < 0.05);
    }
}

#[test]
fn l2_normalized_returns_unit_vector() {
    let v = vec![3.0f32, 4.0];
    let n = shodh_redb::l2_normalized(&v);
    let norm = shodh_redb::l2_norm(&n);
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn l2_norm_known_value() {
    let v = [3.0f32, 4.0];
    assert!((shodh_redb::l2_norm(&v) - 5.0).abs() < 1e-6);
}

// ===========================================================================
// S7 HLC (Hybrid Logical Clock)
// ===========================================================================

#[test]
fn hlc_now_monotonic() {
    let a = shodh_redb::HybridLogicalClock::now();
    let b = shodh_redb::HybridLogicalClock::now();
    assert!(b >= a);
}

#[test]
fn hlc_raw_roundtrip() {
    let hlc = shodh_redb::HybridLogicalClock::now();
    let raw = hlc.to_raw();
    let recovered = shodh_redb::HybridLogicalClock::from_raw(raw);
    assert_eq!(recovered.to_raw(), raw);
}

#[test]
fn hlc_physical_extraction() {
    let hlc = shodh_redb::HybridLogicalClock::from_wall_ns(1_000_000_000);
    let phys = hlc.physical_ms();
    assert_eq!(phys, 1000); // 1 billion ns = 1000 ms
}

#[test]
fn hlc_logical_extraction() {
    let hlc = shodh_redb::HybridLogicalClock::from_parts(42, 7);
    assert_eq!(hlc.physical_ms(), 42);
    assert_eq!(hlc.logical(), 7);
}

#[test]
fn hlc_tick_increments_logical() {
    let hlc = shodh_redb::HybridLogicalClock::from_parts(100, 0);
    let ticked = hlc.tick();
    assert_eq!(ticked.logical(), 1);
    assert_eq!(ticked.physical_ms(), 100);
}

#[test]
fn hlc_merge_takes_greater() {
    let local = shodh_redb::HybridLogicalClock::from_parts(100, 5);
    let remote = shodh_redb::HybridLogicalClock::from_parts(200, 3);
    let merged = local.merge(remote);
    assert!(merged >= local);
    assert!(merged >= remote);
}

#[test]
fn hlc_merge_advances_past_both() {
    let a = shodh_redb::HybridLogicalClock::now();
    let b = a.tick();
    let merged = a.merge(b);
    // Merged must be >= both inputs
    assert!(merged >= a);
    assert!(merged >= b);
}

#[test]
fn hlc_from_wall_ns_zero() {
    let hlc = shodh_redb::HybridLogicalClock::from_wall_ns(0);
    assert_eq!(hlc.physical_ms(), 0);
    assert_eq!(hlc.logical(), 0);
}

#[test]
fn hlc_from_parts_roundtrip() {
    let hlc = shodh_redb::HybridLogicalClock::from_parts(12345, 67);
    assert_eq!(hlc.physical_ms(), 12345);
    assert_eq!(hlc.logical(), 67);
}

#[test]
fn hlc_ordering() {
    let a = shodh_redb::HybridLogicalClock::from_parts(100, 0);
    let b = shodh_redb::HybridLogicalClock::from_parts(200, 0);
    assert!(a < b);
}

#[test]
fn hlc_ordering_tiebreak_on_logical() {
    let a = shodh_redb::HybridLogicalClock::from_parts(100, 5);
    let b = shodh_redb::HybridLogicalClock::from_parts(100, 10);
    assert!(a < b);
}

// ===========================================================================
// S8 BUILDER CONFIGURATION
// ===========================================================================

#[test]
fn builder_default_creates_db() {
    let f = tmpfile();
    let db = shodh_redb::Builder::new().create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("builder");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 1);
}

#[test]
fn builder_cache_size() {
    let f = tmpfile();
    let db = shodh_redb::Builder::new()
        .set_cache_size(1024 * 1024) // 1MB
        .create(f.path())
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("cache");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn builder_cache_size_large() {
    let f = tmpfile();
    let db = shodh_redb::Builder::new()
        .set_cache_size(4 * 1024 * 1024) // 4MB
        .create(f.path())
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("cache_lg");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
}

// ===========================================================================
// S9 RANGE QUERIES EXTENDED
// ===========================================================================

#[test]
fn range_inclusive_exclusive() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_ie");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    // Exclusive end: 3..7 should give [3,4,5,6]
    let vals: Vec<u64> = t
        .range(3u64..7u64)
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(vals, vec![3, 4, 5, 6]);
}

#[test]
fn range_inclusive_inclusive() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_ii");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let vals: Vec<u64> = t
        .range(3u64..=7u64)
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(vals, vec![3, 4, 5, 6, 7]);
}

#[test]
fn range_reverse_order() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_rev");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..5u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let vals: Vec<u64> = t
        .range(0u64..5u64)
        .unwrap()
        .rev()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(vals, vec![4, 3, 2, 1, 0]);
}

#[test]
fn range_full_iteration() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_full");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..50u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.iter().unwrap().count(), 50);
}

#[test]
fn range_empty_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_empty");
    let w = db.begin_write().unwrap();
    {
        w.open_table(T).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.iter().unwrap().count(), 0);
}

#[test]
fn range_single_element() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_single");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&42u64, &42u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let vals: Vec<u64> = t
        .range(42u64..=42u64)
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(vals, vec![42]);
}

// ===========================================================================
// S10 STRESS TESTS
// ===========================================================================

#[test]
fn stress_1000_entries() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stress1k");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..1000u64 {
            t.insert(&i, &(i * i)).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 1000);
    for i in (0..1000u64).step_by(100) {
        assert_eq!(t.get(&i).unwrap().unwrap().value(), i * i);
    }
}

#[test]
fn stress_insert_delete_cycle() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stress_cycle");
    // Insert 500
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..500u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    // Delete odd, insert 500-999
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in (1..500u64).step_by(2) {
            t.remove(&i).unwrap();
        }
        for i in 500..1000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    // Even 0-498 (250) + all 500-999 (500) = 750
    assert_eq!(t.len().unwrap(), 750);
}

#[test]
fn stress_many_small_transactions() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stress_txn");
    for i in 0..100u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.insert(&i, &i).unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 100);
}

// ===========================================================================
// S11 NEAREST-K EXTENDED
// ===========================================================================

#[test]
fn nearest_k_k_zero_returns_empty() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![(1, vec![1.0])];
    let query = [0.0f32];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 0, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert!(results.is_empty());
}

#[test]
fn nearest_k_all_same_distance() {
    let vectors: Vec<(u64, Vec<f32>)> = (0..10).map(|i| (i as u64, vec![1.0, 0.0])).collect();
    let query = [1.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 5, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert_eq!(results.len(), 5);
    for r in &results {
        assert_eq!(r.distance, 0.0);
    }
}

#[test]
fn nearest_k_with_manhattan() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![0.0, 0.0]),
        (2, vec![1.0, 1.0]),
        (3, vec![10.0, 10.0]),
    ];
    let query = [0.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 2, |a, b| {
        shodh_redb::manhattan_distance(a, b)
    });
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].key, 1);
    assert_eq!(results[1].key, 2);
}

#[test]
fn nearest_k_with_cosine() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![1.0, 0.0]),
        (2, vec![0.7, 0.7]),
        (3, vec![0.0, 1.0]),
    ];
    let query = [1.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 2, |a, b| {
        shodh_redb::cosine_distance(a, b)
    });
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].key, 1); // identical direction
}

#[test]
fn nearest_k_fixed_k_zero() {
    let vectors: Vec<(u64, [f32; 2])> = vec![(1, [1.0, 0.0])];
    let query = [0.0f32, 0.0];
    let results = shodh_redb::nearest_k_fixed(vectors.into_iter(), &query, 0, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert!(results.is_empty());
}

// ===========================================================================
// S12 WRITE/READ F32 EXTENDED
// ===========================================================================

#[test]
fn write_f32_le_exact_capacity() {
    let values = [1.0f32, 2.0, 3.0];
    let mut buf = vec![0u8; 12];
    shodh_redb::write_f32_le(&mut buf, &values);
    let recovered = shodh_redb::read_f32_le(&buf);
    assert_eq!(recovered, values);
}

#[test]
fn write_f32_le_special_values() {
    let values = [f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
    let mut buf = vec![0u8; 16];
    shodh_redb::write_f32_le(&mut buf, &values);
    let recovered = shodh_redb::read_f32_le(&buf);
    assert!(recovered[0].is_infinite() && recovered[0] > 0.0);
    assert!(recovered[1].is_infinite() && recovered[1] < 0.0);
}

#[test]
fn read_f32_le_partial_bytes() {
    // 5 bytes = 1 complete f32 + 1 leftover byte
    let buf = [0u8; 5];
    let result = shodh_redb::read_f32_le(&buf);
    assert_eq!(result.len(), 1);
}

// ===========================================================================
// S13 SAVEPOINT
// ===========================================================================

#[test]
fn savepoint_create_and_restore() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("savepoint");

    // Insert initial data
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    w.commit().unwrap();

    // Create savepoint, then modify and commit
    let w = db.begin_write().unwrap();
    let sp = w.ephemeral_savepoint().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&2u64, &200u64).unwrap();
    }
    w.commit().unwrap();

    // Restore savepoint in a new transaction
    let mut w = db.begin_write().unwrap();
    w.restore_savepoint(&sp).unwrap();
    w.commit().unwrap();

    // The insert of key 2 should be undone
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 100);
    assert!(t.get(&2u64).unwrap().is_none());
}

// ===========================================================================
// S14 IN-MEMORY BACKEND
// ===========================================================================

#[test]
fn in_memory_backend_basic() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("mem");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 1);
}

#[test]
fn in_memory_backend_multiple_tables() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T1: TableDefinition<u64, &str> = TableDefinition::new("mem1");
    const T2: TableDefinition<&str, u64> = TableDefinition::new("mem2");
    let w = db.begin_write().unwrap();
    {
        let mut t1 = w.open_table(T1).unwrap();
        let mut t2 = w.open_table(T2).unwrap();
        t1.insert(&1u64, "hello").unwrap();
        t2.insert("world", &42u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert_eq!(
        r.open_table(T1)
            .unwrap()
            .get(&1u64)
            .unwrap()
            .unwrap()
            .value(),
        "hello"
    );
    assert_eq!(
        r.open_table(T2)
            .unwrap()
            .get("world")
            .unwrap()
            .unwrap()
            .value(),
        42
    );
}

#[test]
fn in_memory_backend_stress() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("mem_stress");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..500u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 500);
}

// ===========================================================================
// S15 DATABASE OPERATIONS
// ===========================================================================

#[test]
fn db_list_tables() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T1: TableDefinition<u64, u64> = TableDefinition::new("tbl_a");
    const T2: TableDefinition<u64, u64> = TableDefinition::new("tbl_b");
    let w = db.begin_write().unwrap();
    {
        w.open_table(T1).unwrap();
        w.open_table(T2).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let tables: Vec<String> = r
        .list_tables()
        .unwrap()
        .map(|h| h.name().to_string())
        .collect();
    assert!(tables.contains(&"tbl_a".to_string()));
    assert!(tables.contains(&"tbl_b".to_string()));
}

#[test]
fn db_reopen_persists_data() {
    let f = tmpfile();
    {
        let db = Database::create(f.path()).unwrap();
        const T: TableDefinition<u64, u64> = TableDefinition::new("persist");
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.insert(&1u64, &42u64).unwrap();
        }
        w.commit().unwrap();
    }
    {
        let db = Database::open(f.path()).unwrap();
        const T: TableDefinition<u64, u64> = TableDefinition::new("persist");
        let r = db.begin_read().unwrap();
        let t = r.open_table(T).unwrap();
        assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 42);
    }
}

#[test]
fn db_delete_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("to_delete");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();

    let w = db.begin_write().unwrap();
    assert!(w.delete_table(T).unwrap());
    w.commit().unwrap();

    let r = db.begin_read().unwrap();
    assert!(r.open_table(T).is_err());
}

// ===========================================================================
// S16 LARGE VALUE TYPES
// ===========================================================================

#[test]
fn large_blob_value_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, &[u8]> = TableDefinition::new("blob_large");
    let blob = vec![0xABu8; 100_000];
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, blob.as_slice()).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value().len(), 100_000);
}

#[test]
fn empty_blob_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, &[u8]> = TableDefinition::new("blob_empty");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &[] as &[u8]).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert!(t.get(&1u64).unwrap().unwrap().value().is_empty());
}

#[test]
fn blob_key_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&[u8], u64> = TableDefinition::new("blob_key");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&[1u8, 2, 3] as &[u8], &42u64).unwrap();
        t.insert(&[1u8, 2] as &[u8], &10u64).unwrap();
        t.insert(&[] as &[u8], &0u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&[1u8, 2, 3] as &[u8]).unwrap().unwrap().value(), 42);
    assert_eq!(t.len().unwrap(), 3);
}

// ===========================================================================
// S17 TUPLE TYPES EXTENDED
// ===========================================================================

#[test]
fn tuple_3_element_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<(u32, u32, u32), u64> = TableDefinition::new("tup3");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&(1u32, 2u32, 3u32), &100u64).unwrap();
        t.insert(&(0u32, 0u32, 0u32), &0u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&(1u32, 2u32, 3u32)).unwrap().unwrap().value(), 100);
}

#[test]
fn tuple_mixed_types_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<(u64, &str), u64> = TableDefinition::new("tup_mix");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&(1u64, "a"), &10u64).unwrap();
        t.insert(&(1u64, "b"), &20u64).unwrap();
        t.insert(&(2u64, "a"), &30u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 3);
    assert_eq!(t.get(&(1u64, "b")).unwrap().unwrap().value(), 20);
}

#[test]
fn tuple_key_ordering() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<(u32, u32), u64> = TableDefinition::new("tup_ord");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&(2u32, 1u32), &1u64).unwrap();
        t.insert(&(1u32, 2u32), &2u64).unwrap();
        t.insert(&(1u32, 1u32), &3u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<(u32, u32)> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![(1, 1), (1, 2), (2, 1)]);
}

// ===========================================================================
// S18 BOUNDARY VALUES
// ===========================================================================

#[test]
fn u64_max_min_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("u64_bounds");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&0u64, &u64::MAX).unwrap();
        t.insert(&u64::MAX, &0u64).unwrap();
        t.insert(&(u64::MAX / 2), &(u64::MAX / 2)).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&0u64).unwrap().unwrap().value(), u64::MAX);
    assert_eq!(t.get(&u64::MAX).unwrap().unwrap().value(), 0);
}

#[test]
fn i64_extremes() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<i64, i64> = TableDefinition::new("i64_ext");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&i64::MIN, &i64::MAX).unwrap();
        t.insert(&i64::MAX, &i64::MIN).unwrap();
        t.insert(&0i64, &0i64).unwrap();
        t.insert(&(-1i64), &1i64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<i64> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    assert_eq!(keys, vec![i64::MIN, -1, 0, i64::MAX]);
}

#[test]
fn f32_subnormals() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, f32> = TableDefinition::new("f32_sub");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &f32::MIN_POSITIVE).unwrap();
        t.insert(&2u64, &(-f32::MIN_POSITIVE)).unwrap();
        t.insert(&3u64, &f32::EPSILON).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), f32::MIN_POSITIVE);
    assert_eq!(t.get(&3u64).unwrap().unwrap().value(), f32::EPSILON);
}

// ===========================================================================
// S19 CONCURRENT READ TRANSACTIONS
// ===========================================================================

#[test]
fn multiple_read_transactions() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("multi_read");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    w.commit().unwrap();
    let r1 = db.begin_read().unwrap();
    let r2 = db.begin_read().unwrap();
    let t1 = r1.open_table(T).unwrap();
    let t2 = r2.open_table(T).unwrap();
    assert_eq!(t1.get(&1u64).unwrap().unwrap().value(), 100);
    assert_eq!(t2.get(&1u64).unwrap().unwrap().value(), 100);
}

#[test]
fn read_snapshot_isolation() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("snapshot");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    w.commit().unwrap();
    // Take read snapshot
    let r = db.begin_read().unwrap();
    // Write new data after snapshot
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &200u64).unwrap();
    }
    w.commit().unwrap();
    // Snapshot should still see old value
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 100);
}

// ===========================================================================
// S20 DELETE OPERATIONS EXTENDED
// ===========================================================================

#[test]
fn delete_nonexistent_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("del_nokey");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
        let removed = t.remove(&999u64).unwrap();
        assert!(removed.is_none());
    }
    w.commit().unwrap();
}

#[test]
fn delete_returns_old_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("del_ret");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &42u64).unwrap();
        let old = t.remove(&1u64).unwrap();
        assert_eq!(old.unwrap().value(), 42);
    }
    w.commit().unwrap();
}

#[test]
fn delete_all_entries() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("del_all");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..50u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..50u64 {
            t.remove(&i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 0);
}

// ===========================================================================
// S21 MULTIMAP EXTENDED
// ===========================================================================

#[test]
fn multimap_duplicate_insert_is_idempotent() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_dup");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &10u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let vals: Vec<u64> = t.get(&1u64).unwrap().map(|v| v.unwrap().value()).collect();
    assert_eq!(vals.len(), 1);
}

#[test]
fn multimap_remove_all_for_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_rmall");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &20u64).unwrap();
        t.insert(&1u64, &30u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.remove_all(&1u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let vals: Vec<u64> = t.get(&1u64).unwrap().map(|v| v.unwrap().value()).collect();
    assert!(vals.is_empty());
}

#[test]
fn multimap_len() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_len");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &20u64).unwrap();
        t.insert(&2u64, &30u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 3);
}

#[test]
fn multimap_iter_all() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_iter");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        for k in 0..5u64 {
            for v in 0..3u64 {
                t.insert(&k, &(k * 10 + v)).unwrap();
            }
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let count = t.iter().unwrap().count();
    assert_eq!(count, 5);
}

// ===========================================================================
// S22 NEAREST-K ADVANCED SCENARIOS
// ===========================================================================

#[test]
fn nearest_k_more_than_available() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])];
    let query = [0.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 10, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert_eq!(results.len(), 2);
}

#[test]
fn nearest_k_single_element() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![(42, vec![5.0, 5.0])];
    let query = [0.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 1, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].key, 42);
    assert_eq!(results[0].distance, 50.0);
}

#[test]
fn nearest_k_empty_input() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![];
    let query = [0.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 5, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert!(results.is_empty());
}

#[test]
fn nearest_k_large_k_small_data() {
    let vectors: Vec<(u64, Vec<f32>)> = (0..3).map(|i| (i as u64, vec![i as f32])).collect();
    let query = [1.5f32];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 100, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert_eq!(results.len(), 3);
}

#[test]
fn nearest_k_fixed_correctness() {
    let vectors: Vec<(u64, [f32; 3])> = vec![
        (1, [0.0, 0.0, 0.0]),
        (2, [1.0, 0.0, 0.0]),
        (3, [0.0, 1.0, 0.0]),
        (4, [10.0, 10.0, 10.0]),
    ];
    let query = [0.0f32, 0.0, 0.0];
    let results = shodh_redb::nearest_k_fixed(vectors.into_iter(), &query, 2, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].key, 1);
    assert_eq!(results[0].distance, 0.0);
}

#[test]
fn nearest_k_fixed_single_input() {
    let vectors: Vec<(u64, [f32; 2])> = vec![(99, [3.0, 4.0])];
    let query = [0.0f32, 0.0];
    let results = shodh_redb::nearest_k_fixed(vectors.into_iter(), &query, 5, |a, b| {
        shodh_redb::euclidean_distance_sq(a, b)
    });
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].distance, 25.0);
}

// ===========================================================================
// S23 DISTANCE FUNCTION EDGE CASES
// ===========================================================================

#[test]
fn euclidean_zero_vectors() {
    let a = [0.0f32; 10];
    let b = [0.0f32; 10];
    assert_eq!(shodh_redb::euclidean_distance_sq(&a, &b), 0.0);
}

#[test]
fn manhattan_zero_vectors() {
    let a = [0.0f32; 10];
    let b = [0.0f32; 10];
    assert_eq!(shodh_redb::manhattan_distance(&a, &b), 0.0);
}

#[test]
fn dot_product_zero_vectors() {
    let a = [0.0f32; 10];
    let b = [0.0f32; 10];
    assert_eq!(shodh_redb::dot_product(&a, &b), 0.0);
}

#[test]
fn euclidean_unit_vectors() {
    let mut a = [0.0f32; 3];
    a[0] = 1.0;
    let mut b = [0.0f32; 3];
    b[1] = 1.0;
    assert_eq!(shodh_redb::euclidean_distance_sq(&a, &b), 2.0);
}

#[test]
fn manhattan_unit_vectors() {
    let mut a = [0.0f32; 3];
    a[0] = 1.0;
    let mut b = [0.0f32; 3];
    b[1] = 1.0;
    assert_eq!(shodh_redb::manhattan_distance(&a, &b), 2.0);
}

#[test]
fn cosine_zero_magnitude() {
    let a = [0.0f32, 0.0];
    let b = [1.0f32, 0.0];
    let sim = shodh_redb::cosine_similarity(&a, &b);
    assert!(sim.is_nan() || sim == 0.0);
}

#[test]
fn hamming_single_bit_diff() {
    let a = [0b0000_0001u8];
    let b = [0b0000_0000u8];
    assert_eq!(shodh_redb::hamming_distance(&a, &b), 1);
}

#[test]
fn hamming_large_vectors() {
    let a = vec![0xFFu8; 1024];
    let b = vec![0x00u8; 1024];
    assert_eq!(shodh_redb::hamming_distance(&a, &b), 8192);
}

#[test]
fn l2_normalize_zero_vector() {
    let mut v = [0.0f32; 4];
    shodh_redb::l2_normalize(&mut v);
    for x in &v {
        assert!(x.is_nan() || *x == 0.0);
    }
}

#[test]
fn l2_normalize_already_unit() {
    let mut v = [1.0f32, 0.0, 0.0];
    shodh_redb::l2_normalize(&mut v);
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert_eq!(v[1], 0.0);
    assert_eq!(v[2], 0.0);
}

#[test]
fn l2_norm_single_element() {
    assert_eq!(shodh_redb::l2_norm(&[7.0f32]), 7.0);
}

#[test]
fn l2_norm_negative() {
    assert!((shodh_redb::l2_norm(&[-3.0f32, -4.0]) - 5.0).abs() < 1e-6);
}

// ===========================================================================
// S24 QUANTIZATION EDGE CASES
// ===========================================================================

#[test]
fn quantize_scalar_single_value() {
    let v = [42.0f32];
    let sq = shodh_redb::quantize_scalar(&v);
    let dq = sq.dequantize();
    assert!((dq[0] - 42.0).abs() < 0.1);
}

#[test]
fn quantize_scalar_identical_values() {
    let v = [5.0f32; 16];
    let sq = shodh_redb::quantize_scalar(&v);
    assert_eq!(sq.min_val, 5.0);
    assert_eq!(sq.max_val, 5.0);
}

#[test]
fn quantize_binary_mixed() {
    let v = [1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let bq = shodh_redb::quantize_binary(&v);
    // Positive at indices 0,2,4,6 -> bits 7,5,3,1 set
    // MSB-first: 0b10101010 = 0xAA
    assert_eq!(bq[0], 0xAA);
}

#[test]
fn quantize_binary_16_dims() {
    let v = [1.0f32; 16];
    let bq = shodh_redb::quantize_binary(&v);
    assert_eq!(bq.len(), 2);
    assert_eq!(bq[0], 0xFF);
    assert_eq!(bq[1], 0xFF);
}

#[test]
fn quantize_binary_9_dims() {
    let v = [1.0f32; 9];
    let bq = shodh_redb::quantize_binary(&v);
    assert_eq!(bq.len(), 2);
    assert_eq!(bq[0], 0xFF);
    // bit 7 of byte 1 set (index 8)
    assert_eq!(bq[1] & 0x80, 0x80);
}

// ===========================================================================
// S25 WRITE/READ F32 EDGE CASES
// ===========================================================================

#[test]
fn write_f32_le_empty() {
    let values: [f32; 0] = [];
    let mut buf = vec![];
    shodh_redb::write_f32_le(&mut buf, &values);
    assert!(buf.is_empty());
}

#[test]
fn write_f32_le_nan_roundtrip() {
    let values = [f32::NAN];
    let mut buf = vec![0u8; 4];
    shodh_redb::write_f32_le(&mut buf, &values);
    let recovered = shodh_redb::read_f32_le(&buf);
    assert!(recovered[0].is_nan());
}

#[test]
fn read_f32_le_empty() {
    let buf: [u8; 0] = [];
    let result = shodh_redb::read_f32_le(&buf);
    assert!(result.is_empty());
}

#[test]
fn write_f32_le_large_array() {
    let values: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let mut buf = vec![0u8; 4000];
    shodh_redb::write_f32_le(&mut buf, &values);
    let recovered = shodh_redb::read_f32_le(&buf);
    assert_eq!(recovered.len(), 1000);
    assert_eq!(recovered[999], 999.0);
}

// ===========================================================================
// S26 HLC EXTENDED
// ===========================================================================

#[test]
fn hlc_tick_multiple() {
    let mut hlc = shodh_redb::HybridLogicalClock::from_parts(100, 0);
    for i in 1..=10u16 {
        hlc = hlc.tick();
        assert_eq!(hlc.logical(), i);
    }
}

#[test]
fn hlc_from_raw_zero() {
    let hlc = shodh_redb::HybridLogicalClock::from_raw(0);
    assert_eq!(hlc.physical_ms(), 0);
    assert_eq!(hlc.logical(), 0);
}

#[test]
fn hlc_from_raw_max() {
    let hlc = shodh_redb::HybridLogicalClock::from_raw(u64::MAX);
    assert!(hlc.physical_ms() > 0);
    assert!(hlc.logical() > 0);
}

#[test]
fn hlc_now_has_nonzero_physical() {
    let hlc = shodh_redb::HybridLogicalClock::now();
    assert!(hlc.physical_ms() > 0);
}

#[test]
fn hlc_equality() {
    let a = shodh_redb::HybridLogicalClock::from_parts(100, 5);
    let b = shodh_redb::HybridLogicalClock::from_parts(100, 5);
    assert_eq!(a, b);
}

// ===========================================================================
// S27 RANGE QUERIES ADVANCED
// ===========================================================================

#[test]
fn range_boundary_only_start() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_start");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let vals: Vec<u64> = t
        .range(8u64..)
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(vals, vec![8, 9]);
}

#[test]
fn range_boundary_only_end() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_end");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let vals: Vec<u64> = t
        .range(..3u64)
        .unwrap()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(vals, vec![0, 1, 2]);
}

#[test]
fn range_no_match() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_nomatch");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
        t.insert(&10u64, &10u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let count = t.range(5u64..8u64).unwrap().count();
    assert_eq!(count, 0);
}

#[test]
fn range_str_keys() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("range_str");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("apple", &1u64).unwrap();
        t.insert("banana", &2u64).unwrap();
        t.insert("cherry", &3u64).unwrap();
        t.insert("date", &4u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let vals: Vec<String> = t
        .range("banana".."date")
        .unwrap()
        .map(|e| e.unwrap().0.value().to_string())
        .collect();
    assert_eq!(vals, vec!["banana", "cherry"]);
}

// ===========================================================================
// S28 DATABASE OPERATIONS EXTENDED
// ===========================================================================

#[test]
fn db_delete_nonexistent_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("ghost");
    let w = db.begin_write().unwrap();
    let deleted = w.delete_table(T).unwrap();
    assert!(!deleted);
    w.commit().unwrap();
}

#[test]
fn db_create_many_tables() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    for i in 0..20u64 {
        let name = format!("table_{i}");
        let def: TableDefinition<u64, u64> = TableDefinition::new(&name);
        let mut t = w.open_table(def).unwrap();
        t.insert(&i, &i).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let tables: Vec<String> = r
        .list_tables()
        .unwrap()
        .map(|h| h.name().to_string())
        .collect();
    assert_eq!(tables.len(), 20);
}

#[test]
fn db_empty_has_no_tables() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let tables: Vec<String> = r
        .list_tables()
        .unwrap()
        .map(|h| h.name().to_string())
        .collect();
    assert!(tables.is_empty());
}

#[test]
fn db_reopen_multiple_tables() {
    let f = tmpfile();
    {
        let db = Database::create(f.path()).unwrap();
        const T1: TableDefinition<u64, u64> = TableDefinition::new("rt1");
        const T2: TableDefinition<u64, &str> = TableDefinition::new("rt2");
        let w = db.begin_write().unwrap();
        {
            let mut t1 = w.open_table(T1).unwrap();
            let mut t2 = w.open_table(T2).unwrap();
            t1.insert(&1u64, &100u64).unwrap();
            t2.insert(&1u64, "hello").unwrap();
        }
        w.commit().unwrap();
    }
    {
        let db = Database::open(f.path()).unwrap();
        const T1: TableDefinition<u64, u64> = TableDefinition::new("rt1");
        const T2: TableDefinition<u64, &str> = TableDefinition::new("rt2");
        let r = db.begin_read().unwrap();
        assert_eq!(
            r.open_table(T1)
                .unwrap()
                .get(&1u64)
                .unwrap()
                .unwrap()
                .value(),
            100
        );
        assert_eq!(
            r.open_table(T2)
                .unwrap()
                .get(&1u64)
                .unwrap()
                .unwrap()
                .value(),
            "hello"
        );
    }
}

// ===========================================================================
// S29 STRESS / BULK OPERATIONS
// ===========================================================================

#[test]
fn stress_5000_entries() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stress5k");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..5000u64 {
            t.insert(&i, &(i * 2)).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 5000);
    assert_eq!(t.get(&4999u64).unwrap().unwrap().value(), 9998);
}

#[test]
fn stress_overwrite_same_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stress_ow");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..1000u64 {
            t.insert(&0u64, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 1);
    assert_eq!(t.get(&0u64).unwrap().unwrap().value(), 999);
}

#[test]
fn stress_many_tables() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    for i in 0..50u64 {
        let name = format!("stress_t_{i}");
        let def: TableDefinition<u64, u64> = TableDefinition::new(&name);
        let mut t = w.open_table(def).unwrap();
        t.insert(&i, &i).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let count = r.list_tables().unwrap().count();
    assert_eq!(count, 50);
}

#[test]
fn stress_large_string_keys() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("stress_str");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..100u64 {
            let key = "k".repeat(i as usize + 1);
            t.insert(key.as_str(), &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 100);
}

// ===========================================================================
// S30 IN-MEMORY BACKEND EXTENDED
// ===========================================================================

#[test]
fn in_memory_backend_multimap() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mem_mm");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &20u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let vals: Vec<u64> = t.get(&1u64).unwrap().map(|v| v.unwrap().value()).collect();
    assert_eq!(vals.len(), 2);
}

#[test]
fn in_memory_backend_range_query() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("mem_range");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..20u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let count = t.range(5u64..15u64).unwrap().count();
    assert_eq!(count, 10);
}

#[test]
fn in_memory_backend_delete_and_reinsert() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("mem_del");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.remove(&1u64).unwrap();
        t.insert(&1u64, &200u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 200);
}

// ===========================================================================
// S31 DISTANCE METRIC ENUM EXTENDED
// ===========================================================================

#[test]
fn distance_metric_euclidean_identity() {
    let a = [1.0f32, 2.0, 3.0];
    assert_eq!(shodh_redb::DistanceMetric::EuclideanSq.compute(&a, &a), 0.0);
}

#[test]
fn distance_metric_manhattan_identity() {
    let a = [1.0f32, 2.0, 3.0];
    assert_eq!(shodh_redb::DistanceMetric::Manhattan.compute(&a, &a), 0.0);
}

#[test]
fn distance_metric_cosine_identity() {
    let a = [1.0f32, 2.0, 3.0];
    let d = shodh_redb::DistanceMetric::Cosine.compute(&a, &a);
    assert!(d.abs() < 1e-6);
}

#[test]
fn distance_metric_dot_identity() {
    let a = [1.0f32, 0.0];
    let d = shodh_redb::DistanceMetric::DotProduct.compute(&a, &a);
    assert_eq!(d, -1.0);
}

// ===========================================================================
// S32 MERGE OPERATOR EDGE CASES
// ===========================================================================

#[test]
fn merge_add_multiple_keys() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_multi");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("a", &10u64.to_le_bytes(), &shodh_redb::NumericAdd)
            .unwrap();
        t.merge("b", &20u64.to_le_bytes(), &shodh_redb::NumericAdd)
            .unwrap();
        t.merge("a", &5u64.to_le_bytes(), &shodh_redb::NumericAdd)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("a").unwrap().unwrap().value(), 15);
    assert_eq!(t.get("b").unwrap().unwrap().value(), 20);
}

#[test]
fn merge_add_zero() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_zero");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &42u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &0u64.to_le_bytes(), &shodh_redb::NumericAdd)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 42);
}

#[test]
fn merge_max_equal_values() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_max_eq");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &50u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &50u64.to_le_bytes(), &shodh_redb::NumericMax)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 50);
}

#[test]
fn merge_min_equal_values() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_min_eq");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &50u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &50u64.to_le_bytes(), &shodh_redb::NumericMin)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 50);
}

// ===========================================================================
// S33 TABLE METADATA
// ===========================================================================

#[test]
fn table_is_empty() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("is_empty");
    let w = db.begin_write().unwrap();
    {
        w.open_table(T).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert!(t.is_empty().unwrap());
}

#[test]
fn table_not_empty() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("not_empty");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert!(!t.is_empty().unwrap());
}

#[test]
fn table_len_after_inserts_and_deletes() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("len_ops");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
        t.remove(&5u64).unwrap();
        t.remove(&7u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 8);
}

// ===========================================================================
// S34 COSINE DISTANCE FUNCTION
// ===========================================================================

#[test]
fn cosine_distance_identical() {
    let a = [1.0f32, 2.0, 3.0];
    let d = shodh_redb::cosine_distance(&a, &a);
    assert!(d.abs() < 1e-6);
}

#[test]
fn cosine_distance_orthogonal() {
    let a = [1.0f32, 0.0];
    let b = [0.0f32, 1.0];
    let d = shodh_redb::cosine_distance(&a, &b);
    assert!((d - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_distance_opposite() {
    let a = [1.0f32, 0.0];
    let b = [-1.0f32, 0.0];
    let d = shodh_redb::cosine_distance(&a, &b);
    assert!((d - 2.0).abs() < 1e-6);
}

// ===========================================================================
// S35 ITERATION ORDER VERIFICATION
// ===========================================================================

#[test]
fn iter_ascending_order_u64() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("iter_asc");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        // Insert in reverse order
        for i in (0..100u64).rev() {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<u64> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
    for (i, key) in keys.iter().enumerate().take(100) {
        assert_eq!(*key, i as u64);
    }
}

#[test]
fn iter_ascending_order_str() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("iter_str");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("d", &4u64).unwrap();
        t.insert("b", &2u64).unwrap();
        t.insert("a", &1u64).unwrap();
        t.insert("c", &3u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<String> = t
        .iter()
        .unwrap()
        .map(|e| e.unwrap().0.value().to_string())
        .collect();
    assert_eq!(keys, vec!["a", "b", "c", "d"]);
}

#[test]
fn iter_reverse_range() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("iter_rev2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let vals: Vec<u64> = t
        .range(3u64..=6u64)
        .unwrap()
        .rev()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(vals, vec![6, 5, 4, 3]);
}

// ===========================================================================
// S36 OPTION TYPE EXTENDED
// ===========================================================================

#[test]
fn option_u64_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, Option<u64>> = TableDefinition::new("opt_u64");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &Some(u64::MAX)).unwrap();
        t.insert(&2u64, &None).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), Some(u64::MAX));
    assert_eq!(t.get(&2u64).unwrap().unwrap().value(), None);
}

#[test]
fn option_bool_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, Option<bool>> = TableDefinition::new("opt_bool");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &Some(true)).unwrap();
        t.insert(&2u64, &Some(false)).unwrap();
        t.insert(&3u64, &None).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), Some(true));
    assert_eq!(t.get(&2u64).unwrap().unwrap().value(), Some(false));
    assert_eq!(t.get(&3u64).unwrap().unwrap().value(), None);
}

// ===========================================================================
// S37 BUILDER WITH IN-MEMORY BACKEND EDGE CASES
// ===========================================================================

#[test]
fn in_memory_backend_empty_db() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    let r = db.begin_read().unwrap();
    let tables: Vec<String> = r
        .list_tables()
        .unwrap()
        .map(|h| h.name().to_string())
        .collect();
    assert!(tables.is_empty());
}

#[test]
fn in_memory_backend_savepoint() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("mem_sp");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    let sp = w.ephemeral_savepoint().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&2u64, &200u64).unwrap();
    }
    w.commit().unwrap();
    let mut w = db.begin_write().unwrap();
    w.restore_savepoint(&sp).unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert!(t.get(&2u64).unwrap().is_none());
}

// ===========================================================================
// S38 L2 NORMALIZE EXTENDED
// ===========================================================================

#[test]
fn l2_normalized_preserves_direction() {
    let v = vec![3.0f32, 4.0, 0.0];
    let n = shodh_redb::l2_normalized(&v);
    assert!(n[0] > 0.0);
    assert!(n[1] > 0.0);
    assert_eq!(n[2], 0.0);
    assert!((n[0] / n[1] - 0.75).abs() < 1e-6);
}

#[test]
fn l2_normalize_large_values() {
    let mut v = [1e10f32, 1e10];
    shodh_redb::l2_normalize(&mut v);
    let norm = shodh_redb::l2_norm(&v);
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn l2_normalize_small_values() {
    let mut v = [1e-20f32, 1e-20];
    shodh_redb::l2_normalize(&mut v);
    let norm = shodh_redb::l2_norm(&v);
    assert!((norm - 1.0).abs() < 1e-5);
}

// ===========================================================================
// S39 TTL TABLE BASIC OPERATIONS
// ===========================================================================

#[test]
fn ttl_insert_and_get() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_basic");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert("key1", &42u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert_eq!(t.get("key1").unwrap().unwrap().value(), 42);
}

#[test]
fn ttl_insert_no_expiry_has_zero_expires_at() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_noexp");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert("k", &1u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().expires_at_ms(), 0);
}

#[test]
fn ttl_insert_with_ttl_has_nonzero_expires_at() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_exp");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert_with_ttl("k", &1u64, Duration::from_secs(3600))
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    let guard = t.get("k").unwrap().unwrap();
    assert!(guard.expires_at_ms() > 0);
    assert_eq!(guard.value(), 1);
}

#[test]
fn ttl_expired_entry_not_visible() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_gone");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        // TTL of 0 means already expired
        t.insert_with_ttl("k", &1u64, Duration::ZERO).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert!(t.get("k").unwrap().is_none());
}

#[test]
fn ttl_len_with_expired_includes_expired() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_len");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert("live", &1u64).unwrap();
        t.insert_with_ttl("dead", &2u64, Duration::ZERO).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let t = w.open_ttl_table(T).unwrap();
        assert_eq!(t.len_with_expired().unwrap(), 2);
    }
    w.commit().unwrap();
}

#[test]
fn ttl_purge_expired_removes_dead_entries() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_purge");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert("live", &1u64).unwrap();
        t.insert_with_ttl("dead", &2u64, Duration::ZERO).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        let purged = t.purge_expired().unwrap();
        assert_eq!(purged, 1);
        assert_eq!(t.len_with_expired().unwrap(), 1);
    }
    w.commit().unwrap();
}

#[test]
fn ttl_remove_returns_old_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_rm");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert("k", &99u64).unwrap();
        let old = t.remove("k").unwrap();
        assert_eq!(old.unwrap().value(), 99);
    }
    w.commit().unwrap();
}

#[test]
fn ttl_remove_nonexistent_returns_none() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_rm_none");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        assert!(t.remove("nope").unwrap().is_none());
    }
    w.commit().unwrap();
}

#[test]
fn ttl_iter_skips_expired() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_iter");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert_with_ttl(&2u64, &20u64, Duration::ZERO).unwrap();
        t.insert(&3u64, &30u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let t = w.open_ttl_table(T).unwrap();
        let keys: Vec<u64> = t.iter().unwrap().map(|e| e.unwrap().0.value()).collect();
        assert_eq!(keys, vec![1, 3]);
    }
    w.commit().unwrap();
}

#[test]
fn ttl_range_query() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_range");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &(i * 10)).unwrap();
        }
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let t = w.open_ttl_table(T).unwrap();
        let vals: Vec<u64> = t
            .range(3u64..7u64)
            .unwrap()
            .map(|e| e.unwrap().1.value())
            .collect();
        assert_eq!(vals, vec![30, 40, 50, 60]);
    }
    w.commit().unwrap();
}

#[test]
fn ttl_overwrite_preserves_new_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_ow");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert("k", &1u64).unwrap();
        t.insert("k", &2u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 2);
}

#[test]
fn ttl_long_lived_entry_accessible() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_long");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert_with_ttl("k", &42u64, Duration::from_secs(86400))
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 42);
}

#[test]
fn ttl_definition_name() {
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("my_ttl");
    assert_eq!(T.name(), "my_ttl");
}

#[test]
fn ttl_multiple_keys() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, &str> = TtlTableDefinition::new("ttl_multi");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        for i in 0..50u64 {
            t.insert(&i, "value").unwrap();
        }
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let t = w.open_ttl_table(T).unwrap();
        assert_eq!(t.len_with_expired().unwrap(), 50);
    }
    w.commit().unwrap();
}

#[test]
fn ttl_purge_on_empty_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_purge_empty");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        assert_eq!(t.purge_expired().unwrap(), 0);
    }
    w.commit().unwrap();
}

#[test]
fn ttl_read_only_get() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_ro");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert("a", &1u64).unwrap();
        t.insert("b", &2u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert_eq!(t.get("a").unwrap().unwrap().value(), 1);
    assert_eq!(t.get("b").unwrap().unwrap().value(), 2);
    assert!(t.get("c").unwrap().is_none());
}

#[test]
fn ttl_read_only_iter() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_ro_iter");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        for i in 0..5u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    let count = t.iter().unwrap().count();
    assert_eq!(count, 5);
}

#[test]
fn ttl_read_only_range() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_ro_range");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    let vals: Vec<u64> = t
        .range(2u64..5u64)
        .unwrap()
        .map(|e| e.unwrap().1.value())
        .collect();
    assert_eq!(vals, vec![2, 3, 4]);
}

#[test]
fn ttl_read_only_len_with_expired() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_ro_len");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert("a", &1u64).unwrap();
        t.insert_with_ttl("b", &2u64, Duration::ZERO).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert_eq!(t.len_with_expired().unwrap(), 2);
}

// ===========================================================================
// S40 BLOB STORE TYPES -- BlobId
// ===========================================================================

#[test]
fn blob_id_new_and_fields() {
    let id = shodh_redb::BlobId::new(42, 0xDEAD_BEEF);
    assert_eq!(id.sequence, 42);
    assert_eq!(id.content_prefix_hash, 0xDEAD_BEEF);
}

#[test]
fn blob_id_roundtrip() {
    let id = shodh_redb::BlobId::new(u64::MAX, u64::MAX);
    let bytes = id.to_be_bytes();
    let recovered = shodh_redb::BlobId::from_be_bytes(bytes);
    assert_eq!(recovered.sequence, u64::MAX);
    assert_eq!(recovered.content_prefix_hash, u64::MAX);
}

#[test]
fn blob_id_min() {
    let min = shodh_redb::BlobId::MIN;
    assert_eq!(min.sequence, 0);
    assert_eq!(min.content_prefix_hash, 0);
}

#[test]
fn blob_id_max() {
    let max = shodh_redb::BlobId::MAX;
    assert_eq!(max.sequence, u64::MAX);
    assert_eq!(max.content_prefix_hash, u64::MAX);
}

#[test]
fn blob_id_equality() {
    let a = shodh_redb::BlobId::new(1, 2);
    let b = shodh_redb::BlobId::new(1, 2);
    assert_eq!(a, b);
}

#[test]
fn blob_id_inequality() {
    let a = shodh_redb::BlobId::new(1, 2);
    let b = shodh_redb::BlobId::new(1, 3);
    assert_ne!(a, b);
}

#[test]
fn blob_id_serialized_size() {
    assert_eq!(shodh_redb::BlobId::SERIALIZED_SIZE, 16);
}

#[test]
fn blob_id_zero_roundtrip() {
    let id = shodh_redb::BlobId::new(0, 0);
    let bytes = id.to_be_bytes();
    assert_eq!(bytes, [0u8; 16]);
    let recovered = shodh_redb::BlobId::from_be_bytes(bytes);
    assert_eq!(recovered, id);
}

// ===========================================================================
// S41 BLOB STORE TYPES -- ContentType
// ===========================================================================

#[test]
fn content_type_all_variants_roundtrip() {
    use shodh_redb::ContentType;
    let variants = [
        ContentType::OctetStream,
        ContentType::ImagePng,
        ContentType::ImageJpeg,
        ContentType::AudioWav,
        ContentType::AudioOgg,
        ContentType::VideoMp4,
        ContentType::PointCloudLas,
        ContentType::SensorImu,
        ContentType::Embedding,
        ContentType::Metadata,
    ];
    for ct in &variants {
        let byte = ct.as_byte();
        let recovered = ContentType::from_byte(byte);
        assert_eq!(recovered.as_byte(), byte);
    }
}

#[test]
fn content_type_mime_strings() {
    use shodh_redb::ContentType;
    assert_eq!(
        ContentType::OctetStream.mime_str(),
        "application/octet-stream"
    );
    assert_eq!(ContentType::ImagePng.mime_str(), "image/png");
    assert_eq!(ContentType::ImageJpeg.mime_str(), "image/jpeg");
    assert_eq!(ContentType::AudioWav.mime_str(), "audio/wav");
    assert_eq!(ContentType::AudioOgg.mime_str(), "audio/ogg");
    assert_eq!(ContentType::VideoMp4.mime_str(), "video/mp4");
    assert_eq!(ContentType::Embedding.mime_str(), "application/x-embedding");
    assert_eq!(ContentType::Metadata.mime_str(), "application/json");
}

#[test]
fn content_type_unknown_byte_defaults_to_octet_stream() {
    let ct = shodh_redb::ContentType::from_byte(255);
    assert_eq!(ct.as_byte(), 0);
}

#[test]
fn content_type_byte_values() {
    use shodh_redb::ContentType;
    assert_eq!(ContentType::OctetStream.as_byte(), 0);
    assert_eq!(ContentType::ImagePng.as_byte(), 1);
    assert_eq!(ContentType::ImageJpeg.as_byte(), 2);
    assert_eq!(ContentType::Metadata.as_byte(), 9);
}

// ===========================================================================
// S42 BLOB STORE TYPES -- BlobRef
// ===========================================================================

#[test]
fn blob_ref_roundtrip() {
    use shodh_redb::BlobRef;
    let br = BlobRef {
        offset: 1024,
        length: 4096,
        checksum: 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0,
        ref_count: 3,
        content_type: 2,
        compression: 1,
    };
    let bytes = br.to_le_bytes();
    let recovered = BlobRef::from_le_bytes(bytes);
    assert_eq!(recovered.offset, 1024);
    assert_eq!(recovered.length, 4096);
    assert_eq!(recovered.checksum, br.checksum);
    assert_eq!(recovered.ref_count, 3);
    assert_eq!(recovered.content_type, 2);
    assert_eq!(recovered.compression, 1);
}

#[test]
fn blob_ref_serialized_size() {
    assert_eq!(shodh_redb::BlobRef::SERIALIZED_SIZE, 40);
}

#[test]
fn blob_ref_content_type_enum() {
    use shodh_redb::BlobRef;
    let br = BlobRef {
        offset: 0,
        length: 0,
        checksum: 0,
        ref_count: 1,
        content_type: 8, // Embedding
        compression: 0,
    };
    assert_eq!(
        br.content_type_enum().as_byte(),
        shodh_redb::ContentType::Embedding.as_byte()
    );
}

// ===========================================================================
// S43 BLOB STORE TYPES -- BlobMeta
// ===========================================================================

#[test]
fn blob_meta_roundtrip_with_parent() {
    use shodh_redb::{BlobMeta, BlobRef};
    let br = BlobRef {
        offset: 0,
        length: 100,
        checksum: 42,
        ref_count: 1,
        content_type: 0,
        compression: 0,
    };
    let parent = shodh_redb::BlobId::new(10, 20);
    let meta = BlobMeta::new(br, 1_000_000, 99, Some(parent), "test-label");
    let bytes = meta.to_le_bytes();
    let recovered = BlobMeta::from_le_bytes(bytes);
    assert_eq!(recovered.wall_clock_ns, 1_000_000);
    assert_eq!(recovered.hlc, 99);
    assert_eq!(recovered.causal_parent.unwrap().sequence, 10);
    assert_eq!(recovered.label_str(), "test-label");
}

#[test]
fn blob_meta_roundtrip_no_parent() {
    use shodh_redb::{BlobMeta, BlobRef};
    let br = BlobRef {
        offset: 0,
        length: 0,
        checksum: 0,
        ref_count: 1,
        content_type: 0,
        compression: 0,
    };
    let meta = BlobMeta::new(br, 0, 0, None, "");
    let bytes = meta.to_le_bytes();
    let recovered = BlobMeta::from_le_bytes(bytes);
    assert!(recovered.causal_parent.is_none());
    assert_eq!(recovered.label_str(), "");
}

#[test]
fn blob_meta_serialized_size() {
    assert_eq!(shodh_redb::BlobMeta::SERIALIZED_SIZE, 169);
}

#[test]
fn blob_meta_label_truncation() {
    use shodh_redb::{BlobMeta, BlobRef};
    let br = BlobRef {
        offset: 0,
        length: 0,
        checksum: 0,
        ref_count: 1,
        content_type: 0,
        compression: 0,
    };
    let long_label = "a".repeat(100);
    let meta = BlobMeta::new(br, 0, 0, None, &long_label);
    assert_eq!(meta.label_str().len(), 63);
}

// ===========================================================================
// S44 BLOB STORE TYPES -- RelationType
// ===========================================================================

#[test]
fn relation_type_all_variants() {
    use shodh_redb::RelationType;
    let variants = [
        (RelationType::Derived, 0, "derived"),
        (RelationType::Similar, 1, "similar"),
        (RelationType::Contradicts, 2, "contradicts"),
        (RelationType::Supports, 3, "supports"),
        (RelationType::Supersedes, 4, "supersedes"),
    ];
    for (rt, byte, label) in &variants {
        assert_eq!(rt.as_byte(), *byte);
        assert_eq!(rt.label(), *label);
        assert_eq!(RelationType::from_byte(*byte).as_byte(), *byte);
    }
}

#[test]
fn relation_type_unknown_byte_defaults_to_derived() {
    use shodh_redb::RelationType;
    assert_eq!(RelationType::from_byte(255).as_byte(), 0);
}

// ===========================================================================
// S45 BLOB STORE TYPES -- CausalEdge
// ===========================================================================

#[test]
fn causal_edge_roundtrip() {
    use shodh_redb::{CausalEdge, RelationType};
    let child = shodh_redb::BlobId::new(5, 10);
    let edge = CausalEdge::new(child, RelationType::Supports, "test context");
    let bytes = edge.to_le_bytes();
    let recovered = CausalEdge::from_le_bytes(bytes);
    assert_eq!(recovered.child.sequence, 5);
    assert_eq!(
        recovered.relation.as_byte(),
        RelationType::Supports.as_byte()
    );
    assert_eq!(recovered.context_str(), "test context");
}

#[test]
fn causal_edge_legacy() {
    use shodh_redb::CausalEdge;
    let child = shodh_redb::BlobId::new(1, 2);
    let edge = CausalEdge::legacy(child);
    assert_eq!(edge.relation.as_byte(), 0); // Derived
    assert_eq!(edge.context_str(), "");
}

#[test]
fn causal_edge_serialized_size() {
    assert_eq!(shodh_redb::CausalEdge::SERIALIZED_SIZE, 80);
}

#[test]
fn causal_edge_context_truncation() {
    use shodh_redb::{CausalEdge, RelationType};
    let child = shodh_redb::BlobId::new(0, 0);
    let long_ctx = "x".repeat(100);
    let edge = CausalEdge::new(child, RelationType::Derived, &long_ctx);
    assert_eq!(edge.context_str().len(), 62);
}

// ===========================================================================
// S46 BLOB STORE TYPES -- CausalEdgeKey
// ===========================================================================

#[test]
fn causal_edge_key_roundtrip() {
    use shodh_redb::blob_store::CausalEdgeKey;
    let parent = shodh_redb::BlobId::new(1, 2);
    let child = shodh_redb::BlobId::new(3, 4);
    let key = CausalEdgeKey::new(parent, child);
    let bytes = key.to_be_bytes();
    let recovered = CausalEdgeKey::from_be_bytes(bytes);
    assert_eq!(recovered.parent.sequence, 1);
    assert_eq!(recovered.child.sequence, 3);
}

#[test]
fn causal_edge_key_serialized_size() {
    assert_eq!(shodh_redb::blob_store::CausalEdgeKey::SERIALIZED_SIZE, 32);
}

// ===========================================================================
// S47 BLOB STORE TYPES -- TagKey
// ===========================================================================

#[test]
fn tag_key_roundtrip() {
    use shodh_redb::blob_store::TagKey;
    let blob_id = shodh_redb::BlobId::new(42, 99);
    let tk = TagKey::new("sensor-data", blob_id);
    let bytes = tk.to_be_bytes();
    let recovered = TagKey::from_be_bytes(bytes);
    assert_eq!(recovered.tag_str(), "sensor-data");
    assert_eq!(recovered.blob_id.sequence, 42);
}

#[test]
fn tag_key_serialized_size() {
    assert_eq!(shodh_redb::blob_store::TagKey::SERIALIZED_SIZE, 49);
}

#[test]
fn tag_key_range_bounds() {
    use shodh_redb::blob_store::TagKey;
    let start = TagKey::range_start("my-tag");
    let end = TagKey::range_end("my-tag");
    assert_eq!(start.tag_str(), "my-tag");
    assert_eq!(start.blob_id, shodh_redb::BlobId::MIN);
    assert_eq!(end.blob_id, shodh_redb::BlobId::MAX);
}

#[test]
fn tag_key_truncation() {
    use shodh_redb::blob_store::TagKey;
    let long_tag = "t".repeat(50);
    let tk = TagKey::new(&long_tag, shodh_redb::BlobId::MIN);
    assert_eq!(tk.tag_str().len(), 32);
}

// ===========================================================================
// S48 BLOB STORE TYPES -- NamespaceKey / NamespaceVal
// ===========================================================================

#[test]
fn namespace_key_roundtrip() {
    use shodh_redb::blob_store::NamespaceKey;
    let blob_id = shodh_redb::BlobId::new(7, 8);
    let nk = NamespaceKey::new("prod-session", blob_id);
    let bytes = nk.to_be_bytes();
    let recovered = NamespaceKey::from_be_bytes(bytes);
    assert_eq!(recovered.namespace_str(), "prod-session");
    assert_eq!(recovered.blob_id.sequence, 7);
}

#[test]
fn namespace_key_serialized_size() {
    assert_eq!(shodh_redb::blob_store::NamespaceKey::SERIALIZED_SIZE, 80);
}

#[test]
fn namespace_key_range_bounds() {
    use shodh_redb::blob_store::NamespaceKey;
    let start = NamespaceKey::range_start("ns");
    let end = NamespaceKey::range_end("ns");
    assert_eq!(start.namespace_str(), "ns");
    assert_eq!(start.blob_id, shodh_redb::BlobId::MIN);
    assert_eq!(end.blob_id, shodh_redb::BlobId::MAX);
}

#[test]
fn namespace_val_roundtrip() {
    use shodh_redb::blob_store::NamespaceVal;
    let nv = NamespaceVal::new("my-namespace");
    let bytes = nv.to_le_bytes();
    let recovered = NamespaceVal::from_le_bytes(bytes);
    assert_eq!(recovered.namespace_str(), "my-namespace");
}

#[test]
fn namespace_val_serialized_size() {
    assert_eq!(shodh_redb::blob_store::NamespaceVal::SERIALIZED_SIZE, 64);
}

#[test]
fn namespace_val_truncation() {
    use shodh_redb::blob_store::NamespaceVal;
    let long_ns = "n".repeat(100);
    let nv = NamespaceVal::new(&long_ns);
    assert_eq!(nv.namespace_str().len(), 63);
}

// ===========================================================================
// S49 BLOB STORE TYPES -- TemporalKey
// ===========================================================================

#[test]
fn temporal_key_roundtrip() {
    use shodh_redb::blob_store::TemporalKey;
    let hlc = shodh_redb::HybridLogicalClock::from_parts(500, 3);
    let blob_id = shodh_redb::BlobId::new(10, 20);
    let tk = TemporalKey::new(1_000_000, hlc, blob_id);
    let bytes = tk.to_be_bytes();
    let recovered = TemporalKey::from_be_bytes(bytes);
    assert_eq!(recovered.wall_clock_ns, 1_000_000);
    assert_eq!(recovered.hlc.physical_ms(), 500);
    assert_eq!(recovered.blob_id.sequence, 10);
}

#[test]
fn temporal_key_serialized_size() {
    assert_eq!(shodh_redb::blob_store::TemporalKey::SERIALIZED_SIZE, 32);
}

#[test]
fn temporal_key_range_bounds() {
    use shodh_redb::blob_store::TemporalKey;
    let start = TemporalKey::range_start(100);
    let end = TemporalKey::range_end(200);
    assert_eq!(start.wall_clock_ns, 100);
    assert_eq!(end.wall_clock_ns, 200);
    assert_eq!(start.blob_id, shodh_redb::BlobId::MIN);
    assert_eq!(end.blob_id, shodh_redb::BlobId::MAX);
}

// ===========================================================================
// S50 BLOB STORE TYPES -- StoreOptions
// ===========================================================================

#[test]
fn store_options_with_namespace() {
    let opts = shodh_redb::StoreOptions::with_namespace("test-ns");
    assert_eq!(opts.namespace.as_deref(), Some("test-ns"));
    assert!(opts.causal_link.is_none());
    assert!(opts.tags.is_empty());
}

#[test]
fn store_options_with_tags() {
    let opts = shodh_redb::StoreOptions::with_tags(&["a", "b", "c"]);
    assert_eq!(opts.tags.len(), 3);
    assert_eq!(opts.tags[0], "a");
    assert!(opts.namespace.is_none());
}

#[test]
fn store_options_default() {
    let opts = shodh_redb::StoreOptions::default();
    assert!(opts.causal_link.is_none());
    assert!(opts.namespace.is_none());
    assert!(opts.tags.is_empty());
}

#[test]
fn max_tags_per_blob_constant() {
    assert_eq!(shodh_redb::MAX_TAGS_PER_BLOB, 8);
}

// ===========================================================================
// S51 BLOB STORE TYPES -- CausalLink
// ===========================================================================

#[test]
fn causal_link_new() {
    use shodh_redb::{CausalLink, RelationType};
    let parent = shodh_redb::BlobId::new(1, 2);
    let link = CausalLink::new(parent, RelationType::Similar, "related content");
    assert_eq!(link.parent.sequence, 1);
    assert_eq!(link.relation.as_byte(), 1);
    assert_eq!(link.context, "related content");
}

#[test]
fn causal_link_derived() {
    use shodh_redb::CausalLink;
    let parent = shodh_redb::BlobId::new(5, 6);
    let link = CausalLink::derived(parent);
    assert_eq!(link.parent.sequence, 5);
    assert_eq!(link.relation.as_byte(), 0);
    assert!(link.context.is_empty());
}

// ===========================================================================
// S52 CDC TYPES
// ===========================================================================

#[test]
fn cdc_config_default() {
    let cfg = shodh_redb::CdcConfig::default();
    assert!(!cfg.enabled);
    assert_eq!(cfg.retention_max_txns, 0);
}

#[test]
fn cdc_config_enabled() {
    let cfg = shodh_redb::CdcConfig {
        enabled: true,
        retention_max_txns: 1000,
    };
    assert!(cfg.enabled);
    assert_eq!(cfg.retention_max_txns, 1000);
}

#[test]
fn change_op_variants() {
    assert_ne!(shodh_redb::ChangeOp::Insert, shodh_redb::ChangeOp::Update);
    assert_ne!(shodh_redb::ChangeOp::Update, shodh_redb::ChangeOp::Delete);
    assert_ne!(shodh_redb::ChangeOp::Delete, shodh_redb::ChangeOp::Insert);
}

#[test]
fn change_op_clone() {
    let op = shodh_redb::ChangeOp::Insert;
    let cloned = op;
    assert_eq!(op, cloned);
}

// ===========================================================================
// S54 COMPOSITE TYPES -- SignalWeights
// ===========================================================================

#[test]
fn signal_weights_default() {
    let w = shodh_redb::SignalWeights::default();
    assert_eq!(w.semantic, 0.0);
    assert_eq!(w.temporal, 0.0);
    assert_eq!(w.causal, 0.0);
}

#[test]
fn signal_weights_custom() {
    let w = shodh_redb::SignalWeights {
        semantic: 0.7,
        temporal: 0.2,
        causal: 0.1,
    };
    assert_eq!(w.semantic, 0.7);
    assert_eq!(w.temporal, 0.2);
    assert_eq!(w.causal, 0.1);
}

#[test]
fn signal_weights_clone() {
    let w = shodh_redb::SignalWeights {
        semantic: 1.0,
        temporal: 0.0,
        causal: 0.0,
    };
    let cloned = w.clone();
    assert_eq!(cloned.semantic, 1.0);
}

// ===========================================================================
// S55 ADDITIONAL MERGE OPERATORS -- BytesAppend / BitwiseOr
// ===========================================================================

#[test]
fn merge_bytes_append_new_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, &[u8]> = TableDefinition::new("merge_ba");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("log", b"hello", &shodh_redb::BytesAppend).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("log").unwrap().unwrap().value(), b"hello");
}

#[test]
fn merge_bytes_append_existing() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, &[u8]> = TableDefinition::new("merge_ba2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("log", b"hello".as_slice()).unwrap();
        t.merge("log", b" world", &shodh_redb::BytesAppend).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("log").unwrap().unwrap().value(), b"hello world");
}

#[test]
fn merge_bytes_append_empty() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, &[u8]> = TableDefinition::new("merge_ba3");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", b"data".as_slice()).unwrap();
        t.merge("k", b"", &shodh_redb::BytesAppend).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), b"data");
}

#[test]
fn merge_bitwise_or_new_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_bor");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("flags", &0b1010u64.to_le_bytes(), &shodh_redb::BitwiseOr)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("flags").unwrap().unwrap().value(), 0b1010);
}

#[test]
fn merge_bitwise_or_existing() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_bor2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("flags", &0b1100u64).unwrap();
        t.merge("flags", &0b0011u64.to_le_bytes(), &shodh_redb::BitwiseOr)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("flags").unwrap().unwrap().value(), 0b1111);
}

#[test]
fn merge_fn_custom_subtract() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_sub");
    let subtract = shodh_redb::merge_fn(|_key, existing, operand| {
        let a = existing.map_or(0u64, |b| u64::from_le_bytes(b.try_into().unwrap()));
        let b = u64::from_le_bytes(operand.try_into().unwrap());
        Some(a.saturating_sub(b).to_le_bytes().to_vec())
    });
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &100u64).unwrap();
        t.merge("k", &30u64.to_le_bytes(), &subtract).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 70);
}

#[test]
fn merge_fn_conditional_update() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_cond");
    // Only update if new value is even
    let even_only = shodh_redb::merge_fn(|_key, existing, operand| {
        let new_val = u64::from_le_bytes(operand.try_into().unwrap());
        if new_val % 2 == 0 {
            Some(new_val.to_le_bytes().to_vec())
        } else {
            existing.map(|b| b.to_vec())
        }
    });
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &10u64).unwrap();
        t.merge("k", &7u64.to_le_bytes(), &even_only).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 10); // unchanged
}

// ===========================================================================
// S56 SCALAR QUANTIZATION DISTANCE FUNCTIONS
// ===========================================================================

#[test]
fn sq_euclidean_basic() {
    let v = [1.0f32, 2.0, 3.0, 4.0];
    let sq = shodh_redb::quantize_scalar(&v);
    let dist = shodh_redb::sq_euclidean_distance_sq(&v, &sq);
    // Distance to self through quantization should be very small
    assert!(dist < 1.0);
}

#[test]
fn sq_dot_product_basic() {
    let v = [1.0f32, 0.0, 0.0, 0.0];
    let sq = shodh_redb::quantize_scalar(&v);
    let dp = shodh_redb::sq_dot_product(&v, &sq);
    assert!(dp > 0.0);
}

#[test]
fn sq_roundtrip_accuracy() {
    let v = [0.0f32, 0.5, 1.0, -1.0, -0.5, 0.25, -0.25, 0.75];
    let sq = shodh_redb::quantize_scalar(&v);
    let dq = sq.dequantize();
    for i in 0..v.len() {
        assert!((v[i] - dq[i]).abs() < 0.02);
    }
}

#[test]
fn sq_min_max_preserved() {
    let v = [-10.0f32, 0.0, 5.0, 10.0];
    let sq = shodh_redb::quantize_scalar(&v);
    assert_eq!(sq.min_val, -10.0);
    assert_eq!(sq.max_val, 10.0);
}

#[test]
fn sq_dequantize_matches_function() {
    let v = [1.0f32, 2.0, 3.0, 4.0];
    let sq = shodh_redb::quantize_scalar(&v);
    let dq1 = sq.dequantize();
    let dq2 = shodh_redb::dequantize_scalar(&sq);
    for i in 0..4 {
        assert!((dq1[i] - dq2[i]).abs() < 1e-6);
    }
}

// ===========================================================================
// S57 HLC ADVANCED
// ===========================================================================

#[test]
fn hlc_advance_forward() {
    let old = shodh_redb::HybridLogicalClock::from_parts(1, 0);
    let advanced = old.advance();
    assert!(advanced.physical_ms() >= old.physical_ms());
}

#[test]
fn hlc_merge_both_old() {
    let a = shodh_redb::HybridLogicalClock::from_parts(1, 5);
    let b = shodh_redb::HybridLogicalClock::from_parts(2, 3);
    let merged = a.merge(b);
    // Should be at least as recent as both
    assert!(merged.to_raw() >= a.to_raw());
    assert!(merged.to_raw() >= b.to_raw());
}

#[test]
fn hlc_from_wall_ns() {
    let ns = 1_700_000_000_000_000_000u64; // ~2023 in ns
    let hlc = shodh_redb::HybridLogicalClock::from_wall_ns(ns);
    assert_eq!(hlc.physical_ms(), ns / 1_000_000);
    assert_eq!(hlc.logical(), 0);
}

#[test]
fn hlc_zero_constant() {
    let zero = shodh_redb::HybridLogicalClock::ZERO;
    assert_eq!(zero.physical_ms(), 0);
    assert_eq!(zero.logical(), 0);
    assert_eq!(zero.to_raw(), 0);
}

#[test]
fn hlc_min_max_constants() {
    let min = shodh_redb::HybridLogicalClock::MIN;
    let max = shodh_redb::HybridLogicalClock::MAX;
    assert!(min.to_raw() < max.to_raw());
}

#[test]
fn hlc_ordering_physical_then_logical() {
    let a = shodh_redb::HybridLogicalClock::from_parts(100, 0);
    let b = shodh_redb::HybridLogicalClock::from_parts(100, 1);
    let c = shodh_redb::HybridLogicalClock::from_parts(101, 0);
    assert!(a < b);
    assert!(b < c);
}

// ===========================================================================
// S58 ADDITIONAL TYPE SYSTEM TESTS
// ===========================================================================

#[test]
fn i8_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<i8, i8> = TableDefinition::new("i8rt");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&i8::MIN, &i8::MIN).unwrap();
        t.insert(&0i8, &0i8).unwrap();
        t.insert(&i8::MAX, &i8::MAX).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&i8::MIN).unwrap().unwrap().value(), i8::MIN);
    assert_eq!(t.get(&0i8).unwrap().unwrap().value(), 0);
    assert_eq!(t.get(&i8::MAX).unwrap().unwrap().value(), i8::MAX);
}

#[test]
fn i16_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<i16, i16> = TableDefinition::new("i16rt");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&i16::MIN, &i16::MIN).unwrap();
        t.insert(&i16::MAX, &i16::MAX).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&i16::MIN).unwrap().unwrap().value(), i16::MIN);
    assert_eq!(t.get(&i16::MAX).unwrap().unwrap().value(), i16::MAX);
}

#[test]
fn i32_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<i32, i32> = TableDefinition::new("i32rt");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&i32::MIN, &i32::MIN).unwrap();
        t.insert(&i32::MAX, &i32::MAX).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&i32::MIN).unwrap().unwrap().value(), i32::MIN);
    assert_eq!(t.get(&i32::MAX).unwrap().unwrap().value(), i32::MAX);
}

#[test]
fn i64_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<i64, i64> = TableDefinition::new("i64rt");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&i64::MIN, &i64::MIN).unwrap();
        t.insert(&i64::MAX, &i64::MAX).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&i64::MIN).unwrap().unwrap().value(), i64::MIN);
    assert_eq!(t.get(&i64::MAX).unwrap().unwrap().value(), i64::MAX);
}

#[test]
fn i128_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<i128, i128> = TableDefinition::new("i128rt");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&i128::MIN, &i128::MIN).unwrap();
        t.insert(&i128::MAX, &i128::MAX).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&i128::MIN).unwrap().unwrap().value(), i128::MIN);
    assert_eq!(t.get(&i128::MAX).unwrap().unwrap().value(), i128::MAX);
}

#[test]
fn u128_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u128, u128> = TableDefinition::new("u128rt");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&0u128, &0u128).unwrap();
        t.insert(&u128::MAX, &u128::MAX).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&0u128).unwrap().unwrap().value(), 0);
    assert_eq!(t.get(&u128::MAX).unwrap().unwrap().value(), u128::MAX);
}

#[test]
fn f32_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, f32> = TableDefinition::new("f32rt");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1.234_f32).unwrap();
        t.insert(&2u64, &f32::MIN).unwrap();
        t.insert(&3u64, &f32::MAX).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert!((t.get(&1u64).unwrap().unwrap().value() - 1.234_f32).abs() < 1e-6);
    assert_eq!(t.get(&2u64).unwrap().unwrap().value(), f32::MIN);
    assert_eq!(t.get(&3u64).unwrap().unwrap().value(), f32::MAX);
}

#[test]
fn f64_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, f64> = TableDefinition::new("f64rt");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &std::f64::consts::PI).unwrap();
        t.insert(&2u64, &f64::MIN).unwrap();
        t.insert(&3u64, &f64::MAX).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert!((t.get(&1u64).unwrap().unwrap().value() - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn bool_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, bool> = TableDefinition::new("boolrt");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&0u64, &false).unwrap();
        t.insert(&1u64, &true).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert!(!t.get(&0u64).unwrap().unwrap().value());
    assert!(t.get(&1u64).unwrap().unwrap().value());
}

#[test]
fn bytes_key_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&[u8], u64> = TableDefinition::new("byteskey");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(b"key1".as_slice(), &1u64).unwrap();
        t.insert(b"key2".as_slice(), &2u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(b"key1".as_slice()).unwrap().unwrap().value(), 1);
    assert_eq!(t.get(b"key2".as_slice()).unwrap().unwrap().value(), 2);
}

// ===========================================================================
// S59 TRANSACTION ISOLATION
// ===========================================================================

#[test]
fn write_not_visible_until_commit() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("isolation1");
    // First, seed a value
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    w.commit().unwrap();
    // Start a read txn
    let r = db.begin_read().unwrap();
    // Start a write txn that modifies the value
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &200u64).unwrap();
    }
    w.commit().unwrap();
    // The read txn started before should still see old value
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 100);
}

#[test]
fn abort_discards_changes() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("abort1");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &999u64).unwrap();
    }
    w.abort().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 100);
}

#[test]
fn durability_none_setting() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("dur_none");
    let mut w = db.begin_write().unwrap();
    w.set_durability(Durability::None).unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 1);
}

#[test]
fn multiple_transactions_sequential() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("multi_txn");
    for i in 0..10u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.insert(&i, &i).unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 10);
}

// ===========================================================================
// S60 DELETE TABLE
// ===========================================================================

#[test]
fn delete_existing_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("to_delete");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    assert!(w.delete_table(T).unwrap());
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let tables: Vec<String> = r
        .list_tables()
        .unwrap()
        .map(|h| h.name().to_string())
        .collect();
    assert!(!tables.contains(&"to_delete".to_string()));
}

#[test]
fn delete_multimap_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_del");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    assert!(w.delete_multimap_table(T).unwrap());
    w.commit().unwrap();
}

// ===========================================================================
// S61 ADDITIONAL DISTANCE FUNCTIONS
// ===========================================================================

#[test]
fn manhattan_known_value() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 6.0, 3.0];
    // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
    assert_eq!(shodh_redb::manhattan_distance(&a, &b), 7.0);
}

#[test]
fn euclidean_known_value() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 4.0];
    // 9 + 16 = 25
    assert_eq!(shodh_redb::euclidean_distance_sq(&a, &b), 25.0);
}

#[test]
fn dot_product_known_value() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    // 4 + 10 + 18 = 32
    assert_eq!(shodh_redb::dot_product(&a, &b), 32.0);
}

#[test]
fn cosine_similarity_parallel() {
    let a = [1.0f32, 0.0];
    let b = [5.0f32, 0.0];
    let sim = shodh_redb::cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_similarity_antiparallel() {
    let a = [1.0f32, 0.0];
    let b = [-1.0f32, 0.0];
    let sim = shodh_redb::cosine_similarity(&a, &b);
    assert!((sim + 1.0).abs() < 1e-6);
}

#[test]
fn hamming_identical() {
    let a = [0xABu8, 0xCD, 0xEF];
    assert_eq!(shodh_redb::hamming_distance(&a, &a), 0);
}

#[test]
fn hamming_all_different_bits() {
    let a = [0x00u8];
    let b = [0xFFu8];
    assert_eq!(shodh_redb::hamming_distance(&a, &b), 8);
}

#[test]
fn l2_norm_pythagorean() {
    assert!((shodh_redb::l2_norm(&[3.0f32, 4.0]) - 5.0).abs() < 1e-6);
}

#[test]
fn l2_normalized_unit_length() {
    let v = vec![3.0f32, 4.0, 5.0, 6.0];
    let n = shodh_redb::l2_normalized(&v);
    let norm = shodh_redb::l2_norm(&n);
    assert!((norm - 1.0).abs() < 1e-5);
}

// ===========================================================================
// S62 DISTANCE METRIC ENUM COMPLETENESS
// ===========================================================================

#[test]
fn distance_metric_euclidean_known() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 4.0];
    assert_eq!(
        shodh_redb::DistanceMetric::EuclideanSq.compute(&a, &b),
        25.0
    );
}

#[test]
fn distance_metric_manhattan_known() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 4.0];
    assert_eq!(shodh_redb::DistanceMetric::Manhattan.compute(&a, &b), 7.0);
}

#[test]
fn distance_metric_dot_product_known() {
    let a = [1.0f32, 2.0];
    let b = [3.0f32, 4.0];
    // negated: -(3+8) = -11
    let d = shodh_redb::DistanceMetric::DotProduct.compute(&a, &b);
    assert_eq!(d, -11.0);
}

#[test]
fn distance_metric_cosine_known() {
    let a = [1.0f32, 0.0];
    let b = [0.0f32, 1.0];
    let d = shodh_redb::DistanceMetric::Cosine.compute(&a, &b);
    assert!((d - 1.0).abs() < 1e-6); // orthogonal = cosine distance 1.0
}

// ===========================================================================
// S63 QUANTIZATION ADVANCED
// ===========================================================================

#[test]
fn quantize_binary_all_negative_8dims() {
    let v = [-1.0f32; 8];
    let bq = shodh_redb::quantize_binary(&v);
    assert_eq!(bq[0], 0x00); // all bits 0
}

#[test]
fn quantize_binary_single_dim() {
    let v = [1.0f32];
    let bq = shodh_redb::quantize_binary(&v);
    assert_eq!(bq.len(), 1);
    assert_eq!(bq[0] & 0x80, 0x80); // bit 7 set
}

#[test]
fn quantize_binary_exactly_zero() {
    let v = [0.0f32; 8];
    let bq = shodh_redb::quantize_binary(&v);
    // zero is not positive, so all bits should be 0
    assert_eq!(bq[0], 0x00);
}

#[test]
fn quantize_scalar_wide_range() {
    let v = [-100.0f32, 100.0, 0.0, 50.0];
    let sq = shodh_redb::quantize_scalar(&v);
    assert_eq!(sq.min_val, -100.0);
    assert_eq!(sq.max_val, 100.0);
    let dq = sq.dequantize();
    for i in 0..4 {
        assert!((v[i] - dq[i]).abs() < 1.0);
    }
}

#[test]
fn quantize_scalar_negative_only() {
    let v = [-10.0f32, -5.0, -1.0, -0.1];
    let sq = shodh_redb::quantize_scalar(&v);
    assert!(sq.min_val < 0.0);
    assert!(sq.max_val < 0.0);
}

// ===========================================================================
// S64 BUILDER AND DATABASE CONFIGURATION
// ===========================================================================

#[test]
fn builder_create_file_db() {
    let f = tmpfile();
    let db = shodh_redb::Builder::new().create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("builder1");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn builder_with_cache_size() {
    let f = tmpfile();
    let db = shodh_redb::Builder::new()
        .set_cache_size(8 * 1024 * 1024)
        .create(f.path())
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("cache_test");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn db_open_nonexistent_fails() {
    let result = Database::open("/tmp/nonexistent_db_file_shodh_test_xyz.redb");
    assert!(result.is_err());
}

#[test]
fn db_create_and_reopen() {
    let f = tmpfile();
    {
        let db = Database::create(f.path()).unwrap();
        const T: TableDefinition<u64, u64> = TableDefinition::new("reopen");
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.insert(&1u64, &42u64).unwrap();
        }
        w.commit().unwrap();
    }
    {
        let db = Database::open(f.path()).unwrap();
        const T: TableDefinition<u64, u64> = TableDefinition::new("reopen");
        let r = db.begin_read().unwrap();
        assert_eq!(
            r.open_table(T)
                .unwrap()
                .get(&1u64)
                .unwrap()
                .unwrap()
                .value(),
            42
        );
    }
}

// ===========================================================================
// S65 MULTIMAP ADVANCED
// ===========================================================================

#[test]
fn multimap_remove_specific_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_rm");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &20u64).unwrap();
        t.insert(&1u64, &30u64).unwrap();
        t.remove(&1u64, &20u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let vals: Vec<u64> = t.get(&1u64).unwrap().map(|v| v.unwrap().value()).collect();
    assert_eq!(vals.len(), 2);
    assert!(vals.contains(&10));
    assert!(vals.contains(&30));
}

#[test]
fn multimap_remove_all_values_for_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_rmall2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &20u64).unwrap();
        t.remove_all(&1u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let vals: Vec<u64> = t.get(&1u64).unwrap().map(|v| v.unwrap().value()).collect();
    assert!(vals.is_empty());
}

#[test]
fn multimap_string_values() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<&str, &str> = MultimapTableDefinition::new("mm_str");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert("fruits", "apple").unwrap();
        t.insert("fruits", "banana").unwrap();
        t.insert("fruits", "cherry").unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let vals: Vec<String> = t
        .get("fruits")
        .unwrap()
        .map(|v| v.unwrap().value().to_string())
        .collect();
    assert_eq!(vals.len(), 3);
}

#[test]
fn multimap_iter() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_iter");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&2u64, &20u64).unwrap();
        t.insert(&3u64, &30u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let count = t.iter().unwrap().count();
    assert_eq!(count, 3);
}

// ===========================================================================
// S66 WRITE / READ F32 ADDITIONAL
// ===========================================================================

#[test]
fn write_f32_le_inf_and_neg_inf() {
    let values = [f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
    let mut buf = vec![0u8; 16];
    shodh_redb::write_f32_le(&mut buf, &values);
    let recovered = shodh_redb::read_f32_le(&buf);
    assert_eq!(recovered[0], f32::INFINITY);
    assert_eq!(recovered[1], f32::NEG_INFINITY);
    assert_eq!(recovered[2], 0.0);
}

#[test]
fn write_f32_le_single_value() {
    let values = [42.5f32];
    let mut buf = vec![0u8; 4];
    shodh_redb::write_f32_le(&mut buf, &values);
    let recovered = shodh_redb::read_f32_le(&buf);
    assert_eq!(recovered[0], 42.5);
}

// ===========================================================================
// S67 NEAREST-K WITH DIFFERENT METRICS
// ===========================================================================

#[test]
fn nearest_k_manhattan_top2() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![0.0, 0.0]),
        (2, vec![1.0, 1.0]),
        (3, vec![10.0, 10.0]),
    ];
    let query = [0.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 2, |a, b| {
        shodh_redb::manhattan_distance(a, b)
    });
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].key, 1);
    assert_eq!(results[1].key, 2);
}

#[test]
fn nearest_k_with_dot_product() {
    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![1.0, 0.0]),
        (2, vec![0.0, 1.0]),
        (3, vec![1.0, 1.0]),
    ];
    let query = [1.0f32, 0.0];
    let results = shodh_redb::nearest_k(vectors.into_iter(), &query, 3, |a, b| {
        -shodh_redb::dot_product(a, b) // negate to make lower = better
    });
    assert_eq!(results.len(), 3);
    // (1, [1,0]) has highest dot product with [1,0], so should be first (lowest negated)
    assert_eq!(results[0].key, 1);
}

#[test]
fn nearest_k_fixed_with_manhattan() {
    let vectors: Vec<(u64, [f32; 2])> = vec![(1, [0.0, 0.0]), (2, [5.0, 5.0]), (3, [1.0, 1.0])];
    let query = [0.0f32, 0.0];
    let results = shodh_redb::nearest_k_fixed(vectors.into_iter(), &query, 2, |a, b| {
        shodh_redb::manhattan_distance(a, b)
    });
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].key, 1);
    assert_eq!(results[1].key, 3);
}

// ===========================================================================
// S68 IN-MEMORY BACKEND TTL
// ===========================================================================

#[test]
fn in_memory_ttl_table() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("mem_ttl");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert("live", &1u64).unwrap();
        t.insert_with_ttl("dead", &2u64, Duration::ZERO).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert!(t.get("live").unwrap().is_some());
    assert!(t.get("dead").unwrap().is_none());
}

#[test]
fn in_memory_multiple_tables() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T1: TableDefinition<u64, u64> = TableDefinition::new("mem_t1");
    const T2: TableDefinition<&str, &str> = TableDefinition::new("mem_t2");
    let w = db.begin_write().unwrap();
    {
        let mut t1 = w.open_table(T1).unwrap();
        t1.insert(&1u64, &100u64).unwrap();
        let mut t2 = w.open_table(T2).unwrap();
        t2.insert("hello", "world").unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert_eq!(
        r.open_table(T1)
            .unwrap()
            .get(&1u64)
            .unwrap()
            .unwrap()
            .value(),
        100
    );
    assert_eq!(
        r.open_table(T2)
            .unwrap()
            .get("hello")
            .unwrap()
            .unwrap()
            .value(),
        "world"
    );
}

// ===========================================================================
// S69 STRESS / MIXED OPERATIONS
// ===========================================================================

#[test]
fn stress_interleaved_insert_delete() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stress_id");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
        for i in (0..100u64).step_by(2) {
            t.remove(&i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 50);
    // All remaining keys are odd
    for kv in t.iter().unwrap() {
        let (k, _) = kv.unwrap();
        assert!(k.value() % 2 == 1);
    }
}

#[test]
fn stress_many_transactions() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stress_txn");
    for i in 0..50u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.insert(&i, &(i * i)).unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 50);
    assert_eq!(t.get(&49u64).unwrap().unwrap().value(), 2401);
}

#[test]
fn stress_large_values() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, &[u8]> = TableDefinition::new("stress_lv");
    let big_val = vec![0xABu8; 64 * 1024]; // 64KB
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, big_val.as_slice()).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value().len(), 64 * 1024);
}

// ===========================================================================
// S70 EDGE CASES -- EMPTY AND BOUNDARY
// ===========================================================================

#[test]
fn empty_string_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("empty_str");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("", &42u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("").unwrap().unwrap().value(), 42);
}

#[test]
fn empty_bytes_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, &[u8]> = TableDefinition::new("empty_bytes");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, b"".as_slice()).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert!(t.get(&1u64).unwrap().unwrap().value().is_empty());
}

#[test]
fn table_definition_name() {
    const T: TableDefinition<u64, u64> = TableDefinition::new("my_table");
    assert_eq!(T.name(), "my_table");
}

#[test]
fn multimap_definition_name() {
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("my_mm");
    assert_eq!(T.name(), "my_mm");
}

#[test]
fn get_nonexistent_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("no_key");
    let w = db.begin_write().unwrap();
    {
        w.open_table(T).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert!(t.get(&999u64).unwrap().is_none());
}

#[test]
fn remove_nonexistent_returns_none() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("rm_none");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        assert!(t.remove(&999u64).unwrap().is_none());
    }
    w.commit().unwrap();
}

#[test]
fn insert_returns_old_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("ins_old");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        assert!(t.insert(&1u64, &100u64).unwrap().is_none());
        let old = t.insert(&1u64, &200u64).unwrap();
        assert_eq!(old.unwrap().value(), 100);
    }
    w.commit().unwrap();
}

// ===========================================================================
// S71 REVERSE ITERATION
// ===========================================================================

#[test]
fn iter_reverse_full() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("rev_full");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..5u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let keys: Vec<u64> = t
        .iter()
        .unwrap()
        .rev()
        .map(|e| e.unwrap().0.value())
        .collect();
    assert_eq!(keys, vec![4, 3, 2, 1, 0]);
}

#[test]
fn range_reverse_inclusive() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("rev_inc");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &(i * 10)).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let vals: Vec<u64> = t
        .range(2u64..=5u64)
        .unwrap()
        .rev()
        .map(|e| e.unwrap().1.value())
        .collect();
    assert_eq!(vals, vec![50, 40, 30, 20]);
}

// ===========================================================================
// S72 MULTIMAP RANGE AND ITERATION
// ===========================================================================

#[test]
fn multimap_range_scan() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_range");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        for i in 0..5u64 {
            t.insert(&i, &(i * 10)).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let count = t.range(1u64..4u64).unwrap().count();
    assert_eq!(count, 3);
}

#[test]
fn multimap_duplicate_insert_idempotent() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_dup");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &10u64).unwrap(); // same value again
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let vals: Vec<u64> = t.get(&1u64).unwrap().map(|v| v.unwrap().value()).collect();
    assert_eq!(vals.len(), 1); // multimap deduplicates
}

#[test]
fn multimap_many_values_per_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_many");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        for v in 0..100u64 {
            t.insert(&1u64, &v).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    let count = t.get(&1u64).unwrap().count();
    assert_eq!(count, 100);
}

#[test]
fn multimap_multiple_keys_count() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<&str, u64> = MultimapTableDefinition::new("mm_mk2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert("a", &1u64).unwrap();
        t.insert("a", &2u64).unwrap();
        t.insert("b", &3u64).unwrap();
        t.insert("b", &4u64).unwrap();
        t.insert("b", &5u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    assert_eq!(t.get("a").unwrap().count(), 2);
    assert_eq!(t.get("b").unwrap().count(), 3);
}

// ===========================================================================
// S73 SAVEPOINT PATTERNS
// ===========================================================================

#[test]
fn savepoint_ephemeral_restore() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("sp_eph");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    let sp = w.ephemeral_savepoint().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&2u64, &200u64).unwrap();
    }
    w.commit().unwrap();
    let mut w = db.begin_write().unwrap();
    w.restore_savepoint(&sp).unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert!(t.get(&1u64).unwrap().is_some());
    assert!(t.get(&2u64).unwrap().is_none());
}

#[test]
fn savepoint_restore_multiple_tables() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T1: TableDefinition<u64, u64> = TableDefinition::new("sp_t1");
    const T2: TableDefinition<u64, u64> = TableDefinition::new("sp_t2");
    let w = db.begin_write().unwrap();
    {
        let mut t1 = w.open_table(T1).unwrap();
        t1.insert(&1u64, &10u64).unwrap();
        let mut t2 = w.open_table(T2).unwrap();
        t2.insert(&1u64, &20u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    let sp = w.ephemeral_savepoint().unwrap();
    {
        let mut t1 = w.open_table(T1).unwrap();
        t1.insert(&2u64, &30u64).unwrap();
    }
    w.commit().unwrap();
    let mut w = db.begin_write().unwrap();
    w.restore_savepoint(&sp).unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert!(r.open_table(T1).unwrap().get(&2u64).unwrap().is_none());
    assert_eq!(
        r.open_table(T2)
            .unwrap()
            .get(&1u64)
            .unwrap()
            .unwrap()
            .value(),
        20
    );
}

// ===========================================================================
// S74 TABLE TYPE MIXED KEY/VALUE COMBINATIONS
// ===========================================================================

#[test]
fn str_to_str_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, &str> = TableDefinition::new("s2s");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("greeting", "hello world").unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("greeting").unwrap().unwrap().value(), "hello world");
}

#[test]
fn u32_to_bytes_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u32, &[u8]> = TableDefinition::new("u32b");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&42u32, b"binary-data".as_slice()).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&42u32).unwrap().unwrap().value(), b"binary-data");
}

#[test]
fn u16_key_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u16, u64> = TableDefinition::new("u16k");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&u16::MAX, &42u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&u16::MAX).unwrap().unwrap().value(), 42);
}

#[test]
fn option_str_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, Option<&str>> = TableDefinition::new("opt_str");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &Some("hello")).unwrap();
        t.insert(&2u64, &None::<&str>).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), Some("hello"));
    assert_eq!(t.get(&2u64).unwrap().unwrap().value(), None);
}

// ===========================================================================
// S75 BLOB STORE TYPES -- Sha256Key / DedupVal
// ===========================================================================

#[test]
fn sha256_key_roundtrip() {
    use shodh_redb::blob_store::Sha256Key;
    let key = Sha256Key([0xAB; 32]);
    let bytes = key.to_le_bytes();
    let recovered = Sha256Key::from_le_bytes(bytes);
    assert_eq!(recovered.0, [0xAB; 32]);
}

#[test]
fn sha256_key_serialized_size() {
    assert_eq!(shodh_redb::blob_store::Sha256Key::SERIALIZED_SIZE, 32);
}

#[test]
fn dedup_val_roundtrip() {
    use shodh_redb::blob_store::DedupVal;
    let dv = DedupVal {
        offset: 1024,
        length: 4096,
        checksum: 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0,
        ref_count: 5,
    };
    let bytes = dv.to_le_bytes();
    let recovered = DedupVal::from_le_bytes(bytes);
    assert_eq!(recovered.offset, 1024);
    assert_eq!(recovered.length, 4096);
    assert_eq!(recovered.checksum, dv.checksum);
    assert_eq!(recovered.ref_count, 5);
}

#[test]
fn dedup_val_serialized_size() {
    assert_eq!(shodh_redb::blob_store::DedupVal::SERIALIZED_SIZE, 40);
}

// ===========================================================================
// S76 ADDITIONAL MERGE PATTERNS
// ===========================================================================

#[test]
fn merge_add_across_transactions() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_cross");
    for i in 1..=5u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.merge("counter", &i.to_le_bytes(), &shodh_redb::NumericAdd)
                .unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("counter").unwrap().unwrap().value(), 15);
}

#[test]
fn merge_min_new_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_min_new");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &42u64.to_le_bytes(), &shodh_redb::NumericMin)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 42);
}

#[test]
fn merge_max_new_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_max_new");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &42u64.to_le_bytes(), &shodh_redb::NumericMax)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 42);
}

#[test]
fn merge_bytes_append_multiple() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, &[u8]> = TableDefinition::new("merge_ba_multi");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", b"a", &shodh_redb::BytesAppend).unwrap();
        t.merge("k", b"b", &shodh_redb::BytesAppend).unwrap();
        t.merge("k", b"c", &shodh_redb::BytesAppend).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), b"abc");
}

#[test]
fn merge_bitwise_or_cumulative() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_bor_cum");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("k", &1u64.to_le_bytes(), &shodh_redb::BitwiseOr)
            .unwrap();
        t.merge("k", &2u64.to_le_bytes(), &shodh_redb::BitwiseOr)
            .unwrap();
        t.merge("k", &4u64.to_le_bytes(), &shodh_redb::BitwiseOr)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 7);
}

// ===========================================================================
// S77 TTL TABLE ADVANCED
// ===========================================================================

#[test]
fn ttl_overwrite_with_ttl() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("ttl_ow2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert("k", &1u64).unwrap();
        t.insert_with_ttl("k", &2u64, Duration::from_secs(3600))
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    let guard = t.get("k").unwrap().unwrap();
    assert_eq!(guard.value(), 2);
    assert!(guard.expires_at_ms() > 0);
}

#[test]
fn ttl_bytes_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<&str, &[u8]> = TtlTableDefinition::new("ttl_bytes");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert("data", b"hello".as_slice()).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert_eq!(t.get("data").unwrap().unwrap().value(), b"hello");
}

#[test]
fn ttl_u64_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, &str> = TtlTableDefinition::new("ttl_u64k");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert(&42u64, "answer").unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert_eq!(t.get(&42u64).unwrap().unwrap().value(), "answer");
}

#[test]
fn ttl_purge_multiple_expired() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_purge_m");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        for i in 0..10u64 {
            t.insert_with_ttl(&i, &i, Duration::ZERO).unwrap();
        }
        t.insert(&100u64, &100u64).unwrap(); // this one doesn't expire
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        let purged = t.purge_expired().unwrap();
        assert_eq!(purged, 10);
        assert_eq!(t.len_with_expired().unwrap(), 1);
    }
    w.commit().unwrap();
}

#[test]
fn ttl_iter_reverse() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_rev");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        for i in 0..5u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let t = w.open_ttl_table(T).unwrap();
        let keys: Vec<u64> = t
            .iter()
            .unwrap()
            .rev()
            .map(|e| e.unwrap().0.value())
            .collect();
        assert_eq!(keys, vec![4, 3, 2, 1, 0]);
    }
    w.commit().unwrap();
}

// ===========================================================================
// S78 BLOB TYPES -- ORDERING AND COMPARISON
// ===========================================================================

#[test]
fn blob_id_ordering() {
    let a = shodh_redb::BlobId::new(1, 0);
    let b = shodh_redb::BlobId::new(2, 0);
    assert!(a < b);
}

#[test]
fn blob_id_hash_consistent() {
    use std::collections::HashSet;
    let id = shodh_redb::BlobId::new(1, 2);
    let mut set = HashSet::new();
    set.insert(id);
    set.insert(id); // duplicate
    assert_eq!(set.len(), 1);
}

#[test]
fn blob_id_debug_format() {
    let id = shodh_redb::BlobId::new(42, 99);
    let dbg = format!("{:?}", id);
    assert!(dbg.contains("42"));
}

#[test]
fn causal_edge_all_relation_types() {
    let child = shodh_redb::BlobId::new(1, 1);
    let relations = [
        shodh_redb::RelationType::Derived,
        shodh_redb::RelationType::Similar,
        shodh_redb::RelationType::Contradicts,
        shodh_redb::RelationType::Supports,
        shodh_redb::RelationType::Supersedes,
    ];
    for rel in &relations {
        let edge = shodh_redb::CausalEdge::new(child, *rel, "test");
        let bytes = edge.to_le_bytes();
        let recovered = shodh_redb::CausalEdge::from_le_bytes(bytes);
        assert_eq!(recovered.relation.as_byte(), rel.as_byte());
    }
}

#[test]
fn blob_meta_no_label() {
    let br = shodh_redb::BlobRef {
        offset: 0,
        length: 0,
        checksum: 0,
        ref_count: 1,
        content_type: 0,
        compression: 0,
    };
    let meta = shodh_redb::BlobMeta::new(br, 0, 0, None, "");
    assert_eq!(meta.label_str(), "");
    assert_eq!(meta.label_len, 0);
}

#[test]
fn blob_ref_zero_fields() {
    let br = shodh_redb::BlobRef {
        offset: 0,
        length: 0,
        checksum: 0,
        ref_count: 0,
        content_type: 0,
        compression: 0,
    };
    let bytes = br.to_le_bytes();
    assert_eq!(bytes, [0u8; 40]);
}

// ===========================================================================
// S79 DISTANCE METRIC SYMMETRY PROPERTIES
// ===========================================================================

#[test]
fn euclidean_symmetric() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    assert_eq!(
        shodh_redb::euclidean_distance_sq(&a, &b),
        shodh_redb::euclidean_distance_sq(&b, &a)
    );
}

#[test]
fn manhattan_symmetric() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    assert_eq!(
        shodh_redb::manhattan_distance(&a, &b),
        shodh_redb::manhattan_distance(&b, &a)
    );
}

#[test]
fn dot_product_commutative() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    assert_eq!(
        shodh_redb::dot_product(&a, &b),
        shodh_redb::dot_product(&b, &a)
    );
}

#[test]
fn cosine_distance_symmetric() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    assert!(
        (shodh_redb::cosine_distance(&a, &b) - shodh_redb::cosine_distance(&b, &a)).abs() < 1e-6
    );
}

#[test]
fn hamming_symmetric() {
    let a = [0xABu8, 0xCD];
    let b = [0x12u8, 0x34];
    assert_eq!(
        shodh_redb::hamming_distance(&a, &b),
        shodh_redb::hamming_distance(&b, &a)
    );
}

// ===========================================================================
// S80 EUCLIDEAN TRIANGLE INEQUALITY
// ===========================================================================

#[test]
fn euclidean_triangle_inequality_2d() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 0.0];
    let c = [3.0f32, 4.0];
    let ab = shodh_redb::euclidean_distance_sq(&a, &b).sqrt();
    let bc = shodh_redb::euclidean_distance_sq(&b, &c).sqrt();
    let ac = shodh_redb::euclidean_distance_sq(&a, &c).sqrt();
    assert!(ac <= ab + bc + 1e-6);
}

#[test]
fn manhattan_triangle_inequality_2d() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 0.0];
    let c = [3.0f32, 4.0];
    let ab = shodh_redb::manhattan_distance(&a, &b);
    let bc = shodh_redb::manhattan_distance(&b, &c);
    let ac = shodh_redb::manhattan_distance(&a, &c);
    assert!(ac <= ab + bc + 1e-6);
}

// ===========================================================================
// S81 LARGE DIMENSION VECTORS
// ===========================================================================

#[test]
fn euclidean_high_dim() {
    let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| (i as f32) + 1.0).collect();
    let dist = shodh_redb::euclidean_distance_sq(&a, &b);
    assert_eq!(dist, 256.0); // sum of 256 * 1^2
}

#[test]
fn dot_product_high_dim() {
    let a: Vec<f32> = vec![1.0; 1024];
    let b: Vec<f32> = vec![2.0; 1024];
    assert_eq!(shodh_redb::dot_product(&a, &b), 2048.0);
}

#[test]
fn manhattan_high_dim() {
    let a: Vec<f32> = vec![0.0; 512];
    let b: Vec<f32> = vec![1.0; 512];
    assert_eq!(shodh_redb::manhattan_distance(&a, &b), 512.0);
}

#[test]
fn quantize_binary_high_dim() {
    let v: Vec<f32> = (0..128)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let bq = shodh_redb::quantize_binary(&v);
    assert_eq!(bq.len(), 16); // 128 / 8
}

#[test]
fn hamming_high_dim() {
    let a = vec![0xFFu8; 128];
    let b = vec![0x00u8; 128];
    assert_eq!(shodh_redb::hamming_distance(&a, &b), 1024);
}

// ===========================================================================
// S82 DATABASE STATS
// ===========================================================================

#[test]
fn db_stats_basic() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stats_basic");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let stats = t.stats().unwrap();
    assert!(stats.tree_height() > 0);
    assert!(stats.stored_bytes() > 0);
}

#[test]
fn db_stats_empty_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stats_empty");
    let w = db.begin_write().unwrap();
    {
        w.open_table(T).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let stats = t.stats().unwrap();
    assert_eq!(stats.tree_height(), 0);
}

// ===========================================================================
// S83 CONCURRENT READ TRANSACTIONS
// ===========================================================================

#[test]
fn multiple_concurrent_reads() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("conc_read");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &100u64).unwrap();
    }
    w.commit().unwrap();
    let r1 = db.begin_read().unwrap();
    let r2 = db.begin_read().unwrap();
    let r3 = db.begin_read().unwrap();
    assert_eq!(
        r1.open_table(T)
            .unwrap()
            .get(&1u64)
            .unwrap()
            .unwrap()
            .value(),
        100
    );
    assert_eq!(
        r2.open_table(T)
            .unwrap()
            .get(&1u64)
            .unwrap()
            .unwrap()
            .value(),
        100
    );
    assert_eq!(
        r3.open_table(T)
            .unwrap()
            .get(&1u64)
            .unwrap()
            .unwrap()
            .value(),
        100
    );
}

// ===========================================================================
// S84 HLC ADDITIONAL OPERATIONS
// ===========================================================================

#[test]
fn hlc_tick_overflow_to_physical() {
    // When logical counter is at max, tick should advance physical
    let hlc = shodh_redb::HybridLogicalClock::from_parts(100, u16::MAX);
    let ticked = hlc.tick();
    assert!(ticked.physical_ms() > 100 || ticked.logical() == 0);
}

#[test]
fn hlc_raw_roundtrip_all_bits() {
    let raw = 0xABCD_1234_5678_9ABCu64;
    let hlc = shodh_redb::HybridLogicalClock::from_raw(raw);
    assert_eq!(hlc.to_raw(), raw);
}

#[test]
fn hlc_now_monotonic_successive() {
    let a = shodh_redb::HybridLogicalClock::now();
    let b = shodh_redb::HybridLogicalClock::now();
    assert!(b.to_raw() >= a.to_raw());
}

#[test]
fn hlc_from_parts_fields_check() {
    let hlc = shodh_redb::HybridLogicalClock::from_parts(12345, 67);
    assert_eq!(hlc.physical_ms(), 12345);
    assert_eq!(hlc.logical(), 67);
}

// ===========================================================================
// S85 ADDITIONAL BUILDER TESTS
// ===========================================================================

#[test]
fn builder_open_nonexistent() {
    let result = shodh_redb::Builder::new().open("/tmp/nonexistent_shodh_builder_test.redb");
    assert!(result.is_err());
}

#[test]
fn builder_in_memory_with_cache() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .set_cache_size(1024 * 1024)
        .create_with_backend(backend)
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("mem_cache");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
}

// ===========================================================================
// S86 TABLE OPERATIONS -- RETAIN / POP
// ===========================================================================

#[test]
fn table_retain_even_keys() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("retain");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
        t.retain(|k, _v| k % 2 == 0).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 5);
    for kv in t.iter().unwrap() {
        let (k, _) = kv.unwrap();
        assert_eq!(k.value() % 2, 0);
    }
}

#[test]
fn table_pop_first() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("pop_first");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&2u64, &20u64).unwrap();
        t.insert(&3u64, &30u64).unwrap();
        let first = t.pop_first().unwrap().unwrap();
        assert_eq!(first.0.value(), 1);
        assert_eq!(first.1.value(), 10);
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 2);
}

#[test]
fn table_pop_last() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("pop_last");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&2u64, &20u64).unwrap();
        t.insert(&3u64, &30u64).unwrap();
        let last = t.pop_last().unwrap().unwrap();
        assert_eq!(last.0.value(), 3);
        assert_eq!(last.1.value(), 30);
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 2);
}

#[test]
fn table_pop_first_empty() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("pop_first_empty");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        assert!(t.pop_first().unwrap().is_none());
    }
    w.commit().unwrap();
}

#[test]
fn table_pop_last_empty() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("pop_last_empty");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        assert!(t.pop_last().unwrap().is_none());
    }
    w.commit().unwrap();
}

// ===========================================================================
// S87 READ-ONLY TABLE FIRST/LAST
// ===========================================================================

#[test]
fn read_table_first() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("ro_first");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&5u64, &50u64).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&9u64, &90u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let (k, v) = t.first().unwrap().unwrap();
    assert_eq!(k.value(), 1);
    assert_eq!(v.value(), 10);
}

#[test]
fn read_table_last() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("ro_last");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&5u64, &50u64).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&9u64, &90u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let (k, v) = t.last().unwrap().unwrap();
    assert_eq!(k.value(), 9);
    assert_eq!(v.value(), 90);
}

#[test]
fn read_table_first_empty() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("ro_first_empty");
    let w = db.begin_write().unwrap();
    {
        w.open_table(T).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert!(t.first().unwrap().is_none());
}

// ===========================================================================
// S88 LIST TABLES
// ===========================================================================

#[test]
fn list_tables_after_delete() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T1: TableDefinition<u64, u64> = TableDefinition::new("lt1");
    const T2: TableDefinition<u64, u64> = TableDefinition::new("lt2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T1).unwrap();
        t.insert(&1u64, &1u64).unwrap();
        let mut t = w.open_table(T2).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    w.delete_table(T1).unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let names: Vec<String> = r
        .list_tables()
        .unwrap()
        .map(|h| h.name().to_string())
        .collect();
    assert_eq!(names.len(), 1);
    assert_eq!(names[0], "lt2");
}

// ===========================================================================
// S89 ADDITIONAL VECTOR OPERATIONS
// ===========================================================================

#[test]
fn l2_norm_large_vector() {
    let v: Vec<f32> = vec![1.0; 10000];
    let norm = shodh_redb::l2_norm(&v);
    assert!((norm - 100.0).abs() < 0.01); // sqrt(10000) = 100
}

#[test]
fn cosine_similarity_nearly_parallel() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [1.0f32, 0.001, 0.0];
    let sim = shodh_redb::cosine_similarity(&a, &b);
    assert!(sim > 0.999);
}

#[test]
fn dot_product_negative() {
    let a = [1.0f32, 0.0];
    let b = [-1.0f32, 0.0];
    assert_eq!(shodh_redb::dot_product(&a, &b), -1.0);
}

// ===========================================================================
// S90 STRESS -- MIXED TABLE TYPES
// ===========================================================================

#[test]
fn stress_mixed_types_in_one_db() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T1: TableDefinition<u64, u64> = TableDefinition::new("mix_u64");
    const T2: TableDefinition<&str, &str> = TableDefinition::new("mix_str");
    const T3: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mix_mm");
    const T4: TtlTableDefinition<&str, u64> = TtlTableDefinition::new("mix_ttl");
    let w = db.begin_write().unwrap();
    {
        let mut t1 = w.open_table(T1).unwrap();
        t1.insert(&1u64, &100u64).unwrap();
        let mut t2 = w.open_table(T2).unwrap();
        t2.insert("key", "value").unwrap();
        let mut t3 = w.open_multimap_table(T3).unwrap();
        t3.insert(&1u64, &10u64).unwrap();
        t3.insert(&1u64, &20u64).unwrap();
        let mut t4 = w.open_ttl_table(T4).unwrap();
        t4.insert("cached", &42u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert_eq!(
        r.open_table(T1)
            .unwrap()
            .get(&1u64)
            .unwrap()
            .unwrap()
            .value(),
        100
    );
    assert_eq!(
        r.open_table(T2)
            .unwrap()
            .get("key")
            .unwrap()
            .unwrap()
            .value(),
        "value"
    );
    assert_eq!(
        r.open_multimap_table(T3)
            .unwrap()
            .get(&1u64)
            .unwrap()
            .count(),
        2
    );
    assert_eq!(
        r.open_ttl_table(T4)
            .unwrap()
            .get("cached")
            .unwrap()
            .unwrap()
            .value(),
        42
    );
}

// ===========================================================================
// S91 RETAIN WITH VARIOUS PREDICATES
// ===========================================================================

#[test]
fn retain_all_removes_nothing() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("retain_all");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
        t.retain(|_k, _v| true).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 10);
}

#[test]
fn retain_none_removes_all() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("retain_none");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
        t.retain(|_k, _v| false).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 0);
}

#[test]
fn retain_by_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("retain_val");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &(i * 10)).unwrap();
        }
        t.retain(|_k, v| v >= 50).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 5);
}

// ===========================================================================
// S92 TUPLE KEY TYPES
// ===========================================================================

#[test]
fn tuple_u64_u64_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<(u64, u64), u64> = TableDefinition::new("tup_key");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&(1u64, 2u64), &42u64).unwrap();
        t.insert(&(1u64, 3u64), &43u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&(1u64, 2u64)).unwrap().unwrap().value(), 42);
    assert_eq!(t.get(&(1u64, 3u64)).unwrap().unwrap().value(), 43);
}

#[test]
fn tuple_str_u64_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<(&str, u64), &str> = TableDefinition::new("tup_str");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&("user", 1u64), "alice").unwrap();
        t.insert(&("user", 2u64), "bob").unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&("user", 1u64)).unwrap().unwrap().value(), "alice");
}

#[test]
fn tuple_key_range() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<(u64, u64), u64> = TableDefinition::new("tup_range");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..5u64 {
            for j in 0..3u64 {
                t.insert(&(i, j), &(i * 10 + j)).unwrap();
            }
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let count = t.range((2u64, 0u64)..(4u64, 0u64)).unwrap().count();
    assert_eq!(count, 6); // keys (2,0),(2,1),(2,2),(3,0),(3,1),(3,2)
}

// ===========================================================================
// S93 LARGE SCALE TABLE OPERATIONS
// ===========================================================================

#[test]
fn stress_10000_entries() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stress10k");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10_000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 10_000);
    assert_eq!(t.get(&9999u64).unwrap().unwrap().value(), 9999);
}

#[test]
fn stress_insert_delete_reinsert() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stress_idr");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
        for i in 0..100u64 {
            t.remove(&i).unwrap();
        }
        for i in 0..100u64 {
            t.insert(&i, &(i + 1000)).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 100);
    assert_eq!(t.get(&50u64).unwrap().unwrap().value(), 1050);
}

#[test]
fn stress_100_small_transactions() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("stress_small_txn2");
    for i in 0..100u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.insert(&i, &i).unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 100);
}

// ===========================================================================
// S94 TABLE WITH VARIOUS VALUE SIZES
// ===========================================================================

#[test]
fn value_1kb() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, &[u8]> = TableDefinition::new("val_1kb");
    let data = vec![0x42u8; 1024];
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, data.as_slice()).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value().len(), 1024);
}

#[test]
fn value_1mb() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, &[u8]> = TableDefinition::new("val_1mb");
    let data = vec![0xAB; 1024 * 1024];
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, data.as_slice()).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value().len(), 1024 * 1024);
}

#[test]
fn long_string_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, &str> = TableDefinition::new("long_str");
    let s = "x".repeat(10_000);
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, s.as_str()).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value().len(), 10_000);
}

#[test]
fn many_different_sized_values() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, &[u8]> = TableDefinition::new("varied");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..20u64 {
            let size = (i as usize + 1) * 100;
            let data = vec![i as u8; size];
            t.insert(&i, data.as_slice()).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 20);
}

// ===========================================================================
// S95 CONTENT TYPE EXHAUSTIVE
// ===========================================================================

#[test]
fn content_type_from_byte_0_through_9() {
    for i in 0..10u8 {
        let ct = shodh_redb::ContentType::from_byte(i);
        assert_eq!(ct.as_byte(), i);
    }
}

#[test]
fn content_type_display() {
    let ct = shodh_redb::ContentType::ImagePng;
    let s = format!("{}", ct);
    assert_eq!(s, "image/png");
}

#[test]
fn content_type_debug() {
    let ct = shodh_redb::ContentType::VideoMp4;
    let s = format!("{:?}", ct);
    assert!(s.contains("video/mp4"));
}

// ===========================================================================
// S96 BLOB ID COMPREHENSIVE
// ===========================================================================

#[test]
fn blob_id_copy() {
    let id = shodh_redb::BlobId::new(1, 2);
    let copy = id;
    assert_eq!(id, copy);
}

#[test]
fn blob_id_sequence_zero() {
    let id = shodh_redb::BlobId::new(0, 12345);
    let bytes = id.to_be_bytes();
    let recovered = shodh_redb::BlobId::from_be_bytes(bytes);
    assert_eq!(recovered.sequence, 0);
    assert_eq!(recovered.content_prefix_hash, 12345);
}

#[test]
fn blob_id_ordering_by_sequence() {
    let a = shodh_redb::BlobId::new(1, u64::MAX);
    let b = shodh_redb::BlobId::new(2, 0);
    assert!(a < b); // sequence takes priority
}

// ===========================================================================
// S97 RELATION TYPE EXHAUSTIVE
// ===========================================================================

#[test]
fn relation_type_from_byte_0_through_4() {
    use shodh_redb::RelationType;
    let expected = [0u8, 1, 2, 3, 4];
    for byte in expected {
        let rt = RelationType::from_byte(byte);
        assert_eq!(rt.as_byte(), byte);
    }
}

#[test]
fn relation_type_debug() {
    let rt = shodh_redb::RelationType::Supersedes;
    let dbg = format!("{:?}", rt);
    assert!(dbg.contains("supersedes"));
}

#[test]
fn relation_type_labels_unique() {
    use shodh_redb::RelationType;
    let labels: Vec<&str> = vec![
        RelationType::Derived.label(),
        RelationType::Similar.label(),
        RelationType::Contradicts.label(),
        RelationType::Supports.label(),
        RelationType::Supersedes.label(),
    ];
    for i in 0..labels.len() {
        for j in (i + 1)..labels.len() {
            assert_ne!(labels[i], labels[j]);
        }
    }
}

// ===========================================================================
// S98 CAUSAL EDGE ADDITIONAL
// ===========================================================================

#[test]
fn causal_edge_empty_context() {
    let child = shodh_redb::BlobId::new(1, 1);
    let edge = shodh_redb::CausalEdge::new(child, shodh_redb::RelationType::Derived, "");
    assert_eq!(edge.context_str(), "");
    assert_eq!(edge.context_len, 0);
}

#[test]
fn causal_edge_max_context() {
    let child = shodh_redb::BlobId::new(1, 1);
    let ctx = "a".repeat(62);
    let edge = shodh_redb::CausalEdge::new(child, shodh_redb::RelationType::Similar, &ctx);
    assert_eq!(edge.context_str().len(), 62);
}

#[test]
fn causal_edge_child_preserved() {
    let child = shodh_redb::BlobId::new(42, 99);
    let edge = shodh_redb::CausalEdge::new(child, shodh_redb::RelationType::Supports, "evidence");
    let bytes = edge.to_le_bytes();
    let recovered = shodh_redb::CausalEdge::from_le_bytes(bytes);
    assert_eq!(recovered.child.sequence, 42);
    assert_eq!(recovered.child.content_prefix_hash, 99);
}

// ===========================================================================
// S99 STORE OPTIONS ADDITIONAL
// ===========================================================================

#[test]
fn store_options_with_causal_link() {
    let parent = shodh_redb::BlobId::new(1, 2);
    let link = shodh_redb::CausalLink::derived(parent);
    let opts = shodh_redb::StoreOptions::with_causal_link(link);
    assert!(opts.causal_link.is_some());
    assert_eq!(opts.causal_link.unwrap().parent.sequence, 1);
}

#[test]
fn store_options_empty_tags() {
    let opts = shodh_redb::StoreOptions::with_tags(&[]);
    assert!(opts.tags.is_empty());
}

#[test]
fn store_options_empty_namespace() {
    let opts = shodh_redb::StoreOptions::with_namespace("");
    assert_eq!(opts.namespace.as_deref(), Some(""));
}

// ===========================================================================
// S100 RANGE QUERIES ADDITIONAL
// ===========================================================================

#[test]
fn range_full_scan() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_full");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..20u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.range::<u64>(0..20).unwrap().count(), 20);
}

#[test]
fn range_inclusive_end() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_incl");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let count = t.range(0u64..=9u64).unwrap().count();
    assert_eq!(count, 10);
}

#[test]
fn range_single_element_inclusive_bounds() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("range_single2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    let count = t.range(5u64..=5u64).unwrap().count();
    assert_eq!(count, 1);
}

// ===========================================================================
// S101 MULTIMAP METADATA
// ===========================================================================

#[test]
fn multimap_len_total_pairs() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_len2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_multimap_table(T).unwrap();
        t.insert(&1u64, &10u64).unwrap();
        t.insert(&1u64, &20u64).unwrap();
        t.insert(&2u64, &30u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 3); // 3 total (key,value) pairs
}

#[test]
fn multimap_is_empty() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: MultimapTableDefinition<u64, u64> = MultimapTableDefinition::new("mm_empty2");
    let w = db.begin_write().unwrap();
    {
        w.open_multimap_table(T).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_multimap_table(T).unwrap();
    assert!(t.is_empty().unwrap());
}

// ===========================================================================
// S102 ADDITIONAL QUANTIZATION
// ===========================================================================

#[test]
fn sq_euclidean_different_vectors() {
    let a = [0.0f32, 0.0, 0.0, 0.0];
    let b = [1.0f32, 1.0, 1.0, 1.0];
    let sq_b = shodh_redb::quantize_scalar(&b);
    let dist = shodh_redb::sq_euclidean_distance_sq(&a, &sq_b);
    // Should be approximately 4.0
    assert!((dist - 4.0).abs() < 0.5);
}

#[test]
fn sq_dot_product_orthogonal() {
    let a = [1.0f32, 0.0, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0, 0.0];
    let sq_b = shodh_redb::quantize_scalar(&b);
    let dp = shodh_redb::sq_dot_product(&a, &sq_b);
    assert!(dp.abs() < 0.5); // approximately 0
}

#[test]
fn quantize_scalar_preserves_extremes() {
    let v = [0.0f32, 255.0, 0.0, 255.0];
    let sq = shodh_redb::quantize_scalar(&v);
    let dq = sq.dequantize();
    assert!((dq[0] - 0.0).abs() < 1.0);
    assert!((dq[1] - 255.0).abs() < 1.0);
}

#[test]
fn quantize_binary_large_positive() {
    let v = [100.0f32; 8];
    let bq = shodh_redb::quantize_binary(&v);
    assert_eq!(bq[0], 0xFF);
}

// ===========================================================================
// S103 CROSS-TRANSACTION VISIBILITY
// ===========================================================================

#[test]
fn snapshot_isolation_read_consistency() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("snapshot");
    // Write initial value
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
    // Take read snapshot
    let r = db.begin_read().unwrap();
    // Write new value in separate transaction
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &999u64).unwrap();
    }
    w.commit().unwrap();
    // Old snapshot should still see value 1
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 1);
    // New read sees 999
    let r2 = db.begin_read().unwrap();
    let t2 = r2.open_table(T).unwrap();
    assert_eq!(t2.get(&1u64).unwrap().unwrap().value(), 999);
}

#[test]
fn read_after_multiple_writes() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("multi_write");
    for i in 0..5u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.insert(&1u64, &i).unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get(&1u64).unwrap().unwrap().value(), 4); // last write wins
}

// ===========================================================================
// S104 IN-MEMORY BACKEND ADDITIONAL
// ===========================================================================

#[test]
fn in_memory_stress() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("mem_stress");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        for i in 0..1000u64 {
            t.insert(&i, &(i * 2)).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 1000);
}

#[test]
fn in_memory_list_tables() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T1: TableDefinition<u64, u64> = TableDefinition::new("mem_lt1");
    const T2: TableDefinition<u64, u64> = TableDefinition::new("mem_lt2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T1).unwrap();
        t.insert(&1u64, &1u64).unwrap();
        let mut t = w.open_table(T2).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert_eq!(r.list_tables().unwrap().count(), 2);
}

#[test]
fn in_memory_delete_table() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = shodh_redb::Builder::new()
        .create_with_backend(backend)
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("mem_del2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    assert!(w.delete_table(T).unwrap());
    w.commit().unwrap();
}

// ===========================================================================
// S105 NAMESPACE KEY/VAL ADDITIONAL
// ===========================================================================

#[test]
fn namespace_key_empty_string() {
    use shodh_redb::blob_store::NamespaceKey;
    let nk = NamespaceKey::new("", shodh_redb::BlobId::MIN);
    assert_eq!(nk.namespace_str(), "");
    assert_eq!(nk.ns_len, 0);
}

#[test]
fn namespace_val_empty_string() {
    use shodh_redb::blob_store::NamespaceVal;
    let nv = NamespaceVal::new("");
    assert_eq!(nv.namespace_str(), "");
}

#[test]
fn tag_key_empty_tag() {
    use shodh_redb::blob_store::TagKey;
    let tk = TagKey::new("", shodh_redb::BlobId::MIN);
    assert_eq!(tk.tag_str(), "");
    assert_eq!(tk.tag_len, 0);
}

#[test]
fn temporal_key_zero() {
    use shodh_redb::blob_store::TemporalKey;
    let tk = TemporalKey::new(
        0,
        shodh_redb::HybridLogicalClock::ZERO,
        shodh_redb::BlobId::MIN,
    );
    let bytes = tk.to_be_bytes();
    assert_eq!(bytes, [0u8; 32]);
}

// ===========================================================================
// S106 BLOB META ADDITIONAL
// ===========================================================================

#[test]
fn blob_meta_with_max_label() {
    let br = shodh_redb::BlobRef {
        offset: 0,
        length: 0,
        checksum: 0,
        ref_count: 1,
        content_type: 0,
        compression: 0,
    };
    let label = "b".repeat(63);
    let meta = shodh_redb::BlobMeta::new(br, 0, 0, None, &label);
    assert_eq!(meta.label_str().len(), 63);
    let bytes = meta.to_le_bytes();
    let recovered = shodh_redb::BlobMeta::from_le_bytes(bytes);
    assert_eq!(recovered.label_str(), label);
}

#[test]
fn blob_ref_all_fields_max() {
    let br = shodh_redb::BlobRef {
        offset: u64::MAX,
        length: u64::MAX,
        checksum: u128::MAX,
        ref_count: u32::MAX,
        content_type: 255,
        compression: 255,
    };
    let bytes = br.to_le_bytes();
    let recovered = shodh_redb::BlobRef::from_le_bytes(bytes);
    assert_eq!(recovered.offset, u64::MAX);
    assert_eq!(recovered.length, u64::MAX);
    assert_eq!(recovered.checksum, u128::MAX);
    assert_eq!(recovered.ref_count, u32::MAX);
}

// ===========================================================================
// S107 SIGNAL WEIGHTS / SCORES
// ===========================================================================

#[test]
fn signal_scores_all_none() {
    let scores = shodh_redb::SignalScores {
        semantic: None,
        temporal: None,
        causal: None,
    };
    assert!(scores.semantic.is_none());
    assert!(scores.temporal.is_none());
    assert!(scores.causal.is_none());
}

#[test]
fn signal_scores_mixed() {
    let scores = shodh_redb::SignalScores {
        semantic: Some(0.9),
        temporal: None,
        causal: Some(0.5),
    };
    assert_eq!(scores.semantic, Some(0.9));
    assert!(scores.temporal.is_none());
    assert_eq!(scores.causal, Some(0.5));
}

#[test]
fn signal_weights_all_zero() {
    let w = shodh_redb::SignalWeights {
        semantic: 0.0,
        temporal: 0.0,
        causal: 0.0,
    };
    assert_eq!(w.semantic, 0.0);
}

// ===========================================================================
// S108 MERGE_IN TYPED API
// ===========================================================================

#[test]
fn merge_in_typed_add() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_in");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &10u64).unwrap();
        t.merge_in("k", &5u64, &shodh_redb::NumericAdd).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 15);
}

#[test]
fn merge_in_typed_max() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_in_max");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &50u64).unwrap();
        t.merge_in("k", &30u64, &shodh_redb::NumericMax).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 50);
}

#[test]
fn merge_in_typed_min() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("merge_in_min");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert("k", &50u64).unwrap();
        t.merge_in("k", &30u64, &shodh_redb::NumericMin).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("k").unwrap().unwrap().value(), 30);
}

// ===========================================================================
// S109 CDC CONFIG BUILDER
// ===========================================================================

#[test]
fn builder_with_cdc() {
    let f = tmpfile();
    let cfg = shodh_redb::CdcConfig {
        enabled: true,
        retention_max_txns: 100,
    };
    let db = shodh_redb::Builder::new()
        .set_cdc(cfg)
        .create(f.path())
        .unwrap();
    const T: TableDefinition<u64, u64> = TableDefinition::new("cdc_test");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.insert(&1u64, &1u64).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn cdc_config_clone() {
    let cfg = shodh_redb::CdcConfig {
        enabled: true,
        retention_max_txns: 500,
    };
    let cloned = cfg.clone();
    assert!(cloned.enabled);
    assert_eq!(cloned.retention_max_txns, 500);
}

// ===========================================================================
// S200 CRASH RECOVERY & DURABILITY
// ===========================================================================

#[test]
fn durability_immediate_persists_across_reopen() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    {
        let db = Database::create(&path).unwrap();
        let mut w = db.begin_write().unwrap();
        w.set_durability(Durability::Immediate).unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("dur"))
                .unwrap();
            t.insert(&1, &100).unwrap();
        }
        w.commit().unwrap();
    }
    {
        let db = Database::create(&path).unwrap();
        let r = db.begin_read().unwrap();
        let t = r
            .open_table(TableDefinition::<u64, u64>::new("dur"))
            .unwrap();
        assert_eq!(t.get(&1).unwrap().unwrap().value(), 100);
    }
}

#[test]
fn durability_none_may_not_persist() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let mut w = db.begin_write().unwrap();
    w.set_durability(Durability::None).unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("nondur"))
            .unwrap();
        t.insert(&1, &42).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("nondur"))
        .unwrap();
    assert_eq!(t.get(&1).unwrap().unwrap().value(), 42);
}

#[test]
fn two_phase_commit_enabled() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let mut w = db.begin_write().unwrap();
    w.set_two_phase_commit(true);
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("2pc"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("2pc"))
        .unwrap();
    assert_eq!(t.get(&1).unwrap().unwrap().value(), 1);
}

#[test]
fn two_phase_commit_multi_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let mut w = db.begin_write().unwrap();
    w.set_two_phase_commit(true);
    {
        let mut t1 = w
            .open_table(TableDefinition::<u64, u64>::new("2pc_a"))
            .unwrap();
        t1.insert(&1, &10).unwrap();
    }
    {
        let mut t2 = w
            .open_table(TableDefinition::<&str, &str>::new("2pc_b"))
            .unwrap();
        t2.insert("k", "v").unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t1 = r
        .open_table(TableDefinition::<u64, u64>::new("2pc_a"))
        .unwrap();
    assert_eq!(t1.get(&1).unwrap().unwrap().value(), 10);
    let t2 = r
        .open_table(TableDefinition::<&str, &str>::new("2pc_b"))
        .unwrap();
    assert_eq!(t2.get("k").unwrap().unwrap().value(), "v");
}

#[test]
fn persist_reopen_multiple_tables() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    {
        let db = Database::create(&path).unwrap();
        let w = db.begin_write().unwrap();
        {
            let mut t1 = w
                .open_table(TableDefinition::<u64, u64>::new("tbl_a"))
                .unwrap();
            for i in 0..50 {
                t1.insert(&i, &(i * 10)).unwrap();
            }
        }
        {
            let mut t2 = w
                .open_table(TableDefinition::<&str, u64>::new("tbl_b"))
                .unwrap();
            t2.insert("count", &50).unwrap();
        }
        w.commit().unwrap();
    }
    {
        let db = Database::create(&path).unwrap();
        let r = db.begin_read().unwrap();
        let t1 = r
            .open_table(TableDefinition::<u64, u64>::new("tbl_a"))
            .unwrap();
        assert_eq!(t1.len().unwrap(), 50);
        assert_eq!(t1.get(&49).unwrap().unwrap().value(), 490);
        let t2 = r
            .open_table(TableDefinition::<&str, u64>::new("tbl_b"))
            .unwrap();
        assert_eq!(t2.get("count").unwrap().unwrap().value(), 50);
    }
}

#[test]
fn abort_does_not_persist() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("ab"))
                .unwrap();
            t.insert(&1, &1).unwrap();
        }
        w.commit().unwrap();
    }
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("ab"))
                .unwrap();
            t.insert(&2, &2).unwrap();
        }
        w.abort().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("ab"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 1);
    assert!(t.get(&2).unwrap().is_none());
}

#[test]
fn drop_without_commit_rolls_back() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("drop_rb"))
                .unwrap();
            t.insert(&1, &1).unwrap();
        }
        w.commit().unwrap();
    }
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("drop_rb"))
                .unwrap();
            t.insert(&2, &2).unwrap();
        }
        drop(w);
    }
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("drop_rb"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 1);
}

#[test]
fn reopen_after_many_commits() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    {
        let db = Database::create(&path).unwrap();
        for i in 0u64..100 {
            let w = db.begin_write().unwrap();
            {
                let mut t = w
                    .open_table(TableDefinition::<u64, u64>::new("many_c"))
                    .unwrap();
                t.insert(&i, &i).unwrap();
            }
            w.commit().unwrap();
        }
    }
    {
        let db = Database::create(&path).unwrap();
        let r = db.begin_read().unwrap();
        let t = r
            .open_table(TableDefinition::<u64, u64>::new("many_c"))
            .unwrap();
        assert_eq!(t.len().unwrap(), 100);
        for i in 0u64..100 {
            assert_eq!(t.get(&i).unwrap().unwrap().value(), i);
        }
    }
}

#[test]
fn check_integrity_clean_db() {
    let f = tmpfile();
    let mut db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("integ"))
            .unwrap();
        for i in 0..100 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    assert!(db.check_integrity().unwrap());
}

#[test]
fn check_integrity_empty_db() {
    let f = tmpfile();
    let mut db = Database::create(f.path()).unwrap();
    assert!(db.check_integrity().unwrap());
}

#[test]
fn compact_reduces_or_maintains_size() {
    let f = tmpfile();
    let mut db = Database::create(f.path()).unwrap();
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, &[u8]>::new("compact"))
                .unwrap();
            let data = vec![0xFFu8; 1024];
            for i in 0..200u64 {
                t.insert(&i, data.as_slice()).unwrap();
            }
        }
        w.commit().unwrap();
    }
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, &[u8]>::new("compact"))
                .unwrap();
            for i in 0..100u64 {
                t.remove(&i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    let _compacted = db.compact().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, &[u8]>::new("compact"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 100);
}

#[test]
fn incremental_compaction_steps() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("incr"))
                .unwrap();
            for i in 0..500u64 {
                t.insert(&i, &i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("incr"))
                .unwrap();
            for i in 0..250u64 {
                t.remove(&i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    let handle = db.start_compaction().unwrap();
    let mut steps = 0u64;
    loop {
        let progress = handle.step().unwrap();
        steps += 1;
        if progress.complete {
            break;
        }
        if steps > 10000 {
            panic!("compaction did not complete in 10000 steps");
        }
    }
    assert!(steps >= 1);
}

// ===========================================================================
// S210 TRANSACTION ISOLATION
// ===========================================================================

#[test]
fn snapshot_isolation_read_does_not_see_concurrent_write() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("iso"))
            .unwrap();
        t.insert(&1, &100).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let w2 = db.begin_write().unwrap();
    {
        let mut t = w2
            .open_table(TableDefinition::<u64, u64>::new("iso"))
            .unwrap();
        t.insert(&1, &200).unwrap();
        t.insert(&2, &300).unwrap();
    }
    w2.commit().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("iso"))
        .unwrap();
    assert_eq!(t.get(&1).unwrap().unwrap().value(), 100);
    assert!(t.get(&2).unwrap().is_none());
}

#[test]
fn read_after_commit_sees_new_data() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rac"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let w2 = db.begin_write().unwrap();
    {
        let mut t = w2
            .open_table(TableDefinition::<u64, u64>::new("rac"))
            .unwrap();
        t.insert(&1, &2).unwrap();
    }
    w2.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("rac"))
        .unwrap();
    assert_eq!(t.get(&1).unwrap().unwrap().value(), 2);
}

#[test]
fn multiple_readers_see_same_snapshot() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("mr"))
            .unwrap();
        t.insert(&1, &42).unwrap();
    }
    w.commit().unwrap();
    let r1 = db.begin_read().unwrap();
    let r2 = db.begin_read().unwrap();
    let t1 = r1
        .open_table(TableDefinition::<u64, u64>::new("mr"))
        .unwrap();
    let t2 = r2
        .open_table(TableDefinition::<u64, u64>::new("mr"))
        .unwrap();
    assert_eq!(
        t1.get(&1).unwrap().unwrap().value(),
        t2.get(&1).unwrap().unwrap().value()
    );
}

#[test]
fn delete_not_visible_until_commit() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("dnv"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let w2 = db.begin_write().unwrap();
    {
        let mut t = w2
            .open_table(TableDefinition::<u64, u64>::new("dnv"))
            .unwrap();
        t.remove(&1).unwrap();
    }
    w2.commit().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("dnv"))
        .unwrap();
    assert!(t.get(&1).unwrap().is_some());
}

#[test]
fn isolation_across_table_creation() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let r = db.begin_read().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("new_tbl"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    assert!(
        r.open_table(TableDefinition::<u64, u64>::new("new_tbl"))
            .is_err()
    );
}

#[test]
fn isolation_delete_table_not_visible_to_prior_reader() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("del_iso"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let w2 = db.begin_write().unwrap();
    w2.delete_table(TableDefinition::<u64, u64>::new("del_iso"))
        .unwrap();
    w2.commit().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("del_iso"))
        .unwrap();
    assert_eq!(t.get(&1).unwrap().unwrap().value(), 1);
}

#[test]
fn sequential_transactions_each_see_prior() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    for i in 0u64..10 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("seq"))
                .unwrap();
            t.insert(&i, &i).unwrap();
        }
        w.commit().unwrap();
        let r = db.begin_read().unwrap();
        let t = r
            .open_table(TableDefinition::<u64, u64>::new("seq"))
            .unwrap();
        assert_eq!(t.len().unwrap(), i + 1);
    }
}

#[test]
fn overwrite_preserves_isolation() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("ow"))
            .unwrap();
        t.insert(&1, &100).unwrap();
    }
    w.commit().unwrap();
    let r_old = db.begin_read().unwrap();
    let w2 = db.begin_write().unwrap();
    {
        let mut t = w2
            .open_table(TableDefinition::<u64, u64>::new("ow"))
            .unwrap();
        t.insert(&1, &200).unwrap();
    }
    w2.commit().unwrap();
    let r_new = db.begin_read().unwrap();
    let t_old = r_old
        .open_table(TableDefinition::<u64, u64>::new("ow"))
        .unwrap();
    let t_new = r_new
        .open_table(TableDefinition::<u64, u64>::new("ow"))
        .unwrap();
    assert_eq!(t_old.get(&1).unwrap().unwrap().value(), 100);
    assert_eq!(t_new.get(&1).unwrap().unwrap().value(), 200);
}

// ===========================================================================
// S220 SAVEPOINTS -- DEPTH & PERSISTENCE
// ===========================================================================

#[test]
fn persistent_savepoint_survives_reopen() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    let sp_id;
    {
        let db = Database::create(&path).unwrap();
        {
            let w = db.begin_write().unwrap();
            {
                let mut t = w
                    .open_table(TableDefinition::<u64, u64>::new("psp"))
                    .unwrap();
                t.insert(&1, &100).unwrap();
            }
            w.commit().unwrap();
        }
        let w = db.begin_write().unwrap();
        sp_id = w.persistent_savepoint().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("psp"))
                .unwrap();
            t.insert(&2, &200).unwrap();
        }
        w.commit().unwrap();
    }
    {
        let db = Database::create(&path).unwrap();
        let w = db.begin_write().unwrap();
        let sps: Vec<u64> = w.list_persistent_savepoints().unwrap().collect();
        assert!(sps.contains(&sp_id));
        w.commit().unwrap();
    }
}

#[test]
fn delete_persistent_savepoint() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("dps"))
                .unwrap();
            t.insert(&1, &1).unwrap();
        }
        w.commit().unwrap();
    }
    let w = db.begin_write().unwrap();
    let sp_id = w.persistent_savepoint().unwrap();
    w.commit().unwrap();
    let w2 = db.begin_write().unwrap();
    assert!(w2.delete_persistent_savepoint(sp_id).unwrap());
    assert!(!w2.delete_persistent_savepoint(sp_id).unwrap());
    w2.commit().unwrap();
}

#[test]
fn ephemeral_savepoint_restore() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    // Txn 1: seed data
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("esr"))
            .unwrap();
        t.insert(&1, &10).unwrap();
    }
    w.commit().unwrap();
    // Txn 2: create savepoint then commit more data
    let w = db.begin_write().unwrap();
    let sp = w.ephemeral_savepoint().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("esr"))
            .unwrap();
        t.insert(&2, &20).unwrap();
        t.insert(&3, &30).unwrap();
    }
    w.commit().unwrap();
    // Txn 3: restore savepoint -- rolls back txn 2's writes
    let mut w = db.begin_write().unwrap();
    w.restore_savepoint(&sp).unwrap();
    w.commit().unwrap();
    // Verify: only key=1 from txn 1 remains
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("esr"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 1);
    assert!(t.get(&2).unwrap().is_none());
}

#[test]
fn savepoint_restore_then_continue_writing() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    // Txn 1: seed data
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("src"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    // Txn 2: create savepoint, then add key=2
    let w = db.begin_write().unwrap();
    let sp = w.ephemeral_savepoint().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("src"))
            .unwrap();
        t.insert(&2, &2).unwrap();
    }
    w.commit().unwrap();
    // Txn 3: restore savepoint (undoes key=2), then write key=3
    let mut w = db.begin_write().unwrap();
    w.restore_savepoint(&sp).unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("src"))
            .unwrap();
        t.insert(&3, &3).unwrap();
    }
    w.commit().unwrap();
    // Verify: key=1 (from txn 1), key=3 (from txn 3), no key=2
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("src"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 2);
    assert!(t.get(&1).unwrap().is_some());
    assert!(t.get(&2).unwrap().is_none());
    assert!(t.get(&3).unwrap().is_some());
}

#[test]
fn multiple_savepoints_nested() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    // Txn 1: seed data (key=1)
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("msn"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    // Txn 2: create sp1 (captures state: key=1), then add key=2
    let w = db.begin_write().unwrap();
    let sp1 = w.ephemeral_savepoint().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("msn"))
            .unwrap();
        t.insert(&2, &2).unwrap();
    }
    w.commit().unwrap();
    // Txn 3: create sp2 (captures state: key=1,2), then add key=3
    let w = db.begin_write().unwrap();
    let sp2 = w.ephemeral_savepoint().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("msn"))
            .unwrap();
        t.insert(&3, &3).unwrap();
    }
    w.commit().unwrap();
    // Txn 4: restore sp2 (undoes key=3, keeps key=1,2)
    let mut w = db.begin_write().unwrap();
    w.restore_savepoint(&sp2).unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("msn"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 2);
    drop(t);
    drop(r);
    // Txn 5: restore sp1 (undoes key=2 too, keeps only key=1)
    let mut w = db.begin_write().unwrap();
    w.restore_savepoint(&sp1).unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("msn"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 1);
}

#[test]
fn savepoint_across_multiple_tables() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    // Txn 1: seed both tables
    let w = db.begin_write().unwrap();
    {
        let mut t1 = w
            .open_table(TableDefinition::<u64, u64>::new("sp_a"))
            .unwrap();
        t1.insert(&1, &10).unwrap();
    }
    {
        let mut t2 = w
            .open_table(TableDefinition::<u64, u64>::new("sp_b"))
            .unwrap();
        t2.insert(&1, &20).unwrap();
    }
    w.commit().unwrap();
    // Txn 2: create savepoint then modify both tables
    let w = db.begin_write().unwrap();
    let sp = w.ephemeral_savepoint().unwrap();
    {
        let mut t1 = w
            .open_table(TableDefinition::<u64, u64>::new("sp_a"))
            .unwrap();
        t1.insert(&2, &30).unwrap();
    }
    {
        let mut t2 = w
            .open_table(TableDefinition::<u64, u64>::new("sp_b"))
            .unwrap();
        t2.remove(&1).unwrap();
    }
    w.commit().unwrap();
    // Txn 3: restore savepoint -- undoes all txn 2 changes across both tables
    let mut w = db.begin_write().unwrap();
    w.restore_savepoint(&sp).unwrap();
    w.commit().unwrap();
    // Verify both tables restored
    let r = db.begin_read().unwrap();
    let t1 = r
        .open_table(TableDefinition::<u64, u64>::new("sp_a"))
        .unwrap();
    assert_eq!(t1.len().unwrap(), 1);
    drop(t1);
    let t2 = r
        .open_table(TableDefinition::<u64, u64>::new("sp_b"))
        .unwrap();
    assert_eq!(t2.len().unwrap(), 1);
}

// ===========================================================================
// S230 CONCURRENT ACCESS
// ===========================================================================

#[test]
fn concurrent_readers_10_threads() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cr10"))
            .unwrap();
        for i in 0..1000u64 {
            t.insert(&i, &(i * 2)).unwrap();
        }
    }
    w.commit().unwrap();
    let db = Arc::new(db);
    let barrier = Arc::new(Barrier::new(10));
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let r = db.begin_read().unwrap();
                let t = r
                    .open_table(TableDefinition::<u64, u64>::new("cr10"))
                    .unwrap();
                assert_eq!(t.len().unwrap(), 1000);
                for i in 0..1000u64 {
                    assert_eq!(t.get(&i).unwrap().unwrap().value(), i * 2);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn reader_survives_concurrent_writer() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rscw"))
            .unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let w2 = db.begin_write().unwrap();
    {
        let mut t = w2
            .open_table(TableDefinition::<u64, u64>::new("rscw"))
            .unwrap();
        for i in 100..200u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w2.commit().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("rscw"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 100);
}

#[test]
fn sequential_writers_from_threads() {
    let f = tmpfile();
    let db = Arc::new(Database::create(f.path()).unwrap());
    for i in 0u64..20 {
        let db = Arc::clone(&db);
        let h = std::thread::spawn(move || {
            let w = db.begin_write().unwrap();
            {
                let mut t = w
                    .open_table(TableDefinition::<u64, u64>::new("swt"))
                    .unwrap();
                t.insert(&i, &i).unwrap();
            }
            w.commit().unwrap();
        });
        h.join().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("swt"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 20);
}

#[test]
fn readers_during_bulk_write() {
    let f = tmpfile();
    let db = Arc::new(Database::create(f.path()).unwrap());
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("rdbw"))
                .unwrap();
            for i in 0..500u64 {
                t.insert(&i, &i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    let db2 = Arc::clone(&db);
    let reader = std::thread::spawn(move || {
        for _ in 0..5 {
            let r = db2.begin_read().unwrap();
            let t = r
                .open_table(TableDefinition::<u64, u64>::new("rdbw"))
                .unwrap();
            let len = t.len().unwrap();
            assert!(len >= 500);
        }
    });
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("rdbw"))
                .unwrap();
            for i in 500..1000u64 {
                t.insert(&i, &i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    reader.join().unwrap();
}

// ===========================================================================
// S240 BUILDER CONFIGURATION & MEMORY BUDGET
// ===========================================================================

#[test]
fn builder_cache_size_4mb() {
    let f = tmpfile();
    let db = Builder::new()
        .set_cache_size(4 * 1024 * 1024)
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("bcs"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn builder_small_cache() {
    let f = tmpfile();
    let db = Builder::new()
        .set_cache_size(64 * 1024)
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("bsc"))
            .unwrap();
        for i in 0..1000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("bsc"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 1000);
}

#[test]
fn builder_read_verification_full() {
    let f = tmpfile();
    let db = Builder::new()
        .set_read_verification(ReadVerification::Full)
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rvf"))
            .unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("rvf"))
        .unwrap();
    for i in 0..100u64 {
        assert_eq!(t.get(&i).unwrap().unwrap().value(), i);
    }
}

#[test]
fn builder_read_verification_sampled() {
    let f = tmpfile();
    let db = Builder::new()
        .set_read_verification(ReadVerification::Sampled { rate: 0.5 })
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rvs"))
            .unwrap();
        for i in 0..50u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("rvs"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 50);
}

#[test]
fn builder_read_verification_none() {
    let f = tmpfile();
    let db = Builder::new()
        .set_read_verification(ReadVerification::None)
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rvn"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn builder_blob_dedup_enabled() {
    let f = tmpfile();
    let db = Builder::new()
        .set_blob_dedup(true)
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("bde"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn builder_history_retention() {
    let f = tmpfile();
    let db = Builder::new()
        .set_history_retention(10)
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("bhr"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn builder_cdc_enabled() {
    let f = tmpfile();
    let db = Builder::new()
        .set_cdc(shodh_redb::CdcConfig {
            enabled: true,
            retention_max_txns: 100,
        })
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("bcdc"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn builder_memory_budget() {
    let f = tmpfile();
    let db = Builder::new()
        .set_memory_budget(16 * 1024 * 1024)
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("bmb"))
            .unwrap();
        for i in 0..500u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
}

#[test]
fn builder_all_options_combined() {
    let f = tmpfile();
    let db = Builder::new()
        .set_cache_size(8 * 1024 * 1024)
        .set_memory_budget(32 * 1024 * 1024)
        .set_read_verification(ReadVerification::Sampled { rate: 0.1 })
        .set_blob_dedup(true)
        .set_history_retention(5)
        .set_cdc(shodh_redb::CdcConfig {
            enabled: true,
            retention_max_txns: 50,
        })
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("bao"))
            .unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("bao"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 100);
}

// ===========================================================================
// S250 DATABASE STATS
// ===========================================================================

#[test]
fn database_stats_after_inserts() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("stats"))
            .unwrap();
        for i in 0..1000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let w2 = db.begin_write().unwrap();
    let stats: DatabaseStats = w2.stats().unwrap();
    assert!(stats.tree_height() >= 1);
    assert!(stats.allocated_pages() > 0);
    assert!(stats.leaf_pages() > 0);
    assert!(stats.stored_bytes() > 0);
    assert!(stats.page_size() > 0);
    w2.abort().unwrap();
}

#[test]
fn database_stats_empty() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let stats = w.stats().unwrap();
    assert_eq!(stats.tree_height(), 0);
    w.abort().unwrap();
}

#[test]
fn database_stats_after_delete() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("sdel"))
            .unwrap();
        for i in 0..500u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    let stats1 = w.stats().unwrap();
    w.commit().unwrap();
    let w2 = db.begin_write().unwrap();
    {
        let mut t = w2
            .open_table(TableDefinition::<u64, u64>::new("sdel"))
            .unwrap();
        for i in 0..250u64 {
            t.remove(&i).unwrap();
        }
    }
    let stats2 = w2.stats().unwrap();
    assert!(stats2.stored_bytes() <= stats1.stored_bytes());
    w2.commit().unwrap();
}

#[test]
fn database_stats_free_pages_after_compact() {
    let f = tmpfile();
    let mut db = Database::create(f.path()).unwrap();
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, &[u8]>::new("sfp"))
                .unwrap();
            let data = vec![0u8; 512];
            for i in 0..200u64 {
                t.insert(&i, data.as_slice()).unwrap();
            }
        }
        w.commit().unwrap();
    }
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, &[u8]>::new("sfp"))
                .unwrap();
            for i in 0..200u64 {
                t.remove(&i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    db.compact().unwrap();
    let w = db.begin_write().unwrap();
    let _stats = w.stats().unwrap();
    w.abort().unwrap();
}

// ===========================================================================
// S260 LARGE SCALE DATA
// ===========================================================================

#[test]
fn large_scale_10k_entries() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("ls10k"))
            .unwrap();
        for i in 0..10_000u64 {
            t.insert(&i, &(i * 3)).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("ls10k"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 10_000);
    assert_eq!(t.get(&0).unwrap().unwrap().value(), 0);
    assert_eq!(t.get(&9999).unwrap().unwrap().value(), 29997);
}

#[test]
fn large_scale_50k_entries() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("ls50k"))
            .unwrap();
        for i in 0..50_000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("ls50k"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 50_000);
    assert_eq!(t.first().unwrap().unwrap().1.value(), 0);
    assert_eq!(t.last().unwrap().unwrap().1.value(), 49_999);
}

#[test]
fn large_scale_range_scan_10k() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("lsr"))
            .unwrap();
        for i in 0..10_000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("lsr"))
        .unwrap();
    let count = t.range(5000..6000).unwrap().count();
    assert_eq!(count, 1000);
}

#[test]
fn large_scale_bulk_delete() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("lsbd"))
                .unwrap();
            for i in 0..10_000u64 {
                t.insert(&i, &i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("lsbd"))
                .unwrap();
            for i in 0..5000u64 {
                t.remove(&i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("lsbd"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 5000);
    assert_eq!(t.first().unwrap().unwrap().0.value(), 5000);
}

#[test]
fn large_scale_string_keys_5k() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<&str, u64>::new("lssk"))
            .unwrap();
        for i in 0..5000u64 {
            let key = format!("key_{i:06}");
            t.insert(key.as_str(), &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<&str, u64>::new("lssk"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 5000);
    assert_eq!(t.get("key_000000").unwrap().unwrap().value(), 0);
    assert_eq!(t.get("key_004999").unwrap().unwrap().value(), 4999);
}

#[test]
fn large_values_4kb_x_1000() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let data = vec![0xABu8; 4096];
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, &[u8]>::new("lv4k"))
            .unwrap();
        for i in 0..1000u64 {
            t.insert(&i, data.as_slice()).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, &[u8]>::new("lv4k"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 1000);
    let v = t.get(&500).unwrap().unwrap();
    assert_eq!(v.value().len(), 4096);
    assert!(v.value().iter().all(|&b| b == 0xAB));
}

#[test]
fn large_scale_multimap_10k_pairs() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("lsmm"))
            .unwrap();
        for i in 0..100u64 {
            for j in 0..100u64 {
                t.insert(&i, &(i * 100 + j)).unwrap();
            }
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("lsmm"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 10_000);
}

#[test]
fn large_scale_many_tables_50() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    for i in 0..50u32 {
        let name = format!("table_{i:03}");
        let def = TableDefinition::<u64, u64>::new(&*Box::leak(name.into_boxed_str()));
        let mut t = w.open_table(def).unwrap();
        t.insert(&(i as u64), &(i as u64)).unwrap();
    }
    w.commit().unwrap();
    let w2 = db.begin_write().unwrap();
    let tables: Vec<_> = w2.list_tables().unwrap().collect();
    assert!(tables.len() >= 50);
    w2.commit().unwrap();
}

#[test]
fn large_scale_insert_delete_reinsert_cycle() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    for round in 0..5u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("lscyc"))
                .unwrap();
            let base = round * 1000;
            for i in base..base + 1000 {
                t.insert(&i, &i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("lscyc"))
            .unwrap();
        for i in 0..2500u64 {
            t.remove(&i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("lscyc"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 2500);
}

// ===========================================================================
// S270 ERROR HANDLING & EDGE CASES
// ===========================================================================

#[test]
fn table_type_mismatch_is_error() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("tm"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let w2 = db.begin_write().unwrap();
    let result = w2.open_table(TableDefinition::<&str, &str>::new("tm"));
    assert!(result.is_err());
    drop(result);
    w2.abort().unwrap();
}

#[test]
fn open_multimap_as_regular_is_error() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("mm_err"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let w2 = db.begin_write().unwrap();
    let result = w2.open_table(TableDefinition::<u64, u64>::new("mm_err"));
    assert!(result.is_err());
    drop(result);
    w2.abort().unwrap();
}

#[test]
fn open_regular_as_multimap_is_error() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("reg_err"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let w2 = db.begin_write().unwrap();
    let result = w2.open_multimap_table(MultimapTableDefinition::<u64, u64>::new("reg_err"));
    assert!(result.is_err());
    drop(result);
    w2.abort().unwrap();
}

#[test]
fn delete_nonexistent_table_returns_false() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let deleted = w
        .delete_table(TableDefinition::<u64, u64>::new("nonexistent"))
        .unwrap();
    assert!(!deleted);
    w.commit().unwrap();
}

#[test]
fn delete_existing_table_returns_true() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("to_del"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let w2 = db.begin_write().unwrap();
    assert!(
        w2.delete_table(TableDefinition::<u64, u64>::new("to_del"))
            .unwrap()
    );
    w2.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert!(
        r.open_table(TableDefinition::<u64, u64>::new("to_del"))
            .is_err()
    );
}

#[test]
fn get_nonexistent_key_returns_none() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("gnk"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("gnk"))
        .unwrap();
    assert!(t.get(&999).unwrap().is_none());
}

#[test]
fn remove_nonexistent_key_returns_none() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rnk"))
            .unwrap();
        let removed = t.remove(&999).unwrap();
        assert!(removed.is_none());
    }
    w.commit().unwrap();
}

#[test]
fn insert_returns_previous_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("irp"))
            .unwrap();
        let prev = t.insert(&1, &100).unwrap();
        assert!(prev.is_none());
        drop(prev);
        let prev2 = t.insert(&1, &200).unwrap();
        assert_eq!(prev2.unwrap().value(), 100);
    }
    w.commit().unwrap();
}

#[test]
fn range_on_empty_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let t = w
            .open_table(TableDefinition::<u64, u64>::new("ret"))
            .unwrap();
        assert_eq!(t.range(0..100).unwrap().count(), 0);
    }
    w.commit().unwrap();
}

#[test]
fn range_beyond_data() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rbd"))
            .unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
        assert_eq!(t.range(100..200).unwrap().count(), 0);
    }
    w.commit().unwrap();
}

#[test]
fn first_last_on_empty_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let t = w
            .open_table(TableDefinition::<u64, u64>::new("flet"))
            .unwrap();
        assert!(t.first().unwrap().is_none());
        assert!(t.last().unwrap().is_none());
    }
    w.commit().unwrap();
}

#[test]
fn pop_first_pop_last() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("pfpl"))
            .unwrap();
        for i in 0..5u64 {
            t.insert(&i, &i).unwrap();
        }
        let first = t.pop_first().unwrap().unwrap();
        assert_eq!(first.0.value(), 0);
        drop(first);
        let last = t.pop_last().unwrap().unwrap();
        assert_eq!(last.0.value(), 4);
        drop(last);
        assert_eq!(t.len().unwrap(), 3);
    }
    w.commit().unwrap();
}

#[test]
fn extract_if_removes_matching() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("eif"))
            .unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
        // extract_if returns AccessGuards that borrow the table; count and drop them
        let count = t.extract_if(|k, _v| k % 2 == 0).unwrap().count();
        assert_eq!(count, 5);
        assert_eq!(t.len().unwrap(), 5);
    }
    w.commit().unwrap();
}

#[test]
fn retain_keeps_matching() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rk"))
            .unwrap();
        for i in 0..20u64 {
            t.insert(&i, &i).unwrap();
        }
        t.retain(|k, _v| k >= 10).unwrap();
        assert_eq!(t.len().unwrap(), 10);
        assert_eq!(t.first().unwrap().unwrap().0.value(), 10);
    }
    w.commit().unwrap();
}

// ===========================================================================
// S280 IN-MEMORY BACKEND
// ===========================================================================

#[test]
fn in_memory_basic_crud() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("im"))
            .unwrap();
        t.insert(&1, &10).unwrap();
        t.insert(&2, &20).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("im"))
        .unwrap();
    assert_eq!(t.get(&1).unwrap().unwrap().value(), 10);
    assert_eq!(t.len().unwrap(), 2);
}

#[test]
fn in_memory_multiple_transactions() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();
    for i in 0u64..50 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("immt"))
                .unwrap();
            t.insert(&i, &i).unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("immt"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 50);
}

#[test]
fn in_memory_multimap() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("imm"))
            .unwrap();
        t.insert(&1, &10).unwrap();
        t.insert(&1, &20).unwrap();
        t.insert(&2, &30).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("imm"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 3);
}

#[test]
fn in_memory_savepoint() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();
    // Txn 1: seed data
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("imsp"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    // Txn 2: create savepoint then add key=2
    let w = db.begin_write().unwrap();
    let sp = w.ephemeral_savepoint().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("imsp"))
            .unwrap();
        t.insert(&2, &2).unwrap();
    }
    w.commit().unwrap();
    // Txn 3: restore savepoint -- undoes key=2
    let mut w = db.begin_write().unwrap();
    w.restore_savepoint(&sp).unwrap();
    w.commit().unwrap();
    // Verify only key=1 remains
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("imsp"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 1);
}

#[test]
fn in_memory_large_dataset() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = Builder::new().create_with_backend(backend).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("imld"))
            .unwrap();
        for i in 0..10_000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("imld"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 10_000);
}

#[test]
fn in_memory_concurrent_readers() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let db = Arc::new(Builder::new().create_with_backend(backend).unwrap());
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("imcr"))
                .unwrap();
            for i in 0..100u64 {
                t.insert(&i, &i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    let handles: Vec<_> = (0..5)
        .map(|_| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                let r = db.begin_read().unwrap();
                let t = r
                    .open_table(TableDefinition::<u64, u64>::new("imcr"))
                    .unwrap();
                assert_eq!(t.len().unwrap(), 100);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn in_memory_check_integrity() {
    let backend = shodh_redb::backends::InMemoryBackend::new();
    let mut db = Builder::new().create_with_backend(backend).unwrap();
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("imci"))
                .unwrap();
            for i in 0..100u64 {
                t.insert(&i, &i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    assert!(db.check_integrity().unwrap());
}

// ===========================================================================
// S290 TTL DEEP TESTS
// ===========================================================================

#[test]
fn ttl_insert_many_with_varying_expiry() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_vary");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        for i in 0..100u64 {
            t.insert_with_ttl(&i, &i, Duration::from_secs(i + 1))
                .unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert_eq!(t.len_with_expired().unwrap(), 100);
}

#[test]
fn ttl_no_expiry_always_visible() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_noexp");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert(&1, &42).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    let guard = t.get(&1).unwrap().unwrap();
    assert_eq!(guard.value(), 42);
    assert_eq!(guard.expires_at_ms(), 0);
}

#[test]
fn ttl_overwrite_resets_expiry() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_ow");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert_with_ttl(&1, &10, Duration::from_secs(60)).unwrap();
        t.insert_with_ttl(&1, &20, Duration::from_secs(3600))
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    let guard = t.get(&1).unwrap().unwrap();
    assert_eq!(guard.value(), 20);
    assert!(guard.expires_at_ms() > 0);
}

#[test]
fn ttl_remove_entry() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_rm");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        t.insert(&1, &1).unwrap();
        t.insert(&2, &2).unwrap();
        t.remove(&1).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert!(t.get(&1).unwrap().is_none());
    assert!(t.get(&2).unwrap().is_some());
}

#[test]
fn ttl_range_query_50_entries() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_rq");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        for i in 0..50u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    let count = t.range(10..20).unwrap().count();
    assert_eq!(count, 10);
}

#[test]
fn ttl_iter_full_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TtlTableDefinition<u64, u64> = TtlTableDefinition::new("ttl_it");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_ttl_table(T).unwrap();
        for i in 0..25u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_ttl_table(T).unwrap();
    assert_eq!(t.iter().unwrap().count(), 25);
}

// ===========================================================================
// S300 MERGE OPERATOR DEPTH
// ===========================================================================

#[test]
fn merge_add_100_times() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("m100");
    for _ in 0..100 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.merge("counter", &1u64.to_le_bytes(), &shodh_redb::NumericAdd)
                .unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("counter").unwrap().unwrap().value(), 100);
}

#[test]
fn merge_max_across_transactions() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("mmax");
    let values = [10u64, 50, 30, 80, 20, 90, 40];
    for v in &values {
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.merge("max_val", &v.to_le_bytes(), &shodh_redb::NumericMax)
                .unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("max_val").unwrap().unwrap().value(), 90);
}

#[test]
fn merge_min_across_transactions() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("mmin");
    let values = [50u64, 30, 80, 10, 90, 5, 40];
    for v in &values {
        let w = db.begin_write().unwrap();
        {
            let mut t = w.open_table(T).unwrap();
            t.merge("min_val", &v.to_le_bytes(), &shodh_redb::NumericMin)
                .unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("min_val").unwrap().unwrap().value(), 5);
}

#[test]
fn merge_bitwise_or_accumulate() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, u64> = TableDefinition::new("mbor");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("flags", &1u64.to_le_bytes(), &shodh_redb::BitwiseOr)
            .unwrap();
        t.merge("flags", &2u64.to_le_bytes(), &shodh_redb::BitwiseOr)
            .unwrap();
        t.merge("flags", &4u64.to_le_bytes(), &shodh_redb::BitwiseOr)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("flags").unwrap().unwrap().value(), 7);
}

#[test]
fn merge_bytes_append_concat_two() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    const T: TableDefinition<&str, &[u8]> = TableDefinition::new("mba2");
    let w = db.begin_write().unwrap();
    {
        let mut t = w.open_table(T).unwrap();
        t.merge("log", b"hello", &shodh_redb::BytesAppend).unwrap();
        t.merge("log", b" world", &shodh_redb::BytesAppend).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r.open_table(T).unwrap();
    assert_eq!(t.get("log").unwrap().unwrap().value(), b"hello world");
}

// ===========================================================================
// S310 VECTOR OPS DEPTH -- PROPERTY TESTS
// ===========================================================================

#[test]
fn cosine_distance_commutativity() {
    let a = &[1.0f32, 2.0, 3.0];
    let b = &[4.0, 5.0, 6.0];
    let d1 = shodh_redb::cosine_distance(a, b);
    let d2 = shodh_redb::cosine_distance(b, a);
    assert!((d1 - d2).abs() < 1e-6);
}

#[test]
fn cosine_distance_self_is_zero() {
    let a = &[3.0f32, 4.0, 5.0];
    let d = shodh_redb::cosine_distance(a, a);
    assert!(d.abs() < 1e-6);
}

#[test]
fn euclidean_commutativity() {
    let a = &[1.0f32, 2.0];
    let b = &[4.0, 6.0];
    let d1 = shodh_redb::euclidean_distance_sq(a, b);
    let d2 = shodh_redb::euclidean_distance_sq(b, a);
    assert!((d1 - d2).abs() < 1e-6);
}

#[test]
fn euclidean_self_is_zero() {
    let a = &[7.0f32, 8.0, 9.0];
    assert!(shodh_redb::euclidean_distance_sq(a, a).abs() < 1e-6);
}

#[test]
fn euclidean_triangle_inequality_3d() {
    let a = &[0.0f32, 0.0, 0.0];
    let b = &[1.0, 0.0, 0.0];
    let c = &[0.0, 1.0, 0.0];
    let ab = shodh_redb::euclidean_distance_sq(a, b).sqrt();
    let bc = shodh_redb::euclidean_distance_sq(b, c).sqrt();
    let ac = shodh_redb::euclidean_distance_sq(a, c).sqrt();
    assert!(ac <= ab + bc + 1e-6);
}

#[test]
fn manhattan_commutativity() {
    let a = &[1.0f32, 2.0, 3.0];
    let b = &[4.0, 5.0, 6.0];
    let d1 = shodh_redb::manhattan_distance(a, b);
    let d2 = shodh_redb::manhattan_distance(b, a);
    assert!((d1 - d2).abs() < 1e-6);
}

#[test]
fn manhattan_self_is_zero() {
    let a = &[5.0f32, 10.0];
    assert!(shodh_redb::manhattan_distance(a, a).abs() < 1e-6);
}

#[test]
fn manhattan_triangle_inequality_vec2() {
    let a = &[0.0f32, 0.0];
    let b = &[3.0, 0.0];
    let c = &[0.0, 4.0];
    let ab = shodh_redb::manhattan_distance(a, b);
    let bc = shodh_redb::manhattan_distance(b, c);
    let ac = shodh_redb::manhattan_distance(a, c);
    assert!(ac <= ab + bc + 1e-6);
}

#[test]
fn dot_product_commutativity() {
    let a = &[1.0f32, 2.0, 3.0];
    let b = &[4.0, 5.0, 6.0];
    let d1 = shodh_redb::dot_product(a, b);
    let d2 = shodh_redb::dot_product(b, a);
    assert!((d1 - d2).abs() < 1e-6);
}

#[test]
fn dot_product_orthogonal_zero_2d() {
    let a = &[1.0f32, 0.0];
    let b = &[0.0, 1.0];
    assert!(shodh_redb::dot_product(a, b).abs() < 1e-6);
}

#[test]
fn hamming_commutativity() {
    let a = &[0b1010u8, 0b1100];
    let b = &[0b0110u8, 0b1001];
    assert_eq!(
        shodh_redb::hamming_distance(a, b),
        shodh_redb::hamming_distance(b, a)
    );
}

#[test]
fn hamming_self_is_zero() {
    let a = &[0xFFu8, 0x00, 0xAB];
    assert_eq!(shodh_redb::hamming_distance(a, a), 0);
}

#[test]
fn l2_normalize_unit_vector() {
    let v = vec![1.0f32, 0.0, 0.0];
    let n = shodh_redb::l2_normalized(&v);
    assert!((n[0] - 1.0).abs() < 1e-6);
    assert!(n[1].abs() < 1e-6);
}

#[test]
fn l2_normalize_preserves_direction() {
    let v = vec![3.0f32, 4.0];
    let n = shodh_redb::l2_normalized(&v);
    let norm = shodh_redb::l2_norm(&n);
    assert!((norm - 1.0).abs() < 1e-6);
    assert!((n[0] / n[1] - 0.75).abs() < 1e-6);
}

#[test]
fn nearest_k_correctness_euclidean() {
    let vecs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![10.0, 10.0],
        vec![0.5, 0.5],
    ];
    let query = &[0.0f32, 0.0];
    let results = shodh_redb::nearest_k(
        vecs.into_iter().enumerate().map(|(i, v)| (i as u64, v)),
        query,
        3,
        shodh_redb::euclidean_distance_sq,
    );
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].key, 0);
}

#[test]
fn nearest_k_correctness_cosine() {
    let vecs: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0],
        vec![0.707, 0.707],
        vec![0.0, 1.0],
        vec![-1.0, 0.0],
    ];
    let query = &[1.0f32, 0.0];
    let results = shodh_redb::nearest_k(
        vecs.into_iter().enumerate().map(|(i, v)| (i as u64, v)),
        query,
        2,
        shodh_redb::cosine_distance,
    );
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].key, 0);
}

#[test]
fn nearest_k_returns_all_when_k_exceeds() {
    let vecs: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0]];
    let query = &[0.0f32];
    let results = shodh_redb::nearest_k(
        vecs.into_iter().enumerate().map(|(i, v)| (i as u64, v)),
        query,
        100,
        shodh_redb::euclidean_distance_sq,
    );
    assert_eq!(results.len(), 2);
}

#[test]
fn quantize_binary_roundtrip_property() {
    let v = vec![1.0f32, -1.0, 0.5, -0.5, 0.0, 2.0, -3.0, 0.1];
    let bq = shodh_redb::quantize_binary(&v);
    assert_eq!(bq.len(), 1);
    assert_eq!(bq[0] & 0x80, 0x80);
    assert_eq!(bq[0] & 0x40, 0);
}

#[test]
fn quantize_scalar_preserves_ordering() {
    let v: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    let sq = shodh_redb::quantize_scalar(&v);
    let dq = shodh_redb::dequantize_scalar(&sq);
    for i in 0..4 {
        assert!(dq[i] <= dq[i + 1]);
    }
}

#[test]
fn read_write_f32_le_roundtrip_many() {
    let values: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
    let mut buf = vec![0u8; values.len() * 4];
    shodh_redb::write_f32_le(&mut buf, &values);
    let out = shodh_redb::read_f32_le(&buf);
    for (a, b) in values.iter().zip(out.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

// ===========================================================================
// S320 HLC DEPTH
// ===========================================================================

#[test]
fn hlc_monotonic_100_ticks() {
    let mut prev = shodh_redb::HybridLogicalClock::from_parts(1000, 0);
    for _ in 0..100 {
        let curr = prev.tick();
        assert!(curr > prev);
        prev = curr;
    }
}

#[test]
fn hlc_merge_always_advances() {
    let c1 = shodh_redb::HybridLogicalClock::from_parts(100, 5);
    let c2 = shodh_redb::HybridLogicalClock::from_parts(200, 3);
    let merged = c1.merge(c2);
    assert!(merged > c1);
    assert!(merged > c2);
}

#[test]
fn hlc_raw_roundtrip_multiple() {
    let mut ts = shodh_redb::HybridLogicalClock::from_parts(500, 0);
    for _ in 0..50 {
        ts = ts.tick();
        let raw = ts.to_raw();
        let restored = shodh_redb::HybridLogicalClock::from_raw(raw);
        assert_eq!(ts, restored);
    }
}

#[test]
fn hlc_ordering_is_total() {
    let mut timestamps = Vec::new();
    let mut ts = shodh_redb::HybridLogicalClock::from_parts(1000, 0);
    for _ in 0..50 {
        ts = ts.tick();
        timestamps.push(ts);
    }
    for i in 0..49 {
        assert!(timestamps[i] < timestamps[i + 1]);
    }
    timestamps.reverse();
    for i in 0..49 {
        assert!(timestamps[i] > timestamps[i + 1]);
    }
}

#[test]
fn hlc_from_parts_various() {
    for phys in [0u64, 1, 1000, u64::MAX >> 16] {
        for logical in [0u16, 1, 100, u16::MAX] {
            let ts = shodh_redb::HybridLogicalClock::from_parts(phys, logical);
            assert_eq!(ts.physical_ms(), phys);
            assert_eq!(ts.logical(), logical);
        }
    }
}

// ===========================================================================
// S330 STRESS -- MIXED WORKLOADS
// ===========================================================================

#[test]
fn stress_interleaved_read_write_100_txns() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    for i in 0u64..100 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("sirw"))
                .unwrap();
            t.insert(&i, &(i * i)).unwrap();
        }
        w.commit().unwrap();
        let r = db.begin_read().unwrap();
        let t = r
            .open_table(TableDefinition::<u64, u64>::new("sirw"))
            .unwrap();
        assert_eq!(t.get(&i).unwrap().unwrap().value(), i * i);
    }
}

#[test]
fn stress_overwrite_same_key_1000() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("sok1k"))
            .unwrap();
        for i in 0..1000u64 {
            t.insert(&0, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("sok1k"))
        .unwrap();
    assert_eq!(t.get(&0).unwrap().unwrap().value(), 999);
    assert_eq!(t.len().unwrap(), 1);
}

#[test]
fn stress_delete_reinsert_500_rounds() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    for round in 0..5 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("sdrc2"))
                .unwrap();
            for i in 0..100u64 {
                t.insert(&i, &(round * 100 + i)).unwrap();
            }
        }
        w.commit().unwrap();
        let w2 = db.begin_write().unwrap();
        {
            let mut t = w2
                .open_table(TableDefinition::<u64, u64>::new("sdrc2"))
                .unwrap();
            for i in 0..50u64 {
                t.remove(&i).unwrap();
            }
        }
        w2.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("sdrc2"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 50);
}

#[test]
fn stress_100_tables_one_txn() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    for i in 0..100u32 {
        let name = format!("stress_t_{i:03}");
        let def = TableDefinition::<u64, u64>::new(&*Box::leak(name.into_boxed_str()));
        let mut t = w.open_table(def).unwrap();
        t.insert(&(i as u64), &(i as u64)).unwrap();
    }
    w.commit().unwrap();
    let w2 = db.begin_write().unwrap();
    let tables: Vec<_> = w2.list_tables().unwrap().collect();
    assert!(tables.len() >= 100);
    w2.commit().unwrap();
}

#[test]
fn stress_mixed_key_types_one_db() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t1 = w
            .open_table(TableDefinition::<u64, u64>::new("sm_u64"))
            .unwrap();
        t1.insert(&1, &1).unwrap();
    }
    {
        let mut t2 = w
            .open_table(TableDefinition::<&str, &str>::new("sm_str"))
            .unwrap();
        t2.insert("a", "b").unwrap();
    }
    {
        let mut t3 = w
            .open_table(TableDefinition::<u32, &[u8]>::new("sm_bytes"))
            .unwrap();
        t3.insert(&42, &[1, 2, 3][..]).unwrap();
    }
    {
        let mut t4 = w
            .open_table(TableDefinition::<i64, i64>::new("sm_i64"))
            .unwrap();
        t4.insert(&-1, &-100).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t1 = r
        .open_table(TableDefinition::<u64, u64>::new("sm_u64"))
        .unwrap();
    assert_eq!(t1.get(&1).unwrap().unwrap().value(), 1);
    let t4 = r
        .open_table(TableDefinition::<i64, i64>::new("sm_i64"))
        .unwrap();
    assert_eq!(t4.get(&-1).unwrap().unwrap().value(), -100);
}

#[test]
fn stress_rapid_open_close_20x() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    for i in 0u64..20 {
        let db = Database::create(&path).unwrap();
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("roc2"))
                .unwrap();
            t.insert(&i, &i).unwrap();
        }
        w.commit().unwrap();
    }
    let db = Database::create(&path).unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("roc2"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 20);
}

#[test]
fn stress_5_readers_1_writer_concurrent() {
    let f = tmpfile();
    let db = Arc::new(Database::create(f.path()).unwrap());
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("scrw2"))
                .unwrap();
            for i in 0..1000u64 {
                t.insert(&i, &i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    let barrier = Arc::new(Barrier::new(6));
    let mut handles = Vec::new();
    for _ in 0..5 {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            for _ in 0..10 {
                let r = db.begin_read().unwrap();
                let t = r
                    .open_table(TableDefinition::<u64, u64>::new("scrw2"))
                    .unwrap();
                assert!(t.len().unwrap() >= 1000);
            }
        }));
    }
    {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            for i in 1000u64..1100 {
                let w = db.begin_write().unwrap();
                {
                    let mut t = w
                        .open_table(TableDefinition::<u64, u64>::new("scrw2"))
                        .unwrap();
                    t.insert(&i, &i).unwrap();
                }
                w.commit().unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("scrw2"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 1100);
}

// ===========================================================================
// S340 MULTIMAP DEEP TESTS
// ===========================================================================

#[test]
fn multimap_100_values_per_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("mm100"))
            .unwrap();
        for v in 0..100u64 {
            t.insert(&1, &v).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("mm100"))
        .unwrap();
    let values: Vec<u64> = t.get(&1).unwrap().map(|v| v.unwrap().value()).collect();
    assert_eq!(values.len(), 100);
}

#[test]
fn multimap_remove_single_value_deep() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("mmrsv"))
            .unwrap();
        t.insert(&1, &10).unwrap();
        t.insert(&1, &20).unwrap();
        t.insert(&1, &30).unwrap();
        t.remove(&1, &20).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("mmrsv"))
        .unwrap();
    let values: Vec<u64> = t.get(&1).unwrap().map(|v| v.unwrap().value()).collect();
    assert_eq!(values.len(), 2);
    assert!(!values.contains(&20));
}

#[test]
fn multimap_remove_all_50_values() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("mmra50"))
            .unwrap();
        for v in 0..50u64 {
            t.insert(&1, &v).unwrap();
        }
        t.insert(&2, &100).unwrap();
        t.remove_all(&1).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("mmra50"))
        .unwrap();
    assert_eq!(t.get(&1).unwrap().count(), 0);
    assert_eq!(t.len().unwrap(), 1);
}

#[test]
fn multimap_range_scan_with_multi_values() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("mmrsmv"))
            .unwrap();
        for k in 0..20u64 {
            t.insert(&k, &(k * 10)).unwrap();
            t.insert(&k, &(k * 10 + 1)).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("mmrsmv"))
        .unwrap();
    let range_count: u64 = t
        .range(5..10)
        .unwrap()
        .map(|kv| {
            let (_k, vals) = kv.unwrap();
            vals.len()
        })
        .sum();
    assert_eq!(range_count, 10);
}

// ===========================================================================
// S350 PERSISTENCE & INTEGRITY DEEP
// ===========================================================================

#[test]
fn reopen_preserves_multimap_data() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    {
        let db = Database::create(&path).unwrap();
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("rpmm"))
                .unwrap();
            t.insert(&1, &10).unwrap();
            t.insert(&1, &20).unwrap();
        }
        w.commit().unwrap();
    }
    {
        let db = Database::create(&path).unwrap();
        let r = db.begin_read().unwrap();
        let t = r
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("rpmm"))
            .unwrap();
        assert_eq!(t.len().unwrap(), 2);
    }
}

#[test]
fn reopen_preserves_all_table_names() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    {
        let db = Database::create(&path).unwrap();
        let w = db.begin_write().unwrap();
        for i in 0..10u32 {
            let name = format!("rptl2_{i}");
            let def = TableDefinition::<u64, u64>::new(&*Box::leak(name.into_boxed_str()));
            let mut t = w.open_table(def).unwrap();
            t.insert(&(i as u64), &(i as u64)).unwrap();
        }
        w.commit().unwrap();
    }
    {
        let db = Database::create(&path).unwrap();
        let w = db.begin_write().unwrap();
        let tables: Vec<_> = w.list_tables().unwrap().collect();
        assert!(tables.len() >= 10);
        w.commit().unwrap();
    }
}

#[test]
fn compact_then_reopen_preserves_data() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    {
        let mut db = Database::create(&path).unwrap();
        {
            let w = db.begin_write().unwrap();
            {
                let mut t = w
                    .open_table(TableDefinition::<u64, u64>::new("ctrp"))
                    .unwrap();
                for i in 0..500u64 {
                    t.insert(&i, &i).unwrap();
                }
            }
            w.commit().unwrap();
        }
        {
            let w = db.begin_write().unwrap();
            {
                let mut t = w
                    .open_table(TableDefinition::<u64, u64>::new("ctrp"))
                    .unwrap();
                for i in 0..250u64 {
                    t.remove(&i).unwrap();
                }
            }
            w.commit().unwrap();
        }
        db.compact().unwrap();
    }
    {
        let db = Database::create(&path).unwrap();
        let r = db.begin_read().unwrap();
        let t = r
            .open_table(TableDefinition::<u64, u64>::new("ctrp"))
            .unwrap();
        assert_eq!(t.len().unwrap(), 250);
        assert_eq!(t.first().unwrap().unwrap().0.value(), 250);
    }
}

#[test]
fn integrity_after_heavy_mixed_workload() {
    let f = tmpfile();
    let mut db = Database::create(f.path()).unwrap();
    for round in 0u64..10 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("ihmw"))
                .unwrap();
            let base = round * 100;
            for i in base..base + 100 {
                t.insert(&i, &i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("ihmw"))
                .unwrap();
            for i in 0..500u64 {
                t.remove(&i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    assert!(db.check_integrity().unwrap());
}

#[test]
fn compact_and_integrity_check() {
    let f = tmpfile();
    let mut db = Database::create(f.path()).unwrap();
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, &[u8]>::new("caic"))
                .unwrap();
            let data = vec![0u8; 256];
            for i in 0..300u64 {
                t.insert(&i, data.as_slice()).unwrap();
            }
        }
        w.commit().unwrap();
    }
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, &[u8]>::new("caic"))
                .unwrap();
            for i in 0..300u64 {
                t.remove(&i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    db.compact().unwrap();
    assert!(db.check_integrity().unwrap());
}

// ===========================================================================
// S400 BLOB STORE -- CORE STORAGE & RETRIEVAL
// ===========================================================================

#[test]
fn blob_store_basic_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let blob_id = w
        .store_blob(
            b"hello world",
            ContentType::OctetStream,
            "test",
            StoreOptions::default(),
        )
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let (data, meta) = r.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(data, b"hello world");
    assert_eq!(meta.label_str(), "test");
}

#[test]
fn blob_store_empty_data() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let blob_id = w
        .store_blob(
            b"",
            ContentType::OctetStream,
            "empty",
            StoreOptions::default(),
        )
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let (data, _meta) = r.get_blob(&blob_id).unwrap().unwrap();
    assert!(data.is_empty());
}

#[test]
fn blob_store_large_1mb() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let payload = vec![0xABu8; 1_048_576];
    let w = db.begin_write().unwrap();
    let blob_id = w
        .store_blob(
            &payload,
            ContentType::OctetStream,
            "1mb",
            StoreOptions::default(),
        )
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let (data, _meta) = r.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(data.len(), 1_048_576);
    assert!(data.iter().all(|&b| b == 0xAB));
}

#[test]
fn blob_store_multiple_content_types() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let types = [
        ContentType::OctetStream,
        ContentType::ImagePng,
        ContentType::ImageJpeg,
        ContentType::AudioWav,
        ContentType::Embedding,
        ContentType::Metadata,
    ];
    let w = db.begin_write().unwrap();
    let mut ids = Vec::new();
    for ct in &types {
        let id = w
            .store_blob(b"data", *ct, "ct-test", StoreOptions::default())
            .unwrap();
        ids.push(id);
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    for (i, id) in ids.iter().enumerate() {
        let (_data, meta) = r.get_blob(id).unwrap().unwrap();
        assert_eq!(meta.blob_ref.content_type, types[i].as_byte());
    }
}

#[test]
fn blob_store_metadata_only_retrieval() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let blob_id = w
        .store_blob(
            b"payload",
            ContentType::Embedding,
            "emb-label",
            StoreOptions::default(),
        )
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let meta = r.get_blob_meta(&blob_id).unwrap().unwrap();
    assert_eq!(meta.label_str(), "emb-label");
    assert_eq!(meta.blob_ref.length, 7);
}

#[test]
fn blob_store_delete() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let blob_id = w
        .store_blob(
            b"deleteme",
            ContentType::OctetStream,
            "del",
            StoreOptions::default(),
        )
        .unwrap();
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    assert!(w.delete_blob(&blob_id).unwrap());
    assert!(!w.delete_blob(&blob_id).unwrap());
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert!(r.get_blob(&blob_id).unwrap().is_none());
}

#[test]
fn blob_store_get_nonexistent() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let fake_id = BlobId::new(9999, 0);
    assert!(r.get_blob(&fake_id).unwrap().is_none());
    assert!(r.get_blob_meta(&fake_id).unwrap().is_none());
}

#[test]
fn blob_store_multiple_independent() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let id1 = w
        .store_blob(
            b"first",
            ContentType::OctetStream,
            "a",
            StoreOptions::default(),
        )
        .unwrap();
    let id2 = w
        .store_blob(
            b"second",
            ContentType::OctetStream,
            "b",
            StoreOptions::default(),
        )
        .unwrap();
    let id3 = w
        .store_blob(
            b"third",
            ContentType::OctetStream,
            "c",
            StoreOptions::default(),
        )
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert_eq!(r.get_blob(&id1).unwrap().unwrap().0, b"first");
    assert_eq!(r.get_blob(&id2).unwrap().unwrap().0, b"second");
    assert_eq!(r.get_blob(&id3).unwrap().unwrap().0, b"third");
}

#[test]
fn blob_store_persist_across_reopen() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    let blob_id;
    {
        let db = Database::create(&path).unwrap();
        let w = db.begin_write().unwrap();
        blob_id = w
            .store_blob(
                b"persist",
                ContentType::OctetStream,
                "p",
                StoreOptions::default(),
            )
            .unwrap();
        w.commit().unwrap();
    }
    {
        let db = Database::create(&path).unwrap();
        let r = db.begin_read().unwrap();
        let (data, _) = r.get_blob(&blob_id).unwrap().unwrap();
        assert_eq!(data, b"persist");
    }
}

// ===========================================================================
// S410 BLOB STORE -- STREAMING WRITER
// ===========================================================================

#[test]
fn blob_writer_basic() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let blob_id = {
        let mut writer = w
            .blob_writer(ContentType::OctetStream, "stream", StoreOptions::default())
            .unwrap();
        writer.write(b"chunk1").unwrap();
        writer.write(b"chunk2").unwrap();
        writer.finish().unwrap()
    };
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let (data, _) = r.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(data, b"chunk1chunk2");
}

#[test]
fn blob_writer_large_streaming() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let blob_id = {
        let mut writer = w
            .blob_writer(ContentType::PointCloudLas, "large", StoreOptions::default())
            .unwrap();
        let chunk = vec![0xFFu8; 4096];
        for _ in 0..100 {
            writer.write(&chunk).unwrap();
        }
        writer.finish().unwrap()
    };
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let (data, _) = r.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(data.len(), 4096 * 100);
}

#[test]
fn blob_writer_single_byte_chunks() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let blob_id = {
        let mut writer = w
            .blob_writer(ContentType::OctetStream, "tiny", StoreOptions::default())
            .unwrap();
        for b in b"hello" {
            writer.write(&[*b]).unwrap();
        }
        writer.finish().unwrap()
    };
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let (data, _) = r.get_blob(&blob_id).unwrap().unwrap();
    assert_eq!(data, b"hello");
}

// ===========================================================================
// S420 BLOB STORE -- RANGE READS
// ===========================================================================

#[test]
fn blob_range_read_middle() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let blob_id = w
        .store_blob(
            b"0123456789",
            ContentType::OctetStream,
            "rr",
            StoreOptions::default(),
        )
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let data = r.read_blob_range(&blob_id, 3, 4).unwrap().unwrap();
    assert_eq!(data, b"3456");
}

#[test]
fn blob_range_read_zero_length() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let blob_id = w
        .store_blob(
            b"data",
            ContentType::OctetStream,
            "z",
            StoreOptions::default(),
        )
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let data = r.read_blob_range(&blob_id, 2, 0).unwrap().unwrap();
    assert!(data.is_empty());
}

#[test]
fn blob_range_read_out_of_bounds() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let blob_id = w
        .store_blob(
            b"short",
            ContentType::OctetStream,
            "oob",
            StoreOptions::default(),
        )
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let result = r.read_blob_range(&blob_id, 0, 100);
    assert!(result.is_err());
}

// ===========================================================================
// S430 BLOB STORE -- TAGS & NAMESPACE
// ===========================================================================

#[test]
fn blob_tags_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let opts = StoreOptions {
        causal_link: None,
        namespace: None,
        tags: vec!["sensor".into(), "lidar".into()],
    };
    let w = db.begin_write().unwrap();
    let blob_id = w
        .store_blob(b"tagged", ContentType::SensorImu, "t", opts)
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let tags = r.blob_tags(&blob_id).unwrap();
    assert!(tags.contains(&"sensor".to_string()));
    assert!(tags.contains(&"lidar".to_string()));
}

#[test]
fn blob_tags_query_by_tag() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let id1 = w
        .store_blob(
            b"a",
            ContentType::OctetStream,
            "a",
            StoreOptions {
                causal_link: None,
                namespace: None,
                tags: vec!["alpha".into()],
            },
        )
        .unwrap();
    let _id2 = w
        .store_blob(
            b"b",
            ContentType::OctetStream,
            "b",
            StoreOptions {
                causal_link: None,
                namespace: None,
                tags: vec!["beta".into()],
            },
        )
        .unwrap();
    let id3 = w
        .store_blob(
            b"c",
            ContentType::OctetStream,
            "c",
            StoreOptions {
                causal_link: None,
                namespace: None,
                tags: vec!["alpha".into()],
            },
        )
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let alpha_blobs = r.blobs_by_tag("alpha").unwrap();
    assert_eq!(alpha_blobs.len(), 2);
    assert!(alpha_blobs.contains(&id1));
    assert!(alpha_blobs.contains(&id3));
}

#[test]
fn blob_namespace_roundtrip() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let opts = StoreOptions {
        causal_link: None,
        namespace: Some("session-42".into()),
        tags: vec![],
    };
    let w = db.begin_write().unwrap();
    let blob_id = w
        .store_blob(b"ns-data", ContentType::OctetStream, "ns", opts)
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let ns = r.blob_namespace(&blob_id).unwrap();
    assert_eq!(ns, Some("session-42".to_string()));
}

#[test]
fn blob_namespace_query() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    for i in 0..5 {
        let opts = StoreOptions {
            causal_link: None,
            namespace: Some("ns-a".into()),
            tags: vec![],
        };
        w.store_blob(
            format!("blob-{i}").as_bytes(),
            ContentType::OctetStream,
            "a",
            opts,
        )
        .unwrap();
    }
    for i in 0..3 {
        let opts = StoreOptions {
            causal_link: None,
            namespace: Some("ns-b".into()),
            tags: vec![],
        };
        w.store_blob(
            format!("blob-b-{i}").as_bytes(),
            ContentType::OctetStream,
            "b",
            opts,
        )
        .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let ns_a = r.blobs_in_namespace("ns-a").unwrap();
    assert_eq!(ns_a.len(), 5);
    let ns_b = r.blobs_in_namespace("ns-b").unwrap();
    assert_eq!(ns_b.len(), 3);
}

// ===========================================================================
// S440 BLOB STORE -- CAUSAL LINKS
// ===========================================================================

#[test]
fn blob_causal_parent_child() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let parent_id = w
        .store_blob(
            b"parent",
            ContentType::OctetStream,
            "p",
            StoreOptions::default(),
        )
        .unwrap();
    let child_opts = StoreOptions {
        causal_link: Some(CausalLink::new(
            parent_id,
            RelationType::Derived,
            "derived-from",
        )),
        namespace: None,
        tags: vec![],
    };
    let child_id = w
        .store_blob(b"child", ContentType::OctetStream, "c", child_opts)
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let chain = r.causal_chain(&child_id, 10).unwrap();
    assert!(chain.len() >= 2);
    assert_eq!(chain[0].0, child_id);
    assert_eq!(chain[1].0, parent_id);
}

#[test]
fn blob_causal_chain_three_levels() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let root = w
        .store_blob(
            b"root",
            ContentType::OctetStream,
            "r",
            StoreOptions::default(),
        )
        .unwrap();
    let mid_opts = StoreOptions {
        causal_link: Some(CausalLink::new(root, RelationType::Derived, "")),
        namespace: None,
        tags: vec![],
    };
    let mid = w
        .store_blob(b"mid", ContentType::OctetStream, "m", mid_opts)
        .unwrap();
    let leaf_opts = StoreOptions {
        causal_link: Some(CausalLink::new(mid, RelationType::Supports, "evidence")),
        namespace: None,
        tags: vec![],
    };
    let _leaf = w
        .store_blob(b"leaf", ContentType::OctetStream, "l", leaf_opts)
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let chain = r.causal_chain(&_leaf, 10).unwrap();
    assert!(chain.len() >= 3);
}

#[test]
fn blob_causal_children() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let parent = w
        .store_blob(
            b"parent",
            ContentType::OctetStream,
            "p",
            StoreOptions::default(),
        )
        .unwrap();
    for i in 0..3 {
        let opts = StoreOptions {
            causal_link: Some(CausalLink::new(parent, RelationType::Derived, "")),
            namespace: None,
            tags: vec![],
        };
        w.store_blob(
            format!("child-{i}").as_bytes(),
            ContentType::OctetStream,
            "c",
            opts,
        )
        .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let children = r.causal_children(&parent).unwrap();
    assert_eq!(children.len(), 3);
}

#[test]
fn blob_causal_all_relation_types() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let base = w
        .store_blob(
            b"base",
            ContentType::OctetStream,
            "b",
            StoreOptions::default(),
        )
        .unwrap();
    let relations = [
        RelationType::Derived,
        RelationType::Similar,
        RelationType::Contradicts,
        RelationType::Supports,
        RelationType::Supersedes,
    ];
    for rel in &relations {
        let opts = StoreOptions {
            causal_link: Some(CausalLink::new(base, *rel, "")),
            namespace: None,
            tags: vec![],
        };
        w.store_blob(b"x", ContentType::OctetStream, "x", opts)
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let children = r.causal_children(&base).unwrap();
    assert_eq!(children.len(), 5);
}

// ===========================================================================
// S450 BLOB STORE -- STATS & DEDUP
// ===========================================================================

#[test]
fn blob_stats_empty() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let stats = w.blob_stats().unwrap();
    assert_eq!(stats.blob_count, 0);
    assert_eq!(stats.live_bytes, 0);
    w.abort().unwrap();
}

#[test]
fn blob_stats_after_inserts() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    for i in 0..10 {
        w.store_blob(
            &vec![0u8; 1000],
            ContentType::OctetStream,
            &format!("b{i}"),
            StoreOptions::default(),
        )
        .unwrap();
    }
    let stats = w.blob_stats().unwrap();
    assert_eq!(stats.blob_count, 10);
    assert!(stats.live_bytes >= 10_000);
    w.commit().unwrap();
}

#[test]
fn blob_stats_after_delete() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let _id1 = w
        .store_blob(
            b"keep",
            ContentType::OctetStream,
            "k",
            StoreOptions::default(),
        )
        .unwrap();
    let id2 = w
        .store_blob(
            b"remove",
            ContentType::OctetStream,
            "r",
            StoreOptions::default(),
        )
        .unwrap();
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    w.delete_blob(&id2).unwrap();
    let stats = w.blob_stats().unwrap();
    assert_eq!(stats.blob_count, 1);
    // With chunked B-tree storage, deleted blob chunks are freed immediately —
    // there is no dead space accumulation as with region-based storage.
    assert_eq!(stats.dead_bytes, 0);
    w.commit().unwrap();
}

#[test]
fn blob_dedup_identical_content() {
    let f = tmpfile();
    let db = Builder::new()
        .set_blob_dedup(true)
        .create(f.path())
        .unwrap();
    let data = vec![0xCDu8; 8192];
    let w = db.begin_write().unwrap();
    let id1 = w
        .store_blob(
            &data,
            ContentType::OctetStream,
            "a",
            StoreOptions::default(),
        )
        .unwrap();
    let id2 = w
        .store_blob(
            &data,
            ContentType::OctetStream,
            "b",
            StoreOptions::default(),
        )
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let (d1, _) = r.get_blob(&id1).unwrap().unwrap();
    let (d2, _) = r.get_blob(&id2).unwrap().unwrap();
    assert_eq!(d1, d2);
    let dedup = r.dedup_stats().unwrap();
    assert!(dedup.bytes_saved > 0);
}

// ===========================================================================
// S500 CDC (CHANGE DATA CAPTURE)
// ===========================================================================

#[test]
fn cdc_insert_detected() {
    let f = tmpfile();
    let db = Builder::new()
        .set_cdc(CdcConfig {
            enabled: true,
            retention_max_txns: 0,
        })
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cdc_ins"))
            .unwrap();
        t.insert(&1, &100).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let changes = r.read_cdc_since(0).unwrap();
    assert!(!changes.is_empty());
    let insert = changes.iter().find(|c| c.op == ChangeOp::Insert).unwrap();
    assert_eq!(insert.table_name, "cdc_ins");
}

#[test]
fn cdc_update_detected() {
    let f = tmpfile();
    let db = Builder::new()
        .set_cdc(CdcConfig {
            enabled: true,
            retention_max_txns: 0,
        })
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cdc_upd"))
            .unwrap();
        t.insert(&1, &100).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cdc_upd"))
            .unwrap();
        t.insert(&1, &200).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let changes = r.read_cdc_since(0).unwrap();
    assert!(changes.iter().any(|c| c.op == ChangeOp::Update));
}

#[test]
fn cdc_delete_detected() {
    let f = tmpfile();
    let db = Builder::new()
        .set_cdc(CdcConfig {
            enabled: true,
            retention_max_txns: 0,
        })
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cdc_del"))
            .unwrap();
        t.insert(&1, &100).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cdc_del"))
            .unwrap();
        t.remove(&1).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let changes = r.read_cdc_since(0).unwrap();
    assert!(changes.iter().any(|c| c.op == ChangeOp::Delete));
}

#[test]
fn cdc_multiple_tables_tracked() {
    let f = tmpfile();
    let db = Builder::new()
        .set_cdc(CdcConfig {
            enabled: true,
            retention_max_txns: 0,
        })
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t1 = w
            .open_table(TableDefinition::<u64, u64>::new("cdc_t1"))
            .unwrap();
        t1.insert(&1, &1).unwrap();
    }
    {
        let mut t2 = w
            .open_table(TableDefinition::<u64, u64>::new("cdc_t2"))
            .unwrap();
        t2.insert(&2, &2).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let changes = r.read_cdc_since(0).unwrap();
    let tables: Vec<&str> = changes.iter().map(|c| c.table_name.as_str()).collect();
    assert!(tables.contains(&"cdc_t1"));
    assert!(tables.contains(&"cdc_t2"));
}

#[test]
fn cdc_range_query() {
    let f = tmpfile();
    let db = Builder::new()
        .set_cdc(CdcConfig {
            enabled: true,
            retention_max_txns: 0,
        })
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cdc_rng"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cdc_rng"))
            .unwrap();
        t.insert(&2, &2).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let latest = r.latest_cdc_transaction_id().unwrap().unwrap();
    let changes = r.read_cdc_range(latest, latest).unwrap();
    assert!(!changes.is_empty());
    for c in &changes {
        assert_eq!(c.transaction_id, latest);
    }
}

#[test]
fn cdc_cursor_management() {
    let f = tmpfile();
    let db = Builder::new()
        .set_cdc(CdcConfig {
            enabled: true,
            retention_max_txns: 0,
        })
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cdc_cur"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    w.advance_cdc_cursor("consumer-1", 1).unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert_eq!(r.cdc_cursor("consumer-1").unwrap(), Some(1));
    assert!(r.cdc_cursor("consumer-2").unwrap().is_none());
}

#[test]
fn cdc_empty_when_disabled() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("no_cdc"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert!(r.read_cdc_since(0).unwrap().is_empty());
}

// ===========================================================================
// S510 INTEGRITY VERIFICATION
// ===========================================================================

#[test]
fn verify_integrity_header_level() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("vi_h"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let report = db.verify_integrity(VerifyLevel::Header).unwrap();
    assert!(report.valid);
    assert!(report.header_valid);
    assert_eq!(report.pages_checked, 0);
}

#[test]
fn verify_integrity_checksum_level() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("vi_c"))
            .unwrap();
        for i in 0..100u64 {
            t.insert(&i, &(i * 2)).unwrap();
        }
    }
    w.commit().unwrap();
    let report = db.verify_integrity(VerifyLevel::Pages).unwrap();
    assert!(report.valid);
    assert!(report.pages_checked > 0);
    assert_eq!(report.pages_corrupt, 0);
}

#[test]
fn verify_integrity_full_level() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("vi_f"))
            .unwrap();
        for i in 0..200u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let report = db.verify_integrity(VerifyLevel::Full).unwrap();
    assert!(report.valid);
    assert_eq!(report.pages_corrupt, 0);
    assert_eq!(report.structural_valid, Some(true));
}

#[test]
fn verify_integrity_after_many_transactions() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    for i in 0..50u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("vi_mt"))
                .unwrap();
            t.insert(&i, &(i * 3)).unwrap();
        }
        w.commit().unwrap();
    }
    let report = db.verify_integrity(VerifyLevel::Full).unwrap();
    assert!(report.valid);
}

#[test]
fn check_integrity_clean() {
    let f = tmpfile();
    let mut db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("ci"))
            .unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    assert!(db.check_integrity().unwrap());
}

// ===========================================================================
// S520 COMPACTION
// ===========================================================================

#[test]
fn compact_reduces_file_after_deletes() {
    let f = tmpfile();
    let mut db = Database::create(f.path()).unwrap();
    let data = vec![0u8; 512];
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, &[u8]>::new("cmp"))
            .unwrap();
        for i in 0..500u64 {
            t.insert(&i, data.as_slice()).unwrap();
        }
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, &[u8]>::new("cmp"))
            .unwrap();
        for i in 0..500u64 {
            t.remove(&i).unwrap();
        }
    }
    w.commit().unwrap();
    let size_before = std::fs::metadata(f.path()).unwrap().len();
    db.compact().unwrap();
    let size_after = std::fs::metadata(f.path()).unwrap().len();
    assert!(size_after < size_before);
}

#[test]
fn compact_preserves_data() {
    let f = tmpfile();
    let mut db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cmp_d"))
            .unwrap();
        for i in 0..100u64 {
            t.insert(&i, &(i * 7)).unwrap();
        }
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cmp_d"))
            .unwrap();
        for i in (0..100u64).step_by(2) {
            t.remove(&i).unwrap();
        }
    }
    w.commit().unwrap();
    db.compact().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("cmp_d"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 50);
    for i in (1..100u64).step_by(2) {
        assert_eq!(t.get(&i).unwrap().unwrap().value(), i * 7);
    }
}

#[test]
fn compact_integrity_check_after() {
    let f = tmpfile();
    let mut db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cmp_ic"))
            .unwrap();
        for i in 0..200u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    db.compact().unwrap();
    assert!(db.check_integrity().unwrap());
    let report = db.verify_integrity(VerifyLevel::Full).unwrap();
    assert!(report.valid);
}

#[test]
fn blob_compaction_reclaims_space() {
    let f = tmpfile();
    let mut db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let data = vec![0u8; 4096];
    let mut ids = Vec::new();
    for i in 0..20 {
        ids.push(
            w.store_blob(
                &data,
                ContentType::OctetStream,
                &format!("b{i}"),
                StoreOptions::default(),
            )
            .unwrap(),
        );
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    for id in &ids[..10] {
        w.delete_blob(id).unwrap();
    }
    w.commit().unwrap();
    let report = db.compact_blobs().unwrap();
    assert!(report.bytes_reclaimed > 0 || report.was_noop);
    let r = db.begin_read().unwrap();
    for id in &ids[10..] {
        assert!(r.get_blob(id).unwrap().is_some());
    }
}

// ===========================================================================
// S530 CONCURRENT WRITE CONTENTION
// ===========================================================================

#[test]
fn concurrent_write_serialized() {
    let f = tmpfile();
    let db = Arc::new(Database::create(f.path()).unwrap());
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cws"))
            .unwrap();
        t.insert(&0, &0).unwrap();
    }
    w.commit().unwrap();
    let db2 = Arc::clone(&db);
    let handle = std::thread::spawn(move || {
        let w = db2.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("cws"))
                .unwrap();
            t.insert(&1, &1).unwrap();
        }
        w.commit().unwrap();
    });
    handle.join().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cws"))
            .unwrap();
        t.insert(&2, &2).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("cws"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 3);
}

#[test]
fn concurrent_writers_10_threads() {
    let f = tmpfile();
    let db = Arc::new(Database::create(f.path()).unwrap());
    {
        let w = db.begin_write().unwrap();
        w.open_table(TableDefinition::<u64, u64>::new("cw10"))
            .unwrap();
        w.commit().unwrap();
    }
    let mut handles = Vec::new();
    for i in 0..10u64 {
        let db = Arc::clone(&db);
        handles.push(std::thread::spawn(move || {
            let w = db.begin_write().unwrap();
            {
                let mut t = w
                    .open_table(TableDefinition::<u64, u64>::new("cw10"))
                    .unwrap();
                for j in 0..10u64 {
                    t.insert(&(i * 10 + j), &j).unwrap();
                }
            }
            w.commit().unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("cw10"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 100);
}

#[test]
fn concurrent_read_write_isolation() {
    let f = tmpfile();
    let db = Arc::new(Database::create(f.path()).unwrap());
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("crwi"))
            .unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let db2 = Arc::clone(&db);
    let handle = std::thread::spawn(move || {
        let w = db2.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("crwi"))
                .unwrap();
            for i in 100..200u64 {
                t.insert(&i, &i).unwrap();
            }
        }
        w.commit().unwrap();
    });
    handle.join().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("crwi"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 100); // snapshot isolation
    drop(t);
    drop(r);
    let r2 = db.begin_read().unwrap();
    let t2 = r2
        .open_table(TableDefinition::<u64, u64>::new("crwi"))
        .unwrap();
    assert_eq!(t2.len().unwrap(), 200);
}

// ===========================================================================
// S540 BOUNDARY CONDITIONS & EDGE CASES
// ===========================================================================

#[test]
fn boundary_u64_max_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("bmax"))
            .unwrap();
        t.insert(&u64::MAX, &42).unwrap();
        t.insert(&u64::MIN, &0).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("bmax"))
        .unwrap();
    assert_eq!(t.get(&u64::MAX).unwrap().unwrap().value(), 42);
    assert_eq!(t.get(&u64::MIN).unwrap().unwrap().value(), 0);
}

#[test]
fn boundary_empty_string_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<&str, u64>::new("besk"))
            .unwrap();
        t.insert("", &1).unwrap();
        t.insert("a", &2).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<&str, u64>::new("besk"))
        .unwrap();
    assert_eq!(t.get("").unwrap().unwrap().value(), 1);
    assert_eq!(t.len().unwrap(), 2);
}

#[test]
fn boundary_empty_byte_slice_value() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, &[u8]>::new("bev"))
            .unwrap();
        t.insert(&1, &[][..]).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, &[u8]>::new("bev"))
        .unwrap();
    assert!(t.get(&1).unwrap().unwrap().value().is_empty());
}

#[test]
fn boundary_i128_extremes() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<i128, i128>::new("bi128"))
            .unwrap();
        t.insert(&i128::MIN, &i128::MAX).unwrap();
        t.insert(&i128::MAX, &i128::MIN).unwrap();
        t.insert(&0i128, &0i128).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<i128, i128>::new("bi128"))
        .unwrap();
    assert_eq!(t.get(&i128::MIN).unwrap().unwrap().value(), i128::MAX);
    assert_eq!(t.get(&i128::MAX).unwrap().unwrap().value(), i128::MIN);
    assert_eq!(t.len().unwrap(), 3);
}

#[test]
fn boundary_bool_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<bool, u64>::new("bbool"))
            .unwrap();
        t.insert(&true, &1).unwrap();
        t.insert(&false, &0).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<bool, u64>::new("bbool"))
        .unwrap();
    assert_eq!(t.get(&true).unwrap().unwrap().value(), 1);
    assert_eq!(t.get(&false).unwrap().unwrap().value(), 0);
}

#[test]
fn boundary_overwrite_same_key_100x() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("bow"))
            .unwrap();
        for i in 0..100u64 {
            t.insert(&42, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("bow"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 1);
    assert_eq!(t.get(&42).unwrap().unwrap().value(), 99);
}

#[test]
fn boundary_insert_delete_same_key_50_cycles() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    for cycle in 0..50u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("bidc"))
                .unwrap();
            t.insert(&1, &cycle).unwrap();
        }
        w.commit().unwrap();
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("bidc"))
                .unwrap();
            t.remove(&1).unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("bidc"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 0);
}

#[test]
fn boundary_very_long_string_key() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let long_key: String = "x".repeat(10_000);
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<&str, u64>::new("blsk"))
            .unwrap();
        t.insert(long_key.as_str(), &42).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<&str, u64>::new("blsk"))
        .unwrap();
    assert_eq!(t.get(long_key.as_str()).unwrap().unwrap().value(), 42);
}

#[test]
fn boundary_large_value_64kb() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let big_val = vec![0xFFu8; 65536];
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, &[u8]>::new("blv64"))
            .unwrap();
        t.insert(&1, big_val.as_slice()).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, &[u8]>::new("blv64"))
        .unwrap();
    assert_eq!(t.get(&1).unwrap().unwrap().value().len(), 65536);
}

// ===========================================================================
// S550 ERROR HANDLING & ROBUSTNESS
// ===========================================================================

#[test]
fn error_table_type_mismatch() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("etm"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let result = w.open_table(TableDefinition::<&str, u64>::new("etm"));
        assert!(result.is_err());
    }
    w.abort().unwrap();
}

#[test]
fn error_open_multimap_as_regular() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("emm"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let result = w.open_table(TableDefinition::<u64, u64>::new("emm"));
        assert!(result.is_err());
    }
    w.abort().unwrap();
}

#[test]
fn error_read_nonexistent_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let result = r.open_table(TableDefinition::<u64, u64>::new("nope"));
    assert!(result.is_err());
}

#[test]
fn error_durability_with_persistent_savepoint() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    // Create a persistent savepoint first
    let w = db.begin_write().unwrap();
    let sp_id = w.persistent_savepoint().unwrap();
    w.commit().unwrap();
    // Now try to set Durability::None in a txn that deletes a persistent savepoint
    let mut w = db.begin_write().unwrap();
    w.delete_persistent_savepoint(sp_id).unwrap();
    let result = w.set_durability(Durability::None);
    assert!(result.is_err());
    w.abort().unwrap();
}

#[test]
fn abort_rolls_back() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("arb"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("arb"))
            .unwrap();
        t.insert(&2, &2).unwrap();
    }
    w.abort().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("arb"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 1);
}

// ===========================================================================
// S560 MEMORY BUDGET & CACHE
// ===========================================================================

#[test]
fn memory_budget_tight_still_works() {
    let f = tmpfile();
    let db = Builder::new()
        .set_memory_budget(512 * 1024)
        .create(f.path())
        .unwrap();
    let value = vec![0u8; 4096];
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, &[u8]>::new("mb"))
            .unwrap();
        for i in 0..200u64 {
            t.insert(&i, value.as_slice()).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, &[u8]>::new("mb"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 200);
}

#[test]
fn cache_stats_report() {
    let f = tmpfile();
    let db = Builder::new()
        .set_cache_size(10 * 1024 * 1024)
        .create(f.path())
        .unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("cs"))
            .unwrap();
        for i in 0..1000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let stats = db.cache_stats();
    assert!(stats.used_bytes() > 0);
}

// ===========================================================================
// S570 PERSISTENCE & CRASH SAFETY
// ===========================================================================

#[test]
fn persistence_data_survives_reopen() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    {
        let db = Database::create(&path).unwrap();
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("pdr"))
                .unwrap();
            for i in 0..500u64 {
                t.insert(&i, &(i * 3)).unwrap();
            }
        }
        w.commit().unwrap();
    }
    {
        let db = Database::create(&path).unwrap();
        let r = db.begin_read().unwrap();
        let t = r
            .open_table(TableDefinition::<u64, u64>::new("pdr"))
            .unwrap();
        assert_eq!(t.len().unwrap(), 500);
        assert_eq!(t.get(&499).unwrap().unwrap().value(), 499 * 3);
    }
}

#[test]
fn persistence_multimap_survives_reopen() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    {
        let db = Database::create(&path).unwrap();
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("pmm"))
                .unwrap();
            for i in 0..10u64 {
                for j in 0..5u64 {
                    t.insert(&i, &(i * 10 + j)).unwrap();
                }
            }
        }
        w.commit().unwrap();
    }
    {
        let db = Database::create(&path).unwrap();
        let r = db.begin_read().unwrap();
        let t = r
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("pmm"))
            .unwrap();
        assert_eq!(t.len().unwrap(), 50);
    }
}

#[test]
fn persistence_uncommitted_not_visible() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    {
        let db = Database::create(&path).unwrap();
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("punv"))
                .unwrap();
            t.insert(&1, &1).unwrap();
        }
        w.commit().unwrap();
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("punv"))
                .unwrap();
            t.insert(&2, &2).unwrap();
        }
        w.abort().unwrap();
    }
    {
        let db = Database::create(&path).unwrap();
        let r = db.begin_read().unwrap();
        let t = r
            .open_table(TableDefinition::<u64, u64>::new("punv"))
            .unwrap();
        assert_eq!(t.len().unwrap(), 1);
    }
}

#[test]
fn durability_none_faster_but_works() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let mut w = db.begin_write().unwrap();
    w.set_durability(Durability::None).unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("dn"))
            .unwrap();
        for i in 0..1000u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("dn"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 1000);
}

// ===========================================================================
// S580 REGRESSION-STYLE TESTS
// ===========================================================================

#[test]
fn regression_savepoint_created_before_dirty() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let _sp = w.ephemeral_savepoint().unwrap();
    w.commit().unwrap();
}

#[test]
fn regression_savepoint_after_write_fails() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rsaw"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    let result = w.ephemeral_savepoint();
    assert!(result.is_err());
    w.commit().unwrap();
}

#[test]
fn regression_drop_guards_before_reuse() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rdg"))
            .unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
        let first = t.pop_first().unwrap().unwrap();
        assert_eq!(first.0.value(), 0);
        drop(first);
        let last = t.pop_last().unwrap().unwrap();
        assert_eq!(last.0.value(), 9);
        drop(last);
        assert_eq!(t.len().unwrap(), 8);
    }
    w.commit().unwrap();
}

#[test]
fn regression_extract_if_consumes_iterator() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("reic"))
            .unwrap();
        for i in 0..20u64 {
            t.insert(&i, &i).unwrap();
        }
        let removed = t.extract_if(|k, _| k < 10).unwrap().count();
        assert_eq!(removed, 10);
        assert_eq!(t.len().unwrap(), 10);
    }
    w.commit().unwrap();
}

#[test]
fn regression_multimap_values_len_is_u64() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("rml"))
            .unwrap();
        for v in 0..10u64 {
            t.insert(&1, &v).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("rml"))
        .unwrap();
    let vals = t.get(&1).unwrap();
    let len: u64 = vals.len();
    assert_eq!(len, 10);
}

// ===========================================================================
// S590 STRESS & MIXED WORKLOADS (GAP BATCH)
// ===========================================================================

#[test]
fn stress_alternating_insert_delete_1000() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("said"))
            .unwrap();
        for i in 0..1000u64 {
            t.insert(&i, &i).unwrap();
            if i % 3 == 0 {
                t.remove(&i).unwrap();
            }
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("said"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 666);
}

#[test]
fn stress_200_small_transactions() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    for i in 0..200u64 {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("smst"))
                .unwrap();
            t.insert(&i, &i).unwrap();
        }
        w.commit().unwrap();
    }
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("smst"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 200);
}

#[test]
fn stress_mixed_table_and_blob() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t1 = w
            .open_table(TableDefinition::<u64, u64>::new("smt1"))
            .unwrap();
        for i in 0..100u64 {
            t1.insert(&i, &i).unwrap();
        }
    }
    {
        let mut t2 = w
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("smt2"))
            .unwrap();
        for i in 0..50u64 {
            t2.insert(&i, &(i * 2)).unwrap();
            t2.insert(&i, &(i * 2 + 1)).unwrap();
        }
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    w.store_blob(
        b"mixed-workload",
        ContentType::OctetStream,
        "m",
        StoreOptions::default(),
    )
    .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t1 = r
        .open_table(TableDefinition::<u64, u64>::new("smt1"))
        .unwrap();
    assert_eq!(t1.len().unwrap(), 100);
    drop(t1);
    let t2 = r
        .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("smt2"))
        .unwrap();
    assert_eq!(t2.len().unwrap(), 100);
}

#[test]
fn stress_concurrent_readers_during_write() {
    let f = tmpfile();
    let db = Arc::new(Database::create(f.path()).unwrap());
    {
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("scrdw"))
                .unwrap();
            for i in 0..100u64 {
                t.insert(&i, &i).unwrap();
            }
        }
        w.commit().unwrap();
    }
    let barrier = Arc::new(Barrier::new(6));
    let mut handles = Vec::new();
    for _ in 0..5 {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            for _ in 0..10 {
                let r = db.begin_read().unwrap();
                let t = r
                    .open_table(TableDefinition::<u64, u64>::new("scrdw"))
                    .unwrap();
                assert!(t.len().unwrap() >= 100);
            }
        }));
    }
    barrier.wait();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("scrdw"))
            .unwrap();
        for i in 100..200u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn stress_rapid_open_close_with_data() {
    let f = tmpfile();
    let path = f.path().to_path_buf();
    for round in 0..20u64 {
        let db = Database::create(&path).unwrap();
        let w = db.begin_write().unwrap();
        {
            let mut t = w
                .open_table(TableDefinition::<u64, u64>::new("sroc"))
                .unwrap();
            t.insert(&round, &round).unwrap();
        }
        w.commit().unwrap();
    }
    let db = Database::create(&path).unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("sroc"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 20);
}

// ===========================================================================
// S600 BLOB STORE -- ADVANCED PATTERNS
// ===========================================================================

#[test]
fn blob_delete_cleans_tags() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let opts = StoreOptions {
        causal_link: None,
        namespace: None,
        tags: vec!["cleanup-tag".into()],
    };
    let blob_id = w
        .store_blob(b"to-delete", ContentType::OctetStream, "d", opts)
        .unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert_eq!(r.blobs_by_tag("cleanup-tag").unwrap().len(), 1);
    drop(r);
    let w = db.begin_write().unwrap();
    w.delete_blob(&blob_id).unwrap();
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    assert!(r.blobs_by_tag("cleanup-tag").unwrap().is_empty());
}

#[test]
fn blob_100_blobs_all_readable() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    let mut ids = Vec::new();
    for i in 0..100u32 {
        let data = format!("blob-data-{i}");
        let id = w
            .store_blob(
                data.as_bytes(),
                ContentType::OctetStream,
                &format!("l{i}"),
                StoreOptions::default(),
            )
            .unwrap();
        ids.push(id);
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    for (i, id) in ids.iter().enumerate() {
        let (data, _) = r.get_blob(id).unwrap().unwrap();
        assert_eq!(data, format!("blob-data-{i}").as_bytes());
    }
    let stats = r.blob_stats().unwrap();
    assert_eq!(stats.blob_count, 100);
}

// ===========================================================================
// S610 INCREMENTAL COMPACTION
// ===========================================================================

#[test]
fn incremental_compaction_basic() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, &[u8]>::new("ic_basic"))
            .unwrap();
        let data = vec![0u8; 256];
        for i in 0..200u64 {
            t.insert(&i, data.as_slice()).unwrap();
        }
    }
    w.commit().unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, &[u8]>::new("ic_basic"))
            .unwrap();
        for i in 0..150u64 {
            t.remove(&i).unwrap();
        }
    }
    w.commit().unwrap();
    let handle = db.start_compaction().unwrap();
    let steps = handle.run().unwrap();
    assert!(steps > 0);
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, &[u8]>::new("ic_basic"))
        .unwrap();
    assert_eq!(t.len().unwrap(), 50);
}

// ===========================================================================
// S620 TWO-PHASE COMMIT
// ===========================================================================

#[test]
fn two_phase_commit_basic() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let mut w = db.begin_write().unwrap();
    w.set_two_phase_commit(true);
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("2pc"))
            .unwrap();
        t.insert(&1, &100).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("2pc"))
        .unwrap();
    assert_eq!(t.get(&1).unwrap().unwrap().value(), 100);
}

// ===========================================================================
// S630 TABLE LISTING & METADATA
// ===========================================================================

#[test]
fn list_tables_after_creation() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("list_a"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("list_b"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let tables: Vec<_> = r.list_tables().unwrap().collect();
    let names: Vec<String> = tables.iter().map(|t| t.name().to_string()).collect();
    assert!(names.contains(&"list_a".to_string()));
    assert!(names.contains(&"list_b".to_string()));
}

#[test]
fn list_multimap_tables() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_multimap_table(MultimapTableDefinition::<u64, u64>::new("mm_list"))
            .unwrap();
        t.insert(&1, &1).unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let tables: Vec<_> = r.list_multimap_tables().unwrap().collect();
    let names: Vec<String> = tables.iter().map(|t| t.name().to_string()).collect();
    assert!(names.contains(&"mm_list".to_string()));
}

#[test]
fn table_stats_accuracy() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("tsa"))
            .unwrap();
        for i in 0..500u64 {
            t.insert(&i, &(i * 2)).unwrap();
        }
    }
    let stats = w.stats().unwrap();
    assert!(stats.tree_height() > 0);
    assert!(stats.stored_bytes() > 0);
    assert!(stats.allocated_pages() > 0);
    w.commit().unwrap();
}

// ===========================================================================
// S640 RANGE OPERATIONS DEEP
// ===========================================================================

#[test]
fn range_full_scan_50_entries() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rfs2"))
            .unwrap();
        for i in 0..50u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("rfs2"))
        .unwrap();
    let all: Vec<_> = t.range::<u64>(..).unwrap().collect();
    assert_eq!(all.len(), 50);
}

#[test]
fn range_bounded() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rb"))
            .unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("rb"))
        .unwrap();
    let range: Vec<_> = t.range(10..20).unwrap().collect();
    assert_eq!(range.len(), 10);
}

#[test]
fn range_inclusive() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("ri"))
            .unwrap();
        for i in 0..100u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("ri"))
        .unwrap();
    let range: Vec<_> = t.range(10..=20).unwrap().collect();
    assert_eq!(range.len(), 11);
}

#[test]
fn range_empty_result() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rer"))
            .unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("rer"))
        .unwrap();
    let range: Vec<_> = t.range(100..200).unwrap().collect();
    assert!(range.is_empty());
}

#[test]
fn range_reverse_iteration() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("rri"))
            .unwrap();
        for i in 0..10u64 {
            t.insert(&i, &i).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("rri"))
        .unwrap();
    let range: Vec<_> = t.range::<u64>(..).unwrap().rev().collect();
    assert_eq!(range.len(), 10);
    assert_eq!(range[0].as_ref().unwrap().0.value(), 9);
    assert_eq!(range[9].as_ref().unwrap().0.value(), 0);
}

// ===========================================================================
// S650 FIRST/LAST OPERATIONS
// ===========================================================================

#[test]
fn first_last_on_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("fl"))
            .unwrap();
        for i in 10..20u64 {
            t.insert(&i, &(i * 2)).unwrap();
        }
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("fl"))
        .unwrap();
    let (first_k, first_v) = t.first().unwrap().unwrap();
    assert_eq!(first_k.value(), 10);
    assert_eq!(first_v.value(), 20);
    let (last_k, last_v) = t.last().unwrap().unwrap();
    assert_eq!(last_k.value(), 19);
    assert_eq!(last_v.value(), 38);
}

#[test]
fn first_last_empty_table() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let _t = w
            .open_table(TableDefinition::<u64, u64>::new("fle"))
            .unwrap();
    }
    w.commit().unwrap();
    let r = db.begin_read().unwrap();
    let t = r
        .open_table(TableDefinition::<u64, u64>::new("fle"))
        .unwrap();
    assert!(t.first().unwrap().is_none());
    assert!(t.last().unwrap().is_none());
}

#[test]
fn pop_first_last_drain() {
    let f = tmpfile();
    let db = Database::create(f.path()).unwrap();
    let w = db.begin_write().unwrap();
    {
        let mut t = w
            .open_table(TableDefinition::<u64, u64>::new("pfld"))
            .unwrap();
        for i in 0..5u64 {
            t.insert(&i, &i).unwrap();
        }
        for expected in 0..5u64 {
            let entry = t.pop_first().unwrap().unwrap();
            assert_eq!(entry.0.value(), expected);
            drop(entry);
        }
        assert_eq!(t.len().unwrap(), 0);
    }
    w.commit().unwrap();
}
