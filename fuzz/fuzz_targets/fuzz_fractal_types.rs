#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use redb::fractal::types::{ClusterMeta, HierarchyKey, CLUSTER_META_SIZE};
use redb::{Key, Value};

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    /// ClusterMeta::from_bytes with arbitrary data (short, full, oversized).
    ClusterMetaFromBytes { data: Vec<u8> },
    /// ClusterMeta field set/get roundtrip.
    ClusterMetaRoundtrip {
        cluster_id: u32,
        parent_id: u32,
        level: u8,
        is_root: bool,
        population: u32,
        buffer_count: u32,
        num_children: u16,
        variance_bits: u64,
        oldest_hlc: u64,
        newest_hlc: u64,
        oldest_wall_ns: u64,
        newest_wall_ns: u64,
    },
    /// HierarchyKey::from_be_bytes with arbitrary (possibly truncated) data.
    HierarchyKeyFromBytes { data: Vec<u8> },
    /// HierarchyKey roundtrip and ordering.
    HierarchyKeyRoundtrip { parent_id: u32, child_id: u32 },
    /// HierarchyKey Key::compare consistency with Ord.
    HierarchyKeyCompare {
        a_parent: u32,
        a_child: u32,
        b_parent: u32,
        b_child: u32,
    },
    /// ClusterMeta from_bytes then mutate and re-serialize.
    ClusterMetaMutateRoundtrip {
        initial_data: Vec<u8>,
        new_level: u8,
        new_population: u32,
        new_flags: u8,
    },
}

fuzz_target!(|op: FuzzOp| {
    match op {
        FuzzOp::ClusterMetaFromBytes { data } => {
            // Must not panic for any input.
            let meta = ClusterMeta::from_bytes(&data);
            // All accessors must not panic.
            let _ = meta.cluster_id();
            let _ = meta.parent_id();
            let _ = meta.level();
            let _ = meta.flags();
            let _ = meta.is_root();
            let _ = meta.is_leaf();
            let _ = meta.num_children();
            let _ = meta.population();
            let _ = meta.buffer_count();
            let _ = meta.sum_variance();
            let _ = meta.oldest_hlc();
            let _ = meta.newest_hlc();
            let _ = meta.oldest_wall_ns();
            let _ = meta.newest_wall_ns();
            let _ = meta.has_no_parent();

            // as_bytes should return exactly CLUSTER_META_SIZE bytes.
            let bytes = meta.as_bytes();
            assert_eq!(bytes.len(), CLUSTER_META_SIZE);
        }

        FuzzOp::ClusterMetaRoundtrip {
            cluster_id,
            parent_id,
            level,
            is_root,
            population,
            buffer_count,
            num_children,
            variance_bits,
            oldest_hlc,
            newest_hlc,
            oldest_wall_ns,
            newest_wall_ns,
        } => {
            let mut meta = ClusterMeta::new(cluster_id, parent_id, level, is_root);
            meta.set_population(population);
            meta.set_buffer_count(buffer_count);
            meta.set_num_children(num_children);
            let variance = f64::from_bits(variance_bits);
            meta.set_sum_variance(variance);
            meta.set_oldest_hlc(oldest_hlc);
            meta.set_newest_hlc(newest_hlc);
            meta.set_oldest_wall_ns(oldest_wall_ns);
            meta.set_newest_wall_ns(newest_wall_ns);

            // Serialize and deserialize.
            let bytes = meta.as_bytes();
            let restored = ClusterMeta::from_bytes(bytes);

            assert_eq!(restored.cluster_id(), cluster_id);
            assert_eq!(restored.parent_id(), parent_id);
            assert_eq!(restored.level(), level);
            assert_eq!(restored.population(), population);
            assert_eq!(restored.buffer_count(), buffer_count);
            assert_eq!(restored.num_children(), num_children);
            assert_eq!(restored.oldest_hlc(), oldest_hlc);
            assert_eq!(restored.newest_hlc(), newest_hlc);
            assert_eq!(restored.oldest_wall_ns(), oldest_wall_ns);
            assert_eq!(restored.newest_wall_ns(), newest_wall_ns);

            // Variance roundtrip (NaN-aware).
            let rest_var = restored.sum_variance();
            if variance.is_nan() {
                assert!(rest_var.is_nan());
            } else {
                assert_eq!(rest_var.to_bits(), variance.to_bits());
            }
        }

        FuzzOp::HierarchyKeyFromBytes { data } => {
            // Must not panic. Truncated data returns zeroed key.
            let key = HierarchyKey::from_be_bytes(&data);
            if data.len() < HierarchyKey::SERIALIZED_SIZE {
                assert_eq!(key.parent_id, 0);
                assert_eq!(key.child_id, 0);
            }
        }

        FuzzOp::HierarchyKeyRoundtrip { parent_id, child_id } => {
            let key = HierarchyKey::new(parent_id, child_id);
            let bytes = key.to_be_bytes();
            let restored = HierarchyKey::from_be_bytes(&bytes);
            assert_eq!(restored.parent_id, parent_id);
            assert_eq!(restored.child_id, child_id);

            // Value trait roundtrip.
            let val_bytes = <HierarchyKey as Value>::as_bytes(&key);
            let val_restored = <HierarchyKey as Value>::from_bytes(&val_bytes);
            assert_eq!(val_restored.parent_id, parent_id);
            assert_eq!(val_restored.child_id, child_id);
        }

        FuzzOp::HierarchyKeyCompare {
            a_parent,
            a_child,
            b_parent,
            b_child,
        } => {
            let a = HierarchyKey::new(a_parent, a_child);
            let b = HierarchyKey::new(b_parent, b_child);

            let a_bytes = <HierarchyKey as Value>::as_bytes(&a);
            let b_bytes = <HierarchyKey as Value>::as_bytes(&b);

            // Key::compare must match Ord.
            let key_cmp = <HierarchyKey as Key>::compare(&a_bytes, &b_bytes);
            let ord_cmp = a.cmp(&b);
            assert_eq!(
                key_cmp, ord_cmp,
                "Key::compare disagrees with Ord for {:?} vs {:?}",
                a, b
            );

            // Big-endian byte comparison must also match.
            let byte_cmp = a_bytes.cmp(&b_bytes);
            assert_eq!(
                byte_cmp, ord_cmp,
                "BE byte comparison disagrees with Ord for {:?} vs {:?}",
                a, b
            );
        }

        FuzzOp::ClusterMetaMutateRoundtrip {
            initial_data,
            new_level,
            new_population,
            new_flags,
        } => {
            let mut meta = ClusterMeta::from_bytes(&initial_data);
            meta.set_level(new_level);
            meta.set_population(new_population);
            meta.set_flags(new_flags);

            let bytes = meta.as_bytes();
            let restored = ClusterMeta::from_bytes(bytes);
            assert_eq!(restored.level(), new_level);
            assert_eq!(restored.population(), new_population);
            assert_eq!(restored.flags(), new_flags);
        }
    }
});
