#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use redb::ivfpq::cluster_blob::{
    encode_cluster_blob, merge_into_blob, remove_from_blob, ClusterBlobRef,
};

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    /// Parse arbitrary bytes as a cluster blob — must not panic.
    ParseRaw {
        data: Vec<u8>,
        pq_len: u16,
        dim_sel: u8,
    },
    /// Encode entries then parse — roundtrip.
    EncodeRoundtrip {
        /// Number of entries (clamped to 1..8).
        num_entries: u8,
        /// PQ code length (clamped to 1..16).
        pq_len: u8,
        /// Dimension selector: 0..3 → 2/4/8/16.
        dim_sel: u8,
        /// Raw u32 bits for vector IDs.
        vector_ids: Vec<u64>,
        /// Raw u32 bits for PQ codes.
        pq_data: Vec<u8>,
    },
    /// Merge a new entry into an existing blob.
    MergeEntry {
        pq_len: u8,
        dim_sel: u8,
        existing_ids: Vec<u64>,
        existing_pq: Vec<u8>,
        new_id: u64,
        new_pq: Vec<u8>,
    },
    /// Remove an entry from a blob.
    RemoveEntry {
        pq_len: u8,
        dim_sel: u8,
        ids: Vec<u64>,
        pq_data: Vec<u8>,
        remove_id: u64,
    },
}

fn select_dim(sel: u8) -> usize {
    match sel % 4 {
        0 => 2,
        1 => 4,
        2 => 8,
        _ => 16,
    }
}

fuzz_target!(|op: FuzzOp| {
    match op {
        FuzzOp::ParseRaw {
            data,
            pq_len,
            dim_sel,
        } => {
            let dim = select_dim(dim_sel);
            let pq_len = pq_len.max(1);
            // Must not panic on any input.
            let _ = ClusterBlobRef::new(&data, pq_len, dim);
        }

        FuzzOp::EncodeRoundtrip {
            num_entries,
            pq_len,
            dim_sel,
            vector_ids,
            pq_data,
        } => {
            let dim = select_dim(dim_sel);
            let pq_len = (pq_len as u16).clamp(1, 16);
            let n = (num_entries as usize).clamp(1, 8).min(vector_ids.len());
            if n == 0 {
                return;
            }

            // Build entries with no raw vectors (simpler path).
            let mut entries: Vec<(u64, Vec<u8>, Option<Vec<f32>>)> = Vec::with_capacity(n);
            for i in 0..n {
                let id = vector_ids[i % vector_ids.len()];
                let pq_start = (i * pq_len as usize) % pq_data.len().max(1);
                let codes: Vec<u8> = (0..pq_len as usize)
                    .map(|j| {
                        pq_data
                            .get((pq_start + j) % pq_data.len().max(1))
                            .copied()
                            .unwrap_or(0)
                    })
                    .collect();
                entries.push((id, codes, None));
            }

            // Build the tuple refs that encode_cluster_blob expects.
            let entry_refs: Vec<(u64, &[u8], Option<&[f32]>)> = entries
                .iter()
                .map(|(id, codes, _)| (*id, codes.as_slice(), None))
                .collect();

            let blob = encode_cluster_blob(&entry_refs, pq_len);
            let parsed = ClusterBlobRef::new(&blob, pq_len, dim);
            if let Ok(parsed) = parsed {
                assert_eq!(parsed.count() as usize, n);
                // Verify all IDs are findable.
                for (id, _, _) in &entry_refs {
                    assert!(parsed.find_vector(*id).is_some());
                }
            }
        }

        FuzzOp::MergeEntry {
            pq_len,
            dim_sel,
            existing_ids,
            existing_pq,
            new_id,
            new_pq,
        } => {
            let dim = select_dim(dim_sel);
            let pq_len = (pq_len as u16).clamp(1, 16);
            if existing_ids.is_empty() {
                return;
            }
            let n = existing_ids.len().min(4);

            let mut entries: Vec<(u64, Vec<u8>, Option<Vec<f32>>)> = Vec::new();
            for i in 0..n {
                let codes: Vec<u8> = (0..pq_len as usize)
                    .map(|j| {
                        existing_pq
                            .get((i * pq_len as usize + j) % existing_pq.len().max(1))
                            .copied()
                            .unwrap_or(0)
                    })
                    .collect();
                entries.push((existing_ids[i], codes, None));
            }
            let entry_refs: Vec<(u64, &[u8], Option<&[f32]>)> = entries
                .iter()
                .map(|(id, codes, _)| (*id, codes.as_slice(), None))
                .collect();
            let blob = encode_cluster_blob(&entry_refs, pq_len);

            if let Ok(parsed) = ClusterBlobRef::new(&blob, pq_len, dim) {
                let new_codes: Vec<u8> = (0..pq_len as usize)
                    .map(|j| new_pq.get(j % new_pq.len().max(1)).copied().unwrap_or(0))
                    .collect();
                let mut new_entries = vec![(new_id, new_codes, None)];
                let merged = merge_into_blob(Some(&parsed), &mut new_entries, pq_len);
                // Merged blob must parse.
                let _ = ClusterBlobRef::new(&merged, pq_len, dim);
            }
        }

        FuzzOp::RemoveEntry {
            pq_len,
            dim_sel,
            ids,
            pq_data,
            remove_id,
        } => {
            let dim = select_dim(dim_sel);
            let pq_len = (pq_len as u16).clamp(1, 16);
            if ids.is_empty() {
                return;
            }
            let n = ids.len().min(4);
            let mut entries: Vec<(u64, Vec<u8>, Option<Vec<f32>>)> = Vec::new();
            for i in 0..n {
                let codes: Vec<u8> = (0..pq_len as usize)
                    .map(|j| {
                        pq_data
                            .get((i * pq_len as usize + j) % pq_data.len().max(1))
                            .copied()
                            .unwrap_or(0)
                    })
                    .collect();
                entries.push((ids[i], codes, None));
            }
            let entry_refs: Vec<(u64, &[u8], Option<&[f32]>)> = entries
                .iter()
                .map(|(id, codes, _)| (*id, codes.as_slice(), None))
                .collect();
            let blob = encode_cluster_blob(&entry_refs, pq_len);

            if let Ok(parsed) = ClusterBlobRef::new(&blob, pq_len, dim) {
                let _ = remove_from_blob(&parsed, remove_id, pq_len);
            }
        }
    }
});
