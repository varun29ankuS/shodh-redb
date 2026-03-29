use alloc::string::String;
use alloc::vec::Vec;

use crate::TableDefinition;
use crate::error::StorageError;
use crate::ivfpq::kmeans;
use crate::ivfpq::pq::Codebooks;
use crate::ivfpq::types::PostingKey;
use crate::table::ReadableTable;
use crate::transactions::WriteTransaction;

use super::config::{FractalIndexConfig, NO_PARENT};
use super::types::{ClusterMeta, HierarchyKey};

/// Convert a `TableError` to `StorageError`.
fn te(e: crate::error::TableError) -> StorageError {
    e.into_storage_error_or_corrupted("fractal index internal table error")
}

// ---------------------------------------------------------------------------
// Centroid update (Welford's online algorithm)
// ---------------------------------------------------------------------------

/// Add a vector to a cluster's running centroid.
///
/// Updates the f64 sum accumulator and recomputes the f32 centroid.
/// Returns the new population count.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn centroid_add(
    txn: &WriteTransaction,
    centroid_sums_table: &str,
    centroids_table: &str,
    cluster_id: u32,
    dim: usize,
    vector: &[f32],
    old_population: u32,
) -> crate::Result<u32> {
    let new_pop = old_population.saturating_add(1);

    let sums_def = TableDefinition::<u32, &[u8]>::new(centroid_sums_table);
    let mut sums_tbl = txn.open_table(sums_def).map_err(te)?;

    // Load or initialize f64 sums
    let mut sums = Vec::with_capacity(dim);
    if let Some(existing) = sums_tbl.get(cluster_id)? {
        let bytes = existing.value();
        for i in 0..dim {
            let offset = i * 8;
            sums.push(f64::from_le_bytes(
                bytes[offset..offset + 8].try_into().unwrap(),
            ));
        }
    } else {
        sums.resize(dim, 0.0);
    }

    // Update sums
    for (s, &v) in sums.iter_mut().zip(vector.iter()) {
        *s += f64::from(v);
    }

    // Persist sums
    let mut sum_bytes = Vec::with_capacity(dim * 8);
    for &s in &sums {
        sum_bytes.extend_from_slice(&s.to_le_bytes());
    }
    sums_tbl.insert(cluster_id, sum_bytes.as_slice())?;
    drop(sums_tbl);

    // Recompute centroid
    let pop_f64 = f64::from(new_pop);
    let mut centroid_bytes = Vec::with_capacity(dim * 4);
    for &s in &sums {
        centroid_bytes.extend_from_slice(&((s / pop_f64) as f32).to_le_bytes());
    }

    let centroids_def = TableDefinition::<u32, &[u8]>::new(centroids_table);
    let mut cent_tbl = txn.open_table(centroids_def).map_err(te)?;
    cent_tbl.insert(cluster_id, centroid_bytes.as_slice())?;

    Ok(new_pop)
}

/// Remove a vector from a cluster's running centroid.
///
/// Returns the new population count. If population reaches 0, centroid is zeroed.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn centroid_remove(
    txn: &WriteTransaction,
    centroid_sums_table: &str,
    centroids_table: &str,
    cluster_id: u32,
    dim: usize,
    vector: &[f32],
    old_population: u32,
) -> crate::Result<u32> {
    if old_population == 0 {
        return Ok(0);
    }
    let new_pop = old_population - 1;

    let sums_def = TableDefinition::<u32, &[u8]>::new(centroid_sums_table);
    let mut sums_tbl = txn.open_table(sums_def).map_err(te)?;

    let mut sums = Vec::with_capacity(dim);
    if let Some(existing) = sums_tbl.get(cluster_id)? {
        let bytes = existing.value();
        for i in 0..dim {
            let offset = i * 8;
            sums.push(f64::from_le_bytes(
                bytes[offset..offset + 8].try_into().unwrap(),
            ));
        }
    } else {
        sums.resize(dim, 0.0);
    }

    for (s, &v) in sums.iter_mut().zip(vector.iter()) {
        *s -= f64::from(v);
    }

    let mut sum_bytes = Vec::with_capacity(dim * 8);
    for &s in &sums {
        sum_bytes.extend_from_slice(&s.to_le_bytes());
    }
    sums_tbl.insert(cluster_id, sum_bytes.as_slice())?;
    drop(sums_tbl);

    let centroids_def = TableDefinition::<u32, &[u8]>::new(centroids_table);
    let mut cent_tbl = txn.open_table(centroids_def).map_err(te)?;

    if new_pop == 0 {
        let zeros = alloc::vec![0u8; dim * 4];
        cent_tbl.insert(cluster_id, zeros.as_slice())?;
    } else {
        let pop_f64 = f64::from(new_pop);
        let mut centroid_bytes = Vec::with_capacity(dim * 4);
        for &s in &sums {
            centroid_bytes.extend_from_slice(&((s / pop_f64) as f32).to_le_bytes());
        }
        cent_tbl.insert(cluster_id, centroid_bytes.as_slice())?;
    }

    Ok(new_pop)
}

// ---------------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------------

/// Split a leaf cluster into two children using 2-means.
///
/// The original cluster becomes an internal node with two leaf children.
/// When `store_raw_vectors` is disabled, approximate vectors are reconstructed
/// from PQ codes via the provided codebooks.
/// Returns `(child_a_id, child_b_id)`.
pub(crate) fn split_cluster(
    txn: &WriteTransaction,
    config: &mut FractalIndexConfig,
    cluster_id: u32,
    table_names: &TableNames,
    codebooks: Option<&Codebooks>,
) -> crate::Result<(u32, u32)> {
    let dim = config.dim as usize;

    // 1. Collect all vector IDs from the posting list
    let postings_def = TableDefinition::<PostingKey, &[u8]>::new(&table_names.postings);
    let vectors_def = TableDefinition::<u64, &[u8]>::new(&table_names.vectors);

    let mut vector_ids: Vec<u64> = Vec::new();
    let mut flat_vectors: Vec<f32> = Vec::new();

    {
        let tbl = txn.open_table(postings_def).map_err(te)?;
        let range =
            tbl.range(PostingKey::cluster_start(cluster_id)..=PostingKey::cluster_end(cluster_id))?;
        for entry in range {
            let (key, _pq_codes) = entry?;
            let vid = key.value().vector_id;
            vector_ids.push(vid);
        }
    }

    // Load raw vectors from table if available
    if config.store_raw_vectors {
        let vtbl = txn.open_table(vectors_def).map_err(te)?;
        for &vid in &vector_ids {
            if let Some(raw) = vtbl.get(vid)? {
                let bytes = raw.value();
                for i in 0..dim {
                    let offset = i * 4;
                    flat_vectors.push(f32::from_le_bytes(
                        bytes[offset..offset + 4].try_into().unwrap(),
                    ));
                }
            }
        }
    }

    // Fallback: reconstruct approximate vectors from PQ codes when raw vectors unavailable
    if flat_vectors.is_empty()
        && !vector_ids.is_empty()
        && let Some(cb) = codebooks
    {
        let ptbl = txn.open_table(postings_def).map_err(te)?;
        let sub_dim = cb.sub_dim;
        for &vid in &vector_ids {
            if let Some(pq_guard) = ptbl.get(PostingKey::new(cluster_id, vid))? {
                let pq_codes = pq_guard.value();
                for (m, &code) in pq_codes.iter().enumerate().take(cb.num_subvectors) {
                    let base = m * 256 * sub_dim + (code as usize) * sub_dim;
                    for d in 0..sub_dim {
                        flat_vectors.push(cb.data[base + d]);
                    }
                }
            }
        }
    }

    if flat_vectors.len() / dim < 2 {
        return Err(StorageError::Corrupted(
            "fractal: cannot split cluster with fewer than 2 vectors".into(),
        ));
    }

    // 2. Run 2-means
    let centroids_flat = kmeans::kmeans(&flat_vectors, dim, 2, 10, config.metric);

    // Assign each vector to one of the 2 clusters
    let n = vector_ids.len();
    let mut assignments = Vec::with_capacity(n);
    for i in 0..n {
        let vec_slice = &flat_vectors[i * dim..(i + 1) * dim];
        let (cluster_idx, _) =
            kmeans::assign_nearest(vec_slice, &centroids_flat, dim, 2, config.metric);
        assignments.push(cluster_idx);
    }

    // 3. Allocate new cluster IDs
    let child_a = config.alloc_cluster_id();
    let child_b = config.alloc_cluster_id();

    // 4. Compute per-child statistics
    let mut pop_a: u32 = 0;
    let mut pop_b: u32 = 0;
    let mut sums_a = alloc::vec![0.0f64; dim];
    let mut sums_b = alloc::vec![0.0f64; dim];

    for (idx, &assignment) in assignments.iter().enumerate() {
        let vec_start = idx * dim;
        let vec_slice = &flat_vectors[vec_start..vec_start + dim];
        if assignment == 0 {
            pop_a += 1;
            for (s, &v) in sums_a.iter_mut().zip(vec_slice.iter()) {
                *s += f64::from(v);
            }
        } else {
            pop_b += 1;
            for (s, &v) in sums_b.iter_mut().zip(vec_slice.iter()) {
                *s += f64::from(v);
            }
        }
    }

    // 5. Create child ClusterMetas
    let mut meta_a = ClusterMeta::new(child_a, cluster_id, 0, false);
    meta_a.set_population(pop_a);
    let mut meta_b = ClusterMeta::new(child_b, cluster_id, 0, false);
    meta_b.set_population(pop_b);

    // 6. Persist child metadata
    let clusters_def = TableDefinition::<u32, &[u8]>::new(&table_names.clusters);
    {
        let mut ctbl = txn.open_table(clusters_def).map_err(te)?;
        ctbl.insert(child_a, meta_a.as_bytes().as_slice())?;
        ctbl.insert(child_b, meta_b.as_bytes().as_slice())?;
    }

    // 7. Persist child centroids and sums
    let centroids_def = TableDefinition::<u32, &[u8]>::new(&table_names.centroids);
    let sums_def = TableDefinition::<u32, &[u8]>::new(&table_names.centroid_sums);
    {
        let mut cent_tbl = txn.open_table(centroids_def).map_err(te)?;
        let centroid_a: Vec<u8> = centroids_flat[..dim]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let centroid_b: Vec<u8> = centroids_flat[dim..2 * dim]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        cent_tbl.insert(child_a, centroid_a.as_slice())?;
        cent_tbl.insert(child_b, centroid_b.as_slice())?;

        let mut sum_tbl = txn.open_table(sums_def).map_err(te)?;
        let sums_a_bytes: Vec<u8> = sums_a.iter().flat_map(|f| f.to_le_bytes()).collect();
        let sums_b_bytes: Vec<u8> = sums_b.iter().flat_map(|f| f.to_le_bytes()).collect();
        sum_tbl.insert(child_a, sums_a_bytes.as_slice())?;
        sum_tbl.insert(child_b, sums_b_bytes.as_slice())?;
    }

    // 8. Move postings to children
    {
        let mut ptbl = txn.open_table(postings_def).map_err(te)?;
        let mut atbl = txn
            .open_table(TableDefinition::<u64, u32>::new(&table_names.assignments))
            .map_err(te)?;

        // First collect old postings to avoid borrow conflicts
        let mut old_entries: Vec<(u64, Vec<u8>)> = Vec::new();
        {
            let range = ptbl.range(
                PostingKey::cluster_start(cluster_id)..=PostingKey::cluster_end(cluster_id),
            )?;
            for entry in range {
                let (key, val) = entry?;
                let vid = key.value().vector_id;
                old_entries.push((vid, val.value().to_vec()));
            }
        }

        // Remove old entries and insert into children
        for (idx, (vid, pq_codes)) in old_entries.iter().enumerate() {
            ptbl.remove(PostingKey::new(cluster_id, *vid))?;
            let target = if assignments[idx] == 0 {
                child_a
            } else {
                child_b
            };
            ptbl.insert(PostingKey::new(target, *vid), pq_codes.as_slice())?;
            atbl.insert(*vid, target)?;
        }
    }

    // 9. Insert hierarchy edges
    let hier_def = TableDefinition::<HierarchyKey, ()>::new(&table_names.hierarchy);
    {
        let mut htbl = txn.open_table(hier_def).map_err(te)?;
        htbl.insert(HierarchyKey::new(cluster_id, child_a), ())?;
        htbl.insert(HierarchyKey::new(cluster_id, child_b), ())?;
    }

    // 10. Convert original cluster to internal
    {
        let mut ctbl = txn.open_table(clusters_def).map_err(te)?;
        let meta_opt = ctbl
            .get(cluster_id)?
            .map(|g| ClusterMeta::from_bytes(g.value()));
        if let Some(mut meta) = meta_opt {
            meta.set_level(1);
            meta.set_num_children(2);
            meta.set_population(0); // Vectors moved to children
            meta.set_buffer_count(0);
            ctbl.insert(cluster_id, meta.as_bytes().as_slice())?;
        }
    }

    config.num_clusters += 2;

    Ok((child_a, child_b))
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/// Merge a leaf cluster into its nearest sibling.
///
/// Returns the surviving sibling's cluster ID.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn merge_cluster(
    txn: &WriteTransaction,
    config: &mut FractalIndexConfig,
    cluster_id: u32,
    table_names: &TableNames,
) -> crate::Result<Option<u32>> {
    let clusters_def = TableDefinition::<u32, &[u8]>::new(&table_names.clusters);
    let dim = config.dim as usize;

    // Load the cluster to merge
    let meta = {
        let ctbl = txn.open_table(clusters_def).map_err(te)?;
        match ctbl.get(cluster_id)? {
            Some(g) => ClusterMeta::from_bytes(g.value()),
            None => return Ok(None),
        }
    };

    let parent_id = meta.parent_id();
    if parent_id == NO_PARENT {
        // Can't merge the root
        return Ok(None);
    }

    // Find siblings
    let hier_def = TableDefinition::<HierarchyKey, ()>::new(&table_names.hierarchy);
    let centroids_def = TableDefinition::<u32, &[u8]>::new(&table_names.centroids);

    let mut siblings: Vec<u32> = Vec::new();
    {
        let htbl = txn.open_table(hier_def).map_err(te)?;
        let range = htbl.range(
            HierarchyKey::children_start(parent_id)..=HierarchyKey::children_end(parent_id),
        )?;
        for entry in range {
            let (key, _) = entry?;
            let cid = key.value().child_id;
            if cid != cluster_id {
                siblings.push(cid);
            }
        }
    }

    if siblings.is_empty() {
        return Ok(None);
    }

    // Find nearest sibling by centroid distance
    let my_centroid: Vec<f32> = {
        let ctbl = txn.open_table(centroids_def).map_err(te)?;
        match ctbl.get(cluster_id)? {
            Some(g) => {
                let bytes = g.value();
                (0..dim)
                    .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                    .collect()
            }
            None => return Ok(None),
        }
    };

    let mut best_sibling = siblings[0];
    let mut best_dist = f32::MAX;
    {
        let ctbl = txn.open_table(centroids_def).map_err(te)?;
        for &sib in &siblings {
            if let Some(g) = ctbl.get(sib)? {
                let bytes = g.value();
                let sib_centroid: Vec<f32> = (0..dim)
                    .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                    .collect();
                let dist = config.metric.compute(&my_centroid, &sib_centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_sibling = sib;
                }
            }
        }
    }

    // Check if combined population fits
    let sibling_meta = {
        let ctbl = txn.open_table(clusters_def).map_err(te)?;
        match ctbl.get(best_sibling)? {
            Some(g) => ClusterMeta::from_bytes(g.value()),
            None => return Ok(None),
        }
    };

    let combined_pop = meta.population().saturating_add(sibling_meta.population());
    if combined_pop > config.max_leaf_population {
        return Ok(None);
    }

    // Move all postings from this cluster to the sibling
    let postings_def = TableDefinition::<PostingKey, &[u8]>::new(&table_names.postings);
    let assignments_def = TableDefinition::<u64, u32>::new(&table_names.assignments);
    {
        let mut ptbl = txn.open_table(postings_def).map_err(te)?;
        let mut atbl = txn.open_table(assignments_def).map_err(te)?;

        let mut entries: Vec<(u64, Vec<u8>)> = Vec::new();
        {
            let range = ptbl.range(
                PostingKey::cluster_start(cluster_id)..=PostingKey::cluster_end(cluster_id),
            )?;
            for entry in range {
                let (key, val) = entry?;
                entries.push((key.value().vector_id, val.value().to_vec()));
            }
        }

        for (vid, pq_codes) in &entries {
            ptbl.remove(PostingKey::new(cluster_id, *vid))?;
            ptbl.insert(PostingKey::new(best_sibling, *vid), pq_codes.as_slice())?;
            atbl.insert(*vid, best_sibling)?;
        }
    }

    // Move buffer entries from the merged cluster to the sibling so they are
    // not orphaned. Without this, any vectors in the merged cluster's unflushed
    // buffer would be silently lost.
    let buffer_def = TableDefinition::<PostingKey, &[u8]>::new(&table_names.buffer);
    {
        let mut btbl = txn.open_table(buffer_def).map_err(te)?;
        let mut buf_entries: Vec<(u64, Vec<u8>)> = Vec::new();
        {
            let range = btbl.range(
                PostingKey::cluster_start(cluster_id)..=PostingKey::cluster_end(cluster_id),
            )?;
            for entry in range {
                let (key, val) = entry?;
                buf_entries.push((key.value().vector_id, val.value().to_vec()));
            }
        }
        for (vid, raw_vec) in &buf_entries {
            btbl.remove(PostingKey::new(cluster_id, *vid))?;
            btbl.insert(PostingKey::new(best_sibling, *vid), raw_vec.as_slice())?;
        }
    }

    // Update sibling metadata (weighted centroid merge via sums)
    let sums_def = TableDefinition::<u32, &[u8]>::new(&table_names.centroid_sums);
    {
        let mut sum_tbl = txn.open_table(sums_def).map_err(te)?;

        let my_sums: Vec<f64> = match sum_tbl.get(cluster_id)? {
            Some(g) => {
                let bytes = g.value();
                (0..dim)
                    .map(|i| f64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()))
                    .collect()
            }
            None => alloc::vec![0.0; dim],
        };

        let sib_sums: Vec<f64> = match sum_tbl.get(best_sibling)? {
            Some(g) => {
                let bytes = g.value();
                (0..dim)
                    .map(|i| f64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()))
                    .collect()
            }
            None => alloc::vec![0.0; dim],
        };

        let merged_sums: Vec<f64> = my_sums
            .iter()
            .zip(sib_sums.iter())
            .map(|(a, b)| a + b)
            .collect();
        let merged_bytes: Vec<u8> = merged_sums.iter().flat_map(|f| f.to_le_bytes()).collect();
        sum_tbl.insert(best_sibling, merged_bytes.as_slice())?;

        // Remove merged cluster's sums
        sum_tbl.remove(cluster_id)?;
    }

    // Recompute sibling centroid from merged sums
    {
        let sum_tbl = txn.open_table(sums_def).map_err(te)?;
        let merged_sums: Vec<f64> = match sum_tbl.get(best_sibling)? {
            Some(g) => {
                let bytes = g.value();
                (0..dim)
                    .map(|i| f64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()))
                    .collect()
            }
            None => alloc::vec![0.0; dim],
        };
        drop(sum_tbl);

        let pop_f64 = f64::from(combined_pop);
        let centroid_bytes: Vec<u8> = merged_sums
            .iter()
            .map(|s| (s / pop_f64) as f32)
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let mut cent_tbl = txn.open_table(centroids_def).map_err(te)?;
        cent_tbl.insert(best_sibling, centroid_bytes.as_slice())?;
    }

    // Update sibling population
    {
        let mut ctbl = txn.open_table(clusters_def).map_err(te)?;
        let sib_opt = ctbl
            .get(best_sibling)?
            .map(|g| ClusterMeta::from_bytes(g.value()));
        if let Some(mut sib) = sib_opt {
            sib.set_population(combined_pop);
            ctbl.insert(best_sibling, sib.as_bytes().as_slice())?;
        }
    }

    // Remove merged cluster from hierarchy and metadata
    {
        let mut htbl = txn.open_table(hier_def).map_err(te)?;
        htbl.remove(HierarchyKey::new(parent_id, cluster_id))?;
    }
    {
        let mut ctbl = txn.open_table(clusters_def).map_err(te)?;
        ctbl.remove(cluster_id)?;
    }
    {
        let mut cent_tbl = txn.open_table(centroids_def).map_err(te)?;
        cent_tbl.remove(cluster_id)?;
    }

    // Update parent's children count
    {
        let mut ctbl = txn.open_table(clusters_def).map_err(te)?;
        let parent_opt = ctbl
            .get(parent_id)?
            .map(|g| ClusterMeta::from_bytes(g.value()));
        if let Some(mut parent_meta) = parent_opt {
            let new_count = parent_meta.num_children().saturating_sub(1);
            parent_meta.set_num_children(new_count);
            ctbl.insert(parent_id, parent_meta.as_bytes().as_slice())?;
        }
    }

    config.num_clusters = config.num_clusters.saturating_sub(1);
    Ok(Some(best_sibling))
}

// ---------------------------------------------------------------------------
// Buffer cascade
// ---------------------------------------------------------------------------

/// Flush a leaf cluster's buffer: PQ-encode all buffered vectors and move
/// them to the posting list.
pub(crate) fn cascade_leaf_buffer(
    txn: &WriteTransaction,
    cluster_id: u32,
    codebooks: &Codebooks,
    config: &FractalIndexConfig,
    table_names: &TableNames,
) -> crate::Result<()> {
    let dim = config.dim as usize;
    let buffer_def = TableDefinition::<PostingKey, &[u8]>::new(&table_names.buffer);
    let postings_def = TableDefinition::<PostingKey, &[u8]>::new(&table_names.postings);
    let vectors_def = TableDefinition::<u64, &[u8]>::new(&table_names.vectors);

    // Collect buffered entries
    let mut entries: Vec<(u64, Vec<f32>)> = Vec::new();
    {
        let btbl = txn.open_table(buffer_def).map_err(te)?;
        let range = btbl
            .range(PostingKey::cluster_start(cluster_id)..=PostingKey::cluster_end(cluster_id))?;
        for entry in range {
            let (key, val) = entry?;
            let vid = key.value().vector_id;
            let bytes = val.value();
            let vec: Vec<f32> = (0..dim)
                .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                .collect();
            entries.push((vid, vec));
        }
    }

    if entries.is_empty() {
        return Ok(());
    }

    // PQ-encode and insert into posting list
    {
        let mut ptbl = txn.open_table(postings_def).map_err(te)?;
        let mut vtbl_opt = if config.store_raw_vectors {
            Some(txn.open_table(vectors_def).map_err(te)?)
        } else {
            None
        };

        for (vid, vec) in &entries {
            let pq_codes = codebooks.encode(vec);
            ptbl.insert(PostingKey::new(cluster_id, *vid), pq_codes.as_slice())?;

            if let Some(ref mut vtbl) = vtbl_opt {
                let raw_bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
                vtbl.insert(*vid, raw_bytes.as_slice())?;
            }
        }
    }

    // Remove from buffer
    {
        let mut btbl = txn.open_table(buffer_def).map_err(te)?;
        for (vid, _) in &entries {
            btbl.remove(PostingKey::new(cluster_id, *vid))?;
        }
    }

    // Update cluster meta buffer_count
    let clusters_def = TableDefinition::<u32, &[u8]>::new(&table_names.clusters);
    {
        let mut ctbl = txn.open_table(clusters_def).map_err(te)?;
        let meta_opt = ctbl
            .get(cluster_id)?
            .map(|g| ClusterMeta::from_bytes(g.value()));
        if let Some(mut meta) = meta_opt {
            meta.set_buffer_count(0);
            ctbl.insert(cluster_id, meta.as_bytes().as_slice())?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Table name helper
// ---------------------------------------------------------------------------

/// Pre-computed table names for a fractal index instance.
#[derive(Clone)]
pub(crate) struct TableNames {
    pub meta: String,
    pub clusters: String,
    pub centroids: String,
    pub centroid_sums: String,
    pub hierarchy: String,
    pub buffer: String,
    pub postings: String,
    pub assignments: String,
    pub vectors: String,
    pub codebooks: String,
}

impl TableNames {
    pub fn new(name: &str) -> Self {
        Self {
            meta: alloc::format!("__fractal:{name}:meta"),
            clusters: alloc::format!("__fractal:{name}:clusters"),
            centroids: alloc::format!("__fractal:{name}:centroids"),
            centroid_sums: alloc::format!("__fractal:{name}:centroid_sums"),
            hierarchy: alloc::format!("__fractal:{name}:hierarchy"),
            buffer: alloc::format!("__fractal:{name}:buffer"),
            postings: alloc::format!("__fractal:{name}:postings"),
            assignments: alloc::format!("__fractal:{name}:assignments"),
            vectors: alloc::format!("__fractal:{name}:vectors"),
            codebooks: alloc::format!("__fractal:{name}:codebooks"),
        }
    }
}
