use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Ordering as CmpOrdering;

use crate::TableDefinition;
use crate::error::StorageError;
use crate::probe_select::DiversityConfig;
use crate::table::ReadableTable;
use crate::transactions::{ReadTransaction, WriteTransaction};
use crate::vector_ops::{DistanceMetric, Neighbor, l2_normalize};

use crate::ivfpq::adc::AdcTable;
use crate::ivfpq::types::PostingKey;

use super::cluster::TableNames;
use super::config::FractalSearchParams;
use super::index::{FractalIndex, ReadOnlyFractalIndex};
use super::types::{ClusterMeta, HierarchyKey};

/// Convert a `TableError` to `StorageError`.
fn te(e: crate::error::TableError) -> StorageError {
    e.into_storage_error_or_corrupted("fractal search internal table error")
}

// ---------------------------------------------------------------------------
// CandidateHeap -- fixed-capacity max-heap for top-K selection
// ---------------------------------------------------------------------------

struct CandidateEntry {
    vector_id: u64,
    distance: f32,
}

impl PartialEq for CandidateEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for CandidateEntry {}

impl PartialOrd for CandidateEntry {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for CandidateEntry {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Max-heap: worst (largest) distance at top for eviction
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(CmpOrdering::Equal)
    }
}

struct CandidateHeap {
    capacity: usize,
    heap: BinaryHeap<CandidateEntry>,
}

impl CandidateHeap {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::with_capacity(capacity + 1),
        }
    }

    fn push(&mut self, vector_id: u64, distance: f32) {
        if self.heap.len() < self.capacity {
            self.heap.push(CandidateEntry {
                vector_id,
                distance,
            });
        } else if let Some(worst) = self.heap.peek()
            && distance < worst.distance
        {
            self.heap.pop();
            self.heap.push(CandidateEntry {
                vector_id,
                distance,
            });
        }
    }

    fn into_sorted(self, k: usize) -> Vec<Neighbor<u64>> {
        let mut entries: Vec<_> = self.heap.into_vec();
        entries.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(CmpOrdering::Equal)
        });
        entries
            .into_iter()
            .take(k)
            .map(|e| Neighbor {
                key: e.vector_id,
                distance: e.distance,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Search implementation -- write transaction path
// ---------------------------------------------------------------------------

pub(crate) fn search_write(
    idx: &mut FractalIndex<'_>,
    query: &[f32],
    params: &FractalSearchParams,
) -> crate::Result<Vec<Neighbor<u64>>> {
    let dim = idx.config.dim as usize;
    if query.len() != dim {
        return Err(StorageError::Corrupted(alloc::format!(
            "fractal: search query dim mismatch: expected {dim}, got {}",
            query.len(),
        )));
    }

    let q = if idx.config.metric == DistanceMetric::Cosine {
        let mut v = query.to_vec();
        l2_normalize(&mut v);
        v
    } else {
        query.to_vec()
    };

    let codebooks = idx.codebooks.as_ref().ok_or_else(|| {
        StorageError::Corrupted("fractal: search called on index without codebooks".to_string())
    })?;
    let adc = AdcTable::build(&q, codebooks, idx.config.metric);
    let nprobe = params.nprobe.max(1);
    let heap_cap = if params.rerank {
        params.candidates
    } else {
        params.k
    };
    let mut heap = CandidateHeap::new(heap_cap);

    // Collect leaf clusters to scan via beam search
    let leaves = beam_search_leaves_write(
        idx.txn,
        &idx.names,
        &idx.config,
        idx.config.root_cluster_id,
        &q,
        nprobe,
        params.min_hlc,
        params.diversity,
    )?;

    // Scan posting lists and buffers of selected leaves
    let postings_def = TableDefinition::<PostingKey, &[u8]>::new(&idx.names.postings);
    let buffer_def = TableDefinition::<PostingKey, &[u8]>::new(&idx.names.buffer);

    let ptbl = idx.txn.open_table(postings_def).map_err(te)?;
    let btbl = idx.txn.open_table(buffer_def).map_err(te)?;

    for leaf_id in &leaves {
        // Scan posting list with ADC
        let range =
            ptbl.range(PostingKey::cluster_start(*leaf_id)..=PostingKey::cluster_end(*leaf_id))?;
        for entry in range {
            let (key, val) = entry?;
            let vid = key.value().vector_id;
            let pq_codes = val.value();
            let dist = adc.approximate_distance(pq_codes);
            heap.push(vid, dist);
        }

        // Scan buffer with exact distance
        let brange =
            btbl.range(PostingKey::cluster_start(*leaf_id)..=PostingKey::cluster_end(*leaf_id))?;
        for entry in brange {
            let (key, val) = entry?;
            let vid = key.value().vector_id;
            let bytes = val.value();
            if bytes.len() < dim * 4 {
                continue;
            }
            let vec: Vec<f32> = (0..dim)
                .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                .collect();
            let dist = idx.config.metric.compute(&q, &vec);
            heap.push(vid, dist);
        }
    }

    drop(ptbl);
    drop(btbl);

    // Rerank with raw vectors if available
    if params.rerank && idx.config.store_raw_vectors {
        let candidates = heap.into_sorted(params.candidates);
        let vectors_def = TableDefinition::<u64, &[u8]>::new(&idx.names.vectors);
        let vtbl = idx.txn.open_table(vectors_def).map_err(te)?;

        let mut reranked: Vec<Neighbor<u64>> = Vec::with_capacity(candidates.len());
        for c in &candidates {
            if let Some(g) = vtbl.get(c.key)? {
                let bytes = g.value();
                if bytes.len() < dim * 4 {
                    reranked.push(Neighbor {
                        key: c.key,
                        distance: c.distance,
                    });
                    continue;
                }
                let vec: Vec<f32> = (0..dim)
                    .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                    .collect();
                let dist = idx.config.metric.compute(&q, &vec);
                reranked.push(Neighbor {
                    key: c.key,
                    distance: dist,
                });
            } else {
                reranked.push(Neighbor {
                    key: c.key,
                    distance: c.distance,
                });
            }
        }
        reranked.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(CmpOrdering::Equal)
        });
        reranked.truncate(params.k);
        Ok(reranked)
    } else {
        Ok(heap.into_sorted(params.k))
    }
}

// ---------------------------------------------------------------------------
// Search implementation -- read transaction path
// ---------------------------------------------------------------------------

pub(crate) fn search_read(
    idx: &ReadOnlyFractalIndex,
    txn: &ReadTransaction,
    query: &[f32],
    params: &FractalSearchParams,
) -> crate::Result<Vec<Neighbor<u64>>> {
    let dim = idx.config.dim as usize;
    if query.len() != dim {
        return Err(StorageError::Corrupted(alloc::format!(
            "fractal: search query dim mismatch: expected {dim}, got {}",
            query.len(),
        )));
    }

    let q = if idx.config.metric == DistanceMetric::Cosine {
        let mut v = query.to_vec();
        l2_normalize(&mut v);
        v
    } else {
        query.to_vec()
    };

    let adc = AdcTable::build(&q, &idx.codebooks, idx.config.metric);
    let nprobe = params.nprobe.max(1);
    let heap_cap = if params.rerank {
        params.candidates
    } else {
        params.k
    };
    let mut heap = CandidateHeap::new(heap_cap);

    let leaves = beam_search_leaves_read(
        txn,
        &idx.names,
        &idx.config,
        idx.config.root_cluster_id,
        &q,
        nprobe,
        params.min_hlc,
        params.diversity,
    )?;

    let postings_def = TableDefinition::<PostingKey, &[u8]>::new(&idx.names.postings);
    let buffer_def = TableDefinition::<PostingKey, &[u8]>::new(&idx.names.buffer);

    let ptbl = txn.open_table(postings_def).map_err(te)?;
    let btbl = txn.open_table(buffer_def).map_err(te)?;

    for leaf_id in &leaves {
        let range =
            ptbl.range(PostingKey::cluster_start(*leaf_id)..=PostingKey::cluster_end(*leaf_id))?;
        for entry in range {
            let (key, val) = entry?;
            let vid = key.value().vector_id;
            let pq_codes = val.value();
            let dist = adc.approximate_distance(pq_codes);
            heap.push(vid, dist);
        }

        let brange =
            btbl.range(PostingKey::cluster_start(*leaf_id)..=PostingKey::cluster_end(*leaf_id))?;
        for entry in brange {
            let (key, val) = entry?;
            let vid = key.value().vector_id;
            let bytes = val.value();
            if bytes.len() < dim * 4 {
                continue;
            }
            let vec: Vec<f32> = (0..dim)
                .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                .collect();
            let dist = idx.config.metric.compute(&q, &vec);
            heap.push(vid, dist);
        }
    }

    drop(ptbl);
    drop(btbl);

    if params.rerank && idx.config.store_raw_vectors {
        let candidates = heap.into_sorted(params.candidates);
        let vectors_def = TableDefinition::<u64, &[u8]>::new(&idx.names.vectors);
        let vtbl = txn.open_table(vectors_def).map_err(te)?;

        let mut reranked: Vec<Neighbor<u64>> = Vec::with_capacity(candidates.len());
        for c in &candidates {
            if let Some(g) = vtbl.get(c.key)? {
                let bytes = g.value();
                if bytes.len() < dim * 4 {
                    reranked.push(Neighbor {
                        key: c.key,
                        distance: c.distance,
                    });
                    continue;
                }
                let vec: Vec<f32> = (0..dim)
                    .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                    .collect();
                let dist = idx.config.metric.compute(&q, &vec);
                reranked.push(Neighbor {
                    key: c.key,
                    distance: dist,
                });
            } else {
                reranked.push(Neighbor {
                    key: c.key,
                    distance: c.distance,
                });
            }
        }
        reranked.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(CmpOrdering::Equal)
        });
        reranked.truncate(params.k);
        Ok(reranked)
    } else {
        Ok(heap.into_sorted(params.k))
    }
}

// ---------------------------------------------------------------------------
// Beam search helpers
// ---------------------------------------------------------------------------

/// Beam search: walk the cluster tree from root to leaves, selecting the
/// best `nprobe` children at each internal level.
#[allow(clippy::too_many_arguments)]
fn beam_search_leaves_write(
    txn: &WriteTransaction,
    names: &TableNames,
    config: &super::config::FractalIndexConfig,
    root: u32,
    query: &[f32],
    nprobe: u32,
    min_hlc: u64,
    diversity: DiversityConfig,
) -> crate::Result<Vec<u32>> {
    let dim = config.dim as usize;
    let clusters_def = TableDefinition::<u32, &[u8]>::new(&names.clusters);
    let centroids_def = TableDefinition::<u32, &[u8]>::new(&names.centroids);
    let hier_def = TableDefinition::<HierarchyKey, ()>::new(&names.hierarchy);

    let mut current_level = alloc::vec![root];
    let mut leaves: Vec<u32> = Vec::new();

    loop {
        let mut next_level: Vec<(u32, f32)> = Vec::new();
        let mut next_centroids: Vec<f32> = Vec::new();
        let collect_centroids = diversity.enabled();

        for &node_id in &current_level {
            let meta = {
                let ctbl = txn.open_table(clusters_def).map_err(te)?;
                match ctbl.get(node_id)? {
                    Some(g) => ClusterMeta::from_bytes(g.value()),
                    None => continue,
                }
            };

            if meta.is_leaf() {
                // Apply temporal filter
                if min_hlc > 0 && meta.newest_hlc() > 0 && meta.newest_hlc() < min_hlc {
                    continue;
                }
                leaves.push(node_id);
                continue;
            }

            // Internal: collect children with distances
            let htbl = txn.open_table(hier_def).map_err(te)?;
            let ctbl = txn.open_table(centroids_def).map_err(te)?;
            let cltbl = txn.open_table(clusters_def).map_err(te)?;

            let range = htbl.range(
                HierarchyKey::children_start(node_id)..=HierarchyKey::children_end(node_id),
            )?;

            for entry in range {
                let (key, _) = entry?;
                let child_id = key.value().child_id;

                // Temporal filter on child
                if min_hlc > 0
                    && let Some(cg) = cltbl.get(child_id)?
                {
                    let child_meta = ClusterMeta::from_bytes(cg.value());
                    if child_meta.newest_hlc() > 0 && child_meta.newest_hlc() < min_hlc {
                        continue;
                    }
                }

                if let Some(cg) = ctbl.get(child_id)? {
                    let bytes = cg.value();
                    if bytes.len() < dim * 4 {
                        continue;
                    }
                    let centroid: Vec<f32> = (0..dim)
                        .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                        .collect();
                    let dist = config.metric.compute(query, &centroid);
                    next_level.push((child_id, dist));
                    if collect_centroids {
                        next_centroids.extend_from_slice(&centroid);
                    }
                }
            }
        }

        if next_level.is_empty() {
            break;
        }

        let nprobe_usize = nprobe as usize;

        if collect_centroids && next_level.len() > nprobe_usize {
            // Sort with index tracking so we can reorder centroids in lockstep
            let mut indexed: Vec<(usize, u32, f32)> = next_level
                .iter()
                .enumerate()
                .map(|(i, &(id, d))| (i, id, d))
                .collect();
            indexed.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(CmpOrdering::Equal));

            let sorted_candidates: Vec<(u32, f32)> =
                indexed.iter().map(|&(_, id, d)| (id, d)).collect();
            let mut sorted_centroids: Vec<f32> = Vec::with_capacity(indexed.len() * dim);
            for &(orig_idx, _, _) in &indexed {
                sorted_centroids
                    .extend_from_slice(&next_centroids[orig_idx * dim..(orig_idx + 1) * dim]);
            }

            let selected = crate::probe_select::select_diverse_probes(
                &sorted_candidates,
                &sorted_centroids,
                dim,
                nprobe_usize,
                diversity,
                config.metric,
            );
            current_level = selected.iter().map(|(id, _)| *id).collect();
        } else {
            // Fast path: pure distance ranking
            next_level.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(CmpOrdering::Equal));
            if next_level.len() > nprobe_usize {
                next_level.truncate(nprobe_usize);
            }
            current_level = next_level.iter().map(|(id, _)| *id).collect();
        }
    }

    Ok(leaves)
}

/// Beam search for read transactions (same logic, different table access).
#[allow(clippy::too_many_arguments)]
fn beam_search_leaves_read(
    txn: &ReadTransaction,
    names: &TableNames,
    config: &super::config::FractalIndexConfig,
    root: u32,
    query: &[f32],
    nprobe: u32,
    min_hlc: u64,
    diversity: DiversityConfig,
) -> crate::Result<Vec<u32>> {
    let dim = config.dim as usize;
    let clusters_def = TableDefinition::<u32, &[u8]>::new(&names.clusters);
    let centroids_def = TableDefinition::<u32, &[u8]>::new(&names.centroids);
    let hier_def = TableDefinition::<HierarchyKey, ()>::new(&names.hierarchy);

    let mut current_level = alloc::vec![root];
    let mut leaves: Vec<u32> = Vec::new();

    loop {
        let mut next_level: Vec<(u32, f32)> = Vec::new();
        let mut next_centroids: Vec<f32> = Vec::new();
        let collect_centroids = diversity.enabled();

        for &node_id in &current_level {
            let meta = {
                let ctbl = txn.open_table(clusters_def).map_err(te)?;
                match ctbl.get(node_id)? {
                    Some(g) => ClusterMeta::from_bytes(g.value()),
                    None => continue,
                }
            };

            if meta.is_leaf() {
                if min_hlc > 0 && meta.newest_hlc() > 0 && meta.newest_hlc() < min_hlc {
                    continue;
                }
                leaves.push(node_id);
                continue;
            }

            let htbl = txn.open_table(hier_def).map_err(te)?;
            let ctbl = txn.open_table(centroids_def).map_err(te)?;
            let cltbl = txn.open_table(clusters_def).map_err(te)?;

            let range = htbl.range(
                HierarchyKey::children_start(node_id)..=HierarchyKey::children_end(node_id),
            )?;

            for entry in range {
                let (key, _) = entry?;
                let child_id = key.value().child_id;

                if min_hlc > 0
                    && let Some(cg) = cltbl.get(child_id)?
                {
                    let child_meta = ClusterMeta::from_bytes(cg.value());
                    if child_meta.newest_hlc() > 0 && child_meta.newest_hlc() < min_hlc {
                        continue;
                    }
                }

                if let Some(cg) = ctbl.get(child_id)? {
                    let bytes = cg.value();
                    if bytes.len() < dim * 4 {
                        continue;
                    }
                    let centroid: Vec<f32> = (0..dim)
                        .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                        .collect();
                    let dist = config.metric.compute(query, &centroid);
                    next_level.push((child_id, dist));
                    if collect_centroids {
                        next_centroids.extend_from_slice(&centroid);
                    }
                }
            }
        }

        if next_level.is_empty() {
            break;
        }

        let nprobe_usize = nprobe as usize;

        if collect_centroids && next_level.len() > nprobe_usize {
            let mut indexed: Vec<(usize, u32, f32)> = next_level
                .iter()
                .enumerate()
                .map(|(i, &(id, d))| (i, id, d))
                .collect();
            indexed.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(CmpOrdering::Equal));

            let sorted_candidates: Vec<(u32, f32)> =
                indexed.iter().map(|&(_, id, d)| (id, d)).collect();
            let mut sorted_centroids: Vec<f32> = Vec::with_capacity(indexed.len() * dim);
            for &(orig_idx, _, _) in &indexed {
                sorted_centroids
                    .extend_from_slice(&next_centroids[orig_idx * dim..(orig_idx + 1) * dim]);
            }

            let selected = crate::probe_select::select_diverse_probes(
                &sorted_candidates,
                &sorted_centroids,
                dim,
                nprobe_usize,
                diversity,
                config.metric,
            );
            current_level = selected.iter().map(|(id, _)| *id).collect();
        } else {
            next_level.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(CmpOrdering::Equal));
            if next_level.len() > nprobe_usize {
                next_level.truncate(nprobe_usize);
            }
            current_level = next_level.iter().map(|(id, _)| *id).collect();
        }
    }

    Ok(leaves)
}
