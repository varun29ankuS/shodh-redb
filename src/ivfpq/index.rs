use alloc::collections::BinaryHeap;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering as CmpOrdering;

use crate::TableDefinition;
use crate::error::StorageError;
use crate::storage_traits::{ReadTable, StorageRead, StorageWrite, WriteTable};
use crate::vector_ops::{DistanceMetric, Neighbor, l2_normalize};

use super::adc::IntAdcTable;
use super::cluster_blob::{ClusterBlobRef, merge_into_blob, remove_from_blob};
use super::config::{
    FORMAT_V0_LEGACY, IndexConfig, IvfPqIndexDefinition, STATE_TRAINED, SearchParams,
};
use super::kmeans;
use super::metadata::{MetadataMap, passes_filter};

use super::pq::{self, Codebooks};
use super::types::{decode_index_config, encode_index_config};

/// Owned entry for cluster blob operations: `(vector_id, pq_codes, optional_raw_bytes)`.
type OwnedBlobEntry = (u64, Vec<u8>, Option<Vec<u8>>);

// ---------------------------------------------------------------------------
// Table name helpers
// ---------------------------------------------------------------------------

fn meta_name(name: &str) -> String {
    alloc::format!("__ivfpq:{name}:meta")
}
fn centroids_name(name: &str) -> String {
    alloc::format!("__ivfpq:{name}:centroids")
}
fn codebooks_name(name: &str) -> String {
    alloc::format!("__ivfpq:{name}:codebooks")
}
fn clusters_name(name: &str) -> String {
    alloc::format!("__ivfpq:{name}:clusters")
}
fn vectors_name(name: &str) -> String {
    alloc::format!("__ivfpq:{name}:vectors")
}
fn assignments_name(name: &str) -> String {
    alloc::format!("__ivfpq:{name}:assignments")
}
fn vector_meta_name(name: &str) -> String {
    alloc::format!("__ivfpq:{name}:vector_meta")
}

/// Validate that an index configuration is internally consistent.
fn validate_config(config: &IndexConfig) -> crate::Result<()> {
    if config.num_subvectors == 0 {
        return Err(StorageError::Corrupted(
            "IVF-PQ: num_subvectors must be > 0".to_string(),
        ));
    }
    if config.dim == 0 {
        return Err(StorageError::Corrupted(
            "IVF-PQ: dim must be > 0".to_string(),
        ));
    }
    if config.dim as usize % config.num_subvectors as usize != 0 {
        return Err(StorageError::Corrupted(alloc::format!(
            "IVF-PQ: dim ({}) must be divisible by num_subvectors ({})",
            config.dim,
            config.num_subvectors,
        )));
    }
    if config.num_clusters == 0 {
        return Err(StorageError::Corrupted(
            "IVF-PQ: num_clusters must be > 0".to_string(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// IvfPqIndex -- writable index handle
// ---------------------------------------------------------------------------

/// A writable IVF-PQ index bound to a storage write transaction.
///
/// Obtained via [`crate::WriteTransaction::open_ivfpq_index`].
///
/// The index configuration (including `num_vectors` count) is persisted
/// automatically when the index handle is dropped. You can also call
/// [`flush`](Self::flush) to persist explicitly at any point.
pub struct IvfPqIndex<'txn, T: StorageWrite> {
    txn: &'txn T,
    pub(crate) config: IndexConfig,
    name: String,
    /// The cluster count from the original definition, before any clamping.
    /// Used by re-training to restore the user's requested cluster count.
    requested_num_clusters: u32,
    centroids: Option<Vec<f32>>,
    codebooks: Option<Codebooks>,
    /// Tracks whether config has been modified since last persist.
    config_dirty: bool,
}

impl<'txn, T: StorageWrite> IvfPqIndex<'txn, T> {
    /// Open or create. Called by `WriteTransaction::open_ivfpq_index`.
    pub(crate) fn open(txn: &'txn T, definition: &IvfPqIndexDefinition) -> crate::Result<Self> {
        let name = String::from(definition.name());

        let mn = meta_name(&name);
        let meta_def = TableDefinition::<&str, &[u8]>::new(&mn);
        let mut meta_table = txn.open_storage_table(meta_def)?;

        // Check if config exists; if not, persist the initial config.
        let existing = meta_table.st_get(&"config")?;
        let config = if let Some(guard) = existing {
            decode_index_config(guard.value())
        } else {
            let config = definition.to_config();
            // Validate before persisting a new config.
            validate_config(&config)?;
            let bytes = encode_index_config(&config);
            drop(existing); // release binding
            meta_table.st_insert(&"config", &bytes.as_slice())?;
            config
        };

        // Reject legacy format -- re-training is required.
        if config.format_version == FORMAT_V0_LEGACY && config.state != 0 {
            return Err(StorageError::Corrupted(alloc::format!(
                "IVF-PQ '{name}': legacy format v0 -- re-train required for blob format v1",
            )));
        }

        // Eagerly create the other tables.
        {
            let cn = centroids_name(&name);
            let _ = txn.open_storage_table(TableDefinition::<u32, &[u8]>::new(&cn))?;
            let cb = codebooks_name(&name);
            let _ = txn.open_storage_table(TableDefinition::<u32, &[u8]>::new(&cb))?;
            let cl = clusters_name(&name);
            let _ = txn.open_storage_table(TableDefinition::<u32, &[u8]>::new(&cl))?;
            let vn = vectors_name(&name);
            let _ = txn.open_storage_table(TableDefinition::<u64, &[u8]>::new(&vn))?;
            let an = assignments_name(&name);
            let _ = txn.open_storage_table(TableDefinition::<u64, u32>::new(&an))?;
        }

        let requested_num_clusters = definition.num_clusters();

        Ok(Self {
            txn,
            config,
            name,
            requested_num_clusters,
            centroids: None,
            codebooks: None,
            config_dirty: false,
        })
    }

    /// Returns the current index configuration.
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// Persist any pending configuration changes to the meta table.
    ///
    /// This is called automatically on drop, but you can call it explicitly
    /// if you need to guarantee the config is written at a specific point.
    pub fn flush(&mut self) -> crate::Result<()> {
        if self.config_dirty {
            self.persist_config_inner()?;
            self.config_dirty = false;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train the IVF-PQ index from training vectors.
    ///
    /// `training_vectors` is an iterator of `(vector_id, vector)` pairs.
    /// For large datasets, pass a representative sample (10x-50x `num_clusters`).
    ///
    /// Re-training is supported: calling `train()` again replaces the centroids
    /// and codebooks, and **clears all existing postings, assignments, and raw
    /// vectors**. You must re-insert all vectors after re-training.
    pub fn train<I>(&mut self, training_vectors: I, max_iter: usize) -> crate::Result<()>
    where
        I: Iterator<Item = (u64, Vec<f32>)>,
    {
        validate_config(&self.config)?;
        let dim = self.config.dim as usize;
        // Always use the definition's requested cluster count so re-training
        // doesn't get stuck at a previously clamped value.
        let num_clusters = self.requested_num_clusters as usize;
        let num_subvectors = self.config.num_subvectors as usize;

        let mut flat: Vec<f32> = Vec::new();
        for (_id, mut vec) in training_vectors {
            if vec.len() != dim {
                return Err(StorageError::Corrupted(alloc::format!(
                    "IVF-PQ '{}': training vector dim {} != {}",
                    self.name,
                    vec.len(),
                    dim,
                )));
            }
            if self.config.metric == DistanceMetric::Cosine {
                l2_normalize(&mut vec);
            }
            flat.extend_from_slice(&vec);
        }

        let n = flat.len() / dim;
        if n == 0 {
            return Err(StorageError::Corrupted(alloc::format!(
                "IVF-PQ '{}': no training vectors provided",
                self.name,
            )));
        }

        // 1. Train IVF centroids.
        let centroid_data = kmeans::kmeans(&flat, dim, num_clusters, max_iter, self.config.metric);

        // kmeans clamps k to min(requested, n). Update config to reflect the
        // actual number of centroids so that subsequent opens don't try to
        // read centroids that were never persisted.
        let actual_k = centroid_data.len() / dim;
        let old_k = self.config.num_clusters as usize;
        #[allow(clippy::cast_possible_truncation)]
        {
            self.config.num_clusters = actual_k as u32;
        }

        // 2. Compute residuals (vector - assigned centroid) for PQ training.
        //    Residual encoding is the standard Faiss IVFADC approach: PQ
        //    codebooks learn to quantize the *offset from the centroid* rather
        //    than the raw vector. This concentrates PQ precision on the
        //    intra-cluster structure, dramatically improving recall.
        let mut residuals = Vec::with_capacity(flat.len());
        for i in 0..n {
            let vec_slice = &flat[i * dim..(i + 1) * dim];
            let (cid, _) = kmeans::assign_nearest(
                vec_slice,
                &centroid_data,
                dim,
                actual_k,
                self.config.metric,
            );
            let c_offset = cid as usize * dim;
            for d in 0..dim {
                residuals.push(vec_slice[d] - centroid_data[c_offset + d]);
            }
        }

        // 3. Train PQ codebooks on residuals.
        let codebooks_trained = pq::train_codebooks(
            &residuals,
            dim,
            num_subvectors,
            max_iter,
            self.config.metric,
        )?;

        // 4. Clear stale data from a previous training cycle.
        self.clear_stale_training_data(old_k, actual_k)?;

        // 5. Persist new centroids.
        {
            let tn = centroids_name(&self.name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            for c in 0..actual_k {
                let bytes = f32_slice_to_le_bytes(&centroid_data[c * dim..(c + 1) * dim]);
                #[allow(clippy::cast_possible_truncation)]
                table.st_insert(&(c as u32), &bytes.as_slice())?;
            }
        }

        // 6. Persist PQ codebooks.
        {
            let tn = codebooks_name(&self.name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            for m in 0..num_subvectors {
                let bytes = codebooks_trained.serialize_codebook(m);
                #[allow(clippy::cast_possible_truncation)]
                table.st_insert(&(m as u32), &bytes.as_slice())?;
            }
        }

        // 7. Update config -- persist immediately since training is a major event.
        self.config.state = STATE_TRAINED;
        self.config.num_vectors = 0;
        self.persist_config_inner()?;
        self.config_dirty = false;

        self.centroids = Some(centroid_data);
        self.codebooks = Some(codebooks_trained);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Insert / Remove
    // -----------------------------------------------------------------------

    /// Insert a vector into the index. Index must be trained first.
    ///
    /// If `vector_id` already exists, the old entry is replaced (upsert semantics).
    /// Returns an error if the vector contains NaN or Inf values.
    #[allow(clippy::cast_possible_truncation)]
    pub fn insert(&mut self, vector_id: u64, vector: &[f32]) -> crate::Result<()> {
        self.ensure_trained()?;
        let dim = self.config.dim as usize;
        if vector.len() != dim {
            return Err(StorageError::Corrupted(alloc::format!(
                "IVF-PQ '{}': vector dim {} != {}",
                self.name,
                vector.len(),
                dim,
            )));
        }
        Self::validate_finite(vector, &self.name)?;

        let vec_owned;
        let vec_ref = if self.config.metric == DistanceMetric::Cosine {
            vec_owned = crate::vector_ops::l2_normalized(vector);
            &vec_owned
        } else {
            vector
        };

        let centroids = self.load_centroids()?;

        let (cluster_id, _) = kmeans::assign_nearest(
            vec_ref,
            &centroids,
            dim,
            self.config.num_clusters as usize,
            self.config.metric,
        );
        let c_offset = cluster_id as usize * dim;
        let residual: Vec<f32> = vec_ref
            .iter()
            .enumerate()
            .map(|(d, &v)| v - centroids[c_offset + d])
            .collect();

        let codebooks = self.load_codebooks()?;
        let pq_codes = codebooks.encode(&residual);

        // Check if this vector_id already exists.
        let old_cluster = {
            let tn = assignments_name(&self.name);
            let def = TableDefinition::<u64, u32>::new(&tn);
            let table = self.txn.open_storage_table(def)?;
            table.st_get(&vector_id)?.map(|g| g.value())
        };

        let pq_len = self.config.num_subvectors as u16;

        // Remove from old cluster blob if moving clusters.
        if let Some(old_cid) = old_cluster
            && old_cid != cluster_id
        {
            self.remove_from_cluster_blob(old_cid, vector_id, pq_len)?;
        }

        // Merge into target cluster blob (PQ-only, no raw vectors).
        {
            let tn = clusters_name(&self.name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;

            let existing_blob = table.st_get(&cluster_id)?;
            let existing_ref = match existing_blob {
                Some(ref guard) => Some(ClusterBlobRef::new(guard.value(), pq_len, dim)?),
                None => None,
            };

            let mut new_entries: Vec<OwnedBlobEntry> = vec![(vector_id, pq_codes, None)];
            let merged = merge_into_blob(existing_ref.as_ref(), &mut new_entries, pq_len);
            drop(existing_blob);
            table.st_insert(&cluster_id, &merged.as_slice())?;
        }

        // Store raw vector separately for reranking.
        if self.config.store_raw_vectors {
            let raw_bytes = f32_slice_to_le_bytes(vec_ref);
            let vn = vectors_name(&self.name);
            let vdef = TableDefinition::<u64, &[u8]>::new(&vn);
            let mut vt = self.txn.open_storage_table(vdef)?;
            vt.st_insert(&vector_id, &raw_bytes.as_slice())?;
        }

        // Update assignment.
        {
            let tn = assignments_name(&self.name);
            let def = TableDefinition::<u64, u32>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            table.st_insert(&vector_id, &cluster_id)?;
        }

        if old_cluster.is_none() {
            self.config.num_vectors = self.config.num_vectors.saturating_add(1);
            self.config_dirty = true;
        }
        Ok(())
    }

    /// Bulk insert vectors.
    ///
    /// Groups vectors by cluster and performs one blob read-modify-write per
    /// cluster touched, regardless of how many vectors go into it.
    ///
    /// If a `vector_id` already exists, the old entry is replaced (upsert semantics).
    #[allow(clippy::cast_possible_truncation)]
    pub fn insert_batch<I>(&mut self, vectors: I) -> crate::Result<u64>
    where
        I: Iterator<Item = (u64, Vec<f32>)>,
    {
        self.ensure_trained()?;
        let dim = self.config.dim as usize;
        let centroids = self.load_centroids()?;
        let num_clusters = self.config.num_clusters as usize;
        let metric = self.config.metric;
        let store_raw = self.config.store_raw_vectors;
        let pq_len = self.config.num_subvectors as u16;

        let codebooks = self.load_codebooks()?;

        // Phase 1: Compute cluster assignments and PQ codes for all vectors.
        // grouped[cluster_id] = Vec<(vector_id, pq_codes, None)>  -- PQ only
        let mut grouped: Vec<Vec<OwnedBlobEntry>> = Vec::new();
        grouped.resize_with(num_clusters, Vec::new);

        // Raw vectors stored separately (vector_id, raw_bytes).
        let mut raw_vectors: Vec<(u64, Vec<u8>)> = Vec::new();

        let an = assignments_name(&self.name);
        let ad = TableDefinition::<u64, u32>::new(&an);
        let mut at = self.txn.open_storage_table(ad)?;

        // Track old cluster assignments for upsert cleanup.
        let mut old_assignments: Vec<(u64, u32)> = Vec::new();
        let mut new_count = 0u64;

        for (vector_id, mut vec) in vectors {
            if vec.len() != dim {
                return Err(StorageError::Corrupted(alloc::format!(
                    "IVF-PQ '{}': vector dim {} != {}",
                    self.name,
                    vec.len(),
                    dim,
                )));
            }
            Self::validate_finite(&vec, &self.name)?;
            if metric == DistanceMetric::Cosine {
                l2_normalize(&mut vec);
            }

            let (cluster_id, _) =
                kmeans::assign_nearest(&vec, &centroids, dim, num_clusters, metric);

            let c_offset = cluster_id as usize * dim;
            let residual: Vec<f32> = vec
                .iter()
                .enumerate()
                .map(|(d, &v)| v - centroids[c_offset + d])
                .collect();

            let pq_codes = codebooks.encode(&residual);
            if store_raw {
                raw_vectors.push((
                    vector_id,
                    f32_slice_to_le_bytes(&vec),
                ));
            }

            // Check for existing assignment.
            let old_cluster = at.st_get(&vector_id)?.map(|g| g.value());
            if let Some(old_cid) = old_cluster {
                if old_cid != cluster_id {
                    old_assignments.push((vector_id, old_cid));
                }
            } else {
                new_count += 1;
            }

            at.st_insert(&vector_id, &cluster_id)?;
            grouped[cluster_id as usize].push((vector_id, pq_codes, None));
        }
        drop(at);

        // Phase 2: Remove vectors that moved to different clusters.
        if !old_assignments.is_empty() {
            for &(vid, old_cid) in &old_assignments {
                self.remove_from_cluster_blob(old_cid, vid, pq_len)?;
            }
        }

        // Phase 3: For each cluster touched, read-modify-write the PQ blob.
        {
            let tn = clusters_name(&self.name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;

            for (cid, mut entries) in grouped.into_iter().enumerate() {
                if entries.is_empty() {
                    continue;
                }
                let cid_u32 = cid as u32;

                let existing_blob = table.st_get(&cid_u32)?;
                let existing_ref = match existing_blob {
                    Some(ref guard) => Some(ClusterBlobRef::new(guard.value(), pq_len, dim)?),
                    None => None,
                };

                let merged = merge_into_blob(existing_ref.as_ref(), &mut entries, pq_len);
                drop(existing_blob);
                table.st_insert(&cid_u32, &merged.as_slice())?;
            }
        }

        // Phase 4: Write raw vectors to separate table.
        if !raw_vectors.is_empty() {
            let vn = vectors_name(&self.name);
            let vdef = TableDefinition::<u64, &[u8]>::new(&vn);
            let mut vt = self.txn.open_storage_table(vdef)?;
            for (vid, raw) in &raw_vectors {
                vt.st_insert(vid, &raw.as_slice())?;
            }
        }

        if new_count > 0 {
            self.config.num_vectors = self.config.num_vectors.saturating_add(new_count);
            self.config_dirty = true;
        }
        Ok(new_count)
    }

    /// Remove a vector from the index. Returns `true` if found and removed.
    #[allow(clippy::cast_possible_truncation)]
    pub fn remove(&mut self, vector_id: u64) -> crate::Result<bool> {
        let cluster_id = {
            let tn = assignments_name(&self.name);
            let def = TableDefinition::<u64, u32>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            match table.st_remove(&vector_id)? {
                Some(guard) => guard.value(),
                None => return Ok(false),
            }
        };

        let pq_len = self.config.num_subvectors as u16;
        self.remove_from_cluster_blob(cluster_id, vector_id, pq_len)?;

        // Remove raw vector if stored.
        if self.config.store_raw_vectors {
            let vn = vectors_name(&self.name);
            let vdef = TableDefinition::<u64, &[u8]>::new(&vn);
            let mut vt = self.txn.open_storage_table(vdef)?;
            vt.st_remove(&vector_id)?;
        }

        self.config.num_vectors = self.config.num_vectors.saturating_sub(1);
        self.config_dirty = true;

        // Also remove associated metadata if present.
        {
            let mn = vector_meta_name(&self.name);
            let mdef = TableDefinition::<u64, &[u8]>::new(&mn);
            let mut mt = self.txn.open_storage_table(mdef)?;
            mt.st_remove(&vector_id)?;
        }

        Ok(true)
    }

    /// Insert or replace metadata for a vector.
    ///
    /// The metadata is stored in a separate B-tree table keyed by `vector_id`.
    /// This must be called after inserting the vector itself.
    pub fn insert_metadata(&mut self, vector_id: u64, metadata: &MetadataMap) -> crate::Result<()> {
        let encoded = metadata.encode();
        let mn = vector_meta_name(&self.name);
        let mdef = TableDefinition::<u64, &[u8]>::new(&mn);
        let mut mt = self.txn.open_storage_table(mdef)?;
        mt.st_insert(&vector_id, &encoded.as_slice())?;
        Ok(())
    }

    /// Remove metadata for a vector.
    pub fn remove_metadata(&mut self, vector_id: u64) -> crate::Result<()> {
        let mn = vector_meta_name(&self.name);
        let mdef = TableDefinition::<u64, &[u8]>::new(&mn);
        let mut mt = self.txn.open_storage_table(mdef)?;
        mt.st_remove(&vector_id)?;
        Ok(())
    }

    /// Search within a write transaction.
    #[allow(clippy::cast_possible_truncation)]
    pub fn search(
        &mut self,
        query: &[f32],
        params: &SearchParams,
    ) -> crate::Result<Vec<Neighbor<u64>>> {
        self.ensure_trained()?;
        self.flush()?;
        let dim = self.config.dim as usize;
        if query.len() != dim {
            return Err(StorageError::Corrupted(alloc::format!(
                "IVF-PQ '{}': query dim {} != {}",
                self.name,
                query.len(),
                dim,
            )));
        }

        let centroids = self.load_centroids()?;
        let codebooks = self.load_codebooks()?;

        let query_owned;
        let q = if self.config.metric == DistanceMetric::Cosine {
            if crate::vector_ops::l2_norm(query) == 0.0 {
                return Ok(Vec::new());
            }
            query_owned = crate::vector_ops::l2_normalized(query);
            &query_owned
        } else {
            query
        };

        let nprobe = (params.nprobe).max(1).min(self.config.num_clusters) as usize;
        let probes = kmeans::nearest_clusters(
            q,
            &centroids,
            dim,
            self.config.num_clusters as usize,
            nprobe,
            self.config.metric,
            params.diversity,
        );

        let cap = if params.rerank && self.config.store_raw_vectors {
            params.candidates.max(params.k)
        } else {
            params.k
        };
        let mut heap = CandidateHeap::new(cap);

        let pq_len = self.config.num_subvectors as u16;
        let metric = self.config.metric;
        let want_rerank = params.rerank && self.config.store_raw_vectors;

        {
            let tn = clusters_name(&self.name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let table = self.txn.open_storage_table(def)?;

            let meta_table = if params.filter.is_some() {
                let mn = vector_meta_name(&self.name);
                let mdef = TableDefinition::<u64, &[u8]>::new(&mn);
                Some(self.txn.open_storage_table(mdef)?)
            } else {
                None
            };

            let mut query_residual = vec![0.0f32; dim];
            for &(cid, _) in &probes {
                let c_offset = cid as usize * dim;
                for d in 0..dim {
                    query_residual[d] = q[d] - centroids[c_offset + d];
                }

                let Some(blob_data) = table.st_get(&cid)? else {
                    continue;
                };
                let blob = ClusterBlobRef::new(blob_data.value(), pq_len, dim)?;
                let adc = IntAdcTable::build(&query_residual, &codebooks, metric);

                let pq_block = blob.pq_codes_block();
                let m = pq_len as usize;
                for i in 0..blob.count() {
                    let codes = &pq_block[i as usize * m..(i as usize + 1) * m];
                    let dist = adc.to_f32(adc.approximate_distance(codes));
                    let vid = blob.vector_id(i);

                    if let Some(ref filter) = params.filter
                        && let Some(ref mt) = meta_table
                    {
                        match mt.st_get(&vid)? {
                            Some(guard) => {
                                if !passes_filter(guard.value(), filter) {
                                    continue;
                                }
                            }
                            None => continue,
                        }
                    }
                    heap.push(vid, dist);
                }
            }
        }

        if want_rerank {
            let sorted = heap.into_sorted();
            rerank_from_vectors_table_write(self.txn, q, &sorted, &self.name, dim, metric, params.k)
        } else {
            Ok(heap.into_sorted().into_iter().take(params.k).collect())
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Validate that a vector contains only finite values.
    fn validate_finite(vector: &[f32], name: &str) -> crate::Result<()> {
        for (i, &v) in vector.iter().enumerate() {
            if !v.is_finite() {
                return Err(StorageError::Corrupted(alloc::format!(
                    "IVF-PQ '{name}': vector contains non-finite value ({v}) at index {i}",
                )));
            }
        }
        Ok(())
    }

    /// Remove stale data from a previous training cycle.
    ///
    /// Deletes orphaned centroid rows (indices `new_k..old_k`) and clears all
    /// cluster blobs and assignments -- they reference cluster IDs from
    /// the previous centroid set and are invalid after re-training.
    fn clear_stale_training_data(&self, old_k: usize, new_k: usize) -> crate::Result<()> {
        if old_k > new_k {
            let tn = centroids_name(&self.name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            for c in new_k..old_k {
                #[allow(clippy::cast_possible_truncation)]
                table.st_remove(&(c as u32))?;
            }
        }

        // Clear all cluster blobs.
        {
            let tn = clusters_name(&self.name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            table.st_drain_all()?;
        }

        // Clear all raw vectors.
        {
            let vn = vectors_name(&self.name);
            let vdef = TableDefinition::<u64, &[u8]>::new(&vn);
            let mut vt = self.txn.open_storage_table(vdef)?;
            vt.st_drain_all()?;
        }

        // Clear all assignments.
        {
            let tn = assignments_name(&self.name);
            let def = TableDefinition::<u64, u32>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            table.st_drain_all()?;
        }

        Ok(())
    }

    fn ensure_trained(&self) -> crate::Result<()> {
        if self.config.state != STATE_TRAINED {
            return Err(StorageError::Corrupted(alloc::format!(
                "IVF-PQ '{}' not trained -- call train() first",
                self.name,
            )));
        }
        Ok(())
    }

    fn persist_config_inner(&self) -> crate::Result<()> {
        let tn = meta_name(&self.name);
        let def = TableDefinition::<&str, &[u8]>::new(&tn);
        let mut table = self.txn.open_storage_table(def)?;
        let bytes = encode_index_config(&self.config);
        table.st_insert(&"config", &bytes.as_slice())?;
        Ok(())
    }

    fn load_centroids(&mut self) -> crate::Result<Vec<f32>> {
        if let Some(ref c) = self.centroids {
            return Ok(c.clone());
        }
        let data = self.read_centroids()?;
        self.centroids = Some(data.clone());
        Ok(data)
    }

    fn read_centroids(&self) -> crate::Result<Vec<f32>> {
        let dim = self.config.dim as usize;
        let k = self.config.num_clusters as usize;
        let tn = centroids_name(&self.name);
        let def = TableDefinition::<u32, &[u8]>::new(&tn);
        let table = self.txn.open_storage_table(def)?;

        let mut flat = Vec::with_capacity(k * dim);
        for c in 0..k {
            #[allow(clippy::cast_possible_truncation)]
            let guard = table.st_get(&(c as u32))?.ok_or_else(|| {
                StorageError::Corrupted(alloc::format!(
                    "IVF-PQ '{}': missing centroid {c}",
                    self.name,
                ))
            })?;
            let raw = guard.value();
            if raw.len() % 4 != 0 {
                return Err(StorageError::Corrupted(alloc::format!(
                    "IVF-PQ '{}': centroid {c} has misaligned byte length {} (not a multiple of 4)",
                    self.name,
                    raw.len(),
                )));
            }
            for chunk in raw.chunks_exact(4) {
                if let Ok(bytes) = chunk.try_into() {
                    flat.push(f32::from_le_bytes(bytes));
                }
            }
        }
        Ok(flat)
    }

    fn load_codebooks(&mut self) -> crate::Result<Codebooks> {
        if let Some(ref cb) = self.codebooks {
            return Ok(cb.clone());
        }
        let cb = self.read_codebooks()?;
        self.codebooks = Some(cb.clone());
        Ok(cb)
    }

    fn read_codebooks(&self) -> crate::Result<Codebooks> {
        let m = self.config.num_subvectors as usize;
        let sd = self.config.sub_dim();
        let tn = codebooks_name(&self.name);
        let def = TableDefinition::<u32, &[u8]>::new(&tn);
        let table = self.txn.open_storage_table(def)?;

        let mut data = Vec::with_capacity(m * 256 * sd);
        for i in 0..m {
            #[allow(clippy::cast_possible_truncation)]
            let guard = table.st_get(&(i as u32))?.ok_or_else(|| {
                StorageError::Corrupted(alloc::format!(
                    "IVF-PQ '{}': missing codebook {i}",
                    self.name,
                ))
            })?;
            data.extend_from_slice(&Codebooks::deserialize_codebook(guard.value(), sd));
        }

        Ok(Codebooks {
            data,
            num_subvectors: m,
            sub_dim: sd,
        })
    }

    /// Remove a vector from a cluster blob. Helper for insert upsert + remove.
    #[allow(clippy::cast_possible_truncation)]
    fn remove_from_cluster_blob(
        &self,
        cluster_id: u32,
        vector_id: u64,
        pq_len: u16,
    ) -> crate::Result<()> {
        let dim = self.config.dim as usize;
        let tn = clusters_name(&self.name);
        let def = TableDefinition::<u32, &[u8]>::new(&tn);
        let mut table = self.txn.open_storage_table(def)?;

        let new_blob_or_empty = {
            let existing = table.st_get(&cluster_id)?;
            if let Some(guard) = existing {
                let blob = ClusterBlobRef::new(guard.value(), pq_len, dim)?;
                Some(remove_from_blob(&blob, vector_id, pq_len))
            } else {
                None
            }
        };
        if let Some(result) = new_blob_or_empty {
            match result {
                Some(new_blob) => {
                    table.st_insert(&cluster_id, &new_blob.as_slice())?;
                }
                None => {
                    table.st_remove(&cluster_id)?;
                }
            }
        }
        Ok(())
    }
}

impl<T: StorageWrite> Drop for IvfPqIndex<'_, T> {
    fn drop(&mut self) {
        if self.config_dirty {
            // Best-effort persist on drop. Errors are silently ignored since
            // the transaction will either commit (persisting everything) or
            // abort (discarding everything) regardless.
            let _ = self.persist_config_inner();
        }
    }
}

// ---------------------------------------------------------------------------
// Shared reranking from cluster blobs (on-demand reads)
// ---------------------------------------------------------------------------

/// Rerank candidates by reading raw vectors from the separate vectors table.
///
/// Optimizations over naive per-candidate lookup:
/// 1. Sort candidates by `vector_id` for B-tree page cache locality.
/// 2. Reuse a single f32 buffer via `bytes_to_f32_buf` (one allocation total).
macro_rules! impl_rerank_from_vectors {
    ($fn_name:ident, $trait_bound:path) => {
        #[allow(clippy::too_many_arguments)]
        fn $fn_name<S: $trait_bound>(
            txn: &S,
            query: &[f32],
            candidates: &[Neighbor<u64>],
            index_name: &str,
            dim: usize,
            metric: DistanceMetric,
            k: usize,
        ) -> crate::Result<Vec<Neighbor<u64>>> {
            let vn = vectors_name(index_name);
            let vdef = TableDefinition::<u64, &[u8]>::new(&vn);
            let vt = txn.open_storage_table(vdef)?;

            // Sort by vector_id for sequential B-tree access.
            let mut sorted_cands: Vec<&Neighbor<u64>> = candidates.iter().collect();
            sorted_cands.sort_unstable_by_key(|c| c.key);

            let expected_bytes = dim * 4;
            let mut raw_buf = vec![0.0f32; dim];
            let mut results: Vec<Neighbor<u64>> = Vec::with_capacity(sorted_cands.len());
            for cand in &sorted_cands {
                if let Some(guard) = vt.st_get(&cand.key)? {
                    let raw = guard.value();
                    if raw.len() == expected_bytes {
                        bytes_to_f32_buf(raw, &mut raw_buf);
                        results.push(Neighbor {
                            key: cand.key,
                            distance: metric.compute(query, &raw_buf),
                        });
                    }
                }
            }

            results.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
            results.truncate(k);
            Ok(results)
        }
    };
}

impl_rerank_from_vectors!(rerank_from_vectors_table, StorageRead);
impl_rerank_from_vectors!(rerank_from_vectors_table_write, StorageWrite);

// ---------------------------------------------------------------------------
// ReadOnlyIvfPqIndex -- read-only index handle
// ---------------------------------------------------------------------------

/// A read-only IVF-PQ index.
///
/// Centroids and codebooks are loaded into memory at open time.
pub struct ReadOnlyIvfPqIndex {
    config: IndexConfig,
    name: String,
    centroids: Vec<f32>,
    codebooks: Codebooks,
}

impl ReadOnlyIvfPqIndex {
    /// Open. Called by `ReadTransaction::open_ivfpq_index`.
    pub(crate) fn open<R: StorageRead>(
        txn: &R,
        definition: &IvfPqIndexDefinition,
    ) -> crate::Result<Self> {
        let name = String::from(definition.name());

        let mn = meta_name(&name);
        let md = TableDefinition::<&str, &[u8]>::new(&mn);
        let mt = txn.open_storage_table(md)?;

        let config = match mt.st_get(&"config")? {
            Some(guard) => decode_index_config(guard.value()),
            None => {
                return Err(StorageError::Corrupted(alloc::format!(
                    "IVF-PQ index '{mn}' not found (missing config)",
                )));
            }
        };

        let dim = config.dim as usize;
        let num_clusters = config.num_clusters as usize;
        let centroids = {
            let tn = centroids_name(&name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let table = txn.open_storage_table(def)?;
            let mut flat = Vec::with_capacity(num_clusters * dim);
            for c in 0..num_clusters {
                #[allow(clippy::cast_possible_truncation)]
                let guard = table.st_get(&(c as u32))?.ok_or_else(|| {
                    StorageError::Corrupted(
                        alloc::format!("IVF-PQ '{name}': missing centroid {c}",),
                    )
                })?;
                for chunk in guard.value().chunks_exact(4) {
                    if let Ok(bytes) = chunk.try_into() {
                        flat.push(f32::from_le_bytes(bytes));
                    }
                }
            }
            flat
        };

        // Load PQ codebooks.
        let codebooks = {
            let num_subvectors = config.num_subvectors as usize;
            let sub_dim = config.sub_dim();
            let tn = codebooks_name(&name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let table = txn.open_storage_table(def)?;
            let mut data = Vec::with_capacity(num_subvectors * 256 * sub_dim);
            for m in 0..num_subvectors {
                #[allow(clippy::cast_possible_truncation)]
                let guard = table.st_get(&(m as u32))?.ok_or_else(|| {
                    StorageError::Corrupted(
                        alloc::format!("IVF-PQ '{name}': missing codebook {m}",),
                    )
                })?;
                data.extend_from_slice(&Codebooks::deserialize_codebook(guard.value(), sub_dim));
            }
            Codebooks {
                data,
                num_subvectors,
                sub_dim,
            }
        };

        Ok(Self {
            config,
            name,
            centroids,
            codebooks,
        })
    }

    /// Returns the index configuration.
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// Search for approximate nearest neighbors.
    #[allow(clippy::cast_possible_truncation)]
    pub fn search<R: StorageRead>(
        &self,
        txn: &R,
        query: &[f32],
        params: &SearchParams,
    ) -> crate::Result<Vec<Neighbor<u64>>> {
        if self.config.state != STATE_TRAINED {
            return Err(StorageError::Corrupted(alloc::format!(
                "IVF-PQ '{}' not trained",
                self.name,
            )));
        }

        let dim = self.config.dim as usize;
        if query.len() != dim {
            return Err(StorageError::Corrupted(alloc::format!(
                "IVF-PQ '{}': query dim {} != {}",
                self.name,
                query.len(),
                dim,
            )));
        }

        let query_owned;
        let q = if self.config.metric == DistanceMetric::Cosine {
            if crate::vector_ops::l2_norm(query) == 0.0 {
                return Ok(Vec::new());
            }
            query_owned = crate::vector_ops::l2_normalized(query);
            &query_owned
        } else {
            query
        };

        let nprobe = (params.nprobe).max(1).min(self.config.num_clusters) as usize;
        let probes = kmeans::nearest_clusters(
            q,
            &self.centroids,
            dim,
            self.config.num_clusters as usize,
            nprobe,
            self.config.metric,
            params.diversity,
        );

        let cap = if params.rerank && self.config.store_raw_vectors {
            params.candidates.max(params.k)
        } else {
            params.k
        };
        let mut heap = CandidateHeap::new(cap);

        let pq_len = self.config.num_subvectors as u16;
        let metric = self.config.metric;
        let want_rerank = params.rerank && self.config.store_raw_vectors;

        {
            let tn = clusters_name(&self.name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let table = txn.open_storage_table(def)?;

            let meta_table = if params.filter.is_some() {
                let mn = vector_meta_name(&self.name);
                let mdef = TableDefinition::<u64, &[u8]>::new(&mn);
                Some(txn.open_storage_table(mdef)?)
            } else {
                None
            };

            let mut query_residual = vec![0.0f32; dim];
            for &(cid, _) in &probes {
                let c_offset = cid as usize * dim;
                for d in 0..dim {
                    query_residual[d] = q[d] - self.centroids[c_offset + d];
                }

                let Some(blob_data) = table.st_get(&cid)? else {
                    continue;
                };
                let blob = ClusterBlobRef::new(blob_data.value(), pq_len, dim)?;
                let adc = IntAdcTable::build(&query_residual, &self.codebooks, metric);

                let pq_block = blob.pq_codes_block();
                let m = pq_len as usize;
                for i in 0..blob.count() {
                    let codes = &pq_block[i as usize * m..(i as usize + 1) * m];
                    let dist = adc.to_f32(adc.approximate_distance(codes));
                    let vid = blob.vector_id(i);

                    if let Some(ref filter) = params.filter
                        && let Some(ref mt) = meta_table
                    {
                        match mt.st_get(&vid)? {
                            Some(guard) => {
                                if !passes_filter(guard.value(), filter) {
                                    continue;
                                }
                            }
                            None => continue,
                        }
                    }
                    heap.push(vid, dist);
                }
            }
        }

        if want_rerank {
            let sorted = heap.into_sorted();
            rerank_from_vectors_table(txn, q, &sorted, &self.name, dim, metric, params.k)
        } else {
            Ok(heap.into_sorted().into_iter().take(params.k).collect())
        }
    }
}

// ---------------------------------------------------------------------------
// CandidateHeap -- fixed-size max-heap for top-k tracking
// ---------------------------------------------------------------------------

struct CandidateHeap {
    capacity: usize,
    heap: BinaryHeap<CandidateEntry>,
}

#[derive(PartialEq)]
struct CandidateEntry {
    vector_id: u64,
    distance: f32,
}

impl Eq for CandidateEntry {}

impl PartialOrd for CandidateEntry {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for CandidateEntry {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.distance.total_cmp(&other.distance)
    }
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

    fn into_sorted(self) -> Vec<Neighbor<u64>> {
        let mut items: Vec<Neighbor<u64>> = self
            .heap
            .into_iter()
            .map(|e| Neighbor {
                key: e.vector_id,
                distance: e.distance,
            })
            .collect();
        items.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        items
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Encode a slice of f32s into LE bytes.
///
/// On little-endian targets this is a direct memcpy; on big-endian it
/// converts each float individually.
#[inline]
fn f32_slice_to_le_bytes(floats: &[f32]) -> Vec<u8> {
    let byte_len = floats.len() * 4;
    let mut out = vec![0u8; byte_len];
    #[cfg(target_endian = "little")]
    {
        // SAFETY: On LE targets f32 memory layout matches LE byte order.
        // `floats` has `floats.len() * 4` bytes, `out` has the same size.
        // Both pointers are valid and non-overlapping (Vec owns its buffer).
        unsafe {
            core::ptr::copy_nonoverlapping(
                floats.as_ptr().cast::<u8>(),
                out.as_mut_ptr(),
                byte_len,
            );
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        for (i, &f) in floats.iter().enumerate() {
            let b = f.to_le_bytes();
            out[i * 4..i * 4 + 4].copy_from_slice(&b);
        }
    }
    out
}

/// Decode LE f32s from `bytes` into the pre-allocated `buf`.
/// Caller must ensure `bytes.len() == buf.len() * 4`.
#[inline]
fn bytes_to_f32_buf(bytes: &[u8], buf: &mut [f32]) {
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        // chunks_exact guarantees exactly 4 bytes per chunk.
        buf[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
}
