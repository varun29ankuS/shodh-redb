use alloc::string::ToString;
use alloc::vec::Vec;

use crate::TableDefinition;
use crate::error::{StorageError, TableError};
use crate::table::ReadableTable;
use crate::transactions::{ReadTransaction, WriteTransaction};
use crate::vector_ops::{DistanceMetric, Neighbor, l2_normalize};

use crate::ivfpq::pq::{self, Codebooks};
use crate::ivfpq::types::PostingKey;

use super::cluster::{
    TableNames, cascade_leaf_buffer, centroid_add, centroid_remove, merge_cluster, split_cluster,
};
use super::config::{
    FractalIndexConfig, FractalIndexDefinition, FractalSearchParams, NO_PARENT,
    STATE_CODEBOOKS_TRAINED, STATE_NEW, STATE_OPERATIONAL, decode_fractal_config,
    encode_fractal_config,
};
use super::search;
use super::types::{ClusterMeta, HierarchyKey};

/// Convert a `TableError` to `StorageError`.
fn te(e: TableError) -> StorageError {
    e.into_storage_error_or_corrupted("fractal index internal table error")
}

/// Validate that a fractal index configuration is internally consistent.
fn validate_config(config: &FractalIndexConfig) -> crate::Result<()> {
    if config.num_subvectors == 0 {
        return Err(StorageError::Corrupted(
            "fractal: num_subvectors must be > 0".to_string(),
        ));
    }
    if config.dim == 0 {
        return Err(StorageError::Corrupted(
            "fractal: dim must be > 0".to_string(),
        ));
    }
    if config.dim as usize % config.num_subvectors as usize != 0 {
        return Err(StorageError::Corrupted(alloc::format!(
            "fractal: dim ({}) must be divisible by num_subvectors ({})",
            config.dim,
            config.num_subvectors,
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FractalIndex -- writable index handle
// ---------------------------------------------------------------------------

/// A writable fractal vector index bound to a [`WriteTransaction`].
///
/// Obtained via [`WriteTransaction::open_fractal_index`].
pub struct FractalIndex<'txn> {
    pub(crate) txn: &'txn WriteTransaction,
    pub(crate) config: FractalIndexConfig,
    pub(crate) names: TableNames,
    pub(crate) codebooks: Option<Codebooks>,
    config_dirty: bool,
}

impl<'txn> FractalIndex<'txn> {
    /// Open or create. Called by `WriteTransaction::open_fractal_index`.
    pub(crate) fn open(
        txn: &'txn WriteTransaction,
        def: &FractalIndexDefinition,
    ) -> Result<Self, TableError> {
        let names = TableNames::new(def.name());

        // Open or create meta table
        let meta_def = TableDefinition::<&str, &[u8]>::new(&names.meta);
        let mut meta_tbl = txn.open_table(meta_def)?;

        let config = if let Some(guard) = meta_tbl.get("config")? {
            let data = guard.value();
            decode_fractal_config(data)
        } else {
            let cfg = def.to_config();
            let encoded = encode_fractal_config(&cfg);
            meta_tbl.insert("config", encoded.as_slice())?;
            cfg
        };
        drop(meta_tbl);

        validate_config(&config).map_err(TableError::Storage)?;

        // Ensure all tables exist by opening them
        let _ = txn.open_table(TableDefinition::<u32, &[u8]>::new(&names.clusters))?;
        let _ = txn.open_table(TableDefinition::<u32, &[u8]>::new(&names.centroids))?;
        let _ = txn.open_table(TableDefinition::<u32, &[u8]>::new(&names.centroid_sums))?;
        let _ = txn.open_table(TableDefinition::<HierarchyKey, ()>::new(&names.hierarchy))?;
        let _ = txn.open_table(TableDefinition::<PostingKey, &[u8]>::new(&names.buffer))?;
        let _ = txn.open_table(TableDefinition::<PostingKey, &[u8]>::new(&names.postings))?;
        let _ = txn.open_table(TableDefinition::<u64, u32>::new(&names.assignments))?;
        if config.store_raw_vectors {
            let _ = txn.open_table(TableDefinition::<u64, &[u8]>::new(&names.vectors))?;
        }
        let _ = txn.open_table(TableDefinition::<u32, &[u8]>::new(&names.codebooks))?;

        Ok(Self {
            txn,
            config,
            names,
            codebooks: None,
            config_dirty: false,
        })
    }

    /// Persist the config to the meta table.
    pub fn flush(&mut self) -> crate::Result<()> {
        if self.config_dirty {
            let meta_def = TableDefinition::<&str, &[u8]>::new(&self.names.meta);
            let mut meta_tbl = self.txn.open_table(meta_def).map_err(te)?;
            let encoded = encode_fractal_config(&self.config);
            meta_tbl.insert("config", encoded.as_slice())?;
            self.config_dirty = false;
        }
        Ok(())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn ensure_codebooks(&mut self) -> crate::Result<()> {
        if self.codebooks.is_some() {
            return Ok(());
        }
        if self.config.state == STATE_NEW {
            return Err(StorageError::Corrupted(
                "fractal: index has not been trained yet".to_string(),
            ));
        }
        let cb_def = TableDefinition::<u32, &[u8]>::new(&self.names.codebooks);
        let tbl = self.txn.open_table(cb_def).map_err(te)?;
        let num_sub = self.config.num_subvectors as usize;
        let sub_dim = self.config.sub_dim();
        let mut data = alloc::vec![0.0f32; num_sub * 256 * sub_dim];
        for m in 0..num_sub {
            if let Some(guard) = tbl.get(m as u32)? {
                let bytes = guard.value();
                let expected_len = 256 * sub_dim * 4;
                if bytes.len() < expected_len {
                    return Err(StorageError::Corrupted(alloc::format!(
                        "fractal: codebook {} has insufficient bytes: expected {}, got {}",
                        m,
                        expected_len,
                        bytes.len(),
                    )));
                }
                for k in 0..256 {
                    for d in 0..sub_dim {
                        let offset = (k * sub_dim + d) * 4;
                        data[m * 256 * sub_dim + k * sub_dim + d] =
                            f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                    }
                }
            }
        }
        self.codebooks = Some(Codebooks {
            data,
            num_subvectors: num_sub,
            sub_dim,
        });
        Ok(())
    }

    /// Train PQ codebooks from training data.
    ///
    /// This initializes the index: trains codebooks, creates the root cluster,
    /// bulk-inserts training vectors, and recursively splits until all leaf
    /// clusters are below `max_leaf_population`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn train_codebooks<I>(&mut self, vectors: I, max_iter: usize) -> crate::Result<()>
    where
        I: Iterator<Item = (u64, Vec<f32>)>,
    {
        let dim = self.config.dim as usize;
        let num_sub = self.config.num_subvectors as usize;

        // Collect training data
        let mut ids: Vec<u64> = Vec::new();
        let mut flat: Vec<f32> = Vec::new();
        for (id, vec) in vectors {
            if vec.len() != dim {
                return Err(StorageError::Corrupted(alloc::format!(
                    "fractal: training vector dim mismatch: expected {dim}, got {}",
                    vec.len(),
                )));
            }
            ids.push(id);
            if self.config.metric == DistanceMetric::Cosine {
                let mut normalized = vec;
                l2_normalize(&mut normalized);
                flat.extend_from_slice(&normalized);
            } else {
                flat.extend_from_slice(&vec);
            }
        }

        let n = ids.len();
        if n == 0 {
            return Err(StorageError::Corrupted(
                "fractal: need at least 1 training vector".to_string(),
            ));
        }

        // Train PQ codebooks
        let codebooks = pq::train_codebooks(&flat, dim, num_sub, max_iter, self.config.metric);

        // Persist codebooks
        let cb_def = TableDefinition::<u32, &[u8]>::new(&self.names.codebooks);
        let mut cb_tbl = self.txn.open_table(cb_def).map_err(te)?;
        let sub_dim = codebooks.sub_dim;
        for m in 0..num_sub {
            let mut bytes = Vec::with_capacity(256 * sub_dim * 4);
            for k in 0..256 {
                for d in 0..sub_dim {
                    bytes.extend_from_slice(
                        &codebooks.data[m * 256 * sub_dim + k * sub_dim + d].to_le_bytes(),
                    );
                }
            }
            cb_tbl.insert(m as u32, bytes.as_slice())?;
        }
        drop(cb_tbl);

        self.config.state = STATE_CODEBOOKS_TRAINED;
        self.codebooks = Some(codebooks);

        // Create root cluster
        let root_id = self.config.alloc_cluster_id();
        self.config.root_cluster_id = root_id;
        self.config.num_clusters += 1;

        // Compute root centroid from all training data
        let mut sums = alloc::vec![0.0f64; dim];
        for i in 0..n {
            for d in 0..dim {
                sums[d] += f64::from(flat[i * dim + d]);
            }
        }
        let pop_f64 = n as f64;
        let centroid: Vec<f32> = sums.iter().map(|s| (*s / pop_f64) as f32).collect();

        // Persist root cluster metadata
        let mut root_meta = ClusterMeta::new(root_id, NO_PARENT, 0, true);
        root_meta.set_population(0); // Will be incremented during insert

        let clusters_def = TableDefinition::<u32, &[u8]>::new(&self.names.clusters);
        let centroids_def = TableDefinition::<u32, &[u8]>::new(&self.names.centroids);
        let sums_def = TableDefinition::<u32, &[u8]>::new(&self.names.centroid_sums);
        {
            let mut ctbl = self.txn.open_table(clusters_def).map_err(te)?;
            ctbl.insert(root_id, root_meta.as_bytes().as_slice())?;

            let mut cent_tbl = self.txn.open_table(centroids_def).map_err(te)?;
            let centroid_bytes: Vec<u8> = centroid.iter().flat_map(|f| f.to_le_bytes()).collect();
            cent_tbl.insert(root_id, centroid_bytes.as_slice())?;

            let mut sum_tbl = self.txn.open_table(sums_def).map_err(te)?;
            let sum_bytes: Vec<u8> = sums.iter().flat_map(|f| f.to_le_bytes()).collect();
            sum_tbl.insert(root_id, sum_bytes.as_slice())?;
        }

        // Bulk-insert all training vectors into root cluster
        let codebooks_ref = self.codebooks.as_ref().ok_or_else(|| {
            StorageError::Corrupted(
                "fractal: codebooks not initialized after ensure_codebooks".to_string(),
            )
        })?;
        let postings_def = TableDefinition::<PostingKey, &[u8]>::new(&self.names.postings);
        let assignments_def = TableDefinition::<u64, u32>::new(&self.names.assignments);
        let vectors_def = TableDefinition::<u64, &[u8]>::new(&self.names.vectors);
        {
            let mut ptbl = self.txn.open_table(postings_def).map_err(te)?;
            let mut atbl = self.txn.open_table(assignments_def).map_err(te)?;
            let mut vtbl_opt = if self.config.store_raw_vectors {
                Some(self.txn.open_table(vectors_def).map_err(te)?)
            } else {
                None
            };

            for i in 0..n {
                let vid = ids[i];
                let vec_slice = &flat[i * dim..(i + 1) * dim];
                let pq_codes = codebooks_ref.encode(vec_slice);
                ptbl.insert(PostingKey::new(root_id, vid), pq_codes.as_slice())?;
                atbl.insert(vid, root_id)?;

                if let Some(ref mut vtbl) = vtbl_opt {
                    let raw_bytes: Vec<u8> =
                        vec_slice.iter().flat_map(|f| f.to_le_bytes()).collect();
                    vtbl.insert(vid, raw_bytes.as_slice())?;
                }
            }
        }

        // Update root population
        {
            let mut ctbl = self.txn.open_table(clusters_def).map_err(te)?;
            root_meta.set_population(n as u32);
            ctbl.insert(root_id, root_meta.as_bytes().as_slice())?;
        }

        self.config.num_vectors = n as u64;
        self.config.state = STATE_OPERATIONAL;

        // Recursively split overflowing clusters
        self.split_overflowing_clusters()?;

        self.config_dirty = true;
        self.flush()?;

        Ok(())
    }

    /// Recursively split any leaf cluster that exceeds `max_leaf_population`.
    fn split_overflowing_clusters(&mut self) -> crate::Result<()> {
        loop {
            // Find a leaf cluster that needs splitting
            let clusters_def = TableDefinition::<u32, &[u8]>::new(&self.names.clusters);
            let mut to_split: Option<u32> = None;

            {
                let ctbl = self.txn.open_table(clusters_def).map_err(te)?;
                let range = ctbl.range::<u32>(..)?;
                for entry in range {
                    let (key, val) = entry?;
                    let meta = ClusterMeta::from_bytes(val.value());
                    if meta.is_leaf()
                        && meta.population() > self.config.max_leaf_population
                        && meta.population() >= 2 * self.config.min_leaf_population
                    {
                        to_split = Some(key.value());
                        break;
                    }
                }
            }

            match to_split {
                Some(cid) => {
                    self.ensure_codebooks()?;
                    split_cluster(
                        self.txn,
                        &mut self.config,
                        cid,
                        &self.names,
                        self.codebooks.as_ref(),
                    )?;
                }
                None => break,
            }
        }
        Ok(())
    }

    /// Insert a single vector with associated temporal metadata.
    ///
    /// `hlc` and `wall_ns` are used to maintain per-cluster temporal ranges,
    /// enabling time-window pruning during search. Pass 0 for both if temporal
    /// filtering is not needed.
    pub fn insert_with_time(
        &mut self,
        vector_id: u64,
        vector: &[f32],
        hlc: u64,
        wall_ns: u64,
    ) -> crate::Result<()> {
        let dim = self.config.dim as usize;
        if vector.len() != dim {
            return Err(StorageError::Corrupted(alloc::format!(
                "fractal: expected dim={dim}, got {}",
                vector.len(),
            )));
        }

        // Validate finite
        for &v in vector {
            if !v.is_finite() {
                return Err(StorageError::Corrupted(
                    "fractal: vector contains NaN or Inf".to_string(),
                ));
            }
        }

        if self.config.state != STATE_OPERATIONAL {
            return Err(StorageError::Corrupted(
                "fractal: index not operational (train codebooks first)".to_string(),
            ));
        }

        let vec_ref: Vec<f32> = if self.config.metric == DistanceMetric::Cosine {
            let mut v = vector.to_vec();
            l2_normalize(&mut v);
            v
        } else {
            vector.to_vec()
        };

        // Remove old entry if it exists (upsert)
        let assignments_def = TableDefinition::<u64, u32>::new(&self.names.assignments);
        let old_cluster = {
            let atbl = self.txn.open_table(assignments_def).map_err(te)?;
            atbl.get(vector_id)?.map(|g| g.value())
        };
        if let Some(old_cid) = old_cluster {
            // Load the OLD vector for correct centroid removal (not the new one)
            let old_vec = self.load_vector_any(vector_id, old_cid, dim)?;
            if let Some(ref ov) = old_vec {
                self.remove_from_cluster(vector_id, old_cid, ov)?;
            } else {
                // No raw vector available: remove from posting list, decrement population
                let postings_def = TableDefinition::<PostingKey, &[u8]>::new(&self.names.postings);
                {
                    let mut ptbl = self.txn.open_table(postings_def).map_err(te)?;
                    ptbl.remove(PostingKey::new(old_cid, vector_id))?;
                }
                // Also remove from buffer
                let buffer_def = TableDefinition::<PostingKey, &[u8]>::new(&self.names.buffer);
                {
                    let mut btbl = self.txn.open_table(buffer_def).map_err(te)?;
                    btbl.remove(PostingKey::new(old_cid, vector_id))?;
                }
                // Decrement population without centroid update
                let clusters_def = TableDefinition::<u32, &[u8]>::new(&self.names.clusters);
                let mut ctbl = self.txn.open_table(clusters_def).map_err(te)?;
                let meta_opt = ctbl
                    .get(old_cid)?
                    .map(|g| ClusterMeta::from_bytes(g.value()));
                if let Some(mut meta) = meta_opt {
                    meta.set_population(meta.population().saturating_sub(1));
                    ctbl.insert(old_cid, meta.as_bytes().as_slice())?;
                }
            }
        }

        // Walk tree to find target leaf cluster
        let leaf_id = self.find_nearest_leaf(self.config.root_cluster_id, &vec_ref)?;

        // Insert into leaf's buffer
        let buffer_def = TableDefinition::<PostingKey, &[u8]>::new(&self.names.buffer);
        {
            let mut btbl = self.txn.open_table(buffer_def).map_err(te)?;
            let raw_bytes: Vec<u8> = vec_ref.iter().flat_map(|f| f.to_le_bytes()).collect();
            btbl.insert(PostingKey::new(leaf_id, vector_id), raw_bytes.as_slice())?;
        }

        // Update centroid incrementally
        let clusters_def = TableDefinition::<u32, &[u8]>::new(&self.names.clusters);
        let old_pop = {
            let ctbl = self.txn.open_table(clusters_def).map_err(te)?;
            match ctbl.get(leaf_id)? {
                Some(g) => ClusterMeta::from_bytes(g.value()).population(),
                None => 0,
            }
        };

        let new_pop = centroid_add(
            self.txn,
            &self.names.centroid_sums,
            &self.names.centroids,
            leaf_id,
            dim,
            &vec_ref,
            old_pop,
        )?;

        // Update cluster meta (population, buffer count, temporal range)
        {
            let mut ctbl = self.txn.open_table(clusters_def).map_err(te)?;
            let meta_opt = ctbl
                .get(leaf_id)?
                .map(|g| ClusterMeta::from_bytes(g.value()));
            if let Some(mut meta) = meta_opt {
                meta.set_population(new_pop);
                meta.set_buffer_count(meta.buffer_count() + 1);
                // Update temporal range
                if hlc > 0 {
                    if meta.oldest_hlc() == 0 || hlc < meta.oldest_hlc() {
                        meta.set_oldest_hlc(hlc);
                    }
                    if hlc > meta.newest_hlc() {
                        meta.set_newest_hlc(hlc);
                    }
                }
                if wall_ns > 0 {
                    if meta.oldest_wall_ns() == 0 || wall_ns < meta.oldest_wall_ns() {
                        meta.set_oldest_wall_ns(wall_ns);
                    }
                    if wall_ns > meta.newest_wall_ns() {
                        meta.set_newest_wall_ns(wall_ns);
                    }
                }
                ctbl.insert(leaf_id, meta.as_bytes().as_slice())?;
            }
        }

        // Update assignments
        {
            let mut atbl = self.txn.open_table(assignments_def).map_err(te)?;
            atbl.insert(vector_id, leaf_id)?;
        }

        // Cascade buffer if full
        let buffer_count = {
            let ctbl = self.txn.open_table(clusters_def).map_err(te)?;
            match ctbl.get(leaf_id)? {
                Some(g) => ClusterMeta::from_bytes(g.value()).buffer_count(),
                None => 0,
            }
        };

        if buffer_count >= self.config.max_buffer_size {
            self.ensure_codebooks()?;
            cascade_leaf_buffer(
                self.txn,
                leaf_id,
                self.codebooks.as_ref().ok_or_else(|| {
                    StorageError::Corrupted("fractal: codebooks not initialized".to_string())
                })?,
                &self.config,
                &self.names,
            )?;
        }

        // Split if overflowing
        if new_pop > self.config.max_leaf_population
            && new_pop >= 2 * self.config.min_leaf_population
        {
            // Must cascade buffer first before splitting
            self.ensure_codebooks()?;
            cascade_leaf_buffer(
                self.txn,
                leaf_id,
                self.codebooks.as_ref().ok_or_else(|| {
                    StorageError::Corrupted("fractal: codebooks not initialized".to_string())
                })?,
                &self.config,
                &self.names,
            )?;
            split_cluster(
                self.txn,
                &mut self.config,
                leaf_id,
                &self.names,
                self.codebooks.as_ref(),
            )?;
        }

        if old_cluster.is_none() {
            self.config.num_vectors += 1;
        }
        self.config_dirty = true;

        Ok(())
    }

    /// Insert a single vector into the index (no temporal metadata).
    pub fn insert(&mut self, vector_id: u64, vector: &[f32]) -> crate::Result<()> {
        self.insert_with_time(vector_id, vector, 0, 0)
    }

    /// Insert multiple vectors in a batch.
    pub fn insert_batch<I>(&mut self, vectors: I) -> crate::Result<()>
    where
        I: Iterator<Item = (u64, Vec<f32>)>,
    {
        for (vid, vec) in vectors {
            self.insert(vid, &vec)?;
        }
        Ok(())
    }

    /// Remove a vector from the index.
    pub fn remove(&mut self, vector_id: u64) -> crate::Result<bool> {
        let assignments_def = TableDefinition::<u64, u32>::new(&self.names.assignments);
        let cluster_id = {
            let atbl = self.txn.open_table(assignments_def).map_err(te)?;
            match atbl.get(vector_id)? {
                Some(g) => g.value(),
                None => return Ok(false),
            }
        };

        let dim = self.config.dim as usize;

        // Check if vector is in the buffer before removal (for buffer_count tracking)
        let in_buffer = {
            let buffer_def = TableDefinition::<PostingKey, &[u8]>::new(&self.names.buffer);
            let btbl = self.txn.open_table(buffer_def).map_err(te)?;
            btbl.get(PostingKey::new(cluster_id, vector_id))?.is_some()
        };

        // Try to load vector from any source for centroid update
        let raw_vec = self.load_vector_any(vector_id, cluster_id, dim)?;

        if let Some(ref vec) = raw_vec {
            self.remove_from_cluster(vector_id, cluster_id, vec)?;
        } else {
            // Remove from posting list without centroid update
            let postings_def = TableDefinition::<PostingKey, &[u8]>::new(&self.names.postings);
            let mut ptbl = self.txn.open_table(postings_def).map_err(te)?;
            ptbl.remove(PostingKey::new(cluster_id, vector_id))?;

            // Still decrement population even without centroid update
            let clusters_def = TableDefinition::<u32, &[u8]>::new(&self.names.clusters);
            let mut ctbl = self.txn.open_table(clusters_def).map_err(te)?;
            let meta_opt = ctbl
                .get(cluster_id)?
                .map(|g| ClusterMeta::from_bytes(g.value()));
            if let Some(mut meta) = meta_opt {
                meta.set_population(meta.population().saturating_sub(1));
                ctbl.insert(cluster_id, meta.as_bytes().as_slice())?;
            }
        }

        // Remove from buffer
        {
            let buffer_def = TableDefinition::<PostingKey, &[u8]>::new(&self.names.buffer);
            let mut btbl = self.txn.open_table(buffer_def).map_err(te)?;
            btbl.remove(PostingKey::new(cluster_id, vector_id))?;
        }

        // Decrement buffer_count if the vector was in the buffer
        if in_buffer {
            let clusters_def = TableDefinition::<u32, &[u8]>::new(&self.names.clusters);
            let mut ctbl = self.txn.open_table(clusters_def).map_err(te)?;
            let meta_opt = ctbl
                .get(cluster_id)?
                .map(|g| ClusterMeta::from_bytes(g.value()));
            if let Some(mut meta) = meta_opt {
                meta.set_buffer_count(meta.buffer_count().saturating_sub(1));
                ctbl.insert(cluster_id, meta.as_bytes().as_slice())?;
            }
        }

        // Remove assignment
        {
            let mut atbl = self.txn.open_table(assignments_def).map_err(te)?;
            atbl.remove(vector_id)?;
        }

        // Remove raw vector
        if self.config.store_raw_vectors {
            let vectors_def = TableDefinition::<u64, &[u8]>::new(&self.names.vectors);
            let mut vtbl = self.txn.open_table(vectors_def).map_err(te)?;
            vtbl.remove(vector_id)?;
        }

        self.config.num_vectors = self.config.num_vectors.saturating_sub(1);
        self.config_dirty = true;

        // Check merge condition
        let clusters_def = TableDefinition::<u32, &[u8]>::new(&self.names.clusters);
        let pop = {
            let ctbl = self.txn.open_table(clusters_def).map_err(te)?;
            match ctbl.get(cluster_id)? {
                Some(g) => ClusterMeta::from_bytes(g.value()).population(),
                None => 0,
            }
        };

        if pop < self.config.min_leaf_population && pop > 0 {
            merge_cluster(self.txn, &mut self.config, cluster_id, &self.names)?;
        }

        Ok(true)
    }

    /// Search the index for nearest neighbors.
    pub fn search(
        &mut self,
        query: &[f32],
        params: &FractalSearchParams,
    ) -> crate::Result<Vec<Neighbor<u64>>> {
        self.ensure_codebooks()?;
        self.flush()?;
        search::search_write(self, query, params)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Walk from a cluster node to the nearest leaf.
    fn find_nearest_leaf(&self, start: u32, vector: &[f32]) -> crate::Result<u32> {
        let dim = self.config.dim as usize;
        let clusters_def = TableDefinition::<u32, &[u8]>::new(&self.names.clusters);
        let centroids_def = TableDefinition::<u32, &[u8]>::new(&self.names.centroids);
        let hier_def = TableDefinition::<HierarchyKey, ()>::new(&self.names.hierarchy);

        let mut current = start;
        loop {
            let meta = {
                let ctbl = self.txn.open_table(clusters_def).map_err(te)?;
                match ctbl.get(current)? {
                    Some(g) => ClusterMeta::from_bytes(g.value()),
                    None => {
                        return Err(StorageError::Corrupted(alloc::format!(
                            "fractal: cluster {current} not found"
                        )));
                    }
                }
            };

            if meta.is_leaf() {
                return Ok(current);
            }

            // Internal node: find nearest child
            let htbl = self.txn.open_table(hier_def).map_err(te)?;
            let ctbl = self.txn.open_table(centroids_def).map_err(te)?;

            let mut best_child = current;
            let mut best_dist = f32::MAX;

            let range = htbl.range(
                HierarchyKey::children_start(current)..=HierarchyKey::children_end(current),
            )?;

            for entry in range {
                let (key, _) = entry?;
                let child_id = key.value().child_id;

                if let Some(cg) = ctbl.get(child_id)? {
                    let bytes = cg.value();
                    if bytes.len() < dim * 4 {
                        continue;
                    }
                    let child_centroid: Vec<f32> = (0..dim)
                        .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                        .collect();
                    let dist = self.config.metric.compute(vector, &child_centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_child = child_id;
                    }
                }
            }

            if best_child == current {
                // No children found -- shouldn't happen for internal nodes
                return Err(StorageError::Corrupted(alloc::format!(
                    "fractal: internal cluster {current} has no children"
                )));
            }

            current = best_child;
        }
    }

    fn remove_from_cluster(
        &self,
        vector_id: u64,
        cluster_id: u32,
        vector: &[f32],
    ) -> crate::Result<()> {
        let dim = self.config.dim as usize;

        // Remove from posting list
        let postings_def = TableDefinition::<PostingKey, &[u8]>::new(&self.names.postings);
        {
            let mut ptbl = self.txn.open_table(postings_def).map_err(te)?;
            ptbl.remove(PostingKey::new(cluster_id, vector_id))?;
        }

        // Update centroid
        let clusters_def = TableDefinition::<u32, &[u8]>::new(&self.names.clusters);
        let old_pop = {
            let ctbl = self.txn.open_table(clusters_def).map_err(te)?;
            match ctbl.get(cluster_id)? {
                Some(g) => ClusterMeta::from_bytes(g.value()).population(),
                None => 0,
            }
        };

        let new_pop = centroid_remove(
            self.txn,
            &self.names.centroid_sums,
            &self.names.centroids,
            cluster_id,
            dim,
            vector,
            old_pop,
        )?;

        {
            let mut ctbl = self.txn.open_table(clusters_def).map_err(te)?;
            let meta_opt = ctbl
                .get(cluster_id)?
                .map(|g| ClusterMeta::from_bytes(g.value()));
            if let Some(mut meta) = meta_opt {
                meta.set_population(new_pop);
                ctbl.insert(cluster_id, meta.as_bytes().as_slice())?;
            }
        }

        Ok(())
    }

    fn load_raw_vector(&self, vector_id: u64, dim: usize) -> crate::Result<Option<Vec<f32>>> {
        if !self.config.store_raw_vectors {
            return Ok(None);
        }
        let vectors_def = TableDefinition::<u64, &[u8]>::new(&self.names.vectors);
        let vtbl = self.txn.open_table(vectors_def).map_err(te)?;
        match vtbl.get(vector_id)? {
            Some(g) => {
                let bytes = g.value();
                if bytes.len() < dim * 4 {
                    return Ok(None);
                }
                let vec: Vec<f32> = (0..dim)
                    .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                    .collect();
                Ok(Some(vec))
            }
            None => Ok(None),
        }
    }

    /// Try to load a vector from raw vectors table, then buffer table.
    fn load_vector_any(
        &self,
        vector_id: u64,
        cluster_id: u32,
        dim: usize,
    ) -> crate::Result<Option<Vec<f32>>> {
        // Try raw vectors table first
        if let Some(vec) = self.load_raw_vector(vector_id, dim)? {
            return Ok(Some(vec));
        }
        // Try buffer table (vector may not yet be cascaded)
        let buffer_def = TableDefinition::<PostingKey, &[u8]>::new(&self.names.buffer);
        let btbl = self.txn.open_table(buffer_def).map_err(te)?;
        if let Some(g) = btbl.get(PostingKey::new(cluster_id, vector_id))? {
            let bytes = g.value();
            if bytes.len() < dim * 4 {
                return Ok(None);
            }
            let vec: Vec<f32> = (0..dim)
                .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
                .collect();
            return Ok(Some(vec));
        }
        Ok(None)
    }

    /// Returns the current config (for testing).
    pub fn config(&self) -> &FractalIndexConfig {
        &self.config
    }
}

impl Drop for FractalIndex<'_> {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

// ---------------------------------------------------------------------------
// ReadOnlyFractalIndex -- read-only index handle
// ---------------------------------------------------------------------------

/// A read-only fractal vector index for search within a [`ReadTransaction`].
///
/// Obtained via [`ReadTransaction::open_fractal_index`].
pub struct ReadOnlyFractalIndex {
    pub(crate) config: FractalIndexConfig,
    pub(crate) names: TableNames,
    pub(crate) codebooks: Codebooks,
}

impl ReadOnlyFractalIndex {
    /// Open a fractal index for reading.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn open(
        txn: &ReadTransaction,
        def: &FractalIndexDefinition,
    ) -> Result<Self, TableError> {
        let names = TableNames::new(def.name());

        let meta_def = TableDefinition::<&str, &[u8]>::new(&names.meta);
        let meta_tbl = txn.open_table(meta_def)?;
        let config = match meta_tbl.get("config")? {
            Some(guard) => decode_fractal_config(guard.value()),
            None => {
                return Err(TableError::TableDoesNotExist(def.name().to_string()));
            }
        };
        drop(meta_tbl);

        validate_config(&config).map_err(TableError::Storage)?;

        // Eagerly load codebooks
        let cb_def = TableDefinition::<u32, &[u8]>::new(&names.codebooks);
        let cb_tbl = txn.open_table(cb_def)?;
        let num_sub = config.num_subvectors as usize;
        let sub_dim = config.sub_dim();
        let mut data = alloc::vec![0.0f32; num_sub * 256 * sub_dim];
        for m in 0..num_sub {
            if let Some(guard) = cb_tbl.get(m as u32)? {
                let bytes = guard.value();
                let expected_len = 256 * sub_dim * 4;
                if bytes.len() < expected_len {
                    return Err(TableError::Storage(StorageError::Corrupted(
                        alloc::format!(
                            "fractal: codebook {} has insufficient bytes: expected {}, got {}",
                            m,
                            expected_len,
                            bytes.len(),
                        ),
                    )));
                }
                for k in 0..256 {
                    for d in 0..sub_dim {
                        let offset = (k * sub_dim + d) * 4;
                        data[m * 256 * sub_dim + k * sub_dim + d] =
                            f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                    }
                }
            }
        }

        let codebooks = Codebooks {
            data,
            num_subvectors: num_sub,
            sub_dim,
        };

        Ok(Self {
            config,
            names,
            codebooks,
        })
    }

    /// Search the index for nearest neighbors.
    pub fn search(
        &self,
        txn: &ReadTransaction,
        query: &[f32],
        params: &FractalSearchParams,
    ) -> crate::Result<Vec<Neighbor<u64>>> {
        search::search_read(self, txn, query, params)
    }

    /// Returns the current config (for testing).
    pub fn config(&self) -> &FractalIndexConfig {
        &self.config
    }
}
