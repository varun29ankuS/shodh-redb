use alloc::collections::BinaryHeap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cmp::Ordering as CmpOrdering;

use crate::TableDefinition;
use crate::error::StorageError;
use crate::storage_traits::{ReadTable, StorageRead, StorageWrite, WriteTable};
use crate::vector_ops::{DistanceMetric, Neighbor, l2_normalize};

use super::adc::AdcTable;
use super::config::{IndexConfig, IvfPqIndexDefinition, STATE_TRAINED, SearchParams};
use super::kmeans;
use super::metadata::{MetadataMap, passes_filter};
use super::pq::{self, Codebooks};
use super::types::{PostingKey, decode_index_config, encode_index_config};

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
fn postings_name(name: &str) -> String {
    alloc::format!("__ivfpq:{name}:postings")
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

        // Eagerly create the other tables.
        {
            let cn = centroids_name(&name);
            let _ = txn.open_storage_table(TableDefinition::<u32, &[u8]>::new(&cn))?;
            let cb = codebooks_name(&name);
            let _ = txn.open_storage_table(TableDefinition::<u32, &[u8]>::new(&cb))?;
            let pn = postings_name(&name);
            let _ = txn.open_storage_table(TableDefinition::<PostingKey, &[u8]>::new(&pn))?;
            let an = assignments_name(&name);
            let _ = txn.open_storage_table(TableDefinition::<u64, u32>::new(&an))?;
            if config.store_raw_vectors {
                let vn = vectors_name(&name);
                let _ = txn.open_storage_table(TableDefinition::<u64, &[u8]>::new(&vn))?;
            }
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

        // 2. Train PQ codebooks.
        let codebooks =
            pq::train_codebooks(&flat, dim, num_subvectors, max_iter, self.config.metric)?;

        // 3. Clear stale data from a previous training cycle.
        //    Old centroids beyond actual_k, old postings, assignments, and raw
        //    vectors must be removed -- they reference stale cluster IDs from the
        //    previous centroid set.
        self.clear_stale_training_data(old_k, actual_k)?;

        // 4. Persist new centroids.
        {
            let tn = centroids_name(&self.name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            for c in 0..actual_k {
                let bytes: Vec<u8> = centroid_data[c * dim..(c + 1) * dim]
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                #[allow(clippy::cast_possible_truncation)]
                table.st_insert(&(c as u32), &bytes.as_slice())?;
            }
        }

        // 5. Persist new PQ codebooks.
        {
            let tn = codebooks_name(&self.name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            for m in 0..num_subvectors {
                let bytes = codebooks.serialize_codebook(m);
                #[allow(clippy::cast_possible_truncation)]
                table.st_insert(&(m as u32), &bytes.as_slice())?;
            }
        }

        // 6. Update config -- persist immediately since training is a major event.
        self.config.state = STATE_TRAINED;
        self.config.num_vectors = 0;
        self.persist_config_inner()?;
        self.config_dirty = false;

        self.centroids = Some(centroid_data);
        self.codebooks = Some(codebooks);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Insert / Remove
    // -----------------------------------------------------------------------

    /// Insert a vector into the index. Index must be trained first.
    ///
    /// If `vector_id` already exists, the old entry is replaced (upsert semantics).
    /// Returns an error if the vector contains NaN or Inf values.
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
        let codebooks = self.load_codebooks()?;

        let (cluster_id, _) = kmeans::assign_nearest(
            vec_ref,
            &centroids,
            dim,
            self.config.num_clusters as usize,
            self.config.metric,
        );
        let pq_codes = codebooks.encode(vec_ref);

        // Check if this vector_id already exists (H5/H6: handle duplicates).
        let old_cluster = {
            let tn = assignments_name(&self.name);
            let def = TableDefinition::<u64, u32>::new(&tn);
            let table = self.txn.open_storage_table(def)?;
            table.st_get(&vector_id)?.map(|g| g.value())
        };

        // Remove old posting entry if the vector existed in a different (or same) cluster.
        if let Some(old_cid) = old_cluster {
            let tn = postings_name(&self.name);
            let def = TableDefinition::<PostingKey, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            table.st_remove(&PostingKey::new(old_cid, vector_id))?;
        }

        // Insert the new posting entry.
        {
            let tn = postings_name(&self.name);
            let def = TableDefinition::<PostingKey, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            table.st_insert(
                &PostingKey::new(cluster_id, vector_id),
                &pq_codes.as_slice(),
            )?;
        }
        {
            let tn = assignments_name(&self.name);
            let def = TableDefinition::<u64, u32>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            table.st_insert(&vector_id, &cluster_id)?;
        }
        if self.config.store_raw_vectors {
            let tn = vectors_name(&self.name);
            let def = TableDefinition::<u64, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            let bytes: Vec<u8> = vec_ref.iter().flat_map(|f| f.to_le_bytes()).collect();
            table.st_insert(&vector_id, &bytes.as_slice())?;
        }

        // Only increment count for genuinely new vectors.
        if old_cluster.is_none() {
            self.config.num_vectors = self.config.num_vectors.saturating_add(1);
            self.config_dirty = true;
        }
        Ok(())
    }

    /// Bulk insert vectors.
    ///
    /// If a `vector_id` already exists, the old entry is replaced (upsert semantics).
    pub fn insert_batch<I>(&mut self, vectors: I) -> crate::Result<u64>
    where
        I: Iterator<Item = (u64, Vec<f32>)>,
    {
        self.ensure_trained()?;
        let dim = self.config.dim as usize;
        let centroids = self.load_centroids()?;
        let codebooks = self.load_codebooks()?;
        let num_clusters = self.config.num_clusters as usize;
        let metric = self.config.metric;
        let store_raw = self.config.store_raw_vectors;

        let pn = postings_name(&self.name);
        let pd = TableDefinition::<PostingKey, &[u8]>::new(&pn);
        let mut pt = self.txn.open_storage_table(pd)?;

        let an = assignments_name(&self.name);
        let ad = TableDefinition::<u64, u32>::new(&an);
        let mut at = self.txn.open_storage_table(ad)?;

        // Open vectors table once outside loop (M2 fix).
        let vn;
        let mut vt_opt = if store_raw {
            vn = vectors_name(&self.name);
            let vd = TableDefinition::<u64, &[u8]>::new(&vn);
            Some(self.txn.open_storage_table(vd)?)
        } else {
            None
        };

        let mut count = 0u64;

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
            let pq_codes = codebooks.encode(&vec);

            // Check for existing assignment (handle duplicates).
            let old_cluster = at.st_get(&vector_id)?.map(|g| g.value());
            if let Some(old_cid) = old_cluster {
                pt.st_remove(&PostingKey::new(old_cid, vector_id))?;
            }

            pt.st_insert(
                &PostingKey::new(cluster_id, vector_id),
                &pq_codes.as_slice(),
            )?;
            at.st_insert(&vector_id, &cluster_id)?;

            if let Some(ref mut vt) = vt_opt {
                let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
                vt.st_insert(&vector_id, &bytes.as_slice())?;
            }

            if old_cluster.is_none() {
                count += 1;
            }
        }

        if count > 0 {
            self.config.num_vectors = self.config.num_vectors.saturating_add(count);
            self.config_dirty = true;
        }
        Ok(count)
    }

    /// Remove a vector from the index. Returns `true` if found and removed.
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

        {
            let tn = postings_name(&self.name);
            let def = TableDefinition::<PostingKey, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            table.st_remove(&PostingKey::new(cluster_id, vector_id))?;
        }

        if self.config.store_raw_vectors {
            let tn = vectors_name(&self.name);
            let def = TableDefinition::<u64, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            table.st_remove(&vector_id)?;
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
    pub fn insert_metadata(
        &mut self,
        vector_id: u64,
        metadata: &MetadataMap,
    ) -> crate::Result<()> {
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
    pub fn search(
        &mut self,
        query: &[f32],
        params: &SearchParams,
    ) -> crate::Result<Vec<Neighbor<u64>>> {
        self.ensure_trained()?;
        // Flush pending config before search so reads are consistent.
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

        // Lazy-load centroids and codebooks from disk if not cached (C1 fix).
        let centroids = self.load_centroids()?;
        let codebooks = self.load_codebooks()?;

        let query_owned;
        let q = if self.config.metric == DistanceMetric::Cosine {
            // Zero-norm query in cosine space is undefined (division by zero
            // in the distance computation). Return empty results rather than
            // producing arbitrary rankings.
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

        let adc = AdcTable::build(q, &codebooks, self.config.metric);

        let cap = if params.rerank && self.config.store_raw_vectors {
            params.candidates.max(params.k)
        } else {
            params.k
        };
        let mut heap = CandidateHeap::new(cap);

        {
            let tn = postings_name(&self.name);
            let def = TableDefinition::<PostingKey, &[u8]>::new(&tn);
            let table = self.txn.open_storage_table(def)?;

            // Open metadata table only when a filter is active.
            let meta_table = if params.filter.is_some() {
                let mn = vector_meta_name(&self.name);
                let mdef = TableDefinition::<u64, &[u8]>::new(&mn);
                Some(self.txn.open_storage_table(mdef)?)
            } else {
                None
            };

            for &(cid, _) in &probes {
                let start = PostingKey::cluster_start(cid);
                let end = PostingKey::cluster_end(cid);
                let range_iter = table.st_range(Some(&start), Some(&end), true, true)?;
                for entry in range_iter {
                    let (kg, vg) = entry?;
                    let vid = kg.value().vector_id;
                    let dist = adc.approximate_distance(vg.value());
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

        if params.rerank && self.config.store_raw_vectors {
            self.rerank_write(q, &heap.into_sorted(), params.k)
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
    /// postings, assignments, and raw vectors -- they reference cluster IDs from
    /// the previous centroid set and are invalid after re-training.
    fn clear_stale_training_data(&self, old_k: usize, new_k: usize) -> crate::Result<()> {
        // Remove orphaned centroid rows if cluster count shrank.
        if old_k > new_k {
            let tn = centroids_name(&self.name);
            let def = TableDefinition::<u32, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            for c in new_k..old_k {
                #[allow(clippy::cast_possible_truncation)]
                table.st_remove(&(c as u32))?;
            }
        }

        // Clear all postings -- they reference stale cluster assignments.
        {
            let tn = postings_name(&self.name);
            let def = TableDefinition::<PostingKey, &[u8]>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            table.st_drain_all()?;
        }

        // Clear all assignments.
        {
            let tn = assignments_name(&self.name);
            let def = TableDefinition::<u64, u32>::new(&tn);
            let mut table = self.txn.open_storage_table(def)?;
            table.st_drain_all()?;
        }

        // Clear raw vectors if stored.
        if self.config.store_raw_vectors {
            let tn = vectors_name(&self.name);
            let def = TableDefinition::<u64, &[u8]>::new(&tn);
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

    fn rerank_write(
        &self,
        query: &[f32],
        candidates: &[Neighbor<u64>],
        k: usize,
    ) -> crate::Result<Vec<Neighbor<u64>>> {
        let tn = vectors_name(&self.name);
        let def = TableDefinition::<u64, &[u8]>::new(&tn);
        let table = self.txn.open_storage_table(def)?;
        let metric = self.config.metric;

        let dim = query.len();
        let mut results: Vec<Neighbor<u64>> = Vec::with_capacity(candidates.len());
        for cand in candidates {
            if let Some(guard) = table.st_get(&cand.key)? {
                let vec = bytes_to_f32_vec(guard.value());
                // Skip truncated/corrupted vectors -- fall back to excluding
                // them rather than computing a misleading distance.
                if vec.len() != dim {
                    continue;
                }
                results.push(Neighbor {
                    key: cand.key,
                    distance: metric.compute(query, &vec),
                });
            }
        }
        results.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        results.truncate(k);
        Ok(results)
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

        let num_subvectors = config.num_subvectors as usize;
        let sub_dim = config.sub_dim();
        let codebooks = {
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
            // Zero-norm query in cosine space is undefined (division by zero
            // in the distance computation). Return empty results rather than
            // producing arbitrary rankings.
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

        let adc = AdcTable::build(q, &self.codebooks, self.config.metric);

        let cap = if params.rerank && self.config.store_raw_vectors {
            params.candidates.max(params.k)
        } else {
            params.k
        };
        let mut heap = CandidateHeap::new(cap);

        {
            let tn = postings_name(&self.name);
            let def = TableDefinition::<PostingKey, &[u8]>::new(&tn);
            let table = txn.open_storage_table(def)?;

            let meta_table = if params.filter.is_some() {
                let mn = vector_meta_name(&self.name);
                let mdef = TableDefinition::<u64, &[u8]>::new(&mn);
                Some(txn.open_storage_table(mdef)?)
            } else {
                None
            };

            for &(cid, _) in &probes {
                let start = PostingKey::cluster_start(cid);
                let end = PostingKey::cluster_end(cid);
                let range_iter = table.st_range(Some(&start), Some(&end), true, true)?;
                for entry in range_iter {
                    let (kg, vg) = entry?;
                    let vid = kg.value().vector_id;
                    let dist = adc.approximate_distance(vg.value());
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

        if params.rerank && self.config.store_raw_vectors {
            let tn = vectors_name(&self.name);
            let def = TableDefinition::<u64, &[u8]>::new(&tn);
            let table = txn.open_storage_table(def)?;
            let metric = self.config.metric;

            let sorted = heap.into_sorted();
            let mut results: Vec<Neighbor<u64>> = Vec::with_capacity(sorted.len());
            for cand in &sorted {
                if let Some(guard) = table.st_get(&cand.key)? {
                    let vec = bytes_to_f32_vec(guard.value());
                    // Skip truncated/corrupted vectors -- fall back to excluding
                    // them rather than computing a misleading distance.
                    if vec.len() != dim {
                        continue;
                    }
                    results.push(Neighbor {
                        key: cand.key,
                        distance: metric.compute(q, &vec),
                    });
                }
            }
            results.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
            results.truncate(params.k);
            Ok(results)
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

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .filter_map(|c| c.try_into().ok().map(f32::from_le_bytes))
        .collect()
}
