#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use redb::{Database, DistanceMetric, IvfPqIndexDefinition, ReadableDatabase, SearchParams};
use tempfile::NamedTempFile;

/// Operations the fuzzer can perform on the index.
#[derive(Arbitrary, Debug)]
enum IndexOp {
    /// Insert a vector with the given ID and f32 data.
    Insert { id_sel: u16, vector_data: Vec<u8> },
    /// Remove a vector by ID.
    Remove { id_sel: u16 },
    /// Search with a query vector.
    Search {
        vector_data: Vec<u8>,
        k: u8,
        nprobe: u8,
    },
    /// Commit the current write transaction and start a new one.
    Commit,
}

/// Fuzz input: configuration + sequence of operations.
#[derive(Arbitrary, Debug)]
struct FuzzInput {
    /// Dimension selector: 0..3 -> 4/6/8/12.
    dim_sel: u8,
    /// Number of clusters selector: 0..2 -> 2/4/8.
    cluster_sel: u8,
    /// Metric selector: 0..2 -> EuclideanSq/Cosine/DotProduct.
    metric_sel: u8,
    /// Whether to store raw vectors for reranking.
    store_raw: bool,
    /// Training vectors (raw f32 bytes, will be chunked by dim).
    training_data: Vec<u8>,
    /// Operations to perform after training.
    ops: Vec<IndexOp>,
}

fn select_dim(sel: u8) -> u32 {
    match sel % 4 {
        0 => 4,
        1 => 6,
        2 => 8,
        _ => 12,
    }
}

fn select_clusters(sel: u8) -> u32 {
    match sel % 3 {
        0 => 2,
        1 => 4,
        _ => 8,
    }
}

fn select_metric(sel: u8) -> DistanceMetric {
    match sel % 3 {
        0 => DistanceMetric::EuclideanSq,
        1 => DistanceMetric::Cosine,
        _ => DistanceMetric::DotProduct,
    }
}

/// Build a vector of f32 from raw bytes, replacing non-finite values.
fn build_vector(dim: usize, raw: &[u8]) -> Vec<f32> {
    let mut vec = Vec::with_capacity(dim);
    for i in 0..dim {
        let off = (i * 4) % raw.len().max(1);
        let bytes: [u8; 4] = [
            raw.get(off).copied().unwrap_or(0),
            raw.get(off + 1).copied().unwrap_or(0),
            raw.get(off + 2).copied().unwrap_or(0),
            raw.get(off + 3).copied().unwrap_or(0),
        ];
        let val = f32::from_le_bytes(bytes);
        if val.is_finite() {
            vec.push(val);
        } else {
            vec.push(0.1); // Avoid zero for cosine metric.
        }
    }
    vec
}

fuzz_target!(|input: FuzzInput| {
    let dim = select_dim(input.dim_sel) as usize;
    let num_clusters = select_clusters(input.cluster_sel);
    let metric = select_metric(input.metric_sel);

    // num_subvectors must divide dim evenly. Use 2 if possible, else 1.
    let num_subvectors = if dim % 2 == 0 { 2u32 } else { 1 };

    if input.training_data.len() < 4 {
        return;
    }

    // Limit ops to prevent timeouts.
    let max_ops = input.ops.len().min(32);

    // Create temp database.
    let tmpfile = match NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    let db = match Database::create(tmpfile.path()) {
        Ok(db) => db,
        Err(_) => return,
    };

    // Build training vectors from arbitrary data.
    let num_training = (input.training_data.len() / (dim * 4).max(1))
        .max(num_clusters as usize)
        .min(64);
    let training_vecs: Vec<(u64, Vec<f32>)> = (0..num_training)
        .map(|i| {
            let offset = i * dim * 4;
            let slice = if offset < input.training_data.len() {
                &input.training_data[offset..]
            } else {
                &input.training_data
            };
            (i as u64 + 10000, build_vector(dim, slice))
        })
        .collect();

    // Use a leaked static str for the index name (fuzz targets are short-lived).
    let index_name: &'static str = Box::leak(Box::new(String::from("fuzz_idx")));

    let def = IvfPqIndexDefinition::new(
        index_name,
        dim as u32,
        num_clusters,
        num_subvectors,
        metric,
    );
    let def = if input.store_raw {
        def.with_raw_vectors()
    } else {
        def
    };

    // Train the index.
    {
        let write_txn = match db.begin_write() {
            Ok(t) => t,
            Err(_) => return,
        };
        {
            let mut idx = match write_txn.open_ivfpq_index(&def) {
                Ok(i) => i,
                Err(_) => return,
            };
            if idx.train(training_vecs.into_iter(), 5).is_err() {
                return;
            }
        }
        if write_txn.commit().is_err() {
            return;
        }
    }

    // Execute operations in batches separated by commits.
    let mut op_idx = 0;
    while op_idx < max_ops {
        let write_txn = match db.begin_write() {
            Ok(t) => t,
            Err(_) => return,
        };
        {
            let mut idx = match write_txn.open_ivfpq_index(&def) {
                Ok(i) => i,
                Err(_) => return,
            };

            while op_idx < max_ops {
                match &input.ops[op_idx] {
                    IndexOp::Insert { id_sel, vector_data } => {
                        let id = *id_sel as u64;
                        let vec = build_vector(dim, vector_data);
                        let _ = idx.insert(id, &vec);
                    }

                    IndexOp::Remove { id_sel } => {
                        let _ = idx.remove(*id_sel as u64);
                    }

                    IndexOp::Search {
                        vector_data,
                        k,
                        nprobe,
                    } => {
                        let query = build_vector(dim, vector_data);
                        let k_val = (*k as usize).clamp(1, 20);
                        let mut params = SearchParams::top_k(k_val);
                        params.nprobe = (*nprobe as u32).clamp(1, num_clusters);
                        let results = idx.search(&query, &params);
                        if let Ok(results) = results {
                            for r in &results {
                                assert!(
                                    r.distance.is_finite(),
                                    "non-finite distance: {}",
                                    r.distance
                                );
                            }
                            assert!(results.len() <= k_val);
                        }
                    }

                    IndexOp::Commit => {
                        op_idx += 1;
                        break;
                    }
                }
                op_idx += 1;
            }
        }
        // idx dropped, commit the transaction.
        let _ = write_txn.commit();
    }

    // Final read-transaction search to verify read path.
    {
        let read_txn = match db.begin_read() {
            Ok(t) => t,
            Err(_) => return,
        };
        let ro_idx = match read_txn.open_ivfpq_index(&def) {
            Ok(i) => i,
            Err(_) => return,
        };
        let query = build_vector(dim, &input.training_data);
        let params = SearchParams::top_k(5);
        let results = ro_idx.search(&read_txn, &query, &params);
        if let Ok(results) = results {
            for r in &results {
                assert!(r.distance.is_finite(), "non-finite distance in read path");
            }
        }
    }
});
