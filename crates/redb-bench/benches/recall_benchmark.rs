//! Recall benchmark for IVF-PQ vector indexes.
//!
//! Generates a synthetic clustered dataset (Gaussian blobs), builds the index,
//! and measures recall@1, recall@10, recall@100 against brute-force
//! ground truth. Also reports QPS and build time.
//!
//! Usage:
//!   cargo bench -p shodh-redb-bench --bench recall_benchmark
//!
//! Optional: set `SIFT1M_PATH` env var to a directory containing `sift_base.fvecs`
//! and `sift_query.fvecs` for real-dataset evaluation.

use std::collections::HashSet;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use shodh_redb::bf_tree_store::{BfTreeConfig, BfTreeDatabase, DurabilityMode};
use shodh_redb::{
    Database, DistanceMetric, IvfPqIndexDefinition, ReadableDatabase, SearchParams,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DIM: usize = 128;
const NUM_CLUSTERS: usize = 50;
const NUM_BASE: usize = 100_000;
const NUM_QUERIES: usize = 1_000;
const GROUND_TRUTH_K: usize = 100;

// IVF-PQ parameters
const IVFPQ_CLUSTERS: u32 = 256;
const IVFPQ_SUBVECTORS: u32 = 16; // 128 / 16 = 8 floats per subvector
const KMEANS_ITERS: usize = 15;

// ---------------------------------------------------------------------------
// Synthetic data generation -- Gaussian blobs
// ---------------------------------------------------------------------------

/// Simple LCG PRNG for reproducibility without pulling in rand.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Approximate normal distribution via Box-Muller transform.
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() % max as u64) as usize
    }
}

fn generate_cluster_centers(rng: &mut Rng, n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dim).map(|_| rng.next_normal() * 5.0).collect())
        .collect()
}

fn generate_vectors(
    rng: &mut Rng,
    centers: &[Vec<f32>],
    count: usize,
    _dim: usize,
    stddev: f32,
) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| {
            let center = &centers[rng.next_usize(centers.len())];
            center
                .iter()
                .map(|&c| c + rng.next_normal() * stddev)
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// fvecs / ivecs file readers (SIFT1M format)
// ---------------------------------------------------------------------------

/// Read an fvecs file: each record is [dim: i32 LE][data: f32 x dim LE].
fn read_fvecs(path: &Path) -> Vec<Vec<f32>> {
    let mut file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Cannot open {}: {e}", path.display()));
    let mut buf4 = [0u8; 4];
    let mut vecs = Vec::new();
    loop {
        if file.read_exact(&mut buf4).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(buf4) as usize;
        let mut data = vec![0u8; dim * 4];
        file.read_exact(&mut data)
            .unwrap_or_else(|e| panic!("Truncated fvecs at vec {}: {e}", vecs.len()));
        let vec: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        vecs.push(vec);
    }
    vecs
}

/// Read an ivecs file: each record is [dim: i32 LE][data: i32 x dim LE].
/// Returns Vec<Vec<i32>> (ground-truth neighbor IDs).
fn read_ivecs(path: &Path) -> Vec<Vec<i32>> {
    let mut file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Cannot open {}: {e}", path.display()));
    let mut buf4 = [0u8; 4];
    let mut vecs = Vec::new();
    loop {
        if file.read_exact(&mut buf4).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(buf4) as usize;
        let mut data = vec![0u8; dim * 4];
        file.read_exact(&mut data)
            .unwrap_or_else(|e| panic!("Truncated ivecs at vec {}: {e}", vecs.len()));
        let vec: Vec<i32> = data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        vecs.push(vec);
    }
    vecs
}

// ---------------------------------------------------------------------------
// Brute-force ground truth
// ---------------------------------------------------------------------------

fn brute_force_knn(
    base: &[Vec<f32>],
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<(u64, f32)> {
    let mut dists: Vec<(u64, f32)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, metric.compute(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.total_cmp(&b.1));
    dists.truncate(k);
    dists
}

// ---------------------------------------------------------------------------
// Recall computation
// ---------------------------------------------------------------------------

fn compute_recall(ground_truth: &[(u64, f32)], results: &[(u64, f32)], k: usize) -> f32 {
    let gt_set: HashSet<u64> = ground_truth.iter().take(k).map(|&(id, _)| id).collect();
    let result_set: HashSet<u64> = results.iter().take(k).map(|&(id, _)| id).collect();
    let intersection = gt_set.intersection(&result_set).count();
    intersection as f32 / k as f32
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let metric = DistanceMetric::EuclideanSq;

    // Check for SIFT1M real dataset
    let (base_vecs, query_vecs, ground_truth) = if let Ok(sift_path) = std::env::var("SIFT1M_PATH")
    {
        let dir = Path::new(&sift_path);
        println!("=== Recall Benchmark (SIFT1M real data) ===");

        print!("Loading sift_base.fvecs...");
        let base = read_fvecs(&dir.join("sift_base.fvecs"));
        println!(" {} vectors, dim={}", base.len(), base[0].len());

        print!("Loading sift_query.fvecs...");
        let queries = read_fvecs(&dir.join("sift_query.fvecs"));
        println!(" {} vectors", queries.len());

        // Check for pre-computed ground truth
        let gt_path = dir.join("sift_groundtruth.ivecs");
        let gt = if gt_path.exists() {
            print!("Loading sift_groundtruth.ivecs...");
            let gt_ivecs = read_ivecs(&gt_path);
            let gt: Vec<Vec<(u64, f32)>> = gt_ivecs
                .iter()
                .zip(queries.iter())
                .map(|(ids, q)| {
                    ids.iter()
                        .take(GROUND_TRUTH_K)
                        .map(|&id| {
                            let d = metric.compute(q, &base[id as usize]);
                            (id as u64, d)
                        })
                        .collect()
                })
                .collect();
            println!(" done");
            gt
        } else {
            print!("Computing brute-force ground truth (k={GROUND_TRUTH_K})...");
            let gt_start = Instant::now();
            let gt: Vec<Vec<(u64, f32)>> = queries
                .iter()
                .map(|q| brute_force_knn(&base, q, GROUND_TRUTH_K, metric))
                .collect();
            println!(" done ({:.2}s)", gt_start.elapsed().as_secs_f64());
            gt
        };

        println!();
        (base, queries, gt)
    } else {
        println!("=== Recall Benchmark (synthetic data) ===");
        println!(
            "Base vectors: {NUM_BASE}, Queries: {NUM_QUERIES}, Dim: {DIM}, Clusters: {NUM_CLUSTERS}"
        );
        println!("  (Set SIFT1M_PATH to use real SIFT1M data instead)");
        println!();

        let mut rng = Rng::new(42);
        let centers = generate_cluster_centers(&mut rng, NUM_CLUSTERS, DIM);

        print!("Generating base vectors...");
        let base_vecs = generate_vectors(&mut rng, &centers, NUM_BASE, DIM, 1.0);
        println!(" done");

        print!("Generating query vectors...");
        let query_vecs = generate_vectors(&mut rng, &centers, NUM_QUERIES, DIM, 1.0);
        println!(" done");

        print!("Computing brute-force ground truth (k={GROUND_TRUTH_K})...");
        let gt_start = Instant::now();
        let ground_truth: Vec<Vec<(u64, f32)>> = query_vecs
            .iter()
            .map(|q| brute_force_knn(&base_vecs, q, GROUND_TRUTH_K, metric))
            .collect();
        let gt_time = gt_start.elapsed();
        println!(" done ({:.2}s)", gt_time.as_secs_f64());
        println!();

        (base_vecs, query_vecs, ground_truth)
    };

    // --- IVF-PQ on B-tree ---
    run_ivfpq_benchmark(&base_vecs, &query_vecs, &ground_truth, metric);

    println!();

    // --- IVF-PQ on Bf-Tree ---
    run_bftree_benchmark(&base_vecs, &query_vecs, &ground_truth, metric);
}

fn run_ivfpq_benchmark(
    base_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    ground_truth: &[Vec<(u64, f32)>],
    metric: DistanceMetric,
) {
    println!("--- IVF-PQ Index (B-tree) ---");

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = Database::create(tmp.path()).unwrap();

    let index_def = IvfPqIndexDefinition::new(
        "bench_ivfpq",
        DIM as u32,
        IVFPQ_CLUSTERS,
        IVFPQ_SUBVECTORS,
        metric,
    )
    .with_raw_vectors()
    .with_nprobe(10);

    // Build index
    let build_start = Instant::now();
    {
        let write_txn = db.begin_write().unwrap();
        let mut idx = write_txn.open_ivfpq_index(&index_def).unwrap();

        // Train on a subset
        let training_data: Vec<(u64, Vec<f32>)> = base_vecs
            .iter()
            .enumerate()
            .take(10_000)
            .map(|(i, v)| (i as u64, v.clone()))
            .collect();
        idx.train(training_data.into_iter(), KMEANS_ITERS).unwrap();

        // Insert all vectors
        let batch: Vec<(u64, Vec<f32>)> = base_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, v.clone()))
            .collect();
        idx.insert_batch(batch.into_iter()).unwrap();
        idx.flush().unwrap();
        drop(idx);
        write_txn.commit().unwrap();
    }
    let build_time = build_start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f64());

    // Search configs: (label, nprobe, k, rerank, candidates_override)
    let configs: Vec<(&str, u32, usize, bool, Option<usize>)> = vec![
        // PQ-only (no rerank) — shows raw scan performance
        ("PQ-only", 10, GROUND_TRUTH_K, false, None),
        // Reranked k=100 — full accuracy benchmark (default 10x candidates)
        ("rerank", 1, GROUND_TRUTH_K, true, None),
        ("rerank", 10, GROUND_TRUTH_K, true, None),
        ("rerank", 50, GROUND_TRUTH_K, true, None),
        // Reranked k=10 — realistic use case
        ("k10", 10, 10, true, None),       // default 100 candidates
        ("k10-200c", 10, 10, true, Some(200)),  // 200 candidates
        ("k10-500c", 10, 10, true, Some(500)),  // 500 candidates
    ];

    println!(
        "  {:>10} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "mode", "nprobe", "recall@1", "recall@10", "recall@100", "QPS", "p50(ms)", "p95(ms)", "p99(ms)"
    );

    for &(label, nprobe, k, rerank, cand_override) in &configs {
        let read_txn = db.begin_read().unwrap();
        let idx = read_txn.open_ivfpq_index(&index_def).unwrap();

        let mut params = SearchParams::top_k(k);
        params.nprobe = nprobe;
        params.rerank = rerank;
        if let Some(c) = cand_override {
            params.candidates = c;
        }

        let mut total_recall_1 = 0.0f64;
        let mut total_recall_10 = 0.0f64;
        let mut total_recall_100 = 0.0f64;
        let mut latencies_us: Vec<u64> = Vec::with_capacity(query_vecs.len());

        let search_start = Instant::now();
        for (qi, query) in query_vecs.iter().enumerate() {
            let q_start = Instant::now();
            let results = idx.search(&read_txn, query, &params).unwrap();
            latencies_us.push(q_start.elapsed().as_micros() as u64);

            let result_pairs: Vec<(u64, f32)> =
                results.iter().map(|n| (n.key, n.distance)).collect();

            total_recall_1 += compute_recall(&ground_truth[qi], &result_pairs, 1) as f64;
            total_recall_10 += compute_recall(&ground_truth[qi], &result_pairs, 10) as f64;
            total_recall_100 += compute_recall(&ground_truth[qi], &result_pairs, 100) as f64;
        }
        let search_time = search_start.elapsed();

        latencies_us.sort_unstable();
        let n = query_vecs.len();
        let p50 = latencies_us[n / 2];
        let p95 = latencies_us[n * 95 / 100];
        let p99 = latencies_us[n * 99 / 100];
        let qps = n as f64 / search_time.as_secs_f64();

        println!(
            "  {:>10} {:>6} {:>10.4} {:>10.4} {:>10.4} {:>10.0} {:>10.2} {:>10.2} {:>10.2}",
            label,
            nprobe,
            total_recall_1 / n as f64,
            total_recall_10 / n as f64,
            total_recall_100 / n as f64,
            qps,
            p50 as f64 / 1000.0,
            p95 as f64 / 1000.0,
            p99 as f64 / 1000.0,
        );
    }
}

fn run_bftree_benchmark(
    base_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    ground_truth: &[Vec<(u64, f32)>],
    metric: DistanceMetric,
) {
    println!("--- IVF-PQ Index (Bf-Tree) ---");

    // Configure Bf-Tree with default page sizes + blob indirection. Values above
    // blob_threshold are split into 1024B chunks in a system table, so even 150KB+
    // cluster blobs work with standard 4KB pages and 1568B max_record.
    let config = BfTreeConfig {
        circular_buffer_size: 256 * 1024 * 1024, // 256 MiB
        enable_wal: false,
        durability: DurabilityMode::NoSync,
        blob_threshold: 1024,
        ..BfTreeConfig::default()
    };

    let db = BfTreeDatabase::create(config).unwrap();

    // Use 1024 clusters (same as B-tree benchmark). Blob indirection handles
    // arbitrarily large cluster blobs transparently.
    let bf_clusters: u32 = 1024;
    let index_def = IvfPqIndexDefinition::new(
        "bench_ivfpq_bf",
        DIM as u32,
        bf_clusters,
        IVFPQ_SUBVECTORS,
        metric,
    )
    .with_raw_vectors()
    .with_nprobe(10);

    // Build index -- split into train + batched insert to stay within the
    // write buffer's 1M-entry limit.
    let build_start = Instant::now();
    {
        // Train in its own transaction
        let write_txn = db.begin_write();
        let mut idx = write_txn.open_ivfpq_index(&index_def).unwrap();
        let training_data: Vec<(u64, Vec<f32>)> = base_vecs
            .iter()
            .enumerate()
            .take(10_000)
            .map(|(i, v)| (i as u64, v.clone()))
            .collect();
        idx.train(training_data.into_iter(), KMEANS_ITERS).unwrap();
        idx.flush().unwrap();
        drop(idx);
        write_txn.commit().unwrap();
    }
    {
        // Insert vectors in batches of 500K to avoid write buffer overflow
        const BATCH_SIZE: usize = 500_000;
        for chunk_start in (0..base_vecs.len()).step_by(BATCH_SIZE) {
            let chunk_end = (chunk_start + BATCH_SIZE).min(base_vecs.len());
            let write_txn = db.begin_write();
            let mut idx = write_txn.open_ivfpq_index(&index_def).unwrap();
            let batch: Vec<(u64, Vec<f32>)> = base_vecs[chunk_start..chunk_end]
                .iter()
                .enumerate()
                .map(|(i, v)| ((chunk_start + i) as u64, v.clone()))
                .collect();
            idx.insert_batch(batch.into_iter()).unwrap();
            idx.flush().unwrap();
            drop(idx);
            write_txn.commit().unwrap();
        }
    }
    let build_time = build_start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f64());

    // Full benchmark configs matching B-tree: PQ-only, reranked, and k=10.
    let configs: Vec<(&str, u32, usize, bool, Option<usize>)> = vec![
        ("PQ-only", 10, GROUND_TRUTH_K, false, None),
        ("rerank", 1, GROUND_TRUTH_K, true, None),
        ("rerank", 10, GROUND_TRUTH_K, true, None),
        ("rerank", 50, GROUND_TRUTH_K, true, None),
        ("k10", 10, 10, true, None),
        ("k10-200c", 10, 10, true, Some(200)),
        ("k10-500c", 10, 10, true, Some(500)),
    ];

    println!(
        "  {:>10} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "mode", "nprobe", "recall@1", "recall@10", "recall@100", "QPS", "p50(ms)", "p95(ms)", "p99(ms)"
    );

    for &(label, nprobe, k, rerank, cand_override) in &configs {
        let read_txn = db.begin_read();
        let idx = read_txn.open_ivfpq_index(&index_def).unwrap();

        let mut params = SearchParams::top_k(k);
        params.nprobe = nprobe;
        params.rerank = rerank;
        if let Some(c) = cand_override {
            params.candidates = c;
        }

        let mut total_recall_1 = 0.0f64;
        let mut total_recall_10 = 0.0f64;
        let mut total_recall_100 = 0.0f64;
        let mut latencies_us: Vec<u64> = Vec::with_capacity(query_vecs.len());

        let search_start = Instant::now();
        for (qi, query) in query_vecs.iter().enumerate() {
            let q_start = Instant::now();
            let results = idx.search(&read_txn, query, &params).unwrap();
            latencies_us.push(q_start.elapsed().as_micros() as u64);

            let result_pairs: Vec<(u64, f32)> =
                results.iter().map(|n| (n.key, n.distance)).collect();

            total_recall_1 += compute_recall(&ground_truth[qi], &result_pairs, 1) as f64;
            total_recall_10 += compute_recall(&ground_truth[qi], &result_pairs, 10) as f64;
            total_recall_100 += compute_recall(&ground_truth[qi], &result_pairs, 100) as f64;
        }
        let search_time = search_start.elapsed();

        latencies_us.sort_unstable();
        let n = query_vecs.len();
        let p50 = latencies_us[n / 2];
        let p95 = latencies_us[n * 95 / 100];
        let p99 = latencies_us[n * 99 / 100];
        let qps = n as f64 / search_time.as_secs_f64();

        println!(
            "  {:>10} {:>6} {:>10.4} {:>10.4} {:>10.4} {:>10.0} {:>10.2} {:>10.2} {:>10.2}",
            label,
            nprobe,
            total_recall_1 / n as f64,
            total_recall_10 / n as f64,
            total_recall_100 / n as f64,
            qps,
            p50 as f64 / 1000.0,
            p95 as f64 / 1000.0,
            p99 as f64 / 1000.0,
        );
    }
}
