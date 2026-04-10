//! Recall benchmark for IVF-PQ and Fractal vector indexes.
//!
//! Generates a synthetic clustered dataset (Gaussian blobs), builds both index
//! types, and measures recall@1, recall@10, recall@100 against brute-force
//! ground truth. Also reports QPS and build time.
//!
//! Usage:
//!   cargo bench -p shodh-redb-bench --bench recall_benchmark
//!
//! Optional: set `SIFT1M_PATH` env var to a directory containing `sift_base.fvecs`
//! and `sift_query.fvecs` for real-dataset evaluation.

use std::collections::HashSet;
use std::time::Instant;

use shodh_redb::{
    Database, DistanceMetric, FractalIndexDefinition, FractalSearchParams, IvfPqIndexDefinition,
    ReadableDatabase, SearchParams,
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

// Fractal parameters
const FRACTAL_SUBVECTORS: u32 = 16;

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
    println!("=== Recall Benchmark ===");
    println!(
        "Base vectors: {NUM_BASE}, Queries: {NUM_QUERIES}, Dim: {DIM}, Clusters: {NUM_CLUSTERS}"
    );
    println!();

    // Generate data
    let mut rng = Rng::new(42);
    let centers = generate_cluster_centers(&mut rng, NUM_CLUSTERS, DIM);

    print!("Generating base vectors...");
    let base_vecs = generate_vectors(&mut rng, &centers, NUM_BASE, DIM, 1.0);
    println!(" done");

    print!("Generating query vectors...");
    let query_vecs = generate_vectors(&mut rng, &centers, NUM_QUERIES, DIM, 1.0);
    println!(" done");

    let metric = DistanceMetric::EuclideanSq;

    // Compute ground truth
    print!("Computing brute-force ground truth (k={GROUND_TRUTH_K})...");
    let gt_start = Instant::now();
    let ground_truth: Vec<Vec<(u64, f32)>> = query_vecs
        .iter()
        .map(|q| brute_force_knn(&base_vecs, q, GROUND_TRUTH_K, metric))
        .collect();
    let gt_time = gt_start.elapsed();
    println!(" done ({:.2}s)", gt_time.as_secs_f64());
    println!();

    // --- IVF-PQ ---
    run_ivfpq_benchmark(&base_vecs, &query_vecs, &ground_truth, metric);

    println!();

    // --- Fractal ---
    run_fractal_benchmark(&base_vecs, &query_vecs, &ground_truth, metric);
}

fn run_ivfpq_benchmark(
    base_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    ground_truth: &[Vec<(u64, f32)>],
    metric: DistanceMetric,
) {
    println!("--- IVF-PQ Index ---");

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

    // Search at different nprobe values
    let nprobe_values = [1, 10, 50];

    println!(
        "  {:>8} {:>10} {:>10} {:>10} {:>10}",
        "nprobe", "recall@1", "recall@10", "recall@100", "QPS"
    );

    for &nprobe in &nprobe_values {
        let read_txn = db.begin_read().unwrap();
        let idx = read_txn.open_ivfpq_index(&index_def).unwrap();

        let mut params = SearchParams::top_k(GROUND_TRUTH_K);
        params.nprobe = nprobe;

        let search_start = Instant::now();
        let mut total_recall_1 = 0.0f64;
        let mut total_recall_10 = 0.0f64;
        let mut total_recall_100 = 0.0f64;

        for (qi, query) in query_vecs.iter().enumerate() {
            let results = idx.search(&read_txn, query, &params).unwrap();
            let result_pairs: Vec<(u64, f32)> =
                results.iter().map(|n| (n.key, n.distance)).collect();

            total_recall_1 += compute_recall(&ground_truth[qi], &result_pairs, 1) as f64;
            total_recall_10 += compute_recall(&ground_truth[qi], &result_pairs, 10) as f64;
            total_recall_100 += compute_recall(&ground_truth[qi], &result_pairs, 100) as f64;
        }
        let search_time = search_start.elapsed();

        let n = query_vecs.len() as f64;
        let qps = n / search_time.as_secs_f64();

        println!(
            "  {:>8} {:>10.4} {:>10.4} {:>10.4} {:>10.0}",
            nprobe,
            total_recall_1 / n,
            total_recall_10 / n,
            total_recall_100 / n,
            qps,
        );
    }
}

fn run_fractal_benchmark(
    base_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    ground_truth: &[Vec<(u64, f32)>],
    metric: DistanceMetric,
) {
    println!("--- Fractal Index ---");

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = Database::create(tmp.path()).unwrap();

    let index_def = FractalIndexDefinition::new(
        "bench_fractal",
        DIM as u32,
        FRACTAL_SUBVECTORS,
        metric,
    )
    .with_raw_vectors()
    .with_nprobe(8);

    // Build index
    let build_start = Instant::now();
    {
        let write_txn = db.begin_write().unwrap();
        let mut idx = write_txn.open_fractal_index(&index_def).unwrap();

        // Train codebooks
        let training_data: Vec<(u64, Vec<f32>)> = base_vecs
            .iter()
            .enumerate()
            .take(10_000)
            .map(|(i, v)| (i as u64, v.clone()))
            .collect();
        idx.train_codebooks(training_data.into_iter(), KMEANS_ITERS)
            .unwrap();

        // Insert all vectors
        for (i, v) in base_vecs.iter().enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        idx.flush().unwrap();
        drop(idx);
        write_txn.commit().unwrap();
    }
    let build_time = build_start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f64());

    // Search at different nprobe values
    let nprobe_values = [1, 4, 8];

    println!(
        "  {:>8} {:>10} {:>10} {:>10} {:>10}",
        "nprobe", "recall@1", "recall@10", "recall@100", "QPS"
    );

    for &nprobe in &nprobe_values {
        let read_txn = db.begin_read().unwrap();
        let idx = read_txn.open_fractal_index(&index_def).unwrap();

        let params = FractalSearchParams::top_k(GROUND_TRUTH_K).with_nprobe(nprobe);

        let search_start = Instant::now();
        let mut total_recall_1 = 0.0f64;
        let mut total_recall_10 = 0.0f64;
        let mut total_recall_100 = 0.0f64;

        for (qi, query) in query_vecs.iter().enumerate() {
            let results = idx.search(&read_txn, query, &params).unwrap();
            let result_pairs: Vec<(u64, f32)> =
                results.iter().map(|n| (n.key, n.distance)).collect();

            total_recall_1 += compute_recall(&ground_truth[qi], &result_pairs, 1) as f64;
            total_recall_10 += compute_recall(&ground_truth[qi], &result_pairs, 10) as f64;
            total_recall_100 += compute_recall(&ground_truth[qi], &result_pairs, 100) as f64;
        }
        let search_time = search_start.elapsed();

        let n = query_vecs.len() as f64;
        let qps = n / search_time.as_secs_f64();

        println!(
            "  {:>8} {:>10.4} {:>10.4} {:>10.4} {:>10.0}",
            nprobe,
            total_recall_1 / n,
            total_recall_10 / n,
            total_recall_100 / n,
            qps,
        );
    }
}
