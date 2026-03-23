use shodh_redb::{
    Database, DistanceMetric, FractalIndexDefinition, FractalSearchParams, ReadableDatabase,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_tempfile() -> tempfile::NamedTempFile {
    if cfg!(target_os = "wasi") {
        tempfile::NamedTempFile::new_in("/tmp").unwrap()
    } else {
        tempfile::NamedTempFile::new().unwrap()
    }
}

/// Deterministic pseudo-random f32 vector from a seed.
fn random_vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
        })
        .collect()
}

/// Brute-force k-nearest neighbors.
fn brute_force_knn(
    query: &[f32],
    vectors: &[(u64, Vec<f32>)],
    k: usize,
    metric: DistanceMetric,
) -> Vec<u64> {
    let mut dists: Vec<(u64, f32)> = vectors
        .iter()
        .map(|(id, v)| (*id, metric.compute(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.into_iter().take(k).map(|(id, _)| id).collect()
}

// ---------------------------------------------------------------------------
// Index definitions
// ---------------------------------------------------------------------------

const FRACTAL_8D: FractalIndexDefinition = FractalIndexDefinition::new(
    "test_fractal_8d",
    8, // dim
    2, // subvectors (sub_dim = 4)
    DistanceMetric::EuclideanSq,
)
.with_raw_vectors()
.with_max_leaf_population(20)
.with_min_leaf_population(2)
.with_max_buffer_size(4);

const FRACTAL_32D: FractalIndexDefinition = FractalIndexDefinition::new(
    "test_fractal_32d",
    32, // dim
    4,  // subvectors (sub_dim = 8)
    DistanceMetric::EuclideanSq,
)
.with_raw_vectors()
.with_max_leaf_population(50)
.with_min_leaf_population(5)
.with_max_buffer_size(8);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn train_insert_search_basic() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let training: Vec<(u64, Vec<f32>)> = (0..30).map(|i| (i, random_vector(i + 100, 8))).collect();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&FRACTAL_8D).unwrap();
        idx.train_codebooks(training.iter().map(|(id, v)| (*id, v.clone())), 15)
            .unwrap();
    }
    write_txn.commit().unwrap();

    // Search from a read transaction
    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_fractal_index(&FRACTAL_8D).unwrap();

    let results = idx
        .search(&read_txn, &training[0].1, &FractalSearchParams::top_k(5))
        .unwrap();

    assert!(!results.is_empty(), "search should return results");
    // The query vector itself should be the top result (distance ~0)
    assert_eq!(
        results[0].key, 0,
        "closest result should be the query vector itself"
    );
    assert!(
        results[0].distance < 0.01,
        "self-distance should be near zero, got {}",
        results[0].distance
    );
}

#[test]
fn split_on_overflow() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Use small max_leaf_population to force splits
    let def = FractalIndexDefinition::new("split_test", 8, 2, DistanceMetric::EuclideanSq)
        .with_raw_vectors()
        .with_max_leaf_population(10)
        .with_min_leaf_population(2)
        .with_max_buffer_size(4);

    let training: Vec<(u64, Vec<f32>)> = (0..40).map(|i| (i, random_vector(i + 200, 8))).collect();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&def).unwrap();
        idx.train_codebooks(training.iter().map(|(id, v)| (*id, v.clone())), 10)
            .unwrap();

        let config = idx.config();
        assert!(
            config.num_clusters > 1,
            "with 40 vectors and max_leaf=10, tree should have split. clusters={}",
            config.num_clusters
        );
    }
    write_txn.commit().unwrap();
}

#[test]
fn buffer_cascade() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // max_buffer_size=4, so every 4 inserts should cascade
    let def = FractalIndexDefinition::new("buffer_test", 8, 2, DistanceMetric::EuclideanSq)
        .with_raw_vectors()
        .with_max_leaf_population(1000) // high to avoid splits
        .with_min_leaf_population(2)
        .with_max_buffer_size(4);

    // Train with initial data
    let training: Vec<(u64, Vec<f32>)> = (0..10).map(|i| (i, random_vector(i + 300, 8))).collect();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&def).unwrap();
        idx.train_codebooks(training.iter().map(|(id, v)| (*id, v.clone())), 10)
            .unwrap();

        // Insert more vectors beyond buffer threshold
        for i in 10..20 {
            idx.insert(i, &random_vector(i + 300, 8)).unwrap();
        }

        assert_eq!(idx.config().num_vectors, 20);
    }
    write_txn.commit().unwrap();

    // Verify search still works after cascade
    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_fractal_index(&def).unwrap();
    let results = idx
        .search(&read_txn, &training[0].1, &FractalSearchParams::top_k(5))
        .unwrap();
    assert!(!results.is_empty());
}

#[test]
fn persistence_reopen() {
    let tmpfile = create_tempfile();
    let path = tmpfile.path().to_path_buf();

    // Create and populate
    {
        let db = Database::create(&path).unwrap();
        let training: Vec<(u64, Vec<f32>)> =
            (0..20).map(|i| (i, random_vector(i + 400, 8))).collect();

        let write_txn = db.begin_write().unwrap();
        {
            let mut idx = write_txn.open_fractal_index(&FRACTAL_8D).unwrap();
            idx.train_codebooks(training.iter().map(|(id, v)| (*id, v.clone())), 10)
                .unwrap();
        }
        write_txn.commit().unwrap();
    }

    // Reopen and search
    {
        let db = Database::open(&path).unwrap();
        let read_txn = db.begin_read().unwrap();
        let idx = read_txn.open_fractal_index(&FRACTAL_8D).unwrap();

        let query = random_vector(400, 8); // same seed as vector 0
        let results = idx
            .search(&read_txn, &query, &FractalSearchParams::top_k(5))
            .unwrap();
        assert!(
            !results.is_empty(),
            "search should return results after reopen"
        );
    }
}

#[test]
fn upsert_semantics() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let training: Vec<(u64, Vec<f32>)> = (0..20).map(|i| (i, random_vector(i + 500, 8))).collect();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&FRACTAL_8D).unwrap();
        idx.train_codebooks(training.iter().map(|(id, v)| (*id, v.clone())), 10)
            .unwrap();

        // Insert vector_id=0 with a different vector (upsert)
        let new_vec = random_vector(9999, 8);
        idx.insert(0, &new_vec).unwrap();

        // Count should not increase (upsert replaces)
        assert_eq!(idx.config().num_vectors, 20);
    }
    write_txn.commit().unwrap();
}

#[test]
fn remove_vector() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let training: Vec<(u64, Vec<f32>)> = (0..20).map(|i| (i, random_vector(i + 600, 8))).collect();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&FRACTAL_8D).unwrap();
        idx.train_codebooks(training.iter().map(|(id, v)| (*id, v.clone())), 10)
            .unwrap();

        // Remove vector 5
        let removed = idx.remove(5).unwrap();
        assert!(removed, "vector 5 should have been found and removed");
        assert_eq!(idx.config().num_vectors, 19);

        // Removing again should return false
        let removed_again = idx.remove(5).unwrap();
        assert!(!removed_again, "vector 5 already removed");

        // Search: vector 5 should not appear in results
        let results = idx
            .search(&training[5].1, &FractalSearchParams::top_k(20))
            .unwrap();
        for r in &results {
            assert_ne!(
                r.key, 5,
                "removed vector should not appear in search results"
            );
        }
    }
    write_txn.commit().unwrap();
}

#[test]
fn multi_metric_euclidean() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let def = FractalIndexDefinition::new("metric_eucl", 8, 2, DistanceMetric::EuclideanSq)
        .with_raw_vectors()
        .with_max_leaf_population(100)
        .with_max_buffer_size(8);

    let training: Vec<(u64, Vec<f32>)> = (0..30).map(|i| (i, random_vector(i + 700, 8))).collect();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&def).unwrap();
        idx.train_codebooks(training.iter().map(|(id, v)| (*id, v.clone())), 10)
            .unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_fractal_index(&def).unwrap();
    let results = idx
        .search(&read_txn, &training[0].1, &FractalSearchParams::top_k(3))
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].key, 0);
}

#[test]
fn multi_metric_cosine() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let def = FractalIndexDefinition::new("metric_cos", 8, 2, DistanceMetric::Cosine)
        .with_raw_vectors()
        .with_max_leaf_population(100)
        .with_max_buffer_size(8);

    let training: Vec<(u64, Vec<f32>)> = (0..30).map(|i| (i, random_vector(i + 800, 8))).collect();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&def).unwrap();
        idx.train_codebooks(training.iter().map(|(id, v)| (*id, v.clone())), 10)
            .unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_fractal_index(&def).unwrap();
    let results = idx
        .search(&read_txn, &training[0].1, &FractalSearchParams::top_k(3))
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].key, 0);
}

#[test]
fn multi_metric_dotproduct() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let def = FractalIndexDefinition::new("metric_dot", 8, 2, DistanceMetric::DotProduct)
        .with_raw_vectors()
        .with_max_leaf_population(100)
        .with_max_buffer_size(8);

    let training: Vec<(u64, Vec<f32>)> = (0..30).map(|i| (i, random_vector(i + 900, 8))).collect();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&def).unwrap();
        idx.train_codebooks(training.iter().map(|(id, v)| (*id, v.clone())), 10)
            .unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_fractal_index(&def).unwrap();
    let results = idx
        .search(&read_txn, &training[0].1, &FractalSearchParams::top_k(3))
        .unwrap();
    assert!(!results.is_empty());
}

#[test]
fn multi_metric_manhattan() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let def = FractalIndexDefinition::new("metric_man", 8, 2, DistanceMetric::Manhattan)
        .with_raw_vectors()
        .with_max_leaf_population(100)
        .with_max_buffer_size(8);

    let training: Vec<(u64, Vec<f32>)> = (0..30).map(|i| (i, random_vector(i + 1000, 8))).collect();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&def).unwrap();
        idx.train_codebooks(training.iter().map(|(id, v)| (*id, v.clone())), 10)
            .unwrap();
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_fractal_index(&def).unwrap();
    let results = idx
        .search(&read_txn, &training[0].1, &FractalSearchParams::top_k(3))
        .unwrap();
    assert!(!results.is_empty());
}

#[test]
fn recall_benchmark_32d() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let n = 200;
    let dim = 32;
    let k = 10;
    let vectors: Vec<(u64, Vec<f32>)> = (0..n)
        .map(|i| (i as u64, random_vector(i as u64 + 1100, dim)))
        .collect();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&FRACTAL_32D).unwrap();
        idx.train_codebooks(vectors.iter().map(|(id, v)| (*id, v.clone())), 20)
            .unwrap();
    }
    write_txn.commit().unwrap();

    // Benchmark recall@10 across several queries
    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_fractal_index(&FRACTAL_32D).unwrap();

    let num_queries = 10;
    let mut total_recall = 0.0f64;

    for qi in 0..num_queries {
        let query = random_vector(qi as u64 + 2000, dim);
        let gt = brute_force_knn(&query, &vectors, k, DistanceMetric::EuclideanSq);
        let results = idx
            .search(
                &read_txn,
                &query,
                &FractalSearchParams {
                    nprobe: 16,
                    candidates: 100,
                    k,
                    rerank: true,
                    min_hlc: 0,
                },
            )
            .unwrap();

        let result_ids: Vec<u64> = results.iter().map(|r| r.key).collect();
        let hits = gt.iter().filter(|id| result_ids.contains(id)).count();
        total_recall += hits as f64 / k as f64;
    }

    let avg_recall = total_recall / num_queries as f64;
    assert!(
        avg_recall >= 0.5,
        "recall@{k} should be >= 0.5, got {avg_recall:.3} (200 vectors, 32d)"
    );
}

#[test]
fn insert_batch() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let training: Vec<(u64, Vec<f32>)> = (0..10).map(|i| (i, random_vector(i + 1200, 8))).collect();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&FRACTAL_8D).unwrap();
        idx.train_codebooks(training.iter().map(|(id, v)| (*id, v.clone())), 10)
            .unwrap();

        // Batch insert additional vectors
        let batch: Vec<(u64, Vec<f32>)> = (100..120)
            .map(|i| (i, random_vector(i + 1200, 8)))
            .collect();
        idx.insert_batch(batch.into_iter()).unwrap();

        assert_eq!(idx.config().num_vectors, 30); // 10 training + 20 batch
    }
    write_txn.commit().unwrap();
}

#[test]
fn empty_search_before_training() {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_fractal_index(&FRACTAL_8D).unwrap();
        // Search without training should error
        let result = idx.search(&random_vector(0, 8), &FractalSearchParams::top_k(5));
        assert!(result.is_err(), "search before training should fail");
    }
    // Don't commit -- just testing the error
}
