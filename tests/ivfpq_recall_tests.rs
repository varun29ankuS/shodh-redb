use shodh_redb::{
    Database, DistanceMetric, DiversityConfig, IvfPqIndexDefinition, MetadataFilter, MetadataMap,
    MetadataValue, ReadableDatabase, SearchParams,
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

fn random_vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = seed;
    for _ in 0..dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0);
    }
    v
}

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

fn recall(result_ids: &[u64], ground_truth: &[u64]) -> f64 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let hits = ground_truth
        .iter()
        .filter(|id| result_ids.contains(id))
        .count();
    hits as f64 / ground_truth.len() as f64
}

// ---------------------------------------------------------------------------
// Index definitions
// ---------------------------------------------------------------------------

const INDEX_8D_EUC: IvfPqIndexDefinition =
    IvfPqIndexDefinition::new("recall_8d_euc", 8, 4, 2, DistanceMetric::EuclideanSq)
        .with_raw_vectors()
        .with_nprobe(4);

const INDEX_8D_COS: IvfPqIndexDefinition =
    IvfPqIndexDefinition::new("recall_8d_cos", 8, 4, 2, DistanceMetric::Cosine)
        .with_raw_vectors()
        .with_nprobe(4);

const INDEX_8D_DOT: IvfPqIndexDefinition =
    IvfPqIndexDefinition::new("recall_8d_dot", 8, 4, 2, DistanceMetric::DotProduct)
        .with_raw_vectors()
        .with_nprobe(4);

const INDEX_8D_MAN: IvfPqIndexDefinition =
    IvfPqIndexDefinition::new("recall_8d_man", 8, 4, 2, DistanceMetric::Manhattan)
        .with_raw_vectors()
        .with_nprobe(4);

const INDEX_32D_EUC: IvfPqIndexDefinition =
    IvfPqIndexDefinition::new("recall_32d_euc", 32, 16, 8, DistanceMetric::EuclideanSq)
        .with_raw_vectors()
        .with_nprobe(16);

const INDEX_8D_META: IvfPqIndexDefinition =
    IvfPqIndexDefinition::new("recall_8d_meta", 8, 4, 2, DistanceMetric::EuclideanSq)
        .with_raw_vectors()
        .with_nprobe(4);

// ---------------------------------------------------------------------------
// Shared setup: generate vectors, create DB, train, insert, commit
// ---------------------------------------------------------------------------

fn setup_index(
    def: &IvfPqIndexDefinition,
    vectors: &[(u64, Vec<f32>)],
) -> (tempfile::NamedTempFile, Database) {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_ivfpq_index(def).unwrap();
        idx.train(vectors.iter().map(|(id, v)| (*id, v.clone())), 25)
            .unwrap();
        for (id, vec) in vectors {
            idx.insert(*id, vec).unwrap();
        }
    }
    write_txn.commit().unwrap();
    (tmpfile, db)
}

fn make_vectors(n: u64, dim: usize, seed_offset: u64) -> Vec<(u64, Vec<f32>)> {
    (0..n)
        .map(|i| (i, random_vector(i + seed_offset, dim)))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Train 200 vectors in 8D with Euclidean distance, search top-5, verify
/// recall >= 0.6 averaged over multiple queries.
#[test]
fn train_and_search_euclidean() {
    let vectors = make_vectors(200, 8, 10000);
    let (_tmp, db) = setup_index(&INDEX_8D_EUC, &vectors);

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_8D_EUC).unwrap();

    let k = 5;
    let num_queries = 20;
    let mut total_recall = 0.0f64;

    for q in 0..num_queries {
        let query = &vectors[q * 10].1;
        let gt = brute_force_knn(query, &vectors, k, DistanceMetric::EuclideanSq);
        let results = idx
            .search(&read_txn, query, &SearchParams::top_k(k))
            .unwrap();
        let ids: Vec<u64> = results.iter().map(|r| r.key).collect();
        total_recall += recall(&ids, &gt);
    }

    let avg = total_recall / num_queries as f64;
    assert!(
        avg >= 0.6,
        "euclidean recall@{k} = {avg:.3}, expected >= 0.6"
    );
}

/// Train 200 vectors in 8D with cosine distance, search top-5, verify
/// recall >= 0.6.
#[test]
fn train_and_search_cosine() {
    let vectors = make_vectors(200, 8, 20000);
    let (_tmp, db) = setup_index(&INDEX_8D_COS, &vectors);

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_8D_COS).unwrap();

    let k = 5;
    let num_queries = 20;
    let mut total_recall = 0.0f64;

    for q in 0..num_queries {
        let query = &vectors[q * 10].1;
        let gt = brute_force_knn(query, &vectors, k, DistanceMetric::Cosine);
        let results = idx
            .search(&read_txn, query, &SearchParams::top_k(k))
            .unwrap();
        let ids: Vec<u64> = results.iter().map(|r| r.key).collect();
        total_recall += recall(&ids, &gt);
    }

    let avg = total_recall / num_queries as f64;
    assert!(avg >= 0.6, "cosine recall@{k} = {avg:.3}, expected >= 0.6");
}

/// Train 200 vectors in 8D with dot product distance, search top-5, verify
/// recall >= 0.6.
#[test]
fn train_and_search_dot_product() {
    let vectors = make_vectors(200, 8, 30000);
    let (_tmp, db) = setup_index(&INDEX_8D_DOT, &vectors);

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_8D_DOT).unwrap();

    let k = 5;
    let num_queries = 20;
    let mut total_recall = 0.0f64;

    for q in 0..num_queries {
        let query = &vectors[q * 10].1;
        let gt = brute_force_knn(query, &vectors, k, DistanceMetric::DotProduct);
        let results = idx
            .search(&read_txn, query, &SearchParams::top_k(k))
            .unwrap();
        let ids: Vec<u64> = results.iter().map(|r| r.key).collect();
        total_recall += recall(&ids, &gt);
    }

    let avg = total_recall / num_queries as f64;
    assert!(
        avg >= 0.6,
        "dot_product recall@{k} = {avg:.3}, expected >= 0.6"
    );
}

/// Train 200 vectors in 8D with Manhattan distance, search top-5, verify
/// recall >= 0.6.
#[test]
fn train_and_search_manhattan() {
    let vectors = make_vectors(200, 8, 40000);
    let (_tmp, db) = setup_index(&INDEX_8D_MAN, &vectors);

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_8D_MAN).unwrap();

    let k = 5;
    let num_queries = 20;
    let mut total_recall = 0.0f64;

    for q in 0..num_queries {
        let query = &vectors[q * 10].1;
        let gt = brute_force_knn(query, &vectors, k, DistanceMetric::Manhattan);
        let results = idx
            .search(&read_txn, query, &SearchParams::top_k(k))
            .unwrap();
        let ids: Vec<u64> = results.iter().map(|r| r.key).collect();
        total_recall += recall(&ids, &gt);
    }

    let avg = total_recall / num_queries as f64;
    assert!(
        avg >= 0.6,
        "manhattan recall@{k} = {avg:.3}, expected >= 0.6"
    );
}

/// Verify that reranking with raw vectors produces recall >= the PQ-only path.
/// Index is built with_raw_vectors. We compare rerank=true vs rerank=false
/// across multiple queries, and assert that reranking is at least as good.
#[test]
fn rerank_improves_recall() {
    let vectors = make_vectors(200, 8, 50000);
    let (_tmp, db) = setup_index(&INDEX_8D_EUC, &vectors);

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_8D_EUC).unwrap();

    let k = 5;
    let num_queries = 20;
    let mut rerank_total = 0.0f64;
    let mut pq_total = 0.0f64;

    for q in 0..num_queries {
        let query = &vectors[q * 10].1;
        let gt = brute_force_knn(query, &vectors, k, DistanceMetric::EuclideanSq);

        let params_rerank = SearchParams {
            nprobe: 4,
            candidates: k * 10,
            k,
            rerank: true,
            diversity: DiversityConfig { lambda: 0.0 },
            filter: None,
        };
        let results_rerank = idx.search(&read_txn, query, &params_rerank).unwrap();
        let ids_rerank: Vec<u64> = results_rerank.iter().map(|r| r.key).collect();
        rerank_total += recall(&ids_rerank, &gt);

        let params_pq = SearchParams {
            nprobe: 4,
            candidates: k * 10,
            k,
            rerank: false,
            diversity: DiversityConfig { lambda: 0.0 },
            filter: None,
        };
        let results_pq = idx.search(&read_txn, query, &params_pq).unwrap();
        let ids_pq: Vec<u64> = results_pq.iter().map(|r| r.key).collect();
        pq_total += recall(&ids_pq, &gt);
    }

    let rerank_avg = rerank_total / num_queries as f64;
    let pq_avg = pq_total / num_queries as f64;
    assert!(
        rerank_avg >= pq_avg,
        "rerank recall ({rerank_avg:.3}) should be >= pq-only recall ({pq_avg:.3})"
    );
}

/// Verify that probing more clusters yields better or equal recall.
/// Compare nprobe=1 against nprobe=num_clusters (exhaustive).
#[test]
fn nprobe_affects_recall() {
    let vectors = make_vectors(200, 8, 60000);
    let (_tmp, db) = setup_index(&INDEX_8D_EUC, &vectors);

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_8D_EUC).unwrap();
    let num_clusters = idx.config().num_clusters;

    let k = 5;
    let num_queries = 20;
    let mut recall_low = 0.0f64;
    let mut recall_high = 0.0f64;

    for q in 0..num_queries {
        let query = &vectors[q * 10].1;
        let gt = brute_force_knn(query, &vectors, k, DistanceMetric::EuclideanSq);

        let params_low = SearchParams {
            nprobe: 1,
            candidates: k * 10,
            k,
            rerank: true,
            diversity: DiversityConfig { lambda: 0.0 },
            filter: None,
        };
        let results_low = idx.search(&read_txn, query, &params_low).unwrap();
        let ids_low: Vec<u64> = results_low.iter().map(|r| r.key).collect();
        recall_low += recall(&ids_low, &gt);

        let params_high = SearchParams {
            nprobe: num_clusters,
            candidates: k * 10,
            k,
            rerank: true,
            diversity: DiversityConfig { lambda: 0.0 },
            filter: None,
        };
        let results_high = idx.search(&read_txn, query, &params_high).unwrap();
        let ids_high: Vec<u64> = results_high.iter().map(|r| r.key).collect();
        recall_high += recall(&ids_high, &gt);
    }

    let avg_low = recall_low / num_queries as f64;
    let avg_high = recall_high / num_queries as f64;
    assert!(
        avg_high >= avg_low,
        "nprobe=all ({avg_high:.3}) should have recall >= nprobe=1 ({avg_low:.3})"
    );
}

/// Train on a subset, then insert additional vectors afterward, and verify
/// that search can find the newly inserted vectors.
#[test]
fn insert_after_train() {
    let initial = make_vectors(100, 8, 70000);
    let extra = make_vectors(50, 8, 80000)
        .into_iter()
        .map(|(id, v)| (id + 1000, v))
        .collect::<Vec<_>>();

    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    // Train on initial set only
    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_ivfpq_index(&INDEX_8D_EUC).unwrap();
        idx.train(initial.iter().map(|(id, v)| (*id, v.clone())), 25)
            .unwrap();
        for (id, vec) in &initial {
            idx.insert(*id, vec).unwrap();
        }
    }
    write_txn.commit().unwrap();

    // Insert extra vectors in a new transaction
    let write_txn2 = db.begin_write().unwrap();
    {
        let mut idx = write_txn2.open_ivfpq_index(&INDEX_8D_EUC).unwrap();
        for (id, vec) in &extra {
            idx.insert(*id, vec).unwrap();
        }
    }
    write_txn2.commit().unwrap();

    // Search for one of the extra vectors
    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_8D_EUC).unwrap();
    assert_eq!(idx.config().num_vectors, 150);

    let query = &extra[25].1;
    let results = idx
        .search(&read_txn, query, &SearchParams::top_k(5))
        .unwrap();
    let ids: Vec<u64> = results.iter().map(|r| r.key).collect();
    assert!(
        ids.contains(&extra[25].0),
        "expected to find inserted vector {} in results {:?}",
        extra[25].0,
        ids,
    );
}

/// Insert vectors, remove one, and verify search no longer returns it.
#[test]
fn remove_vector() {
    let vectors = make_vectors(50, 8, 90000);
    let (_tmp, db) = setup_index(&INDEX_8D_EUC, &vectors);

    let target_id = 25u64;
    let target_vec = vectors[target_id as usize].1.clone();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_ivfpq_index(&INDEX_8D_EUC).unwrap();
        let removed = idx.remove(target_id).unwrap();
        assert!(removed, "remove should return true for existing vector");
        assert_eq!(idx.config().num_vectors, 49);
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_8D_EUC).unwrap();
    let results = idx
        .search(&read_txn, &target_vec, &SearchParams::top_k(50))
        .unwrap();
    for r in &results {
        assert_ne!(
            r.key, target_id,
            "removed vector {target_id} should not appear in results"
        );
    }
}

/// Train the index but insert zero vectors, then search. Should return empty.
#[test]
fn search_empty_index() {
    let training = make_vectors(20, 8, 100000);
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_ivfpq_index(&INDEX_8D_EUC).unwrap();
        idx.train(training.iter().map(|(id, v)| (*id, v.clone())), 25)
            .unwrap();
        // Deliberately do not insert any vectors.
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_8D_EUC).unwrap();
    assert_eq!(idx.config().num_vectors, 0);

    let query = random_vector(999999, 8);
    let results = idx
        .search(&read_txn, &query, &SearchParams::top_k(5))
        .unwrap();
    assert!(
        results.is_empty(),
        "search on empty index should return no results, got {}",
        results.len()
    );
}

/// Verify the single closest vector is the correct brute-force nearest
/// neighbor.
#[test]
fn search_top_1() {
    let vectors = make_vectors(200, 8, 110000);
    let (_tmp, db) = setup_index(&INDEX_8D_EUC, &vectors);

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_8D_EUC).unwrap();

    let num_queries = 20;
    let mut correct = 0usize;

    for q in 0..num_queries {
        let query = &vectors[q * 10].1;
        let gt = brute_force_knn(query, &vectors, 1, DistanceMetric::EuclideanSq);
        let results = idx
            .search(&read_txn, query, &SearchParams::top_k(1))
            .unwrap();
        assert_eq!(results.len(), 1);
        if results[0].key == gt[0] {
            correct += 1;
        }
    }

    // With reranking and nprobe=all_clusters on 200 vectors, top-1 accuracy
    // should be high. We use a generous threshold.
    assert!(
        correct >= num_queries / 2,
        "top-1 accuracy = {correct}/{num_queries}, expected at least 50%"
    );
}

/// 1000 vectors in 32D, verify recall@10 >= 0.5 with reranking.
#[test]
fn large_dataset_recall() {
    let vectors = make_vectors(1000, 32, 120000);
    let (_tmp, db) = setup_index(&INDEX_32D_EUC, &vectors);

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_32D_EUC).unwrap();
    assert_eq!(idx.config().num_vectors, 1000);

    let k = 10;
    let num_queries = 30;
    let mut total_recall = 0.0f64;

    for q in 0..num_queries {
        let query = &vectors[q * 33].1;
        let gt = brute_force_knn(query, &vectors, k, DistanceMetric::EuclideanSq);

        let params = SearchParams {
            nprobe: 16,
            candidates: 200,
            k,
            rerank: true,
            diversity: DiversityConfig { lambda: 0.0 },
            filter: None,
        };
        let results = idx.search(&read_txn, query, &params).unwrap();
        let ids: Vec<u64> = results.iter().map(|r| r.key).collect();
        total_recall += recall(&ids, &gt);
    }

    let avg = total_recall / num_queries as f64;
    assert!(
        avg >= 0.5,
        "large dataset recall@{k} = {avg:.3}, expected >= 0.5"
    );
}

/// Insert vectors with metadata, search with a filter, and verify that only
/// matching vectors appear in results.
#[test]
fn metadata_filter_search() {
    let vectors = make_vectors(100, 8, 130000);
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();

    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_ivfpq_index(&INDEX_8D_META).unwrap();
        idx.train(vectors.iter().map(|(id, v)| (*id, v.clone())), 25)
            .unwrap();
        for (id, vec) in &vectors {
            idx.insert(*id, vec).unwrap();

            // Even IDs get category "alpha", odd IDs get "beta".
            let mut meta = MetadataMap::new();
            if id % 2 == 0 {
                meta.insert("category", MetadataValue::String("alpha".into()));
            } else {
                meta.insert("category", MetadataValue::String("beta".into()));
            }
            meta.insert("score", MetadataValue::U64(*id));
            idx.insert_metadata(*id, &meta).unwrap();
        }
    }
    write_txn.commit().unwrap();

    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(&INDEX_8D_META).unwrap();

    // Search with filter: category == "alpha" (even IDs only)
    let filter = MetadataFilter::Eq("category".into(), MetadataValue::String("alpha".into()));
    let params = SearchParams::top_k(10).with_filter(filter);
    let query = &vectors[0].1;
    let results = idx.search(&read_txn, query, &params).unwrap();

    assert!(
        !results.is_empty(),
        "filtered search should return at least one result"
    );
    for r in &results {
        assert_eq!(
            r.key % 2,
            0,
            "filtered result id {} should be even (category=alpha)",
            r.key
        );
    }

    // Search with filter: score > 50 (IDs 51..99)
    let filter_gt = MetadataFilter::Gt("score".into(), MetadataValue::U64(50));
    let params_gt = SearchParams::top_k(10).with_filter(filter_gt);
    let results_gt = idx.search(&read_txn, query, &params_gt).unwrap();

    for r in &results_gt {
        assert!(r.key > 50, "filtered result id {} should be > 50", r.key);
    }
}
