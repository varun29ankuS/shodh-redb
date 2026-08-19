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

/// Smoke test, not a recall gate. 200 vectors at dim 8 with
/// `nprobe == num_clusters`, raw vectors stored and re-ranking on, so it
/// probes everything and repairs quantization error exactly: it measures
/// 1.000 and cannot fail. It verifies that train-then-search works end to
/// end for this metric. The real gate is `gate_recall_euclidean`.
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
        avg >= 0.95,
        "euclidean recall@{k} = {avg:.3}, expected >= 0.95 (measured 1.000)"
    );
}

/// Smoke test, not a recall gate -- see `train_and_search_euclidean`.
/// The real gate is `gate_recall_cosine`.
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
    assert!(
        avg >= 0.95,
        "cosine recall@{k} = {avg:.3}, expected >= 0.95 (measured 1.000)"
    );
}

/// Smoke test, not a recall gate -- see `train_and_search_euclidean`.
/// The real gate is `gate_recall_dot_product`.
///
/// The 0.760 in the assertion message below was measured BEFORE the
/// inner-product residual fix, when this metric was returning near-random
/// results at realistic settings. It survived here only because this test
/// probes every cluster and re-ranks a shortlist covering half the corpus.
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
        avg >= 0.70,
        "dot_product recall@{k} = {avg:.3}, expected >= 0.70 (measured 0.760)"
    );
}

/// Smoke test, not a recall gate -- see `train_and_search_euclidean`.
/// The real gate is `gate_recall_manhattan`.
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
        avg >= 0.95,
        "manhattan recall@{k} = {avg:.3}, expected >= 0.95 (measured 1.000)"
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
        avg >= 0.95,
        "large dataset recall@{k} = {avg:.3}, expected >= 0.95 (measured 1.000)"
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

// ---------------------------------------------------------------------------
// Discriminating recall harness
//
// The tests above run at dim 8 with 4 clusters and nprobe 4. Since nprobe
// equals the cluster count they probe every cluster, and they store raw
// vectors and re-rank, so quantization error is corrected away. They measure
// recall 1.000 and cannot fail: they are end-to-end smoke tests, not a recall
// gate. The harness below is the gate.
//
// Three things make it discriminating:
//   1. nprobe << num_clusters, so IVF pruning can actually lose neighbours.
//   2. No raw vectors stored at all, so PQ is the only thing standing between
//      the query and the answer. `rerank: false` alone is not enough -- an
//      index that stores raw vectors still pays for them.
//   3. Clustered data. Uniform-random vectors are the wrong input: in high
//      dimensions every point is near-equidistant from every other, so recall
//      collapses and turns noisy, which makes a useless gate.
// ---------------------------------------------------------------------------

/// Deterministic PRNG. Reproducibility is the whole point: a recall threshold
/// is only meaningful if the data behind it is identical on every run and
/// every platform.
struct Prng(u64);

impl Prng {
    fn new(seed: u64) -> Self {
        // Any nonzero state works; 0 would stick at 0 for a pure LCG.
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        // Take the high bits: the low bits of an LCG have short periods.
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Standard normal via Box-Muller. Only the first of the pair is used;
    /// the second is discarded to keep the call site simple.
    fn next_normal(&mut self) -> f32 {
        // Clamp away from 0 so ln() stays finite.
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
    }
}

/// Gaussian mixture: `num_centers` latent clusters drawn uniformly from
/// [-1, 1]^dim, with each vector drawn from one center plus isotropic noise of
/// standard deviation `spread`.
///
/// `spread` is the knob that sets how hard the search is. Too small and every
/// true neighbour of a query lands in the same IVF cluster, so recall is 1.000
/// whatever the quantizer does. Too large and the mixture degenerates towards
/// uniform, where high-dimensional distances concentrate and recall becomes
/// noise. `num_centers` is deliberately larger than the index's cluster count
/// so the IVF partition cannot line up exactly with the latent structure --
/// that mismatch is what real data looks like.
fn gaussian_mixture(
    n: u64,
    dim: usize,
    num_centers: usize,
    spread: f32,
    decay: f32,
    seed: u64,
) -> Vec<(u64, Vec<f32>)> {
    let mut rng = Prng::new(seed);

    // Per-dimension scale following a power law, which is what the spectrum of
    // a real embedding corpus looks like: a few directions carry most of the
    // variance and the tail is near-flat. `decay = 0` gives isotropic noise --
    // full intrinsic dimensionality, which is the pathological case for product
    // quantization, since PQ earns its accuracy from correlated structure.
    // Benchmarking only that case understates the index against real data.
    let scale: Vec<f32> = (0..dim)
        .map(|d| ((d + 1) as f32).powf(-decay / 2.0))
        .collect();

    let mut centers = Vec::with_capacity(num_centers);
    for _ in 0..num_centers {
        let c: Vec<f32> = (0..dim)
            .map(|d| (rng.next_f32() * 2.0 - 1.0) * scale[d])
            .collect();
        centers.push(c);
    }

    (0..n)
        .map(|id| {
            let center = &centers[(rng.next_u64() as usize) % num_centers];
            let v = (0..dim)
                .map(|d| center[d] + rng.next_normal() * spread * scale[d])
                .collect();
            (id, v)
        })
        .collect()
}

/// Fast tier geometry. dim 128 with 16 sub-quantizers gives 8 floats per
/// sub-vector, a normal production shape, and compresses 512 bytes of f32 down
/// to 16 bytes. nprobe 4 of 64 clusters means a query sees one sixteenth of the
/// index. No `with_raw_vectors`: the quantizer gets no safety net.
const FAST_DIM: usize = 128;
const FAST_CLUSTERS: u32 = 256;
const FAST_SUBVECTORS: u32 = 32;
const FAST_NPROBE: u32 = 4;
const FAST_N: u64 = 20_000;
/// Latent clusters in the generated data. Deliberately above `FAST_CLUSTERS`
/// so the IVF partition cannot align exactly with the data's own structure.
const FAST_CENTERS: usize = 1;
const FAST_TRAIN_ITERS: usize = 10;

/// Build an index that stores no raw vectors, training on a subsample.
fn setup_pq_only_index(
    def: &IvfPqIndexDefinition,
    vectors: &[(u64, Vec<f32>)],
    train_sample: usize,
    train_iters: usize,
) -> (tempfile::NamedTempFile, Database) {
    let tmpfile = create_tempfile();
    let db = Database::create(tmpfile.path()).unwrap();
    let write_txn = db.begin_write().unwrap();
    {
        let mut idx = write_txn.open_ivfpq_index(def).unwrap();
        idx.train(
            vectors
                .iter()
                .take(train_sample)
                .map(|(id, v)| (*id, v.clone())),
            train_iters,
        )
        .unwrap();
        for (id, vec) in vectors {
            idx.insert(*id, vec).unwrap();
        }
    }
    write_txn.commit().unwrap();
    (tmpfile, db)
}

/// Search params with re-ranking off, so recall reflects PQ plus IVF alone.
fn pq_only_params(k: usize, nprobe: u32) -> SearchParams {
    SearchParams {
        nprobe,
        candidates: k * 10,
        k,
        rerank: false,
        diversity: DiversityConfig { lambda: 0.0 },
        filter: None,
    }
}

/// Mean recall@k over `num_queries` queries drawn from the corpus itself.
fn measure_recall(
    db: &Database,
    def: &IvfPqIndexDefinition,
    vectors: &[(u64, Vec<f32>)],
    metric: DistanceMetric,
    k: usize,
    nprobe: u32,
    num_queries: usize,
) -> f64 {
    let read_txn = db.begin_read().unwrap();
    let idx = read_txn.open_ivfpq_index(def).unwrap();
    let stride = vectors.len() / num_queries;
    let mut total = 0.0f64;
    for q in 0..num_queries {
        let query = &vectors[q * stride].1;
        let gt = brute_force_knn(query, vectors, k, metric);
        let results = idx
            .search(&read_txn, query, &pq_only_params(k, nprobe))
            .unwrap();
        let ids: Vec<u64> = results.iter().map(|r| r.key).collect();
        total += recall(&ids, &gt);
    }
    total / num_queries as f64
}

// ---------------------------------------------------------------------------
// Fast tier: the recall gate
//
// Every test below is `#[ignore]`d and run by a dedicated release-mode CI step:
//
//     cargo test --release --test ivfpq_recall_tests -- --ignored --nocapture
//
// Not because they are optional, but because debug builds are roughly 21x
// slower on this scalar float path (151s vs 7s to build one index at this
// geometry). Left un-ignored they would run inside every debug test job --
// about six of them -- and add roughly ten minutes to each. One release-mode
// step gives the same signal in about thirty seconds total.
//
// Thresholds are the minimum across five seeds, less a deliberately generous
// margin. At recall 1.000 cross-platform float drift is invisible; in this band
// it is not (FMA and autovectorisation differences change k-means convergence)
// and the four-platform spread has never been measured. Tighten these once CI
// data exists across all four platforms. Never loosen them.
// ---------------------------------------------------------------------------

/// The seed whose measurements set the thresholds below.
const GATE_SEED: u64 = 1;

fn gate_recall(label: &'static str, metric: DistanceMetric, subvectors: u32, nprobe: u32) -> f64 {
    let vectors = gaussian_mixture(FAST_N, FAST_DIM, FAST_CENTERS, 0.5, 1.0, GATE_SEED);
    let def = IvfPqIndexDefinition::new(label, FAST_DIM as u32, FAST_CLUSTERS, subvectors, metric)
        .with_nprobe(nprobe);
    let (_tmp, db) = setup_pq_only_index(&def, &vectors, FAST_N as usize, FAST_TRAIN_ITERS);
    measure_recall(&db, &def, &vectors, metric, 10, nprobe, 20)
}

/// Euclidean is the reference metric: its residual path is mathematically
/// exact, so it sets the level the other three are judged against.
#[test]
#[ignore]
fn gate_recall_euclidean() {
    let r = gate_recall(
        "gate_euc",
        DistanceMetric::EuclideanSq,
        FAST_SUBVECTORS,
        FAST_NPROBE,
    );
    assert!(
        r >= 0.35,
        "euclidean recall@10 = {r:.4}, expected >= 0.35 (5-seed min 0.405)"
    );
}

/// Cosine and dot product both run through the negated dot-product
/// accumulation. Before the inner-product fix these measured 0.278 and 0.037
/// -- dot_product barely above the 0.0005 chance rate -- because search built
/// the ADC table from `q - c`, which is not distance-preserving for inner
/// product. These thresholds fail loudly if that ever regresses.
#[test]
#[ignore]
fn gate_recall_cosine() {
    let r = gate_recall(
        "gate_cos",
        DistanceMetric::Cosine,
        FAST_SUBVECTORS,
        FAST_NPROBE,
    );
    assert!(
        r >= 0.25,
        "cosine recall@10 = {r:.4}, expected >= 0.25 (5-seed min 0.310, was 0.278 pre-fix)"
    );
}

#[test]
#[ignore]
fn gate_recall_dot_product() {
    let r = gate_recall(
        "gate_dot",
        DistanceMetric::DotProduct,
        FAST_SUBVECTORS,
        FAST_NPROBE,
    );
    assert!(
        r >= 0.25,
        "dot_product recall@10 = {r:.4}, expected >= 0.25 (5-seed min 0.325, was 0.037 pre-fix)"
    );
}

#[test]
#[ignore]
fn gate_recall_manhattan() {
    let r = gate_recall(
        "gate_man",
        DistanceMetric::Manhattan,
        FAST_SUBVECTORS,
        FAST_NPROBE,
    );
    assert!(
        r >= 0.25,
        "manhattan recall@10 = {r:.4}, expected >= 0.25 (5-seed min 0.310)"
    );
}

// ---------------------------------------------------------------------------
// Monotonicity invariants
//
// Stronger gates than the absolute thresholds above, because they do not depend
// on a calibrated level and so cannot drift with the platform. Both are
// properties the algorithm must satisfy on any machine: probing more clusters
// cannot lose a neighbour that fewer clusters found, and spending more bits per
// vector cannot make the quantizer less accurate.
//
// Measured gaps are wide -- 0.38 to 0.645 across nprobe 1 to 16, and 0.41 to
// 0.70 across m 16 to 32 -- so these are asserted strictly. A gap that size
// cannot be flipped by float noise; a failure here means something is broken.
// ---------------------------------------------------------------------------

/// More probes cannot mean fewer neighbours found. This is the assertion that
/// actually exercises IVF rather than PQ, and it only has teeth because the
/// corpus is a continuum (`FAST_CENTERS = 1`): with one latent blob per IVF
/// cluster the partition lands on the blobs, every query's neighbours sit in
/// its own cell, and nprobe stops mattering at all.
#[test]
#[ignore]
fn recall_is_monotonic_in_nprobe() {
    let low = gate_recall("mono_p_lo", DistanceMetric::EuclideanSq, FAST_SUBVECTORS, 1);
    let high = gate_recall(
        "mono_p_hi",
        DistanceMetric::EuclideanSq,
        FAST_SUBVECTORS,
        16,
    );
    assert!(
        high >= low,
        "recall must not fall as nprobe rises: nprobe=1 gave {low:.4}, nprobe=16 gave {high:.4}"
    );
}

/// More bits per vector cannot mean a worse quantizer. Fails if codebook
/// training or the ADC accumulation mishandles the sub-vector count.
#[test]
#[ignore]
fn recall_is_monotonic_in_bit_budget() {
    let low = gate_recall("mono_m_lo", DistanceMetric::EuclideanSq, 16, FAST_NPROBE);
    let high = gate_recall("mono_m_hi", DistanceMetric::EuclideanSq, 32, FAST_NPROBE);
    assert!(
        high >= low,
        "recall must not fall as the bit budget rises: m=16 gave {low:.4}, m=32 gave {high:.4}"
    );
}

// ---------------------------------------------------------------------------
// Deep tier: realistic geometry, run by hand
//
// dim 384 is a real embedding width (all-MiniLM-L6-v2). 512 clusters over 50k
// vectors is about 2.3x sqrt(n), inside the usual 1-4x range and proportionally
// close to BigANN's 16384 lists over 100M vectors. k-means trains on a
// subsample because full-corpus training dominates the cost for no accuracy
// gain; production systems subsample for the same reason.
//
// Deliberately not wired into CI -- it is the measuring stick for changes to the
// quantization path, not a gate. Run it when touching that path:
//
//     cargo test --release --test ivfpq_recall_tests deep_ -- --ignored --nocapture
// ---------------------------------------------------------------------------

const DEEP_DIM: usize = 384;
const DEEP_N: u64 = 50_000;
const DEEP_CLUSTERS: u32 = 512;
const DEEP_SUBVECTORS: u32 = 48;
const DEEP_NPROBE: u32 = 16;
const DEEP_TRAIN_SAMPLE: usize = 25_000;

#[test]
#[ignore]
fn deep_recall_table() {
    let vectors = gaussian_mixture(DEEP_N, DEEP_DIM, 1, 0.5, 1.0, GATE_SEED);
    let metrics: [(&'static str, DistanceMetric); 4] = [
        ("deep_euc", DistanceMetric::EuclideanSq),
        ("deep_cos", DistanceMetric::Cosine),
        ("deep_dot", DistanceMetric::DotProduct),
        ("deep_man", DistanceMetric::Manhattan),
    ];
    println!(
        "dim={DEEP_DIM} n={DEEP_N} clusters={DEEP_CLUSTERS} m={DEEP_SUBVECTORS} nprobe={DEEP_NPROBE}"
    );
    for (label, metric) in metrics {
        let def = IvfPqIndexDefinition::new(
            label,
            DEEP_DIM as u32,
            DEEP_CLUSTERS,
            DEEP_SUBVECTORS,
            metric,
        )
        .with_nprobe(DEEP_NPROBE);
        let start = std::time::Instant::now();
        let (_tmp, db) = setup_pq_only_index(&def, &vectors, DEEP_TRAIN_SAMPLE, FAST_TRAIN_ITERS);
        let build_s = start.elapsed().as_secs_f64();
        let r = measure_recall(&db, &def, &vectors, metric, 10, DEEP_NPROBE, 20);
        println!("{label:10} recall@10={r:.4} build={build_s:.1}s");
        assert!(
            r >= 0.10,
            "{label} recall@10 = {r:.4} is implausibly low even for the deep tier"
        );
    }
}
