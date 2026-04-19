//! Criterion benchmarks for IVF-PQ index: insert_batch and search.
//!
//! Training is done in setup (outside the timed region) since it is a one-time
//! cost, not the hot path we want to gate on.

use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use shodh_redb::{Database, DistanceMetric, IvfPqIndexDefinition, ReadableDatabase, SearchParams};
use tempfile::NamedTempFile;

const DIM: u32 = 128;
const NUM_CLUSTERS: u32 = 16;
const NUM_SUBVECTORS: u32 = 16;
const INDEX_DEF: IvfPqIndexDefinition = IvfPqIndexDefinition::new(
    "bench_idx",
    DIM,
    NUM_CLUSTERS,
    NUM_SUBVECTORS,
    DistanceMetric::EuclideanSq,
)
.with_raw_vectors();

fn make_vector(id: u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| {
            ((id.wrapping_mul(7)
                .wrapping_add(j as u64 * 11)
                .wrapping_add(13))
                % 1000) as f32
                * 0.001
                - 0.5
        })
        .collect()
}

fn create_trained_db() -> (NamedTempFile, Database) {
    let f = NamedTempFile::new().unwrap();
    let db = Database::create(f.path()).unwrap();
    let dim = DIM as usize;
    {
        let txn = db.begin_write().unwrap();
        let mut idx = txn.open_ivfpq_index(&INDEX_DEF).unwrap();
        let training: Vec<(u64, Vec<f32>)> =
            (0..200u64).map(|id| (id, make_vector(id, dim))).collect();
        idx.train(training.into_iter(), 10).unwrap();
        drop(idx);
        txn.commit().unwrap();
    }
    (f, db)
}

// ---------------------------------------------------------------------------
// Insert batch
// ---------------------------------------------------------------------------

fn bench_insert_batch(c: &mut Criterion) {
    let dim = DIM as usize;
    let mut group = c.benchmark_group("ivfpq/insert_batch");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(3));

    for &n in &[10u64, 100, 1000] {
        let vectors: Vec<(u64, Vec<f32>)> = (0..n)
            .map(|id| (id + 10_000, make_vector(id + 10_000, dim)))
            .collect();
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_batched(
                || {
                    // Setup: create a trained DB (not timed)
                    let (f, db) = create_trained_db();
                    (f, db, vectors.clone())
                },
                |(_f, db, vecs)| {
                    let txn = db.begin_write().unwrap();
                    let mut idx = txn.open_ivfpq_index(&INDEX_DEF).unwrap();
                    idx.insert_batch(vecs.into_iter()).unwrap();
                    drop(idx);
                    txn.commit().unwrap();
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

fn bench_search(c: &mut Criterion) {
    let dim = DIM as usize;
    // Build one trained + populated DB outside measurement loop
    let (_f, db) = create_trained_db();
    {
        let txn = db.begin_write().unwrap();
        let mut idx = txn.open_ivfpq_index(&INDEX_DEF).unwrap();
        let vecs: Vec<(u64, Vec<f32>)> = (0..500u64).map(|id| (id, make_vector(id, dim))).collect();
        idx.insert_batch(vecs.into_iter()).unwrap();
        drop(idx);
        txn.commit().unwrap();
    }

    let query = make_vector(9999, dim);
    let params = SearchParams::top_k(10);

    let mut group = c.benchmark_group("ivfpq/search");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));
    group.bench_function("search_500vecs_k10", |b| {
        b.iter(|| {
            let rtxn = db.begin_read().unwrap();
            let idx = rtxn.open_ivfpq_index(&INDEX_DEF).unwrap();
            idx.search(&rtxn, &query, &params).unwrap()
        });
    });
    group.finish();
}

criterion_group!(benches, bench_insert_batch, bench_search);
criterion_main!(benches);
