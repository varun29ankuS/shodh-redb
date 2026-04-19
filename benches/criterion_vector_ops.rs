//! Criterion benchmarks for vector distance functions and related operations.
//!
//! Ported from crates/redb-bench/benches/vector_ops_benchmark.rs into Criterion
//! for statistical regression detection in CI.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use shodh_redb::{
    DistanceMetric, cosine_distance, cosine_similarity, dot_product, euclidean_distance_sq,
    hamming_distance, l2_norm, l2_normalize, manhattan_distance, nearest_k, quantize_binary,
    quantize_scalar, sq_euclidean_distance_sq,
};

fn make_f32_vecs(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..dim)
        .map(|i| ((i * 7 + 13) % 1000) as f32 * 0.001 - 0.5)
        .collect();
    let b: Vec<f32> = (0..dim)
        .map(|i| ((i * 11 + 37) % 1000) as f32 * 0.001 - 0.5)
        .collect();
    (a, b)
}

fn make_u8_vecs(dim: usize) -> (Vec<u8>, Vec<u8>) {
    let a: Vec<u8> = (0..dim).map(|i| ((i * 37 + 13) & 0xFF) as u8).collect();
    let b: Vec<u8> = (0..dim).map(|i| ((i * 53 + 7) & 0xFF) as u8).collect();
    (a, b)
}

// ---------------------------------------------------------------------------
// Distance functions
// ---------------------------------------------------------------------------

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops/dot_product");
    for &dim in &[128usize, 384, 768, 1536] {
        let (a, b) = make_f32_vecs(dim);
        group.throughput(Throughput::Bytes((dim * 4 * 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| dot_product(&a, &b));
        });
    }
    group.finish();
}

fn bench_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops/euclidean_distance_sq");
    for &dim in &[128usize, 384, 768, 1536] {
        let (a, b) = make_f32_vecs(dim);
        group.throughput(Throughput::Bytes((dim * 4 * 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| euclidean_distance_sq(&a, &b));
        });
    }
    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops/cosine_similarity");
    for &dim in &[128usize, 384, 768, 1536] {
        let (a, b) = make_f32_vecs(dim);
        group.throughput(Throughput::Bytes((dim * 4 * 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| cosine_similarity(&a, &b));
        });
    }
    group.finish();
}

fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops/cosine_distance");
    for &dim in &[128usize, 384, 768, 1536] {
        let (a, b) = make_f32_vecs(dim);
        group.throughput(Throughput::Bytes((dim * 4 * 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| cosine_distance(&a, &b));
        });
    }
    group.finish();
}

fn bench_manhattan(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops/manhattan_distance");
    for &dim in &[128usize, 384, 768, 1536] {
        let (a, b) = make_f32_vecs(dim);
        group.throughput(Throughput::Bytes((dim * 4 * 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| manhattan_distance(&a, &b));
        });
    }
    group.finish();
}

fn bench_hamming(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops/hamming_distance");
    for &dim in &[48usize, 96, 192, 384] {
        let (a, b) = make_u8_vecs(dim);
        group.throughput(Throughput::Bytes((dim * 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| hamming_distance(&a, &b));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// DistanceMetric.compute dispatch
// ---------------------------------------------------------------------------

fn bench_metric_dispatch(c: &mut Criterion) {
    let dim = 384usize;
    let (a, b) = make_f32_vecs(dim);
    let metrics = [
        ("Cosine", DistanceMetric::Cosine),
        ("EuclideanSq", DistanceMetric::EuclideanSq),
        ("DotProduct", DistanceMetric::DotProduct),
        ("Manhattan", DistanceMetric::Manhattan),
    ];
    let mut group = c.benchmark_group("vector_ops/metric_dispatch_384");
    group.throughput(Throughput::Bytes((dim * 4 * 2) as u64));
    for (name, metric) in &metrics {
        let a = a.clone();
        let b = b.clone();
        let m = *metric;
        group.bench_function(*name, move |bench| {
            bench.iter(|| m.compute(&a, &b));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

fn bench_l2_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops/l2_norm");
    for &dim in &[128usize, 384, 768, 1536] {
        let (a, _) = make_f32_vecs(dim);
        group.throughput(Throughput::Bytes((dim * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| l2_norm(&a));
        });
    }
    group.finish();
}

fn bench_l2_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops/l2_normalize");
    for &dim in &[128usize, 384, 768, 1536] {
        let (a, _) = make_f32_vecs(dim);
        group.throughput(Throughput::Bytes((dim * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter_batched(
                || a.clone(),
                |mut v| {
                    l2_normalize(&mut v);
                    v
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

fn bench_quantize_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops/quantize_binary");
    for &dim in &[128usize, 384, 768, 1536] {
        let (a, _) = make_f32_vecs(dim);
        group.throughput(Throughput::Bytes((dim * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| quantize_binary(&a));
        });
    }
    group.finish();
}

fn bench_quantize_scalar(c: &mut Criterion) {
    let dim = 384usize;
    let arr: [f32; 384] = {
        let mut a = [0.0f32; 384];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 7 + 13) % 1000) as f32 * 0.001 - 0.5;
        }
        a
    };
    let mut group = c.benchmark_group("vector_ops/quantize_scalar");
    group.throughput(Throughput::Bytes((dim * 4) as u64));
    group.bench_function("dim_384", |bench| {
        bench.iter(|| quantize_scalar(&arr));
    });
    group.finish();
}

fn bench_sq_euclidean(c: &mut Criterion) {
    let dim = 384usize;
    let query: [f32; 384] = {
        let mut a = [0.0f32; 384];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 11 + 37) % 1000) as f32 * 0.001 - 0.5;
        }
        a
    };
    let arr: [f32; 384] = {
        let mut a = [0.0f32; 384];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 7 + 13) % 1000) as f32 * 0.001 - 0.5;
        }
        a
    };
    let sq = quantize_scalar(&arr);
    let mut group = c.benchmark_group("vector_ops/sq_euclidean_distance_sq");
    group.throughput(Throughput::Bytes((dim + dim * 4) as u64));
    group.bench_function("dim_384", |bench| {
        bench.iter(|| sq_euclidean_distance_sq(&query, &sq));
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Brute-force nearest_k
// ---------------------------------------------------------------------------

fn bench_nearest_k(c: &mut Criterion) {
    let dim = 128usize;
    let query: Vec<f32> = (0..dim)
        .map(|i| ((i * 11 + 37) % 1000) as f32 * 0.001 - 0.5)
        .collect();

    let mut group = c.benchmark_group("vector_ops/nearest_k");
    for &n in &[100u64, 1000] {
        let db: Vec<(u64, Vec<f32>)> = (0..n)
            .map(|id| {
                let v: Vec<f32> = (0..dim)
                    .map(|j| ((id as usize * 7 + j * 13) % 1000) as f32 * 0.001 - 0.5)
                    .collect();
                (id, v)
            })
            .collect();
        let q = query.clone();
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::new("cosine_k10", n), &n, |bench, _| {
            bench.iter_batched(
                || (db.clone(), q.clone()),
                |(db, q)| nearest_k(db.into_iter(), &q, 10, cosine_distance),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_euclidean,
    bench_cosine_similarity,
    bench_cosine_distance,
    bench_manhattan,
    bench_hamming,
    bench_metric_dispatch,
    bench_l2_norm,
    bench_l2_normalize,
    bench_quantize_binary,
    bench_quantize_scalar,
    bench_sq_euclidean,
    bench_nearest_k,
);
criterion_main!(benches);
