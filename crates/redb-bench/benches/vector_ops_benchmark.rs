//! Benchmark for vector distance functions and related operations.
//!
//! Measures throughput of SIMD-accelerated distance computations, quantization,
//! normalization, serialization, and top-k search across common embedding dimensions.
//! Runs each benchmark 3 times and reports the average.

use std::time::{Duration, Instant};

fn black_box<T>(x: T) -> T {
    unsafe {
        let ret = std::ptr::read_volatile(&x);
        std::mem::forget(x);
        ret
    }
}

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

const RUNS: usize = 3;

struct BenchResult {
    _ns_per_op: f64,
    _ops_per_sec: f64,
    _gb_per_sec: f64,
}

fn bench_single<F: FnMut() -> T, T>(iters: u64, mut f: F) -> Duration {
    // Warmup
    for _ in 0..200 {
        black_box(f());
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    start.elapsed()
}

fn bench_fn_avg<F: FnMut() -> T + Clone, T>(
    name: &str,
    dim: usize,
    bytes_per_op: usize,
    iters: u64,
    f: F,
) -> BenchResult {
    let mut total_ns = 0.0f64;

    for _ in 0..RUNS {
        let elapsed = bench_single(iters, f.clone());
        total_ns += elapsed.as_nanos() as f64 / iters as f64;
    }

    let ns_per_op = total_ns / RUNS as f64;
    let ops_per_sec = 1_000_000_000.0 / ns_per_op;
    let gb_per_sec = (bytes_per_op as f64 * ops_per_sec) / 1_000_000_000.0;

    println!(
        "  {name:<35} dim={dim:<5} {ns_per_op:>8.1} ns/op  {ops_per_sec:>12.0} ops/s  {gb_per_sec:>6.2} GB/s"
    );

    BenchResult {
        _ns_per_op: ns_per_op,
        _ops_per_sec: ops_per_sec,
        _gb_per_sec: gb_per_sec,
    }
}

fn main() {
    let dims = [128, 384, 768, 1536];
    let iters = 500_000u64;
    let hamming_dims = [48, 96, 192, 384];
    let hamming_iters = 2_000_000u64;

    println!("=== shodh-redb vector ops benchmark ===");
    println!("Platform: {}", std::env::consts::ARCH);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            println!("AVX2: detected (SIMD dispatch active)");
        } else {
            println!("AVX2: NOT detected (scalar fallback)");
        }
        if is_x86_feature_detected!("avx512f") {
            println!("AVX-512: detected (not used yet)");
        }
        if is_x86_feature_detected!("fma") {
            println!("FMA: detected");
        }
    }

    println!("Runs: {RUNS} (averaged)");
    println!("Iterations per run: {iters} (f32), {hamming_iters} (hamming)");
    println!();

    // =========================================================================
    // Distance functions
    // =========================================================================

    println!("--- dot_product ---");
    for &dim in &dims {
        let (a, b) = make_f32_vecs(dim);
        bench_fn_avg("dot_product", dim, dim * 4 * 2, iters, move || {
            shodh_redb::dot_product(&a, &b)
        });
    }
    println!();

    println!("--- euclidean_distance_sq ---");
    for &dim in &dims {
        let (a, b) = make_f32_vecs(dim);
        bench_fn_avg(
            "euclidean_distance_sq",
            dim,
            dim * 4 * 2,
            iters,
            move || shodh_redb::euclidean_distance_sq(&a, &b),
        );
    }
    println!();

    println!("--- cosine_similarity ---");
    for &dim in &dims {
        let (a, b) = make_f32_vecs(dim);
        bench_fn_avg("cosine_similarity", dim, dim * 4 * 2, iters, move || {
            shodh_redb::cosine_similarity(&a, &b)
        });
    }
    println!();

    println!("--- cosine_distance ---");
    for &dim in &dims {
        let (a, b) = make_f32_vecs(dim);
        bench_fn_avg("cosine_distance", dim, dim * 4 * 2, iters, move || {
            shodh_redb::cosine_distance(&a, &b)
        });
    }
    println!();

    println!("--- manhattan_distance ---");
    for &dim in &dims {
        let (a, b) = make_f32_vecs(dim);
        bench_fn_avg("manhattan_distance", dim, dim * 4 * 2, iters, move || {
            shodh_redb::manhattan_distance(&a, &b)
        });
    }
    println!();

    println!("--- hamming_distance ---");
    for &dim in &hamming_dims {
        let (a, b) = make_u8_vecs(dim);
        bench_fn_avg("hamming_distance", dim, dim * 2, hamming_iters, move || {
            shodh_redb::hamming_distance(&a, &b)
        });
    }
    println!();

    // =========================================================================
    // DistanceMetric.compute (end-to-end with mismatch check + NaN guard)
    // =========================================================================

    println!("--- DistanceMetric.compute (all metrics, dim=384) ---");
    {
        let dim = 384;
        let (a, b) = make_f32_vecs(dim);
        let metrics = [
            ("Cosine", shodh_redb::DistanceMetric::Cosine),
            ("EuclideanSq", shodh_redb::DistanceMetric::EuclideanSq),
            ("DotProduct", shodh_redb::DistanceMetric::DotProduct),
            ("Manhattan", shodh_redb::DistanceMetric::Manhattan),
        ];
        for (name, metric) in &metrics {
            let a = a.clone();
            let b = b.clone();
            let m = *metric;
            bench_fn_avg(
                &format!("DistanceMetric::{name}"),
                dim,
                dim * 4 * 2,
                iters,
                move || m.compute(&a, &b),
            );
        }
    }
    println!();

    // =========================================================================
    // Normalization
    // =========================================================================

    println!("--- l2_norm ---");
    for &dim in &dims {
        let (a, _) = make_f32_vecs(dim);
        bench_fn_avg("l2_norm", dim, dim * 4, iters, move || {
            shodh_redb::l2_norm(&a)
        });
    }
    println!();

    println!("--- l2_normalize (in-place) ---");
    for &dim in &dims {
        let (a, _) = make_f32_vecs(dim);
        bench_fn_avg("l2_normalize", dim, dim * 4, iters, move || {
            let mut v = a.clone();
            shodh_redb::l2_normalize(&mut v);
            v
        });
    }
    println!();

    // =========================================================================
    // Quantization
    // =========================================================================

    println!("--- quantize_binary ---");
    for &dim in &dims {
        let (a, _) = make_f32_vecs(dim);
        bench_fn_avg("quantize_binary", dim, dim * 4, iters, move || {
            shodh_redb::quantize_binary(&a)
        });
    }
    println!();

    println!("--- quantize_scalar (dim=384) ---");
    {
        let dim = 384;
        let arr: [f32; 384] = {
            let mut a = [0.0f32; 384];
            for (i, v) in a.iter_mut().enumerate() {
                *v = ((i * 7 + 13) % 1000) as f32 * 0.001 - 0.5;
            }
            a
        };
        bench_fn_avg("quantize_scalar<384>", dim, dim * 4, iters, move || {
            shodh_redb::quantize_scalar(&arr)
        });
    }
    println!();

    println!("--- sq_euclidean_distance_sq (dim=384) ---");
    {
        let dim = 384;
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
        let sq = shodh_redb::quantize_scalar(&arr);
        bench_fn_avg(
            "sq_euclidean_distance_sq<384>",
            dim,
            dim + dim * 4,
            iters,
            move || shodh_redb::sq_euclidean_distance_sq(&query, &sq),
        );
    }
    println!();

    // =========================================================================
    // Serialization
    // =========================================================================

    println!("--- write_f32_le ---");
    for &dim in &dims {
        let (a, _) = make_f32_vecs(dim);
        let mut buf = vec![0u8; dim * 4];
        bench_fn_avg("write_f32_le", dim, dim * 4, iters, move || {
            shodh_redb::write_f32_le(&mut buf, &a);
        });
    }
    println!();

    println!("--- read_f32_le ---");
    for &dim in &dims {
        let (a, _) = make_f32_vecs(dim);
        let mut buf = vec![0u8; dim * 4];
        shodh_redb::write_f32_le(&mut buf, &a);
        bench_fn_avg("read_f32_le", dim, dim * 4, iters, move || {
            shodh_redb::read_f32_le(&buf)
        });
    }
    println!();

    // =========================================================================
    // Top-K search (brute force)
    // =========================================================================

    println!("--- nearest_k (brute force, cosine, k=10) ---");
    for &(n_vectors, dim) in &[(1_000, 384), (10_000, 384), (1_000, 768), (10_000, 128)] {
        let db: Vec<(u64, Vec<f32>)> = (0..n_vectors)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|j| ((i * 7 + j * 11 + 13) % 1000) as f32 * 0.001 - 0.5)
                    .collect();
                (i as u64, v)
            })
            .collect();
        let query: Vec<f32> = (0..dim)
            .map(|j| ((j * 3 + 7) % 1000) as f32 * 0.001 - 0.5)
            .collect();

        let topk_iters = if n_vectors <= 1_000 { 5_000u64 } else { 500 };
        let metric = shodh_redb::DistanceMetric::Cosine;

        let label = format!("nearest_k n={n_vectors} k=10");
        let db_clone = db.clone();
        let query_clone = query.clone();

        // Manual 3-run average since we need to clone the iterator each time
        let mut total_ns = 0.0f64;
        for _ in 0..RUNS {
            // Warmup
            for _ in 0..10 {
                let results = shodh_redb::nearest_k(
                    db_clone.iter().map(|(k, v)| (*k, v.clone())),
                    &query_clone,
                    10,
                    |a, b| metric.compute(a, b),
                );
                black_box(results);
            }

            let start = Instant::now();
            for _ in 0..topk_iters {
                let results = shodh_redb::nearest_k(
                    db_clone.iter().map(|(k, v)| (*k, v.clone())),
                    &query_clone,
                    10,
                    |a, b| metric.compute(a, b),
                );
                black_box(results);
            }
            let elapsed = start.elapsed();
            total_ns += elapsed.as_nanos() as f64 / topk_iters as f64;
        }

        let ns_per_op = total_ns / RUNS as f64;
        let us_per_op = ns_per_op / 1000.0;
        let qps = 1_000_000_000.0 / ns_per_op;

        println!("  {label:<35} dim={dim:<5} {us_per_op:>8.1} us/query  {qps:>8.0} QPS");
    }
    println!();

    println!("Done.");
}
