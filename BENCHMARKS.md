# Benchmarks

## Vector Search: SIFT1M

IVF-PQ (inverted file with product quantization) on the standard [SIFT1M](http://corpus-texmex.irisa.fr/) dataset -- 1 million 128-dimensional vectors, 10,000 queries, Euclidean distance.

### Index Configuration

| Parameter | Value |
|---|---|
| Dimensions | 128 |
| Clusters (nlist) | 256 |
| Subquantizers (M) | 16 (8 floats each) |
| K-means iterations | 15 |
| Training vectors | 10,000 |
| Raw vectors stored | Yes (for reranking) |
| Storage engine | B-tree (copy-on-write, ACID) |

### Results

**Full recall benchmark (k=100)**

| Mode | nprobe | recall@1 | recall@10 | recall@100 | QPS | p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
| PQ-only | 10 | 0.4245 | 0.5214 | 0.5907 | 1,521 | 0.64 ms | 0.84 ms | 1.12 ms |
| rerank | 1 | 0.5280 | 0.4779 | 0.4108 | 495 | 1.82 ms | 3.43 ms | 5.24 ms |
| rerank | 10 | 0.9567 | 0.9515 | 0.9174 | 337 | 2.72 ms | 4.59 ms | 5.50 ms |
| rerank | 50 | 0.9913 | 0.9990 | 0.9900 | 157 | 5.75 ms | 9.61 ms | 11.17 ms |

**Top-10 queries (realistic use case)**

| Mode | nprobe | recall@1 | recall@10 | QPS | p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| k10 | 10 | 0.9531 | 0.9258 | 1,018 | 0.90 ms | 1.55 ms | 1.92 ms |
| k10 (200 candidates) | 10 | 0.9561 | 0.9457 | 816 | 1.13 ms | 1.91 ms | 2.26 ms |
| k10 (500 candidates) | 10 | 0.9566 | 0.9512 | 484 | 1.88 ms | 3.17 ms | 3.71 ms |

**Build time:** 30.78s (single-threaded, includes k-means training)

### Key Takeaways

- **95.3% recall@1 at 1,018 QPS** for top-10 reranked queries -- the sweet spot for most applications
- **99.1% recall@1** achievable at nprobe=50 (157 QPS) when accuracy is critical
- **PQ-only scan at 1,521 QPS** -- useful for fast approximate filtering before reranking
- Sub-millisecond p50 latency for top-10 queries
- All queries run single-threaded with ACID transactions on an embedded B-tree

### How to Reproduce

```bash
# Download SIFT1M dataset
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xzf sift.tar.gz

# Run benchmark
SIFT1M_PATH=./sift cargo bench -p shodh-redb-bench --bench recall_benchmark
```

### Environment

| | |
|---|---|
| CPU | Intel Core i7-1355U (10 cores, up to 5.0 GHz) |
| RAM | 8 GB DDR4 |
| OS | Windows 11 |
| Rust | 1.89.0 |
| Profile | release (optimized + debuginfo) |
| Date | 2026-04-13 |
