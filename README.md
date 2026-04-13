# shodh-redb

[![Crates.io](https://img.shields.io/crates/v/shodh-redb.svg)](https://crates.io/crates/shodh-redb)
[![Documentation](https://docs.rs/shodh-redb/badge.svg)](https://docs.rs/shodh-redb)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/varun29ankuS/shodh-redb/actions/workflows/ci.yml/badge.svg)](https://github.com/varun29ankuS/shodh-redb/actions)
[![no_std](https://img.shields.io/badge/no__std-compatible-green)](https://doc.rust-lang.org/reference/names/preludes.html#the-no_std-attribute)

**Embedded vector database for edge AI.** IVF-PQ search, blob storage, TTL, causal tracking, CDC -- ACID transactions, single file, `no_std` core, zero dependencies.

95% recall@1 at 1,000+ QPS on SIFT1M. Single-threaded. 8 GB RAM. No GPU.

```toml
[dependencies]
shodh-redb = "0.3"
```

---

## Why shodh-redb?

Every edge vector database (sqlite-vec, LanceDB, Qdrant) bolts vector search onto a storage engine that wasn't designed for it. shodh-redb builds both from the same B-tree:

- **Vector search** -- IVF-PQ with integer ADC, SIMD distance, and ACID-transactional index updates
- **Blob storage** -- content-addressed dedup, chunked writes, causal lineage graph
- **TTL tables** -- per-key expiration for sessions, caches, and ephemeral agent state
- **CDC** -- row-level change capture for replication and reactive pipelines
- **Composite queries** -- ranked fusion across semantic, temporal, and causal signals
- **Runs anywhere** -- `no_std` core compiles to wasm32, ARM Cortex, RISC-V. No allocator, no filesystem required

One binary. One file. ACID crash-safe.

---

## Benchmarks (SIFT1M)

1M vectors, 128 dimensions, Euclidean distance. Single-threaded on Intel i7-1355U, 8 GB RAM.

| Mode | nprobe | recall@1 | recall@10 | QPS | p50 latency |
|---|---|---|---|---|---|
| Top-10 reranked | 10 | 95.3% | 92.6% | 1,018 | 0.90 ms |
| Full rerank | 10 | 95.7% | 95.2% | 337 | 2.72 ms |
| Full rerank | 50 | 99.1% | 99.9% | 157 | 5.75 ms |
| PQ-only scan | 10 | 42.5% | 52.1% | 1,521 | 0.64 ms |

Index build: 30.8s (single-threaded, 256 clusters, M=16 subquantizers).

Full results and reproduction steps: [BENCHMARKS.md](BENCHMARKS.md)

---

## Quick start

```rust
use shodh_redb::{Database, TableDefinition, ReadableDatabase, ReadableTable};

const TABLE: TableDefinition<&str, u64> = TableDefinition::new("agent_state");

let db = Database::create("agent.redb")?;

let wtxn = db.begin_write()?;
{
    let mut table = wtxn.open_table(TABLE)?;
    table.insert("task_count", &42u64)?;
}
wtxn.commit()?;

let rtxn = db.begin_read()?;
let table = rtxn.open_table(TABLE)?;
assert_eq!(table.get("task_count")?.unwrap().value(), 42);
```

---

## Vector search

Native IVF-PQ index with integer ADC, residual encoding, and per-vector metadata filtering.

```rust
use shodh_redb::{FixedVec, DistanceMetric, nearest_k, TableDefinition};

// Brute-force top-k (small datasets)
const EMBEDDINGS: TableDefinition<u64, FixedVec<384>> = TableDefinition::new("memory");
let results = nearest_k(
    table.iter()?.map(|r| { let (k, v) = r.unwrap(); (k.value(), v.value().to_vec()) }),
    &query, 10, |a, b| DistanceMetric::Cosine.compute(a, b),
);
```

Three quantization levels:

| Type | Compression | Distance |
|---|---|---|
| `FixedVec<N>` / `DynVec` | None (f32) | Cosine, L2, dot, Manhattan |
| `ScalarQuantized<N>` | ~4x (u8) | Approximate L2 |
| `BinaryQuantized<N>` | 32x (1-bit) | Hamming |

For datasets beyond brute-force range, use the **IVF-PQ index** -- inverted file with product quantization, integer asymmetric distance tables, and optional reranking from stored raw vectors.

### Blob store

Content-addressed storage with SHA-256 dedup, chunked streaming writes, and causal lineage tracking.

```rust
let blob_id = wtxn.store_blob(data, ContentType::ImagePng, "screenshot", &opts)?;
let reader = rtxn.blob_reader(blob_id)?;
```

Per-blob tags, namespaces, temporal indexing, and causal graph with BFS traversal.

### TTL tables

Per-key expiration. Lazy filtering on read, bulk purge on demand.

```rust
let mut table = wtxn.open_table(CACHE)?;
table.insert_with_ttl(&"result_abc", data, Duration::from_secs(1800))?;
table.purge_expired()?;
```

### Merge operators

Atomic read-modify-write without manual locking.

```rust
table.merge(&"tool_calls", &1u64.to_le_bytes(), &NumericAdd)?;
```

Built-in: `NumericAdd`, `SaturatingAdd`, `FloatAdd`, `NumericMax`, `NumericMin`, `BitwiseOr`, `BytesAppend`. Implement `MergeOperator` for custom logic.

### CDC (change data capture)

Row-level change streaming with cursors, retention pruning, and HLC timestamps.

### Composite queries

Multi-signal ranked retrieval fusing semantic similarity, temporal recency, causal proximity, and namespace/tag filtering into a single scored result set.

### Multimap tables

Multiple values per key for tag indices, inverted lookups, and many-to-many relationships.

### Group commit

Batched fsync for write-heavy workloads. Multiple operations share a single transaction and disk sync.

---

## Performance

### SIMD acceleration

Distance functions use hand-written AVX2 intrinsics on x86_64 with runtime feature detection and scalar fallback.

| Function | Strategy | Throughput |
|---|---|---|
| `dot_product` | `_mm256_mul_ps` + `_mm256_add_ps`, 8 f32/iter | ~8x scalar |
| `euclidean_distance_sq` | Fused sub+mul, 8 f32/iter | ~8x scalar |
| `cosine_similarity` | 3 accumulators, 8 f32/iter | ~8x scalar |
| `manhattan_distance` | `_mm256_andnot_ps` for abs | ~8x scalar |
| `hamming_distance` | Mula's vectorized popcount (`pshufb` + SAD), 32 bytes/iter | ~16-32x scalar |

On `no_std` or non-x86 targets, scalar code is structured for LLVM auto-vectorization.

### Zero-copy serialization

On little-endian targets (x86, ARM LE, RISC-V LE), `FixedVec` and `DynVec` use bulk `copy_nonoverlapping` -- the f32 memory layout matches the on-disk LE format directly.

---

## `no_std` support

The core library compiles with `#![no_std]` (disable the `std` feature). Vector types, distance functions, quantization, IVF-PQ indexing, and the B-tree engine all work without the standard library. Targets: wasm32, ARM Cortex-M, RISC-V bare metal.

The `std` feature adds file backends, TTL tables, group commit, and runtime SIMD dispatch.

---

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `std` | Yes | File backends, group commit, TTL, SIMD dispatch |
| `logging` | No | `log` crate integration |
| `cache_metrics` | No | Cache hit/miss counters |
| `compression_lz4` | No | LZ4 page compression |
| `compression_zstd` | No | Zstandard page compression |
| `compression` | No | All compression algorithms |

---

## Architecture

```
                    +--------------------+
                    |   Database API     |
                    |  (typed, ACID)     |
                    +--------+-----------+
                             |
              +--------------+--------------+
              |              |              |
        +-----+------+ +----+-----+ +------+------+
        | Key-Value  | | Blob     | | Vector      |
        | Tables     | | Store    | | Index       |
        | TTL, Merge | | CDC,     | | IVF-PQ,     |
        | Multimap   | | dedup,   | | int ADC,    |
        |            | | causal   | | SIMD        |
        +-----+------+ +----+-----+ +------+------+
              |              |              |
              +--------------+--------------+
                             |
                   +---------+---------+
                   | B-tree Engine     |
                   | (COW pages, MVCC, |
                   |  crash-safe,      |
                   |  no_std core)     |
                   +-------------------+
```

---

## Credits

Core B-tree page store and crash recovery derived from [redb](https://github.com/cberner/redb). All extensions -- vector indexing, blob store, TTL, merge operators, HLC, CDC, composite queries, group commit, SIMD distance, quantization -- are original work.

## License

[Apache License, Version 2.0](LICENSE)

Copyright 2025-2026 Varun Sharma
