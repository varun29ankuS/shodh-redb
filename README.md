# shodh-redb

[![Crates.io](https://img.shields.io/crates/v/shodh-redb.svg)](https://crates.io/crates/shodh-redb)
[![Documentation](https://docs.rs/shodh-redb/badge.svg)](https://docs.rs/shodh-redb)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/varun29ankuS/shodh-redb/actions/workflows/ci.yml/badge.svg)](https://github.com/varun29ankuS/shodh-redb/actions)
[![no_std](https://img.shields.io/badge/no__std-compatible-green)](https://doc.rust-lang.org/reference/names/preludes.html#the-no_std-attribute)

**The embedded database for AI agents.** Vector search, blob storage, TTL, causal tracking, CDC, concurrent multi-writer -- all ACID, single binary, zero external dependencies.

Give your agents persistent memory, semantic retrieval, and structured state -- without spinning up Postgres, Redis, or a vector database.

```toml
[dependencies]
shodh-redb = { version = "0.2", features = ["bf_tree"] }
```

---

## Why shodh-redb?

AI agents need more than a key-value store. They need to:

- **Remember** -- store and retrieve embeddings for RAG, tool history, and long-term memory
- **Reason over context** -- ranked fusion across semantic similarity, recency, and causal chains
- **Expire stale state** -- TTL tables auto-purge old sessions, tool results, and cached responses
- **Stream changes** -- CDC feeds let downstream systems react to every agent decision
- **Run anywhere** -- single binary, `no_std` compatible core, works in serverless, edge, and embedded

shodh-redb does all of this in a single embedded library.

---

## Quick start

```rust
use shodh_redb::bf_tree_store::{BfTreeDatabase, BfTreeConfig};
use shodh_redb::TableDefinition;

const TABLE: TableDefinition<&str, u64> = TableDefinition::new("agent_state");

let db = BfTreeDatabase::create(BfTreeConfig::new_file("agent.bftree", 32)).unwrap();

// Multiple agents can write concurrently
let wtxn = db.begin_write();
let mut table = wtxn.open_table(TABLE);
table.insert(&"task_count", &42u64).unwrap();
let _ = table;
wtxn.commit().unwrap(); // atomic -- all or nothing

// Readers never block writers
let rtxn = db.begin_read();
let table = rtxn.open_table(TABLE);
let val = table.get(&"task_count").unwrap();
```

Concurrent writers, lock-free readers, buffered atomic transactions. Built on [BfTree](https://www.microsoft.com/en-us/research/publication/bf-tree/) (Microsoft Research, VLDB 2024).

---

## Features

### Vector search

Native approximate nearest neighbor search. Store embeddings and retrieve similar items without an external vector database.

```rust
use shodh_redb::{FixedVec, TableDefinition, DistanceMetric, nearest_k};

// 384-dim embeddings (e.g. all-MiniLM-L6-v2)
const EMBEDDINGS: TableDefinition<u64, FixedVec<384>> = TableDefinition::new("memory");

// Search: brute-force top-k with any distance metric
let query: [f32; 384] = get_embedding("what did I do yesterday?");
let metric = DistanceMetric::Cosine;
let results = nearest_k(
    table.iter().unwrap().map(|r| {
        let (k, v) = r.unwrap();
        (k.value(), v.value().to_vec())
    }),
    &query,
    10,
    |a, b| metric.compute(a, b),
);
```

**Quantization** -- three levels of compression:

```rust
use shodh_redb::{ScalarQuantized, BinaryQuantized, quantize_scalar, quantize_binary};

// Scalar quantized: ~4x compression, bounded error
const SQ: TableDefinition<u64, ScalarQuantized<384>> = TableDefinition::new("sq_memory");

// Binary quantized: 32x compression, hamming distance
const BQ: TableDefinition<u64, BinaryQuantized<48>> = TableDefinition::new("bq_memory");
```

Indexing: **IVF-PQ** (inverted file with product quantization) and **Fractal** (hierarchical cluster index).

### Blob store

Store documents, images, tool outputs, and any binary data with content-addressed dedup and causal tracking.

```rust
use shodh_redb::blob_store::{ContentType, StoreOptions};

// Atomic write with SHA-256 dedup
let blob_id = wtxn.store_blob(data, ContentType::ImagePng, "screenshot", &opts)?;

// Streaming write for large blobs
let mut writer = wtxn.blob_writer(ContentType::ApplicationOctet, "model.bin", opts)?;
writer.write(&chunk1)?;
writer.write(&chunk2)?;
let blob_id = writer.finish()?;

// Read back (seekable)
let reader = rtxn.blob_reader(blob_id)?;
```

Chunked storage, per-blob tags and namespaces, causal graph with BFS traversal, temporal indexing.

### TTL tables

Per-key expiration for sessions, cached tool results, and temporary agent state.

```rust
use shodh_redb::TtlTableDefinition;
use std::time::Duration;

const CACHE: TtlTableDefinition<&str, &[u8]> = TtlTableDefinition::new("tool_cache");

let mut table = wtxn.open_table(CACHE)?;
table.insert_with_ttl(&"result_abc", data, Duration::from_secs(1800))?;
// Expired entries filtered on read, bulk cleanup:
let purged = table.purge_expired()?;
```

### Merge operators

Atomic read-modify-write for counters, accumulators, and running aggregates -- no manual locking.

```rust
use shodh_redb::{NumericAdd, MergeOperator};

// Atomic counter increment -- no read-lock-write cycle
table.merge(&"tool_calls", &1u64.to_le_bytes(), &NumericAdd)?;
```

Built-in operators: `NumericAdd`, `SaturatingAdd`, `FloatAdd`, `NumericMax`, `NumericMin`, `BitwiseOr`, `BytesAppend`. Or implement `MergeOperator` for custom logic.

### Change data capture (CDC)

Row-level change tracking. Stream every insert, update, and delete to downstream consumers -- audit logs, replication, or reactive agent pipelines.

```rust
// Read all changes since transaction 42
let changes = rtxn.read_cdc_since(42)?;
for change in &changes {
    match change.op {
        ChangeOp::Insert => { /* new key */ }
        ChangeOp::Update => { /* overwrite */ }
        ChangeOp::Delete => { /* removal */ }
        _ => {}
    }
}
```

### Multimap tables

Multiple values per key. Tag indices, many-to-many relationships, inverted lookups.

### Group commit

Batched fsync for write-heavy workloads. Single shared transaction for atomicity.

### Composite queries

Multi-signal ranked fusion: semantic similarity + temporal recency + causal relevance + namespace filtering. One query combines all the signals your agent needs to retrieve the right context.

---

## Performance

### Database benchmarks

Measured on a single machine, single-threaded (except mixed workload: 4 readers + 1 writer). All engines use default durability (fsync per commit) unless noted.

| Workload | BfTree (sync) | BfTree (periodic fsync) | RocksDB | Notes |
|---|---|---|---|---|
| Sequential writes (50K keys) | 88K ops/s | 1.55M ops/s | 91K ops/s | BfTree periodic mode batches fsync |
| Point reads (5K keys) | 600K ops/s | -- | 1.9M ops/s | RocksDB block cache advantage |
| Mixed 4R+1W p99 latency | 33.9 us | 42.6 us | 6.9 us | Reader-friendly RwLock, no cascading stalls |
| Read-only p99 latency | 1.2 us | -- | 0.8 us | Both sub-microsecond at p50 |

**Where BfTree wins**: concurrent multi-writer (lock-free readers, no single-writer bottleneck), periodic durability throughput (17x sync mode), embedded simplicity (no background compaction, no LSM write amplification).

**Where RocksDB wins**: raw point-read throughput (mature block cache, bloom filters), mixed p99 latency (optimized for read-heavy OLTP).

**Trade-off**: BfTree is designed for AI agent workloads -- write-heavy, concurrent, embedded. RocksDB is a general-purpose LSM with 10+ years of optimization. The gap narrows under multi-writer contention where RocksDB's single-writer mutex becomes the bottleneck.

### SIMD acceleration

Distance functions (`dot_product`, `euclidean_distance_sq`, `cosine_similarity`, `manhattan_distance`, `hamming_distance`) use hand-written AVX2 intrinsics on x86_64 with runtime feature detection and scalar fallback.

| Function | Strategy | Throughput |
|---|---|---|
| `dot_product` | 8 f32/iter via `_mm256_mul_ps` + `_mm256_add_ps` | ~8x scalar |
| `euclidean_distance_sq` | 8 f32/iter, fused sub+mul | ~8x scalar |
| `cosine_similarity` | 3 accumulators, 8 f32/iter | ~8x scalar |
| `manhattan_distance` | `_mm256_andnot_ps` for abs | ~8x scalar |
| `hamming_distance` | Mula's vectorized popcount (`pshufb` + SAD), 32 bytes/iter | ~16-32x scalar |

On `no_std` or non-x86 targets, the scalar code is structured for LLVM auto-vectorization (equal-length assertions, indexed loops, separate accumulators).

### Zero-copy serialization

On little-endian targets (x86, ARM LE, RISC-V LE), `FixedVec` and `DynVec` serialization uses bulk `copy_nonoverlapping` instead of per-element conversion -- the f32 memory layout matches the on-disk LE format directly.

### `no_std` support

The core library compiles with `#![no_std]` (disable the `std` feature). Vector types, distance functions, quantization, and the B-tree engine all work without the standard library. The `std` feature adds file backends, TTL tables, group commit, and runtime SIMD dispatch.

---

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `bf_tree` | No | BfTree concurrent multi-writer engine |
| `std` | Yes | File backends, group commit, TTL, runtime SIMD dispatch |
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
                    |  (typed, unified)  |
                    +--------+-----------+
                             |
              +--------------+--------------+
              |              |              |
        +-----+------+ +----+-----+ +------+------+
        | Key-Value  | | Blob     | | Vector      |
        | Tables     | | Store    | | Index       |
        | TTL, Merge | | CDC,     | | IVF-PQ,     |
        | Multimap   | | dedup,   | | Fractal     |
        |            | | causal   | |             |
        +-----+------+ +----+-----+ +------+------+
              |              |              |
              +--------------+--------------+
                             |
                   +---------+---------+
                   | BfTree Engine     |
                   | (multi-writer,    |
                   |  CAS, WAL,       |
                   |  lock-free reads) |
                   +-------------------+
```

---

## Credits

BfTree integration wraps [bf-tree](https://crates.io/crates/bf-tree) (Microsoft Research, VLDB 2024). Core page cache and crash recovery inherited from [redb](https://github.com/cberner/redb). All extensions -- vectors, blobs, TTL, merge operators, HLC, CDC, composite queries, group commit -- are original work.

## License

[Apache License, Version 2.0](LICENSE)

Copyright 2025-2026 Varun Sharma
