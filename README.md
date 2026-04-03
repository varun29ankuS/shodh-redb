# shodh-redb

[![Crates.io](https://img.shields.io/crates/v/shodh-redb.svg)](https://crates.io/crates/shodh-redb)
[![Documentation](https://docs.rs/shodh-redb/badge.svg)](https://docs.rs/shodh-redb)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/varun29ankuS/shodh-redb/actions/workflows/ci.yml/badge.svg)](https://github.com/varun29ankuS/shodh-redb/actions)

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
- **Run anywhere** -- single binary, no network dependencies, works in serverless and edge

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
use shodh_redb::{FixedVec, TableDefinition, cosine_distance, nearest_k};

// 384-dim embeddings (e.g. all-MiniLM-L6-v2)
const EMBEDDINGS: TableDefinition<u64, FixedVec<384>> = TableDefinition::new("memory");

// Binary quantized -- 32x smaller, hardware-accelerated hamming distance
use shodh_redb::{BinaryQuantized, quantize_binary, hamming_distance};
const BINARY: TableDefinition<u64, BinaryQuantized<12>> = TableDefinition::new("binary_memory");
```

Indexing: **IVF-PQ** (inverted file with product quantization) and **Fractal** (hierarchical cluster index).

### Blob store

Store documents, images, tool outputs, and any binary data with content-addressed dedup and causal tracking.

```rust
use shodh_redb::{ContentType, StoreOptions};

// SHA-256 dedup, tags, namespaces, causal links
// let blob_id = txn.store_blob(data, ContentType::ImagePng, "screenshot", &opts)?;
// let reader = txn.blob_reader(blob_id)?; // seekable
```

Chunked storage, per-blob tags and namespaces, causal graph with BFS traversal, temporal indexing.

### TTL tables

Per-key expiration for sessions, cached tool results, and temporary agent state.

```rust
use shodh_redb::TtlTableDefinition;
use std::time::Duration;

const CACHE: TtlTableDefinition<&str, &[u8]> = TtlTableDefinition::new("tool_cache");

// table.insert_with_ttl("result_abc", data, Duration::from_secs(1800))?;
// Expired entries filtered on read, bulk cleanup with table.purge_expired()
```

### Merge operators

Atomic read-modify-write for counters, accumulators, and running aggregates -- no manual locking.

```rust
use shodh_redb::{NumericAdd, MergeOperator};

// Atomic counter increment
// table.merge(&"tool_calls", &1u64, &NumericAdd)?;
```

### Multimap tables

Multiple values per key. Tag indices, many-to-many relationships, inverted lookups.

### Change data capture (CDC)

Row-level change tracking. Stream every insert, update, and delete to downstream consumers -- audit logs, replication, or reactive agent pipelines.

### Group commit

Batched fsync for write-heavy workloads. Single shared transaction for atomicity.

### Composite queries

Multi-signal ranked fusion: semantic similarity + temporal recency + causal relevance + namespace filtering. One query combines all the signals your agent needs to retrieve the right context.

---

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `bf_tree` | No | BfTree concurrent multi-writer engine |
| `std` | Yes | File backends, group commit, TTL, full error types |
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
