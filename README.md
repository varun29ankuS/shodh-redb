# shodh-redb

[![Crates.io](https://img.shields.io/crates/v/shodh-redb.svg)](https://crates.io/crates/shodh-redb)
[![Documentation](https://docs.rs/shodh-redb/badge.svg)](https://docs.rs/shodh-redb)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/varun29ankuS/shodh-redb/actions/workflows/ci.yml/badge.svg)](https://github.com/varun29ankuS/shodh-redb/actions)

**Multi-modal embedded database for Rust.** Vectors, blobs, TTL, merge operators, causal tracking, and flash storage -- all on ACID B-trees. Runs on servers, WASM, and $5 microcontrollers.

```toml
[dependencies]
shodh-redb = "0.2"
```

---

## Why shodh-redb?

| Capability | redb | shodh-redb |
|---|---|---|
| ACID key-value store | Yes | Yes |
| Vector search (IVF-PQ, fractal index) | -- | Native ANN with PQ compression |
| Blob store (CDC, dedup, streaming) | -- | Append-only with SHA-256 dedup, seekable reads |
| Composite queries (semantic + temporal + causal) | -- | Multi-signal ranked fusion |
| TTL (per-key expiration) | -- | Lazy filtering + bulk purge |
| Merge operators (atomic RMW) | -- | `NumericAdd`, `NumericMax`, `BitwiseOr`, closures |
| Change data capture (CDC) | -- | Row-level change tracking |
| Hybrid Logical Clock | -- | Causal ordering across distributed writes |
| Flash storage backend | -- | Wear leveling, bad block management, power-loss safe |
| Group commit (batched fsync) | -- | Leader-election batching |
| Memory budget | -- | Hard RAM cap with adaptive cache |
| `no_std` / WASM | -- | `#![no_std]` with `alloc` |

---

## Quick start

### Key-value store

```rust
use shodh_redb::{Database, Error, ReadableDatabase, ReadableTable, TableDefinition};

const TABLE: TableDefinition<&str, u64> = TableDefinition::new("my_data");

fn main() -> Result<(), Error> {
    let db = Database::create("my_db.redb")?;
    let write_txn = db.begin_write()?;
    {
        let mut table = write_txn.open_table(TABLE)?;
        table.insert("my_key", &123)?;
    }
    write_txn.commit()?;

    let read_txn = db.begin_read()?;
    let table = read_txn.open_table(TABLE)?;
    assert_eq!(table.get("my_key")?.unwrap().value(), 123);
    Ok(())
}
```

### Vector search

```rust
use shodh_redb::{Database, FixedVec, TableDefinition, cosine_distance, nearest_k};

// Store 384-dim embeddings
const EMBEDDINGS: TableDefinition<u64, FixedVec<384>> = TableDefinition::new("embeddings");

// Binary quantized (32x smaller, hardware-accelerated hamming distance)
use shodh_redb::{BinaryQuantized, quantize_binary, hamming_distance};
const BINARY: TableDefinition<u64, BinaryQuantized<12>> = TableDefinition::new("binary_vecs");
```

### TTL tables

```rust
use shodh_redb::{Database, TtlTableDefinition};
use std::time::Duration;

const SESSIONS: TtlTableDefinition<&str, &[u8]> = TtlTableDefinition::new("sessions");

// Insert with 30-minute expiry
// table.insert_with_ttl("session_abc", data, Duration::from_secs(1800))?;
// Expired entries are automatically filtered on read
// Bulk cleanup: table.purge_expired()?;
```

### Merge operators

```rust
use shodh_redb::{NumericAdd, MergeOperator};

// Atomic counter increment -- no read-modify-write boilerplate
// table.merge(&"page_views", &1u64, &NumericAdd)?;
```

### Blob store

```rust
use shodh_redb::{Database, ContentType, StoreOptions};

// Store large data with streaming writes and content dedup
// let blob_id = write_txn.store_blob(image_bytes, ContentType::ImagePng, "photo", &StoreOptions::default())?;
// let reader = read_txn.blob_reader(blob_id)?; // Seekable reader
```

---

## Flash storage (bare-metal / embedded)

Run shodh-redb on raw flash with no OS. Implement the `FlashHardware` trait for your chip and the FTL handles everything else:

```rust
use shodh_redb::{Builder, FlashBackend, FlashGeometry, FlashHardware};

// Your chip's flash driver
struct MySpiFlash { /* SPI peripheral, CS pin */ }

impl FlashHardware for MySpiFlash {
    fn geometry(&self) -> FlashGeometry {
        FlashGeometry {
            erase_block_size: 4096,     // 4KB sectors (NOR)
            write_page_size: 256,       // page program size
            total_blocks: 2048,         // 8MB device
            max_erase_cycles: 100_000,
        }
    }
    // implement read, write_page, erase_block, is_bad_block, mark_bad_block, sync
    # // ...
}

let backend = FlashBackend::mount(MySpiFlash::new())?;
let db = Builder::new().create_with_backend(backend)?;
// Use exactly like a normal database
```

The flash translation layer provides:
- **Wear leveling** -- dynamic (lowest-erase-count allocation) + static (hot/cold swap)
- **Bad block management** -- scan on mount, runtime detection, transparent remapping
- **Power-loss safety** -- double-buffered metadata journal with xxh3-128 checksums
- **Copy-on-write** -- all writes go to fresh blocks; old blocks are erased and recycled

---

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `std` | Yes | File backends, group commit, TTL, full error types |
| `logging` | No | `log` crate integration |
| `cache_metrics` | No | Cache hit/miss counters |
| `compression_lz4` | No | LZ4 page compression |
| `compression_zstd` | No | Zstandard page compression |
| `compression` | No | All compression algorithms |

### `no_std` usage

```toml
[dependencies]
shodh-redb = { version = "0.2", default-features = false }
```

Requires `alloc`. Use `InMemoryBackend`, `FlashBackend`, or implement a custom `StorageBackend`.

---

## Architecture

```
                        +------------------+
                        |    Database API   |
                        +--------+---------+
                                 |
              +------------------+------------------+
              |                  |                  |
        +-----+------+   +------+------+   +-------+------+
        | Key-Value  |   | Blob Store  |   | Vector Index |
        | Tables     |   | (CDC, dedup)|   | (IVF-PQ,     |
        |            |   |             |   |  Fractal)    |
        +-----+------+   +------+------+   +-------+------+
              |                  |                  |
              +------------------+------------------+
                                 |
                   +-------------+-------------+
                   | TransactionalMemory (MVCC) |
                   +-------------+-------------+
                                 |
                   +-------------+-------------+
                   |    StorageBackend trait    |
                   +--+--------+--------+------+
                      |        |        |
                +-----+--+ +---+----+ +-+----------+
                |  File  | |InMemory| |FlashBackend|
                |Backend | |Backend | |(bare-metal) |
                +--------+ +--------+ +------------+
```

---

## Credits

Built on [redb](https://github.com/cberner/redb) by Christopher Berner. The core B-tree engine, page cache, MVCC, and crash recovery are inherited from redb. All extensions (vectors, blobs, TTL, merge operators, HLC, CDC, flash backend, composite queries, group commit, memory budget, `no_std`) are original work.

## License

[Apache License, Version 2.0](LICENSE)

Copyright 2025-2026 Varun Sharma
