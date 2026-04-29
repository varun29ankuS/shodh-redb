# Known Limitations

## Concurrency

- **Single-writer, multiple-reader**: Only one write transaction can be active at a time. Multiple read transactions can run concurrently with a single writer. There is no multi-process support or file locking — opening the same database file from multiple processes causes corruption.

## Durability

- **`Durability::None` page reclamation**: Non-durable commits defer page freeing when concurrent readers exist. This prevents MVCC corruption but means page reclamation is delayed until readers close or a durable commit runs. Write-heavy non-durable workloads with long-lived readers may see higher memory/file usage.

## Storage

- **No at-rest encryption**: Data is stored unencrypted on disk. Use filesystem-level encryption (LUKS, FileVault, BitLocker) if encryption at rest is required.

- **Max value size**: Individual values are limited to 3 GiB.

- **Manual compaction**: There is no background auto-compaction. Call `compact()` or `start_compaction()` explicitly. Compaction requires exclusive access (no active readers, no persistent savepoints).

- **File size growth**: Deleted data is not reclaimed until compaction. Write-heavy workloads will grow the file monotonically until `compact()` is called.

## TTL

- **Lazy eviction**: Expired entries are not removed automatically. Call `purge_expired()` on each TTL table to reclaim space from expired entries. Purge frequency should be proportional to write rate — high-throughput tables may need purging every few seconds, low-write tables every few minutes. The cost of `purge_expired()` is O(expired entries), not O(total entries). File size grows monotonically until expired entries are purged and `compact()` is called.

- **`no_std`**: TTL is not available without the `std` feature (requires system clock).

## IVF-PQ Vector Index

- **Non-incremental training**: `train()` retrains all codebooks and cluster centroids from scratch. There is no incremental training or online learning.

- **Savepoint reverts training**: Restoring a savepoint taken before training will revert the index to its pre-training state.

- **Recall vs. speed**: PQ is lossy compression. Recall depends on the number of subvectors, training data quality, and `nprobe` at search time.

## Cache

- **System-aware default**: Default cache size is 25% of physical RAM, clamped to [16 MiB, 1 GiB]. Override with `Builder::set_cache_size()`.

- **Fixed 90/10 read/write split**: The cache is split 90% read / 10% write. This ratio is not configurable at runtime.

## `no_std`

- **Spin-lock with timeout**: The `no_std` build uses a spin-lock with a configurable timeout instead of OS mutexes. Under high contention this can busy-wait.

- **No file I/O**: The `no_std` build requires a user-provided `StorageBackend` implementation. There is no built-in file backend.

- **No TTL**: System clock is unavailable.

## Error Handling

- **`Value` trait**: The `Value::from_bytes` trait method returns `Self`, not `Result<Self>`. If on-disk table definition metadata is corrupted, `InternalTableDefinition::from_bytes` panics with a diagnostic message. On `std` builds, all internal call sites wrap this in `catch_unwind` and convert panics to `StorageError::Corrupted`. On `no_std` builds, corrupt metadata causes an unrecoverable panic. All other deserialization paths return `StorageError::Corrupted` directly.
