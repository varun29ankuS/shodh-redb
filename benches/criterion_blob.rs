//! Criterion benchmarks for blob store operations: store and read.

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use shodh_redb::{ContentType, Database, ReadableDatabase, StoreOptions};
use tempfile::NamedTempFile;

fn make_blob_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i & 0xFF) as u8).collect()
}

// ---------------------------------------------------------------------------
// Store blob
// ---------------------------------------------------------------------------

fn bench_store_blob(c: &mut Criterion) {
    let sizes: &[(usize, &str)] = &[
        (1024, "1KiB"),
        (64 * 1024, "64KiB"),
        (1024 * 1024, "1MiB"),
    ];
    let mut group = c.benchmark_group("blob/store");
    for &(size, label) in sizes {
        let data = make_blob_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("store_blob", label), &size, |b, _| {
            b.iter_batched(
                || {
                    let f = NamedTempFile::new().unwrap();
                    let db = Database::builder()
                        .set_blob_dedup(true)
                        .create(f.path())
                        .unwrap();
                    (f, db)
                },
                |(_f, db)| {
                    let txn = db.begin_write().unwrap();
                    let _id = txn
                        .store_blob(
                            &data,
                            ContentType::OctetStream,
                            "bench",
                            StoreOptions::default(),
                        )
                        .unwrap();
                    txn.commit().unwrap();
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Read blob
// ---------------------------------------------------------------------------

fn bench_get_blob(c: &mut Criterion) {
    let sizes: &[(usize, &str)] = &[
        (1024, "1KiB"),
        (64 * 1024, "64KiB"),
        (1024 * 1024, "1MiB"),
    ];
    let mut group = c.benchmark_group("blob/get");
    for &(size, label) in sizes {
        let data = make_blob_data(size);
        // Pre-populate a DB with one blob of this size
        let f = NamedTempFile::new().unwrap();
        let db = Database::builder()
            .set_blob_dedup(true)
            .create(f.path())
            .unwrap();
        let blob_id = {
            let txn = db.begin_write().unwrap();
            let id = txn
                .store_blob(
                    &data,
                    ContentType::OctetStream,
                    "bench",
                    StoreOptions::default(),
                )
                .unwrap();
            txn.commit().unwrap();
            id
        };

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("get_blob", label), &size, |b, _| {
            b.iter(|| {
                let rtxn = db.begin_read().unwrap();
                let (got, _meta) = rtxn.get_blob(&blob_id).unwrap().unwrap();
                got
            });
        });
        drop(db);
        drop(f);
    }
    group.finish();
}

criterion_group!(benches, bench_store_blob, bench_get_blob);
criterion_main!(benches);
