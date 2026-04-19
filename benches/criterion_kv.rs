//! Criterion benchmarks for B-tree KV operations: insert, get, range scan.
//!
//! Uses Durability::None to isolate B-tree performance from fsync noise.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use shodh_redb::{Database, Durability, ReadableDatabase, TableDefinition};
use tempfile::NamedTempFile;

const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("bench_kv");
const VALUE_SIZE: usize = 150;

fn make_value(seed: u64) -> [u8; VALUE_SIZE] {
    let mut buf = [0u8; VALUE_SIZE];
    let mut state = seed;
    for b in buf.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (state >> 33) as u8;
    }
    buf
}

fn create_populated_db(n: u64) -> (NamedTempFile, Database) {
    let f = NamedTempFile::new().unwrap();
    let db = Database::builder()
        .set_cache_size(4 * 1024 * 1024)
        .create(f.path())
        .unwrap();
    let mut txn = db.begin_write().unwrap();
    txn.set_durability(Durability::None).unwrap();
    {
        let mut table = txn.open_table(TABLE).unwrap();
        for i in 0..n {
            let val = make_value(i);
            table.insert(&i, val.as_ref()).unwrap();
        }
    }
    txn.commit().unwrap();
    (f, db)
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv/insert");
    for &n in &[100u64, 1_000, 10_000] {
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let f = NamedTempFile::new().unwrap();
                    let db = Database::builder()
                        .set_cache_size(4 * 1024 * 1024)
                        .create(f.path())
                        .unwrap();
                    (f, db)
                },
                |(_f, db)| {
                    let mut txn = db.begin_write().unwrap();
                    txn.set_durability(Durability::None).unwrap();
                    {
                        let mut table = txn.open_table(TABLE).unwrap();
                        for i in 0..n {
                            let val = make_value(i);
                            table.insert(&i, val.as_ref()).unwrap();
                        }
                    }
                    txn.commit().unwrap();
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Point get
// ---------------------------------------------------------------------------

fn bench_get(c: &mut Criterion) {
    let (_f, db) = create_populated_db(10_000);

    let mut group = c.benchmark_group("kv/get");
    group.throughput(Throughput::Elements(1));
    group.bench_function("random_get_10k", |b| {
        let mut key = 0u64;
        b.iter(|| {
            let rtxn = db.begin_read().unwrap();
            let table = rtxn.open_table(TABLE).unwrap();
            let _ = table.get(&key).unwrap();
            key = (key.wrapping_add(7919)) % 10_000;
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Range scan
// ---------------------------------------------------------------------------

fn bench_range_scan(c: &mut Criterion) {
    let (_f, db) = create_populated_db(10_000);

    let mut group = c.benchmark_group("kv/range_scan");
    group.throughput(Throughput::Elements(100));
    group.bench_function("scan_100_from_10k", |b| {
        let mut start = 0u64;
        b.iter(|| {
            let rtxn = db.begin_read().unwrap();
            let table = rtxn.open_table(TABLE).unwrap();
            let mut count = 0u64;
            for entry in table.range(start..start + 100).unwrap() {
                let _ = entry.unwrap();
                count += 1;
            }
            start = (start.wrapping_add(137)) % 9_900;
            count
        });
    });
    group.finish();
}

criterion_group!(benches, bench_insert, bench_get, bench_range_scan);
criterion_main!(benches);
