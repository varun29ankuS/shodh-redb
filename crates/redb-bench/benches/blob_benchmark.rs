use std::env::current_dir;
use std::io::Write;
use std::time::{Duration, Instant};
use std::{fs, process};

use comfy_table::presets::ASCII_MARKDOWN;
use shodh_redb::{BlobId, ContentType, Database, ReadableDatabase, StoreOptions};
use tempfile::NamedTempFile;

const SMALL_BLOB_COUNT: usize = 1_000;
const SMALL_BLOB_SIZE: usize = 4_096;
const LARGE_BLOB_COUNT: usize = 100;
const LARGE_BLOB_SIZE: usize = 1_024 * 1_024;
const STREAMING_BLOB_COUNT: usize = 10;
const STREAMING_BLOB_SIZE: usize = 10 * 1_024 * 1_024;
const STREAMING_CHUNK_SIZE: usize = 64 * 1_024;
const DEDUP_BLOB_COUNT: usize = 1_000;
const STATS_ITERATIONS: usize = 1_000;

struct BenchResult {
    name: String,
    duration: Duration,
    ops: usize,
    bytes: u64,
    extra: Option<String>,
}

impl BenchResult {
    fn ops_per_sec(&self) -> f64 {
        self.ops as f64 / self.duration.as_secs_f64()
    }

    fn mb_per_sec(&self) -> f64 {
        (self.bytes as f64 / (1024.0 * 1024.0)) / self.duration.as_secs_f64()
    }
}

fn make_blob_data(size: usize, seed: u8) -> Vec<u8> {
    let mut data = vec![0u8; size];
    let mut rng = fastrand::Rng::with_seed(seed as u64);
    rng.fill(&mut data);
    data
}

fn bench_small_blob_writes(db: &Database) -> (BenchResult, Vec<BlobId>) {
    let mut ids = Vec::with_capacity(SMALL_BLOB_COUNT);
    let start = Instant::now();
    {
        let txn = db.begin_write().unwrap();
        for i in 0..SMALL_BLOB_COUNT {
            let data = make_blob_data(SMALL_BLOB_SIZE, (i & 0xFF) as u8);
            let id = txn
                .store_blob(
                    &data,
                    ContentType::OctetStream,
                    "bench",
                    StoreOptions::default(),
                )
                .unwrap();
            ids.push(id);
        }
        txn.commit().unwrap();
    }
    let duration = start.elapsed();
    let total_bytes = (SMALL_BLOB_COUNT * SMALL_BLOB_SIZE) as u64;

    (
        BenchResult {
            name: format!("small blob writes ({SMALL_BLOB_COUNT} x {SMALL_BLOB_SIZE}B)"),
            duration,
            ops: SMALL_BLOB_COUNT,
            bytes: total_bytes,
            extra: None,
        },
        ids,
    )
}

fn bench_large_blob_writes(db: &Database) -> (BenchResult, Vec<BlobId>) {
    let mut ids = Vec::with_capacity(LARGE_BLOB_COUNT);
    let start = Instant::now();
    {
        let txn = db.begin_write().unwrap();
        for i in 0..LARGE_BLOB_COUNT {
            let data = make_blob_data(LARGE_BLOB_SIZE, (i & 0xFF) as u8);
            let id = txn
                .store_blob(
                    &data,
                    ContentType::OctetStream,
                    "bench",
                    StoreOptions::default(),
                )
                .unwrap();
            ids.push(id);
        }
        txn.commit().unwrap();
    }
    let duration = start.elapsed();
    let total_bytes = (LARGE_BLOB_COUNT * LARGE_BLOB_SIZE) as u64;

    (
        BenchResult {
            name: format!("large blob writes ({LARGE_BLOB_COUNT} x 1MB)"),
            duration,
            ops: LARGE_BLOB_COUNT,
            bytes: total_bytes,
            extra: None,
        },
        ids,
    )
}

fn bench_streaming_writes(db: &Database) -> (BenchResult, Vec<BlobId>) {
    let mut ids = Vec::with_capacity(STREAMING_BLOB_COUNT);
    let start = Instant::now();
    for i in 0..STREAMING_BLOB_COUNT {
        let txn = db.begin_write().unwrap();
        let mut writer = txn
            .blob_writer(
                ContentType::OctetStream,
                "bench-stream",
                StoreOptions::default(),
            )
            .unwrap();
        let data = make_blob_data(STREAMING_BLOB_SIZE, (i & 0xFF) as u8);
        let mut offset = 0;
        while offset < data.len() {
            let end = (offset + STREAMING_CHUNK_SIZE).min(data.len());
            writer.write_all(&data[offset..end]).unwrap();
            offset = end;
        }
        let id = writer.finish().unwrap();
        ids.push(id);
        txn.commit().unwrap();
    }
    let duration = start.elapsed();
    let total_bytes = (STREAMING_BLOB_COUNT * STREAMING_BLOB_SIZE) as u64;

    (
        BenchResult {
            name: format!("streaming writes ({STREAMING_BLOB_COUNT} x 10MB, 64KB chunks)"),
            duration,
            ops: STREAMING_BLOB_COUNT,
            bytes: total_bytes,
            extra: None,
        },
        ids,
    )
}

fn bench_sequential_reads(db: &Database, ids: &[BlobId]) -> BenchResult {
    let start = Instant::now();
    let txn = db.begin_read().unwrap();
    let mut total_bytes_read = 0u64;
    for id in ids {
        let (data, _meta) = txn.get_blob(id).unwrap().unwrap();
        total_bytes_read += data.len() as u64;
    }
    drop(txn);
    let duration = start.elapsed();

    BenchResult {
        name: format!("sequential reads ({} blobs)", ids.len()),
        duration,
        ops: ids.len(),
        bytes: total_bytes_read,
        extra: None,
    }
}

fn bench_range_reads(db: &Database, ids: &[BlobId]) -> BenchResult {
    let range_size = 1024u64;
    let start = Instant::now();
    let txn = db.begin_read().unwrap();
    let mut total_bytes_read = 0u64;
    for id in ids {
        let data = txn.read_blob_range(id, 0, range_size).unwrap().unwrap();
        total_bytes_read += data.len() as u64;
    }
    drop(txn);
    let duration = start.elapsed();

    BenchResult {
        name: format!("range reads ({} x 1KB slice)", ids.len()),
        duration,
        ops: ids.len(),
        bytes: total_bytes_read,
        extra: None,
    }
}

fn bench_dedup_writes(db: &Database) -> BenchResult {
    let data = make_blob_data(SMALL_BLOB_SIZE, 42);
    let start = Instant::now();
    {
        let txn = db.begin_write().unwrap();
        for _ in 0..DEDUP_BLOB_COUNT {
            txn.store_blob(
                &data,
                ContentType::OctetStream,
                "dedup",
                StoreOptions::default(),
            )
            .unwrap();
        }
        txn.commit().unwrap();
    }
    let duration = start.elapsed();

    // Check dedup stats
    let txn = db.begin_read().unwrap();
    let stats = txn.blob_stats().unwrap();
    drop(txn);

    let logical_bytes = (DEDUP_BLOB_COUNT * SMALL_BLOB_SIZE) as u64;
    let dedup_ratio = if stats.live_bytes > 0 {
        logical_bytes as f64 / stats.live_bytes as f64
    } else {
        1.0
    };

    BenchResult {
        name: format!("dedup writes ({DEDUP_BLOB_COUNT} identical {SMALL_BLOB_SIZE}B)"),
        duration,
        ops: DEDUP_BLOB_COUNT,
        bytes: logical_bytes,
        extra: Some(format!("dedup ratio: {dedup_ratio:.1}x")),
    }
}

fn bench_delete_and_compact(db: &mut Database, ids: &[BlobId]) -> BenchResult {
    let delete_count = ids.len() / 2;

    // Delete half
    {
        let txn = db.begin_write().unwrap();
        for id in ids.iter().take(delete_count) {
            txn.delete_blob(id).unwrap();
        }
        txn.commit().unwrap();
    }

    // Compact
    let start = Instant::now();
    let report = db.compact_blobs().unwrap();
    let duration = start.elapsed();

    BenchResult {
        name: format!("delete {delete_count} + compact"),
        duration,
        ops: 1,
        bytes: report.bytes_reclaimed,
        extra: Some(format!(
            "reclaimed: {:.2} MB, blobs relocated: {}",
            report.bytes_reclaimed as f64 / (1024.0 * 1024.0),
            report.blobs_relocated
        )),
    }
}

fn bench_blob_stats(db: &Database) -> BenchResult {
    let start = Instant::now();
    for _ in 0..STATS_ITERATIONS {
        let txn = db.begin_write().unwrap();
        let _stats = txn.blob_stats().unwrap();
        txn.abort().unwrap();
    }
    let duration = start.elapsed();

    BenchResult {
        name: format!("blob_stats() x {STATS_ITERATIONS}"),
        duration,
        ops: STATS_ITERATIONS,
        bytes: 0,
        extra: None,
    }
}

fn main() {
    let _ = env_logger::try_init();
    let tmpdir = current_dir().unwrap().join(".blob_benchmark");
    fs::create_dir_all(&tmpdir).unwrap();

    let tmpdir2 = tmpdir.clone();
    ctrlc::set_handler(move || {
        let _ = fs::remove_dir_all(&tmpdir2);
        process::exit(1);
    })
    .unwrap();

    let tmpfile: NamedTempFile = NamedTempFile::new_in(&tmpdir).unwrap();
    let mut builder = shodh_redb::Builder::new();
    builder.set_cache_size(4 * 1024 * 1024 * 1024);
    builder.set_blob_dedup(true);
    let mut db = builder.create(tmpfile.path()).unwrap();

    let mut results: Vec<BenchResult> = Vec::new();

    // Small blob writes
    let (result, small_ids) = bench_small_blob_writes(&db);
    println!(
        "{}: {:.0} ops/sec, {:.2} MB/s",
        result.name,
        result.ops_per_sec(),
        result.mb_per_sec()
    );
    results.push(result);

    // Large blob writes
    let (result, _large_ids) = bench_large_blob_writes(&db);
    println!(
        "{}: {:.0} ops/sec, {:.2} MB/s",
        result.name,
        result.ops_per_sec(),
        result.mb_per_sec()
    );
    results.push(result);

    // Streaming writes
    let (result, _streaming_ids) = bench_streaming_writes(&db);
    println!(
        "{}: {:.0} ops/sec, {:.2} MB/s",
        result.name,
        result.ops_per_sec(),
        result.mb_per_sec()
    );
    results.push(result);

    // Sequential reads (small blobs)
    let result = bench_sequential_reads(&db, &small_ids);
    println!(
        "{}: {:.0} ops/sec, {:.2} MB/s",
        result.name,
        result.ops_per_sec(),
        result.mb_per_sec()
    );
    results.push(result);

    // Range reads (small blobs)
    let result = bench_range_reads(&db, &small_ids);
    println!(
        "{}: {:.0} ops/sec, {:.2} MB/s",
        result.name,
        result.ops_per_sec(),
        result.mb_per_sec()
    );
    results.push(result);

    // Dedup writes (new DB to isolate)
    {
        let dedup_tmpfile: NamedTempFile = NamedTempFile::new_in(&tmpdir).unwrap();
        let mut dedup_builder = shodh_redb::Builder::new();
        dedup_builder.set_cache_size(4 * 1024 * 1024 * 1024);
        dedup_builder.set_blob_dedup(true);
        let dedup_db = dedup_builder.create(dedup_tmpfile.path()).unwrap();
        let result = bench_dedup_writes(&dedup_db);
        println!(
            "{}: {:.0} ops/sec | {}",
            result.name,
            result.ops_per_sec(),
            result.extra.as_deref().unwrap_or("")
        );
        results.push(result);
    }

    // Delete + compact
    let result = bench_delete_and_compact(&mut db, &small_ids);
    println!(
        "{}: {:?} | {}",
        result.name,
        result.duration,
        result.extra.as_deref().unwrap_or("")
    );
    results.push(result);

    // blob_stats()
    let result = bench_blob_stats(&db);
    println!("{}: {:.0} ops/sec", result.name, result.ops_per_sec());
    results.push(result);

    // Print markdown table
    let mut table = comfy_table::Table::new();
    table.load_preset(ASCII_MARKDOWN);
    table.set_header(["Operation", "Time", "Ops/sec", "Throughput", "Notes"]);
    for r in &results {
        let throughput = if r.bytes > 0 {
            format!("{:.2} MB/s", r.mb_per_sec())
        } else {
            "N/A".to_string()
        };
        table.add_row([
            r.name.clone(),
            format!("{:.2?}", r.duration),
            format!("{:.0}", r.ops_per_sec()),
            throughput,
            r.extra.clone().unwrap_or_default(),
        ]);
    }

    println!("\n{table}");

    let _ = fs::remove_dir_all(&tmpdir);
}
