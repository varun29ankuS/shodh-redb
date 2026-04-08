//! BfTree vs BTree vs RocksDB benchmark.
//!
//! Measures 9 dimensions: latency distribution, concurrent write throughput,
//! mixed read/write, write amplification, space amplification, recovery time,
//! throughput at scale, periodic-mode throughput, and periodic-mode mixed.

use std::env::current_dir;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use comfy_table::{Cell, Table};
use rocksdb::{OptimisticTransactionDB, OptimisticTransactionOptions, WriteOptions};
use shodh_redb::TableDefinition;
use shodh_redb::bf_tree_store::{
    BfTreeConfig, BfTreeDatabase, BfTreeDatabaseWriteTxn, DurabilityMode, WriteBatchFn,
    concurrent_group_commit,
};
use shodh_redb::{Database, Durability, ReadableDatabase};
use tempfile::TempDir;
use walkdir::WalkDir;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const KEY_SIZE: usize = 24;
const VALUE_SIZE: usize = 150;
const ENTRY_SIZE: usize = KEY_SIZE + VALUE_SIZE;
const RNG_SEED: u64 = 3;

const LATENCY_FILL: usize = 10_000;
const LATENCY_SAMPLE_READS: usize = 5_000;
const LATENCY_SAMPLE_WRITES: usize = 100;
const LATENCY_SAMPLE_SCANS: usize = 500;
const LATENCY_SCAN_LEN: usize = 10;

const CONCURRENT_FILL: usize = 5_000;
const CONCURRENT_OPS_PER_THREAD: usize = 2_000;
const CONCURRENT_BATCH_SIZE: usize = 500;
const GROUP_COMMIT_BATCHES: usize = 4;
const GROUP_COMMIT_OPS: usize = 1_000;

const MIXED_FILL: usize = 10_000;
const MIXED_DURATION_SECS: u64 = 3;
const MIXED_READER_THREADS: usize = 4;

const WRITE_AMP_ENTRIES: usize = 10_000;
const SPACE_AMP_ENTRIES: usize = 10_000;
const RECOVERY_ENTRIES: usize = 10_000;

const SCALE_LEVELS: &[usize] = &[5_000, 10_000, 50_000];
const SCALE_SAMPLE_OPS: usize = 5_000;
const SCALE_BATCH_SIZE: usize = 100;
const SCALE_FILL_BATCH: usize = 5000;

const TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("bench");

// ---------------------------------------------------------------------------
// Latency histogram
// ---------------------------------------------------------------------------

struct LatencyHistogram {
    samples: Vec<Duration>,
    sorted: bool,
}

#[allow(dead_code)]
impl LatencyHistogram {
    fn with_capacity(cap: usize) -> Self {
        Self {
            samples: Vec::with_capacity(cap),
            sorted: false,
        }
    }

    fn record(&mut self, d: Duration) {
        self.samples.push(d);
        self.sorted = false;
    }

    fn finalize(&mut self) {
        self.samples.sort_unstable();
        self.sorted = true;
    }

    fn percentile(&self, p: f64) -> Duration {
        assert!(self.sorted, "call finalize() before percentile()");
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let idx = ((p / 100.0) * (self.samples.len() as f64 - 1.0)).ceil() as usize;
        let idx = idx.min(self.samples.len() - 1);
        self.samples[idx]
    }

    fn p50(&self) -> Duration {
        self.percentile(50.0)
    }
    fn p95(&self) -> Duration {
        self.percentile(95.0)
    }
    fn p99(&self) -> Duration {
        self.percentile(99.0)
    }
    fn p999(&self) -> Duration {
        self.percentile(99.9)
    }

    fn merge(&mut self, other: &LatencyHistogram) {
        self.samples.extend_from_slice(&other.samples);
        self.sorted = false;
    }
}

fn fmt_us(d: Duration) -> String {
    let us = d.as_nanos() as f64 / 1000.0;
    if us < 1000.0 {
        format!("{us:.1}us")
    } else {
        format!("{:.1}ms", us / 1000.0)
    }
}

// ---------------------------------------------------------------------------
// RNG + key/value helpers
// ---------------------------------------------------------------------------

fn make_rng() -> fastrand::Rng {
    fastrand::Rng::with_seed(RNG_SEED)
}

fn random_pair(rng: &mut fastrand::Rng) -> ([u8; KEY_SIZE], Vec<u8>) {
    let mut key = [0u8; KEY_SIZE];
    rng.fill(&mut key);
    let mut value = vec![0u8; VALUE_SIZE];
    rng.fill(&mut value[..]);
    (key, value)
}

fn database_size(path: &Path) -> u64 {
    WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok().map(|m| m.len()))
        .sum()
}

#[cfg(target_os = "linux")]
fn proc_write_bytes() -> Option<u64> {
    let data = std::fs::read_to_string("/proc/self/io").ok()?;
    for line in data.lines() {
        if let Some(rest) = line.strip_prefix("write_bytes: ") {
            return rest.trim().parse().ok();
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn proc_write_bytes() -> Option<u64> {
    None
}

// ---------------------------------------------------------------------------
// RocksDB setup
// ---------------------------------------------------------------------------

fn rocksdb_opts() -> rocksdb::Options {
    let cache = rocksdb::Cache::new_lru_cache(256 * 1024 * 1024);
    let mut bb = rocksdb::BlockBasedOptions::default();
    bb.set_block_cache(&cache);
    bb.set_bloom_filter(10.0, false);

    let mut opts = rocksdb::Options::default();
    opts.set_block_based_table_factory(&bb);
    opts.create_if_missing(true);
    opts.increase_parallelism(std::thread::available_parallelism().map_or(1, |n| n.get()) as i32);
    opts
}

fn open_rocksdb(path: &Path) -> OptimisticTransactionDB {
    let opts = rocksdb_opts();
    OptimisticTransactionDB::open(&opts, path).unwrap()
}

fn rocksdb_put(db: &OptimisticTransactionDB, key: &[u8], value: &[u8]) {
    let mut wo = WriteOptions::new();
    wo.set_sync(true);
    let mut to = OptimisticTransactionOptions::new();
    to.set_snapshot(true);
    let txn = db.transaction_opt(&wo, &to);
    txn.put(key, value).unwrap();
    txn.commit().unwrap();
}

fn rocksdb_get(db: &OptimisticTransactionDB, key: &[u8]) -> Option<Vec<u8>> {
    db.get(key).unwrap()
}

fn rocksdb_batch_put(db: &OptimisticTransactionDB, pairs: &[([u8; KEY_SIZE], Vec<u8>)]) {
    let mut wo = WriteOptions::new();
    wo.set_sync(true);
    let mut to = OptimisticTransactionOptions::new();
    to.set_snapshot(true);
    let txn = db.transaction_opt(&wo, &to);
    for (k, v) in pairs {
        txn.put(k, v).unwrap();
    }
    txn.commit().unwrap();
}

fn rocksdb_delete(db: &OptimisticTransactionDB, key: &[u8]) {
    let mut wo = WriteOptions::new();
    wo.set_sync(true);
    let txn = db.transaction_opt(&wo, &OptimisticTransactionOptions::new());
    txn.delete(key).unwrap();
    txn.commit().unwrap();
}

// ---------------------------------------------------------------------------
// BfTree setup
// ---------------------------------------------------------------------------

fn bftree_config(path: &Path) -> BfTreeConfig {
    let mut config = BfTreeConfig::new_file(path.join("bftree.db"), 64);
    // Disable automatic snapshots during benchmarks. We call snapshot()
    // explicitly between fill and measurement phases to avoid a stop-the-world
    // snapshot landing mid-measurement (the default interval=100 caused a
    // 3x regression at the 10K scale point).
    config.snapshot_interval = 0;
    config
}

fn bftree_periodic_config(path: &Path) -> BfTreeConfig {
    let mut config = bftree_config(path);
    config.durability = DurabilityMode::Periodic;
    config
}

fn open_bftree(path: &Path) -> BfTreeDatabase {
    BfTreeDatabase::create(bftree_config(path)).unwrap()
}

fn open_bftree_periodic(path: &Path) -> BfTreeDatabase {
    BfTreeDatabase::create(bftree_periodic_config(path)).unwrap()
}

fn reopen_bftree(path: &Path) -> BfTreeDatabase {
    BfTreeDatabase::open(bftree_config(path)).unwrap()
}

fn bftree_put(db: &BfTreeDatabase, key: &[u8], value: &[u8]) {
    let wtxn = db.begin_write();
    wtxn.insert::<&[u8], &[u8]>(&TABLE, &key, &value).unwrap();
    wtxn.commit().unwrap();
}

fn bftree_get(db: &BfTreeDatabase, key: &[u8]) -> Option<Vec<u8>> {
    let mut rtxn = db.begin_read();
    rtxn.get::<&[u8], &[u8]>(&TABLE, &key).unwrap()
}

fn bftree_batch_put(db: &BfTreeDatabase, pairs: &[([u8; KEY_SIZE], Vec<u8>)]) {
    let wtxn = db.begin_write();
    for (k, v) in pairs {
        wtxn.insert::<&[u8], &[u8]>(&TABLE, &k.as_slice(), &v.as_slice())
            .unwrap();
    }
    wtxn.commit().unwrap();
}

fn bftree_delete(db: &BfTreeDatabase, key: &[u8]) {
    let wtxn = db.begin_write();
    wtxn.delete::<&[u8], &[u8]>(&TABLE, &key);
    wtxn.commit().unwrap();
}

// ---------------------------------------------------------------------------
// BTree (original shodh-redb) setup
// ---------------------------------------------------------------------------

fn open_btree(path: &Path) -> Database {
    Database::builder()
        .set_cache_size(256 * 1024 * 1024) // 256 MiB to match RocksDB
        .create(path.join("btree.redb"))
        .unwrap()
}

#[allow(dead_code)]
fn reopen_btree(path: &Path) -> Database {
    Database::builder()
        .set_cache_size(256 * 1024 * 1024)
        .open(path.join("btree.redb"))
        .unwrap()
}

#[allow(dead_code)]
fn btree_put(db: &Database, key: &[u8], value: &[u8]) {
    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        table.insert(key, value).unwrap();
    }
    write_txn.commit().unwrap();
}

fn btree_get(db: &Database, key: &[u8]) -> Option<Vec<u8>> {
    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(TABLE).unwrap();
    table.get(key).unwrap().map(|v| v.value().to_vec())
}

fn btree_batch_put(db: &Database, pairs: &[([u8; KEY_SIZE], Vec<u8>)]) {
    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        for (k, v) in pairs {
            table.insert(k.as_slice(), v.as_slice()).unwrap();
        }
    }
    write_txn.commit().unwrap();
}

fn btree_batch_put_nosync(db: &Database, pairs: &[([u8; KEY_SIZE], Vec<u8>)]) {
    let mut write_txn = db.begin_write().unwrap();
    let _ = write_txn.set_durability(Durability::None);
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        for (k, v) in pairs {
            table.insert(k.as_slice(), v.as_slice()).unwrap();
        }
    }
    write_txn.commit().unwrap();
}

#[allow(dead_code)]
fn btree_delete(db: &Database, key: &[u8]) {
    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(TABLE).unwrap();
        let _ = table.remove(key);
    }
    write_txn.commit().unwrap();
}

// ---------------------------------------------------------------------------
// Fill helpers
// ---------------------------------------------------------------------------

fn fill_bftree(db: &BfTreeDatabase, count: usize, rng: &mut fastrand::Rng) -> Vec<[u8; KEY_SIZE]> {
    let mut keys = Vec::with_capacity(count);
    let mut batch = Vec::with_capacity(SCALE_FILL_BATCH);
    let start = Instant::now();
    for i in 0..count {
        let (key, value) = random_pair(rng);
        keys.push(key);
        batch.push((key, value));
        if batch.len() >= SCALE_FILL_BATCH || i == count - 1 {
            let t = Instant::now();
            bftree_batch_put(db, &batch);
            let elapsed = t.elapsed();
            if elapsed > Duration::from_secs(1) {
                eprintln!(
                    "  [bftree] batch commit {}/{} took {:.1}s ({} entries)",
                    i + 1,
                    count,
                    elapsed.as_secs_f64(),
                    batch.len()
                );
            }
            batch.clear();
        }
    }
    eprintln!(
        "  [bftree] fill {} entries: {:.1}s",
        count,
        start.elapsed().as_secs_f64()
    );
    keys
}

#[allow(dead_code)]
fn fill_btree(db: &Database, count: usize, rng: &mut fastrand::Rng) -> Vec<[u8; KEY_SIZE]> {
    let mut keys = Vec::with_capacity(count);
    let mut batch = Vec::with_capacity(SCALE_FILL_BATCH);
    let start = Instant::now();
    for i in 0..count {
        let (key, value) = random_pair(rng);
        keys.push(key);
        batch.push((key, value));
        if batch.len() >= SCALE_FILL_BATCH || i == count - 1 {
            btree_batch_put(db, &batch);
            batch.clear();
        }
    }
    eprintln!(
        "  [btree] fill {} entries: {:.1}s",
        count,
        start.elapsed().as_secs_f64()
    );
    keys
}

fn fill_rocksdb(
    db: &OptimisticTransactionDB,
    count: usize,
    rng: &mut fastrand::Rng,
) -> Vec<[u8; KEY_SIZE]> {
    let mut keys = Vec::with_capacity(count);
    let mut batch = Vec::with_capacity(SCALE_FILL_BATCH);
    for i in 0..count {
        let (key, value) = random_pair(rng);
        keys.push(key);
        batch.push((key, value));
        if batch.len() >= SCALE_FILL_BATCH || i == count - 1 {
            rocksdb_batch_put(db, &batch);
            batch.clear();
        }
    }
    keys
}

// ---------------------------------------------------------------------------
// 1. Latency distribution
// ---------------------------------------------------------------------------

fn bench_latency() {
    println!("\n=== 1. Latency Distribution ===");
    println!("Filling {LATENCY_FILL} entries...");

    let bf_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let rk_dir = TempDir::new_in(current_dir().unwrap()).unwrap();

    let bf_db = open_bftree(bf_dir.path());
    let rk_db = open_rocksdb(rk_dir.path());

    let mut rng = make_rng();
    let bf_keys = fill_bftree(&bf_db, LATENCY_FILL, &mut rng);
    let mut rng = make_rng();
    let rk_keys = fill_rocksdb(&rk_db, LATENCY_FILL, &mut rng);

    // Point reads
    let mut bf_read = LatencyHistogram::with_capacity(LATENCY_SAMPLE_READS);
    let mut rk_read = LatencyHistogram::with_capacity(LATENCY_SAMPLE_READS);
    let mut rng = make_rng();
    for _ in 0..LATENCY_SAMPLE_READS {
        let idx = rng.usize(0..bf_keys.len());
        let t = Instant::now();
        let _ = bftree_get(&bf_db, &bf_keys[idx]);
        bf_read.record(t.elapsed());
    }
    let mut rng = make_rng();
    for _ in 0..LATENCY_SAMPLE_READS {
        let idx = rng.usize(0..rk_keys.len());
        let t = Instant::now();
        let _ = rocksdb_get(&rk_db, &rk_keys[idx]);
        rk_read.record(t.elapsed());
    }
    bf_read.finalize();
    rk_read.finalize();

    // Point writes
    let mut bf_write = LatencyHistogram::with_capacity(LATENCY_SAMPLE_WRITES);
    let mut rk_write = LatencyHistogram::with_capacity(LATENCY_SAMPLE_WRITES);
    let mut rng = make_rng();
    for _ in 0..LATENCY_SAMPLE_WRITES {
        let (key, value) = random_pair(&mut rng);
        let t = Instant::now();
        bftree_put(&bf_db, &key, &value);
        bf_write.record(t.elapsed());
    }
    let mut rng = make_rng();
    for _ in 0..LATENCY_SAMPLE_WRITES {
        let (key, value) = random_pair(&mut rng);
        let t = Instant::now();
        rocksdb_put(&rk_db, &key, &value);
        rk_write.record(t.elapsed());
    }
    bf_write.finalize();
    rk_write.finalize();

    // Range scans
    let mut bf_scan = LatencyHistogram::with_capacity(LATENCY_SAMPLE_SCANS);
    let mut rk_scan = LatencyHistogram::with_capacity(LATENCY_SAMPLE_SCANS);
    for _ in 0..LATENCY_SAMPLE_SCANS {
        let t = Instant::now();
        let rtxn = bf_db.begin_read();
        let mut scan = rtxn.scan_table::<&[u8], &[u8]>(&TABLE).unwrap();
        let mut scan_buf = vec![0u8; 2048];
        for _ in 0..LATENCY_SCAN_LEN {
            if scan.next(&mut scan_buf).is_none() {
                break;
            }
        }
        bf_scan.record(t.elapsed());
    }
    for _ in 0..LATENCY_SAMPLE_SCANS {
        let t = Instant::now();
        let mut iter = rk_db.iterator(rocksdb::IteratorMode::Start);
        for _ in 0..LATENCY_SCAN_LEN {
            if iter.next().is_none() {
                break;
            }
        }
        rk_scan.record(t.elapsed());
    }
    bf_scan.finalize();
    rk_scan.finalize();

    let mut table = Table::new();
    table.set_header(vec!["Operation", "Percentile", "BfTree", "RocksDB"]);
    for (name, bf, rk) in [
        ("Point Read", &bf_read, &rk_read),
        ("Point Write", &bf_write, &rk_write),
        ("Range Scan (10)", &bf_scan, &rk_scan),
    ] {
        for (pname, pval) in [("p50", 50.0), ("p95", 95.0), ("p99", 99.0), ("p99.9", 99.9)] {
            table.add_row(vec![
                Cell::new(name),
                Cell::new(pname),
                Cell::new(fmt_us(bf.percentile(pval))),
                Cell::new(fmt_us(rk.percentile(pval))),
            ]);
        }
    }
    println!("{table}");
}

// ---------------------------------------------------------------------------
// 2. Concurrent write throughput
// ---------------------------------------------------------------------------

fn bench_concurrent_writes() {
    println!("\n=== 2. Concurrent Write Throughput ===");
    println!("Filling {CONCURRENT_FILL} entries as baseline...");

    let thread_counts = [1, 2, 4, 8];
    let mut bf_results: Vec<(usize, f64)> = Vec::new();
    let mut rk_results: Vec<(usize, f64)> = Vec::new();

    for &nthreads in &thread_counts {
        // BfTree
        {
            let dir = TempDir::new_in(current_dir().unwrap()).unwrap();
            let db = Arc::new(open_bftree(dir.path()));
            let mut rng = make_rng();
            fill_bftree(&db, CONCURRENT_FILL, &mut rng);

            let ops_per_thread = CONCURRENT_OPS_PER_THREAD;
            let start = Instant::now();
            let mut handles = Vec::new();
            for t in 0..nthreads {
                let db = db.clone();
                handles.push(thread::spawn(move || {
                    let mut rng = fastrand::Rng::with_seed(RNG_SEED + t as u64 + 100);
                    let mut batch = Vec::with_capacity(CONCURRENT_BATCH_SIZE);
                    for i in 0..ops_per_thread {
                        batch.push(random_pair(&mut rng));
                        if batch.len() >= CONCURRENT_BATCH_SIZE || i == ops_per_thread - 1 {
                            let wtxn = db.begin_write();
                            for (k, v) in &batch {
                                wtxn.insert::<&[u8], &[u8]>(&TABLE, &k.as_slice(), &v.as_slice())
                                    .unwrap();
                            }
                            wtxn.commit().unwrap();
                            batch.clear();
                        }
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
            let elapsed = start.elapsed();
            let total_ops = nthreads * ops_per_thread;
            let ops_sec = total_ops as f64 / elapsed.as_secs_f64();
            bf_results.push((nthreads, ops_sec));
        }

        // RocksDB
        {
            let dir = TempDir::new_in(current_dir().unwrap()).unwrap();
            let db = Arc::new(open_rocksdb(dir.path()));
            let mut rng = make_rng();
            fill_rocksdb(&db, CONCURRENT_FILL, &mut rng);

            let ops_per_thread = CONCURRENT_OPS_PER_THREAD;
            let start = Instant::now();
            let mut handles = Vec::new();
            for t in 0..nthreads {
                let db = db.clone();
                handles.push(thread::spawn(move || {
                    let mut rng = fastrand::Rng::with_seed(RNG_SEED + t as u64 + 100);
                    let mut batch = Vec::with_capacity(CONCURRENT_BATCH_SIZE);
                    for i in 0..ops_per_thread {
                        batch.push(random_pair(&mut rng));
                        if batch.len() >= CONCURRENT_BATCH_SIZE || i == ops_per_thread - 1 {
                            let mut wo = WriteOptions::new();
                            wo.set_sync(true);
                            let txn = db.transaction_opt(&wo, &OptimisticTransactionOptions::new());
                            for (k, v) in &batch {
                                txn.put(k, v).unwrap();
                            }
                            txn.commit().unwrap();
                            batch.clear();
                        }
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
            let elapsed = start.elapsed();
            let total_ops = nthreads * ops_per_thread;
            let ops_sec = total_ops as f64 / elapsed.as_secs_f64();
            rk_results.push((nthreads, ops_sec));
        }
    }

    // Group commit test
    let bf_gc_ops_sec = {
        let dir = TempDir::new_in(current_dir().unwrap()).unwrap();
        let db = Arc::new(open_bftree(dir.path()));
        let mut rng = make_rng();
        fill_bftree(&db, CONCURRENT_FILL, &mut rng);

        let batches: Vec<WriteBatchFn> = (0..GROUP_COMMIT_BATCHES)
            .map(|i| {
                let batch: WriteBatchFn = Box::new(move |txn: &BfTreeDatabaseWriteTxn| {
                    let mut rng = fastrand::Rng::with_seed(RNG_SEED + i as u64 + 200);
                    for _ in 0..GROUP_COMMIT_OPS {
                        let (key, value) = random_pair(&mut rng);
                        txn.insert::<&[u8], &[u8]>(&TABLE, &key.as_slice(), &value.as_slice())?;
                    }
                    Ok(())
                });
                batch
            })
            .collect();

        let total = GROUP_COMMIT_BATCHES * GROUP_COMMIT_OPS;
        let start = Instant::now();
        concurrent_group_commit(db, batches).unwrap();
        let elapsed = start.elapsed();
        total as f64 / elapsed.as_secs_f64()
    };

    // Equivalent RocksDB pattern: parallel threads writing, then sequential commits
    let rk_gc_ops_sec = {
        let dir = TempDir::new_in(current_dir().unwrap()).unwrap();
        let db = Arc::new(open_rocksdb(dir.path()));
        let mut rng = make_rng();
        fill_rocksdb(&db, CONCURRENT_FILL, &mut rng);

        let total = GROUP_COMMIT_BATCHES * GROUP_COMMIT_OPS;
        let start = Instant::now();
        let mut handles = Vec::new();
        for i in 0..GROUP_COMMIT_BATCHES {
            let db = db.clone();
            handles.push(thread::spawn(move || {
                let mut rng = fastrand::Rng::with_seed(RNG_SEED + i as u64 + 200);
                let mut wo = WriteOptions::new();
                wo.set_sync(true);
                let txn = db.transaction_opt(&wo, &OptimisticTransactionOptions::new());
                for _ in 0..GROUP_COMMIT_OPS {
                    let (key, value) = random_pair(&mut rng);
                    txn.put(key, value).unwrap();
                }
                txn.commit().unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let elapsed = start.elapsed();
        total as f64 / elapsed.as_secs_f64()
    };

    let mut table = Table::new();
    table.set_header(vec!["Threads", "BfTree ops/s", "RocksDB ops/s"]);
    for (bf, rk) in bf_results.iter().zip(rk_results.iter()) {
        table.add_row(vec![
            Cell::new(format!("{}", bf.0)),
            Cell::new(format!("{:.0}", bf.1)),
            Cell::new(format!("{:.0}", rk.1)),
        ]);
    }
    table.add_row(vec![
        Cell::new("group_commit (4x5K)"),
        Cell::new(format!("{bf_gc_ops_sec:.0}")),
        Cell::new(format!("{rk_gc_ops_sec:.0}")),
    ]);
    println!("{table}");
}

// ---------------------------------------------------------------------------
// 3. Mixed read/write workload
// ---------------------------------------------------------------------------

fn bench_mixed() {
    println!("\n=== 3. Mixed Read/Write Workload ===");
    println!("Filling {MIXED_FILL} entries...");

    // BfTree
    let bf_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let bf_db = Arc::new(open_bftree(bf_dir.path()));
    let mut rng = make_rng();
    let bf_keys = fill_bftree(&bf_db, MIXED_FILL, &mut rng);
    let bf_keys = Arc::new(bf_keys);

    // RocksDB
    let rk_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let rk_db = Arc::new(open_rocksdb(rk_dir.path()));
    let mut rng = make_rng();
    let rk_keys = fill_rocksdb(&rk_db, MIXED_FILL, &mut rng);
    let rk_keys = Arc::new(rk_keys);

    // Phase 1: Read-only baseline
    println!(
        "Phase 1: Read-only baseline ({MIXED_READER_THREADS} readers, {MIXED_DURATION_SECS}s)..."
    );
    let bf_readonly = run_readers(&bf_db, &bf_keys, MIXED_READER_THREADS, MIXED_DURATION_SECS);
    let rk_readonly =
        run_rocksdb_readers(&rk_db, &rk_keys, MIXED_READER_THREADS, MIXED_DURATION_SECS);

    // Phase 2: Mixed
    println!(
        "Phase 2: Mixed ({MIXED_READER_THREADS} readers + 1 writer, {MIXED_DURATION_SECS}s)..."
    );
    let (bf_mixed_reads, bf_writer_ops) =
        run_mixed_bftree(&bf_db, &bf_keys, MIXED_READER_THREADS, MIXED_DURATION_SECS);
    let (rk_mixed_reads, rk_writer_ops) =
        run_mixed_rocksdb(&rk_db, &rk_keys, MIXED_READER_THREADS, MIXED_DURATION_SECS);

    let bf_degrade = if bf_readonly.p99() > Duration::ZERO {
        (bf_mixed_reads.p99().as_nanos() as f64 / bf_readonly.p99().as_nanos() as f64 - 1.0) * 100.0
    } else {
        0.0
    };
    let rk_degrade = if rk_readonly.p99() > Duration::ZERO {
        (rk_mixed_reads.p99().as_nanos() as f64 / rk_readonly.p99().as_nanos() as f64 - 1.0) * 100.0
    } else {
        0.0
    };

    let mut table = Table::new();
    table.set_header(vec!["Metric", "BfTree", "RocksDB"]);
    table.add_row(vec![
        Cell::new("Read-only p50"),
        Cell::new(fmt_us(bf_readonly.p50())),
        Cell::new(fmt_us(rk_readonly.p50())),
    ]);
    table.add_row(vec![
        Cell::new("Read-only p99"),
        Cell::new(fmt_us(bf_readonly.p99())),
        Cell::new(fmt_us(rk_readonly.p99())),
    ]);
    table.add_row(vec![
        Cell::new("Mixed p50"),
        Cell::new(fmt_us(bf_mixed_reads.p50())),
        Cell::new(fmt_us(rk_mixed_reads.p50())),
    ]);
    table.add_row(vec![
        Cell::new("Mixed p99"),
        Cell::new(fmt_us(bf_mixed_reads.p99())),
        Cell::new(fmt_us(rk_mixed_reads.p99())),
    ]);
    table.add_row(vec![
        Cell::new("p99 degradation"),
        Cell::new(format!("{bf_degrade:+.1}%")),
        Cell::new(format!("{rk_degrade:+.1}%")),
    ]);
    table.add_row(vec![
        Cell::new("Writer ops/s"),
        Cell::new(format!("{bf_writer_ops:.0}")),
        Cell::new(format!("{rk_writer_ops:.0}")),
    ]);
    println!("{table}");
}

fn run_readers(
    db: &Arc<BfTreeDatabase>,
    keys: &Arc<Vec<[u8; KEY_SIZE]>>,
    n_readers: usize,
    duration_secs: u64,
) -> LatencyHistogram {
    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    for t in 0..n_readers {
        let db = db.clone();
        let keys = keys.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            let mut hist = LatencyHistogram::with_capacity(100_000);
            let mut rng = fastrand::Rng::with_seed(RNG_SEED + t as u64 + 50);
            while !stop.load(Ordering::Relaxed) {
                let idx = rng.usize(0..keys.len());
                let t = Instant::now();
                let _ = bftree_get(&db, &keys[idx]);
                hist.record(t.elapsed());
            }
            hist
        }));
    }
    thread::sleep(Duration::from_secs(duration_secs));
    stop.store(true, Ordering::Relaxed);
    let mut merged = LatencyHistogram::with_capacity(0);
    for h in handles {
        let hist = h.join().unwrap();
        merged.merge(&hist);
    }
    merged.finalize();
    merged
}

fn run_rocksdb_readers(
    db: &Arc<OptimisticTransactionDB>,
    keys: &Arc<Vec<[u8; KEY_SIZE]>>,
    n_readers: usize,
    duration_secs: u64,
) -> LatencyHistogram {
    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    for t in 0..n_readers {
        let db = db.clone();
        let keys = keys.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            let mut hist = LatencyHistogram::with_capacity(100_000);
            let mut rng = fastrand::Rng::with_seed(RNG_SEED + t as u64 + 50);
            while !stop.load(Ordering::Relaxed) {
                let idx = rng.usize(0..keys.len());
                let t = Instant::now();
                let _ = rocksdb_get(&db, &keys[idx]);
                hist.record(t.elapsed());
            }
            hist
        }));
    }
    thread::sleep(Duration::from_secs(duration_secs));
    stop.store(true, Ordering::Relaxed);
    let mut merged = LatencyHistogram::with_capacity(0);
    for h in handles {
        let hist = h.join().unwrap();
        merged.merge(&hist);
    }
    merged.finalize();
    merged
}

fn run_mixed_bftree(
    db: &Arc<BfTreeDatabase>,
    keys: &Arc<Vec<[u8; KEY_SIZE]>>,
    n_readers: usize,
    duration_secs: u64,
) -> (LatencyHistogram, f64) {
    let stop = Arc::new(AtomicBool::new(false));
    let writer_ops = Arc::new(AtomicU64::new(0));

    // Readers
    let mut handles = Vec::new();
    for t in 0..n_readers {
        let db = db.clone();
        let keys = keys.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            let mut hist = LatencyHistogram::with_capacity(100_000);
            let mut rng = fastrand::Rng::with_seed(RNG_SEED + t as u64 + 50);
            while !stop.load(Ordering::Relaxed) {
                let idx = rng.usize(0..keys.len());
                let t = Instant::now();
                let _ = bftree_get(&db, &keys[idx]);
                hist.record(t.elapsed());
            }
            hist
        }));
    }

    // Writer
    let db_w = db.clone();
    let stop_w = stop.clone();
    let wops = writer_ops.clone();
    let writer = thread::spawn(move || {
        let mut rng = fastrand::Rng::with_seed(RNG_SEED + 999);
        let batch_sz = 100;
        while !stop_w.load(Ordering::Relaxed) {
            let wtxn = db_w.begin_write();
            for _ in 0..batch_sz {
                let (key, value) = random_pair(&mut rng);
                wtxn.insert::<&[u8], &[u8]>(&TABLE, &key.as_slice(), &value.as_slice())
                    .unwrap();
            }
            wtxn.commit().unwrap();
            wops.fetch_add(batch_sz, Ordering::Relaxed);
        }
    });

    thread::sleep(Duration::from_secs(duration_secs));
    stop.store(true, Ordering::Relaxed);

    let mut merged = LatencyHistogram::with_capacity(0);
    for h in handles {
        let hist = h.join().unwrap();
        merged.merge(&hist);
    }
    writer.join().unwrap();
    merged.finalize();

    let ops = writer_ops.load(Ordering::Relaxed) as f64 / duration_secs as f64;
    (merged, ops)
}

fn run_mixed_rocksdb(
    db: &Arc<OptimisticTransactionDB>,
    keys: &Arc<Vec<[u8; KEY_SIZE]>>,
    n_readers: usize,
    duration_secs: u64,
) -> (LatencyHistogram, f64) {
    let stop = Arc::new(AtomicBool::new(false));
    let writer_ops = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();
    for t in 0..n_readers {
        let db = db.clone();
        let keys = keys.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            let mut hist = LatencyHistogram::with_capacity(100_000);
            let mut rng = fastrand::Rng::with_seed(RNG_SEED + t as u64 + 50);
            while !stop.load(Ordering::Relaxed) {
                let idx = rng.usize(0..keys.len());
                let t = Instant::now();
                let _ = rocksdb_get(&db, &keys[idx]);
                hist.record(t.elapsed());
            }
            hist
        }));
    }

    let db_w = db.clone();
    let stop_w = stop.clone();
    let wops = writer_ops.clone();
    let writer = thread::spawn(move || {
        let mut rng = fastrand::Rng::with_seed(RNG_SEED + 999);
        let batch_sz = 100usize;
        while !stop_w.load(Ordering::Relaxed) {
            let mut wo = WriteOptions::new();
            wo.set_sync(true);
            let txn = db_w.transaction_opt(&wo, &OptimisticTransactionOptions::new());
            for _ in 0..batch_sz {
                let (key, value) = random_pair(&mut rng);
                txn.put(key, value).unwrap();
            }
            txn.commit().unwrap();
            wops.fetch_add(batch_sz as u64, Ordering::Relaxed);
        }
    });

    thread::sleep(Duration::from_secs(duration_secs));
    stop.store(true, Ordering::Relaxed);

    let mut merged = LatencyHistogram::with_capacity(0);
    for h in handles {
        let hist = h.join().unwrap();
        merged.merge(&hist);
    }
    writer.join().unwrap();
    merged.finalize();

    let ops = writer_ops.load(Ordering::Relaxed) as f64 / duration_secs as f64;
    (merged, ops)
}

// ---------------------------------------------------------------------------
// 4. Write amplification
// ---------------------------------------------------------------------------

fn bench_write_amp() {
    println!("\n=== 4. Write Amplification ===");
    println!("Inserting {WRITE_AMP_ENTRIES} entries...");

    let bf_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let rk_dir = TempDir::new_in(current_dir().unwrap()).unwrap();

    let user_bytes = (WRITE_AMP_ENTRIES * ENTRY_SIZE) as f64;

    // BfTree
    let bf_amp = {
        let db = open_bftree(bf_dir.path());
        let before = proc_write_bytes();
        let mut rng = make_rng();
        fill_bftree(&db, WRITE_AMP_ENTRIES, &mut rng);
        let after = proc_write_bytes();
        match (before, after) {
            (Some(b), Some(a)) => (a - b) as f64 / user_bytes,
            _ => {
                let sz = database_size(bf_dir.path()) as f64;
                sz / user_bytes
            }
        }
    };

    // RocksDB
    let rk_amp = {
        let db = open_rocksdb(rk_dir.path());
        let before = proc_write_bytes();
        let mut rng = make_rng();
        fill_rocksdb(&db, WRITE_AMP_ENTRIES, &mut rng);
        let after = proc_write_bytes();
        match (before, after) {
            (Some(b), Some(a)) => (a - b) as f64 / user_bytes,
            _ => {
                let sz = database_size(rk_dir.path()) as f64;
                sz / user_bytes
            }
        }
    };

    let mut table = Table::new();
    table.set_header(vec!["Metric", "BfTree", "RocksDB"]);
    table.add_row(vec![
        Cell::new("Write amplification"),
        Cell::new(format!("{bf_amp:.2}x")),
        Cell::new(format!("{rk_amp:.2}x")),
    ]);
    println!("{table}");
}

// ---------------------------------------------------------------------------
// 5. Space amplification
// ---------------------------------------------------------------------------

fn bench_space_amp() {
    println!("\n=== 5. Space Amplification ===");
    println!("Filling {SPACE_AMP_ENTRIES} entries...");

    let bf_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let rk_dir = TempDir::new_in(current_dir().unwrap()).unwrap();

    let bf_db = open_bftree(bf_dir.path());
    let rk_db = open_rocksdb(rk_dir.path());

    let mut rng = make_rng();
    let bf_keys = fill_bftree(&bf_db, SPACE_AMP_ENTRIES, &mut rng);
    let mut rng = make_rng();
    let rk_keys = fill_rocksdb(&rk_db, SPACE_AMP_ENTRIES, &mut rng);

    let full_user = (SPACE_AMP_ENTRIES * ENTRY_SIZE) as f64;
    let bf_full = database_size(bf_dir.path()) as f64 / full_user;
    let rk_full = database_size(rk_dir.path()) as f64 / full_user;

    // Delete half
    let half = SPACE_AMP_ENTRIES / 2;
    for key in &bf_keys[..half] {
        bftree_delete(&bf_db, key);
    }
    for key in &rk_keys[..half] {
        rocksdb_delete(&rk_db, key);
    }

    let remaining_user = ((SPACE_AMP_ENTRIES - half) * ENTRY_SIZE) as f64;
    let bf_post = database_size(bf_dir.path()) as f64 / remaining_user;
    let rk_post = database_size(rk_dir.path()) as f64 / remaining_user;

    let mut table = Table::new();
    table.set_header(vec!["Metric", "BfTree", "RocksDB"]);
    table.add_row(vec![
        Cell::new("Full (db/user)"),
        Cell::new(format!("{bf_full:.2}x")),
        Cell::new(format!("{rk_full:.2}x")),
    ]);
    table.add_row(vec![
        Cell::new("Post-delete (db/user)"),
        Cell::new(format!("{bf_post:.2}x")),
        Cell::new(format!("{rk_post:.2}x")),
    ]);
    println!("{table}");
}

// ---------------------------------------------------------------------------
// 6. Recovery time
// ---------------------------------------------------------------------------

fn bench_recovery() {
    println!("\n=== 6. Recovery Time ===");
    println!("Filling {RECOVERY_ENTRIES} entries...");

    let bf_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let rk_dir = TempDir::new_in(current_dir().unwrap()).unwrap();

    // BfTree: fill and close
    let bf_first_key = {
        let db = open_bftree(bf_dir.path());
        let mut rng = make_rng();
        let keys = fill_bftree(&db, RECOVERY_ENTRIES, &mut rng);
        keys[0]
    };

    // RocksDB: fill and close
    let rk_first_key = {
        let db = open_rocksdb(rk_dir.path());
        let mut rng = make_rng();
        let keys = fill_rocksdb(&db, RECOVERY_ENTRIES, &mut rng);
        keys[0]
    };

    // BfTree recovery
    let bf_recovery_ms = {
        let start = Instant::now();
        let db = reopen_bftree(bf_dir.path());
        let _ = bftree_get(&db, &bf_first_key);
        start.elapsed().as_secs_f64() * 1000.0
    };

    // RocksDB recovery
    let rk_recovery_ms = {
        let start = Instant::now();
        let db = open_rocksdb(rk_dir.path());
        let _ = rocksdb_get(&db, &rk_first_key);
        start.elapsed().as_secs_f64() * 1000.0
    };

    let mut table = Table::new();
    table.set_header(vec!["Metric", "BfTree", "RocksDB"]);
    table.add_row(vec![
        Cell::new("Recovery (open + read) ms"),
        Cell::new(format!("{bf_recovery_ms:.1}")),
        Cell::new(format!("{rk_recovery_ms:.1}")),
    ]);
    println!("{table}");
}

// ---------------------------------------------------------------------------
// 7. Throughput at scale
// ---------------------------------------------------------------------------

fn bench_throughput_at_scale() {
    println!("\n=== 7. Throughput at Scale (Sync) ===");
    println!("All engines: fsync per commit");

    let bf_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let bt_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let rk_dir = TempDir::new_in(current_dir().unwrap()).unwrap();

    let bf_db = open_bftree(bf_dir.path());
    let bt_db = open_btree(bt_dir.path());
    let rk_db = open_rocksdb(rk_dir.path());

    let mut bf_rng = make_rng();
    let mut bt_rng = make_rng();
    let mut rk_rng = make_rng();
    let mut bf_keys: Vec<[u8; KEY_SIZE]> = Vec::new();
    let mut bt_keys: Vec<[u8; KEY_SIZE]> = Vec::new();
    let mut rk_keys: Vec<[u8; KEY_SIZE]> = Vec::new();

    let mut bf_read_results = Vec::new();
    let mut bf_write_results = Vec::new();
    let mut bt_read_results = Vec::new();
    let mut bt_write_results = Vec::new();
    let mut rk_read_results = Vec::new();
    let mut rk_write_results = Vec::new();

    let mut bf_filled = 0usize;
    let mut bt_filled = 0usize;
    let mut rk_filled = 0usize;

    for &level in SCALE_LEVELS {
        println!("  Filling to {level}...");

        // Fill BfTree to level
        while bf_filled < level {
            let batch_end = (bf_filled + SCALE_FILL_BATCH).min(level);
            let batch_sz = batch_end - bf_filled;
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                let (key, value) = random_pair(&mut bf_rng);
                bf_keys.push(key);
                batch.push((key, value));
            }
            bftree_batch_put(&bf_db, &batch);
            bf_filled = batch_end;
        }
        let _ = bf_db.snapshot();

        // Fill BTree to level
        while bt_filled < level {
            let batch_end = (bt_filled + SCALE_FILL_BATCH).min(level);
            let batch_sz = batch_end - bt_filled;
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                let (key, value) = random_pair(&mut bt_rng);
                bt_keys.push(key);
                batch.push((key, value));
            }
            btree_batch_put(&bt_db, &batch);
            bt_filled = batch_end;
        }

        // Fill RocksDB to level
        while rk_filled < level {
            let batch_end = (rk_filled + SCALE_FILL_BATCH).min(level);
            let batch_sz = batch_end - rk_filled;
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                let (key, value) = random_pair(&mut rk_rng);
                rk_keys.push(key);
                batch.push((key, value));
            }
            rocksdb_batch_put(&rk_db, &batch);
            rk_filled = batch_end;
        }

        // Measure read throughput -- BfTree
        let mut read_rng = fastrand::Rng::with_seed(42);
        let start = Instant::now();
        for _ in 0..SCALE_SAMPLE_OPS {
            let idx = read_rng.usize(0..bf_keys.len());
            let _ = bftree_get(&bf_db, &bf_keys[idx]);
        }
        let bf_read_ops = SCALE_SAMPLE_OPS as f64 / start.elapsed().as_secs_f64();
        bf_read_results.push((level, bf_read_ops));

        // Measure read throughput -- BTree
        let mut read_rng = fastrand::Rng::with_seed(42);
        let start = Instant::now();
        for _ in 0..SCALE_SAMPLE_OPS {
            let idx = read_rng.usize(0..bt_keys.len());
            let _ = btree_get(&bt_db, &bt_keys[idx]);
        }
        let bt_read_ops = SCALE_SAMPLE_OPS as f64 / start.elapsed().as_secs_f64();
        bt_read_results.push((level, bt_read_ops));

        // Measure read throughput -- RocksDB
        let mut read_rng = fastrand::Rng::with_seed(42);
        let start = Instant::now();
        for _ in 0..SCALE_SAMPLE_OPS {
            let idx = read_rng.usize(0..rk_keys.len());
            let _ = rocksdb_get(&rk_db, &rk_keys[idx]);
        }
        let rk_read_ops = SCALE_SAMPLE_OPS as f64 / start.elapsed().as_secs_f64();
        rk_read_results.push((level, rk_read_ops));

        // Measure write throughput -- BfTree
        let mut write_rng = fastrand::Rng::with_seed(77);
        let start = Instant::now();
        let mut done = 0;
        while done < SCALE_SAMPLE_OPS {
            let batch_sz = SCALE_BATCH_SIZE.min(SCALE_SAMPLE_OPS - done);
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                batch.push(random_pair(&mut write_rng));
            }
            bftree_batch_put(&bf_db, &batch);
            done += batch_sz;
        }
        let bf_write_ops = SCALE_SAMPLE_OPS as f64 / start.elapsed().as_secs_f64();
        bf_write_results.push((level, bf_write_ops));

        // Measure write throughput -- BTree
        let mut write_rng = fastrand::Rng::with_seed(77);
        let start = Instant::now();
        let mut done = 0;
        while done < SCALE_SAMPLE_OPS {
            let batch_sz = SCALE_BATCH_SIZE.min(SCALE_SAMPLE_OPS - done);
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                batch.push(random_pair(&mut write_rng));
            }
            btree_batch_put(&bt_db, &batch);
            done += batch_sz;
        }
        let bt_write_ops = SCALE_SAMPLE_OPS as f64 / start.elapsed().as_secs_f64();
        bt_write_results.push((level, bt_write_ops));

        // Measure write throughput -- RocksDB
        let mut write_rng = fastrand::Rng::with_seed(77);
        let start = Instant::now();
        let mut done = 0;
        while done < SCALE_SAMPLE_OPS {
            let batch_sz = SCALE_BATCH_SIZE.min(SCALE_SAMPLE_OPS - done);
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                batch.push(random_pair(&mut write_rng));
            }
            rocksdb_batch_put(&rk_db, &batch);
            done += batch_sz;
        }
        let rk_write_ops = SCALE_SAMPLE_OPS as f64 / start.elapsed().as_secs_f64();
        rk_write_results.push((level, rk_write_ops));
    }

    let mut table = Table::new();
    table.set_header(vec![
        "Level",
        "BfTree read/s",
        "BTree read/s",
        "RocksDB read/s",
        "BfTree write/s",
        "BTree write/s",
        "RocksDB write/s",
    ]);
    for i in 0..SCALE_LEVELS.len() {
        table.add_row(vec![
            Cell::new(format!("{}K", SCALE_LEVELS[i] / 1000)),
            Cell::new(format!("{:.0}", bf_read_results[i].1)),
            Cell::new(format!("{:.0}", bt_read_results[i].1)),
            Cell::new(format!("{:.0}", rk_read_results[i].1)),
            Cell::new(format!("{:.0}", bf_write_results[i].1)),
            Cell::new(format!("{:.0}", bt_write_results[i].1)),
            Cell::new(format!("{:.0}", rk_write_results[i].1)),
        ]);
    }
    println!("{table}");
}

// ---------------------------------------------------------------------------
// 8. Throughput at scale (Periodic durability -- matches RocksDB default)
// ---------------------------------------------------------------------------

fn bench_throughput_at_scale_periodic() {
    println!("\n=== 8. Throughput at Scale (No fsync) ===");
    println!("BfTree: DurabilityMode::Periodic | BTree: Durability::None | RocksDB: sync=false");

    let bf_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let bt_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let rk_dir = TempDir::new_in(current_dir().unwrap()).unwrap();

    let bf_db = open_bftree_periodic(bf_dir.path());
    let bt_db = open_btree(bt_dir.path());
    let rk_db: OptimisticTransactionDB = {
        let opts = rocksdb_opts();
        OptimisticTransactionDB::open(&opts, rk_dir.path()).unwrap()
    };

    let mut bf_rng = make_rng();
    let mut bt_rng = make_rng();
    let mut rk_rng = make_rng();
    let mut bf_keys: Vec<[u8; KEY_SIZE]> = Vec::new();
    let mut bt_keys: Vec<[u8; KEY_SIZE]> = Vec::new();
    let mut rk_keys: Vec<[u8; KEY_SIZE]> = Vec::new();

    let mut bf_write_results = Vec::new();
    let mut bt_write_results = Vec::new();
    let mut rk_write_results = Vec::new();

    let mut bf_filled = 0usize;
    let mut bt_filled = 0usize;
    let mut rk_filled = 0usize;

    for &level in SCALE_LEVELS {
        println!("  Filling to {level}...");

        // Fill BfTree
        while bf_filled < level {
            let batch_end = (bf_filled + SCALE_FILL_BATCH).min(level);
            let batch_sz = batch_end - bf_filled;
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                let (key, value) = random_pair(&mut bf_rng);
                bf_keys.push(key);
                batch.push((key, value));
            }
            bftree_batch_put(&bf_db, &batch);
            bf_filled = batch_end;
        }
        let _ = bf_db.snapshot();

        // Fill BTree (no-sync for fill)
        while bt_filled < level {
            let batch_end = (bt_filled + SCALE_FILL_BATCH).min(level);
            let batch_sz = batch_end - bt_filled;
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                let (key, value) = random_pair(&mut bt_rng);
                bt_keys.push(key);
                batch.push((key, value));
            }
            btree_batch_put_nosync(&bt_db, &batch);
            bt_filled = batch_end;
        }

        // Fill RocksDB (no sync)
        while rk_filled < level {
            let batch_end = (rk_filled + SCALE_FILL_BATCH).min(level);
            let batch_sz = batch_end - rk_filled;
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                let (key, value) = random_pair(&mut rk_rng);
                rk_keys.push(key);
                batch.push((key, value));
            }
            let wo = WriteOptions::new();
            let to = OptimisticTransactionOptions::new();
            let txn = rk_db.transaction_opt(&wo, &to);
            for (k, v) in &batch {
                txn.put(k, v).unwrap();
            }
            txn.commit().unwrap();
            rk_filled = batch_end;
        }

        // Measure write throughput -- BfTree (periodic)
        let mut write_rng = fastrand::Rng::with_seed(77);
        let start = Instant::now();
        let mut done = 0;
        while done < SCALE_SAMPLE_OPS {
            let batch_sz = SCALE_BATCH_SIZE.min(SCALE_SAMPLE_OPS - done);
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                batch.push(random_pair(&mut write_rng));
            }
            bftree_batch_put(&bf_db, &batch);
            done += batch_sz;
        }
        let bf_write_ops = SCALE_SAMPLE_OPS as f64 / start.elapsed().as_secs_f64();
        bf_write_results.push((level, bf_write_ops));

        // Measure write throughput -- BTree (no-sync)
        let mut write_rng = fastrand::Rng::with_seed(77);
        let start = Instant::now();
        let mut done = 0;
        while done < SCALE_SAMPLE_OPS {
            let batch_sz = SCALE_BATCH_SIZE.min(SCALE_SAMPLE_OPS - done);
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                batch.push(random_pair(&mut write_rng));
            }
            btree_batch_put_nosync(&bt_db, &batch);
            done += batch_sz;
        }
        let bt_write_ops = SCALE_SAMPLE_OPS as f64 / start.elapsed().as_secs_f64();
        bt_write_results.push((level, bt_write_ops));

        // Measure write throughput -- RocksDB (no-sync)
        let mut write_rng = fastrand::Rng::with_seed(77);
        let start = Instant::now();
        let mut done = 0;
        while done < SCALE_SAMPLE_OPS {
            let batch_sz = SCALE_BATCH_SIZE.min(SCALE_SAMPLE_OPS - done);
            let mut batch = Vec::with_capacity(batch_sz);
            for _ in 0..batch_sz {
                batch.push(random_pair(&mut write_rng));
            }
            let wo = WriteOptions::new();
            let txn = rk_db.transaction_opt(&wo, &OptimisticTransactionOptions::new());
            for (k, v) in &batch {
                txn.put(k, v).unwrap();
            }
            txn.commit().unwrap();
            done += batch_sz;
        }
        let rk_write_ops = SCALE_SAMPLE_OPS as f64 / start.elapsed().as_secs_f64();
        rk_write_results.push((level, rk_write_ops));
    }

    let mut table = Table::new();
    table.set_header(vec![
        "Level",
        "BfTree write/s",
        "BTree write/s",
        "RocksDB write/s",
    ]);
    for i in 0..SCALE_LEVELS.len() {
        let bf_vs_rk = bf_write_results[i].1 / rk_write_results[i].1;
        let bf_vs_bt = bf_write_results[i].1 / bt_write_results[i].1;
        table.add_row(vec![
            Cell::new(format!("{}K", SCALE_LEVELS[i] / 1000)),
            Cell::new(format!(
                "{:.0} ({bf_vs_rk:.1}x RocksDB, {bf_vs_bt:.1}x BTree)",
                bf_write_results[i].1
            )),
            Cell::new(format!("{:.0}", bt_write_results[i].1)),
            Cell::new(format!("{:.0}", rk_write_results[i].1)),
        ]);
    }
    println!("{table}");
}

// ---------------------------------------------------------------------------
// 9. Mixed read/write (Periodic durability)
// ---------------------------------------------------------------------------

fn bench_mixed_periodic() {
    println!("\n=== 9. Mixed Read/Write (Periodic Durability) ===");
    println!("BfTree: DurabilityMode::Periodic | RocksDB: sync=false");

    // BfTree periodic
    let bf_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let bf_db = Arc::new(open_bftree_periodic(bf_dir.path()));
    let mut rng = make_rng();
    let bf_keys = fill_bftree(&bf_db, MIXED_FILL, &mut rng);
    let bf_keys = Arc::new(bf_keys);

    // RocksDB no-sync
    let rk_dir = TempDir::new_in(current_dir().unwrap()).unwrap();
    let rk_db: Arc<OptimisticTransactionDB> = Arc::new({
        let opts = rocksdb_opts();
        OptimisticTransactionDB::open(&opts, rk_dir.path()).unwrap()
    });
    let mut rng = make_rng();
    let rk_keys = fill_rocksdb(&rk_db, MIXED_FILL, &mut rng);
    let rk_keys = Arc::new(rk_keys);

    // Read-only baseline
    let bf_readonly = run_readers(&bf_db, &bf_keys, MIXED_READER_THREADS, MIXED_DURATION_SECS);
    let rk_readonly =
        run_rocksdb_readers(&rk_db, &rk_keys, MIXED_READER_THREADS, MIXED_DURATION_SECS);

    // Mixed: BfTree periodic writer
    let (bf_mixed_reads, bf_writer_ops) =
        run_mixed_bftree(&bf_db, &bf_keys, MIXED_READER_THREADS, MIXED_DURATION_SECS);

    // Mixed: RocksDB no-sync writer
    let (rk_mixed_reads, rk_writer_ops) = {
        let stop = Arc::new(AtomicBool::new(false));
        let writer_ops = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();

        for t in 0..MIXED_READER_THREADS {
            let db = rk_db.clone();
            let keys = rk_keys.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                let mut hist = LatencyHistogram::with_capacity(100_000);
                let mut rng = fastrand::Rng::with_seed(RNG_SEED + t as u64 + 50);
                while !stop.load(Ordering::Relaxed) {
                    let idx = rng.usize(0..keys.len());
                    let t = Instant::now();
                    let _ = rocksdb_get(&db, &keys[idx]);
                    hist.record(t.elapsed());
                }
                hist
            }));
        }

        let db_w = rk_db.clone();
        let stop_w = stop.clone();
        let wops = writer_ops.clone();
        let writer = thread::spawn(move || {
            let mut rng = fastrand::Rng::with_seed(RNG_SEED + 999);
            let batch_sz = 100usize;
            while !stop_w.load(Ordering::Relaxed) {
                let wo = WriteOptions::new(); // sync=false
                let txn = db_w.transaction_opt(&wo, &OptimisticTransactionOptions::new());
                for _ in 0..batch_sz {
                    let (key, value) = random_pair(&mut rng);
                    txn.put(key, value).unwrap();
                }
                txn.commit().unwrap();
                wops.fetch_add(batch_sz as u64, Ordering::Relaxed);
            }
        });

        thread::sleep(Duration::from_secs(MIXED_DURATION_SECS));
        stop.store(true, Ordering::Relaxed);

        let mut merged = LatencyHistogram::with_capacity(0);
        for h in handles {
            let hist = h.join().unwrap();
            merged.merge(&hist);
        }
        writer.join().unwrap();
        merged.finalize();

        let ops = writer_ops.load(Ordering::Relaxed) as f64 / MIXED_DURATION_SECS as f64;
        (merged, ops)
    };

    let bf_degrade = if bf_readonly.p99() > Duration::ZERO {
        (bf_mixed_reads.p99().as_nanos() as f64 / bf_readonly.p99().as_nanos() as f64 - 1.0) * 100.0
    } else {
        0.0
    };
    let rk_degrade = if rk_readonly.p99() > Duration::ZERO {
        (rk_mixed_reads.p99().as_nanos() as f64 / rk_readonly.p99().as_nanos() as f64 - 1.0) * 100.0
    } else {
        0.0
    };

    let mut table = Table::new();
    table.set_header(vec!["Metric", "BfTree (Periodic)", "RocksDB (no-sync)"]);
    table.add_row(vec![
        Cell::new("Read-only p50"),
        Cell::new(fmt_us(bf_readonly.p50())),
        Cell::new(fmt_us(rk_readonly.p50())),
    ]);
    table.add_row(vec![
        Cell::new("Read-only p99"),
        Cell::new(fmt_us(bf_readonly.p99())),
        Cell::new(fmt_us(rk_readonly.p99())),
    ]);
    table.add_row(vec![
        Cell::new("Mixed p50"),
        Cell::new(fmt_us(bf_mixed_reads.p50())),
        Cell::new(fmt_us(rk_mixed_reads.p50())),
    ]);
    table.add_row(vec![
        Cell::new("Mixed p99"),
        Cell::new(fmt_us(bf_mixed_reads.p99())),
        Cell::new(fmt_us(rk_mixed_reads.p99())),
    ]);
    table.add_row(vec![
        Cell::new("p99 degradation"),
        Cell::new(format!("{bf_degrade:+.1}%")),
        Cell::new(format!("{rk_degrade:+.1}%")),
    ]);
    table.add_row(vec![
        Cell::new("Writer ops/s"),
        Cell::new(format!("{bf_writer_ops:.0}")),
        Cell::new(format!("{rk_writer_ops:.0}")),
    ]);
    println!("{table}");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== shodh-redb BfTree vs RocksDB Benchmark ===");
    println!("Platform: {}", std::env::consts::ARCH);
    println!(
        "CPUs: {}",
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    println!("Key: {KEY_SIZE}B, Value: {VALUE_SIZE}B, Entry: {ENTRY_SIZE}B");
    println!();

    bench_latency();
    bench_concurrent_writes();
    bench_mixed();
    bench_write_amp();
    bench_space_amp();
    bench_recovery();
    bench_throughput_at_scale();
    bench_throughput_at_scale_periodic();
    bench_mixed_periodic();

    println!("\nDone.");
}
