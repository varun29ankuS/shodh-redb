//! End-to-end drive of the multi-modal surface, as a user would meet it.
//!
//! Exercises KV, blobs, IVF-PQ train/insert/search, backup, and reopen-verify
//! against a real file on disk -- the paths the CLI can then inspect.
//!
//! Run: `cargo run --example smoke_drive -- <db-path> <backup-path>`

use shodh_redb::{
    ContentType, Database, DistanceMetric, IvfPqIndexDefinition, ReadableDatabase, SearchParams,
    StoreOptions, TableDefinition, VerifyLevel,
};

const KV: TableDefinition<u64, &str> = TableDefinition::new("demo_kv");
const INDEX: IvfPqIndexDefinition =
    IvfPqIndexDefinition::new("demo_vec", 32, 16, 8, DistanceMetric::EuclideanSq)
        .with_raw_vectors()
        .with_nprobe(8);

fn embedding(seed: u64) -> Vec<f32> {
    (0..32)
        .map(|i| {
            let x = (seed.wrapping_mul(2_654_435_761).wrapping_add(i * 40_503)) % 1000;
            x as f32 / 1000.0
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let db_path = args.get(1).cloned().unwrap_or_else(|| "demo.redb".into());
    let backup_path = args.get(2).cloned().unwrap_or_else(|| "demo.bak".into());

    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_file(&backup_path);

    let db = Database::create(&db_path)?;
    println!("created {db_path}");

    // 1. Key-value
    let txn = db.begin_write()?;
    {
        let mut t = txn.open_table(KV)?;
        for i in 0..5_000u64 {
            t.insert(i, format!("value-{i}").as_str())?;
        }
    }
    txn.commit()?;
    println!("kv:      inserted 5000 rows");

    // 2. Blobs
    let txn = db.begin_write()?;
    let mut blob_ids = Vec::new();
    {
        for i in 0..20u64 {
            let payload = vec![(i % 251) as u8; 64 * 1024];
            let id = txn.store_blob(
                &payload,
                ContentType::OctetStream,
                &format!("blob-{i}"),
                StoreOptions::default(),
            )?;
            blob_ids.push(id);
        }
    }
    txn.commit()?;
    println!("blobs:   wrote {} x 64KiB", blob_ids.len());

    // 3. IVF-PQ train + insert + search
    let txn = db.begin_write()?;
    {
        let mut idx = txn.open_ivfpq_index(&INDEX)?;
        let training: Vec<(u64, Vec<f32>)> = (0..2_000u64).map(|i| (i, embedding(i))).collect();
        idx.train(training.into_iter(), 15)?;
        for i in 0..2_000u64 {
            idx.insert(i, &embedding(i))?;
        }
    }
    txn.commit()?;
    println!("vectors: trained + inserted 2000");

    let read = db.begin_read()?;
    {
        let idx = read.open_ivfpq_index(&INDEX)?;
        let q = embedding(42);
        let hits = idx.search(&read, &q, &SearchParams::top_k(5))?;
        let ids: Vec<u64> = hits.iter().map(|h| h.key).collect();
        println!("search:  top-5 for id 42 -> {ids:?}");
        assert!(ids.contains(&42), "exact match must be found, got {ids:?}");
    }
    drop(read);

    // 4. Backup (now holds the write lock) and verify the copy
    db.backup(&backup_path)?;
    println!("backup:  wrote {backup_path}");

    let report = Database::verify_backup(&backup_path, VerifyLevel::Full)?;
    println!(
        "verify:  valid={} pages_checked={} corrupt={} structural={:?}",
        report.valid, report.pages_checked, report.pages_corrupt, report.structural_valid
    );
    assert!(report.valid, "backup must verify clean: {report:?}");

    // 5. Reopen the backup and confirm the data survived the round trip
    drop(db);
    let restored = Database::open(&backup_path)?;
    let read = restored.begin_read()?;
    {
        let t = read.open_table(KV)?;
        let v = t.get(&4_999u64)?.expect("row 4999 must survive restore");
        assert_eq!(v.value(), "value-4999");
        println!("restore: row 4999 = {:?}", v.value());
    }
    println!("\nOK -- all stages passed");
    Ok(())
}
