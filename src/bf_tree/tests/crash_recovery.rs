// Crash recovery tests for BfTree.
//
// These tests validate that the BfTree correctly handles and recovers from
// various failure scenarios: truncated WAL, corrupt snapshot, mid-operation
// crashes, and concurrent snapshot safety.

#![cfg(all(test, feature = "std", not(feature = "shuttle")))]

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use crate::bf_tree::config::WalConfig;
use crate::bf_tree::nodes::leaf_node::LeafReadResult;
use crate::bf_tree::utils::test_util::install_value_to_buffer;
use crate::bf_tree::{BfTree, Config, StorageBackend};

const MIN_RECORD_SIZE: usize = 64;
const MAX_RECORD_SIZE: usize = 2048;
const LEAF_PAGE_SIZE: usize = 8192;
const CB_SIZE: usize = LEAF_PAGE_SIZE * 64;
const WAL_SEGMENT_SIZE: usize = 1024 * 1024;
const KEY_LEN: usize = MIN_RECORD_SIZE / 2;

/// Helper: create a config for file-backed tree with WAL.
fn make_config(snapshot_path: &std::path::Path, wal_path: &std::path::Path) -> Config {
    let mut config = Config::new(snapshot_path, CB_SIZE);
    config.storage_backend(StorageBackend::Std);
    config.cb_min_record_size = MIN_RECORD_SIZE;
    config.cb_max_record_size = MAX_RECORD_SIZE;
    config.leaf_page_size = LEAF_PAGE_SIZE;
    config.max_fence_len = MAX_RECORD_SIZE;
    let mut wal_config = WalConfig::new(wal_path);
    wal_config.segment_size(WAL_SEGMENT_SIZE);
    wal_config.flush_interval(std::time::Duration::from_micros(1));
    config.enable_write_ahead_log(Arc::new(wal_config));
    config
}

/// Helper: create a config without WAL (snapshot-only).
fn make_config_no_wal(snapshot_path: &std::path::Path) -> Config {
    let mut config = Config::new(snapshot_path, CB_SIZE);
    config.storage_backend(StorageBackend::Std);
    config.cb_min_record_size = MIN_RECORD_SIZE;
    config.cb_max_record_size = MAX_RECORD_SIZE;
    config.leaf_page_size = LEAF_PAGE_SIZE;
    config.max_fence_len = MAX_RECORD_SIZE;
    config
}

/// Helper: write TOML config file for recovery().
fn write_config_toml(config_path: &std::path::Path, snapshot_path: &std::path::Path) {
    let max_key_len = MAX_RECORD_SIZE / 2;
    let config_toml = format!(
        "cb_size_byte = {CB_SIZE}\n\
         cb_min_record_size = {MIN_RECORD_SIZE}\n\
         cb_max_record_size = {MAX_RECORD_SIZE}\n\
         cb_max_key_len = {max_key_len}\n\
         leaf_page_size = {LEAF_PAGE_SIZE}\n\
         index_file_path = \"{}\"\n\
         backend_storage = \"disk\"\n\
         read_promotion_rate = 50\n\
         write_load_full_page = true\n\
         cache_only = false\n",
        snapshot_path.to_string_lossy().replace('\\', "\\\\"),
    );
    std::fs::write(config_path, &config_toml).unwrap();
}

/// Helper: insert N records with deterministic keys.
fn insert_records(tree: &BfTree, start: usize, count: usize) {
    let mut key_buffer = vec![0usize; KEY_LEN / 8];
    for r in start..(start + count) {
        let key = install_value_to_buffer(&mut key_buffer, r);
        tree.insert(key, key);
    }
}

/// Helper: verify N records are present.
fn verify_records(tree: &BfTree, start: usize, count: usize) {
    let mut key_buffer = vec![0usize; KEY_LEN / 8];
    let mut out_buffer = vec![0u8; KEY_LEN];
    for r in start..(start + count) {
        let key = install_value_to_buffer(&mut key_buffer, r);
        match tree.read(key, &mut out_buffer) {
            LeafReadResult::Found(v) => {
                assert_eq!(v as usize, KEY_LEN, "wrong value size for key {r}");
                assert_eq!(&out_buffer[..KEY_LEN], key, "wrong value for key {r}");
            }
            other => panic!("key {r} not found: {other:?}"),
        }
    }
}

/// Helper: verify a key is NOT present (either NotFound or Deleted).
fn verify_key_absent(tree: &BfTree, idx: usize) {
    let mut key_buffer = vec![0usize; KEY_LEN / 8];
    let mut out_buffer = vec![0u8; KEY_LEN];
    let key = install_value_to_buffer(&mut key_buffer, idx);
    match tree.read(key, &mut out_buffer) {
        LeafReadResult::NotFound | LeafReadResult::Deleted => {}
        other => panic!("key {idx} should be absent but got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 1: Truncated WAL -- corrupt tail should be skipped during recovery
// ---------------------------------------------------------------------------
#[test]
fn truncated_wal_recovery() {
    let pid = std::process::id();
    let test_dir = std::path::PathBuf::from(format!("target/test_truncated_wal_{pid}"));
    let _ = std::fs::remove_dir_all(&test_dir);
    std::fs::create_dir_all(&test_dir).unwrap();

    let snapshot_path = test_dir.join("data.bftree");
    let wal_path = test_dir.join("wal.log");
    let config_path = test_dir.join("config.toml");

    let pre_count = 200;
    let post_count = 100;

    // Phase 1: Insert records, snapshot, add more to WAL, then crash.
    {
        let tree = BfTree::with_config(make_config(&snapshot_path, &wal_path), None).unwrap();
        insert_records(&tree, 0, pre_count);
        tree.snapshot().unwrap();
        insert_records(&tree, pre_count, post_count);
        // WAL flush to ensure entries are on disk
        std::thread::sleep(std::time::Duration::from_millis(10));
        // Drop = crash
    }

    // Phase 2: Truncate WAL file to corrupt the tail.
    {
        let wal_meta = std::fs::metadata(&wal_path).unwrap();
        let orig_len = wal_meta.len();
        if orig_len > 128 {
            // Truncate ~30% from the end to corrupt some entries.
            let truncate_to = orig_len * 7 / 10;
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(&wal_path)
                .unwrap();
            file.set_len(truncate_to).unwrap();
        }
    }

    // Phase 3: Recovery should succeed -- corrupt tail entries are lost
    // but snapshot data + valid WAL entries survive.
    write_config_toml(&config_path, &snapshot_path);
    let tree = BfTree::recovery(&config_path, &wal_path, WAL_SEGMENT_SIZE, None)
        .expect("recovery should succeed despite truncated WAL");

    // At minimum, all pre-snapshot records must be present.
    verify_records(&tree, 0, pre_count);

    drop(tree);
    let _ = std::fs::remove_dir_all(&test_dir);
}

// ---------------------------------------------------------------------------
// Test 2: Corrupt snapshot falls back to fresh tree
// ---------------------------------------------------------------------------
#[test]
fn corrupt_snapshot_falls_back_to_fresh() {
    let pid = std::process::id();
    let test_dir = std::path::PathBuf::from(format!("target/test_corrupt_snapshot_{pid}"));
    let _ = std::fs::remove_dir_all(&test_dir);
    std::fs::create_dir_all(&test_dir).unwrap();

    let snapshot_path = test_dir.join("data.bftree");

    // Phase 1: Create a tree, snapshot it.
    {
        let tree = BfTree::with_config(make_config_no_wal(&snapshot_path), None).unwrap();
        insert_records(&tree, 0, 100);
        tree.snapshot().unwrap();
    }

    // Phase 2: Corrupt the snapshot file header (first 64 bytes).
    {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&snapshot_path)
            .unwrap();
        let mut header = [0u8; 64];
        file.read_exact(&mut header).unwrap();
        // XOR all bytes to corrupt magic markers
        for b in &mut header {
            *b ^= 0xFF;
        }
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&header).unwrap();
    }

    // Phase 3: Opening from corrupt snapshot should produce a fresh tree.
    let tree = BfTree::new_from_snapshot(make_config_no_wal(&snapshot_path), None).unwrap();

    // The fresh tree should be empty -- old data is lost.
    verify_key_absent(&tree, 0);
    verify_key_absent(&tree, 99);

    drop(tree);
    let _ = std::fs::remove_dir_all(&test_dir);
}

// ---------------------------------------------------------------------------
// Test 3: Post-snapshot WAL entries survive crash
// ---------------------------------------------------------------------------
#[test]
fn post_snapshot_wal_entries_survive_crash() {
    let pid = std::process::id();
    let test_dir = std::path::PathBuf::from(format!("target/test_post_snap_wal_{pid}"));
    let _ = std::fs::remove_dir_all(&test_dir);
    std::fs::create_dir_all(&test_dir).unwrap();

    let snapshot_path = test_dir.join("data.bftree");
    let wal_path = test_dir.join("wal.log");
    let config_path = test_dir.join("config.toml");

    let pre_count = 300;
    let post_count = 200;

    // Phase 1: Insert, snapshot, insert more, crash.
    {
        let tree = BfTree::with_config(make_config(&snapshot_path, &wal_path), None).unwrap();
        insert_records(&tree, 0, pre_count);
        tree.snapshot().unwrap();
        insert_records(&tree, pre_count, post_count);
        // Ensure WAL flush
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Phase 2: Recover and verify ALL entries.
    write_config_toml(&config_path, &snapshot_path);
    let tree =
        BfTree::recovery(&config_path, &wal_path, WAL_SEGMENT_SIZE, None).expect("recovery failed");

    verify_records(&tree, 0, pre_count + post_count);

    drop(tree);
    let _ = std::fs::remove_dir_all(&test_dir);
}

// ---------------------------------------------------------------------------
// Test 4: Empty WAL recovery (snapshot only, no WAL activity)
// ---------------------------------------------------------------------------
#[test]
fn empty_wal_recovery() {
    let pid = std::process::id();
    let test_dir = std::path::PathBuf::from(format!("target/test_empty_wal_{pid}"));
    let _ = std::fs::remove_dir_all(&test_dir);
    std::fs::create_dir_all(&test_dir).unwrap();

    let snapshot_path = test_dir.join("data.bftree");
    let wal_path = test_dir.join("wal.log");
    let config_path = test_dir.join("config.toml");

    let record_count = 500;

    // Phase 1: Insert records with WAL, snapshot, then close cleanly.
    {
        let tree = BfTree::with_config(make_config(&snapshot_path, &wal_path), None).unwrap();
        insert_records(&tree, 0, record_count);
        tree.snapshot().unwrap();
        // No post-snapshot writes -- WAL is empty relative to snapshot.
    }

    // Phase 2: Recover -- should load snapshot, replay zero WAL entries.
    write_config_toml(&config_path, &snapshot_path);
    let tree = BfTree::recovery(&config_path, &wal_path, WAL_SEGMENT_SIZE, None)
        .expect("recovery from empty WAL failed");

    verify_records(&tree, 0, record_count);

    drop(tree);
    let _ = std::fs::remove_dir_all(&test_dir);
}

// ---------------------------------------------------------------------------
// Test 5: WAL replay is idempotent -- replaying twice produces same state
// ---------------------------------------------------------------------------
#[test]
fn wal_replay_idempotency() {
    let pid = std::process::id();
    let test_dir = std::path::PathBuf::from(format!("target/test_wal_idempotent_{pid}"));
    let _ = std::fs::remove_dir_all(&test_dir);
    std::fs::create_dir_all(&test_dir).unwrap();

    let snapshot_path = test_dir.join("data.bftree");
    let wal_path = test_dir.join("wal.log");
    let config_path = test_dir.join("config.toml");

    let pre_count = 200;
    let post_count = 300;

    // Phase 1: Create tree, snapshot, add WAL entries, crash.
    {
        let tree = BfTree::with_config(make_config(&snapshot_path, &wal_path), None).unwrap();
        insert_records(&tree, 0, pre_count);
        tree.snapshot().unwrap();
        insert_records(&tree, pre_count, post_count);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    write_config_toml(&config_path, &snapshot_path);

    // Phase 2: First recovery.
    {
        let tree = BfTree::recovery(&config_path, &wal_path, WAL_SEGMENT_SIZE, None)
            .expect("first recovery failed");
        verify_records(&tree, 0, pre_count + post_count);
        // recovery() takes a fresh snapshot internally
    }

    // Phase 3: Second recovery from the same WAL -- must produce same state.
    {
        let tree = BfTree::recovery(&config_path, &wal_path, WAL_SEGMENT_SIZE, None)
            .expect("second recovery failed");
        verify_records(&tree, 0, pre_count + post_count);
    }

    let _ = std::fs::remove_dir_all(&test_dir);
}

// ---------------------------------------------------------------------------
// Test 6: Concurrent writes during snapshot don't crash
// ---------------------------------------------------------------------------
#[test]
fn concurrent_writes_during_snapshot() {
    let pid = std::process::id();
    let test_dir = std::path::PathBuf::from(format!("target/test_concurrent_snap_{pid}"));
    let _ = std::fs::remove_dir_all(&test_dir);
    std::fs::create_dir_all(&test_dir).unwrap();

    let snapshot_path = test_dir.join("data.bftree");

    let tree = Arc::new(BfTree::with_config(make_config_no_wal(&snapshot_path), None).unwrap());

    // Pre-populate
    let mut key_buffer = vec![0usize; KEY_LEN / 8];
    for r in 0..500 {
        let key = install_value_to_buffer(&mut key_buffer, r);
        tree.insert(key, key);
    }
    tree.snapshot().unwrap();

    let writer_count = 4;
    let ops_per_writer = 100;
    let barrier = Arc::new(std::sync::Barrier::new(writer_count + 1));

    // Spawn writer threads.
    let handles: Vec<_> = (0..writer_count)
        .map(|t| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let mut buf = vec![0usize; KEY_LEN / 8];
                let base = 1000 + t * ops_per_writer;
                for r in base..(base + ops_per_writer) {
                    let key = install_value_to_buffer(&mut buf, r);
                    tree.insert(key, key);
                }
            })
        })
        .collect();

    // Snapshot concurrently with writers.
    barrier.wait();
    let snap_result = tree.snapshot();
    // Snapshot should succeed (or return error), never panic.
    if let Err(e) = &snap_result {
        // Some transient errors are acceptable under contention.
        eprintln!("snapshot under contention returned error (acceptable): {e:?}");
    }

    for h in handles {
        h.join().unwrap();
    }

    // Verify original records are still readable.
    let mut out = vec![0u8; KEY_LEN];
    for r in 0..500 {
        let key = install_value_to_buffer(&mut key_buffer, r);
        match tree.read(key, &mut out) {
            LeafReadResult::Found(_) => {}
            other => panic!("key {r} missing after concurrent snapshot: {other:?}"),
        }
    }

    drop(tree);
    let _ = std::fs::remove_dir_all(&test_dir);
}

// ---------------------------------------------------------------------------
// Test 7: Large values near max record size survive snapshot + recovery
// ---------------------------------------------------------------------------
#[test]
fn large_value_snapshot_recovery() {
    let pid = std::process::id();
    let test_dir = std::path::PathBuf::from(format!("target/test_large_val_{pid}"));
    let _ = std::fs::remove_dir_all(&test_dir);
    std::fs::create_dir_all(&test_dir).unwrap();

    let snapshot_path = test_dir.join("data.bftree");

    // Use values close to max record size.
    let value_len = MAX_RECORD_SIZE - KEY_LEN - 64; // leave room for metadata
    let record_count = 50;

    // Phase 1: Insert large records and snapshot.
    {
        let tree = BfTree::with_config(make_config_no_wal(&snapshot_path), None).unwrap();
        let mut key_buffer = vec![0usize; KEY_LEN / 8];
        let value = vec![0xABu8; value_len];
        for r in 0..record_count {
            let key = install_value_to_buffer(&mut key_buffer, r);
            tree.insert(key, &value);
        }
        tree.snapshot().unwrap();
    }

    // Phase 2: Recover and verify values intact.
    let tree = BfTree::new_from_snapshot(make_config_no_wal(&snapshot_path), None).unwrap();

    let mut key_buffer = vec![0usize; KEY_LEN / 8];
    let mut out = vec![0u8; value_len + 64];
    for r in 0..record_count {
        let key = install_value_to_buffer(&mut key_buffer, r);
        match tree.read(key, &mut out) {
            LeafReadResult::Found(v) => {
                assert_eq!(v as usize, value_len, "wrong value size for key {r}");
                assert!(
                    out[..value_len].iter().all(|&b| b == 0xAB),
                    "value corruption for key {r}"
                );
            }
            other => panic!("key {r} not found: {other:?}"),
        }
    }

    drop(tree);
    let _ = std::fs::remove_dir_all(&test_dir);
}

// ---------------------------------------------------------------------------
// Test 8: Delete operations persist across snapshot + recovery
// ---------------------------------------------------------------------------
#[test]
fn delete_survives_snapshot_recovery() {
    let pid = std::process::id();
    let test_dir = std::path::PathBuf::from(format!("target/test_delete_recovery_{pid}"));
    let _ = std::fs::remove_dir_all(&test_dir);
    std::fs::create_dir_all(&test_dir).unwrap();

    let snapshot_path = test_dir.join("data.bftree");
    let wal_path = test_dir.join("wal.log");
    let config_path = test_dir.join("config.toml");

    let record_count = 200;
    let delete_start = 50;
    let delete_end = 150;

    // Phase 1: Insert, delete some, snapshot, crash.
    {
        let tree = BfTree::with_config(make_config(&snapshot_path, &wal_path), None).unwrap();
        insert_records(&tree, 0, record_count);
        // Delete records 50..150
        let mut key_buffer = vec![0usize; KEY_LEN / 8];
        for r in delete_start..delete_end {
            let key = install_value_to_buffer(&mut key_buffer, r);
            tree.delete(key);
        }
        tree.snapshot().unwrap();
    }

    // Phase 2: Recover and verify deletions persisted.
    write_config_toml(&config_path, &snapshot_path);
    let tree =
        BfTree::recovery(&config_path, &wal_path, WAL_SEGMENT_SIZE, None).expect("recovery failed");

    // Records 0..50 and 150..200 should be present.
    verify_records(&tree, 0, delete_start);
    verify_records(&tree, delete_end, record_count - delete_end);

    // Records 50..150 should be absent.
    for r in delete_start..delete_end {
        verify_key_absent(&tree, r);
    }

    drop(tree);
    let _ = std::fs::remove_dir_all(&test_dir);
}
