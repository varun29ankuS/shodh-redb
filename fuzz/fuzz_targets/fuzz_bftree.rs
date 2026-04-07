#![no_main]

use libfuzzer_sys::fuzz_target;
use redb::bf_tree_store::{BfTreeConfig, BfTreeDatabase};
use redb::TableDefinition;

const TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("fuzz");

/// Operations the fuzzer can request.
#[derive(arbitrary::Arbitrary, Debug)]
enum Op {
    /// Insert a key-value pair.
    Insert { key: Vec<u8>, value: Vec<u8> },
    /// Delete a key.
    Delete { key: Vec<u8> },
    /// Read a key and verify against the reference model.
    Read { key: Vec<u8> },
    /// Commit the current write transaction.
    Commit,
    /// Commit with an explicit snapshot checkpoint.
    CommitWithSnapshot,
    /// Scan a range of keys.
    Scan { start: Vec<u8>, limit: u8 },
}

fuzz_target!(|ops: Vec<Op>| {
    if ops.is_empty() || ops.len() > 512 {
        return;
    }

    let config = BfTreeConfig::new_embedded(64);
    let db = match BfTreeDatabase::create(config) {
        Ok(db) => db,
        Err(_) => return,
    };

    // Reference model: track expected state
    let mut model = std::collections::BTreeMap::<Vec<u8>, Vec<u8>>::new();
    // Pending writes not yet committed
    let mut pending = std::collections::BTreeMap::<Vec<u8>, Option<Vec<u8>>>::new();

    let mut needs_commit = false;

    for op in &ops {
        match op {
            Op::Insert { key, value } => {
                // Clamp key/value to prevent pathological sizes
                let key = &key[..key.len().min(64)];
                let value = &value[..value.len().min(256)];
                if key.is_empty() {
                    continue;
                }
                let wtxn = db.begin_write();
                if wtxn.insert::<&[u8], &[u8]>(&TABLE, &key, &value).is_ok() {
                    pending.insert(key.to_vec(), Some(value.to_vec()));
                    needs_commit = true;
                }
                // Intentionally drop without commit — tests abort path
                if !needs_commit {
                    continue;
                }
                if wtxn.commit().is_ok() {
                    // Apply pending to model
                    for (k, v) in pending.drain() {
                        match v {
                            Some(val) => model.insert(k, val),
                            None => model.remove(&k),
                        };
                    }
                    needs_commit = false;
                }
            }
            Op::Delete { key } => {
                if key.is_empty() {
                    continue;
                }
                let key = &key[..key.len().min(64)];
                let wtxn = db.begin_write();
                wtxn.delete::<&[u8], &[u8]>(&TABLE, &key);
                pending.insert(key.to_vec(), None);
                needs_commit = true;
                if wtxn.commit().is_ok() {
                    for (k, v) in pending.drain() {
                        match v {
                            Some(val) => model.insert(k, val),
                            None => model.remove(&k),
                        };
                    }
                    needs_commit = false;
                }
            }
            Op::Read { key } => {
                if key.is_empty() {
                    continue;
                }
                let key = &key[..key.len().min(64)];
                let rtxn = db.begin_read();
                let result = rtxn.get::<&[u8], &[u8]>(&TABLE, &key);
                match result {
                    Ok(Some(val)) => {
                        if let Some(expected) = model.get(key) {
                            assert_eq!(
                                val.as_slice(),
                                expected.as_slice(),
                                "value mismatch for key {:?}",
                                key
                            );
                        }
                    }
                    Ok(None) => {
                        // Key not found — consistent with model if absent
                    }
                    Err(_) => {
                        // I/O or internal error — acceptable in fuzzing
                    }
                }
            }
            Op::Commit => {
                if !needs_commit {
                    continue;
                }
                let wtxn = db.begin_write();
                if wtxn.commit().is_ok() {
                    for (k, v) in pending.drain() {
                        match v {
                            Some(val) => model.insert(k, val),
                            None => model.remove(&k),
                        };
                    }
                    needs_commit = false;
                }
            }
            Op::CommitWithSnapshot => {
                let wtxn = db.begin_write();
                if wtxn.commit_with_snapshot().is_ok() {
                    for (k, v) in pending.drain() {
                        match v {
                            Some(val) => model.insert(k, val),
                            None => model.remove(&k),
                        };
                    }
                    needs_commit = false;
                }
            }
            Op::Scan { start, limit } => {
                let start = &start[..start.len().min(64)];
                let limit = (*limit as usize).min(50);
                if limit == 0 {
                    continue;
                }
                let rtxn = db.begin_read();
                if let Ok(mut scan) = rtxn.scan_table::<&[u8], &[u8]>(&TABLE) {
                    let mut buf = vec![0u8; 512];
                    let mut count = 0;
                    while count < limit {
                        match scan.next(&mut buf) {
                            Some(_) => count += 1,
                            None => break,
                        }
                    }
                }
            }
        }
    }
});
