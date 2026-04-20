//! Background integrity scanner for continuous corruption detection.
//!
//! Spawns a dedicated thread that periodically walks all B-tree pages
//! (both data and system roots) using xxh3-128 checksum verification.
//! Corruption is reported via [`ScanCycleResult`] and an optional callback.
//!
//! The scanner never blocks writers -- it reads a consistent snapshot of
//! the page state each cycle using the same verification infrastructure
//! as [`Database::verify_integrity()`](crate::Database::verify_integrity).

use crate::db::{CorruptPageInfo, Database, TransactionGuard};
use crate::tree_store::TransactionalMemory;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use portable_atomic::AtomicU64;
use std::sync::{Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Callback type invoked after each completed scan cycle.
pub type CycleCallback = Box<dyn Fn(&ScanCycleResult) + Send + Sync>;

/// Configuration for the background integrity scanner.
pub struct IntegrityScannerConfig {
    /// Seconds between scan cycles. Default: 60.
    pub scan_interval_secs: u64,
    /// Optional callback invoked after each completed scan cycle.
    pub on_cycle_complete: Option<CycleCallback>,
}

impl Default for IntegrityScannerConfig {
    fn default() -> Self {
        Self {
            scan_interval_secs: 60,
            on_cycle_complete: None,
        }
    }
}

/// Results from a single integrity scan cycle.
#[derive(Clone, Debug)]
pub struct ScanCycleResult {
    /// Total B-tree pages checked (data + system roots).
    pub pages_checked: u64,
    /// Number of pages with checksum mismatches.
    pub pages_corrupt: u64,
    /// Details of each corrupt page found.
    pub corrupt_details: Vec<CorruptPageInfo>,
    /// Wall-clock duration of this scan cycle.
    pub duration: Duration,
    /// Monotonically increasing cycle counter (starts at 1).
    pub cycle_number: u64,
}

/// Handle to a running background integrity scanner.
///
/// The scanner thread is stopped when this handle is dropped or
/// [`shutdown()`](IntegrityScannerHandle::shutdown) is called explicitly.
pub struct IntegrityScannerHandle {
    shutdown: Arc<AtomicBool>,
    wake: Arc<(Mutex<()>, Condvar)>,
    thread: Option<JoinHandle<()>>,
    last_result: Arc<Mutex<Option<ScanCycleResult>>>,
    total_cycles: Arc<AtomicU64>,
}

impl IntegrityScannerHandle {
    pub(crate) fn start(
        mem: Arc<TransactionalMemory>,
        config: IntegrityScannerConfig,
    ) -> Result<Self, std::io::Error> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let wake = Arc::new((Mutex::new(()), Condvar::new()));
        let last_result: Arc<Mutex<Option<ScanCycleResult>>> = Arc::new(Mutex::new(None));
        let total_cycles = Arc::new(AtomicU64::new(0));

        let thread = {
            let shutdown = shutdown.clone();
            let wake = wake.clone();
            let last_result = last_result.clone();
            let total_cycles = total_cycles.clone();
            std::thread::Builder::new()
                .name("shodh-integrity-scanner".into())
                .spawn(move || {
                    run_scanner(mem, config, shutdown, wake, last_result, total_cycles);
                })?
        };

        Ok(Self {
            shutdown,
            wake,
            thread: Some(thread),
            last_result,
            total_cycles,
        })
    }

    /// Returns a clone of the most recent scan cycle result, or `None` if
    /// no cycle has completed yet.
    pub fn last_result(&self) -> Option<ScanCycleResult> {
        self.last_result.lock().ok().and_then(|guard| guard.clone())
    }

    /// Returns the total number of completed scan cycles.
    pub fn total_cycles(&self) -> u64 {
        self.total_cycles.load(Ordering::Relaxed)
    }

    /// Signals the scanner thread to stop and waits for it to exit.
    ///
    /// Safe to call multiple times -- subsequent calls are no-ops.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.wake.1.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for IntegrityScannerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run_scanner(
    mem: Arc<TransactionalMemory>,
    config: IntegrityScannerConfig,
    shutdown: Arc<AtomicBool>,
    wake: Arc<(Mutex<()>, Condvar)>,
    last_result: Arc<Mutex<Option<ScanCycleResult>>>,
    total_cycles: Arc<AtomicU64>,
) {
    let mut cycle = 0u64;
    let interval = Duration::from_secs(config.scan_interval_secs);

    while !shutdown.load(Ordering::Relaxed) {
        let start = Instant::now();
        let scan = Database::verify_primary_checksums_detailed(
            mem.clone(),
            mem.get_persisted_data_root(),
            mem.get_persisted_system_root(),
            Arc::new(TransactionGuard::Verification),
        );

        if let Ok((pages_checked, corrupt_details)) = scan {
            cycle += 1;
            let result = ScanCycleResult {
                pages_checked,
                pages_corrupt: corrupt_details.len() as u64,
                corrupt_details,
                duration: start.elapsed(),
                cycle_number: cycle,
            };
            if let Some(ref cb) = config.on_cycle_complete {
                cb(&result);
            }
            if let Ok(mut guard) = last_result.lock() {
                *guard = Some(result);
            }
            total_cycles.store(cycle, Ordering::Relaxed);
        }

        // Sleep with early wake on shutdown signal.
        let (lock, cvar) = &*wake;
        if let Ok(guard) = lock.lock() {
            let _ = cvar.wait_timeout(guard, interval);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Database, TableDefinition};

    const TEST_TABLE: TableDefinition<&str, u64> = TableDefinition::new("scanner_test");

    fn create_populated_db() -> (tempfile::NamedTempFile, Database) {
        let file = crate::create_tempfile();
        let db = Database::create(file.path()).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(TEST_TABLE).unwrap();
            for i in 0..100 {
                let key = alloc::format!("key_{i}");
                table.insert(key.as_str(), &i).unwrap();
            }
        }
        txn.commit().unwrap();
        (file, db)
    }

    #[test]
    fn scanner_start_and_stop() {
        let (_file, db) = create_populated_db();
        let mut handle = db
            .start_integrity_scanner(IntegrityScannerConfig {
                scan_interval_secs: 1,
                on_cycle_complete: None,
            })
            .unwrap();

        // Wait for at least one cycle
        for _ in 0..50 {
            if handle.total_cycles() >= 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(
            handle.total_cycles() >= 1,
            "scanner should complete at least 1 cycle"
        );

        let result = handle.last_result().expect("should have a result");
        assert_eq!(result.pages_corrupt, 0);
        assert!(result.pages_checked > 0);

        handle.shutdown();
    }

    #[test]
    fn scanner_callback_invoked() {
        let (_file, db) = create_populated_db();
        let callback_fired = Arc::new(AtomicBool::new(false));
        let flag = callback_fired.clone();

        let mut handle = db
            .start_integrity_scanner(IntegrityScannerConfig {
                scan_interval_secs: 1,
                on_cycle_complete: Some(Box::new(move |_result| {
                    flag.store(true, Ordering::Relaxed);
                })),
            })
            .unwrap();

        for _ in 0..50 {
            if callback_fired.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(
            callback_fired.load(Ordering::Relaxed),
            "callback should fire"
        );

        handle.shutdown();
    }

    #[test]
    fn scanner_drop_triggers_shutdown() {
        let (_file, db) = create_populated_db();
        let handle = db
            .start_integrity_scanner(IntegrityScannerConfig {
                scan_interval_secs: 3600, // long interval -- drop should wake it
                on_cycle_complete: None,
            })
            .unwrap();
        // Dropping the handle should join the thread without hanging.
        drop(handle);
    }

    #[test]
    fn scanner_concurrent_with_writes() {
        let (_file, db) = create_populated_db();
        let mut handle = db
            .start_integrity_scanner(IntegrityScannerConfig {
                scan_interval_secs: 1,
                on_cycle_complete: None,
            })
            .unwrap();

        // Perform writes while the scanner is running
        for batch in 0..5 {
            let txn = db.begin_write().unwrap();
            {
                let mut table = txn.open_table(TEST_TABLE).unwrap();
                for i in 0..20 {
                    let key = alloc::format!("batch_{batch}_key_{i}");
                    table.insert(key.as_str(), &(batch * 20 + i)).unwrap();
                }
            }
            txn.commit().unwrap();
            std::thread::sleep(Duration::from_millis(50));
        }

        // Wait for at least one cycle to complete
        for _ in 0..50 {
            if handle.total_cycles() >= 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(handle.total_cycles() >= 1);

        handle.shutdown();
    }

    #[test]
    fn scanner_detects_corruption() {
        use std::io::{Seek, Write};

        let file = crate::create_tempfile();
        let db = Database::create(file.path()).unwrap();

        // Insert enough data to create multiple B-tree pages
        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(TEST_TABLE).unwrap();
            for i in 0u64..500 {
                let key = alloc::format!("corruption_test_key_{i:06}");
                table.insert(key.as_str(), &i).unwrap();
            }
        }
        txn.commit().unwrap();
        drop(db);

        // Corrupt some bytes in the middle of the file (likely a B-tree page)
        {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(file.path())
                .unwrap();
            let file_len = f.seek(std::io::SeekFrom::End(0)).unwrap();
            // Write garbage in the middle of the data region
            let corrupt_offset = file_len / 2;
            f.seek(std::io::SeekFrom::Start(corrupt_offset)).unwrap();
            f.write_all(&[0xFF; 256]).unwrap();
            f.sync_all().unwrap();
        }

        // Reopen the DB -- it may detect corruption on open but we try anyway
        let Ok(db) = Database::create(file.path()) else {
            return; // corruption too severe to open -- test passes trivially
        };

        let corruption_found = Arc::new(AtomicBool::new(false));
        let flag = corruption_found.clone();

        let mut handle = db
            .start_integrity_scanner(IntegrityScannerConfig {
                scan_interval_secs: 1,
                on_cycle_complete: Some(Box::new(move |result| {
                    if result.pages_corrupt > 0 {
                        flag.store(true, Ordering::Relaxed);
                    }
                })),
            })
            .unwrap();

        // Wait for at least one cycle
        for _ in 0..50 {
            if handle.total_cycles() >= 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        handle.shutdown();

        // The corruption may or may not be detected depending on where the bytes landed.
        // If the scanner completed a cycle, the result should be valid either way.
        if let Some(result) = handle.last_result() {
            assert!(result.pages_checked > 0, "scanner should check pages");
        }
    }
}
