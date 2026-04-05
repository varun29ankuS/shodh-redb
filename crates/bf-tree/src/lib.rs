// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
// Forked with no_std support by Roshera.
//
// # no_std requirements
// - Global allocator (`#[global_allocator]`) that supports 4096-byte alignment.
// - Multi-core target or RTOS with preemptive scheduling (spin-wait deadlocks
//   on single-core without preemption).

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "std"), allow(dead_code, unused_imports))]

extern crate alloc;

pub mod circular_buffer;
mod config;
mod error;
mod fs;
mod nodes;

#[cfg(feature = "std")]
mod wal;

#[cfg(any(
    feature = "metrics",
    feature = "metrics-rt",
    feature = "metrics-rt-debug-all",
    feature = "metrics-rt-debug-timer"
))]
pub mod metric;

mod mini_page_op;
mod range_scan;
mod storage;
pub(crate) mod sync;
#[cfg(test)]
mod tests;

#[cfg(feature = "std")]
mod snapshot;
mod tree;
mod utils;

#[cfg(feature = "std")]
pub use config::WalConfig;
pub use config::{Config, StorageBackend};
pub use error::ConfigError;
pub use nodes::leaf_node::LeafReadResult;
pub use range_scan::{ScanIter, ScanReturnField};
pub use tree::{BfTree, LeafInsertResult, ScanIterError};
#[cfg(feature = "std")]
pub use wal::WalReader;

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        #[cfg(all(feature = "tracing", debug_assertions))]
        {
            tracing::info!($($arg)*);
        }

        #[cfg(not(all(feature = "tracing", debug_assertions)))]
        {
        }
    };
}

#[macro_export]
macro_rules! counter {
    ($event:ident) => {
        #[cfg(feature = "metrics-rt")]
        {
            $crate::metric::get_tls_recorder()
                .increment_counter($crate::metric::Counter::$event, 1);
        }
    };
    ($event:ident, $value:literal) => {
        #[cfg(feature = "metrics-rt")]
        {
            $crate::metric::get_tls_recorder()
                .increment_counter($crate::metric::Counter::$event, $value);
        }
    };
}

#[macro_export]
macro_rules! histogram {
    ($event:ident, $value:expr) => {
        #[cfg(feature = "metrics-rt")]
        {
            $crate::metric::get_tls_recorder()
                .hit_histogram($crate::metric::Histogram::$event, $value);
        }
    };
}

#[macro_export]
macro_rules! timer {
    ($event:expr) => {
        let _timer_guard = if cfg!(feature = "metrics-rt") {
            Some($crate::metric::get_tls_recorder().timer_guard($event))
        } else {
            None
        };
    };
}

#[macro_export]
macro_rules! check_parent {
    ($self:ident, $node:expr, $parent:expr) => {
        if let Some(ref p) = $parent {
            p.check_version()?;
        } else if $node != $self.get_root_page().0 {
            return Err(TreeError::Locked);
        }
    };
}
