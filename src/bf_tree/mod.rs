// Originally from bf-tree (MIT, Microsoft Research). See LICENSE-MIT-BFTREE.
// Forked with no_std support by Roshera.
//
// # no_std requirements
// - Global allocator (`#[global_allocator]`) that supports 4096-byte alignment.
// - Multi-core target or RTOS with preemptive scheduling (spin-wait deadlocks
//   on single-core without preemption).

// Edition 2024 makes unsafe_op_in_unsafe_fn deny-by-default. The vendored
// bf-tree code has not been audited for this yet -- allow for now.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_imports)]
// Pre-existing MSR bf-tree lint issues -- allow in forked code to avoid churn.
// The vendored code was written against bf-tree's own lint config; suppress
// shodh-redb's stricter deny(clippy::all, clippy::pedantic) for this module.
#![allow(clippy::all, clippy::pedantic, clippy::disallowed_methods, dead_code)]

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
#[cfg(all(test, feature = "std", not(feature = "shuttle")))]
mod tests;

#[cfg(feature = "std")]
mod snapshot;
mod tree;
mod utils;

#[cfg(feature = "std")]
pub use config::WalConfig;
pub use config::{Config, StorageBackend};
pub use error::{BfTreeError, ConfigError, IoErrorKind};
pub use nodes::leaf_node::LeafReadResult;
pub use range_scan::{ScanIter, ScanReturnField};
pub use tree::{BfTree, LeafInsertResult, ScanIterError};
#[cfg(feature = "std")]
pub use wal::WalReader;

// These macros are used throughout the bf_tree submodules. They must be
// #[macro_export] so they're visible across file boundaries, but we hide
// them from the public API with #[doc(hidden)].

#[doc(hidden)]
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

#[doc(hidden)]
#[macro_export]
macro_rules! counter {
    ($event:ident) => {
        #[cfg(feature = "metrics-rt")]
        {
            $crate::bf_tree::metric::get_tls_recorder()
                .increment_counter($crate::bf_tree::metric::Counter::$event, 1);
        }
    };
    ($event:ident, $value:literal) => {
        #[cfg(feature = "metrics-rt")]
        {
            $crate::bf_tree::metric::get_tls_recorder()
                .increment_counter($crate::bf_tree::metric::Counter::$event, $value);
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! histogram {
    ($event:ident, $value:expr) => {
        #[cfg(feature = "metrics-rt")]
        {
            $crate::bf_tree::metric::get_tls_recorder()
                .hit_histogram($crate::bf_tree::metric::Histogram::$event, $value);
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! timer {
    ($event:expr) => {
        let _timer_guard = if cfg!(feature = "metrics-rt") {
            Some($crate::bf_tree::metric::get_tls_recorder().timer_guard($event))
        } else {
            None
        };
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! check_parent {
    ($self:ident, $node:expr, $parent:expr) => {
        if let Some(p) = &$parent {
            p.check_version()?;
        } else if $node != $self.get_root_page().0 {
            return Err(TreeError::Locked);
        }
    };
}
