// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
// Modified for no_std: use spin::Mutex always, core::sync::atomic always.

pub(crate) mod atomic {
    pub use core::sync::atomic::*;
}

pub(crate) use alloc::sync::Arc;

// Use spin::Mutex unconditionally -- no poison semantics needed.
// .lock() returns the guard directly (no Result).
// .try_lock() returns Option<MutexGuard> (no TryLockError).
pub(crate) use spin::Mutex;
pub(crate) use spin::MutexGuard;

#[cfg(all(feature = "shuttle", test))]
pub(crate) use shuttle::thread;

#[cfg(all(feature = "std", not(all(feature = "shuttle", test))))]
#[allow(unused_imports)]
pub(crate) use std::thread;

#[cfg(all(feature = "std", test, not(all(feature = "shuttle", test))))]
#[allow(unused_imports)]
pub(crate) use std::sync::Barrier;
