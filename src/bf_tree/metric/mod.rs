// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use serde::Serialize;
use std::cell::UnsafeCell;
use std::ops::AddAssign;

pub(crate) mod counter;
pub(crate) mod histogram;
pub mod recorders;
pub mod timer;

#[allow(unused_imports)]
pub use crate::bf_tree::metric::counter::Counter;
#[allow(unused_imports)]
pub use crate::bf_tree::metric::histogram::Histogram;
#[allow(unused_imports)]
pub use crate::bf_tree::metric::timer::Timer;

pub use recorders::TlsRecorder;
pub use timer::DebugTimerGuard;

thread_local! {
    static LOCAL_RECORDER: UnsafeCell<TlsRecorder> = UnsafeCell::new(TlsRecorder::default());
}

pub trait RecorderImpl: Serialize + AddAssign + Sized {
    fn reset(&mut self);
}

pub fn get_tls_recorder() -> &'static mut TlsRecorder {
    // SAFETY: The UnsafeCell is thread-local, so only one thread can access it at a time.
    // The returned &'static mut is valid for the thread's lifetime. No other references to
    // the inner TlsRecorder exist since this is the only accessor.
    LOCAL_RECORDER.with(|id| unsafe { &mut *id.get() })
}
