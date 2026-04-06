// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/// Adapted from https://github.com/m-ou-se/atomic-wait
use crate::sync::atomic::AtomicU32;

#[cfg(all(
    feature = "std",
    target_os = "linux",
    not(all(feature = "shuttle", test))
))]
mod platform {
    use core::sync::atomic::AtomicU32;
    #[inline]
    pub fn wait(a: &AtomicU32, expected: u32) {
        // SAFETY: a is a valid reference to an AtomicU32, providing a valid address for the
        // futex syscall. FUTEX_WAIT with FUTEX_PRIVATE_FLAG is a well-defined Linux syscall.
        // A null timespec pointer means wait indefinitely. Spurious wakeups are handled by callers.
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                a,
                libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                expected,
                core::ptr::null::<libc::timespec>(),
            );
        };
    }

    #[inline]
    pub fn wake_one(ptr: *const AtomicU32) {
        // SAFETY: The futex syscall tolerates dangling/null pointers for FUTEX_WAKE (it simply
        // finds no waiters). When valid, ptr points to an AtomicU32 used as the futex word.
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                ptr,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1i32,
            );
        };
    }

    #[inline]
    pub fn wake_all(ptr: *const AtomicU32) {
        // SAFETY: Same as wake_one; FUTEX_WAKE with i32::MAX wakes all waiting threads.
        // The futex syscall is safe with dangling/null pointers (no waiters found).
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                ptr,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                i32::MAX,
            );
        };
    }
}

#[cfg(all(feature = "shuttle", test))]
mod platform {
    use core::sync::atomic::AtomicU32;

    #[inline]
    pub fn wait(_a: &AtomicU32, _expected: u32) {
        shuttle::thread::yield_now();
    }

    #[inline]
    pub fn wake_one(_ptr: *const AtomicU32) {
        shuttle::thread::yield_now();
    }

    #[inline]
    pub fn wake_all(_ptr: *const AtomicU32) {
        shuttle::thread::yield_now();
    }
}

#[cfg(all(
    feature = "std",
    target_os = "macos",
    not(all(feature = "shuttle", test))
))]
/// We don't do anything for macOS, just to make sure it compiles and correct.
mod platform {
    use core::sync::atomic::AtomicU32;

    #[inline]
    pub fn wait(_a: &AtomicU32, _expected: u32) {}

    #[inline]
    pub fn wake_one(_ptr: *const AtomicU32) {}

    #[inline]
    pub fn wake_all(_ptr: *const AtomicU32) {}
}

#[cfg(all(
    feature = "std",
    target_os = "windows",
    not(all(feature = "shuttle", test))
))]
mod platform {
    use core::sync::atomic::AtomicU32;
    use windows_sys::Win32::System::Threading::{
        WaitOnAddress, WakeByAddressAll, WakeByAddressSingle, INFINITE,
    };

    #[inline]
    pub fn wait(a: &AtomicU32, expected: u32) {
        let ptr: *const AtomicU32 = a;
        let expected_ptr: *const u32 = &expected;
        // SAFETY: ptr is derived from a valid &AtomicU32 reference, so it is non-null and
        // aligned. expected_ptr points to a local stack variable. Size 4 matches sizeof(u32).
        // WaitOnAddress is a well-defined Windows API; spurious wakeups are handled by callers.
        unsafe { WaitOnAddress(ptr.cast(), expected_ptr.cast(), 4, INFINITE) };
    }

    #[inline]
    pub fn wake_one(ptr: *const AtomicU32) {
        // SAFETY: WakeByAddressSingle tolerates any pointer (wakes matching waiters or is a no-op).
        // When valid, ptr points to an AtomicU32 used as the wait address.
        unsafe { WakeByAddressSingle(ptr.cast()) };
    }

    #[inline]
    pub fn wake_all(ptr: *const AtomicU32) {
        // SAFETY: Same as WakeByAddressSingle; wakes all threads waiting on this address.
        unsafe { WakeByAddressAll(ptr.cast()) };
    }
}

/// Fallback: spin-loop wait (no OS primitives available or no_std mode).
///
/// # Single-core targets
/// On single-core MCUs without preemption (e.g. Cortex-M), spin-waiting can
/// deadlock if the lock holder shares the same core. Users on single-core
/// bare-metal MUST run under an RTOS with preemptive scheduling, or replace
/// this module with an interrupt-based yield (e.g. `cortex_m::asm::wfe()`).
#[cfg(not(any(
    all(
        feature = "std",
        target_os = "linux",
        not(all(feature = "shuttle", test))
    ),
    all(
        feature = "std",
        target_os = "macos",
        not(all(feature = "shuttle", test))
    ),
    all(
        feature = "std",
        target_os = "windows",
        not(all(feature = "shuttle", test))
    ),
    all(feature = "shuttle", test)
)))]
mod platform {
    use core::sync::atomic::AtomicU32;
    #[inline]
    pub fn wait(_a: &AtomicU32, _expected: u32) {
        core::hint::spin_loop();
    }
    #[inline]
    pub fn wake_one(_ptr: *const AtomicU32) {}
    #[inline]
    pub fn wake_all(_ptr: *const AtomicU32) {}
}

/// If the value is `value`, wait until woken up.
///
/// This function might also return spuriously,
/// without a corresponding wake operation.
#[inline]
pub fn wait(atomic: &AtomicU32, value: u32) {
    platform::wait(atomic, value)
}

/// Wake one thread that is waiting on this atomic.
///
/// It's okay if the pointer dangles or is null.
#[inline]
pub fn wake_one(atomic: *const AtomicU32) {
    platform::wake_one(atomic);
}

/// Wake all threads that are waiting on this atomic.
///
/// It's okay if the pointer dangles or is null.
#[inline]
pub fn wake_all(atomic: *const AtomicU32) {
    platform::wake_all(atomic);
}
