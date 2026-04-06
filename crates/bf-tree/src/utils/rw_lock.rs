// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use core::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
};

use crate::sync::atomic::{AtomicU32, Ordering};
use crate::utils::atomic_wait;

/// A RWLock with upgrade operation.
/// We use this because std RWLock don't allow lock upgrade.
///
/// Borrowed from https://marabos.nl/atomics/building-locks.html#reader-writer-lock
pub struct RwLock<T> {
    val: UnsafeCell<T>,
    lock_val: AtomicU32,
    writer_wake_counter: AtomicU32,
}

impl<T> RwLock<T> {
    pub fn new(val: T) -> Self {
        Self {
            val: UnsafeCell::new(val),
            lock_val: AtomicU32::new(0),
            writer_wake_counter: AtomicU32::new(0),
        }
    }

    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        let mut v = self.lock_val.load(Ordering::Relaxed);
        loop {
            if v.is_multiple_of(2) {
                match self.lock_val.compare_exchange_weak(
                    v,
                    v + 2,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return RwLockReadGuard { lock: self },
                    Err(e) => v = e,
                }
            }

            if !v.is_multiple_of(2) {
                atomic_wait::wait(&self.lock_val, v);
                v = self.lock_val.load(Ordering::Relaxed);
            }
        }
    }

    pub fn try_read(&self) -> Result<RwLockReadGuard<'_, T>, ()> {
        let v = self.lock_val.load(Ordering::Relaxed);

        if v.is_multiple_of(2) {
            let new_v = v + 2;

            match self.lock_val.compare_exchange_weak(
                v,
                new_v,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => Ok(RwLockReadGuard { lock: self }),
                Err(_) => Err(()),
            }
        } else {
            Err(())
        }
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        let mut s = self.lock_val.load(Ordering::Relaxed);
        loop {
            if s <= 1 {
                match self.lock_val.compare_exchange(
                    s,
                    u32::MAX,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return RwLockWriteGuard { lock: self },
                    Err(e) => {
                        s = e;
                        continue;
                    }
                }
            }

            if s.is_multiple_of(2) {
                match self
                    .lock_val
                    .compare_exchange(s, s + 1, Ordering::Relaxed, Ordering::Relaxed)
                {
                    Ok(_) => {}
                    Err(e) => {
                        s = e;
                        continue;
                    }
                }
            }

            let w = self.writer_wake_counter.load(Ordering::Acquire);
            s = self.lock_val.load(Ordering::Relaxed);
            if s >= 2 && !s.is_multiple_of(2) {
                atomic_wait::wait(&self.writer_wake_counter, w);
                s = self.lock_val.load(Ordering::Relaxed);
            }
        }
    }

    pub fn try_write(&self) -> Result<RwLockWriteGuard<'_, T>, ()> {
        let s = self.lock_val.load(Ordering::Relaxed);
        if s <= 1 {
            match self.lock_val.compare_exchange_weak(
                s,
                u32::MAX,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(RwLockWriteGuard { lock: self }),
                Err(_) => return Err(()),
            }
        }
        Err(())
    }
}

pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Drop for RwLockReadGuard<'_, T> {
    fn drop(&mut self) {
        if self.lock.lock_val.fetch_sub(2, Ordering::Release) == 3 {
            self.lock
                .writer_wake_counter
                .fetch_add(1, Ordering::Release);
            atomic_wait::wake_one(&self.lock.writer_wake_counter);
        }
    }
}

impl<T> Deref for RwLockReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, T> RwLockReadGuard<'a, T> {
    /// Attempt to atomically upgrade a read lock to a write lock.
    ///
    /// Succeeds only when the calling thread holds the **sole** read lock
    /// (`lock_val == 2`). With any other readers, the CAS fails and the
    /// read guard is returned in `Err`.
    ///
    /// This prevents deadlock: if two readers tried to upgrade simultaneously,
    /// neither could succeed. By requiring sole-reader status, at most one
    /// upgrade can succeed.
    ///
    /// On failure, release the read lock and retry with `write()`.
    pub fn try_upgrade(self) -> Result<RwLockWriteGuard<'a, T>, RwLockReadGuard<'a, T>> {
        // Each reader adds 2 to lock_val, so a single reader means lock_val == 2.
        // We CAS from 2 -> u32::MAX (write-locked).
        let old_v = 2;

        match self.lock.lock_val.compare_exchange_weak(
            old_v,
            u32::MAX,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                let lock = self.lock;
                // SAFETY: We successfully CAS'd lock_val from 2 (one reader) to
                // u32::MAX (writer). We must forget `self` to avoid the read-guard
                // Drop decrementing lock_val, which would corrupt the lock state.
                core::mem::forget(self);
                Ok(RwLockWriteGuard { lock })
            }
            Err(_e) => Err(self),
        }
    }

    pub(crate) fn as_ref(&self) -> &T {
        // SAFETY: We hold a read lock (lock_val >= 2, even), so no writer
        // exists and shared access is safe.
        unsafe { &*self.lock.val.get() }
    }
}

pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T> DerefMut for RwLockWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<T> Drop for RwLockWriteGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.lock_val.store(0, Ordering::Release);
        self.lock
            .writer_wake_counter
            .fetch_add(1, Ordering::Release);
        atomic_wait::wake_one(&self.lock.writer_wake_counter);
        atomic_wait::wake_all(&self.lock.lock_val);
    }
}

impl<T> RwLockWriteGuard<'_, T> {
    pub(crate) fn as_mut(&mut self) -> &mut T {
        // SAFETY: We hold the write lock (lock_val == u32::MAX), so we have
        // exclusive access to the inner value.
        unsafe { &mut *self.lock.val.get() }
    }

    pub(crate) fn as_ref(&self) -> &T {
        // SAFETY: We hold the write lock (lock_val == u32::MAX), so no other
        // reader or writer can access the value.
        unsafe { &*self.lock.val.get() }
    }
}

#[cfg(all(test, feature = "std", not(feature = "shuttle")))]
mod tests {
    use super::*;

    #[test]
    fn single_reader_upgrade_succeeds() {
        let lock = RwLock::new(42u32);
        let read_guard = lock.read();
        assert_eq!(*read_guard, 42);

        let write_guard = match read_guard.try_upgrade() {
            Ok(w) => w,
            Err(_) => panic!("sole reader upgrade must succeed"),
        };
        assert_eq!(*write_guard, 42);
        drop(write_guard);

        // Lock is usable again after drop.
        let g = lock.read();
        assert_eq!(*g, 42);
    }

    #[test]
    fn upgrade_fails_with_multiple_readers() {
        let lock = RwLock::new(0u32);
        let r1 = lock.read();
        let r2 = lock.read();

        // With two readers, upgrade must fail.
        let r1 = match r1.try_upgrade() {
            Err(guard) => guard,
            Ok(_) => panic!("upgrade must fail with two readers"),
        };
        drop(r1);
        drop(r2);

        // After releasing both, a fresh write lock succeeds.
        let mut w = lock.write();
        *w = 99;
        drop(w);
        assert_eq!(*lock.read(), 99);
    }

    #[test]
    fn upgrade_failure_preserves_read_guard() {
        let lock = RwLock::new(7u32);
        let r1 = lock.read();
        let r2 = lock.read();

        // Upgrade fails, we get our read guard back.
        let r1_returned = match r1.try_upgrade() {
            Err(guard) => guard,
            Ok(_) => panic!("upgrade must fail with two readers"),
        };
        assert_eq!(*r1_returned, 7);

        drop(r1_returned);
        drop(r2);
    }

    #[test]
    fn read_write_mutual_exclusion() {
        let lock = RwLock::new(0u32);
        {
            let mut w = lock.write();
            *w = 10;
            // While write lock is held, try_read must fail.
            assert!(lock.try_read().is_err());
        }
        // After dropping write lock, read succeeds.
        assert_eq!(*lock.read(), 10);
    }

    #[test]
    fn concurrent_readers_allowed() {
        let lock = RwLock::new(42u32);
        let r1 = lock.read();
        let r2 = lock.try_read().expect("concurrent read must succeed");
        let r3 = lock.try_read().expect("concurrent read must succeed");
        assert_eq!(*r1, 42);
        assert_eq!(*r2, 42);
        assert_eq!(*r3, 42);
        drop(r1);
        drop(r2);
        drop(r3);
    }

    #[test]
    fn try_write_fails_while_read_held() {
        let lock = RwLock::new(0u32);
        let _r = lock.read();
        assert!(lock.try_write().is_err());
    }
}
