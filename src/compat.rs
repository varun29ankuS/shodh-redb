pub(crate) use hashbrown::HashMap;
pub(crate) use hashbrown::HashSet;
pub(crate) use spin::Mutex;
pub(crate) use spin::RwLock;
pub(crate) use spin::RwLockReadGuard;
pub(crate) use spin::RwLockWriteGuard;

/// Reference-counted pointer, portable across targets without atomic CAS.
///
/// `alloc::sync::Arc` requires native pointer-width compare-and-swap, so it
/// does not exist on targets such as `thumbv6m-none-eabi` (Cortex-M0/M0+,
/// which includes the RP2040). On those targets we substitute
/// `portable_atomic_util::Arc`, which builds the same semantics on top of
/// `portable-atomic`'s CAS emulation.
///
/// # Constraint
///
/// `portable_atomic_util::Arc` cannot perform unsized coercion on stable Rust
/// -- `Arc<Concrete> -> Arc<dyn Trait>` needs the nightly-only
/// `portable_atomic_unstable_coerce_unsized` cfg (see
/// <https://github.com/taiki-e/portable-atomic/issues/143>). Constructing
/// `Arc<[T]>` from a `Vec<T>` is fine, since that is a `From` impl rather than
/// a coercion.
///
/// Consequently the two `Arc<dyn Trait>` users in this crate -- the database
/// observer and the read-verification callback -- are compiled out on targets
/// without CAS. See `crate::observer` and `crate::db::ReadVerificationCallback`.
#[cfg(target_has_atomic = "ptr")]
pub(crate) use alloc::sync::Arc;
#[cfg(not(target_has_atomic = "ptr"))]
pub(crate) use portable_atomic_util::Arc;

/// `true` when this target supports the dynamic-dispatch extension points
/// (observer, read-verification callback). These require `Arc<dyn Trait>`,
/// which is unavailable without atomic CAS.
#[allow(dead_code)]
pub(crate) const HAS_DYN_EXTENSIONS: bool = cfg!(target_has_atomic = "ptr");
