//! Smoke test: verify bf-tree core types compile and instantiate without std.
//! Run with: cargo test -p bf-tree --no-default-features --test no_std_smoke

use bf_tree::Config;

#[test]
fn config_default_is_memory() {
    let cfg = Config::default();
    assert!(cfg.is_memory_backend());
}

#[test]
fn config_memory_roundtrip() {
    let cfg = Config::new_memory(1024 * 1024);
    assert!(cfg.is_memory_backend());
    assert_eq!(cfg.get_cb_size_byte(), 1024 * 1024);
}
