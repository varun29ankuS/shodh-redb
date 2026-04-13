# shodh-redb
Embedded multi-modal database engine in Rust. B-tree page store with vector indexing (IVF-PQ), blob store, CDC, TTL, merge operators. no_std compatible core, std feature for file backends.

# Build & Test
- IMPORTANT: Do NOT run `cargo build` or `trunk serve` -- user runs builds in background
- Test single: `cargo test <name> --lib`
- Clippy: `cargo clippy -- -D warnings`
- Fuzz: `cargo fuzz run --sanitizer=none <target> -- -max_len=10000 -max_total_time=30`

# Code
- MUST be production grade: no TODOs, placeholders, mocks, or stubs
- Understand architecture and data flow BEFORE modifying anything
- All unsafe blocks MUST have a `// SAFETY:` comment
- Match existing error patterns in `src/error.rs`
- Non-ASCII characters are banned (CI check enforces this)

# Architecture
- `src/tree_store/` -- B-tree core, page store, buddy allocator, flash FTL
- `src/ivfpq/` -- IVF-PQ vector indexing (kmeans, codebooks)
- `src/blob_store/` -- Content-addressed blob storage with causal tracking
- `src/vector.rs`, `src/vector_ops.rs` -- Vector types, distance metrics, quantization

# Git
- IMPORTANT: Do NOT add "Generated with Claude Code" or "Co-Authored-By" to commits
- Branch: `fix/`, `feat/`, `refactor/` prefixes
- Message format: `<type>: <description>`

# Context Preservation
When compacting, preserve: modified file list, test results, open issue numbers, current branch name.
