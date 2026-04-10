// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use alloc::string::String;
use core::fmt;

#[derive(Debug)]
pub(crate) enum TreeError {
    Locked,
    CircularBufferFull,
    NeedRestart, // need to restart the operation, potentially will do SMO operations
    IoError(IoErrorKind),
}

/// Describes the kind of I/O failure that occurred within bf-tree internals.
///
/// Kept `no_std`-compatible (no `std::io::Error` dependency).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoErrorKind {
    VfsRead {
        offset: usize,
    },
    VfsWrite {
        offset: usize,
    },
    VfsFlush,
    WalAppend,
    WalFlush,
    SnapshotRead,
    SnapshotWrite,
    ConfigRead,
    ConfigParse,
    Corruption,
    ChecksumMismatch {
        offset: usize,
    },
    /// Operation attempted on a deallocated or uninitialized (Null) page.
    NullPage,
    /// Internal state machine invariant violated (indicates a bug or corruption).
    InvariantViolation,
    /// Disk operation attempted on a cache-only (in-memory) tree.
    CacheOnlyViolation,
    /// Record exceeds the maximum size supported by a mini-page.
    RecordTooLarge,
}

impl fmt::Display for IoErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoErrorKind::VfsRead { offset } => write!(f, "VFS read failed at offset {}", offset),
            IoErrorKind::VfsWrite { offset } => write!(f, "VFS write failed at offset {}", offset),
            IoErrorKind::VfsFlush => write!(f, "VFS flush/sync failed"),
            IoErrorKind::WalAppend => write!(f, "WAL append failed"),
            IoErrorKind::WalFlush => write!(f, "WAL flush failed"),
            IoErrorKind::SnapshotRead => write!(f, "snapshot read failed"),
            IoErrorKind::SnapshotWrite => write!(f, "snapshot write failed"),
            IoErrorKind::ConfigRead => write!(f, "config file read failed"),
            IoErrorKind::ConfigParse => write!(f, "config file parse failed"),
            IoErrorKind::Corruption => write!(f, "data corruption detected"),
            IoErrorKind::ChecksumMismatch { offset } => {
                write!(f, "CRC-32 checksum mismatch at disk page offset {}", offset)
            }
            IoErrorKind::NullPage => write!(f, "operation on deallocated/uninitialized page"),
            IoErrorKind::InvariantViolation => {
                write!(f, "internal state machine invariant violated")
            }
            IoErrorKind::CacheOnlyViolation => {
                write!(f, "disk operation on cache-only tree")
            }
            IoErrorKind::RecordTooLarge => write!(f, "record exceeds mini-page capacity"),
        }
    }
}

/// Public error type for bf-tree API boundaries.
#[derive(Debug)]
pub enum BfTreeError {
    Config(ConfigError),
    Io(IoErrorKind),
}

impl fmt::Display for BfTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BfTreeError::Config(e) => write!(f, "config error: {:?}", e),
            BfTreeError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl From<ConfigError> for BfTreeError {
    fn from(e: ConfigError) -> Self {
        BfTreeError::Config(e)
    }
}

impl From<IoErrorKind> for BfTreeError {
    fn from(e: IoErrorKind) -> Self {
        BfTreeError::Io(e)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ConfigError {
    MinimumRecordSize(String),
    MaximumRecordSize(String),
    LeafPageSize(String),
    MaxKeyLen(String),
    CircularBufferSize(String),
}
