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
    VfsRead { offset: usize },
    VfsWrite { offset: usize },
    WalAppend,
    WalFlush,
    SnapshotRead,
    SnapshotWrite,
    Corruption,
}

impl fmt::Display for IoErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoErrorKind::VfsRead { offset } => write!(f, "VFS read failed at offset {}", offset),
            IoErrorKind::VfsWrite { offset } => write!(f, "VFS write failed at offset {}", offset),
            IoErrorKind::WalAppend => write!(f, "WAL append failed"),
            IoErrorKind::WalFlush => write!(f, "WAL flush failed"),
            IoErrorKind::SnapshotRead => write!(f, "snapshot read failed"),
            IoErrorKind::SnapshotWrite => write!(f, "snapshot write failed"),
            IoErrorKind::Corruption => write!(f, "data corruption detected"),
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
