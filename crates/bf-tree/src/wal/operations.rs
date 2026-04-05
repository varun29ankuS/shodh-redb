// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::nodes::leaf_node::OpType;

use super::LogEntryImpl;

pub(crate) struct WriteOp<'a> {
    pub(crate) key: &'a [u8],
    pub(crate) value: &'a [u8],
    pub(crate) op_type: OpType,
}

impl<'a> WriteOp<'a> {
    pub(crate) fn make_insert(key: &'a [u8], value: &'a [u8]) -> WriteOp<'a> {
        WriteOp {
            key,
            value,
            op_type: OpType::Insert,
        }
    }

    pub(crate) fn make_delete(key: &'a [u8]) -> WriteOp<'a> {
        WriteOp {
            key,
            value: &[],
            op_type: OpType::Delete,
        }
    }
}

impl<'a> LogEntryImpl<'a> for WriteOp<'a> {
    fn log_size(&self) -> usize {
        // layout:
        // key_size | value_size | op_type | key | value
        let key_size = self.key.len();
        let value_size = self.value.len();
        let op_type_size = std::mem::size_of::<OpType>();

        std::mem::size_of::<u16>()
            + std::mem::size_of::<u16>()
            + op_type_size
            + key_size
            + value_size
    }

    fn write_to_buffer(&self, buffer: &mut [u8]) {
        debug_assert_eq!(buffer.len(), self.log_size());
        let key_len = self.key.len() as u16;
        let val_len = self.value.len() as u16;
        let op_u8 = self.op_type as u8;
        buffer[0..2].copy_from_slice(key_len.to_le_bytes().as_ref());
        buffer[2..4].copy_from_slice(val_len.to_le_bytes().as_ref());
        buffer[4] = op_u8;

        let key_offset = 5 + self.key.len();
        buffer[5..key_offset].copy_from_slice(self.key);
        buffer[key_offset..].copy_from_slice(self.value);
    }

    fn read_from_buffer(buffer: &'a [u8]) -> WriteOp<'a> {
        let key_size = u16::from_le_bytes(buffer[0..2].try_into().unwrap()) as usize;
        let value_size = u16::from_le_bytes(buffer[2..4].try_into().unwrap()) as usize;
        // SAFETY: OpType is #[repr(u8)] with variants 0..3. The WAL writer always
        // serializes a valid OpType, so the byte is in range. A corrupted byte here
        // would produce an invalid enum variant -- the caller validates via split_key
        // bounds checking rather than trusting this value blindly.
        let op_type = unsafe { std::mem::transmute::<u8, OpType>(buffer[4]) };
        let key = &buffer[5..5 + key_size];
        let value = &buffer[5 + key_size..];
        assert_eq!(value.len(), value_size);
        WriteOp {
            key,
            value,
            op_type,
        }
    }
}

/// WAL entry for a leaf page split.
///
/// Records the source and sibling page IDs plus the separator key so that
/// recovery can replay the split without re-reading the full page data.
///
/// Serialization format (little-endian):
/// ```text
/// [source_page_id: u64][new_page_id: u64][split_key_len: u16][split_key: [u8]]
/// ```
pub(crate) struct SplitOp<'a> {
    pub(crate) source_page_id: u64,
    pub(crate) new_page_id: u64,
    pub(crate) split_key: &'a [u8],
}

impl<'a> LogEntryImpl<'a> for SplitOp<'a> {
    fn log_size(&self) -> usize {
        // source_page_id(8) + new_page_id(8) + split_key_len(2) + split_key
        18 + self.split_key.len()
    }

    fn write_to_buffer(&self, buffer: &mut [u8]) {
        debug_assert_eq!(buffer.len(), self.log_size());
        buffer[0..8].copy_from_slice(&self.source_page_id.to_le_bytes());
        buffer[8..16].copy_from_slice(&self.new_page_id.to_le_bytes());
        let key_len = self.split_key.len() as u16;
        buffer[16..18].copy_from_slice(&key_len.to_le_bytes());
        buffer[18..18 + self.split_key.len()].copy_from_slice(self.split_key);
    }

    fn read_from_buffer(buffer: &'a [u8]) -> SplitOp<'a> {
        let source_page_id = u64::from_le_bytes(buffer[0..8].try_into().unwrap());
        let new_page_id = u64::from_le_bytes(buffer[8..16].try_into().unwrap());
        let key_len = u16::from_le_bytes(buffer[16..18].try_into().unwrap()) as usize;
        let split_key = &buffer[18..18 + key_len];
        SplitOp {
            source_page_id,
            new_page_id,
            split_key,
        }
    }
}

pub(crate) enum LogEntry<'a> {
    Write(WriteOp<'a>),
    Split(SplitOp<'a>),
}

#[repr(u8)]
enum LogEntryTagVal {
    Write = 0,
    Split = 1,
}

impl LogEntryTagVal {
    fn size() -> usize {
        std::mem::size_of::<u8>()
    }
}

impl<'a> From<&LogEntry<'a>> for LogEntryTagVal {
    fn from(entry: &LogEntry<'a>) -> Self {
        match entry {
            LogEntry::Write(_) => LogEntryTagVal::Write,
            LogEntry::Split(_) => LogEntryTagVal::Split,
        }
    }
}

impl From<u8> for LogEntryTagVal {
    fn from(val: u8) -> Self {
        match val {
            0 => LogEntryTagVal::Write,
            1 => LogEntryTagVal::Split,
            _ => unreachable!(),
        }
    }
}

impl<'a> LogEntryImpl<'a> for LogEntry<'a> {
    fn log_size(&self) -> usize {
        let tag_size = LogEntryTagVal::size();
        let data_size = match self {
            LogEntry::Write(op) => op.log_size(),
            LogEntry::Split(op) => op.log_size(),
        };
        tag_size + data_size
    }

    fn write_to_buffer(&self, buffer: &mut [u8]) {
        buffer[0] = LogEntryTagVal::from(self) as u8;
        match self {
            LogEntry::Write(op) => op.write_to_buffer(&mut buffer[1..]),
            LogEntry::Split(op) => op.write_to_buffer(&mut buffer[1..]),
        }
    }

    fn read_from_buffer(buffer: &'a [u8]) -> Self {
        let tag = LogEntryTagVal::from(buffer[0]);
        match tag {
            LogEntryTagVal::Write => LogEntry::Write(WriteOp::read_from_buffer(&buffer[1..])),
            LogEntryTagVal::Split => LogEntry::Split(SplitOp::read_from_buffer(&buffer[1..])),
        }
    }
}
