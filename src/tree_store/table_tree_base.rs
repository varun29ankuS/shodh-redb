use crate::compat::Arc;
use crate::compat::{HashMap, Mutex};
use crate::multimap_table::{UntypedMultiBtree, relocate_subtrees};
use crate::tree_store::{
    BtreeHeader, PageNumber, PagePath, TransactionalMemory, UntypedBtree, UntypedBtreeMut,
};
use crate::{Key, Result, TableError, TypeName, Value};
use alloc::format;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;

// Forward compatibility feature in case alignment can be supported in the future
// See https://github.com/cberner/redb/issues/360
const ALIGNMENT: usize = 1;

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub(crate) enum TableType {
    Normal,
    Multimap,
}

impl TableType {
    fn is_legacy(value: u8) -> bool {
        value == 1 || value == 2
    }
}

#[allow(clippy::from_over_into)]
impl Into<u8> for TableType {
    fn into(self) -> u8 {
        match self {
            // 1 & 2 were used in the v1 file format
            // TableType::Normal => 1,
            // TableType::Multimap => 2,
            TableType::Normal => 3,
            TableType::Multimap => 4,
        }
    }
}

impl TableType {
    fn try_from_byte(value: u8) -> crate::Result<Self> {
        match value {
            3 => Ok(TableType::Normal),
            4 => Ok(TableType::Multimap),
            _ => Err(crate::StorageError::Corrupted(format!(
                "invalid TableType byte: {value}, expected 3 (Normal) or 4 (Multimap)"
            ))),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) enum InternalTableDefinition {
    Normal {
        table_root: Option<BtreeHeader>,
        table_length: u64,
        fixed_key_size: Option<usize>,
        fixed_value_size: Option<usize>,
        key_alignment: usize,
        value_alignment: usize,
        key_type: TypeName,
        value_type: TypeName,
    },
    Multimap {
        table_root: Option<BtreeHeader>,
        table_length: u64,
        fixed_key_size: Option<usize>,
        fixed_value_size: Option<usize>,
        key_alignment: usize,
        value_alignment: usize,
        key_type: TypeName,
        value_type: TypeName,
    },
}

impl InternalTableDefinition {
    pub(super) fn new<K: Key, V: Value>(
        table_type: TableType,
        table_root: Option<BtreeHeader>,
        table_length: u64,
    ) -> Self {
        match table_type {
            TableType::Normal => InternalTableDefinition::Normal {
                table_root,
                table_length,
                fixed_key_size: K::fixed_width(),
                fixed_value_size: V::fixed_width(),
                key_alignment: ALIGNMENT,
                value_alignment: ALIGNMENT,
                key_type: K::type_name(),
                value_type: V::type_name(),
            },
            TableType::Multimap => InternalTableDefinition::Multimap {
                table_root,
                table_length,
                fixed_key_size: K::fixed_width(),
                fixed_value_size: V::fixed_width(),
                key_alignment: ALIGNMENT,
                value_alignment: ALIGNMENT,
                key_type: K::type_name(),
                value_type: V::type_name(),
            },
        }
    }

    pub(super) fn set_header(&mut self, root: Option<BtreeHeader>, length: u64) {
        match self {
            InternalTableDefinition::Normal {
                table_root,
                table_length,
                ..
            }
            | InternalTableDefinition::Multimap {
                table_root,
                table_length,
                ..
            } => {
                *table_root = root;
                *table_length = length;
            }
        }
    }

    pub(super) fn check_match_untyped(
        &self,
        table_type: TableType,
        name: &str,
    ) -> Result<(), TableError> {
        if self.get_type() != table_type {
            return if self.get_type() == TableType::Multimap {
                Err(TableError::TableIsMultimap(name.to_string()))
            } else {
                Err(TableError::TableIsNotMultimap(name.to_string()))
            };
        }
        if self.private_get_key_alignment() != ALIGNMENT {
            return Err(TableError::TypeDefinitionChanged {
                name: self.private_key_type(),
                alignment: self.private_get_key_alignment(),
                width: self.private_get_fixed_key_size(),
            });
        }
        if self.private_get_value_alignment() != ALIGNMENT {
            return Err(TableError::TypeDefinitionChanged {
                name: self.private_value_type(),
                alignment: self.private_get_value_alignment(),
                width: self.private_get_fixed_value_size(),
            });
        }

        Ok(())
    }

    pub(super) fn check_match<K: Key, V: Value>(
        &self,
        table_type: TableType,
        name: &str,
    ) -> Result<(), TableError> {
        self.check_match_untyped(table_type, name)?;

        if self.private_key_type() != K::type_name() || self.private_value_type() != V::type_name()
        {
            return Err(TableError::TableTypeMismatch {
                table: name.to_string(),
                key: self.private_key_type(),
                value: self.private_value_type(),
            });
        }
        if self.private_get_fixed_key_size() != K::fixed_width() {
            return Err(TableError::TypeDefinitionChanged {
                name: K::type_name(),
                alignment: self.private_get_key_alignment(),
                width: self.private_get_fixed_key_size(),
            });
        }
        if self.private_get_fixed_value_size() != V::fixed_width() {
            return Err(TableError::TypeDefinitionChanged {
                name: V::type_name(),
                alignment: self.private_get_value_alignment(),
                width: self.private_get_fixed_value_size(),
            });
        }

        Ok(())
    }

    pub(crate) fn visit_all_pages<'a, F>(&self, mem: Arc<TransactionalMemory>, visitor: F) -> Result
    where
        F: FnMut(&PagePath) -> Result + 'a,
    {
        match self {
            InternalTableDefinition::Normal {
                table_root,
                fixed_key_size,
                fixed_value_size,
                ..
            } => {
                let effective_value_size = if mem.compression().is_enabled() {
                    None
                } else {
                    *fixed_value_size
                };
                let tree =
                    UntypedBtree::new(*table_root, mem, *fixed_key_size, effective_value_size);
                tree.visit_all_pages(visitor)?;
            }
            InternalTableDefinition::Multimap {
                table_root,
                fixed_key_size,
                fixed_value_size,
                ..
            } => {
                let effective_value_size = if mem.compression().is_enabled() {
                    None
                } else {
                    *fixed_value_size
                };
                let tree =
                    UntypedMultiBtree::new(*table_root, mem, *fixed_key_size, effective_value_size);
                tree.visit_all_pages(visitor)?;
            }
        }

        Ok(())
    }

    pub(crate) fn relocate_tree(
        &mut self,
        mem: Arc<TransactionalMemory>,
        freed_pages: Arc<Mutex<Vec<PageNumber>>>,
        relocation_map: &HashMap<PageNumber, PageNumber>,
    ) -> Result<Option<BtreeHeader>> {
        let original_root = self.private_get_root();
        let effective_value_size = if mem.compression().is_enabled() {
            None
        } else {
            self.private_get_fixed_value_size()
        };
        let relocated_root = match self {
            InternalTableDefinition::Normal { table_root, .. } => *table_root,
            InternalTableDefinition::Multimap {
                table_root,
                fixed_key_size,
                ..
            } => {
                if let Some(header) = table_root {
                    let (page_number, checksum) = relocate_subtrees(
                        (header.root, header.checksum),
                        *fixed_key_size,
                        effective_value_size,
                        mem.clone(),
                        freed_pages.clone(),
                        relocation_map,
                    )?;
                    Some(BtreeHeader::new(page_number, checksum, header.length))
                } else {
                    None
                }
            }
        };
        let mut tree = UntypedBtreeMut::new(
            relocated_root,
            mem,
            freed_pages,
            self.private_get_fixed_key_size(),
            effective_value_size,
        );
        tree.relocate(relocation_map)?;
        if tree.get_root() != original_root {
            self.set_header(tree.get_root(), self.get_length());
            Ok(tree.get_root())
        } else {
            Ok(None)
        }
    }

    fn private_get_root(&self) -> Option<BtreeHeader> {
        match self {
            InternalTableDefinition::Normal { table_root, .. }
            | InternalTableDefinition::Multimap { table_root, .. } => *table_root,
        }
    }

    pub(crate) fn get_length(&self) -> u64 {
        match self {
            InternalTableDefinition::Normal { table_length, .. }
            | InternalTableDefinition::Multimap { table_length, .. } => *table_length,
        }
    }

    fn private_get_fixed_key_size(&self) -> Option<usize> {
        match self {
            InternalTableDefinition::Normal { fixed_key_size, .. }
            | InternalTableDefinition::Multimap { fixed_key_size, .. } => *fixed_key_size,
        }
    }

    fn private_get_fixed_value_size(&self) -> Option<usize> {
        match self {
            InternalTableDefinition::Normal {
                fixed_value_size, ..
            }
            | InternalTableDefinition::Multimap {
                fixed_value_size, ..
            } => *fixed_value_size,
        }
    }

    fn private_get_key_alignment(&self) -> usize {
        match self {
            InternalTableDefinition::Normal { key_alignment, .. }
            | InternalTableDefinition::Multimap { key_alignment, .. } => *key_alignment,
        }
    }

    fn private_get_value_alignment(&self) -> usize {
        match self {
            InternalTableDefinition::Normal {
                value_alignment, ..
            }
            | InternalTableDefinition::Multimap {
                value_alignment, ..
            } => *value_alignment,
        }
    }

    pub(crate) fn get_type(&self) -> TableType {
        match self {
            InternalTableDefinition::Normal { .. } => TableType::Normal,
            InternalTableDefinition::Multimap { .. } => TableType::Multimap,
        }
    }

    fn private_key_type(&self) -> TypeName {
        match self {
            InternalTableDefinition::Normal { key_type, .. }
            | InternalTableDefinition::Multimap { key_type, .. } => key_type.clone(),
        }
    }

    fn private_value_type(&self) -> TypeName {
        match self {
            InternalTableDefinition::Normal { value_type, .. }
            | InternalTableDefinition::Multimap { value_type, .. } => value_type.clone(),
        }
    }
}

impl Value for InternalTableDefinition {
    type SelfType<'a> = InternalTableDefinition;
    type AsBytes<'a> = Vec<u8>;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self
    where
        Self: 'a,
    {
        // Minimum length: 1 (type) + 8 (length) + 1 (root null) + BtreeHeader::serialized_size()
        // + 1 (key null) + 4 (key size) + 1 (val null) + 4 (val size) + 4 (key align)
        // + 4 (val align) + 4 (key_type_len) = 32 + BtreeHeader::serialized_size()
        // NOTE: Value trait prevents returning Result; controlled panic is the best we can do.
        let min_len = 1
            + size_of::<u64>()
            + 1
            + BtreeHeader::serialized_size()
            + 1
            + size_of::<u32>()
            + 1
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>();
        if data.len() < min_len {
            // Truncated metadata. The Value trait forces this function to
            // return Self, and the previous "controlled panic" here was
            // reachable from a crafted database image, so degrade instead.
            //
            // Alignment 0 is deliberately invalid -- ALIGNMENT is 1 -- so
            // check_match_untyped rejects this with TypeDefinitionChanged
            // before any caller can act on it. That matters because the
            // untyped API does not compare type names, so returning a
            // plausible-looking empty table would silently mask corruption.
            return InternalTableDefinition::Normal {
                table_root: None,
                table_length: 0,
                fixed_key_size: None,
                fixed_value_size: None,
                key_alignment: 0,
                value_alignment: 0,
                key_type: TypeName::from_bytes(&[]),
                value_type: TypeName::from_bytes(&[]),
            };
        }
        let mut offset = 0;
        let legacy = TableType::is_legacy(data[offset]);
        debug_assert!(!legacy);
        let table_type = TableType::try_from_byte(data[offset]).unwrap_or(TableType::Normal);
        offset += 1;

        let table_length = u64::from_le_bytes(
            data[offset..(offset + size_of::<u64>())]
                .try_into()
                .unwrap(),
        );
        offset += size_of::<u64>();

        let non_null = data[offset] != 0;
        offset += 1;
        let table_root = if non_null {
            Some(BtreeHeader::from_le_bytes(
                data[offset..(offset + BtreeHeader::serialized_size())]
                    .try_into()
                    .unwrap(),
            ))
        } else {
            None
        };
        offset += BtreeHeader::serialized_size();

        let non_null = data[offset] != 0;
        offset += 1;
        let fixed_key_size = if non_null {
            let fixed = u32::from_le_bytes(
                data[offset..(offset + size_of::<u32>())]
                    .try_into()
                    .unwrap(),
            ) as usize;
            Some(fixed)
        } else {
            None
        };
        offset += size_of::<u32>();

        let non_null = data[offset] != 0;
        offset += 1;
        let fixed_value_size = if non_null {
            let fixed = u32::from_le_bytes(
                data[offset..(offset + size_of::<u32>())]
                    .try_into()
                    .unwrap(),
            ) as usize;
            Some(fixed)
        } else {
            None
        };
        offset += size_of::<u32>();
        let key_alignment = u32::from_le_bytes(
            data[offset..(offset + size_of::<u32>())]
                .try_into()
                .unwrap(),
        ) as usize;
        offset += size_of::<u32>();
        let value_alignment = u32::from_le_bytes(
            data[offset..(offset + size_of::<u32>())]
                .try_into()
                .unwrap(),
        ) as usize;
        offset += size_of::<u32>();

        let key_type_len = u32::from_le_bytes(
            data[offset..(offset + size_of::<u32>())]
                .try_into()
                .unwrap(),
        ) as usize;
        offset += size_of::<u32>();
        // `key_type_len` is read off disk and is NOT covered by the min_len
        // assert above, which only spans the fixed-size prefix. A corrupted
        // value sliced past the buffer:
        //   range end index 12648546 out of range for slice of length 113
        // The Value trait forces this function to return Self, so there is no
        // error to return -- but clamping is strictly better than panicking.
        // A truncated type name simply fails the later type comparison, which
        // reports corruption through a normal error path.
        let key_type_end = offset.saturating_add(key_type_len).min(data.len());
        let key_type = TypeName::from_bytes(&data[offset..key_type_end]);
        offset = key_type_end;
        let value_type = TypeName::from_bytes(&data[offset..]);

        match table_type {
            TableType::Normal => InternalTableDefinition::Normal {
                table_root,
                table_length,
                fixed_key_size,
                fixed_value_size,
                key_alignment,
                value_alignment,
                key_type,
                value_type,
            },
            TableType::Multimap => InternalTableDefinition::Multimap {
                table_root,
                table_length,
                fixed_key_size,
                fixed_value_size,
                key_alignment,
                value_alignment,
                key_type,
                value_type,
            },
        }
    }

    // Be careful if you change this serialization format! The InternalTableDefinition for
    // a given table needs to have a consistent serialized length, regardless of the table
    // contents, so that create_table_and_flush_table_root() can update the allocator state
    // table without causing more allocations
    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Vec<u8>
    where
        Self: 'b,
    {
        let mut result = vec![value.get_type().into()];
        result.extend_from_slice(&value.get_length().to_le_bytes());
        if let Some(header) = value.private_get_root() {
            result.push(1);
            result.extend_from_slice(&header.to_le_bytes());
        } else {
            result.push(0);
            result.extend_from_slice(&[0; BtreeHeader::serialized_size()]);
        }
        if let Some(fixed) = value.private_get_fixed_key_size() {
            result.push(1);
            result.extend_from_slice(&u32::try_from(fixed).unwrap().to_le_bytes());
        } else {
            result.push(0);
            result.extend_from_slice(&[0; size_of::<u32>()]);
        }
        if let Some(fixed) = value.private_get_fixed_value_size() {
            result.push(1);
            result.extend_from_slice(&u32::try_from(fixed).unwrap().to_le_bytes());
        } else {
            result.push(0);
            result.extend_from_slice(&[0; size_of::<u32>()]);
        }
        result.extend_from_slice(
            &u32::try_from(value.private_get_key_alignment())
                .unwrap()
                .to_le_bytes(),
        );
        result.extend_from_slice(
            &u32::try_from(value.private_get_value_alignment())
                .unwrap()
                .to_le_bytes(),
        );
        let key_type_bytes = value.private_key_type().to_bytes();
        result.extend_from_slice(&u32::try_from(key_type_bytes.len()).unwrap().to_le_bytes());
        result.extend_from_slice(&key_type_bytes);
        result.extend_from_slice(&value.private_value_type().to_bytes());

        result
    }

    fn type_name() -> TypeName {
        TypeName::internal("redb::InternalTableDefinition")
    }
}

#[cfg(test)]
mod corrupt_metadata_tests {
    use super::*;

    /// `key_type_len` is read off disk and lies outside the `min_len` assert,
    /// which only spans the fixed-size prefix. A corrupted value used to slice
    /// past the buffer:
    ///
    /// ```text
    /// range end index 12648546 out of range for slice of length 113
    /// ```
    ///
    /// Found by the `fuzz_db_image` target. `Value::from_bytes` cannot return
    /// an error, so the requirement is simply that it does not panic.
    /// Truncated metadata used to hit a deliberate `assert!`, reachable from a
    /// crafted database image. It must degrade instead -- and the degradation
    /// must be rejected, not merely plausible: the untyped API does not compare
    /// type names, so an empty-looking table would mask the corruption.
    #[test]
    fn truncated_metadata_degrades_to_a_rejected_definition() {
        for len in [0usize, 1, 8, 32, 63] {
            let data = vec![0u8; len];
            let def = <InternalTableDefinition as Value>::from_bytes(&data);
            // Alignment 0 is invalid (ALIGNMENT is 1), so every caller path
            // through check_match_untyped refuses it.
            assert!(
                def.check_match_untyped(TableType::Normal, "t").is_err(),
                "truncated metadata of {len} bytes must be rejected"
            );
        }
    }

    #[test]
    fn absurd_key_type_len_does_not_slice_past_the_buffer() {
        // Fixed-size prefix, matching the layout from_bytes expects.
        let mut data = vec![0u8; 64];
        data[0] = 3; // TableType::Normal (1 and 2 are the legacy v1 encodings)
        // table_length at 1..9, root null flag at 9, BtreeHeader at 10..42,
        // fixed key/value flags and sizes at 42..60 -- all zero is fine.
        // key_type_len at 60..64: absurd.
        data[60..64].copy_from_slice(&0xC0FFEEu32.to_le_bytes());
        // A little trailing data, so the buffer is longer than min_len but far
        // shorter than the claimed key type name.
        data.extend_from_slice(&[0u8; 49]);

        // Must return, not panic.
        let def = <InternalTableDefinition as Value>::from_bytes(&data);
        // The type name is necessarily truncated; it just has to be produced.
        let _ = format!("{def:?}");
    }
}
