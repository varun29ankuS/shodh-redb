use crate::types::{TypeName, Value};
use alloc::format;
use alloc::vec::Vec;

// Encode len as a varint and store it at the end of output
pub(super) fn encode_varint_len(len: usize, output: &mut Vec<u8>) {
    if len < 254 {
        output.push(len.try_into().unwrap());
    } else if len <= u16::MAX.into() {
        let u16_len: u16 = len.try_into().unwrap();
        output.push(254);
        output.extend_from_slice(&u16_len.to_le_bytes());
    } else {
        let u32_len: u32 = len.try_into().unwrap();
        output.push(255);
        output.extend_from_slice(&u32_len.to_le_bytes());
    }
}

// Decode a variable length int starting at the beginning of data
// Returns (decoded length, length consumed of `data`)
// Returns (0, 0) if data is too short to decode, preventing panics on corrupted input.
pub(super) fn decode_varint_len(data: &[u8]) -> (usize, usize) {
    if data.is_empty() {
        return (0, 0);
    }
    match data[0] {
        0..=253 => (data[0] as usize, 1),
        254 => {
            if data.len() < 3 {
                return (0, data.len());
            }
            (u16::from_le_bytes([data[1], data[2]]) as usize, 3)
        }
        255 => {
            if data.len() < 5 {
                return (0, data.len());
            }
            (
                u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize,
                5,
            )
        }
    }
}

impl<T: Value> Value for Vec<T> {
    type SelfType<'a>
        = Vec<T::SelfType<'a>>
    where
        Self: 'a;
    type AsBytes<'a>
        = Vec<u8>
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Vec<T::SelfType<'a>>
    where
        Self: 'a,
    {
        let (elements, mut offset) = decode_varint_len(data);
        if offset > data.len() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(elements.min(data.len()));
        for _ in 0..elements {
            if offset >= data.len() {
                break;
            }
            let element_len = if let Some(len) = T::fixed_width() {
                len
            } else {
                let (len, consumed) = decode_varint_len(&data[offset..]);
                offset += consumed;
                len
            };
            if offset + element_len > data.len() {
                break;
            }
            result.push(T::from_bytes(&data[offset..(offset + element_len)]));
            offset += element_len;
        }
        result
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Vec<T::SelfType<'b>>) -> Vec<u8>
    where
        Self: 'b,
    {
        let mut result = if let Some(width) = T::fixed_width() {
            Vec::with_capacity(value.len() * width + 5)
        } else {
            Vec::with_capacity(value.len() * 2 + 5)
        };
        encode_varint_len(value.len(), &mut result);

        for element in value {
            let serialized = T::as_bytes(element);
            if T::fixed_width().is_none() {
                encode_varint_len(serialized.as_ref().len(), &mut result);
            }
            result.extend_from_slice(serialized.as_ref());
        }
        result
    }

    fn type_name() -> TypeName {
        TypeName::internal(&format!("Vec<{}>", T::type_name().name()))
    }
}
