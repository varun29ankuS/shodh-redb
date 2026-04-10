use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::convert::TryInto;
use core::fmt::Debug;
use core::mem::size_of;
#[cfg(feature = "chrono_v0_4")]
mod chrono_v0_4;
#[cfg(feature = "uuid")]
mod uuid;

#[derive(Eq, PartialEq, Clone, Debug)]
enum TypeClassification {
    Internal,
    UserDefined,
    // Used by variable width tuple encoding in version 3.0 and newer. This differentiates the encoding
    // from the old encoding used previously
    Internal2,
}

impl TypeClassification {
    fn to_byte(&self) -> u8 {
        match self {
            TypeClassification::Internal => 1,
            TypeClassification::UserDefined => 2,
            TypeClassification::Internal2 => 3,
        }
    }

    fn from_byte(value: u8) -> crate::Result<Self> {
        match value {
            1 => Ok(TypeClassification::Internal),
            2 => Ok(TypeClassification::UserDefined),
            3 => Ok(TypeClassification::Internal2),
            _ => Err(crate::StorageError::Corrupted(format!(
                "invalid TypeClassification byte: {value}"
            ))),
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct TypeName {
    classification: TypeClassification,
    name: String,
}

impl TypeName {
    /// It is recommended that `name` be prefixed with the crate name to minimize the chance of
    /// it coliding with another user defined type
    pub fn new(name: &str) -> Self {
        Self {
            classification: TypeClassification::UserDefined,
            name: name.to_string(),
        }
    }

    pub(crate) fn internal(name: &str) -> Self {
        Self {
            classification: TypeClassification::Internal,
            name: name.to_string(),
        }
    }

    pub(crate) fn internal2(name: &str) -> Self {
        Self {
            classification: TypeClassification::Internal2,
            name: name.to_string(),
        }
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.name.len() + 1);
        result.push(self.classification.to_byte());
        result.extend_from_slice(self.name.as_bytes());
        result
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self {
                classification: TypeClassification::UserDefined,
                name: alloc::string::String::from("<empty>"),
            };
        }
        let classification =
            TypeClassification::from_byte(bytes[0]).unwrap_or(TypeClassification::UserDefined);
        let name = core::str::from_utf8(&bytes[1..])
            .unwrap_or("<corrupted>")
            .to_string();

        Self {
            classification,
            name,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Types that implement this trait can be used as values in a redb table
pub trait Value: Debug {
    /// `SelfType<'a>` must be the same type as Self with all lifetimes replaced with 'a
    type SelfType<'a>: Debug + 'a
    where
        Self: 'a;

    type AsBytes<'a>: AsRef<[u8]> + 'a
    where
        Self: 'a;

    /// Width of a fixed type, or None for variable width
    fn fixed_width() -> Option<usize>;

    /// Deserializes data
    /// Implementations may return a view over data, or an owned type
    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a;

    /// Serialize the value to a slice
    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b;

    /// Globally unique identifier for this type
    fn type_name() -> TypeName;
}

/// Implementing this trait indicates that the type can be mutated in-place as a &mut [u8].
/// This enables the `.insert_reserve()` method on Table
pub trait MutInPlaceValue: Value {
    /// The base type such that &mut [u8] can be safely transmuted to `&mut BaseRefType`
    type BaseRefType: Debug + ?Sized;

    // Initialize `data` to a valid value. This method will be called (at some point, not necessarily immediately)
    // before from_bytes_mut() is called on a slice.
    fn initialize(data: &mut [u8]);

    fn from_bytes_mut(data: &mut [u8]) -> &mut Self::BaseRefType;
}

impl MutInPlaceValue for &[u8] {
    type BaseRefType = [u8];

    fn initialize(_data: &mut [u8]) {
        // no-op. All values are valid.
    }

    fn from_bytes_mut(data: &mut [u8]) -> &mut Self::BaseRefType {
        data
    }
}

/// Trait which allows the type to be used as a key in a redb table
pub trait Key: Value {
    /// Compare data1 with data2
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering;

    /// Transform key bytes in-place to byte-comparable form.
    ///
    /// Byte-ordered storage engines (e.g., `BfTree`) compare keys lexicographically
    /// on raw bytes. For types where `as_bytes()` output does not preserve semantic
    /// ordering under byte comparison (e.g., signed integers in little-endian
    /// encoding), this method transforms the bytes so that lexicographic byte
    /// comparison matches `Self::compare()` ordering.
    ///
    /// The default implementation is a no-op (correct for unsigned integers,
    /// strings, and other types whose byte representation already sorts correctly).
    fn to_byte_ordered_in_place(_data: &mut [u8]) {}

    /// Reverse the transformation applied by [`to_byte_ordered_in_place`].
    ///
    /// Called when reading key bytes back from a byte-ordered storage engine
    /// before passing them to `from_bytes()`.
    fn from_byte_ordered_in_place(_data: &mut [u8]) {}
}

impl Value for () {
    type SelfType<'a>
        = ()
    where
        Self: 'a;
    type AsBytes<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        Some(0)
    }

    #[allow(clippy::unused_unit, clippy::semicolon_if_nothing_returned)]
    fn from_bytes<'a>(_data: &'a [u8]) -> ()
    where
        Self: 'a,
    {
        ()
    }

    #[allow(clippy::ignored_unit_patterns)]
    fn as_bytes<'a, 'b: 'a>(_: &'a Self::SelfType<'b>) -> &'a [u8]
    where
        Self: 'b,
    {
        &[]
    }

    fn type_name() -> TypeName {
        TypeName::internal("()")
    }
}

impl Key for () {
    fn compare(_data1: &[u8], _data2: &[u8]) -> Ordering {
        Ordering::Equal
    }
}

impl Value for bool {
    type SelfType<'a>
        = bool
    where
        Self: 'a;
    type AsBytes<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        Some(1)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> bool
    where
        Self: 'a,
    {
        // Treat any non-zero as true (like C); empty slice yields false on corrupted data
        matches!(data.first(), Some(&b) if b != 0)
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> &'a [u8]
    where
        Self: 'b,
    {
        match value {
            true => &[1],
            false => &[0],
        }
    }

    fn type_name() -> TypeName {
        TypeName::internal("bool")
    }
}

impl Key for bool {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        let value1 = Self::from_bytes(data1);
        let value2 = Self::from_bytes(data2);
        value1.cmp(&value2)
    }
}

impl<T: Value> Value for Option<T> {
    type SelfType<'a>
        = Option<T::SelfType<'a>>
    where
        Self: 'a;
    type AsBytes<'a>
        = Vec<u8>
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        T::fixed_width().map(|x| x + 1)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Option<T::SelfType<'a>>
    where
        Self: 'a,
    {
        if data.is_empty() {
            return None;
        }
        match data[0] {
            0 => None,
            // Treat any non-zero discriminator as Some to avoid crashing on corrupted data
            _ => Some(T::from_bytes(&data[1..])),
        }
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Vec<u8>
    where
        Self: 'b,
    {
        let mut result = vec![0];
        if let Some(x) = value {
            result[0] = 1;
            result.extend_from_slice(T::as_bytes(x).as_ref());
        } else if let Some(fixed_width) = T::fixed_width() {
            result.resize(1 + fixed_width, 0);
        }
        result
    }

    fn type_name() -> TypeName {
        TypeName::internal(&format!("Option<{}>", T::type_name().name()))
    }
}

impl<T: Key> Key for Option<T> {
    #[allow(clippy::collapsible_else_if)]
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        let d1 = data1.first().copied().unwrap_or(0);
        let d2 = data2.first().copied().unwrap_or(0);
        if d1 == 0 {
            if d2 == 0 {
                Ordering::Equal
            } else {
                Ordering::Less
            }
        } else {
            if d2 == 0 {
                Ordering::Greater
            } else {
                T::compare(data1.get(1..).unwrap_or(&[]), data2.get(1..).unwrap_or(&[]))
            }
        }
    }
}

impl Value for &[u8] {
    type SelfType<'a>
        = &'a [u8]
    where
        Self: 'a;
    type AsBytes<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> &'a [u8]
    where
        Self: 'a,
    {
        data
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> &'a [u8]
    where
        Self: 'b,
    {
        value
    }

    fn type_name() -> TypeName {
        TypeName::internal("&[u8]")
    }
}

impl Key for &[u8] {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        data1.cmp(data2)
    }
}

impl<const N: usize> Value for &[u8; N] {
    type SelfType<'a>
        = &'a [u8; N]
    where
        Self: 'a;
    type AsBytes<'a>
        = &'a [u8; N]
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        Some(N)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> &'a [u8; N]
    where
        Self: 'a,
    {
        // first_chunk returns None if data.len() < N. The fallback uses an
        // inline const to produce a &'static [u8; N] (which outlives 'a).
        data.first_chunk::<N>().unwrap_or(const { &[0u8; N] })
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> &'a [u8; N]
    where
        Self: 'b,
    {
        value
    }

    fn type_name() -> TypeName {
        TypeName::internal(&format!("[u8;{N}]"))
    }
}

impl<const N: usize> Key for &[u8; N] {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        data1.cmp(data2)
    }
}

impl<const N: usize, T: Value> Value for [T; N] {
    type SelfType<'a>
        = [T::SelfType<'a>; N]
    where
        Self: 'a;
    type AsBytes<'a>
        = Vec<u8>
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        T::fixed_width().map(|x| x * N)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> [T::SelfType<'a>; N]
    where
        Self: 'a,
    {
        let mut result = Vec::with_capacity(N);
        if let Some(fixed) = T::fixed_width() {
            for i in 0..N {
                let slice_start = fixed * i;
                let slice_end = fixed * (i + 1);
                result.push(T::from_bytes(
                    data.get(slice_start..slice_end).unwrap_or(&[]),
                ));
            }
        } else {
            // Set offset to the first data item
            let mut start = size_of::<u32>() * N;
            for i in 0..N {
                let range = size_of::<u32>() * i..size_of::<u32>() * (i + 1);
                let end = data
                    .get(range)
                    .and_then(|s| <[u8; 4]>::try_from(s).ok())
                    .map_or(0, u32::from_le_bytes) as usize;
                // Clamp to data bounds and enforce monotonically non-decreasing offsets
                let end = end.min(data.len()).max(start);
                let slice_start = start.min(data.len());
                result.push(T::from_bytes(data.get(slice_start..end).unwrap_or(&[])));
                start = end;
            }
        }
        // The loop above pushes exactly N elements; the conversion is infallible.
        result
            .try_into()
            .unwrap_or_else(|_| unreachable!("loop pushes exactly N elements"))
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Vec<u8>
    where
        Self: 'b,
    {
        if let Some(fixed) = T::fixed_width() {
            let mut result = Vec::with_capacity(fixed * N);
            for item in value {
                result.extend_from_slice(T::as_bytes(item).as_ref());
            }
            result
        } else {
            // Reserve space for the end offsets
            let mut result = vec![0u8; size_of::<u32>() * N];
            for i in 0..N {
                result.extend_from_slice(T::as_bytes(&value[i]).as_ref());
                debug_assert!(
                    u32::try_from(result.len()).is_ok(),
                    "[T; N] as_bytes: serialized size {} exceeds u32::MAX offset limit",
                    result.len(),
                );
                // Saturate at u32::MAX rather than panic; reader will see
                // clamped offsets and produce a shorter-than-expected element.
                #[allow(clippy::cast_possible_truncation)]
                let end = u32::try_from(result.len()).unwrap_or(u32::MAX);
                result[size_of::<u32>() * i..size_of::<u32>() * (i + 1)]
                    .copy_from_slice(&end.to_le_bytes());
            }
            result
        }
    }

    fn type_name() -> TypeName {
        // Uses the same type name as [T;N] so that tables are compatible with [u8;N] and &[u8;N] types
        // This requires that the binary encoding be the same
        TypeName::internal(&format!("[{};{N}]", T::type_name().name()))
    }
}

impl<const N: usize, T: Key> Key for [T; N] {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        if let Some(fixed) = T::fixed_width() {
            for i in 0..N {
                let s = fixed * i;
                let e = fixed * (i + 1);
                let s1 = data1.get(s..e).unwrap_or(&[]);
                let s2 = data2.get(s..e).unwrap_or(&[]);
                let comparison = T::compare(s1, s2);
                if !comparison.is_eq() {
                    return comparison;
                }
            }
        } else {
            // Set offset to the first data item
            let mut start1 = size_of::<u32>() * N;
            let mut start2 = size_of::<u32>() * N;
            for i in 0..N {
                let range = size_of::<u32>() * i..size_of::<u32>() * (i + 1);
                let end1 = data1
                    .get(range.clone())
                    .and_then(|s| <[u8; 4]>::try_from(s).ok())
                    .map_or(0, u32::from_le_bytes) as usize;
                let end2 = data2
                    .get(range)
                    .and_then(|s| <[u8; 4]>::try_from(s).ok())
                    .map_or(0, u32::from_le_bytes) as usize;
                // Enforce monotonic offsets and clamp to data bounds
                let end1 = end1.min(data1.len()).max(start1);
                let end2 = end2.min(data2.len()).max(start2);
                let s1 = data1.get(start1..end1).unwrap_or(&[]);
                let s2 = data2.get(start2..end2).unwrap_or(&[]);
                let comparison = T::compare(s1, s2);
                if !comparison.is_eq() {
                    return comparison;
                }
                start1 = end1;
                start2 = end2;
            }
        }
        Ordering::Equal
    }
}

impl Value for &str {
    type SelfType<'a>
        = &'a str
    where
        Self: 'a;
    type AsBytes<'a>
        = &'a str
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> &'a str
    where
        Self: 'a,
    {
        // Graceful fallback on corrupted UTF-8: return empty string rather than crash
        core::str::from_utf8(data).unwrap_or("")
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> &'a str
    where
        Self: 'b,
    {
        value
    }

    fn type_name() -> TypeName {
        TypeName::internal("&str")
    }
}

impl Key for &str {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        // Compare raw bytes directly -- avoids UTF-8 validation overhead and
        // produces consistent ordering even on corrupted data
        data1.cmp(data2)
    }
}

impl Value for String {
    type SelfType<'a>
        = String
    where
        Self: 'a;
    type AsBytes<'a>
        = &'a str
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> String
    where
        Self: 'a,
    {
        // Lossy conversion to avoid crashing on corrupted UTF-8 data
        String::from_utf8_lossy(data).into_owned()
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> &'a str
    where
        Self: 'b,
    {
        value.as_str()
    }

    fn type_name() -> TypeName {
        TypeName::internal("String")
    }
}

impl Key for String {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        // Compare raw bytes directly -- avoids UTF-8 validation overhead and
        // produces consistent ordering even on corrupted data
        data1.cmp(data2)
    }
}

impl Value for char {
    type SelfType<'a> = char;
    type AsBytes<'a>
        = [u8; 3]
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        Some(3)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> char
    where
        Self: 'a,
    {
        if data.len() < 3 {
            return '\u{FFFD}';
        }
        char::from_u32(u32::from_le_bytes([data[0], data[1], data[2], 0])).unwrap_or('\u{FFFD}')
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> [u8; 3]
    where
        Self: 'b,
    {
        let bytes = u32::from(*value).to_le_bytes();
        [bytes[0], bytes[1], bytes[2]]
    }

    fn type_name() -> TypeName {
        TypeName::internal(stringify!(char))
    }
}

impl Key for char {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        Self::from_bytes(data1).cmp(&Self::from_bytes(data2))
    }
}

macro_rules! le_value {
    ($t:ty) => {
        impl Value for $t {
            type SelfType<'a> = $t;
            type AsBytes<'a>
                = [u8; core::mem::size_of::<$t>()]
            where
                Self: 'a;

            fn fixed_width() -> Option<usize> {
                Some(core::mem::size_of::<$t>())
            }

            fn from_bytes<'a>(data: &'a [u8]) -> $t
            where
                Self: 'a,
            {
                let bytes: [u8; core::mem::size_of::<$t>()] = match data.try_into() {
                    Ok(b) => b,
                    Err(_) => return <$t>::from_le_bytes([0u8; core::mem::size_of::<$t>()]),
                };
                <$t>::from_le_bytes(bytes)
            }

            fn as_bytes<'a, 'b: 'a>(
                value: &'a Self::SelfType<'b>,
            ) -> [u8; core::mem::size_of::<$t>()]
            where
                Self: 'a,
                Self: 'b,
            {
                value.to_le_bytes()
            }

            fn type_name() -> TypeName {
                TypeName::internal(stringify!($t))
            }
        }
    };
}

macro_rules! le_impl {
    ($t:ty) => {
        le_value!($t);

        impl Key for $t {
            fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
                Self::from_bytes(data1).cmp(&Self::from_bytes(data2))
            }
        }
    };
}

/// Implement `Key` for signed integer types with byte-order transformation.
///
/// Signed integers in two's-complement little-endian encoding do not sort
/// correctly under lexicographic byte comparison (negative values have high
/// bytes and sort after positive values). The transformation flips the sign
/// bit and reverses to big-endian, producing a byte sequence where
/// lexicographic order matches numeric order.
macro_rules! le_signed_impl {
    ($t:ty) => {
        le_value!($t);

        impl Key for $t {
            fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
                Self::from_bytes(data1).cmp(&Self::from_bytes(data2))
            }

            fn to_byte_ordered_in_place(data: &mut [u8]) {
                // Flip the sign bit (MSB in LE = last byte) then reverse to
                // big-endian. This maps i_MIN to 0x00..00, 0 to 0x80..00,
                // i_MAX to 0xFF..FF, preserving numeric order under byte comparison.
                if let Some(last) = data.last_mut() {
                    *last ^= 0x80;
                }
                data.reverse();
            }

            fn from_byte_ordered_in_place(data: &mut [u8]) {
                data.reverse();
                if let Some(last) = data.last_mut() {
                    *last ^= 0x80;
                }
            }
        }
    };
}

le_impl!(u8);
le_impl!(u16);
le_impl!(u32);
le_impl!(u64);
le_impl!(u128);
le_signed_impl!(i8);
le_signed_impl!(i16);
le_signed_impl!(i32);
le_signed_impl!(i64);
le_signed_impl!(i128);
le_value!(f32);
le_value!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, TestRunner};

    macro_rules! proptest_signed_ordering {
        ($test_name:ident, $t:ty) => {
            #[test]
            fn $test_name() {
                let config = Config {
                    cases: 10_000,
                    ..Config::default()
                };
                let mut runner = TestRunner::new(config);
                runner
                    .run(&(any::<$t>(), any::<$t>()), |(a, b)| {
                        let mut a_bytes = <$t as Value>::as_bytes(&a).as_ref().to_vec();
                        let mut b_bytes = <$t as Value>::as_bytes(&b).as_ref().to_vec();
                        <$t as Key>::to_byte_ordered_in_place(&mut a_bytes);
                        <$t as Key>::to_byte_ordered_in_place(&mut b_bytes);
                        prop_assert_eq!(
                            a_bytes.cmp(&b_bytes),
                            a.cmp(&b),
                            "byte ordering mismatch: a={}, b={}",
                            a,
                            b
                        );
                        Ok(())
                    })
                    .unwrap();
            }
        };
    }

    proptest_signed_ordering!(proptest_i8_byte_ordering, i8);
    proptest_signed_ordering!(proptest_i16_byte_ordering, i16);
    proptest_signed_ordering!(proptest_i32_byte_ordering, i32);
    proptest_signed_ordering!(proptest_i64_byte_ordering, i64);
    proptest_signed_ordering!(proptest_i128_byte_ordering, i128);

    macro_rules! proptest_signed_roundtrip {
        ($test_name:ident, $t:ty) => {
            #[test]
            fn $test_name() {
                let config = Config {
                    cases: 10_000,
                    ..Config::default()
                };
                let mut runner = TestRunner::new(config);
                runner
                    .run(&any::<$t>(), |v| {
                        let mut bytes = <$t as Value>::as_bytes(&v).as_ref().to_vec();
                        <$t as Key>::to_byte_ordered_in_place(&mut bytes);
                        <$t as Key>::from_byte_ordered_in_place(&mut bytes);
                        let recovered = <$t as Value>::from_bytes(&bytes);
                        prop_assert_eq!(recovered, v, "round-trip failed");
                        Ok(())
                    })
                    .unwrap();
            }
        };
    }

    proptest_signed_roundtrip!(proptest_i8_roundtrip, i8);
    proptest_signed_roundtrip!(proptest_i16_roundtrip, i16);
    proptest_signed_roundtrip!(proptest_i32_roundtrip, i32);
    proptest_signed_roundtrip!(proptest_i64_roundtrip, i64);
    proptest_signed_roundtrip!(proptest_i128_roundtrip, i128);
}
