use crate::tree_store::page_store::xxh3_checksum;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;

const HEIGHT_OFFSET: usize = 0;
const END_OFFSETS: usize = HEIGHT_OFFSET + size_of::<u32>();

pub(crate) struct BtreeBitmap {
    heights: Vec<U64GroupedBitmap>,
}

// Stores a 64-way bit-tree of allocated ids.
//
// Data structure format:
// height: u32
// layer_ends: array of u32, ending offset in bytes of layers.
// layer data: u64s
// ...consecutive layers. Except for the last level, all sub-trees of the root must be complete
impl BtreeBitmap {
    pub(crate) fn count_unset(&self) -> u32 {
        self.get_level(self.get_height() - 1).count_unset()
    }

    pub(crate) fn has_unset(&self) -> bool {
        self.get_level(self.get_height() - 1).any_unset()
    }

    pub(crate) fn get(&self, i: u32) -> Result<bool, crate::StorageError> {
        self.get_level(self.get_height() - 1).get(i)
    }

    pub(crate) fn len(&self) -> u32 {
        self.get_level(self.get_height() - 1).len()
    }

    pub(crate) fn find_first_unset(&self) -> Option<u32> {
        if let Some(mut entry) = self.get_level(0).first_unset(0, 64) {
            let mut height = 0;

            while height < self.get_height() - 1 {
                height += 1;
                entry *= 64;
                // The tree invariant says a parent bit is unset only when some
                // child below it is unset, so this "cannot" fail -- but the
                // levels are rebuilt from disk, and corrupt data breaks it.
                // Reporting "nothing free" makes the allocator fail cleanly
                // rather than panicking here.
                entry = self.get_level(height).first_unset(entry, entry + 64)?;
            }

            Some(entry)
        } else {
            None
        }
    }

    fn get_level(&self, i: u32) -> &U64GroupedBitmap {
        assert!(i < self.get_height());
        &self.heights[i as usize]
    }

    #[allow(clippy::cast_possible_truncation)]
    fn get_height(&self) -> u32 {
        // Height is bounded by page order (max ~20), so this always fits in u32.
        self.heights.len() as u32
    }

    pub(crate) fn xxh3_hash(&self) -> u128 {
        let mut result = 0;
        for height in &self.heights {
            result ^= height.xxh3_hash();
        }
        result
    }

    pub(crate) fn to_vec(&self) -> crate::Result<Vec<u8>> {
        let mut result = vec![];
        #[allow(clippy::cast_possible_truncation)]
        let height: u32 = self.heights.len() as u32;
        result.extend(height.to_le_bytes());

        let vecs: Vec<Vec<u8>> = self.heights.iter().map(|x| x.to_vec()).collect();
        let mut data_offset = END_OFFSETS + self.heights.len() * size_of::<u32>();
        let end_metadata = data_offset;
        for serialized in &vecs {
            data_offset += serialized.len();
            let offset_u32: u32 = u32::try_from(data_offset).map_err(|_| {
                crate::StorageError::Internal(
                    "BtreeBitmap serialized data exceeds u32 range".into(),
                )
            })?;
            result.extend(offset_u32.to_le_bytes());
        }

        assert_eq!(end_metadata, result.len());
        for serialized in &vecs {
            result.extend(serialized);
        }

        Ok(result)
    }

    pub(crate) fn from_bytes(data: &[u8]) -> Result<Self, crate::StorageError> {
        if data.len() < HEIGHT_OFFSET + size_of::<u32>() {
            return Err(crate::StorageError::Corrupted(
                "BtreeBitmap: buffer too small for header".into(),
            ));
        }
        let height = u32::from_le_bytes(
            data[HEIGHT_OFFSET..(HEIGHT_OFFSET + size_of::<u32>())]
                .try_into()
                .map_err(|_| {
                    crate::StorageError::Corrupted("BtreeBitmap: failed to read height".into())
                })?,
        );
        // `new()` always pushes at least the leaf level, so a bitmap written by
        // this crate has height >= 1. A crafted image can still claim 0, and
        // every accessor -- count_unset, has_unset, get, len, clear -- indexes
        // `get_level(self.get_height() - 1)`. With no levels that subtraction
        // wraps to u32::MAX and trips the bounds assert in get_level. Reject it
        // here: an empty bitmap has no meaningful leaf level to read.
        if height == 0 {
            return Err(crate::StorageError::Corrupted(
                "BtreeBitmap: height of 0 has no leaf level".into(),
            ));
        }

        let mut metadata = END_OFFSETS;
        // `height` comes off disk, and `usize` is 32 bits on wasm32/WASI and
        // Cortex-M. `height * 4` therefore overflows there for a large value,
        // where on x86-64 it cannot -- so this panicked only under WASI:
        //   bitmap.rs:130: attempt to multiply with overflow
        // Reject rather than wrap: a height whose offset table cannot even be
        // addressed is corrupt by definition.
        let Some(data_start) = (height as usize)
            .checked_mul(size_of::<u32>())
            .and_then(|table| END_OFFSETS.checked_add(table))
        else {
            return Err(crate::StorageError::Corrupted(alloc::format!(
                "BtreeBitmap: height {height} overflows the offset table size"
            )));
        };
        let mut data_start = data_start;

        let mut heights = vec![];
        for _ in 0..height {
            if metadata + size_of::<u32>() > data.len() {
                return Err(crate::StorageError::Corrupted(
                    "BtreeBitmap: truncated offset table".into(),
                ));
            }
            let data_end = u32::from_le_bytes(
                data[metadata..(metadata + size_of::<u32>())]
                    .try_into()
                    .map_err(|_| {
                        crate::StorageError::Corrupted("BtreeBitmap: failed to read offset".into())
                    })?,
            ) as usize;
            if data_end > data.len() || data_start > data_end {
                return Err(crate::StorageError::Corrupted(
                    "BtreeBitmap: offset out of bounds".into(),
                ));
            }
            heights.push(U64GroupedBitmap::from_bytes(&data[data_start..data_end])?);
            data_start = data_end;
            metadata += size_of::<u32>();
        }

        // The levels form a 64-way tree: each level is the parent summary of
        // the one below it, so `parent.len() == ceil(child.len() / 64)`, and the
        // root fits in a single 64-bit group. `new()` and `resize()` both
        // maintain that; a crafted image need not.
        //
        // It matters because `resize()` asserts `get_level(0).len() <= 64` in
        // release builds, and `find_first_unset` walks parent to child assuming
        // each entry addresses 64 children. A level count inconsistent with the
        // leaf capacity turns the first file growth into a panic, and a tree
        // walk into a lookup of entries that were never summarised.
        for i in (1..heights.len()).rev() {
            let child = heights[i].len();
            let expected_parent = child.div_ceil(64);
            let actual_parent = heights[i - 1].len();
            if actual_parent != expected_parent {
                return Err(crate::StorageError::Corrupted(alloc::format!(
                    "BtreeBitmap: level {} has {actual_parent} entries, expected {expected_parent}                      to summarise {child} entries at level {i}",
                    i - 1
                )));
            }
        }
        if let Some(root) = heights.first()
            && root.len() > 64
        {
            return Err(crate::StorageError::Corrupted(alloc::format!(
                "BtreeBitmap: root level has {} entries, which exceeds one 64-bit group",
                root.len()
            )));
        }

        Ok(Self { heights })
    }

    // Initializes a new allocator, with no ids free
    pub(crate) fn new(mut num_pages: u32, mut capacity: u32) -> Self {
        let mut heights = vec![];

        // Build from the leaf to root
        loop {
            heights.push(U64GroupedBitmap::new_full(num_pages, capacity));
            if capacity <= 64 {
                break;
            }
            capacity = capacity.div_ceil(64);
            num_pages = num_pages.div_ceil(64);
        }

        // Reverse so that the root is at index 0
        heights.reverse();

        Self { heights }
    }

    // Like new(), but pads the tree height for max_capacity so resize()
    // never needs to insert new levels.
    pub(crate) fn new_padded(num_pages: u32, capacity: u32, max_capacity: u32) -> Self {
        let mut result = Self::new(num_pages, capacity);

        let max_height = Self::height_for_capacity(max_capacity);
        while result.heights.len() < max_height {
            let root_len = result.heights[0].len();
            let parent_len = root_len.div_ceil(64);
            result
                .heights
                .insert(0, U64GroupedBitmap::new_full(parent_len, parent_len));
        }

        result
    }

    fn height_for_capacity(mut capacity: u32) -> usize {
        let mut height = 1;
        while capacity > 64 {
            capacity = capacity.div_ceil(64);
            height += 1;
        }
        height
    }

    pub(crate) fn resize(&mut self, mut new_len: u32, full: bool) {
        for height in self.heights.iter_mut().rev() {
            height.resize(new_len, full);
            new_len = new_len.div_ceil(64);
        }
        assert!(self.get_level(0).len() <= 64);
    }

    // Returns the first unset id, and sets it
    pub(crate) fn alloc(&mut self) -> Result<Option<u32>, crate::StorageError> {
        let Some(entry) = self.find_first_unset() else {
            return Ok(None);
        };
        self.set(entry)?;
        Ok(Some(entry))
    }

    pub(crate) fn set(&mut self, i: u32) -> Result<bool, crate::StorageError> {
        let full = self.get_level_mut(self.get_height() - 1).set(i)?;
        self.update_to_root(i, full)?;
        Ok(full)
    }

    pub(crate) fn clear(&mut self, i: u32) -> Result<(), crate::StorageError> {
        self.get_level_mut(self.get_height() - 1).clear(i)?;
        self.update_to_root(i, false)
    }

    fn get_level_mut(&mut self, i: u32) -> &mut U64GroupedBitmap {
        assert!(i < self.get_height());
        &mut self.heights[i as usize]
    }

    // Recursively update to the root, starting at the given entry in the given height
    // full parameter must be set if all bits in the entry's group of u64 are full
    fn update_to_root(&mut self, i: u32, mut full: bool) -> Result<(), crate::StorageError> {
        if self.get_height() == 1 {
            return Ok(());
        }

        let mut parent_height = self.get_height() - 2;
        let mut parent_entry = i / 64;
        loop {
            full = if full {
                self.get_level_mut(parent_height).set(parent_entry)?
            } else {
                self.get_level_mut(parent_height).clear(parent_entry)?;
                false
            };

            if parent_height == 0 {
                break;
            }
            parent_height -= 1;
            parent_entry /= 64;
        }
        Ok(())
    }
}

// A bitmap which groups consecutive groups of 64bits together
pub(crate) struct U64GroupedBitmap {
    len: u32,
    data: Vec<u64>,
}

impl U64GroupedBitmap {
    fn required_words(elements: u32) -> usize {
        let words = elements.div_ceil(64);
        words as usize
    }

    pub fn new_full(len: u32, capacity: u32) -> Self {
        let data = vec![u64::MAX; Self::required_words(capacity)];
        Self { len, data }
    }

    pub fn xxh3_hash(&self) -> u128 {
        if self.len == 0 {
            return 0;
        }
        let mut bytes = vec![];
        bytes.extend(self.len.to_le_bytes());
        // Hash all the whole words
        for x in &self.data[0..Self::required_words(self.len) - 1] {
            bytes.extend(x.to_le_bytes());
        }
        let (index, bit) = Self::data_index_of(self.len - 1);
        // Select the bit and all lower ones
        let mask = ((1 << bit) - 1) | (1 << bit);
        let group = self.data[index];
        let group = group & mask;
        bytes.extend(group.to_le_bytes());

        xxh3_checksum(&bytes)
    }

    // Format:
    // 4 bytes: number of elements
    // n bytes: serialized groups
    pub fn to_vec(&self) -> Vec<u8> {
        let mut result = vec![];
        result.extend(self.len.to_le_bytes());
        for x in &self.data[..Self::required_words(self.len)] {
            result.extend(x.to_le_bytes());
        }
        result
    }

    pub fn from_bytes(serialized: &[u8]) -> Result<Self, crate::StorageError> {
        if serialized.len() < size_of::<u32>() {
            return Err(crate::StorageError::Corrupted(
                "U64GroupedBitmap: buffer too small for header".into(),
            ));
        }
        if (serialized.len() - size_of::<u32>()) % size_of::<u64>() != 0 {
            return Err(crate::StorageError::Corrupted(
                "U64GroupedBitmap: buffer size not aligned to u64".into(),
            ));
        }
        let len = u32::from_le_bytes(serialized[..size_of::<u32>()].try_into().map_err(|_| {
            crate::StorageError::Corrupted("U64GroupedBitmap: failed to read length".into())
        })?);
        let words = (serialized.len() - size_of::<u32>()) / size_of::<u64>();
        // `len` comes off disk, the word count comes from the buffer size.
        // `set`/`clear` bounds-check the bit against `len` and then index
        // `data`, so a `len` that outruns the words present turns that guard
        // into an out-of-bounds index. `to_vec` always writes
        // `required_words(len)`, so this only fails on corrupted input.
        if words < Self::required_words(len) {
            return Err(crate::StorageError::Corrupted(
                "U64GroupedBitmap: length exceeds the serialized data".into(),
            ));
        }
        let mut data = Vec::with_capacity(words);
        for i in 0..words {
            let start = size_of::<u32>() + i * size_of::<u64>();
            let value = u64::from_le_bytes(
                serialized[start..(start + size_of::<u64>())]
                    .try_into()
                    .map_err(|_| {
                        crate::StorageError::Corrupted(
                            "U64GroupedBitmap: failed to read word".into(),
                        )
                    })?,
            );
            data.push(value);
        }

        Ok(Self { len, data })
    }

    fn data_index_of(bit: u32) -> (usize, usize) {
        ((bit as usize) / 64, (bit as usize) % 64)
    }

    fn select_mask(bit: usize) -> u64 {
        1u64 << (bit as u64)
    }

    fn count_unset(&self) -> u32 {
        self.data.iter().map(|x| x.count_zeros()).sum()
    }

    fn any_unset(&self) -> bool {
        self.data.iter().any(|x| x.count_zeros() > 0)
    }

    fn first_unset(&self, start_bit: u32, end_bit: u32) -> Option<u32> {
        assert_eq!(end_bit, (start_bit - start_bit % 64) + 64);
        // Unlike get/set/clear, `start_bit` comes from the caller rather than
        // being checked against `len` first, and the buddy allocator derives it
        // from on-disk state. A bit past the end simply has nothing unset in
        // it, so report that instead of indexing out of the word vector.
        if self.len == 0 || start_bit >= self.len {
            return None;
        }

        let (index, bit) = Self::data_index_of(start_bit);
        let mask = (1 << bit) - 1;
        let group = *self.data.get(index)?;
        let group = group | mask;
        match group.trailing_ones() {
            64 => None,
            x => Some(start_bit + x - u32::try_from(bit).unwrap()),
        }
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn resize(&mut self, new_len: u32, full: bool) {
        if self.data.len() < Self::required_words(new_len) {
            let default_value = if full { u64::MAX } else { 0 };
            self.data
                .resize(Self::required_words(new_len), default_value);
        }
        let old_len = self.len;
        self.len = new_len;
        if old_len < new_len {
            // Handle the partial boundary word at old_len (if not word-aligned).
            // All fully new words beyond this are already correct from Vec::resize.
            let old_bit = u64::from(old_len % 64);
            if old_bit != 0 {
                let word_idx = (old_len / 64) as usize;
                if full {
                    // Set bits [old_bit..64) in the boundary word
                    self.data[word_idx] |= !((1u64 << old_bit) - 1);
                } else {
                    // Clear bits [old_bit..64) in the boundary word
                    self.data[word_idx] &= (1u64 << old_bit) - 1;
                }
            }
        }
    }

    pub fn get(&self, bit: u32) -> Result<bool, crate::StorageError> {
        if bit >= self.len {
            return Err(crate::StorageError::Corrupted(
                "bitmap: bit index out of bounds".into(),
            ));
        }
        let (index, bit_index) = Self::data_index_of(bit);
        let group = self.data[index];
        Ok(group & U64GroupedBitmap::select_mask(bit_index) != 0)
    }

    // Returns true iff the bit's group is all set
    pub fn set(&mut self, bit: u32) -> Result<bool, crate::StorageError> {
        if bit >= self.len {
            return Err(crate::StorageError::Corrupted(
                "bitmap: bit index out of bounds".into(),
            ));
        }
        let (index, bit_index) = Self::data_index_of(bit);
        let mut group = self.data[index];
        group |= Self::select_mask(bit_index);
        self.data[index] = group;

        Ok(group == u64::MAX)
    }

    pub fn clear(&mut self, bit: u32) -> Result<(), crate::StorageError> {
        if bit >= self.len {
            return Err(crate::StorageError::Corrupted(
                "bitmap: bit index out of bounds".into(),
            ));
        }
        let (index, bit_index) = Self::data_index_of(bit);
        self.data[index] &= !Self::select_mask(bit_index);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::compat::HashSet;
    use crate::tree_store::page_store::bitmap::BtreeBitmap;
    use rand::prelude::IteratorRandom;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Build a serialized `BtreeBitmap` from per-level lengths, root first.
    /// Mirrors `to_vec`: height, then one absolute end offset per level, then
    /// each level as `[len: u32][words: u64...]`.
    fn build_levels(lens: &[u32]) -> Vec<u8> {
        let height = u32::try_from(lens.len()).unwrap();
        let mut bodies: Vec<Vec<u8>> = Vec::new();
        for &len in lens {
            let words = (len as usize).div_ceil(64).max(1);
            let mut body = len.to_le_bytes().to_vec();
            body.extend(core::iter::repeat_n(0u8, words * 8));
            bodies.push(body);
        }
        let mut out = height.to_le_bytes().to_vec();
        let mut offset = 4 + 4 * lens.len();
        let mut ends = Vec::new();
        for body in &bodies {
            offset += body.len();
            ends.push(u32::try_from(offset).unwrap());
        }
        for end in ends {
            out.extend(end.to_le_bytes());
        }
        for body in bodies {
            out.extend(body);
        }
        out
    }

    /// The levels are a 64-way tree: each is the parent summary of the one
    /// below, so `parent.len() == ceil(child.len() / 64)` and the root fits one
    /// 64-bit group. `new()` and `resize()` maintain that; a crafted image need
    /// not, and nothing checked.
    ///
    /// It is not cosmetic. `resize()` carries a release-active
    /// `assert!(get_level(0).len() <= 64)`, so an inconsistent level count turns
    /// the next file growth into a panic, and `find_first_unset` walks parent to
    /// child assuming each entry summarises exactly 64 children.
    #[test]
    fn inconsistent_level_sizes_are_rejected() {
        // Consistent: 100 leaf entries need ceil(100/64) = 2 parent entries.
        assert!(BtreeBitmap::from_bytes(&build_levels(&[2, 100])).is_ok());

        // Parent too small to summarise its children.
        assert!(BtreeBitmap::from_bytes(&build_levels(&[1, 100])).is_err());

        // Parent larger than the level below warrants.
        assert!(BtreeBitmap::from_bytes(&build_levels(&[9, 100])).is_err());

        // Single level whose root cannot fit in one 64-bit group. This is the
        // exact shape that trips the assert inside `resize`.
        assert!(BtreeBitmap::from_bytes(&build_levels(&[100])).is_err());

        // A root of exactly 64 is the largest legal one.
        assert!(BtreeBitmap::from_bytes(&build_levels(&[64])).is_ok());

        // Three levels, consistent throughout: 8192 -> 128 -> 2.
        assert!(BtreeBitmap::from_bytes(&build_levels(&[2, 128, 8192])).is_ok());
        // ...and the same shape with one level perturbed.
        assert!(BtreeBitmap::from_bytes(&build_levels(&[2, 127, 8192])).is_err());
    }

    /// Height is read from disk. `new()` always builds at least the leaf
    /// level, so 0 never occurs in a bitmap this crate wrote -- but a crafted
    /// image can claim it, and then every accessor computes
    /// `get_level(self.get_height() - 1)`, which wraps to `u32::MAX`:
    ///
    /// ```text
    /// panicked at bitmap.rs:59: assertion failed: i < self.get_height()
    /// ```
    ///
    /// Found by the `fuzz_db_image` target (input
    /// `[213, 105, 105, 105, 105, 224, 0, 234, 224]`).
    #[test]
    fn zero_height_is_rejected() {
        // A well-formed header whose height is 0, with an offset table and
        // payload that are otherwise entirely plausible.
        let mut data = 0u32.to_le_bytes().to_vec();
        data.extend_from_slice(&[0u8; 32]);
        assert!(
            BtreeBitmap::from_bytes(&data).is_err(),
            "a bitmap with no levels must be rejected at the parse boundary"
        );

        // A real bitmap still round-trips, so the guard is not over-broad.
        let bitmap = BtreeBitmap::new(64, 64);
        let bytes = bitmap.to_vec().unwrap();
        let parsed = BtreeBitmap::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.len(), bitmap.len());
    }

    /// `len` is read from disk but the word count comes from the buffer size.
    /// A `len` larger than the words present must be rejected: `set`/`clear`
    /// bounds-check against `len` and then index `data`, so an inconsistent
    /// pair indexes out of bounds.
    #[test]
    fn grouped_bitmap_rejects_len_exceeding_data() {
        use crate::tree_store::page_store::bitmap::U64GroupedBitmap;

        // len = 1000 (needs 16 u64 words) but only one word follows.
        let mut serialized = Vec::new();
        serialized.extend(1000u32.to_le_bytes());
        serialized.extend(0u64.to_le_bytes());

        assert!(
            U64GroupedBitmap::from_bytes(&serialized).is_err(),
            "a length larger than the serialized data must be rejected"
        );
    }

    /// `first_unset` takes `start_bit` from its caller rather than checking it
    /// against `len` the way get/set/clear do, and the buddy allocator derives
    /// that value from on-disk state. A bit past the end used to index out of
    /// the word vector: "index out of bounds: the len is 5 but the index is 56".
    /// Found by the `fuzz_db_image` target.
    #[test]
    fn first_unset_past_the_end_returns_none() {
        use crate::tree_store::page_store::bitmap::U64GroupedBitmap;

        // 100 bits -> 2 words of storage.
        let bitmap = U64GroupedBitmap::new_full(100, 100);
        // Well past the end, and word-aligned so the internal assert holds.
        let start = 3584u32;
        assert_eq!(bitmap.first_unset(start, start + 64), None);
    }

    #[test]
    fn alloc() {
        let num_pages = 2;
        let mut allocator = BtreeBitmap::new(num_pages, num_pages);
        for i in 0..num_pages {
            allocator.clear(i).unwrap();
        }
        for i in 0..num_pages {
            assert_eq!(i, allocator.alloc().unwrap().unwrap());
        }
        assert!(allocator.alloc().unwrap().is_none());
    }

    #[test]
    fn record_alloc() {
        let mut allocator = BtreeBitmap::new(2, 2);
        allocator.clear(0).unwrap();
        allocator.clear(1).unwrap();
        allocator.set(0).unwrap();
        assert_eq!(1, allocator.alloc().unwrap().unwrap());
        assert!(allocator.alloc().unwrap().is_none());
    }

    #[test]
    fn free() {
        let mut allocator = BtreeBitmap::new(1, 1);
        allocator.clear(0).unwrap();
        assert_eq!(0, allocator.alloc().unwrap().unwrap());
        assert!(allocator.alloc().unwrap().is_none());
        allocator.clear(0).unwrap();
        assert_eq!(0, allocator.alloc().unwrap().unwrap());
    }

    #[test]
    fn reuse_lowest() {
        let num_pages = 65;
        let mut allocator = BtreeBitmap::new(num_pages, num_pages);
        for i in 0..num_pages {
            allocator.clear(i).unwrap();
        }
        for i in 0..num_pages {
            assert_eq!(i, allocator.alloc().unwrap().unwrap());
        }
        allocator.clear(5).unwrap();
        allocator.clear(15).unwrap();
        assert_eq!(5, allocator.alloc().unwrap().unwrap());
        assert_eq!(15, allocator.alloc().unwrap().unwrap());
        assert!(allocator.alloc().unwrap().is_none());
    }

    #[test]
    fn all_space_used() {
        let num_pages = 65;
        let mut allocator = BtreeBitmap::new(num_pages, num_pages);
        for i in 0..num_pages {
            allocator.clear(i).unwrap();
        }
        // Allocate everything
        while allocator.alloc().unwrap().is_some() {}
        // The last u64 must be used, since the leaf layer is compact
        assert_eq!(
            u64::MAX,
            *allocator.heights.last().unwrap().data.last().unwrap()
        );
    }

    #[test]
    fn find_free() {
        let num_pages = 129;
        let mut allocator = BtreeBitmap::new(num_pages, num_pages);
        assert!(allocator.find_first_unset().is_none());
        allocator.clear(128).unwrap();
        assert_eq!(allocator.find_first_unset().unwrap(), 128);
        allocator.clear(65).unwrap();
        assert_eq!(allocator.find_first_unset().unwrap(), 65);
        allocator.clear(8).unwrap();
        assert_eq!(allocator.find_first_unset().unwrap(), 8);
        allocator.clear(0).unwrap();
        assert_eq!(allocator.find_first_unset().unwrap(), 0);
    }

    #[test]
    fn resize_beyond_initial_capacity() {
        let mut bm = BtreeBitmap::new_padded(60, 60, 1_048_576);
        let height = bm.heights.len();

        bm.clear(0).unwrap();
        bm.clear(30).unwrap();
        bm.clear(59).unwrap();

        bm.resize(5000, true);

        // Height is pre-padded at construction, stays the same
        assert_eq!(bm.heights.len(), height);

        // Previously cleared bits survive the resize
        assert!(!bm.get(0).unwrap());
        assert!(!bm.get(30).unwrap());
        assert!(!bm.get(59).unwrap());

        // Newly grown region should be marked full (true)
        assert!(bm.get(60).unwrap());
        assert!(bm.get(4999).unwrap());

        // Alloc should return the previously cleared entries
        assert_eq!(bm.alloc().unwrap(), Some(0));
        assert_eq!(bm.alloc().unwrap(), Some(30));
        assert_eq!(bm.alloc().unwrap(), Some(59));
        assert!(bm.alloc().unwrap().is_none());

        // Clear a bit in the newly grown area, alloc should find it
        bm.clear(3000).unwrap();
        assert_eq!(bm.alloc().unwrap(), Some(3000));
        assert!(bm.alloc().unwrap().is_none());
    }

    #[test]
    fn random_pattern() {
        let seed = rand::rng().random();
        // Print the seed to debug for reproducibility, in case this test fails
        println!("seed={seed}");
        let mut rng = StdRng::seed_from_u64(seed);

        let num_pages = rng.random_range(2..10000);
        let mut allocator = BtreeBitmap::new(num_pages, num_pages);
        for i in 0..num_pages {
            allocator.clear(i).unwrap();
        }
        let mut allocated = HashSet::new();

        for _ in 0..(num_pages * 2) {
            if rng.random_bool(0.75) {
                if let Some(page) = allocator.alloc().unwrap() {
                    allocated.insert(page);
                } else {
                    assert_eq!(allocated.len(), num_pages as usize);
                }
            } else if let Some(to_free) = allocated.iter().choose(&mut rng).copied() {
                allocator.clear(to_free).unwrap();
                allocated.remove(&to_free);
            }
        }

        for _ in allocated.len()..(num_pages as usize) {
            allocator.alloc().unwrap().unwrap();
        }
        assert!(allocator.alloc().unwrap().is_none());

        for i in 0..num_pages {
            allocator.clear(i).unwrap();
        }

        for _ in 0..num_pages {
            allocator.alloc().unwrap().unwrap();
        }
        assert!(allocator.alloc().unwrap().is_none());
    }
}
