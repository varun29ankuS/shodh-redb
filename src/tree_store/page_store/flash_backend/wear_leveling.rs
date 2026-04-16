use alloc::vec;
use alloc::vec::Vec;

/// Per-block erase counter table, persisted as part of FTL metadata.
///
/// Each physical block has a 32-bit erase count. The table is serialized as
/// a packed little-endian `u32` array and stored in the FTL journal.
pub(super) struct EraseCountTable {
    /// Erase count for each physical block. Index = physical block number.
    counts: Vec<u32>,
}

impl EraseCountTable {
    /// Create a new table with zero counts for all blocks.
    pub fn new(total_blocks: u32) -> Self {
        Self {
            counts: vec![0u32; total_blocks as usize],
        }
    }

    /// Deserialize from packed little-endian `u32` bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        let count = data.len() / 4;
        let mut counts = Vec::with_capacity(count);
        for i in 0..count {
            let offset = i * 4;
            if offset + 4 <= data.len() {
                let val = u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                counts.push(val);
            }
        }
        Self { counts }
    }

    /// Serialize to packed little-endian `u32` bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.counts.len() * 4);
        for &c in &self.counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        out
    }

    /// Record one erase of the given physical block.
    #[inline]
    pub fn increment(&mut self, physical_block: u32) {
        if let Some(c) = self.counts.get_mut(physical_block as usize) {
            *c = c.saturating_add(1);
        }
    }

    /// Get erase count for a physical block.
    #[inline]
    pub fn get(&self, physical_block: u32) -> u32 {
        self.counts
            .get(physical_block as usize)
            .copied()
            .unwrap_or(u32::MAX)
    }

    /// Find the block with the lowest erase count among the given candidate blocks.
    ///
    /// Returns `None` if `candidates` is empty.
    pub fn pick_lowest(&self, candidates: &[u32]) -> Option<u32> {
        candidates.iter().copied().min_by_key(|&b| self.get(b))
    }

    /// Check if a static wear-leveling swap is warranted.
    ///
    /// Returns `Some((hot_physical, cold_physical))` if the erase count delta
    /// between the most-worn and least-worn blocks exceeds `threshold`.
    ///
    /// `in_use_blocks` are physical block indices currently mapped to logical blocks.
    pub fn check_static_swap(
        &self,
        threshold: u32,
        in_use_blocks: &[u32],
        free_blocks: &[u32],
    ) -> Option<(u32, u32)> {
        if in_use_blocks.is_empty() || free_blocks.is_empty() {
            return None;
        }

        // Find the most-worn in-use block
        let hot = in_use_blocks.iter().copied().max_by_key(|&b| self.get(b))?;

        // Find the least-worn free block
        let cold = free_blocks.iter().copied().min_by_key(|&b| self.get(b))?;

        let hot_count = self.get(hot);
        let cold_count = self.get(cold);

        if hot_count.saturating_sub(cold_count) >= threshold {
            Some((hot, cold))
        } else {
            None
        }
    }

    /// Total number of blocks tracked.
    #[allow(dead_code, clippy::cast_possible_truncation)]
    pub fn len(&self) -> u32 {
        self.counts.len() as u32
    }

    /// Compute aggregate wear statistics across all tracked blocks.
    pub fn stats(&self) -> (u32, u32, u64) {
        let mut min_count = u32::MAX;
        let mut max_count = 0u32;
        let mut total = 0u64;
        for &c in &self.counts {
            min_count = min_count.min(c);
            max_count = max_count.max(c);
            total += u64::from(c);
        }
        if self.counts.is_empty() {
            min_count = 0;
        }
        (min_count, max_count, total)
    }
}
