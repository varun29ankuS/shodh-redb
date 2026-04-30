use crate::tree_store::page_store::base::MAX_REGIONS;
use crate::tree_store::page_store::bitmap::BtreeBitmap;
use crate::tree_store::page_store::buddy_allocator::BuddyAllocator;
use crate::tree_store::page_store::layout::DatabaseLayout;
use crate::tree_store::page_store::page_manager::{INITIAL_REGIONS, MAX_MAX_PAGE_ORDER};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::{self, max};
use core::mem::size_of;

// Tracks the page orders that MAY BE free in each region. This data structure is optimistic, so
// a region may not actually have a page free for a given order
pub(crate) struct RegionTracker {
    order_trackers: Vec<BtreeBitmap>,
}

impl RegionTracker {
    pub(crate) fn new(regions: u32, orders: u8) -> Self {
        let mut data = vec![];
        for _ in 0..orders {
            data.push(BtreeBitmap::new_padded(regions, regions, MAX_REGIONS));
        }
        Self {
            order_trackers: data,
        }
    }

    // Format:
    // num_orders: u32 number of order allocators
    // allocator_lens: u32 length of each allocator
    // data: BtreeBitmap data for each order
    pub(super) fn to_vec(&self) -> crate::Result<Vec<u8>> {
        let mut result = vec![];
        let orders: u32 = u32::try_from(self.order_trackers.len()).map_err(|_| {
            crate::StorageError::Internal("RegionTracker: order count exceeds u32 range".into())
        })?;
        let vecs: Vec<Vec<u8>> = self
            .order_trackers
            .iter()
            .map(|x| x.to_vec())
            .collect::<crate::Result<Vec<_>>>()?;
        let mut allocator_lens = Vec::with_capacity(vecs.len());
        for v in &vecs {
            allocator_lens.push(u32::try_from(v.len()).map_err(|_| {
                crate::StorageError::Internal(
                    "RegionTracker: allocator length exceeds u32 range".into(),
                )
            })?);
        }
        result.extend(orders.to_le_bytes());
        for allocator_len in allocator_lens {
            result.extend(allocator_len.to_le_bytes());
        }
        for serialized in &vecs {
            result.extend(serialized);
        }
        Ok(result)
    }

    pub(super) fn from_bytes(page: &[u8]) -> Result<Self, crate::StorageError> {
        if page.len() < size_of::<u32>() {
            return Err(crate::StorageError::Corrupted(
                "RegionTracker: buffer too small for header".into(),
            ));
        }
        let orders = u32::from_le_bytes(page[..size_of::<u32>()].try_into().map_err(|_| {
            crate::StorageError::Corrupted("RegionTracker: failed to read order count".into())
        })?);
        let mut start = size_of::<u32>();
        let mut allocator_lens = vec![];
        for _ in 0..orders {
            if start + size_of::<u32>() > page.len() {
                return Err(crate::StorageError::Corrupted(
                    "RegionTracker: truncated allocator length table".into(),
                ));
            }
            let allocator_len =
                u32::from_le_bytes(page[start..start + size_of::<u32>()].try_into().map_err(
                    |_| {
                        crate::StorageError::Corrupted(
                            "RegionTracker: failed to read allocator length".into(),
                        )
                    },
                )?) as usize;
            allocator_lens.push(allocator_len);
            start += size_of::<u32>();
        }
        let mut data = vec![];
        for allocator_len in allocator_lens {
            if start + allocator_len > page.len() {
                return Err(crate::StorageError::Corrupted(
                    "RegionTracker: allocator data extends beyond buffer".into(),
                ));
            }
            data.push(BtreeBitmap::from_bytes(
                &page[start..(start + allocator_len)],
            )?);
            start += allocator_len;
        }

        Ok(Self {
            order_trackers: data,
        })
    }

    pub(crate) fn find_free(&self, order: u8) -> Option<u32> {
        self.order_trackers
            .get(order as usize)
            .and_then(|tracker| tracker.find_first_unset())
    }

    pub(crate) fn mark_free(&mut self, order: u8, region: u32) -> crate::Result<()> {
        let order: usize = order.into();
        if order >= self.order_trackers.len() {
            return Err(crate::StorageError::Corrupted(
                "RegionTracker::mark_free: order exceeds tracker count".into(),
            ));
        }
        for i in 0..=order {
            self.order_trackers[i].clear(region)?;
        }
        Ok(())
    }

    pub(crate) fn mark_full(&mut self, order: u8, region: u32) -> crate::Result<()> {
        let order: usize = order.into();
        if order >= self.order_trackers.len() {
            return Err(crate::StorageError::Corrupted(
                "RegionTracker::mark_full: order exceeds tracker count".into(),
            ));
        }
        for i in order..self.order_trackers.len() {
            self.order_trackers[i].set(region)?;
        }
        Ok(())
    }

    fn resize(&mut self, new_capacity: u32) {
        for order in &mut self.order_trackers {
            order.resize(new_capacity, true);
        }
    }

    fn len(&self) -> u32 {
        self.order_trackers.first().map_or(0, |t| t.len())
    }
}

pub(super) struct Allocators {
    pub(super) region_tracker: RegionTracker,
    pub(super) region_allocators: Vec<BuddyAllocator>,
}

impl Allocators {
    pub(super) fn new(layout: DatabaseLayout) -> crate::Result<Self> {
        let mut region_allocators = vec![];
        let initial_regions = max(INITIAL_REGIONS, layout.num_regions());
        let mut region_tracker = RegionTracker::new(initial_regions, MAX_MAX_PAGE_ORDER + 1);
        for i in 0..layout.num_regions() {
            let region_layout = layout.region_layout(i);
            let allocator = BuddyAllocator::new(
                region_layout.num_pages(),
                layout.full_region_layout().num_pages(),
            );
            let max_order = allocator.get_max_order();
            region_tracker.mark_free(max_order, i)?;
            region_allocators.push(allocator);
        }

        Ok(Self {
            region_tracker,
            region_allocators,
        })
    }

    pub(crate) fn xxh3_hash(&self) -> u128 {
        // Ignore the region tracker because it is an optimistic cache, and so may not match
        // between repairs of the allocators
        let mut result = 0;
        for allocator in &self.region_allocators {
            result ^= allocator.xxh3_hash();
        }
        result
    }

    pub(super) fn resize_to(&mut self, new_layout: DatabaseLayout) -> crate::Result<()> {
        let shrink = match (new_layout.num_regions() as usize).cmp(&self.region_allocators.len()) {
            cmp::Ordering::Less => true,
            cmp::Ordering::Equal => {
                let allocator = self.region_allocators.last().ok_or_else(|| {
                    crate::StorageError::Corrupted(
                        "Allocators::resize_to: no region allocators present".into(),
                    )
                })?;
                let last_region = new_layout
                    .trailing_region_layout()
                    .unwrap_or_else(|| new_layout.full_region_layout());
                match last_region.num_pages().cmp(&allocator.len()) {
                    cmp::Ordering::Less => true,
                    cmp::Ordering::Equal => {
                        // No-op
                        return Ok(());
                    }
                    cmp::Ordering::Greater => false,
                }
            }
            cmp::Ordering::Greater => false,
        };

        if shrink {
            // Drop all regions that were removed
            let old_count = u32::try_from(self.region_allocators.len()).map_err(|_| {
                crate::StorageError::Internal(
                    "Allocators::resize_to: region count exceeds u32 range".into(),
                )
            })?;
            for i in new_layout.num_regions()..old_count {
                self.region_tracker.mark_full(0, i)?;
            }
            self.region_allocators
                .drain((new_layout.num_regions() as usize)..);

            // Resize the last region
            let last_region = new_layout
                .trailing_region_layout()
                .unwrap_or_else(|| new_layout.full_region_layout());
            let allocator = self.region_allocators.last_mut().ok_or_else(|| {
                crate::StorageError::Corrupted(
                    "Allocators::resize_to: no region allocators after drain".into(),
                )
            })?;
            if allocator.len() > last_region.num_pages() {
                allocator.resize(last_region.num_pages())?;
            }
        } else {
            let old_num_regions = self.region_allocators.len();
            for i in 0..new_layout.num_regions() {
                let new_region = new_layout.region_layout(i);
                if (i as usize) < old_num_regions {
                    let allocator = &mut self.region_allocators[i as usize];
                    if new_region.num_pages() < allocator.len() {
                        return Err(crate::StorageError::Corrupted(
                            "Allocators::resize_to: new region smaller than existing allocator"
                                .into(),
                        ));
                    }
                    if new_region.num_pages() != allocator.len() {
                        allocator.resize(new_region.num_pages())?;
                        let highest_free = allocator.highest_free_order().ok_or_else(|| {
                            crate::StorageError::Corrupted(
                                "Allocators::resize_to: no free order after resize".into(),
                            )
                        })?;
                        self.region_tracker.mark_free(highest_free, i)?;
                    }
                } else {
                    // brand new region
                    let allocator = BuddyAllocator::new(
                        new_region.num_pages(),
                        new_layout.full_region_layout().num_pages(),
                    );
                    let highest_free = allocator.highest_free_order().ok_or_else(|| {
                        crate::StorageError::Corrupted(
                            "Allocators::resize_to: new region has no free pages".into(),
                        )
                    })?;
                    if i >= self.region_tracker.len() {
                        self.region_tracker.resize(i + 1);
                    }
                    self.region_tracker.mark_free(highest_free, i)?;
                    self.region_allocators.push(allocator);
                }
            }
        }
        Ok(())
    }
}
