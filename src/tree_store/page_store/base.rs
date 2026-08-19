use crate::compat::Arc;
use crate::compat::HashSet;
#[cfg(debug_assertions)]
use crate::compat::{HashMap, Mutex};
use crate::error::StorageError;
use crate::tree_store::page_store::cached_file::WritablePage;
use crate::tree_store::page_store::page_manager::MAX_MAX_PAGE_ORDER;
use alloc::format;
#[cfg(test)]
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt::{Debug, Formatter};
use core::hash::{Hash, Hasher};
use core::mem;
use core::ops::Range;

pub(crate) const MAX_VALUE_LENGTH: usize = 3 * 1024 * 1024 * 1024;
pub(crate) const MAX_PAIR_LENGTH: usize = 3 * 1024 * 1024 * 1024 + 768 * 1024 * 1024;
pub(crate) const MAX_PAGE_INDEX: u32 = 0x000F_FFFF;
pub(crate) const MAX_REGIONS: u32 = 0x0010_0000;

// On-disk format is:
// Performance note: storing the count of order-0 pages per region could
// avoid scanning the buddy bitmap. Deferred until profiling shows region
// scanning is a bottleneck.
// lowest 20bits: page index within the region. Only the lowest `20 - order_exponent` bits may be read.
// The remaining bits are reserved for future use and must be ignored
// second 20bits: region number
// 19bits: reserved
// highest 5bits: page order exponent
//
// Assuming a reasonable page size, like 4kiB, this allows for 4kiB * 2^20 * 2^20 = 4PiB of usable space
#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) struct PageNumber {
    pub(crate) region: u32,
    pub(crate) page_index: u32,
    pub(crate) page_order: u8,
}

impl Hash for PageNumber {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Layout: packing region/page_index/page_order into a single u64 would
        // reduce field access overhead but hurt readability. Current layout preferred.
        let mut temp = 0x000F_FFFF & u64::from(self.page_index);
        temp |= (0x000F_FFFF & u64::from(self.region)) << 20;
        temp |= (0b0001_1111 & u64::from(self.page_order)) << 59;
        state.write_u64(temp);
    }
}

// PageNumbers are ordered as determined by their starting address in the database file
impl Ord for PageNumber {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.region.cmp(&other.region) {
            Ordering::Less => Ordering::Less,
            Ordering::Equal => {
                let self_order0 =
                    u64::from(self.page_index).saturating_mul(1u64 << self.page_order);
                let other_order0 =
                    u64::from(other.page_index).saturating_mul(1u64 << other.page_order);
                debug_assert!(
                    self_order0 != other_order0 || self.page_order == other.page_order,
                    "{self:?} overlaps {other:?}, but is not equal"
                );
                self_order0.cmp(&other_order0)
            }
            Ordering::Greater => Ordering::Greater,
        }
    }
}

impl PartialOrd for PageNumber {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PageNumber {
    pub(crate) const fn serialized_size() -> usize {
        8
    }

    pub(crate) fn new(region: u32, page_index: u32, page_order: u8) -> Self {
        debug_assert!(region <= 0x000F_FFFF);
        debug_assert!(page_index <= MAX_PAGE_INDEX);
        debug_assert!(page_order <= MAX_MAX_PAGE_ORDER);
        Self {
            region,
            page_index,
            page_order,
        }
    }

    pub(crate) fn to_le_bytes(self) -> [u8; 8] {
        let mut temp = 0x000F_FFFF & u64::from(self.page_index);
        temp |= (0x000F_FFFF & u64::from(self.region)) << 20;
        temp |= (0b0001_1111 & u64::from(self.page_order)) << 59;
        temp.to_le_bytes()
    }

    pub(crate) fn from_le_bytes(bytes: [u8; 8]) -> Self {
        let temp = u64::from_le_bytes(bytes);
        let order = (temp >> 59) as u8;
        // Clamp order to the maximum valid page order to prevent downstream
        // overflow when corrupted data contains an out-of-range value.
        let order = if order > MAX_MAX_PAGE_ORDER {
            MAX_MAX_PAGE_ORDER
        } else {
            order
        };
        // The mask is at most 20 bits (0x000F_FFFF) shifted right by order,
        // so the result always fits in u32.
        #[allow(clippy::cast_possible_truncation)]
        let index = (temp & (0x000F_FFFF >> order)) as u32;
        #[allow(clippy::cast_possible_truncation)]
        let region = ((temp >> 20) & 0x000F_FFFF) as u32;

        Self {
            region,
            page_index: index,
            page_order: order,
        }
    }

    #[cfg(test)]
    pub(crate) fn to_order0(self) -> Vec<PageNumber> {
        let mut pages = vec![self];
        loop {
            let mut progress = false;
            let mut new_pages = vec![];
            for page in pages {
                if page.page_order == 0 {
                    new_pages.push(page);
                } else {
                    progress = true;
                    new_pages.push(PageNumber::new(
                        page.region,
                        page.page_index * 2,
                        page.page_order - 1,
                    ));
                    new_pages.push(PageNumber::new(
                        page.region,
                        page.page_index * 2 + 1,
                        page.page_order - 1,
                    ));
                }
            }
            pages = new_pages;
            if !progress {
                break;
            }
        }

        pages
    }

    /// Byte range this page occupies in the file.
    ///
    /// `page_index` and `page_order` both come off disk, so the range is only
    /// meaningful if the page actually fits inside its own region. The index is
    /// masked to 20 bits on decode, which at a 4 KiB page size still permits a
    /// regional offset of nearly 4 GiB -- far past the end of a small region.
    ///
    /// This was a `debug_assert!(regional_start < region_size)`, so release
    /// builds computed the address anyway and read whatever lived there,
    /// which for an out-of-region offset is another region's data. Report the
    /// corruption instead. The check covers the page's end as well as its
    /// start: a page may begin inside the region and still extend past it.
    pub(crate) fn address_range(
        &self,
        data_section_offset: u64,
        region_size: u64,
        region_pages_start: u64,
        page_size: u32,
    ) -> Result<Range<u64>, StorageError> {
        let page_bytes = self.page_size_bytes(page_size);
        let regional_start = u64::from(self.page_index)
            .checked_mul(page_bytes)
            .and_then(|offset| region_pages_start.checked_add(offset))
            .ok_or_else(|| {
                StorageError::Corrupted(format!(
                    "page {self:?}: regional offset overflows u64 at page size {page_size}"
                ))
            })?;
        let regional_end = regional_start.checked_add(page_bytes).ok_or_else(|| {
            StorageError::Corrupted(format!(
                "page {self:?}: regional end overflows u64 at page size {page_size}"
            ))
        })?;
        if regional_end > region_size {
            return Err(StorageError::Corrupted(format!(
                "page {self:?}: occupies {regional_start}..{regional_end} which does not fit in a region of {region_size} bytes"
            )));
        }
        let region_base = u64::from(self.region)
            .checked_mul(region_size)
            .ok_or_else(|| {
                StorageError::Corrupted(format!(
                    "page {self:?}: region base overflows u64 at region size {region_size}"
                ))
            })?;
        let start = data_section_offset
            .checked_add(region_base)
            .and_then(|base| base.checked_add(regional_start))
            .ok_or_else(|| {
                StorageError::Corrupted(format!("page {self:?}: absolute offset overflows u64"))
            })?;
        let end = start.checked_add(page_bytes).ok_or_else(|| {
            StorageError::Corrupted(format!("page {self:?}: absolute end overflows u64"))
        })?;
        Ok(start..end)
    }

    pub(crate) fn page_size_bytes(&self, page_size: u32) -> u64 {
        let pages = 1u64 << self.page_order;
        pages * u64::from(page_size)
    }
}

impl Debug for PageNumber {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "r{}.{}/{}",
            self.region, self.page_index, self.page_order
        )
    }
}

pub(crate) trait Page {
    fn memory(&self) -> &[u8];

    fn get_page_number(&self) -> PageNumber;
}

pub struct PageImpl {
    pub(super) mem: Arc<[u8]>,
    pub(super) page_number: PageNumber,
    #[cfg(debug_assertions)]
    pub(super) open_pages: Arc<Mutex<HashMap<PageNumber, u64>>>,
}

impl PageImpl {
    pub(crate) fn to_arc(&self) -> Arc<[u8]> {
        self.mem.clone()
    }
}

impl Debug for PageImpl {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("PageImpl: page_number={:?}", self.page_number))
    }
}

#[cfg(debug_assertions)]
impl Drop for PageImpl {
    fn drop(&mut self) {
        let mut open_pages = self.open_pages.lock();
        if let Some(value) = open_pages.get_mut(&self.page_number) {
            debug_assert!(*value > 0);
            *value -= 1;
            if *value == 0 {
                open_pages.remove(&self.page_number);
            }
        }
    }
}

impl Page for PageImpl {
    fn memory(&self) -> &[u8] {
        self.mem.as_ref()
    }

    fn get_page_number(&self) -> PageNumber {
        self.page_number
    }
}

impl Clone for PageImpl {
    fn clone(&self) -> Self {
        // Debug-only ref-count bookkeeping: the `open_pages` field exists only
        // under `cfg(debug_assertions)`. The Drop impl above is similarly
        // gated, so this Clone must match it. In release builds, Arc<[u8]>
        // already keeps the page memory alive after a free; the bookkeeping
        // is a debug-time use-after-free detector, not a correctness primitive.
        #[cfg(debug_assertions)]
        {
            *self.open_pages.lock().entry(self.page_number).or_insert(0) += 1;
        }
        Self {
            mem: self.mem.clone(),
            page_number: self.page_number,
            #[cfg(debug_assertions)]
            open_pages: self.open_pages.clone(),
        }
    }
}

pub(crate) struct PageMut {
    pub(super) mem: WritablePage,
    pub(super) page_number: PageNumber,
    #[cfg(debug_assertions)]
    pub(super) open_pages: Arc<Mutex<HashSet<PageNumber>>>,
}

impl PageMut {
    pub(crate) fn memory_mut(&mut self) -> crate::Result<&mut [u8]> {
        self.mem.mem_mut()
    }
}

impl Page for PageMut {
    fn memory(&self) -> &[u8] {
        self.mem.mem()
    }

    fn get_page_number(&self) -> PageNumber {
        self.page_number
    }
}

#[cfg(debug_assertions)]
impl Drop for PageMut {
    fn drop(&mut self) {
        assert!(self.open_pages.lock().remove(&self.page_number));
    }
}

#[derive(Copy, Clone)]
pub(crate) enum PageHint {
    None,
    Clean,
}

pub(crate) enum PageTrackerPolicy {
    Ignore,
    Track(HashSet<PageNumber>),
    Closed,
}

impl PageTrackerPolicy {
    pub(crate) fn new_tracking() -> Self {
        PageTrackerPolicy::Track(HashSet::new())
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            PageTrackerPolicy::Ignore | PageTrackerPolicy::Closed => true,
            PageTrackerPolicy::Track(x) => x.is_empty(),
        }
    }

    pub(super) fn remove(&mut self, page: PageNumber) {
        match self {
            PageTrackerPolicy::Ignore | PageTrackerPolicy::Closed => {}
            PageTrackerPolicy::Track(x) => {
                x.remove(&page);
            }
        }
    }

    pub(super) fn insert(&mut self, page: PageNumber) {
        match self {
            PageTrackerPolicy::Ignore | PageTrackerPolicy::Closed => {}
            PageTrackerPolicy::Track(x) => {
                x.insert(page);
            }
        }
    }

    pub(crate) fn close(&mut self) -> HashSet<PageNumber> {
        let old = mem::replace(self, PageTrackerPolicy::Closed);
        match old {
            PageTrackerPolicy::Ignore | PageTrackerPolicy::Closed => HashSet::new(),
            PageTrackerPolicy::Track(x) => x,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::tree_store::PageNumber;
    use crate::tree_store::page_store::base::MAX_PAGE_INDEX;

    #[test]
    fn last_page() {
        let region_data_size = 2u64.pow(32);
        let page_size = 4096;
        let pages_per_region = region_data_size / page_size;
        let region_header_size = 2u64.pow(16);
        let last_page_index = pages_per_region - 1;
        let page_number = PageNumber::new(1, last_page_index.try_into().unwrap(), 0);
        page_number
            .address_range(
                4096,
                region_data_size + region_header_size,
                region_header_size,
                page_size.try_into().unwrap(),
            )
            .unwrap();
    }

    /// `page_index` is read off disk and masked only to 20 bits, which at a
    /// 4 KiB page size still permits a regional offset of nearly 4 GiB -- far
    /// past the end of a small region. This was a
    /// `debug_assert!(regional_start < region_size)`, so release builds went on
    /// to compute an address inside a *different* region and read it.
    ///
    /// Found by the `fuzz_db_image` target:
    ///
    /// ```text
    /// panicked at base.rs:164: assertion failed: regional_start < region_size
    /// ```
    #[test]
    fn page_outside_its_region_is_rejected() {
        let page_size = 4096u32;
        // 1 MiB, the region size the fuzzer's generated configs reach.
        let region_size = 1024 * 1024u64;
        let region_header = 4096u64;
        let pages_per_region = (region_size - region_header) / u64::from(page_size);

        // Far past the region end.
        let far = PageNumber::new(0, MAX_PAGE_INDEX, 0);
        assert!(
            far.address_range(4096, region_size, region_header, page_size)
                .is_err()
        );

        // Starts inside the region but extends past its end -- checking only
        // the start would let this through.
        let straddling = PageNumber::new(0, u32::try_from(pages_per_region).unwrap(), 0);
        assert!(
            straddling
                .address_range(4096, region_size, region_header, page_size)
                .is_err()
        );

        // The last page that genuinely fits is still accepted, so the guard is
        // not off by one.
        let last = PageNumber::new(0, u32::try_from(pages_per_region - 1).unwrap(), 0);
        assert!(
            last.address_range(4096, region_size, region_header, page_size)
                .is_ok()
        );
    }

    #[test]
    fn reserved_bits() {
        let page_number = PageNumber::new(0, 0, 12);
        let mut bytes = page_number.to_le_bytes();
        bytes[1] = 0xFF;
        let page_number2 = PageNumber::from_le_bytes(bytes);
        assert_eq!(page_number, page_number2);
    }
}
