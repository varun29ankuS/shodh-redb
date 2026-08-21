use crate::StorageError;
use core::ops::Range;

fn round_up_to_multiple_of(value: u64, multiple: u64) -> u64 {
    if value % multiple == 0 {
        value
    } else {
        value + multiple - value % multiple
    }
}

// Regions are laid out starting with the allocator state header, followed by the pages aligned
// to the next page
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RegionLayout {
    num_pages: u32,
    // Offset where data pages start
    header_pages: u32,
    page_size: u32,
}

#[allow(clippy::cast_possible_truncation)]
impl RegionLayout {
    pub(super) fn new(num_pages: u32, header_pages: u32, page_size: u32) -> Self {
        assert!(num_pages > 0);
        Self {
            num_pages,
            header_pages,
            page_size,
        }
    }

    pub(super) fn calculate(
        desired_usable_bytes: u64,
        page_capacity: u32,
        region_header_pages: u32,
        page_size: u32,
    ) -> RegionLayout {
        assert!(desired_usable_bytes <= u64::from(page_capacity) * u64::from(page_size));
        let num_pages =
            round_up_to_multiple_of(desired_usable_bytes, page_size.into()) / u64::from(page_size);

        Self {
            num_pages: num_pages as u32,
            header_pages: region_header_pages,
            page_size,
        }
    }

    pub(super) fn data_section(&self) -> Range<u64> {
        let header_bytes = u64::from(self.header_pages) * u64::from(self.page_size);
        header_bytes..(header_bytes + self.usable_bytes())
    }

    pub(super) fn get_header_pages(&self) -> u32 {
        self.header_pages
    }

    pub(super) fn num_pages(&self) -> u32 {
        self.num_pages
    }

    pub(super) fn page_size(&self) -> u32 {
        self.page_size
    }

    pub(super) fn len(&self) -> u64 {
        u64::from(self.header_pages) * u64::from(self.page_size) + self.usable_bytes()
    }

    pub(super) fn usable_bytes(&self) -> u64 {
        u64::from(self.page_size) * u64::from(self.num_pages)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DatabaseLayout {
    full_region_layout: RegionLayout,
    num_full_regions: u32,
    trailing_partial_region: Option<RegionLayout>,
}

#[allow(clippy::cast_possible_truncation)]
impl DatabaseLayout {
    pub(super) fn new(
        full_regions: u32,
        full_region: RegionLayout,
        trailing_region: Option<RegionLayout>,
    ) -> Self {
        Self {
            full_region_layout: full_region,
            num_full_regions: full_regions,
            trailing_partial_region: trailing_region,
        }
    }

    pub(super) fn reduce_last_region(&mut self, pages: u32) {
        if let Some(ref mut trailing) = self.trailing_partial_region {
            assert!(pages <= trailing.num_pages);
            trailing.num_pages -= pages;
            if trailing.num_pages == 0 {
                self.trailing_partial_region = None;
            }
        } else {
            self.num_full_regions -= 1;
            let full_layout = self.full_region_layout;
            if full_layout.num_pages > pages {
                self.trailing_partial_region = Some(RegionLayout::new(
                    full_layout.num_pages - pages,
                    full_layout.header_pages,
                    full_layout.page_size,
                ));
            }
        }
    }

    // Every input except `file_len` comes from the database header, and at two
    // of the three call sites `file_len` is `blob_region_offset`, which is read
    // off disk as well. So none of this arithmetic may assume sane operands.
    // Release builds have overflow checks off, so what used to happen on a
    // crafted file was a silent wrap followed by one of two `assert!`s -- not
    // `debug_assert!`, so they are compiled into release and panic there.
    //
    // Unlike the allocator state, which is a redundant cache that can be
    // rebuilt from the tree, the header's layout parameters are the source of
    // truth for where every page lives. There is nothing to rebuild them from,
    // so an impossible layout is `Corrupted` rather than a recovery request.
    pub(super) fn recalculate(
        file_len: u64,
        region_header_pages_u32: u32,
        region_max_data_pages_u32: u32,
        page_size_u32: u32,
    ) -> Result<Self, StorageError> {
        let page_size = u64::from(page_size_u32);
        let region_header_pages = u64::from(region_header_pages_u32);
        let region_max_data_pages = u64::from(region_max_data_pages_u32);

        let invalid =
            |what: &str| StorageError::Corrupted(alloc::format!("Database layout: {what}"));

        if page_size == 0 {
            return Err(invalid("page size is zero"));
        }
        // `RegionLayout::new` asserts this, and that assertion is right for its
        // trusted callers. Report it here instead of reaching it.
        if region_max_data_pages_u32 == 0 {
            return Err(invalid("a full region has no data pages"));
        }

        // Super-header
        let mut remaining = file_len.checked_sub(page_size).ok_or_else(|| {
            invalid(&alloc::format!(
                "file is {file_len} bytes, shorter than the {page_size}-byte super-header"
            ))
        })?;

        let full_region_size = region_header_pages
            .checked_add(region_max_data_pages)
            .and_then(|pages| pages.checked_mul(page_size))
            .ok_or_else(|| invalid("region size overflows a 64-bit byte count"))?;
        let full_regions = remaining / full_region_size;
        remaining -= full_regions * full_region_size;

        let header_bytes = region_header_pages
            .checked_mul(page_size)
            .ok_or_else(|| invalid("region header size overflows a 64-bit byte count"))?;
        // A trailing region needs its header plus at least one data page.
        let trailing_threshold = header_bytes
            .checked_add(page_size)
            .ok_or_else(|| invalid("region header size overflows a 64-bit byte count"))?;

        let trailing = if remaining >= trailing_threshold {
            // Divide before narrowing. The original narrowed first, with an
            // `as u32` that silently discarded the high bits and could
            // manufacture a zero page count out of a multiple of 2^32.
            let data_pages = (remaining - header_bytes) / page_size;
            let data_pages = u32::try_from(data_pages).map_err(|_| {
                invalid(&alloc::format!(
                    "trailing region of {data_pages} pages exceeds a 32-bit page index"
                ))
            })?;
            // The trailing region is partial by definition, and a region with
            // no data pages is not a region.
            if data_pages == 0 || data_pages >= region_max_data_pages_u32 {
                return Err(invalid(&alloc::format!(
                    "trailing region of {data_pages} pages is not a partial region of the                      {region_max_data_pages_u32} pages in a full one"
                )));
            }
            Some(RegionLayout::new(
                data_pages,
                region_header_pages_u32,
                page_size_u32,
            ))
        } else {
            None
        };

        let num_full_regions = u32::try_from(full_regions).map_err(|_| {
            invalid(&alloc::format!(
                "{full_regions} regions exceeds a 32-bit region index"
            ))
        })?;

        Ok(Self {
            full_region_layout: RegionLayout::new(
                region_max_data_pages_u32,
                region_header_pages_u32,
                page_size_u32,
            ),
            num_full_regions,
            trailing_partial_region: trailing,
        })
    }

    pub(super) fn calculate(
        desired_usable_bytes: u64,
        page_capacity: u32,
        region_header_pages: u32,
        page_size: u32,
    ) -> Self {
        let full_region_layout = RegionLayout::new(page_capacity, region_header_pages, page_size);
        if desired_usable_bytes <= full_region_layout.usable_bytes() {
            // Single region layout
            let region_layout = RegionLayout::calculate(
                desired_usable_bytes,
                page_capacity,
                region_header_pages,
                page_size,
            );
            DatabaseLayout {
                full_region_layout,
                num_full_regions: 0,
                trailing_partial_region: Some(region_layout),
            }
        } else {
            // Multi region layout
            let full_regions = desired_usable_bytes / full_region_layout.usable_bytes();
            let remaining_desired =
                desired_usable_bytes - full_regions * full_region_layout.usable_bytes();
            assert!(full_regions > 0);
            let trailing_region = if remaining_desired > 0 {
                Some(RegionLayout::calculate(
                    remaining_desired,
                    page_capacity,
                    region_header_pages,
                    page_size,
                ))
            } else {
                None
            };
            if let Some(ref region) = trailing_region {
                // All regions must have the same header size
                assert_eq!(region.header_pages, full_region_layout.header_pages);
            }
            DatabaseLayout {
                full_region_layout,
                num_full_regions: full_regions as u32,
                trailing_partial_region: trailing_region,
            }
        }
    }

    pub(super) fn full_region_layout(&self) -> &RegionLayout {
        &self.full_region_layout
    }

    pub(super) fn trailing_region_layout(&self) -> Option<&RegionLayout> {
        self.trailing_partial_region.as_ref()
    }

    pub(super) fn num_full_regions(&self) -> u32 {
        self.num_full_regions
    }

    pub(super) fn num_regions(&self) -> u32 {
        if self.trailing_partial_region.is_some() {
            self.num_full_regions + 1
        } else {
            self.num_full_regions
        }
    }

    pub(super) fn len(&self) -> u64 {
        let last = self.num_regions() - 1;
        self.region_base_address(last) + self.region_layout(last).len()
    }

    pub(super) fn usable_bytes(&self) -> u64 {
        let trailing = self
            .trailing_partial_region
            .as_ref()
            .map(RegionLayout::usable_bytes)
            .unwrap_or_default();
        u64::from(self.num_full_regions) * self.full_region_layout.usable_bytes() + trailing
    }

    pub(super) fn region_base_address(&self, region: u32) -> u64 {
        assert!(region < self.num_regions());
        u64::from(self.full_region_layout.page_size())
            + u64::from(region) * self.full_region_layout.len()
    }

    pub(super) fn region_layout(&self, region: u32) -> RegionLayout {
        assert!(region < self.num_regions());
        if region == self.num_full_regions {
            // Safety: if region == num_full_regions and region < num_regions(),
            // then num_regions() > num_full_regions, which implies trailing exists.
            match self.trailing_partial_region {
                Some(layout) => layout,
                None => self.full_region_layout,
            }
        } else {
            self.full_region_layout
        }
    }
}

#[cfg(test)]
mod recalculate_test {
    use super::DatabaseLayout;

    const PAGE: u32 = 4096;

    // An ordinary reopen: one full region plus a partial trailing one. This is
    // the control -- the checks below must not reject a layout the database
    // itself would write.
    #[test]
    fn an_ordinary_layout_is_accepted() {
        let region_pages = 256u32;
        let full_region = u64::from(region_pages) * u64::from(PAGE);
        let file_len = u64::from(PAGE) + full_region + 64 * u64::from(PAGE);

        let layout = DatabaseLayout::recalculate(file_len, 0, region_pages, PAGE).unwrap();
        assert_eq!(layout.num_full_regions(), 1);
        assert_eq!(
            layout.trailing_region_layout().map(|r| r.num_pages()),
            Some(64)
        );
        assert_eq!(layout.len(), file_len);
    }

    // The largest trailing region that is still partial. `remaining` here is
    // just under 2^32, which is precisely where narrowing to u32 stops being
    // safe -- so this is the case a too-eager bound would wrongly reject.
    #[test]
    fn a_maximal_partial_region_is_accepted() {
        let region_pages = 0x10_0000u32;
        let trailing_pages = region_pages - 1;
        let file_len = u64::from(PAGE) + u64::from(trailing_pages) * u64::from(PAGE);

        let layout = DatabaseLayout::recalculate(file_len, 0, region_pages, PAGE).unwrap();
        assert_eq!(layout.num_full_regions(), 0);
        assert_eq!(
            layout.trailing_region_layout().map(|r| r.num_pages()),
            Some(trailing_pages)
        );
    }

    // `file_len - page_size`. A file shorter than its own super-header used to
    // wrap to about 2^64 in release and compute every later quantity from it.
    #[test]
    fn a_file_shorter_than_the_super_header_is_rejected() {
        assert!(DatabaseLayout::recalculate(100, 0, 3, u32::MAX).is_err());
        assert!(DatabaseLayout::recalculate(0, 0, 3, PAGE).is_err());
    }

    // `(region_header_pages + region_max_data_pages) * page_size`. The sum
    // reaches 2^32 and the product 2^64, which wrapped to a garbage-small
    // region size and so to a wildly wrong region count.
    #[test]
    fn a_region_size_that_overflows_is_rejected() {
        assert!(DatabaseLayout::recalculate(1 << 40, u32::MAX, 3, u32::MAX).is_err());
    }

    // `remaining as u32`. This is the case ox's audit flagged and it is worth
    // stating precisely, because the fix is not a rejection. With 2^32 bytes
    // left over, narrowing before dividing gave zero, and a trailing region of
    // zero pages tripped `assert!(num_pages > 0)` inside `RegionLayout::new` --
    // a plain `assert!`, so it panicked in release builds too. Dividing first
    // yields the true count of one page, which is a perfectly valid partial
    // region. The defect was the narrowing, not the input.
    #[test]
    fn a_trailing_region_larger_than_a_32_bit_byte_count_is_measured_not_narrowed() {
        let file_len = 0x1_FFFF_FFFFu64;
        let layout = DatabaseLayout::recalculate(file_len, 0, 2, u32::MAX).unwrap();
        assert_eq!(layout.num_full_regions(), 0);
        assert_eq!(
            layout.trailing_region_layout().map(|r| r.num_pages()),
            Some(1)
        );
    }

    // `full_regions as u32`. `file_len` is `blob_region_offset` at two of the
    // three call sites and is read off disk, so a region count past 2^32 does
    // not need a real file of that size.
    #[test]
    fn a_region_count_past_a_32_bit_index_is_rejected() {
        assert!(DatabaseLayout::recalculate(u64::MAX, 0, 1, 512).is_err());
    }

    // `RegionLayout::new` asserts a non-zero page count. Report it rather than
    // reaching it.
    #[test]
    fn a_full_region_with_no_data_pages_is_rejected() {
        assert!(DatabaseLayout::recalculate(1 << 20, 0, 0, PAGE).is_err());
    }
}
