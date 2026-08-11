// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Lazy demand paging for mmap regions.
//!
//! `SYS_MMAP` reserves a contiguous virtual address range and records it as a
//! *region*.  The first access that triggers a page fault causes the kernel to
//! allocate a physical frame, zero it, and map it on-the-fly.  This is true
//! demand paging — pages are not backed until they are touched.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;
use x86_64::{
    structures::paging::{PageTable, PageTableFlags},
    VirtAddr,
};

/// A lazily-backed mmap region.
#[derive(Clone)]
pub struct MmapRegion {
    pub start: VirtAddr,
    pub pages: usize,
    pub flags: PageTableFlags,
    /// Physical address of the PML4 this region belongs to.
    pub page_table: u64,
}

/// Global list of mmap regions.  `pub(super)` so `swap.rs` can iterate over them.
pub(super) static REGIONS: Mutex<Vec<MmapRegion>> = Mutex::new(Vec::new());

/// Lowest address the bump may hand out. 1 MiB, clear of the null page and the
/// low-memory BIOS/UEFI areas.
const USER_VIRT_BASE: u64 = 0x10_0000;

/// Exclusive upper bound on the bump: stay well below the user-stack and
/// kernel-half boundaries.
const USER_VIRT_END: u64 = 0x0000_7FFF_0000_0000;

/// Largest single reservation, in pages (1 GiB).
///
/// `mmap_user` backs nothing at reservation time — pages are faulted in
/// lazily — so a reservation costs a caller no physical memory at all. Without
/// a per-call bound, a handful of `SYS_MMAP` calls asking for terabytes each
/// would consume the whole 128 TiB range at zero cost to the caller and, since
/// the bump is global and nothing returns a range to it, leave every process on
/// the machine permanently unable to `mmap` or grow its break. The bound does
/// not remove that attack, it only prices it: exhausting the range now takes
/// 131,072 successful reservations, each of which also pins an `MmapRegion` in
/// `REGIONS`. Nothing in this kernel reserves anywhere near 1 GiB in one call.
const MAX_RESERVATION_PAGES: usize = 1 << 18;

/// User-space virtual bump allocator for mmap.
///
/// One global counter, not one per address space, so two processes never see
/// the same address even where they could. It only grows: `munmap_user` frees
/// the physical frames but the virtual range is not returned here.
///
/// ponytail: no free list. A released range is consulted by nobody, because
/// nothing can release one — `munmap_user` has no callers and no `SYS_MUNMAP`
/// exists, so a free list would be unreachable code with an O(n) scan on the
/// reservation path. Upgrade path, in order: wire `SYS_MUNMAP` in
/// `syscall/table.rs`, then have `munmap_user` push `(start, pages)` onto a
/// coalescing free list that `mmap_user` first-fits before advancing the bump.
/// Note when doing so that the bump leaves no guard gap between reservations
/// today — successive ranges are already exactly adjacent — so returning an
/// exact range to a free list introduces no adjacency that does not already
/// occur.
static USER_VIRT_BUMP: AtomicU64 = AtomicU64::new(USER_VIRT_BASE);

/// Reserve a contiguous virtual address range for the given page table.
///
/// No physical frames are allocated — they are fetched on the first page fault.
///
/// Returns `None` for an empty or oversized request, and for any request the
/// remaining range cannot satisfy. The bump only moves when a reservation
/// succeeds: the range is bounds-checked before the compare-exchange commits
/// it, so a refused call leaves the counter exactly where it was. Advancing
/// first and checking afterwards — which is what a bare `fetch_add` does — let
/// one refused request permanently consume address space it was never granted,
/// and since `USER_VIRT_BUMP` is one global counter shared by every address
/// space, that loss is not the caller's alone.
///
/// Every arithmetic step is checked. `pages` arrives from userspace by way of
/// `len.div_ceil(4096)` in `SYS_MMAP`, so it reaches 2^52 for `len = u64::MAX`;
/// `pages * 4096` then overflows, and the kernel's release profile leaves
/// overflow checks off, so it wraps to 0. A zero-size reservation returns an
/// address without reserving it and the next call hands the same address out
/// again — two live regions at one address, which is worse than exhaustion.
/// `pages == 0` (from `mmap(len = 0)`) produced the same duplicate.
pub fn mmap_user(pages: usize, flags: PageTableFlags, page_table: u64) -> Option<VirtAddr> {
    if pages == 0 || pages > MAX_RESERVATION_PAGES {
        return None;
    }
    let size = (pages as u64).checked_mul(4096)?;

    // The range is computed and bounds-checked before the store is committed,
    // so every rejection path returns without touching the counter. `fetch_add`
    // cannot express that ordering; compare-exchange can. A lost race only
    // re-reads the counter and retries, so the loop is bounded by contention,
    // not by the size of the request.
    let mut cur = USER_VIRT_BUMP.load(Ordering::SeqCst);
    let addr = loop {
        let end = cur.checked_add(size)?;
        if end > USER_VIRT_END {
            return None;
        }
        match USER_VIRT_BUMP.compare_exchange_weak(cur, end, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => break cur,
            Err(observed) => cur = observed,
        }
    };

    let virt = VirtAddr::new(addr);
    REGIONS.lock().push(MmapRegion {
        start: virt,
        pages,
        flags,
        page_table,
    });
    Some(virt)
}

/// Unmap `pages` pages starting at `start` and free backing frames.
///
/// The lookup only matches a region registered under the *caller's own*
/// page table. `REGIONS` is one global list shared by every process, and a
/// region's `(start, pages)` pair is only unique within the address space
/// that reserved it — two unrelated processes (or a COW-forked parent and
/// child, which alias the same virtual addresses) can legitimately race the
/// same key. Without the `page_table` check, a call in one address space
/// could match another process's region, unmap and free frames out from
/// under *that* process, while leaving the caller's own mapping at the same
/// address untouched. A region owned by a different page table is refused
/// (`false`) rather than acted on.
///
/// # Safety
/// Caller must ensure the range matches a previously-mapped region.
pub unsafe fn munmap_user(start: VirtAddr, pages: usize) -> bool {
    let current_pt = x86_64::registers::control::Cr3::read()
        .0
        .start_address()
        .as_u64();

    let mut regions = REGIONS.lock();
    let idx = regions
        .iter()
        .position(|r| r.start == start && r.pages == pages && r.page_table == current_pt);
    let idx = match idx {
        Some(i) => i,
        None => return false,
    };
    let region = regions.swap_remove(idx);
    drop(regions);

    // `region.page_table == current_pt` is now guaranteed by the match
    // above, so every unmap in this loop targets the caller's own live
    // page table.
    for i in 0..pages {
        let v = VirtAddr::new(start.as_u64() + i as u64 * 4096);
        if let Some(frame) =
            crate::memory::virt::unmap_page_in_pml4(v, x86_64::PhysAddr::new(region.page_table))
        {
            crate::memory::phys::free_frame(frame);
        }
    }
    true
}

/// Attempt to satisfy a page fault by lazily backing the faulting page.
///
/// Returns `true` if the fault was handled (the page is now mapped and the
/// faulting instruction can be retried).  Returns `false` if the address does
/// not belong to any known mmap region — the caller should treat it as a
/// hard fault.
pub fn handle_page_fault(fault_addr: VirtAddr) -> bool {
    let current_pt = x86_64::registers::control::Cr3::read()
        .0
        .start_address()
        .as_u64();

    let regions = REGIONS.lock();
    for region in regions.iter() {
        if region.page_table != current_pt {
            continue;
        }
        let end = region.start.as_u64() + region.pages as u64 * 4096;
        if fault_addr.as_u64() >= region.start.as_u64() && fault_addr.as_u64() < end {
            let page_start = VirtAddr::new(fault_addr.as_u64() & !0xFFF);
            let frame = crate::memory::phys::alloc_frame();
            let Some(frame) = frame else {
                return false;
            };
            unsafe {
                core::ptr::write_bytes(frame.as_u64() as *mut u8, 0, 4096);
                let pml4 = &mut *(current_pt as *mut PageTable);
                // `map_page_inner` refuses to overwrite an already-present
                // leaf instead of silently corrupting it (see
                // `virt::map_page_inner`). That only happens here when
                // another racing fault on this exact page already won and
                // mapped it, so the page is present either way and retrying
                // the faulting instruction below will succeed against the
                // winner's mapping. This call's freshly allocated `frame`
                // was never installed on that path, so it must be freed
                // here or every losing racer leaks a frame.
                if crate::memory::virt::map_page_to_pml4(page_start, frame, region.flags, pml4)
                    .is_err()
                {
                    crate::memory::phys::free_frame(frame);
                }
            }
            return true;
        }
    }
    false
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// Defect regression: `munmap_user` must never act on a region owned by
    /// a page table other than the caller's own. Before the `page_table`
    /// check was added to the lookup, this call matched on `(start, pages)`
    /// alone, removed the region from the shared global `REGIONS` list, and
    /// then unmapped/freed frames through `foreign_pt` — corrupting a
    /// different address space's mapping while leaving the caller's own
    /// mapping (if any existed) untouched.
    ///
    /// `foreign_pt` is a zeroed `PageTable` living on this function's own
    /// stack, never installed as anyone's CR3. It stands in for "some other
    /// process's PML4": distinct from the real `current_pt`, and safe to
    /// pass to `unmap_page_in_pml4` because every level is `is_unused()`, so
    /// a walk into it (if the bug under test is still present) bottoms out
    /// immediately instead of dereferencing unrelated memory.
    fn test_munmap_user_refuses_foreign_page_table() -> TestResult {
        let current_pt = x86_64::registers::control::Cr3::read()
            .0
            .start_address()
            .as_u64();

        let foreign_pml4 = PageTable::new();
        let foreign_pt = &foreign_pml4 as *const PageTable as u64;
        test_assert!(foreign_pt != current_pt);

        // Deep into the user pml4 slot (index 253 of 0..256), never touched
        // by boot-time identity mapping (pml4 index 0) or kernel mappings
        // (indices 256..512), and distinct from every other scratch address
        // used in this file's tests.
        let start = VirtAddr::new(0x0000_7EFE_0000_0000);
        let pages = 1;

        REGIONS.lock().push(MmapRegion {
            start,
            pages,
            flags: PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            page_table: foreign_pt,
        });

        let result = unsafe { munmap_user(start, pages) };

        let still_present = REGIONS
            .lock()
            .iter()
            .any(|r| r.start == start && r.pages == pages && r.page_table == foreign_pt);
        // Clean up before asserting so a failure doesn't leave the phantom
        // region behind for later tests.
        REGIONS.lock().retain(|r| r.page_table != foreign_pt);

        test_assert!(
            !result,
            "munmap_user must refuse a region owned by another page table"
        );
        test_assert!(
            still_present,
            "a refused region must not be removed from REGIONS"
        );
        TestResult::Pass
    }

    /// Companion to the refusal test above: the `page_table` guard must not
    /// block a genuinely self-owned region, or `munmap_user` would be dead
    /// on arrival for every legitimate caller.
    fn test_munmap_user_accepts_own_page_table() -> TestResult {
        let current_pt = x86_64::registers::control::Cr3::read()
            .0
            .start_address()
            .as_u64();

        let start = VirtAddr::new(0x0000_7EFD_0000_0000);
        let pages = 1;

        REGIONS.lock().push(MmapRegion {
            start,
            pages,
            flags: PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            page_table: current_pt,
        });

        let result = unsafe { munmap_user(start, pages) };
        test_assert!(
            result,
            "munmap_user must succeed for a region the caller's own page table registered"
        );

        let still_present = REGIONS
            .lock()
            .iter()
            .any(|r| r.start == start && r.pages == pages);
        test_assert!(!still_present, "a matched region must be removed from REGIONS");
        TestResult::Pass
    }

    /// Defect regression: a losing racer in `handle_page_fault` must free
    /// the frame it allocated when `map_page_to_pml4` refuses an
    /// already-present leaf, instead of leaking it. Simulated here without
    /// real concurrency by faulting the same fresh address twice: the first
    /// call legitimately maps a frame, so the second call's mapping attempt
    /// is guaranteed to hit the same "leaf already present" refusal a real
    /// losing racer would hit.
    fn test_handle_page_fault_frees_frame_on_map_failure() -> TestResult {
        let current_pt = x86_64::registers::control::Cr3::read()
            .0
            .start_address()
            .as_u64();

        let start = VirtAddr::new(0x0000_7EFC_0000_0000);
        let pages = 1;

        REGIONS.lock().push(MmapRegion {
            start,
            pages,
            flags: PageTableFlags::PRESENT | PageTableFlags::WRITABLE,
            page_table: current_pt,
        });

        let first = handle_page_fault(start);
        let free_before_second = crate::memory::phys::free_count();
        let second = handle_page_fault(start);
        let free_after_second = crate::memory::phys::free_count();

        // Cleanup regardless of outcome: drop the real mapping/frame this
        // test installed and remove the synthetic region.
        unsafe {
            if let Some(phys) = crate::memory::virt::unmap_page(start) {
                crate::memory::phys::free_frame(phys);
            }
        }
        REGIONS.lock().retain(|r| r.start != start);

        test_assert!(first, "first fault on a fresh mmap region must be handled");
        test_assert!(
            second,
            "a fault on an already-mapped page must still be reported handled"
        );
        test_assert_eq!(free_before_second, free_after_second);
        TestResult::Pass
    }

    /// Defect regression: a refused reservation must leave `USER_VIRT_BUMP`
    /// exactly where it was.
    ///
    /// The old body called `fetch_add` first and bounds-checked the result
    /// afterwards, with no rollback on the failing path. Because the counter is
    /// one global shared by every address space, a single `SYS_MMAP` asking for
    /// the whole 128 TiB range returned `ENOMEM` to its caller and left the bump
    /// past the cap — after which every `mmap` and every `brk` growth in every
    /// process failed permanently. One unprivileged syscall, system-wide, with
    /// no way to undo it.
    ///
    /// The four requests below are the ones that reach the arithmetic from
    /// userspace. `SYS_MMAP` computes `pages` as `len.div_ceil(4096)`, so
    /// `len = u64::MAX` yields 2^52 pages; `2^52 * 4096` is exactly 2^64 and the
    /// release profile sets no `overflow-checks`, so it wrapped to a zero-size
    /// reservation that advanced nothing and handed the next caller the same
    /// address. One page below that wrapped the sum to `2^64 - 4096`, driving
    /// the bump backward below `USER_VIRT_BASE` while `addr + size` wrapped back
    /// under the cap, so the check passed.
    fn test_mmap_user_refuses_hostile_request_without_moving_bump() -> TestResult {
        let current_pt = x86_64::registers::control::Cr3::read()
            .0
            .start_address()
            .as_u64();
        let flags = PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE;

        let before = USER_VIRT_BUMP.load(Ordering::SeqCst);

        let empty = mmap_user(0, flags, current_pt);
        let overflow = mmap_user(1usize << 52, flags, current_pt);
        let backward = mmap_user((1usize << 52) - 1, flags, current_pt);
        let whole_range = mmap_user((USER_VIRT_END / 4096) as usize, flags, current_pt);

        let after = USER_VIRT_BUMP.load(Ordering::SeqCst);

        // Any request that wrongly succeeded also left a region behind.
        for start in [empty, overflow, backward, whole_range].into_iter().flatten() {
            REGIONS.lock().retain(|r| r.start != start);
        }

        test_assert!(empty.is_none(), "a zero-page reservation must be refused");
        test_assert!(
            overflow.is_none(),
            "a reservation whose byte size overflows must be refused"
        );
        test_assert!(
            backward.is_none(),
            "a reservation that would wrap the bump backward must be refused"
        );
        test_assert!(
            whole_range.is_none(),
            "a reservation spanning the whole user range must be refused"
        );
        test_assert_eq!(before, after);
        TestResult::Pass
    }

    /// The property the bump exists to hold: an address it hands out never
    /// overlaps a region that is already live.
    ///
    /// Snapshots every live region in the caller's own address space first, then
    /// checks both fresh reservations against that snapshot and against each
    /// other, byte-wise. Regions belonging to other page tables are excluded
    /// because addresses in different address spaces are permitted to coincide.
    /// Pre-fix, `mmap(len = 0)` and `mmap(len = u64::MAX)` both reserved zero
    /// bytes, so the following call was handed an address inside the region the
    /// previous call had just registered.
    fn test_mmap_user_hands_out_disjoint_addresses() -> TestResult {
        let current_pt = x86_64::registers::control::Cr3::read()
            .0
            .start_address()
            .as_u64();
        let flags = PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE;

        let live: Vec<(u64, u64)> = REGIONS
            .lock()
            .iter()
            .filter(|r| r.page_table == current_pt)
            .map(|r| (r.start.as_u64(), r.pages as u64 * 4096))
            .collect();

        let first = mmap_user(1, flags, current_pt);
        let second = mmap_user(2, flags, current_pt);

        // Release the regions before asserting. The bump itself cannot be
        // rewound — nothing returns a range to it — so this test permanently
        // consumes 12 KiB of the 128 TiB range each time it runs.
        for start in [first, second].into_iter().flatten() {
            REGIONS.lock().retain(|r| r.start != start);
        }

        let (a, b) = match (first, second) {
            (Some(a), Some(b)) => (a.as_u64(), b.as_u64()),
            _ => return TestResult::Fail("mmap_user must serve two small reservations"),
        };

        for (start, len) in [(a, 4096u64), (b, 8192u64)] {
            test_assert!(
                start >= USER_VIRT_BASE && start + len <= USER_VIRT_END,
                "a reservation must land inside the user range"
            );
            for (other, other_len) in live.iter().copied() {
                test_assert!(
                    start >= other + other_len || other >= start + len,
                    "a fresh reservation overlaps a region that was already live"
                );
            }
        }
        test_assert!(
            b >= a + 4096,
            "two successive reservations must not overlap each other"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "mmap::mmap_user_refuses_hostile_request_without_moving_bump",
            test_mmap_user_refuses_hostile_request_without_moving_bump,
        );
        crate::testing::register_test(
            "mmap::mmap_user_hands_out_disjoint_addresses",
            test_mmap_user_hands_out_disjoint_addresses,
        );
        crate::testing::register_test(
            "mmap::munmap_user_refuses_foreign_page_table",
            test_munmap_user_refuses_foreign_page_table,
        );
        crate::testing::register_test(
            "mmap::munmap_user_accepts_own_page_table",
            test_munmap_user_accepts_own_page_table,
        );
        crate::testing::register_test(
            "mmap::handle_page_fault_frees_frame_on_map_failure",
            test_handle_page_fault_frees_frame_on_map_failure,
        );
    }
}
