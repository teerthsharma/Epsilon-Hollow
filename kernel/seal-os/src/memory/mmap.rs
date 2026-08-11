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

/// User-space virtual bump allocator for mmap.
/// Starts at 1 MiB to avoid the null-page and low-memory BIOS/UEFI areas.
/// Grows upward; no holes are reclaimed in this minimal implementation.
static USER_VIRT_BUMP: AtomicU64 = AtomicU64::new(0x10_0000);

/// Reserve a contiguous virtual address range for the given page table.
///
/// No physical frames are allocated — they are fetched on the first page fault.
pub fn mmap_user(pages: usize, flags: PageTableFlags, page_table: u64) -> Option<VirtAddr> {
    let size = pages as u64 * 4096;
    let addr = USER_VIRT_BUMP.fetch_add(size, Ordering::SeqCst);

    // Hard cap: stay well below the user-stack and kernel-half boundaries.
    if addr + size > 0x0000_7FFF_0000_0000 {
        return None;
    }

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

    pub fn register_all() {
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
