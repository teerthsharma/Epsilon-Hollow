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
    PhysAddr, VirtAddr,
};

/// A lazily-backed mmap region.
pub struct MmapRegion {
    pub start: VirtAddr,
    pub pages: usize,
    pub flags: PageTableFlags,
    /// Physical address of the PML4 this region belongs to.
    pub page_table: u64,
}

static REGIONS: Mutex<Vec<MmapRegion>> = Mutex::new(Vec::new());

/// User-space virtual bump allocator for mmap.
/// Starts at 1 MiB to avoid the null-page and low-memory BIOS/UEFI areas.
/// Grows upward; no holes are reclaimed in this minimal implementation.
static USER_VIRT_BUMP: AtomicU64 = AtomicU64::new(0x10_0000);

/// Reserve a contiguous virtual address range for the given page table.
///
/// No physical frames are allocated — they are fetched on the first page fault.
pub fn mmap_user(
    pages: usize,
    flags: PageTableFlags,
    page_table: u64,
) -> Option<VirtAddr> {
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

/// Attempt to satisfy a page fault by lazily backing the faulting page.
///
/// Returns `true` if the fault was handled (the page is now mapped and the
/// faulting instruction can be retried).  Returns `false` if the address does
/// not belong to any known mmap region — the caller should treat it as a
/// hard fault.
pub fn handle_page_fault(fault_addr: VirtAddr) -> bool {
    let current_pt = unsafe { x86_64::registers::control::Cr3::read() }
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
                // Ignore map errors — if the page table structure is broken
                // the fault will recur and we'll return false next time.
                let _ = crate::memory::virt::map_page_to_pml4(
                    page_start,
                    frame,
                    region.flags,
                    pml4,
                );
            }
            return true;
        }
    }
    false
}
