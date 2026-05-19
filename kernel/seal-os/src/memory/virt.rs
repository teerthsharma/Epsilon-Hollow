// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Virtual memory manager — 4-level page tables.
//!
//! On init we:
//! 1. Allocate a fresh PML4.
//! 2. Identity-map the first 4 GiB using 2 MiB huge pages (boot compatibility).
//! 3. Map the kernel image to `0xffffffff80000000` using 4 KiB pages.
//! 4. Switch CR3.
//!
//! All page-table pages are allocated from the physical allocator and accessed
//! through the identity map (they live below 4 GiB).

use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;
use x86_64::{
    PhysAddr, VirtAddr,
    registers::control::{Cr3, Cr3Flags},
    structures::paging::{PageTable, PageTableFlags},
    structures::paging::page_table::PageTableEntry,
};

const KERNEL_HIGHER_HALF: u64 = 0xffff_ffff_8000_0000;
const IDENTITY_MAP_SIZE: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB
const HUGE_PAGE_SIZE: u64 = 2 * 1024 * 1024; // 2 MiB

const HEAP_VIRTUAL_BASE: u64 = 0xffff_9000_0000_0000;

/// Error returned when a page-table allocation or mapping fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapError;

/// Virtual address of the current PML4 (accessed via identity-mapped address).
static PML4_VIRT: Mutex<VirtAddr> = Mutex::new(VirtAddr::new_truncate(0));

/// Physical address of the BSP's PML4 (set during init, shared with APs).
static BSP_PML4_PHYS: AtomicU64 = AtomicU64::new(0);

/// Return the physical address of the BSP's PML4.
pub fn bsp_pml4() -> u64 {
    BSP_PML4_PHYS.load(Ordering::SeqCst)
}

/// Simple bump allocator for heap virtual addresses.
static VIRTUAL_BUMP: Mutex<u64> = Mutex::new(HEAP_VIRTUAL_BASE);

/// Allocate a range of virtual pages for the heap.
pub fn alloc_virtual_pages(pages: usize, align_pages: usize) -> Option<VirtAddr> {
    let align = align_pages.max(1) as u64 * 4096;
    let mut bump = VIRTUAL_BUMP.lock();
    let addr = (*bump + align - 1) & !(align - 1);
    let size = (pages as u64).checked_mul(4096)?;
    let new_bump = addr.checked_add(size)?;
    if new_bump > 0xFFFF_A000_0000_0000 {
        return None;
    }
    *bump = new_bump;
    Some(VirtAddr::new(addr))
}

/// Initialise virtual memory.
///
/// # Safety
/// Must be called once after the physical allocator is ready.
pub unsafe fn init(kernel_base: PhysAddr, kernel_size: u64) -> Result<(), MapError> {
    let pml4_frame = crate::memory::phys::alloc_frame().ok_or(MapError)?;
    let pml4 = &mut *(pml4_frame.as_u64() as *mut PageTable);
    pml4.zero();

    // Identity-map first 16 GiB with 2 MiB huge pages.
    let id_flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
    for phys in (0..IDENTITY_MAP_SIZE).step_by(HUGE_PAGE_SIZE as usize) {
        map_huge_page_2m(
            VirtAddr::new(phys),
            PhysAddr::new(phys),
            id_flags,
            pml4,
        )?;
    }

    // Map kernel to higher half with 4 KiB pages.
    let kernel_flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
    for offset in (0..kernel_size).step_by(4096) {
        let phys = kernel_base + offset;
        let virt = VirtAddr::new(KERNEL_HIGHER_HALF + offset);
        map_page_inner(virt, phys, kernel_flags, pml4)?;
    }

    // Install the new page-table root.
    let pml4_phys = x86_64::structures::paging::PhysFrame::containing_address(pml4_frame);
    Cr3::write(pml4_phys, Cr3Flags::empty());

    *PML4_VIRT.lock() = VirtAddr::new(pml4 as *mut _ as u64);
    BSP_PML4_PHYS.store(pml4_frame.as_u64(), Ordering::SeqCst);
    Ok(())
}

/// Map a 2 MiB huge page at the PD level.
unsafe fn map_huge_page_2m(
    virt: VirtAddr,
    phys: PhysAddr,
    flags: PageTableFlags,
    pml4: &mut PageTable,
) -> Result<(), MapError> {
    let pml4_idx = ((virt.as_u64() >> 39) & 0x1FF) as usize;
    let pdpt_idx = ((virt.as_u64() >> 30) & 0x1FF) as usize;
    let pd_idx = ((virt.as_u64() >> 21) & 0x1FF) as usize;

    let pdpt = get_or_create_table(&mut pml4[pml4_idx])?;
    let pd = get_or_create_table(&mut pdpt[pdpt_idx])?;
    pd[pd_idx].set_addr(phys, flags | PageTableFlags::HUGE_PAGE);
    Ok(())
}

/// Internal `map_page` that works on an arbitrary PML4.
unsafe fn map_page_inner(virt: VirtAddr, phys: PhysAddr, flags: PageTableFlags, pml4: &mut PageTable) -> Result<(), MapError> {
    let pml4_idx = ((virt.as_u64() >> 39) & 0x1FF) as usize;
    let pdpt_idx = ((virt.as_u64() >> 30) & 0x1FF) as usize;
    let pd_idx = ((virt.as_u64() >> 21) & 0x1FF) as usize;
    let pt_idx = ((virt.as_u64() >> 12) & 0x1FF) as usize;

    let pdpt = get_or_create_table(&mut pml4[pml4_idx])?;
    let pd = get_or_create_table(&mut pdpt[pdpt_idx])?;
    let pt = get_or_create_table(&mut pd[pd_idx])?;
    pt[pt_idx].set_addr(phys, flags);
    Ok(())
}

/// Return the virtual address at which the kernel PML4 is mapped.
pub fn current_pml4_virt() -> VirtAddr {
    *PML4_VIRT.lock()
}

/// Map a 4 KiB page into an arbitrary PML4.
///
/// # Safety
/// `pml4` must point to a valid, active or soon-to-be-active page table.
pub unsafe fn map_page_to_pml4(virt: VirtAddr, phys: PhysAddr, flags: PageTableFlags, pml4: &mut PageTable) -> Result<(), MapError> {
    map_page_inner(virt, phys, flags, pml4)
}

/// Map a 4 KiB page into the current address space.
///
/// # Safety
/// Caller must ensure the mapping does not violate invariants.
pub unsafe fn map_page(virt: VirtAddr, phys: PhysAddr, flags: PageTableFlags) -> Result<(), MapError> {
    let mut ptr_opt = PML4_VIRT.lock();
    let pml4 = unsafe { &mut *(ptr_opt.as_u64() as *mut PageTable) };
    map_page_inner(virt, phys, flags, pml4)
}

/// Unmap a 4 KiB page and return the previously mapped physical address.
pub unsafe fn unmap_page(virt: VirtAddr) -> Option<PhysAddr> {
    let mut ptr_opt = PML4_VIRT.lock();
    let pml4 = unsafe { &mut *(ptr_opt.as_u64() as *mut PageTable) };

    let pml4_idx = ((virt.as_u64() >> 39) & 0x1FF) as usize;
    let pdpt_idx = ((virt.as_u64() >> 30) & 0x1FF) as usize;
    let pd_idx = ((virt.as_u64() >> 21) & 0x1FF) as usize;
    let pt_idx = ((virt.as_u64() >> 12) & 0x1FF) as usize;

    let pdpt_entry = &pml4[pml4_idx];
    if pdpt_entry.is_unused() {
        return None;
    }
    let pdpt = &mut *(pdpt_entry.addr().as_u64() as *mut PageTable);

    let pd_entry = &pdpt[pdpt_idx];
    if pd_entry.is_unused() {
        return None;
    }
    let pd = &mut *(pd_entry.addr().as_u64() as *mut PageTable);

    let pt_entry = &pd[pd_idx];
    if pt_entry.is_unused() {
        return None;
    }
    let pt = &mut *(pt_entry.addr().as_u64() as *mut PageTable);

    let entry = &mut pt[pt_idx];
    if entry.is_unused() {
        return None;
    }
    let phys = entry.addr();
    entry.set_unused();

    // TLB shootdown for this page on all CPUs.
    crate::sync::tlb::shootdown(virt);

    Some(phys)
}

/// Walk the page tables to find the physical address mapped at `virt`.
pub fn translate(virt: VirtAddr) -> Option<PhysAddr> {
    let mut ptr_opt = PML4_VIRT.lock();
    let pml4 = unsafe { &mut *(ptr_opt.as_u64() as *mut PageTable) };

    let pml4_idx = ((virt.as_u64() >> 39) & 0x1FF) as usize;
    let pdpt_idx = ((virt.as_u64() >> 30) & 0x1FF) as usize;
    let pd_idx = ((virt.as_u64() >> 21) & 0x1FF) as usize;
    let pt_idx = ((virt.as_u64() >> 12) & 0x1FF) as usize;

    let pdpt_entry = &pml4[pml4_idx];
    if pdpt_entry.is_unused() {
        return None;
    }
    let pdpt = unsafe { &*(pdpt_entry.addr().as_u64() as *const PageTable) };

    let pd_entry = &pdpt[pdpt_idx];
    if pd_entry.is_unused() {
        return None;
    }
    // Check for 1 GiB huge page at PDPT level.
    if pd_entry.flags().contains(PageTableFlags::HUGE_PAGE) {
        return Some(pd_entry.addr() + (virt.as_u64() & 0x3FFF_FFFF));
    }
    let pd = unsafe { &*(pd_entry.addr().as_u64() as *const PageTable) };

    let pt_entry = &pd[pd_idx];
    if pt_entry.is_unused() {
        return None;
    }
    // Check for 2 MiB huge page at PD level.
    if pt_entry.flags().contains(PageTableFlags::HUGE_PAGE) {
        return Some(pt_entry.addr() + (virt.as_u64() & 0x1F_FFFF));
    }
    let pt = unsafe { &*(pt_entry.addr().as_u64() as *const PageTable) };

    let page_entry = &pt[pt_idx];
    if page_entry.is_unused() {
        return None;
    }
    Some(page_entry.addr() + (virt.as_u64() & 0xFFF))
}

/// Helper: allocate a new zeroed page table if `entry` is unused,
/// then return a mutable reference to it (via identity map).
fn get_or_create_table(entry: &mut PageTableEntry) -> Result<&'static mut PageTable, MapError> {
    if entry.is_unused() {
        let frame = crate::memory::phys::alloc_frame().ok_or(MapError)?;
        let table = unsafe { &mut *(frame.as_u64() as *mut PageTable) };
        table.zero();
        entry.set_addr(
            frame,
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE,
        );
        Ok(table)
    } else {
        let addr = entry.addr();
        Ok(unsafe { &mut *(addr.as_u64() as *mut PageTable) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_virtual_pages() {
        // These tests run on the host; the bump allocator is just a counter.
        let base = *VIRTUAL_BUMP.lock();
        let v1 = alloc_virtual_pages(1, 1).unwrap();
        assert_eq!(v1.as_u64(), base);
        let v2 = alloc_virtual_pages(4, 2).unwrap();
        assert!(v2.as_u64() >= base + 4096);
    }
}
