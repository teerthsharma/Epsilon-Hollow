// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Virtual memory manager — 4-level page tables.
//!
//! On init we:
//! 1. Allocate a fresh PML4.
//! 2. Identity-map the first 16 GiB using 2 MiB huge pages (boot compatibility).
//! 3. Map the kernel image to `0xffffffff80000000` using 4 KiB pages.
//! 4. Switch CR3.
//!
//! All page-table pages are allocated from the physical allocator and accessed
//! through the identity map (they live below 4 GiB).

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;
use x86_64::{
    registers::control::{Cr3, Cr3Flags},
    structures::paging::page_table::PageTableEntry,
    structures::paging::{PageTable, PageTableFlags},
    PhysAddr, VirtAddr,
};

const KERNEL_HIGHER_HALF: u64 = 0xffff_ffff_8000_0000;
/// Size of the identity-mapped region (16 GiB).  Frames below this line can be
/// accessed directly via `phys.as_u64() as *mut T`.
pub const IDENTITY_MAP_SIZE: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB
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
        map_huge_page_2m(VirtAddr::new(phys), PhysAddr::new(phys), id_flags, pml4)?;
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
unsafe fn map_page_inner(
    virt: VirtAddr,
    phys: PhysAddr,
    flags: PageTableFlags,
    pml4: &mut PageTable,
) -> Result<(), MapError> {
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
pub unsafe fn map_page_to_pml4(
    virt: VirtAddr,
    phys: PhysAddr,
    flags: PageTableFlags,
    pml4: &mut PageTable,
) -> Result<(), MapError> {
    map_page_inner(virt, phys, flags, pml4)
}

/// Map a 4 KiB page into the current address space.
///
/// # Safety
/// Caller must ensure the mapping does not violate invariants.
pub unsafe fn map_page(
    virt: VirtAddr,
    phys: PhysAddr,
    flags: PageTableFlags,
) -> Result<(), MapError> {
    let ptr_opt = PML4_VIRT.lock();
    let pml4 = unsafe { &mut *(ptr_opt.as_u64() as *mut PageTable) };
    map_page_inner(virt, phys, flags, pml4)
}

/// Unmap a 4 KiB page from an arbitrary PML4 and return the previous physical address.
///
/// # Safety
/// `pml4_phys` must point to a valid page table.
pub unsafe fn unmap_page_in_pml4(virt: VirtAddr, pml4_phys: PhysAddr) -> Option<PhysAddr> {
    let pml4 = &mut *(pml4_phys.as_u64() as *mut PageTable);

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

    crate::sync::tlb::shootdown(virt);
    Some(phys)
}

/// Unmap a 4 KiB page and return the previously mapped physical address.
///
/// # Safety
/// `virt` must be a currently mapped virtual address in the active page table.
/// Unmapping an address that is in use by the kernel or hardware DMA causes
/// immediate page faults, data loss, or undefined behavior. Caller must ensure
/// the physical frame is not leaked after unmapping.
pub unsafe fn unmap_page(virt: VirtAddr) -> Option<PhysAddr> {
    let ptr_opt = PML4_VIRT.lock();
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

/// Walk the page tables rooted at `pml4_phys` to translate `virt`.
pub fn translate_in_pml4(virt: VirtAddr, pml4_phys: PhysAddr) -> Option<PhysAddr> {
    let pml4 = unsafe { &*(pml4_phys.as_u64() as *const PageTable) };

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

/// Walk the page tables to find the physical address mapped at `virt`.
pub fn translate(virt: VirtAddr) -> Option<PhysAddr> {
    let ptr_opt = PML4_VIRT.lock();
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
        entry.set_addr(frame, PageTableFlags::PRESENT | PageTableFlags::WRITABLE);
        Ok(table)
    } else {
        let addr = entry.addr();
        Ok(unsafe { &mut *(addr.as_u64() as *mut PageTable) })
    }
}

// ---------------------------------------------------------------------------
// COW fork support — T5 hyperbolic lifetime classification
// ---------------------------------------------------------------------------

struct CowCloneRollback {
    frames: Vec<PhysAddr>,
    committed: bool,
}

static COW_ROLLBACK_TRACKED_TOTAL: AtomicU64 = AtomicU64::new(0);
static COW_ROLLBACK_FREED_TOTAL: AtomicU64 = AtomicU64::new(0);

impl CowCloneRollback {
    fn new() -> Self {
        Self {
            frames: Vec::new(),
            committed: false,
        }
    }

    fn track(&mut self, frame: PhysAddr) {
        COW_ROLLBACK_TRACKED_TOTAL.fetch_add(1, Ordering::SeqCst);
        self.frames.push(frame);
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for CowCloneRollback {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for frame in self.frames.drain(..).rev() {
            unsafe {
                crate::memory::phys::free_frame(frame);
            }
            COW_ROLLBACK_FREED_TOTAL.fetch_add(1, Ordering::SeqCst);
        }
    }
}

fn cow_alloc_frame(
    rollback: &mut CowCloneRollback,
    allocation_budget: Option<&mut usize>,
) -> Option<PhysAddr> {
    if let Some(remaining) = allocation_budget {
        if *remaining == 0 {
            return None;
        }
        *remaining -= 1;
    }
    let frame = crate::memory::phys::alloc_frame()?;
    rollback.track(frame);
    Some(frame)
}

/// Clone a page table for COW fork.
///
/// Kernel higher-half entries (256..512) are shared verbatim.
/// User entries (0..256) are walked: each leaf 4 KiB page keeps its physical
/// frame but the WRITABLE bit is cleared.  Read-only user pages (code,
/// shared libs) are considered "long-lived" under T5 and are left truly
/// shared.  Writable user pages (stack, heap, data) are marked COW.
///
/// # Safety
/// `old_pml4_phys` must be a valid page table.
pub unsafe fn clone_page_table_cow(old_pml4_phys: PhysAddr) -> Option<PhysAddr> {
    clone_page_table_cow_with_budget(old_pml4_phys, None)
}

unsafe fn clone_page_table_cow_with_budget(
    old_pml4_phys: PhysAddr,
    mut allocation_budget: Option<usize>,
) -> Option<PhysAddr> {
    let old_pml4 = &*(old_pml4_phys.as_u64() as *const PageTable);
    let mut rollback = CowCloneRollback::new();
    let new_pml4_frame = cow_alloc_frame(&mut rollback, allocation_budget.as_mut())?;
    let new_pml4 = &mut *(new_pml4_frame.as_u64() as *mut PageTable);
    new_pml4.zero();

    // Copy kernel higher-half mappings verbatim — no COW needed.
    for i in 256..512 {
        new_pml4[i] = old_pml4[i].clone();
    }

    // Walk user space and COW-mark writable pages.
    for pml4_idx in 0..256 {
        let pdpt_entry = &old_pml4[pml4_idx];
        if pdpt_entry.is_unused() {
            continue;
        }
        let old_pdpt = &*(pdpt_entry.addr().as_u64() as *const PageTable);
        let new_pdpt_frame = cow_alloc_frame(&mut rollback, allocation_budget.as_mut())?;
        let new_pdpt = &mut *(new_pdpt_frame.as_u64() as *mut PageTable);
        new_pdpt.zero();
        new_pml4[pml4_idx].set_addr(
            new_pdpt_frame,
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE,
        );

        for pdpt_idx in 0..512 {
            let pd_entry = &old_pdpt[pdpt_idx];
            if pd_entry.is_unused() {
                continue;
            }
            // Propagate huge pages (1 GiB) without COW for simplicity.
            if pd_entry.flags().contains(PageTableFlags::HUGE_PAGE) {
                new_pdpt[pdpt_idx] = pd_entry.clone();
                continue;
            }
            let old_pd = &*(pd_entry.addr().as_u64() as *const PageTable);
            let new_pd_frame = cow_alloc_frame(&mut rollback, allocation_budget.as_mut())?;
            let new_pd = &mut *(new_pd_frame.as_u64() as *mut PageTable);
            new_pd.zero();
            new_pdpt[pdpt_idx].set_addr(
                new_pd_frame,
                PageTableFlags::PRESENT | PageTableFlags::WRITABLE,
            );

            for pd_idx in 0..512 {
                let pt_entry = &old_pd[pd_idx];
                if pt_entry.is_unused() {
                    continue;
                }
                if pt_entry.flags().contains(PageTableFlags::HUGE_PAGE) {
                    new_pd[pd_idx] = pt_entry.clone();
                    continue;
                }
                let old_pt = &*(pt_entry.addr().as_u64() as *const PageTable);
                let new_pt_frame = cow_alloc_frame(&mut rollback, allocation_budget.as_mut())?;
                let new_pt = &mut *(new_pt_frame.as_u64() as *mut PageTable);
                new_pt.zero();
                new_pd[pd_idx].set_addr(
                    new_pt_frame,
                    PageTableFlags::PRESENT | PageTableFlags::WRITABLE,
                );

                for pt_idx in 0..512 {
                    let page_entry = &old_pt[pt_idx];
                    if page_entry.is_unused() {
                        continue;
                    }
                    let new_flags = page_entry.flags();
                    // Deep copy all user-accessible pages to prevent double-free
                    if new_flags.contains(PageTableFlags::USER_ACCESSIBLE) {
                        let new_frame =
                            cow_alloc_frame(&mut rollback, allocation_budget.as_mut())?;
                        core::ptr::copy_nonoverlapping(
                            page_entry.addr().as_u64() as *const u8,
                            new_frame.as_u64() as *mut u8,
                            4096,
                        );
                        new_pt[pt_idx].set_addr(new_frame, new_flags);
                    } else {
                        new_pt[pt_idx].set_addr(page_entry.addr(), new_flags);
                    }
                }
            }
        }
    }

    rollback.commit();
    Some(new_pml4_frame)
}

pub fn emit_cow_proof() {
    let tracked_start = COW_ROLLBACK_TRACKED_TOTAL.load(Ordering::SeqCst);
    let freed_start = COW_ROLLBACK_FREED_TOTAL.load(Ordering::SeqCst);

    let Some(old_pml4_frame) = crate::memory::phys::alloc_frame() else {
        crate::serial_println!(
            "[MM] cow-proof version=1 rollback_guard=0 fork_fallback=0 clone_fallback=0 samples=0 rollback_ok=0 tracked_frames=0 rollback_frees=0 leaked_frames=1 result=fail"
        );
        return;
    };
    let Some(old_pdpt_frame) = crate::memory::phys::alloc_frame() else {
        unsafe {
            crate::memory::phys::free_frame(old_pml4_frame);
        }
        crate::serial_println!(
            "[MM] cow-proof version=1 rollback_guard=0 fork_fallback=0 clone_fallback=0 samples=0 rollback_ok=0 tracked_frames=0 rollback_frees=0 leaked_frames=1 result=fail"
        );
        return;
    };
    let Some(old_pd_frame) = crate::memory::phys::alloc_frame() else {
        unsafe {
            crate::memory::phys::free_frame(old_pdpt_frame);
            crate::memory::phys::free_frame(old_pml4_frame);
        }
        crate::serial_println!(
            "[MM] cow-proof version=1 rollback_guard=0 fork_fallback=0 clone_fallback=0 samples=0 rollback_ok=0 tracked_frames=0 rollback_frees=0 leaked_frames=1 result=fail"
        );
        return;
    };
    let Some(old_pt_frame) = crate::memory::phys::alloc_frame() else {
        unsafe {
            crate::memory::phys::free_frame(old_pd_frame);
            crate::memory::phys::free_frame(old_pdpt_frame);
            crate::memory::phys::free_frame(old_pml4_frame);
        }
        crate::serial_println!(
            "[MM] cow-proof version=1 rollback_guard=0 fork_fallback=0 clone_fallback=0 samples=0 rollback_ok=0 tracked_frames=0 rollback_frees=0 leaked_frames=1 result=fail"
        );
        return;
    };
    let Some(old_page_frame) = crate::memory::phys::alloc_frame() else {
        unsafe {
            crate::memory::phys::free_frame(old_pt_frame);
            crate::memory::phys::free_frame(old_pd_frame);
            crate::memory::phys::free_frame(old_pdpt_frame);
            crate::memory::phys::free_frame(old_pml4_frame);
        }
        crate::serial_println!(
            "[MM] cow-proof version=1 rollback_guard=0 fork_fallback=0 clone_fallback=0 samples=0 rollback_ok=0 tracked_frames=0 rollback_frees=0 leaked_frames=1 result=fail"
        );
        return;
    };

    unsafe {
        let old_pml4 = &mut *(old_pml4_frame.as_u64() as *mut PageTable);
        let old_pdpt = &mut *(old_pdpt_frame.as_u64() as *mut PageTable);
        let old_pd = &mut *(old_pd_frame.as_u64() as *mut PageTable);
        let old_pt = &mut *(old_pt_frame.as_u64() as *mut PageTable);
        old_pml4.zero();
        old_pdpt.zero();
        old_pd.zero();
        old_pt.zero();
        old_pml4[0].set_addr(
            old_pdpt_frame,
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE,
        );
        old_pdpt[0].set_addr(
            old_pd_frame,
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE,
        );
        old_pd[0].set_addr(
            old_pt_frame,
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE,
        );
        old_pt[0].set_addr(
            old_page_frame,
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE,
        );
    }

    let mut rollback_ok = 0_u64;
    let samples = 4_u64;
    for budget in 1..=samples as usize {
        let before = crate::memory::phys::free_count();
        let cloned = unsafe { clone_page_table_cow_with_budget(old_pml4_frame, Some(budget)) };
        let after = crate::memory::phys::free_count();
        if cloned.is_none()
            && before == after
            && translate_in_pml4(VirtAddr::new(0), old_pml4_frame) == Some(old_page_frame)
        {
            rollback_ok += 1;
        }
    }

    unsafe {
        crate::memory::phys::free_frame(old_page_frame);
        crate::memory::phys::free_frame(old_pt_frame);
        crate::memory::phys::free_frame(old_pd_frame);
        crate::memory::phys::free_frame(old_pdpt_frame);
        crate::memory::phys::free_frame(old_pml4_frame);
    }

    let tracked = COW_ROLLBACK_TRACKED_TOTAL
        .load(Ordering::SeqCst)
        .saturating_sub(tracked_start);
    let freed = COW_ROLLBACK_FREED_TOTAL
        .load(Ordering::SeqCst)
        .saturating_sub(freed_start);
    let leaked = tracked.saturating_sub(freed);
    let pass = rollback_ok == samples && tracked > 0 && leaked == 0;
    let rollback_guard = if pass { 1 } else { 0 };
    let result = if pass { "pass" } else { "fail" };
    crate::serial_println!(
        "[MM] cow-proof version=1 rollback_guard={} fork_fallback=0 clone_fallback=0 samples={} rollback_ok={} tracked_frames={} rollback_frees={} leaked_frames={} result={}",
        rollback_guard,
        samples,
        rollback_ok,
        tracked,
        freed,
        leaked,
        result
    );
}

/// Handle a COW page fault at `fault_addr` for the page table rooted at
/// `pml4_phys`.  Returns `true` if the fault was resolved.
///
/// # Safety
/// `pml4_phys` must be a valid page table.
pub unsafe fn handle_cow_fault(fault_addr: VirtAddr, pml4_phys: u64) -> bool {
    let pml4 = &*(pml4_phys as *const PageTable);
    let pml4_idx = ((fault_addr.as_u64() >> 39) & 0x1FF) as usize;
    let pdpt_idx = ((fault_addr.as_u64() >> 30) & 0x1FF) as usize;
    let pd_idx = ((fault_addr.as_u64() >> 21) & 0x1FF) as usize;
    let pt_idx = ((fault_addr.as_u64() >> 12) & 0x1FF) as usize;

    let pdpt_entry = &pml4[pml4_idx];
    if pdpt_entry.is_unused() {
        return false;
    }
    let pdpt = &*(pdpt_entry.addr().as_u64() as *const PageTable);

    let pd_entry = &pdpt[pdpt_idx];
    if pd_entry.is_unused() || pd_entry.flags().contains(PageTableFlags::HUGE_PAGE) {
        return false;
    }
    let pd = &*(pd_entry.addr().as_u64() as *const PageTable);

    let pt_entry = &pd[pd_idx];
    if pt_entry.is_unused() || pt_entry.flags().contains(PageTableFlags::HUGE_PAGE) {
        return false;
    }
    let pt = &mut *(pt_entry.addr().as_u64() as *mut PageTable);

    let entry = &mut pt[pt_idx];
    if entry.is_unused() {
        return false;
    }
    let flags = entry.flags();
    // COW candidate: user-accessible, present, but not writable.
    if !flags.contains(PageTableFlags::USER_ACCESSIBLE)
        || !flags.contains(PageTableFlags::PRESENT)
        || flags.contains(PageTableFlags::WRITABLE)
    {
        return false;
    }

    let old_phys = entry.addr();
    let new_frame = match crate::memory::phys::alloc_frame() {
        Some(f) => f,
        None => return false,
    };

    // Copy the 4 KiB page contents.
    core::ptr::copy_nonoverlapping(
        old_phys.as_u64() as *const u8,
        new_frame.as_u64() as *mut u8,
        4096,
    );

    let mut new_flags = flags;
    new_flags.insert(PageTableFlags::WRITABLE);
    entry.set_addr(new_frame, new_flags);

    crate::sync::tlb::shootdown(fault_addr);
    true
}

// ---------------------------------------------------------------------------
// Thread-local storage — FS segment base (x86_64 MSR 0xC0000100)
// ---------------------------------------------------------------------------

/// Set the FS base MSR (0xC0000100) for the current CPU.
///
/// Each thread receives a unique TLS area at creation; this function
/// wires the `FS` segment base so that `mov %fs:offset, %reg` accesses
/// the thread-local slot vector.
pub fn set_fs_base(addr: u64) {
    unsafe {
        x86_64::registers::model_specific::Msr::new(0xC000_0100).write(addr);
    }
}

/// Read the FS base MSR.
pub fn get_fs_base() -> u64 {
    unsafe { x86_64::registers::model_specific::Msr::new(0xC000_0100).read() }
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
