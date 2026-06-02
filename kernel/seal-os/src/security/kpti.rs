// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Kernel Page-Table Isolation (KPTI) — real CR3 swap via pgtable-asm.
//!
//! Captures the kernel CR3 at boot and provides fast CR3 switching for
//! syscall entry/exit and interrupt dispatch.  The user shadow page table
//! is still a TODO (requires per-CPU allocation); the swap primitive itself
//! is real x86_64 assembly.

use core::sync::atomic::{AtomicU64, Ordering};
use x86_64::registers::control::Cr3;

static KERNEL_CR3: AtomicU64 = AtomicU64::new(0);
static USER_CR3: AtomicU64 = AtomicU64::new(0);

/// Initialize KPTI state.
///
/// Captures the currently active kernel CR3 so that `switch_to_kernel_pt()`
/// can restore it later.
pub fn init() {
    let (frame, _) = Cr3::read();
    let phys = frame.start_address().as_u64();
    KERNEL_CR3.store(phys, Ordering::SeqCst);
    crate::serial_println!(
        "[KPTI] Kernel Page-Table Isolation initialized — kernel CR3 = {:#x}",
        phys
    );
    init_trampoline_pt();
}

/// Switch from the current page table to the kernel page table.
///
/// This is called on syscall entry and when handling interrupts from
/// userspace.  If the kernel CR3 has not been captured, this is a no-op.
pub fn switch_to_kernel_pt() {
    let cr3 = KERNEL_CR3.load(Ordering::SeqCst);
    if cr3 != 0 {
        unsafe {
            crate::memory::pgtable_asm::swap_cr3(cr3);
        }
    }
}

/// Switch from the kernel page table to the user shadow page table.
///
/// Called on syscall exit (`sysret`) just before returning to ring 3.
/// If no user shadow PT has been installed, this logs a warning and
/// does not swap (the user continues running with kernel mappings).
pub fn switch_to_user_pt() {
    let cr3 = USER_CR3.load(Ordering::SeqCst);
    if cr3 != 0 {
        unsafe {
            crate::memory::pgtable_asm::swap_cr3(cr3);
        }
    } else {
        crate::serial_println!("[KPTI] No user shadow PT installed — skipping user CR3 swap");
    }
}

/// Install a user shadow page-table root.
///
/// # Safety
/// `pt_phys` must be the physical address of a valid page table that maps
/// at least the trampoline, user memory, and minimal kernel entry stubs.
pub unsafe fn set_user_cr3(pt_phys: u64) {
    USER_CR3.store(pt_phys, Ordering::SeqCst);
}

/// Allocate and install a trampoline page table that copies kernel PML4
/// entries (upper half) but leaves the lower half unmapped.
pub fn init_trampoline_pt() {
    // Allocate a PML4 frame
    let frame = match crate::memory::phys::alloc_frame() {
        Some(f) => f,
        None => {
            crate::serial_println!("[KPTI] Failed to allocate trampoline PML4");
            return;
        }
    };
    // Physical memory below 4 GiB is identity-mapped by the boot page tables.
    let pml4 = frame.as_u64() as *mut u64;

    // Zero it
    unsafe {
        core::ptr::write_bytes(pml4, 0, 4096);
    }

    // Copy kernel PML4 entries (upper half) — entry 256+ map kernel
    let (kernel_frame, _) = Cr3::read();
    let kernel_pml4 = kernel_frame.start_address().as_u64();
    let kernel_virt = kernel_pml4 as *mut u64;
    for i in 256..512 {
        unsafe {
            let entry = *kernel_virt.add(i);
            *pml4.add(i) = entry;
        }
    }

    // Set as user CR3
    unsafe {
        set_user_cr3(frame.as_u64());
    }
    crate::serial_println!(
        "[KPTI] Trampoline page table installed @ {:#x}",
        frame.as_u64()
    );
}

/// Return the captured kernel CR3 physical address.
pub fn kernel_cr3() -> u64 {
    KERNEL_CR3.load(Ordering::SeqCst)
}

/// Return the installed user shadow CR3 physical address.
pub fn user_cr3() -> u64 {
    USER_CR3.load(Ordering::SeqCst)
}

/// Return true if both kernel and user CR3 values have been initialized.
pub fn has_kpti() -> bool {
    kernel_cr3() != 0 && user_cr3() != 0
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use crate::testing::TestResult;

    pub fn register_all() {
        // KPTI tests require a full page-table setup; tested via integration.
    }
}
