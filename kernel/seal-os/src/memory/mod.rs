// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Memory subsystem — physical allocator, virtual memory manager, slab allocator,
//! and global heap allocator.

use x86_64::PhysAddr;

use crate::boot::boot_info::BootInfo;

pub mod gdt;
pub mod heap;
pub mod mmap;
pub mod pgtable_asm;
pub mod phys;
pub mod slab;
pub mod virt;

pub use heap::SealAllocator;

#[global_allocator]
static GLOBAL: SealAllocator = SealAllocator;

/// One-time initialisation of the entire memory subsystem.
///
/// # Safety
/// Must be called exactly once, early in kernel boot, with valid `BootInfo`.
pub unsafe fn init(boot_info: &BootInfo) {
    let kernel_start = PhysAddr::new(boot_info.kernel_base);
    let kernel_end = PhysAddr::new(boot_info.kernel_base + boot_info.kernel_size);

    let fb_start = PhysAddr::new(boot_info.fb_addr);
    let fb_size = boot_info.fb_pitch as u64 * boot_info.fb_height as u64;
    let fb_end = PhysAddr::new(boot_info.fb_addr + fb_size);

    phys::init(
        &boot_info.memory_map[..boot_info.memory_map_len],
        kernel_start,
        kernel_end,
        fb_start,
        fb_end,
    );

    let free_after_phys = phys::free_count();
    crate::serial_println!("[BOOT] phys::init complete, free frames = {}", free_after_phys);

    if let Err(_) = virt::init(kernel_start, boot_info.kernel_size) {
        crate::serial_println!("[BOOT] FATAL: Failed to initialize virtual memory (out of memory)");
        loop {
            x86_64::instructions::hlt();
        }
    }
    let free_after_virt = phys::free_count();
    crate::serial_println!("[BOOT] virt::init complete, free frames = {}", free_after_virt);
    gdt::init_gdt();
}

/// Return heap diagnostics `(allocated_bytes, total_reserved_bytes)`.
///
/// `total_reserved_bytes` is currently a best-effort estimate based on the
/// free-frame count plus allocated bytes.
pub fn heap_stats() -> (usize, usize) {
    let allocated = heap::total_allocated();
    let free_frames = phys::free_count();
    // total = allocated + free frames * 4096 (approximate)
    let total = allocated + free_frames * 4096;
    (allocated, total)
}

/// Compute total conventional RAM from the UEFI memory map.
pub fn total_ram(boot_info: &BootInfo) -> u64 {
    let mut total = 0u64;
    for i in 0..boot_info.memory_map_len {
        let desc = &boot_info.memory_map[i];
        // UEFI MemoryType::CONVENTIONAL == 7
        if desc.ty == 7 {
            total += desc.page_count * 4096;
        }
    }
    total
}

#[cfg(feature = "test-mode")]
pub mod tests {
    pub fn register_all() {
        // Memory tests will be registered here once implemented
    }
}
