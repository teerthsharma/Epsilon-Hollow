// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Memory subsystem — physical allocator, virtual memory manager, slab allocator,
//! and global heap allocator.

use x86_64::PhysAddr;

use crate::boot::boot_info::BootInfo;

pub mod gdt;
pub mod heap;
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

    virt::init(kernel_start, boot_info.kernel_size);
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

#[cfg(feature = "test-mode")]
pub mod tests {
    pub fn register_all() {
        // Memory tests will be registered here once implemented
    }
}
