// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Global heap allocator.
//!
//! Routes small allocations (≤ 2048 B) to the slab allocator and large
//! allocations to the page allocator + virtual-memory mapper.

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering};

use x86_64::{PhysAddr, VirtAddr, structures::paging::PageTableFlags};

use super::{slab, virt};

static TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

pub struct SealAllocator;

unsafe impl GlobalAlloc for SealAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let req_size = layout.size().max(layout.align());

        if req_size <= 2048 {
            let slab_size = req_size.next_power_of_two().max(64);
            let ptr = slab::slab_alloc(slab_size);
            if !ptr.is_null() {
                TOTAL_ALLOCATED.fetch_add(slab_size, Ordering::Relaxed);
            }
            ptr
        } else {
            let pages = (req_size + 4095) / 4096;
            let align_pages = (layout.align() + 4095) / 4096;
            let total_pages = pages.max(align_pages);

            let virt = virt::alloc_virtual_pages(total_pages, align_pages);
            if virt.is_none() {
                return core::ptr::null_mut();
            }
            let base = virt.unwrap();

            for i in 0..total_pages {
                let page_virt = VirtAddr::new(base.as_u64() + i as u64 * 4096);
                match crate::memory::phys::alloc_frame() {
                    Some(frame) => {
                        let flags = PageTableFlags::PRESENT
                            | PageTableFlags::WRITABLE
                            | PageTableFlags::NO_EXECUTE;
                        if virt::map_page(page_virt, frame, flags).is_err() {
                            // Page-table allocation failed — free this frame and undo prior mappings.
                            crate::memory::phys::free_frame(frame);
                            for j in 0..i {
                                let v = VirtAddr::new(base.as_u64() + j as u64 * 4096);
                                if let Some(f) = virt::unmap_page(v) {
                                    crate::memory::phys::free_frame(f);
                                }
                            }
                            return core::ptr::null_mut();
                        }
                    }
                    None => {
                        // OOM — unmap and free what we already acquired.
                        for j in 0..i {
                            let v = VirtAddr::new(base.as_u64() + j as u64 * 4096);
                            if let Some(frame) = virt::unmap_page(v) {
                                crate::memory::phys::free_frame(frame);
                            }
                        }
                        return core::ptr::null_mut();
                    }
                }
            }

            TOTAL_ALLOCATED.fetch_add(total_pages * 4096, Ordering::Relaxed);
            base.as_mut_ptr()
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ptr.is_null() {
            return;
        }

        let req_size = layout.size().max(layout.align());

        if req_size <= 2048 {
            let slab_size = req_size.next_power_of_two().max(64);
            slab::slab_free(ptr, slab_size);
            TOTAL_ALLOCATED.fetch_sub(slab_size, Ordering::Relaxed);
        } else {
            let pages = (req_size + 4095) / 4096;
            let align_pages = (layout.align() + 4095) / 4096;
            let total_pages = pages.max(align_pages);

            let base = VirtAddr::new(ptr as u64);
            for i in 0..total_pages {
                let v = VirtAddr::new(base.as_u64() + i as u64 * 4096);
                if let Some(frame) = virt::unmap_page(v) {
                    crate::memory::phys::free_frame(frame);
                }
            }

            TOTAL_ALLOCATED.fetch_sub(total_pages * 4096, Ordering::Relaxed);
        }
    }
}

/// Return the number of bytes currently allocated through the global allocator.
pub fn total_allocated() -> usize {
    TOTAL_ALLOCATED.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_allocated_starts_at_zero() {
        assert_eq!(total_allocated(), 0);
    }
}
