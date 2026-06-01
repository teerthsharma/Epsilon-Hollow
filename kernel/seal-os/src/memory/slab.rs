// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Slab allocator for kernel objects.
//!
//! Maintains a cache for each power-of-two size from 64 B to 2048 B.
//! Objects are carved out of 4 KiB pages allocated from the physical frame
//! allocator.  Free objects are kept on an intrusive singly-linked list.

use spin::Mutex;

const SLAB_SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048];

struct SlabCache {
    obj_size: usize,
    free_list: Option<*mut u8>,
}

struct SlabAllocator {
    caches: [SlabCache; SLAB_SIZES.len()],
}

impl SlabAllocator {
    const fn new() -> Self {
        Self {
            caches: [
                SlabCache {
                    obj_size: 64,
                    free_list: None,
                },
                SlabCache {
                    obj_size: 128,
                    free_list: None,
                },
                SlabCache {
                    obj_size: 256,
                    free_list: None,
                },
                SlabCache {
                    obj_size: 512,
                    free_list: None,
                },
                SlabCache {
                    obj_size: 1024,
                    free_list: None,
                },
                SlabCache {
                    obj_size: 2048,
                    free_list: None,
                },
            ],
        }
    }

    unsafe fn alloc(&mut self, size: usize) -> Option<*mut u8> {
        let idx = size_to_index(size)?;
        let cache = &mut self.caches[idx];

        if let Some(ptr) = cache.free_list {
            let next = *(ptr as *const *mut u8);
            cache.free_list = if next.is_null() { None } else { Some(next) };
            return Some(ptr);
        }

        let frame = crate::memory::phys::alloc_frame()?;
        let page_start = frame.as_u64() as *mut u8;
        let objs_per_page = 4096 / cache.obj_size;

        // Initialise free list in reverse so the first returned object
        // is at page_start (nice cache behaviour).
        for i in (0..objs_per_page).rev() {
            let obj = page_start.add(i * cache.obj_size);
            *(obj as *mut *mut u8) = cache.free_list.unwrap_or(core::ptr::null_mut());
            cache.free_list = Some(obj);
        }

        self.alloc(size)
    }

    unsafe fn free(&mut self, ptr: *mut u8, size: usize) {
        if let Some(idx) = size_to_index(size) {
            let cache = &mut self.caches[idx];
            *(ptr as *mut *mut u8) = cache.free_list.unwrap_or(core::ptr::null_mut());
            cache.free_list = Some(ptr);
        }
    }
}

unsafe impl Send for SlabAllocator {}
unsafe impl Sync for SlabAllocator {}

fn size_to_index(size: usize) -> Option<usize> {
    match size {
        64 => Some(0),
        128 => Some(1),
        256 => Some(2),
        512 => Some(3),
        1024 => Some(4),
        2048 => Some(5),
        _ => None,
    }
}

static SLAB: Mutex<SlabAllocator> = Mutex::new(SlabAllocator::new());

/// Allocate an object of exactly `size` bytes from the slab.
/// `size` must be one of the supported slab sizes.
///
/// # Safety
/// The returned pointer is uninitialized. The caller must ensure `size` is one
/// of the supported slab sizes (64, 128, 256, 512, 1024, or 2048). Using an
/// unsupported size returns a null pointer; dereferencing it without a check
/// causes undefined behavior.
pub unsafe fn slab_alloc(size: usize) -> *mut u8 {
    let result = SLAB.lock().alloc(size);
    if result.is_none() {
        crate::serial_println!("[DEBUG] slab_alloc({}) failed", size);
    }
    result.unwrap_or(core::ptr::null_mut())
}

/// Free an object previously allocated with `slab_alloc`.
///
/// # Safety
/// `ptr` must be a non-null pointer previously returned by `slab_alloc` with
/// the same `size`. Passing a mismatched size, a null pointer, or a pointer
/// not owned by the slab allocator corrupts the free list and causes heap
/// memory corruption or use-after-free.
pub unsafe fn slab_free(ptr: *mut u8, size: usize) {
    if !ptr.is_null() {
        SLAB.lock().free(ptr, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_to_index() {
        assert_eq!(size_to_index(64), Some(0));
        assert_eq!(size_to_index(128), Some(1));
        assert_eq!(size_to_index(256), Some(2));
        assert_eq!(size_to_index(512), Some(3));
        assert_eq!(size_to_index(1024), Some(4));
        assert_eq!(size_to_index(2048), Some(5));
        assert_eq!(size_to_index(100), None);
    }
}
