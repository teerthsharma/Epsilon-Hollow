// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Kernel heap allocator (static 4MB bump allocator for Phase 1).

use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use spin::Mutex;

const HEAP_SIZE: usize = 4 * 1024 * 1024; // 4 MB

#[repr(C, align(4096))]
struct HeapStorage([u8; HEAP_SIZE]);

static mut HEAP: HeapStorage = HeapStorage([0; HEAP_SIZE]);

struct BumpAllocator {
    next: usize,
}

impl BumpAllocator {
    const fn new() -> Self {
        Self { next: 0 }
    }

    fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let start = (self.next + layout.align() - 1) & !(layout.align() - 1);
        let end = start + layout.size();
        if end > HEAP_SIZE {
            return null_mut();
        }
        self.next = end;
        unsafe { (core::ptr::addr_of_mut!(HEAP) as *mut u8).add(start) }
    }
}

static ALLOCATOR: Mutex<BumpAllocator> = Mutex::new(BumpAllocator::new());

struct SealAllocator;

unsafe impl GlobalAlloc for SealAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATOR.lock().alloc(layout)
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static GLOBAL: SealAllocator = SealAllocator;

pub fn init_heap() {
    // Static buffer — nothing to init, but this marks the milestone
}
