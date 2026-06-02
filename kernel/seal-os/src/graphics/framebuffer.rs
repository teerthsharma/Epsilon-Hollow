// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

pub struct Framebuffer {
    buffer: *mut u8,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub bpp: u8,
    back_buffer: AtomicPtr<u32>,
    back_buffer_len: AtomicUsize,
}

// SAFETY: The framebuffer pointer comes from UEFI firmware and remains valid for
// the kernel's lifetime. All pixel writes use volatile ops and bounds-check x/y.
unsafe impl Send for Framebuffer {}
unsafe impl Sync for Framebuffer {}

impl Framebuffer {
    pub const fn null() -> Self {
        Self {
            buffer: ptr::null_mut(),
            width: 0,
            height: 0,
            pitch: 0,
            bpp: 0,
            back_buffer: AtomicPtr::new(ptr::null_mut()),
            back_buffer_len: AtomicUsize::new(0),
        }
    }

    pub unsafe fn new(addr: u64, width: u32, height: u32, pitch: u32, bpp: u8) -> Self {
        Self {
            buffer: addr as *mut u8,
            width,
            height,
            pitch,
            bpp,
            back_buffer: AtomicPtr::new(ptr::null_mut()),
            back_buffer_len: AtomicUsize::new(0),
        }
    }

    pub fn is_available(&self) -> bool {
        !self.buffer.is_null()
    }

    pub fn is_ready(&self) -> bool {
        if !self.is_available() {
            return false;
        }
        let test_x = self.width / 2;
        let test_y = self.height / 2;
        let test_color = 0xDEADBEEF;
        self.put_pixel(test_x, test_y, test_color);
        let read_back = self.get_pixel(test_x, test_y);
        self.put_pixel(test_x, test_y, 0);
        read_back == test_color
    }

    /// Allocate the back buffer. Safe to call multiple times; subsequent calls
    /// are ignored. Must only be called after the heap is initialised.
    pub fn init_back_buffer(&self) {
        if !self.back_buffer.load(Ordering::Relaxed).is_null() {
            return;
        }
        if self.width == 0 || self.height == 0 {
            return;
        }
        let count = (self.width as usize).saturating_mul(self.height as usize);
        let mut v = alloc::vec![0u32; count];
        let ptr = v.as_mut_ptr();
        self.back_buffer.store(ptr, Ordering::Relaxed);
        self.back_buffer_len.store(count, Ordering::Relaxed);
        core::mem::forget(v);
    }

    pub fn has_back_buffer(&self) -> bool {
        !self.back_buffer.load(Ordering::Relaxed).is_null()
    }

    pub fn back_buffer_ptr(&self) -> *mut u32 {
        self.back_buffer.load(Ordering::Relaxed)
    }

    pub fn put_pixel(&self, x: u32, y: u32, color: u32) {
        if x >= self.width || y >= self.height || self.buffer.is_null() {
            return;
        }
        let bb = self.back_buffer.load(Ordering::Relaxed);
        if !bb.is_null() {
            let idx = (y * self.width + x) as usize;
            let len = self.back_buffer_len.load(Ordering::Relaxed);
            if idx < len {
                unsafe {
                    ptr::write_volatile(bb.add(idx), color);
                }
            }
            return;
        }
        let bytes_per_pixel = (self.bpp as u32) / 8;
        let offset = (y * self.pitch + x * bytes_per_pixel) as isize;
        unsafe {
            let pixel = self.buffer.offset(offset) as *mut u32;
            ptr::write_volatile(pixel, color);
        }
    }

    pub fn fill_rect(&self, x: u32, y: u32, w: u32, h: u32, color: u32) {
        let x_end = (x + w).min(self.width);
        let y_end = (y + h).min(self.height);
        for row in y..y_end {
            for col in x..x_end {
                self.put_pixel(col, row, color);
            }
        }
    }

    pub fn clear(&self, color: u32) {
        self.fill_rect(0, 0, self.width, self.height, color);
    }

    pub fn clear_back(&self, color: u32) {
        let bb = self.back_buffer.load(Ordering::Relaxed);
        if bb.is_null() {
            return;
        }
        let len = self.back_buffer_len.load(Ordering::Relaxed);
        for i in 0..len {
            unsafe {
                ptr::write_volatile(bb.add(i), color);
            }
        }
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> u32 {
        if x >= self.width || y >= self.height || self.buffer.is_null() {
            return 0;
        }
        let bb = self.back_buffer.load(Ordering::Relaxed);
        if !bb.is_null() {
            let idx = (y * self.width + x) as usize;
            let len = self.back_buffer_len.load(Ordering::Relaxed);
            if idx < len {
                return unsafe { ptr::read_volatile(bb.add(idx)) };
            }
            return 0;
        }
        let bytes_per_pixel = (self.bpp as u32) / 8;
        let offset = (y * self.pitch + x * bytes_per_pixel) as isize;
        unsafe {
            let pixel = self.buffer.offset(offset) as *mut u32;
            ptr::read_volatile(pixel)
        }
    }

    /// Copy the back buffer to VRAM.
    /// Accounts for `pitch` vs `width * bpp/8`.
    pub fn blit(&self) -> bool {
        let bb = self.back_buffer.load(Ordering::Relaxed);
        if bb.is_null() || self.buffer.is_null() {
            return false;
        }
        let bytes_per_pixel = (self.bpp as u32) / 8;
        let row_pixels = self.width as usize;
        if bytes_per_pixel == 4 {
            unsafe {
                for y in 0..self.height as usize {
                    let src = bb.add(y * row_pixels);
                    let dst = self.buffer.add(y * self.pitch as usize) as *mut u32;
                    for x in 0..row_pixels {
                        let color = ptr::read_volatile(src.add(x));
                        ptr::write_volatile(dst.add(x), color);
                    }
                }
            }
        } else {
            for y in 0..self.height {
                for x in 0..self.width {
                    let idx = (y * self.width + x) as usize;
                    let color = unsafe { ptr::read_volatile(bb.add(idx)) };
                    let offset = (y * self.pitch + x * bytes_per_pixel) as isize;
                    unsafe {
                        let pixel = self.buffer.offset(offset) as *mut u32;
                        ptr::write_volatile(pixel, color);
                    }
                }
            }
        }
        true
    }

    pub fn buffer_ptr(&self) -> *mut u8 {
        self.buffer
    }
}
