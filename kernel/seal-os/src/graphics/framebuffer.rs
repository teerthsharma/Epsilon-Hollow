// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

use core::ptr;

pub struct Framebuffer {
    buffer: *mut u8,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub bpp: u8,
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
        }
    }

    pub unsafe fn new(addr: u64, width: u32, height: u32, pitch: u32, bpp: u8) -> Self {
        Self {
            buffer: addr as *mut u8,
            width,
            height,
            pitch,
            bpp,
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

    pub fn put_pixel(&self, x: u32, y: u32, color: u32) {
        if x >= self.width || y >= self.height || self.buffer.is_null() {
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

    pub fn get_pixel(&self, x: u32, y: u32) -> u32 {
        if x >= self.width || y >= self.height || self.buffer.is_null() {
            return 0;
        }
        let bytes_per_pixel = (self.bpp as u32) / 8;
        let offset = (y * self.pitch + x * bytes_per_pixel) as isize;
        unsafe {
            let pixel = self.buffer.offset(offset) as *mut u32;
            ptr::read_volatile(pixel)
        }
    }

    pub fn buffer_ptr(&self) -> *mut u8 {
        self.buffer
    }
}
