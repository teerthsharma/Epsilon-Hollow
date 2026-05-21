// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Framebuffer text console with scrolling.

use super::font::{self, CHAR_HEIGHT, CHAR_WIDTH};
use super::framebuffer::Framebuffer;
use core::fmt;

// Colors are read from the active theme at runtime so the console
// respects theme changes.  During early boot the dark theme (index 0)
// is active, yielding the same colors that were previously hard-coded.

pub struct Console {
    fb: &'static Framebuffer,
    col: u32,
    row: u32,
    cols: u32,
    rows: u32,
}

impl Console {
    pub fn new(fb: &'static Framebuffer) -> Self {
        let cols = fb.width / CHAR_WIDTH;
        let rows = fb.height / CHAR_HEIGHT;
        Self {
            fb,
            col: 0,
            row: 0,
            cols,
            rows,
        }
    }

    pub fn clear(&mut self) {
        let theme = crate::wm::themes::current_theme();
        self.fb.clear(theme.console_bg);
        self.col = 0;
        self.row = 0;
    }

    pub fn put_char(&mut self, ch: u8) {
        match ch {
            b'\n' => {
                self.col = 0;
                self.row += 1;
                if self.row >= self.rows {
                    self.scroll();
                }
            }
            b'\r' => {
                self.col = 0;
            }
            b'\t' => {
                self.col = (self.col + 4) & !3;
                if self.col >= self.cols {
                    self.col = 0;
                    self.row += 1;
                    if self.row >= self.rows {
                        self.scroll();
                    }
                }
            }
            _ => {
                let x = self.col * CHAR_WIDTH;
                let y = self.row * CHAR_HEIGHT;
                self.draw_glyph(x, y, ch);
                self.col += 1;
                if self.col >= self.cols {
                    self.col = 0;
                    self.row += 1;
                    if self.row >= self.rows {
                        self.scroll();
                    }
                }
            }
        }
    }

    fn draw_glyph(&self, x: u32, y: u32, ch: u8) {
        let theme = crate::wm::themes::current_theme();
        let glyph = font::glyph(ch);
        for row in 0..CHAR_HEIGHT {
            let bits = glyph[row as usize];
            for col in 0..CHAR_WIDTH {
                let color = if bits & (0x80 >> col) != 0 {
                    theme.console_fg
                } else {
                    theme.console_bg
                };
                self.fb.put_pixel(x + col, y + row, color);
            }
        }
    }

    fn scroll(&mut self) {
        // Shift all rows up by one character height using raw pixel copy
        let bytes_per_pixel = (self.fb.bpp as u32) / 8;
        let row_bytes = self.fb.pitch;
        let scroll_lines = CHAR_HEIGHT;
        let total_lines = self.fb.height;

        unsafe {
            let buf = self.fb.buffer_ptr();
            for line in 0..(total_lines - scroll_lines) {
                let dst = buf.add((line * row_bytes) as usize);
                let src = buf.add(((line + scroll_lines) * row_bytes) as usize);
                core::ptr::copy(src, dst, row_bytes as usize);
            }
            // Clear the last row
            let theme = crate::wm::themes::current_theme();
            for line in (total_lines - scroll_lines)..total_lines {
                let dst = buf.add((line * row_bytes) as usize);
                for x in 0..self.fb.width {
                    let pixel = dst.add((x * bytes_per_pixel) as usize) as *mut u32;
                    pixel.write_volatile(theme.console_bg);
                }
            }
        }
        self.row = self.rows - 1;
    }

    pub fn write_str(&mut self, s: &str) {
        for b in s.bytes() {
            self.put_char(b);
        }
    }

    pub fn write_colored(&mut self, s: &str, color: u32) {
        // Temporarily swap colors — simple approach for boot messages
        for b in s.bytes() {
            match b {
                b'\n' | b'\r' | b'\t' => self.put_char(b),
                _ => {
                    let x = self.col * CHAR_WIDTH;
                    let y = self.row * CHAR_HEIGHT;
                    self.draw_glyph_colored(x, y, b, color);
                    self.col += 1;
                    if self.col >= self.cols {
                        self.col = 0;
                        self.row += 1;
                        if self.row >= self.rows {
                            self.scroll();
                        }
                    }
                }
            }
        }
    }

    fn draw_glyph_colored(&self, x: u32, y: u32, ch: u8, fg: u32) {
        let theme = crate::wm::themes::current_theme();
        let glyph = font::glyph(ch);
        for row in 0..CHAR_HEIGHT {
            let bits = glyph[row as usize];
            for col in 0..CHAR_WIDTH {
                let color = if bits & (0x80 >> col) != 0 { fg } else { theme.console_bg };
                self.fb.put_pixel(x + col, y + row, color);
            }
        }
    }
}

impl fmt::Write for Console {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        Console::write_str(self, s);
        Ok(())
    }
}
