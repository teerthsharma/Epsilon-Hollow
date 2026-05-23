// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![no_std]
#![allow(dead_code, unused_unsafe, improper_ctypes_definitions, unused_imports, unused_features)]
#![no_main]
#![feature(abi_x86_interrupt)]

#[macro_use]
extern crate alloc;

use core::fmt::{self, Write};
use core::panic::PanicInfo;
use uefi::prelude::*;

#[entry]
fn efi_main() -> Status {
    seal_os::boot::uefi_entry::run()
}

/// Small no-alloc formatter to capture the panic message into a stack buffer.
struct BufWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> Write for BufWriter<'a> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.bytes() {
            if self.pos >= self.buf.len() {
                return Err(fmt::Error);
            }
            self.buf[self.pos] = b;
            self.pos += 1;
        }
        Ok(())
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    seal_os::serial_println!("!!! SEAL OS KERNEL PANIC !!!");
    seal_os::serial_println!("{}", info);

    if let Some(fb) = seal_os::framebuffer() {
        // Fill screen red
        for y in 0..fb.height {
            for x in 0..fb.width {
                fb.put_pixel(x, y, 0xFF0000);
            }
        }

        let white = 0xFFFFFF;
        let char_w = seal_os::graphics::font::CHAR_WIDTH;
        let char_h = seal_os::graphics::font::CHAR_HEIGHT;

        // Draw "PANIC" centered near the top
        let panic_text = b"PANIC";
        let text_width = panic_text.len() as u32 * char_w;
        let px = (fb.width.saturating_sub(text_width)) / 2;
        let py = fb.height / 2 - 60;
        for (i, &ch) in panic_text.iter().enumerate() {
            seal_os::graphics::font::draw_char(
                fb,
                px + (i as u32) * char_w,
                py,
                ch,
                white,
            );
        }

        // Capture panic message to a stack buffer and render it
        let mut msg_buf = [0u8; 256];
        let mut writer = BufWriter {
            buf: &mut msg_buf,
            pos: 0,
        };
        let _ = write!(writer, "{}", info);
        let msg_len = writer.pos;

        let msg_y = py + char_h + 16;
        for i in 0..msg_len {
            let col = i as u32 % 60;
            let row = i as u32 / 60;
            let mx = 20 + col * char_w;
            let my = msg_y + row * char_h;
            seal_os::graphics::font::draw_char(fb, mx, my, msg_buf[i], white);
        }

        fb.blit();
    }

    loop {
        x86_64::instructions::hlt();
    }
}
