// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! UEFI entry point — pure Rust, zero assembly.
//! UEFI firmware loads this as a PE/COFF binary in 64-bit long mode.
//! We query GOP for framebuffer, then exit boot services and run the kernel.

use uefi::prelude::*;
use uefi::proto::console::gop::GraphicsOutput;

use super::boot_info::BootInfo;
use crate::serial_println;

#[entry]
fn efi_main() -> Status {
    // Initialize UEFI helpers (logger, allocator)
    uefi::helpers::init().expect("Failed to initialize UEFI helpers");

    // Query GOP for framebuffer
    let boot_info = get_framebuffer();

    // Exit boot services — after this, UEFI is gone and we own the machine
    unsafe {
        uefi::boot::exit_boot_services(None);
    }

    // Initialize serial port (COM1) for debug output
    crate::drivers::serial::init();
    serial_println!("Seal OS — UEFI boot complete, boot services exited");

    if boot_info.fb_addr != 0 {
        serial_println!(
            "[BOOT] Framebuffer: {}x{}x{}bpp @ {:#X}, pitch={}",
            boot_info.fb_width, boot_info.fb_height, boot_info.fb_bpp,
            boot_info.fb_addr, boot_info.fb_pitch
        );
    } else {
        serial_println!("[BOOT] No GOP framebuffer available");
    }

    // Hand off to kernel
    crate::kernel_main(&boot_info);
}

fn get_framebuffer() -> BootInfo {
    let gop_handle = match uefi::boot::get_handle_for_protocol::<GraphicsOutput>() {
        Ok(h) => h,
        Err(_) => {
            return BootInfo {
                fb_addr: 0,
                fb_width: 0,
                fb_height: 0,
                fb_pitch: 0,
                fb_bpp: 0,
            };
        }
    };

    let mut gop = uefi::boot::open_protocol_exclusive::<GraphicsOutput>(gop_handle)
        .expect("Failed to open GOP");

    // Try to find 1024x768x32 mode, fall back to current mode
    let target_w = 1024;
    let target_h = 768;

    let mut best_mode = None;
    for mode in gop.modes() {
        let info = mode.info();
        let (w, h) = info.resolution();
        if w == target_w && h == target_h {
            best_mode = Some(mode);
            break;
        }
    }

    if let Some(mode) = best_mode {
        let _ = gop.set_mode(&mode);
    }

    let mode_info = gop.current_mode_info();
    let (width, height) = mode_info.resolution();
    let stride = mode_info.stride();
    let mut fb = gop.frame_buffer();
    let fb_addr = fb.as_mut_ptr() as u64;

    BootInfo {
        fb_addr,
        fb_width: width as u32,
        fb_height: height as u32,
        fb_pitch: (stride * 4) as u32, // 4 bytes per pixel (32bpp)
        fb_bpp: 32,
    }
}
