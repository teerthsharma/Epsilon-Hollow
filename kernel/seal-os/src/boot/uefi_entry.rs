// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! UEFI entry point — pure Rust, zero assembly.
//! UEFI firmware loads this as a PE/COFF binary in 64-bit long mode.
//! We query GOP for framebuffer, then exit boot services and run the kernel.

use uefi::prelude::*;
use uefi::mem::memory_map::MemoryMap;
use uefi::proto::console::gop::GraphicsOutput;
use uefi::proto::loaded_image::LoadedImage;

use super::boot_info::{BootInfo, MemoryDescriptor, MAX_MEMORY_DESCRIPTORS};
use crate::serial_println;

pub fn run() -> Status {
    // Initialize serial port FIRST — before any fallible UEFI operation.
    // If anything panics below, the panic handler can use serial output.
    crate::drivers::serial::init();
    serial_println!("Seal OS — UEFI entry reached, serial online");

    // Initialize UEFI helpers (logger, allocator)
    if let Err(e) = uefi::helpers::init() {
        serial_println!("[WARN] UEFI helpers init failed: {:?}", e.status());
    }

    // Search UEFI config tables for ACPI 2.0 RSDP
    let acpi_rsdp = uefi::system::with_config_table(|tables| {
        for table in tables {
            if table.guid == uefi::table::cfg::ACPI2_GUID {
                let addr = table.address as u64;
                serial_println!("[BOOT] ACPI 2.0 RSDP found @ {:#X}", addr);
                return addr;
            }
        }
        0u64
    });

    // Query GOP for framebuffer
    let boot_info = get_framebuffer();

    // Get kernel image base and size before exiting boot services
    let image_handle = uefi::boot::image_handle();
    let (kernel_base, kernel_size) = match uefi::boot::open_protocol_exclusive::<LoadedImage>(image_handle) {
        Ok(img) => {
            let (base, size) = img.info();
            (base as u64, size)
        }
        Err(_) => {
            serial_println!("[WARN] Failed to get LoadedImage protocol");
            (0, 0)
        }
    };

    // Exit boot services — after this, UEFI is gone and we own the machine
    let memory_map = unsafe {
        uefi::boot::exit_boot_services(uefi::boot::MemoryType::LOADER_DATA)
    };

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

    // Copy memory map descriptors into BootInfo
    let mut descriptors = [MemoryDescriptor { phys_start: 0, page_count: 0, ty: 0 }; MAX_MEMORY_DESCRIPTORS];
    let mut len = 0;
    for desc in memory_map.entries() {
        if len < MAX_MEMORY_DESCRIPTORS {
            descriptors[len] = MemoryDescriptor {
                phys_start: desc.phys_start,
                page_count: desc.page_count,
                ty: desc.ty.0,
            };
            len += 1;
        }
    }

    let boot_info = BootInfo {
        fb_addr: boot_info.fb_addr,
        fb_width: boot_info.fb_width,
        fb_height: boot_info.fb_height,
        fb_pitch: boot_info.fb_pitch,
        fb_bpp: boot_info.fb_bpp,
        kernel_base,
        kernel_size,
        memory_map: descriptors,
        memory_map_len: len,
        acpi_rsdp,
    };

    // Hand off to kernel
    crate::kernel_main(&boot_info);
}

fn get_framebuffer() -> BootInfo {
    let gop_handle = match uefi::boot::get_handle_for_protocol::<GraphicsOutput>() {
        Ok(h) => h,
        Err(_) => {
            serial_println!("[BOOT] No GOP handle — falling back to serial-only");
            return BootInfo {
                fb_addr: 0,
                fb_width: 0,
                fb_height: 0,
                fb_pitch: 0,
                fb_bpp: 0,
                kernel_base: 0,
                kernel_size: 0,
                memory_map: [MemoryDescriptor { phys_start: 0, page_count: 0, ty: 0 }; MAX_MEMORY_DESCRIPTORS],
                memory_map_len: 0,
                acpi_rsdp: 0,
            };
        }
    };

    let mut gop = match uefi::boot::open_protocol_exclusive::<GraphicsOutput>(gop_handle) {
        Ok(g) => g,
        Err(_) => {
            serial_println!("[BOOT] Failed to open GOP protocol — falling back to serial-only");
            return BootInfo {
                fb_addr: 0,
                fb_width: 0,
                fb_height: 0,
                fb_pitch: 0,
                fb_bpp: 0,
                kernel_base: 0,
                kernel_size: 0,
                memory_map: [MemoryDescriptor { phys_start: 0, page_count: 0, ty: 0 }; MAX_MEMORY_DESCRIPTORS],
                memory_map_len: 0,
                acpi_rsdp: 0,
            };
        }
    };

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
        kernel_base: 0,
        kernel_size: 0,
        memory_map: [MemoryDescriptor { phys_start: 0, page_count: 0, ty: 0 }; MAX_MEMORY_DESCRIPTORS],
        memory_map_len: 0,
        acpi_rsdp: 0,
    }
}
