// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

extern crate alloc;

mod boot;
mod drivers;
mod graphics;
mod memory;

use core::panic::PanicInfo;

use aether_core::governor::GeometricGovernor;
use aether_core::tss::SphericalVoronoiIndex;

use graphics::console::Console;
use graphics::framebuffer::Framebuffer;

const VERSION: &str = "1.0.0-alpha";

static mut FRAMEBUFFER: Framebuffer = Framebuffer::null();

#[no_mangle]
pub extern "C" fn _start(multiboot_info_addr: u64) -> ! {
    drivers::serial::init();
    serial_println!("Seal OS v{} — The Geometrical Operating System", VERSION);
    serial_println!("All data = geometry on S^2. File moves = O(1) topological surgery.");
    serial_println!("");

    // Phase 1: Memory
    memory::init_heap();
    serial_println!("[BOOT] Heap initialized (4 MB)");

    // Phase 2: Interrupts
    drivers::interrupts::init();
    serial_println!("[BOOT] IDT + PIC initialized");

    // Phase 2b: Framebuffer
    if multiboot_info_addr != 0 {
        parse_multiboot2(multiboot_info_addr);
    }

    let fb = unsafe { &*core::ptr::addr_of!(FRAMEBUFFER) };

    if fb.is_available() {
        serial_println!(
            "[BOOT] Framebuffer: {}x{}x{}bpp",
            fb.width,
            fb.height,
            fb.bpp
        );

        // Boot splash
        let mut console = Console::new(fb);
        graphics::splash::render_splash(&mut console);
        graphics::splash::draw_progress_bar(fb, 30);

        // Phase 3: Theorems
        let governor = GeometricGovernor::new();
        serial_println!(
            "[T4/AGCR] Governor online: epsilon = {:.4}",
            governor.epsilon()
        );
        graphics::splash::draw_progress_bar(fb, 60);

        let centroids = [
            (0.0, 0.0),
            (1.57, 0.0),
            (3.14, 0.0),
            (0.0, 1.57),
            (1.57, 1.57),
            (3.14, 1.57),
            (0.0, 3.14),
            (1.57, 3.14),
        ];
        let voronoi = SphericalVoronoiIndex::<8>::new(centroids);
        let cell = voronoi.locate((0.5, 0.5));
        serial_println!(
            "[T1/TSS]  Voronoi index: 8 cells, test lookup -> cell {}",
            cell
        );
        graphics::splash::draw_progress_bar(fb, 90);

        serial_println!("");
        serial_println!("[BOOT] All theorems ACTIVE. Seal OS ready.");
        graphics::splash::draw_progress_bar(fb, 100);

        // Brief pause on splash, then show wallpaper
        for _ in 0..50_000_000u64 {
            core::hint::spin_loop();
        }

        // Render desktop wallpaper
        graphics::wallpaper::render(&mut console);
        serial_println!("[BOOT] Desktop wallpaper rendered.");
    } else {
        serial_println!("[BOOT] No framebuffer — serial-only mode");

        let governor = GeometricGovernor::new();
        serial_println!(
            "[T4/AGCR] Governor online: epsilon = {:.4}",
            governor.epsilon()
        );

        let centroids = [
            (0.0, 0.0),
            (1.57, 0.0),
            (3.14, 0.0),
            (0.0, 1.57),
            (1.57, 1.57),
            (3.14, 1.57),
            (0.0, 3.14),
            (1.57, 3.14),
        ];
        let voronoi = SphericalVoronoiIndex::<8>::new(centroids);
        let cell = voronoi.locate((0.5, 0.5));
        serial_println!(
            "[T1/TSS]  Voronoi index: 8 cells, test lookup -> cell {}",
            cell
        );

        serial_println!("");
        serial_println!("[BOOT] All theorems ACTIVE. Seal OS ready.");
    }

    serial_println!("[BOOT] Entering sparse event loop (WFI until deviation >= epsilon)");

    loop {
        x86_64::instructions::hlt();
    }
}

fn parse_multiboot2(addr: u64) {
    unsafe {
        let ptr = addr as *const u8;
        let total_size = *(ptr as *const u32) as usize;
        let mut offset = 8usize; // skip total_size + reserved

        while offset < total_size {
            let tag_ptr = ptr.add(offset);
            let tag_type = *(tag_ptr as *const u32);
            let tag_size = *(tag_ptr.add(4) as *const u32) as usize;

            if tag_type == 0 {
                break; // end tag
            }

            // Tag type 8 = framebuffer info
            if tag_type == 8 && tag_size >= 32 {
                let fb_addr = *(tag_ptr.add(8) as *const u64);
                let fb_pitch = *(tag_ptr.add(16) as *const u32);
                let fb_width = *(tag_ptr.add(20) as *const u32);
                let fb_height = *(tag_ptr.add(24) as *const u32);
                let fb_bpp = *(tag_ptr.add(28) as *const u8);
                let fb_type = *(tag_ptr.add(29) as *const u8);

                // Type 1 = RGB framebuffer (what we want)
                if fb_type == 1 {
                    serial_println!(
                        "[FB] Found framebuffer: {}x{} @ {:#X}, pitch={}, bpp={}",
                        fb_width,
                        fb_height,
                        fb_addr,
                        fb_pitch,
                        fb_bpp
                    );
                    FRAMEBUFFER = Framebuffer::new(fb_addr, fb_width, fb_height, fb_pitch, fb_bpp);
                }
            }

            // Tags are 8-byte aligned
            offset += ((tag_size + 7) & !7).max(8);
        }
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("!!! SEAL OS KERNEL PANIC !!!");
    serial_println!("{}", info);
    loop {
        x86_64::instructions::hlt();
    }
}
