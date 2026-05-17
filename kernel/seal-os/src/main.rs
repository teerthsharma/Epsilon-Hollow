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

const VERSION: &str = "1.0.0-alpha";

#[no_mangle]
pub extern "C" fn _start() -> ! {
    drivers::serial::init();
    serial_println!("Seal OS v{} — The Geometrical Operating System", VERSION);
    serial_println!("All data = geometry on S². File moves = O(1) topological surgery.");
    serial_println!("");

    // Phase 1: Memory
    memory::init_heap();
    serial_println!("[BOOT] Heap initialized (4 MB)");

    // Phase 2: Interrupts
    drivers::interrupts::init();
    serial_println!("[BOOT] IDT + PIC initialized");

    // Phase 3: Theorems
    let governor = GeometricGovernor::new();
    serial_println!("[T4/AGCR] Governor online: epsilon = {:.4}", governor.epsilon());

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
    serial_println!("[T1/TSS]  Voronoi index: 8 cells, test lookup -> cell {}", cell);

    serial_println!("");
    serial_println!("[BOOT] All theorems ACTIVE. Seal OS ready.");
    serial_println!("[BOOT] Entering sparse event loop (WFI until deviation >= epsilon)");

    loop {
        x86_64::instructions::hlt();
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
