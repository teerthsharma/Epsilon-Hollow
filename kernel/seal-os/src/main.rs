// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

#[macro_use]
extern crate alloc;

use core::panic::PanicInfo;
use uefi::prelude::*;

#[entry]
fn efi_main() -> Status {
    seal_os::boot::uefi_entry::run()
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    seal_os::serial_println!("!!! SEAL OS KERNEL PANIC !!!");
    seal_os::serial_println!("{}", info);
    loop {
        x86_64::instructions::hlt();
    }
}
