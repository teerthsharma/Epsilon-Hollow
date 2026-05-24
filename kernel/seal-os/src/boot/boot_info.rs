// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Boot information passed from the UEFI entry point to the kernel.

pub const MAX_MEMORY_DESCRIPTORS: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct MemoryDescriptor {
    pub phys_start: u64,
    pub page_count: u64,
    pub ty: u32,
}

#[derive(Clone, Copy)]
pub struct BootInfo {
    pub fb_addr: u64,
    pub fb_width: u32,
    pub fb_height: u32,
    pub fb_pitch: u32,
    pub fb_bpp: u8,
    pub kernel_base: u64,
    pub kernel_size: u64,
    pub memory_map: [MemoryDescriptor; MAX_MEMORY_DESCRIPTORS],
    pub memory_map_len: usize,
    pub acpi_rsdp: u64,
}
