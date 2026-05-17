// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Multiboot2 header with framebuffer request tag.

const MULTIBOOT2_MAGIC: u32 = 0xE85250D6;
const ARCHITECTURE_I386: u32 = 0;

// Header (16) + FB tag (20) + padding (4) + end tag (8) = 48 bytes
const HEADER_LENGTH: u32 = 48;
const CHECKSUM: u32 =
    0u32.wrapping_sub(MULTIBOOT2_MAGIC.wrapping_add(ARCHITECTURE_I386).wrapping_add(HEADER_LENGTH));

#[used]
#[link_section = ".multiboot_header"]
static MULTIBOOT2_HEADER: [u32; 12] = [
    // --- Header (16 bytes) ---
    MULTIBOOT2_MAGIC,
    ARCHITECTURE_I386,
    HEADER_LENGTH,
    CHECKSUM,
    // --- Framebuffer tag (type=5, flags=0 → packed as type(u16)|flags(u16), size=20) ---
    // type=5 (u16 LE) | flags=0 (u16 LE) → 0x0000_0005
    5,
    // size of this tag = 20
    20,
    // width
    1024,
    // height
    768,
    // depth
    32,
    // --- Padding to 8-byte alignment (tags must be 8-byte aligned) ---
    0,
    // --- End tag (type=0, size=8) ---
    0, // type=0 | flags=0
    8, // size=8
];
