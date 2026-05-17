// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Multiboot2 header — placed at the start of the binary so GRUB recognizes it.

const MULTIBOOT2_MAGIC: u32 = 0xE85250D6;
const ARCHITECTURE_I386: u32 = 0; // i386 protected mode
const HEADER_LENGTH: u32 = 24; // 6 * 4 bytes (header + end tag)
const CHECKSUM: u32 = (0u32.wrapping_sub(MULTIBOOT2_MAGIC.wrapping_add(ARCHITECTURE_I386).wrapping_add(HEADER_LENGTH)));

#[used]
#[link_section = ".multiboot_header"]
static MULTIBOOT2_HEADER: [u32; 6] = [
    MULTIBOOT2_MAGIC,
    ARCHITECTURE_I386,
    HEADER_LENGTH,
    CHECKSUM,
    // End tag
    0, // type = 0 (end)
    8, // size = 8
];
