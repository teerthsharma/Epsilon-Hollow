// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Desktop wallpaper — renders Schwarzschild metric and Faraday tensor.

use super::console::Console;

const GOLD: u32 = 0x00D4A847;
const WHITE: u32 = 0x00C8C8D0;
const DIM: u32 = 0x00606070;

pub fn render(console: &mut Console) {
    console.clear();

    for _ in 0..8 {
        console.write_str("\n");
    }

    console.write_colored("             Schwarzschild Metric (Black Hole)", DIM);
    console.write_str("\n\n");
    console.write_colored(
        "    ds^2 = -(1 - 2GM/rc^2)dt^2 + (1 - 2GM/rc^2)^-1 dr^2 + r^2 dO^2",
        GOLD,
    );
    console.write_str("\n\n\n");

    console.write_colored("             Electromagnetic Field Tensor F^uv", DIM);
    console.write_str("\n\n");
    console.write_colored("          |  0    -Ex   -Ey   -Ez |", WHITE);
    console.write_str("\n");
    console.write_colored("   F^uv = |  Ex    0    -Bz    By |", WHITE);
    console.write_str("\n");
    console.write_colored("          |  Ey    Bz    0    -Bx |", WHITE);
    console.write_str("\n");
    console.write_colored("          |  Ez   -By    Bx    0  |", WHITE);
    console.write_str("\n\n\n\n");

    console.write_colored("        All data = geometry on S^2", DIM);
    console.write_str("\n");
    console.write_colored("        File moves = O(1) topological surgery", DIM);
    console.write_str("\n\n");

    console.write_colored(
        "  [T1:*] [T2:*] [T3:*] [T4:*] [T5:*]           Seal OS v1.0",
        0x004488CC,
    );
    console.write_str("\n");
}
