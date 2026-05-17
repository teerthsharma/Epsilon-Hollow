// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Bottom taskbar: theorem status indicators, clock, seal logo.

use crate::drivers::interrupts;
use crate::graphics::console::Console;
use crate::graphics::framebuffer::Framebuffer;

const TASKBAR_HEIGHT: u32 = 28;
const TASKBAR_BG: u32 = 0x00101018;
const TASKBAR_TEXT: u32 = 0x00A0A0B0;
const THEOREM_ACTIVE: u32 = 0x0044CC66;
const SEAL_BLUE: u32 = 0x004488CC;

pub fn draw_taskbar(fb: &Framebuffer, theorem_states: &[bool; 5], epsilon: f64) {
    let y = fb.height - TASKBAR_HEIGHT;

    // Background
    fb.fill_rect(0, y, fb.width, TASKBAR_HEIGHT, TASKBAR_BG);

    // Top border line
    fb.fill_rect(0, y, fb.width, 1, 0x00303040);

    // Theorem indicators (small colored squares)
    let names = ["T1", "T2", "T3", "T4", "T5"];
    for (i, (&active, name)) in theorem_states.iter().zip(names.iter()).enumerate() {
        let x = 10 + i as u32 * 50;
        let color = if active { THEOREM_ACTIVE } else { 0x00404040 };
        // Indicator dot
        fb.fill_rect(x, y + 8, 10, 10, color);
    }

    // Epsilon value area
    fb.fill_rect(270, y + 6, 80, 14, 0x00181828);

    // Seal OS branding on the right
    let brand_x = fb.width - 120;
    fb.fill_rect(brand_x, y + 6, 110, 14, 0x00181828);
}

pub fn taskbar_height() -> u32 {
    TASKBAR_HEIGHT
}
