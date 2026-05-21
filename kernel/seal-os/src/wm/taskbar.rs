// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Bottom taskbar: theorem status indicators, clock, seal logo.

use core::sync::atomic::Ordering;

use crate::graphics::framebuffer::Framebuffer;
use crate::{GOVERNOR_EPSILON, THEOREM_STATES};

const TASKBAR_HEIGHT: u32 = 28;

pub fn draw_taskbar(fb: &Framebuffer) {
    let theme = crate::wm::themes::current_theme();
    let y = fb.height - TASKBAR_HEIGHT;

    // Background
    fb.fill_rect(0, y, fb.width, TASKBAR_HEIGHT, theme.taskbar);

    // Top border line
    fb.fill_rect(0, y, fb.width, 1, theme.border);

    // Theorem indicators (small colored squares)
    let names = ["T1", "T2", "T3", "T4", "T5"];
    for (i, name) in names.iter().enumerate() {
        let active = THEOREM_STATES[i].load(Ordering::Relaxed);
        let x = 10 + i as u32 * 50;
        let color = if active { theme.accent } else { 0x00404040 };
        // Indicator dot
        fb.fill_rect(x, y + 8, 10, 10, color);
    }

    // Epsilon value area
    let _epsilon = f64::from_bits(GOVERNOR_EPSILON.load(Ordering::Relaxed));
    fb.fill_rect(270, y + 6, 80, 14, theme.bg);

    // Seal OS branding on the right
    let brand_x = fb.width - 120;
    fb.fill_rect(brand_x, y + 6, 110, 14, theme.bg);
}

pub fn taskbar_height() -> u32 {
    TASKBAR_HEIGHT
}
