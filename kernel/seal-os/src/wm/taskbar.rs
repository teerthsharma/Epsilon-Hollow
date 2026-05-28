// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Bottom taskbar: theorem status indicators, clock, start/power buttons.

use alloc::string::String;
use core::sync::atomic::Ordering;

use crate::graphics::font;
use crate::graphics::framebuffer::Framebuffer;
use crate::{GOVERNOR_EPSILON, THEOREM_COUNT, THEOREM_STATES};

const TASKBAR_HEIGHT: u32 = 28;

pub fn draw_taskbar(fb: &Framebuffer) {
    let theme = crate::wm::themes::current_theme();
    let y = fb.height - TASKBAR_HEIGHT;

    // Background
    fb.fill_rect(0, y, fb.width, TASKBAR_HEIGHT, theme.taskbar);

    // Top border line
    fb.fill_rect(0, y, fb.width, 1, theme.border);

    // Start button (left)
    fb.fill_rect(4, y + 4, 72, 20, theme.accent);
    let start_label = "Seal";
    let start_w = start_label.len() as u32 * font::CHAR_WIDTH;
    let start_x = 4 + (72 - start_w) / 2;
    for (i, ch) in start_label.bytes().enumerate() {
        font::draw_char(
            fb,
            start_x + i as u32 * font::CHAR_WIDTH,
            y + 6,
            ch,
            0xFFFFFF,
        );
    }

    // Theorem indicators (small colored squares)
    for i in 0..THEOREM_COUNT {
        let active = THEOREM_STATES[i].load(Ordering::Relaxed);
        let x = 90 + i as u32 * 18;
        let color = if active { theme.accent } else { 0x00404040 };
        fb.fill_rect(x, y + 8, 10, 10, color);
    }

    // Epsilon value area
    let _epsilon = f64::from_bits(GOVERNOR_EPSILON.load(Ordering::Relaxed));
    fb.fill_rect(292, y + 6, 80, 14, theme.bg);

    // Clock (center)
    let time_str = format_time();
    let clock_w = time_str.len() as u32 * font::CHAR_WIDTH;
    let clock_x = fb.width / 2 - clock_w / 2;
    for (i, ch) in time_str.bytes().enumerate() {
        font::draw_char(
            fb,
            clock_x + i as u32 * font::CHAR_WIDTH,
            y + 6,
            ch,
            theme.fg,
        );
    }

    // Power button (right)
    let power_x = fb.width - 36;
    fb.fill_rect(power_x, y + 6, 24, 16, 0xFF4444);
    font::draw_char(fb, power_x + 8, y + 6, b'P', 0xFFFFFF);
}

pub fn taskbar_height() -> u32 {
    TASKBAR_HEIGHT
}

fn format_time() -> String {
    let t = crate::drivers::rtc::read_time();
    alloc::format!("{:02}:{:02}:{:02}", t.hour, t.min, t.sec)
}
