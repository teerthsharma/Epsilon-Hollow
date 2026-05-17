// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Desktop surface: equation wallpaper + taskbar integration.

use crate::graphics::console::Console;
use crate::graphics::framebuffer::Framebuffer;

use super::taskbar;

const BG_COLOR: u32 = 0x000A0A0F;
const GOLD: u32 = 0x00D4A847;
const WHITE: u32 = 0x00C8C8D0;
const DIM: u32 = 0x00404050;

pub fn render_desktop(fb: &Framebuffer) {
    // Clear to dark background
    fb.clear(BG_COLOR);

    // Render equations directly to framebuffer using pixel art
    render_equation_wallpaper(fb);

    // Draw taskbar
    taskbar::draw_taskbar(fb, &[true; 5], 0.042);
}

fn render_equation_wallpaper(fb: &Framebuffer) {
    // Decorative grid pattern (subtle)
    for y in (0..fb.height - 28).step_by(64) {
        for x in 0..fb.width {
            fb.put_pixel(x, y, 0x00101018);
        }
    }
    for x in (0..fb.width).step_by(64) {
        for y in 0..fb.height - 28 {
            fb.put_pixel(x, y, 0x00101018);
        }
    }

    // Central glow effect (radial gradient from center)
    let cx = fb.width / 2;
    let cy = (fb.height - 28) / 2;
    for y in (cy.saturating_sub(200))..((cy + 200).min(fb.height - 28)) {
        for x in (cx.saturating_sub(300))..((cx + 300).min(fb.width)) {
            let dx = (x as i32 - cx as i32) as f64;
            let dy = (y as i32 - cy as i32) as f64;
            let dist = libm::sqrt(dx * dx + dy * dy);
            if dist < 250.0 {
                let intensity = ((1.0 - dist / 250.0) * 12.0) as u32;
                let r = (intensity).min(20);
                let g = (intensity).min(15);
                let b = (intensity + 5).min(30);
                fb.put_pixel(x, y, (r << 16) | (g << 8) | b);
            }
        }
    }

    // Draw a simple circular "black hole" visualization at center
    for angle_i in 0..360 {
        let angle = (angle_i as f64) * core::f64::consts::PI / 180.0;
        for r in 40..60 {
            let x = cx as f64 + r as f64 * libm::cos(angle);
            let y = cy as f64 + r as f64 * libm::sin(angle);
            if x >= 0.0 && y >= 0.0 {
                let px = x as u32;
                let py = y as u32;
                if px < fb.width && py < fb.height - 28 {
                    let fade = ((r - 40) as f64 / 20.0 * 80.0) as u32;
                    let color = (fade << 16) | ((fade / 2) << 8) | (fade + 20).min(255);
                    fb.put_pixel(px, py, color);
                }
            }
        }
    }

    // Accretion disk rings
    for ring in 0..3 {
        let base_r = 70 + ring * 25;
        for angle_i in 0..720 {
            let angle = (angle_i as f64) * core::f64::consts::PI / 360.0;
            let r = base_r as f64 + 5.0 * libm::sin(angle * 3.0);
            let x = cx as f64 + r * libm::cos(angle);
            let y = cy as f64 + r * libm::sin(angle) * 0.4; // elliptical
            if x >= 0.0 && y >= 0.0 {
                let px = x as u32;
                let py = y as u32;
                if px < fb.width && py < fb.height - 28 {
                    let intensity = 180 - ring * 50;
                    fb.put_pixel(px, py, (intensity << 16) | ((intensity / 2) << 8) | 0x20);
                }
            }
        }
    }
}
