// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Desktop surface: equation wallpaper + taskbar integration.

use crate::graphics::font;
use crate::graphics::framebuffer::Framebuffer;

use super::taskbar;

const BG_COLOR: u32 = 0x000A0A0F;

pub fn render_desktop(fb: &Framebuffer) {
    // Clear to dark background
    fb.clear(BG_COLOR);

    // Render decorative background art
    render_equation_wallpaper(fb);

    // Render equations as text in the upper-left quadrant
    render_equations_text(fb);

    // Draw taskbar
    taskbar::draw_taskbar(fb);
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

fn render_equations_text(fb: &Framebuffer) {
    const TEXT_X: u32 = 20;
    const TEXT_Y: u32 = 20;
    const LINE_HEIGHT: u32 = 18;
    const PAD: u32 = 8;
    const BG_ALPHA: u8 = 180;
    const BG_PANEL: u32 = 0x0005080A;
    const TEXT_COLOR: u32 = 0x00C8C8D0;
    const LABEL_COLOR: u32 = 0x00606070;
    const GOLD_COLOR: u32 = 0x00D4A847;

    let lines: &[(u32, &str)] = &[
        (LABEL_COLOR, "Schwarzschild metric:"),
        (GOLD_COLOR, "ds^2 = -(1-2GM/rc^2)dt^2 + (1-2GM/rc^2)^-1 dr^2 + r^2 dO^2"),
        (TEXT_COLOR, ""),
        (LABEL_COLOR, "Faraday tensor F^uv:"),
        (TEXT_COLOR, "[ 0   -Ex  -Ey  -Ez ]"),
        (TEXT_COLOR, "[ Ex   0   -Bz   By ]"),
        (TEXT_COLOR, "[ Ey   Bz   0   -Bx ]"),
        (TEXT_COLOR, "[ Ez  -By   Bx   0  ]"),
    ];

    // Compute bounding box from longest line
    let mut max_w = 0u32;
    for (_, text) in lines {
        let w = text.len() as u32 * font::CHAR_WIDTH;
        if w > max_w {
            max_w = w;
        }
    }
    let total_h = lines.len() as u32 * LINE_HEIGHT;

    let bg_x = TEXT_X.saturating_sub(PAD);
    let bg_y = TEXT_Y.saturating_sub(PAD);
    let bg_w = max_w + PAD * 2;
    let bg_h = total_h + PAD * 2;

    // Draw semi-transparent dark background
    for row in bg_y..(bg_y + bg_h).min(fb.height) {
        for col in bg_x..(bg_x + bg_w).min(fb.width) {
            let bg_pixel = fb.get_pixel(col, row);
            let blended = alpha_blend(bg_pixel, BG_PANEL, BG_ALPHA);
            fb.put_pixel(col, row, blended);
        }
    }

    // Draw each line of text
    for (i, &(color, text)) in lines.iter().enumerate() {
        let y = TEXT_Y + i as u32 * LINE_HEIGHT;
        draw_text(fb, TEXT_X, y, text, color);
    }
}

fn draw_text(fb: &Framebuffer, x: u32, y: u32, text: &str, color: u32) {
    for (i, ch) in text.bytes().enumerate() {
        font::draw_char(fb, x + (i as u32) * font::CHAR_WIDTH, y, ch, color);
    }
}

fn alpha_blend(bg: u32, fg: u32, alpha: u8) -> u32 {
    if alpha == 255 {
        return fg;
    }
    if alpha == 0 {
        return bg;
    }
    let a = alpha as u32;
    let inv = 255 - a;
    let r = ((fg >> 16 & 0xFF) * a + (bg >> 16 & 0xFF) * inv) / 255;
    let g = ((fg >> 8 & 0xFF) * a + (bg >> 8 & 0xFF) * inv) / 255;
    let b = ((fg & 0xFF) * a + (bg & 0xFF) * inv) / 255;
    (r << 16) | (g << 8) | b
}
