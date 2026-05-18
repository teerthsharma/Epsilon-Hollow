// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Shared game engine utilities: tick loop, collision, score rendering.

use crate::graphics::font;
use crate::wm::window::Window;

pub const GAME_BG: u32 = 0x00101018;
pub const GAME_FG: u32 = 0x00D0D0D8;
pub const SCORE_COLOR: u32 = 0x0088CC44;

pub fn render_text(win: &mut Window, x: u32, y: u32, text: &str, color: u32) {
    for (i, ch) in text.bytes().enumerate() {
        let glyph = font::glyph(ch);
        let px = x + (i as u32) * font::CHAR_WIDTH;
        for gy in 0..font::CHAR_HEIGHT {
            let bits = glyph[gy as usize];
            for gx in 0..font::CHAR_WIDTH {
                if bits & (0x80 >> gx) != 0 {
                    win.set_client_pixel(px + gx, y + gy, color);
                }
            }
        }
    }
}

pub fn fill_rect(win: &mut Window, x: u32, y: u32, w: u32, h: u32, color: u32) {
    let cw = win.client_width();
    let ch = win.client_height();
    for dy in 0..h {
        for dx in 0..w {
            let px = x + dx;
            let py = y + dy;
            if px < cw && py < ch {
                win.set_client_pixel(px, py, color);
            }
        }
    }
}

pub fn clear_window(win: &mut Window) {
    fill_rect(win, 0, 0, win.client_width(), win.client_height(), GAME_BG);
}

#[derive(Clone, Copy)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.x + other.w
            && self.x + self.w > other.x
            && self.y < other.y + other.h
            && self.y + self.h > other.y
    }
}

pub fn simple_rng(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}
