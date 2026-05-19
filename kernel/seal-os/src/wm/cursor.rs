// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Software mouse cursor (10×16 arrow sprite, XOR-blended).

use crate::graphics::framebuffer::Framebuffer;

const CURSOR_W: u32 = 10;
const CURSOR_H: u32 = 16;
const CURSOR_XOR_COLOR: u32 = 0x00FFFFFF;

// 1 = pixel (drawn with XOR), 0 = transparent
#[rustfmt::skip]
static CURSOR_DATA: [[u8; 10]; 16] = [
    [1,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0],
    [1,1,1,0,0,0,0,0,0,0],
    [1,1,1,1,0,0,0,0,0,0],
    [1,1,1,1,1,0,0,0,0,0],
    [1,1,1,1,1,1,0,0,0,0],
    [1,1,1,1,1,1,1,0,0,0],
    [1,1,1,1,1,1,1,1,0,0],
    [1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,0,0,0],
    [0,1,1,1,1,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
];

pub struct Cursor {
    pub x: i32,
    pub y: i32,
}

impl Cursor {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn draw(&self, fb: &Framebuffer) {
        for row in 0..CURSOR_H {
            for col in 0..CURSOR_W {
                if CURSOR_DATA[row as usize][col as usize] == 0 {
                    continue;
                }
                let px = self.x + col as i32;
                let py = self.y + row as i32;
                if px < 0 || py < 0 || px >= fb.width as i32 || py >= fb.height as i32 {
                    continue;
                }
                let underlying = fb.get_pixel(px as u32, py as u32);
                let blended = underlying ^ CURSOR_XOR_COLOR;
                fb.put_pixel(px as u32, py as u32, blended);
            }
        }
    }
}

/// Convenience free function that draws the cursor at the given coordinates.
pub fn draw_cursor(fb: &Framebuffer, x: i32, y: i32) {
    Cursor::new(x, y).draw(fb);
}
