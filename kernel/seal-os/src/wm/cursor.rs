// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Software mouse cursor (12x19 arrow sprite).

use crate::graphics::framebuffer::Framebuffer;

const CURSOR_W: u32 = 12;
const CURSOR_H: u32 = 19;

// 1=white outline, 2=black fill, 0=transparent
#[rustfmt::skip]
static CURSOR_DATA: [[u8; 12]; 19] = [
    [1,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,0,0],
    [1,2,1,0,0,0,0,0,0,0,0,0],
    [1,2,2,1,0,0,0,0,0,0,0,0],
    [1,2,2,2,1,0,0,0,0,0,0,0],
    [1,2,2,2,2,1,0,0,0,0,0,0],
    [1,2,2,2,2,2,1,0,0,0,0,0],
    [1,2,2,2,2,2,2,1,0,0,0,0],
    [1,2,2,2,2,2,2,2,1,0,0,0],
    [1,2,2,2,2,2,2,2,2,1,0,0],
    [1,2,2,2,2,2,2,2,2,2,1,0],
    [1,2,2,2,2,2,2,1,1,1,1,0],
    [1,2,2,2,1,2,2,1,0,0,0,0],
    [1,2,2,1,0,1,2,2,1,0,0,0],
    [1,2,1,0,0,1,2,2,1,0,0,0],
    [1,1,0,0,0,0,1,2,2,1,0,0],
    [1,0,0,0,0,0,1,2,2,1,0,0],
    [0,0,0,0,0,0,0,1,2,1,0,0],
    [0,0,0,0,0,0,0,1,1,0,0,0],
];

pub fn draw_cursor(fb: &Framebuffer, x: i32, y: i32) {
    for row in 0..CURSOR_H {
        for col in 0..CURSOR_W {
            let px = x + col as i32;
            let py = y + row as i32;
            if px < 0 || py < 0 || px >= fb.width as i32 || py >= fb.height as i32 {
                continue;
            }
            let color = match CURSOR_DATA[row as usize][col as usize] {
                1 => 0x00FFFFFF,
                2 => 0x00000000,
                _ => continue,
            };
            fb.put_pixel(px as u32, py as u32, color);
        }
    }
}
