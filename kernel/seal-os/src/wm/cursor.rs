// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Software mouse cursor (10×16 sprites, XOR-blended).

use core::sync::atomic::{AtomicU8, Ordering};

use crate::graphics::framebuffer::Framebuffer;

const CURSOR_W: u32 = 10;
const CURSOR_H: u32 = 16;


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CursorShape {
    Arrow,
    IBeam,
    Hand,
    ResizeH,
    ResizeV,
}

static CURRENT_SHAPE: AtomicU8 = AtomicU8::new(0);

pub fn set_shape(shape: CursorShape) {
    CURRENT_SHAPE.store(shape as u8, Ordering::Relaxed);
}

pub fn current_shape() -> CursorShape {
    match CURRENT_SHAPE.load(Ordering::Relaxed) {
        1 => CursorShape::IBeam,
        2 => CursorShape::Hand,
        3 => CursorShape::ResizeH,
        4 => CursorShape::ResizeV,
        _ => CursorShape::Arrow,
    }
}

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

#[rustfmt::skip]
static IBEAM_DATA: [[u8; 10]; 16] = [
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,0,0],
];

#[rustfmt::skip]
static HAND_DATA: [[u8; 10]; 16] = [
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
];

#[rustfmt::skip]
static RESIZE_H_DATA: [[u8; 10]; 16] = [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,0,0],
    [0,0,1,1,1,0,1,1,1,0],
    [0,1,1,1,0,0,0,1,1,1],
    [0,0,1,1,1,0,1,1,1,0],
    [0,0,0,1,1,1,1,1,0,0],
    [0,0,0,0,1,1,1,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
];

#[rustfmt::skip]
static RESIZE_V_DATA: [[u8; 10]; 16] = [
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
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
        let data: &[[u8; 10]; 16] = match current_shape() {
            CursorShape::IBeam => &IBEAM_DATA,
            CursorShape::Hand => &HAND_DATA,
            CursorShape::ResizeH => &RESIZE_H_DATA,
            CursorShape::ResizeV => &RESIZE_V_DATA,
            _ => &CURSOR_DATA,
        };
        for row in 0..CURSOR_H {
            for col in 0..CURSOR_W {
                if data[row as usize][col as usize] == 0 {
                    continue;
                }
                let px = self.x + col as i32;
                let py = self.y + row as i32;
                if px < 0 || py < 0 || px >= fb.width as i32 || py >= fb.height as i32 {
                    continue;
                }
                let underlying = fb.get_pixel(px as u32, py as u32);
                let theme = crate::wm::themes::current_theme();
                let blended = underlying ^ theme.cursor;
                fb.put_pixel(px as u32, py as u32, blended);
            }
        }
    }
}

/// Convenience free function that draws the cursor at the given coordinates.
pub fn draw_cursor(fb: &Framebuffer, x: i32, y: i32) {
    Cursor::new(x, y).draw(fb);
}
