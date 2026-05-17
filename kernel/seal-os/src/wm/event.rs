// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Unified input event model — all devices → InputEvent.

#[derive(Debug, Clone, Copy)]
pub enum InputEvent {
    KeyPress(u8),
    KeyRelease(u8),
    MouseMove { dx: i32, dy: i32 },
    MouseButton { button: u8, pressed: bool },
    Scroll { dy: i32 },
}

#[derive(Debug, Clone, Copy)]
pub struct MouseState {
    pub x: i32,
    pub y: i32,
    pub buttons: u8,
}

impl MouseState {
    pub const fn new() -> Self {
        Self {
            x: 512,
            y: 384,
            buttons: 0,
        }
    }

    pub fn apply_move(&mut self, dx: i32, dy: i32, max_x: i32, max_y: i32) {
        self.x = (self.x + dx).clamp(0, max_x - 1);
        self.y = (self.y + dy).clamp(0, max_y - 1);
    }
}
