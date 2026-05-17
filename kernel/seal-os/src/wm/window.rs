// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Window representation with framebuffer region and decorations.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowState {
    Normal,
    Minimized,
    Maximized,
    Closed,
}

pub struct Window {
    pub id: u32,
    pub title: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub z_order: u32,
    pub state: WindowState,
    pub focused: bool,
    pub buffer: Vec<u32>,
    pub dirty: bool,
}

pub const TITLE_BAR_HEIGHT: u32 = 24;
pub const BORDER_WIDTH: u32 = 2;
const TITLE_BG: u32 = 0x002040A0;
const TITLE_BG_UNFOCUSED: u32 = 0x00404050;
const BORDER_COLOR: u32 = 0x00505060;
const CLOSE_BTN: u32 = 0x00CC4444;

impl Window {
    pub fn new(id: u32, title: &str, x: u32, y: u32, width: u32, height: u32) -> Self {
        let total_w = width + BORDER_WIDTH * 2;
        let total_h = height + TITLE_BAR_HEIGHT + BORDER_WIDTH;
        Self {
            id,
            title: String::from(title),
            x,
            y,
            width: total_w,
            height: total_h,
            z_order: id,
            state: WindowState::Normal,
            focused: false,
            buffer: vec![0x00181820; (total_w * total_h) as usize],
            dirty: true,
        }
    }

    pub fn client_x(&self) -> u32 {
        self.x + BORDER_WIDTH
    }

    pub fn client_y(&self) -> u32 {
        self.y + TITLE_BAR_HEIGHT
    }

    pub fn client_width(&self) -> u32 {
        self.width - BORDER_WIDTH * 2
    }

    pub fn client_height(&self) -> u32 {
        self.height - TITLE_BAR_HEIGHT - BORDER_WIDTH
    }

    pub fn contains(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    pub fn in_title_bar(&self, px: u32, py: u32) -> bool {
        px >= self.x
            && px < self.x + self.width
            && py >= self.y
            && py < self.y + TITLE_BAR_HEIGHT
    }

    pub fn in_close_button(&self, px: u32, py: u32) -> bool {
        let btn_x = self.x + self.width - 20;
        let btn_y = self.y + 4;
        px >= btn_x && px < btn_x + 16 && py >= btn_y && py < btn_y + 16
    }

    pub fn set_client_pixel(&mut self, x: u32, y: u32, color: u32) {
        let bx = x + BORDER_WIDTH;
        let by = y + TITLE_BAR_HEIGHT;
        if bx < self.width && by < self.height {
            self.buffer[(by * self.width + bx) as usize] = color;
            self.dirty = true;
        }
    }

    pub fn render_decorations(&mut self) {
        let w = self.width;
        let h = self.height;
        let title_bg = if self.focused {
            TITLE_BG
        } else {
            TITLE_BG_UNFOCUSED
        };

        // Title bar
        for y in 0..TITLE_BAR_HEIGHT {
            for x in 0..w {
                self.buffer[(y * w + x) as usize] = title_bg;
            }
        }

        // Close button
        let btn_x = w - 20;
        for y in 4..20 {
            for x in btn_x..(btn_x + 16) {
                self.buffer[(y * w + x) as usize] = CLOSE_BTN;
            }
        }

        // Title text (simple: just mark the pixels, actual rendering via font in compositor)
        // The compositor will overlay the title text

        // Borders
        for y in 0..h {
            for bx in 0..BORDER_WIDTH {
                self.buffer[(y * w + bx) as usize] = BORDER_COLOR;
                if bx + w - BORDER_WIDTH < w {
                    self.buffer[(y * w + w - BORDER_WIDTH + bx) as usize] = BORDER_COLOR;
                }
            }
        }
        // Bottom border
        for y in (h - BORDER_WIDTH)..h {
            for x in 0..w {
                self.buffer[(y * w + x) as usize] = BORDER_COLOR;
            }
        }

        self.dirty = true;
    }
}
