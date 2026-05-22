// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Window representation with framebuffer region and decorations.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::wm::themes;

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
    pub prev_x: u32,
    pub prev_y: u32,
    pub prev_w: u32,
    pub prev_h: u32,
    pub z_order: u32,
    pub state: WindowState,
    pub focused: bool,
    pub buffer: Vec<u32>,
    pub dirty: bool,
}

pub const TITLE_BAR_HEIGHT: u32 = 24;
pub const BORDER_WIDTH: u32 = 2;

impl Window {
    pub fn new(id: u32, title: &str, x: u32, y: u32, width: u32, height: u32) -> Self {
        let total_w = width + BORDER_WIDTH * 2;
        let total_h = height + TITLE_BAR_HEIGHT + BORDER_WIDTH;
        let theme = themes::current_theme();
        Self {
            id,
            title: String::from(title),
            x,
            y,
            width: total_w,
            height: total_h,
            prev_x: x,
            prev_y: y,
            prev_w: total_w,
            prev_h: total_h,
            z_order: id,
            state: WindowState::Normal,
            focused: false,
            buffer: vec![theme.bg; (total_w * total_h) as usize],
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

    pub fn in_minimize_button(&self, px: u32, py: u32) -> bool {
        let btn_x = self.x + self.width - 40;
        let btn_y = self.y + 4;
        px >= btn_x && px < btn_x + 16 && py >= btn_y && py < btn_y + 16
    }

    pub fn in_maximize_button(&self, px: u32, py: u32) -> bool {
        let btn_x = self.x + self.width - 60;
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
        let theme = themes::current_theme();
        let title_bg = if self.focused {
            theme.titlebar
        } else {
            theme.titlebar_unfocused
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
                self.buffer[(y * w + x) as usize] = theme.close_btn;
            }
        }

        // Minimize button
        let min_x = w - 40;
        for y in 4..20 {
            for x in min_x..(min_x + 16) {
                self.buffer[(y * w + x) as usize] = theme.titlebar;
            }
        }
        // Minimize symbol (_)
        for x in min_x + 4..min_x + 12 {
            self.buffer[(17 * w + x) as usize] = theme.fg;
        }

        // Maximize button
        let max_x = w - 60;
        for y in 4..20 {
            for x in max_x..(max_x + 16) {
                self.buffer[(y * w + x) as usize] = theme.titlebar;
            }
        }
        // Maximize symbol (square)
        for x in max_x + 3..max_x + 13 {
            self.buffer[(6 * w + x) as usize] = theme.fg;
            self.buffer[(15 * w + x) as usize] = theme.fg;
        }
        for y in 6..16 {
            self.buffer[(y * w + max_x + 3) as usize] = theme.fg;
            self.buffer[(y * w + max_x + 12) as usize] = theme.fg;
        }

        // Title text (simple: just mark the pixels, actual rendering via font in compositor)
        // The compositor will overlay the title text

        // Borders
        for y in 0..h {
            for bx in 0..BORDER_WIDTH {
                self.buffer[(y * w + bx) as usize] = theme.border;
                if bx + w - BORDER_WIDTH < w {
                    self.buffer[(y * w + w - BORDER_WIDTH + bx) as usize] = theme.border;
                }
            }
        }
        // Bottom border
        for y in (h - BORDER_WIDTH)..h {
            for x in 0..w {
                self.buffer[(y * w + x) as usize] = theme.border;
            }
        }

        self.dirty = true;
    }
}
