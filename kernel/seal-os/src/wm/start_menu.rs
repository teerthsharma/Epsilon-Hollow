// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Start menu popup.

use crate::graphics::font;
use crate::graphics::framebuffer::Framebuffer;
use crate::wm::themes;

pub struct StartMenu {
    pub open: bool,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

struct MenuItem {
    name: &'static str,
    app_id: u8,
}

const ITEMS: &[MenuItem] = &[
    MenuItem {
        name: "LAAMBA Governor",
        app_id: 10,
    },
    MenuItem {
        name: "Terminal",
        app_id: 1,
    },
    MenuItem {
        name: "IDE",
        app_id: 2,
    },
    MenuItem {
        name: "Files",
        app_id: 3,
    },
    MenuItem {
        name: "Calc",
        app_id: 4,
    },
    MenuItem {
        name: "Theorems",
        app_id: 5,
    },
    MenuItem {
        name: "Snake",
        app_id: 6,
    },
    MenuItem {
        name: "Breakout",
        app_id: 7,
    },
    MenuItem {
        name: "Warp Racer",
        app_id: 8,
    },
    MenuItem {
        name: "Tensor Viewer",
        app_id: 9,
    },
    MenuItem {
        name: "Aether App",
        app_id: 11,
    },
];

const ITEM_H: u32 = 20;
const START_MENU_HEIGHT: u32 = 8 + (ITEMS.len() as u32) * ITEM_H;

impl StartMenu {
    pub const fn new() -> Self {
        Self {
            open: false,
            x: 4,
            y: 0,
            width: 140,
            height: START_MENU_HEIGHT,
        }
    }

    pub fn toggle(&mut self, taskbar_y: u32) {
        self.open = !self.open;
        self.y = taskbar_y.saturating_sub(self.height);
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    pub fn draw(&self, fb: &Framebuffer) {
        if !self.open {
            return;
        }
        let theme = themes::current_theme();
        fb.fill_rect(self.x, self.y, self.width, self.height, theme.bg);
        fb.fill_rect(self.x, self.y, self.width, 1, theme.border);
        fb.fill_rect(
            self.x,
            self.y + self.height - 1,
            self.width,
            1,
            theme.border,
        );
        fb.fill_rect(self.x, self.y, 1, self.height, theme.border);
        fb.fill_rect(
            self.x + self.width - 1,
            self.y,
            1,
            self.height,
            theme.border,
        );

        for (i, item) in ITEMS.iter().enumerate() {
            let iy = self.y + 4 + i as u32 * ITEM_H;
            for (j, ch) in item.name.bytes().enumerate() {
                font::draw_char(
                    fb,
                    self.x + 8 + j as u32 * font::CHAR_WIDTH,
                    iy,
                    ch,
                    theme.fg,
                );
            }
        }
    }

    pub fn handle_click(&self, mx: u32, my: u32) -> Option<u8> {
        if !self.open {
            return None;
        }
        if mx < self.x || mx >= self.x + self.width || my < self.y || my >= self.y + self.height {
            return None;
        }
        let idx = ((my.saturating_sub(self.y + 4)) / ITEM_H) as usize;
        ITEMS.get(idx).map(|item| item.app_id)
    }
}
