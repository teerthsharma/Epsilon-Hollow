// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Power menu popup.

use crate::graphics::font;
use crate::graphics::framebuffer::Framebuffer;
use crate::wm::themes;

pub struct PowerMenu {
    pub open: bool,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum PowerAction {
    Shutdown,
    Reboot,
    Logout,
}

struct MenuItem {
    name: &'static str,
    action: PowerAction,
}

const ITEMS: &[MenuItem] = &[
    MenuItem {
        name: "Shutdown",
        action: PowerAction::Shutdown,
    },
    MenuItem {
        name: "Reboot",
        action: PowerAction::Reboot,
    },
    MenuItem {
        name: "Logout",
        action: PowerAction::Logout,
    },
];

const ITEM_H: u32 = 24;
const POWER_MENU_HEIGHT: u32 = 8 + (ITEMS.len() as u32) * ITEM_H;

impl PowerMenu {
    pub const fn new() -> Self {
        Self {
            open: false,
            x: 0,
            y: 0,
            width: 150,
            height: POWER_MENU_HEIGHT,
        }
    }

    pub fn toggle(&mut self, x: u32, taskbar_y: u32) {
        self.open = !self.open;
        self.x = x;
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
            let iy = self.y + 6 + i as u32 * ITEM_H;
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

    pub fn handle_click(&self, mx: u32, my: u32) -> Option<PowerAction> {
        if !self.open {
            return None;
        }
        if mx < self.x || mx >= self.x + self.width || my < self.y || my >= self.y + self.height {
            return None;
        }
        let idx = ((my.saturating_sub(self.y + 6)) / ITEM_H) as usize;
        ITEMS.get(idx).map(|item| item.action)
    }
}
