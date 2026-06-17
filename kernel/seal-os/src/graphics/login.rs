// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Real password login screen with username and password fields.

use super::font::{CHAR_HEIGHT, CHAR_WIDTH};
use super::framebuffer::Framebuffer;

// Mascot palette — fixed 16-color palette, DO NOT change these values
const DARK_BG: u32 = 0x00080810;
const PENGUIN_BODY: u32 = 0x002A2A5A;
const PENGUIN_BELLY: u32 = 0x00E8E8F0;
const PENGUIN_BEAK: u32 = 0x00E8B830;
const PENGUIN_FEET: u32 = 0x00E8B830;
const PENGUIN_CHEEK: u32 = 0x00F0A0B0;
const PENGUIN_EYE: u32 = 0x00101020;
const PENGUIN_EYE_SHINE: u32 = 0x00FFFFFF;
const SEAL_BODY: u32 = 0x00D0D0D8;
const SEAL_DARK: u32 = 0x00A0A0B0;
const SEAL_SPOTS: u32 = 0x00909098;
const SEAL_EYE: u32 = 0x00101020;
const SEAL_NOSE: u32 = 0x00404050;
const OUTLINE: u32 = 0x0040D0F0;
const SHADOW: u32 = 0x00202030;

// 32x32 mascot sprite — each u8 indexes into the palette below
// Simplified pixel art of a penguin + seal hugging
static MASCOT_PALETTE: [u32; 16] = [
    DARK_BG,           // 0 - transparent/bg
    PENGUIN_BODY,      // 1
    PENGUIN_BELLY,     // 2
    PENGUIN_BEAK,      // 3
    PENGUIN_FEET,      // 4
    PENGUIN_CHEEK,     // 5
    PENGUIN_EYE,       // 6
    PENGUIN_EYE_SHINE, // 7
    SEAL_BODY,         // 8
    SEAL_DARK,         // 9
    SEAL_SPOTS,        // A
    SEAL_EYE,          // B
    SEAL_NOSE,         // C
    OUTLINE,           // D
    SHADOW,            // E
    SEAL_BODY,         // F (duplicate for convenience)
];

#[rustfmt::skip]
static MASCOT_32X32: [u8; 32 * 32] = [
//  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 0
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, // row 2
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8,10, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, // row 3
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8,11, 8, 8, 8,11, 8, 8, 8, 0, 0, 0, 0, 0, 0, // row 4
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8,12, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, // row 5
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, // row 6
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, // row 7
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 8, 9, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 8
    0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 7, 1, 1, 6, 7, 1, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 9
    0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 6, 1, 1, 6, 6, 1, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 10
    0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 1, 3, 3, 1, 5, 1, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 11
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 1, 1, 1, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 12
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 13
    0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 8, 8, 9, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, // row 14
    0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, // row 15
    0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, // row 16
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 17
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 1, 1, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 18
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 1, 8, 8, 8, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 19
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 20
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 21
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 22
    0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 23
    0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 4, 4, 4, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 24
    0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 25
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 26
    0, 0, 0, 0, 0, 0,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 27
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 28
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 29
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 30
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 31
];

const SCALE: u32 = 6;

pub struct LoginScreen {
    username: [u8; 32],
    username_len: usize,
    password: [u8; 32],
    password_len: usize,
    active_field: u8,
    error_msg: [u8; 64],
    error_len: usize,
    authenticated_user: Option<crate::security::passwd::UserEntry>,
}

impl LoginScreen {
    pub const fn new() -> Self {
        Self {
            username: [0u8; 32],
            username_len: 0,
            password: [0u8; 32],
            password_len: 0,
            active_field: 0,
            error_msg: [0u8; 64],
            error_len: 0,
            authenticated_user: None,
        }
    }

    pub fn render(&self, fb: &Framebuffer) {
        let theme = crate::wm::themes::current_theme();
        fb.clear(theme.login_bg);

        // Draw the mascot sprite scaled up, centered horizontally
        let sprite_w = 32 * SCALE;
        let sprite_x = (fb.width.saturating_sub(sprite_w)) / 2;
        let sprite_y = fb.height / 8;

        for row in 0..32u32 {
            for col in 0..32u32 {
                let idx = MASCOT_32X32[(row * 32 + col) as usize];
                if idx == 0 {
                    continue;
                }
                let color = MASCOT_PALETTE[idx as usize];
                fb.fill_rect(
                    sprite_x + col * SCALE,
                    sprite_y + row * SCALE,
                    SCALE,
                    SCALE,
                    color,
                );
            }
        }

        // Draw outline glow around non-transparent pixels
        for row in 0..32u32 {
            for col in 0..32u32 {
                let idx = MASCOT_32X32[(row * 32 + col) as usize];
                if idx == 0 {
                    continue;
                }
                let is_edge = [(0i32, -1i32), (0, 1), (-1, 0), (1, 0)]
                    .iter()
                    .any(|&(dr, dc)| {
                        let nr = row as i32 + dr;
                        let nc = col as i32 + dc;
                        if nr < 0 || nr >= 32 || nc < 0 || nc >= 32 {
                            return true;
                        }
                        MASCOT_32X32[(nr * 32 + nc) as usize] == 0
                    });
                if is_edge {
                    let px = sprite_x + col * SCALE;
                    let py = sprite_y + row * SCALE;
                    fb.fill_rect(px, py, SCALE, 1, OUTLINE);
                    fb.fill_rect(px, py + SCALE - 1, SCALE, 1, OUTLINE);
                    fb.fill_rect(px, py, 1, SCALE, OUTLINE);
                    fb.fill_rect(px + SCALE - 1, py, 1, SCALE, OUTLINE);
                }
            }
        }

        // Title below mascot
        let title_y = sprite_y + 32 * SCALE + 20;
        self.draw_centered_text(fb, "S E A L   O S", title_y, theme.fg);
        self.draw_centered_text(
            fb,
            "The Geometrical Operating System",
            title_y + CHAR_HEIGHT + 4,
            theme.fg,
        );

        // Input boxes
        let box_w = 240u32;
        let box_h = CHAR_HEIGHT + 12;
        let box_x = (fb.width.saturating_sub(box_w)) / 2;

        // Username field
        let user_label_y = title_y + CHAR_HEIGHT * 3 + 20;
        self.draw_centered_text(fb, "Username:", user_label_y, theme.fg);
        let user_box_y = user_label_y + CHAR_HEIGHT + 10;
        self.draw_input_box(fb, box_x, user_box_y, box_w, box_h, self.active_field == 0);

        let text_padding = 8u32;
        let text_x = box_x + text_padding;
        let text_y = user_box_y + (box_h - CHAR_HEIGHT) / 2;
        for i in 0..self.username_len as u32 {
            super::font::draw_char(
                fb,
                text_x + i * CHAR_WIDTH,
                text_y,
                self.username[i as usize],
                theme.fg,
            );
        }
        if self.active_field == 0 {
            let cursor_x = text_x + (self.username_len as u32) * CHAR_WIDTH;
            fb.fill_rect(cursor_x, text_y, 2, CHAR_HEIGHT, theme.fg);
        }

        // Password field
        let pwd_label_y = user_box_y + box_h + 20;
        self.draw_centered_text(fb, "Password:", pwd_label_y, theme.fg);
        let pwd_box_y = pwd_label_y + CHAR_HEIGHT + 10;
        self.draw_input_box(fb, box_x, pwd_box_y, box_w, box_h, self.active_field == 1);

        let dot_size = 6u32;
        let dot_gap = 12u32;
        let dots_total_w = if self.password_len > 0 {
            (self.password_len as u32) * dot_gap - (dot_gap - dot_size)
        } else {
            0
        };
        let dots_x = box_x + (box_w.saturating_sub(dots_total_w)) / 2;
        let dots_y = pwd_box_y + (box_h - dot_size) / 2;
        for i in 0..self.password_len as u32 {
            fb.fill_rect(dots_x + i * dot_gap, dots_y, dot_size, dot_size, theme.fg);
        }
        if self.active_field == 1 {
            let cursor_x = if self.password_len > 0 {
                dots_x + (self.password_len as u32) * dot_gap
            } else {
                dots_x
            };
            fb.fill_rect(cursor_x, dots_y, 2, dot_size, theme.fg);
        }

        // Error message
        if self.error_len > 0 {
            let err_y = pwd_box_y + box_h + 12;
            let err_text =
                core::str::from_utf8(&self.error_msg[..self.error_len]).unwrap_or("Error");
            self.draw_centered_text(fb, err_text, err_y, theme.close_btn);
        }
    }

    fn draw_input_box(&self, fb: &Framebuffer, x: u32, y: u32, w: u32, h: u32, active: bool) {
        let theme = crate::wm::themes::current_theme();
        let border_color = if active { OUTLINE } else { theme.border };
        fb.fill_rect(x - 2, y - 2, w + 4, h + 4, border_color);
        fb.fill_rect(x, y, w, h, theme.bg);
    }

    fn draw_centered_text(&self, fb: &Framebuffer, text: &str, y: u32, color: u32) {
        let text_w = text.len() as u32 * CHAR_WIDTH;
        let x = (fb.width.saturating_sub(text_w)) / 2;
        for (i, ch) in text.bytes().enumerate() {
            super::font::draw_char(fb, x + (i as u32) * CHAR_WIDTH, y, ch, color);
        }
    }

    pub fn key_press(&mut self, key: u8) -> bool {
        match key {
            b'\t' => {
                self.active_field = if self.active_field == 0 { 1 } else { 0 };
            }
            b'\n' | 13 => {
                let username =
                    core::str::from_utf8(&self.username[..self.username_len]).unwrap_or("");
                let password =
                    core::str::from_utf8(&self.password[..self.password_len]).unwrap_or("");
                if let Some(user) = crate::security::passwd::verify_login(username, password) {
                    self.authenticated_user = Some(user);
                    return true;
                } else {
                    self.set_error("Invalid username or password");
                    self.password_len = 0;
                }
            }
            8 | 127 => {
                if self.active_field == 0 {
                    if self.username_len > 0 {
                        self.username_len -= 1;
                    }
                } else {
                    if self.password_len > 0 {
                        self.password_len -= 1;
                    }
                }
            }
            32..=126 => {
                if self.active_field == 0 {
                    if self.username_len < 32 {
                        self.username[self.username_len] = key;
                        self.username_len += 1;
                    }
                } else {
                    if self.password_len < 32 {
                        self.password[self.password_len] = key;
                        self.password_len += 1;
                    }
                }
            }
            _ => {
                // Unhandled input; no-op
            }
        }
        false
    }

    fn set_error(&mut self, msg: &str) {
        let bytes = msg.as_bytes();
        let len = bytes.len().min(64);
        self.error_msg[..len].copy_from_slice(&bytes[..len]);
        self.error_len = len;
    }

    pub fn authenticated_user(&self) -> Option<&crate::security::passwd::UserEntry> {
        self.authenticated_user.as_ref()
    }
}
