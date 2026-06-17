// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! First-run welcome wizard with theme picker and capability summary.

use core::sync::atomic::{AtomicBool, Ordering};

use crate::graphics::font::{CHAR_HEIGHT, CHAR_WIDTH};
use crate::graphics::framebuffer::Framebuffer;
use crate::wm::event::InputEvent;
use crate::wm::themes;

static FIRST_BOOT_DONE: AtomicBool = AtomicBool::new(false);

pub fn mark_first_boot_done() {
    FIRST_BOOT_DONE.store(true, Ordering::Relaxed);
}

pub fn is_first_boot() -> bool {
    !FIRST_BOOT_DONE.load(Ordering::Relaxed)
}

const TITLE_TEXT: &str = "Welcome to Seal OS v0.4.6";
const SUBTITLE_TEXT: &str = "The Geometrical Operating System";

const CAPABILITY_LINES: &[&str] = &[
    "Real: context switch, e1000 NIC, SMP, userspace init",
    "Coming: disk persistence, USB, GPU acceleration",
];

const THEME_NAMES: &[&str] = &["dark", "light", "seal", "matrix"];

const LARGE_SCALE: u32 = 2;
const LARGE_CHAR_WIDTH: u32 = CHAR_WIDTH * LARGE_SCALE;
const LARGE_CHAR_HEIGHT: u32 = CHAR_HEIGHT * LARGE_SCALE;

pub struct WelcomeScreen {
    mouse_x: i32,
    mouse_y: i32,
    fb_width: u32,
    fb_height: u32,
    selected_theme: &'static str,
}

impl WelcomeScreen {
    pub const fn new() -> Self {
        Self {
            mouse_x: 512,
            mouse_y: 384,
            fb_width: 1024,
            fb_height: 768,
            selected_theme: "dark",
        }
    }

    pub fn render(&mut self, fb: &Framebuffer) {
        self.fb_width = fb.width;
        self.fb_height = fb.height;

        let theme = themes::current_theme();
        fb.clear(theme.bg);

        // Title (large, centered)
        let title_w = TITLE_TEXT.len() as u32 * LARGE_CHAR_WIDTH;
        let title_x = (fb.width.saturating_sub(title_w)) / 2;
        let title_y = 50u32;
        self.draw_large_text(fb, title_x, title_y, TITLE_TEXT, theme.accent);

        // Subtitle
        let subtitle_y = title_y + LARGE_CHAR_HEIGHT + 12;
        self.draw_centered_text(fb, SUBTITLE_TEXT, subtitle_y, theme.fg);

        // Theme picker label
        let picker_label_y = subtitle_y + CHAR_HEIGHT + 30;
        self.draw_centered_text(fb, "Select a theme:", picker_label_y, theme.fg);

        // Theme buttons
        let btn_w = 80u32;
        let btn_h = 50u32;
        let btn_gap = 12u32;
        let total_w = THEME_NAMES.len() as u32 * btn_w + (THEME_NAMES.len() as u32 - 1) * btn_gap;
        let start_x = (fb.width.saturating_sub(total_w)) / 2;
        let btn_y = picker_label_y + CHAR_HEIGHT + 16;

        for (i, &name) in THEME_NAMES.iter().enumerate() {
            let bx = start_x + i as u32 * (btn_w + btn_gap);
            let t = themes::get_theme(name).unwrap_or(&themes::DARK);

            // Button background
            fb.fill_rect(bx, btn_y, btn_w, btn_h, t.bg);

            // Color preview strip at bottom
            let strip_h = 8u32;
            fb.fill_rect(bx, btn_y + btn_h - strip_h, btn_w, strip_h, t.accent);

            // Button border (highlight if selected)
            let border_color = if name == self.selected_theme {
                theme.accent
            } else {
                theme.border
            };
            // Top
            fb.fill_rect(bx, btn_y, btn_w, 2, border_color);
            // Bottom
            fb.fill_rect(bx, btn_y + btn_h - 2, btn_w, 2, border_color);
            // Left
            fb.fill_rect(bx, btn_y, 2, btn_h, border_color);
            // Right
            fb.fill_rect(bx + btn_w - 2, btn_y, 2, btn_h, border_color);

            // Theme name
            let label = match name {
                "dark" => "Dark",
                "light" => "Light",
                "seal" => "Seal",
                "matrix" => "Matrix",
                _ => name,
            };
            let text_w = label.len() as u32 * CHAR_WIDTH;
            let text_x = bx + (btn_w.saturating_sub(text_w)) / 2;
            let text_y = btn_y + (btn_h.saturating_sub(CHAR_HEIGHT)) / 2 - 4;
            self.draw_text(fb, text_x, text_y, label, t.fg);
        }

        // Capability summary
        let summary_y = btn_y + btn_h + 40;
        for (i, line) in CAPABILITY_LINES.iter().enumerate() {
            let y = summary_y + i as u32 * (CHAR_HEIGHT + 6);
            self.draw_centered_text(fb, line, y, theme.fg);
        }

        // "Get Started" button
        let gs_w = 160u32;
        let gs_h = 36u32;
        let gs_x = (fb.width.saturating_sub(gs_w)) / 2;
        let gs_y = fb.height.saturating_sub(gs_h + 60);

        // Button background
        fb.fill_rect(gs_x, gs_y, gs_w, gs_h, theme.accent);
        // Button border
        fb.fill_rect(gs_x, gs_y, gs_w, 2, theme.fg);
        fb.fill_rect(gs_x, gs_y + gs_h - 2, gs_w, 2, theme.fg);
        fb.fill_rect(gs_x, gs_y, 2, gs_h, theme.fg);
        fb.fill_rect(gs_x + gs_w - 2, gs_y, 2, gs_h, theme.fg);

        let gs_text = "Get Started";
        let gs_text_w = gs_text.len() as u32 * CHAR_WIDTH;
        let gs_text_x = gs_x + (gs_w.saturating_sub(gs_text_w)) / 2;
        let gs_text_y = gs_y + (gs_h.saturating_sub(CHAR_HEIGHT)) / 2;
        self.draw_text(fb, gs_text_x, gs_text_y, gs_text, theme.bg);
    }

    pub fn handle_event(&mut self, event: InputEvent) -> bool {
        match event {
            InputEvent::KeyPress(key) => {
                if key == b'\n' || key == 13 {
                    return true;
                }
            }
            InputEvent::MouseMove { dx, dy } => {
                let max_x = self.fb_width.saturating_sub(1) as i32;
                let max_y = self.fb_height.saturating_sub(1) as i32;
                self.mouse_x = (self.mouse_x + dx).clamp(0, max_x);
                self.mouse_y = (self.mouse_y + dy).clamp(0, max_y);
            }
            InputEvent::MouseButton {
                button: 1,
                pressed: true,
            } => {
                let mx = self.mouse_x as u32;
                let my = self.mouse_y as u32;

                // Hit-test theme buttons
                let btn_w = 80u32;
                let btn_h = 50u32;
                let btn_gap = 12u32;
                let total_w =
                    THEME_NAMES.len() as u32 * btn_w + (THEME_NAMES.len() as u32 - 1) * btn_gap;
                let start_x = (self.fb_width.saturating_sub(total_w)) / 2;
                let subtitle_y = 50 + LARGE_CHAR_HEIGHT + 12;
                let picker_label_y = subtitle_y + CHAR_HEIGHT + 30;
                let btn_y = picker_label_y + CHAR_HEIGHT + 16;

                for (i, &name) in THEME_NAMES.iter().enumerate() {
                    let bx = start_x + i as u32 * (btn_w + btn_gap);
                    if mx >= bx && mx < bx + btn_w && my >= btn_y && my < btn_y + btn_h {
                        self.selected_theme = name;
                        themes::set_theme(name);
                        return false;
                    }
                }

                // Hit-test "Get Started" button
                let gs_w = 160u32;
                let gs_h = 36u32;
                let gs_x = (self.fb_width.saturating_sub(gs_w)) / 2;
                let gs_y = self.fb_height.saturating_sub(gs_h + 60);
                if mx >= gs_x && mx < gs_x + gs_w && my >= gs_y && my < gs_y + gs_h {
                    return true;
                }
            }
            _ => {
                // Unhandled input; no-op
            }
        }
        false
    }

    fn draw_centered_text(&self, fb: &Framebuffer, text: &str, y: u32, color: u32) {
        let text_w = text.len() as u32 * CHAR_WIDTH;
        let x = (fb.width.saturating_sub(text_w)) / 2;
        self.draw_text(fb, x, y, text, color);
    }

    fn draw_text(&self, fb: &Framebuffer, x: u32, y: u32, text: &str, color: u32) {
        for (i, ch) in text.bytes().enumerate() {
            crate::graphics::font::draw_char(fb, x + (i as u32) * CHAR_WIDTH, y, ch, color);
        }
    }

    fn draw_large_char(&self, fb: &Framebuffer, x: u32, y: u32, ch: u8, color: u32) {
        let g = crate::graphics::font::glyph(ch);
        for row in 0..CHAR_HEIGHT {
            let bits = g[row as usize];
            for col in 0..CHAR_WIDTH {
                if bits & (0x80 >> col) != 0 {
                    fb.fill_rect(
                        x + col * LARGE_SCALE,
                        y + row * LARGE_SCALE,
                        LARGE_SCALE,
                        LARGE_SCALE,
                        color,
                    );
                }
            }
        }
    }

    fn draw_large_text(&self, fb: &Framebuffer, x: u32, y: u32, text: &str, color: u32) {
        for (i, ch) in text.bytes().enumerate() {
            self.draw_large_char(fb, x + (i as u32) * LARGE_CHAR_WIDTH, y, ch, color);
        }
    }
}
