// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Terminal emulator — renders shell output into a window buffer.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::shell::Shell;
use crate::graphics::font;
use crate::wm::window::Window;

const FG: u32 = 0x00D0D0D8;
const BG: u32 = 0x00101018;
const PROMPT_COLOR: u32 = 0x004488CC;
const CURSOR_COLOR: u32 = 0x00A0A0B0;

pub struct Terminal {
    shell: Shell,
    input_buf: String,
    lines: Vec<String>,
    scroll_offset: usize,
    cursor_visible: bool,
    cols: u32,
    rows: u32,
}

impl Terminal {
    pub fn new(client_w: u32, client_h: u32) -> Self {
        let cols = client_w / font::CHAR_WIDTH;
        let rows = client_h / font::CHAR_HEIGHT;
        let mut term = Self {
            shell: Shell::new(),
            input_buf: String::new(),
            lines: Vec::new(),
            scroll_offset: 0,
            cursor_visible: true,
            cols,
            rows,
        };
        term.lines.push(String::from(
            "Seal OS Terminal v1.0 — Type 'help' for commands",
        ));
        term.lines.push(String::new());
        term
    }

    pub fn key_press(&mut self, ch: u8) {
        match ch {
            b'\n' => {
                let prompt = self.shell.prompt();
                self.lines
                    .push(alloc::format!("{}{}", prompt, self.input_buf));

                let output = self.shell.execute(&self.input_buf);
                self.input_buf.clear();

                if output == "\x1b[clear]" {
                    self.lines.clear();
                } else if !output.is_empty() {
                    for line in output.lines() {
                        self.lines.push(String::from(line));
                    }
                }

                // Auto-scroll
                if self.lines.len() > self.rows as usize {
                    self.scroll_offset = self.lines.len() - self.rows as usize;
                }
            }
            0x08 | 0x7F => {
                // Backspace
                self.input_buf.pop();
            }
            0x09 => {
                // Tab — add spaces
                self.input_buf.push_str("    ");
            }
            ch if ch >= 0x20 && ch < 0x7F => {
                self.input_buf.push(ch as char);
            }
            _ => {}
        }
    }

    pub fn render_to_window(&self, win: &mut Window) {
        let cw = win.client_width();
        let ch = win.client_height();

        // Clear client area
        for y in 0..ch {
            for x in 0..cw {
                win.set_client_pixel(x, y, BG);
            }
        }

        // Render visible lines
        let visible_start = self.scroll_offset;
        let visible_end = (visible_start + self.rows as usize).min(self.lines.len());

        for (row, line) in self.lines[visible_start..visible_end].iter().enumerate() {
            self.render_line(win, row as u32, line, FG);
        }

        // Render current input line
        let input_row = (visible_end - visible_start) as u32;
        if input_row < self.rows {
            let prompt = self.shell.prompt();
            let full = alloc::format!("{}{}", prompt, self.input_buf);
            // Prompt in blue
            self.render_line_colored(win, input_row, &prompt, PROMPT_COLOR);
            // Input after prompt
            let input_x = (prompt.len() as u32) * font::CHAR_WIDTH;
            self.render_text_at(win, input_x, input_row * font::CHAR_HEIGHT, &self.input_buf, FG);

            // Cursor
            if self.cursor_visible {
                let cx = (prompt.len() + self.input_buf.len()) as u32 * font::CHAR_WIDTH;
                let cy = input_row * font::CHAR_HEIGHT;
                for dy in 0..font::CHAR_HEIGHT {
                    win.set_client_pixel(cx, cy + dy, CURSOR_COLOR);
                    win.set_client_pixel(cx + 1, cy + dy, CURSOR_COLOR);
                }
            }
        }
    }

    fn render_line(&self, win: &mut Window, row: u32, text: &str, color: u32) {
        self.render_text_at(win, 0, row * font::CHAR_HEIGHT, text, color);
    }

    fn render_line_colored(&self, win: &mut Window, row: u32, text: &str, color: u32) {
        self.render_text_at(win, 0, row * font::CHAR_HEIGHT, text, color);
    }

    fn render_text_at(&self, win: &mut Window, x: u32, y: u32, text: &str, color: u32) {
        for (i, ch) in text.bytes().enumerate() {
            let glyph = font::glyph(ch);
            let px = x + (i as u32) * font::CHAR_WIDTH;
            for gy in 0..font::CHAR_HEIGHT {
                let bits = glyph[gy as usize];
                for gx in 0..font::CHAR_WIDTH {
                    if bits & (0x80 >> gx) != 0 {
                        win.set_client_pixel(px + gx, y + gy, color);
                    }
                }
            }
        }
    }

    pub fn shell_mut(&mut self) -> &mut Shell {
        &mut self.shell
    }
}
