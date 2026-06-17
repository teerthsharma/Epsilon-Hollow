// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Seal IDE — built-in development environment.
//! Features: syntax highlighting, file explorer, deterministic code completion.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::graphics::font;
use crate::wm::window::Window;

const BG: u32 = 0x001E1E2E;
const SIDEBAR_BG: u32 = 0x00181825;
const GUTTER_BG: u32 = 0x00181825;
const LINE_NUM: u32 = 0x00505060;
const TEXT_FG: u32 = 0x00CDD6F4;
const KEYWORD: u32 = 0x00CBA6F7; // purple
const STRING_LIT: u32 = 0x00A6E3A1; // green
const COMMENT: u32 = 0x00585B70; // gray
const FN_NAME: u32 = 0x0089B4FA; // blue
const STATUS_BG: u32 = 0x00313244;
const STATUS_FG: u32 = 0x00BAC2DE;

const SIDEBAR_WIDTH: u32 = 160;
const GUTTER_WIDTH: u32 = 40;

const COMPLETION_CANDIDATES: &[&str] = &[
    "SphericalVoronoiIndex",
    "GeometricGovernor",
    "serial_println!",
    "kernel_main",
    "epsilon",
    "voronoi",
    "governor",
    "locate",
    "loop",
    "unsafe",
    "crate::",
    "alloc::",
];

pub struct SealIde {
    files: Vec<IdeFile>,
    active_file: usize,
    cursor_line: usize,
    cursor_col: usize,
    sidebar_files: Vec<String>,
}

struct IdeFile {
    name: String,
    lines: Vec<String>,
}

impl SealIde {
    pub fn new() -> Self {
        let demo_file = IdeFile {
            name: String::from("main.rs"),
            lines: vec![
                String::from("// Seal OS — The Geometrical Operating System"),
                String::from("// All data = geometry on S^2"),
                String::from(""),
                String::from("#![no_std]"),
                String::from("#![no_main]"),
                String::from(""),
                String::from("use aether_core::tss::SphericalVoronoiIndex;"),
                String::from("use aether_core::governor::GeometricGovernor;"),
                String::from(""),
                String::from("fn kernel_main() -> ! {"),
                String::from("    let governor = GeometricGovernor::new();"),
                String::from("    let voronoi = SphericalVoronoiIndex::<8>::new(centroids);"),
                String::from(""),
                String::from("    // T1-T10 gate verified; T1-T5 active in kernel"),
                String::from("    let cell = voronoi.locate((0.5, 0.5));"),
                String::from("    let epsilon = governor.epsilon();"),
                String::from(""),
                String::from("    serial_println!(\"Seal OS ready. epsilon={}\", epsilon);"),
                String::from(""),
                String::from("    loop { x86_64::instructions::hlt(); }"),
                String::from("}"),
            ],
        };

        Self {
            files: vec![demo_file],
            active_file: 0,
            cursor_line: 9,
            cursor_col: 0,
            sidebar_files: vec![
                String::from("main.rs"),
                String::from("manifold_fs.rs"),
                String::from("encoder.rs"),
                String::from("scheduler.rs"),
                String::from("compositor.rs"),
            ],
        }
    }

    pub fn key_press(&mut self, ch: u8) {
        let file = match self.files.get_mut(self.active_file) {
            Some(f) => f,
            None => return,
        };

        match ch {
            b'\n' => {
                let line = file
                    .lines
                    .get(self.cursor_line)
                    .cloned()
                    .unwrap_or_default();
                let (before, after) = if self.cursor_col <= line.len() {
                    (
                        String::from(&line[..self.cursor_col]),
                        String::from(&line[self.cursor_col..]),
                    )
                } else {
                    (line.clone(), String::new())
                };
                file.lines[self.cursor_line] = before;
                self.cursor_line += 1;
                file.lines.insert(self.cursor_line, after);
                self.cursor_col = 0;
            }
            0x08 => {
                // Backspace
                if self.cursor_col > 0 {
                    if let Some(line) = file.lines.get_mut(self.cursor_line) {
                        if self.cursor_col <= line.len() {
                            line.remove(self.cursor_col - 1);
                        }
                    }
                    self.cursor_col -= 1;
                } else if self.cursor_line > 0 {
                    let removed = file.lines.remove(self.cursor_line);
                    self.cursor_line -= 1;
                    self.cursor_col = file.lines[self.cursor_line].len();
                    file.lines[self.cursor_line].push_str(&removed);
                }
            }
            b'\t' => {
                if let Some(suffix) =
                    completion_suffix_for_line(file.lines.get(self.cursor_line), self.cursor_col)
                {
                    if let Some(line) = file.lines.get_mut(self.cursor_line) {
                        line.insert_str(self.cursor_col, suffix);
                        self.cursor_col += suffix.len();
                    }
                } else if let Some(line) = file.lines.get_mut(self.cursor_line) {
                    line.insert_str(self.cursor_col, "    ");
                    self.cursor_col += 4;
                }
            }
            ch if ch >= 0x20 && ch < 0x7F => {
                if let Some(line) = file.lines.get_mut(self.cursor_line) {
                    if self.cursor_col > line.len() {
                        self.cursor_col = line.len();
                    }
                    line.insert(self.cursor_col, ch as char);
                    self.cursor_col += 1;
                }
            }
            _ => {}
        }
    }

    pub fn handle_special_key(&mut self, key: crate::drivers::interrupts::SpecialKey) {
        use crate::drivers::interrupts::SpecialKey;
        let line_count = self
            .files
            .get(self.active_file)
            .map(|f| f.lines.len())
            .unwrap_or(0);

        match key {
            SpecialKey::Up => {
                if self.cursor_line > 0 {
                    self.cursor_line -= 1;
                    let len = self
                        .files
                        .get(self.active_file)
                        .map(|f| f.lines[self.cursor_line].len())
                        .unwrap_or(0);
                    self.cursor_col = self.cursor_col.min(len);
                }
            }
            SpecialKey::Down => {
                if self.cursor_line + 1 < line_count {
                    self.cursor_line += 1;
                    let len = self
                        .files
                        .get(self.active_file)
                        .map(|f| f.lines[self.cursor_line].len())
                        .unwrap_or(0);
                    self.cursor_col = self.cursor_col.min(len);
                }
            }
            SpecialKey::Left => {
                if self.cursor_col > 0 {
                    self.cursor_col -= 1;
                }
            }
            SpecialKey::Right => {
                let len = self
                    .files
                    .get(self.active_file)
                    .map(|f| f.lines[self.cursor_line].len())
                    .unwrap_or(0);
                if self.cursor_col < len {
                    self.cursor_col += 1;
                }
            }
            SpecialKey::Home => self.cursor_col = 0,
            SpecialKey::End => {
                self.cursor_col = self
                    .files
                    .get(self.active_file)
                    .map(|f| f.lines[self.cursor_line].len())
                    .unwrap_or(0);
            }
            SpecialKey::Delete => {
                if let Some(file) = self.files.get_mut(self.active_file) {
                    if let Some(line) = file.lines.get_mut(self.cursor_line) {
                        if self.cursor_col < line.len() {
                            line.remove(self.cursor_col);
                        } else if self.cursor_line + 1 < file.lines.len() {
                            let next = file.lines.remove(self.cursor_line + 1);
                            file.lines[self.cursor_line].push_str(&next);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    pub fn render_to_window(&self, win: &mut Window) {
        let cw = win.client_width();
        let ch = win.client_height();

        // Clear
        for y in 0..ch {
            for x in 0..cw {
                win.set_client_pixel(x, y, BG);
            }
        }

        // Sidebar
        for y in 0..ch.saturating_sub(20) {
            for x in 0..SIDEBAR_WIDTH.min(cw) {
                win.set_client_pixel(x, y, SIDEBAR_BG);
            }
        }

        // Sidebar files
        for (i, name) in self.sidebar_files.iter().enumerate() {
            let y = 8 + i as u32 * (font::CHAR_HEIGHT + 4);
            if y + font::CHAR_HEIGHT > ch {
                break;
            }
            let color = if i == self.active_file {
                TEXT_FG
            } else {
                COMMENT
            };
            self.render_text(win, 12, y, name, color);
        }

        // Editor area
        if let Some(file) = self.files.get(self.active_file) {
            let editor_x = SIDEBAR_WIDTH + 2;
            let status_h = 20u32;
            let editor_h = ch.saturating_sub(status_h);

            // Line numbers + code
            for (i, line) in file.lines.iter().enumerate() {
                let y = i as u32 * font::CHAR_HEIGHT;
                if y + font::CHAR_HEIGHT > editor_h {
                    break;
                }

                // Gutter
                for x in editor_x..(editor_x + GUTTER_WIDTH).min(cw) {
                    win.set_client_pixel(x, y, GUTTER_BG);
                }

                // Line number
                let num = format!("{:>3}", i + 1);
                self.render_text(win, editor_x + 4, y, &num, LINE_NUM);

                // Syntax-highlighted code
                let code_x = editor_x + GUTTER_WIDTH;
                self.render_highlighted_line(win, code_x, y, line, cw);

                // Cursor
                if i == self.cursor_line {
                    let cx = code_x + (self.cursor_col as u32) * font::CHAR_WIDTH;
                    for dy in 0..font::CHAR_HEIGHT {
                        if cx < cw {
                            win.set_client_pixel(cx, y + dy, TEXT_FG);
                        }
                    }
                }
            }

            // Status bar
            let sy = ch.saturating_sub(status_h);
            for y in sy..ch {
                for x in editor_x..cw {
                    win.set_client_pixel(x, y, STATUS_BG);
                }
            }
            let status = format!(
                " {} | Ln {}, Col {} | Rust | T1-T10 OK | {}",
                file.name,
                self.cursor_line + 1,
                self.cursor_col + 1,
                self.completion_status()
            );
            self.render_text(win, editor_x + 4, sy + 3, &status, STATUS_FG);
        }
    }

    fn completion_status(&self) -> String {
        let line = self
            .files
            .get(self.active_file)
            .and_then(|file| file.lines.get(self.cursor_line));
        match completion_suffix_for_line(line, self.cursor_col) {
            Some(suffix) => format!("Tab completes {}", suffix),
            None => String::from("Seal IDE"),
        }
    }

    fn render_highlighted_line(&self, win: &mut Window, x: u32, y: u32, line: &str, max_x: u32) {
        let trimmed = line.trim_start();

        // Simple keyword-based highlighting
        if trimmed.starts_with("//") {
            self.render_text(win, x, y, line, COMMENT);
            return;
        }

        let keywords = [
            "fn", "let", "mut", "use", "pub", "mod", "struct", "enum", "impl", "for", "loop", "if",
            "else", "match", "return", "const", "static", "unsafe", "extern", "crate",
        ];

        let mut col = 0u32;
        let mut chars = line.chars().peekable();
        let mut word = String::new();
        let mut in_string = false;

        while let Some(ch) = chars.next() {
            let px = x + col * font::CHAR_WIDTH;
            if px >= max_x {
                break;
            }

            if ch == '"' {
                in_string = !in_string;
                self.render_char(win, px, y, ch as u8, STRING_LIT);
                col += 1;
                continue;
            }

            if in_string {
                self.render_char(win, px, y, ch as u8, STRING_LIT);
                col += 1;
                continue;
            }

            if ch.is_alphanumeric() || ch == '_' || ch == '!' || ch == '#' {
                word.push(ch);
            } else {
                if !word.is_empty() {
                    let color = if keywords.contains(&word.as_str()) {
                        KEYWORD
                    } else if word.ends_with('!') {
                        FN_NAME
                    } else {
                        TEXT_FG
                    };
                    for wch in word.bytes() {
                        let _wpx = x
                            + (col - word.len() as u32
                                + word.bytes().position(|b| b == wch).unwrap_or(0) as u32)
                                * font::CHAR_WIDTH;
                        // Simplified: just render the whole word
                    }
                    // Re-render word with color
                    let start_col = col - word.len() as u32;
                    for (i, wch) in word.bytes().enumerate() {
                        let wpx = x + (start_col + i as u32) * font::CHAR_WIDTH;
                        if wpx < max_x {
                            self.render_char(win, wpx, y, wch, color);
                        }
                    }
                    word.clear();
                }
                self.render_char(win, px, y, ch as u8, TEXT_FG);
                col += 1;
                continue;
            }
            col += 1;
        }

        // Flush remaining word
        if !word.is_empty() {
            let color = if keywords.contains(&word.as_str()) {
                KEYWORD
            } else {
                TEXT_FG
            };
            let start_col = col - word.len() as u32;
            for (i, wch) in word.bytes().enumerate() {
                let wpx = x + (start_col + i as u32) * font::CHAR_WIDTH;
                if wpx < max_x {
                    self.render_char(win, wpx, y, wch, color);
                }
            }
        }
    }

    fn render_text(&self, win: &mut Window, x: u32, y: u32, text: &str, color: u32) {
        for (i, ch) in text.bytes().enumerate() {
            let px = x + (i as u32) * font::CHAR_WIDTH;
            self.render_char(win, px, y, ch, color);
        }
    }

    fn render_char(&self, win: &mut Window, x: u32, y: u32, ch: u8, color: u32) {
        let glyph = font::glyph(ch);
        for gy in 0..font::CHAR_HEIGHT {
            let bits = glyph[gy as usize];
            for gx in 0..font::CHAR_WIDTH {
                if bits & (0x80 >> gx) != 0 {
                    win.set_client_pixel(x + gx, y + gy, color);
                }
            }
        }
    }

    pub fn mouse_click(&mut self, _x: u32, _y: u32, _pressed: bool) {}
    pub fn mouse_move(&mut self, _x: u32, _y: u32) {}
}

fn completion_suffix_for_line(line: Option<&String>, cursor_col: usize) -> Option<&'static str> {
    let prefix = current_word_prefix(line?, cursor_col);
    if prefix.len() < 2 {
        return None;
    }
    for candidate in COMPLETION_CANDIDATES {
        if candidate.starts_with(prefix.as_str()) && candidate.len() > prefix.len() {
            return candidate.get(prefix.len()..);
        }
    }
    None
}

fn current_word_prefix(line: &str, cursor_col: usize) -> String {
    let end = cursor_col.min(line.len());
    let mut start = end;
    let bytes = line.as_bytes();
    while start > 0 {
        let b = bytes[start - 1];
        if b.is_ascii_alphanumeric() || b == b'_' || b == b'!' || b == b':' {
            start -= 1;
        } else {
            break;
        }
    }
    String::from(&line[start..end])
}

pub fn main() {
    loop {
        crate::process::scheduler::yield_current();
    }
}
