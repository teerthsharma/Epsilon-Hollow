// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! File Manager — visual ManifoldFS browser with S^2 projection view.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::fs::manifold_fs::ManifoldFS;
use crate::graphics::font;
use crate::wm::window::Window;

const BG: u32 = 0x00141420;
const HEADER_BG: u32 = 0x001A1A2E;
const FILE_COLOR: u32 = 0x0089B4FA;
const DIR_COLOR: u32 = 0x00F9E2AF;
const TEXT_FG: u32 = 0x00CDD6F4;
const CELL_COLORS: [u32; 8] = [
    0x00F38BA8, 0x00FAB387, 0x00F9E2AF, 0x00A6E3A1,
    0x0094E2D5, 0x0089B4FA, 0x00CBA6F7, 0x00F5C2E7,
];

pub struct FileManager {
    cwd: u64,
    path: String,
    selected: usize,
}

impl FileManager {
    pub fn new() -> Self {
        Self {
            cwd: 0,
            path: String::from("/"),
            selected: 0,
        }
    }

    pub fn render_to_window(&self, win: &mut Window, fs: &ManifoldFS) {
        let cw = win.client_width();
        let ch = win.client_height();

        // Clear
        for y in 0..ch {
            for x in 0..cw {
                win.set_client_pixel(x, y, BG);
            }
        }

        // Header bar
        for y in 0..24 {
            for x in 0..cw {
                win.set_client_pixel(x, y, HEADER_BG);
            }
        }
        let header = format!("ManifoldFS Browser — {}", self.path);
        render_text(win, 8, 4, &header, TEXT_FG);

        // S^2 projection view (right half)
        let sphere_cx = cw * 3 / 4;
        let sphere_cy = ch / 2;
        let sphere_r = ch.min(cw / 2) / 3;
        render_sphere_view(win, sphere_cx, sphere_cy, sphere_r);

        // File list (left half)
        if let Ok(entries) = fs.ls(self.cwd) {
            for (i, entry) in entries.iter().enumerate() {
                let y = 32 + i as u32 * (font::CHAR_HEIGHT + 4);
                if y + font::CHAR_HEIGHT > ch {
                    break;
                }

                let icon_color = if entry.kind == "dir" {
                    DIR_COLOR
                } else {
                    FILE_COLOR
                };

                // Selection highlight
                if i == self.selected {
                    for x in 4..cw / 2 {
                        win.set_client_pixel(x, y, 0x00252540);
                    }
                }

                // Cell color indicator
                let cell_color = CELL_COLORS[entry.voronoi_cell % 8];
                for dy in 2..14 {
                    for dx in 0..4 {
                        win.set_client_pixel(8 + dx, y + dy, cell_color);
                    }
                }

                // Icon
                let icon = if entry.kind == "dir" { "[D]" } else { "[F]" };
                render_text(win, 16, y, icon, icon_color);

                // Name
                render_text(win, 44, y, &entry.name, TEXT_FG);

                // Size + cell info
                let info = format!("{} pts cell={}", entry.payload_points, entry.voronoi_cell);
                render_text(win, cw / 2 - 120, y, &info, 0x00585B70);
            }

            if entries.is_empty() {
                render_text(win, 20, 40, "(empty directory)", 0x00585B70);
            }
        }
    }
}

fn render_sphere_view(win: &mut Window, cx: u32, cy: u32, r: u32) {
    // Draw stereographic projection of S^2
    let rf = r as f64;

    // Sphere outline
    for angle_i in 0..360 {
        let angle = (angle_i as f64) * core::f64::consts::PI / 180.0;
        let x = cx as f64 + rf * libm::cos(angle);
        let y = cy as f64 + rf * libm::sin(angle);
        win.set_client_pixel(x as u32, y as u32, 0x00404060);
    }

    // Latitude/longitude grid
    for lat in 1..4 {
        let lr = rf * (lat as f64) / 4.0;
        for angle_i in 0..360 {
            let angle = (angle_i as f64) * core::f64::consts::PI / 180.0;
            let x = cx as f64 + lr * libm::cos(angle);
            let y = cy as f64 + lr * libm::sin(angle);
            win.set_client_pixel(x as u32, y as u32, 0x00252535);
        }
    }
    for lon in 0..8 {
        let angle = (lon as f64) * core::f64::consts::PI / 4.0;
        for d in 0..r {
            let x = cx as f64 + d as f64 * libm::cos(angle);
            let y = cy as f64 + d as f64 * libm::sin(angle);
            win.set_client_pixel(x as u32, y as u32, 0x00252535);
        }
    }

    // Voronoi cell labels
    for i in 0..8 {
        let angle = (i as f64) * core::f64::consts::PI / 4.0;
        let x = cx as f64 + rf * 0.6 * libm::cos(angle);
        let y = cy as f64 + rf * 0.6 * libm::sin(angle);
        let color = CELL_COLORS[i];
        // Small dot for each cell
        for dy in 0..4u32 {
            for dx in 0..4u32 {
                win.set_client_pixel(x as u32 + dx, y as u32 + dy, color);
            }
        }
    }
}

fn render_text(win: &mut Window, x: u32, y: u32, text: &str, color: u32) {
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
