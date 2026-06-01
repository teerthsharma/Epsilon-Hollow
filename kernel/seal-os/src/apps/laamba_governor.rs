// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![allow(dead_code)] // REASON: UI color constants planned for future toolbar styling

//! LAAMBA Governor — native workstation UI for topology-driven ML governance.
//! Renders toolbar, sample bay, topology scope, parameter roll, and console.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::graphics::htek;
use crate::wm::window::Window;

// ── Palette ────────────────────────────────────────────────────────────────
const BG_TOP: u32 = 0x000A0A14;
const BG_BOTTOM: u32 = 0x0005050C;
const PANEL_BG_TOP: u32 = 0x0012121E;
const PANEL_BG_BOT: u32 = 0x000C0C16;
const PANEL_BORDER: u32 = 0x00303050;
const TOOLBAR_TOP: u32 = 0x00202035;
const TOOLBAR_BOT: u32 = 0x0018182A;
const BTN_ACTIVE_TOP: u32 = 0x00304060;
const BTN_ACTIVE_BOT: u32 = 0x00203050;
const BTN_INACTIVE_TOP: u32 = 0x00202035;
const BTN_INACTIVE_BOT: u32 = 0x00181828;
const BTN_BORDER: u32 = 0x00404060;
const TEXT_FG: u32 = 0x00CDD6F4;
const TEXT_DIM: u32 = 0x00606080;
const TEXT_ACCENT: u32 = 0x00FFAA44;
const TEXT_GREEN: u32 = 0x0055FF88;
const TEXT_RED: u32 = 0x00FF5577;
const SELECTED_BG: u32 = 0x002A2A45;
const CONSOLE_BG: u32 = 0x00080810;
const WIREFRAME: u32 = 0x008888AA;

// ── Layout ─────────────────────────────────────────────────────────────────
const TOOLBAR_Y: u32 = 4;
const TOOLBAR_H: u32 = 32;
const LEFT_X: u32 = 4;
const LEFT_Y: u32 = 40;
const LEFT_W: u32 = 180;
const LEFT_H: u32 = 560;
const CENTER_X: u32 = 188;
const CENTER_Y: u32 = 40;
const CENTER_W: u32 = 480;
const CENTER_H: u32 = 400;
const RIGHT_X: u32 = 672;
const RIGHT_Y: u32 = 40;
const RIGHT_W: u32 = 224;
const RIGHT_H: u32 = 400;
const BOTTOM_X: u32 = 188;
const BOTTOM_Y: u32 = 444;
const BOTTOM_W: u32 = 708;
const BOTTOM_H: u32 = 156;

const TOOLS: &[&str] = &["BATTLE", "ANALYZE", "REGRESS", "RANK", "FORMULA", "ENGINES"];
const TOPOLOGIES: &[&str] = &[
    "euclidean",
    "spherical",
    "hyperbolic",
    "grassmannian",
    "product",
];
const DATASETS: &[&str] = &["ring.csv", "tree.csv", "blob.csv", "swiss_roll.csv"];

pub struct LaambaGovernor {
    active_tool: usize,
    selected_dataset: usize,
    selected_topology: usize,
    console_lines: Vec<String>,
    params: GovernorParams,
}

struct GovernorParams {
    manifold: String,
    curvature: String,
    dim: String,
    lr: String,
    batch: String,
    epochs: String,
}

impl LaambaGovernor {
    pub fn new() -> Self {
        let mut console_lines = Vec::new();
        console_lines.push(String::from("[GOVERNOR] LAAMBA Governor v1.0 initialized"));
        console_lines.push(String::from("[GOVERNOR] Topology scope ready"));
        console_lines.push(String::from("[GOVERNOR] Sample bay loaded: 4 datasets"));
        console_lines.push(String::from("[GOVERNOR] Parameter roll: defaults"));

        Self {
            active_tool: 1,
            selected_dataset: 0,
            selected_topology: 0,
            console_lines,
            params: GovernorParams {
                manifold: String::from("S^2"),
                curvature: String::from("+1.000"),
                dim: String::from("2"),
                lr: String::from("0.001"),
                batch: String::from("32"),
                epochs: String::from("100"),
            },
        }
    }

    pub fn key_press(&mut self, ch: u8) {
        match ch {
            b'1'..=b'5' => {
                let idx = (ch - b'1') as usize;
                if idx < TOPOLOGIES.len() {
                    self.selected_topology = idx;
                    self.log(format!("[TOPO] Selected topology: {}", TOPOLOGIES[idx]));
                }
            }
            b'b' | b'B' => {
                self.active_tool = 0;
                self.log(String::from("[BATTLE] Initiating battle royale..."));
            }
            b'r' | b'R' => {
                self.active_tool = 2;
                self.log(String::from("[REGRESS] Starting regression..."));
            }
            _ => {}
        }
    }

    pub fn mouse_click(&mut self, x: u32, y: u32, pressed: bool) {
        if !pressed {
            return;
        }
        let cw = 900u32;

        // Toolbar buttons
        if y >= TOOLBAR_Y && y < TOOLBAR_Y + TOOLBAR_H {
            let gap = 4u32;
            let btn_w = (cw - 8 - gap * (TOOLS.len() as u32 - 1)) / TOOLS.len() as u32;
            for i in 0..TOOLS.len() {
                let bx = 4 + i as u32 * (btn_w + gap);
                if x >= bx && x < bx + btn_w {
                    self.active_tool = i;
                    self.log(format!("[TOOL] Activated: {}", TOOLS[i]));
                    return;
                }
            }
        }

        // Left panel — dataset selection
        if x >= LEFT_X && x < LEFT_X + LEFT_W && y >= LEFT_Y && y < LEFT_Y + LEFT_H {
            let list_y = LEFT_Y + 28;
            let item_h = htek::TEXT_CHAR_H + 6;
            for i in 0..DATASETS.len() {
                let iy = list_y + i as u32 * item_h;
                if y >= iy && y < iy + item_h {
                    self.selected_dataset = i;
                    self.log(format!("[DATA] Loaded dataset: {}", DATASETS[i]));
                    return;
                }
            }
        }

        // Center panel — topology selection via click
        if x >= CENTER_X && x < CENTER_X + CENTER_W && y >= CENTER_Y && y < CENTER_Y + CENTER_H {
            // Placeholder: cycle topology on click
            self.selected_topology = (self.selected_topology + 1) % TOPOLOGIES.len();
            self.log(format!(
                "[TOPO] Cycling topology: {}",
                TOPOLOGIES[self.selected_topology]
            ));
        }
    }

    pub fn mouse_move(&mut self, _x: u32, _y: u32) {}

    pub fn render_to_window(&self, win: &mut Window) {
        let cw = win.client_width();
        let ch = win.client_height();

        // Background
        htek::fill_gradient_v(win, 0, 0, cw, ch, BG_TOP, BG_BOTTOM);

        // ── Toolbar ──────────────────────────────────────────────────────────
        let gap = 4u32;
        let btn_w = (cw - 8 - gap * (TOOLS.len() as u32 - 1)) / TOOLS.len() as u32;
        for (i, &label) in TOOLS.iter().enumerate() {
            let bx = 4 + i as u32 * (btn_w + gap);
            let (top, bot) = if i == self.active_tool {
                (BTN_ACTIVE_TOP, BTN_ACTIVE_BOT)
            } else {
                (BTN_INACTIVE_TOP, BTN_INACTIVE_BOT)
            };
            htek::fill_rounded_rect_gradient(win, bx, TOOLBAR_Y, btn_w, TOOLBAR_H, 6, top, bot);
            htek::stroke_rounded_rect(win, bx, TOOLBAR_Y, btn_w, TOOLBAR_H, 6, 1, BTN_BORDER);
            let text_w = label.len() as u32 * htek::TEXT_CHAR_W;
            let tx = bx + (btn_w.saturating_sub(text_w)) / 2;
            let ty = TOOLBAR_Y + (TOOLBAR_H.saturating_sub(htek::TEXT_CHAR_H)) / 2;
            htek::render_text_small(win, tx, ty, label, TEXT_FG);
        }

        // ── Left panel: Sample Bay ───────────────────────────────────────────
        htek::fill_rounded_rect_gradient(
            win,
            LEFT_X,
            LEFT_Y,
            LEFT_W,
            LEFT_H,
            6,
            PANEL_BG_TOP,
            PANEL_BG_BOT,
        );
        htek::stroke_rounded_rect(win, LEFT_X, LEFT_Y, LEFT_W, LEFT_H, 6, 1, PANEL_BORDER);
        htek::render_text_small(win, LEFT_X + 8, LEFT_Y + 8, "Sample Bay", TEXT_ACCENT);
        let list_y = LEFT_Y + 28;
        let item_h = htek::TEXT_CHAR_H + 6;
        for (i, &name) in DATASETS.iter().enumerate() {
            let iy = list_y + i as u32 * item_h;
            if iy + htek::TEXT_CHAR_H > LEFT_Y + LEFT_H {
                break;
            }
            if i == self.selected_dataset {
                htek::fill_solid(win, LEFT_X + 4, iy, LEFT_W - 8, item_h, SELECTED_BG);
            }
            htek::render_text_small(win, LEFT_X + 12, iy + 3, name, TEXT_FG);
        }

        // ── Center panel: Topology Scope ─────────────────────────────────────
        htek::fill_rounded_rect_gradient(
            win,
            CENTER_X,
            CENTER_Y,
            CENTER_W,
            CENTER_H,
            6,
            PANEL_BG_TOP,
            PANEL_BG_BOT,
        );
        htek::stroke_rounded_rect(
            win,
            CENTER_X,
            CENTER_Y,
            CENTER_W,
            CENTER_H,
            6,
            1,
            PANEL_BORDER,
        );
        let scope_title = format!("3D Viz: {}", TOPOLOGIES[self.selected_topology]);
        htek::render_text_small(win, CENTER_X + 8, CENTER_Y + 8, &scope_title, TEXT_ACCENT);

        // Wireframe box placeholder
        self.draw_wireframe_box(
            win,
            CENTER_X + 40,
            CENTER_Y + 40,
            CENTER_W - 80,
            CENTER_H - 80,
        );

        // ── Right panel: Parameter Roll ──────────────────────────────────────
        htek::fill_rounded_rect_gradient(
            win,
            RIGHT_X,
            RIGHT_Y,
            RIGHT_W,
            RIGHT_H,
            6,
            PANEL_BG_TOP,
            PANEL_BG_BOT,
        );
        htek::stroke_rounded_rect(win, RIGHT_X, RIGHT_Y, RIGHT_W, RIGHT_H, 6, 1, PANEL_BORDER);
        htek::render_text_small(win, RIGHT_X + 8, RIGHT_Y + 8, "Parameter Roll", TEXT_ACCENT);

        let params = [
            ("Manifold", self.params.manifold.as_str()),
            ("Curvature", self.params.curvature.as_str()),
            ("Dim", self.params.dim.as_str()),
            ("LR", self.params.lr.as_str()),
            ("Batch", self.params.batch.as_str()),
            ("Epochs", self.params.epochs.as_str()),
        ];
        let param_y0 = RIGHT_Y + 32;
        let param_h = htek::TEXT_CHAR_H + 8;
        for (i, (label, value)) in params.iter().enumerate() {
            let py = param_y0 + i as u32 * param_h;
            if py + htek::TEXT_CHAR_H > RIGHT_Y + RIGHT_H {
                break;
            }
            let label_str = format!("{:<10}", label);
            htek::render_text_small(win, RIGHT_X + 10, py, &label_str, TEXT_DIM);
            htek::render_text_small(win, RIGHT_X + 90, py, value, TEXT_GREEN);
        }

        // ── Bottom panel: Console ────────────────────────────────────────────
        htek::fill_rounded_rect_gradient(
            win, BOTTOM_X, BOTTOM_Y, BOTTOM_W, BOTTOM_H, 6, CONSOLE_BG, CONSOLE_BG,
        );
        htek::stroke_rounded_rect(
            win,
            BOTTOM_X,
            BOTTOM_Y,
            BOTTOM_W,
            BOTTOM_H,
            6,
            1,
            PANEL_BORDER,
        );
        htek::render_text_small(win, BOTTOM_X + 8, BOTTOM_Y + 6, "Console", TEXT_DIM);
        let line_h = htek::TEXT_CHAR_H + 4;
        let console_y0 = BOTTOM_Y + 22;
        let start = if self.console_lines.len() > 8 {
            self.console_lines.len() - 8
        } else {
            0
        };
        for (i, line) in self.console_lines[start..].iter().enumerate() {
            let cy = console_y0 + i as u32 * line_h;
            if cy + htek::TEXT_CHAR_H > BOTTOM_Y + BOTTOM_H {
                break;
            }
            htek::render_text_small(win, BOTTOM_X + 10, cy, line, TEXT_FG);
        }
    }

    fn draw_wireframe_box(&self, win: &mut Window, x: u32, y: u32, w: u32, h: u32) {
        // Simple perspective wireframe box
        let x0 = x as i32;
        let y0 = y as i32;
        let x1 = (x + w) as i32;
        let y1 = (y + h) as i32;
        let mx = (x0 + x1) / 2;
        let my = (y0 + y1) / 2;
        let off = (w.min(h) / 6) as i32;

        // Front face
        htek::draw_aa_line(win, x0, y0, x1, y0, WIREFRAME, 200);
        htek::draw_aa_line(win, x1, y0, x1, y1, WIREFRAME, 200);
        htek::draw_aa_line(win, x1, y1, x0, y1, WIREFRAME, 200);
        htek::draw_aa_line(win, x0, y1, x0, y0, WIREFRAME, 200);

        // Back face (offset)
        htek::draw_aa_line(win, x0 + off, y0 - off, x1 + off, y0 - off, WIREFRAME, 120);
        htek::draw_aa_line(win, x1 + off, y0 - off, x1 + off, y1 - off, WIREFRAME, 120);
        htek::draw_aa_line(win, x1 + off, y1 - off, x0 + off, y1 - off, WIREFRAME, 120);
        htek::draw_aa_line(win, x0 + off, y1 - off, x0 + off, y0 - off, WIREFRAME, 120);

        // Connecting edges
        htek::draw_aa_line(win, x0, y0, x0 + off, y0 - off, WIREFRAME, 160);
        htek::draw_aa_line(win, x1, y0, x1 + off, y0 - off, WIREFRAME, 160);
        htek::draw_aa_line(win, x1, y1, x1 + off, y1 - off, WIREFRAME, 160);
        htek::draw_aa_line(win, x0, y1, x0 + off, y1 - off, WIREFRAME, 160);

        // Cross lines for "point cloud" feel
        htek::draw_aa_line(win, x0, y0, x1, y1, WIREFRAME, 80);
        htek::draw_aa_line(win, x1, y0, x0, y1, WIREFRAME, 80);
        htek::draw_aa_line(win, mx, y0 - off, mx, y1, WIREFRAME, 60);
        htek::draw_aa_line(win, x0, my, x1 + off, my, WIREFRAME, 60);
    }

    fn log(&mut self, msg: String) {
        self.console_lines.push(msg);
        if self.console_lines.len() > 64 {
            self.console_lines.remove(0);
        }
    }
}

pub fn main() {
    loop {
        crate::process::scheduler::yield_current();
    }
}
