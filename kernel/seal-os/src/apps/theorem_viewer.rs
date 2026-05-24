// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Real-time theorem activity dashboard — T1-T5 live state visualization.

use alloc::format;

use crate::graphics::font;
use crate::wm::window::Window;

const BG: u32 = 0x00101018;
const HEADER_FG: u32 = 0x00CDD6F4;
const ACTIVE_COLOR: u32 = 0x0044CC66;
const LABEL_FG: u32 = 0x00A0A0B0;
const VALUE_FG: u32 = 0x0089B4FA;
const BAR_BG: u32 = 0x00252535;
const BAR_FG: u32 = 0x00CBA6F7;

pub struct TheoremViewer {
    pub epsilon: f64,
    pub betti_0: usize,
    pub prefetch_accuracy: f64,
    pub entropy: f64,
    pub hyperbolic_ratio: f64,
    pub teleports: u64,
    pub lookups: u64,
    pub schedule_count: u64,
}

impl TheoremViewer {
    pub fn new() -> Self {
        Self {
            epsilon: 0.042,
            betti_0: 3,
            prefetch_accuracy: 75.0,
            entropy: 1.234,
            hyperbolic_ratio: 4.2,
            teleports: 0,
            lookups: 0,
            schedule_count: 0,
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

        let mut y = 8u32;

        // Title
        render_text(win, 8, y, "Seal OS Theorem Dashboard", HEADER_FG);
        y += font::CHAR_HEIGHT + 8;

        // Separator
        for x in 8..cw.saturating_sub(8) {
            win.set_client_pixel(x, y, 0x00303040);
        }
        y += 8;

        // T1: TSS (Voronoi)
        render_theorem(win, 8, y, "T1/TSS", "Topological State Space",
            &format!("Voronoi: 8 cells, {} lookups, O(1)", self.lookups),
            1.0, cw);
        y += 50;

        // T2: SCM (Prefetch)
        render_theorem(win, 8, y, "T2/SCM", "Spectral Contraction Map",
            &format!("Prefetch accuracy: {:.0}%, rho=0.700", self.prefetch_accuracy),
            self.prefetch_accuracy / 100.0, cw);
        y += 50;

        // T3: GMC (Entropy)
        render_theorem(win, 8, y, "T3/GMC", "Geometric Measure Convergence",
            &format!("Entropy: H={:.3} bits, Betti-0={}", self.entropy, self.betti_0),
            (self.entropy / 3.0).min(1.0), cw);
        y += 50;

        // T4: AGCR (Governor)
        render_theorem(win, 8, y, "T4/AGCR", "Adaptive Geometric Convergence Rate",
            &format!("epsilon={:.4}, {} scheduler ticks", self.epsilon, self.schedule_count),
            self.epsilon.min(1.0), cw);
        y += 50;

        // T5: HCS (Hyperbolic)
        render_theorem(win, 8, y, "T5/HCS", "Hyperbolic Curvature Scaling",
            &format!("ratio={:.1}x, {} teleports", self.hyperbolic_ratio, self.teleports),
            (self.hyperbolic_ratio / 10.0).min(1.0), cw);
    }
}

fn render_theorem(win: &mut Window, x: u32, y: u32, id: &str, name: &str,
                  detail: &str, bar_value: f64, max_w: u32) {
    // Active indicator dot
    for dy in 0..8u32 {
        for dx in 0..8u32 {
            win.set_client_pixel(x + dx, y + 3 + dy, ACTIVE_COLOR);
        }
    }

    // Theorem ID
    render_text(win, x + 12, y, id, VALUE_FG);
    render_text(win, x + 80, y, name, LABEL_FG);

    // Detail
    render_text(win, x + 12, y + font::CHAR_HEIGHT + 2, detail, LABEL_FG);

    // Progress bar
    let bar_x = x + 12;
    let bar_y = y + font::CHAR_HEIGHT * 2 + 4;
    let bar_w = (max_w - 32).min(400);
    let bar_h = 6u32;

    for bx in 0..bar_w {
        for by in 0..bar_h {
            win.set_client_pixel(bar_x + bx, bar_y + by, BAR_BG);
        }
    }

    let fill = ((bar_w as f64) * bar_value.clamp(0.0, 1.0)) as u32;
    for bx in 0..fill {
        for by in 0..bar_h {
            win.set_client_pixel(bar_x + bx, bar_y + by, BAR_FG);
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

impl TheoremViewer {
    pub fn mouse_click(&mut self, _x: u32, _y: u32, _pressed: bool) {}
    pub fn mouse_move(&mut self, _x: u32, _y: u32) {}
}

pub fn main() {
    loop {
        crate::process::scheduler::yield_current();
    }
}
