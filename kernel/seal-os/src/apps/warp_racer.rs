// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Warp Racer — aether-link prefetch visualization game.
//! Fly through data blocks; green = prefetched, red = cache miss.

use crate::wm::window::Window;

use super::game_engine::{self, SCORE_COLOR};

const BLOCK_W: u32 = 20;
const BLOCK_H: u32 = 16;
const SHIP_W: u32 = 16;
const SHIP_H: u32 = 12;
const SHIP_COLOR: u32 = 0x0088CCFF;
const PREFETCH_HIT: u32 = 0x0044CC44;
const PREFETCH_MISS: u32 = 0x00CC4444;
const TUNNEL_COLOR: u32 = 0x00202040;

pub struct WarpRacer {
    width: u32,
    height: u32,
    ship_x: i32,
    ship_y: i32,
    scroll_offset: u32,
    score: u32,
    hits: u32,
    misses: u32,
    epsilon: f64,
    blocks: [(u32, u32, bool); 64],
    game_over: bool,
    rng_state: u32,
    tick_count: u64,
}

impl WarpRacer {
    pub fn new(client_w: u32, client_h: u32) -> Self {
        let mut game = Self {
            width: client_w,
            height: client_h,
            ship_x: 20,
            ship_y: (client_h / 2) as i32,
            scroll_offset: 0,
            score: 0,
            hits: 0,
            misses: 0,
            epsilon: 0.40,
            blocks: [(0, 0, false); 64],
            game_over: false,
            rng_state: 12345,
            tick_count: 0,
        };
        game.generate_blocks();
        game
    }

    fn generate_blocks(&mut self) {
        for i in 0..64 {
            let x = game_engine::simple_rng(&mut self.rng_state) % self.width;
            let y = i as u32 * 30 + 50;
            let prefetched = (game_engine::simple_rng(&mut self.rng_state) % 100) < 73;
            self.blocks[i] = (x, y, prefetched);
        }
    }

    pub fn key_press(&mut self, key: u8) {
        if self.game_over {
            return;
        }
        match key {
            b'w' | 0x48 => self.ship_y = (self.ship_y - 8).max(0),
            b's' | 0x50 => self.ship_y = (self.ship_y + 8).min(self.height as i32 - SHIP_H as i32),
            b'a' | 0x4B => self.ship_x = (self.ship_x - 8).max(0),
            b'd' | 0x4D => self.ship_x = (self.ship_x + 8).min(self.width as i32 - SHIP_W as i32),
            _ => {
                // Unhandled input; no-op
            }
        }
    }

    pub fn tick(&mut self) {
        if self.game_over {
            return;
        }
        self.tick_count += 1;
        if self.tick_count % 2 != 0 {
            return;
        }

        self.scroll_offset += 2;
        self.score += 1;

        self.epsilon = 0.40 + 0.01 * libm::sin(self.tick_count as f64 * 0.01);

        let ship_x = self.ship_x as u32;
        for &(bx, by, prefetched) in &self.blocks {
            let screen_y = by as i32 - self.scroll_offset as i32;
            if screen_y > self.ship_y - BLOCK_H as i32
                && screen_y < self.ship_y + SHIP_H as i32
                && bx < ship_x + SHIP_W
                && bx + BLOCK_W > ship_x
            {
                if prefetched {
                    self.hits += 1;
                    self.score += 5;
                } else {
                    self.misses += 1;
                }
            }
        }

        if self.scroll_offset > 64 * 30 {
            self.game_over = true;
        }
    }

    pub fn render_to_window(&self, win: &mut Window) {
        game_engine::clear_window(win);

        // Tunnel walls
        for y in 0..self.height {
            win.set_client_pixel(0, y, TUNNEL_COLOR);
            win.set_client_pixel(self.width - 1, y, TUNNEL_COLOR);
        }

        // Status
        let total = self.hits + self.misses;
        let ratio = if total > 0 {
            self.hits as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        let status = alloc::format!(
            "Score:{} Hits:{} Miss:{} Ratio:{:.0}% e={:.3}",
            self.score,
            self.hits,
            self.misses,
            ratio,
            self.epsilon
        );
        game_engine::render_text(win, 4, 2, &status, SCORE_COLOR);

        // Blocks
        for &(bx, by, prefetched) in &self.blocks {
            let screen_y = by as i32 - self.scroll_offset as i32;
            if screen_y >= 0 && screen_y < self.height as i32 {
                let color = if prefetched {
                    PREFETCH_HIT
                } else {
                    PREFETCH_MISS
                };
                game_engine::fill_rect(win, bx, screen_y as u32, BLOCK_W, BLOCK_H, color);
            }
        }

        // Ship
        game_engine::fill_rect(
            win,
            self.ship_x as u32,
            self.ship_y as u32,
            SHIP_W,
            SHIP_H,
            SHIP_COLOR,
        );

        if self.game_over {
            game_engine::render_text(
                win,
                self.width / 2 - 60,
                self.height / 2,
                "WARP COMPLETE!",
                0x0088CCFF,
            );
        }
    }

    pub fn mouse_click(&mut self, _x: u32, _y: u32, _pressed: bool) {}
    pub fn mouse_move(&mut self, _x: u32, _y: u32) {}
}

pub fn main() {
    loop {
        crate::process::scheduler::yield_current();
    }
}
