// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Breakout game — paddle, ball, and bricks.

use crate::wm::window::Window;

use super::game_engine::{self, SCORE_COLOR};

const PADDLE_W: u32 = 80;
const PADDLE_H: u32 = 10;
const PADDLE_COLOR: u32 = 0x004488CC;
const BALL_SIZE: u32 = 8;
const BALL_COLOR: u32 = 0x00DDDDEE;
const BRICK_W: u32 = 48;
const BRICK_H: u32 = 14;
const BRICK_COLS: u32 = 10;
const BRICK_ROWS: u32 = 5;
const BRICK_GAP: u32 = 4;
const BRICK_TOP: u32 = 30;

const BRICK_COLORS: [u32; 5] = [
    0x00CC4444, 0x00CC8844, 0x00CCCC44, 0x0044CC44, 0x004488CC,
];

pub struct BreakoutGame {
    width: u32,
    height: u32,
    paddle_x: i32,
    ball_x: i32,
    ball_y: i32,
    ball_dx: i32,
    ball_dy: i32,
    bricks: [[bool; BRICK_COLS as usize]; BRICK_ROWS as usize],
    score: u32,
    lives: u32,
    game_over: bool,
    won: bool,
    tick_count: u64,
}

impl BreakoutGame {
    pub fn new(client_w: u32, client_h: u32) -> Self {
        let game = Self {
            width: client_w,
            height: client_h,
            paddle_x: (client_w / 2 - PADDLE_W / 2) as i32,
            ball_x: (client_w / 2) as i32,
            ball_y: (client_h - 40) as i32,
            ball_dx: 0,
            ball_dy: 0,
            bricks: [[true; BRICK_COLS as usize]; BRICK_ROWS as usize],
            score: 0,
            lives: 3,
            game_over: false,
            won: false,
            tick_count: 0,
        };
        game
    }

    pub fn key_press(&mut self, key: u8) {
        if self.game_over {
            return;
        }
        match key {
            b'a' | 0x4B => {
                self.paddle_x = (self.paddle_x - 16).max(0);
            }
            b'd' | 0x4D => {
                self.paddle_x = (self.paddle_x + 16).min((self.width - PADDLE_W) as i32);
            }
            b' ' => {
                if self.ball_dy == 0 {
                    self.ball_dy = -2;
                    if self.ball_dx == 0 {
                        self.ball_dx = 2;
                    }
                }
            }
            _ => {}
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

        self.ball_x += self.ball_dx;
        self.ball_y += self.ball_dy;

        // Wall bounce
        if self.ball_x <= 0 || self.ball_x >= (self.width - BALL_SIZE) as i32 {
            self.ball_dx = -self.ball_dx;
        }
        if self.ball_y <= 0 {
            self.ball_dy = -self.ball_dy;
        }

        // Bottom — lose life
        if self.ball_y >= self.height as i32 {
            self.lives -= 1;
            if self.lives == 0 {
                self.game_over = true;
                return;
            }
            self.ball_x = (self.width / 2) as i32;
            self.ball_y = (self.height - 40) as i32;
            self.ball_dy = -2;
        }

        // Paddle bounce
        if self.ball_dy > 0
            && self.ball_y + BALL_SIZE as i32 >= (self.height - 20) as i32
            && self.ball_x + BALL_SIZE as i32 >= self.paddle_x
            && self.ball_x <= self.paddle_x + PADDLE_W as i32
        {
            self.ball_dy = -self.ball_dy;
            let hit_pos = self.ball_x - self.paddle_x;
            let center = PADDLE_W as i32 / 2;
            self.ball_dx = (hit_pos - center) / 8;
            if self.ball_dx == 0 {
                self.ball_dx = 1;
            }
        }

        // Brick collision
        let brick_area_w = BRICK_COLS * (BRICK_W + BRICK_GAP);
        let brick_start_x = (self.width - brick_area_w) / 2;

        for row in 0..BRICK_ROWS as usize {
            for col in 0..BRICK_COLS as usize {
                if !self.bricks[row][col] {
                    continue;
                }
                let bx = brick_start_x as i32 + (col as i32) * (BRICK_W + BRICK_GAP) as i32;
                let by = BRICK_TOP as i32 + (row as i32) * (BRICK_H + BRICK_GAP) as i32;

                if self.ball_x + BALL_SIZE as i32 >= bx
                    && self.ball_x <= bx + BRICK_W as i32
                    && self.ball_y + BALL_SIZE as i32 >= by
                    && self.ball_y <= by + BRICK_H as i32
                {
                    self.bricks[row][col] = false;
                    self.ball_dy = -self.ball_dy;
                    self.score += (BRICK_ROWS as u32 - row as u32) * 10;
                }
            }
        }

        // Win check
        let remaining: usize = self.bricks.iter().flatten().filter(|&&b| b).count();
        if remaining == 0 {
            self.game_over = true;
            self.won = true;
        }
    }

    pub fn render_to_window(&self, win: &mut Window) {
        game_engine::clear_window(win);

        let status = alloc::format!("Score: {}  Lives: {}", self.score, self.lives);
        game_engine::render_text(win, 4, 2, &status, SCORE_COLOR);

        // Bricks
        let brick_area_w = BRICK_COLS * (BRICK_W + BRICK_GAP);
        let brick_start_x = (self.width - brick_area_w) / 2;

        for row in 0..BRICK_ROWS as usize {
            for col in 0..BRICK_COLS as usize {
                if !self.bricks[row][col] {
                    continue;
                }
                let bx = brick_start_x + (col as u32) * (BRICK_W + BRICK_GAP);
                let by = BRICK_TOP + (row as u32) * (BRICK_H + BRICK_GAP);
                game_engine::fill_rect(win, bx, by, BRICK_W, BRICK_H, BRICK_COLORS[row]);
            }
        }

        // Paddle
        game_engine::fill_rect(
            win,
            self.paddle_x as u32,
            self.height - 20,
            PADDLE_W,
            PADDLE_H,
            PADDLE_COLOR,
        );

        // Ball
        if !self.game_over {
            game_engine::fill_rect(
                win,
                self.ball_x as u32,
                self.ball_y as u32,
                BALL_SIZE,
                BALL_SIZE,
                BALL_COLOR,
            );
        }

        if self.game_over {
            let msg = if self.won { "YOU WIN!" } else { "GAME OVER" };
            game_engine::render_text(win, self.width / 2 - 40, self.height / 2, msg, 0x00FF4444);
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
