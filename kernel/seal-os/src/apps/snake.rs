// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Snake game — classic snake + apple, rendered into a compositor window.

use alloc::vec::Vec;

use crate::wm::window::Window;

use super::game_engine::{self, SCORE_COLOR};

const CELL_SIZE: i32 = 12;
const SNAKE_COLOR: u32 = 0x0044CC88;
const APPLE_COLOR: u32 = 0x00CC4444;
const BORDER_COLOR: u32 = 0x00404060;

#[derive(Clone, Copy, PartialEq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

pub struct SnakeGame {
    grid_w: i32,
    grid_h: i32,
    snake: Vec<(i32, i32)>,
    direction: Direction,
    apple: (i32, i32),
    score: u32,
    game_over: bool,
    paused: bool,
    rng_state: u32,
    tick_count: u64,
}

impl SnakeGame {
    pub fn new(client_w: u32, client_h: u32) -> Self {
        let grid_w = (client_w as i32) / CELL_SIZE;
        let grid_h = (client_h as i32 - 20) / CELL_SIZE;
        let mid_x = grid_w / 2;
        let mid_y = grid_h / 2;

        let mut game = Self {
            grid_w,
            grid_h,
            snake: Vec::new(),
            direction: Direction::Right,
            apple: (0, 0),
            score: 0,
            game_over: false,
            paused: false,
            rng_state: 42,
            tick_count: 0,
        };

        game.snake.push((mid_x, mid_y));
        game.snake.push((mid_x - 1, mid_y));
        game.snake.push((mid_x - 2, mid_y));
        game.spawn_apple();
        game
    }

    pub fn key_press(&mut self, key: u8) {
        if self.game_over {
            return;
        }
        match key {
            b'w' | 0x48 => {
                if self.direction != Direction::Down {
                    self.direction = Direction::Up;
                }
            }
            b's' | 0x50 => {
                if self.direction != Direction::Up {
                    self.direction = Direction::Down;
                }
            }
            b'a' | 0x4B => {
                if self.direction != Direction::Right {
                    self.direction = Direction::Left;
                }
            }
            b'd' | 0x4D => {
                if self.direction != Direction::Left {
                    self.direction = Direction::Right;
                }
            }
            b' ' => self.paused = !self.paused,
            _ => {}
        }
    }

    pub fn tick(&mut self) {
        if self.game_over || self.paused {
            return;
        }
        self.tick_count += 1;
        if self.tick_count % 5 != 0 {
            return;
        }

        let (hx, hy) = self.snake[0];
        let (nx, ny) = match self.direction {
            Direction::Up => (hx, hy - 1),
            Direction::Down => (hx, hy + 1),
            Direction::Left => (hx - 1, hy),
            Direction::Right => (hx + 1, hy),
        };

        if nx < 0 || nx >= self.grid_w || ny < 0 || ny >= self.grid_h {
            self.game_over = true;
            return;
        }
        if self.snake.iter().any(|&(sx, sy)| sx == nx && sy == ny) {
            self.game_over = true;
            return;
        }

        self.snake.insert(0, (nx, ny));

        if nx == self.apple.0 && ny == self.apple.1 {
            self.score += 10;
            self.spawn_apple();
        } else {
            self.snake.pop();
        }
    }

    fn spawn_apple(&mut self) {
        loop {
            let x = (game_engine::simple_rng(&mut self.rng_state) as i32).abs() % self.grid_w;
            let y = (game_engine::simple_rng(&mut self.rng_state) as i32).abs() % self.grid_h;
            if !self.snake.iter().any(|&(sx, sy)| sx == x && sy == y) {
                self.apple = (x, y);
                break;
            }
        }
    }

    pub fn render_to_window(&self, win: &mut Window) {
        game_engine::clear_window(win);

        let score_text = alloc::format!("Score: {}", self.score);
        game_engine::render_text(win, 4, 2, &score_text, SCORE_COLOR);

        let offset_y = 20;

        for gy in 0..self.grid_h {
            for gx in 0..self.grid_w {
                let px = (gx * CELL_SIZE) as u32;
                let py = (gy * CELL_SIZE) as u32 + offset_y;
                if gx == 0 || gx == self.grid_w - 1 || gy == 0 || gy == self.grid_h - 1 {
                    game_engine::fill_rect(
                        win,
                        px,
                        py,
                        CELL_SIZE as u32,
                        CELL_SIZE as u32,
                        BORDER_COLOR,
                    );
                }
            }
        }

        let (ax, ay) = self.apple;
        game_engine::fill_rect(
            win,
            (ax * CELL_SIZE + 1) as u32,
            (ay * CELL_SIZE + 1) as u32 + offset_y,
            (CELL_SIZE - 2) as u32,
            (CELL_SIZE - 2) as u32,
            APPLE_COLOR,
        );

        for (i, &(sx, sy)) in self.snake.iter().enumerate() {
            let color = if i == 0 { 0x0066EEAA } else { SNAKE_COLOR };
            game_engine::fill_rect(
                win,
                (sx * CELL_SIZE + 1) as u32,
                (sy * CELL_SIZE + 1) as u32 + offset_y,
                (CELL_SIZE - 2) as u32,
                (CELL_SIZE - 2) as u32,
                color,
            );
        }

        if self.game_over {
            game_engine::render_text(win, 4, win.client_height() / 2, "GAME OVER!", 0x00FF4444);
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
