// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Desktop application state manager — owns all running apps, routes input events
//! to the focused window, ticks games, and re-renders dirty windows.

use crate::apps::breakout::BreakoutGame;
use crate::apps::calculator::Calculator;
use crate::apps::file_manager::FileManager;
use crate::apps::media_player::MediaPlayer;
use crate::apps::seal_ide::SealIde;
use crate::apps::snake::SnakeGame;
use crate::apps::terminal::Terminal;
use crate::apps::theorem_viewer::TheoremViewer;
use crate::apps::warp_racer::WarpRacer;
use crate::drivers::interrupts::{scancode_to_char, scancode_to_special, SpecialKey};
use crate::fs::manifold_fs::ManifoldFS;
use crate::wm::compositor::Compositor;
use crate::wm::event::InputEvent;

pub struct AppState {
    pub terminal: Terminal,
    pub ide: SealIde,
    pub theorem_viewer: TheoremViewer,
    pub calculator: Calculator,
    pub media_player: MediaPlayer,
    pub file_manager: FileManager,
    pub snake: SnakeGame,
    pub breakout: BreakoutGame,
    pub warp_racer: WarpRacer,
    pub fs: ManifoldFS,

    // Window IDs (assigned by compositor)
    pub term_id: u32,
    pub ide_id: u32,
    pub tv_id: u32,
    pub calc_id: u32,
    pub player_id: u32,

    // Track which windows are dirty and need re-render
    dirty: [bool; 5],
}

#[derive(Clone, Copy, PartialEq)]
pub enum AppId {
    Terminal,
    Ide,
    TheoremViewer,
    Calculator,
    MediaPlayer,
}

impl AppState {
    pub fn new() -> Self {
        let mut fs = ManifoldFS::new();

        // Set up initial filesystem content
        let vol_a = fs.mkdir("vol_a", 0).unwrap_or(1);
        let _vol_b = fs.mkdir("vol_b", 0).unwrap_or(2);
        let docs = fs.mkdir("docs", 0).unwrap_or(3);
        fs.store_text("hello.txt", "Hello from Seal OS!", vol_a).ok();
        fs.store_text("kernel.rs", "fn _start() -> ! { loop {} }", vol_a).ok();
        fs.store_text("readme.txt", "Welcome to Seal OS", docs).ok();
        fs.store_text("notes.txt", "Topological file surgery", docs).ok();

        Self {
            terminal: Terminal::new(600, 400),
            ide: SealIde::new(),
            theorem_viewer: TheoremViewer::new(),
            calculator: Calculator::new(),
            media_player: MediaPlayer::new(),
            file_manager: FileManager::new(),
            snake: SnakeGame::new(400, 380),
            breakout: BreakoutGame::new(400, 380),
            warp_racer: WarpRacer::new(400, 380),
            fs,
            term_id: 0,
            ide_id: 0,
            tv_id: 0,
            calc_id: 0,
            player_id: 0,
            dirty: [true; 5],
        }
    }

    pub fn create_windows(&mut self, compositor: &mut Compositor) {
        self.term_id = compositor.create_window("Terminal", 40, 40, 600, 400);
        self.ide_id = compositor.create_window("Seal IDE", 120, 80, 640, 440);
        self.tv_id = compositor.create_window("Theorems", 500, 60, 420, 340);
        self.calc_id = compositor.create_window("Calculator", 680, 100, 280, 440);
        self.player_id = compositor.create_window("SealPlayer", 200, 160, 520, 380);

        // Initial render of all windows
        self.render_all(compositor);
    }

    pub fn handle_event(&mut self, event: InputEvent, compositor: &mut Compositor) {
        match event {
            InputEvent::KeyPress(scancode) => {
                self.handle_key(scancode, compositor);
            }
            InputEvent::KeyRelease(_) => {}
            InputEvent::MouseMove { dx, dy } => {
                compositor.mouse_move(dx, dy, 1024, 768);
            }
            InputEvent::MouseButton { button: 0, pressed } => {
                compositor.mouse_click(pressed);
            }
            InputEvent::MouseButton { .. } => {}
            InputEvent::Scroll { .. } => {}
        }
    }

    fn handle_key(&mut self, scancode: u8, compositor: &mut Compositor) {
        let focused_id = compositor.focused_window_id();

        if focused_id == self.term_id {
            if let Some(special) = scancode_to_special(scancode) {
                match special {
                    SpecialKey::Up => self.terminal.key_press(0x48),
                    SpecialKey::Down => self.terminal.key_press(0x50),
                    _ => {}
                }
            } else {
                let ch = scancode_to_char(scancode);
                if ch != 0 {
                    self.terminal.key_press(ch);
                }
            }
            self.dirty[0] = true;
        } else if focused_id == self.ide_id {
            if let Some(special) = scancode_to_special(scancode) {
                self.ide.handle_special_key(special);
            } else {
                let ch = scancode_to_char(scancode);
                if ch != 0 {
                    self.ide.key_press(ch);
                }
            }
            self.dirty[1] = true;
        } else if focused_id == self.calc_id {
            let ch = scancode_to_char(scancode);
            if ch != 0 {
                self.calculator.key_press(ch);
            }
            self.dirty[3] = true;
        } else if focused_id == self.player_id {
            let ch = scancode_to_char(scancode);
            if ch != 0 {
                self.media_player.key_press(ch);
            }
            self.dirty[4] = true;
        }
    }

    pub fn tick(&mut self) {
        self.snake.tick();
        self.breakout.tick();
        self.warp_racer.tick();
    }

    pub fn render_dirty(&mut self, compositor: &mut Compositor) {
        if self.dirty[0] {
            if let Some(win) = compositor.window_mut(self.term_id) {
                self.terminal.render_to_window(win);
            }
            self.dirty[0] = false;
        }
        if self.dirty[1] {
            if let Some(win) = compositor.window_mut(self.ide_id) {
                self.ide.render_to_window(win);
            }
            self.dirty[1] = false;
        }
        if self.dirty[2] {
            if let Some(win) = compositor.window_mut(self.tv_id) {
                self.theorem_viewer.render_to_window(win);
            }
            self.dirty[2] = false;
        }
        if self.dirty[3] {
            if let Some(win) = compositor.window_mut(self.calc_id) {
                self.calculator.render_to_window(win);
            }
            self.dirty[3] = false;
        }
        if self.dirty[4] {
            if let Some(win) = compositor.window_mut(self.player_id) {
                self.media_player.render_to_window(win);
            }
            self.dirty[4] = false;
        }
    }

    fn render_all(&mut self, compositor: &mut Compositor) {
        if let Some(win) = compositor.window_mut(self.term_id) {
            self.terminal.render_to_window(win);
        }
        if let Some(win) = compositor.window_mut(self.ide_id) {
            self.ide.render_to_window(win);
        }
        if let Some(win) = compositor.window_mut(self.tv_id) {
            self.theorem_viewer.render_to_window(win);
        }
        if let Some(win) = compositor.window_mut(self.calc_id) {
            self.calculator.render_to_window(win);
        }
        if let Some(win) = compositor.window_mut(self.player_id) {
            self.media_player.render_to_window(win);
        }
    }

    pub fn mark_all_dirty(&mut self) {
        self.dirty = [true; 5];
    }
}
