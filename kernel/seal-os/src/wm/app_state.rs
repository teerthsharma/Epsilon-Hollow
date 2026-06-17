// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Desktop application state manager — owns all running apps, routes input events
//! to the focused window, ticks games, and re-renders dirty windows.

use crate::apps::aether_app_host::AetherAppHost;
use crate::apps::breakout::BreakoutGame;
use crate::apps::calculator::Calculator;
use crate::apps::file_manager::FileManager;
use crate::apps::laamba_governor::LaambaGovernor;
use crate::apps::media_player::MediaPlayer;
use crate::apps::seal_ide::SealIde;
use crate::apps::snake::SnakeGame;
use crate::apps::tensor_viewer::TensorViewer;
use crate::apps::terminal::Terminal;
use crate::apps::theorem_viewer::TheoremViewer;
use crate::apps::warp_racer::WarpRacer;
use crate::drivers::interrupts::{scancode_to_char, scancode_to_special, SpecialKey};
use crate::fs::manifold_fs::ManifoldFS;
use crate::wm::compositor::Compositor;
use crate::wm::event::InputEvent;
use crate::wm::window::WindowState;

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
    pub tensor_viewer: TensorViewer,
    pub laamba: LaambaGovernor,
    pub aether_app_host: AetherAppHost,
    pub fs: ManifoldFS,

    // Window IDs (assigned by compositor)
    pub term_id: u32,
    pub ide_id: u32,
    pub tv_id: u32,
    pub calc_id: u32,
    pub player_id: u32,
    pub snake_id: u32,
    pub breakout_id: u32,
    pub warp_racer_id: u32,
    pub fm_id: u32,
    pub tensor_id: u32,
    pub laamba_id: u32,
    pub aether_id: u32,

    // Track which windows are dirty and need re-render
    dirty: [bool; 12],
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
        fs.store_text("hello.txt", "Hello from Seal OS!", vol_a)
            .ok();
        fs.store_text("kernel.rs", "fn _start() -> ! { loop {} }", vol_a)
            .ok();
        fs.store_text("readme.txt", "Welcome to Seal OS", docs).ok();
        fs.store_text("notes.txt", "Topological file surgery", docs)
            .ok();

        let terminal = Terminal::new(600, 400);
        let ide = SealIde::new();
        let theorem_viewer = TheoremViewer::new();
        let calculator = Calculator::new();
        let media_player = MediaPlayer::new();
        let file_manager = FileManager::new();
        let snake = SnakeGame::new(400, 380);
        let breakout = BreakoutGame::new(400, 380);
        let warp_racer = WarpRacer::new(400, 380);
        let tensor_viewer = TensorViewer::new();
        let laamba = LaambaGovernor::new();
        let aether_app_host = AetherAppHost::new();

        Self {
            terminal,
            ide,
            theorem_viewer,
            calculator,
            media_player,
            file_manager,
            snake,
            breakout,
            warp_racer,
            tensor_viewer,
            laamba,
            aether_app_host,
            fs,
            term_id: 0,
            ide_id: 0,
            tv_id: 0,
            calc_id: 0,
            player_id: 0,
            snake_id: 0,
            breakout_id: 0,
            warp_racer_id: 0,
            fm_id: 0,
            tensor_id: 0,
            laamba_id: 0,
            aether_id: 0,
            dirty: [true; 12],
        }
    }

    pub fn create_windows(&mut self, compositor: &mut Compositor) {
        self.term_id = compositor.create_window("Terminal", 48, 132, 600, 400);
        self.ide_id = compositor.create_window("Seal IDE", 120, 80, 640, 440);
        self.tv_id = compositor.create_window("Theorems", 500, 60, 460, 560);
        self.calc_id = compositor.create_window("Calculator", 680, 100, 280, 440);
        self.player_id = compositor.create_window("SealPlayer", 200, 160, 520, 380);
        self.snake_id = compositor.create_window("Snake", 100, 100, 400, 380);
        self.breakout_id = compositor.create_window("Breakout", 150, 150, 400, 380);
        self.warp_racer_id = compositor.create_window("Warp Racer", 200, 200, 400, 380);
        self.fm_id = compositor.create_window("Files", 160, 120, 560, 400);
        self.tensor_id = compositor.create_window("Tensor Viewer", 180, 140, 520, 360);
        self.laamba_id = compositor.create_window("LAAMBA Governor", 80, 40, 900, 650);
        self.aether_id = compositor.create_window("Aether App", 200, 200, 640, 480);

        self.minimize_secondary_boot_apps(compositor);
        if let Some(win) = compositor.window_mut(self.term_id) {
            self.terminal.render_to_window(win);
        }
        self.dirty = [false; 12];
        self.dirty[0] = true;
    }

    fn minimize_secondary_boot_apps(&mut self, compositor: &mut Compositor) {
        let ids = [
            self.ide_id,
            self.tv_id,
            self.calc_id,
            self.player_id,
            self.snake_id,
            self.breakout_id,
            self.warp_racer_id,
            self.fm_id,
            self.tensor_id,
            self.laamba_id,
            self.aether_id,
        ];

        for id in ids {
            if let Some(win) = compositor.window_mut(id) {
                win.state = WindowState::Minimized;
                win.focused = false;
                win.dirty = false;
            }
        }
    }

    pub fn handle_event(&mut self, event: InputEvent, compositor: &mut Compositor) {
        match event {
            InputEvent::KeyPress(scancode) => {
                self.handle_key(scancode, compositor);
            }
            InputEvent::KeyRelease(_) => {}
            InputEvent::MouseMove { dx, dy } => {
                compositor.mouse_move(dx, dy, 1024, 768);
                self.route_mouse_move(compositor);
            }
            InputEvent::MouseButton { button: 0, pressed } => {
                compositor.mouse_click(pressed);
                self.route_mouse_click(compositor, pressed);
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
                    _ => {
                // Unhandled input; no-op
            }
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
        } else if focused_id == self.tv_id {
            self.dirty[2] = true;
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
        } else if focused_id == self.snake_id {
            send_key_to_game(scancode, |k| self.snake.key_press(k));
            self.dirty[5] = true;
        } else if focused_id == self.breakout_id {
            send_key_to_game(scancode, |k| self.breakout.key_press(k));
            self.dirty[6] = true;
        } else if focused_id == self.warp_racer_id {
            send_key_to_game(scancode, |k| self.warp_racer.key_press(k));
            self.dirty[7] = true;
        } else if focused_id == self.fm_id {
            self.dirty[8] = true;
        } else if focused_id == self.tensor_id {
            self.dirty[9] = true;
        } else if focused_id == self.laamba_id {
            let ch = scancode_to_char(scancode);
            if ch != 0 {
                self.laamba.key_press(ch);
            }
            self.dirty[10] = true;
        } else if focused_id == self.aether_id {
            let ch = scancode_to_char(scancode);
            if ch != 0 {
                self.aether_app_host.key_press(ch);
            }
            self.dirty[11] = true;
        }
    }

    pub fn tick(&mut self) {
        self.snake.tick();
        self.breakout.tick();
        self.warp_racer.tick();
        self.tensor_viewer.tick();
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
        if self.dirty[5] {
            if let Some(win) = compositor.window_mut(self.snake_id) {
                self.snake.render_to_window(win);
            }
            self.dirty[5] = false;
        }
        if self.dirty[6] {
            if let Some(win) = compositor.window_mut(self.breakout_id) {
                self.breakout.render_to_window(win);
            }
            self.dirty[6] = false;
        }
        if self.dirty[7] {
            if let Some(win) = compositor.window_mut(self.warp_racer_id) {
                self.warp_racer.render_to_window(win);
            }
            self.dirty[7] = false;
        }
        if self.dirty[8] {
            if let Some(win) = compositor.window_mut(self.fm_id) {
                let fs = &self.fs;
                self.file_manager.render_to_window(win, fs);
            }
            self.dirty[8] = false;
        }
        if self.dirty[9] {
            if let Some(win) = compositor.window_mut(self.tensor_id) {
                self.tensor_viewer.render_to_window(win);
            }
            self.dirty[9] = false;
        }
        if self.dirty[10] {
            if let Some(win) = compositor.window_mut(self.laamba_id) {
                self.laamba.render_to_window(win);
            }
            self.dirty[10] = false;
        }
        if self.dirty[11] {
            self.aether_app_host.render_window(self.aether_id);
            self.dirty[11] = false;
        }
    }

    pub fn mark_all_dirty(&mut self) {
        self.dirty = [true; 12];
    }

    pub fn focus_app(&mut self, app_id: u8, compositor: &mut Compositor) {
        let win_id = match app_id {
            1 => self.term_id,
            2 => self.ide_id,
            3 => self.fm_id,
            4 => self.calc_id,
            5 => self.tv_id,
            6 => self.snake_id,
            7 => self.breakout_id,
            8 => self.warp_racer_id,
            9 => self.tensor_id,
            10 => self.laamba_id,
            11 => self.aether_id,
            _ => return,
        };
        if win_id == 0 {
            return;
        }
        compositor.focus_window(win_id);
        match app_id {
            1 => self.dirty[0] = true,
            2 => self.dirty[1] = true,
            3 => self.dirty[8] = true,
            4 => self.dirty[3] = true,
            5 => self.dirty[2] = true,
            6 => self.dirty[5] = true,
            7 => self.dirty[6] = true,
            8 => self.dirty[7] = true,
            9 => self.dirty[9] = true,
            10 => self.dirty[10] = true,
            11 => self.dirty[11] = true,
            _ => {
                // Unhandled input; no-op
            }
        }
    }

    fn route_mouse_move(&mut self, compositor: &mut Compositor) {
        let focused_id = compositor.focused_window_id();
        if focused_id == 0 {
            return;
        }
        let mx = compositor.mouse_state().x as u32;
        let my = compositor.mouse_state().y as u32;
        if let Some(win) = compositor.window_mut(focused_id) {
            let cx = win.client_x();
            let cy = win.client_y();
            let cw = win.client_width();
            let ch = win.client_height();
            if mx >= cx && mx < cx + cw && my >= cy && my < cy + ch {
                let local_x = mx - cx;
                let local_y = my - cy;
                if focused_id == self.term_id {
                    self.terminal.mouse_move(local_x, local_y);
                    self.dirty[0] = true;
                } else if focused_id == self.ide_id {
                    self.ide.mouse_move(local_x, local_y);
                    self.dirty[1] = true;
                } else if focused_id == self.tv_id {
                    self.theorem_viewer.mouse_move(local_x, local_y);
                    self.dirty[2] = true;
                } else if focused_id == self.calc_id {
                    self.calculator.mouse_move(local_x, local_y);
                    self.dirty[3] = true;
                } else if focused_id == self.player_id {
                    self.media_player.mouse_move(local_x, local_y);
                    self.dirty[4] = true;
                } else if focused_id == self.snake_id {
                    self.snake.mouse_move(local_x, local_y);
                    self.dirty[5] = true;
                } else if focused_id == self.breakout_id {
                    self.breakout.mouse_move(local_x, local_y);
                    self.dirty[6] = true;
                } else if focused_id == self.warp_racer_id {
                    self.warp_racer.mouse_move(local_x, local_y);
                    self.dirty[7] = true;
                } else if focused_id == self.fm_id {
                    self.dirty[8] = true;
                } else if focused_id == self.tensor_id {
                    self.dirty[9] = true;
                } else if focused_id == self.laamba_id {
                    self.laamba.mouse_move(local_x, local_y);
                    self.dirty[10] = true;
                } else if focused_id == self.aether_id {
                    self.aether_app_host.mouse_move(local_x, local_y);
                    self.dirty[11] = true;
                }
            }
        }
    }

    fn route_mouse_click(&mut self, compositor: &mut Compositor, pressed: bool) {
        let focused_id = compositor.focused_window_id();
        if focused_id == 0 {
            return;
        }
        let mx = compositor.mouse_state().x as u32;
        let my = compositor.mouse_state().y as u32;
        if let Some(win) = compositor.window_mut(focused_id) {
            let cx = win.client_x();
            let cy = win.client_y();
            let cw = win.client_width();
            let ch = win.client_height();
            if mx >= cx && mx < cx + cw && my >= cy && my < cy + ch {
                let local_x = mx - cx;
                let local_y = my - cy;
                if focused_id == self.term_id {
                    self.terminal.mouse_click(local_x, local_y, pressed);
                    self.dirty[0] = true;
                } else if focused_id == self.ide_id {
                    self.ide.mouse_click(local_x, local_y, pressed);
                    self.dirty[1] = true;
                } else if focused_id == self.tv_id {
                    self.theorem_viewer.mouse_click(local_x, local_y, pressed);
                    self.dirty[2] = true;
                } else if focused_id == self.calc_id {
                    self.calculator.mouse_click(local_x, local_y, pressed);
                    self.dirty[3] = true;
                } else if focused_id == self.player_id {
                    self.media_player.mouse_click(local_x, local_y, pressed);
                    self.dirty[4] = true;
                } else if focused_id == self.snake_id {
                    self.snake.mouse_click(local_x, local_y, pressed);
                    self.dirty[5] = true;
                } else if focused_id == self.breakout_id {
                    self.breakout.mouse_click(local_x, local_y, pressed);
                    self.dirty[6] = true;
                } else if focused_id == self.warp_racer_id {
                    self.warp_racer.mouse_click(local_x, local_y, pressed);
                    self.dirty[7] = true;
                } else if focused_id == self.fm_id {
                    if pressed {
                        if let Some(msg) = self.file_manager.click(local_x, local_y, &self.fs) {
                            crate::serial_println!("{}", msg);
                        }
                    }
                    self.dirty[8] = true;
                } else if focused_id == self.tensor_id {
                    self.dirty[9] = true;
                } else if focused_id == self.laamba_id {
                    self.laamba.mouse_click(local_x, local_y, pressed);
                    self.dirty[10] = true;
                } else if focused_id == self.aether_id {
                    self.aether_app_host.mouse_click(local_x, local_y, pressed);
                    self.dirty[11] = true;
                }
            }
        }
    }
}

fn send_key_to_game(scancode: u8, mut key_press: impl FnMut(u8)) {
    let ch = if scancode == 0x39 {
        b' '
    } else {
        scancode_to_char(scancode)
    };
    if ch != 0 {
        key_press(ch);
    }
    if let Some(special) = scancode_to_special(scancode) {
        let raw = match special {
            SpecialKey::Up => 0x48,
            SpecialKey::Down => 0x50,
            SpecialKey::Left => 0x4B,
            SpecialKey::Right => 0x4D,
            _ => 0,
        };
        if raw != 0 {
            key_press(raw);
        }
    }
}
