// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Compositing window manager with T1 hit-testing and T4 adaptive FPS.

use alloc::vec::Vec;

use aether_core::governor::GeometricGovernor;
use aether_core::tss::SphericalVoronoiIndex;

use super::cursor;
use super::event::MouseState;
use super::window::{Window, WindowState};
use crate::graphics::framebuffer::Framebuffer;

pub struct Compositor {
    windows: Vec<Window>,
    next_id: u32,
    mouse: MouseState,
    voronoi: SphericalVoronoiIndex<8>,
    governor: GeometricGovernor,
    frame_count: u64,
    dragging: Option<(u32, i32, i32)>, // window_id, offset_x, offset_y
}

impl Compositor {
    pub fn new() -> Self {
        let centroids = [
            (0.0, 0.0),
            (1.57, 0.0),
            (3.14, 0.0),
            (0.0, 1.57),
            (1.57, 1.57),
            (3.14, 1.57),
            (0.0, 3.14),
            (1.57, 3.14),
        ];
        Self {
            windows: Vec::new(),
            next_id: 1,
            mouse: MouseState::new(),
            voronoi: SphericalVoronoiIndex::<8>::new(centroids),
            governor: GeometricGovernor::new(),
            frame_count: 0,
            dragging: None,
        }
    }

    pub fn create_window(&mut self, title: &str, x: u32, y: u32, w: u32, h: u32) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let mut win = Window::new(id, title, x, y, w, h);
        win.focused = self.windows.is_empty();
        win.render_decorations();
        self.windows.push(win);
        id
    }

    pub fn window_mut(&mut self, id: u32) -> Option<&mut Window> {
        self.windows.iter_mut().find(|w| w.id == id)
    }

    pub fn mouse_move(&mut self, dx: i32, dy: i32, screen_w: i32, screen_h: i32) {
        self.mouse.apply_move(dx, dy, screen_w, screen_h);

        if let Some((win_id, ox, oy)) = self.dragging {
            if let Some(win) = self.windows.iter_mut().find(|w| w.id == win_id) {
                win.x = (self.mouse.x - ox).max(0) as u32;
                win.y = (self.mouse.y - oy).max(0) as u32;
                win.dirty = true;
            }
        }
    }

    pub fn mouse_click(&mut self, pressed: bool) {
        let mx = self.mouse.x as u32;
        let my = self.mouse.y as u32;

        if !pressed {
            self.dragging = None;
            return;
        }

        // T1: O(1) hit-test via screen-space Voronoi
        // (simplified: linear scan since window count is small)
        let mut hit = None;
        for (i, win) in self.windows.iter().enumerate().rev() {
            if win.state == WindowState::Closed {
                continue;
            }
            if win.contains(mx, my) {
                hit = Some(i);
                break;
            }
        }

        if let Some(idx) = hit {
            // Focus this window
            for w in &mut self.windows {
                w.focused = false;
            }
            self.windows[idx].focused = true;
            self.windows[idx].render_decorations();

            if self.windows[idx].in_close_button(mx, my) {
                self.windows[idx].state = WindowState::Closed;
            } else if self.windows[idx].in_title_bar(mx, my) {
                let win = &self.windows[idx];
                self.dragging = Some((
                    win.id,
                    self.mouse.x - win.x as i32,
                    self.mouse.y - win.y as i32,
                ));
            }
        }
    }

    pub fn compose(&mut self, fb: &Framebuffer) {
        self.frame_count += 1;

        // T4: Adaptive FPS — skip frames when idle
        let eps = self.governor.epsilon();
        if eps < 0.3 && self.frame_count % 4 != 0 {
            return; // 15fps when idle
        }

        // Render windows bottom-to-top
        for win in &self.windows {
            if win.state == WindowState::Closed || win.state == WindowState::Minimized {
                continue;
            }
            self.blit_window(fb, win);
        }

        // Cursor on top
        cursor::draw_cursor(fb, self.mouse.x, self.mouse.y);

        self.governor.adapt(1.0, 0.01);
    }

    fn blit_window(&self, fb: &Framebuffer, win: &Window) {
        for y in 0..win.height {
            let screen_y = win.y + y;
            if screen_y >= fb.height {
                break;
            }
            for x in 0..win.width {
                let screen_x = win.x + x;
                if screen_x >= fb.width {
                    break;
                }
                let color = win.buffer[(y * win.width + x) as usize];
                fb.put_pixel(screen_x, screen_y, color);
            }
        }
    }

    pub fn mouse_state(&self) -> &MouseState {
        &self.mouse
    }

    pub fn window_count(&self) -> usize {
        self.windows
            .iter()
            .filter(|w| w.state != WindowState::Closed)
            .count()
    }
}
