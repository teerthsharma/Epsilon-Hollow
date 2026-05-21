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
use crate::wm::themes;

#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub fn area(&self) -> u64 {
        self.w as u64 * self.h as u64
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.x + other.w
            && self.x + self.w > other.x
            && self.y < other.y + other.h
            && self.y + self.h > other.y
    }

    pub fn union(&self, other: &Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let x2 = (self.x + self.w).max(other.x + other.w);
        let y2 = (self.y + self.h).max(other.y + other.h);
        Rect { x, y, w: x2 - x, h: y2 - y }
    }
}

pub struct Compositor {
    windows: Vec<Window>,
    next_id: u32,
    mouse: MouseState,
    voronoi: SphericalVoronoiIndex<8>,
    governor: GeometricGovernor,
    frame_count: u64,
    dragging: Option<(u32, i32, i32)>, // window_id, offset_x, offset_y
    dirty_rects: Vec<Rect>,
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
            dirty_rects: Vec::new(),
        }
    }

    pub fn create_window(&mut self, title: &str, x: u32, y: u32, w: u32, h: u32) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let mut win = Window::new(id, title, x, y, w, h);
        win.focused = self.windows.is_empty();
        win.render_decorations();
        let win_rect = Rect { x: win.x, y: win.y, w: win.width, h: win.height };
        self.windows.push(win);
        self.mark_dirty(win_rect.x, win_rect.y, win_rect.w, win_rect.h);
        id
    }

    pub fn mark_dirty(&mut self, x: u32, y: u32, w: u32, h: u32) {
        self.dirty_rects.push(Rect { x, y, w, h });
    }

    pub fn window_mut(&mut self, id: u32) -> Option<&mut Window> {
        self.windows.iter_mut().find(|w| w.id == id)
    }

    pub fn mouse_move(&mut self, dx: i32, dy: i32, screen_w: i32, screen_h: i32) {
        self.mouse.apply_move(dx, dy, screen_w, screen_h);

        if let Some((win_id, ox, oy)) = self.dragging {
            if let Some(idx) = self.windows.iter().position(|w| w.id == win_id) {
                let old_x = self.windows[idx].x;
                let old_y = self.windows[idx].y;
                self.windows[idx].x = (self.mouse.x - ox).max(0) as u32;
                self.windows[idx].y = (self.mouse.y - oy).max(0) as u32;
                let new_x = self.windows[idx].x;
                let new_y = self.windows[idx].y;
                let w = self.windows[idx].width;
                let h = self.windows[idx].height;
                if new_x != old_x || new_y != old_y {
                    self.mark_dirty(old_x, old_y, w, h);
                    self.mark_dirty(new_x, new_y, w, h);
                }
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
            let win_id = self.windows[idx].id;
            let win_x = self.windows[idx].x;
            let win_y = self.windows[idx].y;
            let win_w = self.windows[idx].width;
            let win_h = self.windows[idx].height;
            let in_close = self.windows[idx].in_close_button(mx, my);
            let in_title = self.windows[idx].in_title_bar(mx, my);

            self.mark_dirty(win_x, win_y, win_w, win_h);

            if in_close {
                self.windows[idx].state = WindowState::Closed;
                self.mark_dirty(win_x, win_y, win_w, win_h);
            } else if in_title {
                self.dragging = Some((
                    win_id,
                    self.mouse.x - win_x as i32,
                    self.mouse.y - win_y as i32,
                ));
            }
        }
    }

    pub fn request_full_redraw(&mut self) {
        for win in &mut self.windows {
            if win.state != WindowState::Closed && win.state != WindowState::Minimized {
                win.render_decorations();
            }
        }
    }

    pub fn compose(&mut self, fb: &Framebuffer) {
        self.frame_count += 1;

        let theme_dirty = themes::theme_changed();
        if theme_dirty {
            self.request_full_redraw();
            self.mark_dirty(0, 0, fb.width, fb.height);
        }

        // T4: Adaptive FPS — skip frames when idle
        let eps = self.governor.epsilon();
        if !theme_dirty && eps < 0.3 && self.frame_count % 4 != 0 {
            return; // 15fps when idle
        }

        // Propagate per-window dirty flags to compositor-level dirty rects
        for i in 0..self.windows.len() {
            if self.windows[i].dirty {
                let x = self.windows[i].x;
                let y = self.windows[i].y;
                let w = self.windows[i].width;
                let h = self.windows[i].height;
                self.windows[i].dirty = false;
                self.mark_dirty(x, y, w, h);
            }
        }

        let screen_area = fb.width as u64 * fb.height as u64;
        let dirty_area: u64 = self.dirty_rects.iter().map(|r| r.area()).sum();
        let full_redraw = dirty_area > screen_area / 2;

        let clip = if self.dirty_rects.is_empty() || full_redraw {
            Rect { x: 0, y: 0, w: fb.width, h: fb.height }
        } else {
            let mut r = self.dirty_rects[0];
            for d in &self.dirty_rects[1..] {
                r = r.union(d);
            }
            r
        };

        // Render windows bottom-to-top using z_order
        self.windows.sort_by_key(|w| w.z_order);

        for win in &self.windows {
            if win.state == WindowState::Closed || win.state == WindowState::Minimized {
                continue;
            }
            if !clip.intersects(&Rect { x: win.x, y: win.y, w: win.width, h: win.height }) {
                continue;
            }
            self.blit_window(fb, win, &clip);
        }

        // Context-aware cursor shape
        self.update_cursor_shape();
        // Cursor on top
        cursor::draw_cursor(fb, self.mouse.x, self.mouse.y);

        self.dirty_rects.clear();
        self.governor.adapt(1.0, 0.01);
    }

    fn blit_window(&self, fb: &Framebuffer, win: &Window, clip: &Rect) {
        let start_y = win.y.max(clip.y);
        let end_y = (win.y + win.height).min(clip.y + clip.h).min(fb.height);
        let start_x = win.x.max(clip.x);
        let end_x = (win.x + win.width).min(clip.x + clip.w).min(fb.width);

        for y in start_y..end_y {
            let buf_y = y - win.y;
            let row_offset = (buf_y * win.width) as usize;
            for x in start_x..end_x {
                let buf_x = x - win.x;
                let color = win.buffer[row_offset + buf_x as usize];
                fb.put_pixel(x, y, color);
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

    pub fn focused_window_id(&self) -> u32 {
        self.windows
            .iter()
            .find(|w| w.focused && w.state != WindowState::Closed)
            .map(|w| w.id)
            .unwrap_or(0)
    }

    fn update_cursor_shape(&self) {
        let mx = self.mouse.x as u32;
        let my = self.mouse.y as u32;

        for win in self.windows.iter().rev() {
            if win.state == WindowState::Closed || win.state == WindowState::Minimized {
                continue;
            }
            if !win.contains(mx, my) {
                continue;
            }

            if win.in_close_button(mx, my) {
                cursor::set_shape(cursor::CursorShape::Hand);
                return;
            }
            if win.in_title_bar(mx, my) {
                cursor::set_shape(cursor::CursorShape::Arrow);
                return;
            }

            // Edges within 4px of border
            let in_left = mx >= win.x && mx < win.x + 4;
            let in_right = mx >= win.x + win.width.saturating_sub(4);
            let in_top = my >= win.y && my < win.y + 4;
            let in_bottom = my >= win.y + win.height.saturating_sub(4);

            if in_left || in_right {
                cursor::set_shape(cursor::CursorShape::ResizeH);
                return;
            }
            if in_top || in_bottom {
                cursor::set_shape(cursor::CursorShape::ResizeV);
                return;
            }

            // Client area
            cursor::set_shape(cursor::CursorShape::IBeam);
            return;
        }

        cursor::set_shape(cursor::CursorShape::Arrow);
    }
}
