// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Compositing window manager with T1 hit-testing and T4 adaptive FPS.

use alloc::vec::Vec;

use aether_core::governor::GeometricGovernor;
use aether_core::tss::SphericalVoronoiIndex;

use super::cursor;
use super::event::MouseState;
use super::taskbar;
use super::window::{self, Window, WindowState};
use crate::graphics::font;
use crate::graphics::framebuffer::Framebuffer;
use crate::wm::themes;

#[derive(Debug, Clone, Copy)]
pub enum ResizeEdge {
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

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
        Rect {
            x,
            y,
            w: x2 - x,
            h: y2 - y,
        }
    }
}

/// Width of the taskbar button that stands for one minimized window.
const TASKBAR_BTN_W: u32 = 80;
/// Gap between two adjacent taskbar buttons.
const TASKBAR_BTN_GAP: u32 = 4;
/// Left edge of the first taskbar button, clear of the start button.
const TASKBAR_BTN_X0: u32 = 350;
/// Columns held back at the right end of the taskbar for the clock and the
/// power button. See `taskbar::power_button_rect`.
const TASKBAR_BTN_RESERVED: u32 = 132;

/// Interactive floor for an in-progress resize drag, in total window pixels.
///
/// Deliberately above [`window::MIN_WIDTH`] — that one is the floor
/// `render_decorations` needs to keep its three button columns inside the
/// buffer, this one is the smallest window a user should be able to drag
/// themselves into. The assertions below fail the build if either value is
/// edited to the point where the interactive floor stops satisfying the
/// paintable one, which is the only way the two can disagree.
const MIN_W: i64 = 100;
/// Companion to [`MIN_W`]; see there.
const MIN_H: i64 = 60;
const _: () = assert!(MIN_W >= window::MIN_WIDTH as i64 && MIN_W <= window::MAX_WIDTH as i64);
const _: () = assert!(MIN_H >= window::MIN_HEIGHT as i64 && MIN_H <= window::MAX_HEIGHT as i64);

/// Point a window's backing buffer at its current dimensions.
///
/// The product is computed in `usize`. Three call sites here wrote
/// `(width * height) as usize`, which is a `u32` multiply: the release profile
/// carries no overflow checks, so a large enough window wrapped it to a short
/// length while `blit_window` went on indexing `height * width` elements.
fn fit_buffer(win: &mut Window) {
    let needed = win.width as usize * win.height as usize;
    if win.buffer.len() != needed {
        let theme = themes::current_theme();
        win.buffer.resize(needed, theme.bg);
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
    dragging_resize: Option<(u32, ResizeEdge)>,
    dirty_rects: Vec<Rect>,
    // Geometry of the framebuffer this compositor draws into. Written only by
    // `adopt_screen`, from the framebuffer handed to `compose`/`compose_full`,
    // so that hit-testing and drawing cannot disagree about where the taskbar
    // row is or how large a maximized window may be. The initial values are a
    // placeholder for the window between `new()` and the first composed frame;
    // `boot_graphical` composes once before it accepts any input.
    screen_w: u32,
    screen_h: u32,
}

impl Compositor {
    pub fn new() -> Self {
        // (theta, phi) centroids on the {0, pi/2, pi} lattice. Spelled with the
        // exact constants rather than 1.57 / 3.14 so screen-space cell
        // boundaries match the scheduler's; the truncated literals were off by
        // ~1.6e-3 rad.
        use core::f64::consts::{FRAC_PI_2, PI};
        let centroids = [
            (0.0, 0.0),
            (FRAC_PI_2, 0.0),
            (PI, 0.0),
            (0.0, FRAC_PI_2),
            (FRAC_PI_2, FRAC_PI_2),
            (PI, FRAC_PI_2),
            (0.0, PI),
            (FRAC_PI_2, PI),
        ];
        Self {
            windows: Vec::new(),
            next_id: 1,
            mouse: MouseState::new(),
            voronoi: SphericalVoronoiIndex::<8>::new(centroids),
            governor: GeometricGovernor::new(),
            frame_count: 0,
            dragging: None,
            dragging_resize: None,
            dirty_rects: Vec::new(),
            screen_w: 1024,
            screen_h: 768,
        }
    }

    pub fn create_window(&mut self, title: &str, x: u32, y: u32, w: u32, h: u32) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let mut win = Window::new(id, title, x, y, w, h);
        win.focused = self.windows.is_empty();
        win.render_decorations();
        let win_rect = Rect {
            x: win.x,
            y: win.y,
            w: win.width,
            h: win.height,
        };
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

    /// Move the cursor by a device delta.
    ///
    /// The cursor is clamped to the framebuffer last composed into, not to the
    /// bounds a caller supplies: the clamp has to agree with the bounds every
    /// drawing path uses, and a caller that invents a screen size is exactly
    /// how the cursor came to stop at 1023x767 on every other mode. The
    /// trailing two arguments are vestigial and ignored; they stay only until
    /// `run_desktop_soak_probe`, the last caller that still passes them, drops
    /// them.
    pub fn mouse_move(&mut self, dx: i32, dy: i32, _screen_w: i32, _screen_h: i32) {
        self.mouse
            .apply_move(dx, dy, self.screen_w as i32, self.screen_h as i32);

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

        if let Some((win_id, edge)) = self.dragging_resize {
            if let Some(idx) = self.windows.iter().position(|w| w.id == win_id) {
                let old_x = self.windows[idx].x;
                let old_y = self.windows[idx].y;
                let old_w = self.windows[idx].width;
                let old_h = self.windows[idx].height;

                // The edge opposite the one being dragged does not move, so the
                // whole new geometry is a function of the clamped cursor and
                // that fixed edge — the same absolute derivation the drag path
                // above uses, for the same reason. Applying `dx`/`dy` instead
                // kept widening the window after `apply_move` had already
                // pinned the cursor at the screen edge: the pointer stood still
                // and the window still grew by a further 255 pixels per PS/2
                // packet, each of which reallocated `width * height` u32s. A
                // sustained right-edge drag on a 1024x768 screen requested more
                // than 4 GiB by packet 9,883, which is allocation failure and
                // therefore the machine, some 29,000 packets before the `u32`
                // product could have wrapped.
                //
                // i64 throughout: `x + width` can exceed `u32::MAX`, because
                // `Window::new` clamps dimensions but not position and
                // `create_window` takes both from an unprivileged Aether script.
                let mx = self.mouse.x as i64;
                let my = self.mouse.y as i64;
                let left = old_x as i64;
                let top = old_y as i64;
                let right = left + old_w as i64;
                let bottom = top + old_h as i64;

                let (mut new_x, mut new_w) = match edge {
                    ResizeEdge::Left | ResizeEdge::TopLeft | ResizeEdge::BottomLeft => {
                        (mx, right - mx)
                    }
                    ResizeEdge::Right | ResizeEdge::TopRight | ResizeEdge::BottomRight => {
                        (left, mx - left + 1)
                    }
                    ResizeEdge::Top | ResizeEdge::Bottom => (left, old_w as i64),
                };
                let (mut new_y, mut new_h) = match edge {
                    ResizeEdge::Top | ResizeEdge::TopLeft | ResizeEdge::TopRight => {
                        (my, bottom - my)
                    }
                    ResizeEdge::Bottom | ResizeEdge::BottomLeft | ResizeEdge::BottomRight => {
                        (top, my - top + 1)
                    }
                    ResizeEdge::Left | ResizeEdge::Right => (top, old_h as i64),
                };

                // Both bounds come from `wm::window`, so the size a resize can
                // reach and the size the constructor can hand out are one pair
                // of numbers rather than two that drift apart.
                let clamped_w = new_w.clamp(MIN_W, window::MAX_WIDTH as i64);
                if clamped_w != new_w {
                    new_w = clamped_w;
                    if matches!(
                        edge,
                        ResizeEdge::Left | ResizeEdge::TopLeft | ResizeEdge::BottomLeft
                    ) {
                        new_x = right - new_w;
                    }
                }
                let clamped_h = new_h.clamp(MIN_H, window::MAX_HEIGHT as i64);
                if clamped_h != new_h {
                    new_h = clamped_h;
                    if matches!(
                        edge,
                        ResizeEdge::Top | ResizeEdge::TopLeft | ResizeEdge::TopRight
                    ) {
                        new_y = bottom - new_h;
                    }
                }

                let new_win_x = new_x.clamp(0, u32::MAX as i64) as u32;
                let new_win_y = new_y.clamp(0, u32::MAX as i64) as u32;
                let new_win_w = new_w as u32;
                let new_win_h = new_h as u32;

                let win = &mut self.windows[idx];
                win.x = new_win_x;
                win.y = new_win_y;
                win.width = new_win_w;
                win.height = new_win_h;

                fit_buffer(win);
                win.render_decorations();
                win.dirty = true;

                self.mark_dirty(old_x, old_y, old_w, old_h);
                self.mark_dirty(new_win_x, new_win_y, new_win_w, new_win_h);
            }
        }
    }

    pub fn mouse_click(&mut self, pressed: bool) {
        let mx = self.mouse.x as u32;
        let my = self.mouse.y as u32;

        if !pressed {
            self.dragging = None;
            self.dragging_resize = None;
            return;
        }

        // Taskbar click: check for minimized window restore
        let taskbar_h = taskbar::taskbar_height();
        if my >= self.screen_h.saturating_sub(taskbar_h) {
            self.handle_taskbar_click(mx, my);
            return;
        }

        // T1: screen-space Voronoi chooses the first hit-test cell.
        let click_cell = self.screen_point_cell(mx, my);
        let hit = self
            .window_hit_in_cell(mx, my, click_cell)
            .or_else(|| self.window_hit_any(mx, my));

        if let Some(idx) = hit {
            // Focus this window. `raise` keeps `idx` valid: it rewrites one
            // `z_order` and never reorders `self.windows`.
            for w in &mut self.windows {
                w.focused = false;
            }
            self.raise(idx);
            self.windows[idx].focused = true;
            self.windows[idx].render_decorations();
            let win_id = self.windows[idx].id;
            let win_x = self.windows[idx].x;
            let win_y = self.windows[idx].y;
            let win_w = self.windows[idx].width;
            let win_h = self.windows[idx].height;
            let in_close = self.windows[idx].in_close_button(mx, my);
            let in_minimize = self.windows[idx].in_minimize_button(mx, my);
            let in_maximize = self.windows[idx].in_maximize_button(mx, my);
            let in_title = self.windows[idx].in_title_bar(mx, my);

            self.mark_dirty(win_x, win_y, win_w, win_h);

            if in_close {
                self.windows[idx].state = WindowState::Closed;
                self.mark_dirty(win_x, win_y, win_w, win_h);
            } else if in_minimize {
                self.windows[idx].state = WindowState::Minimized;
                self.mark_dirty(win_x, win_y, win_w, win_h);
            } else if in_maximize {
                if self.windows[idx].state == WindowState::Maximized {
                    let old_x = self.windows[idx].x;
                    let old_y = self.windows[idx].y;
                    let old_w = self.windows[idx].width;
                    let old_h = self.windows[idx].height;
                    let prev_x = self.windows[idx].prev_x;
                    let prev_y = self.windows[idx].prev_y;
                    let prev_w = self.windows[idx].prev_w;
                    let prev_h = self.windows[idx].prev_h;
                    self.windows[idx].state = WindowState::Normal;
                    self.windows[idx].x = prev_x;
                    self.windows[idx].y = prev_y;
                    self.windows[idx].width = prev_w;
                    self.windows[idx].height = prev_h;
                    fit_buffer(&mut self.windows[idx]);
                    self.windows[idx].render_decorations();
                    self.mark_dirty(old_x, old_y, old_w, old_h);
                    self.mark_dirty(prev_x, prev_y, prev_w, prev_h);
                } else {
                    let prev_x = self.windows[idx].x;
                    let prev_y = self.windows[idx].y;
                    let prev_w = self.windows[idx].width;
                    let prev_h = self.windows[idx].height;
                    self.windows[idx].prev_x = prev_x;
                    self.windows[idx].prev_y = prev_y;
                    self.windows[idx].prev_w = prev_w;
                    self.windows[idx].prev_h = prev_h;

                    self.windows[idx].state = WindowState::Maximized;
                    self.windows[idx].x = 0;
                    self.windows[idx].y = 0;
                    self.windows[idx].width = self.screen_w;
                    self.windows[idx].height = self.screen_h.saturating_sub(taskbar_h);
                    fit_buffer(&mut self.windows[idx]);
                    self.windows[idx].render_decorations();
                    self.mark_dirty(prev_x, prev_y, prev_w, prev_h);
                    self.mark_dirty(0, 0, self.screen_w, self.screen_h.saturating_sub(taskbar_h));
                }
            } else if in_title {
                self.dragging = Some((
                    win_id,
                    self.mouse.x - win_x as i32,
                    self.mouse.y - win_y as i32,
                ));
            } else {
                // Check for resize edge hit
                let win = &self.windows[idx];
                let in_left = mx >= win.x && mx < win.x + 4;
                let in_right = mx >= win.x + win.width.saturating_sub(4);
                let in_top = my >= win.y && my < win.y + 4;
                let in_bottom = my >= win.y + win.height.saturating_sub(4);

                let edge = if in_top && in_left {
                    Some(ResizeEdge::TopLeft)
                } else if in_top && in_right {
                    Some(ResizeEdge::TopRight)
                } else if in_bottom && in_left {
                    Some(ResizeEdge::BottomLeft)
                } else if in_bottom && in_right {
                    Some(ResizeEdge::BottomRight)
                } else if in_left {
                    Some(ResizeEdge::Left)
                } else if in_right {
                    Some(ResizeEdge::Right)
                } else if in_top {
                    Some(ResizeEdge::Top)
                } else if in_bottom {
                    Some(ResizeEdge::Bottom)
                } else {
                    None
                };

                if let Some(e) = edge {
                    self.dragging_resize = Some((win_id, e));
                }
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

    /// Adopt the geometry of the framebuffer about to be drawn into.
    ///
    /// A zero-sized framebuffer carries no usable geometry and is ignored: a
    /// zero height would put the taskbar hit row at 0 and swallow every click.
    fn adopt_screen(&mut self, fb: &Framebuffer) {
        if fb.width == 0 || fb.height == 0 {
            return;
        }
        self.screen_w = fb.width;
        self.screen_h = fb.height;
    }

    pub fn compose(&mut self, fb: &Framebuffer) {
        self.adopt_screen(fb);
        self.frame_count += 1;

        let theme_dirty = themes::theme_changed();
        if theme_dirty {
            self.request_full_redraw();
            self.mark_dirty(0, 0, fb.width, fb.height);
        }

        // T4: Adaptive FPS — skip frames when idle
        let has_dirty_windows = self.windows.iter().any(|w| w.dirty);
        let eps = self.governor.epsilon();
        if !theme_dirty
            && !has_dirty_windows
            && self.dirty_rects.is_empty()
            && eps < 0.3
            && self.frame_count % 4 != 0
        {
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
            Rect {
                x: 0,
                y: 0,
                w: fb.width,
                h: fb.height,
            }
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
            if !clip.intersects(&Rect {
                x: win.x,
                y: win.y,
                w: win.width,
                h: win.height,
            }) {
                continue;
            }
            self.blit_window(fb, win, &clip);
        }

        // Context-aware cursor shape
        self.update_cursor_shape();
        // Cursor on top
        cursor::draw_cursor(fb, self.mouse.x, self.mouse.y);

        // Minimized window indicators on taskbar
        self.draw_taskbar_indicators(fb);

        self.dirty_rects.clear();
        self.governor.adapt(1.0, 0.01);
    }

    pub fn compose_full(&mut self, fb: &Framebuffer) {
        self.adopt_screen(fb);
        self.frame_count += 1;
        self.windows.sort_by_key(|w| w.z_order);

        let clip = Rect {
            x: 0,
            y: 0,
            w: fb.width,
            h: fb.height,
        };

        for win in &mut self.windows {
            win.dirty = false;
        }
        for win in &self.windows {
            if win.state == WindowState::Closed || win.state == WindowState::Minimized {
                continue;
            }
            self.blit_window(fb, win, &clip);
        }

        self.update_cursor_shape();
        cursor::draw_cursor(fb, self.mouse.x, self.mouse.y);
        self.draw_taskbar_indicators(fb);
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

    pub fn window_at(&self, mx: u32, my: u32) -> Option<u32> {
        let cell = self.screen_point_cell(mx, my);
        self.window_hit_in_cell(mx, my, cell)
            .or_else(|| self.window_hit_any(mx, my))
            .map(|idx| self.windows[idx].id)
    }

    fn screen_point_cell(&self, x: u32, y: u32) -> usize {
        let w = self.screen_w.max(1) as f64;
        let h = self.screen_h.max(1) as f64;
        let theta = core::f64::consts::PI * (y as f64 / h);
        let phi = core::f64::consts::TAU * (x as f64 / w);
        self.voronoi.locate((theta, phi))
    }

    fn window_primary_cell(&self, win: &Window) -> usize {
        self.screen_point_cell(win.x + win.width / 2, win.y + win.height / 2)
    }

    fn window_hit_in_cell(&self, mx: u32, my: u32, cell: usize) -> Option<usize> {
        self.windows
            .iter()
            .enumerate()
            .rev()
            .find(|(_, win)| {
                win.state != WindowState::Closed
                    && win.state != WindowState::Minimized
                    && self.window_primary_cell(win) == cell
                    && win.contains(mx, my)
            })
            .map(|(idx, _)| idx)
    }

    fn window_hit_any(&self, mx: u32, my: u32) -> Option<usize> {
        self.windows
            .iter()
            .enumerate()
            .rev()
            .find(|(_, win)| {
                win.state != WindowState::Closed
                    && win.state != WindowState::Minimized
                    && win.contains(mx, my)
            })
            .map(|(idx, _)| idx)
    }

    /// Bring a window to the top of the paint order.
    ///
    /// `z_order` is assigned once, in `Window::new`, and both render paths sort
    /// by it, so without a second writer the paint order is permanently
    /// creation order: a clicked window took focus and every keystroke while an
    /// older window kept painting over it, and the user typed into a window
    /// they could not see. Only `self.windows[idx].z_order` is written — the
    /// vector is not reordered, so an index held across this call stays valid.
    fn raise(&mut self, idx: usize) {
        let top = self.windows.iter().map(|w| w.z_order).max().unwrap_or(0);
        if self.windows[idx].z_order < top {
            // ponytail: z_order saturates at u32::MAX, after which raising
            // stops working. It only advances when a window that is not
            // already on top is raised, so reaching that takes 4.29e9 such
            // clicks; renumber the stack by sort rank if it ever matters.
            self.windows[idx].z_order = top.saturating_add(1);
        }
    }

    /// Focus a window, restore it if it was minimized, and raise it.
    ///
    /// The chokepoint every focus change routes through, including the taskbar
    /// restore path, so that exactly one window carries `focused` and the
    /// focused window is the one on top.
    pub fn focus_window(&mut self, id: u32) {
        // An unknown id leaves the focus where it is rather than clearing it
        // off every window and leaving the desktop with none.
        let Some(idx) = self.windows.iter().position(|w| w.id == id) else {
            return;
        };
        for w in &mut self.windows {
            w.focused = false;
        }
        self.raise(idx);
        let win = &mut self.windows[idx];
        win.state = WindowState::Normal;
        win.focused = true;
        win.render_decorations();
        let r = Rect {
            x: win.x,
            y: win.y,
            w: win.width,
            h: win.height,
        };
        self.mark_dirty(r.x, r.y, r.w, r.h);
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

            if win.in_close_button(mx, my)
                || win.in_minimize_button(mx, my)
                || win.in_maximize_button(mx, my)
            {
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

    /// Left edge of the taskbar button for the `n`th minimized window, or
    /// `None` once the row has no room for another one.
    ///
    /// The single source of truth for that geometry, in the screen coordinates
    /// `adopt_screen` took from the framebuffer: `draw_taskbar_indicators`
    /// paints exactly these bands and `handle_taskbar_click` hit-tests exactly
    /// these bands. The two used to bound themselves separately — the drawing
    /// stopped where the clock begins, the hit-testing never stopped — so at
    /// 1024x768 with eleven windows minimized, six buttons were painted, the
    /// last ending at x=850, and a click anywhere in [854, 1018) on bare
    /// taskbar restored a window that had no button.
    fn taskbar_button_x(&self, n: u32) -> Option<u32> {
        let x = TASKBAR_BTN_X0.checked_add(n.checked_mul(TASKBAR_BTN_W + TASKBAR_BTN_GAP)?)?;
        let end_x = self.screen_w.saturating_sub(TASKBAR_BTN_RESERVED);
        (x.saturating_add(TASKBAR_BTN_W) <= end_x).then_some(x)
    }

    fn handle_taskbar_click(&mut self, mx: u32, my: u32) {
        let taskbar_h = taskbar::taskbar_height();
        let y = self.screen_h.saturating_sub(taskbar_h);
        if my < y + 4 || my >= y + 24 {
            return;
        }

        let minimized: Vec<_> = self
            .windows
            .iter()
            .filter(|w| w.state == WindowState::Minimized)
            .map(|w| w.id)
            .collect();

        for (n, win_id) in minimized.iter().enumerate() {
            let Some(x) = self.taskbar_button_x(n as u32) else {
                break;
            };
            if mx >= x && mx < x + TASKBAR_BTN_W {
                // Restoring is a focus change, and focus changes go through one
                // function: this loop used to set `focused` without clearing it
                // on the others, leaving two windows claiming focus and sending
                // the keystrokes to whichever had the lower z_order.
                self.focus_window(*win_id);
                break;
            }
        }
    }

    fn draw_taskbar_indicators(&self, fb: &Framebuffer) {
        let theme = themes::current_theme();
        let taskbar_h = taskbar::taskbar_height();
        // `screen_h`, not `fb.height`: `adopt_screen` has already taken both
        // from this framebuffer, and reading the same field the click path
        // reads is what keeps the drawn row and the clickable row identical.
        let y = self.screen_h.saturating_sub(taskbar_h);

        let minimized = self
            .windows
            .iter()
            .filter(|w| w.state == WindowState::Minimized);

        for (n, win) in minimized.enumerate() {
            let Some(x) = self.taskbar_button_x(n as u32) else {
                break;
            };
            // Button background
            fb.fill_rect(x, y + 4, TASKBAR_BTN_W, 20, theme.bg);
            fb.fill_rect(x, y + 4, TASKBAR_BTN_W, 1, theme.border);
            fb.fill_rect(x, y + 23, TASKBAR_BTN_W, 1, 0);
            fb.fill_rect(x, y + 4, 1, 20, theme.border);
            fb.fill_rect(x + TASKBAR_BTN_W - 1, y + 4, 1, 20, 0);
            // Accent strip on left
            fb.fill_rect(x, y + 4, 3, 20, theme.accent);
            // Draw truncated title
            let name = if win.title.len() > 7 {
                &win.title[..7]
            } else {
                &win.title
            };
            let text_x = x + 6;
            let text_y = y + 6;
            for (i, ch) in name.bytes().enumerate() {
                font::draw_char(
                    fb,
                    text_x + (i as u32) * font::CHAR_WIDTH,
                    text_y,
                    ch,
                    theme.fg,
                );
            }
        }
    }
}

impl Default for Compositor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// A framebuffer whose pixels can be read back.
    ///
    /// The kernel offers no other way to get one: `Framebuffer::null()` reports
    /// zero width and every drawing path bounds-checks against it, so a test
    /// that wants to see what was painted has to supply its own storage. The
    /// returned `Vec` must stay alive for as long as the framebuffer is used.
    fn screen(w: u32, h: u32) -> (Vec<u32>, Framebuffer) {
        let mut pixels = alloc::vec![0u32; (w as usize) * (h as usize)];
        // SAFETY: `pixels` holds exactly `w * h` u32s, so the buffer is
        // `w * 4` bytes per scanline at 32 bpp for `h` scanlines, matching the
        // pitch and depth passed here. Moving the `Vec` out of this function
        // does not move its heap allocation, and the caller holds it for the
        // lifetime of the returned `Framebuffer`.
        let fb = unsafe { Framebuffer::new(pixels.as_mut_ptr() as u64, w, h, w * 4, 32) };
        (pixels, fb)
    }

    /// Park the cursor at an absolute screen position.
    fn move_to(comp: &mut Compositor, x: i32, y: i32) {
        let dx = x - comp.mouse.x;
        let dy = y - comp.mouse.y;
        comp.mouse_move(dx, dy, 0, 0);
    }

    /// The middle of the taskbar button row for a 768-pixel-tall screen.
    fn taskbar_row_y() -> u32 {
        768 - taskbar::taskbar_height() + 10
    }

    /// The draw loop stops once a button would run into the clock; the click
    /// loop used to keep handing out 84-pixel bands past the end of the row.
    /// At 1024x768 with eleven windows minimized, six buttons are painted and
    /// the last ends at x=850, yet a click at x=856 — bare taskbar — restored
    /// the seventh window.
    fn test_taskbar_click_stops_where_the_row_stops() -> TestResult {
        let (_pixels, fb) = screen(1024, 768);
        let mut comp = Compositor::new();
        for _ in 0..11 {
            let id = comp.create_window("win", 10, 10, 100, 80);
            comp.window_mut(id).unwrap().state = WindowState::Minimized;
        }
        comp.compose_full(&fb);

        let row_y = taskbar_row_y();
        // The accent strip of the sixth button, at x in [770, 773).
        test_assert!(
            fb.get_pixel(771, row_y) != 0,
            "sixth taskbar button was not drawn"
        );
        // A seventh button would start at x=854 and is never painted.
        test_assert_eq!(fb.get_pixel(855, row_y), 0);

        move_to(&mut comp, 856, row_y as i32);
        comp.mouse_click(true);
        test_assert!(
            comp.windows
                .iter()
                .all(|w| w.state == WindowState::Minimized),
            "a click past the last drawn button restored a window"
        );

        // Control: the last button that is drawn still restores its window.
        move_to(&mut comp, 772, row_y as i32);
        comp.mouse_click(true);
        test_assert_eq!(
            comp.windows
                .iter()
                .filter(|w| w.state == WindowState::Minimized)
                .count(),
            10
        );
        TestResult::Pass
    }

    /// `handle_taskbar_click` set `focused` without clearing it anywhere else,
    /// so a restore left two windows claiming focus. `focused_window_id` takes
    /// the first match in ascending z_order, i.e. the older window: the
    /// restored window drew the focused title bar while every keystroke went
    /// to the other one.
    fn test_taskbar_restore_focuses_one_window() -> TestResult {
        let (_pixels, fb) = screen(1024, 768);
        let mut comp = Compositor::new();
        let first = comp.create_window("first", 10, 10, 200, 100);
        let second = comp.create_window("second", 300, 10, 200, 100);
        comp.window_mut(second).unwrap().state = WindowState::Minimized;
        comp.compose_full(&fb);

        // The first taskbar button spans x in [350, 430).
        move_to(&mut comp, 360, taskbar_row_y() as i32);
        comp.mouse_click(true);

        test_assert_eq!(comp.windows.iter().filter(|w| w.focused).count(), 1);
        test_assert_eq!(comp.focused_window_id(), second);
        test_assert!(!comp.window_mut(first).unwrap().focused);
        test_assert_eq!(
            comp.window_mut(second).unwrap().state,
            WindowState::Normal
        );
        TestResult::Pass
    }

    /// `z_order` was written once, in `Window::new`, and both render paths sort
    /// by it, so paint order was permanently creation order. Clicking a window
    /// another one overlaps gave it focus and every keystroke while the other
    /// kept painting on top of it — the user typed into a window they could
    /// not see.
    fn test_click_raises_window_above_its_neighbour() -> TestResult {
        const PROBE_X: u32 = 200;
        const PROBE_Y: u32 = 174;
        const LOWER_INK: u32 = 0x00FF_0000;
        const UPPER_INK: u32 = 0x0000_FF00;

        let (_pixels, fb) = screen(1024, 768);
        let mut comp = Compositor::new();
        let lower = comp.create_window("lower", 100, 100, 200, 200);
        let upper = comp.create_window("upper", 150, 100, 200, 200);

        // One client pixel from each window at the same screen point, inside
        // the region where the two overlap.
        for (id, ink) in [(lower, LOWER_INK), (upper, UPPER_INK)] {
            let win = comp.window_mut(id).unwrap();
            let (cx, cy) = (win.client_x(), win.client_y());
            win.set_client_pixel(PROBE_X - cx, PROBE_Y - cy, ink);
        }

        comp.compose_full(&fb);
        test_assert_eq!(fb.get_pixel(PROBE_X, PROBE_Y), UPPER_INK);

        // Click the lower window where the upper one does not cover it.
        move_to(&mut comp, 110, PROBE_Y as i32);
        comp.mouse_click(true);
        comp.mouse_click(false);
        comp.compose_full(&fb);

        test_assert_eq!(comp.focused_window_id(), lower);
        test_assert_eq!(fb.get_pixel(PROBE_X, PROBE_Y), LOWER_INK);
        test_assert!(
            comp.window_mut(lower).unwrap().z_order > comp.window_mut(upper).unwrap().z_order,
            "the clicked window did not come to the top of the paint order"
        );
        TestResult::Pass
    }

    /// `mouse_move` clamps the cursor to the screen and then, in the resize
    /// path, applied the *unclamped* `dx` of the same packet to the width. Once
    /// the pointer pinned at x=1023 the window went on growing 200 pixels per
    /// packet with the pointer standing still, and the size could only be
    /// walked back with an equal number of opposite packets. Every packet also
    /// reallocated the buffer: a sustained drag at the PS/2 maximum of 255 per
    /// packet requested 4 GiB by packet 9,883.
    fn test_resize_tracks_the_clamped_cursor() -> TestResult {
        let (_pixels, fb) = screen(1024, 768);
        let mut comp = Compositor::new();
        let id = comp.create_window("resize", 100, 100, 600, 400);
        comp.compose_full(&fb);
        // Window::new(600, 400) -> 604x426, so the right edge is at x=704 and
        // its 4-pixel grab band is [700, 704).
        test_assert_eq!(comp.window_mut(id).unwrap().width, 604);

        move_to(&mut comp, 702, 300);
        comp.mouse_click(true);
        test_assert!(
            comp.dragging_resize.is_some(),
            "the right edge was not grabbed"
        );

        // Eight packets of +200. Only the first two move the cursor; it pins at
        // 1023 and the remaining six are pointer-stationary.
        for _ in 0..8 {
            comp.mouse_move(200, 0, 0, 0);
        }
        test_assert_eq!(comp.mouse.x, 1023);
        let win = comp.window_mut(id).unwrap();
        test_assert_eq!(win.x, 100);
        test_assert_eq!(win.width, 924); // right edge exactly under the cursor
        test_assert_eq!(win.buffer.len(), (win.width as usize) * (win.height as usize));

        // One opposite packet is enough to bring the edge back to the pointer.
        comp.mouse_move(-500, 0, 0, 0);
        test_assert_eq!(comp.mouse.x, 523);
        test_assert_eq!(comp.window_mut(id).unwrap().width, 424);

        // The floor still holds when the pointer crosses the fixed edge.
        comp.mouse_move(-1000, 0, 0, 0);
        test_assert_eq!(comp.mouse.x, 0);
        let win = comp.window_mut(id).unwrap();
        test_assert_eq!(win.x, 100);
        test_assert_eq!(win.width, MIN_W as u32);
        test_assert_eq!(win.buffer.len(), (win.width as usize) * (win.height as usize));
        TestResult::Pass
    }

    /// `Window::new` clamps dimensions but not position, and `create_window`
    /// takes both from an unprivileged Aether script, so `x + width` can land
    /// near `u32::MAX`. A left-edge drag derives the width from that sum minus
    /// the cursor, which is the one input the screen clamp does not bound — so
    /// the resize has to fail closed against the same maximum the constructor
    /// uses rather than hand a wrapped product to `Vec::resize`.
    fn test_resize_clamps_to_the_window_maximum() -> TestResult {
        let (_pixels, fb) = screen(1024, 768);
        let mut comp = Compositor::new();
        comp.compose_full(&fb);

        let id = comp.create_window("far", u32::MAX - 1000, 100, 600, 400);
        comp.dragging_resize = Some((id, ResizeEdge::Left));
        comp.mouse_move(-2000, 0, 0, 0);
        test_assert_eq!(comp.mouse.x, 0);

        let win = comp.window_mut(id).unwrap();
        test_assert_eq!(win.width, window::MAX_WIDTH);
        test_assert_eq!(win.height, 426);
        test_assert_eq!(win.buffer.len(), (win.width as usize) * (win.height as usize));
        // The product the old code handed to `Vec::resize` as a u32.
        test_assert!((win.width as u64) * (win.height as u64) <= u32::MAX as u64);
        // The dragged edge moved; the opposite one did not.
        test_assert_eq!(win.x + win.width, u32::MAX - 1000 + 604);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "compositor::resize_tracks_the_clamped_cursor",
            test_resize_tracks_the_clamped_cursor,
        );
        crate::testing::register_test(
            "compositor::resize_clamps_to_the_window_maximum",
            test_resize_clamps_to_the_window_maximum,
        );
        crate::testing::register_test(
            "compositor::taskbar_click_stops_where_the_row_stops",
            test_taskbar_click_stops_where_the_row_stops,
        );
        crate::testing::register_test(
            "compositor::taskbar_restore_focuses_one_window",
            test_taskbar_restore_focuses_one_window,
        );
        crate::testing::register_test(
            "compositor::click_raises_window_above_its_neighbour",
            test_click_raises_window_above_its_neighbour,
        );
    }
}
