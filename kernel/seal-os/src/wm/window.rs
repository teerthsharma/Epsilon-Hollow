// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Window representation with framebuffer region and decorations.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::wm::themes;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowState {
    Normal,
    Minimized,
    Maximized,
    Closed,
}

pub struct Window {
    pub id: u32,
    pub title: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub prev_x: u32,
    pub prev_y: u32,
    pub prev_w: u32,
    pub prev_h: u32,
    pub z_order: u32,
    pub state: WindowState,
    pub focused: bool,
    pub buffer: Vec<u32>,
    pub dirty: bool,
}

pub const TITLE_BAR_HEIGHT: u32 = 24;
pub const BORDER_WIDTH: u32 = 2;

/// Smallest total window width `render_decorations` can paint safely.
///
/// The maximize button starts at `width - 60`; the release profile builds with
/// `panic = "abort"` and no overflow checks, so a narrower window wraps that
/// subtraction to near `u32::MAX`. The resulting index either lands past the end
/// of `buffer` (index-out-of-bounds with no unwind, i.e. the machine stops) or
/// wraps back inside it and silently paints over unrelated pixels. Sixty is the
/// measured floor for the arithmetic; the two side borders on top of it keep the
/// button column off the frame.
pub const MIN_WIDTH: u32 = 60 + BORDER_WIDTH * 2;

/// Smallest total window height: the title bar plus the bottom border, i.e. a
/// window with a zero-height client area. Anything shorter runs the title-bar
/// loop and `client_height` past the end of the buffer.
pub const MIN_HEIGHT: u32 = TITLE_BAR_HEIGHT + BORDER_WIDTH;

/// Largest total window width. A window is never useful beyond the framebuffer,
/// and the largest mode the boot path is offered in practice is 7680x4320 (see
/// the mode-selection note in `boot/uefi_entry.rs`). The cap also keeps
/// `width * height` — computed as `u32` here, and again in `compositor.rs` when a
/// window is resized or maximized — three orders of magnitude clear of wrapping.
pub const MAX_WIDTH: u32 = 7680;

/// Largest total window height. Companion to [`MAX_WIDTH`]; same 7680x4320 mode.
pub const MAX_HEIGHT: u32 = 4320;

impl Window {
    /// Creates a window whose dimensions are always paintable.
    ///
    /// `width` and `height` are the requested *client* size and are clamped, not
    /// rejected: this constructor returns `Self`, so an error return would have
    /// to be handled by every call site, and the only sources of out-of-range
    /// values are unprivileged Aether scripts calling `window.create` with
    /// nonsense. A clamped window still runs; a rejected one would need new
    /// failure handling in the compositor for no gain.
    pub fn new(id: u32, title: &str, x: u32, y: u32, width: u32, height: u32) -> Self {
        let total_w = width
            .saturating_add(BORDER_WIDTH * 2)
            .clamp(MIN_WIDTH, MAX_WIDTH);
        let total_h = height
            .saturating_add(TITLE_BAR_HEIGHT + BORDER_WIDTH)
            .clamp(MIN_HEIGHT, MAX_HEIGHT);
        let theme = themes::current_theme();
        Self {
            id,
            title: String::from(title),
            x,
            y,
            width: total_w,
            height: total_h,
            prev_x: x,
            prev_y: y,
            prev_w: total_w,
            prev_h: total_h,
            z_order: id,
            state: WindowState::Normal,
            focused: false,
            buffer: vec![theme.bg; (total_w * total_h) as usize],
            dirty: true,
        }
    }

    pub fn client_x(&self) -> u32 {
        self.x + BORDER_WIDTH
    }

    pub fn client_y(&self) -> u32 {
        self.y + TITLE_BAR_HEIGHT
    }

    pub fn client_width(&self) -> u32 {
        self.width - BORDER_WIDTH * 2
    }

    pub fn client_height(&self) -> u32 {
        self.height - TITLE_BAR_HEIGHT - BORDER_WIDTH
    }

    pub fn contains(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    pub fn in_title_bar(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + TITLE_BAR_HEIGHT
    }

    pub fn in_close_button(&self, px: u32, py: u32) -> bool {
        let btn_x = self.x + self.width - 20;
        let btn_y = self.y + 4;
        px >= btn_x && px < btn_x + 16 && py >= btn_y && py < btn_y + 16
    }

    pub fn in_minimize_button(&self, px: u32, py: u32) -> bool {
        let btn_x = self.x + self.width - 40;
        let btn_y = self.y + 4;
        px >= btn_x && px < btn_x + 16 && py >= btn_y && py < btn_y + 16
    }

    pub fn in_maximize_button(&self, px: u32, py: u32) -> bool {
        let btn_x = self.x + self.width - 60;
        let btn_y = self.y + 4;
        px >= btn_x && px < btn_x + 16 && py >= btn_y && py < btn_y + 16
    }

    pub fn set_client_pixel(&mut self, x: u32, y: u32, color: u32) {
        let bx = x + BORDER_WIDTH;
        let by = y + TITLE_BAR_HEIGHT;
        if bx < self.width && by < self.height {
            self.buffer[(by * self.width + bx) as usize] = color;
            self.dirty = true;
        }
    }

    pub fn render_decorations(&mut self) {
        let w = self.width;
        let h = self.height;
        let theme = themes::current_theme();
        let title_bg = if self.focused {
            theme.titlebar
        } else {
            theme.titlebar_unfocused
        };

        // Title bar
        for y in 0..TITLE_BAR_HEIGHT {
            for x in 0..w {
                self.buffer[(y * w + x) as usize] = title_bg;
            }
        }

        // Close button
        let btn_x = w - 20;
        for y in 4..20 {
            for x in btn_x..(btn_x + 16) {
                self.buffer[(y * w + x) as usize] = theme.close_btn;
            }
        }

        // Minimize button
        let min_x = w - 40;
        for y in 4..20 {
            for x in min_x..(min_x + 16) {
                self.buffer[(y * w + x) as usize] = theme.titlebar;
            }
        }
        // Minimize symbol (_)
        for x in min_x + 4..min_x + 12 {
            self.buffer[(17 * w + x) as usize] = theme.fg;
        }

        // Maximize button
        let max_x = w - 60;
        for y in 4..20 {
            for x in max_x..(max_x + 16) {
                self.buffer[(y * w + x) as usize] = theme.titlebar;
            }
        }
        // Maximize symbol (square)
        for x in max_x + 3..max_x + 13 {
            self.buffer[(6 * w + x) as usize] = theme.fg;
            self.buffer[(15 * w + x) as usize] = theme.fg;
        }
        for y in 6..16 {
            self.buffer[(y * w + max_x + 3) as usize] = theme.fg;
            self.buffer[(y * w + max_x + 12) as usize] = theme.fg;
        }

        // Title text (simple: just mark the pixels, actual rendering via font in compositor)
        // The compositor will overlay the title text

        // Borders
        for y in 0..h {
            for bx in 0..BORDER_WIDTH {
                self.buffer[(y * w + bx) as usize] = theme.border;
                if bx + w - BORDER_WIDTH < w {
                    self.buffer[(y * w + w - BORDER_WIDTH + bx) as usize] = theme.border;
                }
            }
        }
        // Bottom border
        for y in (h - BORDER_WIDTH)..h {
            for x in 0..w {
                self.buffer[(y * w + x) as usize] = theme.border;
            }
        }

        self.dirty = true;
    }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// The exact path an Aether script takes: `window.create("t", 0, 0)` reaches
    /// `create_window(title, 100, 100, 0, 0)`, which used to build a 4x26 window.
    /// `render_decorations` then computed `min_x = 4 - 40 = 4294967260` and
    /// indexed a 104-element buffer at 4294967276.
    fn test_zero_dimensions_are_floored() -> TestResult {
        let win = Window::new(1, "script", 100, 100, 0, 0);
        test_assert_eq!(win.width, MIN_WIDTH);
        test_assert_eq!(win.height, MIN_HEIGHT);
        test_assert_eq!(win.buffer.len(), (MIN_WIDTH * MIN_HEIGHT) as usize);
        test_assert_eq!(win.prev_w, MIN_WIDTH);
        test_assert_eq!(win.prev_h, MIN_HEIGHT);
        TestResult::Pass
    }

    /// `width + 4` and `height + 26` used to wrap: `u32::MAX - 25` produced a
    /// total height of 0 and therefore a zero-length buffer.
    fn test_wrapping_dimensions_are_capped() -> TestResult {
        for (w, h) in [
            (u32::MAX, u32::MAX),
            (u32::MAX - 3, u32::MAX - 25),
            (65532, 65510),
        ] {
            let win = Window::new(2, "wrap", 0, 0, w, h);
            test_assert!(win.width >= MIN_WIDTH && win.width <= MAX_WIDTH);
            test_assert!(win.height >= MIN_HEIGHT && win.height <= MAX_HEIGHT);
            test_assert_eq!(win.buffer.len(), (win.width as usize) * (win.height as usize));
        }
        TestResult::Pass
    }

    /// Every index `render_decorations` computes must be inside `buffer`, for
    /// the smallest window the constructor can now produce. The call itself is
    /// the real assertion: an out-of-bounds index aborts rather than unwinds.
    fn test_decorations_stay_inside_buffer() -> TestResult {
        let mut win = Window::new(3, "min", 0, 0, 0, 0);
        let w = win.width;
        let h = win.height;
        // The three button origins, none of which may wrap.
        test_assert!(w >= 60);
        let max_x = w - 60;
        // Highest index any decoration loop reaches: title-bar row 19, one past
        // the right edge of the maximize button's 16-pixel span.
        let highest = 19 * w + max_x + 15;
        test_assert!((highest as usize) < win.buffer.len());
        test_assert!(((h * w - 1) as usize) < win.buffer.len());
        win.render_decorations();
        test_assert!(win.dirty);
        TestResult::Pass
    }

    /// Ordinary windows keep their exact dimensions; the clamp only touches the
    /// degenerate ends. 600x400 is the terminal window `app_state` creates.
    fn test_normal_dimensions_unchanged() -> TestResult {
        let win = Window::new(4, "Terminal", 48, 132, 600, 400);
        test_assert_eq!(win.width, 604);
        test_assert_eq!(win.height, 426);
        test_assert_eq!(win.client_width(), 600);
        test_assert_eq!(win.client_height(), 400);
        test_assert_eq!(win.buffer.len(), 604 * 426);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "window::zero_dimensions_are_floored",
            test_zero_dimensions_are_floored,
        );
        crate::testing::register_test(
            "window::wrapping_dimensions_are_capped",
            test_wrapping_dimensions_are_capped,
        );
        crate::testing::register_test(
            "window::decorations_stay_inside_buffer",
            test_decorations_stay_inside_buffer,
        );
        crate::testing::register_test(
            "window::normal_dimensions_unchanged",
            test_normal_dimensions_unchanged,
        );
    }
}
