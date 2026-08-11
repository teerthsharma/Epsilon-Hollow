// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Bottom taskbar: theorem status indicators, clock, start/power buttons.

use alloc::string::String;
use core::sync::atomic::Ordering;

use crate::graphics::font;
use crate::graphics::framebuffer::Framebuffer;
use crate::{GOVERNOR_EPSILON, THEOREM_COUNT, THEOREM_STATES};

const TASKBAR_HEIGHT: u32 = 28;

const POWER_BUTTON_W: u32 = 24;
const POWER_BUTTON_H: u32 = 16;
/// Gap between the right edge of the power button and the right edge of the
/// screen.
const POWER_BUTTON_MARGIN: u32 = 12;

/// Screen rectangle of the power button, as `(x, y, width, height)`.
///
/// The single source of truth for that geometry: `draw_taskbar` fills exactly
/// this rectangle and `wm::desktop::handle_click` hit-tests exactly this
/// rectangle. Both sides used to derive "near the right edge" independently and
/// had drifted four pixels apart, so a click on the right end of the visible
/// button did nothing while a click on bare taskbar to its left opened the menu.
pub fn power_button_rect(fb_width: u32, fb_height: u32) -> (u32, u32, u32, u32) {
    let taskbar_y = fb_height.saturating_sub(TASKBAR_HEIGHT);
    (
        fb_width.saturating_sub(POWER_BUTTON_MARGIN + POWER_BUTTON_W),
        taskbar_y + 6,
        POWER_BUTTON_W,
        POWER_BUTTON_H,
    )
}

pub fn draw_taskbar(fb: &Framebuffer) {
    let theme = crate::wm::themes::current_theme();
    let y = fb.height - TASKBAR_HEIGHT;

    // Background
    fb.fill_rect(0, y, fb.width, TASKBAR_HEIGHT, theme.taskbar);

    // Top border line
    fb.fill_rect(0, y, fb.width, 1, theme.border);

    // Start button (left)
    fb.fill_rect(4, y + 4, 72, 20, theme.accent);
    let start_label = "Seal";
    let start_w = start_label.len() as u32 * font::CHAR_WIDTH;
    let start_x = 4 + (72 - start_w) / 2;
    for (i, ch) in start_label.bytes().enumerate() {
        font::draw_char(
            fb,
            start_x + i as u32 * font::CHAR_WIDTH,
            y + 6,
            ch,
            0xFFFFFF,
        );
    }

    // Theorem indicators (small colored squares)
    for i in 0..THEOREM_COUNT {
        let active = THEOREM_STATES[i].load(Ordering::Relaxed);
        let x = 90 + i as u32 * 18;
        let color = if active { theme.accent } else { 0x00404040 };
        fb.fill_rect(x, y + 8, 10, 10, color);
    }

    // Epsilon value area
    let _epsilon = f64::from_bits(GOVERNOR_EPSILON.load(Ordering::Relaxed));
    fb.fill_rect(292, y + 6, 80, 14, theme.bg);

    // Clock (center)
    let time_str = format_time();
    let clock_w = time_str.len() as u32 * font::CHAR_WIDTH;
    let clock_x = fb.width / 2 - clock_w / 2;
    for (i, ch) in time_str.bytes().enumerate() {
        font::draw_char(
            fb,
            clock_x + i as u32 * font::CHAR_WIDTH,
            y + 6,
            ch,
            theme.fg,
        );
    }

    // Power button (right)
    let (power_x, power_y, power_w, power_h) = power_button_rect(fb.width, fb.height);
    fb.fill_rect(power_x, power_y, power_w, power_h, 0xFF4444);
    font::draw_char(fb, power_x + 8, power_y, b'P', 0xFFFFFF);
}

pub fn taskbar_height() -> u32 {
    TASKBAR_HEIGHT
}

fn format_time() -> String {
    let t = crate::drivers::rtc::read_time();
    alloc::format!("{:02}:{:02}:{:02}", t.hour, t.min, t.sec)
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::wm::power_menu::{PowerAction, PowerMenu};
    use crate::{test_assert, test_assert_eq};

    /// At 1024x768 the button was drawn over x in `[988, 1012)` and hit-tested
    /// over `[984, 1008)`: a click at x=1010 landed on visible red and did
    /// nothing, and a click at x=985 landed on bare taskbar and opened the menu.
    /// The rectangle is now computed once; the drawn extent is unchanged, so the
    /// two ends of the old hit region are the interesting coordinates.
    fn test_power_button_rect_is_the_hit_region() -> TestResult {
        let rect = power_button_rect(1024, 768);
        test_assert_eq!(rect, (988, 746, 24, 16));
        let (x, y, w, h) = rect;
        let my = y + 4;
        let hit = |mx: u32, my: u32| mx >= x && mx < x + w && my >= y && my < y + h;
        test_assert!(hit(1010, my));
        test_assert!(hit(988, my));
        test_assert!(hit(1011, my));
        test_assert!(!hit(985, my));
        test_assert!(!hit(1012, my));
        test_assert!(!hit(1000, y - 1));
        test_assert!(!hit(1000, y + h));
        TestResult::Pass
    }

    /// A framebuffer narrower than the button must not wrap the origin.
    fn test_power_button_rect_survives_tiny_framebuffer() -> TestResult {
        for (w, h) in [(0u32, 0u32), (10, 10), (36, 28)] {
            let (x, y, bw, bh) = power_button_rect(w, h);
            test_assert!(x == 0);
            test_assert_eq!(y, 6);
            test_assert_eq!(bw, 24);
            test_assert_eq!(bh, 16);
        }
        TestResult::Pass
    }

    /// The menu was anchored at the button and ran 110 columns past the right
    /// edge, so `fill_rect` clipped away most of every item. It must now fit,
    /// and `handle_click` must still resolve rows at wherever it is drawn.
    fn test_power_menu_fits_and_stays_hit_testable() -> TestResult {
        let fb_w = 1024;
        let fb_h = 768;
        let taskbar_y = fb_h - taskbar_height();
        let (power_x, _, _, _) = power_button_rect(fb_w, fb_h);

        let mut menu = PowerMenu::new();
        menu.toggle(power_x, taskbar_y, fb_w);
        test_assert!(menu.open);
        test_assert!(menu.x + menu.width <= fb_w);
        test_assert_eq!(menu.y + menu.height, taskbar_y);
        // "Shutdown" is the longest item: 8 characters drawn from `x + 8`.
        test_assert!(menu.x + 8 + 8 * font::CHAR_WIDTH <= fb_w);

        // Same origin the draw uses, so the rows follow the box.
        test_assert!(matches!(
            menu.handle_click(menu.x + 4, menu.y + 8),
            Some(PowerAction::Shutdown)
        ));
        test_assert!(matches!(
            menu.handle_click(menu.x + 4, menu.y + 32),
            Some(PowerAction::Reboot)
        ));
        test_assert!(matches!(
            menu.handle_click(menu.x + 4, menu.y + 56),
            Some(PowerAction::Logout)
        ));
        test_assert!(menu.handle_click(menu.x - 1, menu.y + 8).is_none());
        test_assert!(menu
            .handle_click(menu.x + menu.width, menu.y + 8)
            .is_none());
        TestResult::Pass
    }

    /// A framebuffer narrower than the menu clamps to the left edge instead of
    /// wrapping the origin to near `u32::MAX`.
    fn test_power_menu_clamps_on_narrow_framebuffer() -> TestResult {
        for fb_w in [0u32, 120, 149] {
            let mut menu = PowerMenu::new();
            menu.toggle(power_button_rect(fb_w, 768).0, 740, fb_w);
            test_assert_eq!(menu.x, 0);
        }
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "taskbar::power_button_rect_is_the_hit_region",
            test_power_button_rect_is_the_hit_region,
        );
        crate::testing::register_test(
            "taskbar::power_button_rect_survives_tiny_framebuffer",
            test_power_button_rect_survives_tiny_framebuffer,
        );
        crate::testing::register_test(
            "taskbar::power_menu_fits_and_stays_hit_testable",
            test_power_menu_fits_and_stays_hit_testable,
        );
        crate::testing::register_test(
            "taskbar::power_menu_clamps_on_narrow_framebuffer",
            test_power_menu_clamps_on_narrow_framebuffer,
        );
    }
}
