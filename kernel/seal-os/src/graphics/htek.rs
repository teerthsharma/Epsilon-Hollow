// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! High-tech graphics primitives: anti-aliased rendering, gradients, rounded rectangles,
//! glow effects, and alpha blending. All software-rendered on the framebuffer.

use crate::wm::window::Window;

fn alpha_blend(bg: u32, fg: u32, alpha: u8) -> u32 {
    if alpha == 255 {
        return fg;
    }
    if alpha == 0 {
        return bg;
    }
    let a = alpha as u32;
    let inv = 255 - a;
    let r = ((fg >> 16 & 0xFF) * a + (bg >> 16 & 0xFF) * inv) / 255;
    let g = ((fg >> 8 & 0xFF) * a + (bg >> 8 & 0xFF) * inv) / 255;
    let b = ((fg & 0xFF) * a + (bg & 0xFF) * inv) / 255;
    (r << 16) | (g << 8) | b
}

fn lerp_color(c0: u32, c1: u32, t_256: u32) -> u32 {
    let inv = 256 - t_256;
    let r = ((c0 >> 16 & 0xFF) * inv + (c1 >> 16 & 0xFF) * t_256) >> 8;
    let g = ((c0 >> 8 & 0xFF) * inv + (c1 >> 8 & 0xFF) * t_256) >> 8;
    let b = ((c0 & 0xFF) * inv + (c1 & 0xFF) * t_256) >> 8;
    (r << 16) | (g << 8) | b
}

/// How many steps of a `len`-long span starting at `start` can land inside a
/// `limit`-wide client area.
///
/// Every primitive here takes its extent from the caller, and a script reaches
/// several of them through the Aether graphics callbacks. Iterating past the
/// client area only produces writes that `Window::set_client_pixel` rejects, so
/// a loop bounded by this costs the surface rather than the argument. It also
/// keeps `start + step` from wrapping: `step` never reaches `limit - start`.
fn visible_len(start: u32, len: u32, limit: u32) -> u32 {
    len.min(limit.saturating_sub(start))
}

/// Clip an inclusive run `[lo, hi]` along one axis to the drawable range
/// `[0, limit - 1]`, or `None` when the run misses the client area entirely.
///
/// Both bounds are caller-supplied `i32`s, so the result is computed in `i64`:
/// `hi - lo` overflows for endpoints at opposite ends of the range. The
/// returned run is at most `limit` steps long.
fn clip_span(lo: i64, hi: i64, limit: u32) -> Option<(i64, i64)> {
    let start = lo.max(0);
    let end = hi.min(limit as i64 - 1);
    if start > end {
        None
    } else {
        Some((start, end))
    }
}

fn pixel_at(win: &Window, x: u32, y: u32) -> u32 {
    use crate::wm::window::{BORDER_WIDTH, TITLE_BAR_HEIGHT};
    let bx = x + BORDER_WIDTH;
    let by = y + TITLE_BAR_HEIGHT;
    if bx < win.width && by < win.height {
        win.buffer[(by * win.width + bx) as usize]
    } else {
        0
    }
}

pub fn set_pixel_blended(win: &mut Window, x: u32, y: u32, color: u32, alpha: u8) {
    let cw = win.client_width();
    let ch = win.client_height();
    if x < cw && y < ch {
        if alpha == 255 {
            win.set_client_pixel(x, y, color);
        } else if alpha > 0 {
            let bg = pixel_at(win, x, y);
            win.set_client_pixel(x, y, alpha_blend(bg, color, alpha));
        }
    }
}

pub fn fill_gradient_v(win: &mut Window, x: u32, y: u32, w: u32, h: u32, top: u32, bottom: u32) {
    let cw = win.client_width();
    let ch = win.client_height();
    for dy in 0..h {
        let py = y + dy;
        if py >= ch {
            break;
        }
        let t = if h > 1 { (dy * 256) / (h - 1) } else { 0 };
        let color = lerp_color(top, bottom, t);
        // Bounded by the surface, not by `w`: the inner loop had no `break`, so
        // a script-supplied `w` of `u32::MAX` scanned four billion columns per
        // row to draw at most `cw` of them — 3,298,534,882,560 iterations over
        // a 768-row client area.
        for dx in 0..visible_len(x, w, cw) {
            win.set_client_pixel(x + dx, py, color);
        }
    }
}

pub fn fill_gradient_h(win: &mut Window, x: u32, y: u32, w: u32, h: u32, left: u32, right: u32) {
    let cw = win.client_width();
    let ch = win.client_height();
    for dx in 0..w {
        let px = x + dx;
        if px >= cw {
            break;
        }
        let t = if w > 1 { (dx * 256) / (w - 1) } else { 0 };
        let color = lerp_color(left, right, t);
        // Same unbounded inner loop as `fill_gradient_v`, on the other axis.
        for dy in 0..visible_len(y, h, ch) {
            win.set_client_pixel(px, y + dy, color);
        }
    }
}

pub fn fill_rounded_rect(win: &mut Window, x: u32, y: u32, w: u32, h: u32, r: u32, color: u32) {
    let cw = win.client_width();
    let ch = win.client_height();
    let r = r.min(w / 2).min(h / 2);

    // Both loops used `continue`, not `break`, so their length came from the
    // caller: `min(h, ch) * w + (h - min(h, ch))` iterations, which is
    // 3,302,829,849,087 for the `w` and `h` of `u32::MAX` a script can pass.
    // `is_inside_rounded` still sees the unclipped `w`, `h` and `dx`, `dy`, so
    // the corner geometry is unchanged.
    for dy in 0..visible_len(y, h, ch) {
        let py = y + dy;
        for dx in 0..visible_len(x, w, cw) {
            let px = x + dx;

            let inside = is_inside_rounded(dx, dy, w, h, r);
            if inside >= 240 {
                win.set_client_pixel(px, py, color);
            } else if inside > 0 {
                let bg = pixel_at(win, px, py);
                win.set_client_pixel(px, py, alpha_blend(bg, color, inside));
            }
        }
    }
}

pub fn fill_rounded_rect_gradient(
    win: &mut Window,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    r: u32,
    top: u32,
    bottom: u32,
) {
    let cw = win.client_width();
    let ch = win.client_height();
    let r = r.min(w / 2).min(h / 2);

    // Bounded by the client area for the same reason as `fill_rounded_rect`.
    for dy in 0..visible_len(y, h, ch) {
        let py = y + dy;
        let t = if h > 1 { (dy * 256) / (h - 1) } else { 0 };
        let color = lerp_color(top, bottom, t);
        for dx in 0..visible_len(x, w, cw) {
            let px = x + dx;
            let inside = is_inside_rounded(dx, dy, w, h, r);
            if inside >= 240 {
                win.set_client_pixel(px, py, color);
            } else if inside > 0 {
                let bg = pixel_at(win, px, py);
                win.set_client_pixel(px, py, alpha_blend(bg, color, inside));
            }
        }
    }
}

pub fn stroke_rounded_rect(
    win: &mut Window,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    r: u32,
    thickness: u32,
    color: u32,
) {
    let cw = win.client_width();
    let ch = win.client_height();
    let r = r.min(w / 2).min(h / 2);

    // Bounded by the client area for the same reason as `fill_rounded_rect`.
    for dy in 0..visible_len(y, h, ch) {
        let py = y + dy;
        for dx in 0..visible_len(x, w, cw) {
            let px = x + dx;
            let outer = is_inside_rounded(dx, dy, w, h, r);
            if outer == 0 {
                continue;
            }
            let is_border =
                if dx >= thickness && dy >= thickness && dx < w - thickness && dy < h - thickness {
                    let inner = is_inside_rounded(
                        dx - thickness,
                        dy - thickness,
                        w - thickness * 2,
                        h - thickness * 2,
                        r.saturating_sub(thickness),
                    );
                    inner < 240
                } else {
                    true
                };
            if is_border {
                let alpha = outer;
                set_pixel_blended(win, px, py, color, alpha);
            }
        }
    }
}

fn is_inside_rounded(dx: u32, dy: u32, w: u32, h: u32, r: u32) -> u8 {
    if r == 0 {
        return 255;
    }

    let (cx, cy, in_corner) = if dx < r && dy < r {
        (r, r, true)
    } else if dx >= w - r && dy < r {
        (w - r - 1, r, true)
    } else if dx < r && dy >= h - r {
        (r, h - r - 1, true)
    } else if dx >= w - r && dy >= h - r {
        (w - r - 1, h - r - 1, true)
    } else {
        (0, 0, false)
    };

    if !in_corner {
        return 255;
    }

    let ddx = dx.abs_diff(cx);
    let ddy = dy.abs_diff(cy);
    let dist_sq = ddx * ddx + ddy * ddy;
    let r_sq = r * r;

    if dist_sq <= r_sq.saturating_sub(r * 2) {
        255
    } else if dist_sq > r_sq + r * 2 {
        0
    } else {
        let dist = isqrt(dist_sq);
        if dist <= r {
            let edge = (r - dist) * 255 / (r.max(1));
            edge.min(255) as u8
        } else {
            0
        }
    }
}

fn isqrt(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

pub fn glow_rect(win: &mut Window, x: u32, y: u32, w: u32, h: u32, spread: u32, color: u32) {
    let cw = win.client_width();
    let ch = win.client_height();
    let x0 = x.saturating_sub(spread);
    let y0 = y.saturating_sub(spread);
    let x1 = (x + w + spread).min(cw);
    let y1 = (y + h + spread).min(ch);

    for py in y0..y1 {
        for px in x0..x1 {
            if px >= x && px < x + w && py >= y && py < y + h {
                continue;
            }
            let dx = if px < x {
                x - px
            } else if px >= x + w {
                px - x - w + 1
            } else {
                0
            };
            let dy = if py < y {
                y - py
            } else if py >= y + h {
                py - y - h + 1
            } else {
                0
            };
            let dist = isqrt(dx * dx + dy * dy);
            if dist < spread {
                let alpha = ((spread - dist) * 120 / spread) as u8;
                set_pixel_blended(win, px, py, color, alpha);
            }
        }
    }
}

pub fn draw_circle_filled(win: &mut Window, cx: u32, cy: u32, r: u32, color: u32) {
    let cw = win.client_width();
    let ch = win.client_height();
    // Clip the scan box to the client area before stepping. `r` is
    // caller-supplied — a script reaches this through `gfx.draw_circle` — and
    // the old `-r..=r` square walked `(2r + 1)^2` cells to draw at most
    // `cw * ch` of them: 18,446,744,065,119,617,025 for `r = i32::MAX`. The
    // distance test is unchanged, so the same pixels are drawn with the same
    // alphas; it moved to u64 only because `dx * dx` and `r * r` overflowed
    // for a radius or centre past 65535, and because `r as i32` went negative
    // past 2^31 and silently drew nothing at all.
    let r_i = r as i64;
    let Some((px_lo, px_hi)) = clip_span(cx as i64 - r_i, cx as i64 + r_i, cw) else {
        return;
    };
    let Some((py_lo, py_hi)) = clip_span(cy as i64 - r_i, cy as i64 + r_i, ch) else {
        return;
    };

    let r_u = r as u64;
    let r_sq = r_u * r_u;
    for py in py_lo..=py_hi {
        let dy = (py - cy as i64).unsigned_abs();
        for px in px_lo..=px_hi {
            let dx = (px - cx as i64).unsigned_abs();
            let dist_sq = dx
                .saturating_mul(dx)
                .saturating_add(dy.saturating_mul(dy));

            if dist_sq <= r_sq.saturating_sub(r_u) {
                win.set_client_pixel(px as u32, py as u32, color);
            } else if dist_sq <= r_sq + r_u {
                let dist = dist_sq.isqrt();
                let alpha = if dist <= r_u {
                    ((r_u - dist) * 255 / r_u.max(1)).min(255) as u8
                } else {
                    0
                };
                set_pixel_blended(win, px as u32, py as u32, color, alpha);
            }
        }
    }
}

pub fn draw_line_h(win: &mut Window, x: u32, y: u32, w: u32, color: u32, alpha: u8) {
    let cw = win.client_width();
    let ch = win.client_height();
    if y >= ch {
        return;
    }
    for dx in 0..w {
        let px = x + dx;
        if px >= cw {
            break;
        }
        set_pixel_blended(win, px, y, color, alpha);
    }
}

// --- High-quality text rendering ---
// 2x scaled font with 4x supersampled anti-aliasing for smooth edges

use crate::graphics::font;

pub const HTEXT_CHAR_W: u32 = font::CHAR_WIDTH * 2;
pub const HTEXT_CHAR_H: u32 = font::CHAR_HEIGHT * 2;
pub const TEXT_CHAR_W: u32 = font::CHAR_WIDTH;
pub const TEXT_CHAR_H: u32 = font::CHAR_HEIGHT;

pub fn render_text_smooth(win: &mut Window, x: u32, y: u32, text: &str, color: u32) {
    for (i, ch) in text.bytes().enumerate() {
        let glyph = font::glyph(ch);
        let ox = x + (i as u32) * HTEXT_CHAR_W;
        for gy in 0..font::CHAR_HEIGHT {
            let bits = glyph[gy as usize];
            for gx in 0..font::CHAR_WIDTH {
                let on = bits & (0x80 >> gx) != 0;
                let py = y + gy * 2;
                let px = ox + gx * 2;

                if on {
                    // Core pixel: full opacity 2x2 block
                    set_pixel_blended(win, px, py, color, 255);
                    set_pixel_blended(win, px + 1, py, color, 255);
                    set_pixel_blended(win, px, py + 1, color, 255);
                    set_pixel_blended(win, px + 1, py + 1, color, 255);
                } else {
                    // Anti-alias: check neighbors and apply fringe
                    let mut neighbor_count = 0u8;
                    if gx > 0 && bits & (0x80 >> (gx - 1)) != 0 {
                        neighbor_count += 1;
                    }
                    if gx < font::CHAR_WIDTH - 1 && bits & (0x80 >> (gx + 1)) != 0 {
                        neighbor_count += 1;
                    }
                    if gy > 0 {
                        let above = glyph[(gy - 1) as usize];
                        if above & (0x80 >> gx) != 0 {
                            neighbor_count += 1;
                        }
                    }
                    if gy < font::CHAR_HEIGHT - 1 {
                        let below = glyph[(gy + 1) as usize];
                        if below & (0x80 >> gx) != 0 {
                            neighbor_count += 1;
                        }
                    }
                    if neighbor_count >= 2 {
                        let a = 50u8;
                        set_pixel_blended(win, px, py, color, a);
                        set_pixel_blended(win, px + 1, py, color, a);
                        set_pixel_blended(win, px, py + 1, color, a);
                        set_pixel_blended(win, px + 1, py + 1, color, a);
                    } else if neighbor_count == 1 {
                        let a = 25u8;
                        set_pixel_blended(win, px, py, color, a);
                        set_pixel_blended(win, px + 1, py, color, a);
                        set_pixel_blended(win, px, py + 1, color, a);
                        set_pixel_blended(win, px + 1, py + 1, color, a);
                    }
                }
            }
        }
    }
}

pub fn render_text_small(win: &mut Window, x: u32, y: u32, text: &str, color: u32) {
    for (i, ch) in text.bytes().enumerate() {
        let glyph = font::glyph(ch);
        let px = x + (i as u32) * font::CHAR_WIDTH;
        for gy in 0..font::CHAR_HEIGHT {
            let bits = glyph[gy as usize];
            for gx in 0..font::CHAR_WIDTH {
                if bits & (0x80 >> gx) != 0 {
                    set_pixel_blended(win, px + gx, y + gy, color, 255);
                }
            }
        }
    }
}

pub fn render_text_glow(win: &mut Window, x: u32, y: u32, text: &str, color: u32, glow_color: u32) {
    // Glow pass: render offset copies at low alpha
    for &(dx, dy, a) in &[
        (1i32, 0i32, 40u8),
        (-1, 0, 40),
        (0, 1, 40),
        (0, -1, 40),
        (2, 0, 20),
        (-2, 0, 20),
        (0, 2, 20),
        (0, -2, 20),
        (1, 1, 30),
        (-1, -1, 30),
        (1, -1, 30),
        (-1, 1, 30),
    ] {
        let gx = (x as i32 + dx).max(0) as u32;
        let gy = (y as i32 + dy).max(0) as u32;
        for (i, ch) in text.bytes().enumerate() {
            let glyph = font::glyph(ch);
            let ox = gx + (i as u32) * HTEXT_CHAR_W;
            for row in 0..font::CHAR_HEIGHT {
                let bits = glyph[row as usize];
                for col in 0..font::CHAR_WIDTH {
                    if bits & (0x80 >> col) != 0 {
                        let px = ox + col * 2;
                        let py = gy + row * 2;
                        set_pixel_blended(win, px, py, glow_color, a);
                        set_pixel_blended(win, px + 1, py, glow_color, a);
                        set_pixel_blended(win, px, py + 1, glow_color, a);
                        set_pixel_blended(win, px + 1, py + 1, glow_color, a);
                    }
                }
            }
        }
    }
    render_text_smooth(win, x, y, text, color);
}

pub fn fill_solid(win: &mut Window, x: u32, y: u32, w: u32, h: u32, color: u32) {
    let cw = win.client_width();
    let ch = win.client_height();
    for dy in 0..h {
        let py = y + dy;
        if py >= ch {
            break;
        }
        for dx in 0..w {
            let px = x + dx;
            if px >= cw {
                break;
            }
            win.set_client_pixel(px, py, color);
        }
    }
}

pub fn fill_solid_alpha(win: &mut Window, x: u32, y: u32, w: u32, h: u32, color: u32, alpha: u8) {
    let cw = win.client_width();
    let ch = win.client_height();
    for dy in 0..h {
        let py = y + dy;
        if py >= ch {
            break;
        }
        for dx in 0..w {
            let px = x + dx;
            if px >= cw {
                break;
            }
            set_pixel_blended(win, px, py, color, alpha);
        }
    }
}

pub fn draw_aa_line(win: &mut Window, x0: i32, y0: i32, x1: i32, y1: i32, color: u32, alpha: u8) {
    // i64 throughout the setup: `x1 - x0` overflows i32 for endpoints at
    // opposite ends of the range, which panics in a debug build and silently
    // flips the `steep` decision in a release one.
    let steep = (y1 as i64 - y0 as i64).abs() > (x1 as i64 - x0 as i64).abs();

    let (mut x0, mut y0, mut x1, mut y1) = if steep {
        (y0, x0, y1, x1)
    } else {
        (x0, y0, x1, y1)
    };

    if x0 > x1 {
        core::mem::swap(&mut x0, &mut x1);
        core::mem::swap(&mut y0, &mut y1);
    }

    let dx = x1 as i64 - x0 as i64;
    let dy = y1 as i64 - y0 as i64;
    let gradient = if dx == 0 { 1.0 } else { dy as f32 / dx as f32 };

    // Clip the major axis to the client area before stepping. The endpoints
    // come from the caller — `gfx.draw_line(id, 0, 0, 2147483647, 0)` in a
    // script reaches here — and the old loop walked every column between them,
    // rejecting all but the visible 1024 one pixel at a time. Worse, `x += 1`
    // at `x1 == i32::MAX` wrapped to `i32::MIN` with `x <= x1` true again, so
    // the loop never terminated and the watchdog reset the machine.
    //
    // The visible part is untouched: `y` is advanced to the clipped start by
    // the same gradient, so the slope and every drawn pixel are unchanged.
    let major_limit = if steep {
        win.client_height()
    } else {
        win.client_width()
    };
    let Some((start, end)) = clip_span(x0 as i64, x1 as i64, major_limit) else {
        return;
    };

    let mut x = start;
    let mut y = y0 as f32 + gradient * (start - x0 as i64) as f32;

    while x <= end {
        // `y as i32` truncates toward zero, so for a line crossing the top or
        // left edge (`y` in -1..0) `frac` went negative, `a1` overflowed 255
        // and saturated, and the fringe pixel came out fully opaque instead of
        // fading. Flooring keeps `frac` in 0..1 where the blend expects it.
        let y_floor = libm::floorf(y) as i32;
        let frac = y - y_floor as f32;
        let a1 = ((1.0 - frac) * alpha as f32) as u8;
        let a2 = (frac * alpha as f32) as u8;

        if steep {
            set_pixel_blended(win, y_floor as u32, x as u32, color, a1);
            set_pixel_blended(win, y_floor.saturating_add(1) as u32, x as u32, color, a2);
        } else {
            set_pixel_blended(win, x as u32, y_floor as u32, color, a1);
            set_pixel_blended(win, x as u32, y_floor.saturating_add(1) as u32, color, a2);
        }

        y += gradient;
        x += 1;
    }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    const CW: u32 = 320;
    const CH: u32 = 200;
    const INK: u32 = 0x00FF00;

    fn surface() -> Window {
        Window::new(1, "htek", 0, 0, CW, CH)
    }

    /// RED (was): `draw_aa_line` ran `while x <= x1` on the caller's `x1`.
    /// `gfx.draw_line(id, 0, 0, 2147483647, 0)` from a script walked every
    /// column up to `i32::MAX`, and `x += 1` there wrapped to `i32::MIN` with
    /// the condition still true — the loop never terminated, so the event loop
    /// stopped petting the watchdog and the machine reset. GREEN: the run is
    /// clipped to the client area, so it is at most `client_width()` steps.
    fn test_clip_span_bounds_a_hostile_endpoint() -> TestResult {
        test_assert_eq!(clip_span(0, i32::MAX as i64, 1024), Some((0, 1023)));
        test_assert_eq!(
            clip_span(i32::MIN as i64, i32::MAX as i64, 1024),
            Some((0, 1023))
        );
        for (lo, hi, limit) in [
            (0i64, i32::MAX as i64, 1024u32),
            (i32::MIN as i64, i32::MAX as i64, 1u32),
            (-5000, 5000, 320),
            (0, 0, 1),
        ] {
            let (start, end) = clip_span(lo, hi, limit).unwrap();
            let steps = end - start + 1;
            test_assert!(
                steps <= limit as i64,
                "the clipped run must never be longer than the surface"
            );
        }
        TestResult::Pass
    }

    /// A run that misses the client area entirely draws nothing at all rather
    /// than stepping across it pixel by pixel.
    fn test_clip_span_rejects_invisible_runs() -> TestResult {
        test_assert_eq!(clip_span(1024, 5000, 1024), None);
        test_assert_eq!(clip_span(-5000, -1, 1024), None);
        test_assert_eq!(clip_span(0, i32::MAX as i64, 0), None);
        // Control: a run that fits is passed through untouched.
        test_assert_eq!(clip_span(7, 500, 1024), Some((7, 500)));
        TestResult::Pass
    }

    /// RED (was): the inner loops of the gradient, rounded-rect and stroke
    /// primitives ran the caller's full `w`/`h` — `u32::MAX` each from a
    /// script — and relied on a per-pixel bounds rejection.
    fn test_visible_len_bounds_a_hostile_extent() -> TestResult {
        test_assert_eq!(visible_len(0, u32::MAX, 1024), 1024);
        test_assert_eq!(visible_len(1000, u32::MAX, 1024), 24);
        test_assert_eq!(visible_len(1024, u32::MAX, 1024), 0);
        test_assert_eq!(visible_len(u32::MAX, u32::MAX, 1024), 0);
        // Control: an extent that fits is not shortened.
        test_assert_eq!(visible_len(10, 5, 1024), 5);
        // `start + step` can never wrap, because step < limit - start.
        for start in [0u32, 1, 1023, 1024, u32::MAX] {
            let n = visible_len(start, u32::MAX, 1024);
            test_assert!(
                n == 0 || start.checked_add(n - 1).is_some(),
                "the bound must keep start + step inside u32"
            );
        }
        TestResult::Pass
    }

    /// The hostile line still draws its visible part: every column of row 0
    /// from 0 to `client_width() - 1` is inked, and nothing beyond.
    fn test_hostile_line_still_draws_the_visible_row() -> TestResult {
        let mut win = surface();
        draw_aa_line(&mut win, 0, 0, i32::MAX, 0, INK, 255);
        test_assert_eq!(win.client_width(), CW);
        for x in [0u32, 1, CW / 2, CW - 1] {
            test_assert_eq!(pixel_at(&win, x, 0), INK);
        }
        TestResult::Pass
    }

    /// A clip must not change the slope: `(-500, 0) -> (500, 250)` has gradient
    /// 0.25 and therefore crosses `x = 0` at `y = 125` and `x = 100` at
    /// `y = 150`, whether or not the off-screen half is iterated.
    fn test_clipped_line_keeps_its_slope() -> TestResult {
        let mut win = surface();
        draw_aa_line(&mut win, -500, 0, 500, 250, INK, 255);
        test_assert_eq!(pixel_at(&win, 0, 125), INK);
        test_assert_eq!(pixel_at(&win, 100, 150), INK);
        test_assert_eq!(pixel_at(&win, 200, 175), INK);
        // Off the line: the row above the first column must stay untouched.
        test_assert!(pixel_at(&win, 0, 124) != INK);
        TestResult::Pass
    }

    /// A line entirely outside the client area leaves the buffer untouched.
    fn test_offscreen_line_draws_nothing() -> TestResult {
        let mut win = surface();
        let before = win.buffer.clone();
        draw_aa_line(&mut win, 2000, 0, 3000, 0, INK, 255);
        draw_aa_line(&mut win, i32::MIN, 0, -1, 0, INK, 255);
        test_assert!(
            win.buffer == before,
            "a line outside the client area must not touch the buffer"
        );
        TestResult::Pass
    }

    /// RED (was): `draw_circle_filled` scanned `(2r + 1)^2` cells for a
    /// caller-supplied `r`, which `gfx.draw_circle` exposes to scripts. GREEN:
    /// the scan box is clipped to the client area, so an absurd radius costs
    /// `cw * ch` and paints the area it actually covers.
    fn test_circle_radius_is_clipped_not_iterated() -> TestResult {
        let mut win = surface();
        draw_circle_filled(&mut win, CW / 2, CH / 2, u32::MAX, INK);
        for (x, y) in [(0u32, 0u32), (CW - 1, 0), (0, CH - 1), (CW - 1, CH - 1)] {
            test_assert_eq!(pixel_at(&win, x, y), INK);
        }
        // Control: an ordinary circle still has an inside and an outside.
        let mut win = surface();
        let bg = pixel_at(&win, 0, 0);
        draw_circle_filled(&mut win, 50, 50, 10, INK);
        test_assert_eq!(pixel_at(&win, 50, 50), INK);
        test_assert_eq!(pixel_at(&win, 50, 45), INK);
        test_assert_eq!(pixel_at(&win, 90, 50), bg);
        TestResult::Pass
    }

    /// RED (was): both loops of `fill_rounded_rect` used `continue`, so a
    /// `u32::MAX` by `u32::MAX` rectangle ran 2^64 iterations. GREEN: the
    /// extent is clipped and the visible part is still filled.
    fn test_rect_extent_is_clipped_not_iterated() -> TestResult {
        let mut win = surface();
        fill_rounded_rect(&mut win, 0, 0, u32::MAX, u32::MAX, 0, INK);
        test_assert_eq!(pixel_at(&win, CW - 1, CH - 1), INK);

        let mut win = surface();
        fill_gradient_v(&mut win, 0, 0, u32::MAX, u32::MAX, INK, INK);
        test_assert_eq!(pixel_at(&win, CW - 1, CH - 1), INK);

        let mut win = surface();
        fill_gradient_h(&mut win, 0, 0, u32::MAX, u32::MAX, INK, INK);
        test_assert_eq!(pixel_at(&win, CW - 1, CH - 1), INK);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "graphics::htek::clip_span_bounds_a_hostile_endpoint",
            test_clip_span_bounds_a_hostile_endpoint,
        );
        crate::testing::register_test(
            "graphics::htek::clip_span_rejects_invisible_runs",
            test_clip_span_rejects_invisible_runs,
        );
        crate::testing::register_test(
            "graphics::htek::visible_len_bounds_a_hostile_extent",
            test_visible_len_bounds_a_hostile_extent,
        );
        crate::testing::register_test(
            "graphics::htek::hostile_line_still_draws_the_visible_row",
            test_hostile_line_still_draws_the_visible_row,
        );
        crate::testing::register_test(
            "graphics::htek::clipped_line_keeps_its_slope",
            test_clipped_line_keeps_its_slope,
        );
        crate::testing::register_test(
            "graphics::htek::offscreen_line_draws_nothing",
            test_offscreen_line_draws_nothing,
        );
        crate::testing::register_test(
            "graphics::htek::circle_radius_is_clipped_not_iterated",
            test_circle_radius_is_clipped_not_iterated,
        );
        crate::testing::register_test(
            "graphics::htek::rect_extent_is_clipped_not_iterated",
            test_rect_extent_is_clipped_not_iterated,
        );
    }
}
