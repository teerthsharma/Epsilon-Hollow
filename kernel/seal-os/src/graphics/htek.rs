// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! High-tech graphics primitives: anti-aliased rendering, gradients, rounded rectangles,
//! glow effects, and alpha blending. All software-rendered on the framebuffer.

use crate::wm::window::Window;

fn alpha_blend(bg: u32, fg: u32, alpha: u8) -> u32 {
    if alpha == 255 { return fg; }
    if alpha == 0 { return bg; }
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
        if py >= ch { break; }
        let t = if h > 1 { (dy * 256) / (h - 1) } else { 0 };
        let color = lerp_color(top, bottom, t);
        for dx in 0..w {
            let px = x + dx;
            if px < cw {
                win.set_client_pixel(px, py, color);
            }
        }
    }
}

pub fn fill_gradient_h(win: &mut Window, x: u32, y: u32, w: u32, h: u32, left: u32, right: u32) {
    let cw = win.client_width();
    let ch = win.client_height();
    for dx in 0..w {
        let px = x + dx;
        if px >= cw { break; }
        let t = if w > 1 { (dx * 256) / (w - 1) } else { 0 };
        let color = lerp_color(left, right, t);
        for dy in 0..h {
            let py = y + dy;
            if py < ch {
                win.set_client_pixel(px, py, color);
            }
        }
    }
}

pub fn fill_rounded_rect(win: &mut Window, x: u32, y: u32, w: u32, h: u32, r: u32, color: u32) {
    let cw = win.client_width();
    let ch = win.client_height();
    let r = r.min(w / 2).min(h / 2);

    for dy in 0..h {
        let py = y + dy;
        if py >= ch { continue; }
        for dx in 0..w {
            let px = x + dx;
            if px >= cw { continue; }

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
    win: &mut Window, x: u32, y: u32, w: u32, h: u32, r: u32, top: u32, bottom: u32,
) {
    let cw = win.client_width();
    let ch = win.client_height();
    let r = r.min(w / 2).min(h / 2);

    for dy in 0..h {
        let py = y + dy;
        if py >= ch { continue; }
        let t = if h > 1 { (dy * 256) / (h - 1) } else { 0 };
        let color = lerp_color(top, bottom, t);
        for dx in 0..w {
            let px = x + dx;
            if px >= cw { continue; }
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
    win: &mut Window, x: u32, y: u32, w: u32, h: u32, r: u32, thickness: u32, color: u32,
) {
    let cw = win.client_width();
    let ch = win.client_height();
    let r = r.min(w / 2).min(h / 2);

    for dy in 0..h {
        let py = y + dy;
        if py >= ch { continue; }
        for dx in 0..w {
            let px = x + dx;
            if px >= cw { continue; }
            let outer = is_inside_rounded(dx, dy, w, h, r);
            if outer == 0 { continue; }
            let is_border = if dx >= thickness && dy >= thickness
                && dx < w - thickness && dy < h - thickness
            {
                let inner = is_inside_rounded(
                    dx - thickness, dy - thickness,
                    w - thickness * 2, h - thickness * 2,
                    r.saturating_sub(thickness),
                );
                inner < 240
            } else {
                true
            };
            if is_border {
                let alpha = outer.min(255);
                set_pixel_blended(win, px, py, color, alpha);
            }
        }
    }
}

fn is_inside_rounded(dx: u32, dy: u32, w: u32, h: u32, r: u32) -> u8 {
    if r == 0 { return 255; }

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

    if !in_corner { return 255; }

    let ddx = if dx > cx { dx - cx } else { cx - dx };
    let ddy = if dy > cy { dy - cy } else { cy - dy };
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
    if n == 0 { return 0; }
    let mut x = n;
    let mut y = (x + 1) / 2;
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
            if px >= x && px < x + w && py >= y && py < y + h { continue; }
            let dx = if px < x { x - px } else if px >= x + w { px - x - w + 1 } else { 0 };
            let dy = if py < y { y - py } else if py >= y + h { py - y - h + 1 } else { 0 };
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
    let r_i = r as i32;

    for dy in -r_i..=r_i {
        for dx in -r_i..=r_i {
            let dist_sq = (dx * dx + dy * dy) as u32;
            let r_sq = r * r;
            let px = cx as i32 + dx;
            let py = cy as i32 + dy;
            if px < 0 || py < 0 { continue; }
            let px = px as u32;
            let py = py as u32;
            if px >= cw || py >= ch { continue; }

            if dist_sq <= r_sq.saturating_sub(r) {
                win.set_client_pixel(px, py, color);
            } else if dist_sq <= r_sq + r {
                let dist = isqrt(dist_sq);
                let alpha = if dist <= r {
                    ((r - dist) * 255 / r.max(1)).min(255) as u8
                } else {
                    0
                };
                set_pixel_blended(win, px, py, color, alpha);
            }
        }
    }
}

pub fn draw_line_h(win: &mut Window, x: u32, y: u32, w: u32, color: u32, alpha: u8) {
    let cw = win.client_width();
    let ch = win.client_height();
    if y >= ch { return; }
    for dx in 0..w {
        let px = x + dx;
        if px >= cw { break; }
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
                    if gx > 0 && bits & (0x80 >> (gx - 1)) != 0 { neighbor_count += 1; }
                    if gx < font::CHAR_WIDTH - 1 && bits & (0x80 >> (gx + 1)) != 0 { neighbor_count += 1; }
                    if gy > 0 {
                        let above = glyph[(gy - 1) as usize];
                        if above & (0x80 >> gx) != 0 { neighbor_count += 1; }
                    }
                    if gy < font::CHAR_HEIGHT - 1 {
                        let below = glyph[(gy + 1) as usize];
                        if below & (0x80 >> gx) != 0 { neighbor_count += 1; }
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
    for &(dx, dy, a) in &[(1i32, 0i32, 40u8), (-1, 0, 40), (0, 1, 40), (0, -1, 40),
                           (2, 0, 20), (-2, 0, 20), (0, 2, 20), (0, -2, 20),
                           (1, 1, 30), (-1, -1, 30), (1, -1, 30), (-1, 1, 30)] {
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
        if py >= ch { break; }
        for dx in 0..w {
            let px = x + dx;
            if px >= cw { break; }
            win.set_client_pixel(px, py, color);
        }
    }
}

pub fn fill_solid_alpha(win: &mut Window, x: u32, y: u32, w: u32, h: u32, color: u32, alpha: u8) {
    let cw = win.client_width();
    let ch = win.client_height();
    for dy in 0..h {
        let py = y + dy;
        if py >= ch { break; }
        for dx in 0..w {
            let px = x + dx;
            if px >= cw { break; }
            set_pixel_blended(win, px, py, color, alpha);
        }
    }
}

pub fn draw_aa_line(win: &mut Window, x0: i32, y0: i32, x1: i32, y1: i32, color: u32, alpha: u8) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steep = dy.abs() > dx.abs();

    let (mut x0, mut y0, mut x1, mut y1) = if steep {
        (y0, x0, y1, x1)
    } else {
        (x0, y0, x1, y1)
    };

    if x0 > x1 {
        core::mem::swap(&mut x0, &mut x1);
        core::mem::swap(&mut y0, &mut y1);
    }

    let dx = x1 - x0;
    let dy = y1 - y0;
    let gradient = if dx == 0 { 1.0 } else { dy as f32 / dx as f32 };

    let mut x = x0;
    let mut y = y0 as f32;

    while x <= x1 {
        let y_floor = y as i32;
        let frac = y - y_floor as f32;
        let a1 = ((1.0 - frac) * alpha as f32) as u8;
        let a2 = (frac * alpha as f32) as u8;

        if steep {
            set_pixel_blended(win, y_floor as u32, x as u32, color, a1);
            set_pixel_blended(win, (y_floor + 1) as u32, x as u32, color, a2);
        } else {
            set_pixel_blended(win, x as u32, y_floor as u32, color, a1);
            set_pixel_blended(win, x as u32, (y_floor + 1) as u32, color, a2);
        }

        y += gradient;
        x += 1;
    }
}
