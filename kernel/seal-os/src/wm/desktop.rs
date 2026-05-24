// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Desktop surface: equation wallpaper + taskbar integration + app icons + menus.

use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

use crate::graphics::font;
use crate::graphics::framebuffer::Framebuffer;
use crate::wm::app_state::AppState;
use crate::wm::compositor::Compositor;
use crate::wm::event::InputEvent;
use crate::wm::power_menu::{PowerAction, PowerMenu};
use crate::wm::start_menu::StartMenu;
use crate::wm::taskbar;

pub struct DesktopIcon {
    pub name: &'static str,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub color: u32,
    pub app_id: u8,
}

pub struct DesktopState {
    pub icons: Vec<DesktopIcon>,
    pub start_menu: StartMenu,
    pub power_menu: PowerMenu,
}

impl DesktopState {
    pub const fn new() -> Self {
        Self {
            icons: Vec::new(),
            start_menu: StartMenu::new(),
            power_menu: PowerMenu::new(),
        }
    }
}

static DESKTOP_STATE: Mutex<DesktopState> = Mutex::new(DesktopState::new());

static NOTIFICATION: Mutex<Option<(String, u64)>> = Mutex::new(None);

/// Show a transient desktop notification (v1: top-right banner, 3 s).
pub fn show_notification(msg: &str) {
    let now = crate::drivers::interrupts::ticks();
    *NOTIFICATION.lock() = Some((String::from(msg), now));
}

pub fn render_notifications(fb: &Framebuffer) {
    let guard = NOTIFICATION.lock();
    let (msg, start) = match guard.as_ref() {
        Some(v) => (v.0.clone(), v.1),
        None => return,
    };
    drop(guard);
    let now = crate::drivers::interrupts::ticks();
    if now.saturating_sub(start) >= 3000 {
        return;
    }
    let w = msg.len() as u32 * font::CHAR_WIDTH + 16;
    let h = font::CHAR_HEIGHT + 8;
    let x = fb.width.saturating_sub(w + 10);
    let y = 10;
    fb.fill_rect(x, y, w, h, 0x001A1A2E);
    fb.fill_rect(x, y, w, 1, 0x00CBA6F7);
    fb.fill_rect(x, y + h - 1, w, 1, 0x00CBA6F7);
    draw_text(fb, x + 8, y + 4, &msg, 0x00CDD6F4);
}

fn init_icons(state: &mut DesktopState) {
    if !state.icons.is_empty() {
        return;
    }
    state.icons = vec![
        DesktopIcon { name: "Terminal", x: 20, y: 50, width: 48, height: 48, color: 0x0044CC44, app_id: 1 },
        DesktopIcon { name: "IDE", x: 120, y: 50, width: 48, height: 48, color: 0x004444CC, app_id: 2 },
        DesktopIcon { name: "Files", x: 220, y: 50, width: 48, height: 48, color: 0x00CCCC44, app_id: 3 },
        DesktopIcon { name: "Calc", x: 320, y: 50, width: 48, height: 48, color: 0x00CC8844, app_id: 4 },
        DesktopIcon { name: "Theorems", x: 420, y: 50, width: 48, height: 48, color: 0x008844CC, app_id: 5 },
        DesktopIcon { name: "Snake", x: 520, y: 50, width: 48, height: 48, color: 0x00CC4444, app_id: 6 },
        DesktopIcon { name: "Breakout", x: 620, y: 50, width: 48, height: 48, color: 0x0044CCCC, app_id: 7 },
        DesktopIcon { name: "Warp Racer", x: 720, y: 50, width: 48, height: 48, color: 0x00CC44CC, app_id: 8 },
    ];
}

pub fn render_desktop(fb: &Framebuffer) {
    let theme = crate::wm::themes::current_theme();
    fb.clear(theme.desktop_bg);

    render_equation_wallpaper(fb);
    render_equations_text(fb);
    taskbar::draw_taskbar(fb);

    let mut state = DESKTOP_STATE.lock();
    init_icons(&mut state);
    for icon in &state.icons {
        draw_icon(fb, icon);
    }
}

pub fn render_overlays(fb: &Framebuffer) {
    let state = DESKTOP_STATE.lock();
    state.start_menu.draw(fb);
    state.power_menu.draw(fb);
}

fn draw_icon(fb: &Framebuffer, icon: &DesktopIcon) {
    fb.fill_rect(icon.x, icon.y, icon.width, icon.height, icon.color);
    let text_y = icon.y + icon.height + 4;
    for (i, ch) in icon.name.bytes().enumerate() {
        font::draw_char(fb, icon.x + i as u32 * font::CHAR_WIDTH, text_y, ch, 0x00FFFFFF);
    }
}

pub fn handle_input(
    event: InputEvent,
    fb: &Framebuffer,
    compositor: &mut Compositor,
    app_state: &mut AppState,
) -> bool {
    match event {
        InputEvent::MouseButton { button: 0, pressed: true } => {
            let mx = compositor.mouse_state().x as u32;
            let my = compositor.mouse_state().y as u32;
            handle_click(mx, my, fb, compositor, app_state)
        }
        _ => false,
    }
}

fn handle_click(
    mx: u32,
    my: u32,
    fb: &Framebuffer,
    compositor: &mut Compositor,
    app_state: &mut AppState,
) -> bool {
    let mut state = DESKTOP_STATE.lock();
    init_icons(&mut state);
    let taskbar_y = fb.height.saturating_sub(taskbar::taskbar_height());

    // Menu item clicks take highest priority
    if state.start_menu.open {
        if let Some(app_id) = state.start_menu.handle_click(mx, my) {
            state.start_menu.close();
            drop(state);
            crate::wm::app_launcher::launch_app(app_id);
            app_state.focus_app(app_id, compositor);
            return true;
        }
        state.start_menu.close();
    }

    if state.power_menu.open {
        if let Some(action) = state.power_menu.handle_click(mx, my) {
            state.power_menu.close();
            drop(state);
            match action {
                PowerAction::Shutdown => {
                    crate::serial_println!("[Power] Shutdown requested");
                    if !crate::drivers::acpi::power_off() {
                        unsafe {
                            let mut port = x86_64::instructions::port::Port::<u8>::new(0x64);
                            port.write(0xFE);
                        }
                    }
                }
                PowerAction::Reboot => {
                    crate::serial_println!("[Power] Reboot requested");
                    unsafe {
                        let mut port = x86_64::instructions::port::Port::<u8>::new(0x64);
                        port.write(0xFE);
                    }
                }
                PowerAction::Logout => {
                    crate::serial_println!("[Power] Logout requested");
                    request_logout();
                }
            }
            return true;
        }
        state.power_menu.close();
    }

    // Start button
    if mx >= 4 && mx < 76 && my >= taskbar_y + 4 && my < taskbar_y + 24 {
        state.start_menu.toggle(taskbar_y);
        state.power_menu.close();
        return true;
    }

    // Power button
    let power_x = fb.width.saturating_sub(40);
    if mx >= power_x && mx < power_x + 24 && my >= taskbar_y + 6 && my < taskbar_y + 22 {
        state.power_menu.toggle(power_x, taskbar_y);
        state.start_menu.close();
        return true;
    }

    // Desktop icons
    if my < taskbar_y {
        if compositor.window_at(mx, my).is_none() {
            for icon in &state.icons {
                if mx >= icon.x && mx < icon.x + icon.width && my >= icon.y && my < icon.y + icon.height {
                    let app_id = icon.app_id;
                    drop(state);
                    crate::wm::app_launcher::launch_app(app_id);
                    app_state.focus_app(app_id, compositor);
                    return true;
                }
            }
        }
    }

    false
}

static LOGOUT_REQUESTED: spin::Mutex<bool> = spin::Mutex::new(false);

pub fn request_logout() {
    *LOGOUT_REQUESTED.lock() = true;
}

pub fn is_logout_requested() -> bool {
    *LOGOUT_REQUESTED.lock()
}

pub fn clear_logout_request() {
    *LOGOUT_REQUESTED.lock() = false;
}

// ---------------------------------------------------------------------------
// Existing wallpaper / text rendering
// ---------------------------------------------------------------------------

fn render_equation_wallpaper(fb: &Framebuffer) {
    let theme = crate::wm::themes::current_theme();
    // Decorative grid pattern (subtle)
    for y in (0..fb.height - 28).step_by(64) {
        for x in 0..fb.width {
            fb.put_pixel(x, y, theme.bg);
        }
    }
    for x in (0..fb.width).step_by(64) {
        for y in 0..fb.height - 28 {
            fb.put_pixel(x, y, theme.bg);
        }
    }

    // Central glow effect (radial gradient from center)
    let cx = fb.width / 2;
    let cy = (fb.height - 28) / 2;
    for y in (cy.saturating_sub(200))..((cy + 200).min(fb.height - 28)) {
        for x in (cx.saturating_sub(300))..((cx + 300).min(fb.width)) {
            let dx = (x as i32 - cx as i32) as f64;
            let dy = (y as i32 - cy as i32) as f64;
            let dist = libm::sqrt(dx * dx + dy * dy);
            if dist < 250.0 {
                let intensity = ((1.0 - dist / 250.0) * 12.0) as u32;
                let r = (intensity).min(20);
                let g = (intensity).min(15);
                let b = (intensity + 5).min(30);
                fb.put_pixel(x, y, (r << 16) | (g << 8) | b);
            }
        }
    }

    // Draw a simple circular "black hole" visualization at center
    for angle_i in 0..360 {
        let angle = (angle_i as f64) * core::f64::consts::PI / 180.0;
        for r in 40..60 {
            let x = cx as f64 + r as f64 * libm::cos(angle);
            let y = cy as f64 + r as f64 * libm::sin(angle);
            if x >= 0.0 && y >= 0.0 {
                let px = x as u32;
                let py = y as u32;
                if px < fb.width && py < fb.height - 28 {
                    let fade = ((r - 40) as f64 / 20.0 * 80.0) as u32;
                    let color = (fade << 16) | ((fade / 2) << 8) | (fade + 20).min(255);
                    fb.put_pixel(px, py, color);
                }
            }
        }
    }

    // Accretion disk rings
    for ring in 0..3 {
        let base_r = 70 + ring * 25;
        for angle_i in 0..720 {
            let angle = (angle_i as f64) * core::f64::consts::PI / 360.0;
            let r = base_r as f64 + 5.0 * libm::sin(angle * 3.0);
            let x = cx as f64 + r * libm::cos(angle);
            let y = cy as f64 + r * libm::sin(angle) * 0.4; // elliptical
            if x >= 0.0 && y >= 0.0 {
                let px = x as u32;
                let py = y as u32;
                if px < fb.width && py < fb.height - 28 {
                    let intensity = 180 - ring * 50;
                    fb.put_pixel(px, py, (intensity << 16) | ((intensity / 2) << 8) | 0x20);
                }
            }
        }
    }
}

fn render_equations_text(fb: &Framebuffer) {
    let theme = crate::wm::themes::current_theme();
    const TEXT_X: u32 = 20;
    const TEXT_Y: u32 = 20;
    const LINE_HEIGHT: u32 = 18;
    const PAD: u32 = 8;
    const BG_ALPHA: u8 = 180;

    let lines: &[(u32, &str)] = &[
        (theme.border, "Schwarzschild metric:"),
        (theme.accent, "ds^2 = -(1-2GM/rc^2)dt^2 + (1-2GM/rc^2)^-1 dr^2 + r^2 dO^2"),
        (theme.fg, ""),
        (theme.border, "Faraday tensor F^uv:"),
        (theme.fg, "[ 0   -Ex  -Ey  -Ez ]"),
        (theme.fg, "[ Ex   0   -Bz   By ]"),
        (theme.fg, "[ Ey   Bz   0   -Bx ]"),
        (theme.fg, "[ Ez  -By   Bx   0  ]"),
    ];

    // Compute bounding box from longest line
    let mut max_w = 0u32;
    for (_, text) in lines {
        let w = text.len() as u32 * font::CHAR_WIDTH;
        if w > max_w {
            max_w = w;
        }
    }
    let total_h = lines.len() as u32 * LINE_HEIGHT;

    let bg_x = TEXT_X.saturating_sub(PAD);
    let bg_y = TEXT_Y.saturating_sub(PAD);
    let bg_w = max_w + PAD * 2;
    let bg_h = total_h + PAD * 2;

    // Draw semi-transparent dark background
    for row in bg_y..(bg_y + bg_h).min(fb.height) {
        for col in bg_x..(bg_x + bg_w).min(fb.width) {
            let bg_pixel = fb.get_pixel(col, row);
            let blended = alpha_blend(bg_pixel, theme.bg, BG_ALPHA);
            fb.put_pixel(col, row, blended);
        }
    }

    // Draw each line of text
    for (i, &(color, text)) in lines.iter().enumerate() {
        let y = TEXT_Y + i as u32 * LINE_HEIGHT;
        draw_text(fb, TEXT_X, y, text, color);
    }
}

fn draw_text(fb: &Framebuffer, x: u32, y: u32, text: &str, color: u32) {
    for (i, ch) in text.bytes().enumerate() {
        font::draw_char(fb, x + (i as u32) * font::CHAR_WIDTH, y, ch, color);
    }
}

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
