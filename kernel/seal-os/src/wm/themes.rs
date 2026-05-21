// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Desktop themes — color schemes for window manager.

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy)]
pub struct Theme {
    pub name: &'static str,
    pub bg: u32,
    pub fg: u32,
    pub accent: u32,
    pub titlebar: u32,
    pub titlebar_unfocused: u32,
    pub border: u32,
    pub taskbar: u32,
    pub close_btn: u32,
    pub desktop_bg: u32,
    pub splash_color: u32,
    pub progress_bg: u32,
    pub progress_fg: u32,
    pub login_bg: u32,
    pub login_accent: u32,
    pub console_bg: u32,
    pub console_fg: u32,
    pub cursor: u32,
}

pub const DARK: Theme = Theme {
    name: "dark",
    bg: 0x00181820,
    fg: 0x00E0E0E0,
    accent: 0x004488CC,
    titlebar: 0x002040A0,
    titlebar_unfocused: 0x00404050,
    border: 0x00505060,
    taskbar: 0x00101018,
    close_btn: 0x00CC4444,
    desktop_bg: 0x000A0A0F,
    splash_color: 0x0090B0D0,
    progress_bg: 0x00303040,
    progress_fg: 0x004488CC,
    login_bg: 0x00080810,
    login_accent: 0x004488CC,
    console_bg: 0x000A0A0F,
    console_fg: 0x00E0E0E0,
    cursor: 0x00FFFFFF,
};

pub const LIGHT: Theme = Theme {
    name: "light",
    bg: 0x00F0F0F4,
    fg: 0x00202030,
    accent: 0x003366AA,
    titlebar: 0x00DDDDE8,
    titlebar_unfocused: 0x00CCCCD8,
    border: 0x00C0C0C8,
    taskbar: 0x00E0E0E8,
    close_btn: 0x00CC4444,
    desktop_bg: 0x00F0F0F4,
    splash_color: 0x006080A0,
    progress_bg: 0x00D0D0D8,
    progress_fg: 0x003366AA,
    login_bg: 0x00F0F0F4,
    login_accent: 0x003366AA,
    console_bg: 0x00F0F0F4,
    console_fg: 0x00202030,
    cursor: 0x00FFFFFF,
};

pub const SEAL: Theme = Theme {
    name: "seal",
    bg: 0x00102030,
    fg: 0x00A0D8E0,
    accent: 0x0044AACC,
    titlebar: 0x001A3040,
    titlebar_unfocused: 0x002A4050,
    border: 0x00305060,
    taskbar: 0x00152535,
    close_btn: 0x00CC4444,
    desktop_bg: 0x00102030,
    splash_color: 0x006090A0,
    progress_bg: 0x00203040,
    progress_fg: 0x0044AACC,
    login_bg: 0x00102030,
    login_accent: 0x0044AACC,
    console_bg: 0x00102030,
    console_fg: 0x00A0D8E0,
    cursor: 0x00FFFFFF,
};

pub const MATRIX: Theme = Theme {
    name: "matrix",
    bg: 0x00000800,
    fg: 0x0000FF00,
    accent: 0x0000CC00,
    titlebar: 0x00001800,
    titlebar_unfocused: 0x00002200,
    border: 0x00003300,
    taskbar: 0x00001000,
    close_btn: 0x00CC4444,
    desktop_bg: 0x00000800,
    splash_color: 0x0000CC00,
    progress_bg: 0x00002200,
    progress_fg: 0x0000CC00,
    login_bg: 0x00000800,
    login_accent: 0x0000CC00,
    console_bg: 0x00000800,
    console_fg: 0x0000FF00,
    cursor: 0x00FFFFFF,
};

static ACTIVE_THEME_IDX: AtomicUsize = AtomicUsize::new(0);
static THEME_CHANGED: AtomicBool = AtomicBool::new(false);
const THEMES: &[Theme] = &[DARK, LIGHT, SEAL, MATRIX];

pub fn set_theme(name: &str) -> bool {
    if let Some(idx) = THEMES.iter().position(|t| t.name == name) {
        ACTIVE_THEME_IDX.store(idx, Ordering::Relaxed);
        THEME_CHANGED.store(true, Ordering::Relaxed);
        true
    } else {
        false
    }
}

pub fn current_theme() -> &'static Theme {
    &THEMES[ACTIVE_THEME_IDX.load(Ordering::Relaxed)]
}

pub fn theme_changed() -> bool {
    THEME_CHANGED.swap(false, Ordering::Relaxed)
}

pub fn get_theme(name: &str) -> Option<&'static Theme> {
    match name {
        "dark" => Some(&DARK),
        "light" => Some(&LIGHT),
        "seal" => Some(&SEAL),
        "matrix" => Some(&MATRIX),
        _ => None,
    }
}

pub fn available() -> &'static [&'static str] {
    &["dark", "light", "seal", "matrix"]
}
