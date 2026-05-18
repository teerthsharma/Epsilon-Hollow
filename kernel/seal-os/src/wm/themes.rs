// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Desktop themes — color schemes for window manager.

#[derive(Debug, Clone, Copy)]
pub struct Theme {
    pub name: &'static str,
    pub bg: u32,
    pub fg: u32,
    pub accent: u32,
    pub titlebar: u32,
    pub border: u32,
    pub taskbar: u32,
}

pub const DARK: Theme = Theme {
    name: "dark",
    bg: 0x00101018,
    fg: 0x00D0D0D8,
    accent: 0x004488CC,
    titlebar: 0x002040A0,
    border: 0x00505060,
    taskbar: 0x00181828,
};

pub const LIGHT: Theme = Theme {
    name: "light",
    bg: 0x00F0F0F4,
    fg: 0x00202030,
    accent: 0x003366AA,
    titlebar: 0x00DDDDE8,
    border: 0x00C0C0C8,
    taskbar: 0x00E0E0E8,
};

pub const SEAL: Theme = Theme {
    name: "seal",
    bg: 0x00102030,
    fg: 0x00A0D8E0,
    accent: 0x0044AACC,
    titlebar: 0x001A3040,
    border: 0x00305060,
    taskbar: 0x00152535,
};

pub const MATRIX: Theme = Theme {
    name: "matrix",
    bg: 0x00000800,
    fg: 0x0000FF00,
    accent: 0x0000CC00,
    titlebar: 0x00001800,
    border: 0x00003300,
    taskbar: 0x00001000,
};

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
