// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Aether-Lang App Host — runs `.aether` scripts as native windowed applications.
//!
//! The host loads an Aether-Lang script and forwards input events to the runtime.
//! Each frame, the host calls the script's `render()` function, which can use
//! `graphics`, `window`, `ui`, and `input` modules.

use alloc::format;
use alloc::string::String;

const DEFAULT_SCRIPT: &str = r#"
import graphics~
import window~
import ui~
import input~

fn render() {
    graphics.clear(0x001E1E2E)~
    graphics.gradient_v(0, 0, 640, 480, 0x001E1E2E, 0x000A0A18)~

    ui.panel(10, 10, 200, 120, "Controls", 0x002A2A48)~
    ui.label(20, 40, "Aether-Lang App Host", 0x00FFFFFF)~

    if ui.button(20, 70, 80, 28, "Click", 0x002A2A48, 0x00DDDDEE) {
        graphics.fill_rect(250, 70, 40, 40, 0x00FF4444)~
    }

    ui.slider(20, 110, 160, 0, 100, 50, 0x00CBA6F7)~

    let mx = input.mouse_x()~
    let my = input.mouse_y()~
    graphics.draw_circle(mx, my, 4, 0x00FFFFFF)~
}
"#;

pub struct AetherAppHost {
    runtime: crate::lang::AetherRuntime,
    mouse_x: u32,
    mouse_y: u32,
    mouse_pressed: bool,
    mouse_clicked: bool,
    last_key: String,
}

impl AetherAppHost {
    fn from_runtime(runtime: crate::lang::AetherRuntime) -> Self {
        Self {
            runtime,
            mouse_x: 0,
            mouse_y: 0,
            mouse_pressed: false,
            mouse_clicked: false,
            last_key: String::new(),
        }
    }

    pub fn try_new() -> Result<Self, String> {
        let mut runtime = crate::lang::AetherRuntime::new();
        runtime.execute_source(DEFAULT_SCRIPT)?;
        Ok(Self::from_runtime(runtime))
    }

    pub fn new() -> Self {
        match Self::try_new() {
            Ok(host) => host,
            Err(e) => {
                let runtime = crate::lang::AetherRuntime::new();
                crate::serial_print!("[AetherAppHost] script init error: {}\n", e);
                Self::from_runtime(runtime)
            }
        }
    }

    pub fn boot_probe() -> Result<String, String> {
        let mut host = Self::try_new()?;
        let result = host.load_script(
            r#"
let aether_boot_probe = "seal-topology-ok"~
let result = aether_boot_probe~
"#,
        )?;
        if result == "Str(\"seal-topology-ok\")" {
            core::mem::forget(host);
            Ok(result)
        } else {
            Err(format!("unexpected boot probe result: {}", result))
        }
    }

    pub fn load_script(&mut self, source: &str) -> Result<String, String> {
        self.runtime.execute_source(source)
    }

    pub fn key_press(&mut self, ch: u8) {
        self.last_key.clear();
        self.last_key.push(ch as char);
        self.runtime.set_key(&self.last_key);
    }

    pub fn mouse_click(&mut self, x: u32, y: u32, pressed: bool) {
        self.mouse_x = x;
        self.mouse_y = y;
        self.mouse_pressed = pressed;
        if pressed {
            self.mouse_clicked = true;
        }
        self.runtime
            .set_mouse_state(x as i32, y as i32, pressed, self.mouse_clicked);
    }

    pub fn mouse_move(&mut self, x: u32, y: u32) {
        self.mouse_x = x;
        self.mouse_y = y;
        self.runtime
            .set_mouse_state(x as i32, y as i32, self.mouse_pressed, self.mouse_clicked);
    }

    pub fn render_window(&mut self, window_id: u32) {
        self.runtime.set_current_window(window_id);
        let _ = self.runtime.call_render();
        self.mouse_clicked = false;
        self.runtime.set_mouse_state(
            self.mouse_x as i32,
            self.mouse_y as i32,
            self.mouse_pressed,
            false,
        );
    }
}

pub fn main() {
    loop {
        crate::process::scheduler::yield_current();
    }
}
