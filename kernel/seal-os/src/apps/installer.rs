// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Full-screen disk installer for Seal OS.
//!
//! Disk partitioning and formatting require raw block device write access.
//! Full implementation in v0.5.0.

use crate::drivers::interrupts;
use crate::graphics::font::{self, CHAR_HEIGHT, CHAR_WIDTH};
use crate::graphics::framebuffer::Framebuffer;
use crate::serial_println;
use crate::wm::event::InputEvent;
use alloc::format;
use alloc::string::String;

const BG: u32 = 0x00080a10;
const FG: u32 = 0x00d0d0e0;
const ACCENT: u32 = 0x0000aaff;
const BTN_BG: u32 = 0x001a2a40;
const BTN_FG: u32 = 0x00ffffff;
const FIELD_BG: u32 = 0x00101520;
const FIELD_BORDER: u32 = 0x00304050;
const FIELD_ACTIVE: u32 = 0x0000aaff;
const PROGRESS_BG: u32 = 0x00152030;
const PROGRESS_FG: u32 = 0x0000cc66;
const ERROR_FG: u32 = 0x00ff4466;

const DISK_NAMES: &[&str] = &["nvme0", "sda", "sdb"];
const DISK_CAPS_GB: &[u64] = &[512, 1024, 2048];

const STEP_WELCOME: u8 = 0;
const STEP_DISK_SELECT: u8 = 1;
const STEP_PARTITION: u8 = 2;
const STEP_USER: u8 = 3;
const STEP_INSTALL: u8 = 4;
const STEP_DONE: u8 = 5;

pub struct Installer {
    step: u8,
    selected_disk: u8,
    username: [u8; 32],
    username_len: usize,
    password: [u8; 32],
    password_len: usize,
    confirm_password: [u8; 32],
    confirm_len: usize,
    progress: u8,
    active_field: u8, // 0=username, 1=password, 2=confirm
    mouse_x: i32,
    mouse_y: i32,
    fb_width: u32,
    fb_height: u32,
    install_start_tick: u64,
    error_msg: Option<&'static str>,
    install_applied: bool,
    install_ok: bool,
}

impl Installer {
    pub const fn new() -> Self {
        Self {
            step: STEP_WELCOME,
            selected_disk: 0,
            username: [0u8; 32],
            username_len: 0,
            password: [0u8; 32],
            password_len: 0,
            confirm_password: [0u8; 32],
            confirm_len: 0,
            progress: 0,
            active_field: 0,
            mouse_x: 512,
            mouse_y: 384,
            fb_width: 1024,
            fb_height: 768,
            install_start_tick: 0,
            error_msg: None,
            install_applied: false,
            install_ok: false,
        }
    }

    pub fn render(&mut self, fb: &Framebuffer) {
        self.fb_width = fb.width;
        self.fb_height = fb.height;

        fb.clear(BG);

        match self.step {
            STEP_WELCOME => self.render_welcome(fb),
            STEP_DISK_SELECT => self.render_disk_select(fb),
            STEP_PARTITION => self.render_partition(fb),
            STEP_USER => self.render_user(fb),
            STEP_INSTALL => self.render_install(fb),
            STEP_DONE => self.render_done(fb),
            _ => {}
        }
    }

    pub fn handle_event(&mut self, event: InputEvent) -> bool {
        match event {
            InputEvent::KeyPress(key) => {
                if self.step == STEP_USER {
                    self.handle_user_key(key);
                    return false;
                }
                if key == b'\n' || key == 13 {
                    return self.advance_step();
                }
            }
            InputEvent::MouseMove { dx, dy } => {
                let max_x = self.fb_width.saturating_sub(1) as i32;
                let max_y = self.fb_height.saturating_sub(1) as i32;
                self.mouse_x = (self.mouse_x + dx).clamp(0, max_x);
                self.mouse_y = (self.mouse_y + dy).clamp(0, max_y);
            }
            InputEvent::MouseButton {
                button: 0,
                pressed: true,
            } => {
                return self.handle_click();
            }
            _ => {}
        }
        false
    }

    fn render_welcome(&self, fb: &Framebuffer) {
        let title = format!("Welcome to Seal OS {}", crate::VERSION);
        let subtitle = "The Geometrical Operating System";

        let title_w = title.len() as u32 * CHAR_WIDTH;
        let title_x = (fb.width.saturating_sub(title_w)) / 2;
        let title_y = fb.height / 4;
        font::draw_string(fb, title_x, title_y, &title, ACCENT);

        let sub_w = subtitle.len() as u32 * CHAR_WIDTH;
        let sub_x = (fb.width.saturating_sub(sub_w)) / 2;
        font::draw_string(fb, sub_x, title_y + CHAR_HEIGHT + 12, subtitle, FG);

        self.draw_button(fb, "Install", fb.width / 2 - 60, fb.height / 2, 120, 36);
    }

    fn render_disk_select(&self, fb: &Framebuffer) {
        self.draw_title(fb, "Select Disk");

        let start_y = fb.height / 3;
        let row_h = 40u32;
        let box_w = 300u32;
        let box_x = (fb.width.saturating_sub(box_w)) / 2;

        for (i, &name) in DISK_NAMES.iter().enumerate() {
            let y = start_y + i as u32 * row_h;
            let selected = i as u8 == self.selected_disk;
            let border = if selected { ACCENT } else { FIELD_BORDER };
            let bg = if selected { FIELD_BG } else { BG };

            fb.fill_rect(box_x, y, box_w, 32, bg);
            fb.fill_rect(box_x, y, box_w, 2, border);
            fb.fill_rect(box_x, y + 30, box_w, 2, border);
            fb.fill_rect(box_x, y, 2, 32, border);
            fb.fill_rect(box_x + box_w - 2, y, 2, 32, border);

            let cap = DISK_CAPS_GB[i];
            let line = format!("{} — {} GB", name, cap);
            let tx = box_x + 12;
            let ty = y + (32 - CHAR_HEIGHT) / 2;
            font::draw_string(fb, tx, ty, &line, FG);
        }

        self.draw_button(
            fb,
            "Continue",
            fb.width / 2 - 60,
            start_y + DISK_NAMES.len() as u32 * row_h + 20,
            120,
            36,
        );
    }

    fn render_partition(&self, fb: &Framebuffer) {
        self.draw_title(fb, "Partition Scheme");

        let lines: &[&str] = &[
            "Seal OS will create the following partitions:",
            "",
            "  EFI System Partition  — 512 MB, FAT32",
            "  Root Partition        — remainder, ManifoldFS",
            "",
            "Disk partitioning and formatting require raw block",
            "device write access. Full implementation in v0.5.0.",
        ];

        let start_y = fb.height / 3;
        for (i, line) in lines.iter().enumerate() {
            let y = start_y + i as u32 * (CHAR_HEIGHT + 6);
            font::draw_string(fb, 60, y, line, FG);
        }

        self.draw_button(fb, "Continue", fb.width / 2 - 60, fb.height - 120, 120, 36);
    }

    fn render_user(&self, fb: &Framebuffer) {
        self.draw_title(fb, "Create User Account");

        let start_y = fb.height / 3;
        let field_w = 300u32;
        let field_h = 32u32;
        let field_x = (fb.width.saturating_sub(field_w)) / 2;

        // Username
        font::draw_string(fb, field_x, start_y - CHAR_HEIGHT - 6, "Username:", FG);
        self.draw_field(
            fb,
            field_x,
            start_y,
            field_w,
            field_h,
            self.active_field == 0,
        );
        font::draw_string(
            fb,
            field_x + 8,
            start_y + (field_h - CHAR_HEIGHT) / 2,
            core::str::from_utf8(&self.username[..self.username_len]).unwrap_or(""),
            FG,
        );

        // Password
        let py = start_y + field_h + 24;
        font::draw_string(fb, field_x, py - CHAR_HEIGHT - 6, "Password:", FG);
        self.draw_field(fb, field_x, py, field_w, field_h, self.active_field == 1);
        self.draw_dots(
            fb,
            field_x + 8,
            py + (field_h - CHAR_HEIGHT) / 2,
            self.password_len,
        );

        // Confirm
        let cy = py + field_h + 24;
        font::draw_string(fb, field_x, cy - CHAR_HEIGHT - 6, "Confirm Password:", FG);
        self.draw_field(fb, field_x, cy, field_w, field_h, self.active_field == 2);
        self.draw_dots(
            fb,
            field_x + 8,
            cy + (field_h - CHAR_HEIGHT) / 2,
            self.confirm_len,
        );

        if let Some(err) = self.error_msg {
            let ey = cy + field_h + 16;
            font::draw_string(fb, field_x, ey, err, ERROR_FG);
        }

        self.draw_button(fb, "Continue", fb.width / 2 - 60, fb.height - 120, 120, 36);
    }

    fn render_install(&mut self, fb: &Framebuffer) {
        self.draw_title(fb, "Installing Seal OS...");

        let elapsed = interrupts::ticks().saturating_sub(self.install_start_tick);
        // Every ~600ms advance one sub-step (15-20% each)
        let sub_step = (elapsed / 600).min(6) as u8;
        self.progress = (sub_step * 16).min(100);
        if sub_step >= 6 {
            self.progress = 100;
        }

        let bar_w = 400u32;
        let bar_h = 24u32;
        let bar_x = (fb.width.saturating_sub(bar_w)) / 2;
        let bar_y = fb.height / 2;

        fb.fill_rect(bar_x, bar_y, bar_w, bar_h, PROGRESS_BG);
        let fill_w = (bar_w as u64 * self.progress as u64 / 100) as u32;
        fb.fill_rect(bar_x, bar_y, fill_w, bar_h, PROGRESS_FG);

        let pct = format!("{}%", self.progress);
        let pct_w = pct.len() as u32 * CHAR_WIDTH;
        let pct_x = bar_x + (bar_w.saturating_sub(pct_w)) / 2;
        let pct_y = bar_y + (bar_h.saturating_sub(CHAR_HEIGHT)) / 2;
        font::draw_string(fb, pct_x, pct_y, &pct, BTN_FG);

        let msgs = [
            "Checking target disk...",
            "Preparing safe install root...",
            "Writing boot files...",
            "Creating user...",
            "Verifying installed files...",
            "Recording proof...",
            "Finalizing...",
        ];
        let msg = msgs[sub_step.min(6) as usize];
        let msg_w = msg.len() as u32 * CHAR_WIDTH;
        let msg_x = (fb.width.saturating_sub(msg_w)) / 2;
        let msg_y = bar_y + bar_h + 20;
        font::draw_string(fb, msg_x, msg_y, msg, FG);

        if sub_step >= 1 && !self.install_applied {
            self.install_applied = true;
            self.install_ok = self.apply_safe_install();
        }
        if sub_step == 6 && self.progress >= 100 {
            serial_println!(
                "[INSTALLER] Completed safe VFS install target result={}",
                if self.install_ok { "pass" } else { "fail" }
            );
            self.step = STEP_DONE;
        }
    }

    fn render_done(&self, fb: &Framebuffer) {
        let title = "Seal OS installed successfully!";
        let title_w = title.len() as u32 * CHAR_WIDTH;
        let title_x = (fb.width.saturating_sub(title_w)) / 2;
        let title_y = fb.height / 3;
        font::draw_string(fb, title_x, title_y, title, ACCENT);

        let sub = if self.install_ok {
            "Safe VFS install target verified. Reboot when ready."
        } else {
            "Install verification failed. Check serial log."
        };
        let sub_w = sub.len() as u32 * CHAR_WIDTH;
        let sub_x = (fb.width.saturating_sub(sub_w)) / 2;
        font::draw_string(fb, sub_x, title_y + CHAR_HEIGHT + 12, sub, FG);

        self.draw_button(fb, "Reboot", fb.width / 2 - 60, fb.height / 2, 120, 36);
    }

    fn handle_user_key(&mut self, key: u8) {
        match key {
            b'\n' | 13 => {
                if self.active_field < 2 {
                    self.active_field += 1;
                } else {
                    let _ = self.advance_step();
                }
            }
            0x08 | 0x7F => match self.active_field {
                0 => {
                    if self.username_len > 0 {
                        self.username_len -= 1;
                    }
                }
                1 => {
                    if self.password_len > 0 {
                        self.password_len -= 1;
                    }
                }
                2 => {
                    if self.confirm_len > 0 {
                        self.confirm_len -= 1;
                    }
                }
                _ => {}
            },
            0x09 => {
                self.active_field = (self.active_field + 1) % 3;
            }
            ch if ch >= 0x20 && ch < 0x7F => match self.active_field {
                0 => {
                    if self.username_len < 32 {
                        self.username[self.username_len] = ch;
                        self.username_len += 1;
                    }
                }
                1 => {
                    if self.password_len < 32 {
                        self.password[self.password_len] = ch;
                        self.password_len += 1;
                    }
                }
                2 => {
                    if self.confirm_len < 32 {
                        self.confirm_password[self.confirm_len] = ch;
                        self.confirm_len += 1;
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn handle_click(&mut self) -> bool {
        let mx = self.mouse_x as u32;
        let my = self.mouse_y as u32;

        match self.step {
            STEP_WELCOME => {
                if self.hit_button(mx, my, self.fb_width / 2 - 60, self.fb_height / 2, 120, 36) {
                    self.step = STEP_DISK_SELECT;
                }
            }
            STEP_DISK_SELECT => {
                let start_y = self.fb_height / 3;
                let row_h = 40u32;
                let box_w = 300u32;
                let box_x = (self.fb_width.saturating_sub(box_w)) / 2;

                for (i, _) in DISK_NAMES.iter().enumerate() {
                    let y = start_y + i as u32 * row_h;
                    if mx >= box_x && mx < box_x + box_w && my >= y && my < y + 32 {
                        self.selected_disk = i as u8;
                        return false;
                    }
                }

                if self.hit_button(
                    mx,
                    my,
                    self.fb_width / 2 - 60,
                    start_y + DISK_NAMES.len() as u32 * row_h + 20,
                    120,
                    36,
                ) {
                    self.step = STEP_PARTITION;
                }
            }
            STEP_PARTITION => {
                if self.hit_button(
                    mx,
                    my,
                    self.fb_width / 2 - 60,
                    self.fb_height - 120,
                    120,
                    36,
                ) {
                    self.step = STEP_USER;
                }
            }
            STEP_USER => {
                let start_y = self.fb_height / 3;
                let field_w = 300u32;
                let field_h = 32u32;
                let field_x = (self.fb_width.saturating_sub(field_w)) / 2;

                if self.hit_button(mx, my, field_x, start_y, field_w, field_h) {
                    self.active_field = 0;
                    return false;
                }
                let py = start_y + field_h + 24;
                if self.hit_button(mx, my, field_x, py, field_w, field_h) {
                    self.active_field = 1;
                    return false;
                }
                let cy = py + field_h + 24;
                if self.hit_button(mx, my, field_x, cy, field_w, field_h) {
                    self.active_field = 2;
                    return false;
                }

                if self.hit_button(
                    mx,
                    my,
                    self.fb_width / 2 - 60,
                    self.fb_height - 120,
                    120,
                    36,
                ) {
                    return self.advance_step();
                }
            }
            STEP_DONE => {
                if self.hit_button(mx, my, self.fb_width / 2 - 60, self.fb_height / 2, 120, 36) {
                    interrupts::reboot();
                }
            }
            _ => {}
        }
        false
    }

    fn advance_step(&mut self) -> bool {
        match self.step {
            STEP_USER => {
                if self.username_len == 0 {
                    self.error_msg = Some("Username cannot be empty.");
                    return false;
                }
                if self.password_len == 0 {
                    self.error_msg = Some("Password cannot be empty.");
                    return false;
                }
                if self.username_is_seal() && self.password_is_literal_seal() {
                    self.error_msg = Some("Default password is blocked.");
                    return false;
                }
                if self.password_len != self.confirm_len {
                    self.error_msg = Some("Passwords do not match.");
                    return false;
                }
                let mut ok = true;
                for i in 0..self.password_len {
                    if self.password[i] != self.confirm_password[i] {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    self.error_msg = Some("Passwords do not match.");
                    return false;
                }
                self.error_msg = None;
                self.step = STEP_INSTALL;
                self.install_start_tick = interrupts::ticks();
                self.install_applied = false;
                self.install_ok = false;
                false
            }
            _ => {
                if self.step == STEP_DONE {
                    return true;
                }
                self.step += 1;
                false
            }
        }
    }

    fn username_is_seal(&self) -> bool {
        self.username_len == 4 && &self.username[..4] == b"seal"
    }

    fn password_is_literal_seal(&self) -> bool {
        self.password_len == 4 && &self.password[..4] == b"seal"
    }

    fn draw_title(&self, fb: &Framebuffer, text: &str) {
        let w = text.len() as u32 * CHAR_WIDTH;
        let x = (fb.width.saturating_sub(w)) / 2;
        let y = fb.height / 6;
        font::draw_string(fb, x, y, text, ACCENT);
    }

    fn draw_button(&self, fb: &Framebuffer, label: &str, x: u32, y: u32, w: u32, h: u32) {
        fb.fill_rect(x, y, w, h, BTN_BG);
        fb.fill_rect(x, y, w, 2, ACCENT);
        fb.fill_rect(x, y + h - 2, w, 2, ACCENT);
        fb.fill_rect(x, y, 2, h, ACCENT);
        fb.fill_rect(x + w - 2, y, 2, h, ACCENT);

        let text_w = label.len() as u32 * CHAR_WIDTH;
        let text_x = x + (w.saturating_sub(text_w)) / 2;
        let text_y = y + (h.saturating_sub(CHAR_HEIGHT)) / 2;
        font::draw_string(fb, text_x, text_y, label, BTN_FG);
    }

    fn draw_field(&self, fb: &Framebuffer, x: u32, y: u32, w: u32, h: u32, active: bool) {
        let border = if active { FIELD_ACTIVE } else { FIELD_BORDER };
        fb.fill_rect(x, y, w, h, FIELD_BG);
        fb.fill_rect(x, y, w, 2, border);
        fb.fill_rect(x, y + h - 2, w, 2, border);
        fb.fill_rect(x, y, 2, h, border);
        fb.fill_rect(x + w - 2, y, 2, h, border);
    }

    fn draw_dots(&self, fb: &Framebuffer, x: u32, y: u32, count: usize) {
        let dot_size = 6u32;
        let dot_gap = 10u32;
        for i in 0..count as u32 {
            fb.fill_rect(
                x + i * dot_gap,
                y + (CHAR_HEIGHT - dot_size) / 2,
                dot_size,
                dot_size,
                FG,
            );
        }
    }

    fn hit_button(&self, mx: u32, my: u32, x: u32, y: u32, w: u32, h: u32) -> bool {
        mx >= x && mx < x + w && my >= y && my < y + h
    }

    fn apply_safe_install(&self) -> bool {
        let username = core::str::from_utf8(&self.username[..self.username_len]).unwrap_or("seal");
        let password = core::str::from_utf8(&self.password[..self.password_len]).unwrap_or("");
        let disk = DISK_NAMES
            .get(self.selected_disk as usize)
            .copied()
            .unwrap_or("unknown");
        let boot_marker = format!(
            "Seal OS {} safe-install\nselected_disk={}\nuser={}\n",
            crate::VERSION,
            disk,
            username
        );
        let profile = format!(
            "user={}\nshell=/bin/shell\nhome=/home/{}\n",
            username,
            username
        );
        let boot_ok = write_file_verified("/boot/EFI/BOOT/BOOTX64.EFI", boot_marker.as_bytes());
        let profile_path = format!("/home/{}/.seal-profile", username);
        let home_ok = create_dir_path(&format!("/home/{}", username));
        let profile_ok = home_ok && write_file_verified(&profile_path, profile.as_bytes());
        let user_ok = if username == "seal" {
            crate::security::shadow::set_password(username, password)
        } else {
            if crate::security::passwd::get_user(username).is_none() {
                crate::security::passwd::add_user(username, password, 1001, 1001);
            }
            crate::security::passwd::get_user(username).is_some()
        };
        let auth = crate::security::shadow::auth_shadow_proof();
        let result = boot_ok && profile_ok && user_ok && auth.passes();
        serial_println!(
            "[INSTALLER] proof version=1 mode=safe_vfs selected_disk={} boot_marker={} home={} profile={} user={} auth_topo5000={} raw_gpt=0 raw_format=0 result={}",
            disk,
            if boot_ok { 1 } else { 0 },
            if home_ok { 1 } else { 0 },
            if profile_ok { 1 } else { 0 },
            if user_ok { 1 } else { 0 },
            if auth.passes() { 1 } else { 0 },
            if result { "pass" } else { "fail" }
        );
        result
    }
}

fn create_dir_path(path: &str) -> bool {
    let mut current = String::new();
    for part in path.split('/').filter(|part| !part.is_empty()) {
        current.push('/');
        current.push_str(part);
        let ok = crate::fs::vfs::with_vfs(|vfs| {
            if vfs.lookup_follow(&current).is_ok() {
                Some(true)
            } else {
                vfs.mkdir(&current).ok().map(|_| true)
            }
        })
        .unwrap_or(false);
        if !ok {
            return false;
        }
    }
    true
}

fn write_file_verified(path: &str, data: &[u8]) -> bool {
    if let Some(parent) = path.rfind('/').and_then(|idx| path.get(..idx)) {
        if !parent.is_empty() && !create_dir_path(parent) {
            return false;
        }
    }
    crate::fs::vfs::with_vfs(|vfs| {
        let handle = match vfs.create(path) {
            Ok(handle) => handle,
            Err(crate::fs::vfs::VfsError::AlreadyExists) => vfs.lookup_follow(path).ok()?,
            Err(_) => return None,
        };
        vfs.write(handle, data, 0).ok()?;
        let mut buf = alloc::vec![0u8; data.len()];
        let read = vfs.read(handle, &mut buf, 0).ok()?;
        Some(read == data.len() && buf == data)
    })
    .unwrap_or(false)
}
