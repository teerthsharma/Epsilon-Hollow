// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! USB HID class driver — keyboards and mice via interrupt transfers.

use crate::drivers::interrupts::push_event;
use crate::wm::event::InputEvent;

pub const HID_CLASS: u8 = 3;
pub const HID_BOOT_KEYBOARD: u8 = 1;
pub const HID_BOOT_MOUSE: u8 = 2;

#[derive(Debug, Clone, Copy)]
pub enum HidDeviceType {
    Keyboard,
    Mouse,
    Gamepad,
    Unknown,
}

pub struct HidDevice {
    pub device_type: HidDeviceType,
    pub slot_id: u8,
    pub ep_addr: u8,
    pub max_packet: u16,
    pub report_buf: [u8; 8],
    pub last_buttons: u8,
}

impl HidDevice {
    pub fn new(protocol: u8, slot_id: u8, ep_addr: u8, max_packet: u16) -> Self {
        let device_type = match protocol {
            HID_BOOT_KEYBOARD => HidDeviceType::Keyboard,
            HID_BOOT_MOUSE => HidDeviceType::Mouse,
            _ => HidDeviceType::Unknown,
        };
        Self {
            device_type,
            slot_id,
            ep_addr,
            max_packet,
            report_buf: [0u8; 8],
            last_buttons: 0,
        }
    }

    pub fn process_report(&mut self, report: &[u8]) {
        match self.device_type {
            HidDeviceType::Keyboard => self.process_keyboard(report),
            HidDeviceType::Mouse => self.process_mouse(report),
            _ => {}
        }
    }

    fn process_keyboard(&mut self, report: &[u8]) {
        if report.len() < 8 {
            return;
        }
        let modifiers = report[0];
        let shift = (modifiers & 0x22) != 0; // LShift or RShift
        for i in 2..8 {
            let keycode = report[i];
            if keycode == 0 {
                continue;
            }
            if let Some(ch) = hid_keycode_to_char(keycode, shift) {
                push_event(InputEvent::KeyPress(ch));
            }
        }
    }

    fn process_mouse(&mut self, report: &[u8]) {
        if report.len() < 3 {
            return;
        }
        let buttons = report[0];
        let dx = report[1] as i8 as i32;
        let dy = report[2] as i8 as i32;

        if dx != 0 || dy != 0 {
            push_event(InputEvent::MouseMove { dx, dy });
        }

        if buttons != self.last_buttons {
            for btn in 0..2 {
                let mask = 1 << btn;
                if (buttons & mask) != (self.last_buttons & mask) {
                    push_event(InputEvent::MouseButton {
                        button: btn,
                        pressed: (buttons & mask) != 0,
                    });
                }
            }
            self.last_buttons = buttons;
        }
    }

    pub fn device_type(&self) -> HidDeviceType {
        self.device_type
    }
}

fn hid_keycode_to_char(keycode: u8, shift: bool) -> Option<u8> {
    let unshifted: [u8; 57] = [
        0, 0, 0, 0, b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l',
        b'm', b'n', b'o', b'p', b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z',
        b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'0', b'\n', 0x1B, b'\x08',
        b'\t', b' ', b'-', b'=', b'[', b']', b'\\', b'#', b';', b'\'', b'`', b',', b'.', b'/',
    ];
    let shifted: [u8; 57] = [
        0, 0, 0, 0, b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'J', b'K', b'L',
        b'M', b'N', b'O', b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y', b'Z',
        b'!', b'@', b'#', b'$', b'%', b'^', b'&', b'*', b'(', b')', b'\n', 0x1B, b'\x08',
        b'\t', b' ', b'_', b'+', b'{', b'}', b'|', b'~', b':', b'"', b'~', b'<', b'>', b'?',
    ];
    if keycode < 4 || keycode >= 58 {
        return None;
    }
    let idx = keycode as usize;
    if shift {
        Some(shifted[idx])
    } else {
        Some(unshifted[idx])
    }
}
