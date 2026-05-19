// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! USB HID class driver — keyboards and mice via interrupt transfers.
//! NOTE: Protocol constants and struct defined; interrupt transfer processing not yet implemented.

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

pub struct HidDriver {
    device_type: HidDeviceType,
    report_size: u8,
}

impl HidDriver {
    pub fn new(protocol: u8) -> Self {
        let device_type = match protocol {
            HID_BOOT_KEYBOARD => HidDeviceType::Keyboard,
            HID_BOOT_MOUSE => HidDeviceType::Mouse,
            _ => HidDeviceType::Unknown,
        };
        Self {
            device_type,
            report_size: 8,
        }
    }

    pub fn process_report(&self, _report: &[u8]) {
        // Not yet implemented — requires xHCI interrupt transfer setup
    }

    pub fn device_type(&self) -> HidDeviceType {
        self.device_type
    }
}
