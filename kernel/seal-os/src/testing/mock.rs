// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Mock infrastructure for unit tests.

use alloc::vec;
use alloc::vec::Vec;
use spin::Mutex;

/// Mock physical memory region backed by a Vec.
pub struct MockMemoryRegion {
    data: Vec<u8>,
}

impl MockMemoryRegion {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    pub fn read(&self, offset: usize) -> u8 {
        self.data[offset]
    }

    pub fn write(&mut self, offset: usize, value: u8) {
        self.data[offset] = value;
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }
}

/// Mock interrupt controller for testing without real hardware.
pub struct MockInterruptController;

impl MockInterruptController {
    pub const fn new() -> Self {
        Self
    }
    pub fn enable_irq(&self, _irq: u8) {}
    pub fn disable_irq(&self, _irq: u8) {}
    pub fn send_eoi(&self, _irq: u8) {}
}

/// Mock timer that can be manually advanced.
pub struct MockTimer {
    ticks: Mutex<u64>,
}

impl MockTimer {
    pub const fn new() -> Self {
        Self {
            ticks: Mutex::new(0),
        }
    }

    pub fn advance(&self, delta: u64) {
        *self.ticks.lock() += delta;
    }

    pub fn set(&self, value: u64) {
        *self.ticks.lock() = value;
    }

    pub fn now(&self) -> u64 {
        *self.ticks.lock()
    }
}

/// Mock serial output buffer for capturing test output.
pub struct MockSerialBuffer {
    buffer: Mutex<Vec<u8>>,
}

impl MockSerialBuffer {
    pub const fn new() -> Self {
        Self {
            buffer: Mutex::new(Vec::new()),
        }
    }

    pub fn push(&self, byte: u8) {
        self.buffer.lock().push(byte);
    }

    pub fn as_string(&self) -> alloc::string::String {
        let bytes = self.buffer.lock();
        alloc::string::String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn clear(&self) {
        self.buffer.lock().clear();
    }
}
