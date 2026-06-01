// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! COM1 serial port driver for debug output.

use core::fmt;
use spin::Mutex;
use x86_64::instructions::port::Port;

const COM1: u16 = 0x3F8;

pub static SERIAL: Mutex<SerialPort> = Mutex::new(SerialPort::new(COM1));

pub struct SerialPort {
    base: u16,
}

impl SerialPort {
    const fn new(base: u16) -> Self {
        Self { base }
    }

    pub fn init(&self) {
        unsafe {
            let port = |offset: u16| Port::<u8>::new(self.base + offset);
            port(1).write(0x00); // Disable interrupts
            port(3).write(0x80); // Enable DLAB
            port(0).write(0x01); // 115200 baud (divisor low)
            port(1).write(0x00); // (divisor high)
            port(3).write(0x03); // 8N1
            port(2).write(0xC7); // Enable FIFO
            port(4).write(0x0B); // IRQs enabled, RTS/DSR set
        }
    }

    fn write_byte(&self, byte: u8) {
        unsafe {
            let mut status_port = Port::<u8>::new(self.base + 5);
            let mut data_port = Port::<u8>::new(self.base);
            // Wait for transmit buffer empty
            while status_port.read() & 0x20 == 0 {}
            data_port.write(byte);
        }
    }
}

impl fmt::Write for SerialPort {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            if byte == b'\n' {
                self.write_byte(b'\r');
            }
            self.write_byte(byte);
        }
        Ok(())
    }
}

pub fn init() {
    SERIAL.lock().init();
}

#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => {
        $crate::drivers::serial::_print(format_args!($($arg)*))
    };
}

#[macro_export]
macro_rules! serial_println {
    () => ($crate::serial_print!("\n"));
    ($($arg:tt)*) => ($crate::serial_print!("{}\n", format_args!($($arg)*)))
}

#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    x86_64::instructions::interrupts::without_interrupts(|| {
        let _ = SERIAL.lock().write_fmt(args);
    });
}
