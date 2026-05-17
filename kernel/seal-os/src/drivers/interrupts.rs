// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! IDT + PIC initialization for keyboard, timer, and mouse.

use spin::Lazy;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};

use crate::serial_println;

const PIC1_CMD: u16 = 0x20;
const PIC1_DATA: u16 = 0x21;
const PIC2_CMD: u16 = 0xA0;
const PIC2_DATA: u16 = 0xA1;

const IRQ_OFFSET: u8 = 32;

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum Irq {
    Timer = IRQ_OFFSET,
    Keyboard = IRQ_OFFSET + 1,
    Mouse = IRQ_OFFSET + 12,
}

static IDT: Lazy<InterruptDescriptorTable> = Lazy::new(|| {
    let mut idt = InterruptDescriptorTable::new();
    idt.breakpoint.set_handler_fn(breakpoint_handler);
    idt.double_fault.set_handler_fn(double_fault_handler);
    idt[Irq::Timer as u8].set_handler_fn(timer_handler);
    idt[Irq::Keyboard as u8].set_handler_fn(keyboard_handler);
    idt[Irq::Mouse as u8].set_handler_fn(mouse_handler);
    idt
});

static TICKS: spin::Mutex<u64> = spin::Mutex::new(0);

pub fn init() {
    init_pic();
    IDT.load();
    x86_64::instructions::interrupts::enable();
}

fn init_pic() {
    unsafe {
        use x86_64::instructions::port::Port;
        let mut cmd1 = Port::<u8>::new(PIC1_CMD);
        let mut data1 = Port::<u8>::new(PIC1_DATA);
        let mut cmd2 = Port::<u8>::new(PIC2_CMD);
        let mut data2 = Port::<u8>::new(PIC2_DATA);

        // ICW1: begin init
        cmd1.write(0x11);
        cmd2.write(0x11);
        // ICW2: IRQ offset
        data1.write(IRQ_OFFSET);
        data2.write(IRQ_OFFSET + 8);
        // ICW3: cascade
        data1.write(4); // slave on IRQ2
        data2.write(2);
        // ICW4: 8086 mode
        data1.write(0x01);
        data2.write(0x01);
        // Unmask: timer (0), keyboard (1), cascade (2), mouse (12)
        data1.write(0b1111_1000); // unmask IRQ 0,1,2
        data2.write(0b1110_1111); // unmask IRQ 12 (mouse)
    }
}

fn send_eoi(irq: u8) {
    unsafe {
        use x86_64::instructions::port::Port;
        if irq >= 8 {
            Port::<u8>::new(PIC2_CMD).write(0x20);
        }
        Port::<u8>::new(PIC1_CMD).write(0x20);
    }
}

pub fn ticks() -> u64 {
    *TICKS.lock()
}

extern "x86-interrupt" fn timer_handler(_frame: InterruptStackFrame) {
    *TICKS.lock() += 1;
    send_eoi(0);
}

extern "x86-interrupt" fn keyboard_handler(_frame: InterruptStackFrame) {
    let scancode: u8 = unsafe { x86_64::instructions::port::Port::new(0x60).read() };
    // Convert scancode to ASCII for basic keys
    if scancode < 128 {
        let ch = scancode_to_char(scancode);
        if ch != 0 as char {
            serial_println!("[KB] key: '{}'", ch);
        }
    }
    send_eoi(1);
}

extern "x86-interrupt" fn mouse_handler(_frame: InterruptStackFrame) {
    let _data: u8 = unsafe { x86_64::instructions::port::Port::new(0x60).read() };
    send_eoi(12);
}

extern "x86-interrupt" fn breakpoint_handler(frame: InterruptStackFrame) {
    serial_println!("[INT] BREAKPOINT\n{:#?}", frame);
}

extern "x86-interrupt" fn double_fault_handler(frame: InterruptStackFrame, _code: u64) -> ! {
    serial_println!("[FATAL] DOUBLE FAULT\n{:#?}", frame);
    loop { x86_64::instructions::hlt(); }
}

fn scancode_to_char(code: u8) -> char {
    const MAP: &[u8; 58] = b"\0\x1b1234567890-=\x08\tqwertyuiop[]\n\0asdfghjkl;'`\0\\zxcvbnm,./\0*\0 ";
    if (code as usize) < MAP.len() { MAP[code as usize] as char } else { '\0' }
}
