// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! IDT + PIC initialization, keyboard/mouse interrupt handlers,
//! and global input event queue for routing to the window manager.

use spin::Lazy;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};

use crate::serial_println;
use crate::wm::event::InputEvent;

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

// Fixed-size ring buffer for input events — no heap allocation in IRQ context
const EVENT_QUEUE_SIZE: usize = 256;
static EVENT_QUEUE: spin::Mutex<RingBuffer> = spin::Mutex::new(RingBuffer::new());

struct RingBuffer {
    buf: [Option<InputEvent>; EVENT_QUEUE_SIZE],
    head: usize,
    tail: usize,
    count: usize,
}

impl RingBuffer {
    const fn new() -> Self {
        Self {
            buf: [None; EVENT_QUEUE_SIZE],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    fn push(&mut self, event: InputEvent) {
        if self.count < EVENT_QUEUE_SIZE {
            self.buf[self.tail] = Some(event);
            self.tail = (self.tail + 1) % EVENT_QUEUE_SIZE;
            self.count += 1;
        }
    }

    fn pop(&mut self) -> Option<InputEvent> {
        if self.count == 0 {
            return None;
        }
        let event = self.buf[self.head].take();
        self.head = (self.head + 1) % EVENT_QUEUE_SIZE;
        self.count -= 1;
        event
    }
}

// PS/2 mouse state machine for 3-byte packet accumulation
static MOUSE_STATE: spin::Mutex<MousePacketState> = spin::Mutex::new(MousePacketState::new());

struct MousePacketState {
    bytes: [u8; 3],
    index: u8,
}

impl MousePacketState {
    const fn new() -> Self {
        Self {
            bytes: [0; 3],
            index: 0,
        }
    }

    fn feed(&mut self, byte: u8) -> Option<(i32, i32, u8)> {
        match self.index {
            0 => {
                // First byte must have bit 3 set (always-1 bit in PS/2 protocol)
                if byte & 0x08 != 0 {
                    self.bytes[0] = byte;
                    self.index = 1;
                }
                None
            }
            1 => {
                self.bytes[1] = byte;
                self.index = 2;
                None
            }
            2 => {
                self.bytes[2] = byte;
                self.index = 0;

                let flags = self.bytes[0];
                let buttons = flags & 0x07;

                let mut dx = self.bytes[1] as i32;
                let mut dy = self.bytes[2] as i32;

                // Sign extension
                if flags & 0x10 != 0 { dx -= 256; }
                if flags & 0x20 != 0 { dy -= 256; }

                // PS/2 mouse Y is inverted
                dy = -dy;

                Some((dx, dy, buttons))
            }
            _ => {
                self.index = 0;
                None
            }
        }
    }
}

pub fn init() {
    init_pic();
    init_mouse();
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
        data1.write(4);
        data2.write(2);
        // ICW4: 8086 mode
        data1.write(0x01);
        data2.write(0x01);
        // Unmask: timer (0), keyboard (1), cascade (2), mouse (12)
        data1.write(0b1111_1000);
        data2.write(0b1110_1111);
    }
}

fn init_mouse() {
    unsafe {
        use x86_64::instructions::port::Port;
        let mut cmd = Port::<u8>::new(0x64);
        let mut data = Port::<u8>::new(0x60);

        // Enable auxiliary device (mouse)
        wait_write();
        cmd.write(0xA8);

        // Enable interrupts for mouse
        wait_write();
        cmd.write(0x20); // read controller config
        wait_read();
        let status = data.read();
        wait_write();
        cmd.write(0x60); // write controller config
        wait_write();
        data.write(status | 0x02); // enable IRQ12

        // Tell mouse to use default settings
        mouse_write(0xF6);
        mouse_read();

        // Enable mouse data reporting
        mouse_write(0xF4);
        mouse_read();
    }
}

unsafe fn wait_write() {
    use x86_64::instructions::port::Port;
    let mut status = Port::<u8>::new(0x64);
    for _ in 0..10000 {
        if status.read() & 0x02 == 0 { return; }
    }
}

unsafe fn wait_read() {
    use x86_64::instructions::port::Port;
    let mut status = Port::<u8>::new(0x64);
    for _ in 0..10000 {
        if status.read() & 0x01 != 0 { return; }
    }
}

unsafe fn mouse_write(cmd: u8) {
    use x86_64::instructions::port::Port;
    let mut port_cmd = Port::<u8>::new(0x64);
    let mut port_data = Port::<u8>::new(0x60);
    wait_write();
    port_cmd.write(0xD4); // next byte goes to mouse
    wait_write();
    port_data.write(cmd);
}

unsafe fn mouse_read() -> u8 {
    use x86_64::instructions::port::Port;
    let mut data = Port::<u8>::new(0x60);
    wait_read();
    data.read()
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

pub fn poll_event() -> Option<InputEvent> {
    EVENT_QUEUE.lock().pop()
}

extern "x86-interrupt" fn timer_handler(_frame: InterruptStackFrame) {
    *TICKS.lock() += 1;
    send_eoi(0);
}

extern "x86-interrupt" fn keyboard_handler(_frame: InterruptStackFrame) {
    let scancode: u8 = unsafe { x86_64::instructions::port::Port::new(0x60).read() };

    // Key release scancodes have bit 7 set
    if scancode & 0x80 != 0 {
        let release_code = scancode & 0x7F;
        EVENT_QUEUE.lock().push(InputEvent::KeyRelease(release_code));
    } else {
        EVENT_QUEUE.lock().push(InputEvent::KeyPress(scancode));
    }

    send_eoi(1);
}

extern "x86-interrupt" fn mouse_handler(_frame: InterruptStackFrame) {
    let byte: u8 = unsafe { x86_64::instructions::port::Port::new(0x60).read() };

    if let Some((dx, dy, buttons)) = MOUSE_STATE.lock().feed(byte) {
        if dx != 0 || dy != 0 {
            EVENT_QUEUE.lock().push(InputEvent::MouseMove { dx, dy });
        }
        // Track button state changes
        static LAST_BUTTONS: spin::Mutex<u8> = spin::Mutex::new(0);
        let mut last = LAST_BUTTONS.lock();
        if buttons != *last {
            // Left button
            if (buttons & 1) != (*last & 1) {
                EVENT_QUEUE.lock().push(InputEvent::MouseButton {
                    button: 0,
                    pressed: buttons & 1 != 0,
                });
            }
            // Right button
            if (buttons & 2) != (*last & 2) {
                EVENT_QUEUE.lock().push(InputEvent::MouseButton {
                    button: 1,
                    pressed: buttons & 2 != 0,
                });
            }
            *last = buttons;
        }
    }

    send_eoi(12);
}

extern "x86-interrupt" fn breakpoint_handler(frame: InterruptStackFrame) {
    serial_println!("[INT] BREAKPOINT\n{:#?}", frame);
}

extern "x86-interrupt" fn double_fault_handler(frame: InterruptStackFrame, _code: u64) -> ! {
    serial_println!("[FATAL] DOUBLE FAULT\n{:#?}", frame);
    loop { x86_64::instructions::hlt(); }
}

pub fn scancode_to_char(code: u8) -> u8 {
    const MAP: &[u8; 58] = b"\0\x1b1234567890-=\x08\tqwertyuiop[]\n\0asdfghjkl;'`\0\\zxcvbnm,./\0*\0 ";
    if (code as usize) < MAP.len() { MAP[code as usize] } else { 0 }
}

pub fn scancode_to_special(code: u8) -> Option<SpecialKey> {
    match code {
        0x48 => Some(SpecialKey::Up),
        0x50 => Some(SpecialKey::Down),
        0x4B => Some(SpecialKey::Left),
        0x4D => Some(SpecialKey::Right),
        0x47 => Some(SpecialKey::Home),
        0x4F => Some(SpecialKey::End),
        0x49 => Some(SpecialKey::PageUp),
        0x51 => Some(SpecialKey::PageDown),
        0x53 => Some(SpecialKey::Delete),
        0x01 => Some(SpecialKey::Escape),
        0x3B => Some(SpecialKey::F1),
        0x3C => Some(SpecialKey::F2),
        0x3D => Some(SpecialKey::F3),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecialKey {
    Up, Down, Left, Right,
    Home, End, PageUp, PageDown,
    Delete, Escape,
    F1, F2, F3,
}
