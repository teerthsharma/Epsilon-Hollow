// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! IDT + PIC initialization, keyboard/mouse interrupt handlers,
//! and global input event queue for routing to the window manager.

#[cfg(not(test))]
use spin::Lazy;
#[cfg(not(test))]
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};
use x86_64::registers::control::Cr2;

use core::sync::atomic::{AtomicU64, Ordering};

use crate::memory::gdt;
use crate::serial_println;
#[cfg(not(test))]
use crate::wm::event::InputEvent;

const IRQ_OFFSET: u8 = 32;

const VECTOR_TIMER_APIC: u8 = 48;
const VECTOR_IPI_TLB_SHOOTDOWN: u8 = 0xFD;
pub const VECTOR_IPI_RESCHEDULE: u8 = 0xFE;

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum Irq {
    Timer = IRQ_OFFSET,
    Keyboard = IRQ_OFFSET + 1,
    Mouse = IRQ_OFFSET + 12,
}

#[cfg(not(test))]
pub static IDT: Lazy<InterruptDescriptorTable> = Lazy::new(|| {
    let mut idt = InterruptDescriptorTable::new();
    idt.breakpoint.set_handler_fn(breakpoint_handler);
    unsafe {
        idt.double_fault
            .set_handler_fn(double_fault_handler)
            .set_stack_index(gdt::DOUBLE_FAULT_IST_INDEX);
    }
    idt.invalid_opcode.set_handler_fn(invalid_opcode_handler);
    idt.segment_not_present.set_handler_fn(segment_not_present_handler);
    idt.stack_segment_fault.set_handler_fn(stack_segment_fault_handler);
    idt.general_protection_fault.set_handler_fn(general_protection_fault_handler);
    idt.page_fault.set_handler_fn(page_fault_handler);
    idt[VECTOR_TIMER_APIC].set_handler_fn(timer_handler_apic);
    idt[Irq::Keyboard as u8].set_handler_fn(keyboard_handler);
    idt[Irq::Mouse as u8].set_handler_fn(mouse_handler);
    idt[VECTOR_IPI_RESCHEDULE].set_handler_fn(crate::cpu::smp::reschedule_ipi_handler);
    #[cfg(feature = "test-mode")]
    idt[0xFD].set_handler_fn(crate::drivers::apic::apic_test_handler);
    #[cfg(not(feature = "test-mode"))]
    idt[VECTOR_IPI_TLB_SHOOTDOWN].set_handler_fn(crate::cpu::smp::tlb_shootdown_ipi_handler);
    idt
});

static TICKS: AtomicU64 = AtomicU64::new(0);

// Fixed-size ring buffer for input events — no heap allocation in IRQ context
#[cfg(not(test))]
const EVENT_QUEUE_SIZE: usize = 256;
#[cfg(not(test))]
static EVENT_QUEUE: spin::Mutex<RingBuffer> = spin::Mutex::new(RingBuffer::new());

#[cfg(not(test))]
struct RingBuffer {
    buf: [Option<InputEvent>; EVENT_QUEUE_SIZE],
    head: usize,
    tail: usize,
    count: usize,
}

#[cfg(not(test))]
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
#[cfg(not(test))]
static MOUSE_STATE: spin::Mutex<MousePacketState> = spin::Mutex::new(MousePacketState::new());

#[cfg(not(test))]
struct MousePacketState {
    bytes: [u8; 3],
    index: u8,
}

#[cfg(not(test))]
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

#[cfg(not(test))]
pub fn init() {
    unsafe {
        if !crate::drivers::apic::local_apic_init_done() {
            crate::drivers::apic::local_apic().init();
        }
        crate::drivers::apic::ioapic().init();

        let bsp_id = crate::drivers::apic::bsp_apic_id();
        crate::drivers::apic::ioapic().redirect_irq(1, Irq::Keyboard as u8, bsp_id);
        crate::drivers::apic::ioapic().redirect_irq(12, Irq::Mouse as u8, bsp_id);
        crate::drivers::apic::ioapic().set_mask(1, false);
        crate::drivers::apic::ioapic().set_mask(12, false);
    }

    init_mouse();
    IDT.load();
    x86_64::instructions::interrupts::enable();
}

#[cfg(not(test))]
/// Reload the IDT (used by APs after they enter long mode).
pub unsafe fn reload_idt() {
    IDT.load();
}

#[cfg(test)]
pub fn init() {}

#[cfg(not(test))]
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

#[cfg(not(test))]
unsafe fn wait_write() {
    use x86_64::instructions::port::Port;
    let mut status = Port::<u8>::new(0x64);
    for _ in 0..10000 {
        if status.read() & 0x02 == 0 { return; }
    }
}

#[cfg(not(test))]
unsafe fn wait_read() {
    use x86_64::instructions::port::Port;
    let mut status = Port::<u8>::new(0x64);
    for _ in 0..10000 {
        if status.read() & 0x01 != 0 { return; }
    }
}

#[cfg(not(test))]
unsafe fn mouse_write(cmd: u8) {
    use x86_64::instructions::port::Port;
    let mut port_cmd = Port::<u8>::new(0x64);
    let mut port_data = Port::<u8>::new(0x60);
    wait_write();
    port_cmd.write(0xD4); // next byte goes to mouse
    wait_write();
    port_data.write(cmd);
}

#[cfg(not(test))]
unsafe fn mouse_read() -> u8 {
    use x86_64::instructions::port::Port;
    let mut data = Port::<u8>::new(0x60);
    wait_read();
    data.read()
}

#[cfg(not(test))]
fn send_eoi(_irq: u8) {
    unsafe {
        crate::drivers::apic::LOCAL_APIC.eoi();
    }
}

pub fn ticks() -> u64 {
    TICKS.load(Ordering::Relaxed)
}

#[cfg(test)]
pub fn mock_advance_ticks(delta: u64) {
    TICKS.fetch_add(delta, Ordering::Relaxed);
}

#[cfg(not(test))]
pub fn poll_event() -> Option<InputEvent> {
    if let Some(mut queue) = EVENT_QUEUE.try_lock() {
        queue.pop()
    } else {
        None
    }
}

#[cfg(not(test))]
pub fn push_event(event: InputEvent) {
    if let Some(mut queue) = EVENT_QUEUE.try_lock() {
        queue.push(event);
    }
}

#[cfg(test)]
pub fn poll_event() -> Option<crate::wm::event::InputEvent> {
    None
}

#[cfg(test)]
pub fn push_event(_event: crate::wm::event::InputEvent) {}

#[cfg(not(test))]
extern "x86-interrupt" fn timer_handler_pic(_frame: InterruptStackFrame) {
    TICKS.fetch_add(1, Ordering::Relaxed);
    crate::process::scheduler::scheduler_tick();
    // Legacy PIC EOI (fallback only)
    unsafe {
        x86_64::instructions::port::Port::<u8>::new(0x20).write(0x20);
    }
}

#[cfg(not(test))]
extern "x86-interrupt" fn timer_handler_apic(_frame: InterruptStackFrame) {
    TICKS.fetch_add(1, Ordering::Relaxed);
    crate::process::scheduler::scheduler_tick();
    crate::drivers::watchdog::check();
    unsafe {
        crate::drivers::apic::LOCAL_APIC.eoi();
    }
}

#[cfg(not(test))]
extern "x86-interrupt" fn keyboard_handler(_frame: InterruptStackFrame) {
    let scancode: u8 = unsafe { x86_64::instructions::port::Port::new(0x60).read() };

    if let Some(mut queue) = EVENT_QUEUE.try_lock() {
        // Key release scancodes have bit 7 set
        if scancode & 0x80 != 0 {
            let release_code = scancode & 0x7F;
            queue.push(InputEvent::KeyRelease(release_code));
        } else {
            queue.push(InputEvent::KeyPress(scancode));
        }
    }

    send_eoi(1);
}

#[cfg(not(test))]
extern "x86-interrupt" fn mouse_handler(_frame: InterruptStackFrame) {
    let byte: u8 = unsafe { x86_64::instructions::port::Port::new(0x60).read() };

    let mouse_event = if let Some(mut state) = MOUSE_STATE.try_lock() {
        state.feed(byte)
    } else {
        None
    };

    if let Some((dx, dy, buttons)) = mouse_event {
        if let Some(mut queue) = EVENT_QUEUE.try_lock() {
            if dx != 0 || dy != 0 {
                queue.push(InputEvent::MouseMove { dx, dy });
            }
            // Track button state changes
            static LAST_BUTTONS: spin::Mutex<u8> = spin::Mutex::new(0);
            if let Some(mut last) = LAST_BUTTONS.try_lock() {
                if buttons != *last {
                    // Left button
                    if (buttons & 1) != (*last & 1) {
                        queue.push(InputEvent::MouseButton {
                            button: 0,
                            pressed: buttons & 1 != 0,
                        });
                    }
                    // Right button
                    if (buttons & 2) != (*last & 2) {
                        queue.push(InputEvent::MouseButton {
                            button: 1,
                            pressed: buttons & 2 != 0,
                        });
                    }
                    *last = buttons;
                }
            }
        }
    }

    send_eoi(12);
}

#[cfg(not(test))]
extern "x86-interrupt" fn breakpoint_handler(frame: InterruptStackFrame) {
    serial_println!("[INT] BREAKPOINT\n{:#?}", frame);
}

#[cfg(not(test))]
extern "x86-interrupt" fn page_fault_handler(frame: InterruptStackFrame, error_code: PageFaultErrorCode) {
    let addr = Cr2::read_raw();

    // Attempt demand paging for known mmap regions.
    if crate::memory::mmap::handle_page_fault(x86_64::VirtAddr::new(addr)) {
        return;
    }

    serial_println!("[FAULT] Page fault at {:#x}, error code: {:?}", addr, error_code);
    serial_println!("{:#?}", frame);
    loop { x86_64::instructions::hlt(); }
}

extern "x86-interrupt" fn double_fault_handler(frame: InterruptStackFrame, _code: u64) -> ! {
    serial_println!("[FATAL] DOUBLE FAULT\n{:#?}", frame);
    loop { x86_64::instructions::hlt(); }
}

#[cfg(not(test))]
extern "x86-interrupt" fn invalid_opcode_handler(frame: InterruptStackFrame) {
    serial_println!("[FAULT] Invalid Opcode (#UD)\n{:#?}", frame);
    loop { x86_64::instructions::hlt(); }
}

#[cfg(not(test))]
extern "x86-interrupt" fn segment_not_present_handler(frame: InterruptStackFrame, error_code: u64) {
    serial_println!("[FAULT] Segment Not Present (#NP), error code: {:#x}\n{:#?}", error_code, frame);
    loop { x86_64::instructions::hlt(); }
}

#[cfg(not(test))]
extern "x86-interrupt" fn stack_segment_fault_handler(frame: InterruptStackFrame, error_code: u64) {
    serial_println!("[FAULT] Stack Segment Fault (#SS), error code: {:#x}\n{:#?}", error_code, frame);
    loop { x86_64::instructions::hlt(); }
}

#[cfg(not(test))]
extern "x86-interrupt" fn general_protection_fault_handler(frame: InterruptStackFrame, error_code: u64) {
    serial_println!("[FAULT] General Protection Fault (#GP), error code: {:#x}\n{:#?}", error_code, frame);
    loop { x86_64::instructions::hlt(); }
}

pub fn scancode_to_char(code: u8) -> u8 {
    const MAP: &[u8; 59] = b"\0\x1b1234567890-==\x08\tqwertyuiop[]\n\0asdfghjkl;'`\0\\zxcvbnm,./\0*\0 ";
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
