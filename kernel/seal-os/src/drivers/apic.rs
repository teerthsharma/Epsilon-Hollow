// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Local APIC and I/O APIC driver for x86_64 SMP.
//!
//! Replaces the legacy 8259 PIC with memory-mapped APIC controllers.

use core::cell::UnsafeCell;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

#[cfg(all(not(test), feature = "test-mode"))]
use x86_64::structures::idt::InterruptStackFrame;

// ---------------------------------------------------------------------------
// Local APIC register offsets
// ---------------------------------------------------------------------------

const APIC_ID: u32 = 0x020;
const APIC_VER: u32 = 0x030;
const TPR: u32 = 0x080;
const EOI: u32 = 0x0B0;
const SIV: u32 = 0x0F0;
const ICR_LOW: u32 = 0x300;
const ICR_HIGH: u32 = 0x310;
const TIMER_LVT: u32 = 0x320;
const THERMAL_LVT: u32 = 0x330;
const PERF_LVT: u32 = 0x340;
const LINT0_LVT: u32 = 0x350;
const LINT1_LVT: u32 = 0x360;
const ERROR_LVT: u32 = 0x370;
const TIMER_INITCNT: u32 = 0x380;
const TIMER_CURRCNT: u32 = 0x390;
const TIMER_DIV: u32 = 0x3E0;

// ---------------------------------------------------------------------------
// Local APIC
// ---------------------------------------------------------------------------

/// x86 Local APIC driver.
pub struct LocalApic {
    base: usize,
}

impl LocalApic {
    pub const fn new(base: usize) -> Self {
        Self { base }
    }

    /// Read a 32-bit Local APIC register.
    pub unsafe fn read_reg(&self, offset: u32) -> u32 {
        read_volatile((self.base + offset as usize) as *const u32)
    }

    /// Write a 32-bit Local APIC register.
    pub unsafe fn write_reg(&self, offset: u32, val: u32) {
        write_volatile((self.base + offset as usize) as *mut u32, val);
    }

    /// Enable the Local APIC and set task priority to accept all interrupts.
    pub unsafe fn init(&self) {
        // Software enable APIC (bit 8), vector 0xFF for spurious
        self.write_reg(SIV, self.read_reg(SIV) | 0x1FF);
        // Task Priority Register = 0
        self.write_reg(TPR, 0);
    }

    /// Send End-Of-Interrupt.
    pub unsafe fn eoi(&self) {
        self.write_reg(EOI, 0);
    }

    /// Send an Inter-Processor Interrupt.
    pub unsafe fn send_ipi(&self, apic_id: u32, vector: u8) {
        // Wait for delivery status idle (bit 12)
        while self.read_reg(ICR_LOW) & 0x1000 != 0 {}
        self.write_reg(ICR_HIGH, apic_id << 24);
        self.write_reg(ICR_LOW, vector as u32 | 0x4000);
    }

    /// Initialise the Local APIC timer in periodic mode.
    pub unsafe fn init_timer(&self, divide: u32, initial_count: u32, vector: u8) {
        self.write_reg(TIMER_DIV, divide);
        // Periodic mode (bit 17), unmasked, vector
        self.write_reg(TIMER_LVT, (1 << 17) | vector as u32);
        self.write_reg(TIMER_INITCNT, initial_count);
    }

    /// Return the Local APIC ID (bits 24-31 of the ID register).
    pub unsafe fn id(&self) -> u32 {
        self.read_reg(APIC_ID) >> 24
    }

    /// Calibrate the APIC timer against the PIT.
    ///
    /// `pit_ticks_for_100ms` is the PIT reload value for a ~100 ms delay.
    /// Returns the `initial_count` value to program for a ~1000 Hz tick rate.
    pub unsafe fn calibrate_timer(&self, pit_ticks_for_100ms: u32) -> u32 {
        use x86_64::instructions::port::Port;

        // Temporarily configure PIT channel 0 as one-shot
        let mut cmd = Port::<u8>::new(0x43);
        let mut ch0 = Port::<u8>::new(0x40);

        cmd.write(0x30); // channel 0, lobyte/hibyte, mode 0, binary
        ch0.write((pit_ticks_for_100ms & 0xFF) as u8);
        ch0.write(((pit_ticks_for_100ms >> 8) & 0xFF) as u8);

        // Save old timer LVT and program one-shot with vector 0xFE
        let old_lvt = self.read_reg(TIMER_LVT);
        self.write_reg(TIMER_LVT, 0xFE);
        self.write_reg(TIMER_INITCNT, 0xFFFFFFFF);

        // Wait for PIT to reach terminal count
        loop {
            cmd.write(0x00); // latch channel 0
            let lo = ch0.read();
            let hi = ch0.read();
            let count = ((hi as u32) << 8) | (lo as u32);
            if count == 0 {
                break;
            }
        }

        let current = self.read_reg(TIMER_CURRCNT);
        let elapsed = 0xFFFFFFFFu32 - current;

        // Restore timer LVT
        self.write_reg(TIMER_LVT, old_lvt);

        // initial_count for ~1000 Hz (1 ms period = 100 ms / 100)
        elapsed / 100
    }
}

// ---------------------------------------------------------------------------
// I/O APIC
// ---------------------------------------------------------------------------

const IOAPIC_IOREGSEL: u32 = 0x00;
const IOAPIC_IOWIN: u32 = 0x10;

/// I/O APIC driver.
pub struct IoApic {
    base: usize,
}

impl IoApic {
    pub const fn new(base: usize) -> Self {
        Self { base }
    }

    unsafe fn read_reg(&self, reg: u8) -> u32 {
        write_volatile((self.base + IOAPIC_IOREGSEL as usize) as *mut u32, reg as u32);
        read_volatile((self.base + IOAPIC_IOWIN as usize) as *const u32)
    }

    unsafe fn write_reg(&self, reg: u8, val: u32) {
        write_volatile((self.base + IOAPIC_IOREGSEL as usize) as *mut u32, reg as u32);
        write_volatile((self.base + IOAPIC_IOWIN as usize) as *mut u32, val);
    }

    /// Read ID and version registers.
    pub unsafe fn init(&self) {
        let id = self.read_reg(0x00);
        let ver = self.read_reg(0x01);
        let _max_redir = ((ver >> 16) & 0xFF) as u8;
        let _ = id;
    }

    /// Route an IRQ to a vector on a specific Local APIC.
    pub unsafe fn redirect_irq(&self, irq: u8, vector: u8, destination_apic_id: u32) {
        let reg_low = 0x10 + irq * 2;
        let reg_high = reg_low + 1;

        let low = vector as u32;
        let high = destination_apic_id << 24;

        self.write_reg(reg_high, high);
        self.write_reg(reg_low, low);
    }

    /// Mask or unmask an IRQ line.
    pub unsafe fn set_mask(&self, irq: u8, masked: bool) {
        let reg_low = 0x10 + irq * 2;
        let mut low = self.read_reg(reg_low);
        if masked {
            low |= 1 << 16;
        } else {
            low &= !(1 << 16);
        }
        self.write_reg(reg_low, low);
    }
}

// ---------------------------------------------------------------------------
// Global instances
// ---------------------------------------------------------------------------

#[cfg(not(test))]
pub(crate) static LOCAL_APIC: LocalApic = LocalApic::new(0xFEE00000);

#[cfg(not(test))]
pub(crate) static IO_APIC: IoApic = IoApic::new(0xFEC00000);

static LOCAL_APIC_INIT: AtomicBool = AtomicBool::new(false);
static IOAPIC_INIT: AtomicBool = AtomicBool::new(false);
static BSP_APIC_ID: AtomicU32 = AtomicU32::new(0);

const PIT_100MS_TICKS: u32 = 119318; // ~100 ms at 1.193182 MHz

/// Access the global Local APIC instance.
///
/// # Safety
/// All operations are volatile MMIO reads/writes to hardware registers.
/// The hardware itself serializes these operations, so `&self` is sufficient.
#[cfg(not(test))]
pub unsafe fn local_apic() -> &'static LocalApic {
    &LOCAL_APIC
}

/// Access the global I/O APIC instance.
///
/// # Safety
/// Same as `local_apic` — volatile MMIO only.
#[cfg(not(test))]
pub unsafe fn ioapic() -> &'static IoApic {
    &IO_APIC
}

/// Whether the Local APIC has already been initialised.
pub fn local_apic_init_done() -> bool {
    LOCAL_APIC_INIT.load(Ordering::SeqCst)
}

/// Return the BSP APIC ID saved during initialisation.
pub fn bsp_apic_id() -> u32 {
    BSP_APIC_ID.load(Ordering::SeqCst)
}

/// Initialise the global Local APIC and I/O APIC instances.
#[cfg(not(test))]
pub unsafe fn init() {
    if !LOCAL_APIC_INIT.swap(true, Ordering::SeqCst) {
        LOCAL_APIC.init();
    }
    if !IOAPIC_INIT.swap(true, Ordering::SeqCst) {
        IO_APIC.init();
    }
    let id = LOCAL_APIC.id();
    BSP_APIC_ID.store(id, Ordering::SeqCst);
    crate::serial_println!(
        "[APIC] Local APIC ID={}, IO APIC ready",
        id
    );
}

/// Calibrate and start the Local APIC timer on the BSP.
pub unsafe fn init_local_apic_timer_for_bsp() {
    let initial_count = LOCAL_APIC.calibrate_timer(PIT_100MS_TICKS);
    LOCAL_APIC.init_timer(3, initial_count, 48);
    crate::serial_println!("[APIC] BSP timer calibrated, initial_count={}", initial_count);
}

/// Calibrate and start the Local APIC timer on an AP.
pub unsafe fn init_local_apic_timer_for_ap() {
    let initial_count = LOCAL_APIC.calibrate_timer(PIT_100MS_TICKS);
    LOCAL_APIC.init_timer(3, initial_count, 48);
}

// ---------------------------------------------------------------------------
// In-kernel test support
// ---------------------------------------------------------------------------

#[cfg(all(not(test), feature = "test-mode"))]
pub(crate) static TEST_IPI_FIRED: AtomicBool = AtomicBool::new(false);

#[cfg(all(not(test), feature = "test-mode"))]
pub(crate) extern "x86-interrupt" fn apic_test_handler(_frame: InterruptStackFrame) {
    TEST_IPI_FIRED.store(true, Ordering::SeqCst);
    unsafe { LOCAL_APIC.eoi(); }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    #[cfg(all(not(test), feature = "test-mode"))]
    fn test_apic() -> TestResult {
        use core::sync::atomic::Ordering;

        unsafe {
            // Initialise local APIC
            super::LOCAL_APIC.init();
            super::TEST_IPI_FIRED.store(false, Ordering::SeqCst);

            // Send self-IPI on vector 0xFD (TLB shootdown / test vector)
            let my_id = super::LOCAL_APIC.id();
            super::LOCAL_APIC.send_ipi(my_id, 0xFD);

            // Brief spin-wait for interrupt delivery
            for _ in 0..1_000_000 {
                core::hint::spin_loop();
            }

            test_assert!(
                super::TEST_IPI_FIRED.load(Ordering::SeqCst),
                "self-IPI was not received"
            );
        }

        TestResult::Pass
    }

    #[cfg(not(all(not(test), feature = "test-mode")))]
    fn test_apic() -> TestResult {
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("apic::test_apic", test_apic);
    }
}
