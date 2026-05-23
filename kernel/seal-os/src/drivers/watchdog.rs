// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! APIC-timer-based watchdog.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use x86_64::instructions::port::Port;

static WATCHDOG_ENABLED: AtomicBool = AtomicBool::new(false);
static WATCHDOG_DEADLINE: AtomicU64 = AtomicU64::new(0);
static WATCHDOG_PETTED: AtomicBool = AtomicBool::new(false);
static WATCHDOG_PERIOD_MS: AtomicU32 = AtomicU32::new(0);

/// Initialise the watchdog with a given period in milliseconds.
pub fn init(period_ms: u32) {
    let now = crate::drivers::interrupts::ticks();
    WATCHDOG_PERIOD_MS.store(period_ms, Ordering::Relaxed);
    WATCHDOG_DEADLINE.store(now.saturating_add(period_ms as u64), Ordering::Relaxed);
    WATCHDOG_PETTED.store(true, Ordering::Relaxed);
    WATCHDOG_ENABLED.store(true, Ordering::Relaxed);
}

/// Pet the watchdog, resetting the deadline.
pub fn pet() {
    if !WATCHDOG_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let period = WATCHDOG_PERIOD_MS.load(Ordering::Relaxed) as u64;
    let now = crate::drivers::interrupts::ticks();
    WATCHDOG_DEADLINE.store(now.saturating_add(period), Ordering::Relaxed);
    WATCHDOG_PETTED.store(true, Ordering::Relaxed);
}

/// Return `true` if the watchdog has not expired.
pub fn is_alive() -> bool {
    if !WATCHDOG_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    let now = crate::drivers::interrupts::ticks();
    let deadline = WATCHDOG_DEADLINE.load(Ordering::Relaxed);
    let petted = WATCHDOG_PETTED.load(Ordering::Relaxed);
    now <= deadline || petted
}

/// Check the watchdog deadline and reset the system if it has expired
/// without being petted.
pub fn check() {
    if !WATCHDOG_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let now = crate::drivers::interrupts::ticks();
    let deadline = WATCHDOG_DEADLINE.load(Ordering::Relaxed);
    let petted = WATCHDOG_PETTED.load(Ordering::Relaxed);

    if now > deadline {
        if !petted {
            force_reset();
        } else {
            let period = WATCHDOG_PERIOD_MS.load(Ordering::Relaxed) as u64;
            WATCHDOG_DEADLINE.store(now.saturating_add(period), Ordering::Relaxed);
            WATCHDOG_PETTED.store(false, Ordering::Relaxed);
        }
    }
}

/// Force an immediate system reset via the keyboard controller.
pub fn force_reset() {
    unsafe {
        let mut cmd: Port<u8> = Port::new(0x64);
        cmd.write(0xFE);
    }
}
