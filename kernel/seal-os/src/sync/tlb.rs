// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! TLB shootdown — broadcast `invlpg` to all CPUs via IPI.
//!
//! Used after a page table entry is cleared so that stale mappings are
//! invalidated on every core.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use x86_64::VirtAddr;

use crate::cpu::{self, MAX_CPUS};

/// Address that needs to be invalidated (0 = none).
static SHOOTDOWN_ADDR: AtomicU64 = AtomicU64::new(0);

/// Per-CPU acknowledgement array.
static SHOOTDOWN_ACK: [AtomicBool; MAX_CPUS] = [const { AtomicBool::new(false) }; MAX_CPUS];

/// Request a TLB shootdown for `virt_addr` on all CPUs.
///
/// 1. Flushes the local TLB immediately.
/// 2. Sends IPI vector `0xFD` to every other CPU.
/// 3. Spins until all CPUs ack (or a ~1 ms timeout expires).
/// 4. Clears the shared shootdown address.
pub fn shootdown(virt_addr: VirtAddr) {
    // Flush locally first.
    unsafe { x86_64::instructions::tlb::flush(virt_addr); }

    let current = cpu::current_cpu_num() as usize;
    let cpu_count = cpu::CPU_COUNT.load(Ordering::SeqCst);

    if cpu_count <= 1 {
        return;
    }

    SHOOTDOWN_ADDR.store(virt_addr.as_u64(), Ordering::SeqCst);

    for cpu in 0..cpu_count {
        if cpu == current {
            continue;
        }
        SHOOTDOWN_ACK[cpu].store(false, Ordering::SeqCst);
        if let Some(apic_id) = cpu::apic_id_for_cpu(cpu as u32) {
            unsafe {
                crate::drivers::apic::local_apic().send_ipi(apic_id, 0xFD);
            }
        }
    }

    // Wait for ACKs with a ~1 ms timeout.
    let start = crate::drivers::interrupts::ticks();
    for cpu in 0..cpu_count {
        if cpu == current {
            continue;
        }
        while !SHOOTDOWN_ACK[cpu].load(Ordering::SeqCst) {
            let elapsed = crate::drivers::interrupts::ticks().wrapping_sub(start);
            if elapsed >= 2 {
                // Timeout — do not deadlock.
                break;
            }
            core::hint::spin_loop();
        }
    }

    SHOOTDOWN_ADDR.store(0, Ordering::SeqCst);
}

/// Handler for TLB shootdown IPI (vector `0xFD`).
///
/// Must be called from the interrupt handler on each CPU.
pub fn handle_tlb_shootdown_ipi() {
    let addr = SHOOTDOWN_ADDR.load(Ordering::SeqCst);
    if addr != 0 {
        let va = VirtAddr::new(addr);
        unsafe {
            x86_64::instructions::tlb::flush(va);
        }
    }

    let cpu = cpu::current_cpu_num() as usize;
    if cpu < MAX_CPUS {
        SHOOTDOWN_ACK[cpu].store(true, Ordering::SeqCst);
    }

    unsafe {
        crate::drivers::apic::local_apic().eoi();
    }
}
