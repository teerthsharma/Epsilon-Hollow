// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! SMP bring-up: INIT-SIPI-SIPI protocol and AP idle loop.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use x86_64::registers::model_specific::Msr;
use x86_64::structures::idt::InterruptStackFrame;

use super::{alloc_ap_cpu, PerCpu, CPU_COUNT};
use crate::boot::ap_trampoline::{
    ap_trampoline, OFF_32BIT_CODE, OFF_64BIT_CODE, OFF_AP_MAIN_ADDR, OFF_AP_PER_CPU_PTR,
    OFF_BSP_PML4, OFF_GDTR, OFF_GDT_START, OFF_LONG64_PTR, OFF_PROT32_PTR, TRAMPOLINE_PAGE,
};
use crate::serial_println;

// ---------------------------------------------------------------------------
// Shared variables between BSP and AP
// ---------------------------------------------------------------------------

pub static AP_READY_FLAG: AtomicBool = AtomicBool::new(false);
pub static AP_PER_CPU_PTR: AtomicU64 = AtomicU64::new(0);

pub static TLB_SHOOTDOWN_ADDR: spin::Mutex<u64> = spin::Mutex::new(0);
pub static TLB_SHOOTDOWN_ACK: spin::Mutex<bool> = spin::Mutex::new(false);

// ---------------------------------------------------------------------------
// SMP bring-up
// ---------------------------------------------------------------------------

pub fn smp_init() {
    unsafe {
        super::init_bsp();
    }
    serial_println!("[SMP] BSP per-CPU data initialized");

    // Query ACPI for CPU topology.
    let cpu_count = crate::drivers::acpi::cpu_count();
    let apic_ids = crate::drivers::acpi::apic_ids();

    if cpu_count <= 1 {
        serial_println!("[SMP] Only BSP present ({} CPUs total)", cpu_count);
        return;
    }

    let bsp_apic_id = apic_ids[0];

    // Prepare trampoline page at 0x8000.
    prepare_trampoline_page();

    for i in 1..cpu_count {
        let apic_id = apic_ids[i];
        if apic_id == bsp_apic_id {
            continue;
        }

        let cpu_num = i as u32;
        let per_cpu = alloc_ap_cpu(apic_id, cpu_num);

        unsafe {
            AP_PER_CPU_PTR.store(per_cpu as *mut _ as u64, Ordering::SeqCst);
            // Patch the trampoline page so the AP gets the correct pointer.
            ((TRAMPOLINE_PAGE + OFF_AP_PER_CPU_PTR) as *mut u64)
                .write_unaligned(per_cpu as *mut _ as u64);
        }
        AP_READY_FLAG.store(false, Ordering::SeqCst);

        // INIT IPI
        send_ipi(apic_id, 0x00);

        // Wait ~10 ms (spin on PIT ticks)
        let start = crate::drivers::interrupts::ticks();
        while crate::drivers::interrupts::ticks().wrapping_sub(start) < 10 {
            core::hint::spin_loop();
        }

        // STARTUP IPI
        let vector = (TRAMPOLINE_PAGE >> 12) as u8;
        send_startup_ipi(apic_id, vector);

        // Wait for AP ready (timeout ~1s)
        let mut ready = false;
        let wait_start = crate::drivers::interrupts::ticks();
        while crate::drivers::interrupts::ticks().wrapping_sub(wait_start) < 1000 {
            if AP_READY_FLAG.load(Ordering::SeqCst) {
                ready = true;
                break;
            }
            core::hint::spin_loop();
        }

        if !ready {
            // Retry STARTUP IPI once
            send_startup_ipi(apic_id, vector);
            let wait_start = crate::drivers::interrupts::ticks();
            while crate::drivers::interrupts::ticks().wrapping_sub(wait_start) < 1000 {
                if AP_READY_FLAG.load(Ordering::SeqCst) {
                    ready = true;
                    break;
                }
                core::hint::spin_loop();
            }
        }

        if ready {
            serial_println!("[SMP] AP {} (APIC {}) online", cpu_num, apic_id);
        } else {
            serial_println!(
                "[SMP] AP {} (APIC {}) FAILED to come online",
                cpu_num,
                apic_id
            );
        }
    }

    serial_println!("[SMP] {} CPUs online", CPU_COUNT.load(Ordering::SeqCst));
}

/// AP entry point after the trampoline.
pub extern "C" fn ap_main() {
    unsafe {
        let per_cpu = &mut *(AP_PER_CPU_PTR.load(Ordering::SeqCst) as *mut PerCpu);

        // Set GS base
        Msr::new(0xC000_0101).write(per_cpu as *mut _ as u64);

        // Load TSS
        crate::memory::gdt::load_tss(per_cpu.tss_selector);

        // Load IDT (shared with BSP)
        #[cfg(not(test))]
        crate::drivers::interrupts::reload_idt();

        // Initialize local APIC timer (same frequency as BSP)
        crate::drivers::apic::init_local_apic_timer_for_ap();

        AP_READY_FLAG.store(true, Ordering::SeqCst);

        // Idle loop
        loop {
            crate::process::scheduler::scheduler_tick();
            let cpu = crate::cpu::this_cpu();
            if cpu.pending_reschedule {
                cpu.pending_reschedule = false;
                crate::process::scheduler::yield_current();
            }
            if cpu.is_idle {
                x86_64::instructions::hlt();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IPI handlers
// ---------------------------------------------------------------------------

#[cfg(not(test))]
pub extern "x86-interrupt" fn reschedule_ipi_handler(_frame: InterruptStackFrame) {
    unsafe {
        crate::cpu::this_cpu().pending_reschedule = true;
        crate::drivers::apic::LOCAL_APIC.eoi();
    }
}

#[cfg(not(test))]
pub extern "x86-interrupt" fn tlb_shootdown_ipi_handler(_frame: InterruptStackFrame) {
    unsafe {
        let addr = *TLB_SHOOTDOWN_ADDR.lock();
        if addr != 0 {
            x86_64::instructions::tlb::flush(x86_64::VirtAddr::new(addr));
        }
        *TLB_SHOOTDOWN_ACK.lock() = true;
        crate::drivers::apic::LOCAL_APIC.eoi();
    }
}

// ---------------------------------------------------------------------------
// Local APIC IPI helpers
// ---------------------------------------------------------------------------

fn local_apic_base() -> u64 {
    unsafe { Msr::new(0x1B).read() & !0xFFF }
}

fn send_ipi(apic_id: u32, vector: u8) {
    let base = local_apic_base();
    unsafe {
        let icr_high = (base + 0x310) as *mut u32;
        let icr_low = (base + 0x300) as *mut u32;
        icr_high.write_volatile(apic_id << 24);
        // Delivery mode 101 (INIT), level=1, assert=1
        icr_low.write_volatile(0x0000_4500 | (vector as u32));
    }
}

fn send_startup_ipi(apic_id: u32, vector: u8) {
    let base = local_apic_base();
    unsafe {
        let icr_high = (base + 0x310) as *mut u32;
        let icr_low = (base + 0x300) as *mut u32;
        icr_high.write_volatile(apic_id << 24);
        // Delivery mode 110 (STARTUP), level=1, assert=1
        icr_low.write_volatile(0x0000_4600 | (vector as u32));
    }
}

// ---------------------------------------------------------------------------
// Trampoline page construction
// ---------------------------------------------------------------------------

fn prepare_trampoline_page() {
    let page = TRAMPOLINE_PAGE as *mut u8;

    unsafe {
        // Zero the page
        core::ptr::write_bytes(page, 0, 4096);

        // Copy the raw bytes of the naked trampoline function into the page.
        let tramp = ap_trampoline as *const u8;
        let tramp_size = trampoline_size();
        core::ptr::copy_nonoverlapping(tramp, page, tramp_size.min(4096));

        // GDTR descriptor at OFF_GDTR
        let gdt_start = TRAMPOLINE_PAGE + OFF_GDT_START;
        let gdt_limit = (5 * 8 - 1) as u16; // 5 entries
        let gdtr = (TRAMPOLINE_PAGE + OFF_GDTR) as *mut u8;
        gdtr.cast::<u16>().write_unaligned(gdt_limit);
        gdtr.add(2).cast::<u32>().write_unaligned(gdt_start as u32);

        // Temporary GDT entries
        let gdt = gdt_start as *mut u64;
        // 0x00: null
        gdt.add(0).write_unaligned(0);
        // 0x08: 32-bit code (ring 0)
        gdt.add(1).write_unaligned(0x00CF_9A00_0000_FFFF);
        // 0x10: 32-bit data (ring 0)
        gdt.add(2).write_unaligned(0x00CF_9200_0000_FFFF);
        // 0x18: 64-bit code (ring 0)
        gdt.add(3).write_unaligned(0x0020_9A00_0000_0000);
        // 0x20: 64-bit data (ring 0)
        gdt.add(4).write_unaligned(0x0000_9200_0000_0000);

        // Far-jump pointers
        let prot32_ptr = (TRAMPOLINE_PAGE + OFF_PROT32_PTR) as *mut u8;
        prot32_ptr
            .cast::<u16>()
            .write_unaligned(OFF_32BIT_CODE as u16);
        prot32_ptr.add(2).cast::<u16>().write_unaligned(0x0008);

        let long64_ptr = (TRAMPOLINE_PAGE + OFF_LONG64_PTR) as *mut u8;
        long64_ptr
            .cast::<u32>()
            .write_unaligned(OFF_64BIT_CODE as u32);
        long64_ptr.add(4).cast::<u16>().write_unaligned(0x0018);

        // BSP PML4
        let pml4 = crate::memory::virt::bsp_pml4();
        ((TRAMPOLINE_PAGE + OFF_BSP_PML4) as *mut u64).write_unaligned(pml4);

        // AP per-cpu pointer (filled per-AP at runtime)
        ((TRAMPOLINE_PAGE + OFF_AP_PER_CPU_PTR) as *mut u64)
            .write_unaligned(AP_PER_CPU_PTR.load(Ordering::SeqCst));

        // ap_main address
        ((TRAMPOLINE_PAGE + OFF_AP_MAIN_ADDR) as *mut u64)
            .write_unaligned(ap_main as *const () as u64);
    }
}

/// Heuristic size of the trampoline code.
/// The naked function is small (< 512 bytes); we copy a conservative amount.
fn trampoline_size() -> usize {
    512
}
