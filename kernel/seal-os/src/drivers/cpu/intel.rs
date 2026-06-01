// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Intel CPU power management driver.
//!
//! Provides P-state monitoring, thermal readout, Turbo Boost control,
//! and MPERF/APERF-based utilization estimation.

use crate::serial_println;
use core::arch::x86_64::__cpuid;
use core::sync::atomic::{AtomicU8, Ordering};
use x86_64::registers::model_specific::Msr;

// ---------------------------------------------------------------------------
// MSR addresses
// ---------------------------------------------------------------------------

const IA32_MISC_ENABLE: u32 = 0x1A0;
const IA32_PERF_STATUS: u32 = 0x198;
const IA32_PERF_CTL: u32 = 0x199;
const IA32_THERM_STATUS: u32 = 0x19C;
const IA32_MPERF: u32 = 0xE7;
const IA32_APERF: u32 = 0xE8;

// ---------------------------------------------------------------------------
// Driver state
// ---------------------------------------------------------------------------

/// Intel CPU power-management driver.
pub struct IntelCpuDriver {
    family: u8,
    model: u8,
    stepping: u8,
    max_p_state: u16,
}

static INTEL_DRIVER: spin::Mutex<Option<IntelCpuDriver>> = spin::Mutex::new(None);
static TURBO_REQUEST: AtomicU8 = AtomicU8::new(0);

impl IntelCpuDriver {
    /// Detect an Intel CPU and initialize the driver.
    ///
    /// Returns `None` if the CPU vendor string is not `"GenuineIntel"`.
    pub fn init() -> Option<Self> {
        // ----- CPUID leaf 0: vendor string ---------------------------------
        let cpuid0 = __cpuid(0);
        let mut vendor = [0u8; 12];
        vendor[0..4].copy_from_slice(&cpuid0.ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&cpuid0.edx.to_le_bytes());
        vendor[8..12].copy_from_slice(&cpuid0.ecx.to_le_bytes());

        if &vendor != b"GenuineIntel" {
            return None;
        }

        // ----- CPUID leaf 1: family/model/stepping -------------------------
        let cpuid1 = __cpuid(1);
        let stepping = (cpuid1.eax & 0xF) as u8;
        let model = ((cpuid1.eax >> 4) & 0xF) as u8;
        let family = ((cpuid1.eax >> 8) & 0xF) as u8;
        let ext_model = ((cpuid1.eax >> 16) & 0xF) as u8;
        let ext_family = ((cpuid1.eax >> 20) & 0xFF) as u8;

        let display_family = if family == 0x0F {
            family + ext_family
        } else {
            family
        };
        let display_model = if family == 0x06 || family == 0x0F {
            (ext_model << 4) | model
        } else {
            model
        };

        // ----- Initial MSR reads -------------------------------------------
        let max_p_state = read_msr(IA32_PERF_STATUS)
            .map(|v| (v & 0xFFFF) as u16)
            .unwrap_or(0);

        let driver = IntelCpuDriver {
            family: display_family,
            model: display_model,
            stepping,
            max_p_state,
        };

        // ----- Diagnostics -------------------------------------------------
        serial_println!(
            "[IntelCpu] Detected Intel CPU family={:#02x} model={:#02x} stepping={}",
            driver.family,
            driver.model,
            driver.stepping,
        );
        serial_println!(
            "[IntelCpu] Max P-state: {} ({} MHz), Turbo: {}",
            driver.max_p_state,
            driver.max_p_state * 100,
            if driver.turbo_enabled() {
                "enabled"
            } else {
                "disabled"
            }
        );

        if let Some(therm) = read_msr(IA32_THERM_STATUS) {
            let valid = (therm >> 31) & 1 != 0;
            let trip = therm & 1 != 0;
            serial_println!(
                "[IntelCpu] Thermal status: valid={}, thermal-trip={}",
                valid,
                trip
            );
        }

        Some(driver)
    }

    /// Current core frequency in MHz (from IA32_PERF_STATUS bits 15:0 × 100).
    pub fn current_frequency_mhz(&self) -> u16 {
        read_msr(IA32_PERF_STATUS)
            .map(|v| (v & 0xFFFF) as u16)
            .unwrap_or(0)
            .saturating_mul(100)
    }

    /// Target core frequency in MHz (from IA32_PERF_CTL bits 15:0 × 100).
    pub fn target_frequency_mhz(&self) -> u16 {
        read_msr(IA32_PERF_CTL)
            .map(|v| (v & 0xFFFF) as u16)
            .unwrap_or(0)
            .saturating_mul(100)
    }

    /// Core temperature in °C, if the digital thermal sensor readout is valid.
    ///
    /// TjMax is assumed to be 100 °C, which is true for the vast majority of
    /// modern Intel parts.  `readout` in IA32_THERM_STATUS[22:16] is the number
    /// of degrees below TjMax.
    pub fn temperature_celsius(&self) -> Option<u8> {
        read_msr(IA32_THERM_STATUS).and_then(|v| {
            let valid = (v >> 31) & 1 != 0;
            if !valid {
                return None;
            }
            let readout = ((v >> 16) & 0x7F) as u8;
            let tj_max = 100u8;
            Some(tj_max.saturating_sub(readout))
        })
    }

    /// Returns `true` if Turbo Boost is enabled (IA32_MISC_ENABLE bit 38 == 0).
    pub fn turbo_enabled(&self) -> bool {
        read_msr(IA32_MISC_ENABLE)
            .map(|v| (v >> 38) & 1 == 0)
            .unwrap_or(false)
    }

    /// Enable or disable Turbo Boost by flipping IA32_MISC_ENABLE bit 38.
    pub fn set_turbo(&self, enabled: bool) {
        let requested = if enabled { 2 } else { 1 };
        if TURBO_REQUEST.load(Ordering::Relaxed) == requested {
            return;
        }
        if let Some(mut v) = read_msr(IA32_MISC_ENABLE) {
            let currently_enabled = (v >> 38) & 1 == 0;
            if currently_enabled == enabled {
                TURBO_REQUEST.store(requested, Ordering::Relaxed);
                return;
            }
            if enabled {
                v &= !(1u64 << 38);
            } else {
                v |= 1u64 << 38;
            }
            write_msr(IA32_MISC_ENABLE, v);
            serial_println!(
                "[IntelCpu] Turbo {}",
                if enabled { "enabled" } else { "disabled" }
            );
        }
        TURBO_REQUEST.store(requested, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Standalone helpers
// ---------------------------------------------------------------------------

/// Read the core utilization as a percentage using MPERF/APERF.
///
/// APERF counts at the actual core frequency; MPERF counts at a fixed
/// TSC rate.  Their ratio (×100) gives the average utilization over the
/// measurement window.
///
/// Returns `None` if the MSRs cannot be read or MPERF did not advance.
pub fn read_utilization() -> Option<u8> {
    let mperf_before = read_msr(IA32_MPERF)?;
    let aperf_before = read_msr(IA32_APERF)?;

    // Short busy-wait to give the counters time to diverge.
    for _ in 0..1_000_000 {
        core::hint::spin_loop();
    }

    let mperf_after = read_msr(IA32_MPERF)?;
    let aperf_after = read_msr(IA32_APERF)?;

    let mperf_delta = mperf_after.saturating_sub(mperf_before);
    let aperf_delta = aperf_after.saturating_sub(aperf_before);

    if mperf_delta == 0 {
        return None;
    }

    let ratio = (aperf_delta * 100) / mperf_delta;
    Some(ratio.min(255) as u8)
}

/// Enable Turbo Boost (clear bit 38 of IA32_MISC_ENABLE).
pub fn enable_turbo() {
    if let Some(ref driver) = *INTEL_DRIVER.lock() {
        driver.set_turbo(true);
    }
}

/// Disable Turbo Boost (set bit 38 of IA32_MISC_ENABLE).
pub fn disable_turbo() {
    if let Some(ref driver) = *INTEL_DRIVER.lock() {
        driver.set_turbo(false);
    }
}

// ---------------------------------------------------------------------------
// MSR wrappers
// ---------------------------------------------------------------------------

fn read_msr(addr: u32) -> Option<u64> {
    let msr = Msr::new(addr);
    unsafe { Some(msr.read()) }
}

fn write_msr(addr: u32, value: u64) {
    let mut msr = Msr::new(addr);
    unsafe { msr.write(value) }
}

// ---------------------------------------------------------------------------
// Global init
// ---------------------------------------------------------------------------

/// Probe for an Intel CPU and, if found, initialize the power-management driver.
///
/// Safe to call on any x86_64 CPU — non-Intel CPUs are silently skipped.
pub fn init() {
    serial_println!("[IntelCpu] Probing Intel CPU power management...");
    if let Some(driver) = IntelCpuDriver::init() {
        *INTEL_DRIVER.lock() = Some(driver);
        serial_println!("[IntelCpu] Driver initialized");
    } else {
        serial_println!("[IntelCpu] No Intel CPU detected, skipping");
    }
}

/// Access the global Intel CPU driver, if initialized.
pub fn driver() -> Option<spin::MutexGuard<'static, Option<IntelCpuDriver>>> {
    Some(INTEL_DRIVER.lock())
}
