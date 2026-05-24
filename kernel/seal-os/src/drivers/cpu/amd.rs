// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! AMD CPU power-management driver.
//!
//! Detects AMD processors via CPUID, exposes Core Performance Boost (CPB)
//! control, P-state frequency calculation, thermal monitoring, and
//! APERF/MPERF-based utilisation.

use core::arch::x86_64::__cpuid;
use core::sync::atomic::{AtomicBool, Ordering};
use x86_64::registers::model_specific::Msr;

use crate::serial_println;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MSR_K7_HWCR: u32 = 0xC001_0015;
const MSR_PSTATE_CUR_LIMIT: u32 = 0xC001_0061;
const MSR_PSTATE_STATUS: u32 = 0xC001_0063;
const MSR_PSTATE_DEF_0: u32 = 0xC001_0064; // … through 0xC001_006B
const MSR_THM_TCON_CUR_TMP: u32 = 0xC001_00A3;
const MSR_MPERF: u32 = 0xE7;
const MSR_APERF: u32 = 0xE8;

const HWCR_CPB_DISABLE_BIT: u64 = 1 << 25;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static AMD_DETECTED: AtomicBool = AtomicBool::new(false);
static CPB_SUPPORTED: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// CPUID helpers
// ---------------------------------------------------------------------------

/// Read the vendor string from CPUID leaf 0.
fn cpuid_vendor() -> [u8; 12] {
    let cpuid = unsafe { __cpuid(0) };
    let mut buf = [0u8; 12];
    let ebx = cpuid.ebx.to_le_bytes();
    let edx = cpuid.edx.to_le_bytes();
    let ecx = cpuid.ecx.to_le_bytes();
    buf[0..4].copy_from_slice(&ebx);
    buf[4..8].copy_from_slice(&edx);
    buf[8..12].copy_from_slice(&ecx);
    buf
}

/// True when the vendor string reads "AuthenticAMD".
fn is_amd() -> bool {
    cpuid_vendor() == *b"AuthenticAMD"
}

/// Returns (family, model, stepping) from CPUID leaf 1.
fn amd_family_model_stepping() -> (u32, u32, u32) {
    let leaf1 = unsafe { __cpuid(1) };
    let eax = leaf1.eax;
    let stepping = eax & 0xF;
    let base_family = (eax >> 8) & 0xF;
    let ext_family = (eax >> 20) & 0xFF;
    let base_model = (eax >> 4) & 0xF;
    let ext_model = (eax >> 12) & 0xF0;

    let family = if base_family == 0xF {
        base_family + ext_family
    } else {
        base_family
    };
    let model = if base_family == 0xF || base_family == 0x6 {
        base_model | ext_model
    } else {
        base_model
    };
    (family, model, stepping)
}

/// Check whether CPB is supported (CPUID leaf 0x8000_0001, EDX bit 9).
fn cpb_supported() -> bool {
    let leaf = unsafe { __cpuid(0x8000_0001) };
    (leaf.edx >> 9) & 1 == 1
}

// ---------------------------------------------------------------------------
// MSR helpers
// ---------------------------------------------------------------------------

/// Safely read a 64-bit MSR. Returns `None` if the value is all-zeros,
/// which often indicates the MSR is not implemented on this core.
unsafe fn read_msr_safe(addr: u32) -> Option<u64> {
    let val = Msr::new(addr).read();
    if val == 0 {
        None
    } else {
        Some(val)
    }
}

/// Read a 64-bit MSR, logging a warning on zero/failure.
unsafe fn read_msr_warn(addr: u32, name: &str) -> Option<u64> {
    match read_msr_safe(addr) {
        Some(v) => Some(v),
        None => {
            serial_println!("[AMD] Warning: {} MSR ({:#x}) read 0 or unavailable", name, addr);
            None
        }
    }
}

// ---------------------------------------------------------------------------
// P-state helpers
// ---------------------------------------------------------------------------

/// Decode a PSTATE_DEF MSR into frequency (MHz).
///
/// Bits:
///   7:0   = CPUfid
///   15:8  = CPUdid
///   21:16 = CPUvid
///
/// Frequency = 100 MHz * (CPUfid + 0x10) / (2^CPUdid)
fn pstate_to_mhz(def: u64) -> u64 {
    let cpufid = (def & 0xFF) as u64;
    let cpudid = ((def >> 8) & 0xFF) as u64;
    // Avoid division by zero on impossible did values; cap at 7 (128x divider)
    let did = cpudid.min(7);
    let divider = 1u64 << did;
    100 * (cpufid + 0x10) / divider
}

/// Return the current P-state index (0-7), or `None` if unavailable.
fn current_pstate_index() -> Option<u8> {
    unsafe {
        let status = read_msr_safe(MSR_PSTATE_STATUS)?;
        // Bits 2:0 hold the current P-state.
        Some((status & 0x7) as u8)
    }
}

/// Return the frequency (MHz) for a given P-state index.
fn pstate_freq_mhz(index: u8) -> Option<u64> {
    if index > 7 {
        return None;
    }
    unsafe {
        let def = read_msr_safe(MSR_PSTATE_DEF_0 + index as u32)?;
        Some(pstate_to_mhz(def))
    }
}

// ---------------------------------------------------------------------------
// APERF / MPERF
// ---------------------------------------------------------------------------

/// Read the raw APERF and MPERF counters.
///
/// # Safety
/// Caller must ensure RDMSR is safe on this platform.
unsafe fn read_aperf_mperf() -> (u64, u64) {
    let aperf = Msr::new(MSR_APERF).read();
    let mperf = Msr::new(MSR_MPERF).read();
    (aperf, mperf)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// AMD CPU power-management driver singleton.
pub struct AmdCpuDriver;

impl AmdCpuDriver {
    /// Create a new driver handle. Does **not** perform hardware init.
    pub const fn new() -> Self {
        Self
    }

    /// One-time hardware detection and diagnostics.
    ///
    /// # Safety
    /// Must be called on the BSP, after serial output is ready.
    pub unsafe fn init(&self) {
        if !is_amd() {
            serial_println!("[AMD] CPU not detected (vendor mismatch), skipping AMD driver init");
            return;
        }

        AMD_DETECTED.store(true, Ordering::Relaxed);
        let cpb = cpb_supported();
        CPB_SUPPORTED.store(cpb, Ordering::Relaxed);

        let (family, model, stepping) = amd_family_model_stepping();
        serial_println!(
            "[AMD] Detected AMD processor — family={:#x} model={:#x} stepping={}",
            family, model, stepping
        );

        if cpb {
            serial_println!("[AMD] Core Performance Boost (CPB) supported");
            serial_println!("[AMD] CPB currently {}", if self.cpb_enabled() { "enabled" } else { "disabled" });
        } else {
            serial_println!("[AMD] Core Performance Boost (CPB) not supported");
        }

        match self.current_frequency_mhz() {
            Some(freq) => serial_println!("[AMD] Current P-state frequency: {} MHz", freq),
            None => serial_println!("[AMD] Could not determine current P-state frequency"),
        }

        match self.temperature_celsius() {
            Some(t) => serial_println!("[AMD] Current temperature: {} °C", t),
            None => serial_println!("[AMD] Thermal sensor not available or reading zero"),
        }
    }

    /// Returns the current P-state frequency in MHz, if readable.
    pub fn current_frequency_mhz(&self) -> Option<u64> {
        if !AMD_DETECTED.load(Ordering::Relaxed) {
            return None;
        }
        let idx = current_pstate_index()?;
        pstate_freq_mhz(idx)
    }

    /// Returns `true` if CPB is currently enabled (bit 25 == 0).
    pub fn cpb_enabled(&self) -> bool {
        if !AMD_DETECTED.load(Ordering::Relaxed) {
            return false;
        }
        unsafe {
            match read_msr_safe(MSR_K7_HWCR) {
                Some(val) => val & HWCR_CPB_DISABLE_BIT == 0,
                None => false,
            }
        }
    }

    /// Enable (`true`) or disable (`false`) Core Performance Boost.
    pub fn set_cpb(&self, enable: bool) {
        if !AMD_DETECTED.load(Ordering::Relaxed) {
            serial_println!("[AMD] set_cpb ignored: not an AMD CPU");
            return;
        }
        if !CPB_SUPPORTED.load(Ordering::Relaxed) {
            serial_println!("[AMD] set_cpb ignored: CPB not supported");
            return;
        }
        unsafe {
            let mut hwcr = Msr::new(MSR_K7_HWCR).read();
            if enable {
                hwcr &= !HWCR_CPB_DISABLE_BIT; // clear bit 25 => enabled
            } else {
                hwcr |= HWCR_CPB_DISABLE_BIT; // set bit 25 => disabled
            }
            Msr::new(MSR_K7_HWCR).write(hwcr);
        }
        serial_println!("[AMD] CPB {}", if enable { "enabled" } else { "disabled" });
    }

    /// Read the on-die temperature in Celsius, if available.
    ///
    /// Tries MSR `0xC001_00A3` (THM_TCON_CUR_TMP) bits 21:14 first.
    /// Falls back to CPUID leaf 6 thermal sensor presence flag.
    pub fn temperature_celsius(&self) -> Option<u8> {
        if !AMD_DETECTED.load(Ordering::Relaxed) {
            return None;
        }
        // Primary path: AMD THM_TCON_CUR_TMP
        unsafe {
            if let Some(raw) = read_msr_safe(MSR_THM_TCON_CUR_TMP) {
                let temp = ((raw >> 14) & 0xFF) as u8;
                if temp != 0 {
                    return Some(temp);
                }
            }
        }
        // Fallback: CPUID leaf 6, ECX bit 0 = digital temperature sensor supported
        let leaf6 = unsafe { __cpuid(6) };
        if (leaf6.ecx & 1) != 0 {
            // Sensor present but no direct MSR reading on this path;
            // return None because we don't have an accurate value.
            serial_println!("[AMD] Thermal sensor present (CPUID leaf 6) but MSR read 0");
        }
        None
    }
}

/// Read the current utilisation ratio (APERF / MPERF).
///
/// Returns a value in the range `0.0..=1.0`, or `None` if the counters
/// are unavailable.
///
/// # Safety
/// Caller must ensure MSRs are accessible.
pub unsafe fn read_utilization() -> Option<f64> {
    if !AMD_DETECTED.load(Ordering::Relaxed) {
        return None;
    }
    let (aperf, mperf) = read_aperf_mperf();
    if mperf == 0 {
        return None;
    }
    Some(aperf as f64 / mperf as f64)
}

/// Convenience: enable CPB.
pub fn enable_cpb() {
    AmdCpuDriver::new().set_cpb(true);
}

/// Convenience: disable CPB.
pub fn disable_cpb() {
    AmdCpuDriver::new().set_cpb(false);
}

/// Global driver instance.
static DRIVER: AmdCpuDriver = AmdCpuDriver::new();

/// Initialise the AMD CPU driver and print diagnostics.
///
/// # Safety
/// Must be called once on the BSP.
pub unsafe fn init() {
    DRIVER.init();
}
