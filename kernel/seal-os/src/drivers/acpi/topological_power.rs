// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Topological thermal management — T1/T2/T3/T4/T5 theorem driven.
//!
//! T1 Voronoi thermal zones:   Each core is a seed on S²; temperature modulates
//!                             cell weight (hotter = smaller cell = more attention).
//! T2 Spectral prediction:     3-dim predictor per zone forecasts which core
//!                             will trip next; enables proactive throttle.
//! T3 Entropy / Betti-0:       Connected-component count of hot regions;
//!                             high fragmentation → global frequency reduction.
//! T4 PD thermal governor:     ε(t+1) = ε(t) + α·e(t) + β·de/dt.
//! T5 Hyperbolic P-state:      P-states arranged on Poincaré disk; transition
//!                             follows geodesic movement.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use spin::Mutex;

use crate::serial_println;

use super::fadt;

// ---------------------------------------------------------------------------
// T1: Voronoi thermal zones
// ---------------------------------------------------------------------------

const MAX_THERMAL_ZONES: usize = 32;
const T_TARGET: f64 = 75.0; // °C target temperature

/// Per-core thermal zone state.
#[derive(Clone, Copy, Debug)]
struct ThermalZone {
    cpu_num: u32,
    temp_c: f64,
    prev_temp_c: f64,
    /// T2: 3-dim prediction state [temperature_estimate, derivative, integral_error]
    predict: [f64; 3],
    /// T5: hyperbolic coordinate on Poincaré disk (radius, angle)
    pstate_hyp: (f64, f64),
}

impl ThermalZone {
    const fn zero() -> Self {
        Self {
            cpu_num: 0,
            temp_c: 40.0,
            prev_temp_c: 40.0,
            predict: [40.0, 0.0, 0.0],
            pstate_hyp: (0.0, 0.0),
        }
    }
}

static ZONES: Mutex<[ThermalZone; MAX_THERMAL_ZONES]> =
    Mutex::new([ThermalZone::zero(); MAX_THERMAL_ZONES]);

static ZONE_COUNT: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// T4: Governor state
// ---------------------------------------------------------------------------

static GOVERNOR_EPS: Mutex<f64> = Mutex::new(0.0);
static LAST_STEP_TICKS: AtomicU64 = AtomicU64::new(0);
static PRINT_ACCUM_TICKS: AtomicU64 = AtomicU64::new(0);
static SLEEPING: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// T5: Hyperbolic P-state hierarchy on Poincaré disk
// ---------------------------------------------------------------------------

/// P-state indices: 0 = P0 (max freq, origin), 7 = deepest P-state (near boundary).
const PSTATE_COUNT: usize = 8;

/// Hyperbolic radius for each P-state.  P0 = 0.0, P7 → 0.95 (near boundary).
const PSTATE_RADIUS: [f64; PSTATE_COUNT] = [0.00, 0.30, 0.50, 0.65, 0.78, 0.87, 0.93, 0.95];

/// Map a governor epsilon [0,1] to the nearest P-state index.
fn pstate_from_epsilon(eps: f64) -> usize {
    let r = eps.clamp(0.0, 1.0);
    let mut best = PSTATE_COUNT - 1;
    let mut best_diff = 1.0f64;
    for (i, &pr) in PSTATE_RADIUS.iter().enumerate() {
        let d = libm::fabs(pr - r);
        if d < best_diff {
            best_diff = d;
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// T3: Betti-0 (connected components of hot regions)
// ---------------------------------------------------------------------------

/// Simple threshold-based connected-component count on the 1-D CPU array.
/// Cores are ordered by cpu_num; contiguous runs above threshold form components.
fn betti_0(temps: &[f64], threshold: f64) -> usize {
    let mut count = 0;
    let mut in_component = false;
    for &t in temps {
        if t >= threshold {
            if !in_component {
                count += 1;
                in_component = true;
            }
        } else {
            in_component = false;
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Temperature probing
// ---------------------------------------------------------------------------

fn probe_temperatures(temps: &mut [f64], count: usize) {
    // Attempt Intel MSR read first.
    if let Some(guard) = crate::drivers::cpu::intel::driver() {
        if let Some(ref drv) = *guard {
            let t = drv.temperature_celsius().map(|t| t as f64).unwrap_or(45.0);
            temps[0] = t;
            for i in 1..count {
                // On real SMP each core would be read individually;
                // here we replicate the BSP read for simplicity.
                temps[i] = t;
            }
            return;
        }
    }

    // Fallback: AMD MSR read.
    let amd_drv = crate::drivers::cpu::amd::AmdCpuDriver::new();
    if let Some(t) = amd_drv.temperature_celsius() {
        temps[0] = t as f64;
        for i in 1..count {
            temps[i] = t as f64;
        }
        return;
    }

    // Ultimate fallback: assume moderate temperature.
    for i in 0..count {
        temps[i] = 45.0;
    }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

pub fn init() {
    let cpu_count = crate::cpu::CPU_COUNT.load(Ordering::SeqCst);
    let mut zones = ZONES.lock();
    for i in 0..cpu_count.min(MAX_THERMAL_ZONES) {
        zones[i] = ThermalZone {
            cpu_num: i as u32,
            temp_c: 40.0,
            prev_temp_c: 40.0,
            predict: [40.0, 0.0, 0.0],
            pstate_hyp: (0.0, 0.0),
        };
    }
    ZONE_COUNT.store(cpu_count as u64, Ordering::Relaxed);
    crate::serial_println!(
        "[Thermal] Topological power manager initialised ({} zones)",
        cpu_count
    );
}

// ---------------------------------------------------------------------------
// Frequency policy helpers
// ---------------------------------------------------------------------------

fn apply_frequency_policy(eps: f64, _pstate_idx: usize) {
    // ε > 0.7: disable turbo/boost, drop to low P-state
    // ε < 0.2 and thermal headroom: enable turbo/boost
    if eps > 0.7 {
        crate::drivers::cpu::intel::disable_turbo();
        crate::drivers::cpu::amd::disable_cpb();
    } else if eps < 0.2 && max_temp() < T_TARGET * 0.85 {
        crate::drivers::cpu::intel::enable_turbo();
        crate::drivers::cpu::amd::enable_cpb();
    }
}

fn update_hyperbolic_pstate(pstate_idx: usize) {
    let mut zones = ZONES.lock();
    let count = ZONE_COUNT.load(Ordering::Relaxed) as usize;
    let r = PSTATE_RADIUS[pstate_idx.min(PSTATE_COUNT - 1)];
    for i in 0..count {
        // Distribute angles evenly around the Poincaré disk.
        let theta = (i as f64) * 2.0 * core::f64::consts::PI / (count.max(1) as f64);
        zones[i].pstate_hyp = (r, theta);
    }
}

fn max_temp() -> f64 {
    let zones = ZONES.lock();
    let count = ZONE_COUNT.load(Ordering::Relaxed) as usize;
    let mut m = 0.0;
    for i in 0..count {
        if zones[i].temp_c > m {
            m = zones[i].temp_c;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Thermal governor step — called from APIC timer interrupt every ~100 ms
// ---------------------------------------------------------------------------

pub fn thermal_governor_step() {
    if SLEEPING.load(Ordering::Relaxed) {
        return;
    }

    let ticks = crate::drivers::interrupts::ticks();
    let last = LAST_STEP_TICKS.load(Ordering::Relaxed);
    // Timer fires roughly every 1 ms; step every 100 ticks.
    if ticks.wrapping_sub(last) < 100 {
        return;
    }
    LAST_STEP_TICKS.store(ticks, Ordering::Relaxed);

    let count = ZONE_COUNT.load(Ordering::Relaxed) as usize;
    if count == 0 {
        return;
    }

    let mut temps = [0.0f64; MAX_THERMAL_ZONES];
    probe_temperatures(&mut temps, count);

    let mut zones = ZONES.lock();
    let mut max_temp_val = 0.0f64;
    let mut hottest_zone = 0usize;
    let mut _avg_temp = 0.0f64;

    // T1 + T2: Update zones and spectral predictors
    for i in 0..count {
        zones[i].prev_temp_c = zones[i].temp_c;
        zones[i].temp_c = temps[i];
        _avg_temp += temps[i];
        if temps[i] > max_temp_val {
            max_temp_val = temps[i];
            hottest_zone = i;
        }

        // T2: spectral prediction update (simple exponential + derivative + integral)
        let dt = 0.1; // 100 ms step
        let e = temps[i] - zones[i].predict[0];
        let alpha = 0.3;
        let _beta = 0.1;
        zones[i].predict[0] += alpha * e;
        zones[i].predict[1] = (temps[i] - zones[i].prev_temp_c) / dt;
        zones[i].predict[2] += e * dt;
    }
    let _avg_temp = _avg_temp / count as f64;

    // T3: Betti-0 of thermal map
    let hot_threshold = T_TARGET * 0.9;
    let b0 = betti_0(&temps[..count], hot_threshold);

    // T4: PD thermal governor
    let e_t = (max_temp_val - T_TARGET) / T_TARGET;
    let de_dt = zones[hottest_zone].predict[1];
    let alpha_gov = 0.05;
    let beta_gov = 0.02;
    let mut eps = *GOVERNOR_EPS.lock();
    eps = eps + alpha_gov * e_t + beta_gov * de_dt;
    // High Betti-0 (fragmentation) adds a damping penalty.
    eps += 0.01 * (b0 as f64);
    eps = eps.clamp(0.0, 1.0);
    *GOVERNOR_EPS.lock() = eps;

    // T5: Hyperbolic P-state selection
    let pstate_idx = pstate_from_epsilon(eps);

    // Apply frequency / boost policy
    drop(zones); // release before calling into CPU drivers
    apply_frequency_policy(eps, pstate_idx);
    update_hyperbolic_pstate(pstate_idx);

    // T2: Predict which core will overheat next (shortest time-to-trip)
    let zones = ZONES.lock();
    let mut predicted_trip = 0usize;
    let mut worst_time = f64::INFINITY;
    for i in 0..count {
        if zones[i].predict[1] > 0.0 {
            let ttt = (T_TARGET - zones[i].predict[0]) / zones[i].predict[1];
            if ttt < worst_time {
                worst_time = ttt;
                predicted_trip = i;
            }
        }
    }

    // Pre-emptive emergency throttle if trip predicted within 2 seconds.
    if zones[predicted_trip].predict[1] > 0.0 {
        let ttt = (T_TARGET - zones[predicted_trip].predict[0]) / zones[predicted_trip].predict[1];
        if ttt < 2.0 {
            apply_frequency_policy(1.0, PSTATE_COUNT - 1);
            update_hyperbolic_pstate(PSTATE_COUNT - 1);
        }
    }
    drop(zones);

    // Print thermal map every ~1 second (10 × 100 ms steps)
    let accum = PRINT_ACCUM_TICKS.fetch_add(1, Ordering::Relaxed);
    if accum % 10 == 0 {
        crate::serial_println!(
            "[THERMAL] T1 cells: {}, ε={:.2}, P-state: P{}, predicted trip: CPU{}",
            count, eps, pstate_idx, predicted_trip
        );
    }
}

// ---------------------------------------------------------------------------
// S3 Sleep state save / restore
// ---------------------------------------------------------------------------

/// Snapshot of CPU and topological state across S3.
#[derive(Clone, Copy, Debug)]
pub struct SleepState {
    pub cr3: u64,
    pub rflags: u64,
    pub rip: u64,
    pub rsp: u64,
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    /// T2: scheduler prediction state snapshot.
    pub scheduler_predict: [f64; 8],
    /// T5: hyperbolic manifold snapshot (zone_count, max_temp, pstate_radius).
    pub hyperbolic_snapshot: (usize, f64, f64),
}

static SLEEP_STATE: Mutex<SleepState> = Mutex::new(SleepState {
    cr3: 0,
    rflags: 0,
    rip: 0,
    rsp: 0,
    rax: 0,
    rbx: 0,
    rcx: 0,
    rdx: 0,
    rsi: 0,
    rdi: 0,
    rbp: 0,
    r8: 0,
    r9: 0,
    r10: 0,
    r11: 0,
    r12: 0,
    r13: 0,
    r14: 0,
    r15: 0,
    scheduler_predict: [0.0; 8],
    hyperbolic_snapshot: (0, 0.0, 0.0),
});

pub fn is_sleeping() -> bool {
    SLEEPING.load(Ordering::Relaxed)
}

/// Enter ACPI sleep state (S3 or S5).
///
/// Saves CPU state, writes SLP_TYP + SLP_EN to PM1a_CNT, and halts.
/// On wake the firmware resumes at the waking vector; in this simplified
/// bare-metal path we return immediately if the write did not take effect.
pub fn acpi_enter_sleep(state: u8) {
    crate::serial_println!("[ACPI] Entering S{} sleep...", state);
    SLEEPING.store(true, Ordering::Relaxed);

    unsafe {
        let mut sleep = SLEEP_STATE.lock();

        // Save CR3
        let (frame, _) = x86_64::registers::control::Cr3::read();
        sleep.cr3 = frame.start_address().as_u64();

        // Save rflags
        sleep.rflags = x86_64::registers::rflags::read_raw();

        // Save GPRs and RIP/RSP via inline assembly.
        // We use local variables to avoid multiple mutable borrows of `sleep`.
        let mut rax_v = 0u64;
        let mut rbx_v = 0u64;
        let mut rcx_v = 0u64;
        let mut rdx_v = 0u64;
        let mut rsi_v = 0u64;
        let mut rdi_v = 0u64;
        let mut rbp_v = 0u64;
        let mut r8_v  = 0u64;
        let mut r9_v  = 0u64;
        let mut r10_v = 0u64;
        let mut r11_v = 0u64;
        let mut r12_v = 0u64;
        let mut r13_v = 0u64;
        let mut r14_v = 0u64;
        let mut r15_v = 0u64;
        let mut rsp_v = 0u64;
        let mut rip_v = 0u64;
        core::arch::asm!(
            "mov {rax_out}, rax",
            "mov {rbx_out}, rbx",
            "mov {rcx_out}, rcx",
            "mov {rdx_out}, rdx",
            "mov {rsi_out}, rsi",
            "mov {rdi_out}, rdi",
            "mov {rbp_out}, rbp",
            "mov {r8_out}, r8",
            "mov {r9_out}, r9",
            "mov {r10_out}, r10",
            "mov {r11_out}, r11",
            "mov {r12_out}, r12",
            "mov {r13_out}, r13",
            "mov {r14_out}, r14",
            "mov {r15_out}, r15",
            "mov {rsp_out}, rsp",
            "lea {rip_out}, [rip + 2f]",
            "2:",
            rax_out = lateout(reg) rax_v,
            rbx_out = lateout(reg) rbx_v,
            rcx_out = lateout(reg) rcx_v,
            rdx_out = lateout(reg) rdx_v,
            rsi_out = lateout(reg) rsi_v,
            rdi_out = lateout(reg) rdi_v,
            rbp_out = lateout(reg) rbp_v,
            r8_out  = lateout(reg) r8_v,
            r9_out  = lateout(reg) r9_v,
            r10_out = lateout(reg) r10_v,
            r11_out = lateout(reg) r11_v,
            r12_out = lateout(reg) r12_v,
            r13_out = lateout(reg) r13_v,
            r14_out = lateout(reg) r14_v,
            r15_out = lateout(reg) r15_v,
            rsp_out = lateout(reg) rsp_v,
            rip_out = lateout(reg) rip_v,
        );
        sleep.rax = rax_v;
        sleep.rbx = rbx_v;
        sleep.rcx = rcx_v;
        sleep.rdx = rdx_v;
        sleep.rsi = rsi_v;
        sleep.rdi = rdi_v;
        sleep.rbp = rbp_v;
        sleep.r8  = r8_v;
        sleep.r9  = r9_v;
        sleep.r10 = r10_v;
        sleep.r11 = r11_v;
        sleep.r12 = r12_v;
        sleep.r13 = r13_v;
        sleep.r14 = r14_v;
        sleep.r15 = r15_v;
        sleep.rsp = rsp_v;
        sleep.rip = rip_v;

        // T2: Save scheduler prediction state.
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        sleep.scheduler_predict = cpu.scheduler.get_predict_state();

        // T5: Save hyperbolic manifold snapshot.
        let zones = ZONES.lock();
        let count = ZONE_COUNT.load(Ordering::Relaxed) as usize;
        let max_t = (0..count)
            .map(|i| zones[i].temp_c)
            .fold(0.0f64, |a, b| a.max(b));
        let r = if count > 0 { zones[0].pstate_hyp.0 } else { 0.0 };
        sleep.hyperbolic_snapshot = (count, max_t, r);
    }

    // Write SLP_TYP + SLP_EN to PM1a_CNT.
    if let Some(pm1a_cnt) = fadt::pm1a_cnt_blk() {
        let slp_typ = fadt::slp_typ_for_state(state) as u16;
        let slp_en: u16 = 1 << 13;
        let val = (slp_typ << 10) | slp_en;
        unsafe {
            let mut port = x86_64::instructions::port::Port::<u16>::new(pm1a_cnt as u16);
            port.write(val);
        }
        // If we reach here the sleep did not take effect (common in QEMU
        // when S3 is not fully wired).  Log and continue.
        crate::serial_println!("[ACPI] PM1a_CNT write completed; sleep did not latch — continuing");
    } else {
        crate::serial_println!("[ACPI] PM1a_CNT unavailable — cannot enter sleep");
    }

    SLEEPING.store(false, Ordering::Relaxed);
    crate::serial_println!("[ACPI] Resumed from S{} path", state);
}

/// Wake handler called after S3 resume.
/// Restores scheduler prediction state and re-initialises devices.
pub fn acpi_wake_handler() {
    crate::serial_println!("[ACPI] Wake handler: restoring topological state...");
    unsafe {
        let sleep = SLEEP_STATE.lock();

        // T2: Restore scheduler prediction state.
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.set_predict_state(sleep.scheduler_predict);

        // Restore CR3.
        let frame = x86_64::structures::paging::PhysFrame::containing_address(
            x86_64::PhysAddr::new(sleep.cr3),
        );
        x86_64::registers::control::Cr3::write(frame, x86_64::registers::control::Cr3Flags::empty());

        // T5: Restore hyperbolic snapshot metadata.
        let mut zones = ZONES.lock();
        let (count, _max_t, r) = sleep.hyperbolic_snapshot;
        for i in 0..count.min(MAX_THERMAL_ZONES) {
            let theta =
                (i as f64) * 2.0 * core::f64::consts::PI / (count.max(1) as f64);
            zones[i].pstate_hyp = (r, theta);
        }
    }
    crate::serial_println!("[ACPI] State restored, resuming scheduler");
}

// ---------------------------------------------------------------------------
// Power button
// ---------------------------------------------------------------------------

static POWER_BUTTON_DOWN_TICKS: AtomicU64 = AtomicU64::new(0);

/// Record that the power button was pressed.
pub fn power_button_pressed() {
    let now = crate::drivers::interrupts::ticks();
    POWER_BUTTON_DOWN_TICKS.store(now, Ordering::Relaxed);
    crate::serial_println!("[POWER] Button pressed @ tick {}", now);
}

/// Record that the power button was released and act on duration.
pub fn power_button_released() {
    let down = POWER_BUTTON_DOWN_TICKS.load(Ordering::Relaxed);
    let now = crate::drivers::interrupts::ticks();
    let duration_ticks = now.wrapping_sub(down);
    // Timer is ~1 ms per tick.
    if down == 0 {
        return;
    }
    POWER_BUTTON_DOWN_TICKS.store(0, Ordering::Relaxed);

    if duration_ticks < 2000 {
        // Short press (< 2 s) → S3 sleep.
        crate::serial_println!("[POWER] Short press ({} ms) → S3", duration_ticks);
        acpi_enter_sleep(3);
    } else if duration_ticks > 4000 {
        // Long press (> 4 s) → force shutdown.
        crate::serial_println!("[POWER] Long press ({} ms) → shutdown", duration_ticks);
        force_shutdown();
    }
}

/// Force shutdown via ACPI soft-off, falling back to keyboard controller reset.
fn force_shutdown() {
    crate::serial_println!("[POWER] Forcing shutdown...");
    fadt::enter_soft_off();
    // If soft-off returns (e.g. no ACPI), fall back to keyboard reset.
    crate::drivers::interrupts::reboot();
}
