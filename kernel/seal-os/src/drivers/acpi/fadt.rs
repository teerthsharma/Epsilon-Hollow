// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Fixed ACPI Description Table (FADT / FACP) parser.
//!
//! Extracts PM1a_EVT_BLK, PM1b_EVT_BLK, PM1a_CNT_BLK, PM1b_CNT_BLK,
//! and PM_TMR_BLK required for sleep state transitions.

use core::sync::atomic::{AtomicU16, AtomicU64, Ordering};

use super::rsdp::{SdtHeader, walk_sdt};

/// ACPI Generic Address Structure (12 bytes).
#[repr(C, packed)]
struct GenericAddress {
    address_space_id: u8,
    bit_width: u8,
    bit_offset: u8,
    access_size: u8,
    address: u64,
}

/// Minimal FADT layout containing the fields we need.
/// The full table is 268+ bytes; we only declare offsets up to the
/// X_* Generic Address fields (ACPI 2.0+).
#[repr(C, packed)]
struct Fadt {
    header: SdtHeader,
    _firmware_ctrl: u32,
    _dsdt: u32,
    _reserved0: u8,
    _preferred_pm_profile: u8,
    _sci_int: u16,
    _smi_cmd: u32,
    _acpi_enable: u8,
    _acpi_disable: u8,
    _s4bios_req: u8,
    _pstate_cnt: u8,
    pm1a_evt_blk: u32,
    pm1b_evt_blk: u32,
    pm1a_cnt_blk: u32,
    pm1b_cnt_blk: u32,
    _pm2_cnt_blk: u32,
    pm_tmr_blk: u32,
    _gpe0_blk: u32,
    _gpe1_blk: u32,
    pm1_evt_len: u8,
    pm1_cnt_len: u8,
    _pm2_cnt_len: u8,
    _pm_tmr_len: u8,
    _gpe0_blk_len: u8,
    _gpe1_blk_len: u8,
    _gpe1_base: u8,
    _cst_cnt: u8,
    _p_lvl2_lat: u16,
    _p_lvl3_lat: u16,
    _flush_size: u16,
    _flush_stride: u16,
    _duty_offset: u8,
    _duty_width: u8,
    _day_alrm: u8,
    _mon_alrm: u8,
    _century: u8,
    _iapc_boot_arch: u16,
    _reserved1: u8,
    _flags: u32,
    // ACPI 2.0+ X_* fields at offset 244
    x_pm1a_evt_blk: GenericAddress,
    x_pm1b_evt_blk: GenericAddress,
    x_pm1a_cnt_blk: GenericAddress,
    x_pm1b_cnt_blk: GenericAddress,
    _x_pm2_cnt_blk: GenericAddress,
    x_pm_tmr_blk: GenericAddress,
}

// Global parsed addresses (0 = unknown / unavailable).
static PM1A_EVT_BLK: AtomicU64 = AtomicU64::new(0);
static PM1B_EVT_BLK: AtomicU64 = AtomicU64::new(0);
static PM1A_CNT_BLK: AtomicU64 = AtomicU64::new(0);
static PM1B_CNT_BLK: AtomicU64 = AtomicU64::new(0);
static PM_TMR_BLK: AtomicU64 = AtomicU64::new(0);
static PM1_EVT_LEN: AtomicU16 = AtomicU16::new(0);
static PM1_CNT_LEN: AtomicU16 = AtomicU16::new(0);

/// Hard-coded SLP_TYP values for common x86 firmware.
///
/// A production OS extracts these from the DSDT AML by evaluating the
/// `\_S3` and `\_S5` objects.  The values below (SLP_TYPa=1, SLP_TYPb=1
/// for S3; SLP_TYPa=5, SLP_TYPb=5 for S5) are the QEMU defaults and
/// work on a wide range of real hardware.
const SLP_TYP_S3: u16 = 1;
const SLP_TYP_S5: u16 = 5;

/// Parse the FADT from the RSDP/XSDT chain.
pub fn init(rsdp: u64) {
    let fadt_phys = match walk_sdt(rsdp, b"FACP") {
        Some(addr) => addr,
        None => {
            crate::serial_println!(
                "[ACPI/FADT] FACP not found — ACPI sleep / power-off unavailable"
            );
            return;
        }
    };
    crate::serial_println!("[ACPI/FADT] FACP found @ {:#X}", fadt_phys);

    unsafe {
        let fadt = fadt_phys as *const Fadt;
        let revision = core::ptr::addr_of!((*fadt).header.revision).read_unaligned();

        let read_x_gas = |gas: &GenericAddress| -> u64 {
            let addr = core::ptr::addr_of!(gas.address).read_unaligned();
            addr
        };

        // Prefer 64-bit X_* fields when ACPI 2.0+ and address is non-zero.
        macro_rules! parse_block {
            ($x_field:ident, $leg_field:ident, $global:ident) => {{
                let leg = core::ptr::addr_of!((*fadt).$leg_field).read_unaligned() as u64;
                let x = read_x_gas(&core::ptr::addr_of!((*fadt).$x_field).read_unaligned());
                let val = if revision >= 2 && x != 0 { x } else { leg };
                $global.store(val, Ordering::Relaxed);
            }};
        }

        parse_block!(x_pm1a_evt_blk, pm1a_evt_blk, PM1A_EVT_BLK);
        parse_block!(x_pm1b_evt_blk, pm1b_evt_blk, PM1B_EVT_BLK);
        parse_block!(x_pm1a_cnt_blk, pm1a_cnt_blk, PM1A_CNT_BLK);
        parse_block!(x_pm1b_cnt_blk, pm1b_cnt_blk, PM1B_CNT_BLK);
        parse_block!(x_pm_tmr_blk, pm_tmr_blk, PM_TMR_BLK);

        let pm1_evt_len = core::ptr::addr_of!((*fadt).pm1_evt_len).read_unaligned() as u16;
        let pm1_cnt_len = core::ptr::addr_of!((*fadt).pm1_cnt_len).read_unaligned() as u16;
        PM1_EVT_LEN.store(pm1_evt_len, Ordering::Relaxed);
        PM1_CNT_LEN.store(pm1_cnt_len, Ordering::Relaxed);
    }

    crate::serial_println!(
        "[ACPI/FADT] PM1a_CNT={:#X} PM_TMR={:#X}",
        PM1A_CNT_BLK.load(Ordering::Relaxed),
        PM_TMR_BLK.load(Ordering::Relaxed)
    );
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

fn opt_addr(raw: u64) -> Option<u64> {
    if raw != 0 {
        Some(raw)
    } else {
        None
    }
}

pub fn pm1a_evt_blk() -> Option<u64> {
    opt_addr(PM1A_EVT_BLK.load(Ordering::Relaxed))
}
pub fn pm1b_evt_blk() -> Option<u64> {
    opt_addr(PM1B_EVT_BLK.load(Ordering::Relaxed))
}
pub fn pm1a_cnt_blk() -> Option<u64> {
    opt_addr(PM1A_CNT_BLK.load(Ordering::Relaxed))
}
pub fn pm1b_cnt_blk() -> Option<u64> {
    opt_addr(PM1B_CNT_BLK.load(Ordering::Relaxed))
}
pub fn pm_tmr_blk() -> Option<u64> {
    opt_addr(PM_TMR_BLK.load(Ordering::Relaxed))
}

pub fn pm1_evt_len() -> u16 {
    PM1_EVT_LEN.load(Ordering::Relaxed)
}
pub fn pm1_cnt_len() -> u16 {
    PM1_CNT_LEN.load(Ordering::Relaxed)
}

/// Return the packed (SLP_TYPa, SLP_TYPb) value for a given sleep state.
/// The caller must shift these into the correct bit positions before ORing
/// with SLP_EN.
pub fn slp_typ_for_state(state: u8) -> u16 {
    let typ = match state {
        3 => SLP_TYP_S3,
        5 => SLP_TYP_S5,
        _ => SLP_TYP_S3,
    };
    // Pack into PM1_CNT format: bits 10..12 = SLP_TYPa, bits 12..14 = SLP_TYPb
    // (Each is 3 bits on most hardware; we replicate the same typ value.)
    (typ & 0x7) | ((typ & 0x7) << 3)
}

/// Issue an ACPI soft-off (S5) by writing PM1a_CNT.
pub fn enter_soft_off() {
    if let Some(pm1a_cnt) = pm1a_cnt_blk() {
        let slp_typ = slp_typ_for_state(5);
        let slp_en: u16 = 1 << 13;
        let val = (slp_typ << 10) | slp_en;
        unsafe {
            let mut port = x86_64::instructions::port::Port::<u16>::new(pm1a_cnt as u16);
            port.write(val);
        }
    }
}
