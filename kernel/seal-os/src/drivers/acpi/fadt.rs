// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Fixed ACPI Description Table (FADT / FACP) parser.
//!
//! Extracts PM1a_EVT_BLK, PM1b_EVT_BLK, PM1a_CNT_BLK, PM1b_CNT_BLK,
//! and PM_TMR_BLK required for sleep state transitions.

use core::mem;
use core::sync::atomic::{AtomicU16, AtomicU64, Ordering};

use super::rsdp::{walk_sdt, SdtHeader};

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
    // ACPI 2.0+ X_* fields — offset 116 (this struct's own layout; the real
    // ACPI FADT has them at 208, but this struct only declares the fields it
    // needs, so 116 is where they land here) through 188 (the last field,
    // x_pm_tmr_blk, ends at 188 = size_of::<Fadt>()).
    x_pm1a_evt_blk: GenericAddress,
    x_pm1b_evt_blk: GenericAddress,
    x_pm1a_cnt_blk: GenericAddress,
    x_pm1b_cnt_blk: GenericAddress,
    _x_pm2_cnt_blk: GenericAddress,
    x_pm_tmr_blk: GenericAddress,
}

/// Whether it's safe to read this struct's `X_*` Generic Address fields.
///
/// An ACPI 1.0 FADT is 116 bytes and does not have them at all — offsets
/// 116..size_of::<Fadt>() (188) don't exist in the real table no matter what
/// `revision` claims. Gate on *both*: `revision` alone is what the table
/// says about itself, `header_length` is what the table's own declared
/// extent can actually support. A table lying about its revision (claims
/// >= 2 with a 116-byte length) is still refused.
fn x_fields_available(revision: u8, header_length: usize) -> bool {
    revision >= 2 && header_length >= mem::size_of::<Fadt>()
}

/// Offset immediately past the last legacy field this driver reads
/// (`pm1_cnt_len`), derived from the struct's own layout rather than a
/// hand-written number.
const LEGACY_FIELDS_END: usize = mem::offset_of!(Fadt, pm1_cnt_len) + mem::size_of::<u8>();

/// Whether it's safe to read this struct's legacy 32-bit fields
/// (`pm1a_evt_blk` through `pm1_cnt_len`).
///
/// Unlike the `X_*` fields these exist in every FADT revision — ACPI 1.0
/// included — so there is no `revision` gate, only a length one:
/// `walk_sdt` validates the table's checksum over exactly `header.length`
/// declared bytes (`SdtHeader::is_valid`, rsdp.rs), so a table shorter than
/// `LEGACY_FIELDS_END` doesn't actually contain them no matter what this
/// struct's `repr(C, packed)` layout implies is there.
fn legacy_fields_available(header_length: usize) -> bool {
    header_length >= LEGACY_FIELDS_END
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
        let header_length =
            core::ptr::addr_of!((*fadt).header.length).read_unaligned() as usize;

        // `walk_sdt` already validated this table's checksum over exactly
        // `header_length` declared bytes (SdtHeader::is_valid, rsdp.rs) —
        // reading anything within that extent is safe, anything past it
        // is not. A table too short to hold even the legacy 32-bit PM
        // blocks is degraded the same way a missing FACP is: log and back
        // out before touching any field, rather than reading past the
        // table's validated extent.
        if !legacy_fields_available(header_length) {
            crate::serial_println!(
                "[ACPI/FADT] FACP length {} too short for legacy PM blocks (need {}) — ACPI sleep / power-off unavailable",
                header_length,
                LEGACY_FIELDS_END
            );
            return;
        }

        // The X_* fields at offsets 116..188 are outside that extent for an
        // ACPI 1.0 (116-byte) table even though the legacy fields above are
        // covered.
        let has_x_fields = x_fields_available(revision, header_length);

        let read_x_gas =
            |gas: &GenericAddress| -> u64 { core::ptr::addr_of!(gas.address).read_unaligned() };

        // Prefer 64-bit X_* fields when ACPI 2.0+, the table is long enough
        // to actually contain them, and the address is non-zero.
        macro_rules! parse_block {
            ($x_field:ident, $leg_field:ident, $global:ident) => {{
                let leg = core::ptr::addr_of!((*fadt).$leg_field).read_unaligned() as u64;
                let val = if has_x_fields {
                    let x = read_x_gas(&core::ptr::addr_of!((*fadt).$x_field).read_unaligned());
                    if x != 0 {
                        x
                    } else {
                        leg
                    }
                } else {
                    leg
                };
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

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;

    const ACPI_1_0_LEN: usize = 116; // real ACPI 1.0 FADT size — no X_* fields exist
    const FULL_LEN: usize = mem::size_of::<Fadt>(); // 188 — covers all X_* fields declared here

    /// RED (was): `parse_block!` read the `X_*` Generic Address fields at
    /// offsets 116..188 unconditionally, gated only on `revision >= 2`
    /// *after* the read already happened — for an ACPI 1.0 FADT
    /// (`header_length == 116`) the last block read offsets 176..188, up to
    /// 72 bytes past the table's real 116-byte extent. GREEN:
    /// `x_fields_available` must be false whenever `header_length` doesn't
    /// cover the fields, regardless of what `revision` claims.
    fn test_acpi_1_0_length_rejects_x_fields() -> TestResult {
        test_assert!(
            !x_fields_available(0, ACPI_1_0_LEN),
            "ACPI 1.0 FADT (revision 0, length 116) must not read X_* fields"
        );
        test_assert!(
            !x_fields_available(1, ACPI_1_0_LEN),
            "ACPI 1.0 FADT (revision 1, length 116) must not read X_* fields"
        );
        TestResult::Pass
    }

    /// A table that lies about its revision (claims >= 2) but is still
    /// only 116 bytes long must not have its X_* fields read either — the
    /// bytes at offset 116+ simply aren't part of this table.
    fn test_revision_lie_still_gated_by_length() -> TestResult {
        test_assert!(
            !x_fields_available(2, ACPI_1_0_LEN),
            "revision >= 2 alone must not be enough; length must cover the X_* fields too"
        );
        TestResult::Pass
    }

    /// Positive control: a genuine ACPI 2.0+ FADT whose declared length
    /// covers the X_* fields must read them.
    fn test_full_length_v2_allows_x_fields() -> TestResult {
        test_assert!(
            x_fields_available(2, FULL_LEN),
            "ACPI 2.0+ FADT with a length covering the X_* fields must use them"
        );
        TestResult::Pass
    }

    /// Revision 1 with a full-length table must still prefer the legacy
    /// fields — `revision` is the other half of the gate, not just length.
    fn test_v1_with_full_length_still_uses_legacy() -> TestResult {
        test_assert!(
            !x_fields_available(1, FULL_LEN),
            "revision < 2 must not read X_* fields even if length would allow it"
        );
        TestResult::Pass
    }

    /// RED (was): `parse_block!` read `pm1a_evt_blk` .. `pm_tmr_blk`, and
    /// `init` read `pm1_evt_len`/`pm1_cnt_len` right after it, all
    /// unconditionally — with no check that `header.length` covered offsets
    /// up to `LEGACY_FIELDS_END` (90). A table truncated below that (e.g. a
    /// bare 36-byte `SdtHeader` with nothing after it, which still passes
    /// `SdtHeader::is_valid`'s floor) went straight through to those reads.
    /// GREEN: `legacy_fields_available` must be false whenever `header_length`
    /// doesn't reach `LEGACY_FIELDS_END`.
    fn test_short_length_rejects_legacy_fields() -> TestResult {
        test_assert!(
            !legacy_fields_available(mem::size_of::<SdtHeader>()),
            "a header-only (36-byte) FADT must not read the legacy PM blocks"
        );
        test_assert!(
            !legacy_fields_available(LEGACY_FIELDS_END - 1),
            "one byte short of LEGACY_FIELDS_END must still be rejected"
        );
        TestResult::Pass
    }

    /// Boundary control: a length that exactly reaches the end of the last
    /// legacy field read (`pm1_cnt_len`) must be accepted — the check is
    /// `>=`, not `>`.
    fn test_boundary_length_allows_legacy_fields() -> TestResult {
        test_assert!(
            legacy_fields_available(LEGACY_FIELDS_END),
            "a length exactly covering the legacy PM blocks must be accepted"
        );
        TestResult::Pass
    }

    /// Positive control: both a real ACPI 1.0 FADT (116 bytes) and a full
    /// ACPI 2.0+ FADT (188 bytes) comfortably cover the legacy fields —
    /// unlike the `X_*` fields, there is no revision gate here.
    fn test_real_world_lengths_allow_legacy_fields() -> TestResult {
        test_assert!(
            legacy_fields_available(ACPI_1_0_LEN),
            "an ACPI 1.0 FADT must still expose its legacy PM blocks"
        );
        test_assert!(
            legacy_fields_available(FULL_LEN),
            "an ACPI 2.0+ FADT must still expose its legacy PM blocks"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "acpi::fadt::acpi_1_0_length_rejects_x_fields",
            test_acpi_1_0_length_rejects_x_fields,
        );
        crate::testing::register_test(
            "acpi::fadt::revision_lie_still_gated_by_length",
            test_revision_lie_still_gated_by_length,
        );
        crate::testing::register_test(
            "acpi::fadt::full_length_v2_allows_x_fields",
            test_full_length_v2_allows_x_fields,
        );
        crate::testing::register_test(
            "acpi::fadt::v1_with_full_length_still_uses_legacy",
            test_v1_with_full_length_still_uses_legacy,
        );
        crate::testing::register_test(
            "acpi::fadt::short_length_rejects_legacy_fields",
            test_short_length_rejects_legacy_fields,
        );
        crate::testing::register_test(
            "acpi::fadt::boundary_length_allows_legacy_fields",
            test_boundary_length_allows_legacy_fields,
        );
        crate::testing::register_test(
            "acpi::fadt::real_world_lengths_allow_legacy_fields",
            test_real_world_lengths_allow_legacy_fields,
        );
    }
}
