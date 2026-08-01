// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Per-feature security probes.
//!
//! `[SECURITY] hardening proof` lumps KPTI and SMAP/SMEP into one verdict, so a
//! regression in one mitigation is indistinguishable from a regression in the
//! other and mitigations that are merely *compiled in* look identical to
//! mitigations that are *live on this CPU*. This module splits them: every
//! mitigation reports its own supported/active pair, and every "active" bit is
//! read back out of CR0/CR4/EFER, CPUID, the live page tables, or live kernel
//! state — never from a `cfg!`, except where the mitigation genuinely has no
//! hardware bit, in which case the probe kind says so.
//!
//! Probe kinds:
//! - `cpuid+cr4`  — CPUID support bit plus the live CR4 enable bit.
//! - `cpuid+efer` — CPUID support bit plus the live EFER enable bit.
//! - `cr0`        — a live CR0 bit.
//! - `runtime-cr3`, `runtime-entropy`, `runtime-pagewalk`, `runtime-guardband`,
//!   `runtime-vfs` — measured from live kernel state.
//! - `compile-time` — no hardware bit exists; the field reports what was built,
//!   and must not be read as a hardware guarantee.

use x86_64::registers::control::{Cr0, Cr4};
use x86_64::structures::paging::{PageTable, PageTableFlags};

/// Upper bound on leaf entries the W^X page walk will examine, so a corrupt or
/// unexpectedly dense kernel page table cannot stall the boot proof.
const WX_SCAN_BUDGET: u64 = 1 << 16;

/// A single mitigation's measured state.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FeatureState {
    /// Hardware/toolchain reports the mitigation exists.
    pub supported: bool,
    /// The mitigation is enabled right now on this CPU.
    pub active: bool,
    /// How `active` was determined.
    pub probe: &'static str,
}

impl FeatureState {
    /// A mitigation is acceptable when it is active, or genuinely absent from
    /// this CPU. Silently supported-but-off is the failure this catches.
    pub fn ok(self) -> bool {
        self.active || !self.supported
    }
}

fn cpuid7_ebx() -> u32 {
    core::arch::x86_64::__cpuid_count(7, 0).ebx
}

fn cpuid7_edx() -> u32 {
    core::arch::x86_64::__cpuid_count(7, 0).edx
}

/// Read the live EFER MSR (0xC000_0080).
fn efer_raw() -> u64 {
    // SAFETY: 0xC000_0080 (IA32_EFER) is architecturally present on every
    // x86_64 CPU and this is a read of a control MSR with no side effects.
    unsafe { x86_64::registers::model_specific::Msr::new(0xC000_0080).read() }
}

/// SMEP — supervisor may not execute user pages. CPUID.7:0.EBX[7] / CR4[20].
pub fn smep() -> FeatureState {
    FeatureState {
        supported: cpuid7_ebx() & (1 << 7) != 0,
        active: Cr4::read_raw() & (1 << 20) != 0,
        probe: "cpuid+cr4",
    }
}

/// SMAP — supervisor may not read/write user pages. CPUID.7:0.EBX[20] / CR4[21].
pub fn smap() -> FeatureState {
    FeatureState {
        supported: cpuid7_ebx() & (1 << 20) != 0,
        active: Cr4::read_raw() & (1 << 21) != 0,
        probe: "cpuid+cr4",
    }
}

/// NX — no-execute page bit. CPUID.8000_0001:EDX[20] / EFER.NXE (bit 11).
pub fn nx() -> FeatureState {
    let max_ext = core::arch::x86_64::__cpuid(0x8000_0000).eax;
    let supported =
        max_ext >= 0x8000_0001 && core::arch::x86_64::__cpuid(0x8000_0001).edx & (1 << 20) != 0;
    FeatureState {
        supported,
        active: efer_raw() & (1 << 11) != 0,
        probe: "cpuid+efer",
    }
}

/// CR0.WP — supervisor writes honour the page read-only bit. Without it, W^X
/// on kernel pages is unenforceable regardless of how the tables are marked.
pub fn write_protect() -> FeatureState {
    FeatureState {
        supported: true,
        active: Cr0::read_raw() & (1 << 16) != 0,
        probe: "cr0",
    }
}

/// KPTI — distinct kernel/user page-table roots with the expected shape.
pub fn kpti() -> FeatureState {
    FeatureState {
        supported: true,
        active: super::kpti::runtime_proof().passes(),
        probe: "runtime-cr3",
    }
}

/// KASLR — kernel mapping-window randomisation drawn from hardware entropy.
pub fn kaslr() -> FeatureState {
    FeatureState {
        supported: true,
        active: super::kaslr::is_active(),
        probe: "runtime-entropy",
    }
}

/// Retpoline — Spectre-v2 thunks. There is no control register for "the
/// compiler emitted thunks", so instead of trusting a `cfg!` this reads the
/// thunk's own machine code back out of the live image and checks it is still
/// the retpoline sequence. The CPU's IBRS/IBPB support (CPUID.7:0.EDX[26]) is
/// reported alongside as `supported`, so the gate can tell a thunked build on a
/// bare CPU from one on a CPU with microcode mitigations available.
pub fn retpoline() -> FeatureState {
    FeatureState {
        supported: cpuid7_edx() & (1 << 26) != 0,
        active: thunk_bytes_intact(),
        probe: "runtime-thunk-bytes",
    }
}

/// `mov [rsp], rax` followed by `ret` — the tail of every retpoline thunk. If
/// the thunk were optimised away or replaced by a plain indirect jump, this
/// sequence would not be there.
const THUNK_TAIL: [u8; 5] = [0x48, 0x89, 0x04, 0x24, 0xc3];

/// Read the first bytes of the RAX retpoline thunk out of the live image and
/// confirm they are still `call .+0` followed by the thunk tail.
pub fn thunk_bytes_intact() -> bool {
    let entry = super::retpoline::indirect_branch_thunk as *const () as *const u8;
    // SAFETY: `entry` is the address of a `#[naked]` function in the kernel's
    // own text, which is mapped readable, and only the first 16 bytes of that
    // function are read. Nothing is written and no reference outlives the call.
    let window = unsafe { core::slice::from_raw_parts(entry, 16) };
    window[0] == 0xe8 && thunk_window_matches(window)
}

/// Does a byte window contain the retpoline thunk tail?
pub fn thunk_window_matches(window: &[u8]) -> bool {
    window.windows(THUNK_TAIL.len()).any(|w| w == THUNK_TAIL)
}

/// Kernel stack guard band — 16 KiB of zeroed bytes sitting immediately below
/// each per-CPU kernel stack. Stacks grow down, so any byte written there is a
/// stack overflow that already happened. There is no `-Z stack-protector` in
/// this build, so this band is the only stack protection that exists.
pub fn stack_guard() -> FeatureState {
    FeatureState {
        supported: true,
        active: guard_dirty_bytes(current_guard_band()) == 0,
        probe: "runtime-guardband",
    }
}

/// Number of bytes in a guard band that are no longer zero.
pub fn guard_dirty_bytes(guard: &[u8]) -> usize {
    guard.iter().filter(|b| **b != 0).count()
}

fn current_guard_band() -> &'static [u8] {
    // SAFETY: `this_cpu` resolves the per-CPU block through GS base, which the
    // BSP and every AP install before reaching any security probe. The returned
    // reference is reborrowed immediately as a shared slice of the guard band,
    // which no other code writes to by design.
    unsafe { &crate::cpu::this_cpu().kernel_stack_guard[..] }
}

/// Audit-log flush — measured by reading `/var/log/audit.log` back through the
/// VFS, not by trusting the in-memory buffer.
pub fn audit_log() -> FeatureState {
    let readable = crate::fs::vfs::with_vfs(|vfs| {
        let handle = vfs.lookup_follow("/var/log/audit.log").ok()?;
        Some(vfs.stat(handle).ok()?.size > 0)
    })
    .unwrap_or(false);
    FeatureState {
        supported: true,
        active: readable,
        probe: "runtime-vfs",
    }
}

/// W^X over the kernel image alias — count leaf pages in the kernel's
/// higher-half PML4 slot that are both writable and executable.
///
/// Returns `(violations, pages_scanned)`. A page is a violation when it is
/// PRESENT and WRITABLE and does not carry NO_EXECUTE.
pub fn wx_violations() -> (u64, u64) {
    let pml4_virt = crate::memory::virt::current_pml4_virt().as_u64();
    if pml4_virt == 0 {
        return (0, 0);
    }
    let mut violations = 0u64;
    let mut scanned = 0u64;

    // SAFETY: `current_pml4_virt` returns the identity-mapped address of the
    // live PML4, and every table address reached from it is a physical frame
    // below the 16 GiB identity map. All accesses here are aligned read-only
    // loads of 512-entry tables, bounded by WX_SCAN_BUDGET leaf visits.
    unsafe {
        let pml4 = &*(pml4_virt as *const PageTable);
        let pml4_entry = &pml4[511];
        if pml4_entry.is_unused() {
            return (0, 0);
        }
        let pdpt = &*(pml4_entry.addr().as_u64() as *const PageTable);
        for pdpt_idx in 0..512 {
            let pd_entry = &pdpt[pdpt_idx];
            if pd_entry.is_unused() {
                continue;
            }
            if pd_entry.flags().contains(PageTableFlags::HUGE_PAGE) {
                scanned += 1;
                if is_wx(pd_entry.flags()) {
                    violations += 1;
                }
                continue;
            }
            let pd = &*(pd_entry.addr().as_u64() as *const PageTable);
            for pd_idx in 0..512 {
                let pt_entry = &pd[pd_idx];
                if pt_entry.is_unused() {
                    continue;
                }
                if pt_entry.flags().contains(PageTableFlags::HUGE_PAGE) {
                    scanned += 1;
                    if is_wx(pt_entry.flags()) {
                        violations += 1;
                    }
                    continue;
                }
                let pt = &*(pt_entry.addr().as_u64() as *const PageTable);
                for pt_idx in 0..512 {
                    let page = &pt[pt_idx];
                    if page.is_unused() {
                        continue;
                    }
                    scanned += 1;
                    if is_wx(page.flags()) {
                        violations += 1;
                    }
                    if scanned >= WX_SCAN_BUDGET {
                        return (violations, scanned);
                    }
                }
            }
        }
    }

    (violations, scanned)
}

/// A mapping is a W^X violation when it is present, writable, and executable.
pub fn is_wx(flags: PageTableFlags) -> bool {
    flags.contains(PageTableFlags::PRESENT)
        && flags.contains(PageTableFlags::WRITABLE)
        && !flags.contains(PageTableFlags::NO_EXECUTE)
}

/// `[SECURITY-FEATURES] proof version=1 ...` — one supported/active pair per
/// mitigation, each read back from live CPU or kernel state.
///
/// `result=pass` requires every hardware-probed mitigation that the CPU
/// supports to be active, plus KPTI, KASLR and the stack guard band. `wx` is
/// measured and reported but not gated (`wx_enforced=0`): the kernel image
/// alias is currently mapped writable+executable, and pretending otherwise by
/// re-flagging an alias nothing executes from would be a fake win.
pub fn security_feature_proof_line() -> alloc::string::String {
    let kpti = kpti();
    let smep = smep();
    let smap = smap();
    let nx = nx();
    let wp = write_protect();
    let retpoline = retpoline();
    let kaslr = kaslr();
    let guard = stack_guard();
    let audit = audit_log();
    let (wx_violations, wx_scanned) = wx_violations();

    let pass = kpti.active
        && kaslr.active
        && guard.active
        && wp.active
        && smep.ok()
        && smap.ok()
        && nx.ok()
        && retpoline.active;

    alloc::format!(
        "[SECURITY-FEATURES] proof version=1 kpti={} kpti_probe={} smep_supported={} smep={} smep_probe={} smap_supported={} smap={} smap_probe={} nx_supported={} nx={} nx_probe={} wp={} wp_probe={} retpoline={} retpoline_ibpb_supported={} retpoline_probe={} kaslr={} kaslr_bits={} kaslr_probe={} wx={} wx_violations={} wx_pages_scanned={} wx_scope=kernel-alias wx_enforced=0 wx_probe=runtime-pagewalk stackguard={} stackguard_dirty={} stackguard_probe={} audit={} audit_probe={} cr0={:#x} cr4={:#x} efer={:#x} result={}",
        b(kpti.active),
        kpti.probe,
        b(smep.supported),
        b(smep.active),
        smep.probe,
        b(smap.supported),
        b(smap.active),
        smap.probe,
        b(nx.supported),
        b(nx.active),
        nx.probe,
        b(wp.active),
        wp.probe,
        b(retpoline.active),
        b(retpoline.supported),
        retpoline.probe,
        b(kaslr.active),
        super::kaslr::total_entropy_bits(),
        kaslr.probe,
        b(wx_violations == 0),
        wx_violations,
        wx_scanned,
        b(guard.active),
        guard_dirty_bytes(current_guard_band()),
        guard.probe,
        b(audit.active),
        audit.probe,
        Cr0::read_raw(),
        Cr4::read_raw(),
        efer_raw(),
        if pass { "pass" } else { "fail" }
    )
}

fn b(v: bool) -> u32 {
    if v {
        1
    } else {
        0
    }
}

/// Emit the per-feature proof line to the serial console.
pub fn emit_security_feature_proof() {
    crate::serial_println!("{}", security_feature_proof_line());
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// Each probe must read the control-register bit it claims to read, not a
    /// neighbouring one. Compare against an independent decode of the same
    /// register.
    fn test_probes_read_the_right_bits() -> TestResult {
        let cr4 = Cr4::read_raw();
        test_assert_eq!(smep().active, cr4 & 0x0010_0000 != 0); // CR4 bit 20
        test_assert_eq!(smap().active, cr4 & 0x0020_0000 != 0); // CR4 bit 21
        test_assert_eq!(write_protect().active, Cr0::read_raw() & 0x0001_0000 != 0); // CR0 bit 16
        test_assert_eq!(nx().active, efer_raw() & 0x0000_0800 != 0); // EFER bit 11
        let ebx = cpuid7_ebx();
        test_assert_eq!(smep().supported, ebx & 0x0000_0080 != 0); // CPUID.7:0.EBX bit 7
        test_assert_eq!(smap().supported, ebx & 0x0010_0000 != 0); // CPUID.7:0.EBX bit 20
        TestResult::Pass
    }

    /// A supported-but-disabled mitigation must not be reported as acceptable —
    /// this is the whole point of splitting supported from active.
    fn test_supported_but_off_is_not_ok() -> TestResult {
        let off = FeatureState {
            supported: true,
            active: false,
            probe: "cpuid+cr4",
        };
        test_assert!(!off.ok(), "supported-but-off must not pass");
        let absent = FeatureState {
            supported: false,
            active: false,
            probe: "cpuid+cr4",
        };
        test_assert!(absent.ok(), "absent hardware feature must not fail the gate");
        let on = FeatureState {
            supported: true,
            active: true,
            probe: "cpuid+cr4",
        };
        test_assert!(on.ok(), "active mitigation must pass");
        TestResult::Pass
    }

    /// The retpoline byte probe must reject a thunk that no longer ends in
    /// `mov [rsp], rax; ret` — otherwise it would pass on any function.
    fn test_retpoline_byte_probe() -> TestResult {
        test_assert!(
            thunk_window_matches(&[0xe8, 0, 0, 0, 0, 0x48, 0x89, 0x04, 0x24, 0xc3]),
            "real thunk encoding must match"
        );
        test_assert!(
            !thunk_window_matches(&[0xff, 0xe0, 0xc3, 0x90, 0x90, 0x90, 0x90, 0x90]),
            "a bare `jmp rax` must not pass as a retpoline thunk"
        );
        test_assert!(
            !thunk_window_matches(&[0x48, 0x89, 0x04, 0x24]),
            "a truncated tail must not match"
        );
        // The live image must still carry the sequence.
        test_assert!(thunk_bytes_intact(), "retpoline thunk bytes are not intact");
        TestResult::Pass
    }

    /// The W^X classifier must flag writable+executable and nothing else.
    fn test_wx_classifier() -> TestResult {
        let rw_nx = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE;
        let rw_x = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
        let ro_x = PageTableFlags::PRESENT;
        test_assert!(!is_wx(rw_nx), "writable + NX is not a W^X violation");
        test_assert!(is_wx(rw_x), "writable + executable is a W^X violation");
        test_assert!(!is_wx(ro_x), "read-only + executable is not a W^X violation");
        test_assert!(
            !is_wx(PageTableFlags::WRITABLE),
            "non-present mapping is not a W^X violation"
        );
        TestResult::Pass
    }

    /// The guard-band probe must actually notice a dirtied byte.
    fn test_guard_band_detects_overflow() -> TestResult {
        let clean = [0u8; 64];
        test_assert_eq!(guard_dirty_bytes(&clean), 0);
        let mut dirty = [0u8; 64];
        dirty[63] = 0xAA;
        test_assert_eq!(guard_dirty_bytes(&dirty), 1);
        dirty[0] = 0x01;
        test_assert_eq!(guard_dirty_bytes(&dirty), 2);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("security::feature_probe_bits", test_probes_read_the_right_bits);
        crate::testing::register_test("security::feature_supported_off", test_supported_but_off_is_not_ok);
        crate::testing::register_test("security::feature_retpoline_bytes", test_retpoline_byte_probe);
        crate::testing::register_test("security::feature_wx_classifier", test_wx_classifier);
        crate::testing::register_test("security::feature_stack_guard", test_guard_band_detects_overflow);
    }
}
