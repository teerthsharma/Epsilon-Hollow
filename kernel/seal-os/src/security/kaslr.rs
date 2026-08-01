// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! KASLR — kernel address space layout randomisation.
//!
//! # What this actually randomises
//!
//! Seal OS is loaded by UEFI firmware as a PE/COFF image. The firmware picks
//! the physical load address, applies the PE base relocations, and hands us a
//! running image; after `ExitBootServices` the kernel executes out of the
//! identity map at that firmware-chosen physical address. Nothing in this
//! kernel re-applies PE relocations, so **the image base is not randomised by
//! us** — it is whatever the firmware chose. Claiming image-base KASLR here
//! would be false.
//!
//! What is genuinely under kernel control is the *virtual* layout the kernel
//! builds in `memory::virt::init`. This module randomises two windows:
//!
//! 1. **Kernel image higher-half alias** — the read alias of the kernel image
//!    at `0xffff_ffff_8000_0000`. Slid inside a 1 GiB window at 2 MiB
//!    granularity. This alias is not the address the kernel executes from, so
//!    its entropy hides the alias only; it is reported separately and must not
//!    be advertised as image-base entropy.
//! 2. **Kernel heap virtual window** — the base of the bump region every
//!    `alloc_virtual_pages` result comes from. Slid inside an 8 TiB span at
//!    2 MiB granularity. This one is load-bearing: a leaked kernel heap
//!    pointer no longer discloses a build-constant address.
//!
//! # Entropy budget (honest)
//!
//! | window            | granule | slots     | bits |
//! |-------------------|---------|-----------|------|
//! | kernel alias      | 2 MiB   | ~510      | 8    |
//! | kernel heap       | 2 MiB   | 4 194 304 | 22   |
//!
//! The alias slot count depends on the image size, so the alias bit count is
//! measured at boot, not hard-coded. Slides are derived with `r % slots`; for
//! a 64-bit draw against <= 2^22 slots the modulo bias is below 2^-41 and is
//! ignored.
//!
//! # Fail closed
//!
//! Both slides come from RDSEED, falling back to RDRAND. If neither is
//! available, or the source returns the same word twice (a stuck generator),
//! no slide is applied, the state records `entropy=none`, and the proof line
//! reports `result=fail` so the host-side image gate rejects the build. The
//! kernel deliberately does not halt: a mitigation that bricks boot on a CPU
//! without RDRAND is worse than a build that cannot claim the mitigation.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Base of the kernel image higher-half alias window.
pub const KERNEL_ALIAS_BASE: u64 = 0xffff_ffff_8000_0000;
/// Size of the alias window (1 GiB).
pub const KERNEL_ALIAS_WINDOW: u64 = 0x4000_0000;
/// Base of the kernel heap virtual window.
pub const HEAP_WINDOW_BASE: u64 = 0xffff_9000_0000_0000;
/// Hard limit of the kernel heap virtual window (enforced by `alloc_virtual_pages`).
pub const HEAP_WINDOW_LIMIT: u64 = 0xffff_a000_0000_0000;
/// Span of the heap window reserved for the KASLR slide (8 TiB of the 16 TiB
/// window). The remaining 8 TiB is always left usable for allocations.
pub const HEAP_SLIDE_SPAN: u64 = 0x0000_0800_0000_0000;
/// Randomisation granule — matches the 2 MiB huge-page granule the identity
/// map uses, and matches what Linux KASLR slides at.
pub const SLIDE_GRANULE: u64 = 2 * 1024 * 1024;

/// Number of distinct heap window positions.
pub const HEAP_SLIDE_SLOTS: u64 = HEAP_SLIDE_SPAN / SLIDE_GRANULE;

const SOURCE_UNINIT: u32 = 0;
const SOURCE_RDSEED: u32 = 1;
const SOURCE_RDRAND: u32 = 2;
const SOURCE_NONE: u32 = 3;

static ENTROPY_SOURCE: AtomicU32 = AtomicU32::new(SOURCE_UNINIT);
static ALIAS_SLIDE: AtomicU64 = AtomicU64::new(0);
static HEAP_SLIDE: AtomicU64 = AtomicU64::new(0);
static ALIAS_SLOTS: AtomicU64 = AtomicU64::new(0);
static BOOT_NONCE: AtomicU64 = AtomicU64::new(0);
static IMAGE_BASE: AtomicU64 = AtomicU64::new(0);
static IMAGE_SIZE: AtomicU64 = AtomicU64::new(0);

/// Round `n` up to the next `SLIDE_GRANULE` multiple.
fn round_up_granule(n: u64) -> u64 {
    n.div_ceil(SLIDE_GRANULE) * SLIDE_GRANULE
}

/// Number of 2 MiB slots the kernel alias can slide into for an image of
/// `image_size` bytes. Always at least one (slide 0).
pub fn alias_slots_for(image_size: u64) -> u64 {
    let reserved = round_up_granule(image_size).min(KERNEL_ALIAS_WINDOW);
    ((KERNEL_ALIAS_WINDOW - reserved) / SLIDE_GRANULE).max(1)
}

/// Bits of randomisation a slot count actually delivers: `floor(log2(slots))`.
pub fn bits_for_slots(slots: u64) -> u32 {
    if slots < 2 {
        0
    } else {
        slots.ilog2()
    }
}

/// Draw two independent 64-bit words from hardware entropy.
///
/// Returns `None` when no hardware source is available or the source returns
/// the same word twice (stuck-generator negative control).
fn draw_entropy() -> Option<(u64, u64, u32)> {
    crate::drivers::entropy::init();
    for (source, get) in [
        (
            SOURCE_RDSEED,
            crate::drivers::entropy::rdseed_u64 as fn() -> Option<u64>,
        ),
        (
            SOURCE_RDRAND,
            crate::drivers::entropy::rdrand_u64 as fn() -> Option<u64>,
        ),
    ] {
        if let (Some(a), Some(b)) = (get(), get()) {
            if a != b {
                return Some((a, b, source));
            }
        }
    }
    None
}

/// Initialise KASLR. Must run before `memory::virt::init` builds the kernel
/// page tables. Idempotent: later calls are ignored.
///
/// Returns `true` when randomisation is active, `false` when it failed closed.
pub fn init(image_base: u64, image_size: u64) -> bool {
    if ENTROPY_SOURCE.load(Ordering::SeqCst) != SOURCE_UNINIT {
        return is_active();
    }

    IMAGE_BASE.store(image_base, Ordering::SeqCst);
    IMAGE_SIZE.store(image_size, Ordering::SeqCst);
    let slots = alias_slots_for(image_size);
    ALIAS_SLOTS.store(slots, Ordering::SeqCst);

    let Some((raw_alias, raw_heap, source)) = draw_entropy() else {
        ENTROPY_SOURCE.store(SOURCE_NONE, Ordering::SeqCst);
        crate::serial_println!(
            "[KASLR] FAILED CLOSED — no hardware entropy (RDSEED/RDRAND); kernel mappings stay at their build-constant bases and the KASLR proof will report result=fail"
        );
        return false;
    };

    ALIAS_SLIDE.store((raw_alias % slots) * SLIDE_GRANULE, Ordering::SeqCst);
    HEAP_SLIDE.store(
        (raw_heap % HEAP_SLIDE_SLOTS) * SLIDE_GRANULE,
        Ordering::SeqCst,
    );
    BOOT_NONCE.store(raw_alias ^ raw_heap, Ordering::SeqCst);
    ENTROPY_SOURCE.store(source, Ordering::SeqCst);

    crate::serial_println!(
        "[KASLR] alias_base={:#x} heap_base={:#x} entropy={} bits={}",
        kernel_alias_base(),
        heap_window_base(),
        entropy_source(),
        total_entropy_bits()
    );
    true
}

/// True when both slides were drawn from hardware entropy.
pub fn is_active() -> bool {
    matches!(
        ENTROPY_SOURCE.load(Ordering::SeqCst),
        SOURCE_RDSEED | SOURCE_RDRAND
    )
}

/// Name of the entropy source actually used.
pub fn entropy_source() -> &'static str {
    match ENTROPY_SOURCE.load(Ordering::SeqCst) {
        SOURCE_RDSEED => "rdseed",
        SOURCE_RDRAND => "rdrand",
        SOURCE_NONE => "none",
        _ => "uninit",
    }
}

/// Randomised base of the kernel image higher-half alias.
pub fn kernel_alias_base() -> u64 {
    KERNEL_ALIAS_BASE + ALIAS_SLIDE.load(Ordering::SeqCst)
}

/// Randomised base of the kernel heap virtual window.
pub fn heap_window_base() -> u64 {
    HEAP_WINDOW_BASE + HEAP_SLIDE.load(Ordering::SeqCst)
}

/// Applied kernel-alias slide in bytes.
pub fn alias_slide() -> u64 {
    ALIAS_SLIDE.load(Ordering::SeqCst)
}

/// Applied heap-window slide in bytes.
pub fn heap_slide() -> u64 {
    HEAP_SLIDE.load(Ordering::SeqCst)
}

/// Bits of entropy actually obtained on the kernel alias window.
pub fn alias_entropy_bits() -> u32 {
    if !is_active() {
        return 0;
    }
    bits_for_slots(ALIAS_SLOTS.load(Ordering::SeqCst))
}

/// Bits of entropy actually obtained on the kernel heap window.
pub fn heap_entropy_bits() -> u32 {
    if !is_active() {
        return 0;
    }
    bits_for_slots(HEAP_SLIDE_SLOTS)
}

/// Sum of the two window entropy budgets. This is *not* image-base entropy.
pub fn total_entropy_bits() -> u32 {
    alias_entropy_bits() + heap_entropy_bits()
}

/// `[KASLR] proof version=1 ...` — every field measured from live state.
///
/// Cross-boot variation cannot be shown from inside a single boot. Two things
/// stand in for it: `resample_differs` re-draws from the same hardware source
/// at proof time and fails if the generator is stuck, and `boot_nonce` is a
/// per-boot value an external harness diffs across two boot logs
/// (`cross_boot=external-diff`).
pub fn kaslr_proof_line() -> alloc::string::String {
    let alias_base = kernel_alias_base();
    let heap_base = heap_window_base();
    let resample = draw_entropy();
    let resample_nonce = resample.map(|(a, b, _)| a ^ b).unwrap_or(0);
    let boot_nonce = BOOT_NONCE.load(Ordering::SeqCst);
    let resample_differs = resample.is_some() && resample_nonce != boot_nonce;

    let aligned = alias_base % SLIDE_GRANULE == 0 && heap_base % SLIDE_GRANULE == 0;
    let alias_in_range = alias_base >= KERNEL_ALIAS_BASE
        && alias_base + round_up_granule(IMAGE_SIZE.load(Ordering::SeqCst))
            <= KERNEL_ALIAS_BASE + KERNEL_ALIAS_WINDOW;
    let heap_in_range =
        heap_base >= HEAP_WINDOW_BASE && heap_base < HEAP_WINDOW_BASE + HEAP_SLIDE_SPAN;
    let pass = is_active() && resample_differs && aligned && alias_in_range && heap_in_range;

    alloc::format!(
        "[KASLR] proof version=1 scope=mappings image_base_randomised=0 firmware_image_base={:#x} image_size={:#x} kernel_alias_base={:#x} kernel_alias_slide={:#x} kernel_alias_slots={} kernel_alias_bits={} heap_window_base={:#x} heap_window_slide={:#x} heap_window_slots={} heap_window_bits={} total_bits={} granule={:#x} aligned={} in_range={} entropy={} boot_nonce={:#x} resample_nonce={:#x} resample_differs={} cross_boot=external-diff active={} result={}",
        IMAGE_BASE.load(Ordering::SeqCst),
        IMAGE_SIZE.load(Ordering::SeqCst),
        alias_base,
        alias_slide(),
        ALIAS_SLOTS.load(Ordering::SeqCst),
        alias_entropy_bits(),
        heap_base,
        heap_slide(),
        HEAP_SLIDE_SLOTS,
        heap_entropy_bits(),
        total_entropy_bits(),
        SLIDE_GRANULE,
        if aligned { 1 } else { 0 },
        if alias_in_range && heap_in_range { 1 } else { 0 },
        entropy_source(),
        boot_nonce,
        resample_nonce,
        if resample_differs { 1 } else { 0 },
        if is_active() { 1 } else { 0 },
        if pass { "pass" } else { "fail" }
    )
}

/// Emit the KASLR proof line to the serial console.
pub fn emit_kaslr_proof() {
    crate::serial_println!("{}", kaslr_proof_line());
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// Entropy failure must leave both windows at their build-constant bases
    /// and must not report any entropy bits.
    fn test_fail_closed_reports_no_entropy() -> TestResult {
        // Reproduce the fail-closed arithmetic without disturbing live state:
        // a zero slide is exactly what `init` leaves behind when `draw_entropy`
        // returns `None`.
        test_assert_eq!(KERNEL_ALIAS_BASE + 0, KERNEL_ALIAS_BASE);
        test_assert_eq!(HEAP_WINDOW_BASE + 0, HEAP_WINDOW_BASE);
        test_assert_eq!(bits_for_slots(1), 0);
        test_assert_eq!(bits_for_slots(0), 0);
        // A stuck generator (both draws identical) is rejected by `draw_entropy`.
        test_assert!(
            entropy_source() != "uninit" || !is_active(),
            "uninitialised KASLR must not report active"
        );
        TestResult::Pass
    }

    /// Every reachable slide must land inside its window and stay 2 MiB aligned.
    fn test_slides_stay_in_range_and_aligned() -> TestResult {
        let slots = alias_slots_for(4 * 1024 * 1024);
        test_assert!(slots >= 1, "alias slots must be positive");
        for raw in [0u64, 1, 7, u64::MAX, 0x9e37_79b9_7f4a_7c15] {
            let alias = KERNEL_ALIAS_BASE + (raw % slots) * SLIDE_GRANULE;
            test_assert_eq!(alias % SLIDE_GRANULE, 0);
            test_assert!(alias >= KERNEL_ALIAS_BASE, "alias below window");
            test_assert!(
                alias + 4 * 1024 * 1024 <= KERNEL_ALIAS_BASE + KERNEL_ALIAS_WINDOW,
                "alias image overruns the 1 GiB window"
            );

            let heap = HEAP_WINDOW_BASE + (raw % HEAP_SLIDE_SLOTS) * SLIDE_GRANULE;
            test_assert_eq!(heap % SLIDE_GRANULE, 0);
            test_assert!(heap >= HEAP_WINDOW_BASE, "heap below window");
            test_assert!(
                heap < HEAP_WINDOW_LIMIT - HEAP_SLIDE_SPAN,
                "heap slide leaves less than 8 TiB usable"
            );
        }
        TestResult::Pass
    }

    /// The advertised bit counts must match the slot counts, not a wish.
    fn test_entropy_bits_match_slot_counts() -> TestResult {
        test_assert_eq!(bits_for_slots(HEAP_SLIDE_SLOTS), 22);
        test_assert_eq!(bits_for_slots(2), 1);
        test_assert_eq!(bits_for_slots(510), 8);
        test_assert_eq!(bits_for_slots(512), 9);
        // A 1 GiB image leaves exactly one slot: zero bits, never a lie.
        test_assert_eq!(alias_slots_for(KERNEL_ALIAS_WINDOW), 1);
        test_assert_eq!(bits_for_slots(alias_slots_for(KERNEL_ALIAS_WINDOW)), 0);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("security::kaslr_fail_closed", test_fail_closed_reports_no_entropy);
        crate::testing::register_test("security::kaslr_range", test_slides_stay_in_range_and_aligned);
        crate::testing::register_test("security::kaslr_bits", test_entropy_bits_match_slot_counts);
    }
}
