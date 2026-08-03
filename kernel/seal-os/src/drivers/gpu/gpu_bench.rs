// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! `[GPU-BENCH]` proof — what the GPU stack can actually be shown to do.
//!
//! The proof answers four questions at runtime, and is allowed to answer any of
//! them badly:
//!
//! 1. Is a real, non-empty GCN ISA blob embedded in the image, and does it match
//!    what [`super::gcn_asm`] assembles from source?
//! 2. Does that blob decode back to the instruction sequence it claims to be?
//! 3. Is an AMD GPU present, and if so did PM4 dispatch actually run?
//! 4. Does whichever backend ran agree bit-for-bit with a hand-computed
//!    reference on a fixed input?
//!
//! On a machine with no AMD GPU — which includes every CI runner this project
//! uses — question 3 answers `gpu_present=0 hw_attempted=0 backend=cpu_fallback`
//! and stays that way.  It is never reported as success.  When an AMD GPU *is*
//! present the PM4 path is driven for real and a failure there fails the proof.

use alloc::format;
use alloc::string::String;

use super::amd::AmdGpu;
use super::compute_queue::ComputeRing;
use super::gcn_asm;
use super::gpu_mem::{download, upload, GpuBuffer};
use super::shader_binaries::{find_kernel, KERNELS};

/// Problem size for the fixed-input numerical check.
pub const REF_DIM: usize = 64;

/// Contraction factor for the fixed-input check.  Exactly representable.
pub const REF_ALPHA: f64 = 0.25;

// ------------------------------------------------------------------
// CPU reference
// ------------------------------------------------------------------

/// CPU reference for `shaders/spectral_contract.cl`.
///
/// Deliberately written as two separately-rounded products followed by an add,
/// matching the OpenCL source and the GCN instruction sequence: no FMA
/// contraction, so CPU and GPU results are bit-identical rather than merely
/// close.
pub fn spectral_step_cpu(state: &[f64], target: &[f64], alpha: f64, out: &mut [f64]) {
    let one_minus_alpha = 1.0 - alpha;
    for i in 0..out.len() {
        out[i] = one_minus_alpha * state[i] + alpha * target[i];
    }
}

/// The fixed input used by the proof and the tests.
///
/// `state[i] = i`, `target[i] = 100 + i`, `alpha = 0.25`.  Every intermediate is
/// exactly representable in binary64, so the correct answer is exactly
/// `i + 25.0` with no rounding to argue about.
pub fn reference_inputs() -> ([f64; REF_DIM], [f64; REF_DIM], [f64; REF_DIM]) {
    let mut state = [0.0f64; REF_DIM];
    let mut target = [0.0f64; REF_DIM];
    let mut golden = [0.0f64; REF_DIM];
    for i in 0..REF_DIM {
        state[i] = i as f64;
        target[i] = (100 + i) as f64;
        golden[i] = (i + 25) as f64;
    }
    (state, target, golden)
}

/// FNV-1a (64-bit) over a byte slice — used to fingerprint the blob.
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ------------------------------------------------------------------
// Blob / encoder self-checks
// ------------------------------------------------------------------

/// Result of checking the embedded blob against the in-tree assembler.
#[derive(Debug, Clone, Copy)]
pub struct BlobCheck {
    /// Bytes actually embedded from `shaders/spectral_step.bin` (0 if missing).
    pub blob_bytes: usize,
    /// FNV-1a of the embedded blob.
    pub blob_fnv: u64,
    /// FNV-1a of what `gcn_asm::spectral_step_bytes()` produces right now.
    pub encoder_fnv: u64,
    /// Blob and encoder agree byte-for-byte.
    pub matches: bool,
    /// Words that equal the LLVM-verified golden table.
    pub golden_ok: usize,
    /// Instructions the disassembler could decode.
    pub decoded: usize,
    /// Words that survived decode → re-encode unchanged.
    pub roundtrip_ok: usize,
    /// Decoded mnemonic sequence matches `gcn_asm::SPECTRAL_STEP_ASM`.
    pub mnemonics_ok: bool,
}

/// Check the embedded `spectral_step` blob against the in-tree assembler,
/// the LLVM-verified golden words, and the in-tree disassembler.
pub fn check_spectral_step_blob() -> BlobCheck {
    let encoded = gcn_asm::spectral_step_bytes();
    let words = gcn_asm::assemble_spectral_step();

    let blob: &[u8] = match find_kernel("spectral_step") {
        Some(m) => m.binary,
        None => &[],
    };

    let golden_ok = words
        .iter()
        .zip(gcn_asm::GOLDEN_SPECTRAL_STEP.iter())
        .filter(|(a, b)| a == b)
        .count();

    let mut decoded = 0usize;
    let mut roundtrip_ok = 0usize;
    let mut mnemonics_ok = true;
    let mut i = 0usize;
    while i < words.len() {
        let d = match gcn_asm::decode(&words, i) {
            Some(d) => d,
            None => {
                mnemonics_ok = false;
                break;
            }
        };
        let (re, n) = gcn_asm::reencode(&d);
        for j in 0..n {
            if re[j] == words[i + j] {
                roundtrip_ok += 1;
            }
        }
        match gcn_asm::SPECTRAL_STEP_ASM.get(decoded) {
            // The listing entry is `mnemonic operands...`; compare the mnemonic.
            Some(line) if line.split(' ').next() == Some(d.mnemonic) => {}
            _ => mnemonics_ok = false,
        }
        decoded += 1;
        i += d.words as usize;
    }

    BlobCheck {
        blob_bytes: blob.len(),
        blob_fnv: fnv1a64(blob),
        encoder_fnv: fnv1a64(&encoded),
        matches: blob == &encoded[..],
        golden_ok,
        decoded,
        roundtrip_ok,
        mnemonics_ok,
    }
}

// ------------------------------------------------------------------
// Hardware path
// ------------------------------------------------------------------

/// True when PCI enumeration found an AMD display-class device.
pub fn amd_gpu_present() -> bool {
    crate::drivers::pci::get_devices()
        .iter()
        .any(|d| d.class == 0x03 && d.vendor_id == 0x1002)
}

/// Drive one real PM4 `spectral_step` dispatch and read the result back.
///
/// Every failure is returned as a distinct reason string so the proof line can
/// name it; nothing here falls back to the CPU on the quiet.
///
/// # Safety
/// Touches raw BAR0 MMIO and physically-addressed GPU memory.  PCI enumeration
/// must already be complete and no other code may be driving the same ring.
pub unsafe fn run_pm4_spectral_step(
    state: &[f64],
    target: &[f64],
    alpha: f64,
    out: &mut [f64],
) -> Result<u64, &'static str> {
    let devices = crate::drivers::pci::get_devices();
    let dev = devices
        .iter()
        .find(|d| d.class == 0x03 && d.vendor_id == 0x1002)
        .ok_or("no_amd_gpu")?;
    let gpu = AmdGpu::probe(dev).ok_or("probe_failed")?;

    let meta = find_kernel("spectral_step").ok_or("kernel_not_found")?;
    if meta.binary.is_empty() {
        return Err("kernel_empty");
    }

    let ring_pages = 4usize;
    let ring_phys = crate::memory::phys::alloc_frames_contiguous(ring_pages)
        .ok_or("ring_alloc_failed")?
        .as_u64();
    let ring_size_dw = (ring_pages * 4096 / 4) as u32;
    let mut ring = ComputeRing::new(
        gpu.bar0_phys,
        ring_phys as *mut u32,
        ring_size_dw,
        ring_phys,
    )
    .ok_or("ring_init_failed")?;

    let nbytes = out.len() * 8;
    let fence_buf = GpuBuffer::alloc(8).ok_or("fence_alloc_failed")?;
    let kernel_buf = match GpuBuffer::alloc(meta.code_size_bytes) {
        Some(b) => b,
        None => {
            fence_buf.free();
            return Err("kernel_alloc_failed");
        }
    };
    let bufs = (
        GpuBuffer::alloc(nbytes),
        GpuBuffer::alloc(nbytes),
        GpuBuffer::alloc(nbytes),
    );
    let (state_buf, target_buf, out_buf) = match bufs {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        (a, b, c) => {
            if let Some(x) = a {
                x.free();
            }
            if let Some(x) = b {
                x.free();
            }
            if let Some(x) = c {
                x.free();
            }
            kernel_buf.free();
            fence_buf.free();
            return Err("buffer_alloc_failed");
        }
    };

    upload(&kernel_buf, meta.binary);
    upload(&state_buf, f64_as_bytes(state));
    upload(&target_buf, f64_as_bytes(target));

    let alpha_bits = alpha.to_bits();
    let (s, t, o) = (state_buf.phys, target_buf.phys, out_buf.phys);

    ring.set_compute_pgm(kernel_buf.phys);
    ring.set_compute_pgm_rsrc(gcn_asm::SPECTRAL_STEP_RSRC1, gcn_asm::SPECTRAL_STEP_RSRC2);
    ring.set_num_threads(64, 1, 1);
    // Must match gcn_asm::SPECTRAL_STEP_USER_SGPRS and the ABI table there.
    ring.set_user_data(
        0,
        &[
            s as u32,
            (s >> 32) as u32,
            t as u32,
            (t >> 32) as u32,
            o as u32,
            (o >> 32) as u32,
            out.len() as u32,
            0, // padding: keeps `alpha` on an even SGPR pair
            alpha_bits as u32,
            (alpha_bits >> 32) as u32,
        ],
    );
    ring.dispatch_direct(out.len().div_ceil(64) as u32, 1, 1);

    let t0 = core::arch::x86_64::_rdtsc();
    let fence = ring.submit(fence_buf.phys);
    let waited = ring.wait_fence(&fence, 5000);
    let cycles = core::arch::x86_64::_rdtsc().wrapping_sub(t0);

    let result = if waited {
        download(f64_as_bytes_mut(out), &out_buf);
        Ok(cycles)
    } else {
        Err("fence_timeout")
    };

    out_buf.free();
    target_buf.free();
    state_buf.free();
    kernel_buf.free();
    fence_buf.free();
    result
}

fn f64_as_bytes(v: &[f64]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) }
}

fn f64_as_bytes_mut(v: &mut [f64]) -> &mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 8) }
}

/// Count of `out` entries that are bit-identical to `want`, and the largest
/// ULP gap over the rest.
fn compare(out: &[f64], want: &[f64]) -> (usize, u64) {
    let mut exact = 0usize;
    let mut max_ulp = 0u64;
    for i in 0..want.len() {
        let (a, b) = (out[i].to_bits(), want[i].to_bits());
        if a == b {
            exact += 1;
        } else {
            let d = a.abs_diff(b);
            if d > max_ulp {
                max_ulp = d;
            }
        }
    }
    (exact, max_ulp)
}

// ------------------------------------------------------------------
// Proof
// ------------------------------------------------------------------

/// Emit the `[GPU-BENCH]` proof line.  Every field is measured on this boot.
pub fn gpu_bench_proof_line() -> String {
    let check = check_spectral_step_blob();
    let (state, target, golden) = reference_inputs();

    let mut cpu = [0.0f64; REF_DIM];
    spectral_step_cpu(&state, &target, REF_ALPHA, &mut cpu);
    let (cpu_exact, cpu_ulp) = compare(&cpu, &golden);

    let gpu_present = amd_gpu_present();
    let mut backend = "cpu_fallback";
    let mut hw_attempted = 0u32;
    let mut hw_reason = "no_amd_gpu";
    let mut cycles = 0u64;
    let mut backend_out = cpu;

    if gpu_present {
        hw_attempted = 1;
        let mut hw = [0.0f64; REF_DIM];
        match unsafe { run_pm4_spectral_step(&state, &target, REF_ALPHA, &mut hw) } {
            Ok(c) => {
                backend = "pm4_hw";
                hw_reason = "ok";
                cycles = c;
                backend_out = hw;
            }
            Err(e) => {
                hw_reason = e;
            }
        }
    }

    let (backend_exact, backend_ulp) = compare(&backend_out, &golden);

    // A GPU that is present but did not execute is a failure, not a fallback.
    let hw_ok = !gpu_present || backend == "pm4_hw";
    let pass = check.blob_bytes == gcn_asm::SPECTRAL_STEP_BYTES
        && check.matches
        && check.golden_ok == gcn_asm::SPECTRAL_STEP_WORDS
        && check.decoded == gcn_asm::SPECTRAL_STEP_INSTS
        && check.roundtrip_ok == gcn_asm::SPECTRAL_STEP_WORDS
        && check.mnemonics_ok
        && cpu_exact == REF_DIM
        && backend_exact == REF_DIM
        && hw_ok;

    // `KERNELS` carries every declared kernel, including the ones whose `.bin`
    // is still a zero-length placeholder; only the non-empty ones are real.
    let kernels_real = KERNELS.iter().filter(|k| !k.binary.is_empty()).count();
    let kernels_total = KERNELS.len();

    format!(
        "[GPU-BENCH] proof version=1 arch=gfx900 backend={} gpu_present={} hw_attempted={} \
         hw_reason={} cycles={} kernels_real={}/{} spectral_step_bytes={} blob_fnv1a={:#018x} \
         encoder_fnv1a={:#018x} blob_matches_encoder={} golden_words={}/{} decoded_insts={}/{} \
         roundtrip_words={}/{} mnemonics_match={} rsrc1={:#010x} rsrc2={:#010x} ref_dim={} \
         ref_alpha_num=1 ref_alpha_den=4 cpu_ref_exact={}/{} cpu_ref_max_ulp={} \
         backend_exact={}/{} backend_max_ulp={} result={}",
        backend,
        gpu_present as u32,
        hw_attempted,
        hw_reason,
        cycles,
        kernels_real,
        kernels_total,
        check.blob_bytes,
        check.blob_fnv,
        check.encoder_fnv,
        check.matches as u32,
        check.golden_ok,
        gcn_asm::SPECTRAL_STEP_WORDS,
        check.decoded,
        gcn_asm::SPECTRAL_STEP_INSTS,
        check.roundtrip_ok,
        gcn_asm::SPECTRAL_STEP_WORDS,
        check.mnemonics_ok as u32,
        gcn_asm::SPECTRAL_STEP_RSRC1,
        gcn_asm::SPECTRAL_STEP_RSRC2,
        REF_DIM,
        cpu_exact,
        REF_DIM,
        cpu_ulp,
        backend_exact,
        REF_DIM,
        backend_ulp,
        if pass { "pass" } else { "fail" }
    )
}

/// Print the proof line and report whether it passed.
pub fn run_gpu_bench_proof() -> bool {
    let line = gpu_bench_proof_line();
    let pass = line.ends_with("result=pass");
    crate::serial_println!("{}", line);
    for (name, why) in gcn_asm::NOT_IMPLEMENTED.iter() {
        crate::serial_println!("[GPU-BENCH] kernel={} isa=absent reason={}", name, why);
    }
    pass
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;

    /// Each encoded word must equal the value LLVM's AMDGPU assembler produced
    /// for the same mnemonic at `-C target-cpu=gfx900`.
    fn test_encoder_matches_golden() -> TestResult {
        let w = gcn_asm::assemble_spectral_step();
        for i in 0..gcn_asm::SPECTRAL_STEP_WORDS {
            test_assert!(
                w[i] == gcn_asm::GOLDEN_SPECTRAL_STEP[i],
                "encoded word differs from LLVM-verified golden"
            );
        }
        TestResult::Pass
    }

    /// Every word must survive decode → re-encode, and the decoded mnemonic
    /// sequence must equal the recorded assembly listing.
    fn test_decode_roundtrip() -> TestResult {
        let c = check_spectral_step_blob();
        test_assert!(
            c.decoded == gcn_asm::SPECTRAL_STEP_INSTS,
            "all 17 instructions decode"
        );
        test_assert!(
            c.roundtrip_ok == gcn_asm::SPECTRAL_STEP_WORDS,
            "all 24 words survive decode then re-encode"
        );
        test_assert!(c.mnemonics_ok, "decoded mnemonics match the asm listing");
        TestResult::Pass
    }

    /// The checked-in `.bin` must be non-empty and byte-identical to what the
    /// in-tree assembler emits.
    fn test_blob_matches_encoder() -> TestResult {
        let c = check_spectral_step_blob();
        test_assert!(
            c.blob_bytes == gcn_asm::SPECTRAL_STEP_BYTES,
            "embedded blob is 96 bytes, not an empty placeholder"
        );
        test_assert!(
            c.blob_fnv == c.encoder_fnv,
            "blob digest equals assembler-output digest"
        );
        test_assert!(c.matches, "blob is byte-identical to the assembler output");
        TestResult::Pass
    }

    /// The CPU reference must reproduce the hand-computed exact answer.
    fn test_cpu_reference_exact() -> TestResult {
        let (state, target, golden) = reference_inputs();
        let mut out = [0.0f64; REF_DIM];
        spectral_step_cpu(&state, &target, REF_ALPHA, &mut out);
        for i in 0..REF_DIM {
            test_assert!(
                out[i].to_bits() == golden[i].to_bits(),
                "spectral step value differs from the exact hand-computed answer"
            );
        }
        TestResult::Pass
    }

    /// With no AMD GPU the PM4 path must return an error, and the proof line
    /// must say `backend=cpu_fallback`.  It must never report hardware success.
    fn test_absent_gpu_reports_failure() -> TestResult {
        if amd_gpu_present() {
            // Real AMD hardware: the proof drives it for real; nothing to assert here.
            return TestResult::Pass;
        }
        let (state, target, _) = reference_inputs();
        let mut out = [0.0f64; REF_DIM];
        let r = unsafe { run_pm4_spectral_step(&state, &target, REF_ALPHA, &mut out) };
        test_assert!(r.is_err(), "PM4 dispatch must fail with no AMD GPU");
        let line = gpu_bench_proof_line();
        test_assert!(
            line.contains("backend=cpu_fallback"),
            "proof must report the CPU fallback"
        );
        test_assert!(
            line.contains("hw_reason=no_amd_gpu"),
            "proof must name the missing GPU"
        );
        test_assert!(!line.contains("backend=pm4_hw"), "no fabricated hw backend");
        TestResult::Pass
    }

    /// Kernels whose `.bin` is still a zero-length placeholder must look absent
    /// to every dispatcher, not like a zero-byte shader worth uploading.
    fn test_placeholder_kernels_absent() -> TestResult {
        for (name, _) in gcn_asm::NOT_IMPLEMENTED.iter() {
            test_assert!(
                find_kernel(name).is_none(),
                "placeholder kernel must report kernel_not_found"
            );
        }
        test_assert!(
            find_kernel("spectral_step").is_some(),
            "the real kernel must still be found"
        );
        TestResult::Pass
    }

    /// The proof line itself must pass on this boot.
    fn test_proof_line_passes() -> TestResult {
        let line = gpu_bench_proof_line();
        test_assert!(line.ends_with("result=pass"), "GPU-BENCH proof result");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("gpu::encoder_matches_golden", test_encoder_matches_golden);
        crate::testing::register_test("gpu::decode_roundtrip", test_decode_roundtrip);
        crate::testing::register_test("gpu::blob_matches_encoder", test_blob_matches_encoder);
        crate::testing::register_test("gpu::cpu_reference_exact", test_cpu_reference_exact);
        crate::testing::register_test(
            "gpu::absent_gpu_reports_failure",
            test_absent_gpu_reports_failure,
        );
        crate::testing::register_test(
            "gpu::placeholder_kernels_absent",
            test_placeholder_kernels_absent,
        );
        crate::testing::register_test("gpu::proof_line_passes", test_proof_line_passes);
    }
}
