// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! GFX9 (Vega / `gfx900`) machine-code assembler and disassembler.
//!
//! This module is the *source* of the checked-in `shaders/*.bin` GPU kernels.
//! Nothing here is a stub: [`assemble_spectral_step`] emits the exact 96 bytes
//! that `shaders/spectral_step.bin` contains, and the kernel test-suite asserts
//! that equality at runtime.  A reviewer therefore never has to trust an opaque
//! binary — the binary is a function of this file.
//!
//! # Scope
//!
//! Only `spectral_step` (from `shaders/spectral_contract.cl`) is implemented.
//! `voronoi_assign` and `jl_project` are deliberately still absent; see
//! `NOT_IMPLEMENTED` at the bottom of this file for the specific reasons.
//!
//! # Reference
//!
//! All bit layouts and opcode numbers below are from the *"Vega" Instruction
//! Set Architecture Reference Guide* (AMD, rev 1.1, July 2017), chapter 12
//! ("Instruction Formats"), specifically §12.1 SOP2, §12.3 SOP1, §12.5 SOPP,
//! §12.8 SMEM (unused), §12.9 VOP2, §12.10 VOP1, §12.11 VOPC, §12.12 VOP3A and
//! §12.15 FLAT/GLOBAL.  Opcode numbers are the GFX9 column; several differ from
//! GFX8 (notably `S_AND_SAVEEXEC_B64`, which is 32 on GFX9 and 33 on GFX8, and
//! the carry-less `V_ADD_U32` at VOP2 opcode 0x34 which does not exist before
//! GFX9).
//!
//! # Independent verification (no AMD hardware required)
//!
//! Every word emitted here was cross-checked against LLVM's AMDGPU assembler,
//! which ships inside the Rust nightly toolchain — no ROCm, no clang, no GPU.
//! Reproduce with:
//!
//! ```text
//! cargo new --lib gcnprobe && cd gcnprobe
//! # src/lib.rs:
//! #   #![no_std]
//! #   #![feature(asm_experimental_arch)]
//! #   #[no_mangle] pub extern "C" fn probe() {
//! #       unsafe { core::arch::asm!( <the ASM_LISTING lines below>, options(nomem, nostack)) }
//! #   }
//! RUSTFLAGS="-C target-cpu=gfx900" cargo +nightly rustc --release \
//!     -Zbuild-std=core --target amdgcn-amd-amdhsa -- \
//!     -C target-cpu=gfx900 --emit=obj -o probe.o
//! # then read section `.text.probe` of the emitted object
//! ```
//!
//! LLVM emits a 4-byte `s_waitcnt 0` function prologue and a trailing
//! `s_setpc_b64 s[30:31]` return because the block is wrapped in a function;
//! the 24 words between them are byte-identical to [`GOLDEN_SPECTRAL_STEP`].

// ------------------------------------------------------------------
// Operand encoding — Vega ISA §12.16 "Instruction Operand Sources"
// ------------------------------------------------------------------

/// Scalar GPR `n` as a source/destination operand (SGPR 0..101 encode as 0..101).
pub const fn sgpr(n: u32) -> u32 {
    n
}

/// Vector GPR `n` as a *source* operand (VGPRs start at operand code 256).
pub const fn vgpr_src(n: u32) -> u32 {
    256 + n
}

/// `VCC_LO` — operand code 106.
pub const VCC: u32 = 106;

/// Inline constant for a small non-negative integer (0..=64 → 128..=192).
pub const fn inline_int(v: u32) -> u32 {
    128 + v
}

/// Inline constant `1.0` (operand code 242).  For a 64-bit operand the hardware
/// widens this to the double `1.0`.
pub const INLINE_F_ONE: u32 = 242;

// ------------------------------------------------------------------
// Encoding-class constants
// ------------------------------------------------------------------

const ENC_SOP2: u32 = 0b10 << 30;
const ENC_SOP1: u32 = 0b1_0111_1101 << 23;
const ENC_SOPP: u32 = 0b1_0111_1111 << 23;
const ENC_VOP1: u32 = 0b011_1111 << 25;
const ENC_VOPC: u32 = 0b011_1110 << 25;
const ENC_VOP3A: u32 = 0b110100 << 26;
const ENC_FLAT: u32 = 0b110111 << 26;

/// FLAT `SEG` field: global memory (Vega ISA §12.15). GFX9 only.
pub const SEG_GLOBAL: u32 = 2;

// ------------------------------------------------------------------
// Opcodes actually used by the kernels in this file
// ------------------------------------------------------------------

/// SOP2 `S_LSHL_B32`.
pub const OP_S_LSHL_B32: u32 = 28;
/// SOP1 `S_AND_SAVEEXEC_B64` (GFX9 numbering; GFX8 uses 33).
pub const OP_S_AND_SAVEEXEC_B64: u32 = 32;
/// SOPP `S_ENDPGM`.
pub const OP_S_ENDPGM: u32 = 1;
/// SOPP `S_CBRANCH_EXECZ`.
pub const OP_S_CBRANCH_EXECZ: u32 = 8;
/// SOPP `S_WAITCNT`.
pub const OP_S_WAITCNT: u32 = 12;
/// VOP1 `V_MOV_B32`.
pub const OP_V_MOV_B32: u32 = 1;
/// VOP2 `V_LSHLREV_B32`.
pub const OP_V_LSHLREV_B32: u32 = 0x12;
/// VOP2 `V_ADD_U32` — the GFX9 carry-less form.
pub const OP_V_ADD_U32: u32 = 0x34;
/// VOPC `V_CMP_GT_I32` (base 0xC0 for the signed-32 group, `GT` is +4).
pub const OP_V_CMP_GT_I32: u32 = 0xC4;
/// VOP3A `V_ADD_F64`.
pub const OP_V_ADD_F64: u32 = 0x280;
/// VOP3A `V_MUL_F64`.
pub const OP_V_MUL_F64: u32 = 0x281;
/// FLAT `LOAD_DWORDX2` (with `SEG=2`: `global_load_dwordx2`).
pub const OP_FLAT_LOAD_DWORDX2: u32 = 0x15;
/// FLAT `STORE_DWORDX2` (with `SEG=2`: `global_store_dwordx2`).
pub const OP_FLAT_STORE_DWORDX2: u32 = 0x1D;

/// `s_waitcnt vmcnt(0)` immediate: vmcnt=0, expcnt=7 (don't care),
/// lgkmcnt=15 (don't care).  Vega ISA §"S_WAITCNT": VM_CNT is
/// `simm16[3:0]` plus `simm16[15:14]`, EXP_CNT `[6:4]`, LGKM_CNT `[11:8]`.
pub const WAITCNT_VMCNT0: u16 = 0x0F70;

// ------------------------------------------------------------------
// Encoders — one per instruction format
// ------------------------------------------------------------------

/// SOP2: `[31:30]=0b10, OP[29:23], SDST[22:16], SSRC1[15:8], SSRC0[7:0]`.
pub const fn sop2(op: u32, sdst: u32, ssrc0: u32, ssrc1: u32) -> u32 {
    ENC_SOP2 | ((op & 0x7F) << 23) | ((sdst & 0x7F) << 16) | ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF)
}

/// SOP1: `[31:23]=0b101111101, SDST[22:16], OP[15:8], SSRC0[7:0]`.
pub const fn sop1(op: u32, sdst: u32, ssrc0: u32) -> u32 {
    ENC_SOP1 | ((sdst & 0x7F) << 16) | ((op & 0xFF) << 8) | (ssrc0 & 0xFF)
}

/// SOPP: `[31:23]=0b101111111, OP[22:16], SIMM16[15:0]`.
pub const fn sopp(op: u32, simm16: u16) -> u32 {
    ENC_SOPP | ((op & 0x7F) << 16) | (simm16 as u32)
}

/// VOP1: `[31:25]=0b0111111, VDST[24:17], OP[16:9], SRC0[8:0]`.
pub const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    ENC_VOP1 | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)
}

/// VOP2: `[31]=0, OP[30:25], VDST[24:17], VSRC1[16:9], SRC0[8:0]`.
pub const fn vop2(op: u32, vdst: u32, vsrc1: u32, src0: u32) -> u32 {
    ((op & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)
}

/// VOPC: `[31:25]=0b0111110, OP[24:17], VSRC1[16:9], SRC0[8:0]`.  Result → VCC.
pub const fn vopc(op: u32, vsrc1: u32, src0: u32) -> u32 {
    ENC_VOPC | ((op & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)
}

/// VOP3A, 64-bit:
/// word0 `[31:26]=0b110100, OP[25:16], CLAMP[15], OPSEL[14:11], ABS[10:8], VDST[7:0]`
/// word1 `NEG[31:29], OMOD[28:27], SRC2[26:18], SRC1[17:9], SRC0[8:0]`.
///
/// `neg` is a 3-bit mask selecting which of src0/src1/src2 are negated.
pub const fn vop3a(op: u32, vdst: u32, src0: u32, src1: u32, src2: u32, neg: u32) -> [u32; 2] {
    [
        ENC_VOP3A | ((op & 0x3FF) << 16) | (vdst & 0xFF),
        ((neg & 0x7) << 29) | ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF),
    ]
}

/// FLAT / GLOBAL / SCRATCH:
/// word0 `[31:26]=0b110111, OP[24:18], SLC[17], GLC[16], SEG[15:14], LDS[13], OFFSET[12:0]`
/// word1 `VDST[31:24], NV[23], SADDR[22:16], DATA[15:8], ADDR[7:0]`.
///
/// With `SEG=SEG_GLOBAL` and `saddr` naming an even-aligned SGPR pair, `vaddr`
/// holds a 32-bit byte offset added to the 64-bit scalar base.
#[allow(clippy::too_many_arguments)]
pub const fn flat(
    op: u32,
    seg: u32,
    offset: u32,
    vaddr: u32,
    vdata: u32,
    saddr: u32,
    vdst: u32,
) -> [u32; 2] {
    [
        ENC_FLAT | ((op & 0x7F) << 18) | ((seg & 0x3) << 14) | (offset & 0x1FFF),
        ((vdst & 0xFF) << 24) | ((saddr & 0x7F) << 16) | ((vdata & 0xFF) << 8) | (vaddr & 0xFF),
    ]
}

// ------------------------------------------------------------------
// `spectral_step` — from shaders/spectral_contract.cl
// ------------------------------------------------------------------

/// Number of instructions in the `spectral_step` kernel.
pub const SPECTRAL_STEP_INSTS: usize = 17;

/// Number of 32-bit words in the `spectral_step` kernel.
pub const SPECTRAL_STEP_WORDS: usize = 24;

/// Byte length of `shaders/spectral_step.bin`.
pub const SPECTRAL_STEP_BYTES: usize = SPECTRAL_STEP_WORDS * 4;

/// Kernel argument ABI, as user-data SGPRs written to `COMPUTE_USER_DATA_0..9`.
///
/// | SGPR      | contents                              |
/// |-----------|---------------------------------------|
/// | `s[0:1]`  | `__global const double* state`        |
/// | `s[2:3]`  | `__global const double* target`       |
/// | `s[4:5]`  | `__global double* output`             |
/// | `s6`      | `int dim`                             |
/// | `s7`      | padding (keeps `alpha` even-aligned)  |
/// | `s[8:9]`  | `double alpha`                        |
/// | `s10`     | `workgroup_id_x` (first system SGPR)  |
/// | `v0`      | `workitem_id_x`                       |
///
/// `s7` exists only so that `alpha` lands on an even SGPR pair, which the
/// hardware requires for any 64-bit scalar operand.
pub const SPECTRAL_STEP_USER_SGPRS: u32 = 10;

/// `COMPUTE_PGM_RSRC1` for this kernel.
///
/// `VGPRS[5:0]=2` (10 VGPRs → ceil(10/4)−1), `SGPRS[9:6]=1` (14 SGPRs + VCC →
/// ceil(16/8)−1), `FLOAT_MODE[17:10]=0xC0` (round-nearest-even, f64 denormals
/// enabled), `DX10_CLAMP[21]=1`, `IEEE_MODE[23]=1`.
pub const SPECTRAL_STEP_RSRC1: u32 = 2 | (1 << 6) | (0xC0 << 10) | (1 << 21) | (1 << 23);

/// `COMPUTE_PGM_RSRC2` for this kernel: `USER_SGPR[5:1]=10`, `TGID_X_EN[7]=1`.
pub const SPECTRAL_STEP_RSRC2: u32 = (SPECTRAL_STEP_USER_SGPRS << 1) | (1 << 7);

/// The assembly this kernel corresponds to, in LLVM AMDGPU syntax.
///
/// This is the literal text fed to LLVM's assembler to produce
/// [`GOLDEN_SPECTRAL_STEP`], and the disassembler in this module reproduces the
/// same mnemonic sequence from the encoded words.
pub const SPECTRAL_STEP_ASM: [&str; SPECTRAL_STEP_INSTS] = [
    "s_lshl_b32 s11, s10, 6",
    "v_add_u32_e32 v0, s11, v0",
    "v_cmp_gt_i32_e32 vcc, s6, v0",
    "s_and_saveexec_b64 s[12:13], vcc",
    "s_cbranch_execz 18",
    "v_lshlrev_b32_e32 v0, 3, v0",
    "global_load_dwordx2 v[2:3], v0, s[0:1]",
    "global_load_dwordx2 v[4:5], v0, s[2:3]",
    "v_mov_b32_e32 v6, s8",
    "v_mov_b32_e32 v7, s9",
    "v_add_f64 v[8:9], 1.0, -v[6:7]",
    "s_waitcnt vmcnt(0)",
    "v_mul_f64 v[8:9], v[8:9], v[2:3]",
    "v_mul_f64 v[4:5], v[6:7], v[4:5]",
    "v_add_f64 v[8:9], v[8:9], v[4:5]",
    "global_store_dwordx2 v0, v[8:9], s[4:5]",
    "s_endpgm",
];

/// Machine code for `spectral_step`, as verified by LLVM's AMDGPU assembler
/// targeting `gfx900`.  This table is *not* produced by the encoder above; it
/// is the external reference the encoder is checked against.
pub const GOLDEN_SPECTRAL_STEP: [u32; SPECTRAL_STEP_WORDS] = [
    0x8E0B_860A, // s_lshl_b32 s11, s10, 6
    0x6800_000B, // v_add_u32_e32 v0, s11, v0
    0x7D88_0006, // v_cmp_gt_i32_e32 vcc, s6, v0
    0xBE8C_206A, // s_and_saveexec_b64 s[12:13], vcc
    0xBF88_0012, // s_cbranch_execz 18
    0x2400_0083, // v_lshlrev_b32_e32 v0, 3, v0
    0xDC54_8000, // global_load_dwordx2 v[2:3], v0, s[0:1]
    0x0200_0000,
    0xDC54_8000, // global_load_dwordx2 v[4:5], v0, s[2:3]
    0x0402_0000,
    0x7E0C_0208, // v_mov_b32_e32 v6, s8
    0x7E0E_0209, // v_mov_b32_e32 v7, s9
    0xD280_0008, // v_add_f64 v[8:9], 1.0, -v[6:7]
    0x4002_0CF2,
    0xBF8C_0F70, // s_waitcnt vmcnt(0)
    0xD281_0008, // v_mul_f64 v[8:9], v[8:9], v[2:3]
    0x0002_0508,
    0xD281_0004, // v_mul_f64 v[4:5], v[6:7], v[4:5]
    0x0002_0906,
    0xD280_0008, // v_add_f64 v[8:9], v[8:9], v[4:5]
    0x0002_0908,
    0xDC74_8000, // global_store_dwordx2 v0, v[8:9], s[4:5]
    0x0004_0800,
    0xBF81_0000, // s_endpgm
];

/// Assemble the `spectral_step` GFX9 kernel.
///
/// Implements `shaders/spectral_contract.cl` exactly, including the operation
/// order — `(1-alpha)*x` and `alpha*t` are rounded separately and then added,
/// rather than being contracted into an FMA, so the result is bit-identical to
/// the CPU reference in [`super::gpu_bench::spectral_step_cpu`].
pub fn assemble_spectral_step() -> [u32; SPECTRAL_STEP_WORDS] {
    let mut w = [0u32; SPECTRAL_STEP_WORDS];
    let mut n = 0usize;
    let push = |w: &mut [u32; SPECTRAL_STEP_WORDS], n: &mut usize, v: u32| {
        w[*n] = v;
        *n += 1;
    };
    let push2 = |w: &mut [u32; SPECTRAL_STEP_WORDS], n: &mut usize, v: [u32; 2]| {
        w[*n] = v[0];
        w[*n + 1] = v[1];
        *n += 2;
    };

    // gid = workgroup_id_x * 64 + workitem_id_x
    push(
        &mut w,
        &mut n,
        sop2(OP_S_LSHL_B32, 11, sgpr(10), inline_int(6)),
    );
    push(&mut w, &mut n, vop2(OP_V_ADD_U32, 0, 0, sgpr(11)));
    // if (gid >= dim) goto end;   ->   exec &= (dim > gid)
    push(&mut w, &mut n, vopc(OP_V_CMP_GT_I32, 0, sgpr(6)));
    push(&mut w, &mut n, sop1(OP_S_AND_SAVEEXEC_B64, 12, VCC));
    // Branch target is the trailing s_endpgm: 18 dwords past the next PC.
    push(&mut w, &mut n, sopp(OP_S_CBRANCH_EXECZ, 18));
    // byte offset = gid * 8
    push(&mut w, &mut n, vop2(OP_V_LSHLREV_B32, 0, 0, inline_int(3)));
    // v[2:3] = state[gid]; v[4:5] = target[gid]
    push2(
        &mut w,
        &mut n,
        flat(OP_FLAT_LOAD_DWORDX2, SEG_GLOBAL, 0, 0, 0, sgpr(0), 2),
    );
    push2(
        &mut w,
        &mut n,
        flat(OP_FLAT_LOAD_DWORDX2, SEG_GLOBAL, 0, 0, 0, sgpr(2), 4),
    );
    // v[6:7] = alpha
    push(&mut w, &mut n, vop1(OP_V_MOV_B32, 6, sgpr(8)));
    push(&mut w, &mut n, vop1(OP_V_MOV_B32, 7, sgpr(9)));
    // v[8:9] = 1.0 - alpha            (neg mask 0b010 negates src1)
    push2(
        &mut w,
        &mut n,
        vop3a(OP_V_ADD_F64, 8, INLINE_F_ONE, vgpr_src(6), 0, 0b010),
    );
    push(&mut w, &mut n, sopp(OP_S_WAITCNT, WAITCNT_VMCNT0));
    // v[8:9] = (1-alpha) * x
    push2(
        &mut w,
        &mut n,
        vop3a(OP_V_MUL_F64, 8, vgpr_src(8), vgpr_src(2), 0, 0),
    );
    // v[4:5] = alpha * t
    push2(
        &mut w,
        &mut n,
        vop3a(OP_V_MUL_F64, 4, vgpr_src(6), vgpr_src(4), 0, 0),
    );
    // v[8:9] = (1-alpha)*x + alpha*t
    push2(
        &mut w,
        &mut n,
        vop3a(OP_V_ADD_F64, 8, vgpr_src(8), vgpr_src(4), 0, 0),
    );
    // output[gid] = v[8:9]
    push2(
        &mut w,
        &mut n,
        flat(OP_FLAT_STORE_DWORDX2, SEG_GLOBAL, 0, 0, 8, sgpr(4), 0),
    );
    push(&mut w, &mut n, sopp(OP_S_ENDPGM, 0));

    debug_assert!(n == SPECTRAL_STEP_WORDS);
    w
}

/// The same kernel as a little-endian byte image — this is byte-for-byte the
/// content of `shaders/spectral_step.bin`.
pub fn spectral_step_bytes() -> [u8; SPECTRAL_STEP_BYTES] {
    let words = assemble_spectral_step();
    let mut out = [0u8; SPECTRAL_STEP_BYTES];
    let mut i = 0;
    while i < SPECTRAL_STEP_WORDS {
        let b = words[i].to_le_bytes();
        out[i * 4] = b[0];
        out[i * 4 + 1] = b[1];
        out[i * 4 + 2] = b[2];
        out[i * 4 + 3] = b[3];
        i += 1;
    }
    out
}

// ------------------------------------------------------------------
// Disassembler
// ------------------------------------------------------------------

/// A decoded GFX9 instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decoded {
    /// Mnemonic, as LLVM's AMDGPU assembler spells it.
    pub mnemonic: &'static str,
    /// Opcode within the instruction's encoding class.
    pub op: u32,
    /// Destination register index (SGPR or VGPR index, not operand code).
    pub dst: u32,
    /// Source operand codes, in instruction order.
    pub src: [u32; 3],
    /// Class-specific extra: SOPP simm16, VOP3A neg mask, FLAT saddr.
    pub extra: u32,
    /// Instruction length in 32-bit words.
    pub words: u8,
}

/// Decode one instruction starting at `words[i]`.
///
/// Field extraction is written from the ISA tables independently of the
/// encoders above (different expression forms, opposite direction), but the
/// two share the `OP_*` opcode constants — see the honesty note on
/// [`GOLDEN_SPECTRAL_STEP`]: the load-bearing external check is the golden
/// table, which LLVM produced from mnemonics.
pub fn decode(words: &[u32], i: usize) -> Option<Decoded> {
    let w = *words.get(i)?;

    // FLAT: [31:26] == 0b110111
    if w >> 26 == 0b110111 {
        let w1 = *words.get(i + 1)?;
        let op = (w >> 18) & 0x7F;
        let seg = (w >> 14) & 0x3;
        let mnemonic = match (op, seg) {
            (OP_FLAT_LOAD_DWORDX2, SEG_GLOBAL) => "global_load_dwordx2",
            (OP_FLAT_STORE_DWORDX2, SEG_GLOBAL) => "global_store_dwordx2",
            _ => return None,
        };
        return Some(Decoded {
            mnemonic,
            op,
            dst: (w1 >> 24) & 0xFF,                         // VDST
            src: [w1 & 0xFF, (w1 >> 8) & 0xFF, w & 0x1FFF], // ADDR, DATA, OFFSET
            extra: (w1 >> 16) & 0x7F,                       // SADDR
            words: 2,
        });
    }

    // VOP3A: [31:26] == 0b110100
    if w >> 26 == 0b110100 {
        let w1 = *words.get(i + 1)?;
        let op = (w >> 16) & 0x3FF;
        let mnemonic = match op {
            OP_V_ADD_F64 => "v_add_f64",
            OP_V_MUL_F64 => "v_mul_f64",
            _ => return None,
        };
        return Some(Decoded {
            mnemonic,
            op,
            dst: w & 0xFF,
            src: [w1 & 0x1FF, (w1 >> 9) & 0x1FF, (w1 >> 18) & 0x1FF],
            extra: w1 >> 29,
            words: 2,
        });
    }

    // SOPP: [31:23] == 0b101111111
    if w >> 23 == 0b1_0111_1111 {
        let op = (w >> 16) & 0x7F;
        let mnemonic = match op {
            OP_S_ENDPGM => "s_endpgm",
            OP_S_CBRANCH_EXECZ => "s_cbranch_execz",
            OP_S_WAITCNT => "s_waitcnt",
            _ => return None,
        };
        return Some(Decoded {
            mnemonic,
            op,
            dst: 0,
            src: [0, 0, 0],
            extra: w & 0xFFFF,
            words: 1,
        });
    }

    // SOP1: [31:23] == 0b101111101
    if w >> 23 == 0b1_0111_1101 {
        let op = (w >> 8) & 0xFF;
        let mnemonic = match op {
            OP_S_AND_SAVEEXEC_B64 => "s_and_saveexec_b64",
            _ => return None,
        };
        return Some(Decoded {
            mnemonic,
            op,
            dst: (w >> 16) & 0x7F,
            src: [w & 0xFF, 0, 0],
            extra: 0,
            words: 1,
        });
    }

    // SOP2: [31:30] == 0b10 (checked after the SOP1/SOPP prefixes, which are
    // longer and would otherwise be swallowed by this test).
    if w >> 30 == 0b10 {
        let op = (w >> 23) & 0x7F;
        let mnemonic = match op {
            OP_S_LSHL_B32 => "s_lshl_b32",
            _ => return None,
        };
        return Some(Decoded {
            mnemonic,
            op,
            dst: (w >> 16) & 0x7F,
            src: [w & 0xFF, (w >> 8) & 0xFF, 0],
            extra: 0,
            words: 1,
        });
    }

    // VOP1: [31:25] == 0b0111111
    if w >> 25 == 0b011_1111 {
        let op = (w >> 9) & 0xFF;
        let mnemonic = match op {
            OP_V_MOV_B32 => "v_mov_b32_e32",
            _ => return None,
        };
        return Some(Decoded {
            mnemonic,
            op,
            dst: (w >> 17) & 0xFF,
            src: [w & 0x1FF, 0, 0],
            extra: 0,
            words: 1,
        });
    }

    // VOPC: [31:25] == 0b0111110
    if w >> 25 == 0b011_1110 {
        let op = (w >> 17) & 0xFF;
        let mnemonic = match op {
            OP_V_CMP_GT_I32 => "v_cmp_gt_i32_e32",
            _ => return None,
        };
        return Some(Decoded {
            mnemonic,
            op,
            dst: VCC,
            src: [w & 0x1FF, (w >> 9) & 0xFF, 0],
            extra: 0,
            words: 1,
        });
    }

    // VOP2: [31] == 0 (last, because VOP1/VOPC share the leading zero bit).
    if w >> 31 == 0 {
        let op = (w >> 25) & 0x3F;
        let mnemonic = match op {
            OP_V_ADD_U32 => "v_add_u32_e32",
            OP_V_LSHLREV_B32 => "v_lshlrev_b32_e32",
            _ => return None,
        };
        return Some(Decoded {
            mnemonic,
            op,
            dst: (w >> 17) & 0xFF,
            src: [w & 0x1FF, (w >> 9) & 0xFF, 0],
            extra: 0,
            words: 1,
        });
    }

    None
}

/// Re-encode a [`Decoded`] instruction back to its word form.
///
/// Used by the round-trip test; returns `(words, count)`.
pub fn reencode(d: &Decoded) -> ([u32; 2], usize) {
    match d.mnemonic {
        "s_lshl_b32" => ([sop2(d.op, d.dst, d.src[0], d.src[1]), 0], 1),
        "s_and_saveexec_b64" => ([sop1(d.op, d.dst, d.src[0]), 0], 1),
        "s_endpgm" | "s_cbranch_execz" | "s_waitcnt" => ([sopp(d.op, d.extra as u16), 0], 1),
        "v_mov_b32_e32" => ([vop1(d.op, d.dst, d.src[0]), 0], 1),
        "v_add_u32_e32" | "v_lshlrev_b32_e32" => ([vop2(d.op, d.dst, d.src[1], d.src[0]), 0], 1),
        "v_cmp_gt_i32_e32" => ([vopc(d.op, d.src[1], d.src[0]), 0], 1),
        "v_add_f64" | "v_mul_f64" => (vop3a(d.op, d.dst, d.src[0], d.src[1], d.src[2], d.extra), 2),
        "global_load_dwordx2" | "global_store_dwordx2" => (
            flat(
                d.op, SEG_GLOBAL, d.src[2], d.src[0], d.src[1], d.extra, d.dst,
            ),
            2,
        ),
        _ => ([0, 0], 0),
    }
}

// ------------------------------------------------------------------
// What is deliberately not here
// ------------------------------------------------------------------

/// Kernels from `shaders/*.cl` that intentionally have no `.bin` yet, with the
/// reason each was not attempted.  These names must stay in sync with the
/// zero-length placeholders that `build.rs` skips.
pub const NOT_IMPLEMENTED: [(&str, &str); 3] = [
    (
        "voronoi_assign",
        "needs sin/cos/acos; GFX9 V_SIN_F32/V_COS_F32 are low-precision and \
         acos has no hardware instruction, so a faithful port needs a \
         polynomial approximation plus an n_cells loop that cannot be \
         validated for numerical agreement without an AMD GPU",
    ),
    (
        "jl_project",
        "divides a double by the constant 2147483647.0; IEEE-correct f64 \
         division on GFX9 is a V_RCP_F64 + Newton + V_DIV_FMAS_F64 sequence \
         whose edge cases cannot be checked without an AMD GPU, and \
         multiplying by a reciprocal would change the result",
    ),
    (
        "s2_distance",
        "no OpenCL source is checked in for this kernel name",
    ),
];
