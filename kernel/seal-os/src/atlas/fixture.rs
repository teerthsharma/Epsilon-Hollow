// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! The embedded proof chart.
//!
//! # How this object is produced
//!
//! There is no cross-linker in the build, so the ELF64 `ET_REL` object is
//! emitted byte-for-byte by [`build_chart`] below. The generator *is* the
//! source of truth: it is checked in, deterministic, and takes no input except
//! the name of the germ the chart imports. Anyone can regenerate the exact same
//! bytes by calling `build_chart("atlas_germ_probe")`, and a reviewer can dump
//! them and run `readelf -a` / `objdump -dr` over the result.
//!
//! To regenerate and verify on a host: copy this file into a `std` binary
//! crate, drop the two `alloc` imports, replace the `super::relobj` import with
//! the four `R_X86_64_*` constants, add
//! `fn main() { std::fs::write("proof.chart.o", proof_chart()).unwrap(); }`,
//! then `cargo run && llvm-readobj --file-headers --section-headers --symbols
//! --relocations proof.chart.o && llvm-objdump -dr --section=.text
//! proof.chart.o`. The object is 1120 bytes; the disassembly must match the
//! listing below line for line, and the relocation table must read exactly
//! `.rela.text {PLT32 atlas_germ_probe -4, PC32 .data +4, PC32 .data -4,
//! PC32 .text -4}` and `.rela.data {64 .text +0, 32S SEAL_CHART_ABI +0}`.
//!
//! The machine code in [`TEXT`] was hand-assembled from this listing; the
//! comment on each line is the byte encoding, so the array can be checked
//! against any x86_64 assembler:
//!
//! ```text
//! chart_init:                     ; .text + 0x00
//!   0x00  55                      push rbp
//!   0x01  48 89 e5                mov  rbp, rsp
//!   0x04  e8 <rel32>              call atlas_germ_probe   ; R_X86_64_PLT32 @0x05, A=-4
//!   0x09  8b 0d <rel32>           mov  ecx, [rip+chart_state+8] ; R_X86_64_PC32 @0x0b, A=+4
//!   0x0f  01 c8                   add  eax, ecx
//!   0x11  48 98                   cdqe
//!   0x13  5d                      pop  rbp
//!   0x14  c3                      ret
//!                                 ; returns germ_probe() + *(u32*)(chart_state+8)
//!                                 ;       = 0x5EA1_0000 + 0x42 = 0x5EA1_0042
//!
//! chart_exit:                     ; .text + 0x15
//!   0x15  55                      push rbp
//!   0x16  48 89 e5                mov  rbp, rsp
//!   0x19  48 8b 05 <rel32>        mov  rax, [rip+chart_state+0] ; R_X86_64_PC32 @0x1c, A=-4
//!   0x20  48 8d 15 <rel32>        lea  rdx, [rip+chart_init]    ; R_X86_64_PC32 @0x23, A=-4
//!   0x27  48 39 d0                cmp  rax, rdx
//!   0x2a  0f 95 c0                setne al
//!   0x2d  0f b6 c0                movzx eax, al
//!   0x30  f7 d8                   neg  eax
//!   0x32  48 98                   cdqe
//!   0x34  5d                      pop  rbp
//!   0x35  c3                      ret
//!                                 ; returns 0 iff the R_X86_64_64 slot in .data
//!                                 ; really holds &chart_init, else -1
//! ```
//!
//! `.data` is 16 bytes: `[0..8)` is an 8-byte slot relocated with
//! `R_X86_64_64` against the `.text` section symbol (so it ends up holding
//! `&chart_init`), `[8..12)` is a 4-byte slot relocated with `R_X86_64_32S`
//! against the absolute symbol `SEAL_CHART_ABI = 0x42`, `[12..16)` is padding.
//!
//! Between them the two entry points therefore *observe* all four relocation
//! classes: their return codes are wrong unless `R_X86_64_PLT32`,
//! `R_X86_64_PC32`, `R_X86_64_64` and `R_X86_64_32S` were all applied
//! correctly. That is what makes the proof line falsifiable.

use alloc::vec;
use alloc::vec::Vec;

use super::relobj::{R_X86_64_32S, R_X86_64_64, R_X86_64_PC32, R_X86_64_PLT32};

pub const PROOF_CHART_NAME: &str = "seal-proof-chart";
pub const PROOF_GERM_NAME: &str = "atlas_germ_probe";
/// Value the exported germ returns; the low half of the expected init code.
pub const GERM_PROBE_VALUE: i64 = 0x5EA1_0000;
/// Value of the `SHN_ABS` symbol the `R_X86_64_32S` relocation resolves to.
pub const CHART_ABI_ABS_VALUE: u64 = 0x42;
/// `chart_init` returns this iff every relocation landed.
pub const PROOF_INIT_EXPECT: i64 = GERM_PROBE_VALUE + CHART_ABI_ABS_VALUE as i64;
/// `chart_exit` returns this iff the `R_X86_64_64` slot holds `&chart_init`.
pub const PROOF_EXIT_EXPECT: i64 = 0;

const TEXT: [u8; 0x36] = [
    // chart_init
    0x55, // push rbp
    0x48, 0x89, 0xe5, // mov rbp, rsp
    0xe8, 0x00, 0x00, 0x00, 0x00, // call rel32   (reloc @0x05)
    0x8b, 0x0d, 0x00, 0x00, 0x00, 0x00, // mov ecx, [rip+d32] (reloc @0x0b)
    0x01, 0xc8, // add eax, ecx
    0x48, 0x98, // cdqe
    0x5d, // pop rbp
    0xc3, // ret
    // chart_exit
    0x55, // push rbp
    0x48, 0x89, 0xe5, // mov rbp, rsp
    0x48, 0x8b, 0x05, 0x00, 0x00, 0x00, 0x00, // mov rax, [rip+d32] (reloc @0x1c)
    0x48, 0x8d, 0x15, 0x00, 0x00, 0x00, 0x00, // lea rdx, [rip+d32] (reloc @0x23)
    0x48, 0x39, 0xd0, // cmp rax, rdx
    0x0f, 0x95, 0xc0, // setne al
    0x0f, 0xb6, 0xc0, // movzx eax, al
    0xf7, 0xd8, // neg eax
    0x48, 0x98, // cdqe
    0x5d, // pop rbp
    0xc3, // ret
];

const INIT_OFF: u64 = 0x00;
const EXIT_OFF: u64 = 0x15;
const DATA_SIZE: u64 = 16;

fn push_name(tab: &mut Vec<u8>, name: &str) -> u32 {
    let off = tab.len() as u32;
    tab.extend_from_slice(name.as_bytes());
    tab.push(0);
    off
}

fn push_sym(out: &mut Vec<u8>, name: u32, info: u8, shndx: u16, value: u64, size: u64) {
    out.extend_from_slice(&name.to_le_bytes());
    out.push(info);
    out.push(0); // st_other
    out.extend_from_slice(&shndx.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
}

fn push_rela(out: &mut Vec<u8>, offset: u64, sym: u32, rtype: u32, addend: i64) {
    out.extend_from_slice(&offset.to_le_bytes());
    out.extend_from_slice(&(((sym as u64) << 32) | rtype as u64).to_le_bytes());
    out.extend_from_slice(&addend.to_le_bytes());
}

#[allow(clippy::too_many_arguments)] // REASON: mirrors the 10-field ELF64 Shdr layout
fn push_shdr(
    out: &mut Vec<u8>,
    name: u32,
    kind: u32,
    flags: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    addralign: u64,
    entsize: u64,
) {
    out.extend_from_slice(&name.to_le_bytes());
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // sh_addr
    out.extend_from_slice(&offset.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&link.to_le_bytes());
    out.extend_from_slice(&info.to_le_bytes());
    out.extend_from_slice(&addralign.to_le_bytes());
    out.extend_from_slice(&entsize.to_le_bytes());
}

/// Append `payload` at the next 8-byte boundary; return the offset it landed at.
fn place(file: &mut Vec<u8>, payload: &[u8]) -> u64 {
    while file.len() % 8 != 0 {
        file.push(0);
    }
    let off = file.len() as u64;
    file.extend_from_slice(payload);
    off
}

/// Emit the proof chart as a complete ELF64 `ET_REL` object.
///
/// `germ_name` is the single undefined symbol the chart imports. Passing a name
/// the kernel does not publish yields a valid object that must fail to load —
/// that is how the unresolved-symbol path is exercised.
pub fn build_chart(germ_name: &str) -> Vec<u8> {
    let mut strtab = vec![0u8];
    let n_init = push_name(&mut strtab, "chart_init");
    let n_exit = push_name(&mut strtab, "chart_exit");
    let n_state = push_name(&mut strtab, "chart_state");
    let n_germ = push_name(&mut strtab, germ_name);
    let n_abi = push_name(&mut strtab, "SEAL_CHART_ABI");

    let mut shstrtab = vec![0u8];
    let s_text = push_name(&mut shstrtab, ".text");
    let s_rela_text = push_name(&mut shstrtab, ".rela.text");
    let s_data = push_name(&mut shstrtab, ".data");
    let s_rela_data = push_name(&mut shstrtab, ".rela.data");
    let s_symtab = push_name(&mut shstrtab, ".symtab");
    let s_strtab = push_name(&mut shstrtab, ".strtab");
    let s_shstrtab = push_name(&mut shstrtab, ".shstrtab");

    // Symbol indices: 0 null, 1 .text, 2 .data, 3 init, 4 exit, 5 state,
    // 6 germ (UNDEF), 7 SEAL_CHART_ABI (ABS). First global is index 3.
    let mut symtab = Vec::new();
    push_sym(&mut symtab, 0, 0x00, 0, 0, 0);
    push_sym(&mut symtab, 0, 0x03, 1, 0, 0); // LOCAL SECTION -> .text
    push_sym(&mut symtab, 0, 0x03, 3, 0, 0); // LOCAL SECTION -> .data
    push_sym(&mut symtab, n_init, 0x12, 1, INIT_OFF, EXIT_OFF); // GLOBAL FUNC
    push_sym(
        &mut symtab,
        n_exit,
        0x12,
        1,
        EXIT_OFF,
        TEXT.len() as u64 - EXIT_OFF,
    );
    push_sym(&mut symtab, n_state, 0x11, 3, 0, DATA_SIZE); // GLOBAL OBJECT
    push_sym(&mut symtab, n_germ, 0x10, 0, 0, 0); // GLOBAL NOTYPE, SHN_UNDEF
    push_sym(&mut symtab, n_abi, 0x10, 0xFFF1, CHART_ABI_ABS_VALUE, 0); // SHN_ABS

    let mut rela_text = Vec::new();
    push_rela(&mut rela_text, 0x05, 6, R_X86_64_PLT32, -4);
    push_rela(&mut rela_text, 0x0b, 2, R_X86_64_PC32, 4); // .data + 8 - 4
    push_rela(&mut rela_text, 0x1c, 2, R_X86_64_PC32, -4); // .data + 0 - 4
    push_rela(&mut rela_text, 0x23, 1, R_X86_64_PC32, -4); // .text + 0 - 4

    let mut rela_data = Vec::new();
    push_rela(&mut rela_data, 0x00, 1, R_X86_64_64, 0);
    push_rela(&mut rela_data, 0x08, 7, R_X86_64_32S, 0);

    let mut file: Vec<u8> = vec![0u8; 64]; // ELF header, patched at the end
    let off_text = place(&mut file, &TEXT);
    let off_rela_text = place(&mut file, &rela_text);
    let off_data = place(&mut file, &[0u8; DATA_SIZE as usize]);
    let off_rela_data = place(&mut file, &rela_data);
    let off_symtab = place(&mut file, &symtab);
    let off_strtab = place(&mut file, &strtab);
    let off_shstrtab = place(&mut file, &shstrtab);

    while file.len() % 8 != 0 {
        file.push(0);
    }
    let shoff = file.len() as u64;
    let mut sh = Vec::new();
    push_shdr(&mut sh, 0, 0, 0, 0, 0, 0, 0, 0, 0); // 0 NULL
    push_shdr(
        &mut sh,
        s_text,
        1,
        0x6,
        off_text,
        TEXT.len() as u64,
        0,
        0,
        16,
        0,
    );
    push_shdr(
        &mut sh,
        s_rela_text,
        4,
        0,
        off_rela_text,
        rela_text.len() as u64,
        5,
        1,
        8,
        24,
    );
    push_shdr(&mut sh, s_data, 1, 0x3, off_data, DATA_SIZE, 0, 0, 8, 0);
    push_shdr(
        &mut sh,
        s_rela_data,
        4,
        0,
        off_rela_data,
        rela_data.len() as u64,
        5,
        3,
        8,
        24,
    );
    push_shdr(
        &mut sh,
        s_symtab,
        2,
        0,
        off_symtab,
        symtab.len() as u64,
        6,
        3,
        8,
        24,
    );
    push_shdr(
        &mut sh,
        s_strtab,
        3,
        0,
        off_strtab,
        strtab.len() as u64,
        0,
        0,
        1,
        0,
    );
    push_shdr(
        &mut sh,
        s_shstrtab,
        3,
        0,
        off_shstrtab,
        shstrtab.len() as u64,
        0,
        0,
        1,
        0,
    );
    file.extend_from_slice(&sh);

    // ELF64 header
    file[0..4].copy_from_slice(b"\x7FELF");
    file[4] = 2; // ELFCLASS64
    file[5] = 1; // ELFDATA2LSB
    file[6] = 1; // EV_CURRENT
    file[16..18].copy_from_slice(&1u16.to_le_bytes()); // e_type = ET_REL
    file[18..20].copy_from_slice(&62u16.to_le_bytes()); // e_machine = EM_X86_64
    file[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
    file[40..48].copy_from_slice(&shoff.to_le_bytes()); // e_shoff
    file[52..54].copy_from_slice(&64u16.to_le_bytes()); // e_ehsize
    file[58..60].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
    file[60..62].copy_from_slice(&8u16.to_le_bytes()); // e_shnum
    file[62..64].copy_from_slice(&7u16.to_le_bytes()); // e_shstrndx
    file
}

/// The proof chart as grafted at boot.
pub fn proof_chart() -> Vec<u8> {
    build_chart(PROOF_GERM_NAME)
}

/// A well-formed header over a body that stops mid-object.
pub fn truncated_chart() -> Vec<u8> {
    let mut bytes = proof_chart();
    bytes.truncate(96);
    bytes
}

/// A structurally valid object importing a germ the kernel never publishes.
pub fn unresolved_chart() -> Vec<u8> {
    build_chart("atlas_germ_that_does_not_exist")
}
