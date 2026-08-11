// Seal OS - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ELF64 loader for executables and simple shared objects.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use x86_64::structures::paging::{PageTable, PageTableFlags};
use x86_64::{PhysAddr, VirtAddr};

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_INTERP: u32 = 3;

const DT_NULL: u64 = 0;
const DT_NEEDED: u64 = 1;
const DT_STRTAB: u64 = 5;
const DT_RELA: u64 = 7;
const DT_RELASZ: u64 = 8;
const DT_RELAENT: u64 = 9;
const DT_STRSZ: u64 = 10;

const R_X86_64_RELATIVE: u64 = 8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElfError {
    InvalidMagic,
    InvalidClass,
    InvalidType,
    TooSmall,
    LoadFailed,
    MissingDependency,
    /// A program-header field (address arithmetic or segment size) failed
    /// validation before it could reach page-table or allocator code.
    InvalidSegment,
}

#[derive(Clone, Copy)]
struct ProgramHeader {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_filesz: u64,
    p_memsz: u64,
}

pub struct DynamicLinkInfo {
    pub interpreter: Option<String>,
    pub needed: Vec<String>,
    rela_addr: u64,
    rela_size: u64,
    rela_ent: u64,
}

impl DynamicLinkInfo {
    fn empty() -> Self {
        Self {
            interpreter: None,
            needed: Vec::new(),
            rela_addr: 0,
            rela_size: 0,
            rela_ent: 24,
        }
    }
}

pub struct LoadedElf {
    pub entry_point: u64,
    pub stack_pointer: u64,
    pub page_table: u64,
    pub file_mode: u16,
    pub file_uid: u32,
    pub file_gid: u32,
    pub dynamic: DynamicLinkInfo,
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16, ElfError> {
    let bytes = data.get(offset..offset + 2).ok_or(ElfError::TooSmall)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, ElfError> {
    let bytes = data.get(offset..offset + 4).ok_or(ElfError::TooSmall)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64, ElfError> {
    let bytes = data.get(offset..offset + 8).ok_or(ElfError::TooSmall)?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn read_i64(data: &[u8], offset: usize) -> Result<i64, ElfError> {
    Ok(read_u64(data, offset)? as i64)
}

fn validate_elf64(elf_data: &[u8]) -> Result<u16, ElfError> {
    if !elf_data.get(0..4).map(|s| s == b"\x7FELF").unwrap_or(false) {
        return Err(ElfError::InvalidMagic);
    }
    if elf_data.get(4).copied().unwrap_or(0) != 2 {
        return Err(ElfError::InvalidClass);
    }
    let e_type = read_u16(elf_data, 16)?;
    if e_type != 2 && e_type != 3 {
        return Err(ElfError::InvalidType);
    }
    Ok(e_type)
}

pub fn load(
    elf_data: &[u8],
    aslr_base: u64,
    file_mode: u16,
    file_uid: u32,
    file_gid: u32,
) -> Result<LoadedElf, ElfError> {
    let e_type = validate_elf64(elf_data)?;
    let entry = read_u64(elf_data, 24)?;
    let headers = read_program_headers(elf_data)?;

    let base = if e_type == 3 { aslr_base } else { 0 };
    let dynamic = parse_dynamic_link_info(elf_data, &headers)?;

    // Tracks every frame allocated below (PML4, segment pages, stack pages)
    // and frees them all on any early return. See `LoadRollback` for why.
    let mut rollback = LoadRollback::new();

    let pml4_frame = crate::memory::phys::alloc_frame().ok_or(ElfError::LoadFailed)?;
    rollback.track(pml4_frame);
    let pml4 = unsafe { &mut *(pml4_frame.as_u64() as *mut PageTable) };
    pml4.zero();

    unsafe {
        let kernel_pml4_virt = crate::memory::virt::current_pml4_virt();
        let kernel_pml4 = &*(kernel_pml4_virt.as_u64() as *const PageTable);
        for i in 256..512 {
            pml4[i] = kernel_pml4[i].clone();
        }
    }

    map_load_segments(elf_data, &headers, base, pml4, &mut rollback)?;
    apply_relative_relocations(elf_data, &headers, &dynamic, base, pml4_frame.as_u64())?;

    const USER_STACK_PAGES: usize = 4;
    const USER_STACK_TOP: u64 = 0x0000_7fff_ffff_0000;
    let mut user_stack_top_frame = None;

    for i in 0..USER_STACK_PAGES {
        let frame = crate::memory::phys::alloc_frame().ok_or(ElfError::LoadFailed)?;
        rollback.track(frame);
        unsafe {
            core::ptr::write_bytes(frame.as_u64() as *mut u8, 0, 4096);
        }
        if i == USER_STACK_PAGES - 1 {
            user_stack_top_frame = Some(frame.as_u64());
        }
        let virt = VirtAddr::new(USER_STACK_TOP - ((USER_STACK_PAGES - i) as u64) * 4096);
        unsafe {
            if crate::memory::virt::map_page_to_pml4(
                virt,
                frame,
                PageTableFlags::PRESENT
                    | PageTableFlags::WRITABLE
                    | PageTableFlags::USER_ACCESSIBLE,
                pml4,
            )
            .is_err()
            {
                // `frame` stays tracked in `rollback` — it is freed when
                // `rollback` drops on this early return, same as every
                // other frame allocated so far in this call.
                return Err(ElfError::LoadFailed);
            }
        }
    }

    let top_frame = user_stack_top_frame.ok_or(ElfError::LoadFailed)?;
    let mut sp = USER_STACK_TOP;
    sp = push_stack_u64(sp, USER_STACK_TOP, top_frame, 0)?;
    sp = push_stack_u64(sp, USER_STACK_TOP, top_frame, 0)?;
    sp = push_stack_u64(sp, USER_STACK_TOP, top_frame, 0)?;

    // Everything succeeded: the frames now belong to the new address space.
    rollback.commit();

    Ok(LoadedElf {
        entry_point: entry + base,
        stack_pointer: sp,
        page_table: pml4_frame.as_u64(),
        file_mode,
        file_uid,
        file_gid,
        dynamic,
    })
}

pub fn load_shared_object(
    elf_data: &[u8],
    aslr_base: u64,
    page_table: u64,
) -> Result<DynamicLinkInfo, ElfError> {
    let e_type = validate_elf64(elf_data)?;
    if e_type != 3 {
        return Err(ElfError::InvalidType);
    }
    let headers = read_program_headers(elf_data)?;
    let dynamic = parse_dynamic_link_info(elf_data, &headers)?;
    let pml4 = unsafe { &mut *(page_table as *mut PageTable) };

    // `page_table` is owned by the caller (it already exists), so unlike
    // `load()` this rollback never tracks a PML4 frame — only the segment
    // frames this call allocates. On failure those are freed; on success
    // they become part of the caller's live address space.
    let mut rollback = LoadRollback::new();
    map_load_segments(elf_data, &headers, aslr_base, pml4, &mut rollback)?;
    apply_relative_relocations(elf_data, &headers, &dynamic, aslr_base, page_table)?;
    rollback.commit();
    Ok(dynamic)
}

pub fn load_dynamic_dependencies(
    dynamic: &DynamicLinkInfo,
    page_table: u64,
) -> Result<(), ElfError> {
    let mut slot = 1u64;

    if let Some(interpreter) = dynamic.interpreter.as_ref() {
        if let Some(bytes) = read_vfs_file(interpreter) {
            let base = crate::security::aslr::randomize_mmap_base().wrapping_add(slot << 28);
            let nested = load_shared_object(&bytes, base, page_table)?;
            load_needed_objects(&nested, page_table, &mut slot)?;
            slot += 1;
        } else {
            return Err(ElfError::MissingDependency);
        }
    }

    load_needed_objects(dynamic, page_table, &mut slot)
}

fn load_needed_objects(
    dynamic: &DynamicLinkInfo,
    page_table: u64,
    slot: &mut u64,
) -> Result<(), ElfError> {
    for name in dynamic.needed.iter() {
        let bytes = read_shared_object(name).ok_or(ElfError::MissingDependency)?;
        let base = crate::security::aslr::randomize_mmap_base().wrapping_add(*slot << 28);
        let nested = load_shared_object(&bytes, base, page_table)?;
        *slot += 1;

        for nested_name in nested.needed.iter() {
            if dynamic.needed.iter().any(|known| known == nested_name) {
                continue;
            }
            let nested_bytes =
                read_shared_object(nested_name).ok_or(ElfError::MissingDependency)?;
            let nested_base =
                crate::security::aslr::randomize_mmap_base().wrapping_add(*slot << 28);
            let _ = load_shared_object(&nested_bytes, nested_base, page_table)?;
            *slot += 1;
        }
    }
    Ok(())
}

fn read_shared_object(name: &str) -> Option<Vec<u8>> {
    if name.starts_with('/') {
        return read_vfs_file(name);
    }
    read_vfs_file(&format!("/lib/{}", name))
        .or_else(|| read_vfs_file(&format!("/usr/lib/{}", name)))
}

fn read_vfs_file(path: &str) -> Option<Vec<u8>> {
    let handle = crate::fs::vfs::with_vfs(|vfs| vfs.lookup_follow(path)).ok()?;
    let mut out = Vec::new();
    let mut offset = 0u64;
    let mut chunk = [0u8; 4096];
    loop {
        match crate::fs::vfs::with_vfs(|vfs| vfs.read(handle, &mut chunk, offset)) {
            Ok(0) => break,
            Ok(n) => {
                out.extend_from_slice(&chunk[..n]);
                offset += n as u64;
            }
            Err(_) => return None,
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn read_program_headers(elf_data: &[u8]) -> Result<Vec<ProgramHeader>, ElfError> {
    let phoff = read_u64(elf_data, 32)?;
    let phentsize = read_u16(elf_data, 54)?;
    let phnum = read_u16(elf_data, 56)?;
    let mut headers = Vec::new();
    for i in 0..phnum {
        let off = phoff
            .checked_add(
                (i as u64)
                    .checked_mul(phentsize as u64)
                    .ok_or(ElfError::TooSmall)?,
            )
            .ok_or(ElfError::TooSmall)?;
        let start = off as usize;
        let end = start.checked_add(56).ok_or(ElfError::TooSmall)?;
        let ph = elf_data.get(start..end).ok_or(ElfError::TooSmall)?;
        headers.push(ProgramHeader {
            p_type: read_u32(ph, 0)?,
            p_flags: read_u32(ph, 4)?,
            p_offset: read_u64(ph, 8)?,
            p_vaddr: read_u64(ph, 16)?,
            p_filesz: read_u64(ph, 32)?,
            p_memsz: read_u64(ph, 40)?,
        });
    }
    Ok(headers)
}

/// Frees every physical frame it was told about, in reverse allocation
/// order, unless [`commit`](Self::commit) is called first.
///
/// Mirrors `memory::virt::CowCloneRollback`. `load` and `load_shared_object`
/// can fail after allocating the PML4 frame and/or any number of PT_LOAD
/// segment/stack frames — OOM, a rejected mapping, a bad relocation — and
/// every `?` after such an allocation used to return straight past it with
/// no cleanup, leaking a physical frame per allocation on every failed exec
/// of a malformed binary. Tracking each frame here and committing only on
/// the single success path covers every current and future early return
/// without hand-auditing each one.
struct LoadRollback {
    frames: Vec<PhysAddr>,
    committed: bool,
}

impl LoadRollback {
    fn new() -> Self {
        Self {
            frames: Vec::new(),
            committed: false,
        }
    }

    fn track(&mut self, frame: PhysAddr) {
        self.frames.push(frame);
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for LoadRollback {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for frame in self.frames.drain(..).rev() {
            unsafe {
                crate::memory::phys::free_frame(frame);
            }
        }
    }
}

/// Validate a PT_LOAD segment's address range before it drives any page
/// allocation or `VirtAddr` construction.
///
/// `p_vaddr`, `base`, and `p_memsz` are attacker-controlled: `p_vaddr` and
/// `p_memsz` come straight from a program header in a binary passed to
/// `spawn_user`/`exec`, and `base` is only trusted for ET_EXEC (0) — for
/// ET_DYN it is ASLR-chosen but the *addition* still needs checking. Two
/// separate hazards live in the two lines this replaces:
///
/// - `ph.p_vaddr + base` and `seg_start + ph.p_memsz` were plain `u64`
///   additions that can wrap, and the result feeds `VirtAddr::new` (in the
///   caller's page loop), which *panics* on a non-canonical address. This
///   process runs with `panic = "abort"`, so a single malformed header
///   would abort the whole kernel. `checked_add` plus `VirtAddr::try_new`
///   turn that into a rejected load.
/// - `p_memsz` had no upper bound, so a crafted segment could drive the
///   per-page allocation loop to try to allocate most of physical RAM
///   before any failure (including the leak in Defect B) was reached. The
///   bound here is the allocator's own currently-free frame count
///   (`memory::phys::free_count`) — a segment can never legitimately need
///   more memory than the system currently has free, and checking this
///   once up front turns a long allocate/zero/fail churn into a single
///   rejection.
fn validate_segment_range(p_vaddr: u64, base: u64, p_memsz: u64) -> Result<(u64, u64), ElfError> {
    let max_bytes = (crate::memory::phys::free_count() as u64).saturating_mul(4096);
    if p_memsz > max_bytes {
        return Err(ElfError::InvalidSegment);
    }
    let seg_start = p_vaddr.checked_add(base).ok_or(ElfError::InvalidSegment)?;
    let seg_end = seg_start
        .checked_add(p_memsz)
        .ok_or(ElfError::InvalidSegment)?;
    VirtAddr::try_new(seg_start).map_err(|_| ElfError::InvalidSegment)?;
    VirtAddr::try_new(seg_end).map_err(|_| ElfError::InvalidSegment)?;
    Ok((seg_start, seg_end))
}

fn map_load_segments(
    elf_data: &[u8],
    headers: &[ProgramHeader],
    base: u64,
    pml4: &mut PageTable,
    rollback: &mut LoadRollback,
) -> Result<(), ElfError> {
    for ph in headers {
        if ph.p_type != PT_LOAD {
            continue;
        }

        let mut flags = PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE;
        if ph.p_flags & 0x2 != 0 {
            flags |= PageTableFlags::WRITABLE;
        }
        if ph.p_flags & 0x1 == 0 {
            flags |= PageTableFlags::NO_EXECUTE;
        }

        let (seg_start, seg_end) = validate_segment_range(ph.p_vaddr, base, ph.p_memsz)?;
        let filesz_end = seg_start
            .checked_add(ph.p_filesz)
            .ok_or(ElfError::InvalidSegment)?;
        let first_page = seg_start / 4096;
        let last_page = seg_end.div_ceil(4096);

        for page in first_page..last_page {
            let page_virt = VirtAddr::new(page * 4096);
            let frame = crate::memory::phys::alloc_frame().ok_or(ElfError::LoadFailed)?;
            rollback.track(frame);
            unsafe {
                core::ptr::write_bytes(frame.as_u64() as *mut u8, 0, 4096);
            }

            let page_start = page * 4096;
            let page_end = page_start + 4096;
            let seg_page_start = seg_start.max(page_start);
            let file_backed_end = filesz_end.min(page_end);
            let seg_page_end = file_backed_end.min(seg_end);
            let len = seg_page_end.saturating_sub(seg_page_start);

            if len > 0 {
                let frame_offset = (seg_page_start - page_start) as usize;
                let file_offset = ph
                    .p_offset
                    .checked_add(seg_page_start - seg_start)
                    .ok_or(ElfError::TooSmall)?;
                let file_start = file_offset as usize;
                let file_end = file_start
                    .checked_add(len as usize)
                    .ok_or(ElfError::TooSmall)?;
                let src = elf_data
                    .get(file_start..file_end)
                    .ok_or(ElfError::TooSmall)?;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        (frame.as_u64() + frame_offset as u64) as *mut u8,
                        len as usize,
                    );
                }
            }

            unsafe {
                if crate::memory::virt::map_page_to_pml4(page_virt, frame, flags, pml4).is_err() {
                    // `frame` stays tracked in `rollback` — freed when the
                    // caller's rollback drops on this early return.
                    return Err(ElfError::LoadFailed);
                }
            }
        }
    }
    Ok(())
}

fn parse_dynamic_link_info(
    elf_data: &[u8],
    headers: &[ProgramHeader],
) -> Result<DynamicLinkInfo, ElfError> {
    let mut info = DynamicLinkInfo::empty();

    if let Some(interp) = headers.iter().find(|ph| ph.p_type == PT_INTERP) {
        let start = interp.p_offset as usize;
        let end = start
            .checked_add(interp.p_filesz as usize)
            .ok_or(ElfError::TooSmall)?;
        info.interpreter = read_c_string(elf_data.get(start..end).ok_or(ElfError::TooSmall)?);
    }

    let Some(dynamic) = headers.iter().find(|ph| ph.p_type == PT_DYNAMIC) else {
        return Ok(info);
    };

    let dyn_start = dynamic.p_offset as usize;
    let dyn_end = dyn_start
        .checked_add(dynamic.p_filesz as usize)
        .ok_or(ElfError::TooSmall)?;
    let dyn_bytes = elf_data.get(dyn_start..dyn_end).ok_or(ElfError::TooSmall)?;

    let mut strtab = 0;
    let mut strsz = 0;
    let mut needed_offsets = Vec::new();

    for off in (0..dyn_bytes.len()).step_by(16) {
        if off + 16 > dyn_bytes.len() {
            break;
        }
        let tag = read_u64(dyn_bytes, off)?;
        let val = read_u64(dyn_bytes, off + 8)?;
        match tag {
            DT_NULL => break,
            DT_NEEDED => needed_offsets.push(val),
            DT_STRTAB => strtab = val,
            DT_STRSZ => strsz = val,
            DT_RELA => info.rela_addr = val,
            DT_RELASZ => info.rela_size = val,
            DT_RELAENT => info.rela_ent = val.max(24),
            _ => {
                // Unknown dynamic tag; skip per ELF spec
            }
        }
    }

    if strtab != 0 && strsz != 0 {
        for off in needed_offsets {
            let Some(file_off) = va_to_file_offset(headers, strtab + off) else {
                continue;
            };
            let max_len = strsz.saturating_sub(off) as usize;
            if let Some(bytes) = elf_data.get(file_off..file_off.saturating_add(max_len)) {
                if let Some(name) = read_c_string(bytes) {
                    info.needed.push(name);
                }
            }
        }
    }

    Ok(info)
}

fn apply_relative_relocations(
    elf_data: &[u8],
    headers: &[ProgramHeader],
    dynamic: &DynamicLinkInfo,
    base: u64,
    pml4_phys: u64,
) -> Result<(), ElfError> {
    if dynamic.rela_addr == 0 || dynamic.rela_size == 0 {
        return Ok(());
    }
    let Some(rela_file_off) = va_to_file_offset(headers, dynamic.rela_addr) else {
        return Err(ElfError::TooSmall);
    };
    let count = dynamic.rela_size / dynamic.rela_ent.max(24);
    for i in 0..count {
        let off = rela_file_off + (i * dynamic.rela_ent) as usize;
        let r_offset = read_u64(elf_data, off)?;
        let r_info = read_u64(elf_data, off + 8)?;
        let r_addend = read_i64(elf_data, off + 16)?;
        let r_type = r_info & 0xffff_ffff;
        if r_type != R_X86_64_RELATIVE {
            continue;
        }
        // `r_offset` comes straight from an Elf64_Rela entry in the object
        // (attacker-controlled, same as p_vaddr above): `base + r_offset`
        // must not wrap, and the result must be canonical before it reaches
        // `VirtAddr`, which panics on a non-canonical address.
        let target_addr = base
            .checked_add(r_offset)
            .ok_or(ElfError::InvalidSegment)?;
        let target = VirtAddr::try_new(target_addr).map_err(|_| ElfError::InvalidSegment)?;
        let value = base.wrapping_add(r_addend as u64);
        let phys = crate::memory::virt::translate_in_pml4(target, PhysAddr::new(pml4_phys))
            .ok_or(ElfError::LoadFailed)?;
        unsafe {
            (phys.as_u64() as *mut u64).write_unaligned(value);
        }
    }
    Ok(())
}

fn va_to_file_offset(headers: &[ProgramHeader], va: u64) -> Option<usize> {
    headers
        .iter()
        .filter(|ph| ph.p_type == PT_LOAD)
        .find(|ph| va >= ph.p_vaddr && va < ph.p_vaddr.saturating_add(ph.p_filesz))
        .map(|ph| (ph.p_offset + (va - ph.p_vaddr)) as usize)
}

fn read_c_string(bytes: &[u8]) -> Option<String> {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    if end == 0 {
        return None;
    }
    core::str::from_utf8(&bytes[..end]).ok().map(String::from)
}

fn push_stack_u64(sp: u64, stack_top: u64, top_frame: u64, val: u64) -> Result<u64, ElfError> {
    let sp = sp - 8;
    let top_page = stack_top - 4096;
    if sp < top_page || sp >= stack_top {
        return Err(ElfError::LoadFailed);
    }
    let offset = sp - top_page;
    unsafe {
        *((top_frame + offset) as *mut u64) = val;
    }
    Ok(sp)
}

// ---------------------------------------------------------------------------
// Test-only helpers
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// Defect A, first site: `p_vaddr + base` landing one value past the top
    /// of the canonical low half (0x0000_7FFF_FFFF_FFFF) used to panic
    /// straight through to `VirtAddr::new` in `map_load_segments`. With
    /// `base = 0`, `p_vaddr = 0x0000_8000_0000_0000` is exactly that value —
    /// the smallest non-canonical `u64`. Must now be rejected, not panic.
    fn test_rejects_non_canonical_vaddr() -> TestResult {
        let result = validate_segment_range(0x0000_8000_0000_0000, 0, 4096);
        test_assert_eq!(result, Err(ElfError::InvalidSegment));
        TestResult::Pass
    }

    /// Defect C: `p_memsz` beyond the number of bytes the allocator could
    /// ever hand out must be rejected before the per-page allocation loop
    /// ever runs, not after churning through `alloc_frame` until it returns
    /// `None`.
    fn test_rejects_oversized_memsz() -> TestResult {
        let free_bytes = (crate::memory::phys::free_count() as u64).saturating_mul(4096);
        let oversized = free_bytes.saturating_add(4096 * 16);
        let result = validate_segment_range(0x1000, 0, oversized);
        test_assert_eq!(result, Err(ElfError::InvalidSegment));
        TestResult::Pass
    }

    /// Contrast case: an ordinary in-range, canonical, small segment must
    /// still validate — the checks above must not reject legitimate input.
    fn test_accepts_sane_segment() -> TestResult {
        let result = validate_segment_range(0x1000, 0, 4096);
        test_assert_eq!(result, Ok((0x1000, 0x2000)));
        TestResult::Pass
    }

    /// Defect B: a frame tracked in `LoadRollback` but never committed
    /// (the shape every early-return failure path in `load` and
    /// `load_shared_object` now takes) must be freed back to the allocator
    /// when the rollback drops, not leaked.
    fn test_uncommitted_rollback_frees_tracked_frames() -> TestResult {
        let before = crate::memory::phys::free_count();
        let frame = crate::memory::phys::alloc_frame();
        test_assert!(frame.is_some(), "test needs at least one free frame");
        {
            let mut rollback = LoadRollback::new();
            rollback.track(frame.unwrap());
            // Dropped here without calling `commit()` — must free `frame`.
        }
        test_assert_eq!(crate::memory::phys::free_count(), before);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "elf_loader::rejects_non_canonical_vaddr",
            test_rejects_non_canonical_vaddr,
        );
        crate::testing::register_test(
            "elf_loader::rejects_oversized_memsz",
            test_rejects_oversized_memsz,
        );
        crate::testing::register_test(
            "elf_loader::accepts_sane_segment",
            test_accepts_sane_segment,
        );
        crate::testing::register_test(
            "elf_loader::uncommitted_rollback_frees_tracked_frames",
            test_uncommitted_rollback_frees_tracked_frames,
        );
    }
}
