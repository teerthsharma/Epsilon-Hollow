// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Chart object decoding.
//!
//! A *chart* ships as an ELF64 relocatable object (`ET_REL`, `EM_X86_64`).
//! This module turns those bytes into a live coordinate patch inside the
//! kernel manifold: sections are placed into a W^X image, undefined symbols
//! are bound to kernel *germs*, and x86_64 relocations are applied.
//!
//! Every offset taken from the object is bounds-checked before it is used to
//! index. Nothing here panics on malformed input; failures return `ObjError`.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use x86_64::{structures::paging::PageTableFlags, PhysAddr, VirtAddr};

/// Version of the chart object contract this loader implements.
pub const CHART_FORMAT_VERSION: u32 = 1;

/// Bounds applied to untrusted objects *before* any allocation happens.
const MAX_SECTIONS: usize = 256;
const MAX_SYMBOLS: usize = 4096;
const MAX_RELOCS: usize = 8192;
const MAX_IMAGE_BYTES: u64 = 1 << 20;
const MAX_SECTION_ALIGN: u64 = 4096;

const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;

const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;

const SHN_UNDEF: u16 = 0;
const SHN_ABS: u16 = 0xFFF1;

pub const R_X86_64_64: u32 = 1;
pub const R_X86_64_PC32: u32 = 2;
pub const R_X86_64_PLT32: u32 = 4;
pub const R_X86_64_32S: u32 = 11;

/// `movabs r11, imm64; jmp r11` — 13 bytes, padded to 16.
const PLT_SLOT_SIZE: u64 = 16;

/// Entry-point symbol names a chart must export.
pub const CHART_INIT_SYMBOL: &str = "chart_init";
pub const CHART_EXIT_SYMBOL: &str = "chart_exit";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjError {
    Truncated,
    BadMagic,
    NotElf64,
    NotLittleEndian,
    BadElfVersion,
    NotRelocatable,
    NotX86_64,
    BadSectionTable,
    NoSymbolTable,
    BadStringTable,
    BadSectionAlign,
    SectionIndexOutOfRange,
    SymbolIndexOutOfRange,
    RelocationOutOfBounds,
    RelocationOverflow,
    UnsupportedRelocation(u32),
    UnresolvedGerm(String),
    MissingEntryPoints,
    ImageTooLarge,
    TooManySections,
    TooManySymbols,
    TooManyRelocations,
    OutOfMemory,
    MapFailed,
}

// ---------------------------------------------------------------------------
// Bounds-checked primitive readers
// ---------------------------------------------------------------------------

fn at(bytes: &[u8], off: u64, len: u64) -> Result<&[u8], ObjError> {
    let off = usize::try_from(off).map_err(|_| ObjError::Truncated)?;
    let len = usize::try_from(len).map_err(|_| ObjError::Truncated)?;
    let end = off.checked_add(len).ok_or(ObjError::Truncated)?;
    bytes.get(off..end).ok_or(ObjError::Truncated)
}

fn rd_u16(bytes: &[u8], off: u64) -> Result<u16, ObjError> {
    let s = at(bytes, off, 2)?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn rd_u32(bytes: &[u8], off: u64) -> Result<u32, ObjError> {
    let s = at(bytes, off, 4)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn rd_u64(bytes: &[u8], off: u64) -> Result<u64, ObjError> {
    let s = at(bytes, off, 8)?;
    Ok(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

/// Read a NUL-terminated name out of a string-table section.
fn cstr(bytes: &[u8], tab_off: u64, tab_size: u64, name_off: u32) -> Result<String, ObjError> {
    let tab = at(bytes, tab_off, tab_size)?;
    let start = name_off as usize;
    if start >= tab.len() {
        return Err(ObjError::BadStringTable);
    }
    let rest = &tab[start..];
    let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
    Ok(String::from_utf8_lossy(&rest[..end]).into_owned())
}

fn align_up(value: u64, align: u64) -> Option<u64> {
    let align = align.max(1);
    value.checked_add(align - 1).map(|v| v & !(align - 1))
}

// ---------------------------------------------------------------------------
// Relocation arithmetic — pure, so it can be unit-tested without memory
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocWrite {
    /// 8-byte little-endian store.
    W8(u64),
    /// 4-byte little-endian store.
    W4(u32),
}

/// Compute the value a relocation writes.
///
/// `s` is the resolved symbol address (for `PLT32` against an out-of-range
/// germ this is the address of the chart's own PLT veneer), `a` the addend and
/// `p` the address of the relocated slot itself.
pub fn compute_reloc(rtype: u32, s: u64, a: i64, p: u64) -> Result<RelocWrite, ObjError> {
    match rtype {
        R_X86_64_64 => Ok(RelocWrite::W8((s as i64).wrapping_add(a) as u64)),
        R_X86_64_PC32 | R_X86_64_PLT32 => {
            let v = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
            if v < i32::MIN as i64 || v > i32::MAX as i64 {
                return Err(ObjError::RelocationOverflow);
            }
            Ok(RelocWrite::W4(v as i32 as u32))
        }
        R_X86_64_32S => {
            let v = (s as i64).wrapping_add(a);
            if v < i32::MIN as i64 || v > i32::MAX as i64 {
                return Err(ObjError::RelocationOverflow);
            }
            Ok(RelocWrite::W4(v as i32 as u32))
        }
        other => Err(ObjError::UnsupportedRelocation(other)),
    }
}

// ---------------------------------------------------------------------------
// Parsed object
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SectionHeader {
    kind: u32,
    flags: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    addralign: u64,
    entsize: u64,
}

struct Symbol {
    name: String,
    shndx: u16,
    value: u64,
}

struct RelObject {
    sections: Vec<SectionHeader>,
    symbols: Vec<Symbol>,
    symtab_index: usize,
}

fn parse(bytes: &[u8]) -> Result<RelObject, ObjError> {
    if bytes.len() < 64 {
        return Err(ObjError::Truncated);
    }
    if &bytes[0..4] != b"\x7FELF" {
        return Err(ObjError::BadMagic);
    }
    if bytes[4] != 2 {
        return Err(ObjError::NotElf64);
    }
    if bytes[5] != 1 {
        return Err(ObjError::NotLittleEndian);
    }
    if bytes[6] != 1 {
        return Err(ObjError::BadElfVersion);
    }
    if rd_u16(bytes, 16)? != 1 {
        return Err(ObjError::NotRelocatable);
    }
    if rd_u16(bytes, 18)? != 62 {
        return Err(ObjError::NotX86_64);
    }

    let shoff = rd_u64(bytes, 40)?;
    let shentsize = rd_u16(bytes, 58)? as u64;
    let shnum = rd_u16(bytes, 60)? as usize;
    if shentsize != 64 {
        return Err(ObjError::BadSectionTable);
    }
    if shnum == 0 || shnum > MAX_SECTIONS {
        return Err(ObjError::TooManySections);
    }
    // Force the whole section header table to be in range up front.
    at(bytes, shoff, shentsize * shnum as u64)?;

    let mut sections = Vec::with_capacity(shnum);
    for i in 0..shnum {
        let base = shoff + shentsize * i as u64;
        let sh = SectionHeader {
            kind: rd_u32(bytes, base + 4)?,
            flags: rd_u64(bytes, base + 8)?,
            offset: rd_u64(bytes, base + 16)?,
            size: rd_u64(bytes, base + 24)?,
            link: rd_u32(bytes, base + 40)?,
            info: rd_u32(bytes, base + 44)?,
            addralign: rd_u64(bytes, base + 48)?,
            entsize: rd_u64(bytes, base + 56)?,
        };
        // Every non-NOBITS section must actually live inside the file.
        if sh.kind != SHT_NOBITS && sh.kind != 0 {
            at(bytes, sh.offset, sh.size)?;
        }
        sections.push(sh);
    }

    let symtab_index = sections
        .iter()
        .position(|s| s.kind == SHT_SYMTAB)
        .ok_or(ObjError::NoSymbolTable)?;
    let symtab = sections[symtab_index];
    if symtab.entsize != 24 || symtab.size % 24 != 0 {
        return Err(ObjError::NoSymbolTable);
    }
    let strtab = *sections
        .get(symtab.link as usize)
        .ok_or(ObjError::SectionIndexOutOfRange)?;
    if strtab.kind != SHT_STRTAB {
        return Err(ObjError::BadStringTable);
    }

    let symcount = (symtab.size / 24) as usize;
    if symcount > MAX_SYMBOLS {
        return Err(ObjError::TooManySymbols);
    }
    let mut symbols = Vec::with_capacity(symcount);
    for i in 0..symcount {
        let base = symtab.offset + 24 * i as u64;
        let name_off = rd_u32(bytes, base)?;
        let shndx = rd_u16(bytes, base + 6)?;
        let value = rd_u64(bytes, base + 8)?;
        let name = if name_off == 0 {
            String::new()
        } else {
            cstr(bytes, strtab.offset, strtab.size, name_off)?
        };
        symbols.push(Symbol { name, shndx, value });
    }

    Ok(RelObject {
        sections,
        symbols,
        symtab_index,
    })
}

// ---------------------------------------------------------------------------
// W^X chart image
// ---------------------------------------------------------------------------

/// Writable, never executable — used while relocations are being applied and
/// for the data half permanently.
fn writable_flags() -> PageTableFlags {
    PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE
}

/// Executable, never writable — the sealed state of the text half.
fn executable_flags() -> PageTableFlags {
    PageTableFlags::PRESENT
}

/// Kernel-resident memory backing one chart.
///
/// The executable half is mapped `RW + NX` while relocations are being applied
/// and flipped to `RX` by [`ChartImage::seal`] before anything is called; it is
/// never writable and executable at the same time. The data half stays `RW+NX`
/// for its whole life.
pub struct ChartImage {
    exec_base: u64,
    exec_frames: Vec<PhysAddr>,
    data_base: u64,
    data_frames: Vec<PhysAddr>,
    sealed: bool,
}

fn map_region(pages: usize, flags: PageTableFlags) -> Result<(u64, Vec<PhysAddr>), ObjError> {
    let base = crate::memory::virt::alloc_virtual_pages(pages, 1).ok_or(ObjError::OutOfMemory)?;
    let mut frames: Vec<PhysAddr> = Vec::with_capacity(pages);
    for i in 0..pages {
        let virt = VirtAddr::new(base.as_u64() + (i as u64) * 4096);
        let frame = match crate::memory::phys::alloc_frame() {
            Some(f) => f,
            None => {
                unmap_frames(base.as_u64(), &frames);
                return Err(ObjError::OutOfMemory);
            }
        };
        if unsafe { crate::memory::virt::map_page(virt, frame, flags) }.is_err() {
            unsafe { crate::memory::phys::free_frame(frame) };
            unmap_frames(base.as_u64(), &frames);
            return Err(ObjError::MapFailed);
        }
        frames.push(frame);
        unsafe {
            core::ptr::write_bytes(virt.as_u64() as *mut u8, 0, 4096);
        }
    }
    Ok((base.as_u64(), frames))
}

fn unmap_frames(base: u64, frames: &[PhysAddr]) {
    for (i, frame) in frames.iter().enumerate() {
        let virt = VirtAddr::new(base + (i as u64) * 4096);
        unsafe {
            crate::memory::virt::unmap_page(virt);
            crate::memory::phys::free_frame(*frame);
        }
    }
}

fn remap_region(base: u64, frames: &[PhysAddr], flags: PageTableFlags) -> bool {
    for (i, frame) in frames.iter().enumerate() {
        let virt = VirtAddr::new(base + (i as u64) * 4096);
        // `unmap_page` performs the TLB shootdown; the fresh mapping below then
        // takes effect without a stale entry.
        unsafe {
            if crate::memory::virt::unmap_page(virt).is_none() {
                return false;
            }
            if crate::memory::virt::map_page(virt, *frame, flags).is_err() {
                return false;
            }
        }
    }
    true
}

impl ChartImage {
    fn alloc(exec_bytes: u64, data_bytes: u64) -> Result<Self, ObjError> {
        if exec_bytes > MAX_IMAGE_BYTES || data_bytes > MAX_IMAGE_BYTES {
            return Err(ObjError::ImageTooLarge);
        }
        let exec_pages = exec_bytes.div_ceil(4096).max(1) as usize;
        let data_pages = data_bytes.div_ceil(4096).max(1) as usize;
        let (exec_base, exec_frames) = map_region(exec_pages, writable_flags())?;
        let (data_base, data_frames) = match map_region(data_pages, writable_flags()) {
            Ok(v) => v,
            Err(e) => {
                unmap_frames(exec_base, &exec_frames);
                return Err(e);
            }
        };
        Ok(Self {
            exec_base,
            exec_frames,
            data_base,
            data_frames,
            sealed: false,
        })
    }

    /// Flip the executable half from `RW+NX` to `RX`. Returns false if the
    /// remap failed, in which case the image must be discarded.
    pub fn seal(&mut self) -> bool {
        if self.sealed {
            return true;
        }
        self.sealed = remap_region(self.exec_base, &self.exec_frames, executable_flags());
        self.sealed
    }

    /// True once the text half has been flipped to read+execute.
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// Total resident bytes (both halves, page granular).
    pub fn resident_bytes(&self) -> usize {
        (self.exec_frames.len() + self.data_frames.len()) * 4096
    }
}

impl Drop for ChartImage {
    fn drop(&mut self) {
        // ponytail: frames are returned, the virtual range is not — the kernel
        // VA bump allocator has no free list. Add one if chart churn matters.
        unmap_frames(self.exec_base, &self.exec_frames);
        unmap_frames(self.data_base, &self.data_frames);
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadReport {
    pub sections_placed: usize,
    pub symbols_resolved: usize,
    pub germs_bound: usize,
    pub plt_slots: usize,
    pub relocations_applied: usize,
    pub reloc_64: usize,
    pub reloc_pc32: usize,
    pub reloc_plt32: usize,
    pub reloc_32s: usize,
    pub init_addr: u64,
    pub exit_addr: u64,
    pub image_bytes: usize,
}

/// Decode, place and relocate a chart object.
///
/// `resolve` maps an undefined symbol name to a kernel germ address; returning
/// `None` aborts the load with [`ObjError::UnresolvedGerm`].
pub fn load(
    bytes: &[u8],
    resolve: &dyn Fn(&str) -> Option<u64>,
) -> Result<(ChartImage, LoadReport), ObjError> {
    let obj = parse(bytes)?;
    let nsec = obj.sections.len();

    // --- placement ---------------------------------------------------------
    // exec half: SHF_ALLOC | SHF_EXECINSTR. data half: everything else that is
    // SHF_ALLOC. `placed[i] = Some((is_exec, offset_within_half))`.
    let mut placed: Vec<Option<(bool, u64)>> = vec![None; nsec];
    let mut exec_size: u64 = 0;
    let mut data_size: u64 = 0;
    let mut sections_placed = 0usize;
    for (i, sh) in obj.sections.iter().enumerate() {
        if sh.flags & SHF_ALLOC == 0 || sh.size == 0 {
            continue;
        }
        let align = sh.addralign.max(1);
        if align > MAX_SECTION_ALIGN || !align.is_power_of_two() {
            return Err(ObjError::BadSectionAlign);
        }
        let is_exec = sh.flags & SHF_EXECINSTR != 0;
        let cursor = if is_exec {
            &mut exec_size
        } else {
            &mut data_size
        };
        let start = align_up(*cursor, align).ok_or(ObjError::ImageTooLarge)?;
        *cursor = start.checked_add(sh.size).ok_or(ObjError::ImageTooLarge)?;
        if *cursor > MAX_IMAGE_BYTES {
            return Err(ObjError::ImageTooLarge);
        }
        placed[i] = Some((is_exec, start));
        sections_placed += 1;
    }

    // One PLT veneer per named undefined symbol. Kernel text sits ~123 TiB from
    // the chart heap range, far outside the +/-2 GiB a `call rel32` can reach,
    // so every germ call goes through `movabs r11, addr; jmp r11`.
    let plt_slots = obj
        .symbols
        .iter()
        .filter(|s| s.shndx == SHN_UNDEF && !s.name.is_empty())
        .count();
    let plt_base_off = align_up(exec_size, 16).ok_or(ObjError::ImageTooLarge)?;
    exec_size = plt_base_off
        .checked_add(plt_slots as u64 * PLT_SLOT_SIZE)
        .ok_or(ObjError::ImageTooLarge)?;
    if exec_size > MAX_IMAGE_BYTES {
        return Err(ObjError::ImageTooLarge);
    }

    let mut image = ChartImage::alloc(exec_size, data_size)?;
    let plt_base = image.exec_base + plt_base_off;

    let exec_base = image.exec_base;
    let data_base = image.data_base;
    let half_base = |is_exec: bool| if is_exec { exec_base } else { data_base };

    // --- copy section contents --------------------------------------------
    for (i, sh) in obj.sections.iter().enumerate() {
        let Some((is_exec, off)) = placed[i] else {
            continue;
        };
        let dst = half_base(is_exec) + off;
        if sh.kind == SHT_NOBITS {
            continue; // pages arrive zeroed
        }
        let src = at(bytes, sh.offset, sh.size)?;
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, src.len());
        }
    }

    // --- symbol resolution -------------------------------------------------
    let mut values: Vec<u64> = Vec::with_capacity(obj.symbols.len());
    let mut plt_addrs: Vec<Option<u64>> = Vec::with_capacity(obj.symbols.len());
    let mut germs_bound = 0usize;
    let mut next_plt = 0u64;
    for sym in &obj.symbols {
        match sym.shndx {
            SHN_UNDEF => {
                if sym.name.is_empty() {
                    values.push(0);
                    plt_addrs.push(None);
                    continue;
                }
                let target =
                    resolve(&sym.name).ok_or_else(|| ObjError::UnresolvedGerm(sym.name.clone()))?;
                let stub = plt_base + next_plt * PLT_SLOT_SIZE;
                next_plt += 1;
                write_plt_stub(stub, target);
                germs_bound += 1;
                values.push(target);
                plt_addrs.push(Some(stub));
            }
            SHN_ABS => {
                values.push(sym.value);
                plt_addrs.push(None);
            }
            shndx => {
                let idx = shndx as usize;
                if idx >= nsec {
                    return Err(ObjError::SectionIndexOutOfRange);
                }
                let Some((is_exec, off)) = placed[idx] else {
                    // Symbol in a non-loaded section: it has no address.
                    values.push(0);
                    plt_addrs.push(None);
                    continue;
                };
                values.push(half_base(is_exec) + off + sym.value);
                plt_addrs.push(None);
            }
        }
    }

    // --- relocation --------------------------------------------------------
    let mut report = LoadReport {
        sections_placed,
        symbols_resolved: obj.symbols.len().saturating_sub(1),
        germs_bound,
        plt_slots,
        image_bytes: image.resident_bytes(),
        ..Default::default()
    };

    let mut total_relocs = 0usize;
    for sh in obj.sections.iter() {
        if sh.kind != SHT_RELA {
            continue;
        }
        if sh.entsize != 24 || sh.size % 24 != 0 {
            return Err(ObjError::RelocationOutOfBounds);
        }
        let target_idx = sh.info as usize;
        if target_idx >= nsec {
            return Err(ObjError::SectionIndexOutOfRange);
        }
        if sh.link as usize != obj.symtab_index {
            return Err(ObjError::SymbolIndexOutOfRange);
        }
        let Some((is_exec, target_off)) = placed[target_idx] else {
            continue; // relocations against a section we did not load
        };
        let target_size = obj.sections[target_idx].size;
        let target_base = half_base(is_exec) + target_off;

        let count = (sh.size / 24) as usize;
        total_relocs = total_relocs.saturating_add(count);
        if total_relocs > MAX_RELOCS {
            return Err(ObjError::TooManyRelocations);
        }

        for i in 0..count {
            let base = sh.offset + 24 * i as u64;
            let r_offset = rd_u64(bytes, base)?;
            let r_info = rd_u64(bytes, base + 8)?;
            let r_addend = rd_u64(bytes, base + 16)? as i64;
            let rtype = (r_info & 0xFFFF_FFFF) as u32;
            let rsym = (r_info >> 32) as usize;

            if rsym >= values.len() {
                return Err(ObjError::SymbolIndexOutOfRange);
            }
            let width: u64 = match rtype {
                R_X86_64_64 => 8,
                R_X86_64_PC32 | R_X86_64_PLT32 | R_X86_64_32S => 4,
                other => return Err(ObjError::UnsupportedRelocation(other)),
            };
            let end = r_offset
                .checked_add(width)
                .ok_or(ObjError::RelocationOutOfBounds)?;
            if end > target_size {
                return Err(ObjError::RelocationOutOfBounds);
            }

            // PLT32 against an undefined germ goes through the veneer.
            let s = if rtype == R_X86_64_PLT32 {
                plt_addrs[rsym].unwrap_or(values[rsym])
            } else {
                values[rsym]
            };
            let p = target_base + r_offset;
            match compute_reloc(rtype, s, r_addend, p)? {
                RelocWrite::W8(v) => unsafe {
                    core::ptr::write_unaligned(p as *mut u64, v.to_le());
                },
                RelocWrite::W4(v) => unsafe {
                    core::ptr::write_unaligned(p as *mut u32, v.to_le());
                },
            }
            match rtype {
                R_X86_64_64 => report.reloc_64 += 1,
                R_X86_64_PC32 => report.reloc_pc32 += 1,
                R_X86_64_PLT32 => report.reloc_plt32 += 1,
                _ => report.reloc_32s += 1,
            }
            report.relocations_applied += 1;
        }
    }

    // --- entry points ------------------------------------------------------
    let mut init_addr = 0u64;
    let mut exit_addr = 0u64;
    for (i, sym) in obj.symbols.iter().enumerate() {
        if sym.name == CHART_INIT_SYMBOL {
            init_addr = values[i];
        } else if sym.name == CHART_EXIT_SYMBOL {
            exit_addr = values[i];
        }
    }
    if init_addr == 0 || exit_addr == 0 {
        return Err(ObjError::MissingEntryPoints);
    }
    report.init_addr = init_addr;
    report.exit_addr = exit_addr;

    if !image.seal() {
        return Err(ObjError::MapFailed);
    }
    Ok((image, report))
}

/// Emit `movabs r11, target; jmp r11` at `addr` (13 bytes of a 16-byte slot).
fn write_plt_stub(addr: u64, target: u64) {
    let mut slot = [0x90u8; PLT_SLOT_SIZE as usize];
    slot[0] = 0x49;
    slot[1] = 0xbb;
    slot[2..10].copy_from_slice(&target.to_le_bytes());
    slot[10] = 0x41;
    slot[11] = 0xff;
    slot[12] = 0xe3;
    unsafe {
        core::ptr::copy_nonoverlapping(slot.as_ptr(), addr as *mut u8, slot.len());
    }
}
