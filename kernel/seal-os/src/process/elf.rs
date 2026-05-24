// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Minimal ELF64 loader for static executables.

use x86_64::VirtAddr;
use x86_64::structures::paging::{PageTable, PageTableFlags};

/// ELF parsing / loading errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElfError {
    InvalidMagic,
    InvalidClass,
    InvalidType,
    TooSmall,
    LoadFailed,
}

/// Read a little-endian u16 from `data` at `offset`, returning `TooSmall` if out of bounds.
fn read_u16(data: &[u8], offset: usize) -> Result<u16, ElfError> {
    let bytes = data.get(offset..offset + 2).ok_or(ElfError::TooSmall)?;
    let mut arr = [0u8; 2];
    arr.copy_from_slice(bytes);
    Ok(u16::from_le_bytes(arr))
}

/// Read a little-endian u32 from `data` at `offset`, returning `TooSmall` if out of bounds.
fn read_u32(data: &[u8], offset: usize) -> Result<u32, ElfError> {
    let bytes = data.get(offset..offset + 4).ok_or(ElfError::TooSmall)?;
    let mut arr = [0u8; 4];
    arr.copy_from_slice(bytes);
    Ok(u32::from_le_bytes(arr))
}

/// Read a little-endian u64 from `data` at `offset`, returning `TooSmall` if out of bounds.
fn read_u64(data: &[u8], offset: usize) -> Result<u64, ElfError> {
    let bytes = data.get(offset..offset + 8).ok_or(ElfError::TooSmall)?;
    let mut arr = [0u8; 8];
    arr.copy_from_slice(bytes);
    Ok(u64::from_le_bytes(arr))
}

/// Result of a successful ELF load.
pub struct LoadedElf {
    pub entry_point: u64,
    pub stack_pointer: u64,
    pub page_table: u64, // physical address of PML4
    pub file_mode: u16,
    pub file_uid: u32,
    pub file_gid: u32,
}

/// Parse and load a static ELF64 image into fresh user page tables.
/// `file_mode`, `file_uid`, and `file_gid` are forwarded for setuid/setgid
/// handling by the scheduler after loading.
pub fn load(elf_data: &[u8], aslr_base: u64, file_mode: u16, file_uid: u32, file_gid: u32) -> Result<LoadedElf, ElfError> {
    if !elf_data.get(0..4).map(|s| s == b"\x7FELF").unwrap_or(false) {
        return Err(ElfError::InvalidMagic);
    }
    if elf_data.get(4).copied().unwrap_or(0) != 2 {
        // ELFCLASS64
        return Err(ElfError::InvalidClass);
    }

    let e_type = read_u16(elf_data, 16)?;
    if e_type != 2 && e_type != 3 {
        // ET_EXEC or ET_DYN only
        return Err(ElfError::InvalidType);
    }

    let entry = read_u64(elf_data, 24)?;
    let phoff = read_u64(elf_data, 32)?;
    let phentsize = read_u16(elf_data, 54)?;
    let phnum = read_u16(elf_data, 56)?;

    let base = if e_type == 3 { aslr_base } else { 0 };
    let entry_point = entry + base;

    // Allocate a fresh PML4 for this process.
    let pml4_frame = crate::memory::phys::alloc_frame().ok_or(ElfError::LoadFailed)?;
    let pml4 = unsafe { &mut *(pml4_frame.as_u64() as *mut PageTable) };
    pml4.zero();

    // Clone kernel higher-half mappings (PML4 entries 256..512) so that
    // interrupts and syscalls continue to work while in this address space.
    unsafe {
        let kernel_pml4_virt = crate::memory::virt::current_pml4_virt();
        let kernel_pml4 = &*(kernel_pml4_virt.as_u64() as *const PageTable);
        for i in 256..512 {
            pml4[i] = kernel_pml4[i].clone();
        }
    }

    // Walk program headers and map PT_LOAD segments.
    for i in 0..phnum {
        let off = phoff
            .checked_add((i as u64).checked_mul(phentsize as u64).ok_or(ElfError::TooSmall)?)
            .ok_or(ElfError::TooSmall)?;
        let ph_start = off as usize;
        let ph_end = ph_start.checked_add(56).ok_or(ElfError::TooSmall)?;
        if ph_end > elf_data.len() {
            return Err(ElfError::TooSmall);
        }
        let ph = elf_data.get(ph_start..ph_end).ok_or(ElfError::TooSmall)?;

        let p_type = read_u32(ph, 0)?;
        if p_type != 1 {
            // PT_LOAD
            continue;
        }

        let p_offset = read_u64(ph, 8)?;
        let p_vaddr = read_u64(ph, 16)? + base;
        let _p_filesz = read_u64(ph, 32)?;
        let p_memsz = read_u64(ph, 40)?;
        let p_flags = read_u32(ph, 4)?;

        let mut flags = PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE;
        if p_flags & 0x2 != 0 {
            // PF_W
            flags |= PageTableFlags::WRITABLE;
        }
        if p_flags & 0x1 != 0 {
            // PF_X — clear NO_EXECUTE
            flags &= !PageTableFlags::NO_EXECUTE;
        } else {
            flags |= PageTableFlags::NO_EXECUTE;
        }

        let seg_start = p_vaddr;
        let seg_end = p_vaddr + p_memsz;
        let first_page = seg_start / 4096;
        let last_page = (seg_end + 4095) / 4096;

        for page in first_page..last_page {
            let page_virt = VirtAddr::new(page * 4096);
            let frame = crate::memory::phys::alloc_frame().ok_or(ElfError::LoadFailed)?;

            // Zero the page before copying.
            unsafe {
                core::ptr::write_bytes(frame.as_u64() as *mut u8, 0, 4096);
            }

            // Determine what part of this page comes from the file.
            let page_start = page * 4096;
            let page_end = page_start + 4096;
            let seg_page_start = seg_start.max(page_start);
            let seg_page_end = seg_end.min(page_end);
            let len = seg_page_end.saturating_sub(seg_page_start);

            if len > 0 {
                let frame_offset = (seg_page_start - page_start) as usize;
                let file_offset = p_offset
                    .checked_add(seg_page_start - seg_start)
                    .ok_or(ElfError::TooSmall)?;
                let file_end = file_offset.checked_add(len).ok_or(ElfError::TooSmall)? as usize;
                let file_start = file_offset as usize;
                if file_end > elf_data.len() {
                    return Err(ElfError::TooSmall);
                }
                let src = elf_data.get(file_start..file_end).ok_or(ElfError::TooSmall)?;
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
                    crate::memory::phys::free_frame(frame);
                    return Err(ElfError::LoadFailed);
                }
            }
        }
    }

    // Allocate user stack (16 KiB, grows down from high address).
    const USER_STACK_PAGES: usize = 4;
    const USER_STACK_TOP: u64 = 0x0000_7fff_ffff_0000;

    for i in 0..USER_STACK_PAGES {
        let frame = crate::memory::phys::alloc_frame().ok_or(ElfError::LoadFailed)?;
        unsafe {
            core::ptr::write_bytes(frame.as_u64() as *mut u8, 0, 4096);
        }
        let virt = VirtAddr::new(USER_STACK_TOP - ((USER_STACK_PAGES - i) as u64) * 4096);
        unsafe {
            if crate::memory::virt::map_page_to_pml4(
                virt,
                frame,
                PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE,
                pml4,
            ).is_err() {
                crate::memory::phys::free_frame(frame);
                return Err(ElfError::LoadFailed);
            }
        }
    }

    // Push argc=0, argv=NULL, envp=NULL.
    let mut sp = USER_STACK_TOP;
    sp = push_u64(sp, 0); // envp
    sp = push_u64(sp, 0); // argv
    sp = push_u64(sp, 0); // argc

    Ok(LoadedElf {
        entry_point,
        stack_pointer: sp,
        page_table: pml4_frame.as_u64(),
        file_mode,
        file_uid,
        file_gid,
    })
}

fn push_u64(sp: u64, val: u64) -> u64 {
    let sp = sp - 8;
    unsafe {
        *(sp as *mut u64) = val;
    }
    sp
}
