// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Minimal ELF64 loader for static executables.

use x86_64::{PhysAddr, VirtAddr};
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

/// Result of a successful ELF load.
pub struct LoadedElf {
    pub entry_point: u64,
    pub stack_pointer: u64,
    pub page_table: u64, // physical address of PML4
}

/// Parse and load a static ELF64 image into fresh user page tables.
pub fn load(elf_data: &[u8], aslr_base: u64) -> Result<LoadedElf, ElfError> {
    if elf_data.len() < 64 {
        return Err(ElfError::TooSmall);
    }
    if &elf_data[0..4] != b"\x7FELF" {
        return Err(ElfError::InvalidMagic);
    }
    if elf_data[4] != 2 {
        // ELFCLASS64
        return Err(ElfError::InvalidClass);
    }

    let e_type = u16::from_le_bytes([elf_data[16], elf_data[17]]);
    if e_type != 2 && e_type != 3 {
        // ET_EXEC or ET_DYN only
        return Err(ElfError::InvalidType);
    }

    let entry = u64::from_le_bytes(elf_data[24..32].try_into().unwrap());
    let phoff = u64::from_le_bytes(elf_data[32..40].try_into().unwrap());
    let phentsize = u16::from_le_bytes([elf_data[54], elf_data[55]]);
    let phnum = u16::from_le_bytes([elf_data[56], elf_data[57]]);

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
        let off = phoff + (i as u64) * (phentsize as u64);
        let ph_start = off as usize;
        let ph_end = ph_start + 56;
        if ph_end > elf_data.len() {
            return Err(ElfError::TooSmall);
        }
        let ph = &elf_data[ph_start..ph_end];

        let p_type = u32::from_le_bytes(ph[0..4].try_into().unwrap());
        if p_type != 1 {
            // PT_LOAD
            continue;
        }

        let p_offset = u64::from_le_bytes(ph[8..16].try_into().unwrap());
        let p_vaddr = u64::from_le_bytes(ph[16..24].try_into().unwrap()) + base;
        let _p_filesz = u64::from_le_bytes(ph[32..40].try_into().unwrap());
        let p_memsz = u64::from_le_bytes(ph[40..48].try_into().unwrap());
        let p_flags = u32::from_le_bytes(ph[4..8].try_into().unwrap());

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
                let file_offset = p_offset + (seg_page_start - seg_start);
                let src = &elf_data[file_offset as usize..(file_offset + len) as usize];
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        (frame.as_u64() + frame_offset as u64) as *mut u8,
                        len as usize,
                    );
                }
            }

            unsafe {
                crate::memory::virt::map_page_to_pml4(page_virt, frame, flags, pml4);
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
            crate::memory::virt::map_page_to_pml4(
                virt,
                frame,
                PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE,
                pml4,
            );
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
    })
}

fn push_u64(sp: u64, val: u64) -> u64 {
    let sp = sp - 8;
    unsafe {
        *(sp as *mut u64) = val;
    }
    sp
}
