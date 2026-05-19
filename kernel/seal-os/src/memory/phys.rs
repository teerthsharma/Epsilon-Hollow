// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Physical page allocator using a fixed-size bitmap.
//!
//! Supports up to 16 GiB of RAM (4 194 304 frames). The bitmap lives in
//! `.bss` and is accessed under a `spin::Mutex`.

use spin::Mutex;
use x86_64::PhysAddr;

use crate::boot::boot_info::MemoryDescriptor;

const MAX_RAM_GB: usize = 16;
const MAX_FRAMES: usize = (MAX_RAM_GB * 1024 * 1024 * 1024) / 4096;
/// Frames above 4 GiB are not identity-mapped by the current page tables, so we
/// restrict allocations to the low 4 GiB for simplicity.
const USABLE_FRAMES: usize = (4 * 1024 * 1024 * 1024) / 4096;
const BITMAP_U64S: usize = MAX_FRAMES / 64;

/// One bit per 4 KiB frame.  `1` = used, `0` = free.
static BITMAP: Mutex<[u64; BITMAP_U64S]> = Mutex::new([u64::MAX; BITMAP_U64S]);

/// Number of frames currently marked free.
static FREE_COUNT: Mutex<usize> = Mutex::new(0);

#[inline]
fn frame_bit(frame: usize) -> (usize, usize) {
    (frame / 64, frame % 64)
}

/// Initialise the allocator from a UEFI memory map.
///
/// # Safety
/// Must be called exactly once before any other function in this module.
pub unsafe fn init(
    memory_map: &[MemoryDescriptor],
    kernel_start: PhysAddr,
    kernel_end: PhysAddr,
    fb_start: PhysAddr,
    fb_end: PhysAddr,
) {
    let mut bitmap = BITMAP.lock();
    let mut count = FREE_COUNT.lock();

    // Conservative default: mark everything as used.
    for word in bitmap.iter_mut() {
        *word = u64::MAX;
    }
    *count = 0;

    for desc in memory_map {
        // UEFI MemoryType::CONVENTIONAL == 7
        if desc.ty != 7 {
            continue;
        }
        for page in 0..desc.page_count {
            let addr = desc.phys_start + page * 4096;
            if addr as usize >= USABLE_FRAMES * 4096 {
                break;
            }
            if overlaps(PhysAddr::new(addr), kernel_start, kernel_end)
                || overlaps(PhysAddr::new(addr), fb_start, fb_end)
            {
                continue;
            }
            let frame = (addr / 4096) as usize;
            let (word, bit) = frame_bit(frame);
            bitmap[word] &= !(1 << bit);
            *count += 1;
        }
    }
}

fn overlaps(addr: PhysAddr, start: PhysAddr, end: PhysAddr) -> bool {
    let a = addr.as_u64();
    a >= start.as_u64() && a < end.as_u64()
}

/// Allocate a single 4 KiB physical frame.
pub fn alloc_frame() -> Option<PhysAddr> {
    let mut bitmap = BITMAP.lock();
    for (word_idx, word) in bitmap.iter_mut().enumerate() {
        if *word == u64::MAX {
            continue;
        }
        let bit = word.trailing_ones() as usize;
        if bit < 64 {
            *word |= 1 << bit;
            let frame = word_idx * 64 + bit;
            return Some(PhysAddr::new(frame as u64 * 4096));
        }
    }
    None
}

/// Allocate `count` contiguous 4 KiB frames.
pub fn alloc_frames_contiguous(count: usize) -> Option<PhysAddr> {
    let mut bitmap = BITMAP.lock();
    let mut run_start = 0;
    let mut run_len = 0;

    for word_idx in 0..BITMAP_U64S {
        let word = bitmap[word_idx];
        if word == u64::MAX {
            run_len = 0;
            continue;
        }
        for bit in 0..64 {
            let frame = word_idx * 64 + bit;
            if frame >= USABLE_FRAMES {
                break;
            }
            let is_free = (word >> bit) & 1 == 0;
            if is_free {
                if run_len == 0 {
                    run_start = frame;
                }
                run_len += 1;
                if run_len >= count {
                    for f in run_start..run_start + count {
                        let (w, b) = frame_bit(f);
                        bitmap[w] |= 1 << b;
                    }
                    return Some(PhysAddr::new(run_start as u64 * 4096));
                }
            } else {
                run_len = 0;
            }
        }
    }
    None
}

/// Free a single 4 KiB physical frame.
pub unsafe fn free_frame(addr: PhysAddr) {
    let frame = (addr.as_u64() / 4096) as usize;
    if frame >= USABLE_FRAMES {
        return;
    }
    let (word, bit) = frame_bit(frame);
    let mut bitmap = BITMAP.lock();
    bitmap[word] &= !(1 << bit);
}

/// Return the number of free frames.
pub fn free_count() -> usize {
    *FREE_COUNT.lock()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_bit() {
        assert_eq!(frame_bit(0), (0, 0));
        assert_eq!(frame_bit(63), (0, 63));
        assert_eq!(frame_bit(64), (1, 0));
        assert_eq!(frame_bit(128), (2, 0));
    }

    #[test]
    fn test_overlaps() {
        let s = PhysAddr::new(0x1000);
        let e = PhysAddr::new(0x3000);
        assert!(overlaps(PhysAddr::new(0x1000), s, e));
        assert!(overlaps(PhysAddr::new(0x2000), s, e));
        assert!(!overlaps(PhysAddr::new(0x3000), s, e));
        assert!(!overlaps(PhysAddr::new(0x0), s, e));
    }
}
