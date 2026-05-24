// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Physical page allocator using a fixed-size bitmap.
//!
//! Supports up to 128 GiB of RAM. The bitmap lives in `.bss` (kernel higher-half)
//! and is accessed under a `spin::Mutex`.  Low memory (0–4 GiB) is initialised by
//! `init()`; high memory (>4 GiB) is brought online later by `init_high()` so that
//! the UEFI memory map can be scanned after the higher-half page tables are active.

use core::sync::atomic::{AtomicUsize, Ordering};
use spin::Mutex;
use x86_64::PhysAddr;

use crate::boot::boot_info::MemoryDescriptor;

const MAX_RAM_GB: usize = 128;
const MAX_FRAMES: usize = (MAX_RAM_GB * 1024 * 1024 * 1024) / 4096;
const BITMAP_U64S: usize = MAX_FRAMES / 64;

/// Maximum frames tracked by the topological RAM subsystem (256 MiB).
pub const USABLE_FRAMES: usize = 65536;

/// Frames below 4 GiB are identity-mapped by the bootloader.
pub const LOW_FRAME_LIMIT: usize = (4 * 1024 * 1024 * 1024) / 4096;

/// One bit per 4 KiB frame.  `1` = used, `0` = free.
static BITMAP: Mutex<[u64; BITMAP_U64S]> = Mutex::new([u64::MAX; BITMAP_U64S]);

/// Number of frames currently marked free.
static FREE_COUNT: Mutex<usize> = Mutex::new(0);

/// Runtime-discovered total conventional RAM frames (low + high).
static TOTAL_USABLE_FRAMES: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn frame_bit(frame: usize) -> (usize, usize) {
    (frame / 64, frame % 64)
}

/// Initialise the allocator from a UEFI memory map (low memory only).
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

    let mut max_frame = 0usize;

    for desc in memory_map {
        // UEFI MemoryType::CONVENTIONAL == 7
        if desc.ty != 7 {
            continue;
        }
        for page in 0..desc.page_count {
            let addr = desc.phys_start + page * 4096;
            let frame = (addr / 4096) as usize;
            if frame >= LOW_FRAME_LIMIT {
                // High memory is handled later by init_high().
                if frame > max_frame {
                    max_frame = frame;
                }
                continue;
            }
            if overlaps(PhysAddr::new(addr), kernel_start, kernel_end)
                || overlaps(PhysAddr::new(addr), fb_start, fb_end)
                || addr < 0x10_0000
            {
                continue;
            }
            let (word, bit) = frame_bit(frame);
            let was_used = (bitmap[word] >> bit) & 1 != 0;
            bitmap[word] &= !(1 << bit);
            if was_used {
                *count += 1;
            }
            if frame > max_frame {
                max_frame = frame;
            }
        }
    }

    TOTAL_USABLE_FRAMES.store(max_frame + 1, Ordering::SeqCst);
}

/// Bring high-memory frames online after the kernel higher-half mapping is active.
///
/// # Safety
/// Must be called exactly once, after `init()`.
pub unsafe fn init_high(
    memory_map: &[MemoryDescriptor],
    kernel_start: PhysAddr,
    kernel_end: PhysAddr,
    fb_start: PhysAddr,
    fb_end: PhysAddr,
) {
    let mut bitmap = BITMAP.lock();
    let mut count = FREE_COUNT.lock();
    let mut max_frame = TOTAL_USABLE_FRAMES.load(Ordering::SeqCst);

    for desc in memory_map {
        if desc.ty != 7 {
            continue;
        }
        for page in 0..desc.page_count {
            let addr = desc.phys_start + page * 4096;
            let frame = (addr / 4096) as usize;
            if frame < LOW_FRAME_LIMIT {
                continue;
            }
            if frame >= MAX_FRAMES {
                continue;
            }
            if overlaps(PhysAddr::new(addr), kernel_start, kernel_end)
                || overlaps(PhysAddr::new(addr), fb_start, fb_end)
                || addr < 0x10_0000
            {
                continue;
            }
            let (word, bit) = frame_bit(frame);
            if (bitmap[word] >> bit) & 1 != 0 {
                bitmap[word] &= !(1 << bit);
                *count += 1;
            }
            if frame > max_frame {
                max_frame = frame;
            }
        }
    }

    TOTAL_USABLE_FRAMES.store(max_frame + 1, Ordering::SeqCst);
}

/// Return the number of trackable frames (low + high).
pub fn total_usable_frames() -> usize {
    TOTAL_USABLE_FRAMES.load(Ordering::Relaxed)
}

/// Mark a single frame as used. Returns `true` if the frame was previously free.
pub fn mark_used(addr: PhysAddr) -> bool {
    let frame = (addr.as_u64() / 4096) as usize;
    if frame >= total_usable_frames() {
        return false;
    }
    let (word, bit) = frame_bit(frame);
    let mut bitmap = BITMAP.lock();
    let was_free = (bitmap[word] >> bit) & 1 == 0;
    bitmap[word] |= 1 << bit;
    drop(bitmap);
    if was_free {
        let mut count = FREE_COUNT.lock();
        *count = count.saturating_sub(1);
    }
    was_free
}

/// Run a closure with mutable access to the underlying bitmap words.
pub fn with_bitmap_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut [u64]) -> R,
{
    let mut bitmap = BITMAP.lock();
    f(&mut bitmap[..])
}

/// Decrement the cached free-frame counter by `n`.
pub fn decr_free_count(n: usize) {
    let mut count = FREE_COUNT.lock();
    *count = count.saturating_sub(n);
}

fn overlaps(addr: PhysAddr, start: PhysAddr, end: PhysAddr) -> bool {
    let a = addr.as_u64();
    a >= start.as_u64() && a < end.as_u64()
}

/// Allocate a single 4 KiB physical frame (preferentially from low memory).
pub fn alloc_frame() -> Option<PhysAddr> {
    let limit = total_usable_frames();
    let mut bitmap = BITMAP.lock();
    for (word_idx, word) in bitmap.iter_mut().enumerate() {
        if *word == u64::MAX {
            continue;
        }
        let bit = word.trailing_ones() as usize;
        if bit < 64 {
            let frame = word_idx * 64 + bit;
            if frame >= limit {
                continue;
            }
            *word |= 1 << bit;
            drop(bitmap);
            let mut count = FREE_COUNT.lock();
            *count = count.saturating_sub(1);
            return Some(PhysAddr::new(frame as u64 * 4096));
        }
    }
    drop(bitmap);
    let actual_free = count_free_bits();
    crate::serial_println!(
        "[DEBUG] alloc_frame returning None, actual_free_frames={}, stale_free_count={}",
        actual_free,
        *FREE_COUNT.lock()
    );
    None
}

/// Allocate a single 4 KiB physical frame from high memory (>4 GiB).
pub fn alloc_frame_high() -> Option<PhysAddr> {
    let limit = total_usable_frames();
    let mut bitmap = BITMAP.lock();
    let start_word = LOW_FRAME_LIMIT / 64;
    for word_idx in start_word..BITMAP_U64S {
        let word = &mut bitmap[word_idx];
        if *word == u64::MAX {
            continue;
        }
        let mut bit = word.trailing_ones() as usize;
        // If the first free bit in this word is below the high threshold,
        // find the next one that is above it.
        let base_frame = word_idx * 64;
        if base_frame + bit < LOW_FRAME_LIMIT {
            bit = LOW_FRAME_LIMIT.saturating_sub(base_frame);
            if bit >= 64 {
                continue;
            }
            // Advance to the first free bit at or after `bit`.
            while bit < 64 && ((*word >> bit) & 1) != 0 {
                bit += 1;
            }
            if bit >= 64 {
                continue;
            }
        }
        let frame = base_frame + bit;
        if frame >= limit {
            continue;
        }
        *word |= 1 << bit;
        drop(bitmap);
        let mut count = FREE_COUNT.lock();
        *count = count.saturating_sub(1);
        return Some(PhysAddr::new(frame as u64 * 4096));
    }
    None
}

/// Free a single 4 KiB physical frame (any zone).
pub unsafe fn free_frame(addr: PhysAddr) {
    let frame = (addr.as_u64() / 4096) as usize;
    if frame >= total_usable_frames() {
        return;
    }
    let (word, bit) = frame_bit(frame);
    let mut bitmap = BITMAP.lock();
    let was_used = (bitmap[word] >> bit) & 1 != 0;
    bitmap[word] &= !(1 << bit);
    drop(bitmap);
    if was_used {
        let mut count = FREE_COUNT.lock();
        *count += 1;
    }
}

/// Free a high-memory frame (>4 GiB).  Identical to `free_frame` but documents intent.
pub unsafe fn free_frame_high(addr: PhysAddr) {
    free_frame(addr);
}

fn count_free_bits() -> usize {
    let bitmap = BITMAP.lock();
    let mut total = 0usize;
    for word in bitmap.iter() {
        total += word.count_zeros() as usize;
    }
    total
}

/// Allocate `count` contiguous 4 KiB frames.
pub fn alloc_frames_contiguous(count: usize) -> Option<PhysAddr> {
    let limit = total_usable_frames();
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
            if frame >= limit {
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
                    drop(bitmap);
                    let mut free_count = FREE_COUNT.lock();
                    *free_count = free_count.saturating_sub(count);
                    return Some(PhysAddr::new(run_start as u64 * 4096));
                }
            } else {
                run_len = 0;
            }
        }
    }
    None
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
