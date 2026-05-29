// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Physical page allocator with bitmap truth and a bounded topological index.
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
const FRAME_SIZE: u64 = 4096;

/// Number of topological free-frame cells. Kept power-of-two for cheap mapping.
pub const PHYS_TOPO_CELLS: usize = 8;

const SUMMARY_L1_U64S: usize = (BITMAP_U64S + 63) / 64;
const SUMMARY_L2_U64S: usize = (SUMMARY_L1_U64S + 63) / 64;
const SUMMARY_L3_U64S: usize = (SUMMARY_L2_U64S + 63) / 64;

const CELL_BIT_MASKS: [u64; PHYS_TOPO_CELLS] = [
    0x0101_0101_0101_0101,
    0x0202_0202_0202_0202,
    0x0404_0404_0404_0404,
    0x0808_0808_0808_0808,
    0x1010_1010_1010_1010,
    0x2020_2020_2020_2020,
    0x4040_4040_4040_4040,
    0x8080_8080_8080_8080,
];

/// Bounded candidate probes for contiguous requests. This keeps allocation
/// independent of installed RAM size.
const CONTIGUOUS_CANDIDATE_PROBES: usize = 128;
/// Maximum contiguous run size admitted by the bounded allocator path.
///
/// Larger transfers must use chunked/scatter-gather I/O instead of asking the
/// physical allocator to mutate an unbounded number of page bits in one call.
pub const MAX_CONTIGUOUS_RUN_PAGES: usize = 64;
const MAX_ALLOC_L3_WORD_PROBES: usize = SUMMARY_L3_U64S;
const MAX_ALLOC_TOPO_CELL_PROBES: usize = PHYS_TOPO_CELLS;
const SINGLE_ALLOC_WORD_CANDIDATE_PROBES: usize = 8192;
const MAX_SINGLE_ALLOC_WORD_PROBES_PER_CELL: usize = SINGLE_ALLOC_WORD_CANDIDATE_PROBES;
const _: [(); 1] = [(); (SUMMARY_L3_U64S <= 2) as usize];
const _: [(); 1] = [(); (MAX_ALLOC_TOPO_CELL_PROBES == 8) as usize];
const _: [(); 1] = [(); (MAX_SINGLE_ALLOC_WORD_PROBES_PER_CELL <= 8192) as usize];

pub struct AllocatorO1Proof {
    pub topo_cells: usize,
    pub l3_word_probes_per_cell: usize,
    pub single_alloc_word_probes_per_cell: usize,
    pub contiguous_candidate_probes: usize,
    pub contiguous_max_run_pages: usize,
}

pub const fn allocation_o1_proof() -> AllocatorO1Proof {
    AllocatorO1Proof {
        topo_cells: MAX_ALLOC_TOPO_CELL_PROBES,
        l3_word_probes_per_cell: MAX_ALLOC_L3_WORD_PROBES,
        single_alloc_word_probes_per_cell: MAX_SINGLE_ALLOC_WORD_PROBES_PER_CELL,
        contiguous_candidate_probes: CONTIGUOUS_CANDIDATE_PROBES,
        contiguous_max_run_pages: MAX_CONTIGUOUS_RUN_PAGES,
    }
}

/// Maximum frames tracked by the topological RAM subsystem (256 MiB).
pub const USABLE_FRAMES: usize = 65536;

/// Frames below 4 GiB are identity-mapped by the bootloader.
pub const LOW_FRAME_LIMIT: usize = (4 * 1024 * 1024 * 1024) / 4096;

/// One bit per 4 KiB frame.  `1` = used, `0` = free.
static BITMAP: Mutex<[u64; BITMAP_U64S]> = Mutex::new([u64::MAX; BITMAP_U64S]);

/// Three-level topological free index. The bitmap remains the source of truth.
static CELL_L1: Mutex<[[u64; SUMMARY_L1_U64S]; PHYS_TOPO_CELLS]> =
    Mutex::new([[0; SUMMARY_L1_U64S]; PHYS_TOPO_CELLS]);
static CELL_L2: Mutex<[[u64; SUMMARY_L2_U64S]; PHYS_TOPO_CELLS]> =
    Mutex::new([[0; SUMMARY_L2_U64S]; PHYS_TOPO_CELLS]);
static CELL_L3: Mutex<[[u64; SUMMARY_L3_U64S]; PHYS_TOPO_CELLS]> =
    Mutex::new([[0; SUMMARY_L3_U64S]; PHYS_TOPO_CELLS]);

/// Number of frames currently marked free.
static FREE_COUNT: Mutex<usize> = Mutex::new(0);

/// Runtime-discovered total conventional RAM frames (low + high).
static TOTAL_USABLE_FRAMES: AtomicUsize = AtomicUsize::new(0);

/// Round-robin cell cursor for topology-balanced allocation.
static NEXT_TOPO_CELL: AtomicUsize = AtomicUsize::new(0);

/// Runtime proof counters for the allocator hot paths.
static TOPO_FAST_HITS: AtomicUsize = AtomicUsize::new(0);
static TOPO_BOUNDED_MISSES: AtomicUsize = AtomicUsize::new(0);
static MAX_CONTIGUOUS_PROBES_SEEN: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn frame_bit(frame: usize) -> (usize, usize) {
    (frame / 64, frame % 64)
}

#[inline]
fn set_or_clear(word: &mut u64, bit: usize, present: bool) {
    if present {
        *word |= 1 << bit;
    } else {
        *word &= !(1 << bit);
    }
}

fn set_summary_bit(
    l1: &mut [[u64; SUMMARY_L1_U64S]; PHYS_TOPO_CELLS],
    l2: &mut [[u64; SUMMARY_L2_U64S]; PHYS_TOPO_CELLS],
    l3: &mut [[u64; SUMMARY_L3_U64S]; PHYS_TOPO_CELLS],
    cell: usize,
    bitmap_word_idx: usize,
    present: bool,
) {
    let l1_idx = bitmap_word_idx / 64;
    let l1_bit = bitmap_word_idx % 64;
    set_or_clear(&mut l1[cell][l1_idx], l1_bit, present);

    let l2_idx = l1_idx / 64;
    let l2_bit = l1_idx % 64;
    set_or_clear(&mut l2[cell][l2_idx], l2_bit, l1[cell][l1_idx] != 0);

    let l3_idx = l2_idx / 64;
    let l3_bit = l2_idx % 64;
    set_or_clear(&mut l3[cell][l3_idx], l3_bit, l2[cell][l2_idx] != 0);
}

fn refresh_summary_word(
    bitmap_word: u64,
    bitmap_word_idx: usize,
    l1: &mut [[u64; SUMMARY_L1_U64S]; PHYS_TOPO_CELLS],
    l2: &mut [[u64; SUMMARY_L2_U64S]; PHYS_TOPO_CELLS],
    l3: &mut [[u64; SUMMARY_L3_U64S]; PHYS_TOPO_CELLS],
) {
    let free_bits = !bitmap_word;
    for cell in 0..PHYS_TOPO_CELLS {
        set_summary_bit(
            l1,
            l2,
            l3,
            cell,
            bitmap_word_idx,
            (free_bits & CELL_BIT_MASKS[cell]) != 0,
        );
    }
}

fn rebuild_free_index(
    bitmap: &[u64; BITMAP_U64S],
    l1: &mut [[u64; SUMMARY_L1_U64S]; PHYS_TOPO_CELLS],
    l2: &mut [[u64; SUMMARY_L2_U64S]; PHYS_TOPO_CELLS],
    l3: &mut [[u64; SUMMARY_L3_U64S]; PHYS_TOPO_CELLS],
) {
    // SETUP_SCAN_OK: O(total bitmap words) boot/setup rebuild, not a runtime
    // allocation path. Runtime allocation must use the summary index below.
    for cell in 0..PHYS_TOPO_CELLS {
        l1[cell].fill(0);
        l2[cell].fill(0);
        l3[cell].fill(0);
    }
    for (word_idx, word) in bitmap.iter().enumerate() {
        if *word != u64::MAX {
            refresh_summary_word(*word, word_idx, l1, l2, l3);
        }
    }
}

fn frame_range_mask(word_idx: usize, min_frame: usize, max_frame: usize) -> u64 {
    let word_start = word_idx * 64;
    let word_end = word_start + 64;
    if max_frame <= word_start || min_frame >= word_end {
        return 0;
    }

    let mut mask = u64::MAX;
    if min_frame > word_start {
        mask &= u64::MAX << (min_frame - word_start);
    }
    if max_frame < word_end {
        let keep = max_frame - word_start;
        if keep == 0 {
            return 0;
        }
        mask &= (1u64 << keep) - 1;
    }
    mask
}

fn child_range_mask(parent_idx: usize, min_child: usize, max_child: usize) -> u64 {
    let parent_start = parent_idx * 64;
    let parent_end = parent_start + 63;
    if max_child < parent_start || min_child > parent_end {
        return 0;
    }

    let first = min_child.saturating_sub(parent_start).min(63);
    let last = max_child.saturating_sub(parent_start).min(63);
    let low_mask = u64::MAX << first;
    let high_mask = if last == 63 {
        u64::MAX
    } else {
        (1u64 << (last + 1)) - 1
    };
    low_mask & high_mask
}

fn alloc_indexed_frame_from_cell(
    cell: usize,
    min_frame: usize,
    max_frame: usize,
    bitmap: &mut [u64; BITMAP_U64S],
    l1: &mut [[u64; SUMMARY_L1_U64S]; PHYS_TOPO_CELLS],
    l2: &mut [[u64; SUMMARY_L2_U64S]; PHYS_TOPO_CELLS],
    l3: &mut [[u64; SUMMARY_L3_U64S]; PHYS_TOPO_CELLS],
) -> Option<usize> {
    if min_frame >= max_frame {
        return None;
    }

    let min_word = min_frame / 64;
    let max_word = (max_frame - 1) / 64;
    let min_l1_idx = min_word / 64;
    let max_l1_idx = max_word / 64;
    let min_l2_idx = min_l1_idx / 64;
    let max_l2_idx = max_l1_idx / 64;
    let min_l3_idx = min_l2_idx / 64;
    let max_l3_idx = (max_l2_idx / 64).min(SUMMARY_L3_U64S - 1);
    let mut word_probes = 0usize;

    for l3_idx in min_l3_idx..=max_l3_idx {
        let mut l3_word = l3[cell][l3_idx] & child_range_mask(l3_idx, min_l2_idx, max_l2_idx);
        while l3_word != 0 {
            let l3_bit = l3_word.trailing_zeros() as usize;
            let l2_idx = l3_idx * 64 + l3_bit;
            if l2_idx >= SUMMARY_L2_U64S {
                break;
            }

            let mut l2_word = l2[cell][l2_idx] & child_range_mask(l2_idx, min_l1_idx, max_l1_idx);
            while l2_word != 0 {
                let l2_bit = l2_word.trailing_zeros() as usize;
                let l1_idx = l2_idx * 64 + l2_bit;
                if l1_idx >= SUMMARY_L1_U64S {
                    break;
                }

                let mut l1_word = l1[cell][l1_idx] & child_range_mask(l1_idx, min_word, max_word);
                while l1_word != 0 {
                    let l1_bit = l1_word.trailing_zeros() as usize;
                    let word_idx = l1_idx * 64 + l1_bit;
                    if word_idx > max_word {
                        return None;
                    }
                    if word_idx >= min_word {
                        if word_probes >= SINGLE_ALLOC_WORD_CANDIDATE_PROBES {
                            return None;
                        }
                        word_probes += 1;
                        let range_mask = frame_range_mask(word_idx, min_frame, max_frame);
                        let free_bits = !bitmap[word_idx] & CELL_BIT_MASKS[cell] & range_mask;
                        if free_bits != 0 {
                            let bit = free_bits.trailing_zeros() as usize;
                            bitmap[word_idx] |= 1 << bit;
                            refresh_summary_word(bitmap[word_idx], word_idx, l1, l2, l3);
                            return Some(word_idx * 64 + bit);
                        }
                        refresh_summary_word(bitmap[word_idx], word_idx, l1, l2, l3);
                    }
                    l1_word &= !(1 << l1_bit);
                }
                l2_word &= !(1 << l2_bit);
            }
            l3_word &= !(1 << l3_bit);
        }
    }
    None
}

fn find_indexed_free_frame_from_cell(
    cell: usize,
    min_frame: usize,
    max_frame: usize,
    bitmap: &[u64; BITMAP_U64S],
    l1: &[[u64; SUMMARY_L1_U64S]; PHYS_TOPO_CELLS],
    l2: &[[u64; SUMMARY_L2_U64S]; PHYS_TOPO_CELLS],
    l3: &[[u64; SUMMARY_L3_U64S]; PHYS_TOPO_CELLS],
) -> Option<usize> {
    if min_frame >= max_frame {
        return None;
    }

    let min_word = min_frame / 64;
    let max_word = (max_frame - 1) / 64;
    let min_l1_idx = min_word / 64;
    let max_l1_idx = max_word / 64;
    let min_l2_idx = min_l1_idx / 64;
    let max_l2_idx = max_l1_idx / 64;
    let min_l3_idx = min_l2_idx / 64;
    let max_l3_idx = (max_l2_idx / 64).min(SUMMARY_L3_U64S - 1);
    let mut word_probes = 0usize;

    for l3_idx in min_l3_idx..=max_l3_idx {
        let mut l3_word = l3[cell][l3_idx] & child_range_mask(l3_idx, min_l2_idx, max_l2_idx);
        while l3_word != 0 {
            let l3_bit = l3_word.trailing_zeros() as usize;
            let l2_idx = l3_idx * 64 + l3_bit;
            if l2_idx >= SUMMARY_L2_U64S {
                break;
            }

            let mut l2_word = l2[cell][l2_idx] & child_range_mask(l2_idx, min_l1_idx, max_l1_idx);
            while l2_word != 0 {
                let l2_bit = l2_word.trailing_zeros() as usize;
                let l1_idx = l2_idx * 64 + l2_bit;
                if l1_idx >= SUMMARY_L1_U64S {
                    break;
                }

                let mut l1_word = l1[cell][l1_idx] & child_range_mask(l1_idx, min_word, max_word);
                while l1_word != 0 {
                    let l1_bit = l1_word.trailing_zeros() as usize;
                    let word_idx = l1_idx * 64 + l1_bit;
                    if word_idx > max_word {
                        return None;
                    }
                    if word_idx >= min_word {
                        if word_probes >= SINGLE_ALLOC_WORD_CANDIDATE_PROBES {
                            return None;
                        }
                        word_probes += 1;
                        let range_mask = frame_range_mask(word_idx, min_frame, max_frame);
                        let free_bits = !bitmap[word_idx] & CELL_BIT_MASKS[cell] & range_mask;
                        if free_bits != 0 {
                            let bit = free_bits.trailing_zeros() as usize;
                            return Some(word_idx * 64 + bit);
                        }
                    }
                    l1_word &= !(1 << l1_bit);
                }
                l2_word &= !(1 << l2_bit);
            }
            l3_word &= !(1 << l3_bit);
        }
    }
    None
}

fn contiguous_run_is_free(
    bitmap: &[u64; BITMAP_U64S],
    start_frame: usize,
    count: usize,
    max_frame: usize,
) -> bool {
    // RUNTIME_BOUNDED_BY_REQUEST: caller rejects count > MAX_CONTIGUOUS_RUN_PAGES,
    // so this scan is bounded by a fixed DMA run cap, not installed RAM size.
    let end = match start_frame.checked_add(count) {
        Some(end) => end,
        None => return false,
    };
    if count == 0 || end > max_frame {
        return false;
    }

    for frame in start_frame..end {
        let (word, bit) = frame_bit(frame);
        if (bitmap[word] >> bit) & 1 != 0 {
            return false;
        }
    }
    true
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

    // SETUP_SCAN_OK: O(memory-map pages) boot discovery. This pass establishes
    // allocator truth before runtime O(1) hot paths are enabled.
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

    let mut l1 = CELL_L1.lock();
    let mut l2 = CELL_L2.lock();
    let mut l3 = CELL_L3.lock();
    rebuild_free_index(&bitmap, &mut l1, &mut l2, &mut l3);

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

    // SETUP_SCAN_OK: O(memory-map pages) high-memory discovery after paging is
    // live. Runtime single-frame allocation must not return to this scan.
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

    let mut l1 = CELL_L1.lock();
    let mut l2 = CELL_L2.lock();
    let mut l3 = CELL_L3.lock();
    rebuild_free_index(&bitmap, &mut l1, &mut l2, &mut l3);

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
    if was_free {
        let mut l1 = CELL_L1.lock();
        let mut l2 = CELL_L2.lock();
        let mut l3 = CELL_L3.lock();
        refresh_summary_word(bitmap[word], word, &mut l1, &mut l2, &mut l3);
    }
    drop(bitmap);
    if was_free {
        let mut count = FREE_COUNT.lock();
        *count = count.saturating_sub(1);
    }
    was_free
}

/// Run a bounded read-only query over the underlying bitmap words.
///
/// Mutation stays inside allocator-owned APIs so callers cannot bypass the
/// topological free index or invalidate the O(1) allocation proof.
pub(crate) fn with_bitmap_read<F, R>(f: F) -> R
where
    F: FnOnce(&[u64]) -> R,
{
    let bitmap = BITMAP.lock();
    f(&bitmap[..])
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

fn alloc_indexed_frame_cells(
    min_frame: usize,
    max_frame: usize,
    start_cell: usize,
    scan_all_cells: bool,
) -> Option<PhysAddr> {
    let limit = total_usable_frames().min(MAX_FRAMES);
    let max_frame = max_frame.min(limit);
    if min_frame >= max_frame {
        return None;
    }

    let result = {
        let mut bitmap = BITMAP.lock();
        let mut l1 = CELL_L1.lock();
        let mut l2 = CELL_L2.lock();
        let mut l3 = CELL_L3.lock();

        let probes = if scan_all_cells { PHYS_TOPO_CELLS } else { 1 };
        let mut result = None;
        for offset in 0..probes {
            let cell = (start_cell + offset) & (PHYS_TOPO_CELLS - 1);
            result = alloc_indexed_frame_from_cell(
                cell,
                min_frame,
                max_frame,
                &mut bitmap,
                &mut l1,
                &mut l2,
                &mut l3,
            );
            if result.is_some() {
                break;
            }
        }
        result
    };

    if let Some(frame) = result {
        TOPO_FAST_HITS.fetch_add(1, Ordering::Relaxed);
        decr_free_count(1);
        Some(PhysAddr::new(frame as u64 * FRAME_SIZE))
    } else {
        TOPO_BOUNDED_MISSES.fetch_add(1, Ordering::Relaxed);
        None
    }
}

/// Allocate a frame from one topological cell inside `[min_frame, max_frame)`.
pub fn alloc_frame_in_topo_cell(
    cell: usize,
    min_frame: usize,
    max_frame: usize,
) -> Option<PhysAddr> {
    alloc_indexed_frame_cells(min_frame, max_frame, cell & (PHYS_TOPO_CELLS - 1), false)
}

/// Allocate a frame from any topological cell inside `[min_frame, max_frame)`.
pub fn alloc_frame_in_range(min_frame: usize, max_frame: usize) -> Option<PhysAddr> {
    let start_cell = NEXT_TOPO_CELL.fetch_add(1, Ordering::Relaxed) & (PHYS_TOPO_CELLS - 1);
    alloc_indexed_frame_cells(min_frame, max_frame, start_cell, true)
}

/// Allocate a single 4 KiB physical frame (preferentially from low memory).
pub fn alloc_frame() -> Option<PhysAddr> {
    let limit = total_usable_frames().min(MAX_FRAMES);
    let low_limit = limit.min(LOW_FRAME_LIMIT);
    let start_cell = NEXT_TOPO_CELL.fetch_add(1, Ordering::Relaxed) & (PHYS_TOPO_CELLS - 1);

    if let Some(frame) = alloc_indexed_frame_cells(0, low_limit, start_cell, true) {
        return Some(frame);
    }

    if let Some(frame) =
        alloc_indexed_frame_cells(LOW_FRAME_LIMIT.min(limit), limit, start_cell, true)
    {
        return Some(frame);
    }

    crate::serial_println!(
        "[DEBUG] alloc_frame returning None, cached_free_count={}",
        *FREE_COUNT.lock()
    );
    None
}

/// Allocate a single 4 KiB physical frame from high memory (>4 GiB).
pub fn alloc_frame_high() -> Option<PhysAddr> {
    let limit = total_usable_frames().min(MAX_FRAMES);
    let start_cell = NEXT_TOPO_CELL.fetch_add(1, Ordering::Relaxed) & (PHYS_TOPO_CELLS - 1);
    alloc_indexed_frame_cells(LOW_FRAME_LIMIT.min(limit), limit, start_cell, true)
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
    if was_used {
        let mut l1 = CELL_L1.lock();
        let mut l2 = CELL_L2.lock();
        let mut l3 = CELL_L3.lock();
        refresh_summary_word(bitmap[word], word, &mut l1, &mut l2, &mut l3);
    }
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

/// Allocate `count` contiguous 4 KiB frames inside `[min_frame, max_frame)`.
pub fn alloc_frames_contiguous_in_range(
    count: usize,
    min_frame: usize,
    max_frame: usize,
) -> Option<PhysAddr> {
    if count == 0 || count > MAX_CONTIGUOUS_RUN_PAGES {
        return None;
    }
    if count == 1 {
        return alloc_frame_in_range(min_frame, max_frame);
    }

    let limit = total_usable_frames().min(MAX_FRAMES).min(max_frame);
    if min_frame >= limit {
        return None;
    }
    let mut bitmap = BITMAP.lock();
    let mut l1 = CELL_L1.lock();
    let mut l2 = CELL_L2.lock();
    let mut l3 = CELL_L3.lock();
    let start_cell = NEXT_TOPO_CELL.fetch_add(1, Ordering::Relaxed) & (PHYS_TOPO_CELLS - 1);
    let mut probe_min = [min_frame; PHYS_TOPO_CELLS];

    for probe in 0..CONTIGUOUS_CANDIDATE_PROBES {
        MAX_CONTIGUOUS_PROBES_SEEN.fetch_max(probe + 1, Ordering::Relaxed);
        let cell = (start_cell + probe) & (PHYS_TOPO_CELLS - 1);
        let candidate = match find_indexed_free_frame_from_cell(
            cell,
            probe_min[cell],
            limit,
            &bitmap,
            &l1,
            &l2,
            &l3,
        ) {
            Some(candidate) => candidate,
            None => continue,
        };
        probe_min[cell] = candidate.saturating_add(1);

        if !contiguous_run_is_free(&bitmap, candidate, count, limit) {
            continue;
        }

        for frame in candidate..candidate + count {
            let (word, bit) = frame_bit(frame);
            bitmap[word] |= 1 << bit;
        }

        let first_word = candidate / 64;
        let last_word = (candidate + count - 1) / 64;
        for word_idx in first_word..=last_word {
            refresh_summary_word(bitmap[word_idx], word_idx, &mut l1, &mut l2, &mut l3);
        }
        drop(l3);
        drop(l2);
        drop(l1);
        drop(bitmap);
        let mut free_count = FREE_COUNT.lock();
        *free_count = free_count.saturating_sub(count);
        return Some(PhysAddr::new(candidate as u64 * FRAME_SIZE));
    }
    None
}

/// Allocate `count` contiguous 4 KiB frames.
pub fn alloc_frames_contiguous(count: usize) -> Option<PhysAddr> {
    alloc_frames_contiguous_in_range(count, 0, total_usable_frames().min(MAX_FRAMES))
}

/// Return the number of free frames.
pub fn free_count() -> usize {
    *FREE_COUNT.lock()
}

pub fn topo_alloc_stats() -> (usize, usize, usize) {
    (
        TOPO_FAST_HITS.load(Ordering::Relaxed),
        TOPO_BOUNDED_MISSES.load(Ordering::Relaxed),
        MAX_CONTIGUOUS_PROBES_SEEN.load(Ordering::Relaxed),
    )
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
