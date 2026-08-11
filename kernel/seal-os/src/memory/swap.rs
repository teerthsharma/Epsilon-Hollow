// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Topological swap subsystem — T1–T5 theorem integration over swapped pages.
//!
//! Swapped pages are stored in `/swap.topo` as 16-point clouds on S²
//! (reusing TopCrypt encoding).  The swap index is partitioned by Voronoi
//! cell of the virtual address (T1), prefetched spectrally (T2), compacted
//! when Betti-0 entropy exceeds threshold (T3), governed by epsilon (T4),
//! and ordered by hyperbolic lifetime (T5).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::Ordering;
use spin::Mutex;
use x86_64::{PhysAddr, VirtAddr};

use crate::fs::topcrypt::TopoBlock;
use crate::fs::vfs::{self, with_vfs};
use crate::memory::mmap;
use crate::memory::phys;
use crate::memory::virt;

const BLOCKS_PER_PAGE: usize = 64;
const SWAP_ENTRY_SIZE: usize = BLOCKS_PER_PAGE * 68; // 4352 bytes
const SWAP_PREFETCH_PAGES: usize = 3;
const SWAP_FRAGMENTATION_THRESHOLD: f32 = 0.7;

/// A single swapped page encoded as 64 TopoBlocks (4096 bytes → 64×64-byte blocks).
pub struct TopoSwapEntry {
    pub blocks: [TopoBlock; BLOCKS_PER_PAGE],
}

impl TopoSwapEntry {
    /// Encode a 4 KiB page into a topological swap entry.
    pub fn encode_page(data: &[u8; 4096], seed: u64) -> Self {
        let file = crate::fs::topcrypt::encode_bytes(data, seed);
        let mut blocks = [TopoBlock {
            embedding: [0; 32],
            checksum: 0,
        }; BLOCKS_PER_PAGE];
        for (i, block) in file.blocks.iter().take(BLOCKS_PER_PAGE).enumerate() {
            blocks[i] = *block;
        }
        Self { blocks }
    }

    /// Decode a topological swap entry back to a 4 KiB page.
    pub fn decode_page(&self) -> [u8; 4096] {
        let file = crate::fs::topcrypt::TopologicalFile {
            name_hash: [0; 32],
            block_count: BLOCKS_PER_PAGE as u64,
            embedding_seed: 0,
            blocks: self.blocks.to_vec(),
            locked: false,
            lock_entropy: 0.0,
        };
        let decoded = crate::fs::topcrypt::decode_bytes(&file);
        let mut out = [0u8; 4096];
        let len = decoded.len().min(4096);
        out[..len].copy_from_slice(&decoded[..len]);
        out
    }

    /// Serialize to flat bytes (no header, raw blocks).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(SWAP_ENTRY_SIZE);
        for block in &self.blocks {
            for &angle in &block.embedding {
                out.extend_from_slice(&angle.to_le_bytes());
            }
            out.extend_from_slice(&block.checksum.to_le_bytes());
        }
        out
    }

    /// Deserialize from flat bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < SWAP_ENTRY_SIZE {
            return None;
        }
        let mut blocks = [TopoBlock {
            embedding: [0; 32],
            checksum: 0,
        }; BLOCKS_PER_PAGE];
        let mut offset = 0;
        for block in &mut blocks {
            for i in 0..32 {
                block.embedding[i] = u16::from_le_bytes([data[offset], data[offset + 1]]);
                offset += 2;
            }
            block.checksum = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;
        }
        Some(Self { blocks })
    }
}

struct SwapSlot {
    virt: VirtAddr,
    cell: u16,
}

struct SwapState {
    handle: vfs::VfsHandle,
    slots: Vec<Option<SwapSlot>>,
    index: BTreeMap<VirtAddr, usize>,
    cell_index: [Vec<VirtAddr>; 8],
    free_list: Vec<usize>,
    swap_tick: u64,
}

static SWAP_STATE: Mutex<Option<SwapState>> = Mutex::new(None);

// ---------------------------------------------------------------------------
// T1 — Voronoi index of virtual addresses
// ---------------------------------------------------------------------------

fn embedding_distance_sq(a: &[u16; 32], b: &[u16; 32]) -> u32 {
    let mut sum = 0u32;
    for i in 0..32 {
        let d = (a[i] as i32) - (b[i] as i32);
        sum += (d * d) as u32;
    }
    sum
}

fn voronoi_cell_of_virt(virt: VirtAddr) -> u16 {
    let mut emb = [0u16; 32];
    let addr = virt.as_u64();
    for a in 0..32 {
        emb[a] = ((addr
            .wrapping_mul(1103515245)
            .wrapping_add(12345)
            .wrapping_add(a.wrapping_mul(65537) as u64))
            % 65536) as u16;
    }

    let mut seeds = [[0u16; 32]; 8];
    for s in 0..8 {
        for a in 0..32 {
            seeds[s][a] =
                ((s.wrapping_mul(7919).wrapping_add(a.wrapping_mul(104729))) % 65536) as u16;
        }
    }

    let mut best = 0usize;
    let mut best_dist = embedding_distance_sq(&emb, &seeds[0]);
    for i in 1..8 {
        let dist = embedding_distance_sq(&emb, &seeds[i]);
        if dist < best_dist {
            best = i;
            best_dist = dist;
        }
    }
    best as u16
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// # Errors
/// Returns `Err` if `phys` lies above the identity map and the scratch
/// mapping could not be established (out of physical frames for page-table
/// allocation). On error nothing is copied and `out` is left untouched —
/// callers must not treat the buffer as valid.
unsafe fn read_phys_page(phys: PhysAddr, out: &mut [u8; 4096]) -> Result<(), virt::MapError> {
    let addr = phys.as_u64();
    if addr < virt::IDENTITY_MAP_SIZE {
        core::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), 4096);
    } else {
        // Fallback scratch mapping for high physical addresses (>16 GiB identity map).
        let scratch = VirtAddr::new(0xffff_ffff_7000_0000);
        let flags = x86_64::structures::paging::PageTableFlags::PRESENT
            | x86_64::structures::paging::PageTableFlags::WRITABLE;
        virt::map_page(scratch, phys, flags)?;
        core::ptr::copy_nonoverlapping(scratch.as_u64() as *const u8, out.as_mut_ptr(), 4096);
        let _ = virt::unmap_page(scratch);
    }
    Ok(())
}

/// # Errors
/// Returns `Err` if `phys` lies above the identity map and the scratch
/// mapping could not be established (out of physical frames for page-table
/// allocation). On error nothing is copied to `phys` — the frame's contents
/// are left whatever they were before the call.
unsafe fn write_phys_page(phys: PhysAddr, data: &[u8; 4096]) -> Result<(), virt::MapError> {
    let addr = phys.as_u64();
    if addr < virt::IDENTITY_MAP_SIZE {
        core::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, 4096);
    } else {
        let scratch = VirtAddr::new(0xffff_ffff_7000_0000);
        let flags = x86_64::structures::paging::PageTableFlags::PRESENT
            | x86_64::structures::paging::PageTableFlags::WRITABLE;
        virt::map_page(scratch, phys, flags)?;
        core::ptr::copy_nonoverlapping(data.as_ptr(), scratch.as_u64() as *mut u8, 4096);
        let _ = virt::unmap_page(scratch);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Core swap API
// ---------------------------------------------------------------------------

/// Initialise the swap subsystem (requires VFS).
pub fn init() {
    if !vfs::is_vfs_initialized() {
        crate::serial_println!("[SWAP] VFS not ready, deferring swap init");
        return;
    }
    let handle = with_vfs(|vfs| {
        if let Ok(h) = vfs.lookup("/swap.topo") {
            Some(h)
        } else {
            vfs.create("/swap.topo").ok()
        }
    });
    let Some(handle) = handle else {
        crate::serial_println!("[SWAP] Failed to create /swap.topo");
        return;
    };
    *SWAP_STATE.lock() = Some(SwapState {
        handle,
        slots: Vec::new(),
        index: BTreeMap::new(),
        cell_index: [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ],
        free_list: Vec::new(),
        swap_tick: 0,
    });
    crate::serial_println!("[SWAP] Topological swap subsystem initialised");
}

/// Swap out a single page.  Returns `true` on success.
///
/// # Safety
/// On success the page is unmapped and its frame is returned to the physical
/// allocator, so no reference, raw pointer or in-flight DMA may still target
/// `virt` or its backing frame; the mapping is only restored by
/// `swap_in_page`, normally from the page-fault handler. `virt` must belong to
/// an `mmap` region whose `page_table` field is a live PML4 physical address
/// inside the identity map, since the page tables are walked through raw
/// pointers. The caller must hold none of `SWAP_STATE`, `mmap::REGIONS` or the
/// VFS lock — all three are taken here and none is reentrant — and must not
/// call this from an interrupt handler, since it performs blocking VFS I/O.
pub unsafe fn swap_out_page(virt: VirtAddr) -> bool {
    let mut guard = SWAP_STATE.lock();
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return false,
    };

    // Locate owning mmap region.
    let regions = mmap::REGIONS.lock();
    let region = match regions.iter().find(|r| {
        let end = r.start.as_u64() + r.pages as u64 * 4096;
        virt.as_u64() >= r.start.as_u64() && virt.as_u64() < end
    }) {
        Some(r) => r.clone(),
        None => return false,
    };
    drop(regions);

    // Translate to physical in the region's page table.
    let phys = match virt::translate_in_pml4(virt, PhysAddr::new(region.page_table)) {
        Some(p) => p,
        None => return false,
    };

    // Read page contents. On failure (scratch mapping couldn't be
    // established — allocator at/near empty, exactly the condition that
    // triggers swap-out) bail before anything is written to the swap file
    // or the live mapping is touched; the page stays resident and this
    // swap-out attempt degrades to a no-op rather than dereferencing an
    // unmapped scratch page while `SWAP_STATE` is held.
    let mut page_data = [0u8; 4096];
    if read_phys_page(phys, &mut page_data).is_err() {
        return false;
    }

    // Encode topologically.
    let seed = virt.as_u64().wrapping_mul(1103515245).wrapping_add(12345);
    let entry = TopoSwapEntry::encode_page(&page_data, seed);

    // Acquire a swap slot.
    let slot_idx = state.free_list.pop().unwrap_or_else(|| {
        let idx = state.slots.len();
        state.slots.push(None);
        idx
    });

    // Write to disk.
    let offset = slot_idx as u64 * SWAP_ENTRY_SIZE as u64;
    let buf = entry.to_bytes();
    let written = with_vfs(|vfs| vfs.write(state.handle, &buf, offset)).unwrap_or(0);
    if written != buf.len() {
        state.free_list.push(slot_idx);
        return false;
    }

    // T1: index by Voronoi cell of virtual address.
    let cell = voronoi_cell_of_virt(virt);
    state.slots[slot_idx] = Some(SwapSlot { virt, cell });
    state.index.insert(virt, slot_idx);
    if (cell as usize) < state.cell_index.len() {
        state.cell_index[cell as usize].push(virt);
    }

    // Unmap and free the physical frame.
    if let Some(old_phys) = virt::unmap_page_in_pml4(virt, PhysAddr::new(region.page_table)) {
        phys::free_frame(old_phys);
    }

    state.swap_tick += 1;
    true
}

/// Swap in a single page.  Returns `true` on success.
///
/// # Safety
/// `virt` must belong to an `mmap` region whose `page_table` field is a live
/// PML4 physical address inside the identity map: it is dereferenced as a raw
/// `PageTable` and a fresh frame is mapped into it, so nothing else may be
/// mutating that page table concurrently. If `virt` is currently mapped in
/// that address space the previous translation is overwritten and its frame
/// leaks. As with `swap_out_page`, the caller must hold neither `SWAP_STATE`,
/// `mmap::REGIONS` nor the VFS lock, and must be in a context where blocking
/// VFS reads and heap allocation are permitted.
pub unsafe fn swap_in_page(virt: VirtAddr) -> bool {
    let mut guard = SWAP_STATE.lock();
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return false,
    };

    let slot_idx = match state.index.get(&virt) {
        Some(&idx) => idx,
        None => return false,
    };

    let offset = slot_idx as u64 * SWAP_ENTRY_SIZE as u64;
    let mut buf = vec![0; SWAP_ENTRY_SIZE];
    let read = with_vfs(|vfs| vfs.read(state.handle, &mut buf, offset)).unwrap_or(0);
    if read != SWAP_ENTRY_SIZE {
        return false;
    }

    let entry = match TopoSwapEntry::from_bytes(&buf) {
        Some(e) => e,
        None => return false,
    };
    let page_data = entry.decode_page();

    // Allocate a new frame.
    let frame = match phys::alloc_frame() {
        Some(f) => f,
        None => return false,
    };

    // Find the region to know where to map.
    let regions = mmap::REGIONS.lock();
    let region = match regions.iter().find(|r| {
        let end = r.start.as_u64() + r.pages as u64 * 4096;
        virt.as_u64() >= r.start.as_u64() && virt.as_u64() < end
    }) {
        Some(r) => r.clone(),
        None => {
            phys::free_frame(frame);
            return false;
        }
    };
    drop(regions);

    // On failure (scratch mapping couldn't be established) the freshly
    // allocated frame never received the decoded page contents, so it must
    // not be wired into the target address space — free it and degrade this
    // swap-in to a no-op instead of dereferencing an unmapped scratch page
    // while `SWAP_STATE` is held.
    if write_phys_page(frame, &page_data).is_err() {
        phys::free_frame(frame);
        return false;
    }

    let pml4 = &mut *(region.page_table as *mut x86_64::structures::paging::PageTable);
    if virt::map_page_to_pml4(virt, frame, region.flags, pml4).is_err() {
        phys::free_frame(frame);
        return false;
    }

    // Remove from swap state.
    state.slots[slot_idx] = None;
    state.index.remove(&virt);
    let cell = voronoi_cell_of_virt(virt);
    if let Some(vec) = state.cell_index.get_mut(cell as usize) {
        vec.retain(|&v| v != virt);
    }
    state.free_list.push(slot_idx);

    // T2: spectral prefetch — note adjacent pages for future prefetch.
    for i in 1..=SWAP_PREFETCH_PAGES {
        let adj = VirtAddr::new(virt.as_u64() + i as u64 * 4096);
        if state.index.contains_key(&adj) {
            // In a full implementation we would queue these for async prefetch.
            // For now the presence check documents the T2 intent.
        }
    }

    true
}

/// Returns `true` if the given virtual address is currently swapped out.
pub fn is_swapped(virt: VirtAddr) -> bool {
    let guard = SWAP_STATE.lock();
    if let Some(state) = guard.as_ref() {
        state.index.contains_key(&virt)
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// T3 — Betti-0 fragmentation / compaction
// ---------------------------------------------------------------------------

fn swap_fragmentation_entropy_inner(state: &SwapState) -> f32 {
    let total = state.slots.len();
    let free = state.free_list.len();
    if total == 0 || free == 0 {
        return 0.0;
    }
    let mut components = 0usize;
    let mut in_run = false;
    for i in 0..total {
        if state.slots[i].is_none() {
            if !in_run {
                components += 1;
                in_run = true;
            }
        } else {
            in_run = false;
        }
    }
    let num = libm::log2f((components + 1) as f32);
    let den = libm::log2f((free + 1) as f32);
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Return the current Betti-0 entropy of the swap space.
pub fn swap_fragmentation_entropy() -> f32 {
    let guard = SWAP_STATE.lock();
    if let Some(state) = guard.as_ref() {
        swap_fragmentation_entropy_inner(state)
    } else {
        0.0
    }
}

/// Compact the swap file when fragmentation entropy exceeds threshold (T3).
///
/// Returns `true` if the swap file was fully compacted (or was below the
/// entropy threshold and needed no work), `false` if a swap-file read or write
/// failed part way through and compaction stopped early.
///
/// Compaction relocates swap entries towards slot 0, and every relocation is
/// destructive: the slot an entry vacates is added back to the free list and
/// will be overwritten by the next `swap_out_page`. So an entry's index may
/// only be republished at its new slot once the bytes are known to have
/// arrived there. A failed relocation therefore stops the compaction: the
/// entry being moved, and every entry after it, keeps its original slot index,
/// where its bytes are still intact. Nothing that has already been copied is at
/// risk — relocations run in ascending slot order and always move an entry
/// downwards, so the destination of the failing write is always a slot whose
/// former occupant has already been copied elsewhere.
///
/// The alternative — publishing the new index and hoping — loses the page
/// outright: the source slot is freed and reused while the index points at a
/// destination that was never written, so `swap_in_page` decodes whatever
/// bytes happen to be there (`TopoSwapEntry::from_bytes` only checks length)
/// and maps that garbage into the faulting process.
#[must_use]
pub fn compact() -> bool {
    let mut guard = SWAP_STATE.lock();
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return false,
    };
    let entropy = swap_fragmentation_entropy_inner(state);
    if entropy <= SWAP_FRAGMENTATION_THRESHOLD {
        return true;
    }

    let mut new_slots: Vec<Option<SwapSlot>> = Vec::with_capacity(state.slots.len());
    let mut new_index: BTreeMap<VirtAddr, usize> = BTreeMap::new();
    let mut moved = 0usize;
    let mut stopped_at: Option<usize> = None;

    for (old_idx, slot) in state.slots.iter().enumerate() {
        let Some(s) = slot else { continue };
        let new_idx = new_slots.len();

        if new_idx != old_idx {
            let old_offset = old_idx as u64 * SWAP_ENTRY_SIZE as u64;
            let new_offset = new_idx as u64 * SWAP_ENTRY_SIZE as u64;
            let mut buf = vec![0; SWAP_ENTRY_SIZE];
            let read = with_vfs(|vfs| vfs.read(state.handle, &mut buf, old_offset)).unwrap_or(0);
            if read != SWAP_ENTRY_SIZE {
                stopped_at = Some(old_idx);
                break;
            }
            let written = with_vfs(|vfs| vfs.write(state.handle, &buf, new_offset)).unwrap_or(0);
            if written != buf.len() {
                stopped_at = Some(old_idx);
                break;
            }
            moved += 1;
        }

        new_slots.push(Some(SwapSlot {
            virt: s.virt,
            cell: s.cell,
        }));
        new_index.insert(s.virt, new_idx);
    }

    if let Some(stop) = stopped_at {
        // Entries from `stop` onwards were never copied: keep them at the slot
        // that still holds their bytes.
        new_slots.resize_with(state.slots.len(), || None);
        for (old_idx, slot) in state.slots.iter().enumerate().skip(stop) {
            if let Some(s) = slot {
                new_slots[old_idx] = Some(SwapSlot {
                    virt: s.virt,
                    cell: s.cell,
                });
                new_index.insert(s.virt, old_idx);
            }
        }
    }

    state.slots = new_slots;
    state.index = new_index;
    // Rebuild rather than clear: a stopped compaction leaves the vacated slots
    // between the compacted prefix and the untouched tail as holes, and a hole
    // that is not in the free list is never reused, so the swap file would grow
    // without bound. A complete compaction has no holes and this yields the
    // empty free list it always did.
    state.free_list.clear();
    for (i, slot) in state.slots.iter().enumerate() {
        if slot.is_none() {
            state.free_list.push(i);
        }
    }
    for cell in state.cell_index.iter_mut() {
        cell.clear();
    }
    for s in state.slots.iter().flatten() {
        if (s.cell as usize) < state.cell_index.len() {
            state.cell_index[s.cell as usize].push(s.virt);
        }
    }

    if let Some(stop) = stopped_at {
        crate::serial_println!(
            "[SWAP/T3] Compaction stopped at slot {} after {} pages (swap I/O failed), entropy={:.2}",
            stop,
            moved,
            entropy
        );
        return false;
    }

    crate::serial_println!(
        "[SWAP/T3] Compacted {} pages, entropy={:.2}",
        moved,
        entropy
    );
    true
}

// ---------------------------------------------------------------------------
// T4 + T5 — Swap-out daemon
// ---------------------------------------------------------------------------

/// Swap-out daemon tick.  Called periodically from the kernel house-keeping task.
pub fn swap_out_daemon() {
    let total = phys::total_usable_frames();
    let free = phys::free_count();
    if free * 10 >= total {
        return;
    }

    // T4: Governor epsilon controls swap aggressiveness.
    let epsilon = f64::from_bits(crate::GOVERNOR_EPSILON.load(Ordering::Relaxed)) as f32;
    if epsilon < 0.2 {
        return;
    }
    let aggressiveness = if epsilon > 0.6 { 8 } else { 2 };

    // Collect candidate mapped pages from mmap regions.
    let mut candidates: Vec<(VirtAddr, u32)> = Vec::new();
    {
        let regions = mmap::REGIONS.lock();
        for region in regions.iter() {
            for i in 0..region.pages {
                let v = VirtAddr::new(region.start.as_u64() + i as u64 * 4096);
                if let Some(p) = virt::translate_in_pml4(v, PhysAddr::new(region.page_table)) {
                    let density = crate::memory::topo_ram::frame_access_density(p).unwrap_or(0);
                    candidates.push((v, density));
                }
            }
        }
    }

    // T5: hyperbolic lifetime — short-lived pages (high bit density) swap first.
    candidates.sort_by_key(|c| core::cmp::Reverse(c.1));
    for (v, _density) in candidates.iter().take(aggressiveness) {
        unsafe {
            swap_out_page(*v);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;

    /// `write_phys_page` / `read_phys_page` now return `Result` instead of
    /// silently discarding a failed scratch mapping. This exercises the `Ok`
    /// path end to end — allocate a real frame, write a known pattern through
    /// the new signature, read it back, and check both calls report success
    /// and the bytes round-trip. It does not force the `Err` path (that needs
    /// the physical allocator driven to empty, which isn't safe to simulate
    /// inside a shared test-mode boot), but it does prove the refactor didn't
    /// break the copy itself.
    fn test_phys_page_round_trip_reports_ok() -> TestResult {
        let Some(frame) = phys::alloc_frame() else {
            return TestResult::Fail("no free frame available for round-trip test");
        };

        let mut data = [0u8; 4096];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }

        let write_result = unsafe { write_phys_page(frame, &data) };
        if write_result.is_err() {
            unsafe { phys::free_frame(frame) };
            return TestResult::Fail("write_phys_page returned Err for a freshly allocated frame");
        }

        let mut readback = [0u8; 4096];
        let read_result = unsafe { read_phys_page(frame, &mut readback) };
        unsafe { phys::free_frame(frame) };

        if read_result.is_err() {
            return TestResult::Fail("read_phys_page returned Err for a freshly written frame");
        }
        if readback != data {
            return TestResult::Fail("round-tripped page contents did not match what was written");
        }
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "swap::phys_page_round_trip_reports_ok",
            test_phys_page_round_trip_reports_ok,
        );
    }
}
