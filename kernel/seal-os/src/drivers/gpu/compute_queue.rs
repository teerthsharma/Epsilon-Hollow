// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![allow(dead_code)] // REASON: PM4 register and shader constants for future compute dispatch

//! AMD GCN Compute Queue — PM4 command buffer ring, packet builder, and dispatch.
//!
//! This module implements the bare-metal interface to the AMD Command Processor (CP)
//! via a memory-mapped ring buffer.  It is `no_std`, uses only `alloc` for the ring
//! buffer backing store, and communicates with the GPU through PCI BAR0 MMIO.
//!
//! Reference: AMD GCN ISA manuals, "Southern Islands" / "Sea Islands" / "RDNA"
//! programmer guides, and the `linux/drivers/gpu/drm/amd/amdgpu` kernel driver.

use alloc::vec::Vec;
use core::sync::atomic::{fence, Ordering};

use crate::serial_println;

// ------------------------------------------------------------------
// MMIO register offsets (relative to BAR0)
// ------------------------------------------------------------------

/// Graphics Ring Buffer Manager — status.
const MM_GRBM_STATUS: u64 = 0x8010;

/// CP_RB0_RPTR — ring buffer 0 read pointer (GPU consumes).
const MM_CP_RB0_RPTR: u64 = 0xC100;

/// CP_RB0_WPTR — ring buffer 0 write pointer (CPU produces).
const MM_CP_RB0_WPTR: u64 = 0xC104;

/// CP_RB0_CNTL — ring buffer 0 control (size, enable).
const MM_CP_RB0_CNTL: u64 = 0xC110;

/// CP_RB0_BASE — ring buffer 0 base address (lower 32 bits).
const MM_CP_RB0_BASE: u64 = 0xC114;

/// CP_RB0_BASE_HI — ring buffer 0 base address (upper 32 bits).
const MM_CP_RB0_BASE_HI: u64 = 0xC118;

/// CP_RB0_BUFSZ — ring buffer 0 size in DWords.
const MM_CP_RB0_BUFSZ: u64 = 0xC11C;

/// CP_ME_CNTL — Micro-engine control.
const MM_CP_ME_CNTL: u64 = 0x86C0;

/// SDMA0_SEM_ADDR_LO — SDMA semaphore address low.
const MM_SDMA0_SEM_ADDR_LO: u64 = 0x5028;

/// SDMA0_SEM_ADDR_HI — SDMA semaphore address high.
const MM_SDMA0_SEM_ADDR_HI: u64 = 0x502C;

// ------------------------------------------------------------------
// PM4 packet types and opcodes
// ------------------------------------------------------------------

const PM4_TYPE3: u32 = 3;

/// PM4 opcode: write N consecutive SH (shader) registers.
const PM4_SET_SH_REG: u32 = 0x76;

/// PM4 opcode: dispatch compute shader directly (non-indirect).
const PM4_DISPATCH_DIRECT: u32 = 0x15;

/// PM4 opcode: wait for register memory.
const PM4_WAIT_REG_MEM: u32 = 0x3C;

/// PM4 opcode: event write end-of-pipe (fence).
const PM4_EVENT_WRITE_EOP: u32 = 0x46;

/// PM4 opcode: increment CE (constant engine) counter.
const PM4_INCREMENT_CE_COUNTER: u32 = 0x84;

/// PM4 opcode: increment DE (draw engine) counter.
const PM4_INCREMENT_DE_COUNTER: u32 = 0x85;

/// PM4 opcode: PFP (prefetch parser) sync ME.
const PM4_PFP_SYNC_ME: u32 = 0x86;

/// PM4 opcode: DMA data (SDMA copy).
const PM4_DMA_DATA: u32 = 0x50;

// ------------------------------------------------------------------
// Shader register offsets (SI/CI/VI — may need adjustment for RDNA)
// ------------------------------------------------------------------

/// COMPUTE_PGM_LO — shader address bits 31:6.
const SH_REG_COMPUTE_PGM_LO: u32 = 0x0E40;

/// COMPUTE_PGM_HI — shader address bits 47:32.
const SH_REG_COMPUTE_PGM_HI: u32 = 0x0E41;

/// COMPUTE_TMPRING_SIZE — scratch memory size.
const SH_REG_COMPUTE_TMPRING_SIZE: u32 = 0x0E44;

/// COMPUTE_USER_DATA_0 — first user SGPR.
const SH_REG_COMPUTE_USER_DATA_0: u32 = 0x0E4C;

/// COMPUTE_RESOURCE_LIMITS — CU masking.
const SH_REG_COMPUTE_RESOURCE_LIMITS: u32 = 0x0E4E;

/// COMPUTE_NUM_THREAD_X — thread group size X.
const SH_REG_COMPUTE_NUM_THREAD_X: u32 = 0x0E54;

/// COMPUTE_NUM_THREAD_Y — thread group size Y.
const SH_REG_COMPUTE_NUM_THREAD_Y: u32 = 0x0E55;

/// COMPUTE_NUM_THREAD_Z — thread group size Z.
const SH_REG_COMPUTE_NUM_THREAD_Z: u32 = 0x0E56;

// ------------------------------------------------------------------
// Ring buffer
// ------------------------------------------------------------------

/// A GPU command ring buffer.
pub struct ComputeRing {
    /// BAR0 physical base address for MMIO.
    bar0: u64,
    /// Host-allocated backing store (must be GPU-visible).
    /// This is a raw pointer because the memory is identity-mapped
    /// physical memory owned by the physical frame allocator, not the heap.
    base: *mut u32,
    /// Cached write pointer (DWord index).
    wptr: u32,
    /// Cached read pointer (last known).
    rptr: u32,
    /// Size in DWords (power of two).
    size_dw: u32,
    /// Physical address of `base` (for GPU).
    phys_addr: u64,
}

/// A fence handle returned by dispatch.
#[derive(Debug, Clone, Copy)]
pub struct GpuFence {
    pub wptr_at_submit: u32,
    pub eop_addr: u64,
}

impl ComputeRing {
    /// Create a new compute ring.  `base` must point to GPU-visible physical
    /// memory that is identity-mapped for CPU access.  `size_dw` must be a power of two ≥ 256.
    ///
    /// # Safety
    /// `bar0` must be a valid, mapped PCI BAR0 base address for the target GPU.
    /// `base` must point to identity-mapped physical memory that is visible to
    /// both the CPU and the GPU, and must remain valid for the lifetime of the ring.
    /// `size_dw` must be a power of two and at least 256. Incorrect values can
    /// cause GPU hangs, memory corruption, or system instability.
    pub unsafe fn new(bar0: u64, base: *mut u32, size_dw: u32, phys_addr: u64) -> Option<Self> {
        if size_dw.count_ones() != 1 || size_dw < 256 {
            serial_println!("[COMPUTE] Ring buffer size {} is not a power of two or < 256", size_dw);
            return None;
        }

        let ring = Self {
            bar0,
            base,
            wptr: 0,
            rptr: 0,
            size_dw,
            phys_addr,
        };

        // Reset CP ME.
        core::ptr::write_volatile((bar0 + MM_CP_ME_CNTL) as *mut u32, 0x80000000);
        for _ in 0..1000 {
            core::arch::x86_64::_mm_pause();
        }
        core::ptr::write_volatile((bar0 + MM_CP_ME_CNTL) as *mut u32, 0x00000000);

        // Program ring base address.
        core::ptr::write_volatile(
            (bar0 + MM_CP_RB0_BASE) as *mut u32,
            (phys_addr & 0xFFFFFFFF) as u32,
        );
        core::ptr::write_volatile(
            (bar0 + MM_CP_RB0_BASE_HI) as *mut u32,
            ((phys_addr >> 32) & 0xFFFFFFFF) as u32,
        );

        // Program ring size: size in DWords encoded as log2(size) - 1.
        let rb_bufsz = size_dw.trailing_zeros() - 1;
        core::ptr::write_volatile((bar0 + MM_CP_RB0_BUFSZ) as *mut u32, rb_bufsz);

        // Enable ring buffer.
        core::ptr::write_volatile((bar0 + MM_CP_RB0_CNTL) as *mut u32, 0x80000000 | rb_bufsz);

        // Zero WPTR/RPTR.
        core::ptr::write_volatile((bar0 + MM_CP_RB0_WPTR) as *mut u32, 0);
        core::ptr::write_volatile((bar0 + MM_CP_RB0_RPTR) as *mut u32, 0);

        serial_println!(
            "[COMPUTE] Ring buffer initialised: phys={:016X} size={} DW",
            phys_addr,
            size_dw
        );
        Some(ring)
    }

    /// Write a single DWord to the ring, wrapping as necessary.
    fn write_dw(&mut self, dw: u32) {
        let idx = self.wptr & (self.size_dw - 1);
        unsafe { core::ptr::write_volatile(self.base.add(idx as usize), dw); }
        self.wptr = self.wptr.wrapping_add(1);
    }

    /// Commit the current write pointer to hardware.
    unsafe fn commit(&self) {
        fence(Ordering::SeqCst);
        core::ptr::write_volatile((self.bar0 + MM_CP_RB0_WPTR) as *mut u32, self.wptr);
    }

    /// Read the current hardware RPTR.
    unsafe fn read_rptr(&self) -> u32 {
        core::ptr::read_volatile((self.bar0 + MM_CP_RB0_RPTR) as *const u32)
    }

    /// Return the number of free DWords in the ring.
    fn free_dw(&self) -> u32 {
        let rptr = self.rptr;
        let used = self.wptr.wrapping_sub(rptr);
        self.size_dw - used
    }

    /// Wait until the ring has at least `need` free DWords.
    unsafe fn wait_free(&mut self, need: u32) {
        while self.free_dw() < need {
            self.rptr = self.read_rptr();
            core::arch::x86_64::_mm_pause();
        }
    }

    /// Build and submit a PM4 type-3 packet header.
    /// `opcode` — PM4 opcode (e.g., PM4_DISPATCH_DIRECT)
    /// `count` — number of payload DWords following the header (header itself not counted)
    fn packet3(&mut self, opcode: u32, count: u32) {
        let header = ((count & 0x3FFF) << 16) | ((opcode & 0xFF) << 8) | (PM4_TYPE3 << 30);
        self.write_dw(header);
    }

    // ------------------------------------------------------------------
    // High-level packet builders
    // ------------------------------------------------------------------

    /// Write shader registers.  `reg_offset` is the first SH register index.
    /// `values` are the register values to write consecutively.
    ///
    /// # Safety
    /// `reg_offset` must be a valid shader register offset for the target GPU
    /// architecture. Writing to reserved or incorrect registers can alter GPU
    /// execution state, cause shader hangs, or corrupt GPU memory.
    pub unsafe fn set_sh_regs(&mut self, reg_offset: u32, values: &[u32]) {
        let need = 2 + values.len() as u32;
        self.wait_free(need);
        self.packet3(PM4_SET_SH_REG, 1 + values.len() as u32);
        self.write_dw(reg_offset - 0x2000); // base offset adjustment for SET_SH_REG
        for &v in values {
            self.write_dw(v);
        }
    }

    /// Dispatch a compute grid directly.
    /// `dim_x`, `dim_y`, `dim_z` — number of thread groups in each dimension.
    ///
    /// # Safety
    /// Caller must ensure a valid compute shader has been loaded via
    /// `set_compute_pgm` and that all required resources (user data, thread
    /// counts) are correctly configured. Dispatching without proper setup can
    /// cause GPU hangs, data corruption, or out-of-bounds memory accesses.
    pub unsafe fn dispatch_direct(&mut self, dim_x: u32, dim_y: u32, dim_z: u32) {
        let need = 4;
        self.wait_free(need);
        self.packet3(PM4_DISPATCH_DIRECT, 3);
        self.write_dw(dim_x);
        self.write_dw(dim_y);
        self.write_dw(dim_z);
    }

    /// Insert an end-of-pipeline event that writes a fence value to `addr`.
    /// `addr` must be GPU-visible physical address.
    pub unsafe fn event_write_eop(&mut self, addr: u64, value: u64) {
        let need = 6;
        self.wait_free(need);
        self.packet3(PM4_EVENT_WRITE_EOP, 5);
        // EVENT_TYPE = CACHE_FLUSH_AND_INV_TS_EVENT (0x28)
        // EVENT_INDEX = EOP
        self.write_dw(0x00000028);
        self.write_dw((addr & 0xFFFFFFFF) as u32);
        self.write_dw(((addr >> 32) & 0xFFFFFFFF) as u32);
        self.write_dw((value & 0xFFFFFFFF) as u32);
        self.write_dw(((value >> 32) & 0xFFFFFFFF) as u32);
    }

    /// SDMA copy: host-visible `src` → host-visible `dst`, both physical addresses.
    ///
    /// # Safety
    /// `dst_phys` and `src_phys` must be valid, GPU-visible physical addresses
    /// with at least `size_bytes` of accessible memory. The regions must not
    /// overlap. Incorrect addresses can cause DMA to invalid memory, data
    /// corruption, or system instability.
    pub unsafe fn dma_data(
        &mut self,
        dst_phys: u64,
        src_phys: u64,
        size_bytes: u32,
    ) {
        let need = 7;
        self.wait_free(need);
        self.packet3(PM4_DMA_DATA, 6);
        // SRC_SEL = 0 (SRC_ADDR), DST_SEL = 0 (DST_ADDR), CP_SYNC = 1
        self.write_dw(0x00000001);
        self.write_dw((src_phys & 0xFFFFFFFF) as u32);
        self.write_dw(((src_phys >> 32) & 0xFFFFFFFF) as u32);
        self.write_dw((dst_phys & 0xFFFFFFFF) as u32);
        self.write_dw(((dst_phys >> 32) & 0xFFFFFFFF) as u32);
        self.write_dw(size_bytes);
    }

    /// Upload shader constants (user data SGPRs).
    pub unsafe fn set_user_data(&mut self, start_idx: u32, values: &[u32]) {
        let reg = SH_REG_COMPUTE_USER_DATA_0 + start_idx;
        self.set_sh_regs(reg, values);
    }

    /// Configure thread group dimensions.
    pub unsafe fn set_num_threads(&mut self, x: u32, y: u32, z: u32) {
        self.set_sh_regs(SH_REG_COMPUTE_NUM_THREAD_X, &[x, y, z]);
    }

    /// Set compute shader address.
    ///
    /// # Safety
    /// `shader_phys` must be a valid, GPU-visible physical address pointing to
    /// correctly formed shader code for the target GPU architecture. An invalid
    /// address or malformed shader will cause a GPU hang or undefined behavior
    /// when dispatch is triggered.
    pub unsafe fn set_compute_pgm(&mut self, shader_phys: u64) {
        let lo = ((shader_phys >> 8) & 0xFFFFFFFF) as u32;
        let hi = ((shader_phys >> 40) & 0xFFFF) as u32;
        self.set_sh_regs(SH_REG_COMPUTE_PGM_LO, &[lo, hi]);
    }

    /// Submit all queued packets to the GPU and return a fence.
    ///
    /// # Safety
    /// `fence_addr` must be a valid, GPU-visible physical address with at least
    /// 8 bytes of writable memory. The ring must have been correctly initialized
    /// and all queued packets must form a valid command sequence. Submission of
    /// malformed commands can hang the GPU or corrupt memory.
    pub unsafe fn submit(&mut self, fence_addr: u64) -> GpuFence {
        let wptr_before = self.wptr;
        self.event_write_eop(fence_addr, 0xABCDEF00);
        self.commit();
        GpuFence {
            wptr_at_submit: wptr_before,
            eop_addr: fence_addr,
        }
    }

    /// Poll until the fence value at `fence.eop_addr` becomes non-zero.
    ///
    /// # Safety
    /// `fence.eop_addr` must be a valid, mapped physical address that was
    /// previously passed to `submit` and is writable by the GPU. Reading from
    /// an invalid address causes a page fault or memory corruption.
    pub unsafe fn wait_fence(&self, fence: &GpuFence, timeout_ms: u32) -> bool {
        let start = unsafe { core::arch::x86_64::_rdtsc() };
        // Approximate: assume 2.5 GHz TSC.
        let timeout_ticks = (timeout_ms as u64) * 2_500_000;
        loop {
            let val = core::ptr::read_volatile(fence.eop_addr as *const u32);
            if val != 0 {
                return true;
            }
            if unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(start) > timeout_ticks {
                serial_println!("[COMPUTE] Fence timeout at {:016X}", fence.eop_addr);
                return false;
            }
            core::arch::x86_64::_mm_pause();
        }
    }
}

// ------------------------------------------------------------------
// Command buffer allocator
// ------------------------------------------------------------------

/// A growable command buffer that flushes into a `ComputeRing`.
pub struct CommandBuffer {
    cmds: Vec<u32>,
}

impl CommandBuffer {
    pub fn new() -> Self {
        Self { cmds: Vec::with_capacity(256) }
    }

    pub fn clear(&mut self) {
        self.cmds.clear();
    }

    pub fn emit(&mut self, dw: u32) {
        self.cmds.push(dw);
    }

    pub fn emit_slice(&mut self, dws: &[u32]) {
        self.cmds.extend_from_slice(dws);
    }

    /// Write all accumulated commands into the ring as a single batch.
    pub unsafe fn submit_to_ring(&self, ring: &mut ComputeRing, fence_addr: u64) -> GpuFence {
        for &dw in &self.cmds {
            ring.write_dw(dw);
        }
        ring.submit(fence_addr)
    }
}
