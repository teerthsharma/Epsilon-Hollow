// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Unified Topology Accelerator — trait definition and AMD GCN implementation.
//!
//! This module provides a vendor-agnostic interface for offloading topological
//! computations to the GPU.  Kernel subsystems (TopoRAM, ManifoldFS, Scheduler)
//! use `TopologyAccelerator` without knowing the underlying vendor.

use super::compute_queue::{ComputeRing, GpuFence};
use super::firmware_scanner::{FallbackReason, GpuFirmwareCaps};
use super::gpu_mem::{download, upload, GpuBuffer};
use super::shader_binaries::{find_kernel, KernelMeta};
use crate::drivers::gpu::amd::AmdGpu;
use crate::drivers::pci::PciDevice;
use crate::serial_println;

// ------------------------------------------------------------------
// Error types
// ------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuError {
    NotReady,
    NoFirmware,
    InvalidArgument,
    OutOfMemory,
    RingFull,
    FenceTimeout,
    KernelNotFound,
    VendorError(u32),
}

// ------------------------------------------------------------------
// Trait
// ------------------------------------------------------------------

/// Unified interface for GPU-accelerated topology operations.
pub trait TopologyAccelerator {
    /// Probe firmware and initialise hardware.
    ///
    /// # Safety
    /// `pci_dev` must describe a valid, enumerated PCI device that matches the
    /// vendor specified in `caps`. The caller must ensure that no other code is
    /// concurrently accessing the GPU MMIO registers. Incorrect arguments can
    /// leave the GPU in an undefined state or cause system hangs.
    unsafe fn init(&mut self, pci_dev: &PciDevice, caps: &GpuFirmwareCaps) -> Result<(), GpuError>;

    /// Allocate GPU-visible memory.
    fn alloc_gpu_mem(&mut self, size: usize) -> Result<GpuBuffer, GpuError>;

    /// Free GPU-visible memory.
    fn free_gpu_mem(&mut self, buf: GpuBuffer);

    /// Copy host → GPU.
    fn upload(&mut self, dst: &GpuBuffer, src: &[u8]) -> Result<(), GpuError>;

    /// Copy GPU → host.
    fn download(&mut self, dst: &mut [u8], src: &GpuBuffer) -> Result<(), GpuError>;

    /// Launch Voronoi cell assignment kernel (T1).
    fn dispatch_voronoi(
        &mut self,
        points: &GpuBuffer,
        centroids: &GpuBuffer,
        cell_ids: &GpuBuffer,
        n_points: usize,
        n_cells: usize,
    ) -> Result<GpuFence, GpuError>;

    /// Launch JL projection kernel (128-dim sparse → 3-dim).
    fn dispatch_jl_project(
        &mut self,
        input: &GpuBuffer,
        output: &GpuBuffer,
        n_vectors: usize,
        seed: u32,
    ) -> Result<GpuFence, GpuError>;

    /// Launch one spectral contraction step (T2).
    fn dispatch_spectral_step(
        &mut self,
        state: &GpuBuffer,
        output: &GpuBuffer,
        dim: usize,
        alpha: f64,
    ) -> Result<GpuFence, GpuError>;

    /// Launch S² great-circle distance matrix kernel.
    fn dispatch_s2_distance(
        &mut self,
        points_a: &GpuBuffer,
        points_b: &GpuBuffer,
        out: &GpuBuffer,
        n_a: usize,
        n_b: usize,
    ) -> Result<GpuFence, GpuError>;

    /// Block until fence signals completion.
    fn wait_fence(&mut self, fence: &GpuFence, timeout_ms: u32) -> Result<(), GpuError>;

    /// Returns true if this accelerator is ready.
    fn is_ready(&self) -> bool;

    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Return detected capability vector.
    fn caps(&self) -> GpuFirmwareCaps;
}

// ------------------------------------------------------------------
// AMD GCN Implementation
// ------------------------------------------------------------------

pub struct AmdTopologyAccelerator {
    gpu: Option<AmdGpu>,
    ring: Option<ComputeRing>,
    fence_buf: Option<GpuBuffer>,
    caps: GpuFirmwareCaps,
    ready: bool,
}

// Safe because AmdTopologyAccelerator only stores raw MMIO pointers and
// physical addresses that are never aliased by userpace.
unsafe impl Send for AmdTopologyAccelerator {}

impl AmdTopologyAccelerator {
    pub const fn new() -> Self {
        Self {
            gpu: None,
            ring: None,
            fence_buf: None,
            caps: GpuFirmwareCaps::none(),
            ready: false,
        }
    }

    unsafe fn ensure_ring(&mut self) -> Result<&mut ComputeRing, GpuError> {
        self.ring.as_mut().ok_or(GpuError::NotReady)
    }

    unsafe fn upload_kernel(&mut self, meta: &KernelMeta) -> Result<u64, GpuError> {
        let code_bytes = meta.code_size_bytes;
        let code_buf = GpuBuffer::alloc(code_bytes).ok_or(GpuError::OutOfMemory)?;
        upload(
            &code_buf,
            super::shader_binaries::kernel_as_bytes(meta.binary),
        );
        Ok(code_buf.phys)
    }
}

impl TopologyAccelerator for AmdTopologyAccelerator {
    unsafe fn init(&mut self, pci_dev: &PciDevice, caps: &GpuFirmwareCaps) -> Result<(), GpuError> {
        if caps.vendor != super::firmware_scanner::GpuFirmwareVendor::Amd {
            return Err(GpuError::NoFirmware);
        }
        if !caps.has_compute_queue {
            return Err(GpuError::VendorError(0x01));
        }

        serial_println!("[AMD-ACCEL] Initialising topology accelerator...");

        // Probe the AMD GPU (already done by scanner, but we need the struct).
        let gpu = super::amd::AmdGpu::probe(pci_dev).ok_or(GpuError::VendorError(0x02))?;
        let bar0 = gpu.bar0_phys;

        // Allocate ring buffer (4 pages = 16 KiB = 4096 DWords).
        let ring_pages = 4usize;
        let ring_phys = crate::memory::phys::alloc_frames_contiguous(ring_pages)
            .ok_or(GpuError::OutOfMemory)?
            .as_u64();
        let ring_base = ring_phys as *mut u32;
        let ring_size_dw = (ring_pages * 4096 / 4) as u32;

        let ring = unsafe { ComputeRing::new(bar0, ring_base, ring_size_dw, ring_phys) }
            .ok_or(GpuError::VendorError(0x03))?;

        // Allocate fence buffer (one u64 for EOP signalling).
        let fence_buf = GpuBuffer::alloc(8).ok_or(GpuError::OutOfMemory)?;

        self.gpu = Some(gpu);
        self.ring = Some(ring);
        self.fence_buf = Some(fence_buf);
        self.caps = *caps;
        self.ready = true;

        serial_println!("[AMD-ACCEL] Ready — BAR0={:016X}", bar0);
        Ok(())
    }

    fn alloc_gpu_mem(&mut self, size: usize) -> Result<GpuBuffer, GpuError> {
        GpuBuffer::alloc(size).ok_or(GpuError::OutOfMemory)
    }

    fn free_gpu_mem(&mut self, buf: GpuBuffer) {
        buf.free();
    }

    fn upload(&mut self, dst: &GpuBuffer, src: &[u8]) -> Result<(), GpuError> {
        unsafe {
            upload(dst, src);
        }
        Ok(())
    }

    fn download(&mut self, dst: &mut [u8], src: &GpuBuffer) -> Result<(), GpuError> {
        unsafe {
            download(dst, src);
        }
        Ok(())
    }

    fn dispatch_voronoi(
        &mut self,
        points: &GpuBuffer,
        centroids: &GpuBuffer,
        cell_ids: &GpuBuffer,
        n_points: usize,
        n_cells: usize,
    ) -> Result<GpuFence, GpuError> {
        if !self.ready {
            return Err(GpuError::NotReady);
        }
        let meta = find_kernel("voronoi_assign").ok_or(GpuError::KernelNotFound)?;
        let shader_phys = unsafe { self.upload_kernel(meta)? };

        let fence_phys = self.fence_buf.as_ref().ok_or(GpuError::NotReady)?.phys;
        let ring = unsafe { self.ensure_ring()? };

        unsafe {
            // Set compute shader address.
            ring.set_compute_pgm(shader_phys);
            // Thread group: 64 threads per wavefront.
            let grid_x = ((n_points + 63) / 64) as u32;
            ring.set_num_threads(64, 1, 1);
            // Set user data: argument buffer layout.
            // s4-s5: points, s6-s7: centroids, s8-s9: cell_ids, s10: n_points, s11: n_cells
            ring.set_user_data(
                0,
                &[
                    (points.phys & 0xFFFFFFFF) as u32,
                    ((points.phys >> 32) & 0xFFFFFFFF) as u32,
                    (centroids.phys & 0xFFFFFFFF) as u32,
                    ((centroids.phys >> 32) & 0xFFFFFFFF) as u32,
                    (cell_ids.phys & 0xFFFFFFFF) as u32,
                    ((cell_ids.phys >> 32) & 0xFFFFFFFF) as u32,
                    n_points as u32,
                    n_cells as u32,
                ],
            );
            ring.dispatch_direct(grid_x, 1, 1);
            Ok(ring.submit(fence_phys))
        }
    }

    fn dispatch_jl_project(
        &mut self,
        input: &GpuBuffer,
        output: &GpuBuffer,
        n_vectors: usize,
        seed: u32,
    ) -> Result<GpuFence, GpuError> {
        if !self.ready {
            return Err(GpuError::NotReady);
        }
        let meta = find_kernel("jl_project").ok_or(GpuError::KernelNotFound)?;
        let shader_phys = unsafe { self.upload_kernel(meta)? };
        let fence_phys = self.fence_buf.as_ref().ok_or(GpuError::NotReady)?.phys;
        let ring = unsafe { self.ensure_ring()? };

        unsafe {
            ring.set_compute_pgm(shader_phys);
            let grid_x = ((n_vectors + 63) / 64) as u32;
            ring.set_num_threads(64, 1, 1);
            ring.set_user_data(
                0,
                &[
                    (input.phys & 0xFFFFFFFF) as u32,
                    ((input.phys >> 32) & 0xFFFFFFFF) as u32,
                    (output.phys & 0xFFFFFFFF) as u32,
                    ((output.phys >> 32) & 0xFFFFFFFF) as u32,
                    n_vectors as u32,
                    seed,
                ],
            );
            ring.dispatch_direct(grid_x, 1, 1);
            Ok(ring.submit(fence_phys))
        }
    }

    fn dispatch_spectral_step(
        &mut self,
        state: &GpuBuffer,
        output: &GpuBuffer,
        dim: usize,
        alpha: f64,
    ) -> Result<GpuFence, GpuError> {
        if !self.ready {
            return Err(GpuError::NotReady);
        }
        let meta = find_kernel("spectral_step").ok_or(GpuError::KernelNotFound)?;
        let shader_phys = unsafe { self.upload_kernel(meta)? };
        let fence_phys = self.fence_buf.as_ref().ok_or(GpuError::NotReady)?.phys;
        let ring = unsafe { self.ensure_ring()? };

        let alpha_bits = alpha.to_bits();
        unsafe {
            ring.set_compute_pgm(shader_phys);
            let grid_x = ((dim + 63) / 64) as u32;
            ring.set_num_threads(64, 1, 1);
            ring.set_user_data(
                0,
                &[
                    (state.phys & 0xFFFFFFFF) as u32,
                    ((state.phys >> 32) & 0xFFFFFFFF) as u32,
                    (output.phys & 0xFFFFFFFF) as u32,
                    ((output.phys >> 32) & 0xFFFFFFFF) as u32,
                    dim as u32,
                    alpha_bits as u32,
                    (alpha_bits >> 32) as u32,
                ],
            );
            ring.dispatch_direct(grid_x, 1, 1);
            Ok(ring.submit(fence_phys))
        }
    }

    fn dispatch_s2_distance(
        &mut self,
        points_a: &GpuBuffer,
        points_b: &GpuBuffer,
        out: &GpuBuffer,
        n_a: usize,
        n_b: usize,
    ) -> Result<GpuFence, GpuError> {
        if !self.ready {
            return Err(GpuError::NotReady);
        }
        let meta = find_kernel("s2_distance").ok_or(GpuError::KernelNotFound)?;
        let shader_phys = unsafe { self.upload_kernel(meta)? };
        let fence_phys = self.fence_buf.as_ref().ok_or(GpuError::NotReady)?.phys;
        let ring = unsafe { self.ensure_ring()? };

        unsafe {
            ring.set_compute_pgm(shader_phys);
            let grid_x = ((n_a + 7) / 8) as u32;
            let grid_y = ((n_b + 7) / 8) as u32;
            ring.set_num_threads(8, 8, 1);
            ring.set_user_data(
                0,
                &[
                    (points_a.phys & 0xFFFFFFFF) as u32,
                    ((points_a.phys >> 32) & 0xFFFFFFFF) as u32,
                    (points_b.phys & 0xFFFFFFFF) as u32,
                    ((points_b.phys >> 32) & 0xFFFFFFFF) as u32,
                    (out.phys & 0xFFFFFFFF) as u32,
                    ((out.phys >> 32) & 0xFFFFFFFF) as u32,
                    n_a as u32,
                    n_b as u32,
                ],
            );
            ring.dispatch_direct(grid_x, grid_y, 1);
            Ok(ring.submit(fence_phys))
        }
    }

    fn wait_fence(&mut self, fence: &GpuFence, timeout_ms: u32) -> Result<(), GpuError> {
        let ring = unsafe { self.ensure_ring()? };
        if unsafe { ring.wait_fence(fence, timeout_ms) } {
            Ok(())
        } else {
            Err(GpuError::FenceTimeout)
        }
    }

    fn is_ready(&self) -> bool {
        self.ready
    }

    fn name(&self) -> &'static str {
        "AMD GCN Topology Accelerator"
    }

    fn caps(&self) -> GpuFirmwareCaps {
        self.caps
    }
}

// ------------------------------------------------------------------
// Null / CPU fallback accelerator
// ------------------------------------------------------------------

pub struct CpuFallbackAccelerator;

/// Simple hash-based pseudo-random float in [-1, 1] for JL projection.
fn hash_f64(seed: u32, a: usize, b: usize) -> f64 {
    let mut x = seed
        .wrapping_add((a as u32).wrapping_mul(0x9E37_79B9))
        .wrapping_add((b as u32).wrapping_mul(0x85EB_CA6B));
    x = x.wrapping_mul(0xC2B2_AE35);
    ((x as i32) as f64) / (i32::MAX as f64)
}

impl TopologyAccelerator for CpuFallbackAccelerator {
    unsafe fn init(
        &mut self,
        _pci_dev: &PciDevice,
        _caps: &GpuFirmwareCaps,
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn alloc_gpu_mem(&mut self, size: usize) -> Result<GpuBuffer, GpuError> {
        GpuBuffer::alloc(size).ok_or(GpuError::OutOfMemory)
    }

    fn free_gpu_mem(&mut self, buf: GpuBuffer) {
        buf.free();
    }

    fn upload(&mut self, dst: &GpuBuffer, src: &[u8]) -> Result<(), GpuError> {
        unsafe {
            super::gpu_mem::upload(dst, src);
        }
        Ok(())
    }

    fn download(&mut self, dst: &mut [u8], src: &GpuBuffer) -> Result<(), GpuError> {
        unsafe {
            super::gpu_mem::download(dst, src);
        }
        Ok(())
    }

    fn dispatch_voronoi(
        &mut self,
        points: &GpuBuffer,
        centroids: &GpuBuffer,
        cell_ids: &GpuBuffer,
        n_points: usize,
        n_cells: usize,
    ) -> Result<GpuFence, GpuError> {
        let pts =
            unsafe { core::slice::from_raw_parts(points.phys as *const f64, points.size / 8) };
        let ctr = unsafe {
            core::slice::from_raw_parts(centroids.phys as *const f64, centroids.size / 8)
        };
        let out = unsafe {
            core::slice::from_raw_parts_mut(cell_ids.phys as *mut u32, cell_ids.size / 4)
        };

        for p in 0..n_points {
            let p_theta = pts[p * 2];
            let p_phi = pts[p * 2 + 1];
            let mut best_dist = core::f64::MAX;
            let mut best_cell = 0u32;
            for c in 0..n_cells {
                let c_theta = ctr[c * 2];
                let c_phi = ctr[c * 2 + 1];
                let cos_dist = libm::sin(p_theta) * libm::sin(c_theta)
                    + libm::cos(p_theta) * libm::cos(c_theta) * libm::cos(p_phi - c_phi);
                let dist = libm::acos(cos_dist.clamp(-1.0, 1.0));
                if dist < best_dist {
                    best_dist = dist;
                    best_cell = c as u32;
                }
            }
            out[p] = best_cell;
        }
        Ok(GpuFence {
            wptr_at_submit: 0,
            eop_addr: 0,
        })
    }

    fn dispatch_jl_project(
        &mut self,
        input: &GpuBuffer,
        output: &GpuBuffer,
        n_vectors: usize,
        seed: u32,
    ) -> Result<GpuFence, GpuError> {
        const DIM_IN: usize = 128;
        const DIM_OUT: usize = 3;
        let inp = unsafe { core::slice::from_raw_parts(input.phys as *const f64, input.size / 8) };
        let out =
            unsafe { core::slice::from_raw_parts_mut(output.phys as *mut f64, output.size / 8) };

        for v in 0..n_vectors {
            for d in 0..DIM_OUT {
                let mut sum = 0.0f64;
                for i in 0..DIM_IN {
                    sum += inp[v * DIM_IN + i] * hash_f64(seed, i, d);
                }
                out[v * DIM_OUT + d] = sum / libm::sqrt(DIM_IN as f64);
            }
        }
        Ok(GpuFence {
            wptr_at_submit: 0,
            eop_addr: 0,
        })
    }

    fn dispatch_spectral_step(
        &mut self,
        state: &GpuBuffer,
        output: &GpuBuffer,
        dim: usize,
        alpha: f64,
    ) -> Result<GpuFence, GpuError> {
        let st = unsafe { core::slice::from_raw_parts(state.phys as *const f64, state.size / 8) };
        let out =
            unsafe { core::slice::from_raw_parts_mut(output.phys as *mut f64, output.size / 8) };
        let one_minus_alpha = 1.0 - alpha;
        for i in 0..dim {
            // Damped relaxation step: x' = (1-α)*x
            out[i] = st[i] * one_minus_alpha;
        }
        Ok(GpuFence {
            wptr_at_submit: 0,
            eop_addr: 0,
        })
    }

    fn dispatch_s2_distance(
        &mut self,
        points_a: &GpuBuffer,
        points_b: &GpuBuffer,
        out: &GpuBuffer,
        n_a: usize,
        n_b: usize,
    ) -> Result<GpuFence, GpuError> {
        let a =
            unsafe { core::slice::from_raw_parts(points_a.phys as *const f64, points_a.size / 8) };
        let b =
            unsafe { core::slice::from_raw_parts(points_b.phys as *const f64, points_b.size / 8) };
        let o = unsafe { core::slice::from_raw_parts_mut(out.phys as *mut f64, out.size / 8) };
        for i in 0..n_a {
            let a_theta = a[i * 2];
            let a_phi = a[i * 2 + 1];
            for j in 0..n_b {
                let b_theta = b[j * 2];
                let b_phi = b[j * 2 + 1];
                let cos_dist = libm::sin(a_theta) * libm::sin(b_theta)
                    + libm::cos(a_theta) * libm::cos(b_theta) * libm::cos(a_phi - b_phi);
                o[i * n_b + j] = libm::acos(cos_dist.clamp(-1.0, 1.0));
            }
        }
        Ok(GpuFence {
            wptr_at_submit: 0,
            eop_addr: 0,
        })
    }

    fn wait_fence(&mut self, _fence: &GpuFence, _timeout_ms: u32) -> Result<(), GpuError> {
        // CPU fallback is synchronous; nothing to wait for.
        Ok(())
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "CPU Fallback (software topology)"
    }

    fn caps(&self) -> GpuFirmwareCaps {
        let mut caps = GpuFirmwareCaps::none();
        caps.fallback_reason = Some(FallbackReason::NoGpuFound);
        caps
    }
}

// ------------------------------------------------------------------
// Global accelerator instance
// ------------------------------------------------------------------

use spin::Mutex;

static ACCELERATOR: Mutex<Option<alloc::boxed::Box<dyn TopologyAccelerator + Send>>> =
    Mutex::new(None);

/// Initialise the best available topology accelerator.
/// Called during kernel boot after PCI enumeration.
pub fn init_topology_accelerator() {
    let caps = super::firmware_scanner::scan_all();
    super::firmware_scanner::print_report(&caps);

    let mut accel_guard = ACCELERATOR.lock();

    if !caps.is_accelerated() {
        serial_println!("[TOPO-ACCEL] No GPU acceleration available — using CPU fallback");
        *accel_guard = Some(alloc::boxed::Box::new(CpuFallbackAccelerator));
        return;
    }

    // Find the PCI device that matches the scanned capabilities.
    let devices = crate::drivers::pci::get_devices();
    let matched_dev = devices.iter().find(|d| {
        d.class == 0x03
            && match caps.vendor {
                super::firmware_scanner::GpuFirmwareVendor::Amd => d.vendor_id == 0x1002,
                super::firmware_scanner::GpuFirmwareVendor::Nvidia => d.vendor_id == 0x10DE,
                super::firmware_scanner::GpuFirmwareVendor::Intel => d.vendor_id == 0x8086,
                _ => false,
            }
    });

    let Some(dev) = matched_dev else {
        serial_println!("[TOPO-ACCEL] Scanned caps matched but PCI device disappeared");
        *accel_guard = Some(alloc::boxed::Box::new(CpuFallbackAccelerator));
        return;
    };

    match caps.vendor {
        super::firmware_scanner::GpuFirmwareVendor::Amd => {
            let mut amd = AmdTopologyAccelerator::new();
            if unsafe { amd.init(dev, &caps) }.is_ok() {
                serial_println!("[TOPO-ACCEL] AMD GCN accelerator initialised");
                *accel_guard = Some(alloc::boxed::Box::new(amd));
            } else {
                serial_println!("[TOPO-ACCEL] AMD init failed — falling back to CPU");
                *accel_guard = Some(alloc::boxed::Box::new(CpuFallbackAccelerator));
            }
        }
        _ => {
            serial_println!("[TOPO-ACCEL] Vendor stub — falling back to CPU");
            *accel_guard = Some(alloc::boxed::Box::new(CpuFallbackAccelerator));
        }
    }
}

/// Access the global accelerator (read-only reference).
pub fn with_accelerator<F, R>(f: F) -> R
where
    F: FnOnce(&dyn TopologyAccelerator) -> R,
{
    let guard = ACCELERATOR.lock();
    match guard.as_ref() {
        Some(accel) => f(accel.as_ref()),
        None => {
            let fallback = CpuFallbackAccelerator;
            f(&fallback)
        }
    }
}

/// Access the global accelerator (mutable reference).
pub fn with_accelerator_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut dyn TopologyAccelerator) -> R,
{
    let mut guard = ACCELERATOR.lock();
    match guard.as_mut() {
        Some(accel) => f(accel.as_mut()),
        None => {
            let mut fallback = CpuFallbackAccelerator;
            f(&mut fallback)
        }
    }
}
