// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! AMD GPU Hardware Dispatch Proof — end-to-end PM4 compute packet submission.
//!
//! This module provides a standalone proof that Seal OS can:
//! 1. Map a PM4 command ring on real AMD hardware via BAR0 MMIO.
//! 2. Upload a GCN compute shader to GPU-visible memory.
//! 3. Build and submit a `DISPATCH_DIRECT` packet with user-data SGPRs.
//! 4. Wait for GPU completion via an `EVENT_WRITE_EOP` fence.
//! 5. Read back results and verify they are in bounds.
//!
//! On success it prints the canonical sentinel:
//!   `[GPU-BENCH] voronoi result=OK cycles=<n>`
//!
//! # Safety
//! All public functions in this module are `unsafe` because they touch raw PCI
//! BAR0 MMIO registers and assume the caller has already completed PCI
//! enumeration.  Running this on non-AMD hardware will silently fail or hang.

use super::amd::AmdGpu;
use super::compute_queue::ComputeRing;
use super::gpu_mem::{download, upload, GpuBuffer};
use super::shader_binaries::{find_kernel, kernel_as_bytes};
use crate::serial_println;

const N_POINTS: usize = 64;
const N_CELLS: usize = 8;

/// Run a single end-to-end hardware dispatch proof on the first AMD GPU found.
///
/// Returns `true` if the GPU executed the voronoi kernel and produced valid
/// cell IDs, `false` otherwise.  The sentinel line is printed in both cases.
pub unsafe fn run_amd_hardware_proof() -> bool {
    let devices = crate::drivers::pci::get_devices();
    let dev = match devices
        .iter()
        .find(|d| d.class == 0x03 && d.vendor_id == 0x1002)
    {
        Some(d) => d,
        None => {
            serial_println!("[GPU-BENCH] voronoi result=FAIL reason=no_amd_gpu");
            return false;
        }
    };

    let gpu = match AmdGpu::probe(dev) {
        Some(g) => g,
        None => {
            serial_println!("[GPU-BENCH] voronoi result=FAIL reason=probe_failed");
            return false;
        }
    };

    // -----------------------------------------------------------------
    // 1. Allocate and initialise PM4 ring buffer.
    // -----------------------------------------------------------------
    let ring_pages = 4usize;
    let ring_phys = match crate::memory::phys::alloc_frames_contiguous(ring_pages) {
        Some(p) => p.as_u64(),
        None => {
            serial_println!("[GPU-BENCH] voronoi result=FAIL reason=ring_alloc_failed");
            return false;
        }
    };
    let ring_size_dw = (ring_pages * 4096 / 4) as u32;
    let mut ring = match ComputeRing::new(
        gpu.bar0_phys,
        ring_phys as *mut u32,
        ring_size_dw,
        ring_phys,
    ) {
        Some(r) => r,
        None => {
            serial_println!("[GPU-BENCH] voronoi result=FAIL reason=ring_init_failed");
            return false;
        }
    };

    // -----------------------------------------------------------------
    // 2. Allocate fence and kernel buffers.
    // -----------------------------------------------------------------
    let fence_buf: Option<GpuBuffer> = GpuBuffer::alloc(8);
    let mut kernel_buf: Option<GpuBuffer> = None;
    let mut points_buf: Option<GpuBuffer> = None;
    let mut centroids_buf: Option<GpuBuffer> = None;
    let mut cell_ids_buf: Option<GpuBuffer> = None;

    let meta = match find_kernel("voronoi_assign") {
        Some(m) => m,
        None => {
            serial_println!("[GPU-BENCH] voronoi result=FAIL reason=kernel_not_found");
            cleanup(
                kernel_buf,
                fence_buf,
                points_buf,
                centroids_buf,
                cell_ids_buf,
            );
            return false;
        }
    };

    kernel_buf = GpuBuffer::alloc(meta.code_size_bytes);
    if kernel_buf.is_none() {
        serial_println!("[GPU-BENCH] voronoi result=FAIL reason=kernel_alloc_failed");
        cleanup(
            kernel_buf,
            fence_buf,
            points_buf,
            centroids_buf,
            cell_ids_buf,
        );
        return false;
    }
    upload(kernel_buf.as_ref().unwrap(), kernel_as_bytes(meta.binary));

    points_buf = GpuBuffer::alloc(N_POINTS * 16);
    centroids_buf = GpuBuffer::alloc(N_CELLS * 16);
    cell_ids_buf = GpuBuffer::alloc(N_POINTS * 4);
    if points_buf.is_none() || centroids_buf.is_none() || cell_ids_buf.is_none() {
        serial_println!("[GPU-BENCH] voronoi result=FAIL reason=buffer_alloc_failed");
        cleanup(
            kernel_buf,
            fence_buf,
            points_buf,
            centroids_buf,
            cell_ids_buf,
        );
        return false;
    }

    // -----------------------------------------------------------------
    // 3. Fill host buffers and upload.
    // -----------------------------------------------------------------
    let mut points_host = [0u8; N_POINTS * 16];
    let mut centroids_host = [0u8; N_CELLS * 16];
    for i in 0..N_POINTS {
        let theta = (i as f64 / N_POINTS as f64) * core::f64::consts::PI;
        let phi = (i as f64 * 1.618) % (2.0 * core::f64::consts::PI);
        let off = i * 16;
        points_host[off..off + 8].copy_from_slice(&theta.to_le_bytes());
        points_host[off + 8..off + 16].copy_from_slice(&phi.to_le_bytes());
    }
    for i in 0..N_CELLS {
        let theta = (i as f64 / N_CELLS as f64) * core::f64::consts::PI;
        let phi = (i as f64 * 2.0) % (2.0 * core::f64::consts::PI);
        let off = i * 16;
        centroids_host[off..off + 8].copy_from_slice(&theta.to_le_bytes());
        centroids_host[off + 8..off + 16].copy_from_slice(&phi.to_le_bytes());
    }

    upload(points_buf.as_ref().unwrap(), &points_host);
    upload(centroids_buf.as_ref().unwrap(), &centroids_host);

    // -----------------------------------------------------------------
    // 4. Build PM4 packet: set shader, user data, thread count, dispatch.
    // -----------------------------------------------------------------
    let grid_x = ((N_POINTS + 63) / 64) as u32;
    let p = points_buf.as_ref().unwrap().phys;
    let c = centroids_buf.as_ref().unwrap().phys;
    let o = cell_ids_buf.as_ref().unwrap().phys;

    ring.set_compute_pgm(kernel_buf.as_ref().unwrap().phys);
    ring.set_num_threads(64, 1, 1);
    ring.set_user_data(
        0,
        &[
            (p & 0xFFFFFFFF) as u32,
            ((p >> 32) & 0xFFFFFFFF) as u32,
            (c & 0xFFFFFFFF) as u32,
            ((c >> 32) & 0xFFFFFFFF) as u32,
            (o & 0xFFFFFFFF) as u32,
            ((o >> 32) & 0xFFFFFFFF) as u32,
            N_POINTS as u32,
            N_CELLS as u32,
        ],
    );
    ring.dispatch_direct(grid_x, 1, 1);

    // -----------------------------------------------------------------
    // 5. Submit and wait for EOP fence.
    // -----------------------------------------------------------------
    let t0 = core::arch::x86_64::_rdtsc();
    let fence = ring.submit(fence_buf.as_ref().unwrap().phys);
    let waited = ring.wait_fence(&fence, 5000);
    let t1 = core::arch::x86_64::_rdtsc();
    let cycles = t1 - t0;

    if !waited {
        serial_println!(
            "[GPU-BENCH] voronoi result=FAIL reason=fence_timeout cycles={}",
            cycles
        );
        cleanup(
            kernel_buf,
            fence_buf,
            points_buf,
            centroids_buf,
            cell_ids_buf,
        );
        return false;
    }

    // -----------------------------------------------------------------
    // 6. Download and verify.
    // -----------------------------------------------------------------
    let mut cell_ids_host = [0u8; N_POINTS * 4];
    download(&mut cell_ids_host, cell_ids_buf.as_ref().unwrap());

    let mut verify_ok = true;
    for i in 0..N_POINTS {
        let off = i * 4;
        let cid = u32::from_le_bytes([
            cell_ids_host[off],
            cell_ids_host[off + 1],
            cell_ids_host[off + 2],
            cell_ids_host[off + 3],
        ]);
        if cid >= N_CELLS as u32 {
            verify_ok = false;
            break;
        }
    }

    cleanup(
        kernel_buf,
        fence_buf,
        points_buf,
        centroids_buf,
        cell_ids_buf,
    );

    if verify_ok {
        serial_println!("[GPU-BENCH] voronoi result=OK cycles={}", cycles);
        true
    } else {
        serial_println!(
            "[GPU-BENCH] voronoi result=FAIL reason=verify_error cycles={}",
            cycles
        );
        false
    }
}

/// Free all allocated GPU buffers.  Called on both success and failure paths.
fn cleanup(
    kernel_buf: Option<GpuBuffer>,
    fence_buf: Option<GpuBuffer>,
    points_buf: Option<GpuBuffer>,
    centroids_buf: Option<GpuBuffer>,
    cell_ids_buf: Option<GpuBuffer>,
) {
    if let Some(b) = kernel_buf {
        b.free();
    }
    if let Some(b) = fence_buf {
        b.free();
    }
    if let Some(b) = points_buf {
        b.free();
    }
    if let Some(b) = centroids_buf {
        b.free();
    }
    if let Some(b) = cell_ids_buf {
        b.free();
    }
}
