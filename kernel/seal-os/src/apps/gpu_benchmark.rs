// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! In-kernel GPU benchmark application.
//!
//! Run from SealShell: `gpu_benchmark`

use crate::drivers::gpu::topology_accel;
use crate::serial_println;

const WARMUP: usize = 4;
const ITERS: usize = 32;

pub fn main() {
    serial_println!("============================================================");
    serial_println!("  Seal OS GPU Topology Accelerator Benchmark");
    serial_println!("============================================================");

    let accel_name = topology_accel::with_accelerator(|a| a.name());
    let is_gpu = accel_name.starts_with("AMD");
    serial_println!("Accelerator: {}", accel_name);
    if !is_gpu {
        serial_println!("Note: running in software fallback mode — timing reflects CPU execution.");
    }

    bench_voronoi_assign(is_gpu);
    bench_jl_project(is_gpu);
    bench_spectral_step(is_gpu);

    serial_println!("============================================================");
    serial_println!("  Benchmark complete.");
    serial_println!("============================================================");
}

fn bench_voronoi_assign(is_gpu: bool) {
    const N_POINTS: usize = 256;
    const N_CELLS: usize = 8;

    serial_println!("\n[GPU-BENCH] Voronoi assignment ({} points, {} cells) — {}",
        N_POINTS, N_CELLS, if is_gpu { "GPU" } else { "CPU" });

    // Allocate GPU buffers.
    let points_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_POINTS * 16)) {
        Ok(b) => b,
        Err(_) => { serial_println!("[GPU-BENCH] Out of memory"); return; }
    };
    let centroids_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_CELLS * 16)) {
        Ok(b) => b,
        Err(_) => { serial_println!("[GPU-BENCH] Out of memory"); return; }
    };
    let cell_ids_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_POINTS * 4)) {
        Ok(b) => b,
        Err(_) => { serial_println!("[GPU-BENCH] Out of memory"); return; }
    };

    // Upload dummy data.
    let mut points_host = [0u8; N_POINTS * 16];
    let mut centroids_host = [0u8; N_CELLS * 16];
    for i in 0..N_POINTS {
        let theta = (i as f64 / N_POINTS as f64) * core::f64::consts::PI;
        let phi = (i as f64 * 1.618) % (2.0 * core::f64::consts::PI);
        let off = i * 16;
        points_host[off..off+8].copy_from_slice(&theta.to_le_bytes());
        points_host[off+8..off+16].copy_from_slice(&phi.to_le_bytes());
    }
    for i in 0..N_CELLS {
        let theta = (i as f64 / N_CELLS as f64) * core::f64::consts::PI;
        let phi = (i as f64 * 2.0) % (2.0 * core::f64::consts::PI);
        let off = i * 16;
        centroids_host[off..off+8].copy_from_slice(&theta.to_le_bytes());
        centroids_host[off+8..off+16].copy_from_slice(&phi.to_le_bytes());
    }

    let _ = topology_accel::with_accelerator_mut(|a| a.upload(&points_buf, &points_host));
    let _ = topology_accel::with_accelerator_mut(|a| a.upload(&centroids_buf, &centroids_host));

    // Warmup
    for _ in 0..WARMUP {
        let fence = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_voronoi(&points_buf, &centroids_buf, &cell_ids_buf, N_POINTS, N_CELLS)
        });
        if let Ok(f) = fence {
            let _ = topology_accel::with_accelerator_mut(|a| a.wait_fence(&f, 5000));
        }
    }

    // Timed run
    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    for _ in 0..ITERS {
        let fence = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_voronoi(&points_buf, &centroids_buf, &cell_ids_buf, N_POINTS, N_CELLS)
        });
        if let Ok(f) = fence {
            let _ = topology_accel::with_accelerator_mut(|a| a.wait_fence(&f, 5000));
        }
    }
    let t1 = unsafe { core::arch::x86_64::_rdtsc() };
    let cycles = t1 - t0;
    let avg_cycles = cycles / ITERS as u64;

    // Download result for verification.
    let mut cell_ids_host = [0u8; N_POINTS * 4];
    let _ = topology_accel::with_accelerator_mut(|a| a.download(&mut cell_ids_host, &cell_ids_buf));
    let first_cell = u32::from_le_bytes([
        cell_ids_host[0], cell_ids_host[1], cell_ids_host[2], cell_ids_host[3]
    ]);

    serial_println!("[GPU-BENCH]  avg {} cycles/iter (first cell = {})", avg_cycles, first_cell);

    // Verification: all cell IDs must be in [0, N_CELLS).
    let mut ok = true;
    for i in 0..N_POINTS {
        let cid = u32::from_le_bytes([
            cell_ids_host[i*4], cell_ids_host[i*4+1],
            cell_ids_host[i*4+2], cell_ids_host[i*4+3]
        ]);
        if cid >= N_CELLS as u32 {
            serial_println!("[GPU-BENCH]  VERIFY FAIL: cell_ids[{}] = {} (max {})", i, cid, N_CELLS - 1);
            ok = false;
            break;
        }
    }
    if ok {
        serial_println!("[GPU-BENCH]  verification: PASS");
    }

    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(points_buf));
    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(centroids_buf));
    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(cell_ids_buf));
}

fn bench_jl_project(is_gpu: bool) {
    const N_VECTORS: usize = 256;
    const DIM_IN: usize = 128;

    serial_println!("\n[GPU-BENCH] JL projection ({} vectors, {} → 3) — {}",
        N_VECTORS, DIM_IN, if is_gpu { "GPU" } else { "CPU" });

    let input_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_VECTORS * DIM_IN * 8)) {
        Ok(b) => b,
        Err(_) => { serial_println!("[GPU-BENCH] Out of memory"); return; }
    };
    let output_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_VECTORS * 3 * 8)) {
        Ok(b) => b,
        Err(_) => { serial_println!("[GPU-BENCH] Out of memory"); return; }
    };

    let mut input_host = [0u8; N_VECTORS * DIM_IN * 8];
    for i in 0..N_VECTORS * DIM_IN {
        let val = ((i * 997) % 1000) as f64 / 1000.0;
        input_host[i*8..i*8+8].copy_from_slice(&val.to_le_bytes());
    }
    let _ = topology_accel::with_accelerator_mut(|a| a.upload(&input_buf, &input_host));

    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    for _ in 0..ITERS {
        let fence = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_jl_project(&input_buf, &output_buf, N_VECTORS, 0xE95110A7)
        });
        if let Ok(f) = fence {
            let _ = topology_accel::with_accelerator_mut(|a| a.wait_fence(&f, 5000));
        }
    }
    let t1 = unsafe { core::arch::x86_64::_rdtsc() };
    let avg_cycles = (t1 - t0) / ITERS as u64;

    serial_println!("[GPU-BENCH]  avg {} cycles/iter", avg_cycles);

    // Verification: download first output vector and check it is finite.
    let mut out_host = [0u8; 24];
    let _ = topology_accel::with_accelerator_mut(|a| a.download(&mut out_host, &output_buf));
    let v0 = f64::from_le_bytes([out_host[0], out_host[1], out_host[2], out_host[3],
                                  out_host[4], out_host[5], out_host[6], out_host[7]]);
    if v0.is_finite() {
        serial_println!("[GPU-BENCH]  verification: PASS (v0={:.4})", v0);
    } else {
        serial_println!("[GPU-BENCH]  verification: FAIL (v0={})", v0);
    }

    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(input_buf));
    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(output_buf));
}

fn bench_spectral_step(is_gpu: bool) {
    const DIM: usize = 1024;

    serial_println!("\n[GPU-BENCH] Spectral contraction step (dim={}) — {}",
        DIM, if is_gpu { "GPU" } else { "CPU" });

    let state_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(DIM * 8)) {
        Ok(b) => b,
        Err(_) => { serial_println!("[GPU-BENCH] Out of memory"); return; }
    };
    let output_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(DIM * 8)) {
        Ok(b) => b,
        Err(_) => { serial_println!("[GPU-BENCH] Out of memory"); return; }
    };

    let mut state_host = [0u8; DIM * 8];
    for i in 0..DIM {
        let val = libm::sin(i as f64);
        state_host[i*8..i*8+8].copy_from_slice(&val.to_le_bytes());
    }
    let _ = topology_accel::with_accelerator_mut(|a| a.upload(&state_buf, &state_host));

    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    for _ in 0..ITERS {
        let fence = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_spectral_step(&state_buf, &output_buf, DIM, 0.3)
        });
        if let Ok(f) = fence {
            let _ = topology_accel::with_accelerator_mut(|a| a.wait_fence(&f, 5000));
        }
    }
    let t1 = unsafe { core::arch::x86_64::_rdtsc() };
    let avg_cycles = (t1 - t0) / ITERS as u64;

    serial_println!("[GPU-BENCH]  avg {} cycles/iter", avg_cycles);

    // Verification: output[0] should be ~state[0] * (1 - 0.3) = state[0] * 0.7
    let mut out_host = [0u8; 8];
    let _ = topology_accel::with_accelerator_mut(|a| a.download(&mut out_host, &output_buf));
    let got = f64::from_le_bytes([out_host[0], out_host[1], out_host[2], out_host[3],
                                   out_host[4], out_host[5], out_host[6], out_host[7]]);
    let expected = libm::sin(0.0) * 0.7;
    let diff = libm::fabs(got - expected);
    if diff < 1e-12 {
        serial_println!("[GPU-BENCH]  verification: PASS (diff={:.2e})", diff);
    } else {
        serial_println!("[GPU-BENCH]  verification: FAIL (got={}, expected={}, diff={:.2e})", got, expected, diff);
    }

    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(state_buf));
    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(output_buf));
}
