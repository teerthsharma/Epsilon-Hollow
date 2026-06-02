// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! In-kernel GPU benchmark application.
//!
//! Run from SealShell: `gpu_benchmark`

use crate::drivers::gpu::topology_accel;
use crate::serial_println;

const WARMUP: usize = 4;
const ITERS: usize = 32;
const GPU_BENCH_TIMEOUT_MS: u32 = 5000;

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

pub fn run_structured_topology_proof() {
    let accel_name = topology_accel::with_accelerator(|a| a.name());
    let cpu_fallback = accel_name == "CPU Fallback (software topology)";
    serial_println!(
        "[GPU-BENCH] suite version=1 mode={} backend={} accelerator={} hardware_dispatch={} shader_used={} shader_status={} warmup={} iterations={} kernels=3 status=begin",
        if cpu_fallback { "cpu_fallback" } else { "hardware_unproven" },
        if cpu_fallback { "software" } else { "unknown" },
        if cpu_fallback { "cpu_fallback" } else { "hardware_unproven" },
        if cpu_fallback { 0 } else { 1 },
        if cpu_fallback { 0 } else { 1 },
        if cpu_fallback {
            "placeholder_ok_not_claimed"
        } else {
            "hardware_proof_missing"
        },
        WARMUP,
        ITERS
    );

    if !cpu_fallback {
        // Attempt real AMD hardware dispatch proof (Stream-13 artifact).
        let hw_ok = unsafe { crate::drivers::gpu::dispatch::run_amd_hardware_proof() };
        let passed = if hw_ok { 1 } else { 0 };
        let failed = 3usize.saturating_sub(passed);
        serial_println!(
            "[GPU-BENCH] suite version=1 mode={} backend=pm4 hardware_dispatch=1 shader_used=1 kernels=3 passed={} failed={} result={} claim=amd_hardware_dispatch_proof",
            if hw_ok { "hardware_proven" } else { "hardware_unproven" },
            passed,
            failed,
            if hw_ok { "partial" } else { "fail" }
        );
        return;
    }

    let passed = (structured_voronoi_assign() as usize)
        + (structured_jl_project() as usize)
        + (structured_spectral_step() as usize);
    let failed = 3usize.saturating_sub(passed);
    serial_println!(
        "[GPU-BENCH] suite version=1 mode=cpu_fallback backend=software hardware_dispatch=0 shader_used=0 kernels=3 passed={} failed={} result={} claim=cpu_fallback_correctness_only",
        passed,
        failed,
        if failed == 0 { "pass" } else { "fail" }
    );
}

fn structured_voronoi_assign() -> bool {
    const N_POINTS: usize = 64;
    const N_CELLS: usize = 8;
    const TOTAL_RUNS: usize = WARMUP + ITERS;

    let points_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_POINTS * 16))
    {
        Ok(buf) => buf,
        Err(_) => {
            serial_println!("[GPU-BENCH] kernel=voronoi_assign mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 n_points=64 n_cells=8 warmup=4 iterations=32 dispatch_ok=0 wait_ok=0 upload_ok=0 download_ok=0 verify=fail reference=cpu_recompute checked=0 mismatches=64 invalid_ids=64 checksum=0 first_cell=0 avg_cycles=0");
            return false;
        }
    };
    let centroids_buf = match topology_accel::with_accelerator_mut(|a| {
        a.alloc_gpu_mem(N_CELLS * 16)
    }) {
        Ok(buf) => buf,
        Err(_) => {
            topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(points_buf));
            serial_println!("[GPU-BENCH] kernel=voronoi_assign mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 n_points=64 n_cells=8 warmup=4 iterations=32 dispatch_ok=0 wait_ok=0 upload_ok=0 download_ok=0 verify=fail reference=cpu_recompute checked=0 mismatches=64 invalid_ids=64 checksum=0 first_cell=0 avg_cycles=0");
            return false;
        }
    };
    let cell_ids_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_POINTS * 4))
    {
        Ok(buf) => buf,
        Err(_) => {
            topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(points_buf));
            topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(centroids_buf));
            serial_println!("[GPU-BENCH] kernel=voronoi_assign mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 n_points=64 n_cells=8 warmup=4 iterations=32 dispatch_ok=0 wait_ok=0 upload_ok=0 download_ok=0 verify=fail reference=cpu_recompute checked=0 mismatches=64 invalid_ids=64 checksum=0 first_cell=0 avg_cycles=0");
            return false;
        }
    };

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

    let mut upload_ok = 0usize;
    if topology_accel::with_accelerator_mut(|a| a.upload(&points_buf, &points_host)).is_ok() {
        upload_ok += 1;
    }
    if topology_accel::with_accelerator_mut(|a| a.upload(&centroids_buf, &centroids_host)).is_ok() {
        upload_ok += 1;
    }

    let mut dispatch_ok = 0usize;
    let mut wait_ok = 0usize;
    for _ in 0..WARMUP {
        if let Ok(fence) = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_voronoi(
                &points_buf,
                &centroids_buf,
                &cell_ids_buf,
                N_POINTS,
                N_CELLS,
            )
        }) {
            dispatch_ok += 1;
            if topology_accel::with_accelerator_mut(|a| a.wait_fence(&fence, GPU_BENCH_TIMEOUT_MS))
                .is_ok()
            {
                wait_ok += 1;
            }
        }
    }

    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    for _ in 0..ITERS {
        if let Ok(fence) = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_voronoi(
                &points_buf,
                &centroids_buf,
                &cell_ids_buf,
                N_POINTS,
                N_CELLS,
            )
        }) {
            dispatch_ok += 1;
            if topology_accel::with_accelerator_mut(|a| a.wait_fence(&fence, GPU_BENCH_TIMEOUT_MS))
                .is_ok()
            {
                wait_ok += 1;
            }
        }
    }
    let t1 = unsafe { core::arch::x86_64::_rdtsc() };
    let avg_cycles = (t1 - t0).max(ITERS as u64) / ITERS as u64;

    let mut cell_ids_host = [0u8; N_POINTS * 4];
    let download_ok = usize::from(
        topology_accel::with_accelerator_mut(|a| a.download(&mut cell_ids_host, &cell_ids_buf))
            .is_ok(),
    );

    let mut mismatches = 0usize;
    let mut invalid_ids = 0usize;
    let mut checksum = 0u64;
    let first_cell = u32::from_le_bytes([
        cell_ids_host[0],
        cell_ids_host[1],
        cell_ids_host[2],
        cell_ids_host[3],
    ]);
    for i in 0..N_POINTS {
        let off = i * 4;
        let got = u32::from_le_bytes([
            cell_ids_host[off],
            cell_ids_host[off + 1],
            cell_ids_host[off + 2],
            cell_ids_host[off + 3],
        ]);
        if got >= N_CELLS as u32 {
            invalid_ids += 1;
        }
        if got != voronoi_reference_cell(&points_host, &centroids_host, i, N_CELLS) {
            mismatches += 1;
        }
        checksum = checksum_u32(checksum, got);
    }

    let ok = upload_ok == 2
        && dispatch_ok == TOTAL_RUNS
        && wait_ok == TOTAL_RUNS
        && download_ok == 1
        && mismatches == 0
        && invalid_ids == 0
        && checksum != 0;
    serial_println!(
        "[GPU-BENCH] kernel=voronoi_assign mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 n_points={} n_cells={} warmup={} iterations={} dispatch_ok={} wait_ok={} upload_ok={} download_ok={} verify={} reference=cpu_recompute checked={} mismatches={} invalid_ids={} checksum={} first_cell={} avg_cycles={}",
        N_POINTS,
        N_CELLS,
        WARMUP,
        ITERS,
        dispatch_ok,
        wait_ok,
        upload_ok,
        download_ok,
        if ok { "pass" } else { "fail" },
        N_POINTS,
        mismatches,
        invalid_ids,
        checksum,
        first_cell,
        avg_cycles
    );

    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(points_buf));
    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(centroids_buf));
    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(cell_ids_buf));
    ok
}

fn structured_jl_project() -> bool {
    const N_VECTORS: usize = 4;
    const DIM_IN: usize = 128;
    const DIM_OUT: usize = 3;
    const SEED: u32 = 0xE951_10A7;
    const TOTAL_RUNS: usize = WARMUP + ITERS;

    let input_buf = match topology_accel::with_accelerator_mut(|a| {
        a.alloc_gpu_mem(N_VECTORS * DIM_IN * 8)
    }) {
        Ok(buf) => buf,
        Err(_) => {
            serial_println!("[GPU-BENCH] kernel=jl_project mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 n_vectors=4 dim_in=128 dim_out=3 seed=0xE95110A7 warmup=4 iterations=32 dispatch_ok=0 wait_ok=0 upload_ok=0 download_ok=0 verify=fail reference=cpu_recompute checked=0 mismatches=12 finite=0 checksum=0 max_abs_diff_scaled=0 avg_cycles=0");
            return false;
        }
    };
    let output_buf = match topology_accel::with_accelerator_mut(|a| {
        a.alloc_gpu_mem(N_VECTORS * DIM_OUT * 8)
    }) {
        Ok(buf) => buf,
        Err(_) => {
            topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(input_buf));
            serial_println!("[GPU-BENCH] kernel=jl_project mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 n_vectors=4 dim_in=128 dim_out=3 seed=0xE95110A7 warmup=4 iterations=32 dispatch_ok=0 wait_ok=0 upload_ok=0 download_ok=0 verify=fail reference=cpu_recompute checked=0 mismatches=12 finite=0 checksum=0 max_abs_diff_scaled=0 avg_cycles=0");
            return false;
        }
    };

    let mut input_host = [0u8; N_VECTORS * DIM_IN * 8];
    for i in 0..N_VECTORS * DIM_IN {
        let val = ((i * 997) % 1000) as f64 / 1000.0;
        input_host[i * 8..i * 8 + 8].copy_from_slice(&val.to_le_bytes());
    }
    let upload_ok = usize::from(
        topology_accel::with_accelerator_mut(|a| a.upload(&input_buf, &input_host)).is_ok(),
    );

    let mut dispatch_ok = 0usize;
    let mut wait_ok = 0usize;
    for _ in 0..WARMUP {
        if let Ok(fence) = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_jl_project(&input_buf, &output_buf, N_VECTORS, SEED)
        }) {
            dispatch_ok += 1;
            if topology_accel::with_accelerator_mut(|a| a.wait_fence(&fence, GPU_BENCH_TIMEOUT_MS))
                .is_ok()
            {
                wait_ok += 1;
            }
        }
    }

    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    for _ in 0..ITERS {
        if let Ok(fence) = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_jl_project(&input_buf, &output_buf, N_VECTORS, SEED)
        }) {
            dispatch_ok += 1;
            if topology_accel::with_accelerator_mut(|a| a.wait_fence(&fence, GPU_BENCH_TIMEOUT_MS))
                .is_ok()
            {
                wait_ok += 1;
            }
        }
    }
    let t1 = unsafe { core::arch::x86_64::_rdtsc() };
    let avg_cycles = (t1 - t0).max(ITERS as u64) / ITERS as u64;

    let mut out_host = [0u8; N_VECTORS * DIM_OUT * 8];
    let download_ok = usize::from(
        topology_accel::with_accelerator_mut(|a| a.download(&mut out_host, &output_buf)).is_ok(),
    );

    let mut mismatches = 0usize;
    let mut finite = 0usize;
    let mut checksum = 0u64;
    let mut max_abs_diff_scaled = 0u64;
    for v in 0..N_VECTORS {
        for d in 0..DIM_OUT {
            let idx = v * DIM_OUT + d;
            let off = idx * 8;
            let got = f64::from_le_bytes([
                out_host[off],
                out_host[off + 1],
                out_host[off + 2],
                out_host[off + 3],
                out_host[off + 4],
                out_host[off + 5],
                out_host[off + 6],
                out_host[off + 7],
            ]);
            if got.is_finite() {
                finite += 1;
            }
            let expected = jl_reference_value(&input_host, v, d, SEED);
            let diff = libm::fabs(got - expected);
            let scaled = (diff * 1_000_000_000_000.0) as u64;
            max_abs_diff_scaled = max_abs_diff_scaled.max(scaled);
            if scaled != 0 {
                mismatches += 1;
            }
            checksum = checksum_u64(checksum, got.to_bits());
        }
    }

    let checked = N_VECTORS * DIM_OUT;
    let ok = upload_ok == 1
        && dispatch_ok == TOTAL_RUNS
        && wait_ok == TOTAL_RUNS
        && download_ok == 1
        && mismatches == 0
        && finite == checked
        && checksum != 0;
    serial_println!(
        "[GPU-BENCH] kernel=jl_project mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 n_vectors={} dim_in={} dim_out={} seed=0xE95110A7 warmup={} iterations={} dispatch_ok={} wait_ok={} upload_ok={} download_ok={} verify={} reference=cpu_recompute checked={} mismatches={} finite={} checksum={} max_abs_diff_scaled={} avg_cycles={}",
        N_VECTORS,
        DIM_IN,
        DIM_OUT,
        WARMUP,
        ITERS,
        dispatch_ok,
        wait_ok,
        upload_ok,
        download_ok,
        if ok { "pass" } else { "fail" },
        checked,
        mismatches,
        finite,
        checksum,
        max_abs_diff_scaled,
        avg_cycles
    );

    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(input_buf));
    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(output_buf));
    ok
}

fn structured_spectral_step() -> bool {
    const DIM: usize = 512;
    const ALPHA: f64 = 0.3;
    const TOTAL_RUNS: usize = WARMUP + ITERS;

    let state_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(DIM * 8)) {
        Ok(buf) => buf,
        Err(_) => {
            serial_println!("[GPU-BENCH] kernel=spectral_step mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 dim=512 alpha_ppm=300000 warmup=4 iterations=32 dispatch_ok=0 wait_ok=0 upload_ok=0 download_ok=0 verify=fail reference=cpu_recompute checked=0 mismatches=512 finite=0 checksum=0 max_abs_diff_scaled=0 avg_cycles=0");
            return false;
        }
    };
    let output_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(DIM * 8)) {
        Ok(buf) => buf,
        Err(_) => {
            topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(state_buf));
            serial_println!("[GPU-BENCH] kernel=spectral_step mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 dim=512 alpha_ppm=300000 warmup=4 iterations=32 dispatch_ok=0 wait_ok=0 upload_ok=0 download_ok=0 verify=fail reference=cpu_recompute checked=0 mismatches=512 finite=0 checksum=0 max_abs_diff_scaled=0 avg_cycles=0");
            return false;
        }
    };

    let mut state_host = [0u8; DIM * 8];
    for i in 0..DIM {
        let val = libm::sin(i as f64);
        state_host[i * 8..i * 8 + 8].copy_from_slice(&val.to_le_bytes());
    }
    let upload_ok = usize::from(
        topology_accel::with_accelerator_mut(|a| a.upload(&state_buf, &state_host)).is_ok(),
    );

    let mut dispatch_ok = 0usize;
    let mut wait_ok = 0usize;
    for _ in 0..WARMUP {
        if let Ok(fence) = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_spectral_step(&state_buf, &output_buf, DIM, ALPHA)
        }) {
            dispatch_ok += 1;
            if topology_accel::with_accelerator_mut(|a| a.wait_fence(&fence, GPU_BENCH_TIMEOUT_MS))
                .is_ok()
            {
                wait_ok += 1;
            }
        }
    }

    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    for _ in 0..ITERS {
        if let Ok(fence) = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_spectral_step(&state_buf, &output_buf, DIM, ALPHA)
        }) {
            dispatch_ok += 1;
            if topology_accel::with_accelerator_mut(|a| a.wait_fence(&fence, GPU_BENCH_TIMEOUT_MS))
                .is_ok()
            {
                wait_ok += 1;
            }
        }
    }
    let t1 = unsafe { core::arch::x86_64::_rdtsc() };
    let avg_cycles = (t1 - t0).max(ITERS as u64) / ITERS as u64;

    let mut out_host = [0u8; DIM * 8];
    let download_ok = usize::from(
        topology_accel::with_accelerator_mut(|a| a.download(&mut out_host, &output_buf)).is_ok(),
    );

    let mut mismatches = 0usize;
    let mut finite = 0usize;
    let mut checksum = 0u64;
    let mut max_abs_diff_scaled = 0u64;
    let one_minus_alpha = 1.0 - ALPHA;
    for i in 0..DIM {
        let off = i * 8;
        let got = f64::from_le_bytes([
            out_host[off],
            out_host[off + 1],
            out_host[off + 2],
            out_host[off + 3],
            out_host[off + 4],
            out_host[off + 5],
            out_host[off + 6],
            out_host[off + 7],
        ]);
        if got.is_finite() {
            finite += 1;
        }
        let state = f64::from_le_bytes([
            state_host[off],
            state_host[off + 1],
            state_host[off + 2],
            state_host[off + 3],
            state_host[off + 4],
            state_host[off + 5],
            state_host[off + 6],
            state_host[off + 7],
        ]);
        let expected = state * one_minus_alpha;
        let diff = libm::fabs(got - expected);
        let scaled = (diff * 1_000_000_000_000.0) as u64;
        max_abs_diff_scaled = max_abs_diff_scaled.max(scaled);
        if scaled != 0 {
            mismatches += 1;
        }
        checksum = checksum_u64(checksum, got.to_bits());
    }

    let ok = upload_ok == 1
        && dispatch_ok == TOTAL_RUNS
        && wait_ok == TOTAL_RUNS
        && download_ok == 1
        && mismatches == 0
        && finite == DIM
        && checksum != 0;
    serial_println!(
        "[GPU-BENCH] kernel=spectral_step mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 dim={} alpha_ppm=300000 warmup={} iterations={} dispatch_ok={} wait_ok={} upload_ok={} download_ok={} verify={} reference=cpu_recompute checked={} mismatches={} finite={} checksum={} max_abs_diff_scaled={} avg_cycles={}",
        DIM,
        WARMUP,
        ITERS,
        dispatch_ok,
        wait_ok,
        upload_ok,
        download_ok,
        if ok { "pass" } else { "fail" },
        DIM,
        mismatches,
        finite,
        checksum,
        max_abs_diff_scaled,
        avg_cycles
    );

    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(state_buf));
    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(output_buf));
    ok
}

fn voronoi_reference_cell(
    points: &[u8],
    centroids: &[u8],
    point_idx: usize,
    n_cells: usize,
) -> u32 {
    let p_off = point_idx * 16;
    let p_theta = f64::from_le_bytes([
        points[p_off],
        points[p_off + 1],
        points[p_off + 2],
        points[p_off + 3],
        points[p_off + 4],
        points[p_off + 5],
        points[p_off + 6],
        points[p_off + 7],
    ]);
    let p_phi = f64::from_le_bytes([
        points[p_off + 8],
        points[p_off + 9],
        points[p_off + 10],
        points[p_off + 11],
        points[p_off + 12],
        points[p_off + 13],
        points[p_off + 14],
        points[p_off + 15],
    ]);
    let mut best_dist = core::f64::MAX;
    let mut best_cell = 0u32;
    for c in 0..n_cells {
        let c_off = c * 16;
        let c_theta = f64::from_le_bytes([
            centroids[c_off],
            centroids[c_off + 1],
            centroids[c_off + 2],
            centroids[c_off + 3],
            centroids[c_off + 4],
            centroids[c_off + 5],
            centroids[c_off + 6],
            centroids[c_off + 7],
        ]);
        let c_phi = f64::from_le_bytes([
            centroids[c_off + 8],
            centroids[c_off + 9],
            centroids[c_off + 10],
            centroids[c_off + 11],
            centroids[c_off + 12],
            centroids[c_off + 13],
            centroids[c_off + 14],
            centroids[c_off + 15],
        ]);
        let cos_dist = libm::sin(p_theta) * libm::sin(c_theta)
            + libm::cos(p_theta) * libm::cos(c_theta) * libm::cos(p_phi - c_phi);
        let dist = libm::acos(cos_dist.clamp(-1.0, 1.0));
        if dist < best_dist {
            best_dist = dist;
            best_cell = c as u32;
        }
    }
    best_cell
}

fn jl_reference_value(input: &[u8], vector: usize, out_dim: usize, seed: u32) -> f64 {
    const DIM_IN: usize = 128;
    let mut sum = 0.0f64;
    for i in 0..DIM_IN {
        let idx = vector * DIM_IN + i;
        let off = idx * 8;
        let value = f64::from_le_bytes([
            input[off],
            input[off + 1],
            input[off + 2],
            input[off + 3],
            input[off + 4],
            input[off + 5],
            input[off + 6],
            input[off + 7],
        ]);
        sum += value * jl_hash_f64(seed, i, out_dim);
    }
    sum / libm::sqrt(DIM_IN as f64)
}

fn jl_hash_f64(seed: u32, a: usize, b: usize) -> f64 {
    let mut x = seed
        .wrapping_add((a as u32).wrapping_mul(0x9E37_79B9))
        .wrapping_add((b as u32).wrapping_mul(0x85EB_CA6B));
    x = x.wrapping_mul(0xC2B2_AE35);
    ((x as i32) as f64) / (i32::MAX as f64)
}

fn checksum_u32(acc: u64, value: u32) -> u64 {
    checksum_u64(acc, value as u64)
}

fn checksum_u64(acc: u64, value: u64) -> u64 {
    acc.wrapping_mul(0x100_0000_01B3).wrapping_add(value)
}

fn bench_voronoi_assign(is_gpu: bool) {
    const N_POINTS: usize = 256;
    const N_CELLS: usize = 8;

    serial_println!(
        "\n[GPU-BENCH] Voronoi assignment ({} points, {} cells) — {}",
        N_POINTS,
        N_CELLS,
        if is_gpu { "GPU" } else { "CPU" }
    );

    // Allocate GPU buffers.
    let points_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_POINTS * 16))
    {
        Ok(b) => b,
        Err(_) => {
            serial_println!("[GPU-BENCH] Out of memory");
            return;
        }
    };
    let centroids_buf =
        match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_CELLS * 16)) {
            Ok(b) => b,
            Err(_) => {
                serial_println!("[GPU-BENCH] Out of memory");
                return;
            }
        };
    let cell_ids_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_POINTS * 4))
    {
        Ok(b) => b,
        Err(_) => {
            serial_println!("[GPU-BENCH] Out of memory");
            return;
        }
    };

    // Upload dummy data.
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

    let _ = topology_accel::with_accelerator_mut(|a| a.upload(&points_buf, &points_host));
    let _ = topology_accel::with_accelerator_mut(|a| a.upload(&centroids_buf, &centroids_host));

    // Warmup
    for _ in 0..WARMUP {
        let fence = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_voronoi(
                &points_buf,
                &centroids_buf,
                &cell_ids_buf,
                N_POINTS,
                N_CELLS,
            )
        });
        if let Ok(f) = fence {
            let _ = topology_accel::with_accelerator_mut(|a| a.wait_fence(&f, 5000));
        }
    }

    // Timed run
    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    for _ in 0..ITERS {
        let fence = topology_accel::with_accelerator_mut(|a| {
            a.dispatch_voronoi(
                &points_buf,
                &centroids_buf,
                &cell_ids_buf,
                N_POINTS,
                N_CELLS,
            )
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
        cell_ids_host[0],
        cell_ids_host[1],
        cell_ids_host[2],
        cell_ids_host[3],
    ]);

    serial_println!(
        "[GPU-BENCH]  avg {} cycles/iter (first cell = {})",
        avg_cycles,
        first_cell
    );

    // Verification: all cell IDs must be in [0, N_CELLS).
    let mut ok = true;
    for i in 0..N_POINTS {
        let cid = u32::from_le_bytes([
            cell_ids_host[i * 4],
            cell_ids_host[i * 4 + 1],
            cell_ids_host[i * 4 + 2],
            cell_ids_host[i * 4 + 3],
        ]);
        if cid >= N_CELLS as u32 {
            serial_println!(
                "[GPU-BENCH]  VERIFY FAIL: cell_ids[{}] = {} (max {})",
                i,
                cid,
                N_CELLS - 1
            );
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

    serial_println!(
        "\n[GPU-BENCH] JL projection ({} vectors, {} → 3) — {}",
        N_VECTORS,
        DIM_IN,
        if is_gpu { "GPU" } else { "CPU" }
    );

    let input_buf =
        match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_VECTORS * DIM_IN * 8)) {
            Ok(b) => b,
            Err(_) => {
                serial_println!("[GPU-BENCH] Out of memory");
                return;
            }
        };
    let output_buf =
        match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(N_VECTORS * 3 * 8)) {
            Ok(b) => b,
            Err(_) => {
                serial_println!("[GPU-BENCH] Out of memory");
                return;
            }
        };

    let mut input_host = [0u8; N_VECTORS * DIM_IN * 8];
    for i in 0..N_VECTORS * DIM_IN {
        let val = ((i * 997) % 1000) as f64 / 1000.0;
        input_host[i * 8..i * 8 + 8].copy_from_slice(&val.to_le_bytes());
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
    let v0 = f64::from_le_bytes([
        out_host[0],
        out_host[1],
        out_host[2],
        out_host[3],
        out_host[4],
        out_host[5],
        out_host[6],
        out_host[7],
    ]);
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

    serial_println!(
        "\n[GPU-BENCH] Spectral contraction step (dim={}) — {}",
        DIM,
        if is_gpu { "GPU" } else { "CPU" }
    );

    let state_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(DIM * 8)) {
        Ok(b) => b,
        Err(_) => {
            serial_println!("[GPU-BENCH] Out of memory");
            return;
        }
    };
    let output_buf = match topology_accel::with_accelerator_mut(|a| a.alloc_gpu_mem(DIM * 8)) {
        Ok(b) => b,
        Err(_) => {
            serial_println!("[GPU-BENCH] Out of memory");
            return;
        }
    };

    let mut state_host = [0u8; DIM * 8];
    for i in 0..DIM {
        let val = libm::sin(i as f64);
        state_host[i * 8..i * 8 + 8].copy_from_slice(&val.to_le_bytes());
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
    let got = f64::from_le_bytes([
        out_host[0],
        out_host[1],
        out_host[2],
        out_host[3],
        out_host[4],
        out_host[5],
        out_host[6],
        out_host[7],
    ]);
    let expected = libm::sin(0.0) * 0.7;
    let diff = libm::fabs(got - expected);
    if diff < 1e-12 {
        serial_println!("[GPU-BENCH]  verification: PASS (diff={:.2e})", diff);
    } else {
        serial_println!(
            "[GPU-BENCH]  verification: FAIL (got={}, expected={}, diff={:.2e})",
            got,
            expected,
            diff
        );
    }

    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(state_buf));
    topology_accel::with_accelerator_mut(|a| a.free_gpu_mem(output_buf));
}
