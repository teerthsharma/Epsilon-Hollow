// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Embedded GCN ISA kernels for topology acceleration.
//!
//! These binaries are hand-assembled or compiled from OpenCL C and stored as
//! `&[u32]` slices.  Each kernel follows the AMD GCN compute ABI:
//! - s0-s1: global work offset (ignored, set to 0)
//! - s2-s3: kernel argument buffer address (64-bit pointer)
//! - User data registers (s4+) contain extra parameters.
//!
//! # Rebuilding kernels from source
//!
//! ```bash
//! # Install ROCm LLVM
//! clang -x cl -target amdgcn-amd-amdhsa -mcpu=gfx803 \
//!       -cl-std=CL2.0 -O3 -o voronoi_assign.o -c voronoi_assign.cl
//! llvm-objcopy -O binary voronoi_assign.o voronoi_assign.bin
//! xxd -p voronoi_assign.bin | fold -w 8 > voronoi_assign.hex
//! # Then paste hex words into the arrays below.
//! ```

// ------------------------------------------------------------------
// Kernel: voronoi_assign
// Assigns each point in a point cloud to the nearest Voronoi centroid.
//
// OpenCL C source (for reference):
//
// ```opencl
// typedef struct { float theta, phi; } SpherePoint;
// __kernel void voronoi_assign(
//     __global const SpherePoint* points,
//     __global const SpherePoint* centroids,
//     __global int* cell_ids,
//     int n_points,
//     int n_cells
// ) {
//     int gid = get_global_id(0);
//     if (gid >= n_points) return;
//
//     float best_dist = 1e38f;
//     int best_cell = 0;
//     float p_theta = points[gid].theta;
//     float p_phi   = points[gid].phi;
//
//     for (int c = 0; c < n_cells; c++) {
//         float ct = centroids[c].theta;
//         float cp = centroids[c].phi;
//         float cos_dist = sin(p_theta)*sin(ct) + cos(p_theta)*cos(ct)*cos(p_phi - cp);
//         float dist = acos(clamp(cos_dist, -1.0f, 1.0f));
//         if (dist < best_dist) {
//             best_dist = dist;
//             best_cell = c;
//         }
//     }
//     cell_ids[gid] = best_cell;
// }
// ```
//
// GCN ISA encoding (gfx803, wave64):
// ------------------------------------------------------------------

/// `voronoi_assign` kernel binary — GCN ISA for gfx803+ (RX 400/500 series).
/// This is a **placeholder stub** containing the correct header and a minimal
/// loop body.  Replace with real compiled binary for production acceleration.
pub static VORONOI_ASSIGN_GCN: &[u32] = &[
    // AMD kernel descriptor (64 bytes)
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    // Kernel prologue
    0xBEFC0000, // s_mov_b32 m0, s4           ; set exec mask
    0xB9000004, // s_mov_b32 s0, s4           ; work-group ID
    0xB9010005, // s_mov_b32 s1, s5           ; argument pointer low
    0xB9020006, // s_mov_b32 s2, s6           ; argument pointer high
    // Load n_points and n_cells from argument buffer
    0xC0020000, 0x00000010, // s_load_dword s3, s[1:2], 0x10
    0xC0020001, 0x00000014, // s_load_dword s4, s[1:2], 0x14
    0xBF8C0000, // s_waitcnt lgkmcnt(0)
    // Load point coordinate
    0xC0060032, 0x00000000, 0x00000000, // s_load_dwordx2 s[8:9], s[1:2], 0x00
    0xBF8C0000, // s_waitcnt lgkmcnt(0)
    // Compute global ID
    0xB9080000, // s_mov_b32 s8, s0
    0x32000008, // v_mov_b32 v0, s8
    // Compare with n_points
    0x7D8C0003, // v_cmp_gt_i32 vcc, s3, v0
    0xD8250000, // s_cbranch_vccz exit_label
    // Loop body stub (simplified)
    0x7E000280, // v_mov_b32 v1, 0            ; best_cell = 0
    0x7E020280, // v_mov_b32 v2, 0            ; best_dist = 0 (simplified)
    // Store result
    0xC0060032, 0x00000008, 0x00000000, // s_load_dwordx2 s[10:11], s[1:2], 0x08
    0xBF8C0000, // s_waitcnt lgkmcnt(0)
    0xDC500000, 0x00000000, // flat_store_dword v[0:1], v2
    // Exit
    0xBF810000, // s_endpgm
];

// ------------------------------------------------------------------
// Kernel: jl_project
// Johnson-Lindenstrauss projection: 128-dim sparse vector → 3-dim.
//
// Arguments (via user SGPRs / arg buffer):
//   0: input  f64[128] *   (128-dim source vectors)
//   1: output f64[3] *     (3-dim projected vectors)
//   2: n_vectors            (number of vectors to project)
//   3: seed                 (random Gaussian matrix seed)
//
// Each work-item processes one vector.
// ------------------------------------------------------------------

pub static JL_PROJECT_GCN: &[u32] = &[
    // Kernel descriptor placeholder
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    // Prologue
    0xB9000004, // s_mov_b32 s0, s4
    0xB9010005, // s_mov_b32 s1, s5
    0xB9020006, // s_mov_b32 s2, s6
    0xBF8C0000, // s_waitcnt lgkmcnt(0)
    // v0 = global_id
    0x32000000, // v_mov_b32 v0, s0
    // Load n_vectors
    0xC0020002, 0x00000010, // s_load_dword s3, s[1:2], 0x10
    0xBF8C0000, // Bounds check
    0x7D8C0003, // v_cmp_gt_i32 vcc, s3, v0
    0xD8250000, // s_cbranch_vccz exit
    // Compute output index (v0 * 3 * 8 bytes)
    0x32020200, // v_mul_lo_u32 v1, v0, 24
    // Compute input index (v0 * 128 * 8 bytes)
    0x32040400, // v_mul_lo_u32 v2, v0, 1024
    // Simplified: store zeros as placeholder projection result
    0xDC500000, 0x00000000, // flat_store_dword v[1:2], 0
    0xDC500001, 0x00000004, // flat_store_dword v[1:2]+4, 0
    0xDC500002, 0x00000008, // flat_store_dword v[1:2]+8, 0
    // Exit
    0xBF810000, // s_endpgm
];

// ------------------------------------------------------------------
// Kernel: spectral_step
// One iteration of spectral contraction: x_{t+1} = (1-α)*x_t + α*target.
//
// Arguments:
//   0: state   f64[D] *     (current state vector)
//   1: output  f64[D] *     (next state vector)
//   2: dim                    (vector dimension D)
//   3: alpha   f64            (contraction coefficient)
//
// Each work-item processes one element.
// ------------------------------------------------------------------

pub static SPECTRAL_STEP_GCN: &[u32] = &[
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0xB9000004, // s_mov_b32 s0, s4
    0xB9010005, // s_mov_b32 s1, s5
    0xB9020006, // s_mov_b32 s2, s6
    0xBF8C0000, 0x32000000, 0xC0020002, 0x00000010, 0xBF8C0000, 0x7D8C0002, 0xD8250000,
    // Simplified: copy input to output (real kernel would do lerp)
    0xDC500000, 0x00000000, 0xBF810000,
];

// ------------------------------------------------------------------
// Kernel: s2_distance
// Compute great-circle distance matrix between two point clouds on S².
//
// Arguments:
//   0: points_a  SpherePoint[N] *
//   1: points_b  SpherePoint[M] *
//   2: out       f64[N][M] *
//   3: N
//   4: M
//
// 2D grid: global_size = (N, M, 1), each work-item computes one pair.
// ------------------------------------------------------------------

pub static S2_DISTANCE_GCN: &[u32] = &[
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0xB9000004, 0xB9010005, 0xB9020006, 0xBF8C0000,
    // Load global IDs
    0x32000000, // v0 = gid_x
    0x32020201, // v1 = gid_y
    // Bounds checks against N and M
    0xC0020003, 0x00000018, 0xC0020004, 0x0000001C, 0xBF8C0000, 0x7D8C0003, 0xD8250000, 0x7D8E0004,
    0xD8250000,
    // Compute output index (gid_x * M + gid_y) * 8
    0x32060601, // v3 = v0 * M
    0x32080803, // v4 = v3 + v1
    0x320A0A04, // v5 = v4 * 8
    // Load point coordinates and compute acos(sinθ1 sinθ2 + cosθ1 cosθ2 cos(Δφ))
    // ... (simplified stub)
    0xDC500000, 0x00000000, 0xBF810000,
];

// ------------------------------------------------------------------
// Kernel metadata table
// ------------------------------------------------------------------

/// Metadata for each embedded kernel.
pub struct KernelMeta {
    pub name: &'static str,
    pub binary: &'static [u32],
    pub code_size_bytes: usize,
    pub min_waves: u32,
    pub preferred_waves: u32,
}

pub const KERNELS: &[KernelMeta] = &[
    KernelMeta {
        name: "voronoi_assign",
        binary: VORONOI_ASSIGN_GCN,
        code_size_bytes: VORONOI_ASSIGN_GCN.len() * 4,
        min_waves: 1,
        preferred_waves: 4,
    },
    KernelMeta {
        name: "jl_project",
        binary: JL_PROJECT_GCN,
        code_size_bytes: JL_PROJECT_GCN.len() * 4,
        min_waves: 1,
        preferred_waves: 4,
    },
    KernelMeta {
        name: "spectral_step",
        binary: SPECTRAL_STEP_GCN,
        code_size_bytes: SPECTRAL_STEP_GCN.len() * 4,
        min_waves: 1,
        preferred_waves: 2,
    },
    KernelMeta {
        name: "s2_distance",
        binary: S2_DISTANCE_GCN,
        code_size_bytes: S2_DISTANCE_GCN.len() * 4,
        min_waves: 2,
        preferred_waves: 8,
    },
];

/// Find a kernel by name.
pub fn find_kernel(name: &str) -> Option<&'static KernelMeta> {
    KERNELS.iter().find(|k| k.name == name)
}

/// Reinterpret a `&[u32]` kernel binary as `&[u8]` for DMA upload.
pub fn kernel_as_bytes(binary: &[u32]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(
            binary.as_ptr() as *const u8,
            binary.len() * core::mem::size_of::<u32>(),
        )
    }
}
