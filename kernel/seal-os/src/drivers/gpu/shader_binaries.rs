// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Embedded GCN ISA kernels for topology acceleration.
//!
//! These binaries are compiled from OpenCL C sources in `shaders/` by the build
//! script (`build.rs`) invoking `scripts/compile_shaders.py`.  When ROCm is not
//! available on the host, placeholder stub binaries are generated so that the
//! kernel still compiles.
//!
//! Each kernel follows the AMD GCN compute ABI:
//! - s0-s1: global work offset (ignored, set to 0)
//! - s2-s3: kernel argument buffer address (64-bit pointer)
//! - User data registers (s4+) contain extra parameters.
//!
//! # Rebuilding kernels from source
//!
//! ```bash
//! # Install ROCm LLVM, then from repo root:
//! python3 scripts/compile_shaders.py \
//!     kernel/seal-os/src/drivers/gpu/shaders/ \
//!     kernel/seal-os/src/drivers/gpu/shaders/
//! ```

// ------------------------------------------------------------------
// Compiled kernel binaries (generated at build time from .cl sources)
// ------------------------------------------------------------------

/// `voronoi_assign` kernel binary — GCN ISA for gfx803+ (RX 400/500 series).
pub static VORONOI_ASSIGN_GCN: &[u8] = include_bytes!("shaders/voronoi_assign.bin");

/// `jl_project` kernel binary — GCN ISA for gfx803+.
pub static JL_PROJECT_GCN: &[u8] = include_bytes!("shaders/jl_project.bin");

/// `spectral_step` kernel binary — GCN ISA for gfx803+.
pub static SPECTRAL_STEP_GCN: &[u8] = include_bytes!("shaders/spectral_step.bin");

/// `s2_distance` kernel binary — GCN ISA for gfx803+.
pub static S2_DISTANCE_GCN: &[u8] = include_bytes!("shaders/s2_distance.bin");

// ------------------------------------------------------------------
// Kernel metadata table
// ------------------------------------------------------------------

/// Metadata for each embedded kernel.
pub struct KernelMeta {
    pub name: &'static str,
    pub binary: &'static [u8],
    pub code_size_bytes: usize,
    pub min_waves: u32,
    pub preferred_waves: u32,
}

pub const KERNELS: &[KernelMeta] = &[
    KernelMeta {
        name: "voronoi_assign",
        binary: VORONOI_ASSIGN_GCN,
        code_size_bytes: VORONOI_ASSIGN_GCN.len(),
        min_waves: 1,
        preferred_waves: 4,
    },
    KernelMeta {
        name: "jl_project",
        binary: JL_PROJECT_GCN,
        code_size_bytes: JL_PROJECT_GCN.len(),
        min_waves: 1,
        preferred_waves: 4,
    },
    KernelMeta {
        name: "spectral_step",
        binary: SPECTRAL_STEP_GCN,
        code_size_bytes: SPECTRAL_STEP_GCN.len(),
        min_waves: 1,
        preferred_waves: 2,
    },
    KernelMeta {
        name: "s2_distance",
        binary: S2_DISTANCE_GCN,
        code_size_bytes: S2_DISTANCE_GCN.len(),
        min_waves: 2,
        preferred_waves: 8,
    },
];

/// Find a kernel by name.
pub fn find_kernel(name: &str) -> Option<&'static KernelMeta> {
    KERNELS.iter().find(|k| k.name == name)
}

/// Reinterpret a `&[u8]` kernel binary for DMA upload (identity for &[u8]).
pub fn kernel_as_bytes(binary: &[u8]) -> &[u8] {
    binary
}
