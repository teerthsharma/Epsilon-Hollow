// Seal OS - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Embedded GCN ISA kernels for topology acceleration.
//!
//! The kernel embeds only real checked-in ISA blobs from `shaders/*.bin`.
//! Missing blobs mean the hardware path reports `kernel_not_found` and the OS
//! keeps using the proven CPU topology fallback. The build never fabricates
//! placeholder GPU kernels.
//!
//! Each kernel follows the AMD GCN compute ABI:
//! - s0-s1: global work offset (ignored, set to 0)
//! - s2-s3: kernel argument buffer address (64-bit pointer)
//! - User data registers (s4+) contain extra parameters.
//!
//! To enable hardware dispatch, check in architecture-matched `.bin` files
//! beside the OpenCL sources and boot-gate them with `[GPU-BENCH]`.

/// Metadata for each embedded kernel.
pub struct KernelMeta {
    pub name: &'static str,
    pub binary: &'static [u8],
    pub code_size_bytes: usize,
    pub min_waves: u32,
    pub preferred_waves: u32,
}

include!(concat!(env!("OUT_DIR"), "/generated_shader_binaries.rs"));

/// Find a kernel by name.
pub fn find_kernel(name: &str) -> Option<&'static KernelMeta> {
    KERNELS.iter().find(|k| k.name == name)
}

/// Reinterpret a `&[u8]` kernel binary for DMA upload (identity for &[u8]).
pub fn kernel_as_bytes(binary: &[u8]) -> &[u8] {
    binary
}
