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

/// Find a kernel by name, treating a zero-length blob as absent.
///
/// `build.rs` embeds any `shaders/*.bin` that exists, and three of them are
/// still zero-length placeholders.  Handing an empty blob to a dispatcher would
/// upload nothing and then set `COMPUTE_PGM` to whatever the allocator returned,
/// which fails as a GPU hang rather than as an error.  Filtering here makes
/// every caller take the documented `kernel_not_found` path instead.
pub fn find_kernel(name: &str) -> Option<&'static KernelMeta> {
    KERNELS
        .iter()
        .find(|k| k.name == name && !k.binary.is_empty())
}

/// Reinterpret a `&[u8]` kernel binary for DMA upload (identity for &[u8]).
pub fn kernel_as_bytes(binary: &[u8]) -> &[u8] {
    binary
}
