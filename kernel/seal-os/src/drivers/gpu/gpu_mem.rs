// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! GPU Memory Manager — allocates physically contiguous pages that are visible
//! to both the CPU (identity-mapped) and the GPU (via GART or PCIe BAR aperture).
//!
//! For AMD GCN, we use the GART aperture (BAR4 or BAR2) to map CPU pages into
//! GPU-visible addresses.  For simplicity in this research kernel, we allocate
//! from the standard physical frame allocator and rely on the identity map
//! (first 4 GiB) being reachable by the GPU via 32-bit DMA.

use crate::memory::phys::alloc_frames_contiguous;
use alloc::vec::Vec;

/// A buffer allocated in GPU-visible memory.
pub struct GpuBuffer {
    /// Physical address (CPU and GPU can both access this).
    pub phys: u64,
    /// Size in bytes.
    pub size: usize,
    /// Number of contiguous 4 KiB frames.
    pub frames: usize,
}

impl GpuBuffer {
    /// Allocate a GPU-visible buffer of `size` bytes (rounded up to page).
    /// Returns `None` if the physical allocator cannot satisfy the request.
    pub fn alloc(size: usize) -> Option<Self> {
        let frames = (size + 4095) / 4096;
        let phys = alloc_frames_contiguous(frames)?.as_u64();
        // Zero the memory for determinism.
        unsafe {
            core::ptr::write_bytes(phys as *mut u8, 0, frames * 4096);
        }
        Some(Self { phys, size, frames })
    }

    /// Free the buffer back to the physical allocator.
    pub fn free(self) {
        let frame_size = 4096u64;
        for i in 0..self.frames {
            let addr = x86_64::PhysAddr::new(self.phys + (i as u64) * frame_size);
            unsafe {
                crate::memory::phys::free_frame(addr);
            }
        }
    }

    /// Return a mutable byte slice for CPU access.
    ///
    /// # Safety
    /// The caller must ensure that `self.phys` points to valid, mapped memory
    /// with at least `self.size` bytes accessible. The returned slice must not
    /// outlive the buffer or alias with GPU DMA operations targeting the same
    /// memory. Violating these invariants causes undefined behavior.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        core::slice::from_raw_parts_mut(self.phys as *mut u8, self.size)
    }

    /// Return an immutable byte slice for CPU access.
    ///
    /// # Safety
    /// The caller must ensure that `self.phys` points to valid, mapped memory
    /// with at least `self.size` bytes accessible. The returned slice must not
    /// outlive the buffer or alias with concurrent GPU DMA or CPU writes.
    pub unsafe fn as_slice(&self) -> &[u8] {
        core::slice::from_raw_parts(self.phys as *const u8, self.size)
    }

    /// Cast to a typed mutable slice.
    pub unsafe fn as_mut_typed<T>(&mut self) -> &mut [T] {
        let count = self.size / core::mem::size_of::<T>();
        core::slice::from_raw_parts_mut(self.phys as *mut T, count)
    }

    /// Cast to a typed immutable slice.
    pub unsafe fn as_typed<T>(&self) -> &[T] {
        let count = self.size / core::mem::size_of::<T>();
        core::slice::from_raw_parts(self.phys as *const T, count)
    }
}

/// A pool of pre-allocated GPU buffers for fast reuse.
pub struct GpuBufferPool {
    buffers: Vec<GpuBuffer>,
    buf_size: usize,
}

impl GpuBufferPool {
    pub fn new(buf_size: usize, count: usize) -> Option<Self> {
        let mut buffers = Vec::with_capacity(count);
        for _ in 0..count {
            let buf = GpuBuffer::alloc(buf_size)?;
            buffers.push(buf);
        }
        Some(Self { buffers, buf_size })
    }

    pub fn acquire(&mut self) -> Option<GpuBuffer> {
        self.buffers.pop()
    }

    pub fn release(&mut self, buf: GpuBuffer) {
        if buf.size == self.buf_size {
            self.buffers.push(buf);
        } else {
            buf.free();
        }
    }
}

/// Copy data from host slice into GPU buffer.
///
/// # Safety
/// `dst.phys` must point to valid, writable memory of at least `min(dst.size, src.len())`
/// bytes. The destination must not overlap with `src`. Concurrent GPU access to
/// `dst` during the copy results in a data race and undefined behavior.
pub unsafe fn upload(dst: &GpuBuffer, src: &[u8]) {
    let len = core::cmp::min(dst.size, src.len());
    core::ptr::copy_nonoverlapping(src.as_ptr(), dst.phys as *mut u8, len);
}

/// Copy data from GPU buffer into host slice.
///
/// # Safety
/// `src.phys` must point to valid, readable memory of at least `min(src.size, dst.len())`
/// bytes. The source and destination must not overlap. Concurrent GPU writes to
/// `src` during the copy result in a data race and undefined behavior.
pub unsafe fn download(dst: &mut [u8], src: &GpuBuffer) {
    let len = core::cmp::min(src.size, dst.len());
    core::ptr::copy_nonoverlapping(src.phys as *const u8, dst.as_mut_ptr(), len);
}
