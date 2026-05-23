// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! GPU driver — PCI vendor detection. 2D acceleration and compute dispatch are not yet implemented.

pub mod intel;
pub mod nvidia;
pub mod virtio_gpu;

use crate::serial_println;
use spin::Mutex;

static VIRTIO_GPU: Mutex<Option<virtio_gpu::VirtioGpu>> = Mutex::new(None);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    VirtIO,
    Unknown,
}

impl GpuVendor {
    pub fn from_pci(vendor_id: u16) -> Self {
        match vendor_id {
            0x10DE => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            0x1AF4 => Self::VirtIO,
            _ => Self::Unknown,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::VirtIO => "VirtIO-GPU",
            Self::Unknown => "Unknown",
        }
    }
}

pub struct GpuDriver {
    vendor: GpuVendor,
    vram_mb: u32,
    compute_budget_mb: u32,
    initialized: bool,
}

impl GpuDriver {
    pub fn new() -> Self {
        Self {
            vendor: GpuVendor::Unknown,
            vram_mb: 0,
            compute_budget_mb: 2048,
            initialized: false,
        }
    }

    pub fn init_from_pci(&mut self, vendor_id: u16, _bar0: u32) {
        self.vendor = GpuVendor::from_pci(vendor_id);
        self.initialized = true;
    }

    pub fn blit(&self, _src: *const u32, _dst: *mut u32, w: u32, h: u32) {
        if self.vendor == GpuVendor::VirtIO {
            let mut gpu_lock = VIRTIO_GPU.lock();
            if let Some(ref mut gpu) = *gpu_lock {
                // For simplicity, we assume blit means transfer to host and flush the whole area
                let _ = gpu.transfer_to_host_2d(0, 0, w, h);
                let _ = gpu.flush(0, 0, w, h);
            }
        }
    }

    pub fn fill_rect(&self, _fb: *mut u32, x: u32, y: u32, w: u32, h: u32, _color: u32) {
        if self.vendor == GpuVendor::VirtIO {
            let mut gpu_lock = VIRTIO_GPU.lock();
            if let Some(ref mut gpu) = *gpu_lock {
                let _ = gpu.transfer_to_host_2d(x, y, w, h);
                let _ = gpu.flush(x, y, w, h);
            }
        }
    }

    pub fn vendor(&self) -> GpuVendor {
        self.vendor
    }

    pub fn is_available(&self) -> bool {
        self.initialized
    }
}

pub fn init() {
    // Try vendor-specific drivers first.
    if nvidia::init().is_some() {
        return;
    }
    if intel::init().is_some() {
        return;
    }

    // VirtIO-GPU check
    if let Ok(gpu) = virtio_gpu::VirtioGpu::discover_and_init() {
        serial_println!("[GPU] VirtIO-GPU driver initialized");
        *VIRTIO_GPU.lock() = Some(gpu);
        return;
    }

    // Fallback: generic PCI scan for any display controller.
    let mut driver = GpuDriver::new();
    let devices = crate::drivers::pci::enumerate();
    for dev in &devices {
        if dev.class == 0x03 {
            driver.init_from_pci(dev.vendor_id, dev.bar_address(0) as u32);
            crate::serial_println!(
                "[GPU] {} display controller detected ({:04X}:{:04X}) -- no vendor driver",
                driver.vendor().name(),
                dev.vendor_id,
                dev.device_id
            );
            return;
        }
    }
    crate::serial_println!("[GPU] No display controller detected");
}
