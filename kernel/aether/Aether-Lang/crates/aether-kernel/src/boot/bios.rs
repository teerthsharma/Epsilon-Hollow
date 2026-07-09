// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

use aether_core::os::PhysAddr;
use multiboot2::{BootInformation, BootInformationHeader};

/// A region of physical memory.
#[derive(Debug, Clone, Copy)]
pub struct MemoryRegion {
    pub start: PhysAddr,
    pub end: PhysAddr,
    pub kind: MemoryRegionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegionKind {
    Usable,
    Reserved,
    Acpi,
    Kernel,
    Bootloader,
    Unknown,
}

/// Framebuffer information (Sensory Window).
#[derive(Debug, Clone, Copy)]
pub struct Framebuffer {
    pub address: PhysAddr,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub bpp: u8,
}

/// The interface that the BIOS/Bootloader must satisfy.
/// Acts as the "DNA transcription" layer.
pub trait BiosInterface {
    // Required methods would go here.
}

/// BootInfo passed from the bootloader.
pub struct BootInfo {
    multiboot_start: PhysAddr,
}

impl BootInfo {
    /// Create a new BootInfo from the physical address of the multiboot2 structure.
    ///
    /// # Safety
    /// The caller must ensure that `multiboot_start` points to a valid Multiboot2 information structure.
    pub unsafe fn new(multiboot_start: PhysAddr) -> Self {
        Self { multiboot_start }
    }

    /// Access the raw multiboot information.
    fn raw(&self) -> Option<BootInformation<'_>> {
        unsafe {
            BootInformation::load(self.multiboot_start.0 as *const BootInformationHeader).ok()
        }
    }

    /// Iterate over the memory map using a callback.
    /// This avoids returning complex iterators with lifetimes.
    pub fn walk_memory_map<F>(&self, mut f: F)
    where
        F: FnMut(MemoryRegion),
    {
        if let Some(info) = self.raw() {
            if let Some(tag) = info.memory_map_tag() {
                for area in tag.memory_areas() {
                    f(MemoryRegion {
                        start: PhysAddr(area.start_address()),
                        end: PhysAddr(area.end_address()),
                        kind: match multiboot2::MemoryAreaType::from(area.typ()) {
                            multiboot2::MemoryAreaType::Available => MemoryRegionKind::Usable,
                            multiboot2::MemoryAreaType::Reserved => MemoryRegionKind::Reserved,
                            multiboot2::MemoryAreaType::AcpiAvailable => MemoryRegionKind::Acpi,
                            multiboot2::MemoryAreaType::ReservedHibernate => {
                                MemoryRegionKind::Reserved
                            }
                            _ => MemoryRegionKind::Unknown,
                        },
                    });
                }
            }
        }
    }

    pub fn framebuffer(&self) -> Option<Framebuffer> {
        let info = self.raw()?;
        let tag = info.framebuffer_tag()?.ok()?;
        Some(Framebuffer {
            address: PhysAddr(tag.address()),
            width: tag.width(),
            height: tag.height(),
            pitch: tag.pitch(),
            bpp: tag.bpp(),
        })
    }

    pub fn config_root(&self) -> Option<PhysAddr> {
        let info = self.raw()?;

        // Try RSDP (new ACPI)
        if let Some(tag) = info.rsdp_v2_tag() {
            let sig: &str = tag.signature().ok()?;
            return Some(PhysAddr(sig.as_ptr() as u64));
        }

        // Try RSDP (old ACPI)
        if let Some(_tag) = info.rsdp_v1_tag() {
            return None; // Placeholder
        }

        None
    }
}
