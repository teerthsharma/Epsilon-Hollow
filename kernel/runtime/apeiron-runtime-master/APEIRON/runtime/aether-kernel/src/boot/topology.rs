// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

use alloc::vec::Vec;
use acpi::{AcpiHandler, PhysicalMapping};
use core::ptr::NonNull;

#[derive(Clone)]
pub struct KernelAcpiHandler;

impl AcpiHandler for KernelAcpiHandler {
    unsafe fn map_physical_region<T>(
        &self,
        physical_address: usize,
        size: usize,
    ) -> PhysicalMapping<Self, T> {
        PhysicalMapping::new(
            physical_address,
            NonNull::new(physical_address as *mut _).expect("Failed to map physical address (is null?)"),
            size,
            size,
            self.clone(),
        )
    }

    fn unmap_physical_region<T>(_region: &PhysicalMapping<Self, T>) {
        // Nothing to do for identity mapping
    }
}

/// Represents the I/O capabilities of the organism (hardware).
#[derive(Debug, Clone, Copy, Default)]
pub struct IoCaps {
    pub has_serial: bool,
    pub has_framebuffer: bool,
    pub has_keyboard: bool,
    pub has_network: bool,
}

/// The graph of the hardware organism.
/// Nodes are Cores/Memory, edges are latency/bandwidth.
#[derive(Debug, Clone)]
pub struct HardwareTopology {
    /// Number of neural clusters (CPU Cores)
    pub cpu_cores: usize,
    /// Number of memory localities (NUMA Nodes)
    pub numa_nodes: usize,
    /// Total synaptic space (System Memory) in bytes
    pub total_memory: u64,
    /// Sensory/Motor capabilities (I/O)
    pub io_capabilities: IoCaps,
    /// L3 Cache sharing groups (for manifold parallelism)
    pub cache_groups: Vec<usize>,
}

impl HardwareTopology {
    pub fn new() -> Self {
        Self {
            cpu_cores: 1,
            numa_nodes: 1,
            total_memory: 0,
            io_capabilities: IoCaps::default(),
            cache_groups: Vec::new(),
        }
    }

    /// Perform a "Bio-Scan" to populate the topology from boot info.
    pub fn bio_scan(boot_info: &super::bios::BootInfo) -> Self {
        let mut mem_total = 0;
        // Calculate total memory
        boot_info.walk_memory_map(|region| {
            if region.kind == super::bios::MemoryRegionKind::Usable {
                mem_total += region.end.0 - region.start.0;
            }
        });
        
        let mut caps = IoCaps::default();
        if boot_info.framebuffer().is_some() {
            caps.has_framebuffer = true;
        }

        let mut cpu_cores = 1;
        // Parse ACPI for actual core count
        if let Some(rsdp) = boot_info.config_root() {
            let handler = KernelAcpiHandler;
            // Safety: We trust the RSDP address from Multiboot2
            if let Ok(tables) = unsafe { acpi::AcpiTables::from_rsdp(handler, rsdp.0 as usize) } {
                if let Ok(platform_info) = tables.platform_info() {
                    if let Some(processor_info) = platform_info.processor_info {
                        cpu_cores = 1 + processor_info.application_processors.len();
                    }
                }
            }
        }

        let numa_nodes = if let Some(rsdp_addr) = boot_info.config_root() {
            unsafe { super::acpi::parse_numa_nodes(rsdp_addr.0) }
        } else {
            1
        };

        Self {
            cpu_cores,
            numa_nodes,
            total_memory: mem_total,
            io_capabilities: caps,
            cache_groups: Vec::new(),
        }
    }
}
