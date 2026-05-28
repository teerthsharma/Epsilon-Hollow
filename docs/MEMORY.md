# Memory Subsystem Reference

> Sources: `kernel/seal-os/src/memory/`, `kernel/seal-os/src/boot/uefi_entry.rs`, `kernel/seal-os/src/graphics/framebuffer.rs`

Seal OS no longer uses a GRUB/Multiboot trampoline memory story. UEFI provides the memory map, GOP framebuffer information, and loaded PE/COFF image. After `ExitBootServices`, Seal OS owns conventional RAM and builds its allocator state from `BootInfo`.

## Boot Memory Inputs

| Input | Source | Used by |
|---|---|---|
| UEFI memory map | `uefi_entry.rs` | physical allocator, reserved-region filtering |
| Kernel base/size | `BootInfo` | prevent allocator from reusing kernel image pages |
| Framebuffer addr/pitch/size | UEFI GOP or VBE fallback | reserve MMIO range, graphics output |
| ACPI RSDP | UEFI config tables | ACPI/APIC/SMP setup |

## Allocators

| Layer | Source | Role |
|---|---|---|
| Physical frame bitmap | `memory/phys.rs` | Tracks 4 KiB frame ownership from UEFI conventional RAM |
| Slab allocator | `memory/slab.rs` | Small kernel allocations from 64 B to 2 KiB |
| Heap facade | `memory/heap.rs`, `memory/mod.rs` | Global allocator integration |
| Virtual memory | `memory/virt.rs`, `memory/mmap.rs` | Page tables, user mappings, lazy mmap path |
| Topological RAM | `memory/topo_ram.rs` | T1-T5 metadata over physical frames |

## Topological RAM

`topo_ram.rs` is the theorem-native memory layer:

| Theorem | Memory role |
|---|---|
| T1/TSS | Voronoi cell assignment for locality-aware allocation |
| T2/SCM | Spectral prefetch hints from recent allocation/access vectors |
| T3/GMC | Fragmentation entropy and reseeding pressure |
| T4/AGCR | Allocation pressure governor |
| T5/HCS | Hyperbolic lifetime classification and swap candidate bias |

This is the OS direction: memory is not just a flat page array. It is a measured manifold over frames, access history, lifetime, and pressure.

## Framebuffer MMIO

The framebuffer is created from UEFI GOP/VBE data in `BootInfo`.

```rust
pub struct Framebuffer {
    buffer: *mut u8,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub bpp: u8,
}
```

Pixel writes use volatile MMIO writes so the compiler cannot remove or reorder display output.

## VM Notes

QEMU and Oracle VM VirtualBox may place the framebuffer and MMIO windows in different physical ranges. Seal OS reserves the framebuffer range reported by firmware before handing conventional RAM to the allocator. VirtualBox GOP quirks are handled in `src/boot/uefi_entry.rs` with a VBE fallback.
