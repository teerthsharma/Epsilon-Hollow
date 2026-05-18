# Memory Subsystem Reference

> Source: `kernel/seal-os/src/memory/mod.rs`, `kernel/seal-os/src/boot/boot.S`, `kernel/seal-os/src/graphics/framebuffer.rs`

---

## Physical Memory Layout

The kernel is loaded at 1MB (set by `linker.ld` line 5: `. = 1M`). The boot trampoline identity-maps the first 4GB of physical address space via 2MB huge pages (boot.S lines 47-55).

```
0x00000000 - 0x000FFFFF   First 1MB (BIOS, VGA, legacy I/O)
0x00100000 - ...           Kernel image (.multiboot_header, .multiboot_boot, .text, .rodata, .data, .bss)
  .bss contains:
    pml4          4KB       Page Map Level 4
    pdpt          4KB       Page Directory Pointer Table
    pd0-pd3       16KB      Page Directories (4 x 4KB)
    boot_stack    64KB      Boot stack (boot.S line 119)
    HEAP          16MB      Kernel heap (memory/mod.rs line 10)
...
0xFD000000 (approx)        Framebuffer MMIO (varies by hardware/QEMU config)
...
0xFFFFFFFF                  End of 4GB identity map
```

---

## Bump Allocator Design

Source: `kernel/seal-os/src/memory/mod.rs`

### Constants

| Constant | Value | Location |
|----------|-------|----------|
| `HEAP_SIZE` | `16 * 1024 * 1024` (16 MB) | line 10 |
| Alignment | 4096 bytes (page-aligned) | `#[repr(C, align(4096))]` on `HeapStorage`, line 12 |

### Storage

The heap is a static 16MB byte array in BSS (lines 12-15):

```rust
#[repr(C, align(4096))]
struct HeapStorage([u8; HEAP_SIZE]);

static mut HEAP: HeapStorage = HeapStorage([0; HEAP_SIZE]);
```

### Allocator State

The `BumpAllocator` struct (lines 17-19) contains a single field:
- `next: usize` -- offset of the next free byte within the heap

Protected by a `spin::Mutex` (line 37) for interrupt-safe access.

### Allocation Logic -- `alloc(&mut self, layout: Layout) -> *mut u8` (lines 26-34)

1. **Align up**: `start = (self.next + layout.align() - 1) & !(layout.align() - 1)` (line 27)
   - This rounds `self.next` up to the nearest multiple of the requested alignment
2. **Compute end**: `end = start + layout.size()` (line 28)
3. **Bounds check**: if `end > HEAP_SIZE`, return `null_mut()` (lines 29-31)
4. **Advance pointer**: `self.next = end` (line 32)
5. **Return address**: `HEAP base pointer + start` (line 33)

### Deallocation -- `dealloc(&self, _ptr: *mut u8, _layout: Layout)` (line 45)

Empty body. Deallocation is a no-op.

### GlobalAlloc Implementation

The `SealAllocator` struct (line 39) implements `GlobalAlloc` (lines 41-46) and is registered as `#[global_allocator]` (lines 48-49).

### Initialization

`init_heap()` (lines 51-53) is a no-op marker function. The static buffer requires no runtime initialization; calling this function marks the "heap initialized" boot milestone.

---

## 4GB Identity Map Rationale

Source: `boot.S` lines 19-55

The boot trampoline maps 4GB (not just 2GB) using four page directories (pd0-pd3), each containing 512 entries of 2MB huge pages. The comment on line 33 explains:

> PDPT[0..3] -> PD0..PD3 (identity map first 4GB for MMIO framebuffers)

This is necessary because:

1. **Framebuffer MMIO**: The VGA/VESA framebuffer is memory-mapped at a high physical address (typically around 0xFD000000 in QEMU). Without the 3-4GB range mapped, the kernel cannot write to the framebuffer.

2. **Simplicity**: With identity mapping (virtual == physical), all physical addresses are directly accessible without translation. This eliminates the need for a virtual memory manager in the early kernel.

3. **2MB huge pages**: Using huge pages (flag `0x83` = present + writable + PS bit) instead of 4KB pages means only 2048 page directory entries are needed (vs. 1M page table entries for 4KB pages), fitting in 4 pages of PD space.

---

## Framebuffer MMIO

Source: `kernel/seal-os/src/graphics/framebuffer.rs`

The `Framebuffer` struct (lines 6-12) wraps a raw pointer to the MMIO framebuffer:

```rust
pub struct Framebuffer {
    buffer: *mut u8,     // MMIO base address (e.g., 0xFD000000)
    pub width: u32,      // Pixels (requested: 1024, from multiboot2.rs)
    pub height: u32,     // Pixels (requested: 768)
    pub pitch: u32,      // Bytes per row
    pub bpp: u8,         // Bits per pixel (requested: 32)
}
```

The framebuffer address is provided by GRUB via the multiboot2 info structure. The multiboot2 header requests 1024x768x32bpp (multiboot2.rs lines 27-31).

### Pixel Writing

`put_pixel(&self, x: u32, y: u32, color: u32)` (lines 42-51):
- Bounds check: x < width, y < height, buffer not null (line 43)
- Offset: `y * pitch + x * (bpp / 8)` (line 47)
- Write: `ptr::write_volatile(pixel, color)` -- volatile to prevent compiler optimization of MMIO writes (line 50)

The volatile write is critical: without it, the compiler could reorder or eliminate writes to the framebuffer address, since from the compiler's perspective it looks like writing to unused memory.

---

## Why Bump-Only (No Dealloc) Is Sufficient

The bump allocator never frees memory. This design is appropriate for Seal OS Phase 1 because:

1. **Kernel lifetime model**: The kernel initializes its data structures (ManifoldFS, scheduler, interrupt tables) once at boot and never deallocates them. There is no dynamic process creation/destruction that would require memory reclamation.

2. **16MB is generous**: The kernel's primary consumers are:
   - ManifoldFS inodes and directory entries (BTreeMap nodes)
   - Voronoi cell file lists (Vec<u64>)
   - Scheduler task list (Vec<Task>)
   - Activity log (Vec<TheoremEvent>, capped at 100 entries in manifold_fs.rs line 513)
   
   For a demonstration kernel with a handful of files and tasks, 16MB is more than sufficient.

3. **No fragmentation**: A bump allocator has zero fragmentation overhead. Every allocation is perfectly packed (modulo alignment). This is ideal when total memory usage is bounded and monotonically increasing.

4. **Minimal code**: The allocator is 54 lines total (mod.rs). No free lists, no coalescing, no buddy system. This reduces the attack surface and complexity of the most security-critical kernel subsystem.

5. **Lock simplicity**: A single `spin::Mutex<BumpAllocator>` with one atomic field (`next`) is the simplest possible thread-safe allocator. No lock ordering concerns, no deadlock risk from nested allocation.

When Seal OS needs dynamic memory reclamation (Phase 2+), the bump allocator can be replaced with a slab or buddy allocator behind the same `GlobalAlloc` trait interface.
