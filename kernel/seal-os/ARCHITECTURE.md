# Seal OS Architecture

## Overview

Current 0.4.7.5 proof scope: topology drives hot paths, single-frame allocation
uses a bounded topological free index, and contiguous DMA allocation is capped at
64 pages per request. Larger transfers must use chunked/scatter-gather I/O
instead of one unbounded contiguous request. The TopoRAM proof marker exercises
hint-biased target-cell allocation and fails the benchmark gate if it falls back.

Seal OS is a research operating system that proves topology can drive every layer of a real OS kernel. The core thesis: if you embed all state — files, processes, memory regions, screen areas — onto the unit sphere S², then Voronoi partitions give you O(1) lookup, spectral contraction gives you prediction, and geometric governors give you adaptive control.

## Boot Sequence

```
UEFI firmware -> EFI/BOOT/BOOTX64.EFI
  -> src/boot/uefi_entry.rs
  -> serial online
  -> ACPI RSDP from UEFI config tables
  -> GOP framebuffer or VBE fallback for VirtualBox
  -> exit boot services with UEFI memory map
  -> kernel_main(BootInfo)
```

The first-class artifact is `target/x86_64-unknown-uefi/release/seal-os.img`, a GPT disk with a FAT EFI System Partition. Oracle VM VirtualBox uses a converted `seal-os.vdi`.

### _start() initialization order

1. **Serial** — COM1 at 115200 baud for debug output
2. **Heap** — 16MB static buffer with bump allocator
3. **IDT + PIC** — 256-entry IDT, 8259A PIC remapped to IRQ 32+
4. **Framebuffer** - Use UEFI GOP framebuffer details from `BootInfo`
5. **Boot splash** — ASCII seal art + progress bar
6. **Theorems** - Boot-verify T1-T10 through `aether_verified`; activate T1-T5 runtime paths
7. **ManifoldFS** — Initialize filesystem, demo O(1) teleport
8. **Scheduler** — Spawn kernel/compositor/shell/idle tasks
9. **Syscalls** — Dispatch table verification
10. **Shell** — Populate demo files, execute test commands
11. **Desktop** — Render wallpaper, create windows, compose to FB

## Memory Model

```
Physical Memory Layout:
0x00000000 - 0x000FFFFF  Reserved firmware/VGA area
UEFI provided regions    Kernel image, boot services memory, runtime services memory
Conventional RAM         Physical allocator after ExitBootServices
Framebuffer              UEFI GOP address from BootInfo

Kernel page tables are built from the UEFI memory map and Seal OS allocator state.
```

Current heap state: fixed slab classes serve small allocations, page-backed paths serve large allocations, and the topological physical allocator backs frame ownership. Legacy bump-only notes no longer describe the 0.4.7.5 runtime.

Current 0.4.7.5 allocator note: small objects use fixed slab classes, large
allocations use page-backed paths, and physical frames come from the topological
allocator.

## Interrupt Architecture

```
IRQ 0  (PIT Timer)    → tick counter, scheduler timeslice check
IRQ 1  (Keyboard)     → scancode → ASCII, routed to focused window
IRQ 12 (Mouse)        → packet read (stub — full protocol needs state machine)
IRQ 2  (Cascade)      → PIC2 chain

Exception 3  (Breakpoint)   → serial log
Exception 8  (Double Fault) → kernel panic
```

PIC initialization: ICW1-4 sequence, remap IRQ 0-7 to INT 32-39, IRQ 8-15 to INT 40-47.

## ManifoldFS Design

### Encoding Pipeline

```
Input bytes → chunk into 4KB blocks
  → each block: trigram hash → 128-dim sparse vector
  → JL random projection: 128-dim → 3-dim
  → L2 normalize onto unit sphere S²
  → collect up to 64 points = ManifoldPayload
```

Regardless of input size (1KB or 100GB), the payload is always ≤ 1,536 bytes (64 × 3 × 8).

### Inode Structure

```rust
struct Inode {
    id: u64,
    name: String,
    kind: InodeKind,          // File | Directory
    payload: ManifoldPayload, // 64 points on S²
    metadata: InodeMetadata,  // timestamps, size, permissions
    voronoi_cell: usize,      // T1: which of 8 cells
    cluster_id: i32,          // for T3 merging
    parent: u64,              // parent directory inode
}
```

### Teleportation (O(1) file move)

Moving a file between directories:
1. Remove directory entry from source `BTreeMap` — O(log n)
2. Insert directory entry in destination `BTreeMap` — O(log n)
3. Update inode parent pointer — O(1)

The ManifoldPayload stays in the same inode. No data is copied. For BTreeMap operations, n = number of files in the directory (typically small), making this effectively O(1).

### Theorem Activity in ManifoldFS

- **T1**: `assign_voronoi_cell()` maps each file's first sphere point to one of 8 Voronoi cells. `find()` searches only within the matching cell.
- **T2**: `update_prefetch_state()` contracts the access state vector toward recent accesses, predicting which file will be accessed next.
- **T3**: `check_entropy_and_merge()` computes Shannon entropy of cluster sizes. When H > 2.0 bits, merges the two smallest clusters.
- **T4**: `governor_tick()` adapts epsilon after every store/teleport. Zero deviation during surgery stabilizes the governor.
- **T5**: `update_hyperbolic_ratio()` computes separation_ratio = ln(4) × D × (D-1)/2 for directory depth D.

## Scheduler Design

### ManifoldScheduler

Each task has an 8-dimensional manifold embedding computed from its task ID (deterministic hash → normalize to unit sphere in R⁸).

```
Timer IRQ fires → ticks_in_slice++
  → if ticks >= adaptive_timeslice:
    → T1: select Voronoi cell predicted by T2
    → pick highest-priority Ready task in that cell
    → if none: pick any Ready task (fallback)
    → T2: contract predict_state toward new task's embedding
    → T4: governor adapts timeslice length
    → context switch (currently: just update state fields)
```

### Adaptive Timeslice (T4)

```
timeslice = base_timeslice × scale
where scale = 2.0 if epsilon < 0.5 (system stable → longer slices)
              0.5 if epsilon ≥ 0.5 (system volatile → shorter slices)
```

## Window Manager

### Compositor

Double-buffered compositing: each window has its own pixel buffer. On compose():
1. Render desktop wallpaper (once, cached)
2. Blit windows bottom-to-top (z-order)
3. Draw mouse cursor on top

**T4 adaptive FPS**: When governor epsilon < 0.3 (system idle), only compose every 4th frame (15fps). Otherwise 60fps.

### Window Structure

```
┌──────────────── width ────────────────┐
│ Title Bar (24px) ─────── [X] Close    │ ← TITLE_BAR_HEIGHT
├───────────────────────────────────────│
│ │                                   │ │
│ │  Client Area                      │ │ ← set_client_pixel()
│ │  (app renders here)               │ │
│ │                                   │ │
│ │                                   │ │
├─┴───────────────────────────────────┴─┤
│ Border (2px)                          │ ← BORDER_WIDTH
└───────────────────────────────────────┘
```

### Input Event Model

```rust
enum InputEvent {
    KeyPress(u8),
    KeyRelease(u8),
    MouseMove { dx: i32, dy: i32 },
    MouseButton { button: u8, pressed: bool },
    Scroll { dy: i32 },
}
```

All input devices (PS/2 keyboard, mouse, future USB HID) produce InputEvents. The compositor routes them to the focused window.

## Desktop Environment

### Wallpaper

Procedurally rendered:
- Subtle 64px grid pattern on dark background (#0A0A0F)
- Central radial glow (gaussian falloff, 250px radius)
- Black hole visualization: 40-60px radius ring
- Accretion disk: 3 elliptical rings at 70/95/120px

### Taskbar

28px bottom bar:
- Left: 5 theorem status indicators (green dots when active)
- Center: Epsilon readout
- Right: "Seal OS" branding

## Seal ABI Interface

| Number | Name | Arguments | Returns |
|--------|------|-----------|---------|
| 0 | sys_exit | code | — |
| 1 | sys_write | fd, buf, len | bytes written |
| 2 | sys_read | fd, buf, len | bytes read |
| 3 | sys_open | path, flags | fd |
| 4 | sys_close | fd | 0 |
| 5 | sys_exec | path | — |
| 6 | sys_fork | — | pid |
| 7 | sys_waitpid | pid | status |
| 8 | sys_mmap | addr, len, prot | addr |
| 9 | sys_getpid | — | pid |
| 10 | sys_stat | path, buf | 0 |
| 100 | sys_manifold_query | theorem_id | theorem state |
| 101 | sys_teleport | src, dst | result |
| 102 | sys_theorem_status | — | all states |

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `aether-core` | local | Runtime theorem math for T1-T5 |
| `aether_verified` | local | no_std boot checks for T1-T10 |
| `x86_64` | 0.15 | IDT, port I/O, interrupts |
| `spin` | 0.9 | Spinlock mutexes (no_std) |
| `libm` | 0.2 | Math functions (sqrt, cos, sin, etc.) |
| `volatile` | 0.4 | (available, using core::ptr instead) |

## Future Work

1. **True userspace**: Per-process page tables, ring 3 execution, SYSCALL/SYSRET
2. **Allocator soak**: extend slab/page/topological allocator stress coverage and long-run leak checks
3. **USB HID**: xHCI host controller driver for USB keyboards/mice/gamepads
4. **Disk I/O**: ATA/AHCI driver for persistent ManifoldFS storage
5. **Network**: virtio-net for QEMU, rtl8139 for real hardware
6. **SMP**: Multi-core with per-CPU schedulers, Voronoi cells per core
7. **User-provided boot splash**: Embed actual photo as raw RGBA in initrd
