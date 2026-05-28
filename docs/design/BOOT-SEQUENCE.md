# Boot Sequence — Design Document

> **Phase 2 Task**: BOOT-001
> **Priority**: HIGH (required for physical hardware and QEMU parity)
> **Estimated Effort**: 2 weeks
> **Blocked by**: Nothing; blocked *on* allocator stability
> **Blocks**: All kernel initialization, test harness

---

## Overview

Current boot: QEMU loads `seal-os.efi` via UEFI → `kernel_main()` is called. This design documents the full boot sequence from power-on to `init`, covers the linker script, page table setup, physical allocator bootstrap, and the transition to a proper multi-stage boot.

---

## Architecture

### 1. Boot Stages

```
Power On → UEFI Firmware → Boot Manager → seal-os.efi
                                    ↓
                           ┌────────────────┐
                           │  Stage 1: EFI  │  (UEFI runtime services available)
                           │  - Locate FB   │
                           │  - Map kernel  │
                           │  - ExitBootSvcs│
                           └───────┬────────┘
                                   ↓
                           ┌────────────────┐
                           │  Stage 2: Long │  (no UEFI; full control)
                           │  - Set up GDT  │
                           │  - Set up IDT  │
                           │  - Parse memmap│
                           └───────┬────────┘
                                   ↓
                           ┌────────────────┐
                           │  Stage 3: MM   │  (virtual memory active)
                           │  - Identity map│
                           │  - Heap init   │
                           │  - Frame alloc │
                           └───────┬────────┘
                                   ↓
                           ┌────────────────┐
                           │  Stage 4: Init │  (drivers, FS, scheduler)
                           │  - PCI scan    │
                           │  - AHCI init   │
                           │  - ManifoldFS  │
                           └───────┬────────┘
                                   ↓
                           ┌────────────────┐
                           │  Stage 5: User │  (first process / shell)
                           │  - fork init   │
                           │  - exec /bin/sh│
                           └────────────────┘
```

### 2. Stage 1: EFI Pre-Exit

**UEFI Handoff State:**
- `SystemTable` pointer in `rdx` (Microsoft calling convention)
- `ImageHandle` in `rcx`
- Stack provided by firmware (~128 KiB)
- GDT, IDT, page tables set by firmware (likely 4-level, NX off)
### Stage 1: EFI Pre-Exit (uefi_entry.rs)

- [x] Save `SystemTable` and `ImageHandle` in `.bss` globals
- [x] Verify UEFI runtime services pointer is valid
- [x] Call `GetMemoryMap()` to learn available RAM
- [x] Record memory map in `E820_ENTRIES` array
- [x] Save framebuffer info via GOP (`FrameBufferBase`, dimensions, stride)
- [x] Optionally call `SetVirtualAddressMap()` for runtime services in virtual mode
- [x] Call `ExitBootServices(ImageHandle, MapKey)`
- [x] Verify boot services are gone (any subsequent UEFI boot call must panic)
- [x] Preserve runtime services (ResetSystem, GetTime) for shutdown
**Framebuffer:**
```rust
struct FramebufferInfo {
    base: PhysAddr,
    size: usize,
    width: u32,
    height: u32,
    stride: u32,       // pixels per row
    bpp: u32,          // bytes per pixel (usually 4)
}
```
- [x] Define `FramebufferInfo` struct
- [x] Store framebuffer base identity-mapped for early printk
- [x] Implement `early_putpixel()` for visual boot progress

### 3. Stage 2: Long Mode Setup

After ExitBootServices, we own the machine.

**GDT:**
```
0x00: Null
0x08: Kernel Code (64-bit, execute/read, ring 0)
0x10: Kernel Data (read/write, ring 0)
0x18: User Code (64-bit, execute/read, ring 3)
0x20: User Data (read/write, ring 3)
0x28: TSS (64-bit system segment)
```

- [x] Define GDT with 6 entries
- [x] Load GDT via `lgdt`
- [x] Reload segment registers (CS, DS, ES, FS, GS, SS)
- [x] Verify we are in long mode with correct segments

**IDT:**
- [x] Allocate IDT array `[GateDescriptor; 256]`
- [x] Set up divide-by-zero handler before any division
- [x] Set up GPF handler before any memory operation that could fault
- [x] Set up page fault handler before heap init
- [x] Load IDT via `lidt`
- [x] Verify IDT is active via `int 0x03` self-test

**Memory Map Parsing:**
```rust
#[repr(C)]
struct E820Entry {
    base: u64,
    len: u64,
    typ: E820Type,  // Usable, Reserved, ACPI reclaim, ACPI NVS, Bad
}
```

- [x] Convert UEFI memory map to `E820Entry` array
- [x] Mark `E820Type::Usable` regions for allocator
- [x] Skip first 1 MiB (real mode, SMP trampoline, EBDA)
- [x] Skip region below kernel image (protect loaded code)
- [x] Skip any region marked as ACPI NVS or Reserved
- [x] Calculate total usable RAM
- [x] Log memory map via serial for debugging

### 4. Stage 3: Virtual Memory Bootstrap

**Bootstrap Page Tables (4-level, no 5-level for now):**

```
P4 (CR3)          P3                  P2                  P1
┌─────────┐      ┌─────────┐        ┌─────────┐        ┌─────────┐
│ 0..511  │─────→│ 0 (id)  │───────→│ 0..511  │───────→│ 0..511  │  Identity: 0..1 GiB
│         │      │ 510(hh) │────┐   │         │        │         │
└─────────┘      └─────────┘    │   └─────────┘        └─────────┘
                                │
                                └──────────────────────→ P2 for kernel
                                                        (2 MiB huge pages)
```

Higher half base: `0xffff_8000_0000_0000` (common high-half kernel mapping, not an inherited ABI).

- [x] Build P4 page table (one entry for identity map, one for higher half)
- [x] Build P3 page table for identity map (0-4 GiB)
- [x] Build P3 page table for higher half
- [x] Build P2 tables with 2 MiB huge pages for kernel text/data
- [x] Identity-map low 4 GiB (required for DMA, ACPI, UEFI runtime)
- [x] Map kernel image to higher half
- [x] Set `CR3` to P4 physical address
- [x] Enable `CR4.PAE`, `EFER.LME`, `CR0.PG`
- [x] Jump to higher half code
- [x] Verify `rip` is now in `0xffff8000_00000000+` range
- [x] Implement `phys_to_virt()` and `virt_to_phys()` helpers

**Bootstrap allocator:**
- [x] Implement simple bump allocator from top of first usable E820 region
- [x] Bump allocator only allocates full 4 KiB pages
- [x] Bump allocator discarded once bitmap allocator is initialized
- [x] Track highest allocated address for debugging

**Bitmap allocator init:**
- [x] Compute total frames = max_usable_address / 4096, capped at 1,048,576
- [x] Allocate bitmap array `[u64; 65536]` = 512 KiB via bump allocator
- [x] Mark all frames as used initially
- [x] Iterate E820 usable regions, mark frames as free
- [x] Skip frame 0 (null guard)
- [x] Skip our own image frames
- [x] Skip 0-1 MiB frames
- [x] Initialize `FREE_COUNT` to actual free frames
- [x] Log free frame count at boot

**Heap init:**
- [x] Allocate single 4 KiB frame for slab allocator's initial caches
- [x] Set up `SealAllocator` as `#[global_allocator]`
- [x] Test: `alloc_frame()` returns `Some`
- [x] Test: `Box::new(42)` succeeds
- [x] Test: `Vec::with_capacity(100)` succeeds
- [x] Verify slab caches are populated (64, 128, 256, 512, 1024, 2048)

### 5. Stage 4: Subsystem Initialization Order

```rust
fn stage4_init() {
    interrupts::init_idt();          // Must be first
    memory::heap::init();            // Global allocator online
    cpu::smp::smp_init();            // Bring up APs (needs heap for PerCpu)
    acpi::init();                    // Parse tables, find LAPIC/IOAPIC
    security::init_security();       // MAC policy, audit log buffer
    interrupts::init_irq_routing();  // IO-APIC + Local APIC timer
    apic::init_local_apic_timer_for_bsp();
    syscall::init_syscall_msrs();    // LSTAR, STAR, SFMASK
    drivers::pci::scan_bus();        // Enumerate PCI devices
    drivers::block::ahci::init();    // If SATA controller found
    fs::manifold::init();            // Mount root filesystem
    scheduler::init();               // Idle task, timer tick
}
```

**Invariants:**
- [x] `interrupts::init_idt()` succeeds before any subsystem that could trigger exception
- [x] `memory::heap::init()` succeeds before any subsystem using `Vec`/`Box`/`String`
- [x] `cpu::smp::smp_init()` can use `alloc_frame()` but not `Box` (AP boot code identity-mapped)
- [x] All init functions log their start + completion via serial
- [x] Any init failure causes kernel panic with descriptive message

Per-subsystem checklist:
- [x] `interrupts::init_idt()` — all 256 entries valid
- [x] `memory::heap::init()` — slab caches online
- [x] `cpu::smp::smp_init()` — APs respond to INIT-SIPI-SIPI
- [x] `acpi::init()` — MADT, FADT parsed successfully
- [x] `security::init_security()` — MAC policy loaded
- [x] `interrupts::init_irq_routing()` — IO-APIC + Local APIC timer configured
- [x] `apic::init_local_apic_timer_for_bsp()` — BSP timer firing at 1 kHz
- [x] `syscall::init_syscall_msrs()` — STAR, LSTAR, SFMASK programmed
- [x] `drivers::pci::scan_bus()` — PCI devices enumerated
- [x] `drivers::block::ahci::init()` — AHCI controller found and initialized (if present)
- [x] `fs::manifold::init()` — root filesystem mounted
- [x] `scheduler::init()` — idle task created, ready to schedule

### 6. Stage 5: Init Process

After all kernel init is done:
```rust
fn stage5_userspace() {
    let init = process::create("/bin/init");
    init.execve("/bin/init", &["/bin/init"], &[]);
    scheduler::run();  // Never returns
}
```

- [x] Search for `/bin/init` in root filesystem
- [x] If not found, search for `/bin/sh`
- [x] If neither found, launch embedded emergency shell
- [x] Create first process (PID 1) with kernel privileges dropped
- [x] `execve` loads ELF binary into process address space
- [x] Set up standard file descriptors (0, 1, 2) pointing to `/dev/console`
- [x] Enter scheduler — never returns to kernel_main
- [x] Implement emergency shell in `.rodata` as fallback
- [x] Emergency shell supports: `ls`, `cat`, `echo`, `reboot`, `help`

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_memory_map_parsed` | At least one E820 usable region found, total RAM > 16 MiB | [x] |
| `test_identity_map` | Access to `0x0` through `0xb8000` (VGA) works without fault | [x] |
| `test_higher_half` | Kernel symbol addresses are in `0xffff8000_00000000+` range | [x] |
| `test_bitmap_allocator` | `alloc_frame()` / `free_frame()` round-trip; free count correct | [x] |
| `test_heap_box` | `Box::new([0u8; 1024])` succeeds and is writeable | [x] |
| `test_idt_before_heap` | Deliberate `div 0` before heap init is caught, system survives | [x] |
| `test_uefi_framebuffer` | Framebuffer base is valid, can write pixel without fault | [x] |
| `test_init_exists` | `/bin/init` or `/bin/sh` is found in initrd / rootfs | [x] |
| `test_subsystem_init_order` | All 12 stage-4 inits complete in correct order | [x] |
| `test_ap_bringup` | At least one AP reports online after `smp_init()` | [x] |

---

*Design document produced by Phase 2 Planning*
*2026-05-19*
