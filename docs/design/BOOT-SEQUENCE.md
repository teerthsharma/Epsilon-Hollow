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

- [ ] Save `SystemTable` and `ImageHandle` in `.bss` globals
- [ ] Verify UEFI runtime services pointer is valid
- [ ] Call `GetMemoryMap()` to learn available RAM
- [ ] Record memory map in `E820_ENTRIES` array
- [ ] Save framebuffer info via GOP (`FrameBufferBase`, dimensions, stride)
- [ ] Optionally call `SetVirtualAddressMap()` for runtime services in virtual mode
- [ ] Call `ExitBootServices(ImageHandle, MapKey)`
- [ ] Verify boot services are gone (any subsequent UEFI boot call must panic)
- [ ] Preserve runtime services (ResetSystem, GetTime) for shutdown

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
- [ ] Define `FramebufferInfo` struct
- [ ] Store framebuffer base identity-mapped for early printk
- [ ] Implement `early_putpixel()` for visual boot progress

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

- [ ] Define GDT with 6 entries
- [ ] Load GDT via `lgdt`
- [ ] Reload segment registers (CS, DS, ES, FS, GS, SS)
- [ ] Verify we are in long mode with correct segments

**IDT:**
- [ ] Allocate IDT array `[GateDescriptor; 256]`
- [ ] Set up divide-by-zero handler before any division
- [ ] Set up GPF handler before any memory operation that could fault
- [ ] Set up page fault handler before heap init
- [ ] Load IDT via `lidt`
- [ ] Verify IDT is active via `int 0x03` self-test

**Memory Map Parsing:**
```rust
#[repr(C)]
struct E820Entry {
    base: u64,
    len: u64,
    typ: E820Type,  // Usable, Reserved, ACPI reclaim, ACPI NVS, Bad
}
```

- [ ] Convert UEFI memory map to `E820Entry` array
- [ ] Mark `E820Type::Usable` regions for allocator
- [ ] Skip first 1 MiB (real mode, SMP trampoline, EBDA)
- [ ] Skip region below kernel image (protect loaded code)
- [ ] Skip any region marked as ACPI NVS or Reserved
- [ ] Calculate total usable RAM
- [ ] Log memory map via serial for debugging

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

Higher half base: `0xffff_8000_0000_0000` (standard Linux kernel mapping).

- [ ] Build P4 page table (one entry for identity map, one for higher half)
- [ ] Build P3 page table for identity map (0-4 GiB)
- [ ] Build P3 page table for higher half
- [ ] Build P2 tables with 2 MiB huge pages for kernel text/data
- [ ] Identity-map low 4 GiB (required for DMA, ACPI, UEFI runtime)
- [ ] Map kernel image to higher half
- [ ] Set `CR3` to P4 physical address
- [ ] Enable `CR4.PAE`, `EFER.LME`, `CR0.PG`
- [ ] Jump to higher half code
- [ ] Verify `rip` is now in `0xffff8000_00000000+` range
- [ ] Implement `phys_to_virt()` and `virt_to_phys()` helpers

**Bootstrap allocator:**
- [ ] Implement simple bump allocator from top of first usable E820 region
- [ ] Bump allocator only allocates full 4 KiB pages
- [ ] Bump allocator discarded once bitmap allocator is initialized
- [ ] Track highest allocated address for debugging

**Bitmap allocator init:**
- [ ] Compute total frames = max_usable_address / 4096, capped at 1,048,576
- [ ] Allocate bitmap array `[u64; 65536]` = 512 KiB via bump allocator
- [ ] Mark all frames as used initially
- [ ] Iterate E820 usable regions, mark frames as free
- [ ] Skip frame 0 (null guard)
- [ ] Skip our own image frames
- [ ] Skip 0-1 MiB frames
- [ ] Initialize `FREE_COUNT` to actual free frames
- [ ] Log free frame count at boot

**Heap init:**
- [ ] Allocate single 4 KiB frame for slab allocator's initial caches
- [ ] Set up `SealAllocator` as `#[global_allocator]`
- [ ] Test: `alloc_frame()` returns `Some`
- [ ] Test: `Box::new(42)` succeeds
- [ ] Test: `Vec::with_capacity(100)` succeeds
- [ ] Verify slab caches are populated (64, 128, 256, 512, 1024, 2048)

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
- [ ] `interrupts::init_idt()` succeeds before any subsystem that could trigger exception
- [ ] `memory::heap::init()` succeeds before any subsystem using `Vec`/`Box`/`String`
- [ ] `cpu::smp::smp_init()` can use `alloc_frame()` but not `Box` (AP boot code identity-mapped)
- [ ] All init functions log their start + completion via serial
- [ ] Any init failure causes kernel panic with descriptive message

Per-subsystem checklist:
- [ ] `interrupts::init_idt()` — all 256 entries valid
- [ ] `memory::heap::init()` — slab caches online
- [ ] `cpu::smp::smp_init()` — APs respond to INIT-SIPI-SIPI
- [ ] `acpi::init()` — MADT, FADT parsed successfully
- [ ] `security::init_security()` — MAC policy loaded
- [ ] `interrupts::init_irq_routing()` — IO-APIC + Local APIC timer configured
- [ ] `apic::init_local_apic_timer_for_bsp()` — BSP timer firing at 1 kHz
- [ ] `syscall::init_syscall_msrs()` — STAR, LSTAR, SFMASK programmed
- [ ] `drivers::pci::scan_bus()` — PCI devices enumerated
- [ ] `drivers::block::ahci::init()` — AHCI controller found and initialized (if present)
- [ ] `fs::manifold::init()` — root filesystem mounted
- [ ] `scheduler::init()` — idle task created, ready to schedule

### 6. Stage 5: Init Process

After all kernel init is done:
```rust
fn stage5_userspace() {
    let init = process::create("/bin/init");
    init.execve("/bin/init", &["/bin/init"], &[]);
    scheduler::run();  // Never returns
}
```

- [ ] Search for `/bin/init` in root filesystem
- [ ] If not found, search for `/bin/sh`
- [ ] If neither found, launch embedded emergency shell
- [ ] Create first process (PID 1) with kernel privileges dropped
- [ ] `execve` loads ELF binary into process address space
- [ ] Set up standard file descriptors (0, 1, 2) pointing to `/dev/console`
- [ ] Enter scheduler — never returns to kernel_main
- [ ] Implement emergency shell in `.rodata` as fallback
- [ ] Emergency shell supports: `ls`, `cat`, `echo`, `reboot`, `help`

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_memory_map_parsed` | At least one E820 usable region found, total RAM > 16 MiB | [ ] |
| `test_identity_map` | Access to `0x0` through `0xb8000` (VGA) works without fault | [ ] |
| `test_higher_half` | Kernel symbol addresses are in `0xffff8000_00000000+` range | [ ] |
| `test_bitmap_allocator` | `alloc_frame()` / `free_frame()` round-trip; free count correct | [ ] |
| `test_heap_box` | `Box::new([0u8; 1024])` succeeds and is writeable | [ ] |
| `test_idt_before_heap` | Deliberate `div 0` before heap init is caught, system survives | [ ] |
| `test_uefi_framebuffer` | Framebuffer base is valid, can write pixel without fault | [ ] |
| `test_init_exists` | `/bin/init` or `/bin/sh` is found in initrd / rootfs | [ ] |
| `test_subsystem_init_order` | All 12 stage-4 inits complete in correct order | [ ] |
| `test_ap_bringup` | At least one AP reports online after `smp_init()` | [ ] |

---

*Design document produced by Phase 2 Planning*
*2026-05-19*
