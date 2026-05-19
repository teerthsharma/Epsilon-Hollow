# Interrupt and Exception Handling — Design Document

> **Phase 2 Task**: INT-001
> **Priority**: HIGH (required for all device drivers, scheduling, and system calls)
> **Estimated Effort**: 3 weeks
> **Blocked by**: Nothing — foundational; blocked *on* physical allocator stability
> **Blocks**: AHCI interrupts, timer/scheduler, syscalls, SMP IPIs, keyboard input

---

## Overview

Current kernel has a stub IDT with a divide-by-zero handler. This design covers full interrupt architecture: IDT population, IRQ routing through APIC, exception handlers with frame unwinding, and interrupt-safe kernel abstractions.

---

## Architecture

### 1. IDT Layout (x86_64)

256 entries, divided into:

| Range | Purpose |
|---|---|
| 0–20 | CPU exceptions (Faults, Traps, Aborts) |
| 32–47 | Legacy ISA IRQs (remapped from 0x20) |
| 48–223 | Free for devices / MSI / dynamically assigned |
| 224–239 | Local APIC vectors (timer, thermal, error) |
| 240–255 | Reserved for IPIs (SMP), syscall (AMD64 `syscall`), spurious |

**IDT Entry (`GateDescriptor`):**
```rust
#[repr(C, packed)]
struct GateDescriptor {
    offset_low: u16,
    selector: u16,
    ist: u8,           // IST index (0 = none, 1..7 = TSS IST stacks)
    type_attr: u8,     // 0x8E = interrupt gate, 0x8F = trap gate
    offset_mid: u16,
    offset_high: u32,
    reserved: u32,
}
```

- [ ] Define `GateDescriptor` struct in `kernel/seal-os/src/interrupts/idt.rs`
- [ ] Implement `set_handler()` method on `GateDescriptor`
- [ ] Distinguish trap gates (debug, breakpoint) vs interrupt gates (all others)
- [ ] Define `IDT: [GateDescriptor; 256]` static array with proper alignment (16-byte)
- [ ] Implement `lidt()` wrapper loading `IDTR` with base + limit
- [ ] Define vector number constants for all 256 entries

### 2. Exception Handling

| Vector | Name | Type | Handler Action |
|---|---|---|---|
| 0 | Divide Error | Fault | Kill task; kernel panic if init |
| 1 | Debug | Trap | If kernel: ignore; if user: deliver SIGTRAP |
| 2 | NMI | Abort | Log and halt (unrecoverable hardware error) |
| 3 | Breakpoint | Trap | Deliver SIGTRAP to user task |
| 4 | Overflow | Trap | `#OF` — deliver SIGFPE |
| 5 | Bound Range | Fault | SIGFPE |
| 6 | Invalid Opcode | Fault | SIGILL |
| 7 | Device Not Available | Fault | Lazy FPU switch (set TS bit, trigger) |
| 8 | Double Fault | Abort | **IST1** stack; dump registers and halt |
| 9 | Coprocessor Segment | Abort | Reserved (legacy) |
| 10 | Invalid TSS | Fault | Kernel bug — panic |
| 11 | Segment Not Present | Fault | Kernel bug — panic |
| 12 | Stack-Segment Fault | Fault | SIGSEGV |
| 13 | General Protection | Fault | If user: SIGSEGV; if kernel: panic |
| 14 | Page Fault | Fault | **Resolve or kill** — see §3 |
| 16 | x87 FPU Error | Fault | SIGFPE |
| 17 | Alignment Check | Fault | SIGBUS |
| 18 | Machine Check | Abort | Log MCA banks; halt |
| 19 | SIMD FPE | Fault | SIGFPE |
| 20 | Virtualization | Fault | Ignore (hypervisor hint) |

- [x] Write assembly stubs for all 20 CPU exceptions (using x86_64 crate macros)
- [x] Common handler prologue: save all registers, build `InterruptFrame`
- [x] Dispatch to Rust handlers via vector number
- [x] `#DE` (0): kill current task or panic if no current task
- [x] `#DB` (1): log, continue if kernel; queue SIGTRAP if user
- [x] `#NMI` (2): log MCA, halt
- [x] `#BP` (3): queue SIGTRAP
- [x] `#UD` (6): queue SIGILL
- [x] `#NM` (7): lazy FPU context switch
- [x] `#DF` (8): IST1 handler, register dump, halt
- [x] `#GP` (13): panic if CPL=0, SIGSEGV if user
- [x] `#PF` (14): full page fault handler (see §3)
- [x] `#MC` (18): log, halt
- [x] Common handler epilogue: restore registers, `iretq`
- [x] Add `InterruptFrame` struct matching prologue layout

### 3. Page Fault Handler

```rust
fn handle_page_fault(addr: VirtAddr, error_code: PageFaultErrorCode) {
    if error_code.contains(CAUSE_WRITE) && cow_mapping_exists(addr) {
        cow_break(addr);                    // Copy-on-write
    } else if vma_contains(current_task.mm, addr) {
        if on_demand_zero_page(addr) {
            alloc_zero_page_and_map(addr);  // Lazy allocation
        } else if file_backed(addr) {
            read_page_from_file(addr);      // Demand paging
        }
    } else if is_guard_page(addr) {
        expand_stack(addr);                 // Stack growth
    } else {
        deliver_sigsegv_or_kill();          // Unrecoverable
    }
}
```

- [x] Extract fault address from `CR2`
- [x] Parse error code (P, W/R, U/S, RSVD, I/D, PK, SS)
- [ ] COW break handler
- [ ] Lazy zero-page allocation for anonymous mappings
- [ ] Demand paging for file-backed mappings
- [ ] Stack guard page detection + auto-growth
- [ ] Kernel-to-user fixup tables for `copy_to_user` / `copy_from_user`
- [x] Kernel bad pointer → panic (not silent corruption)
- [x] Double page fault detection (level > 1) → double fault

### 4. IRQ Routing (IO-APIC → Local APIC)

```
Device IRQ → IO-APIC pin → Redirection Entry → Local APIC of target CPU → IDT vector
```

- [x] Parse ACPI MADT to find IO-APIC base addresses
- [x] Parse MADT IRQ→GSI override mappings
- [x] Implement IO-APIC register read/write (`IOREGSEL` + `IOWIN`)
- [x] Program redirection table entries (vector, trigger, polarity, destination)
- [x] Remap PIC vectors from 0x08-0x0F to 0x20-0x2F (legacy compatibility)
- [x] Mask all IO-APIC entries at boot
- [x] Implement `irq_register_handler(vector, handler_fn)`
- [x] Implement `irq_unmask(vector)` / `irq_mask(vector)`
- [x] Implement `lapic_eoi()` (write 0 to EOI register)
- [x] Handle level-triggered IRQs: EOI to both LAPIC and IO-APIC

### 5. Interrupt Stack Table (IST)

TSS provides 7 IST stacks. We use:

| IST Index | Use Case | Size |
|---|---|---|
| IST1 | Double Fault | 1 page |
| IST2 | NMI | 1 page |
| IST3 | Machine Check | 1 page |
| IST4 | Debug / #DB | 1 page |
| IST5 | Reserved (future: stack guard) | — |
| IST6 | Reserved | — |
| IST7 | Reserved | — |

- [ ] Define `TaskStateSegment` struct with 7 IST entries
- [ ] Allocate IST stacks via `alloc_frame()` during CPU bringup
- [ ] Place guard pages between IST stacks (unmapped pages)
- [ ] Load TSS via `ltr` instruction
- [ ] Set IST index in `#DF`, `#NMI`, `#MC`, `#DB` IDT entries
- [ ] Verify IST switch via deliberate stack overflow test

### 6. Interrupt-Safe Abstractions

```rust
pub struct IrqGuard;       // RAII: CLI on construct, STI on drop
impl IrqGuard {
    pub fn disable() -> Self { cli(); Self }
}
impl Drop for IrqGuard { fn drop(&mut self) { sti(); } }
```

- [ ] Implement `IrqGuard` with `cli()` / `sti()` in `Drop`
- [ ] Implement `save_flags()` / `restore_flags()` wrappers
- [ ] Audit all existing `Mutex` usage for IRQ safety
- [ ] Convert spinlocks to save/restore IRQ state (not blind `sti`)
- [ ] Add `assert!(!interrupts_enabled())` in allocator hot paths
- [ ] Implement `pause()` / `rep; nop` hint in spin-wait loops

### 7. Deferred Work: Softirqs and Tasklets

Not all IRQ work runs at top half. Two-tier model:

| Tier | Latency | Use Case |
|---|---|---|
| Top half (IRQ context) | < 10 µs | ACK device, queue buffer, wake bottom half |
| Bottom half (softirq) | < 1 ms | AHCI completion processing, network RX batch |
| Tasklet / workqueue | process context | Heavy lifting: fsync, log flush, page reclaim |

- [ ] Define `Softirq` enum with 8 classes
- [ ] Implement per-CPU `softirq_pending` bitmask
- [ ] Implement `raise_softirq()` — atomic OR of pending bit
- [ ] Implement `do_softirq()` — process all pending softirqs
- [ ] Hook `do_softirq()` into `iret` path and syscall exit
- [ ] Implement softirq handler for block I/O completions
- [ ] Implement softirq handler for network RX
- [ ] Implement softirq handler for timer tick deferred work
- [ ] Define `Tasklet` / `Work` structs for process-context deferred work
- [ ] Implement workqueue (single-threaded for v1, per-CPU for v2)

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_idt_loaded` | IDTR points to valid IDT, all 256 entries present | [ ] |
| `test_divide_by_zero` | Exception 0 delivered, task killed cleanly, system survives | [ ] |
| `test_page_fault_user` | User-mode page fault resolved via COW; no kernel panic | [ ] |
| `test_page_fault_kernel_panic` | Kernel bad pointer → panic (not silent corruption) | [ ] |
| `test_irq_timer_fires` | Local APIC timer IRQ fires at expected rate (1 kHz) | [ ] |
| `test_irq_ahci_completes` | AHCI DMA completion interrupts fire; no polling needed | [ ] |
| `test_irq_disable_latency` | `save_irq_disable()` + `restore_irq()` round-trip < 50 ns | [ ] |
| `test_ist_double_fault` | Deliberate stack overflow triggers #DF, IST1 catches it, halts cleanly | [ ] |

---

*Design document produced by Phase 2 Planning*
*2026-05-19*
Planning*
*2026-05-19*
