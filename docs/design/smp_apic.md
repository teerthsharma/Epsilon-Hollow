# SMP / Multi-Core APIC Support Design Document

> **Phase X Task**: CX-002
> **Priority**: HIGH (required for Ubuntu parity)
> **Estimated Effort**: 2 months

---

## Overview

Current kernel runs on a single CPU (BSP). This design adds support for multiple CPUs via xAPIC/x2APIC.

## Architecture

### Per-CPU Data (`gsbase` or `KGSBASE`)

```rust
#[repr(C)]
struct PerCpu {
    apic_id: u32,
    cpu_number: u32,
    current_task: *mut Task,
    scheduler: ManifoldScheduler,
    idle_context: TaskContext,
    kernel_stack: [u8; KERNEL_STACK_SIZE],
    tss: TaskStateSegment,
}
```

Access via `gs:[offset]` for fast, lock-free retrieval.

### Boot Sequence

1. **BSP** (Boot CPU):
   - Parse ACPI MADT to find all Local APIC IDs
   - Allocate `PerCpu` for each AP
   - Set up I/O APIC routing (timer -> all CPUs, device IRQs -> round-robin)

2. **APs** (Application CPUs):
   - Receive INIT IPI
   - Receive STARTUP IPI -> jump to `ap_trampoline` (real mode -> long mode)
   - Set up `gsbase` via `wrgsbase`
   - Load IDT (shared)
   - Enter scheduler loop

### Scheduling

- Each CPU has its own `ManifoldScheduler` instance
- Load balancing: every 100ms, CPU with >2x runnable tasks steals one task
- IPI for preemptive reschedule: `apic_send_ipi(target, RESCHEDULE_VECTOR)`

### Synchronization

- Spinlocks become ticket locks or MCS locks for fairness under contention
- RCU for read-mostly data (VFS mount table, device list)
- Seqlocks for ` jiffies` / `ticks`

### TLB Shootdown

- When `unmap_page()` called, broadcast IPI to all CPUs
- Each CPU flushes its TLB (`invlpg` or `invvpid`)

## Verification

- `test_smp`: Boot with `-smp 4`, verify 4 CPUs online
- `test_sched_balance`: Spawn 100 tasks, verify all CPUs make progress
- `test_rcu`: 1000 readers + 1 writer, no UAF

---

*Design document produced by Phase X Planning*
*2026-05-18*
