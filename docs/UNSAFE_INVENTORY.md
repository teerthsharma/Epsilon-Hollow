# Seal OS Unsafe Inventory

Seal OS is a bare-metal kernel. Unsafe Rust is expected. Unexplained unsafe Rust
is not.

## Policy

Every touched unsafe site needs a nearby `SAFETY:` comment that states the
invariant. Public `unsafe fn` entries need a `# Safety` doc comment when they
are callable across module boundaries.

Use:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --unsafe-inventory .
```

Default mode prints inventory. Strict mode fails on changed unsafe sites without
nearby `SAFETY:` comments.

## Subsystem Buckets

| Bucket | Typical files | Main invariant |
| --- | --- | --- |
| Boot and UEFI handoff | `boot/`, `main.rs`, `lib.rs` | firmware pointers remain valid until copied or boot services exit |
| Framebuffer and MMIO | `graphics/`, `drivers/`, `lang/mod.rs` | mapped device memory is valid, aligned, and owned |
| Page tables and CR3 | `memory/`, `security/kpti.rs` | page tables are present, aligned, and not aliased mutably |
| Interrupts and APIC | `drivers/interrupts.rs`, `drivers/apic.rs` | handlers preserve CPU state and do not block |
| Syscall entry/exit | `process/userspace.rs`, `syscall/` | ring transition state and user stack are valid |
| User copy | `security/smap_smep.rs` | user spans are mapped, user-accessible, and writable when needed |
| Process context switch | `process/scheduler.rs`, `process/task.rs` | saved contexts and stacks outlive switches |
| Drivers | `drivers/` | PCI/MMIO/DMA resources are owned by the driver |

## Closure Target

The first inventory target is not zero unsafe. The target is known unsafe with
documented invariants, then stricter CI for new unsafe in changed files.
