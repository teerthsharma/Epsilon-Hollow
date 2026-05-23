# Plan: Stable Machine Audit — Close All Gaps

## Objective
Seal OS must have zero functional gaps vs any other OS. If Linux/Windows/macOS/BSD can do it and we can't, we fix it. No edge given away.

## Audit Results

### ✅ Already Real (surprisingly solid)
- PS/2 keyboard + mouse (full init, IRQ handlers)
- ELF loader (`process/elf.rs`) + userspace tasks (`spawn_user`)
- KPTI + GDT ring-3 selectors + `iretq` userspace entry
- /proc + /sys pseudo-filesystems
- Audit logging to VFS
- Context switch with page table swap
- ASLR (`security::aslr`)
- APIC timer, async Sleep/Interval futures

### ❌ Critical Gaps (Tier 1 — other OS has massive edge)

| # | Gap | Impact | Fix Strategy |
|---|-----|--------|-------------|
| 1 | **SYS_EXEC = ENOSYS** | Can't run programs. `spawn_user` exists but never called from syscall. | Wire `SYS_EXEC` to VFS read → `elf::load` → `scheduler::spawn_user`. |
| 2 | **SYS_FORK = fake** | Returns `parent+1`. No real process duplication. | Duplicate current task: copy page table (COW later, full copy now), copy kernel/xsave stacks, new task ID, queue in scheduler. Return 0 to child, pid to parent. |
| 3 | **SYS_CHDIR / SYS_GETCWD / SYS_GETPPID missing** | Constants exist but fall through to ENOSYS. | Add match arms: chdir/getcwd use per-task cwd string; getppid walks scheduler slab. |
| 4 | **SYS_NANOSLEEP missing** | Apps can't sleep. | Add `SYS_NANOSLEEP` — compute target tick from `arg0` ms, block task until `ticks() >= target`. |
| 5 | **SYS_REBOOT missing** | Can't power off or reboot. | Add `SYS_REBOOT` — `arg0=0` → ACPI sleep, `arg0=1` → keyboard controller reset (0xFE to 0x64). |
| 6 | **SYS_LSEEK missing** | Can't seek in files. | Track `offset` in `FdEntry` (already has field but never updated). Update on read/write, implement lseek. |
| 7 | **SYS_UNLINK / SYS_RMDIR / SYS_RENAME missing** | Can't delete/rename files. | VFS has these — just wire syscalls. |
| 8 | **No entropy source** | TLS uses deterministic "random". Security hole. | Add `drivers/entropy.rs` — RDRAND + RdSeed fallback, expose `SYS_GETRANDOM`. |
| 9 | **No double buffering** | Screen tearing. | Add back buffer to `Framebuffer` — allocate shadow buffer, render to it, blit with `copy_nonoverlapping`. |
| 10 | **Panic only serial** | No screen panic message. | Extend panic handler to print to framebuffer if available. |

## Implementation Order

1. **Agent A — Syscall Fixes & Process** (highest impact)
   - Real SYS_FORK
   - Real SYS_EXEC (ELF from VFS)
   - Fix SYS_CHDIR, SYS_GETCWD, SYS_GETPPID
   - Add SYS_NANOSLEEP
   - Add SYS_REBOOT
   - Add SYS_LSEEK
   - Add SYS_UNLINK, SYS_RMDIR, SYS_RENAME

2. **Agent B — Hardware & Quality**
   - RDRAND entropy driver + SYS_GETRANDOM
   - Double buffering for framebuffer
   - Panic screen on framebuffer

## Files to Touch
- `kernel/seal-os/src/syscall/table.rs`
- `kernel/seal-os/src/process/scheduler.rs`
- `kernel/seal-os/src/process/task.rs`
- `kernel/seal-os/src/drivers/entropy.rs` (new)
- `kernel/seal-os/src/graphics/framebuffer.rs`
- `kernel/seal-os/src/main.rs` (panic handler)
- `kernel/seal-os/src/lib.rs` (init calls)
- `kernel/seal-os/src/drivers/mod.rs`
- `kernel/seal-os/src/fs/vfs.rs` (verify unlink/rmdir/rename exist)
