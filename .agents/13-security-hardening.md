# 13 — Security Hardening: Memory Protection, Syscall Validation, Capabilities

## Goal

Add memory protection, syscall input validation, and a capability-based security model — so that a misbehaving application cannot crash the kernel or access unauthorized resources.

## Current State

- **No memory protection** — all code runs in ring 0 (kernel mode), no page table isolation
- **No syscall validation** — `syscall/table.rs` dispatches by number but doesn't validate arguments
- **No capability model** — any process can call any syscall
- **No ASLR** — heap and framebuffer at fixed addresses
- **Flat memory** — bump allocator, no page-level permissions

Syscall table (`kernel/seal-os/src/syscall/table.rs`):
- 11 POSIX-like syscalls (0-10)
- 12 Epsilon extensions (100-111)
- `dispatch()` matches syscall number and calls handler — no bounds checking on arguments

## Implementation Steps

### Phase A: Syscall Validation

1. **Validate all syscall arguments**
   - `SYS_WRITE`: verify buffer pointer is within process address space, length is reasonable
   - `SYS_READ`: verify buffer pointer is writable, length bounded
   - `SYS_OPEN`: verify filename string is null-terminated, no path traversal
   - `SYS_TELEPORT`: verify source and destination inode IDs exist
   - `SYS_MANIFOLD_QUERY`: verify query index is in bounds
   - All string arguments: bounded length (max 256 bytes), valid UTF-8

2. **Return typed errors instead of panicking**
   - `SyscallResult` already has `code` field — use negative codes for errors
   - Define error codes: `-EINVAL`, `-ENOENT`, `-EPERM`, `-ENOMEM`, `-EACCES`

3. **Rate limiting**
   - Track syscall frequency per process
   - If a process issues >10,000 syscalls/second, throttle it

### Phase B: Memory Protection

4. **Set up page tables**
   - Use x86_64 4-level paging (PML4 → PDP → PD → PT)
   - Map kernel at higher half (0xFFFF_8000_0000_0000+)
   - Map heap, framebuffer, MMIO regions with appropriate flags
   - Kernel pages: Present + Writable + No-Execute-for-user
   - Framebuffer: Present + Writable + Write-Through (for MMIO)

5. **Stack guard pages**
   - Allocate guard page (not Present) below each stack
   - Stack overflow → page fault → kernel handles gracefully instead of silent corruption

6. **NX bit**
   - Mark data pages as No-Execute
   - Mark code pages as Execute + Read-Only
   - Heap: No-Execute (prevent shellcode execution)

### Phase C: Capability Model

7. **Define capabilities**
   ```rust
   pub enum Capability {
       FileRead,
       FileWrite,
       FileCreate,
       FileTeleport,
       NetworkAccess,
       DeviceAccess(DeviceId),
       SyscallExtended,  // Epsilon-specific syscalls
       SchedulerAdmin,   // modify scheduler params
   }
   ```

8. **Attach capabilities to processes**
   - Each process has a `CapabilitySet` (bitfield)
   - Kernel process: all capabilities
   - User apps: restricted set (FileRead + FileWrite + FileCreate)
   - Teleport requires `FileTeleport` capability
   - Network commands require `NetworkAccess`

9. **Enforce capabilities in syscall dispatch**
   - Before executing any syscall, check process capabilities
   - Missing capability → return `-EPERM`

### Phase D: Future (Out of Scope)

- User-mode processes (ring 3): requires TSS, syscall/sysret, context switching
- Process isolation via separate page tables
- Full ASLR

## Dependencies

- **01-kernel-safety** (no UB in kernel itself first)
- **02-memory-allocator** (page table allocation needs real dealloc)

## Acceptance Criteria

- [ ] Invalid syscall arguments return error, not panic
- [ ] Negative syscall numbers return `-EINVAL`
- [ ] Page tables are set up (kernel mapped at higher half)
- [ ] Stack guard page triggers page fault handler (not silent corruption)
- [ ] NX bit enforced on heap pages
- [ ] Capability check on every syscall dispatch
- [ ] Uncapable process cannot teleport files
- [ ] `cargo +nightly build --release` passes
- [ ] QEMU smoke test passes (existing functionality preserved)

## Files to Modify

- `kernel/seal-os/src/syscall/table.rs` (validation + capability checks)
- `kernel/seal-os/src/process/scheduler.rs` (add CapabilitySet to tasks)
- `kernel/seal-os/src/memory/mod.rs` (page table setup)

## Files to Create

- `kernel/seal-os/src/security/capabilities.rs`
- `kernel/seal-os/src/security/mod.rs`
- `kernel/seal-os/src/memory/paging.rs` (page table management)
