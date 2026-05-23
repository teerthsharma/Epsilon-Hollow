# Master Plan: Complete All Complex Tasks

## Objective
Zero remaining functional gaps. No OS on Earth gets an edge on Seal OS.

## Architecture
Dispatch 6 independent implementation agents in parallel. Each agent owns a subsystem.
**No agent touches `syscall/table.rs` dispatch match block** — they only add constants and expose helper functions. Master (Rick) wires syscalls after all agents return.

## Agent Assignments

### Agent 1: Signals
**Files**: `process/signal.rs` (new), `process/scheduler.rs` (minimal hooks)
**Scope**:
- Signal definitions: SIGKILL(9), SIGSEGV(11), SIGINT(2), SIGTERM(15), SIGUSR1(10), SIGUSR2(12)
- Per-task signal state: `pending: u64` bitmap, `mask: u64` bitmap, `handlers: [u64; 32]` (function pointers)
- `signal::deliver_signal(target_id, sig)` — sets pending bit, if unmasked and target is running, force reschedule
- `signal::check_and_deliver()` — called before returning to userspace; if pending & ~mask != 0, build signal frame on user stack
- Signal frame: push `UserContext`, push sig number, push trampoline addr, set RIP = handler
- `sys_kill(target, sig)` — `dispatch_kill(target_id, sig)` helper
- `sys_sigaction(sig, act, oldact)` — set/retrieve handler
- `sys_sigreturn()` — restore `UserContext` from stack

### Agent 2: Pipes + brk + dup
**Files**: `fs/pipe.rs` (new), `memory/brk.rs` (new)
**Scope**:
- `PipeFs` — in-memory ring buffer per pipe (64KB), FileSystem trait impl
- `sys_pipe(fds)` — creates pipe, returns two fds (read=0, write=1) via userspace array
- `sys_dup(old_fd)` — copies fd entry to new fd
- `sys_dup2(old_fd, new_fd)` — copies fd entry to specific fd (closes existing)
- `sys_brk(new_addr)` — grows/shrinks user heap. Track `brk_end` per task. If `new_addr > brk_end`, mmap pages. If `new_addr < brk_end`, unmap pages. Return new break.

### Agent 3: RTC + Watchdog
**Files**: `drivers/rtc.rs` (new), `drivers/watchdog.rs` (new)
**Scope**:
- CMOS RTC (ports 0x70/0x71): read seconds, minutes, hours, day, month, year from CMOS registers
- Convert BCD to binary. Handle century register (0x32).
- `sys_gettimeofday(buf)` — fills `struct timeval { sec, usec }` from RTC + tick interpolation
- `sys_settimeofday(sec, usec)` — writes to CMOS (if allowed)
- Watchdog: APIC timer-based. `watchdog_init(period_ms)`. On each tick, check if scheduler made progress. If stuck for > period, trigger `int 3` (break into debugger) or reset.
- `sys_watchdog(pet)` — userspace can pet the watchdog. If not petted within timeout, system resets.

### Agent 4: USB Mass Storage (Real)
**Files**: `drivers/usb/mass_storage.rs` (rewrite)
**Scope**:
- SCSI Command Block Wrapper (CBW) + Command Status Wrapper (CSW)
- SCSI commands: INQUIRY, READ CAPACITY(10), READ(10), WRITE(10)
- Bulk transfer via xHCI: find bulk IN/OUT endpoints on mass storage interface, setup bulk transfer rings, send CBW, receive data, receive CSW
- Integrate with `drivers/block/mod.rs` — implement `BlockDevice` trait
- Register as block device so ext2/FAT can mount it

### Agent 5: FAT Filesystem
**Files**: `fs/fat.rs` (new)
**Scope**:
- FAT12/16/32 parser. Read boot sector (BPB), determine FAT type by cluster count.
- FileSystem trait impl: `lookup`, `read`, `readdir`, `stat`
- Read-only for v1 (write/create can be stubs)
- Mount via VFS: `vfs.mount("/usb", fat_fs)`
- Support LFN (Long File Names) via directory entry sequences

### Agent 6: Ext2 Write + ioctl
**Files**: `fs/ext2.rs` (extend), `drivers/ioctl.rs` (new)
**Scope**:
- Implement `write` — direct block pointers (i_block[0..11]), single indirect (i_block[12]), double indirect (i_block[13])
- Implement `create` — allocate new inode, initialize, add directory entry
- Implement `mkdir` — allocate inode with directory type, create `.` and `..` entries
- Implement `rmdir` — check empty, remove directory entries, free inode
- Implement `rename` — same-directory: update directory entry name
- `drivers/ioctl.rs` — generic ioctl dispatcher. Device-specific ioctls registered by major/minor number.
- `sys_ioctl(fd, request, arg)` — looks up fd, dispatches to device ioctl or returns ENOTTY

## Syscall Wiring (Master Rick does this after agents return)
Add to `syscall/table.rs` constants:
```rust
SYS_KILL = 25
SYS_SIGACTION = 26
SYS_SIGRETURN = 27
SYS_PIPE = 28
SYS_DUP = 29
SYS_DUP2 = 30
SYS_BRK = 31
SYS_GETTIMEOFDAY = 32
SYS_SETTIMEOFDAY = 33
SYS_WATCHDOG = 34
SYS_IOCTL = 35
```

Add match arms calling each agent's exposed helper.
