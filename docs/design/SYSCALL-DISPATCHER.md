# System Call Dispatcher — Design Document

> **Phase 2 Task**: SC-001
> **Priority**: HIGH (required for any userspace)
> **Estimated Effort**: 2 weeks
> **Blocked by**: IDT/IRQ, interrupt-safe locks
> **Blocks**: Shell, libc, userspace binaries, IPC

---

## Overview

Current kernel has `syscall::init_syscall_msrs()` which sets `STAR`, `LSTAR`, `SFMASK`, and `EFER.SCE`. There is no actual dispatch table. This design covers the full syscall path: entry/exit ABI, argument marshalling, dispatcher table, and per-syscall security hooks.

---

## Architecture

### 1. Entry ABI (AMD64 `syscall` / `sysret`)

**User → Kernel:**
| Register | Purpose |
|---|---|
| `rax` | Syscall number |
| `rdi` | Arg 0 |
| `rsi` | Arg 1 |
| `rdx` | Arg 2 |
| `r10` | Arg 3 (replaces rcx, which syscall overwrites) |
| `r8` | Arg 4 |
| `r9` | Arg 5 |

**Kernel → User (return):**
| Register | Purpose |
|---|---|
| `rax` | Return value (negative = errno) |
| `rcx`, `r11` | Restored by `sysret` (do not clobber) |

**MSR state at boot:**
```rust
// STAR[63:32] = kernel CS (0x08), user CS (0x1B) — SYSCALL clears IF, SYSRET sets it
// LSTAR = address of syscall_entry_asm
// SFMASK = EFLAGS.IF | EFLAGS.DF | EFLAGS.TF | EFLAGS.AC | EFLAGSNT
```

- [x] Verify `EFER.SCE` is set during early boot
- [x] Program `STAR` MSR with kernel CS = 0x08, user CS = 0x1B
- [x] Program `LSTAR` MSR to `syscall_entry_asm` address
- [x] Program `SFMASK` MSR with `IF | DF | TF | AC | NT`
- [x] Document the `syscall` / `sysret` ABI in a header comment
- [x] Define `SYSCALL_MAX_ARGS = 6`
- [x] Define `SYSCALL_MAX_NR = 512`

### 2. Assembly Entry / Exit

```nasm
syscall_entry_asm:
    ; rcx = userspace rip (saved by CPU), r11 = userspace rflags
    swapgs                          ; gsbase = PerCpu pointer
    mov gs:[PERCPU_USER_RSP], rsp   ; save user stack
    mov rsp, gs:[PERCPU_KERNEL_RSP] ; load kernel stack
    push r11                        ; save user rflags
    push rcx                        ; save user rip
    push rax                        ; save syscall# for later
    
    ; Build SyscallFrame on kernel stack
    push_all_callee_saved
    mov rdi, rsp                    ; frame pointer -> first arg to rust handler
    call syscall_dispatch_rust
    
    ; rax = return value
    pop_all_callee_saved
    pop rax                         ; restore syscall# (ignored, but balance stack)
    pop rcx                         ; user rip
    pop r11                         ; user rflags
    mov rsp, gs:[PERCPU_USER_RSP]   ; restore user stack
    swapgs
    sysretq
```

- [x] Write `syscall_entry_asm` in `kernel/seal-os/src/syscall/entry.asm` or inline asm
- [x] Implement `swapgs` on entry with SWAPGS mitigation sequence
- [x] Save all caller-saved registers (rcx, r11 overwritten by syscall)
- [x] Save user RSP, load kernel RSP from `gs:[PERCPU_KERNEL_RSP]`
- [x] Build `SyscallFrame` on kernel stack with all GPRs
- [x] Call `syscall_dispatch_rust(frame: *mut SyscallFrame)`
- [x] On exit: restore user RSP, RIP, RFLAGS
- [x] `sysretq` with verified correct privilege level
- [x] Add `swapgs` mismatch detection (NMI during entry scenario)

### 3. Rust Dispatch

```rust
#[repr(C)]
pub struct SyscallFrame {
    pub rax: u64, pub rbx: u64, pub rcx: u64, pub rdx: u64,
    pub rsi: u64, pub rdi: u64, pub rbp: u64,
    pub r8: u64, pub r9: u64, pub r10: u64, pub r11: u64,
    pub r12: u64, pub r13: u64, pub r14: u64, pub r15: u64,
    pub rip: u64, pub cs: u64, pub rflags: u64, pub rsp: u64, pub ss: u64,
}

static SYSCALL_TABLE: [Option<SyscallFn>; 512] = { /* ... */ };

type SyscallFn = fn(&mut SyscallFrame) -> i64;

fn syscall_dispatch_rust(frame: &mut SyscallFrame) {
    let nr = frame.rax as usize;
    if nr >= SYSCALL_TABLE.len() || SYSCALL_TABLE[nr].is_none() {
        frame.rax = (-ENOSYS) as u64;
        return;
    }
    
    // Security hook: MAC check
    if !security::syscall_allowed(nr, current_task().cred) {
        frame.rax = (-EPERM) as u64;
        audit::log_denied_syscall(nr, frame);
        return;
    }
    
    let start = ticks();
    frame.rax = SYSCALL_TABLE[nr].unwrap()(frame) as u64;
    let elapsed = ticks() - start;
    
    // Telemetry: slow syscall warning
    if elapsed > SYSCALL_SLOW_THRESHOLD {
        warn!("syscall {} took {} ticks", nr, elapsed);
    }
}
```

- [x] Define `SyscallFrame` struct matching assembly layout
- [x] Create `SYSCALL_TABLE: [Option<SyscallFn>; 512]` static
- [x] Implement `syscall_register(nr, fn_ptr)` for populating the table
- [x] Implement `syscall_dispatch_rust()` with bounds checking
- [x] Return `-ENOSYS` for unimplemented syscalls
- [x] Integrate MAC `security::syscall_allowed()` hook
- [x] Integrate `audit::log_denied_syscall()` on denial
- [x] Add per-syscall timing telemetry with slow threshold warning
- [x] Define syscall number constants (`SYS_read = 0`, `SYS_write = 1`, etc.)
- [x] Implement syscall argument extraction helpers from `SyscallFrame`

### 4. Syscall Table Implementation (Priority Order)

#### Tier 1 — Essential (init + shell)
- [x] `sys_read(fd, buf, count)`
- [x] `sys_write(fd, buf, count)`
- [x] `sys_open(pathname, flags, mode)`
- [x] `sys_close(fd)`
- [x] `sys_exit(error_code)`
- [x] `sys_brk(addr)`
- [x] `sys_mmap(addr, len, prot, flags, fd, off)`
- [x] `sys_munmap(addr, len)`
- [x] `sys_fork()`
- [x] `sys_execve(filename, argv, envp)`
- [x] `sys_wait4(pid, wstatus, options, rusage)`
- [x] `sys_getpid()`
- [x] `sys_getppid()`
- [x] `sys_chdir(path)`
- [x] `sys_getcwd(buf, size)`

#### Tier 2 — POSIX compatibility
- [x] `sys_stat(pathname, statbuf)`
- [x] `sys_fstat(fd, statbuf)`
- [x] `sys_lstat(pathname, statbuf)`
- [x] `sys_lseek(fd, offset, whence)`
- [x] `sys_ioctl(fd, cmd, arg)`
- [x] `sys_pipe(pipefd)`
- [x] `sys_dup(fd)`
- [x] `sys_dup2(oldfd, newfd)`
- [x] `sys_fcntl(fd, cmd, arg)`
- [x] `sys_access(pathname, mode)`
- [x] `sys_getdents64(fd, dirp, count)`
- [x] `sys_mkdir(pathname, mode)`
- [x] `sys_rmdir(pathname)`
- [x] `sys_unlink(pathname)`
- [x] `sys_rename(oldpath, newpath)`
- [x] `sys_link(oldpath, newpath)`
- [x] `sys_symlink(target, linkpath)`
- [x] `sys_readlink(pathname, buf, bufsiz)`
- [x] `sys_chmod(pathname, mode)`
- [x] `sys_chown(pathname, owner, group)`
- [x] `sys_umask(mask)`

#### Tier 3 — Signals & threads
- [x] `sys_rt_sigaction(sig, act, oldact)`
- [x] `sys_rt_sigprocmask(how, set, oldset)`
- [x] `sys_kill(pid, sig)`
- [x] `sys_rt_sigreturn()`
- [x] `sys_clone(flags, stack, ptid, ctid, tls)`
- [x] `sys_exit_group(error_code)`
- [x] `sys_set_tid_address(tidptr)`
- [x] `sys_gettid()`
- [x] `sys_nanosleep(req, rem)`

#### Tier 4 — Filesystem advanced
- [x] `sys_mount(source, target, fstype, flags, data)`
- [x] `sys_umount2(target, flags)`
- [x] `sys_fsync(fd)`
- [x] `sys_truncate(path, length)`
- [x] `sys_ftruncate(fd, length)`
- [x] `sys_getrlimit(resource, rlim)`
- [x] `sys_setrlimit(resource, rlim)`

#### Tier 5 — Network (after network stack)
- [x] `sys_socket(domain, type, protocol)`
- [x] `sys_bind(sockfd, addr, addrlen)`
- [x] `sys_listen(sockfd, backlog)`
- [x] `sys_accept(sockfd, addr, addrlen)`
- [x] `sys_connect(sockfd, addr, addrlen)`
- [x] `sys_sendto(sockfd, buf, len, flags, addr, addrlen)`
- [x] `sys_recvfrom(sockfd, buf, len, flags, addr, addrlen)`
- [x] `sys_shutdown(sockfd, how)`
- [x] `sys_setsockopt(sockfd, level, optname, optval, optlen)`
- [x] `sys_getsockopt(sockfd, level, optname, optval, optlen)`

#### Tier 6 — Advanced / optional
- [x] `sys_poll(fds, nfds, timeout)`
- [x] `sys_select(nfds, readfds, writefds, exceptfds, timeout)`
- [x] `sys_epoll_create1(flags)`
- [x] `sys_epoll_ctl(epfd, op, fd, event)`
- [x] `sys_epoll_wait(epfd, events, maxevents, timeout)`
- [x] `sys_futex(uaddr, futex_op, val, timeout, uaddr2, val3)`
- [x] `sys_prctl(option, arg2, arg3, arg4, arg5)`
- [x] `sys_uname(buf)`
- [x] `sys_sysinfo(info)`
- [x] `sys_gettimeofday(tv, tz)`
- [x] `sys_times(tbuf)`
- [x] `sys_getrandom(buf, buflen, flags)`
- [x] `sys_getcpu(cpu, node, tcache)`
- [x] `sys_arch_prctl(code, addr)`
- [x] `sys_seccomp(op, flags, args)`

### 5. VDSO — Fast Userspace Syscalls

Certain syscalls do not need kernel entry:
```rust
// VDSO page mapped at a fixed userspace address (e.g., 0x7fff_0000_0000)
pub fn __vdso_clock_gettime(clockid: i32, tp: *mut timespec) -> i32;
pub fn __vdso_getcpu(cpu: *mut u32, node: *mut u32, tcache: *mut c_void) -> i32;
pub fn __vdso_gettimeofday(tv: *mut timeval, tz: *mut timezone) -> i32;
pub fn __vdso_time(t: *mut time_t) -> time_t;
```

- [x] Build ELF shared object at kernel compile time
- [x] Implement `__vdso_clock_gettime` using TSC + offset
- [x] Implement `__vdso_gettimeofday`
- [x] Implement `__vdso_getcpu` using `RDPID` or `CPUID`
- [x] Implement `__vdso_time`
- [x] Map VDSO page into every process at a fixed address
- [x] Populate ELF dynamic section so `ld.so` can resolve symbols
- [x] Verify `clock_gettime` via VDSO is < 10 ns (vs 80 ns syscall)

### 6. Security Integration

Every syscall passes through LSM hooks:
```rust
trait LsmHooks {
    fn syscall_enter(nr: u64, frame: &SyscallFrame) -> Result<(), Errno>;
    fn syscall_exit(nr: u64, frame: &SyscallFrame, retval: i64);
}
```

- [x] Define `LsmHooks` trait with `syscall_enter` / `syscall_exit`
- [x] Implement default permissive hooks
- [x] Integrate MAC policy into `syscall_enter`
- [x] Log all denied syscalls to audit buffer
- [x] Log all `execve` calls with full argv to audit buffer
- [x] Log all `open(O_CREAT)` calls with mode and path

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_syscall_abi` | `syscall` instruction enters kernel, `sysret` returns with correct rax | [x] |
| `test_read_write` | Userspace can read/write file via syscall 0/1 | [x] |
| `test_mmap_anon` | `mmap(0, 4096, PROT_RW, MAP_ANON, -1, 0)` returns valid page, writable | [x] |
| `test_fork` | `fork()` creates new process with independent address space | [x] |
| `test_execve` | `execve("/bin/sh", ...)` loads ELF, replaces image, runs | [x] |
| `test_sigreturn` | Signal handler runs and returns to interrupted instruction | [x] |
| `test_clone_thread` | `clone(CLONE_VM | CLONE_FS | CLONE_FILES)` shares memory | [x] |
| `test_futex_wait_wake` | Two threads synchronize via `futex_wait` / `futex_wake` | [x] |
| `test_epoll` | `epoll_create` + `epoll_ctl` + `epoll_wait` on pipe write | [x] |
| `test_seccomp` | `seccomp(SECCOMP_SET_MODE_STRICT)` kills on disallowed syscall | [x] |
| `test_vdso_clock` | `clock_gettime` via VDSO < 10 ns, time monotonically increases | [x] |

---

*Design document produced by Phase 2 Planning*
*2026-05-19*
