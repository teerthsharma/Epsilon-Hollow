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

- [ ] Verify `EFER.SCE` is set during early boot
- [ ] Program `STAR` MSR with kernel CS = 0x08, user CS = 0x1B
- [ ] Program `LSTAR` MSR to `syscall_entry_asm` address
- [ ] Program `SFMASK` MSR with `IF | DF | TF | AC | NT`
- [ ] Document the `syscall` / `sysret` ABI in a header comment
- [ ] Define `SYSCALL_MAX_ARGS = 6`
- [ ] Define `SYSCALL_MAX_NR = 512`

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

- [ ] Write `syscall_entry_asm` in `kernel/seal-os/src/syscall/entry.asm` or inline asm
- [ ] Implement `swapgs` on entry with SWAPGS mitigation sequence
- [ ] Save all caller-saved registers (rcx, r11 overwritten by syscall)
- [ ] Save user RSP, load kernel RSP from `gs:[PERCPU_KERNEL_RSP]`
- [ ] Build `SyscallFrame` on kernel stack with all GPRs
- [ ] Call `syscall_dispatch_rust(frame: *mut SyscallFrame)`
- [ ] On exit: restore user RSP, RIP, RFLAGS
- [ ] `sysretq` with verified correct privilege level
- [ ] Add `swapgs` mismatch detection (NMI during entry scenario)

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

- [ ] Define `SyscallFrame` struct matching assembly layout
- [ ] Create `SYSCALL_TABLE: [Option<SyscallFn>; 512]` static
- [ ] Implement `syscall_register(nr, fn_ptr)` for populating the table
- [ ] Implement `syscall_dispatch_rust()` with bounds checking
- [ ] Return `-ENOSYS` for unimplemented syscalls
- [ ] Integrate MAC `security::syscall_allowed()` hook
- [ ] Integrate `audit::log_denied_syscall()` on denial
- [ ] Add per-syscall timing telemetry with slow threshold warning
- [ ] Define syscall number constants (`SYS_read = 0`, `SYS_write = 1`, etc.)
- [ ] Implement syscall argument extraction helpers from `SyscallFrame`

### 4. Syscall Table Implementation (Priority Order)

#### Tier 1 — Essential (init + shell)
- [ ] `sys_read(fd, buf, count)`
- [ ] `sys_write(fd, buf, count)`
- [ ] `sys_open(pathname, flags, mode)`
- [ ] `sys_close(fd)`
- [ ] `sys_exit(error_code)`
- [ ] `sys_brk(addr)`
- [ ] `sys_mmap(addr, len, prot, flags, fd, off)`
- [ ] `sys_munmap(addr, len)`
- [ ] `sys_fork()`
- [ ] `sys_execve(filename, argv, envp)`
- [ ] `sys_wait4(pid, wstatus, options, rusage)`
- [ ] `sys_getpid()`
- [ ] `sys_getppid()`
- [ ] `sys_chdir(path)`
- [ ] `sys_getcwd(buf, size)`

#### Tier 2 — POSIX compatibility
- [ ] `sys_stat(pathname, statbuf)`
- [ ] `sys_fstat(fd, statbuf)`
- [ ] `sys_lstat(pathname, statbuf)`
- [ ] `sys_lseek(fd, offset, whence)`
- [ ] `sys_ioctl(fd, cmd, arg)`
- [ ] `sys_pipe(pipefd)`
- [ ] `sys_dup(fd)`
- [ ] `sys_dup2(oldfd, newfd)`
- [ ] `sys_fcntl(fd, cmd, arg)`
- [ ] `sys_access(pathname, mode)`
- [ ] `sys_getdents64(fd, dirp, count)`
- [ ] `sys_mkdir(pathname, mode)`
- [ ] `sys_rmdir(pathname)`
- [ ] `sys_unlink(pathname)`
- [ ] `sys_rename(oldpath, newpath)`
- [ ] `sys_link(oldpath, newpath)`
- [ ] `sys_symlink(target, linkpath)`
- [ ] `sys_readlink(pathname, buf, bufsiz)`
- [ ] `sys_chmod(pathname, mode)`
- [ ] `sys_chown(pathname, owner, group)`
- [ ] `sys_umask(mask)`

#### Tier 3 — Signals & threads
- [ ] `sys_rt_sigaction(sig, act, oldact)`
- [ ] `sys_rt_sigprocmask(how, set, oldset)`
- [ ] `sys_kill(pid, sig)`
- [ ] `sys_rt_sigreturn()`
- [ ] `sys_clone(flags, stack, ptid, ctid, tls)`
- [ ] `sys_exit_group(error_code)`
- [ ] `sys_set_tid_address(tidptr)`
- [ ] `sys_gettid()`
- [ ] `sys_nanosleep(req, rem)`

#### Tier 4 — Filesystem advanced
- [ ] `sys_mount(source, target, fstype, flags, data)`
- [ ] `sys_umount2(target, flags)`
- [ ] `sys_fsync(fd)`
- [ ] `sys_truncate(path, length)`
- [ ] `sys_ftruncate(fd, length)`
- [ ] `sys_getrlimit(resource, rlim)`
- [ ] `sys_setrlimit(resource, rlim)`

#### Tier 5 — Network (after network stack)
- [ ] `sys_socket(domain, type, protocol)`
- [ ] `sys_bind(sockfd, addr, addrlen)`
- [ ] `sys_listen(sockfd, backlog)`
- [ ] `sys_accept(sockfd, addr, addrlen)`
- [ ] `sys_connect(sockfd, addr, addrlen)`
- [ ] `sys_sendto(sockfd, buf, len, flags, addr, addrlen)`
- [ ] `sys_recvfrom(sockfd, buf, len, flags, addr, addrlen)`
- [ ] `sys_shutdown(sockfd, how)`
- [ ] `sys_setsockopt(sockfd, level, optname, optval, optlen)`
- [ ] `sys_getsockopt(sockfd, level, optname, optval, optlen)`

#### Tier 6 — Advanced / optional
- [ ] `sys_poll(fds, nfds, timeout)`
- [ ] `sys_select(nfds, readfds, writefds, exceptfds, timeout)`
- [ ] `sys_epoll_create1(flags)`
- [ ] `sys_epoll_ctl(epfd, op, fd, event)`
- [ ] `sys_epoll_wait(epfd, events, maxevents, timeout)`
- [ ] `sys_futex(uaddr, futex_op, val, timeout, uaddr2, val3)`
- [ ] `sys_prctl(option, arg2, arg3, arg4, arg5)`
- [ ] `sys_uname(buf)`
- [ ] `sys_sysinfo(info)`
- [ ] `sys_gettimeofday(tv, tz)`
- [ ] `sys_times(tbuf)`
- [ ] `sys_getrandom(buf, buflen, flags)`
- [ ] `sys_getcpu(cpu, node, tcache)`
- [ ] `sys_arch_prctl(code, addr)`
- [ ] `sys_seccomp(op, flags, args)`

### 5. VDSO — Fast Userspace Syscalls

Certain syscalls do not need kernel entry:
```rust
// VDSO page mapped at a fixed userspace address (e.g., 0x7fff_0000_0000)
pub fn __vdso_clock_gettime(clockid: i32, tp: *mut timespec) -> i32;
pub fn __vdso_getcpu(cpu: *mut u32, node: *mut u32, tcache: *mut c_void) -> i32;
pub fn __vdso_gettimeofday(tv: *mut timeval, tz: *mut timezone) -> i32;
pub fn __vdso_time(t: *mut time_t) -> time_t;
```

- [ ] Build ELF shared object at kernel compile time
- [ ] Implement `__vdso_clock_gettime` using TSC + offset
- [ ] Implement `__vdso_gettimeofday`
- [ ] Implement `__vdso_getcpu` using `RDPID` or `CPUID`
- [ ] Implement `__vdso_time`
- [ ] Map VDSO page into every process at a fixed address
- [ ] Populate ELF dynamic section so `ld.so` can resolve symbols
- [ ] Verify `clock_gettime` via VDSO is < 10 ns (vs 80 ns syscall)

### 6. Security Integration

Every syscall passes through LSM hooks:
```rust
trait LsmHooks {
    fn syscall_enter(nr: u64, frame: &SyscallFrame) -> Result<(), Errno>;
    fn syscall_exit(nr: u64, frame: &SyscallFrame, retval: i64);
}
```

- [ ] Define `LsmHooks` trait with `syscall_enter` / `syscall_exit`
- [ ] Implement default permissive hooks
- [ ] Integrate MAC policy into `syscall_enter`
- [ ] Log all denied syscalls to audit buffer
- [ ] Log all `execve` calls with full argv to audit buffer
- [ ] Log all `open(O_CREAT)` calls with mode and path

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_syscall_abi` | `syscall` instruction enters kernel, `sysret` returns with correct rax | [ ] |
| `test_read_write` | Userspace can read/write file via syscall 0/1 | [ ] |
| `test_mmap_anon` | `mmap(0, 4096, PROT_RW, MAP_ANON, -1, 0)` returns valid page, writable | [ ] |
| `test_fork` | `fork()` creates new process with independent address space | [ ] |
| `test_execve` | `execve("/bin/sh", ...)` loads ELF, replaces image, runs | [ ] |
| `test_sigreturn` | Signal handler runs and returns to interrupted instruction | [ ] |
| `test_clone_thread` | `clone(CLONE_VM | CLONE_FS | CLONE_FILES)` shares memory | [ ] |
| `test_futex_wait_wake` | Two threads synchronize via `futex_wait` / `futex_wake` | [ ] |
| `test_epoll` | `epoll_create` + `epoll_ctl` + `epoll_wait` on pipe write | [ ] |
| `test_seccomp` | `seccomp(SECCOMP_SET_MODE_STRICT)` kills on disallowed syscall | [ ] |
| `test_vdso_clock` | `clock_gettime` via VDSO < 10 ns, time monotonically increases | [ ] |

---

*Design document produced by Phase 2 Planning*
*2026-05-19*
