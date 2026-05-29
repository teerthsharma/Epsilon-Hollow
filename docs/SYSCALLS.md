# Seal ABI Reference

> Source: `kernel/seal-os/src/syscall/table.rs`

Seal OS exposes a native Seal ABI plus Epsilon theorem extensions. No POSIX ABI or Unix compatibility contract is promised. Familiar names such as `read`, `write`, `open`, or `execve` are Seal-defined entry points with Seal semantics, and they may expose topology and theorem state directly.

## Seal ABI Calls

| # | Name | Args | Status |
|---|---|---|---|
| 0 | `exit` | code | Real |
| 1 | `write` | fd, buf, len | Real |
| 2 | `read` | fd, buf, len | Real |
| 3 | `open` | path, flags, mode | Real |
| 4 | `close` | fd | Real |
| 5 | `exec` | path | Real: ELF64 ET_EXEC/ET_DYN, `PT_INTERP`, `DT_NEEDED`, shebang, `.aether` |
| 6 | `fork` | - | Real |
| 7 | `waitpid` | pid | Stub: returns requested pid |
| 8 | `mmap` | addr, len, prot | Real |
| 9 | `getpid` | - | Real |
| 10 | `stat` | path | Real |
| 11 | `mkdir` | path | Real |
| 14 | `chdir` | path | Real |
| 15 | `getcwd` | - | Real |
| 16 | `setuid` | uid | Real |
| 17 | `setgid` | gid | Real |
| 18 | `reboot` | cmd | Real: ACPI, keyboard reset, triple fault |
| 19 | `lseek` | fd, offset, whence | Real |
| 20 | `unlink` | path | Real |
| 21 | `rmdir` | path | Real |
| 22 | `rename` | old, new | Real |
| 23 | `getrandom` | buf, len | Real |
| 24 | `kmsg_read` | buf, len | Real |
| 25 | `kill` | pid, sig | Real |
| 26 | `sigaction` | sig, act, oldact | Real |
| 27 | `sigreturn` | - | Real, never returns normally |
| 28 | `pipe` | pipefd[2] | Real |
| 29 | `dup` | oldfd | Real |
| 30 | `dup2` | oldfd, newfd | Real |
| 31 | `brk` | addr | Real |
| 32 | `gettimeofday` | timeval | Real |
| 33 | `settimeofday` | sec, usec | Permission-denied by design |
| 34 | `watchdog` | pet | Real |
| 35 | `ioctl` | fd, req, arg | Real |
| 36 | `sleep` | ACPI state | Real |
| 37 | `sync` | - | Real |
| 38 | `getppid` | - | Real |
| 39 | `nanosleep` | ms | Real: spin-yield |
| 40 | `seteuid` | euid | Real |
| 41 | `setegid` | egid | Real |
| 42 | `clone` | flags | Real |
| 43 | `setrlimit` | resource, limit | Real |
| 44 | `getrlimit` | resource | Real |
| 45 | `sigaltstack` | new_stack, old_stack | Real |

## Signal ABI

`sigaction` accepts a 16-byte action record: `{ handler: u64, flags: u64 }`.
The old 8-byte handler-only ABI remains compatible because the handler stays
in the first slot. Supported flags are `SA_ONSTACK` and `SA_RESTART`.

`sigaltstack` uses `{ sp: u64, flags: u64, size: u64 }`. `SS_DISABLE`
disables the alternate stack, and `SS_ONSTACK` is reported while a handler is
running there. Handlers with `SA_ONSTACK` use the alternate stack top when it is
enabled and at least `MINSIGSTKSZ` bytes.

Interrupted restartable syscalls that return `-EINTR` are rewound to the
`syscall` instruction when the pending signal action has `SA_RESTART`.

## Epsilon Extensions

| # | Name | Status |
|---|---|---|
| 100 | `manifold_query` | Real; validates theorem id 1-10 and returns T1-T10 state |
| 101 | `teleport` | Real |
| 102 | `theorem_status` | Real; returns T1-T10 gate/runtime state |
| 103 | `pkg_install` | Real |
| 104 | `pkg_remove` | Real |
| 105 | `pkg_list` | Real |
| 106 | `wifi_scan` | Honest no-hardware path unless adapter exists |
| 107 | `wifi_connect` | Honest no-hardware path unless adapter exists |
| 108 | `bt_scan` | Honest no-hardware path unless adapter exists |
| 109 | `bt_pair` | Honest no-hardware path unless adapter exists |
| 110 | `setting_get` | Real |
| 111 | `setting_set` | Real |

## Return Values

- Success: `code >= 0`
- Error: `code = -errno` using compact numeric error values internally
- Data: `code = 0`, `data = Some(String)` for string-returning calls
