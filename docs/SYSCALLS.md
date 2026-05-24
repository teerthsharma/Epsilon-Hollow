# System Call Reference

> Source: `kernel/seal-os/src/syscall/table.rs`

Seal OS implements 36 syscalls — POSIX-like + Epsilon extensions.

## POSIX-like Syscalls (0–35)

| # | Name | Args | Status |
|---|------|------|--------|
| 0 | `exit` | — | ✅ Real |
| 1 | `write` | fd, buf, len | ✅ Real |
| 2 | `read` | fd, buf, len | ✅ Real |
| 3 | `open` | path, flags, mode | ✅ Real (O_CREAT supported) |
| 4 | `close` | fd | ✅ Real |
| 5 | `exec` | path | ✅ Real (ELF64, shebang, .aether) |
| 6 | `fork` | — | ✅ Real (full duplication) |
| 7 | `waitpid` | pid | ⚠️ Returns arg0 (stub) |
| 8 | `mmap` | addr, len, prot | ✅ Real |
| 9 | `getpid` | — | ✅ Real |
| 10 | `stat` | path | ✅ Real |
| 11 | `mkdir` | path | ✅ Real |
| 12 | `setuid` | uid | ✅ Real |
| 13 | `setgid` | gid | ✅ Real |
| 14 | `chdir` | path | ✅ Real |
| 15 | `getcwd` | — | ✅ Real |
| 16 | `getppid` | — | ⚠️ Returns 1 (no parent tracking) |
| 17 | `nanosleep` | ms | ✅ Real (spin-yield) |
| 18 | `reboot` | cmd | ✅ Real (ACPI / kbd / triple-fault) |
| 19 | `lseek` | fd, offset, whence | ✅ Real (SET/CUR/END) |
| 20 | `unlink` | path | ✅ Real |
| 21 | `rmdir` | path | ✅ Real |
| 22 | `rename` | old, new | ✅ Real (same-dir) |
| 23 | `getrandom` | buf, len | ✅ Real (RDRAND/RDSEED) |
| 24 | `kmsg_read` | buf, len | ✅ Real (32 KiB ring) |
| 25 | `kill` | pid, sig | ✅ Real |
| 26 | `sigaction` | sig, act, oldact | ✅ Real |
| 27 | `sigreturn` | — | ✅ Real (never returns) |
| 28 | `pipe` | pipefd[2] | ✅ Real |
| 29 | `dup` | oldfd | ✅ Real |
| 30 | `dup2` | oldfd, newfd | ✅ Real |
| 31 | `brk` | addr | ✅ Real |
| 32 | `gettimeofday` | timeval | ✅ Real (RTC) |
| 33 | `settimeofday` | sec, usec | ⚠️ Returns EPERM |
| 34 | `watchdog` | pet | ✅ Real |
| 35 | `ioctl` | fd, req, arg | ✅ Real |

## Epsilon Extensions (100–111)

| # | Name | Status |
|---|------|--------|
| 100 | `manifold_query` | ✅ Real |
| 101 | `teleport` | ✅ Real |
| 102 | `theorem_status` | ✅ Real |
| 103 | `pkg_install` | ✅ Real (remote + local) |
| 104 | `pkg_remove` | ✅ Real |
| 105 | `pkg_list` | ✅ Real |
| 106 | `wifi_scan` | ⚠️ Honest "no hardware" |
| 107 | `wifi_connect` | ⚠️ Honest "no hardware" |
| 108 | `bt_scan` | ⚠️ Honest "no hardware" |
| 109 | `bt_pair` | ⚠️ Honest "no hardware" |
| 110 | `setting_get` | ✅ Real |
| 111 | `setting_set` | ✅ Real |

## Return Values

- Success: `code >= 0`
- Error: `code = -errno` (POSIX errno codes)
- Data: `code = 0`, `data = Some(String)` for string-returning syscalls
