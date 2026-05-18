# System Call Reference

> Source: `kernel/seal-os/src/syscall/table.rs`

---

## Dispatch Function

```
pub fn dispatch(num: u64, arg0: u64, arg1: u64, arg2: u64) -> SyscallResult
```
(Source: line 53)

Returns `SyscallResult { code: i64, data: Option<String> }` (lines 28-31).

---

## POSIX-like Syscalls

### 0 -- SYS_EXIT

| Field | Value |
|-------|-------|
| Number | `0` (line 10) |
| Signature | `exit()` -- no arguments used |
| Action | Returns `SyscallResult::ok(0)` (line 55) |
| Subsystem | Process management (terminates current process) |
| Return | code = 0 |

### 1 -- SYS_WRITE

| Field | Value |
|-------|-------|
| Number | `1` (line 11) |
| Signature | `write(fd: u64, ptr: u64, len: u64)` |
| Action | Kernel write to serial; returns len as success (lines 57-59) |
| Subsystem | Serial I/O |
| Return | code = arg2 (number of bytes "written") |

### 2 -- SYS_READ

| Field | Value |
|-------|-------|
| Number | `2` (line 12) |
| Signature | `read(fd: u64, ptr: u64, len: u64)` |
| Action | Stub; returns 0 bytes read (line 60) |
| Subsystem | I/O (not yet implemented) |
| Return | code = 0 |

### 3 -- SYS_OPEN

| Field | Value |
|-------|-------|
| Number | `3` (line 13) |
| Signature | `open(path: u64, flags: u64, mode: u64)` |
| Action | Stub; returns fixed file descriptor 3 (line 61) |
| Subsystem | Filesystem |
| Return | code = 3 (fd) |

### 4 -- SYS_CLOSE

| Field | Value |
|-------|-------|
| Number | `4` (line 14) |
| Signature | `close(fd: u64)` |
| Action | Stub; returns success (line 62) |
| Subsystem | Filesystem |
| Return | code = 0 |

### 5 -- SYS_EXEC

| Field | Value |
|-------|-------|
| Number | `5` (line 15) |
| Signature | `exec(path: u64, argv: u64, envp: u64)` |
| Action | Not implemented; falls through to ENOSYS (line 76) |
| Subsystem | Process management |
| Return | code = -38 (ENOSYS) |

### 6 -- SYS_FORK

| Field | Value |
|-------|-------|
| Number | `6` (line 16) |
| Signature | `fork()` |
| Action | Not implemented; falls through to ENOSYS (line 76) |
| Subsystem | Process management |
| Return | code = -38 (ENOSYS) |

### 7 -- SYS_WAITPID

| Field | Value |
|-------|-------|
| Number | `7` (line 17) |
| Signature | `waitpid(pid: u64, status: u64, options: u64)` |
| Action | Not implemented; falls through to ENOSYS (line 76) |
| Subsystem | Process management |
| Return | code = -38 (ENOSYS) |

### 8 -- SYS_MMAP

| Field | Value |
|-------|-------|
| Number | `8` (line 18) |
| Signature | `mmap(addr: u64, len: u64, prot: u64)` |
| Action | Not implemented; falls through to ENOSYS (line 76) |
| Subsystem | Memory management |
| Return | code = -38 (ENOSYS) |

### 9 -- SYS_GETPID

| Field | Value |
|-------|-------|
| Number | `9` (line 19) |
| Signature | `getpid()` |
| Action | Returns fixed PID 1 (line 63) |
| Subsystem | Process management |
| Return | code = 1 |

### 10 -- SYS_STAT

| Field | Value |
|-------|-------|
| Number | `10` (line 20) |
| Signature | `stat(path: u64, buf: u64)` |
| Action | Not implemented; falls through to ENOSYS (line 76) |
| Subsystem | Filesystem |
| Return | code = -38 (ENOSYS) |

---

## Epsilon Extension Syscalls

### 100 -- SYS_MANIFOLD_QUERY

| Field | Value |
|-------|-------|
| Number | `100` (line 23) |
| Signature | `manifold_query(theorem_id: u64)` |
| Action | Returns theorem status string (lines 64-67) |
| Subsystem | ManifoldFS / Theorem subsystem |
| Return | code = 0, data = `"theorem_{arg0}: ACTIVE"` |

### 101 -- SYS_TELEPORT

| Field | Value |
|-------|-------|
| Number | `101` (line 24) |
| Signature | `teleport(inode_id: u64, dst_dir: u64)` |
| Action | Returns teleport confirmation string (lines 68-71) |
| Subsystem | ManifoldFS |
| Return | code = 0, data = `"teleported inode {arg0} -> dir {arg1}"` |

### 102 -- SYS_THEOREM_STATUS

| Field | Value |
|-------|-------|
| Number | `102` (line 25) |
| Signature | `theorem_status()` -- no arguments used |
| Action | Returns status of all theorems (lines 72-75) |
| Subsystem | Theorem subsystem |
| Return | code = 0, data = `"T1:ACTIVE T2:ACTIVE T3:ACTIVE T4:ACTIVE T5:ACTIVE"` |

---

## Error Handling

Any unrecognized syscall number returns `SyscallResult::err(38)` (line 76), where 38 is the POSIX `ENOSYS` errno ("Function not implemented"). The error is encoded as a negative code: `code = -38`.

---

## Summary Table

| Num | Name | Status | Return |
|-----|------|--------|--------|
| 0 | `SYS_EXIT` | Implemented | 0 |
| 1 | `SYS_WRITE` | Implemented (serial) | bytes written |
| 2 | `SYS_READ` | Stub | 0 |
| 3 | `SYS_OPEN` | Stub | fd=3 |
| 4 | `SYS_CLOSE` | Stub | 0 |
| 5 | `SYS_EXEC` | ENOSYS | -38 |
| 6 | `SYS_FORK` | ENOSYS | -38 |
| 7 | `SYS_WAITPID` | ENOSYS | -38 |
| 8 | `SYS_MMAP` | ENOSYS | -38 |
| 9 | `SYS_GETPID` | Implemented | 1 |
| 10 | `SYS_STAT` | ENOSYS | -38 |
| 100 | `SYS_MANIFOLD_QUERY` | Implemented | theorem status string |
| 101 | `SYS_TELEPORT` | Implemented | teleport confirmation |
| 102 | `SYS_THEOREM_STATUS` | Implemented | all theorem statuses |
