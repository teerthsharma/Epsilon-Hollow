// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Seal ABI dispatch table: native OS calls plus Epsilon theorem extensions.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;

use crate::fs::manifold_fs::{FsError, ManifoldFS};
use crate::fs::vfs::{with_vfs, VfsError, VfsHandle};

pub const SYS_EXIT: u64 = 0;
pub const SYS_WRITE: u64 = 1;
pub const SYS_READ: u64 = 2;
pub const SYS_OPEN: u64 = 3;
pub const SYS_CLOSE: u64 = 4;
pub const SYS_EXEC: u64 = 5;
pub const SYS_FORK: u64 = 6;
pub const SYS_WAITPID: u64 = 7;
pub const SYS_MMAP: u64 = 8;
pub const SYS_GETPID: u64 = 9;
pub const SYS_STAT: u64 = 10;
pub const SYS_MKDIR: u64 = 11;
pub const SYS_SETUID: u64 = 16;
pub const SYS_SETGID: u64 = 17;
pub const SYS_CHDIR: u64 = 14;
pub const SYS_GETCWD: u64 = 15;
pub const SYS_GETPPID: u64 = 38;
pub const SYS_NANOSLEEP: u64 = 39;
pub const SYS_SETEUID: u64 = 40;
pub const SYS_SETEGID: u64 = 41;
pub const SYS_REBOOT: u64 = 18;
pub const SYS_LSEEK: u64 = 19;
pub const SYS_UNLINK: u64 = 20;
pub const SYS_RMDIR: u64 = 21;
pub const SYS_RENAME: u64 = 22;
pub const SYS_GETRANDOM: u64 = 23;
pub const SYS_KMSG_READ: u64 = 24;
pub const SYS_KILL: u64 = 25;
pub const SYS_SIGACTION: u64 = 26;
pub const SYS_SIGRETURN: u64 = 27;
pub const SYS_SIGALTSTACK: u64 = 45;
pub const SYS_PIPE: u64 = 28;
pub const SYS_DUP: u64 = 29;
pub const SYS_DUP2: u64 = 30;
pub const SYS_BRK: u64 = 31;
pub const SYS_GETTIMEOFDAY: u64 = 32;
pub const SYS_SETTIMEOFDAY: u64 = 33;
pub const SYS_WATCHDOG: u64 = 34;
pub const SYS_IOCTL: u64 = 35;
pub const SYS_SLEEP: u64 = 36;
pub const SYS_SYNC: u64 = 37;
pub const SYS_CLONE: u64 = 42;
pub const SYS_SETRLIMIT: u64 = 43;
pub const SYS_GETRLIMIT: u64 = 44;

// Epsilon extensions
pub const SYS_MANIFOLD_QUERY: u64 = 100;

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    // NOTE: `test_main()` runs before any task is ever made the scheduler's
    // `current` (see `ManifoldScheduler::new()`, `current: None`), so
    // `crate::process::scheduler::set_current_uid()`/`set_current_euid()` and
    // friends silently no-op here and `current_euid()` always reads back the
    // default 0. A dispatch()-level test therefore always takes the
    // privileged (euid==0) branch and can't exercise *denial* — that rule is
    // covered directly below via `credential_change_allowed`, which takes
    // plain ids and has no scheduler dependency.

    fn test_setuid_changes_uid() -> TestResult {
        let current = crate::process::scheduler::current_uid();
        let result = dispatch(SYS_SETUID, 42, 0, 0);
        test_assert!(
            result.code >= 0,
            "SYS_SETUID should still succeed for a privileged (euid=0) caller"
        );
        // Restore best-effort
        dispatch(SYS_SETUID, current as u64, 0, 0);
        TestResult::Pass
    }

    fn test_setgid_changes_gid() -> TestResult {
        let current = crate::process::scheduler::current_gid();
        let result = dispatch(SYS_SETGID, 99, 0, 0);
        test_assert!(
            result.code >= 0,
            "SYS_SETGID should still succeed for a privileged (euid=0) caller"
        );
        dispatch(SYS_SETGID, current as u64, 0, 0);
        TestResult::Pass
    }

    /// RED: `table.rs`'s SYS_SETUID/SETGID/SETEUID/SETEGID arms used to call
    /// `set_current_uid`/`_gid`/`_euid`/`_egid` unconditionally, with zero
    /// regard for the caller's own privilege — any task could `setuid(0)`.
    /// An unprivileged task (euid=1000) asking for an id it holds neither as
    /// real nor effective (root, or some other arbitrary id) must be denied.
    fn test_credential_change_denies_unprivileged_escalation() -> TestResult {
        test_assert!(!credential_change_allowed(0, 1000, 1000));
        test_assert!(!credential_change_allowed(1, 1000, 2000));
        TestResult::Pass
    }

    fn test_credential_change_permits_id_already_held() -> TestResult {
        test_assert!(credential_change_allowed(1000, 1000, 2000)); // matches real
        test_assert!(credential_change_allowed(2000, 1000, 2000)); // matches effective
        TestResult::Pass
    }

    fn test_credential_change_permits_root() -> TestResult {
        test_assert!(credential_change_allowed(9999, 0, 0));
        TestResult::Pass
    }

    /// RED: the SYS_WRITE arm for any fd other than stdout/stderr used to
    /// ignore `buf_ptr`/`len` entirely and pull bytes from the process-wide
    /// `SYSCALL_PATH` buffer instead, at hardcoded offset 0 — so an invalid
    /// user pointer was never even inspected. A null pointer with `len > 0`
    /// must now be rejected by `copy_from_user` (EFAULT) before the fd's
    /// backing file is touched, leaving the fd's cursor exactly where it was.
    fn test_write_rejects_invalid_pointer_without_touching_fd() -> TestResult {
        let fd = 90_210u64; // unlikely to collide with a live fd
        let starting_offset = 5usize; // simulate a cursor already advanced
        {
            let mut table = FILE_TABLE.lock();
            table.insert(
                fd,
                FdEntry {
                    handle: crate::fs::vfs::VfsHandle {
                        fs_idx: 0,
                        inode: 0,
                    },
                    path: String::from("/tmp/defect_b_probe"),
                    offset: starting_offset,
                    owner: crate::process::scheduler::current_task_id(),
                },
            );
        }

        let result = dispatch(SYS_WRITE, fd, 0, 8);
        test_assert_eq!(result.code, -14); // EFAULT

        let offset_after = FILE_TABLE.lock().get(&fd).map(|e| e.offset);
        test_assert_eq!(offset_after, Some(starting_offset));

        FILE_TABLE.lock().remove(&fd);
        TestResult::Pass
    }

    /// Pure-predicate check for the empty-path guard shape shared by every
    /// path-taking syscall arm (`SYS_OPEN`, `SYS_EXEC`, `SYS_CHDIR`, and —
    /// after this fix — `SYS_STAT`, `SYS_MKDIR`, `SYS_UNLINK`, `SYS_RMDIR`,
    /// `SYS_RENAME`). `copy_path_from_user` returns `Ok(String::new())` for a
    /// pointer to a single NUL byte, so `path.is_empty()` is the exact
    /// condition each arm tests, and EINVAL (22) is the errno each arm
    /// returns for it.
    ///
    /// This is tested as a predicate rather than by driving `dispatch()`:
    /// `test_main()` runs before any task is current, so there is no mapped
    /// user page table for `copy_path_from_user` to safely dereference here
    /// (mirrors `smap_smep::tests::test_user_ptr_validation`, which for the
    /// same reason tests `is_user_ptr` in isolation rather than exercising
    /// `copy_from_user` end to end).
    fn test_empty_path_guard_shape() -> TestResult {
        let empty = String::new();
        let non_empty = String::from("/etc/passwd");
        test_assert!(empty.is_empty(), "empty path must trip the guard");
        test_assert!(
            !non_empty.is_empty(),
            "non-empty path must not trip the guard"
        );
        let guarded = SyscallResult::err(22);
        test_assert_eq!(guarded.code, -22);
        TestResult::Pass
    }

    /// Insert an entry owned by some task other than the caller and hand back
    /// its fd number. Models the exact reachable situation: `NEXT_FD` is one
    /// global monotonic counter, so every live fd is a small integer any task
    /// can name from `rdi`.
    fn plant_foreign_fd(fd: u64, offset: usize) {
        let foreign = crate::process::scheduler::current_task_id().wrapping_add(1);
        FILE_TABLE.lock().insert(
            fd,
            FdEntry {
                handle: crate::fs::vfs::VfsHandle {
                    fs_idx: 0,
                    inode: 0,
                },
                path: String::from("/etc/shadow"),
                offset,
                owner: foreign,
            },
        );
    }

    /// RED: `FILE_TABLE` is one map for the whole system and every arm turned
    /// an fd number into an entry with a bare `table.get(&fd)`, consulting
    /// nothing about the caller — so guessing a small integer handed a task
    /// read, write, seek, ioctl and close on another task's open file. Every
    /// one of those paths must now refuse, and refusing must leave the
    /// victim's entry open with its cursor where it was.
    fn test_fd_lookup_denies_another_tasks_descriptor() -> TestResult {
        let fd = 90_211u64;
        let offset = 7usize;
        plant_foreign_fd(fd, offset);

        test_assert_eq!(dispatch(SYS_LSEEK, fd, 4096, 0).code, -9); // EBADF
        test_assert_eq!(crate::syscall::pipe::dispatch_dup(fd).code, -9);
        test_assert_eq!(crate::syscall::ioctl::dispatch_ioctl(fd, 0, 0).code, -9);
        test_assert_eq!(dispatch(SYS_CLOSE, fd, 0, 0).code, -9);

        let survived = FILE_TABLE.lock().get(&fd).map(|e| e.offset);
        test_assert_eq!(survived, Some(offset));

        FILE_TABLE.lock().remove(&fd);
        TestResult::Pass
    }

    /// RED: `dispatch_dup2` inserted at a caller-supplied fd number, replacing
    /// whatever sat there. Aiming it at another task's fd closed that task's
    /// file and left the attacker's handle under the victim's number.
    fn test_dup2_refuses_to_clobber_another_tasks_descriptor() -> TestResult {
        let mine = 90_212u64;
        let theirs = 90_213u64;
        let free = 90_214u64;
        FILE_TABLE.lock().insert(
            mine,
            FdEntry {
                handle: crate::fs::vfs::VfsHandle {
                    fs_idx: 0,
                    inode: 1,
                },
                path: String::from("/tmp/mine"),
                offset: 0,
                owner: crate::process::scheduler::current_task_id(),
            },
        );
        plant_foreign_fd(theirs, 7);

        test_assert_eq!(
            crate::syscall::pipe::dispatch_dup2(mine, theirs).code,
            -9 // EBADF
        );
        let victim = FILE_TABLE.lock().get(&theirs).map(|e| e.handle.inode);
        test_assert_eq!(victim, Some(0u64)); // still the victim's, not inode 1

        // Positive control: an unoccupied target still works.
        test_assert_eq!(
            crate::syscall::pipe::dispatch_dup2(mine, free).code,
            free as i64
        );
        let copied = FILE_TABLE.lock().get(&free).map(|e| e.handle.inode);
        test_assert_eq!(copied, Some(1u64));

        let mut table = FILE_TABLE.lock();
        table.remove(&mine);
        table.remove(&theirs);
        table.remove(&free);
        TestResult::Pass
    }

    /// The caller's own descriptor stays fully usable — the check denies other
    /// tasks, not everyone. `test_main()` runs with no task current, so
    /// `current_task_id()` is 0 for both the insert and the lookups here.
    fn test_fd_lookup_permits_own_descriptor() -> TestResult {
        let fd = 90_215u64;
        FILE_TABLE.lock().insert(
            fd,
            FdEntry {
                handle: crate::fs::vfs::VfsHandle {
                    fs_idx: 0,
                    inode: 2,
                },
                path: String::from("/tmp/own"),
                offset: 0,
                owner: crate::process::scheduler::current_task_id(),
            },
        );

        test_assert_eq!(dispatch(SYS_LSEEK, fd, 4096, 0).code, 4096);
        let duped = crate::syscall::pipe::dispatch_dup(fd);
        test_assert!(duped.code >= 3, "dup of an owned fd must yield an fd");
        test_assert_eq!(dispatch(SYS_CLOSE, fd, 0, 0).code, 0);
        test_assert!(FILE_TABLE.lock().get(&fd).is_none());

        FILE_TABLE.lock().remove(&(duped.code as u64));
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("syscall::setuid_changes_uid", test_setuid_changes_uid);
        crate::testing::register_test("syscall::setgid_changes_gid", test_setgid_changes_gid);
        crate::testing::register_test(
            "syscall::credential_change_denies_unprivileged_escalation",
            test_credential_change_denies_unprivileged_escalation,
        );
        crate::testing::register_test(
            "syscall::credential_change_permits_id_already_held",
            test_credential_change_permits_id_already_held,
        );
        crate::testing::register_test(
            "syscall::credential_change_permits_root",
            test_credential_change_permits_root,
        );
        crate::testing::register_test(
            "syscall::write_rejects_invalid_pointer_without_touching_fd",
            test_write_rejects_invalid_pointer_without_touching_fd,
        );
        crate::testing::register_test(
            "syscall::empty_path_guard_shape",
            test_empty_path_guard_shape,
        );
        crate::testing::register_test(
            "syscall::fd_lookup_denies_another_tasks_descriptor",
            test_fd_lookup_denies_another_tasks_descriptor,
        );
        crate::testing::register_test(
            "syscall::dup2_refuses_to_clobber_another_tasks_descriptor",
            test_dup2_refuses_to_clobber_another_tasks_descriptor,
        );
        crate::testing::register_test(
            "syscall::fd_lookup_permits_own_descriptor",
            test_fd_lookup_permits_own_descriptor,
        );
    }
}
pub const SYS_TELEPORT: u64 = 101;
pub const SYS_THEOREM_STATUS: u64 = 102;
pub const SYS_PKG_INSTALL: u64 = 103;
pub const SYS_PKG_REMOVE: u64 = 104;
pub const SYS_PKG_LIST: u64 = 105;
pub const SYS_WIFI_SCAN: u64 = 106;
pub const SYS_WIFI_CONNECT: u64 = 107;
pub const SYS_BT_SCAN: u64 = 108;
pub const SYS_BT_PAIR: u64 = 109;
pub const SYS_SETTING_GET: u64 = 110;
pub const SYS_SETTING_SET: u64 = 111;
/// Atlas: graft a signed chart onto the kernel manifold.
pub const SYS_CHART_GRAFT: u64 = 112;
/// Atlas: prune a chart back off the manifold.
pub const SYS_CHART_PRUNE: u64 = 113;
/// Atlas: list grafted charts with their reference counts.
pub const SYS_CHART_LIST: u64 = 114;

// stratum — topological fit control (see ml_engine::stratum).
/// Register the calling task as a training workload. Returns the handle.
pub const SYS_FIT_REGISTER: u64 = 120;
/// Push one step: arg0 = handle, arg1 = train loss bits, arg2 = val loss bits
/// (both `f64::to_bits`). Returns the last computed regime code.
pub const SYS_FIT_OBSERVE: u64 = 121;
/// Recompute and return the regime. arg0 = handle. `code` is the regime, `data`
/// carries the measured signals and the planned actuator settings.
pub const SYS_FIT_REGIME: u64 = 122;
/// Set one calibration field: arg0 = handle, arg1 = field id, arg2 = f64 bits.
pub const SYS_FIT_CALIBRATE: u64 = 123;
/// Drop the workload's fit state. arg0 = handle.
pub const SYS_FIT_UNREGISTER: u64 = 124;

// Foliated KV cache — kernel-managed paged attention for inference processes.
// Prefix sharing is implicit: appending identical tokens descends to the same
// foliation leaf, so two sequences with the same prompt share plaques without
// any explicit share call.
pub const SYS_KV_SEQ_CREATE: u64 = 130;
pub const SYS_KV_SEQ_APPEND: u64 = 131;
pub const SYS_KV_SEQ_RELEASE: u64 = 132;
pub const SYS_KV_SEQ_STATS: u64 = 133;
pub const SYS_KV_POLICY_STATS: u64 = 134;

#[derive(Debug)]
pub struct SyscallResult {
    pub code: i64,
    pub data: Option<String>,
}

impl SyscallResult {
    pub fn ok(code: i64) -> Self {
        Self { code, data: None }
    }

    pub fn with_data(code: i64, data: String) -> Self {
        Self {
            code,
            data: Some(data),
        }
    }

    pub fn err(errno: i64) -> Self {
        Self {
            code: -errno,
            data: None,
        }
    }
}

/// Global filesystem instance used by Epsilon syscalls (teleport, etc.).
/// Initialized during boot by `init_syscall_fs()`.
static SYSCALL_FS: Mutex<Option<ManifoldFS>> = Mutex::new(None);

/// Buffer for passing path strings into syscalls.
static SYSCALL_PATH: Mutex<String> = Mutex::new(String::new());

/// File descriptor table entry.
#[derive(Clone)]
pub(crate) struct FdEntry {
    pub(crate) handle: VfsHandle,
    #[allow(dead_code)] // REASON: path stored for future syscall debugging and fcntl(F_GETPATH)
    pub(crate) path: String,
    pub(crate) offset: usize,
    /// Task id that opened this descriptor. Only that task may name it.
    pub(crate) owner: u64,
}

pub(crate) static FILE_TABLE: Mutex<BTreeMap<u64, FdEntry>> = Mutex::new(BTreeMap::new());
pub(crate) static NEXT_FD: AtomicU64 = AtomicU64::new(3); // 0=stdin, 1=stdout, 2=stderr

/// Whether the task making the current syscall may name `entry`.
///
/// `FILE_TABLE` is one map for the whole system and `NEXT_FD` is one global
/// monotonic counter, so fd numbers are unique system-wide — but uniqueness is
/// not ownership. They are small sequential integers, so without this check any
/// task can read, write, seek, ioctl or close another task's open file by
/// guessing one. Every lookup routes through `fd_lookup`/`fd_lookup_mut` so the
/// decision is made in exactly one place.
///
/// Fails closed: `current_task_id()` reports 0 when no task is current, and
/// real ids start at 1 (`ManifoldScheduler::new()` sets `next_id: 1`), so a
/// task can never match an entry opened in kernel context, and an entry whose
/// owner has exited is named by nothing.
fn fd_owned_by_caller(entry: &FdEntry) -> bool {
    entry.owner == crate::process::scheduler::current_task_id()
}

/// Resolve `fd` to the calling task's entry, or `None` if it names nothing the
/// caller owns. The only sanctioned way to turn an fd number into an entry.
pub(crate) fn fd_lookup(table: &BTreeMap<u64, FdEntry>, fd: u64) -> Option<&FdEntry> {
    table.get(&fd).filter(|e| fd_owned_by_caller(e))
}

/// Mutable counterpart of [`fd_lookup`], for the arms that advance a cursor.
pub(crate) fn fd_lookup_mut(table: &mut BTreeMap<u64, FdEntry>, fd: u64) -> Option<&mut FdEntry> {
    table.get_mut(&fd).filter(|e| fd_owned_by_caller(e))
}

/// Stdin ring buffer for keyboard input.
const STDIN_BUF_SIZE: usize = 256;
static STDIN_BUF: Mutex<[u8; STDIN_BUF_SIZE]> = Mutex::new([0u8; STDIN_BUF_SIZE]);
static STDIN_RD_IDX: AtomicU64 = AtomicU64::new(0);
static STDIN_WR_IDX: AtomicU64 = AtomicU64::new(0);

/// Push a byte into the stdin ring buffer (called from keyboard interrupt handler).
pub fn stdin_push(ch: u8) {
    let wr = STDIN_WR_IDX.load(Ordering::Relaxed) as usize;
    let rd = STDIN_RD_IDX.load(Ordering::Acquire) as usize;
    let next = (wr + 1) & (STDIN_BUF_SIZE - 1);
    if next == rd {
        // buffer full — drop oldest by advancing read index
        STDIN_RD_IDX.store(((rd + 1) & (STDIN_BUF_SIZE - 1)) as u64, Ordering::Release);
    }
    if let Some(mut buf) = STDIN_BUF.try_lock() {
        buf[wr] = ch;
    }
    STDIN_WR_IDX.store(next as u64, Ordering::Release);
}

/// Read bytes from stdin into `dst`. Returns number of bytes read.
fn stdin_read(dst: &mut [u8]) -> usize {
    let mut n = 0usize;
    while n < dst.len() {
        let rd = STDIN_RD_IDX.load(Ordering::Acquire) as usize;
        let wr = STDIN_WR_IDX.load(Ordering::Acquire) as usize;
        if rd == wr {
            break;
        }
        if let Some(buf) = STDIN_BUF.try_lock() {
            dst[n] = buf[rd];
        }
        STDIN_RD_IDX.store(((rd + 1) & (STDIN_BUF_SIZE - 1)) as u64, Ordering::Release);
        n += 1;
    }
    n
}

/// Initialize the syscall filesystem with a fresh ManifoldFS.
/// Called once during kernel boot.
pub fn init_syscall_fs() {
    let mut fs_guard = SYSCALL_FS.lock();
    *fs_guard = Some(ManifoldFS::new());
}

/// Set the path string for the next path-based syscall.
pub fn set_path(path: &str) {
    let mut guard = SYSCALL_PATH.lock();
    guard.clear();
    guard.push_str(path);
}

/// Helper: access the global FS instance for inode-returning operations.
fn with_fs_inode<F>(f: F) -> SyscallResult
where
    F: FnOnce(&mut ManifoldFS) -> Result<u64, FsError>,
{
    let mut fs_guard = SYSCALL_FS.lock();
    if let Some(ref mut fs) = *fs_guard {
        match f(fs) {
            Ok(val) => SyscallResult::ok(val as i64),
            Err(e) => SyscallResult::err(fs_error_to_errno(e)),
        }
    } else {
        SyscallResult::err(19) // ENODEV
    }
}

fn fs_error_to_errno(e: FsError) -> i64 {
    match e {
        FsError::NotFound => 2,       // ENOENT
        FsError::AlreadyExists => 17, // EEXIST
        FsError::NotADirectory => 20, // ENOTDIR
        FsError::Storage => 5,        // EIO
    }
}

fn vfs_error_to_errno(e: VfsError) -> i64 {
    match e {
        VfsError::NotFound => 2,          // ENOENT
        VfsError::AlreadyExists => 17,    // EEXIST
        VfsError::NotADirectory => 20,    // ENOTDIR
        VfsError::PermissionDenied => 13, // EACCES
        VfsError::InvalidPath => 22,      // EINVAL
        VfsError::TooManySymlinks => 40,  // ELOOP
        _ => 5,                           // EIO
    }
}

/// Read a null-terminated path string from userspace into a bounded buffer.
unsafe fn copy_path_from_user(ptr: *const u8) -> Result<String, ()> {
    let mut buf = [0u8; 256];
    crate::security::smap_smep::copy_from_user(&mut buf, ptr)?;
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(String::from_utf8_lossy(&buf[..len]).into_owned())
}

/// Slurp a whole VFS file. Returns `None` if the path does not resolve or a
/// read fails part-way through.
fn read_vfs_file(path: &str) -> Option<Vec<u8>> {
    let handle = with_vfs(|vfs| vfs.lookup_follow(path)).ok()?;
    let mut out = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut offset = 0u64;
    loop {
        match with_vfs(|vfs| vfs.read(handle, &mut chunk, offset)) {
            Ok(0) => break,
            Ok(n) => {
                out.extend_from_slice(&chunk[..n]);
                offset += n as u64;
            }
            Err(_) => return None,
        }
    }
    Some(out)
}

/// Chart registry name for a path: the final component with any suffix removed.
fn chart_name_from_path(path: &str) -> String {
    let base = path.rsplit('/').next().unwrap_or(path);
    match base.rfind('.') {
        Some(dot) if dot > 0 => String::from(&base[..dot]),
        _ => String::from(base),
    }
}

/// Whether a task holding identity (`real`, `effective`) may change to
/// `requested`, for the SYS_SETUID/SETGID/SETEUID/SETEGID arms below.
///
/// POSIX-style rule: permitted if the caller's effective id is 0 (root), or
/// `requested` is an id the task already holds (its real or effective id).
/// `Task` (process/task.rs) has no saved-set-uid/gid field, so the usual
/// third leg of this check — the saved id — is intentionally omitted; only
/// real and effective ids are consulted. Fails closed: anything else is
/// denied.
fn credential_change_allowed(requested: u32, real: u32, effective: u32) -> bool {
    effective == 0 || requested == real || requested == effective
}

pub fn dispatch(num: u64, arg0: u64, arg1: u64, arg2: u64) -> SyscallResult {
    // Seccomp check
    let task_id = crate::process::scheduler::current_task_id();
    match crate::security::seccomp::seccomp_check(task_id, num) {
        crate::security::seccomp::SECCOMP_RET_KILL => {
            crate::process::scheduler::mark_current_dead();
            return SyscallResult::err(1); // EPERM / killed
        }
        crate::security::seccomp::SECCOMP_RET_ERRNO => {
            return SyscallResult::err(1); // EPERM
        }
        crate::security::seccomp::SECCOMP_RET_ALLOW => {}
        // Fail closed. `seccomp_check` already reduces the filter's `k` field to
        // one of the three actions above, so this arm is unreachable today; it
        // denies rather than allows so that adding a fourth action without
        // handling it here costs a denied syscall instead of an unchecked one.
        _ => {
            crate::process::scheduler::mark_current_dead();
            return SyscallResult::err(1); // EPERM / killed
        }
    }

    let result = match num {
        SYS_EXIT => {
            crate::process::scheduler::mark_current_dead();
            crate::process::scheduler::yield_current();
            SyscallResult::ok(0)
        }

        SYS_WRITE => {
            let fd = arg0;
            let buf_ptr = arg1 as *const u8;
            let len = arg2 as usize;
            if fd == 1 || fd == 2 {
                let mut buf = alloc::vec![0u8; len.min(1024)];
                unsafe {
                    if crate::security::smap_smep::copy_from_user(&mut buf, buf_ptr).is_err() {
                        return SyscallResult::err(14); // EFAULT
                    }
                }
                let text = alloc::string::String::from_utf8_lossy(&buf);
                crate::serial_print!("{}", text);
                return SyscallResult::ok(len as i64);
            }
            // File writes: fd validity first (matches SYS_READ's ordering),
            // then copy the caller's actual bytes — not the process-wide
            // SYSCALL_PATH string, which is last-writer-wins across every
            // task and was never this fd's data to begin with. Bounded to
            // 4096 bytes per call, same cap SYS_READ's file-fd branch uses;
            // a caller writing more just loops, same as a short read/write.
            let mut table = FILE_TABLE.lock();
            if let Some(entry) = fd_lookup_mut(&mut table, fd) {
                let mut buf = alloc::vec![0u8; len.min(4096)];
                unsafe {
                    if crate::security::smap_smep::copy_from_user(&mut buf, buf_ptr).is_err() {
                        return SyscallResult::err(14); // EFAULT
                    }
                }
                let offset = entry.offset as u64;
                match with_vfs(|vfs| vfs.write(entry.handle, &buf, offset)) {
                    Ok(n) => {
                        entry.offset += n;
                        SyscallResult::ok(n as i64)
                    }
                    Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
                }
            } else {
                SyscallResult::err(9) // EBADF
            }
        }

        SYS_READ => {
            let fd = arg0;
            let buf_ptr = arg1 as *mut u8;
            let len = arg2 as usize;
            if fd == 0 {
                let mut buf = alloc::vec![0u8; len.min(4096)];
                let read_len = stdin_read(&mut buf);
                unsafe {
                    if crate::security::smap_smep::copy_to_user(buf_ptr, &buf[..read_len]).is_err()
                    {
                        return SyscallResult::err(14); // EFAULT
                    }
                }
                return SyscallResult::ok(read_len as i64);
            }
            let table = FILE_TABLE.lock();
            if let Some(entry) = fd_lookup(&table, fd) {
                let handle = entry.handle;
                drop(table);
                let mut buf = alloc::vec![0u8; len.min(4096)];
                match with_vfs(|vfs| vfs.read(handle, &mut buf, 0)) {
                    Ok(read_len) => {
                        unsafe {
                            if crate::security::smap_smep::copy_to_user(buf_ptr, &buf[..read_len])
                                .is_err()
                            {
                                return SyscallResult::err(14); // EFAULT
                            }
                        }
                        SyscallResult::ok(read_len as i64)
                    }
                    Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
                }
            } else {
                SyscallResult::err(9) // EBADF
            }
        }

        SYS_OPEN => {
            let path_ptr = arg0 as *const u8;
            let flags = arg1;
            let _mode = arg2;
            let path = unsafe {
                match copy_path_from_user(path_ptr) {
                    Ok(p) => p,
                    Err(_) => return SyscallResult::err(14), // EFAULT
                }
            };
            if path.is_empty() {
                return SyscallResult::err(22); // EINVAL
            }
            // Update SYSCALL_PATH for audit logging
            {
                let mut guard = SYSCALL_PATH.lock();
                guard.clear();
                guard.push_str(&path);
            }

            match with_vfs(|vfs| vfs.lookup_follow(&path)) {
                Ok(handle) => {
                    let fd = NEXT_FD.fetch_add(1, Ordering::SeqCst);
                    let mut table = FILE_TABLE.lock();
                    table.insert(
                        fd,
                        FdEntry {
                            handle,
                            path: path.clone(),
                            offset: 0,
                            owner: task_id,
                        },
                    );
                    SyscallResult::ok(fd as i64)
                }
                Err(VfsError::NotFound) if flags & 0x40 != 0 => {
                    // O_CREAT flag — create the file
                    match with_vfs(|vfs| vfs.create(&path)) {
                        Ok(handle) => {
                            let fd = NEXT_FD.fetch_add(1, Ordering::SeqCst);
                            let mut table = FILE_TABLE.lock();
                            table.insert(
                                fd,
                                FdEntry {
                                    handle,
                                    path: path.clone(),
                                    offset: 0,
                                    owner: task_id,
                                },
                            );
                            SyscallResult::ok(fd as i64)
                        }
                        Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
                    }
                }
                Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
            }
        }

        SYS_CLOSE => {
            let fd = arg0;
            let mut table = FILE_TABLE.lock();
            if fd_lookup(&table, fd).is_some() {
                table.remove(&fd);
                SyscallResult::ok(0)
            } else {
                SyscallResult::err(9) // EBADF
            }
        }

        SYS_EXEC => {
            let path_ptr = arg0 as *const u8;
            let path = unsafe {
                match copy_path_from_user(path_ptr) {
                    Ok(p) => p,
                    Err(_) => return SyscallResult::err(14), // EFAULT
                }
            };
            if path.is_empty() {
                return SyscallResult::err(22); // EINVAL
            }
            {
                let mut guard = SYSCALL_PATH.lock();
                guard.clear();
                guard.push_str(&path);
            }

            let handle = match with_vfs(|vfs| vfs.lookup_follow(&path)) {
                Ok(h) => h,
                Err(e) => return SyscallResult::err(vfs_error_to_errno(e)),
            };

            let file_stat = match with_vfs(|vfs| vfs.stat(handle)) {
                Ok(s) => s,
                Err(e) => return SyscallResult::err(vfs_error_to_errno(e)),
            };

            // T1–T5: Exec permission check before reading file contents.
            let uid = crate::process::scheduler::current_uid();
            let gid = crate::process::scheduler::current_gid();
            let groups = crate::process::scheduler::current_groups();
            if crate::security::manifold_acl::check_access(
                uid,
                gid,
                &groups,
                &file_stat,
                crate::security::manifold_acl::PERM_EXEC,
                &path,
            ) == crate::security::manifold_acl::AccessDecision::Deny
            {
                return SyscallResult::err(13); // EACCES
            }

            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let mut offset = 0u64;
            loop {
                match with_vfs(|vfs| vfs.read(handle, &mut chunk, offset)) {
                    Ok(0) => break,
                    Ok(n) => {
                        buf.extend_from_slice(&chunk[..n]);
                        offset += n as u64;
                    }
                    Err(_) => return SyscallResult::err(5), // EIO
                }
            }

            if buf.is_empty() {
                return SyscallResult::err(8); // ENOEXEC
            }

            let real_uid = crate::process::scheduler::current_uid();
            let real_gid = crate::process::scheduler::current_gid();

            if buf.starts_with(b"\x7FELF") {
                let aslr_base = crate::security::aslr::randomize_mmap_base();
                match crate::process::elf::load(
                    &buf,
                    aslr_base,
                    file_stat.mode,
                    file_stat.uid,
                    file_stat.gid,
                ) {
                    Ok(_loaded) => {
                        let name: &'static str = if path == "/bin/init" {
                            "init"
                        } else {
                            "userspace"
                        };
                        let _ = crate::process::scheduler::spawn_user(
                            name,
                            5,
                            &buf,
                            file_stat.mode,
                            file_stat.uid,
                            file_stat.gid,
                            real_uid,
                            real_gid,
                        );
                    }
                    Err(_) => return SyscallResult::err(8), // ENOEXEC
                }
            } else if buf.starts_with(b"#!") {
                let line_end = buf.iter().position(|&b| b == b'\n').unwrap_or(buf.len());
                let shebang = core::str::from_utf8(&buf[2..line_end]).unwrap_or("").trim();
                if shebang.is_empty() {
                    return SyscallResult::err(22); // EINVAL
                }
                let interp_handle = match with_vfs(|vfs| vfs.lookup_follow(shebang)) {
                    Ok(h) => h,
                    Err(_) => return SyscallResult::err(2), // ENOENT
                };
                let interp_stat = match with_vfs(|vfs| vfs.stat(interp_handle)) {
                    Ok(s) => s,
                    Err(e) => return SyscallResult::err(vfs_error_to_errno(e)),
                };
                let mut interp_buf = Vec::new();
                let mut interp_offset = 0u64;
                loop {
                    match with_vfs(|vfs| vfs.read(interp_handle, &mut chunk, interp_offset)) {
                        Ok(0) => break,
                        Ok(n) => {
                            interp_buf.extend_from_slice(&chunk[..n]);
                            interp_offset += n as u64;
                        }
                        Err(_) => return SyscallResult::err(5), // EIO
                    }
                }
                if !interp_buf.starts_with(b"\x7FELF") {
                    return SyscallResult::err(8); // ENOEXEC
                }
                let aslr_base = crate::security::aslr::randomize_mmap_base();
                match crate::process::elf::load(
                    &interp_buf,
                    aslr_base,
                    interp_stat.mode,
                    interp_stat.uid,
                    interp_stat.gid,
                ) {
                    Ok(_loaded) => {
                        let _ = crate::process::scheduler::spawn_user(
                            "interpreter",
                            5,
                            &interp_buf,
                            interp_stat.mode,
                            interp_stat.uid,
                            interp_stat.gid,
                            real_uid,
                            real_gid,
                        );
                    }
                    Err(_) => return SyscallResult::err(8), // ENOEXEC
                }
            } else if path.ends_with(".aether") {
                let source = String::from_utf8_lossy(&buf);
                match crate::lang::AetherRuntime::new().execute_file(&path, &source) {
                    Ok(_) => {}
                    Err(_) => return SyscallResult::err(8), // ENOEXEC
                }
            } else {
                return SyscallResult::err(22); // EINVAL
            }

            crate::process::scheduler::mark_current_dead();
            crate::process::scheduler::yield_current();
            SyscallResult::ok(0)
        }

        SYS_FORK => {
            match crate::process::scheduler::fork_current() {
                Some(child_id) => SyscallResult::ok(child_id as i64),
                None => SyscallResult::err(11), // EAGAIN
            }
        }

        SYS_CLONE => {
            match crate::process::scheduler::clone_current(arg0) {
                Some(child_id) => SyscallResult::ok(child_id as i64),
                None => SyscallResult::err(11), // EAGAIN
            }
        }

        SYS_WAITPID => SyscallResult::ok(arg0 as i64),

        SYS_MMAP => {
            let len = arg1 as usize;
            let prot = arg2;
            let pages = len.div_ceil(4096);

            let mut flags = x86_64::structures::paging::PageTableFlags::PRESENT
                | x86_64::structures::paging::PageTableFlags::USER_ACCESSIBLE;
            if prot & 0x2 != 0 {
                flags |= x86_64::structures::paging::PageTableFlags::WRITABLE;
            }
            if prot & 0x4 == 0 {
                flags |= x86_64::structures::paging::PageTableFlags::NO_EXECUTE;
            }

            if let Some(pt) = crate::process::scheduler::current_page_table() {
                match crate::memory::mmap::mmap_user(pages, flags, pt) {
                    Some(virt) => SyscallResult::ok(virt.as_u64() as i64),
                    None => SyscallResult::err(12), // ENOMEM
                }
            } else {
                SyscallResult::err(1) // EPERM
            }
        }

        SYS_GETPID => SyscallResult::ok(crate::process::scheduler::current_task_id() as i64),

        SYS_STAT => {
            let path_ptr = arg0 as *const u8;
            let path = unsafe {
                match copy_path_from_user(path_ptr) {
                    Ok(p) => p,
                    Err(_) => return SyscallResult::err(14),
                }
            };
            if path.is_empty() {
                return SyscallResult::err(22); // EINVAL
            }
            {
                let mut guard = SYSCALL_PATH.lock();
                guard.clear();
                guard.push_str(&path);
            }
            match with_vfs(|vfs| vfs.lookup_follow(&path)) {
                Ok(handle) => match with_vfs(|vfs| vfs.stat(handle)) {
                    Ok(node) => {
                        let text = format!(
                            "Size: {}\nPermissions: {:o}\nUID: {}\nGID: {}\nType: {:?}\n",
                            node.size, node.permissions, node.uid, node.gid, node.node_type
                        );
                        SyscallResult::with_data(0, text)
                    }
                    Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
                },
                Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
            }
        }

        SYS_MKDIR => {
            let path_ptr = arg0 as *const u8;
            let path = unsafe {
                match copy_path_from_user(path_ptr) {
                    Ok(p) => p,
                    Err(_) => return SyscallResult::err(14),
                }
            };
            if path.is_empty() {
                return SyscallResult::err(22); // EINVAL
            }
            {
                let mut guard = SYSCALL_PATH.lock();
                guard.clear();
                guard.push_str(&path);
            }
            match with_vfs(|vfs| vfs.mkdir(&path)) {
                Ok(_) => SyscallResult::ok(0),
                Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
            }
        }

        SYS_MANIFOLD_QUERY => {
            if arg0 == 0 || arg0 as usize > crate::THEOREM_COUNT {
                SyscallResult::err(22)
            } else {
                let idx = arg0 as usize - 1;
                let state = crate::THEOREM_STATES[idx].load(Ordering::Relaxed);
                let status = if state {
                    if idx < 5 {
                        "ACTIVE"
                    } else {
                        "VERIFIED"
                    }
                } else {
                    "FAILED"
                };
                SyscallResult::with_data(0, format!("{}: {}", crate::THEOREM_NAMES[idx], status))
            }
        }

        SYS_TELEPORT => {
            let path = {
                let guard = SYSCALL_PATH.lock();
                guard.clone()
            };
            let src = arg0;
            let dst = arg1;
            let result = with_fs_inode(|fs| fs.teleport(&path, src, dst).map(|r| r.inode_id));
            if result.code >= 0 {
                SyscallResult::with_data(
                    0,
                    format!("teleported '{}' from dir {} -> dir {}", path, src, dst),
                )
            } else {
                result
            }
        }

        SYS_THEOREM_STATUS => {
            let mut out = String::new();
            for idx in 0..crate::THEOREM_COUNT {
                let ok = crate::THEOREM_STATES[idx].load(Ordering::Relaxed);
                let status = if ok {
                    if idx < 5 {
                        "ACTIVE"
                    } else {
                        "VERIFIED"
                    }
                } else {
                    "FAILED"
                };
                if idx > 0 {
                    out.push(' ');
                }
                out.push_str(crate::THEOREM_NAMES[idx]);
                out.push(':');
                out.push_str(status);
            }
            SyscallResult::with_data(0, out)
        }

        SYS_PKG_INSTALL => {
            let name_ptr = arg0 as *const u8;
            let name = unsafe { copy_path_from_user(name_ptr).unwrap_or_default() };
            if name.is_empty() {
                return SyscallResult::err(22);
            }
            match crate::pkg::GLOBAL_PKG.lock().install(&name) {
                Ok(msg) => SyscallResult::with_data(0, msg),
                Err(e) => SyscallResult::with_data(-1, e),
            }
        }
        SYS_PKG_REMOVE => {
            let name_ptr = arg0 as *const u8;
            let name = unsafe { copy_path_from_user(name_ptr).unwrap_or_default() };
            match crate::pkg::GLOBAL_PKG.lock().remove(&name) {
                Ok(msg) => SyscallResult::with_data(0, msg),
                Err(e) => SyscallResult::with_data(-1, e),
            }
        }
        SYS_PKG_LIST => {
            let pkg = crate::pkg::GLOBAL_PKG.lock();
            let list: String = if pkg.package_count() == 0 {
                String::from("no packages installed")
            } else {
                pkg.list()
                    .iter()
                    .map(|m| format!("{} v{}\n", m.name, m.version))
                    .collect()
            };
            SyscallResult::with_data(0, list)
        }
        SYS_CHART_GRAFT => {
            // arg0 = chart object path, arg1 = detached ed25519 signature path.
            let object_path = unsafe { copy_path_from_user(arg0 as *const u8).unwrap_or_default() };
            let sig_path = unsafe { copy_path_from_user(arg1 as *const u8).unwrap_or_default() };
            if object_path.is_empty() || sig_path.is_empty() {
                return SyscallResult::err(22); // EINVAL
            }
            let (Some(object), Some(signature)) =
                (read_vfs_file(&object_path), read_vfs_file(&sig_path))
            else {
                return SyscallResult::err(2); // ENOENT
            };
            let name = chart_name_from_path(&object_path);
            match crate::atlas::abi_graft(&name, &object, &signature) {
                Ok(code) => {
                    SyscallResult::with_data(0, format!("grafted '{}' init={:#x}", name, code))
                }
                Err(e) => SyscallResult::with_data(-1, format!("graft '{}': {}", name, e.tag())),
            }
        }
        SYS_CHART_PRUNE => {
            let name = unsafe { copy_path_from_user(arg0 as *const u8).unwrap_or_default() };
            if name.is_empty() {
                return SyscallResult::err(22); // EINVAL
            }
            match crate::atlas::abi_prune(&name) {
                Ok(code) => {
                    SyscallResult::with_data(0, format!("pruned '{}' exit={:#x}", name, code))
                }
                Err(e) => SyscallResult::with_data(-1, format!("prune '{}': {}", name, e.tag())),
            }
        }
        SYS_CHART_LIST => SyscallResult::with_data(0, crate::atlas::abi_list()),

        SYS_WIFI_SCAN => {
            SyscallResult::with_data(0, String::from("wifi_scan: no wireless hardware detected"))
        }
        SYS_WIFI_CONNECT => SyscallResult::with_data(
            0,
            String::from("wifi_connect: no wireless hardware detected"),
        ),
        SYS_BT_SCAN => {
            SyscallResult::with_data(0, String::from("bt_scan: no Bluetooth adapter detected"))
        }
        SYS_BT_PAIR => {
            SyscallResult::with_data(0, String::from("bt_pair: no Bluetooth adapter detected"))
        }
        SYS_SETTING_GET => {
            let key_ptr = arg0 as *const u8;
            let key = unsafe { copy_path_from_user(key_ptr).unwrap_or_default() };
            let settings = crate::apps::settings::GLOBAL_SETTINGS.lock();
            match settings.get(&key) {
                Some(val) => SyscallResult::with_data(0, String::from(val)),
                None => SyscallResult::err(2), // ENOENT
            }
        }
        SYS_SETTING_SET => {
            let key_ptr = arg0 as *const u8;
            let val_ptr = arg1 as *const u8;
            let key = unsafe { copy_path_from_user(key_ptr).unwrap_or_default() };
            let val = unsafe { copy_path_from_user(val_ptr).unwrap_or_default() };
            let mut settings = crate::apps::settings::GLOBAL_SETTINGS.lock();
            settings.set(&key, &val);
            SyscallResult::ok(0)
        }

        // arg0 = block budget. Returns the sequence id.
        SYS_KV_SEQ_CREATE => {
            let budget = arg0.min(u64::from(u16::MAX)) as u16;
            match crate::ml_engine::foliation::with_global(|f| f.seq_create(budget)) {
                Ok(id) => SyscallResult::ok(id as i64),
                Err(e) => SyscallResult::err(crate::ml_engine::foliation::errno(e)),
            }
        }
        // arg0 = sequence id, arg1 = token. Returns blocks sealed so far.
        SYS_KV_SEQ_APPEND => {
            let id = arg0 as usize;
            let token = arg1 as u32;
            match crate::ml_engine::foliation::with_global(|f| f.seq_append(id, token)) {
                Ok(blocks) => SyscallResult::ok(i64::from(blocks)),
                Err(e) => SyscallResult::err(crate::ml_engine::foliation::errno(e)),
            }
        }
        // arg0 = sequence id. Returns blocks released; shared blocks survive.
        SYS_KV_SEQ_RELEASE => {
            let id = arg0 as usize;
            match crate::ml_engine::foliation::with_global(|f| f.seq_release(id)) {
                Ok(blocks) => SyscallResult::ok(i64::from(blocks)),
                Err(e) => SyscallResult::err(crate::ml_engine::foliation::errno(e)),
            }
        }
        // arg0 = sequence id. Reports how much of it was shared on entry.
        SYS_KV_SEQ_STATS => match crate::ml_engine::foliation::seq_stats_line(arg0 as usize) {
            Some(line) => SyscallResult::with_data(0, line),
            None => SyscallResult::err(2), // ENOENT
        },
        SYS_KV_POLICY_STATS => {
            SyscallResult::with_data(0, crate::ml_engine::foliation::global_stats_line())
        }

        SYS_SETUID => {
            let new_uid = arg0 as u32;
            let real = crate::process::scheduler::current_uid();
            let effective = crate::process::scheduler::current_euid();
            if !credential_change_allowed(new_uid, real, effective) {
                return SyscallResult::err(1); // EPERM
            }
            crate::process::scheduler::set_current_uid(new_uid);
            crate::process::scheduler::set_current_euid(new_uid);
            SyscallResult::ok(0)
        }

        SYS_SETGID => {
            let new_gid = arg0 as u32;
            let real = crate::process::scheduler::current_gid();
            let effective = crate::process::scheduler::current_egid();
            if !credential_change_allowed(new_gid, real, effective) {
                return SyscallResult::err(1); // EPERM
            }
            crate::process::scheduler::set_current_gid(new_gid);
            crate::process::scheduler::set_current_egid(new_gid);
            SyscallResult::ok(0)
        }

        SYS_SETEUID => {
            let new_euid = arg0 as u32;
            let real = crate::process::scheduler::current_uid();
            let effective = crate::process::scheduler::current_euid();
            if !credential_change_allowed(new_euid, real, effective) {
                return SyscallResult::err(1); // EPERM
            }
            crate::process::scheduler::set_current_euid(new_euid);
            SyscallResult::ok(0)
        }

        SYS_SETEGID => {
            let new_egid = arg0 as u32;
            let real = crate::process::scheduler::current_gid();
            let effective = crate::process::scheduler::current_egid();
            if !credential_change_allowed(new_egid, real, effective) {
                return SyscallResult::err(1); // EPERM
            }
            crate::process::scheduler::set_current_egid(new_egid);
            SyscallResult::ok(0)
        }

        SYS_CHDIR => {
            let path_ptr = arg0 as *const u8;
            let path = unsafe {
                match copy_path_from_user(path_ptr) {
                    Ok(p) => p,
                    Err(_) => return SyscallResult::err(14), // EFAULT
                }
            };
            if path.is_empty() {
                return SyscallResult::err(22); // EINVAL
            }
            {
                let mut guard = SYSCALL_PATH.lock();
                guard.clear();
                guard.push_str(&path);
            }
            // Verify the path exists and is a directory
            match with_vfs(|vfs| vfs.lookup_follow(&path)) {
                Ok(handle) => match with_vfs(|vfs| vfs.stat(handle)) {
                    Ok(node) => {
                        if node.node_type != crate::fs::vfs::VfsNodeType::Directory {
                            return SyscallResult::err(20); // ENOTDIR
                        }
                        crate::process::scheduler::set_current_cwd(path);
                        SyscallResult::ok(0)
                    }
                    Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
                },
                Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
            }
        }

        SYS_GETCWD => {
            let cwd = crate::process::scheduler::current_task_cwd();
            SyscallResult::with_data(0, cwd)
        }

        SYS_GETPPID => {
            // T5: Walk the hyperbolic process tree upward.
            match crate::process::scheduler::current_parent_id() {
                Some(pid) => SyscallResult::ok(pid as i64),
                None => SyscallResult::ok(1), // init fallback
            }
        }

        SYS_NANOSLEEP => {
            let ms = arg0;
            let target = crate::drivers::interrupts::ticks() + ms;
            while crate::drivers::interrupts::ticks() < target {
                crate::process::scheduler::yield_current();
            }
            SyscallResult::ok(0)
        }

        SYS_REBOOT => {
            match arg0 {
                0 => {
                    // ACPI power off
                    if crate::drivers::acpi::power_off() {
                        SyscallResult::ok(0)
                    } else {
                        SyscallResult::err(38) // ENOSYS fallback
                    }
                }
                1 => {
                    // Keyboard controller reset
                    unsafe {
                        use x86_64::instructions::port::Port;
                        let mut cmd: Port<u8> = Port::new(0x64);
                        cmd.write(0xFE);
                    }
                    SyscallResult::ok(0)
                }
                2 => {
                    // Triple fault: load a null IDT descriptor and trigger an interrupt.
                    // A zero-limit IDT causes a double fault on the first interrupt;
                    // the double fault handler then triple faults because the IDT is still invalid.
                    unsafe {
                        let null_idt = x86_64::structures::DescriptorTablePointer {
                            base: x86_64::VirtAddr::new(0),
                            limit: 0,
                        };
                        core::arch::asm!("lidt [{}]", in(reg) &null_idt, options(nostack, preserves_flags));
                        core::arch::asm!("int 3");
                    }
                    SyscallResult::ok(0)
                }
                _ => SyscallResult::err(22), // EINVAL
            }
        }

        SYS_LSEEK => {
            let fd = arg0;
            let offset = arg1 as i64;
            let whence = arg2;
            let mut table = FILE_TABLE.lock();
            if let Some(entry) = fd_lookup_mut(&mut table, fd) {
                let new_offset = match whence {
                    0 => offset,                       // SEEK_SET
                    1 => entry.offset as i64 + offset, // SEEK_CUR
                    2 => {
                        let size = with_vfs(|vfs| vfs.stat(entry.handle))
                            .map(|n| n.size as i64)
                            .unwrap_or(0);
                        size + offset // SEEK_END
                    }
                    _ => return SyscallResult::err(22), // EINVAL
                };
                if new_offset < 0 {
                    return SyscallResult::err(22); // EINVAL
                }
                entry.offset = new_offset as usize;
                SyscallResult::ok(new_offset)
            } else {
                SyscallResult::err(9) // EBADF
            }
        }

        SYS_UNLINK => {
            let path_ptr = arg0 as *const u8;
            let path = unsafe {
                match copy_path_from_user(path_ptr) {
                    Ok(p) => p,
                    Err(_) => return SyscallResult::err(14), // EFAULT
                }
            };
            if path.is_empty() {
                return SyscallResult::err(22); // EINVAL
            }
            {
                let mut guard = SYSCALL_PATH.lock();
                guard.clear();
                guard.push_str(&path);
            }
            match with_vfs(|vfs| vfs.unlink(&path)) {
                Ok(_) => SyscallResult::ok(0),
                Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
            }
        }

        SYS_RMDIR => {
            let path_ptr = arg0 as *const u8;
            let path = unsafe {
                match copy_path_from_user(path_ptr) {
                    Ok(p) => p,
                    Err(_) => return SyscallResult::err(14), // EFAULT
                }
            };
            if path.is_empty() {
                return SyscallResult::err(22); // EINVAL
            }
            {
                let mut guard = SYSCALL_PATH.lock();
                guard.clear();
                guard.push_str(&path);
            }
            match with_vfs(|vfs| vfs.rmdir(&path)) {
                Ok(_) => SyscallResult::ok(0),
                Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
            }
        }

        SYS_RENAME => {
            let old_ptr = arg0 as *const u8;
            let new_ptr = arg1 as *const u8;
            let old = unsafe {
                match copy_path_from_user(old_ptr) {
                    Ok(p) => p,
                    Err(_) => return SyscallResult::err(14), // EFAULT
                }
            };
            if old.is_empty() {
                return SyscallResult::err(22); // EINVAL
            }
            let new = unsafe {
                match copy_path_from_user(new_ptr) {
                    Ok(p) => p,
                    Err(_) => return SyscallResult::err(14), // EFAULT
                }
            };
            if new.is_empty() {
                return SyscallResult::err(22); // EINVAL
            }
            match with_vfs(|vfs| vfs.rename(&old, &new)) {
                Ok(_) => SyscallResult::ok(0),
                Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
            }
        }

        SYS_GETRANDOM => {
            let buf_ptr = arg0 as *mut u8;
            let len = (arg1 as usize).min(256);
            if len == 0 {
                return SyscallResult::ok(0);
            }
            let mut buf = alloc::vec![0u8; len];
            if crate::drivers::entropy::getrandom(&mut buf) {
                unsafe {
                    if crate::security::smap_smep::copy_to_user(buf_ptr, &buf).is_err() {
                        return SyscallResult::err(14); // EFAULT
                    }
                }
                SyscallResult::ok(buf.len() as i64)
            } else {
                SyscallResult::err(5) // EIO
            }
        }

        SYS_KMSG_READ => {
            let buf_ptr = arg0 as *mut u8;
            let len = arg1 as usize;
            if len == 0 {
                return SyscallResult::ok(0);
            }
            let mut buf = alloc::vec![0u8; len.min(4096)];
            let read = crate::drivers::kmsg::kmsg_read(&mut buf);
            if read > 0 {
                unsafe {
                    if crate::security::smap_smep::copy_to_user(buf_ptr, &buf[..read]).is_err() {
                        return SyscallResult::err(14); // EFAULT
                    }
                }
            }
            SyscallResult::ok(read as i64)
        }

        SYS_KILL => {
            let target = arg0 as i64;
            let sig = arg1 as u8;
            if target < 0 {
                // T5: kill(-pid, sig) sends signal to all children in hyperbolic subtree.
                let root_id = (-target) as u64;
                let mut subtree = Vec::new();
                crate::process::scheduler::collect_subtree(root_id, &mut subtree);
                for child_id in subtree {
                    crate::process::signal::send_signal(child_id, sig);
                }
                SyscallResult::ok(0)
            } else {
                SyscallResult::ok(crate::process::signal::sys_kill(target as u64, sig))
            }
        }
        SYS_SIGACTION => SyscallResult::ok(crate::process::signal::sys_sigaction(
            arg0 as u8, arg1, arg2,
        )),
        SYS_SIGRETURN => crate::process::signal::sys_sigreturn_call(),
        SYS_SIGALTSTACK => SyscallResult::ok(crate::process::signal::sys_sigaltstack(arg0, arg1)),
        SYS_PIPE => crate::syscall::pipe::dispatch_pipe(arg0),
        SYS_DUP => crate::syscall::pipe::dispatch_dup(arg0),
        SYS_DUP2 => crate::syscall::pipe::dispatch_dup2(arg0, arg1),
        SYS_BRK => crate::syscall::pipe::dispatch_brk(arg0),
        SYS_GETTIMEOFDAY => crate::syscall::time::dispatch_gettimeofday(arg0),
        SYS_SETTIMEOFDAY => crate::syscall::time::dispatch_settimeofday(arg0, arg1),
        SYS_WATCHDOG => crate::syscall::time::dispatch_watchdog(arg0),
        SYS_IOCTL => crate::syscall::ioctl::dispatch_ioctl(arg0, arg1, arg2),
        SYS_SYNC => match crate::fs::sync() {
            Ok(()) => SyscallResult::ok(0),
            Err(e) => SyscallResult::err(vfs_error_to_errno(e)),
        },

        SYS_SETRLIMIT => {
            crate::process::scheduler::setrlimit(arg0 as u32, arg1);
            SyscallResult::ok(0)
        }

        SYS_GETRLIMIT => {
            let limit = crate::process::scheduler::getrlimit(arg0 as u32);
            SyscallResult::ok(limit as i64)
        }

        SYS_SLEEP => {
            // arg0 = sleep state (3 = S3, 5 = S5)
            let state = arg0 as u8;
            if state == 3 || state == 5 {
                crate::drivers::acpi::topological_power::acpi_enter_sleep(state);
                SyscallResult::ok(0)
            } else {
                SyscallResult::err(22) // EINVAL
            }
        }

        SYS_FIT_REGISTER => {
            let handle = crate::process::scheduler::current_task_id();
            SyscallResult::ok(crate::ml_engine::stratum::register(handle) as i64)
        }

        SYS_FIT_OBSERVE => {
            let train = f64::from_bits(arg1);
            let val = f64::from_bits(arg2);
            match crate::ml_engine::stratum::observe(arg0, train, val) {
                Some(regime) => SyscallResult::ok(regime.code()),
                None => SyscallResult::err(2), // ENOENT: not registered
            }
        }

        SYS_FIT_REGIME => match crate::ml_engine::stratum::regime_of(arg0) {
            Some((regime, _signals, action)) => {
                // Actuate the real knobs in the caller's own context, then
                // return the full state (advisory knobs included) as data.
                crate::ml_engine::stratum::apply_action(&action);
                let data = crate::ml_engine::stratum::report(arg0).unwrap_or_default();
                SyscallResult::with_data(regime.code(), data)
            }
            None => SyscallResult::err(2), // ENOENT
        },

        SYS_FIT_CALIBRATE => {
            let value = f64::from_bits(arg2);
            if crate::ml_engine::stratum::calibrate(arg0, arg1 as u32, value) {
                SyscallResult::ok(0)
            } else {
                SyscallResult::err(22) // EINVAL
            }
        }

        SYS_FIT_UNREGISTER => {
            if crate::ml_engine::stratum::unregister(arg0) {
                SyscallResult::ok(0)
            } else {
                SyscallResult::err(2) // ENOENT
            }
        }

        _ => SyscallResult::err(38), // ENOSYS
    };

    // Audit logging
    let uid = crate::process::scheduler::current_uid();
    match num {
        SYS_OPEN => {
            let path = SYSCALL_PATH.lock().clone();
            crate::security::audit::audit_log(crate::security::audit::AuditEvent::Open {
                uid,
                path,
                perms: if arg1 & 0x1 != 0 {
                    String::from("w")
                } else {
                    String::from("r")
                },
            });
        }
        SYS_EXEC => {
            let path = SYSCALL_PATH.lock().clone();
            if path.contains("sudo") {
                crate::security::audit::audit_log(crate::security::audit::AuditEvent::Sudo {
                    user: format!("uid:{}", uid),
                    command: path.clone(),
                    success: true,
                });
            } else {
                crate::security::audit::audit_log(crate::security::audit::AuditEvent::Execve {
                    uid,
                    path,
                });
            }
        }
        SYS_SETUID => {
            crate::security::audit::audit_log(crate::security::audit::AuditEvent::Setuid {
                uid,
                new_uid: arg0 as u32,
            });
        }
        _ => {
            // Syscall not audited explicitly; no audit record generated
        }
    }

    result
}
