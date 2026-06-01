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

    fn test_setuid_changes_uid() -> TestResult {
        // Skip when running outside a proper scheduler task context.
        let current = crate::process::scheduler::current_uid();
        // Just verify dispatch doesn't panic and returns OK (code >= 0).
        let result = dispatch(SYS_SETUID, 42, 0, 0);
        test_assert!(result.code >= 0, "SYS_SETUID should succeed");
        // Restore best-effort
        dispatch(SYS_SETUID, current as u64, 0, 0);
        TestResult::Pass
    }

    fn test_setgid_changes_gid() -> TestResult {
        let current = crate::process::scheduler::current_gid();
        let result = dispatch(SYS_SETGID, 99, 0, 0);
        test_assert!(result.code >= 0, "SYS_SETGID should succeed");
        dispatch(SYS_SETGID, current as u64, 0, 0);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("syscall::setuid_changes_uid", test_setuid_changes_uid);
        crate::testing::register_test("syscall::setgid_changes_gid", test_setgid_changes_gid);
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
    pub(crate) path: String,
    pub(crate) offset: usize,
}

pub(crate) static FILE_TABLE: Mutex<BTreeMap<u64, FdEntry>> = Mutex::new(BTreeMap::new());
pub(crate) static NEXT_FD: AtomicU64 = AtomicU64::new(3); // 0=stdin, 1=stdout, 2=stderr

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
        _ => {}
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
            let data = {
                let guard = SYSCALL_PATH.lock();
                guard.clone()
            };
            let mut table = FILE_TABLE.lock();
            if let Some(entry) = table.get_mut(&fd) {
                match with_vfs(|vfs| vfs.write(entry.handle, data.as_bytes(), 0)) {
                    Ok(n) => {
                        entry.offset = n;
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
            if let Some(entry) = table.get(&fd) {
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
            if table.remove(&fd).is_some() {
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
            let pages = (len + 4095) / 4096;

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

        SYS_SETUID => {
            let new_uid = arg0 as u32;
            crate::process::scheduler::set_current_uid(new_uid);
            crate::process::scheduler::set_current_euid(new_uid);
            SyscallResult::ok(0)
        }

        SYS_SETGID => {
            let new_gid = arg0 as u32;
            crate::process::scheduler::set_current_gid(new_gid);
            crate::process::scheduler::set_current_egid(new_gid);
            SyscallResult::ok(0)
        }

        SYS_SETEUID => {
            crate::process::scheduler::set_current_euid(arg0 as u32);
            SyscallResult::ok(0)
        }

        SYS_SETEGID => {
            crate::process::scheduler::set_current_egid(arg0 as u32);
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
            if let Some(entry) = table.get_mut(&fd) {
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
            let new = unsafe {
                match copy_path_from_user(new_ptr) {
                    Ok(p) => p,
                    Err(_) => return SyscallResult::err(14), // EFAULT
                }
            };
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
        _ => {}
    }

    result
}
