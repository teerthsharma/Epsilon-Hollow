// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Syscall dispatch table. Standard POSIX-like + Epsilon extensions.

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
pub const SYS_SETUID: u64 = 12;
pub const SYS_SETGID: u64 = 13;

// Epsilon extensions
pub const SYS_MANIFOLD_QUERY: u64 = 100;

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    fn test_setuid_changes_uid() -> TestResult {
        let old = crate::process::scheduler::current_uid();
        dispatch(SYS_SETUID, 42, 0, 0);
        let new = crate::process::scheduler::current_uid();
        test_assert_eq!(new, 42);
        // Restore
        dispatch(SYS_SETUID, old as u64, 0, 0);
        TestResult::Pass
    }

    fn test_setgid_changes_gid() -> TestResult {
        let old = crate::process::scheduler::current_gid();
        dispatch(SYS_SETGID, 99, 0, 0);
        let new = crate::process::scheduler::current_gid();
        test_assert_eq!(new, 99);
        // Restore
        dispatch(SYS_SETGID, old as u64, 0, 0);
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
struct FdEntry {
    handle: VfsHandle,
    path: String,
    offset: usize,
}

static FILE_TABLE: Mutex<BTreeMap<u64, FdEntry>> = Mutex::new(BTreeMap::new());
static NEXT_FD: AtomicU64 = AtomicU64::new(3); // 0=stdin, 1=stdout, 2=stderr

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
                return SyscallResult::err(5); // EIO — stdin not implemented
            }
            let mut table = FILE_TABLE.lock();
            if let Some(entry) = table.get(&fd) {
                let handle = entry.handle;
                drop(table);
                let mut buf = alloc::vec![0u8; len.min(4096)];
                match with_vfs(|vfs| vfs.read(handle, &mut buf, 0)) {
                    Ok(read_len) => {
                        unsafe {
                            if crate::security::smap_smep::copy_to_user(buf_ptr, &buf[..read_len]).is_err() {
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
                    // O_CREAT flag simulated — create the file
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
            // Simplified exec — return ENOSYS for now.
            SyscallResult::err(38)
        }

        SYS_FORK => {
            let parent = crate::process::scheduler::current_task_id();
            SyscallResult::ok((parent + 1) as i64)
        }

        SYS_WAITPID => {
            SyscallResult::ok(arg0 as i64)
        }

        SYS_MMAP => {
            let addr = arg0;
            let len = arg1 as usize;
            let _prot = arg2;
            let pages = (len + 4095) / 4096;
            let map_addr = if addr != 0 { addr } else { 0x0000_0000_0010_0000 };

            if let Some(pt) = crate::process::scheduler::current_page_table() {
                let pml4 = unsafe { &mut *(pt as *mut x86_64::structures::paging::PageTable) };
                for i in 0..pages {
                    let virt = x86_64::VirtAddr::new(map_addr + (i as u64) * 4096);
                    if let Some(frame) = crate::memory::phys::alloc_frame() {
                        unsafe {
                            core::ptr::write_bytes(frame.as_u64() as *mut u8, 0, 4096);
                            crate::memory::virt::map_page_to_pml4(
                                virt,
                                frame,
                                x86_64::structures::paging::PageTableFlags::PRESENT
                                    | x86_64::structures::paging::PageTableFlags::WRITABLE
                                    | x86_64::structures::paging::PageTableFlags::USER_ACCESSIBLE,
                                pml4,
                            );
                        }
                    } else {
                        return SyscallResult::err(12); // ENOMEM
                    }
                }
                SyscallResult::ok(map_addr as i64)
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

        SYS_MANIFOLD_QUERY => SyscallResult::with_data(
            0,
            format!("theorem_{}: ACTIVE", arg0),
        ),

        SYS_TELEPORT => {
            let path = {
                let guard = SYSCALL_PATH.lock();
                guard.clone()
            };
            let src = arg0;
            let dst = arg1;
            let result = with_fs_inode(|fs| {
                fs.teleport(&path, src, dst).map(|r| r.inode_id)
            });
            if result.code >= 0 {
                SyscallResult::with_data(
                    0,
                    format!("teleported '{}' from dir {} -> dir {}", path, src, dst),
                )
            } else {
                result
            }
        }

        SYS_THEOREM_STATUS => SyscallResult::with_data(
            0,
            String::from("T1:ACTIVE T2:ACTIVE T3:ACTIVE T4:ACTIVE T5:ACTIVE"),
        ),

        SYS_PKG_INSTALL => SyscallResult::with_data(
            0,
            String::from("pkg_install: not yet implemented"),
        ),
        SYS_PKG_REMOVE => SyscallResult::with_data(
            0,
            String::from("pkg_remove: not yet implemented"),
        ),
        SYS_PKG_LIST => SyscallResult::with_data(
            0,
            String::from("pkg_list: no packages installed"),
        ),
        SYS_WIFI_SCAN => SyscallResult::with_data(
            0,
            String::from("wifi_scan: no wireless hardware detected"),
        ),
        SYS_WIFI_CONNECT => SyscallResult::with_data(
            0,
            String::from("wifi_connect: no wireless hardware detected"),
        ),
        SYS_BT_SCAN => SyscallResult::with_data(
            0,
            String::from("bt_scan: no Bluetooth adapter detected"),
        ),
        SYS_BT_PAIR => SyscallResult::with_data(
            0,
            String::from("bt_pair: no Bluetooth adapter detected"),
        ),
        SYS_SETTING_GET => SyscallResult::with_data(
            0,
            String::from("setting_get: not yet implemented"),
        ),
        SYS_SETTING_SET => SyscallResult::with_data(
            0,
            String::from("setting_set: not yet implemented"),
        ),

        SYS_SETUID => {
            crate::process::scheduler::set_current_uid(arg0 as u32);
            SyscallResult::ok(0)
        }

        SYS_SETGID => {
            crate::process::scheduler::set_current_gid(arg0 as u32);
            SyscallResult::ok(0)
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
                perms: if arg1 & 0x1 != 0 { String::from("w") } else { String::from("r") },
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
