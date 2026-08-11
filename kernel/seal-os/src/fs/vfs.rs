// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Virtual File System core — mounts multiple filesystems and routes operations.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use spin::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VfsNodeType {
    File,
    Directory,
    Symlink,
    CharDevice,
    BlockDevice,
    Pipe,
}

#[derive(Debug, Clone)]
pub struct VfsNode {
    pub size: u64,
    pub permissions: u16,
    pub uid: u32,
    pub gid: u32,
    pub mode: u16,
    pub atime: u64,
    pub mtime: u64,
    pub node_type: VfsNodeType,
    pub major: u32,
    pub minor: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfsHandle {
    pub fs_idx: usize,
    pub inode: u64,
}

#[derive(Debug, Clone)]
pub struct VfsDirEntry {
    pub name: String,
    pub node_type: VfsNodeType,
}

#[derive(Debug, Clone)]
pub enum VfsError {
    NotFound,
    NotADirectory,
    AlreadyExists,
    InvalidPath,
    TooManySymlinks,
    NotSupported,
    IoError,
    PermissionDenied,
    InvalidOperation,
}

impl core::fmt::Display for VfsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotFound => write!(f, "not found"),
            Self::NotADirectory => write!(f, "not a directory"),
            Self::AlreadyExists => write!(f, "already exists"),
            Self::InvalidPath => write!(f, "invalid path"),
            Self::TooManySymlinks => write!(f, "too many symlinks"),
            Self::NotSupported => write!(f, "not supported"),
            Self::IoError => write!(f, "I/O error"),
            Self::PermissionDenied => write!(f, "permission denied"),
            Self::InvalidOperation => write!(f, "invalid operation"),
        }
    }
}

pub trait FileSystem: Send + Sync {
    fn lookup(&self, path: &str) -> Result<VfsHandle, VfsError>;
    fn read(&self, handle: VfsHandle, buf: &mut [u8], offset: u64) -> Result<usize, VfsError>;
    fn write(&mut self, handle: VfsHandle, buf: &[u8], offset: u64) -> Result<usize, VfsError>;
    fn create(&mut self, path: &str) -> Result<VfsHandle, VfsError>;
    fn mkdir(&mut self, path: &str) -> Result<VfsHandle, VfsError>;
    fn unlink(&mut self, path: &str) -> Result<(), VfsError>;
    fn rmdir(&mut self, path: &str) -> Result<(), VfsError>;
    fn rename(&mut self, old: &str, new: &str) -> Result<(), VfsError>;
    fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError>;
    fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError>;
    fn mknod(
        &mut self,
        path: &str,
        node_type: VfsNodeType,
        major: u32,
        minor: u32,
    ) -> Result<VfsHandle, VfsError>;
    /// Synchronize all dirty buffers and metadata to persistent storage.
    ///
    /// WARNING: the default implementation writes nothing and reports success.
    /// `fat.rs`, `procfs.rs`, `sysfs.rs` and `pipe.rs` do not override it, so a
    /// `sync()` on a mounted FAT volume returns `Ok(())` without flushing.
    /// Implementors backed by real storage MUST override this.
    fn sync(&mut self) -> Result<(), VfsError> {
        Ok(())
    }
    /// Synchronize a single file's dirty data to persistent storage.
    ///
    /// WARNING: same caveat as `sync` — the default is a no-op that reports
    /// success, and the on-disk filesystems do not override it.
    fn fsync(&mut self, _handle: VfsHandle) -> Result<(), VfsError> {
        Ok(())
    }
}

pub struct MountPoint {
    pub prefix: String,
    pub fs: Mutex<Box<dyn FileSystem>>,
}

pub struct Vfs {
    pub(crate) mounts: Vec<MountPoint>,
}

impl Vfs {
    pub fn new() -> Self {
        Self { mounts: Vec::new() }
    }

    /// Check Mandatory Access Control for `path` with `perms`.
    fn check_mac(
        &self,
        path: &str,
        perms: crate::security::mac::Permissions,
    ) -> Result<(), VfsError> {
        let uid = crate::process::scheduler::current_uid();
        if !crate::security::mac::check_file_permission(uid, path, perms) {
            return Err(VfsError::PermissionDenied);
        }
        Ok(())
    }

    /// Canonicalize `path` into the one string every entry point uses for
    /// *both* the MAC check and mount routing. This is the single choke
    /// point: `check_mac` and `find_mount` are always called on the exact
    /// same value produced here, so they can no longer disagree about
    /// which file a string denotes — which is what let `//root/secret`
    /// slip past a `Deny /root` rule while the backend resolved it to the
    /// same inode as `/root/secret`.
    ///
    /// Rejects non-absolute input outright (`None`), before anything can
    /// fall through a fallback — same reasoning as `find_mount`'s own
    /// guard, kept independently here since callers now canonicalize
    /// *before* routing, not after.
    ///
    /// Collapses repeated `/` separators, drops `.` components, and
    /// resolves `..` by popping the last pushed component (clamped at
    /// root once the stack is empty — mirrors `ManifoldFS`'s own root
    /// `".."` entry, which points back to itself). `..` is resolved here,
    /// not left textual: this backend gives every directory a real `".."`
    /// entry pointing at its true parent (`manifold_fs.rs`: `dirs.insert`
    /// calls for `"."`/`".."` on every `mkdir` and on root), and
    /// `resolve_path` walks components as literal directory-entry lookups
    /// with no mid-path symlink substitution — so a textual `..` pop
    /// lands on the identical inode the backend's own structural walk
    /// would reach. That equivalence is what makes textual `..`
    /// resolution safe *here*; it depends on the backend never following
    /// a symlink in the middle of a path before this canonical form is
    /// computed. `Vfs::lookup_follow` only substitutes a symlink target
    /// and re-resolves the *whole* path afterward, never mid-walk, so the
    /// assumption holds for this codebase today. If a backend ever grows
    /// mid-path symlink following, this must move to per-component
    /// resolution instead.
    fn canonicalize_path(path: &str) -> Option<String> {
        if !path.starts_with('/') {
            return None;
        }
        let mut stack: Vec<&str> = Vec::new();
        for part in path.split('/') {
            match part {
                "" | "." => continue,
                ".." => {
                    stack.pop();
                }
                _ => stack.push(part),
            }
        }
        if stack.is_empty() {
            return Some(String::from("/"));
        }
        let mut out = String::with_capacity(path.len());
        for part in stack {
            out.push('/');
            out.push_str(part);
        }
        Some(out)
    }

    pub fn mount(&mut self, prefix: &str, fs: Box<dyn FileSystem>) -> Result<(), VfsError> {
        let prefix = if prefix == "/" {
            String::from("/")
        } else {
            prefix.trim_end_matches('/').to_string()
        };
        self.mounts.push(MountPoint {
            prefix,
            fs: Mutex::new(fs),
        });
        Ok(())
    }

    /// Find the mount whose prefix matches `path` at a path-component
    /// boundary, returning `(mount index, path relative to that mount)`.
    ///
    /// A non-root prefix matches only if `path` equals the prefix exactly,
    /// or the byte immediately after the prefix in `path` is `/`. This is
    /// what a raw `starts_with` test gets wrong: `/devfoo` is not inside
    /// `/dev` just because the bytes happen to line up. The root mount
    /// (`/`) has no such boundary to test — stripping it consumes the
    /// separator itself — so it is handled explicitly as the fallback
    /// that wins only when no longer, boundary-respecting prefix claims
    /// the path.
    ///
    /// Only absolute paths resolve. This must hold *before* the root
    /// fallback can fire: the fallback is keyed on "no non-root mount
    /// claimed it", which `""` and `"dev/null"` also satisfy, and the
    /// root filesystem does not itself guard the empty-name case (see
    /// `manifold_fs::split_path`'s unguarded `None` arm). Without this
    /// check the fallback would hand a syscall's unvalidated `""` or
    /// relative string straight to a backend that trusts VFS routing to
    /// have already rejected it.
    fn find_mount<'a>(&self, path: &'a str) -> Option<(usize, &'a str)> {
        if !path.starts_with('/') {
            return None;
        }
        let mut best: Option<(usize, usize, &'a str)> = None; // (mount idx, prefix len, rel)
        let mut root_idx = None;
        for (i, mp) in self.mounts.iter().enumerate() {
            if mp.prefix == "/" {
                root_idx = Some(i);
                continue;
            }
            let Some(stripped) = path.strip_prefix(mp.prefix.as_str()) else {
                continue;
            };
            if !stripped.is_empty() && !stripped.starts_with('/') {
                continue; // e.g. "/devfoo" vs prefix "/dev" — not a component boundary
            }
            if mp.prefix.len() > best.map_or(0, |(_, len, _)| len) {
                let rel = if stripped.is_empty() { "/" } else { stripped };
                best = Some((i, mp.prefix.len(), rel));
            }
        }
        best.map(|(i, _, rel)| (i, rel))
            .or_else(|| root_idx.map(|i| (i, path)))
    }

    pub fn lookup(&self, path: &str) -> Result<VfsHandle, VfsError> {
        let path = Self::canonicalize_path(path).ok_or(VfsError::InvalidPath)?;
        self.check_mac(&path, crate::security::mac::Permissions::R)?;
        let (fs_idx, rel) = self.find_mount(&path).ok_or(VfsError::NotFound)?;
        let mount = &self.mounts[fs_idx];
        let fs_guard = mount.fs.lock();
        let handle = fs_guard.lookup(rel)?;
        let vfs_handle = VfsHandle {
            fs_idx,
            inode: handle.inode,
        };
        drop(fs_guard);
        HANDLE_PATHS
            .lock()
            .insert((fs_idx, handle.inode), path.clone());
        // T1–T5: Enforce node-level permissions on lookup (open / exec).
        let node = self.stat(vfs_handle)?;
        let uid = crate::process::scheduler::current_uid();
        let gid = crate::process::scheduler::current_gid();
        let groups = crate::process::scheduler::current_groups();
        if crate::security::manifold_acl::check_access(
            uid,
            gid,
            &groups,
            &node,
            crate::security::manifold_acl::PERM_READ,
            &path,
        ) == crate::security::manifold_acl::AccessDecision::Deny
        {
            return Err(VfsError::PermissionDenied);
        }
        Ok(vfs_handle)
    }

    pub fn lookup_follow(&self, path: &str) -> Result<VfsHandle, VfsError> {
        let mut current = path.to_string();
        for _ in 0..8 {
            let handle = self.lookup(&current)?;
            let node = self.stat(handle)?;
            if node.node_type != VfsNodeType::Symlink {
                return Ok(handle);
            }
            let mut buf = [0u8; 256];
            let len = self.read(handle, &mut buf, 0)?;
            let target = core::str::from_utf8(&buf[..len]).map_err(|_| VfsError::InvalidPath)?;
            current = if target.starts_with('/') {
                target.to_string()
            } else {
                let dir = current.rfind('/').map(|i| &current[..i + 1]).unwrap_or("/");
                format!("{}{}", dir, target)
            };
        }
        Err(VfsError::TooManySymlinks)
    }

    pub fn read(&self, handle: VfsHandle, buf: &mut [u8], offset: u64) -> Result<usize, VfsError> {
        if let Some(path) = HANDLE_PATHS.lock().get(&(handle.fs_idx, handle.inode)) {
            self.check_mac(path, crate::security::mac::Permissions::R)?;
        }
        // T1–T5: Node-level ACL before acquiring filesystem lock.
        if let Some(path) = HANDLE_PATHS.lock().get(&(handle.fs_idx, handle.inode)) {
            let node = self.stat(handle)?;
            let uid = crate::process::scheduler::current_uid();
            let gid = crate::process::scheduler::current_gid();
            let groups = crate::process::scheduler::current_groups();
            if crate::security::manifold_acl::check_access(
                uid,
                gid,
                &groups,
                &node,
                crate::security::manifold_acl::PERM_READ,
                path,
            ) == crate::security::manifold_acl::AccessDecision::Deny
            {
                return Err(VfsError::PermissionDenied);
            }
        }
        let mount = self.mounts.get(handle.fs_idx).ok_or(VfsError::NotFound)?;
        let fs_guard = mount.fs.lock();
        fs_guard.read(
            VfsHandle {
                fs_idx: 0,
                inode: handle.inode,
            },
            buf,
            offset,
        )
    }

    pub fn write(&self, handle: VfsHandle, buf: &[u8], offset: u64) -> Result<usize, VfsError> {
        if let Some(path) = HANDLE_PATHS.lock().get(&(handle.fs_idx, handle.inode)) {
            self.check_mac(path, crate::security::mac::Permissions::W)?;
        }
        // T1–T5: Node-level ACL before acquiring filesystem lock.
        if let Some(path) = HANDLE_PATHS.lock().get(&(handle.fs_idx, handle.inode)) {
            let node = self.stat(handle)?;
            let uid = crate::process::scheduler::current_uid();
            let gid = crate::process::scheduler::current_gid();
            let groups = crate::process::scheduler::current_groups();
            if crate::security::manifold_acl::check_access(
                uid,
                gid,
                &groups,
                &node,
                crate::security::manifold_acl::PERM_WRITE,
                path,
            ) == crate::security::manifold_acl::AccessDecision::Deny
            {
                return Err(VfsError::PermissionDenied);
            }
        }
        let mount = self.mounts.get(handle.fs_idx).ok_or(VfsError::NotFound)?;
        let mut fs_guard = mount.fs.lock();
        fs_guard.write(
            VfsHandle {
                fs_idx: 0,
                inode: handle.inode,
            },
            buf,
            offset,
        )
    }

    pub fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> {
        let mount = self.mounts.get(handle.fs_idx).ok_or(VfsError::NotFound)?;
        let fs_guard = mount.fs.lock();
        fs_guard.readdir(VfsHandle {
            fs_idx: 0,
            inode: handle.inode,
        })
    }

    pub fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError> {
        let mount = self.mounts.get(handle.fs_idx).ok_or(VfsError::NotFound)?;
        let fs_guard = mount.fs.lock();
        fs_guard.stat(VfsHandle {
            fs_idx: 0,
            inode: handle.inode,
        })
    }

    pub fn create(&self, path: &str) -> Result<VfsHandle, VfsError> {
        let path = Self::canonicalize_path(path).ok_or(VfsError::InvalidPath)?;
        self.check_mac(&path, crate::security::mac::Permissions::W)?;
        let (fs_idx, rel) = self.find_mount(&path).ok_or(VfsError::NotFound)?;
        let mount = &self.mounts[fs_idx];
        let mut fs_guard = mount.fs.lock();
        let handle = fs_guard.create(rel)?;
        HANDLE_PATHS.lock().insert((fs_idx, handle.inode), path);
        Ok(VfsHandle {
            fs_idx,
            inode: handle.inode,
        })
    }

    pub fn mkdir(&self, path: &str) -> Result<VfsHandle, VfsError> {
        let path = Self::canonicalize_path(path).ok_or(VfsError::InvalidPath)?;
        self.check_mac(&path, crate::security::mac::Permissions::W)?;
        let (fs_idx, rel) = self.find_mount(&path).ok_or(VfsError::NotFound)?;
        let mount = &self.mounts[fs_idx];
        let mut fs_guard = mount.fs.lock();
        let handle = fs_guard.mkdir(rel)?;
        HANDLE_PATHS.lock().insert((fs_idx, handle.inode), path);
        Ok(VfsHandle {
            fs_idx,
            inode: handle.inode,
        })
    }

    pub fn unlink(&self, path: &str) -> Result<(), VfsError> {
        let path = Self::canonicalize_path(path).ok_or(VfsError::InvalidPath)?;
        self.check_mac(&path, crate::security::mac::Permissions::W)?;
        let (fs_idx, rel) = self.find_mount(&path).ok_or(VfsError::NotFound)?;
        let mount = &self.mounts[fs_idx];
        let mut fs_guard = mount.fs.lock();
        fs_guard.unlink(rel)?;
        HANDLE_PATHS
            .lock()
            .remove(&(fs_idx, fs_guard.lookup(rel).map(|h| h.inode).unwrap_or(0)));
        Ok(())
    }

    pub fn rmdir(&self, path: &str) -> Result<(), VfsError> {
        let path = Self::canonicalize_path(path).ok_or(VfsError::InvalidPath)?;
        self.check_mac(&path, crate::security::mac::Permissions::W)?;
        let (fs_idx, rel) = self.find_mount(&path).ok_or(VfsError::NotFound)?;
        let mount = &self.mounts[fs_idx];
        let mut fs_guard = mount.fs.lock();
        fs_guard.rmdir(rel)?;
        HANDLE_PATHS
            .lock()
            .remove(&(fs_idx, fs_guard.lookup(rel).map(|h| h.inode).unwrap_or(0)));
        Ok(())
    }

    pub fn rename(&self, old: &str, new: &str) -> Result<(), VfsError> {
        let old = Self::canonicalize_path(old).ok_or(VfsError::InvalidPath)?;
        let new = Self::canonicalize_path(new).ok_or(VfsError::InvalidPath)?;
        self.check_mac(&old, crate::security::mac::Permissions::W)?;
        self.check_mac(&new, crate::security::mac::Permissions::W)?;
        let (old_fs_idx, old_rel) = self.find_mount(&old).ok_or(VfsError::NotFound)?;
        let (new_fs_idx, new_rel) = self.find_mount(&new).ok_or(VfsError::NotFound)?;
        if old_fs_idx == new_fs_idx {
            let mount = &self.mounts[old_fs_idx];
            let mut fs_guard = mount.fs.lock();
            fs_guard.rename(old_rel, new_rel)?;
            if let Ok(handle) = fs_guard.lookup(new_rel) {
                HANDLE_PATHS
                    .lock()
                    .insert((old_fs_idx, handle.inode), new.clone());
            }
            if let Ok(old_handle) = fs_guard.lookup(old_rel) {
                HANDLE_PATHS.lock().remove(&(old_fs_idx, old_handle.inode));
            }
            Ok(())
        } else {
            // Cross-mount rename: copy file content then delete original.
            // Directories cannot be moved across filesystems.
            let old_mount = &self.mounts[old_fs_idx];
            let old_fs = old_mount.fs.lock();
            let old_handle = old_fs.lookup(old_rel)?;
            let node = old_fs.stat(old_handle)?;
            if node.node_type == VfsNodeType::Directory {
                return Err(VfsError::InvalidOperation);
            }
            let mut data = alloc::vec![0u8; node.size as usize];
            let mut total_read = 0usize;
            while total_read < data.len() {
                let n = old_fs.read(
                    VfsHandle {
                        fs_idx: 0,
                        inode: old_handle.inode,
                    },
                    &mut data[total_read..],
                    total_read as u64,
                )?;
                if n == 0 {
                    break;
                }
                total_read += n;
            }
            drop(old_fs);

            let new_mount = &self.mounts[new_fs_idx];
            let mut new_fs = new_mount.fs.lock();
            // If destination exists, remove it.
            if new_fs.lookup(new_rel).is_ok() {
                new_fs.unlink(new_rel)?;
            }
            let new_handle = new_fs.create(new_rel)?;
            new_fs.write(new_handle, &data, 0)?;
            drop(new_fs);

            let old_mount2 = &self.mounts[old_fs_idx];
            let mut old_fs2 = old_mount2.fs.lock();
            old_fs2.unlink(old_rel)?;
            HANDLE_PATHS.lock().remove(&(old_fs_idx, old_handle.inode));
            HANDLE_PATHS
                .lock()
                .insert((new_fs_idx, new_handle.inode), new.clone());
            Ok(())
        }
    }

    pub fn mknod(
        &self,
        path: &str,
        node_type: VfsNodeType,
        major: u32,
        minor: u32,
    ) -> Result<VfsHandle, VfsError> {
        let path = Self::canonicalize_path(path).ok_or(VfsError::InvalidPath)?;
        self.check_mac(&path, crate::security::mac::Permissions::W)?;
        let (fs_idx, rel) = self.find_mount(&path).ok_or(VfsError::NotFound)?;
        let mount = &self.mounts[fs_idx];
        let mut fs_guard = mount.fs.lock();
        let handle = fs_guard.mknod(rel, node_type, major, minor)?;
        HANDLE_PATHS.lock().insert((fs_idx, handle.inode), path);
        Ok(VfsHandle {
            fs_idx,
            inode: handle.inode,
        })
    }

    pub fn fsync(&self, handle: VfsHandle) -> Result<(), VfsError> {
        let mount = self.mounts.get(handle.fs_idx).ok_or(VfsError::NotFound)?;
        let mut fs_guard = mount.fs.lock();
        fs_guard.fsync(VfsHandle {
            fs_idx: 0,
            inode: handle.inode,
        })
    }
}

impl Default for Vfs {
    fn default() -> Self {
        Self::new()
    }
}

pub static VFS: Mutex<Option<Vfs>> = Mutex::new(None);

/// Reverse mapping from handle to path for MAC checks on read/write.
static HANDLE_PATHS: Mutex<BTreeMap<(usize, u64), String>> = Mutex::new(BTreeMap::new());

pub fn with_vfs<F, R>(f: F) -> R
where
    F: FnOnce(&Vfs) -> R,
{
    #[cfg(any(test, feature = "test-mode"))]
    ensure_test_vfs();
    let guard = VFS.lock();
    let vfs = guard.as_ref().expect("VFS not initialized");
    f(vfs)
}

/// Mount a RAM-backed root the first time a `test-mode` build reaches
/// `with_vfs` with no VFS.
///
/// `testing::runner::test_main()` is called from `kernel_main_continue`
/// immediately after the entropy driver — layers before `fs::init_vfs()` runs
/// — so under `test-mode` the VFS is never initialized at all. The first suite
/// that touches a file (`bundle`, reaching the store through
/// `pkg::ManifoldPkg::install_file`) hit the `expect` above, and with
/// `panic = "abort"` that took down every suite registered after it: 23
/// registration groups never executed.
///
/// Compiled only into test builds. The production boot path keeps the panic,
/// because a kernel that quietly mounts an empty filesystem under a caller
/// that expects the real disk is worse than one that stops. The mount is
/// announced on serial so a test-mode run can never mistake this root for the
/// real one.
///
/// Root only: the pseudo-filesystems `init_vfs` mounts at `/proc`, `/sys`,
/// `/dev` and `/pipe` need driver state the harness has not brought up. A
/// lookup under those prefixes routes to this root and returns `NotFound`,
/// which a test can handle; it cannot abort the machine.
///
/// ponytail: on-first-use rather than an explicit call at the top of
/// `test_main`, which is the honest place for it — `testing/runner.rs` is
/// owned by another change this round. Upgrade path: call `fs::init_vfs()` (or
/// this function) from `test_main` before `register_all`, then delete this.
#[cfg(any(test, feature = "test-mode"))]
fn ensure_test_vfs() {
    let mut guard = VFS.lock();
    if guard.is_some() {
        return;
    }
    let mut v = Vfs::new();
    let _ = v.mount(
        "/",
        Box::new(crate::fs::manifold_fs::ManifoldFS::new_ramfs()),
    );
    *guard = Some(v);
    drop(guard);
    crate::serial_println!(
        "[VFS] test-mode: mounted ramfs root on first use (harness runs before fs::init_vfs)"
    );
}

/// Returns `true` if the VFS has been initialized.
pub fn is_vfs_initialized() -> bool {
    VFS.lock().is_some()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// Minimal `FileSystem` stand-in — `find_mount` never calls into the
    /// backend, it only decides *which* mount and *what* relative path,
    /// so the mounted filesystem's behavior is irrelevant to these tests.
    struct NullFs;

    impl FileSystem for NullFs {
        fn lookup(&self, _path: &str) -> Result<VfsHandle, VfsError> {
            Err(VfsError::NotFound)
        }
        fn read(&self, _h: VfsHandle, _buf: &mut [u8], _off: u64) -> Result<usize, VfsError> {
            Err(VfsError::NotSupported)
        }
        fn write(&mut self, _h: VfsHandle, _buf: &[u8], _off: u64) -> Result<usize, VfsError> {
            Err(VfsError::NotSupported)
        }
        fn create(&mut self, _path: &str) -> Result<VfsHandle, VfsError> {
            Err(VfsError::NotSupported)
        }
        fn mkdir(&mut self, _path: &str) -> Result<VfsHandle, VfsError> {
            Err(VfsError::NotSupported)
        }
        fn unlink(&mut self, _path: &str) -> Result<(), VfsError> {
            Err(VfsError::NotSupported)
        }
        fn rmdir(&mut self, _path: &str) -> Result<(), VfsError> {
            Err(VfsError::NotSupported)
        }
        fn rename(&mut self, _old: &str, _new: &str) -> Result<(), VfsError> {
            Err(VfsError::NotSupported)
        }
        fn readdir(&self, _h: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> {
            Err(VfsError::NotSupported)
        }
        fn stat(&self, _h: VfsHandle) -> Result<VfsNode, VfsError> {
            Err(VfsError::NotSupported)
        }
        fn mknod(
            &mut self,
            _path: &str,
            _node_type: VfsNodeType,
            _major: u32,
            _minor: u32,
        ) -> Result<VfsHandle, VfsError> {
            Err(VfsError::NotSupported)
        }
    }

    /// Mirrors the real mount table wired in `fs/mod.rs::init_vfs`:
    /// `/`, `/proc`, `/sys`, `/dev`, `/pipe`, in that order.
    fn mounted_vfs() -> Vfs {
        let mut v = Vfs::new();
        v.mount("/", Box::new(NullFs)).unwrap();
        v.mount("/proc", Box::new(NullFs)).unwrap();
        v.mount("/sys", Box::new(NullFs)).unwrap();
        v.mount("/dev", Box::new(NullFs)).unwrap();
        v.mount("/pipe", Box::new(NullFs)).unwrap();
        v
    }

    /// `/devfoo` shares a byte-prefix with `/dev` but is not inside it —
    /// the character after `/dev` in `/devfoo` is `f`, not `/`. It must
    /// route through the root mount (index 0), not devtmpfs (index 3).
    fn test_devfoo_is_not_inside_dev_mount() -> TestResult {
        let v = mounted_vfs();
        let (idx, rel) = match v.find_mount("/devfoo") {
            Some(x) => x,
            None => return TestResult::Fail("no mount matched /devfoo"),
        };
        test_assert_eq!(idx, 0); // root mount, not /dev (idx 3)
        test_assert_eq!(rel, "/devfoo");
        TestResult::Pass
    }

    /// A genuine path under `/dev` must still route to devtmpfs with the
    /// mount prefix stripped.
    fn test_dev_null_still_routes_to_devfs() -> TestResult {
        let v = mounted_vfs();
        let (idx, rel) = match v.find_mount("/dev/null") {
            Some(x) => x,
            None => return TestResult::Fail("no mount matched /dev/null"),
        };
        test_assert_eq!(idx, 3); // /dev
        test_assert_eq!(rel, "/null");
        TestResult::Pass
    }

    /// Same collision class as `/devfoo` vs `/dev`, for the other three
    /// non-root mounts: `/procedures`, `/systemd`, `/pipeline`.
    fn test_sibling_prefix_collisions_route_to_root() -> TestResult {
        let v = mounted_vfs();
        for path in ["/procedures", "/systemd", "/pipeline"] {
            let (idx, rel) = match v.find_mount(path) {
                Some(x) => x,
                None => return TestResult::Fail("no mount matched"),
            };
            test_assert_eq!(idx, 0);
            test_assert_eq!(rel, path);
        }
        TestResult::Pass
    }

    /// The mount root itself (no trailing component) must resolve to `/`,
    /// not an empty string, for every FileSystem's `lookup` convention.
    fn test_mount_root_exact_match_yields_slash() -> TestResult {
        let v = mounted_vfs();
        let (idx, rel) = match v.find_mount("/dev") {
            Some(x) => x,
            None => return TestResult::Fail("no mount matched /dev"),
        };
        test_assert_eq!(idx, 3);
        test_assert_eq!(rel, "/");
        TestResult::Pass
    }

    /// The empty path is not absolute and must never fall through to the
    /// root-mount fallback: `ManifoldFS::split_path`'s `None` arm (no `/`
    /// found) returns `Ok(("/", ""))` with no empty-name guard, so a
    /// caller that skipped its own validation could otherwise create an
    /// inode literally named `""` at the filesystem root.
    fn test_empty_path_does_not_resolve() -> TestResult {
        let v = mounted_vfs();
        test_assert!(v.find_mount("").is_none(), "empty path must not resolve");
        TestResult::Pass
    }

    /// A relative path (no leading `/`) is not absolute either, and must
    /// be rejected for the same reason as the empty-path case above.
    fn test_relative_path_does_not_resolve() -> TestResult {
        let v = mounted_vfs();
        test_assert!(
            v.find_mount("dev/null").is_none(),
            "relative path must not resolve"
        );
        TestResult::Pass
    }

    /// `/` alone is the root mount itself: no non-root prefix can match
    /// it (every non-root prefix is longer than one byte), so it must
    /// fall through to the root-mount fallback with `rel == "/"`.
    fn test_root_alone_yields_root_mount() -> TestResult {
        let v = mounted_vfs();
        let (idx, rel) = match v.find_mount("/") {
            Some(x) => x,
            None => return TestResult::Fail("no mount matched /"),
        };
        test_assert_eq!(idx, 0);
        test_assert_eq!(rel, "/");
        TestResult::Pass
    }

    /// A doubled separator breaks the literal-byte-prefix match on
    /// purpose: `"//dev/null".strip_prefix("/dev")` is `None` because
    /// byte 1 is `/` where the prefix expects `d`. `find_mount` does not
    /// normalize paths — that is the caller's job — so this is not a
    /// `/dev` match and correctly falls through to the root mount with
    /// the path passed through unchanged. It is not the empty-path hole:
    /// `rel` here is `"//dev/null"`, never empty, so the unguarded
    /// `ManifoldFS::split_path` `None` arm is never reached.
    fn test_doubled_separator_falls_through_to_root() -> TestResult {
        let v = mounted_vfs();
        let (idx, rel) = match v.find_mount("//dev/null") {
            Some(x) => x,
            None => return TestResult::Fail("no mount matched //dev/null"),
        };
        test_assert_eq!(idx, 0); // root, not devtmpfs — no literal-prefix match
        test_assert_eq!(rel, "//dev/null"); // unchanged, unnormalized
        TestResult::Pass
    }

    /// A trailing slash after a mount prefix still hits the boundary
    /// rule (`stripped == "/"`, which starts with `/`), and the empty
    /// remainder after that separator collapses to `"/"` — same result
    /// as referencing the mount with no trailing slash at all.
    fn test_trailing_slash_after_mount_yields_slash() -> TestResult {
        let v = mounted_vfs();
        let (idx, rel) = match v.find_mount("/dev/") {
            Some(x) => x,
            None => return TestResult::Fail("no mount matched /dev/"),
        };
        test_assert_eq!(idx, 3);
        test_assert_eq!(rel, "/");
        TestResult::Pass
    }

    /// `canonicalize_path` must reject the same inputs `find_mount` does
    /// — proof that adding canonicalization did not reopen the regression
    /// closed last round. Collapsing repeated separators must never turn
    /// an empty or relative string into something absolute.
    fn test_canonicalize_rejects_empty_and_relative() -> TestResult {
        test_assert!(Vfs::canonicalize_path("").is_none());
        test_assert!(Vfs::canonicalize_path("dev/null").is_none());
        TestResult::Pass
    }

    /// Repeated separators collapse, a trailing slash is dropped, and the
    /// path-only-mount-root case yields `/` — this is the exact string
    /// `check_mac` and `find_mount` now agree on for each input.
    fn test_canonicalize_collapses_separators_and_trims() -> TestResult {
        test_assert_eq!(Vfs::canonicalize_path("//root/secret").unwrap(), "/root/secret");
        test_assert_eq!(Vfs::canonicalize_path("///root/secret").unwrap(), "/root/secret");
        test_assert_eq!(Vfs::canonicalize_path("/dev/").unwrap(), "/dev");
        test_assert_eq!(Vfs::canonicalize_path("/").unwrap(), "/");
        TestResult::Pass
    }

    /// `..` is resolved by popping the last component, clamped at root —
    /// matching `ManifoldFS`'s own root `".."` entry, which points back
    /// to itself. `.` is dropped as a no-op.
    fn test_canonicalize_resolves_dotdot_and_dot() -> TestResult {
        test_assert_eq!(
            Vfs::canonicalize_path("/tmp/../root/secret").unwrap(),
            "/root/secret"
        );
        test_assert_eq!(Vfs::canonicalize_path("/../../etc/passwd").unwrap(), "/etc/passwd");
        test_assert_eq!(Vfs::canonicalize_path("/foo/./bar").unwrap(), "/foo/bar");
        TestResult::Pass
    }

    /// The bug as reported: `//root/secret` must be denied for a
    /// non-root uid once it is canonicalized before reaching the MAC
    /// check — exactly what `Vfs::lookup` and friends now do. This
    /// chains `canonicalize_path` + `check_file_permission` directly,
    /// the same two calls `check_mac` makes, without needing a live
    /// scheduler for `current_uid()`.
    fn test_doubled_slash_root_secret_denied_for_user() -> TestResult {
        crate::security::mac::init_default_policy();
        let canonical = Vfs::canonicalize_path("//root/secret").unwrap();
        test_assert!(!crate::security::mac::check_file_permission(
            1000,
            &canonical,
            crate::security::mac::Permissions::R
        ));
        TestResult::Pass
    }

    /// `/root/secret` (no doubling) is still denied — the canonicalize
    /// step must be a no-op on already-canonical input.
    fn test_plain_root_secret_still_denied_for_user() -> TestResult {
        crate::security::mac::init_default_policy();
        let canonical = Vfs::canonicalize_path("/root/secret").unwrap();
        test_assert!(!crate::security::mac::check_file_permission(
            1000,
            &canonical,
            crate::security::mac::Permissions::R
        ));
        TestResult::Pass
    }

    /// `/data/x` and `/tmp/x` remain allowed after canonicalization.
    fn test_data_and_tmp_still_allowed_for_user() -> TestResult {
        crate::security::mac::init_default_policy();
        let data = Vfs::canonicalize_path("/data/x").unwrap();
        let tmp = Vfs::canonicalize_path("/tmp/x").unwrap();
        test_assert!(crate::security::mac::check_file_permission(
            1000,
            &data,
            crate::security::mac::Permissions::R
        ));
        test_assert!(crate::security::mac::check_file_permission(
            1000,
            &tmp,
            crate::security::mac::Permissions::W
        ));
        TestResult::Pass
    }

    /// uid 0 bypasses MAC regardless of canonicalization.
    fn test_uid_zero_bypasses_after_canonicalize() -> TestResult {
        crate::security::mac::init_default_policy();
        let canonical = Vfs::canonicalize_path("//root/secret").unwrap();
        test_assert!(crate::security::mac::check_file_permission(
            0,
            &canonical,
            crate::security::mac::Permissions::R
        ));
        TestResult::Pass
    }

    /// `///root/secret` (three separators) and a trailing-slash form
    /// (`/root/`) both canonicalize to the exact string the `Deny /root`
    /// rule matches, and are denied for a non-root uid.
    fn test_triple_slash_and_trailing_slash_root_denied() -> TestResult {
        crate::security::mac::init_default_policy();
        let triple = Vfs::canonicalize_path("///root/secret").unwrap();
        test_assert_eq!(triple, "/root/secret");
        test_assert!(!crate::security::mac::check_file_permission(
            1000,
            &triple,
            crate::security::mac::Permissions::R
        ));
        let trailing = Vfs::canonicalize_path("/root/").unwrap();
        test_assert_eq!(trailing, "/root");
        test_assert!(!crate::security::mac::check_file_permission(
            1000,
            &trailing,
            crate::security::mac::Permissions::W
        ));
        TestResult::Pass
    }

    /// `/tmp/../root/secret`: without `..` resolution this would hit the
    /// `/tmp` `Allow` rule as a literal string while the backend's own
    /// real `".."` directory entries still resolve it to `/root/secret`.
    /// Canonicalizing `..` before the MAC check closes that route too.
    fn test_dotdot_escape_through_allowed_prefix_denied() -> TestResult {
        crate::security::mac::init_default_policy();
        let canonical = Vfs::canonicalize_path("/tmp/../root/secret").unwrap();
        test_assert_eq!(canonical, "/root/secret");
        test_assert!(!crate::security::mac::check_file_permission(
            1000,
            &canonical,
            crate::security::mac::Permissions::R
        ));
        TestResult::Pass
    }

    /// `testing::runner::test_main()` runs several boot layers before
    /// `fs::init_vfs()`, so every suite reaches `with_vfs` with `VFS == None`.
    /// Before `ensure_test_vfs`, that `expect` aborted the machine — under
    /// `panic = "abort"` there is no unwinding, so the run stopped dead and
    /// the 23 registration groups after `bundle` never executed. This asserts
    /// the harness has a usable root, and asserts it in the `vfs` suite, which
    /// registers before every suite that depends on it.
    fn test_with_vfs_usable_before_init_vfs() -> TestResult {
        test_assert!(
            with_vfs(|vfs| !vfs.mounts.is_empty()),
            "with_vfs must expose at least a root mount"
        );
        test_assert!(
            is_vfs_initialized(),
            "VFS must report initialized once with_vfs has returned"
        );
        // The root must hold bytes, not merely exist: this is the path
        // `bundle`'s store provisioning takes through `pkg::install_file`.
        let path = "/vfs-selftest.bin";
        let handle = match with_vfs(|vfs| vfs.create(path))
            .or_else(|_| with_vfs(|vfs| vfs.lookup_follow(path)))
        {
            Ok(h) => h,
            Err(_) => return TestResult::Fail("root mount must accept a create"),
        };
        let bytes = b"seal";
        test_assert_eq!(with_vfs(|vfs| vfs.write(handle, bytes, 0)).unwrap_or(0), 4);
        let mut buf = [0u8; 4];
        test_assert_eq!(
            with_vfs(|vfs| vfs.read(handle, &mut buf, 0)).unwrap_or(0),
            4
        );
        test_assert_eq!(&buf, bytes);
        let _ = with_vfs(|vfs| vfs.unlink(path));
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "vfs::with_vfs_usable_before_init_vfs",
            test_with_vfs_usable_before_init_vfs,
        );
        crate::testing::register_test(
            "vfs::devfoo_is_not_inside_dev_mount",
            test_devfoo_is_not_inside_dev_mount,
        );
        crate::testing::register_test(
            "vfs::dev_null_still_routes_to_devfs",
            test_dev_null_still_routes_to_devfs,
        );
        crate::testing::register_test(
            "vfs::sibling_prefix_collisions_route_to_root",
            test_sibling_prefix_collisions_route_to_root,
        );
        crate::testing::register_test(
            "vfs::mount_root_exact_match_yields_slash",
            test_mount_root_exact_match_yields_slash,
        );
        crate::testing::register_test(
            "vfs::empty_path_does_not_resolve",
            test_empty_path_does_not_resolve,
        );
        crate::testing::register_test(
            "vfs::relative_path_does_not_resolve",
            test_relative_path_does_not_resolve,
        );
        crate::testing::register_test(
            "vfs::root_alone_yields_root_mount",
            test_root_alone_yields_root_mount,
        );
        crate::testing::register_test(
            "vfs::doubled_separator_falls_through_to_root",
            test_doubled_separator_falls_through_to_root,
        );
        crate::testing::register_test(
            "vfs::trailing_slash_after_mount_yields_slash",
            test_trailing_slash_after_mount_yields_slash,
        );
        crate::testing::register_test(
            "vfs::canonicalize_rejects_empty_and_relative",
            test_canonicalize_rejects_empty_and_relative,
        );
        crate::testing::register_test(
            "vfs::canonicalize_collapses_separators_and_trims",
            test_canonicalize_collapses_separators_and_trims,
        );
        crate::testing::register_test(
            "vfs::canonicalize_resolves_dotdot_and_dot",
            test_canonicalize_resolves_dotdot_and_dot,
        );
        crate::testing::register_test(
            "vfs::doubled_slash_root_secret_denied_for_user",
            test_doubled_slash_root_secret_denied_for_user,
        );
        crate::testing::register_test(
            "vfs::plain_root_secret_still_denied_for_user",
            test_plain_root_secret_still_denied_for_user,
        );
        crate::testing::register_test(
            "vfs::data_and_tmp_still_allowed_for_user",
            test_data_and_tmp_still_allowed_for_user,
        );
        crate::testing::register_test(
            "vfs::uid_zero_bypasses_after_canonicalize",
            test_uid_zero_bypasses_after_canonicalize,
        );
        crate::testing::register_test(
            "vfs::triple_slash_and_trailing_slash_root_denied",
            test_triple_slash_and_trailing_slash_root_denied,
        );
        crate::testing::register_test(
            "vfs::dotdot_escape_through_allowed_prefix_denied",
            test_dotdot_escape_through_allowed_prefix_denied,
        );
    }
}
