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
    fn sync(&mut self) -> Result<(), VfsError> {
        Ok(())
    }
    /// Synchronize a single file's dirty data to persistent storage.
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

    fn find_mount<'a>(&self, path: &'a str) -> Option<(usize, &'a str)> {
        let mut best_idx = None;
        let mut best_len = 0;
        for (i, mp) in self.mounts.iter().enumerate() {
            if path.starts_with(&mp.prefix) && mp.prefix.len() > best_len {
                best_len = mp.prefix.len();
                best_idx = Some(i);
            }
        }
        best_idx.map(move |i| {
            let rel = &path[best_len..];
            let rel = if rel.is_empty() { "/" } else { rel };
            (i, rel)
        })
    }

    pub fn lookup(&self, path: &str) -> Result<VfsHandle, VfsError> {
        self.check_mac(path, crate::security::mac::Permissions::R)?;
        let (fs_idx, rel) = self.find_mount(path).ok_or(VfsError::NotFound)?;
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
            .insert((fs_idx, handle.inode), String::from(path));
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
            path,
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
        self.check_mac(path, crate::security::mac::Permissions::W)?;
        let (fs_idx, rel) = self.find_mount(path).ok_or(VfsError::NotFound)?;
        let mount = &self.mounts[fs_idx];
        let mut fs_guard = mount.fs.lock();
        let handle = fs_guard.create(rel)?;
        HANDLE_PATHS
            .lock()
            .insert((fs_idx, handle.inode), String::from(path));
        Ok(VfsHandle {
            fs_idx,
            inode: handle.inode,
        })
    }

    pub fn mkdir(&self, path: &str) -> Result<VfsHandle, VfsError> {
        self.check_mac(path, crate::security::mac::Permissions::W)?;
        let (fs_idx, rel) = self.find_mount(path).ok_or(VfsError::NotFound)?;
        let mount = &self.mounts[fs_idx];
        let mut fs_guard = mount.fs.lock();
        let handle = fs_guard.mkdir(rel)?;
        HANDLE_PATHS
            .lock()
            .insert((fs_idx, handle.inode), String::from(path));
        Ok(VfsHandle {
            fs_idx,
            inode: handle.inode,
        })
    }

    pub fn unlink(&self, path: &str) -> Result<(), VfsError> {
        self.check_mac(path, crate::security::mac::Permissions::W)?;
        let (fs_idx, rel) = self.find_mount(path).ok_or(VfsError::NotFound)?;
        let mount = &self.mounts[fs_idx];
        let mut fs_guard = mount.fs.lock();
        fs_guard.unlink(rel)?;
        HANDLE_PATHS
            .lock()
            .remove(&(fs_idx, fs_guard.lookup(rel).map(|h| h.inode).unwrap_or(0)));
        Ok(())
    }

    pub fn rmdir(&self, path: &str) -> Result<(), VfsError> {
        self.check_mac(path, crate::security::mac::Permissions::W)?;
        let (fs_idx, rel) = self.find_mount(path).ok_or(VfsError::NotFound)?;
        let mount = &self.mounts[fs_idx];
        let mut fs_guard = mount.fs.lock();
        fs_guard.rmdir(rel)?;
        HANDLE_PATHS
            .lock()
            .remove(&(fs_idx, fs_guard.lookup(rel).map(|h| h.inode).unwrap_or(0)));
        Ok(())
    }

    pub fn rename(&self, old: &str, new: &str) -> Result<(), VfsError> {
        self.check_mac(old, crate::security::mac::Permissions::W)?;
        self.check_mac(new, crate::security::mac::Permissions::W)?;
        let (old_fs_idx, old_rel) = self.find_mount(old).ok_or(VfsError::NotFound)?;
        let (new_fs_idx, new_rel) = self.find_mount(new).ok_or(VfsError::NotFound)?;
        if old_fs_idx == new_fs_idx {
            let mount = &self.mounts[old_fs_idx];
            let mut fs_guard = mount.fs.lock();
            fs_guard.rename(old_rel, new_rel)?;
            if let Ok(handle) = fs_guard.lookup(new_rel) {
                HANDLE_PATHS
                    .lock()
                    .insert((old_fs_idx, handle.inode), String::from(new));
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
                .insert((new_fs_idx, new_handle.inode), String::from(new));
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
        self.check_mac(path, crate::security::mac::Permissions::W)?;
        let (fs_idx, rel) = self.find_mount(path).ok_or(VfsError::NotFound)?;
        let mount = &self.mounts[fs_idx];
        let mut fs_guard = mount.fs.lock();
        let handle = fs_guard.mknod(rel, node_type, major, minor)?;
        HANDLE_PATHS
            .lock()
            .insert((fs_idx, handle.inode), String::from(path));
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

pub static VFS: Mutex<Option<Vfs>> = Mutex::new(None);

/// Reverse mapping from handle to path for MAC checks on read/write.
static HANDLE_PATHS: Mutex<BTreeMap<(usize, u64), String>> = Mutex::new(BTreeMap::new());

pub fn with_vfs<F, R>(f: F) -> R
where
    F: FnOnce(&Vfs) -> R,
{
    let guard = VFS.lock();
    let vfs = guard.as_ref().expect("VFS not initialized");
    f(vfs)
}

/// Returns `true` if the VFS has been initialized.
pub fn is_vfs_initialized() -> bool {
    VFS.lock().is_some()
}
