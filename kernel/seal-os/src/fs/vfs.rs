// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Virtual File System core — mounts multiple filesystems and routes operations.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use spin::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VfsNodeType {
    File,
    Directory,
    Symlink,
    CharDevice,
    BlockDevice,
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
    fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError>;
    fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError>;
    fn mknod(
        &mut self,
        path: &str,
        node_type: VfsNodeType,
        major: u32,
        minor: u32,
    ) -> Result<VfsHandle, VfsError>;
}

pub struct MountPoint {
    pub prefix: String,
    pub fs: Mutex<Box<dyn FileSystem>>,
}

pub struct Vfs {
    mounts: Vec<MountPoint>,
}

impl Vfs {
    pub fn new() -> Self {
        Self {
            mounts: Vec::new(),
        }
    }

    /// Check Mandatory Access Control for `path` with `perms`.
    fn check_mac(&self, path: &str, perms: crate::security::mac::Permissions) -> Result<(), VfsError> {
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
        HANDLE_PATHS.lock().insert((fs_idx, handle.inode), String::from(path));
        Ok(VfsHandle {
            fs_idx,
            inode: handle.inode,
        })
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
                let dir = current
                    .rfind('/')
                    .map(|i| &current[..i + 1])
                    .unwrap_or("/");
                format!("{}{}", dir, target)
            };
        }
        Err(VfsError::TooManySymlinks)
    }

    pub fn read(&self, handle: VfsHandle, buf: &mut [u8], offset: u64) -> Result<usize, VfsError> {
        if let Some(path) = HANDLE_PATHS.lock().get(&(handle.fs_idx, handle.inode)) {
            self.check_mac(path, crate::security::mac::Permissions::R)?;
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
        HANDLE_PATHS.lock().insert((fs_idx, handle.inode), String::from(path));
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
        HANDLE_PATHS.lock().insert((fs_idx, handle.inode), String::from(path));
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
        HANDLE_PATHS.lock().remove(&(fs_idx, fs_guard.lookup(rel).map(|h| h.inode).unwrap_or(0)));
        Ok(())
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
        HANDLE_PATHS.lock().insert((fs_idx, handle.inode), String::from(path));
        Ok(VfsHandle {
            fs_idx,
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
