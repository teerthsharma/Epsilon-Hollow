// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Filesystem layer — VFS + ManifoldFS + pseudo-filesystems.

pub mod block_store;
pub mod buffer_cache;
pub mod devtmpfs;
pub mod dir_hash;
pub mod encoder;
pub mod ext2;
pub mod fat;
pub mod inode_slab;
pub mod journal;
pub mod manifold_fs;
pub mod path_cache;
pub mod pipe;
pub mod prefetch;
pub mod procfs;
pub mod sysfs;
pub mod topcrypt;
pub mod vfs;
pub mod voronoi_cap;

use alloc::boxed::Box;

/// Flush all mounted filesystems.
pub fn sync() -> Result<(), vfs::VfsError> {
    let mut guard = vfs::VFS.lock();
    if let Some(ref mut v) = *guard {
        for mount in &mut v.mounts {
            let mut fs_guard = mount.fs.lock();
            fs_guard.sync()?;
        }
    }
    Ok(())
}

/// Initialize the global VFS and mount all default filesystems.
/// Must be called after driver init (so sysfs sees PCI devices).
pub fn init_vfs() -> Result<(), vfs::VfsError> {
    let mut v = vfs::Vfs::new();

    let mut manifold = match manifold_fs::ManifoldFS::try_mount_disk() {
        Ok(fs) => {
            crate::serial_println!("[VFS] ManifoldFS mounted from disk");
            Some(fs)
        }
        Err(_) => None,
    };

    // If ManifoldFS is primary and ext2 is available, use ext2 as the raw-byte backend.
    let root_fs: Box<dyn vfs::FileSystem> = if let Some(ref mut mfs) = manifold {
        let mut ext2 = ext2::Ext2Fs::new(0x800);
        if ext2.mount().is_ok() {
            crate::serial_println!("[VFS] Ext2 attached as ManifoldFS persistence backend");
            mfs.set_ext2_backend(ext2);
        }
        Box::new(manifold.take().unwrap())
    } else {
        let mut ext2 = ext2::Ext2Fs::new(0x800);
        match ext2.mount() {
            Ok(_) => {
                crate::serial_println!("[VFS] Ext2 mounted from AHCI port 0");
                Box::new(ext2)
            }
            Err(_) => {
                crate::serial_println!("[VFS] No persistent disk found. Falling back to ramfs.");
                Box::new(manifold_fs::ManifoldFS::new_ramfs())
            }
        }
    };

    v.mount("/", root_fs)?;
    v.mount("/proc", Box::new(procfs::ProcFs::new()))?;
    v.mount("/sys", Box::new(sysfs::SysFs::new()))?;
    let mut devfs = devtmpfs::DevTmpFs::new();
    devfs.init_default_devices();
    v.mount("/dev", Box::new(devfs))?;
    let pipe_fs_idx = v.mounts.len();
    v.mount("/pipe", Box::new(pipe::PipeFs::new()))?;
    pipe::set_pipe_fs_idx(pipe_fs_idx);
    *vfs::VFS.lock() = Some(v);
    Ok(())
}
