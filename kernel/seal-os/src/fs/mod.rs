// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Filesystem layer — VFS + ManifoldFS + pseudo-filesystems.

pub mod block_store;
pub mod devtmpfs;
pub mod dir_hash;
pub mod encoder;
pub mod inode_slab;
pub mod manifold_fs;
pub mod path_cache;
pub mod prefetch;
pub mod procfs;
pub mod sysfs;
pub mod vfs;
pub mod voronoi_cap;

use alloc::boxed::Box;

/// Initialize the global VFS and mount all default filesystems.
/// Must be called after driver init (so sysfs sees PCI devices).
pub fn init_vfs() -> Result<(), vfs::VfsError> {
    let mut v = vfs::Vfs::new();

    let root_fs = match manifold_fs::ManifoldFS::try_mount_disk() {
        Ok(fs) => {
            crate::serial_println!("[VFS] ManifoldFS mounted from disk");
            fs
        }
        Err(e) => {
            crate::serial_println!("[VFS] ManifoldFS disk mount failed: {}. Falling back to ramfs.", e);
            manifold_fs::ManifoldFS::new_ramfs()
        }
    };

    v.mount("/", Box::new(root_fs))?;
    v.mount("/proc", Box::new(procfs::ProcFs::new()))?;
    v.mount("/sys", Box::new(sysfs::SysFs::new()))?;
    let mut devfs = devtmpfs::DevTmpFs::new();
    devfs.init_default_devices();
    v.mount("/dev", Box::new(devfs))?;
    *vfs::VFS.lock() = Some(v);
    Ok(())
}
