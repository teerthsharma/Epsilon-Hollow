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
pub mod manifold_fs;
pub mod path_cache;
pub mod prefetch;
pub mod pipe;
pub mod procfs;
pub mod sysfs;
pub mod vfs;
pub mod voronoi_cap;

use alloc::boxed::Box;

/// Initialize the global VFS and mount all default filesystems.
/// Must be called after driver init (so sysfs sees PCI devices).
pub fn init_vfs() -> Result<(), vfs::VfsError> {
    let mut v = vfs::Vfs::new();

    let root_fs: Box<dyn vfs::FileSystem> = match manifold_fs::ManifoldFS::try_mount_disk() {
        Ok(fs) => {
            crate::serial_println!("[VFS] ManifoldFS mounted from disk");
            Box::new(fs)
        }
        Err(_) => {
            // Try Ext2
            let mut ext2 = ext2::Ext2Fs::new(0x800); // 0x800 is the AHCI device number
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
