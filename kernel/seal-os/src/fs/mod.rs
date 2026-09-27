// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Filesystem layer — VFS + ManifoldFS + pseudo-filesystems.

pub mod block_store;
pub mod buffer_cache;
pub mod devtmpfs;
pub mod dir_hash;
pub mod encoder;
pub mod ext2;
pub mod ext2_format;
pub mod fat;
pub mod gpt;
pub mod inode_slab;
pub mod journal;
pub mod manifold_fs;
pub mod parity;
pub mod path_cache;
pub mod pipe;
pub mod prefetch;
pub mod procfs;
pub mod sysfs;
pub mod topcrypt;
pub mod vfs;
pub mod voronoi_cap;

use alloc::boxed::Box;
use vfs::FileSystem as _;

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

/// Disks probed for a root filesystem, in order, with the name the boot log
/// gives each.
// ponytail: the two transports that register disks today; NVMe and USB mass
// storage join this list when they can carry a root.
const ROOT_DISKS: [(u32, &str); 2] = [
    (crate::drivers::block::ahci::AHCI_DEV_NUM, "AHCI port 0"),
    (
        crate::drivers::block::virtio_blk::VIRTIO_BLK_DEV_NUM,
        "virtio-blk 0xfd00",
    ),
];

/// Initialize the global VFS and mount all default filesystems.
/// Must be called after driver init (so sysfs sees PCI devices).
pub fn init_vfs() -> Result<(), vfs::VfsError> {
    let mut v = vfs::Vfs::new();

    // Logs `[disk::ahci] First disk readable`, which the VM proof requires.
    let _ = crate::drivers::disk::ahci::first_disk();
    let manifold = ROOT_DISKS.iter().find_map(|&(dev, name)| {
        let fs = manifold_fs::ManifoldFS::try_mount_disk(dev).ok()?;
        crate::serial_println!("[VFS] ManifoldFS mounted from disk ({})", name);
        Some((dev, name, fs))
    });

    // If ManifoldFS is primary and ext2 is available, use ext2 as the raw-byte backend.
    // A volume `Ext2Fs::mount` refuses (unsupported INCOMPAT features) is never
    // attached or mounted; the refusal and its feature bits are logged by `mount`.
    let root_fs: Box<dyn vfs::FileSystem> = if let Some((dev, name, mut mfs)) = manifold {
        let mut ext2 = ext2::Ext2Fs::new(dev);
        if ext2.mount().is_ok() {
            if ext2.is_read_only() {
                crate::serial_println!(
                    "[VFS] Ext2 on {} is read-only; not attached as ManifoldFS persistence backend",
                    name
                );
            } else {
                crate::serial_println!("[VFS] Ext2 attached as ManifoldFS persistence backend");
                mfs.set_ext2_backend(ext2);
            }
        }
        Box::new(mfs)
    } else {
        let ext2_root = ROOT_DISKS.iter().find_map(|&(dev, name)| {
            let mut ext2 = ext2::Ext2Fs::new(dev);
            ext2.mount().ok()?;
            Some((name, ext2))
        });
        match ext2_root {
            Some((name, ext2)) if ext2.is_read_only() => {
                crate::serial_println!("[VFS] Ext2 mounted read-only from {}", name);
                Box::new(ext2)
            }
            Some((name, ext2)) => {
                crate::serial_println!("[VFS] Ext2 mounted from {}", name);
                Box::new(ext2)
            }
            None => {
                crate::serial_println!("[VFS] No persistent disk found. Falling back to ramfs.");
                Box::new(manifold_fs::ManifoldFS::new_ramfs())
            }
        }
    };

    v.mount("/", root_fs)?;
    // The audit log is written on every audited syscall. On a read-only root
    // it goes to memory instead of being re-buffered forever against a volume
    // that refuses it.
    if v.is_read_only("/var/log") {
        v.mount("/var/log", Box::new(manifold_fs::ManifoldFS::new_ramfs()))?;
        crate::serial_println!("[VFS] Root is read-only; /var/log is a ramfs");
    }
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
