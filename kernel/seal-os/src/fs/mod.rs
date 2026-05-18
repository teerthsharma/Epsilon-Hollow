// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Filesystem layer — VFS + ManifoldFS + pseudo-filesystems.

pub mod devtmpfs;
pub mod encoder;
pub mod manifold_fs;
pub mod prefetch;
pub mod procfs;
pub mod sysfs;
pub mod vfs;

use alloc::boxed::Box;

/// Initialize the global VFS and mount all default filesystems.
/// Must be called after driver init (so sysfs sees PCI devices).
pub fn init_vfs() {
    let mut v = vfs::Vfs::new();
    v.mount("/", Box::new(manifold_fs::ManifoldFS::new()))
        .unwrap();
    v.mount("/proc", Box::new(procfs::ProcFs::new()))
        .unwrap();
    v.mount("/sys", Box::new(sysfs::SysFs::new()))
        .unwrap();
    let mut devfs = devtmpfs::DevTmpFs::new();
    devfs.init_default_devices();
    v.mount("/dev", Box::new(devfs)).unwrap();
    *vfs::VFS.lock() = Some(v);
}
