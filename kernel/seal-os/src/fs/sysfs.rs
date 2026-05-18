// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! /sys pseudo-filesystem — hardware and driver state.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::vfs::{FileSystem, VfsDirEntry, VfsError, VfsHandle, VfsNode, VfsNodeType};
use crate::fs::encoder::hash_bytes;

pub struct SysFs {
    devices: Vec<crate::drivers::pci::PciDevice>,
}

impl SysFs {
    pub fn new() -> Self {
        Self {
            devices: crate::drivers::pci::get_devices(),
        }
    }

    fn path_inode(path: &str) -> u64 {
        hash_bytes(path.as_bytes())
    }
}

impl FileSystem for SysFs {
    fn lookup(&self, path: &str) -> Result<VfsHandle, VfsError> {
        let path = path.trim_start_matches('/').trim_end_matches('/');
        match path {
            "" | "bus" | "bus/pci" | "bus/pci/devices" => Ok(VfsHandle {
                fs_idx: 0,
                inode: Self::path_inode(path),
            }),
            p if p.starts_with("bus/pci/devices/") => {
                let name = &p["bus/pci/devices/".len()..];
                if name.is_empty() {
                    return Ok(VfsHandle {
                        fs_idx: 0,
                        inode: Self::path_inode(p),
                    });
                }
                if self
                    .devices
                    .iter()
                    .any(|d| format!("{:04x}:{:04x}", d.vendor_id, d.device_id) == name)
                {
                    Ok(VfsHandle {
                        fs_idx: 0,
                        inode: Self::path_inode(p),
                    })
                } else {
                    Err(VfsError::NotFound)
                }
            }
            _ => Err(VfsError::NotFound),
        }
    }

    fn read(&self, handle: VfsHandle, buf: &mut [u8], _offset: u64) -> Result<usize, VfsError> {
        for dev in &self.devices {
            let path = format!("bus/pci/devices/{:04x}:{:04x}", dev.vendor_id, dev.device_id);
            if Self::path_inode(&path) == handle.inode {
                let content = format!(
                    "vendor: 0x{:04x}\ndevice: 0x{:04x}\nclass: 0x{:02x}\nsubclass: 0x{:02x}\nprog_if: 0x{:02x}\n",
                    dev.vendor_id, dev.device_id, dev.class, dev.subclass, dev.prog_if
                );
                let bytes = content.as_bytes();
                let len = buf.len().min(bytes.len());
                buf[..len].copy_from_slice(&bytes[..len]);
                return Ok(len);
            }
        }
        Err(VfsError::NotFound)
    }

    fn write(&mut self, _handle: VfsHandle, _buf: &[u8], _offset: u64) -> Result<usize, VfsError> {
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

    fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> {
        match handle.inode {
            i if i == Self::path_inode("") => Ok(vec![VfsDirEntry {
                name: String::from("bus"),
                node_type: VfsNodeType::Directory,
            }]),
            i if i == Self::path_inode("bus") => Ok(vec![VfsDirEntry {
                name: String::from("pci"),
                node_type: VfsNodeType::Directory,
            }]),
            i if i == Self::path_inode("bus/pci") => Ok(vec![VfsDirEntry {
                name: String::from("devices"),
                node_type: VfsNodeType::Directory,
            }]),
            i if i == Self::path_inode("bus/pci/devices") => Ok(self
                .devices
                .iter()
                .map(|d| VfsDirEntry {
                    name: format!("{:04x}:{:04x}", d.vendor_id, d.device_id),
                    node_type: VfsNodeType::File,
                })
                .collect()),
            _ => Err(VfsError::NotFound),
        }
    }

    fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError> {
        let dirs = ["", "bus", "bus/pci", "bus/pci/devices"];
        for dir in &dirs {
            if handle.inode == Self::path_inode(dir) {
                return Ok(VfsNode {
                    size: 0,
                    permissions: 0o555,
                    uid: 0,
                    gid: 0,
                    mode: 0o040555,
                    atime: 0,
                    mtime: 0,
                    node_type: VfsNodeType::Directory,
                });
            }
        }
        for dev in &self.devices {
            let path = format!("bus/pci/devices/{:04x}:{:04x}", dev.vendor_id, dev.device_id);
            if Self::path_inode(&path) == handle.inode {
                return Ok(VfsNode {
                    size: 0,
                    permissions: 0o444,
                    uid: 0,
                    gid: 0,
                    mode: 0o100444,
                    atime: 0,
                    mtime: 0,
                    node_type: VfsNodeType::File,
                });
            }
        }
        Err(VfsError::NotFound)
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
