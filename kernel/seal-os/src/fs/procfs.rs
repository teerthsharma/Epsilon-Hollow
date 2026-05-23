// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! /proc pseudo-filesystem — dynamically generated kernel state.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::vfs::{FileSystem, VfsDirEntry, VfsError, VfsHandle, VfsNode, VfsNodeType};

pub struct ProcFs;

impl ProcFs {
    pub fn new() -> Self {
        Self
    }
}

fn cpu_vendor_string() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let cpuid = unsafe { core::arch::x86_64::__cpuid(0) };
        let mut bytes = [0u8; 12];
        bytes[0..4].copy_from_slice(&cpuid.ebx.to_le_bytes());
        bytes[4..8].copy_from_slice(&cpuid.edx.to_le_bytes());
        bytes[8..12].copy_from_slice(&cpuid.ecx.to_le_bytes());
        String::from_utf8_lossy(&bytes).into_owned()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        String::from("unknown")
    }
}

impl FileSystem for ProcFs {
    fn lookup(&self, path: &str) -> Result<VfsHandle, VfsError> {
        let path = path.trim_start_matches('/').trim_end_matches('/');
        let inode = match path {
            "" | "." => 1,
            "version" | "1/version" => 2,
            "uptime" | "1/uptime" => 3,
            "cpuinfo" | "1/cpuinfo" => 4,
            "meminfo" | "1/meminfo" => 5,
            "self" => 6,
            "1" => 7,
            _ => return Err(VfsError::NotFound),
        };
        Ok(VfsHandle { fs_idx: 0, inode })
    }

    fn read(&self, handle: VfsHandle, buf: &mut [u8], _offset: u64) -> Result<usize, VfsError> {
        let content = match handle.inode {
            1 => return Err(VfsError::NotADirectory),
            2 => format!("Seal OS v{}\n", crate::VERSION),
            3 => format!("{}\n", crate::drivers::interrupts::ticks()),
            4 => format!(
                "processor: 0\nvendor_id: {}\ncpu cores: 1\n",
                cpu_vendor_string()
            ),
            5 => {
                let (allocated, total) = crate::memory::heap_stats();
                let free = total - allocated;
                format!("total: {}\nfree: {}\nallocated: {}\n", total, free, allocated)
            }
            6 => String::from("1/"),
            _ => return Err(VfsError::NotFound),
        };
        let bytes = content.as_bytes();
        let len = buf.len().min(bytes.len());
        buf[..len].copy_from_slice(&bytes[..len]);
        Ok(len)
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

    fn rmdir(&mut self, _path: &str) -> Result<(), VfsError> {
        Err(VfsError::NotSupported)
    }

    fn rename(&mut self, _old: &str, _new: &str) -> Result<(), VfsError> {
        Err(VfsError::NotSupported)
    }

    fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> {
        if handle.inode != 1 && handle.inode != 7 {
            return Err(VfsError::NotADirectory);
        }
        Ok(vec![
            VfsDirEntry {
                name: String::from("version"),
                node_type: VfsNodeType::File,
            },
            VfsDirEntry {
                name: String::from("uptime"),
                node_type: VfsNodeType::File,
            },
            VfsDirEntry {
                name: String::from("cpuinfo"),
                node_type: VfsNodeType::File,
            },
            VfsDirEntry {
                name: String::from("meminfo"),
                node_type: VfsNodeType::File,
            },
            VfsDirEntry {
                name: String::from("self"),
                node_type: VfsNodeType::Symlink,
            },
        ])
    }

    fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError> {
        match handle.inode {
            1 => Ok(VfsNode {
                size: 0,
                permissions: 0o555,
                uid: 0,
                gid: 0,
                mode: 0o040555,
                atime: 0,
                mtime: 0,
                node_type: VfsNodeType::Directory,
                major: 0,
                minor: 0,
            }),
            2 | 3 | 4 | 5 => Ok(VfsNode {
                size: 0,
                permissions: 0o444,
                uid: 0,
                gid: 0,
                mode: 0o100444,
                atime: 0,
                mtime: 0,
                node_type: VfsNodeType::File,
                major: 0,
                minor: 0,
            }),
            6 => Ok(VfsNode {
                size: 2,
                permissions: 0o777,
                uid: 0,
                gid: 0,
                mode: 0o120777,
                atime: 0,
                mtime: 0,
                node_type: VfsNodeType::Symlink,
                major: 0,
                minor: 0,
            }),
            7 => Ok(VfsNode {
                size: 0,
                permissions: 0o555,
                uid: 0,
                gid: 0,
                mode: 0o040555,
                atime: 0,
                mtime: 0,
                node_type: VfsNodeType::Directory,
                major: 0,
                minor: 0,
            }),
            _ => Err(VfsError::NotFound),
        }
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
