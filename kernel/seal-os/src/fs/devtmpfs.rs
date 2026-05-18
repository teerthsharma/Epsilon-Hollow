// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! /dev pseudo-filesystem — device nodes with special read/write callbacks.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::vfs::{FileSystem, VfsDirEntry, VfsError, VfsHandle, VfsNode, VfsNodeType};

#[derive(Debug, Clone)]
struct DevNode {
    name: String,
    node_type: VfsNodeType,
    major: u32,
    minor: u32,
}

pub struct DevTmpFs {
    nodes: BTreeMap<u64, DevNode>,
    dir_entries: BTreeMap<u64, BTreeMap<String, u64>>,
    root_id: u64,
    next_inode: u64,
    content: BTreeMap<u64, Vec<u8>>,
}

impl DevTmpFs {
    pub fn new() -> Self {
        let mut fs = Self {
            nodes: BTreeMap::new(),
            dir_entries: BTreeMap::new(),
            root_id: 1,
            next_inode: 2,
            content: BTreeMap::new(),
        };
        fs.dir_entries.insert(1, BTreeMap::new());
        fs
    }

    pub fn init_default_devices(&mut self) {
        self.mknod("/zero", VfsNodeType::CharDevice, 1, 5).unwrap();
        self.mknod("/null", VfsNodeType::CharDevice, 1, 3).unwrap();
        self.mknod("/random", VfsNodeType::CharDevice, 1, 8).unwrap();
        self.mknod("/console", VfsNodeType::CharDevice, 5, 1).unwrap();
    }

    fn resolve(&self, path: &str) -> Option<u64> {
        let path = path.trim_start_matches('/').trim_end_matches('/');
        if path.is_empty() {
            return Some(self.root_id);
        }
        self.dir_entries.get(&self.root_id)?.get(path).copied()
    }
}

impl FileSystem for DevTmpFs {
    fn lookup(&self, path: &str) -> Result<VfsHandle, VfsError> {
        let id = self.resolve(path).ok_or(VfsError::NotFound)?;
        Ok(VfsHandle { fs_idx: 0, inode: id })
    }

    fn read(&self, handle: VfsHandle, buf: &mut [u8], _offset: u64) -> Result<usize, VfsError> {
        let node = self.nodes.get(&handle.inode).ok_or(VfsError::NotFound)?;
        match node.name.as_str() {
            "zero" => {
                for b in buf.iter_mut() {
                    *b = 0;
                }
                Ok(buf.len())
            }
            "null" => Ok(0),
            "random" => {
                let mut seed = 0x1234_5678u64.wrapping_add(handle.inode);
                for b in buf.iter_mut() {
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    *b = (seed >> 16) as u8;
                }
                Ok(buf.len())
            }
            "console" => Ok(0),
            _ => {
                if let Some(content) = self.content.get(&handle.inode) {
                    let len = buf.len().min(content.len());
                    buf[..len].copy_from_slice(&content[..len]);
                    Ok(len)
                } else {
                    Ok(0)
                }
            }
        }
    }

    fn write(&mut self, handle: VfsHandle, buf: &[u8], _offset: u64) -> Result<usize, VfsError> {
        let node = self.nodes.get(&handle.inode).ok_or(VfsError::NotFound)?;
        match node.name.as_str() {
            "null" => Ok(buf.len()),
            "zero" => Ok(buf.len()),
            "random" => Ok(buf.len()),
            "console" => {
                for &b in buf {
                    crate::serial_print!("{}", b as char);
                }
                Ok(buf.len())
            }
            _ => {
                let content = self
                    .content
                    .entry(handle.inode)
                    .or_insert_with(Vec::new);
                content.clear();
                content.extend_from_slice(buf);
                Ok(buf.len())
            }
        }
    }

    fn create(&mut self, path: &str) -> Result<VfsHandle, VfsError> {
        let path = path.trim_start_matches('/').trim_end_matches('/');
        if path.is_empty() {
            return Err(VfsError::InvalidPath);
        }
        let id = self.next_inode;
        self.next_inode += 1;
        let node = DevNode {
            name: String::from(path),
            node_type: VfsNodeType::File,
            major: 0,
            minor: 0,
        };
        self.nodes.insert(id, node);
        self.dir_entries
            .get_mut(&self.root_id)
            .unwrap()
            .insert(String::from(path), id);
        Ok(VfsHandle { fs_idx: 0, inode: id })
    }

    fn mkdir(&mut self, path: &str) -> Result<VfsHandle, VfsError> {
        let path = path.trim_start_matches('/').trim_end_matches('/');
        if path.is_empty() {
            return Err(VfsError::InvalidPath);
        }
        let id = self.next_inode;
        self.next_inode += 1;
        let node = DevNode {
            name: String::from(path),
            node_type: VfsNodeType::Directory,
            major: 0,
            minor: 0,
        };
        self.nodes.insert(id, node);
        self.dir_entries.insert(id, BTreeMap::new());
        self.dir_entries
            .get_mut(&self.root_id)
            .unwrap()
            .insert(String::from(path), id);
        Ok(VfsHandle { fs_idx: 0, inode: id })
    }

    fn unlink(&mut self, path: &str) -> Result<(), VfsError> {
        let path = path.trim_start_matches('/').trim_end_matches('/');
        let id = self
            .dir_entries
            .get(&self.root_id)
            .and_then(|e| e.get(path).copied())
            .ok_or(VfsError::NotFound)?;
        self.nodes.remove(&id);
        self.dir_entries.get_mut(&self.root_id).unwrap().remove(path);
        self.content.remove(&id);
        Ok(())
    }

    fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> {
        if handle.inode != self.root_id {
            return Err(VfsError::NotSupported);
        }
        let entries = self
            .dir_entries
            .get(&self.root_id)
            .ok_or(VfsError::NotFound)?;
        let mut result = Vec::new();
        for (name, &id) in entries.iter() {
            if let Some(node) = self.nodes.get(&id) {
                result.push(VfsDirEntry {
                    name: name.clone(),
                    node_type: node.node_type,
                });
            }
        }
        Ok(result)
    }

    fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError> {
        let node = self.nodes.get(&handle.inode).ok_or(VfsError::NotFound)?;
        let size = self
            .content
            .get(&handle.inode)
            .map(|c| c.len() as u64)
            .unwrap_or(0);
        let permissions = match node.node_type {
            VfsNodeType::CharDevice => 0o666,
            VfsNodeType::Directory => 0o755,
            _ => 0o644,
        };
        Ok(VfsNode {
            size,
            permissions,
            uid: 0,
            gid: 0,
            mode: permissions,
            atime: 0,
            mtime: 0,
            node_type: node.node_type,
        })
    }

    fn mknod(
        &mut self,
        path: &str,
        node_type: VfsNodeType,
        major: u32,
        minor: u32,
    ) -> Result<VfsHandle, VfsError> {
        let path = path.trim_start_matches('/').trim_end_matches('/');
        if path.is_empty() {
            return Err(VfsError::InvalidPath);
        }
        let id = self.next_inode;
        self.next_inode += 1;
        let node = DevNode {
            name: String::from(path),
            node_type,
            major,
            minor,
        };
        self.nodes.insert(id, node);
        self.dir_entries
            .get_mut(&self.root_id)
            .unwrap()
            .insert(String::from(path), id);
        Ok(VfsHandle { fs_idx: 0, inode: id })
    }
}
