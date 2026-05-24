// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! In-memory pipe filesystem — ring-buffer backed FIFO.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use spin::Mutex;

use super::vfs::{FileSystem, VfsDirEntry, VfsError, VfsHandle, VfsNode, VfsNodeType};

const PIPE_BUF_CAP: usize = 64 * 1024;

/// A single pipe — ring buffer with reader/writer reference counts.
pub struct Pipe {
    buf: [u8; PIPE_BUF_CAP],
    read_pos: usize,
    write_pos: usize,
    len: usize,
    readers: u32,
    writers: u32,
}

impl Pipe {
    pub fn new() -> Self {
        Self {
            buf: [0u8; PIPE_BUF_CAP],
            read_pos: 0,
            write_pos: 0,
            len: 0,
            readers: 1,
            writers: 1,
        }
    }

    /// Copy up to `buf.len()` bytes from the pipe into `buf`.
    /// Returns the number of bytes copied (0 if empty).
    pub fn read(&mut self, buf: &mut [u8]) -> usize {
        if self.len == 0 {
            return 0;
        }
        let to_read = buf.len().min(self.len);
        for i in 0..to_read {
            buf[i] = self.buf[self.read_pos];
            self.read_pos = (self.read_pos + 1) % PIPE_BUF_CAP;
        }
        self.len -= to_read;
        to_read
    }

    /// Copy as much of `data` as possible into the pipe.
    /// Returns the number of bytes written (may be less than `data.len()`).
    pub fn write(&mut self, data: &[u8]) -> usize {
        let available = PIPE_BUF_CAP - self.len;
        let to_write = data.len().min(available);
        for i in 0..to_write {
            self.buf[self.write_pos] = data[i];
            self.write_pos = (self.write_pos + 1) % PIPE_BUF_CAP;
        }
        self.len += to_write;
        to_write
    }
}

/// Global registry mapping pipe inode numbers to the shared pipe instance.
static PIPE_REGISTRY: Mutex<BTreeMap<u64, Arc<Mutex<Pipe>>>> = Mutex::new(BTreeMap::new());
static NEXT_PIPE_INODE: AtomicU64 = AtomicU64::new(1);
static PIPE_FS_IDX: AtomicUsize = AtomicUsize::new(0);

/// Set the VFS mount index for the pipe filesystem.
/// Called once during `init_vfs`.
pub fn set_pipe_fs_idx(idx: usize) {
    PIPE_FS_IDX.store(idx, Ordering::SeqCst);
}

/// Create a new pipe, register it, and return read/write handles.
pub fn create_pipe() -> (VfsHandle, VfsHandle) {
    let pipe = Arc::new(Mutex::new(Pipe::new()));
    let inode = NEXT_PIPE_INODE.fetch_add(1, Ordering::SeqCst);
    PIPE_REGISTRY.lock().insert(inode, pipe);
    let fs_idx = PIPE_FS_IDX.load(Ordering::SeqCst);
    let read_handle = VfsHandle { fs_idx, inode };
    let write_handle = VfsHandle { fs_idx, inode };
    (read_handle, write_handle)
}

/// Look up a pipe by inode.
pub fn get_pipe(inode: u64) -> Option<Arc<Mutex<Pipe>>> {
    PIPE_REGISTRY.lock().get(&inode).cloned()
}

/// Remove a pipe from the registry.
pub fn remove_pipe(inode: u64) {
    PIPE_REGISTRY.lock().remove(&inode);
}

pub struct PipeFs;

impl PipeFs {
    pub fn new() -> Self {
        Self
    }
}

impl FileSystem for PipeFs {
    fn lookup(&self, _path: &str) -> Result<VfsHandle, VfsError> {
        Err(VfsError::InvalidOperation)
    }

    fn read(&self, handle: VfsHandle, buf: &mut [u8], _offset: u64) -> Result<usize, VfsError> {
        let pipe = get_pipe(handle.inode).ok_or(VfsError::NotFound)?;
        let mut guard = pipe.lock();
        Ok(guard.read(buf))
    }

    fn write(&mut self, handle: VfsHandle, buf: &[u8], _offset: u64) -> Result<usize, VfsError> {
        let pipe = get_pipe(handle.inode).ok_or(VfsError::NotFound)?;
        let mut guard = pipe.lock();
        Ok(guard.write(buf))
    }

    fn create(&mut self, _path: &str) -> Result<VfsHandle, VfsError> {
        Err(VfsError::InvalidOperation)
    }

    fn mkdir(&mut self, _path: &str) -> Result<VfsHandle, VfsError> {
        Err(VfsError::InvalidOperation)
    }

    fn unlink(&mut self, _path: &str) -> Result<(), VfsError> {
        Err(VfsError::InvalidOperation)
    }

    fn rmdir(&mut self, _path: &str) -> Result<(), VfsError> {
        Err(VfsError::InvalidOperation)
    }

    fn rename(&mut self, _old: &str, _new: &str) -> Result<(), VfsError> {
        Err(VfsError::InvalidOperation)
    }

    fn readdir(&self, _handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> {
        Err(VfsError::InvalidOperation)
    }

    fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError> {
        let pipe = get_pipe(handle.inode).ok_or(VfsError::NotFound)?;
        let guard = pipe.lock();
        Ok(VfsNode {
            size: guard.len as u64,
            permissions: 0o600,
            uid: 0,
            gid: 0,
            mode: 0o10600, // S_IFIFO | 0600
            atime: 0,
            mtime: 0,
            node_type: VfsNodeType::Pipe,
            major: 0,
            minor: 0,
        })
    }

    fn mknod(
        &mut self,
        _path: &str,
        _node_type: VfsNodeType,
        _major: u32,
        _minor: u32,
    ) -> Result<VfsHandle, VfsError> {
        Err(VfsError::InvalidOperation)
    }
}
