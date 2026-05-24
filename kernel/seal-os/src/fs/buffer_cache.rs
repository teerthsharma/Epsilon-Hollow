use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use crate::drivers::block::{read_block, write_block};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    Invalid,
    Clean,
    Dirty,
    Locked,
}

pub struct Buffer {
    pub dev: u32,
    pub block_id: u64,
    pub data: Vec<u8>,
    pub state: BufferState,
    pub dirty: bool,
}

impl Buffer {
    pub fn new(dev: u32, block_id: u64, block_size: usize) -> Self {
        let mut data = Vec::with_capacity(block_size);
        data.resize(block_size, 0);
        Self {
            dev,
            block_id,
            data,
            state: BufferState::Clean,
            dirty: false,
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn mark_dirty(&mut self) {
        self.state = BufferState::Dirty;
        self.dirty = true;
    }
}

pub struct BufferCache {
    // We use BTreeMap as HashMap is typically unavailable in no_std
    cache: BTreeMap<(u32, u64), Buffer>,
    lru: Vec<(u32, u64)>, // Front is oldest, back is newest
    capacity: usize,
    block_size: usize,
}

impl BufferCache {
    pub fn new(capacity: usize, block_size: usize) -> Self {
        Self {
            cache: BTreeMap::new(),
            lru: Vec::new(),
            capacity,
            block_size,
        }
    }

    pub fn get_block(&mut self, dev: u32, block_id: u64) -> Option<&mut Buffer> {
        if !self.cache.contains_key(&(dev, block_id)) {
            if self.cache.len() >= self.capacity {
                self.evict();
            }
            let mut new_buf = Buffer::new(dev, block_id, self.block_size);
            if read_block(dev, block_id, &mut new_buf.data).is_err() {
                return None;
            }
            self.cache.insert((dev, block_id), new_buf);
        }

        self.update_lru(dev, block_id);
        self.cache.get_mut(&(dev, block_id))
    }

    pub fn write_block(&mut self, dev: u32, block_id: u64, data: &[u8]) {
        if !self.cache.contains_key(&(dev, block_id)) {
            if self.cache.len() >= self.capacity {
                self.evict();
            }
            self.cache.insert(
                (dev, block_id),
                Buffer::new(dev, block_id, self.block_size),
            );
        }

        self.update_lru(dev, block_id);

        let buf = self.cache.get_mut(&(dev, block_id)).unwrap();
        let len = data.len().min(buf.data.len());
        buf.data[..len].copy_from_slice(&data[..len]);
        buf.state = BufferState::Dirty;
        buf.dirty = true;
    }

    pub fn flush(&mut self, dev: Option<u32>) {
        for buf in self.cache.values_mut() {
            if buf.dirty {
                if let Some(d) = dev {
                    if buf.dev != d {
                        continue;
                    }
                }
                // Flush data to the block device.
                if write_block(buf.dev, buf.block_id, &buf.data).is_ok() {
                    buf.state = BufferState::Clean;
                    buf.dirty = false;
                }
            }
        }
    }

    pub fn sync(&mut self) {
        self.flush(None);
    }

    fn update_lru(&mut self, dev: u32, block_id: u64) {
        if let Some(pos) = self.lru.iter().position(|&k| k == (dev, block_id)) {
            self.lru.remove(pos);
        }
        self.lru.push((dev, block_id));
    }

    fn evict(&mut self) {
        if self.lru.is_empty() {
            return;
        }

        let mut evict_idx = None;

        // Prefer evicting the oldest clean block
        for (i, key) in self.lru.iter().enumerate() {
            if let Some(buf) = self.cache.get(key) {
                if !buf.dirty {
                    evict_idx = Some(i);
                    break;
                }
            }
        }

        // If no clean blocks are available, fallback to the absolute oldest block.
        // We must flush it before eviction.
        let evict_idx = evict_idx.unwrap_or(0);

        let key = self.lru.remove(evict_idx);
        if let Some(buf) = self.cache.remove(&key) {
            if buf.dirty {
                let _ = write_block(buf.dev, buf.block_id, &buf.data);
            }
        }
    }
}
