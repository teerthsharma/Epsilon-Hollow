use crate::drivers::block::{read_block, write_block, BlockError};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

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
        let data = vec![0; block_size];
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

    /// Sectors spanned by one cache block. Callers index by filesystem block
    /// number; block devices are addressed in 512-byte sectors, so the
    /// translation belongs here rather than at every call site.
    fn sectors_per_block(&self) -> u64 {
        (self.block_size as u64 / 512).max(1)
    }

    pub fn get_block(&mut self, dev: u32, block_id: u64) -> Option<&mut Buffer> {
        if !self.cache.contains_key(&(dev, block_id)) {
            if self.cache.len() >= self.capacity {
                self.evict().ok()?;
            }
            let lba = block_id * self.sectors_per_block();
            let mut new_buf = Buffer::new(dev, block_id, self.block_size);
            if read_block(dev, lba, &mut new_buf.data).is_err() {
                return None;
            }
            self.cache.insert((dev, block_id), new_buf);
        }

        self.update_lru(dev, block_id);
        self.cache.get_mut(&(dev, block_id))
    }

    pub fn write_block(&mut self, dev: u32, block_id: u64, data: &[u8]) -> Result<(), BlockError> {
        if !self.cache.contains_key(&(dev, block_id)) {
            if self.cache.len() >= self.capacity {
                self.evict()?;
            }
            self.cache
                .insert((dev, block_id), Buffer::new(dev, block_id, self.block_size));
        }

        self.update_lru(dev, block_id);

        let buf = self.cache.get_mut(&(dev, block_id)).unwrap();
        let len = data.len().min(buf.data.len());
        buf.data[..len].copy_from_slice(&data[..len]);
        buf.state = BufferState::Dirty;
        buf.dirty = true;
        Ok(())
    }

    pub fn flush(&mut self, dev: Option<u32>) {
        let sectors_per_block = self.sectors_per_block();
        for buf in self.cache.values_mut() {
            if buf.dirty {
                if let Some(d) = dev {
                    if buf.dev != d {
                        continue;
                    }
                }
                // Flush data to the block device.
                if write_block(buf.dev, buf.block_id * sectors_per_block, &buf.data).is_ok() {
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

    fn evict(&mut self) -> Result<(), BlockError> {
        if self.lru.is_empty() {
            return Ok(());
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

        // Otherwise evict the oldest dirty block that writes back. A block
        // whose write fails stays cached and dirty; if none writes back, the
        // error goes to the caller and the insert that needed the slot fails.
        let sectors_per_block = self.sectors_per_block();
        let mut err = BlockError::IoError;
        if evict_idx.is_none() {
            for (i, key) in self.lru.iter().enumerate() {
                if let Some(buf) = self.cache.get(key) {
                    match write_block(buf.dev, buf.block_id * sectors_per_block, &buf.data) {
                        Ok(()) => {
                            evict_idx = Some(i);
                            break;
                        }
                        Err(e) => err = e,
                    }
                }
            }
        }

        let key = self.lru.remove(evict_idx.ok_or(err)?);
        self.cache.remove(&key);
        Ok(())
    }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::drivers::block::{register_block_device, BlockDevice, BlockError};
    use crate::test_assert;
    use crate::testing::TestResult;

    /// Test-only device number; no driver registers it.
    const WRITE_PROTECTED_DEV: u32 = 0xBC00;

    /// Reads return zeros and every write fails, like write-protected media.
    struct WriteProtected;

    impl BlockDevice for WriteProtected {
        fn sector_size(&self) -> u64 {
            512
        }
        fn num_sectors(&self) -> u64 {
            16
        }
        fn read_sectors(&self, _lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
            buf.fill(0);
            Ok(())
        }
        fn write_sectors(&self, _lba: u64, _buf: &[u8]) -> Result<(), BlockError> {
            Err(BlockError::IoError)
        }
        fn flush(&self) -> Result<(), BlockError> {
            Ok(())
        }
    }

    static WRITE_PROTECTED: WriteProtected = WriteProtected;

    /// Regression: `evict` dropped a dirty buffer even when its write-back
    /// failed, so data `write_block` had already accepted vanished with no
    /// error anywhere. A one-slot cache holding one dirty block must evict it
    /// to take a second block; on a device that rejects writes, block 0 has to
    /// survive that attempt still cached and still dirty.
    fn test_evict_keeps_dirty_block_when_write_back_fails() -> TestResult {
        register_block_device(WRITE_PROTECTED_DEV, &WRITE_PROTECTED);
        let mut cache = BufferCache::new(1, 512);
        let _ = cache.write_block(WRITE_PROTECTED_DEV, 0, &[0xAB; 512]);
        let _ = cache.write_block(WRITE_PROTECTED_DEV, 1, &[0xCD; 512]);
        let Some(buf) = cache.get_block(WRITE_PROTECTED_DEV, 0) else {
            return TestResult::Fail("block 0 is no longer readable through the cache");
        };
        test_assert!(
            buf.data[0] == 0xAB,
            "dirty block 0 was evicted although its write-back failed"
        );
        test_assert!(buf.dirty, "block 0 lost its dirty flag");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "buffer_cache::evict_keeps_dirty_block_when_write_back_fails",
            test_evict_keeps_dirty_block_when_write_back_fails,
        );
    }
}
