# 03 — ManifoldFS: Persistence, Journal, Block Device Format

## Goal

Transform ManifoldFS from an in-memory-only filesystem into a persistent, journaled filesystem backed by a block device, while preserving the S² point-cloud topology and O(1) teleport semantics.

## Current State

`kernel/seal-os/src/fs/manifold_fs.rs` (~700 lines):
- `ManifoldFS` struct: `BTreeMap<u64, Inode>` for inode storage, `next_inode: u64`
- `Inode`: name, data (`Vec<u8>`), point cloud (`Vec<(f64, f64)>` — 64 points on S²), parent, children, metadata
- Operations: `create_file`, `write_file`, `read_file`, `delete_file`, `teleport` (O(1) pointer surgery via BTreeMap remove/insert), `list_dir`, `stat`, `similarity_search`
- `ManifoldIndex`: `SphericalVoronoiIndex` for nearest-cell lookups
- **No persistence** — all data lost on reboot
- **No journal** — no crash recovery
- **No block device** — no disk format, no sectors, no superblock

## Gap Analysis

| Feature | Status |
|---------|--------|
| In-memory inode storage | Working |
| Point cloud generation | Working (random on S²) |
| Teleport (O(1) surgery) | Working (BTreeMap pointer swap) |
| Similarity search | Working (cosine distance) |
| Disk persistence | Missing |
| Journal/WAL | Missing |
| Block device format | Missing |
| Superblock/metadata | Missing |
| fsync/flush | Missing |
| Directory hard links | Missing |
| Permissions/ownership | Missing |

## Implementation Steps

### Phase A: Block Device Abstraction

1. **Create `BlockDevice` trait**
   ```rust
   pub trait BlockDevice {
       fn read_block(&self, block_id: u64, buf: &mut [u8]) -> Result<(), FsError>;
       fn write_block(&mut self, block_id: u64, data: &[u8]) -> Result<(), FsError>;
       fn block_size(&self) -> usize; // 4096
       fn block_count(&self) -> u64;
   }
   ```

2. **Implement `RamDisk`** — in-memory block device for testing (backed by `Vec<[u8; 4096]>`)

3. **Implement `VirtioBlock`** — virtio-blk driver for QEMU (reads/writes to virtio PCI device)

### Phase B: On-Disk Format

4. **Define superblock** (block 0):
   - Magic: `b"MNFD"` (ManifoldFS)
   - Version, block size, total blocks, free blocks
   - Root inode number, inode table start block, data region start block
   - Point cloud index root block

5. **Define inode on-disk layout**:
   - Fixed-size inode entry (256 bytes): name (64 bytes), size, point cloud offset, data block pointers (direct + indirect), timestamps, parent inode, flags

6. **Define point cloud storage**:
   - Each inode's 64-point S² cloud stored as 64 × 2 × f64 = 1024 bytes (fits in one block)
   - Indexed by the SphericalVoronoiIndex for nearest-cell queries

7. **Implement block allocator** — bitmap-based free block tracking

### Phase C: Journal / Write-Ahead Log

8. **Implement WAL** (journal region: blocks 1-1024):
   - Before any metadata write: log the old and new block contents to the journal
   - After write completes: mark journal entry as committed
   - On mount: replay any uncommitted journal entries (crash recovery)

9. **Journal entry format**:
   - Transaction ID, block number, old data hash, new data, commit flag

### Phase D: Persistence Integration

10. **Modify `ManifoldFS::new()` to accept a `BlockDevice`**
    - On mount: read superblock, load inode table into BTreeMap cache
    - On create/write/delete: write-through to block device via journal

11. **Add `sync()` / `flush()`** — force all cached writes to block device

12. **Preserve teleport semantics** — O(1) teleport remains a metadata-only operation (update parent pointer in inode, no data copy). On disk: journal the inode parent field change.

## Dependencies

- **01-kernel-safety** (no unwrap panics in fs code)
- **02-memory-allocator** (BTreeMap cache needs real dealloc for eviction)

## Acceptance Criteria

- [ ] Files survive reboot (write file → reboot QEMU → file still readable)
- [ ] Teleport is still O(1) (metadata pointer swap, journaled)
- [ ] Journal replay works: kill QEMU mid-write → restart → filesystem consistent
- [ ] `similarity_search` still works (point cloud index persisted)
- [ ] Block allocator doesn't leak blocks on delete
- [ ] RamDisk passes all existing ManifoldFS tests
- [ ] VirtioBlock works under QEMU with `-drive file=disk.img,format=raw,if=virtio`

## Files to Modify

- `kernel/seal-os/src/fs/manifold_fs.rs` (major refactor)
- `kernel/seal-os/src/fs/mod.rs`

## Files to Create

- `kernel/seal-os/src/fs/block_device.rs` (trait + RamDisk)
- `kernel/seal-os/src/fs/journal.rs` (WAL implementation)
- `kernel/seal-os/src/fs/disk_format.rs` (superblock, on-disk inode layout)
- `kernel/seal-os/src/drivers/virtio_blk.rs` (virtio block device driver)
