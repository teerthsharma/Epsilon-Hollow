# Ext2 Driver — Design Document

> **Phase 4 Task**: FS-002
> **Priority**: MEDIUM (legacy ext2 disk image interop)
> **Estimated Effort**: 4 weeks
> **Blocked by**: Block device abstraction, buffer cache, physical allocator
> **Blocks**: package manager import/export and legacy ext2 data exchange

---

## Overview

Ext2 is a simple legacy on-disk format worth reading for data exchange. Supporting ext2 lets Seal OS import files from existing ext2 images without adopting another OS ABI or boot model. This design covers read-write ext2 v1.0 with standard inode/block sizes.

---

## Architecture

### 1. On-Disk Layout

```
Block 0:       Boot sector (reserved, 1 block)
Block 1:       Superblock (1024 bytes at offset 1024)
Block 2..N:    Block group descriptors
Block N+1..M:  Block bitmap
Block M+1..O:  Inode bitmap
Block O+1..P:  Inode table
Block P+1..end: Data blocks
```

**Superblock fields (v1.0):**
```rust
#[repr(C, packed)]
struct Ext2Superblock {
    s_inodes_count: u32,
    s_blocks_count: u32,
    s_r_blocks_count: u32,
    s_free_blocks_count: u32,
    s_free_inodes_count: u32,
    s_first_data_block: u32,     // 0 for 1k+ block size, 1 for 1k
    s_log_block_size: u32,       // block_size = 1024 << s_log_block_size
    s_log_frag_size: u32,
    s_blocks_per_group: u32,
    s_frags_per_group: u32,
    s_inodes_per_group: u32,
    s_magic: u16,                // 0xEF53
    s_state: u16,                // 1 = clean, 2 = errors
    s_rev_level: u32,            // 0 = original, 1 = v1 with dynamic rev
    // ... dynamic rev fields below
    s_first_ino: u32,            // First non-reserved inode (11 for v1)
    s_inode_size: u16,           // Usually 128
    s_block_group_nr: u16,
    s_feature_compat: u32,
    s_feature_incompat: u32,
    s_feature_ro_compat: u32,
    s_uuid: [u8; 16],
    s_volume_name: [u8; 16],
}
```

- [x] Define `Ext2Superblock` struct with `#[repr(C, packed)]`
- [x] Read superblock from LBA 2 (offset 1024)
- [x] Verify magic == 0xEF53
- [x] Verify `s_rev_level` is 0 or 1
- [x] Calculate `block_size = 1024 << s_log_block_size`
- [x] Calculate number of block groups
- [x] Read block group descriptor table
- [x] Verify `s_state` == 1 (clean) or force fsck warning
- [x] Store superblock in memory for fast access
- [x] Implement `sync_superblock()` for writing updates

**Block group descriptor:**
```rust
#[repr(C, packed)]
struct BlockGroupDescriptor {
    bg_block_bitmap: u32,
    bg_inode_bitmap: u32,
    bg_inode_table: u32,
    bg_free_blocks_count: u16,
    bg_free_inodes_count: u16,
    bg_used_dirs_count: u16,
    bg_pad: u16,
    bg_reserved: [u32; 3],
}
```

- [x] Define `BlockGroupDescriptor` struct
- [x] Read descriptor table from block 2 (or block 1 for 1k block size)
- [x] Verify descriptors point to valid blocks
- [x] Store descriptor table in memory

### 2. Inode Structure

```rust
#[repr(C, packed)]
struct Ext2Inode {
    i_mode: u16,       // File type + permissions
    i_uid: u16,
    i_size: u32,       // Low 32 bits of size
    i_atime: u32,
    i_ctime: u32,
    i_mtime: u32,
    i_dtime: u32,
    i_gid: u16,
    i_links_count: u16,
    i_blocks: u32,     // 512-byte blocks allocated
    i_flags: u32,
    i_osd1: u32,
    i_block: [u32; 15], // Direct (0-11), indirect (12), dbl indirect (13), tpl indirect (14)
    i_generation: u32,
    i_file_acl: u32,
    i_dir_acl: u32,    // High 32 bits of size (for files > 2 GiB)
    i_faddr: u32,
    i_osd2: [u8; 12],
}
```

- [x] Define `Ext2Inode` struct
- [x] Implement `read_inode(inode_num)` — calculate block group, read from inode table
- [x] Implement `write_inode(inode_num, inode)` — write back to inode table
- [x] Handle `i_dir_acl` for large files (> 2 GiB)
- [x] Parse `i_mode` into file type and permissions

**Block pointers:**
- Direct blocks 0–11: 12 × block_size bytes
- Indirect block 12: block_size / 4 pointers → 256 × block_size bytes
- Double indirect 13: 256² × block_size bytes
- Triple indirect 14: 256³ × block_size bytes

- [x] Implement `resolve_block(inode, block_idx)` for direct blocks
- [x] Implement single indirect resolution
- [x] Implement double indirect resolution
- [x] Implement triple indirect resolution (v2)
- [x] Handle block_idx out of range

### 3. Directory Entry

```rust
#[repr(C, packed)]
struct Ext2DirEntry {
    inode: u32,        // 0 = unused entry
    rec_len: u16,      // Total record length (padded to 4-byte boundary)
    name_len: u8,      // Actual name length
    file_type: u8,     // 0 = unknown (v0), explicit type (v1 with feature)
    name: [u8],        // Variable length, no NUL terminator
}
```

- [x] Define `Ext2DirEntry` struct
- [x] Implement directory iteration (advance by `rec_len`)
- [x] Skip entries with `inode == 0` (deleted)
- [x] Parse `file_type` field when `EXT2_FEATURE_INCOMPAT_FILETYPE` is set
- [x] Implement `find_dentry(dir_block, name)` — linear search
- [x] Implement `add_dentry()` — find deleted entry or append
- [x] Implement `remove_dentry()` — set `inode = 0`
- [x] Handle directory block full → allocate new block

### 4. Read Path

```rust
fn read_inode_data(inode: &Ext2Inode, offset: u64, buf: &mut [u8]) -> Result<usize, Errno> {
    let block_size = self.block_size;
    let mut remaining = buf.len();
    let mut buf_offset = 0;
    let mut file_offset = offset;
    
    while remaining > 0 && file_offset < inode.size() {
        let block_idx = (file_offset / block_size as u64) as u32;
        let block_offset = (file_offset % block_size as u64) as usize;
        let to_read = min(remaining, block_size - block_offset);
        
        let phys_block = resolve_block(inode, block_idx)?;
        let cache_buf = bread(self.dev, phys_block as u64)?;
        buf[buf_offset..buf_offset + to_read]
            .copy_from_slice(&cache_buf.data()[block_offset..block_offset + to_read]);
        
        remaining -= to_read;
        buf_offset += to_read;
        file_offset += to_read as u64;
    }
    
    Ok(buf_offset)
}
```

- [x] Implement `read_inode_data(inode, offset, buf)`
- [x] Use buffer cache (`bread`) for all block reads
- [x] Handle partial block reads at start/end
- [x] Return actual bytes read (may be less than buf.len.)
- [x] Handle EOF correctly

### 5. Write Path

**Allocate block:**
```rust
fn alloc_block(&self, group_hint: u32) -> Result<u32, Errno> {
    // 1. Load block bitmap for group_hint
    // 2. Scan for first 0 bit
    // 3. Set bit, update bg_free_blocks_count, update superblock
    // 4. Return block number = group * blocks_per_group + bit_index
}
```

- [x] Implement `alloc_block(group_hint)`
- [x] Load block bitmap from buffer cache
- [x] Atomically find and set first free bit
- [x] Update block group descriptor free count
- [x] Update superblock free count
- [x] Mark bitmap buffer dirty
- [x] Mark superblock dirty
- [x] Handle all groups full → return -ENOSPC

**Write:**
```rust
fn write_inode_data(inode: &mut Ext2Inode, offset: u64, buf: &[u8]) -> Result<usize, Errno> {
    // For each block touched:
    //   If block not allocated: alloc_block(), update inode.i_block[]
    //   bread() or alloc new buffer
    //   memcpy data into buffer
    //   bwrite() to mark dirty
    // Update inode size if extended
    // Mark inode dirty
}
```

- [x] Implement `write_inode_data(inode, offset, buf)`
- [x] Allocate new blocks on write past EOF
- [x] Update `i_block[]` array with new block numbers
- [x] Handle indirect block allocation
- [x] Update inode size if extended
- [x] Mark inode dirty for later sync

**Inode allocation:**
```rust
fn alloc_inode(&self, group_hint: u32) -> Result<u32, Errno> {
    // Scan inode bitmap for first 0 bit
    // Set bit, update counts
    // Zero out inode table entry
    // Return inode number (1-based)
}
```

- [x] Implement `alloc_inode(group_hint)`
- [x] Load inode bitmap
- [x] Find and set first free inode bit
- [x] Zero out inode table entry
- [x] Update group descriptor and superblock counts

### 6. Directory Operations

**Create:**
```rust
fn create_file(dir_inode: &mut Ext2Inode, name: &str, mode: u16) -> Result<u32, Errno> {
    let inode_num = self.alloc_inode(dir_group)?;
    let inode = self.get_inode(inode_num)?;
    inode.i_mode = mode | S_IFREG;
    inode.i_links_count = 1;
    self.write_inode(inode_num, inode)?;
    
    self.add_dir_entry(dir_inode, name, inode_num, EXT2_FT_REG_FILE)?;
    Ok(inode_num)
}
```

- [x] Implement `create_file(dir, name, mode)`
- [x] Allocate inode
- [x] Initialize inode metadata
- [x] Add directory entry
- [x] Update directory inode mtime/ctime

**Remove:**
- [x] Set directory entry `inode = 0`
- [x] Decrement link count
- [x] If link count reaches 0, free all blocks and inode
- [x] Update block bitmap on block free
- [x] Update inode bitmap on inode free

### 7. Mount Integration

```rust
impl FileSystem for Ext2Fs {
    fn name(&self) -> &'static str { "ext2" }
    // ... delegates to read/write functions above
}
```

- [x] Implement `FileSystem` trait for `Ext2Fs`
- [x] Implement mount: read superblock, validate, construct `Ext2Fs`
- [x] Implement `mount("/dev/sda1", "/mnt", "ext2", MS_RDONLY, None)`
- [x] Support read-only mounts
- [x] Support read-write mounts
- [x] Implement `sync()` flushing all dirty buffers + superblock

### 8. Feature Support Matrix

| Feature | Status | Notes |
|---|---|---|
| 1024-byte blocks | v1 | Required for boot compatibility |
| 4096-byte blocks | v1 | Default for modern systems |
| Dynamic revision | v1 | Required for 32-bit UIDs, extended fields |
| Directory file type | v1 | Speeds up readdir |
| Sparse superblocks | v1 | Skip backup superblock in some groups |
| Large file (> 2 GiB) | v1 | Uses i_dir_acl for high size bits |
| Ext3 journal | v2 | Not implemented; mount as ext2 (no journal replay) |
| Extents | v2 | Not implemented; v1 inodes only |
| xattr | v2 | Not implemented |
| ACLs | v2 | Not implemented |

- [x] Support 1024-byte blocks
- [x] Support 4096-byte blocks
- [x] Support dynamic revision (v1 superblock)
- [x] Support directory file type feature
- [x] Support sparse superblocks
- [x] Support large files via `i_dir_acl`
- [x] Reject mounts with incompatible features (extents, journal, compression)

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_ext2_mount` | Mount ext2 image, superblock magic 0xEF53 valid | [x] |
| `test_ext2_read_file` | Read known file from ext2 image, SHA256 matches | [x] |
| `test_ext2_read_dir` | readdir on root returns ., .., bin, etc, lib | [x] |
| `test_ext2_large_file` | Read 100 MiB file, all blocks resolved correctly | [x] |
| `test_ext2_write_file` | Create file, write data, unmount, remount, read back matches | [x] |
| `test_ext2_mkdir` | Create directory, readdir sees it, can create files inside | [x] |
| `test_ext2_unlink` | Delete file, inode freed, directory entry zeroed | [x] |
| `test_ext2_symlink` | Create symlink, readlink returns target | [x] |
| `test_ext2_hardlink` | Link file, link count 2, same inode | [x] |
| `test_ext2_fill_fs` | Write until ENOSPC, verify free block count reaches 0 | [x] |
| `test_ext2_linux_compat` | Mount same image in Linux VM, files identical | [x] |

---

*Design document produced by Phase 4 Planning*
*2026-05-19*
