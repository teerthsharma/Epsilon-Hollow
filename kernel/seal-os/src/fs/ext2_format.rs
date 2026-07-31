// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Ext2 formatter.
//!
//! Lays down a superblock, block group descriptor table, block bitmap, inode
//! bitmap, inode table and a root directory holding `.` and `..`, so that the
//! reader in [`crate::fs::ext2`] can mount the result and walk it.
//!
//! Every write goes through `drivers::block::write_install_block`, so the
//! formatter cannot touch a device that is not the armed install target.

use alloc::vec;

use crate::drivers::block::{write_install_block, BlockError};
use crate::fs::ext2::{EXT2_MAGIC, EXT2_S_IFDIR};

/// 1 KiB blocks: `s_log_block_size = 0`.
pub const BLOCK_SIZE: u32 = 1024;
pub const LOG_BLOCK_SIZE: u32 = 0;
pub const INODE_SIZE: u32 = 128;
pub const FIRST_DATA_BLOCK: u32 = 1;
pub const FIRST_INO: u32 = 11;
pub const ROOT_INO: u32 = 2;
/// Inodes 1..=10 are reserved by the ext2 revision-1 layout.
pub const RESERVED_INODES: u32 = 10;
/// 8 bits per byte over a one-block bitmap.
pub const BLOCKS_PER_GROUP: u32 = BLOCK_SIZE * 8;

const SECTORS_PER_BLOCK: u64 = (BLOCK_SIZE / 512) as u64;

/// What the formatter actually wrote, measured from its own inputs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FormatReport {
    pub block_size: u32,
    pub blocks_count: u32,
    pub inodes_count: u32,
    pub blocks_per_group: u32,
    pub inodes_per_group: u32,
    pub free_blocks: u32,
    pub free_inodes: u32,
    pub inode_table_block: u32,
    pub root_data_block: u32,
    pub root_ino: u32,
}

fn put_u16(buf: &mut [u8], off: usize, value: u16) {
    buf[off..off + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(buf: &mut [u8], off: usize, value: u32) {
    buf[off..off + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_block(dev_num: u32, block: u32, data: &[u8]) -> Result<(), BlockError> {
    write_install_block(dev_num, block as u64 * SECTORS_PER_BLOCK, data)
}

/// Inode count for a filesystem of `blocks_count` blocks: one inode per four
/// blocks, rounded up to a whole bitmap byte and a whole inode-table block.
fn inode_count_for(blocks_count: u32) -> u32 {
    let raw = (blocks_count / 4).clamp(16, 2048);
    // 8 inodes per 1 KiB table block, 8 bits per bitmap byte — same rounding.
    (raw + 7) & !7
}

/// Format the device numbered `dev_num` as ext2.
///
/// `total_sectors` is the extent of the device (a partition view is expected,
/// so LBA 0 here is the start of the partition).
///
/// ponytail: single block group, so the ceiling is `BLOCKS_PER_GROUP` blocks
/// (8 MiB at 1 KiB blocks). Multi-group needs a BGD entry and a bitmap pair per
/// group plus superblock backups; add it when a target larger than 8 MiB has to
/// be formatted.
pub fn format_ext2(
    dev_num: u32,
    total_sectors: u64,
    volume_name: &str,
    uuid: &[u8; 16],
) -> Result<FormatReport, BlockError> {
    let blocks_count = (total_sectors / SECTORS_PER_BLOCK).min(BLOCKS_PER_GROUP as u64) as u32;
    let inodes_count = inode_count_for(blocks_count);
    let inode_table_blocks = inodes_count * INODE_SIZE / BLOCK_SIZE;

    let block_bitmap_block = FIRST_DATA_BLOCK + 2;
    let inode_bitmap_block = FIRST_DATA_BLOCK + 3;
    let inode_table_block = FIRST_DATA_BLOCK + 4;
    let root_data_block = inode_table_block + inode_table_blocks;

    // Everything from FIRST_DATA_BLOCK through root_data_block is metadata.
    let used_blocks = root_data_block;
    if blocks_count <= used_blocks + 8 {
        return Err(BlockError::InvalidLba);
    }
    let group_blocks = blocks_count - FIRST_DATA_BLOCK;
    let free_blocks = group_blocks - used_blocks;
    let free_inodes = inodes_count - RESERVED_INODES;

    // ── Block 0: boot block, zeroed so a stale MBR cannot be mistaken for one.
    put_block(dev_num, 0, &vec![0u8; BLOCK_SIZE as usize])?;

    // ── Superblock.
    let mut sb = vec![0u8; BLOCK_SIZE as usize];
    put_u32(&mut sb, 0, inodes_count);
    put_u32(&mut sb, 4, blocks_count);
    put_u32(&mut sb, 8, 0); // reserved blocks
    put_u32(&mut sb, 12, free_blocks);
    put_u32(&mut sb, 16, free_inodes);
    put_u32(&mut sb, 20, FIRST_DATA_BLOCK);
    put_u32(&mut sb, 24, LOG_BLOCK_SIZE);
    put_u32(&mut sb, 28, LOG_BLOCK_SIZE); // fragment size == block size
    put_u32(&mut sb, 32, BLOCKS_PER_GROUP);
    put_u32(&mut sb, 36, BLOCKS_PER_GROUP);
    put_u32(&mut sb, 40, inodes_count); // one group: all inodes live in it
    put_u32(&mut sb, 44, 0); // s_mtime
    put_u32(&mut sb, 48, 0); // s_wtime
    put_u16(&mut sb, 52, 0); // s_mnt_count
    put_u16(&mut sb, 54, 0xFFFF); // s_max_mnt_count: never force a check
    put_u16(&mut sb, 56, EXT2_MAGIC);
    put_u16(&mut sb, 58, 1); // s_state: clean
    put_u16(&mut sb, 60, 1); // s_errors: continue
    put_u16(&mut sb, 62, 0); // s_minor_rev_level
    put_u32(&mut sb, 64, 0); // s_lastcheck
    put_u32(&mut sb, 68, 0); // s_checkinterval
    put_u32(&mut sb, 72, 0); // s_creator_os: 0 keeps host e2fsck able to read it
    put_u32(&mut sb, 76, 1); // s_rev_level: dynamic, needed for s_inode_size
    put_u16(&mut sb, 80, 0); // s_def_resuid
    put_u16(&mut sb, 82, 0); // s_def_resgid
    put_u32(&mut sb, 84, FIRST_INO);
    put_u16(&mut sb, 88, INODE_SIZE as u16);
    put_u16(&mut sb, 90, 0); // s_block_group_nr
    put_u32(&mut sb, 92, 0); // s_feature_compat
    put_u32(&mut sb, 96, 0); // s_feature_incompat
    put_u32(&mut sb, 100, 0); // s_feature_ro_compat
    sb[104..120].copy_from_slice(uuid);
    for (i, byte) in volume_name.bytes().take(15).enumerate() {
        sb[120 + i] = byte;
    }
    put_block(dev_num, FIRST_DATA_BLOCK, &sb)?;

    // ── Block group descriptor table (one group).
    let mut bgd = vec![0u8; BLOCK_SIZE as usize];
    put_u32(&mut bgd, 0, block_bitmap_block);
    put_u32(&mut bgd, 4, inode_bitmap_block);
    put_u32(&mut bgd, 8, inode_table_block);
    put_u16(&mut bgd, 12, free_blocks as u16);
    put_u16(&mut bgd, 14, free_inodes as u16);
    put_u16(&mut bgd, 16, 1); // used_dirs_count: the root directory
    put_block(dev_num, FIRST_DATA_BLOCK + 1, &bgd)?;

    // ── Block bitmap. Bit i covers block FIRST_DATA_BLOCK + i.
    let mut block_bitmap = vec![0u8; BLOCK_SIZE as usize];
    for bit in 0..used_blocks {
        block_bitmap[(bit / 8) as usize] |= 1 << (bit % 8);
    }
    // Bits past the end of the filesystem must read as allocated.
    for bit in group_blocks..BLOCKS_PER_GROUP {
        block_bitmap[(bit / 8) as usize] |= 1 << (bit % 8);
    }
    put_block(dev_num, block_bitmap_block, &block_bitmap)?;

    // ── Inode bitmap. Bit i covers inode i + 1.
    let mut inode_bitmap = vec![0u8; BLOCK_SIZE as usize];
    for bit in 0..RESERVED_INODES {
        inode_bitmap[(bit / 8) as usize] |= 1 << (bit % 8);
    }
    for bit in inodes_count..BLOCKS_PER_GROUP {
        inode_bitmap[(bit / 8) as usize] |= 1 << (bit % 8);
    }
    put_block(dev_num, inode_bitmap_block, &inode_bitmap)?;

    // ── Inode table, zeroed except for the root inode.
    let zero = vec![0u8; BLOCK_SIZE as usize];
    for i in 1..inode_table_blocks {
        put_block(dev_num, inode_table_block + i, &zero)?;
    }
    let mut table0 = vec![0u8; BLOCK_SIZE as usize];
    let root_off = ((ROOT_INO - 1) * INODE_SIZE) as usize;
    put_u16(&mut table0, root_off, EXT2_S_IFDIR | 0o755);
    put_u16(&mut table0, root_off + 2, 0); // i_uid
    put_u32(&mut table0, root_off + 4, BLOCK_SIZE); // i_size
    put_u16(&mut table0, root_off + 24, 0); // i_gid
    put_u16(&mut table0, root_off + 26, 2); // i_links_count: "." and the name
    put_u32(&mut table0, root_off + 28, SECTORS_PER_BLOCK as u32); // i_blocks, 512 B units
    put_u32(&mut table0, root_off + 40, root_data_block); // i_block[0]
    put_block(dev_num, inode_table_block, &table0)?;

    // ── Root directory data: "." then ".." padded to the end of the block.
    let mut root = vec![0u8; BLOCK_SIZE as usize];
    put_u32(&mut root, 0, ROOT_INO);
    put_u16(&mut root, 4, 12); // rec_len
    root[6] = 1; // name_len
    root[7] = 2; // file_type: directory
    root[8] = b'.';
    put_u32(&mut root, 12, ROOT_INO);
    put_u16(&mut root, 16, (BLOCK_SIZE - 12) as u16);
    root[18] = 2;
    root[19] = 2;
    root[20] = b'.';
    root[21] = b'.';
    put_block(dev_num, root_data_block, &root)?;

    Ok(FormatReport {
        block_size: BLOCK_SIZE,
        blocks_count,
        inodes_count,
        blocks_per_group: BLOCKS_PER_GROUP,
        inodes_per_group: inodes_count,
        free_blocks,
        free_inodes,
        inode_table_block,
        root_data_block,
        root_ino: ROOT_INO,
    })
}
