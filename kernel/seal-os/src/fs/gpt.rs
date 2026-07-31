// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! GUID Partition Table writer and verifier.
//!
//! Writes a protective MBR, a primary header at LBA 1 with its entry array at
//! LBA 2, and a backup entry array plus backup header at the tail of the disk.
//! Every write goes through `drivers::block::write_install_block`, so a disk
//! that is not the armed install target is refused before any I/O happens.

use alloc::vec;
use alloc::vec::Vec;

use crate::drivers::block::{read_block, write_install_block, BlockError};

pub const SECTOR: usize = 512;
pub const SIGNATURE: &[u8; 8] = b"EFI PART";
pub const REVISION: u32 = 0x0001_0000;
pub const HEADER_SIZE: u32 = 92;
pub const ENTRY_COUNT: u32 = 128;
pub const ENTRY_SIZE: u32 = 128;
/// 128 entries x 128 bytes = 16 KiB = 32 sectors.
pub const ENTRY_SECTORS: u64 = (ENTRY_COUNT as u64 * ENTRY_SIZE as u64) / SECTOR as u64;

/// EFI System Partition, C12A7328-F81F-11D2-BA4B-00A0C93EC93B (mixed endian).
pub const TYPE_ESP: [u8; 16] = [
    0x28, 0x73, 0x2A, 0xC1, 0x1F, 0xF8, 0xD2, 0x11, 0xBA, 0x4B, 0x00, 0xA0, 0xC9, 0x3E, 0xC9, 0x3B,
];

/// Filesystem data, 0FC63DAF-8483-4772-8E79-3D69D8477DE4 (mixed endian).
pub const TYPE_DATA: [u8; 16] = [
    0xAF, 0x3D, 0xC6, 0x0F, 0x83, 0x84, 0x72, 0x47, 0x8E, 0x79, 0x3D, 0x69, 0xD8, 0x47, 0x7D, 0xE4,
];

/// CRC-32 (reflected, polynomial 0xEDB88320, pre/post inverted) — the variant
/// UEFI specifies for GPT, and the same one `seal-mkimage` computes with.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// One partition to lay down.
#[derive(Clone, Copy)]
pub struct PartSpec {
    pub type_guid: [u8; 16],
    pub first_lba: u64,
    pub last_lba: u64,
    pub name: &'static str,
}

/// Where every GPT structure lands on a disk of `total_sectors` sectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GptLayout {
    pub total_sectors: u64,
    pub header_lba: u64,
    pub backup_header_lba: u64,
    pub entry_lba: u64,
    pub backup_entry_lba: u64,
    pub first_usable_lba: u64,
    pub last_usable_lba: u64,
}

impl GptLayout {
    pub fn for_disk(total_sectors: u64) -> Result<Self, BlockError> {
        // pMBR + header + entries at each end, plus at least one usable sector.
        if total_sectors < 2 * (1 + ENTRY_SECTORS) + 2 {
            return Err(BlockError::InvalidLba);
        }
        let backup_header_lba = total_sectors - 1;
        let backup_entry_lba = backup_header_lba - ENTRY_SECTORS;
        Ok(Self {
            total_sectors,
            header_lba: 1,
            backup_header_lba,
            entry_lba: 2,
            backup_entry_lba,
            first_usable_lba: 2 + ENTRY_SECTORS,
            last_usable_lba: backup_entry_lba - 1,
        })
    }
}

/// What a readback of an on-disk GPT actually measured.
#[derive(Debug, Clone, Copy, Default)]
pub struct GptVerify {
    pub partitions: u32,
    pub header_crc: u32,
    pub header_crc_ok: bool,
    pub entries_crc_ok: bool,
    pub backup_header_crc_ok: bool,
    pub backup_agrees: bool,
    pub alt_lba_ok: bool,
    pub protective_mbr_ok: bool,
    pub first_usable_lba: u64,
    pub last_usable_lba: u64,
    pub first_part_first_lba: u64,
    pub first_part_last_lba: u64,
}

impl GptVerify {
    pub fn passes(&self) -> bool {
        self.header_crc_ok
            && self.entries_crc_ok
            && self.backup_header_crc_ok
            && self.backup_agrees
            && self.alt_lba_ok
            && self.protective_mbr_ok
            && self.partitions > 0
    }
}

fn put_u32(buf: &mut [u8], off: usize, value: u32) {
    buf[off..off + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(buf: &mut [u8], off: usize, value: u64) {
    buf[off..off + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn get_u64(buf: &[u8], off: usize) -> u64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&buf[off..off + 8]);
    u64::from_le_bytes(raw)
}

/// Protective MBR: one 0xEE partition spanning the disk.
pub fn build_protective_mbr(total_sectors: u64) -> [u8; SECTOR] {
    let mut mbr = [0u8; SECTOR];
    mbr[446] = 0x00; // not bootable
    mbr[447] = 0x00; // starting head
    mbr[448] = 0x02; // starting sector 2
    mbr[449] = 0x00; // starting cylinder
    mbr[450] = 0xEE; // GPT protective
    mbr[451] = 0xFF;
    mbr[452] = 0xFF;
    mbr[453] = 0xFF;
    put_u32(&mut mbr, 454, 1);
    let span = (total_sectors - 1).min(0xFFFF_FFFF) as u32;
    put_u32(&mut mbr, 458, span);
    mbr[510] = 0x55;
    mbr[511] = 0xAA;
    mbr
}

/// Serialize the partition entry array (always `ENTRY_COUNT` slots).
pub fn build_entries(parts: &[PartSpec], unique_guids: &[[u8; 16]]) -> Vec<u8> {
    let mut entries = vec![0u8; (ENTRY_COUNT * ENTRY_SIZE) as usize];
    for (idx, part) in parts.iter().take(ENTRY_COUNT as usize).enumerate() {
        let base = idx * ENTRY_SIZE as usize;
        entries[base..base + 16].copy_from_slice(&part.type_guid);
        let guid = unique_guids.get(idx).copied().unwrap_or([0u8; 16]);
        entries[base + 16..base + 32].copy_from_slice(&guid);
        put_u64(&mut entries, base + 32, part.first_lba);
        put_u64(&mut entries, base + 40, part.last_lba);
        put_u64(&mut entries, base + 48, 0); // attributes
        for (i, unit) in part.name.encode_utf16().take(35).enumerate() {
            let off = base + 56 + i * 2;
            entries[off..off + 2].copy_from_slice(&unit.to_le_bytes());
        }
    }
    entries
}

/// Serialize one GPT header. `my_lba`/`alt_lba` decide primary versus backup.
pub fn build_header(
    layout: &GptLayout,
    my_lba: u64,
    alt_lba: u64,
    entry_lba: u64,
    disk_guid: &[u8; 16],
    entries_crc: u32,
) -> [u8; SECTOR] {
    let mut hdr = [0u8; SECTOR];
    hdr[0..8].copy_from_slice(SIGNATURE);
    put_u32(&mut hdr, 8, REVISION);
    put_u32(&mut hdr, 12, HEADER_SIZE);
    put_u32(&mut hdr, 16, 0); // header CRC, filled in below
    put_u32(&mut hdr, 20, 0); // reserved
    put_u64(&mut hdr, 24, my_lba);
    put_u64(&mut hdr, 32, alt_lba);
    put_u64(&mut hdr, 40, layout.first_usable_lba);
    put_u64(&mut hdr, 48, layout.last_usable_lba);
    hdr[56..72].copy_from_slice(disk_guid);
    put_u64(&mut hdr, 72, entry_lba);
    put_u32(&mut hdr, 80, ENTRY_COUNT);
    put_u32(&mut hdr, 84, ENTRY_SIZE);
    put_u32(&mut hdr, 88, entries_crc);
    let crc = crc32(&hdr[..HEADER_SIZE as usize]);
    put_u32(&mut hdr, 16, crc);
    hdr
}

/// Recompute a header's CRC-32 the way UEFI defines it: over the first
/// `header_size` bytes with the CRC field itself zeroed.
pub fn header_crc_of(hdr: &[u8]) -> Option<u32> {
    let size = get_u32(hdr, 12) as usize;
    if size < 92 || size > hdr.len() {
        return None;
    }
    let mut scratch = Vec::from(&hdr[..size]);
    put_u32(&mut scratch, 16, 0);
    Some(crc32(&scratch))
}

/// Write a complete GPT (protective MBR, both headers, both entry arrays).
pub fn write_gpt(
    dev_num: u32,
    total_sectors: u64,
    parts: &[PartSpec],
    disk_guid: &[u8; 16],
    unique_guids: &[[u8; 16]],
) -> Result<GptLayout, BlockError> {
    let layout = GptLayout::for_disk(total_sectors)?;
    for part in parts {
        if part.first_lba < layout.first_usable_lba
            || part.last_lba > layout.last_usable_lba
            || part.first_lba > part.last_lba
        {
            return Err(BlockError::InvalidLba);
        }
    }

    let entries = build_entries(parts, unique_guids);
    let entries_crc = crc32(&entries);

    write_install_block(dev_num, 0, &build_protective_mbr(total_sectors))?;
    write_install_block(dev_num, layout.entry_lba, &entries)?;
    write_install_block(dev_num, layout.backup_entry_lba, &entries)?;
    write_install_block(
        dev_num,
        layout.header_lba,
        &build_header(
            &layout,
            layout.header_lba,
            layout.backup_header_lba,
            layout.entry_lba,
            disk_guid,
            entries_crc,
        ),
    )?;
    write_install_block(
        dev_num,
        layout.backup_header_lba,
        &build_header(
            &layout,
            layout.backup_header_lba,
            layout.header_lba,
            layout.backup_entry_lba,
            disk_guid,
            entries_crc,
        ),
    )?;
    Ok(layout)
}

/// Read a GPT back off the disk and measure every self-consistency claim.
pub fn verify_gpt(dev_num: u32) -> Result<GptVerify, BlockError> {
    let mut mbr = [0u8; SECTOR];
    read_block(dev_num, 0, &mut mbr)?;
    let protective_mbr_ok = mbr[450] == 0xEE && mbr[510] == 0x55 && mbr[511] == 0xAA;

    let mut hdr = [0u8; SECTOR];
    read_block(dev_num, 1, &mut hdr)?;
    if &hdr[0..8] != SIGNATURE {
        return Err(BlockError::IoError);
    }
    let stored_crc = get_u32(&hdr, 16);
    let header_crc_ok = header_crc_of(&hdr) == Some(stored_crc);

    let entry_lba = get_u64(&hdr, 72);
    let entry_count = get_u32(&hdr, 80);
    let entry_size = get_u32(&hdr, 84);
    if entry_size == 0 || entry_size % 8 != 0 || entry_count > 4096 {
        return Err(BlockError::IoError);
    }
    let array_bytes = (entry_count as usize) * (entry_size as usize);
    if array_bytes % SECTOR != 0 {
        return Err(BlockError::IoError);
    }
    let mut entries = vec![0u8; array_bytes];
    read_block(dev_num, entry_lba, &mut entries)?;
    let entries_crc_ok = crc32(&entries) == get_u32(&hdr, 88);

    let mut partitions = 0u32;
    let mut first_part_first_lba = 0u64;
    let mut first_part_last_lba = 0u64;
    for idx in 0..entry_count as usize {
        let base = idx * entry_size as usize;
        if entries[base..base + 16].iter().all(|&b| b == 0) {
            continue;
        }
        if partitions == 0 {
            first_part_first_lba = get_u64(&entries, base + 32);
            first_part_last_lba = get_u64(&entries, base + 40);
        }
        partitions += 1;
    }

    let backup_header_lba = get_u64(&hdr, 32);
    let mut backup = [0u8; SECTOR];
    read_block(dev_num, backup_header_lba, &mut backup)?;
    let backup_crc_stored = get_u32(&backup, 16);
    let backup_header_crc_ok =
        &backup[0..8] == SIGNATURE && header_crc_of(&backup) == Some(backup_crc_stored);

    // The two headers must cross-link and agree on every shared field.
    let alt_lba_ok = get_u64(&backup, 24) == backup_header_lba
        && get_u64(&backup, 32) == get_u64(&hdr, 24)
        && get_u64(&hdr, 24) == 1;
    let backup_agrees = get_u64(&backup, 40) == get_u64(&hdr, 40)
        && get_u64(&backup, 48) == get_u64(&hdr, 48)
        && backup[56..72] == hdr[56..72]
        && get_u32(&backup, 80) == entry_count
        && get_u32(&backup, 84) == entry_size
        && get_u32(&backup, 88) == get_u32(&hdr, 88);

    let mut backup_entries = vec![0u8; array_bytes];
    read_block(dev_num, get_u64(&backup, 72), &mut backup_entries)?;
    let backup_agrees = backup_agrees && backup_entries == entries;

    Ok(GptVerify {
        partitions,
        header_crc: stored_crc,
        header_crc_ok,
        entries_crc_ok,
        backup_header_crc_ok,
        backup_agrees,
        alt_lba_ok,
        protective_mbr_ok,
        first_usable_lba: get_u64(&hdr, 40),
        last_usable_lba: get_u64(&hdr, 48),
        first_part_first_lba,
        first_part_last_lba,
    })
}
