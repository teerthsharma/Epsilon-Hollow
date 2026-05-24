// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Creates bootable UEFI disk images and ISOs for Seal OS.
//! VirtualBox-compatible: GPT + FAT32 ESP with proper boot sector.

use std::fs;
use std::io::{Cursor, Write};
use std::path::PathBuf;

const DISK_SIZE: u64 = 128 * 1024 * 1024; // 128MB
const SECTOR_SIZE: u64 = 512;
const ESP_START_LBA: u64 = 2048;
const ESP_SIZE_SECTORS: u64 = (DISK_SIZE / SECTOR_SIZE) - ESP_START_LBA - 34;

fn main() {
    let efi_binary = find_efi_binary();
    let efi_data = fs::read(&efi_binary).expect("Failed to read .efi binary");
    println!("[mkimage] Input: {} ({} bytes)", efi_binary.display(), efi_data.len());

    let out_dir = efi_binary.parent().unwrap_or_else(|| std::path::Path::new("."));

    // 1. Raw GPT disk image (for QEMU, dd to USB, or VirtualBox SATA)
    let img_path = out_dir.join("seal-os.img");
    let disk = create_disk_image(&efi_data);
    fs::write(&img_path, &disk).expect("Failed to write disk image");
    println!("[mkimage] IMG: {} ({:.1} MB)", img_path.display(), disk.len() as f64 / 1024.0 / 1024.0);

    // 2. FAT12 EFI boot image for El Torito ISO
    let efi_boot_img = create_fat12_efi_image(&efi_data);
    let efi_img_path = out_dir.join("seal-os-efi.img");
    fs::write(&efi_img_path, &efi_boot_img).expect("Failed to write EFI boot image");
    println!("[mkimage] EFI boot image: {} ({} bytes)", efi_img_path.display(), efi_boot_img.len());

    println!("[mkimage] Boot with:");
    println!("  QEMU:    qemu-system-x86_64 -bios OVMF.fd -drive file={},format=raw -serial stdio -m 512M", img_path.display());
    println!("  VBox:    Attach {} as SATA hard disk (type: Other)", img_path.display());
}

fn find_efi_binary() -> PathBuf {
    let candidates = [
        "../seal-os/target/x86_64-unknown-uefi/release/seal-os.efi",
        "../seal-os/target/x86_64-unknown-uefi/debug/seal-os.efi",
        "target/x86_64-unknown-uefi/release/seal-os.efi",
        "target/x86_64-unknown-uefi/debug/seal-os.efi",
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    eprintln!("Error: seal-os.efi not found. Run `cargo +nightly build --release` first.");
    std::process::exit(1);
}

fn create_disk_image(efi_data: &[u8]) -> Vec<u8> {
    let mut disk = vec![0u8; DISK_SIZE as usize];

    write_protective_mbr(&mut disk);
    write_gpt(&mut disk);

    let esp_offset = (ESP_START_LBA * SECTOR_SIZE) as usize;
    let esp_size = (ESP_SIZE_SECTORS * SECTOR_SIZE) as usize;
    let fat_img = create_fat32_esp(efi_data, esp_size);
    disk[esp_offset..esp_offset + fat_img.len()].copy_from_slice(&fat_img);

    disk
}

fn write_protective_mbr(disk: &mut [u8]) {
    // Add a simple MBR boot stub that jumps to the bootloader
    // Some UEFI firmware (including VirtualBox) checks this
    let mut boot_stub = vec![0x00u8; 440];
    boot_stub[0] = 0xEB; // jmp short
    boot_stub[1] = 0x5C;
    boot_stub[2] = 0x90;
    boot_stub[3..11].copy_from_slice(b"SEALOS  ");
    disk[0..440].copy_from_slice(&boot_stub);

    let entry_offset = 446;
    disk[entry_offset] = 0x00;      // status
    disk[entry_offset + 1] = 0x00;  // CHS start head
    disk[entry_offset + 2] = 0x02;  // CHS start sector (sector 2)
    disk[entry_offset + 3] = 0x00;  // CHS start cylinder
    disk[entry_offset + 4] = 0xEE;  // GPT protective
    disk[entry_offset + 5] = 0xFF;  // CHS end head
    disk[entry_offset + 6] = 0xFF;  // CHS end sector
    disk[entry_offset + 7] = 0xFF;  // CHS end cylinder
    disk[entry_offset + 8..entry_offset + 12].copy_from_slice(&1u32.to_le_bytes());
    let total_sectors = (DISK_SIZE / SECTOR_SIZE - 1) as u32;
    disk[entry_offset + 12..entry_offset + 16].copy_from_slice(&total_sectors.to_le_bytes());

    // Boot signature
    disk[510] = 0x55;
    disk[511] = 0xAA;
}

fn write_gpt(disk: &mut [u8]) {
    let total_sectors = DISK_SIZE / SECTOR_SIZE;
    let partition_entries_lba: u64 = 2;
    let partition_entry_size: u32 = 128;
    let num_partition_entries: u32 = 128;

    let esp_type_guid: [u8; 16] = [
        0x28, 0x73, 0x2A, 0xC1, 0x1F, 0xF8, 0xD2, 0x11,
        0xBA, 0x4B, 0x00, 0xA0, 0xC9, 0x3E, 0xC9, 0x3B,
    ];

    let part_guid: [u8; 16] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x53, 0x45, 0x41, 0x4C, 0x4F, 0x53, 0x00, 0x01,
    ];

    let disk_guid: [u8; 16] = [
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80,
        0x53, 0x45, 0x41, 0x4C, 0x4F, 0x53, 0x44, 0x4B,
    ];

    let pe_offset = (partition_entries_lba * SECTOR_SIZE) as usize;
    disk[pe_offset..pe_offset + 16].copy_from_slice(&esp_type_guid);
    disk[pe_offset + 16..pe_offset + 32].copy_from_slice(&part_guid);
    disk[pe_offset + 32..pe_offset + 40].copy_from_slice(&ESP_START_LBA.to_le_bytes());
    let esp_end_lba = ESP_START_LBA + ESP_SIZE_SECTORS - 1;
    disk[pe_offset + 40..pe_offset + 48].copy_from_slice(&esp_end_lba.to_le_bytes());

    let name = "EFI System";
    for (i, ch) in name.encode_utf16().enumerate() {
        let off = pe_offset + 56 + i * 2;
        disk[off..off + 2].copy_from_slice(&ch.to_le_bytes());
    }

    let pe_size = (num_partition_entries * partition_entry_size) as usize;
    let pe_crc = crc32(&disk[pe_offset..pe_offset + pe_size]);

    let hdr_offset = SECTOR_SIZE as usize;
    write_gpt_header(
        &mut disk[hdr_offset..hdr_offset + 92],
        1, total_sectors - 1, ESP_START_LBA, total_sectors - 34,
        &disk_guid, partition_entries_lba, num_partition_entries, partition_entry_size, pe_crc,
    );

    let backup_pe_lba = total_sectors - 33;
    let backup_pe_offset = (backup_pe_lba * SECTOR_SIZE) as usize;
    let pe_data: Vec<u8> = disk[pe_offset..pe_offset + pe_size].to_vec();
    disk[backup_pe_offset..backup_pe_offset + pe_size].copy_from_slice(&pe_data);

    let backup_hdr_offset = ((total_sectors - 1) * SECTOR_SIZE) as usize;
    write_gpt_header(
        &mut disk[backup_hdr_offset..backup_hdr_offset + 92],
        total_sectors - 1, 1, ESP_START_LBA, total_sectors - 34,
        &disk_guid, backup_pe_lba, num_partition_entries, partition_entry_size, pe_crc,
    );
}

fn write_gpt_header(
    buf: &mut [u8], my_lba: u64, alt_lba: u64, first_usable: u64, last_usable: u64,
    disk_guid: &[u8; 16], pe_start_lba: u64, num_pe: u32, pe_size: u32, pe_crc: u32,
) {
    buf[0..8].copy_from_slice(b"EFI PART");
    buf[8..12].copy_from_slice(&0x00010000u32.to_le_bytes());
    buf[12..16].copy_from_slice(&92u32.to_le_bytes());
    buf[16..20].copy_from_slice(&0u32.to_le_bytes());
    buf[20..24].copy_from_slice(&0u32.to_le_bytes());
    buf[24..32].copy_from_slice(&my_lba.to_le_bytes());
    buf[32..40].copy_from_slice(&alt_lba.to_le_bytes());
    buf[40..48].copy_from_slice(&first_usable.to_le_bytes());
    buf[48..56].copy_from_slice(&last_usable.to_le_bytes());
    buf[56..72].copy_from_slice(disk_guid);
    buf[72..80].copy_from_slice(&pe_start_lba.to_le_bytes());
    buf[80..84].copy_from_slice(&num_pe.to_le_bytes());
    buf[84..88].copy_from_slice(&pe_size.to_le_bytes());
    buf[88..92].copy_from_slice(&pe_crc.to_le_bytes());

    let hdr_crc = crc32(&buf[0..92]);
    buf[16..20].copy_from_slice(&hdr_crc.to_le_bytes());
}

fn create_fat32_esp(efi_data: &[u8], size: usize) -> Vec<u8> {
    let mut img = vec![0u8; size];
    let cursor = Cursor::new(&mut img[..]);
    let options = fatfs::FormatVolumeOptions::new()
        .fat_type(fatfs::FatType::Fat32)
        .volume_label([b'S', b'E', b'A', b'L', b'-', b'O', b'S', b'-', b'E', b'F', b'I']);
    fatfs::format_volume(cursor, options).expect("Failed to format FAT32");

    {
        let cursor = Cursor::new(&mut img[..]);
        let fs = fatfs::FileSystem::new(cursor, fatfs::FsOptions::new()).expect("Failed to mount FAT32");
        let root = fs.root_dir();

        root.create_dir("EFI").expect("mkdir EFI");
        root.create_dir("EFI/BOOT").expect("mkdir EFI/BOOT");

        let mut file = root.create_file("EFI/BOOT/BOOTX64.EFI").expect("create BOOTX64.EFI");
        file.write_all(efi_data).expect("write EFI binary");
    }
    img
}

/// Create a FAT12 EFI boot image for El Torito UEFI CD/DVD boot.
/// Size is 16MB to fit the ~6MB EFI binary comfortably.
fn create_fat12_efi_image(efi_data: &[u8]) -> Vec<u8> {
    let sectors = 32768; // 16MB
    let size = sectors * 512;
    let mut img = vec![0u8; size];
    let cursor = Cursor::new(&mut img[..]);
    let options = fatfs::FormatVolumeOptions::new()
        .fat_type(fatfs::FatType::Fat12)
        .volume_label([b'E', b'F', b'I', b'B', b'O', b'O', b'T', b' ', b' ', b' ', b' ']);
    fatfs::format_volume(cursor, options).expect("Failed to format FAT12");

    {
        let cursor = Cursor::new(&mut img[..]);
        let fs = fatfs::FileSystem::new(cursor, fatfs::FsOptions::new()).expect("Failed to mount FAT12");
        let root = fs.root_dir();

        root.create_dir("EFI").expect("mkdir EFI");
        root.create_dir("EFI/BOOT").expect("mkdir EFI/BOOT");

        let mut file = root.create_file("EFI/BOOT/BOOTX64.EFI").expect("create BOOTX64.EFI");
        file.write_all(efi_data).expect("write EFI binary");
    }
    img
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}
