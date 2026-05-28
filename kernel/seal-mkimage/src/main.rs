// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Creates bootable UEFI disk images and ISOs for Seal OS.
//! VirtualBox-compatible: GPT + FAT32 ESP with proper boot sector.

use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

const DISK_SIZE: u64 = 128 * 1024 * 1024; // 128MB
const SECTOR_SIZE: u64 = 512;
const ESP_START_LBA: u64 = 2048;
const ESP_SIZE_SECTORS: u64 = (DISK_SIZE / SECTOR_SIZE) - ESP_START_LBA - 34;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("--check-theorem-log") {
        let log_path = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp/seal-os.log"));
        check_theorem_log(&log_path).unwrap_or_else(|e| {
            eprintln!("[seal-audit] THEOREM LOG FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] THEOREM LOG OK: {}", log_path.display());
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-seal-abi") {
        let root = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        });
        check_seal_abi(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] SEAL ABI FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] SEAL ABI OK");
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--unsafe-inventory") {
        let root = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        });
        unsafe_inventory(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] UNSAFE INVENTORY FAIL: {e}");
            std::process::exit(1);
        });
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--verify") {
        let img_path = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| find_disk_image());
        let efi_path = args.get(3).map(PathBuf::from);
        let expected = efi_path
            .as_ref()
            .map(|p| fs::read(p).unwrap_or_else(|e| panic!("Failed to read {}: {e}", p.display())));
        verify_disk_image(&img_path, expected.as_deref()).unwrap_or_else(|e| {
            eprintln!("[mkimage] VERIFY FAIL: {e}");
            std::process::exit(1);
        });
        println!("[mkimage] VERIFY OK: {}", img_path.display());
        return;
    }

    let efi_binary = find_efi_binary();
    let efi_data = fs::read(&efi_binary).expect("Failed to read .efi binary");
    println!(
        "[mkimage] Input: {} ({} bytes)",
        efi_binary.display(),
        efi_data.len()
    );

    let out_dir = efi_binary
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));

    // 1. Raw GPT disk image (for QEMU, dd to USB, or VirtualBox SATA)
    let img_path = out_dir.join("seal-os.img");
    let disk = create_disk_image(&efi_data);
    fs::write(&img_path, &disk).expect("Failed to write disk image");
    verify_disk_image(&img_path, Some(&efi_data)).expect("Disk image self-verify failed");
    println!(
        "[mkimage] IMG: {} ({:.1} MB)",
        img_path.display(),
        disk.len() as f64 / 1024.0 / 1024.0
    );

    // 2. FAT12 EFI boot image for El Torito ISO
    let efi_boot_img = create_fat12_efi_image(&efi_data);
    let efi_img_path = out_dir.join("seal-os-efi.img");
    fs::write(&efi_img_path, &efi_boot_img).expect("Failed to write EFI boot image");
    println!(
        "[mkimage] EFI boot image: {} ({} bytes)",
        efi_img_path.display(),
        efi_boot_img.len()
    );

    println!("[mkimage] Boot with:");
    println!("  QEMU:    qemu-system-x86_64 -bios OVMF.fd -drive file={},format=raw -serial stdio -m 512M", img_path.display());
    println!(
        "  VBox:    convert raw image first: VBoxManage convertfromraw --format VDI {} seal-os.vdi",
        img_path.display()
    );
}

fn find_disk_image() -> PathBuf {
    let candidates = [
        "../seal-os/target/x86_64-unknown-uefi/release/seal-os.img",
        "../seal-os/target/x86_64-unknown-uefi/debug/seal-os.img",
        "target/x86_64-unknown-uefi/release/seal-os.img",
        "target/x86_64-unknown-uefi/debug/seal-os.img",
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    eprintln!("Error: seal-os.img not found. Run `cargo +stable run --release` first.");
    std::process::exit(1);
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
    disk[entry_offset] = 0x00; // status
    disk[entry_offset + 1] = 0x00; // CHS start head
    disk[entry_offset + 2] = 0x02; // CHS start sector (sector 2)
    disk[entry_offset + 3] = 0x00; // CHS start cylinder
    disk[entry_offset + 4] = 0xEE; // GPT protective
    disk[entry_offset + 5] = 0xFF; // CHS end head
    disk[entry_offset + 6] = 0xFF; // CHS end sector
    disk[entry_offset + 7] = 0xFF; // CHS end cylinder
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
        0x28, 0x73, 0x2A, 0xC1, 0x1F, 0xF8, 0xD2, 0x11, 0xBA, 0x4B, 0x00, 0xA0, 0xC9, 0x3E, 0xC9,
        0x3B,
    ];

    let part_guid: [u8; 16] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x53, 0x45, 0x41, 0x4C, 0x4F, 0x53, 0x00,
        0x01,
    ];

    let disk_guid: [u8; 16] = [
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x53, 0x45, 0x41, 0x4C, 0x4F, 0x53, 0x44,
        0x4B,
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
        1,
        total_sectors - 1,
        ESP_START_LBA,
        total_sectors - 34,
        &disk_guid,
        partition_entries_lba,
        num_partition_entries,
        partition_entry_size,
        pe_crc,
    );

    let backup_pe_lba = total_sectors - 33;
    let backup_pe_offset = (backup_pe_lba * SECTOR_SIZE) as usize;
    let pe_data: Vec<u8> = disk[pe_offset..pe_offset + pe_size].to_vec();
    disk[backup_pe_offset..backup_pe_offset + pe_size].copy_from_slice(&pe_data);

    let backup_hdr_offset = ((total_sectors - 1) * SECTOR_SIZE) as usize;
    write_gpt_header(
        &mut disk[backup_hdr_offset..backup_hdr_offset + 92],
        total_sectors - 1,
        1,
        ESP_START_LBA,
        total_sectors - 34,
        &disk_guid,
        backup_pe_lba,
        num_partition_entries,
        partition_entry_size,
        pe_crc,
    );
}

fn write_gpt_header(
    buf: &mut [u8],
    my_lba: u64,
    alt_lba: u64,
    first_usable: u64,
    last_usable: u64,
    disk_guid: &[u8; 16],
    pe_start_lba: u64,
    num_pe: u32,
    pe_size: u32,
    pe_crc: u32,
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
        .volume_label([
            b'S', b'E', b'A', b'L', b'-', b'O', b'S', b'-', b'E', b'F', b'I',
        ]);
    fatfs::format_volume(cursor, options).expect("Failed to format FAT32");

    {
        let cursor = Cursor::new(&mut img[..]);
        let fs =
            fatfs::FileSystem::new(cursor, fatfs::FsOptions::new()).expect("Failed to mount FAT32");
        let root = fs.root_dir();

        root.create_dir("EFI").expect("mkdir EFI");
        root.create_dir("EFI/BOOT").expect("mkdir EFI/BOOT");

        let mut file = root
            .create_file("EFI/BOOT/BOOTX64.EFI")
            .expect("create BOOTX64.EFI");
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
        .volume_label([
            b'E', b'F', b'I', b'B', b'O', b'O', b'T', b' ', b' ', b' ', b' ',
        ]);
    fatfs::format_volume(cursor, options).expect("Failed to format FAT12");

    {
        let cursor = Cursor::new(&mut img[..]);
        let fs =
            fatfs::FileSystem::new(cursor, fatfs::FsOptions::new()).expect("Failed to mount FAT12");
        let root = fs.root_dir();

        root.create_dir("EFI").expect("mkdir EFI");
        root.create_dir("EFI/BOOT").expect("mkdir EFI/BOOT");

        let mut file = root
            .create_file("EFI/BOOT/BOOTX64.EFI")
            .expect("create BOOTX64.EFI");
        file.write_all(efi_data).expect("write EFI binary");
    }
    img
}

fn verify_disk_image(path: &Path, expected_efi: Option<&[u8]>) -> Result<(), String> {
    let disk = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if disk.len() < DISK_SIZE as usize {
        return Err(format!(
            "image too small: {} bytes, expected at least {}",
            disk.len(),
            DISK_SIZE
        ));
    }
    if disk.get(510..512) != Some(&[0x55, 0xAA][..]) {
        return Err("protective MBR signature missing".into());
    }

    let hdr = disk
        .get(SECTOR_SIZE as usize..SECTOR_SIZE as usize + 92)
        .ok_or("primary GPT header missing")?;
    if &hdr[0..8] != b"EFI PART" {
        return Err("primary GPT signature missing".into());
    }

    let stored_hdr_crc = le_u32(&hdr[16..20]);
    let mut hdr_for_crc = hdr.to_vec();
    hdr_for_crc[16..20].copy_from_slice(&0u32.to_le_bytes());
    let actual_hdr_crc = crc32(&hdr_for_crc);
    if stored_hdr_crc != actual_hdr_crc {
        return Err(format!(
            "primary GPT CRC mismatch: stored {stored_hdr_crc:#x}, actual {actual_hdr_crc:#x}"
        ));
    }

    let pe_start = le_u64(&hdr[72..80]);
    let pe_count = le_u32(&hdr[80..84]) as usize;
    let pe_size = le_u32(&hdr[84..88]) as usize;
    let stored_pe_crc = le_u32(&hdr[88..92]);
    let pe_offset = (pe_start * SECTOR_SIZE) as usize;
    let pe_len = pe_count
        .checked_mul(pe_size)
        .ok_or("partition entry array size overflow")?;
    let pe = disk
        .get(pe_offset..pe_offset + pe_len)
        .ok_or("partition entry array missing")?;
    let actual_pe_crc = crc32(pe);
    if stored_pe_crc != actual_pe_crc {
        return Err(format!(
            "partition entry CRC mismatch: stored {stored_pe_crc:#x}, actual {actual_pe_crc:#x}"
        ));
    }

    let first_entry = &pe[0..pe_size];
    let esp_type_guid: [u8; 16] = [
        0x28, 0x73, 0x2A, 0xC1, 0x1F, 0xF8, 0xD2, 0x11, 0xBA, 0x4B, 0x00, 0xA0, 0xC9, 0x3E, 0xC9,
        0x3B,
    ];
    if first_entry[0..16] != esp_type_guid {
        return Err("first GPT partition is not an EFI System Partition".into());
    }
    let esp_start = le_u64(&first_entry[32..40]);
    let esp_end = le_u64(&first_entry[40..48]);
    if esp_start != ESP_START_LBA || esp_end != ESP_START_LBA + ESP_SIZE_SECTORS - 1 {
        return Err(format!(
            "ESP bounds wrong: start {esp_start}, end {esp_end}"
        ));
    }

    let esp_offset = (esp_start * SECTOR_SIZE) as usize;
    let esp_len = ((esp_end - esp_start + 1) * SECTOR_SIZE) as usize;
    let esp = disk
        .get(esp_offset..esp_offset + esp_len)
        .ok_or("ESP data outside image")?;
    let cursor = Cursor::new(esp.to_vec());
    let fs = fatfs::FileSystem::new(cursor, fatfs::FsOptions::new())
        .map_err(|e| format!("mount ESP FAT32: {e:?}"))?;
    let root = fs.root_dir();
    let mut boot = root
        .open_file("EFI/BOOT/BOOTX64.EFI")
        .map_err(|e| format!("open EFI/BOOT/BOOTX64.EFI: {e:?}"))?;
    let mut boot_data = Vec::new();
    boot.read_to_end(&mut boot_data)
        .map_err(|e| format!("read BOOTX64.EFI: {e}"))?;
    if boot_data.len() < 2 || &boot_data[0..2] != b"MZ" {
        return Err("BOOTX64.EFI missing PE/COFF MZ signature".into());
    }
    if let Some(expected) = expected_efi {
        if boot_data.len() != expected.len() {
            return Err(format!(
                "BOOTX64.EFI size mismatch: image {}, expected {}",
                boot_data.len(),
                expected.len()
            ));
        }
    }

    Ok(())
}

fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("u32 slice length"))
}

fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("u64 slice length"))
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

fn check_theorem_log(log_path: &Path) -> Result<(), String> {
    let text = fs::read_to_string(log_path)
        .map_err(|e| format!("read {}: {e}", log_path.display()))?;
    let required = [
        "[THEOREM] T1/TSS VERIFIED",
        "[THEOREM] T2/SCM VERIFIED",
        "[THEOREM] T3/GMC VERIFIED",
        "[THEOREM] T4/AGCR VERIFIED",
        "[THEOREM] T5/HCS VERIFIED",
        "[THEOREM] T6/RGCS VERIFIED",
        "[THEOREM] T7/PHKP VERIFIED",
        "[THEOREM] T8/TEB VERIFIED",
        "[THEOREM] T9/CMA VERIFIED",
        "[THEOREM] T10/WPHB VERIFIED",
        "[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths",
    ];
    let failed: Vec<&str> = text
        .lines()
        .filter(|line| line.contains("[THEOREM]") && line.contains("FAILED"))
        .collect();
    if !failed.is_empty() {
        return Err(format!("theorem failure lines found: {}", failed.join(" | ")));
    }
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|pattern| !text.contains(pattern))
        .collect();
    if !missing.is_empty() {
        return Err(format!("missing theorem patterns: {}", missing.join(" | ")));
    }
    Ok(())
}

fn check_seal_abi(root: &Path) -> Result<(), String> {
    let src = root.join("kernel").join("seal-os").join("src");
    let mut files = Vec::new();
    collect_rs_files(&src, &mut files)?;

    let banned = [
        ("extern crate libc", "libc crate import"),
        ("use libc", "libc use"),
        ("std::os::unix", "Unix std API"),
        ("std::os::linux", "Linux std API"),
        ("posix_spawn", "POSIX spawn"),
        ("pthread_", "pthread API"),
        ("forkpty", "Unix pty API"),
        ("termios", "termios API"),
    ];

    let mut findings = Vec::new();
    for path in files {
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        for (idx, line) in text.lines().enumerate() {
            let code = strip_allowed_comment(line);
            for (needle, reason) in banned {
                if code.contains(needle) {
                    findings.push(format!(
                        "{}:{}: {reason}: {}",
                        path.display(),
                        idx + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn unsafe_inventory(root: &Path) -> Result<(), String> {
    let src = root.join("kernel").join("seal-os").join("src");
    let mut files = Vec::new();
    collect_rs_files(&src, &mut files)?;

    let mut total = 0usize;
    for path in files {
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let count = text.lines().filter(|line| is_unsafe_site(line)).count();
        if count > 0 {
            total += count;
            println!("[seal-audit] unsafe {count:>3} {}", path.display());
        }
    }
    println!("[seal-audit] unsafe total {total}");
    Ok(())
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir entry {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn strip_allowed_comment(line: &str) -> &str {
    let Some((code, comment)) = line.split_once("//") else {
        return line;
    };
    let comment_lower = comment.to_ascii_lowercase();
    if comment_lower.contains("not posix") || comment_lower.contains("historical") {
        code
    } else {
        code
    }
}

fn is_unsafe_site(line: &str) -> bool {
    line.contains("unsafe {")
        || line.contains("unsafe{")
        || line.contains("unsafe fn")
        || line.contains("unsafe impl")
}
