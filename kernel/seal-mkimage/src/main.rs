// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Creates bootable UEFI disk images and ISOs for Seal OS.
//! VirtualBox-compatible: GPT + FAT32 ESP with proper boot sector.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

const DISK_SIZE: u64 = 128 * 1024 * 1024; // 128MB
const SECTOR_SIZE: u64 = 512;
const ESP_START_LBA: u64 = 2048;
const ESP_SIZE_SECTORS: u64 = (64 * 1024 * 1024) / SECTOR_SIZE;
const MANIFOLD_START_LBA: u64 = ESP_START_LBA + ESP_SIZE_SECTORS;
const MANIFOLD_END_LBA: u64 = (DISK_SIZE / SECTOR_SIZE) - 34;
const MANIFOLD_SIZE_SECTORS: u64 = MANIFOLD_END_LBA - MANIFOLD_START_LBA + 1;
const MNFD_JOURNAL_BLOCKS: u64 = 1024;
const MNFD_FREE_BITMAP_BLOCKS: u64 = 256;
const EXPECTED_SEAL_OS_BANNER: &str = "Seal OS v0.4.5";
const CURRENT_UBUNTU_BASELINE_VERSION: &str = "26.04";

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
    if args.get(1).map(|s| s.as_str()) == Some("--check-vm-proof") {
        let log_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from("../seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log")
        });
        check_vm_proof(&log_path, VmProofTarget::Qemu).unwrap_or_else(|e| {
            eprintln!("[seal-audit] VM PROOF FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] VM PROOF OK: {}", log_path.display());
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-vbox-proof") {
        let log_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from("../seal-os/target/x86_64-unknown-uefi/release/vbox-smoke/serial.log")
        });
        check_vm_proof(&log_path, VmProofTarget::VirtualBox).unwrap_or_else(|e| {
            eprintln!("[seal-audit] VBOX PROOF FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] VBOX PROOF OK: {}", log_path.display());
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-seal-abi") {
        let root = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_seal_abi(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] SEAL ABI FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] SEAL ABI OK");
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--unsafe-inventory") {
        let root = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        unsafe_inventory(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] UNSAFE INVENTORY FAIL: {e}");
            std::process::exit(1);
        });
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-language-hygiene") {
        let root = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_language_hygiene(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] LANGUAGE HYGIENE FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] LANGUAGE HYGIENE OK");
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-doc-claim-contract") {
        let root = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_doc_claim_contract(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] DOC CLAIM CONTRACT FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] DOC CLAIM CONTRACT OK");
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-lean-proof-hygiene") {
        let root = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_lean_proof_hygiene(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] LEAN PROOF HYGIENE FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] LEAN PROOF HYGIENE OK");
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-aether-migration") {
        let root = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_aether_migration(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] AETHER MIGRATION FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] AETHER MIGRATION OK");
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-aether-runtime") {
        let log_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from("../seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log")
        });
        check_aether_runtime_log(&log_path).unwrap_or_else(|e| {
            eprintln!("[seal-audit] AETHER RUNTIME FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] AETHER RUNTIME OK: {}", log_path.display());
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-runtime-theorems") {
        let root = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_runtime_theorems(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] RUNTIME THEOREMS FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] RUNTIME THEOREMS OK");
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-o1-allocator") {
        let root = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_o1_allocator(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] O(1) ALLOCATOR FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] O(1) ALLOCATOR OK");
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-proof-screen") {
        let ppm = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from("../seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.ppm")
        });
        check_proof_screen(&ppm).unwrap_or_else(|e| {
            eprintln!("[seal-audit] PROOF SCREEN FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] PROOF SCREEN OK: {}", ppm.display());
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-proof-manifest") {
        let manifest = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(
                "../seal-os/target/x86_64-unknown-uefi/release/qemu-proof/proof-manifest.txt",
            )
        });
        check_proof_manifest(&manifest).unwrap_or_else(|e| {
            eprintln!("[seal-audit] PROOF MANIFEST FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] PROOF MANIFEST OK: {}", manifest.display());
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-desktop-soak") {
        let log_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from("../seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log")
        });
        check_desktop_soak(&log_path).unwrap_or_else(|e| {
            eprintln!("[seal-audit] DESKTOP SOAK FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] DESKTOP SOAK OK: {}", log_path.display());
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-benchmark-log") {
        let log_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from("../seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log")
        });
        check_benchmark_log(&log_path).unwrap_or_else(|e| {
            eprintln!("[seal-audit] BENCHMARK LOG FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] BENCHMARK LOG OK: {}", log_path.display());
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-ubuntu-benchmark-log") {
        let log_path = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("ubuntu-alloc.log"));
        check_ubuntu_benchmark_log(&log_path).unwrap_or_else(|e| {
            eprintln!("[seal-audit] UBUNTU BENCHMARK FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] UBUNTU BENCHMARK OK: {}", log_path.display());
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--compare-benchmark-logs") {
        let seal_log = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from("../seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log")
        });
        let ubuntu_log = args
            .get(3)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("ubuntu-alloc.log"));
        compare_benchmark_logs(&seal_log, &ubuntu_log).unwrap_or_else(|e| {
            eprintln!("[seal-audit] BENCHMARK COMPARE FAIL: {e}");
            std::process::exit(1);
        });
        println!(
            "[seal-audit] BENCHMARK COMPARE OK: Seal beats Ubuntu for alloc-frame ({}, {})",
            seal_log.display(),
            ubuntu_log.display()
        );
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--verify") {
        let img_path = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(find_disk_image);
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
    println!(
        "  QEMU:    qemu-system-x86_64 -machine q35 -drive if=pflash,format=raw,readonly=on,file=OVMF.fd -device ahci,id=seal_sata -drive if=none,id=seal_disk,file={},format=raw,media=disk -device ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0 -serial stdio -m 4G",
        img_path.display()
    );
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
    write_manifold_partition(&mut disk);

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
    let manifold_type_guid: [u8; 16] = [
        0x4D, 0x4E, 0x46, 0x44, 0x2D, 0x53, 0x45, 0x41, 0x4C, 0x2D, 0x54, 0x4F, 0x50, 0x4F, 0x00,
        0x01,
    ];

    let esp_part_guid: [u8; 16] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x53, 0x45, 0x41, 0x4C, 0x4F, 0x53, 0x00,
        0x01,
    ];
    let manifold_part_guid: [u8; 16] = [
        0x02, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x4D, 0x4E, 0x46, 0x44, 0x4F, 0x53, 0x00,
        0x02,
    ];

    let disk_guid: [u8; 16] = [
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x53, 0x45, 0x41, 0x4C, 0x4F, 0x53, 0x44,
        0x4B,
    ];

    let pe_offset = (partition_entries_lba * SECTOR_SIZE) as usize;
    disk[pe_offset..pe_offset + 16].copy_from_slice(&esp_type_guid);
    disk[pe_offset + 16..pe_offset + 32].copy_from_slice(&esp_part_guid);
    disk[pe_offset + 32..pe_offset + 40].copy_from_slice(&ESP_START_LBA.to_le_bytes());
    let esp_end_lba = ESP_START_LBA + ESP_SIZE_SECTORS - 1;
    disk[pe_offset + 40..pe_offset + 48].copy_from_slice(&esp_end_lba.to_le_bytes());

    let name = "EFI System";
    for (i, ch) in name.encode_utf16().enumerate() {
        let off = pe_offset + 56 + i * 2;
        disk[off..off + 2].copy_from_slice(&ch.to_le_bytes());
    }

    let mnfd_offset = pe_offset + partition_entry_size as usize;
    disk[mnfd_offset..mnfd_offset + 16].copy_from_slice(&manifold_type_guid);
    disk[mnfd_offset + 16..mnfd_offset + 32].copy_from_slice(&manifold_part_guid);
    disk[mnfd_offset + 32..mnfd_offset + 40].copy_from_slice(&MANIFOLD_START_LBA.to_le_bytes());
    disk[mnfd_offset + 40..mnfd_offset + 48].copy_from_slice(&MANIFOLD_END_LBA.to_le_bytes());

    let name = "ManifoldFS";
    for (i, ch) in name.encode_utf16().enumerate() {
        let off = mnfd_offset + 56 + i * 2;
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

fn write_manifold_partition(disk: &mut [u8]) {
    let offset = (MANIFOLD_START_LBA * SECTOR_SIZE) as usize;
    let superblock = create_manifold_superblock(MANIFOLD_SIZE_SECTORS);
    disk[offset..offset + superblock.len()].copy_from_slice(&superblock);
}

fn create_manifold_superblock(total_blocks: u64) -> [u8; SECTOR_SIZE as usize] {
    let mut sb = [0u8; SECTOR_SIZE as usize];
    let block_size = SECTOR_SIZE as u32;
    let journal_start = 1u64;
    let free_bitmap_start = journal_start + MNFD_JOURNAL_BLOCKS;
    let free_bitmap_blocks = MNFD_FREE_BITMAP_BLOCKS;
    let inode_table_start = free_bitmap_start + free_bitmap_blocks;
    let max_inodes = total_blocks.saturating_sub(inode_table_start);
    let inode_table_blocks = (max_inodes * 256).div_ceil(block_size as u64);
    let data_start = inode_table_start + inode_table_blocks;

    sb[0..4].copy_from_slice(b"MNFD");
    sb[4..8].copy_from_slice(&1u32.to_le_bytes());
    sb[8..12].copy_from_slice(&block_size.to_le_bytes());
    sb[16..24].copy_from_slice(&max_inodes.to_le_bytes());
    sb[24..32].copy_from_slice(&free_bitmap_start.to_le_bytes());
    sb[32..40].copy_from_slice(&free_bitmap_blocks.to_le_bytes());
    sb[40..48].copy_from_slice(&journal_start.to_le_bytes());
    sb[48..56].copy_from_slice(&MNFD_JOURNAL_BLOCKS.to_le_bytes());
    sb[56..64].copy_from_slice(&inode_table_start.to_le_bytes());
    sb[64..72].copy_from_slice(&inode_table_blocks.to_le_bytes());
    sb[72..80].copy_from_slice(&data_start.to_le_bytes());
    sb[80..88].copy_from_slice(&total_blocks.to_le_bytes());
    sb[88..96].copy_from_slice(&1u64.to_le_bytes());
    sb[96..100].copy_from_slice(&0u32.to_le_bytes());
    sb
}

#[allow(clippy::too_many_arguments)]
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

    let second_entry = pe
        .get(pe_size..pe_size * 2)
        .ok_or("second GPT partition entry missing")?;
    let manifold_type_guid: [u8; 16] = [
        0x4D, 0x4E, 0x46, 0x44, 0x2D, 0x53, 0x45, 0x41, 0x4C, 0x2D, 0x54, 0x4F, 0x50, 0x4F, 0x00,
        0x01,
    ];
    if second_entry[0..16] != manifold_type_guid {
        return Err("second GPT partition is not ManifoldFS".into());
    }
    let mnfd_start = le_u64(&second_entry[32..40]);
    let mnfd_end = le_u64(&second_entry[40..48]);
    if mnfd_start != MANIFOLD_START_LBA || mnfd_end != MANIFOLD_END_LBA {
        return Err(format!(
            "ManifoldFS bounds wrong: start {mnfd_start}, end {mnfd_end}"
        ));
    }
    let mnfd_offset = (mnfd_start * SECTOR_SIZE) as usize;
    let mnfd_sb = disk
        .get(mnfd_offset..mnfd_offset + SECTOR_SIZE as usize)
        .ok_or("ManifoldFS superblock outside image")?;
    if &mnfd_sb[0..4] != b"MNFD" {
        return Err("ManifoldFS MNFD superblock missing".into());
    }
    if le_u32(&mnfd_sb[4..8]) != 1 || le_u32(&mnfd_sb[8..12]) != SECTOR_SIZE as u32 {
        return Err("ManifoldFS superblock version/block size invalid".into());
    }
    if le_u64(&mnfd_sb[80..88]) != MANIFOLD_SIZE_SECTORS {
        return Err("ManifoldFS superblock total_blocks mismatch".into());
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
    let text =
        fs::read_to_string(log_path).map_err(|e| format!("read {}: {e}", log_path.display()))?;
    let required = [
        EXPECTED_SEAL_OS_BANNER,
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
        "[ALLOC] O(1) proof:",
        "[BENCH] toporam-alloc",
        "[BENCH] alloc-frame",
        "[BENCH] manifold-teleport",
        "[BENCH] scheduler-select-next",
        "[Aether-Lang] runtime proof:",
        "[BOOT] Desktop proof frame blit done",
        "[GFX] desktop-soak",
        "[BOOT] Seal OS desktop ready.",
        "[EVENT] Entering real event loop",
    ];
    let fatal_markers = [
        "panic",
        "PANIC",
        "[FAULT]",
        "PAGE FAULT",
        "DOUBLE FAULT",
        "[WATCHDOG]",
        "WATCHDOG FIRED",
        "triple fault",
        "qemu: fatal",
    ];
    let failed: Vec<&str> = text
        .lines()
        .filter(|line| line.contains("[THEOREM]") && line.contains("FAILED"))
        .collect();
    if !failed.is_empty() {
        return Err(format!(
            "theorem failure lines found: {}",
            failed.join(" | ")
        ));
    }
    let fatal: Vec<&str> = text
        .lines()
        .filter(|line| fatal_markers.iter().any(|marker| line.contains(marker)))
        .collect();
    if !fatal.is_empty() {
        return Err(format!("fatal boot markers found: {}", fatal.join(" | ")));
    }
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|pattern| !text.contains(pattern))
        .collect();
    if !missing.is_empty() {
        return Err(format!("missing theorem patterns: {}", missing.join(" | ")));
    }
    check_desktop_soak_text(&text)?;
    check_benchmark_text(&text)?;
    check_aether_runtime_text(&text)?;
    Ok(())
}

fn check_aether_runtime_log(log_path: &Path) -> Result<(), String> {
    let text =
        fs::read_to_string(log_path).map_err(|e| format!("read {}: {e}", log_path.display()))?;
    check_aether_runtime_text(&text)
}

fn check_aether_runtime_text(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[Aether-Lang] runtime proof:")?;
    let parser = parse_field(line, "parser=")?;
    let interpreter = parse_field(line, "interpreter=")?;
    let app_host = parse_field(line, "app_host=")?;
    let script = parse_field(line, "script=")?;
    let result = parse_field(line, "result=")?;

    if parser != "ok" || interpreter != "ok" || app_host != "ok" {
        return Err(format!(
            "Aether runtime proof must have parser/interpreter/app_host ok: parser={parser}, interpreter={interpreter}, app_host={app_host}"
        ));
    }
    if script != "aether_boot_probe" {
        return Err(format!(
            "Aether runtime proof used unexpected script: script={script}"
        ));
    }
    if result != "seal-topology-ok" {
        return Err(format!(
            "Aether runtime proof returned unexpected result: result={result}"
        ));
    }

    Ok(())
}

fn check_benchmark_log(log_path: &Path) -> Result<(), String> {
    let text =
        fs::read_to_string(log_path).map_err(|e| format!("read {}: {e}", log_path.display()))?;
    check_benchmark_text(&text)
}

fn check_benchmark_text(text: &str) -> Result<(), String> {
    parse_alloc_o1_proof(text)?;
    parse_toporam_alloc_benchmark(text)?;
    parse_seal_alloc_benchmark(text)?;
    parse_manifold_teleport_benchmark(text)?;
    parse_scheduler_select_benchmark(text)?;
    Ok(())
}

fn parse_alloc_o1_proof(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[ALLOC] O(1) proof:")?;
    let topo_cells = parse_metric(line, "topo_cells=")?;
    let l3_word_probes = parse_metric(line, "l3_word_probes_per_cell=")?;
    let single_word_probes = parse_metric(line, "single_word_probes_per_cell=")?;
    let contiguous_candidate_probes = parse_metric(line, "contiguous_candidate_probes=")?;
    let contiguous_max_run_pages = parse_metric(line, "contiguous_max_run_pages=")?;
    let toporam_max_run_pages = parse_metric(line, "toporam_max_run_pages=")?;
    let marking = parse_field(line, "marking=")?;

    if topo_cells != 8 {
        return Err(format!(
            "allocator proof topo_cells must be 8, got {topo_cells}"
        ));
    }
    if l3_word_probes > 2 {
        return Err(format!(
            "allocator proof l3 probes too high: {l3_word_probes}"
        ));
    }
    if single_word_probes > 8192 {
        return Err(format!(
            "allocator proof single word probes too high: {single_word_probes}"
        ));
    }
    if contiguous_candidate_probes > 128 {
        return Err(format!(
            "allocator proof contiguous candidate probes too high: {contiguous_candidate_probes}"
        ));
    }
    if contiguous_max_run_pages > 64 {
        return Err(format!(
            "allocator proof contiguous max run too high: {contiguous_max_run_pages}"
        ));
    }
    if toporam_max_run_pages != contiguous_max_run_pages {
        return Err(format!(
            "TopoRAM run cap must match physical contiguous run cap: toporam={toporam_max_run_pages}, contiguous={contiguous_max_run_pages}"
        ));
    }
    if marking != "bounded_by_contiguous_max_run_pages" {
        return Err(format!(
            "allocator proof marking must be bounded, got `{marking}`"
        ));
    }

    Ok(())
}

fn check_ubuntu_benchmark_log(log_path: &Path) -> Result<(), String> {
    let text =
        fs::read_to_string(log_path).map_err(|e| format!("read {}: {e}", log_path.display()))?;
    parse_ubuntu_alloc_benchmark(&text)?;
    Ok(())
}

fn compare_benchmark_logs(seal_log: &Path, ubuntu_log: &Path) -> Result<(), String> {
    let seal_text =
        fs::read_to_string(seal_log).map_err(|e| format!("read {}: {e}", seal_log.display()))?;
    let ubuntu_text = fs::read_to_string(ubuntu_log)
        .map_err(|e| format!("read {}: {e}", ubuntu_log.display()))?;
    let seal = parse_seal_alloc_benchmark(&seal_text)?;
    let ubuntu = parse_ubuntu_alloc_benchmark(&ubuntu_text)?;
    require_seal_alloc_bench_lead(seal, ubuntu)
}

fn require_seal_alloc_bench_lead(
    seal: AllocBenchMetrics,
    ubuntu: AllocBenchMetrics,
) -> Result<(), String> {
    if seal.p50 >= ubuntu.p50 {
        return Err(format!(
            "Seal alloc-frame p50 does not beat Ubuntu: seal={}, ubuntu={}",
            seal.p50, ubuntu.p50
        ));
    }
    if seal.p95 >= ubuntu.p95 {
        return Err(format!(
            "Seal alloc-frame p95 does not beat Ubuntu: seal={}, ubuntu={}",
            seal.p95, ubuntu.p95
        ));
    }
    if seal.max >= ubuntu.max {
        return Err(format!(
            "Seal alloc-frame max does not beat Ubuntu: seal={}, ubuntu={}",
            seal.max, ubuntu.max
        ));
    }

    Ok(())
}

#[derive(Copy, Clone)]
struct AllocBenchMetrics {
    iterations: u64,
    ok: u64,
    p50: u64,
    p95: u64,
    max: u64,
}

fn parse_seal_alloc_benchmark(text: &str) -> Result<AllocBenchMetrics, String> {
    let line = find_marker_line(text, "[BENCH] alloc-frame")?;
    let bench = parse_alloc_bench_metrics(line, "Seal allocator benchmark")?;
    let fast_hits_delta = parse_metric(line, "fast_hits_delta=")?;
    let misses_delta = parse_metric(line, "bounded_misses_delta=")?;
    let contig_delta = parse_metric(line, "max_contiguous_probes_seen_delta=")?;
    let free_before = parse_metric(line, "free_before=")?;
    let free_after = parse_metric(line, "free_after=")?;
    if fast_hits_delta < bench.iterations {
        return Err(format!(
            "allocator benchmark did not hit the topological fast path enough: fast_hits_delta={fast_hits_delta}, iterations={}",
            bench.iterations
        ));
    }
    if misses_delta != 0 {
        return Err(format!(
            "allocator benchmark saw bounded misses: bounded_misses_delta={misses_delta}"
        ));
    }
    if contig_delta != 0 {
        return Err(format!(
            "single-frame allocator benchmark touched contiguous probe path: delta={contig_delta}"
        ));
    }
    if free_before != free_after {
        return Err(format!(
            "allocator benchmark leaked frames: free_before={free_before}, free_after={free_after}"
        ));
    }
    Ok(bench)
}

fn parse_toporam_alloc_benchmark(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[BENCH] toporam-alloc")?;
    let bench = parse_alloc_bench_metrics(line, "TopoRAM allocation benchmark")?;
    let target_hits = parse_metric(line, "target_cell_hits_delta=")?;
    let target_fallbacks = parse_metric(line, "target_cell_fallbacks_delta=")?;
    let low_to_high = parse_metric(line, "low_to_high_fallbacks_delta=")?;
    let high_to_low = parse_metric(line, "high_to_low_fallbacks_delta=")?;
    let pcie_to_high = parse_metric(line, "pcie_to_high_fallbacks_delta=")?;
    let pcie_to_low = parse_metric(line, "pcie_to_low_fallbacks_delta=")?;
    let free_before = parse_metric(line, "free_before=")?;
    let free_after = parse_metric(line, "free_after=")?;

    if target_hits < bench.iterations {
        return Err(format!(
            "TopoRAM benchmark did not use target-cell fast path enough: target_cell_hits_delta={target_hits}, iterations={}",
            bench.iterations
        ));
    }
    if target_fallbacks != 0 {
        return Err(format!(
            "TopoRAM benchmark fell back from target cell: target_cell_fallbacks_delta={target_fallbacks}"
        ));
    }
    if low_to_high != 0 || high_to_low != 0 || pcie_to_high != 0 || pcie_to_low != 0 {
        return Err(format!(
            "TopoRAM benchmark used zone fallback: low_to_high={low_to_high}, high_to_low={high_to_low}, pcie_to_high={pcie_to_high}, pcie_to_low={pcie_to_low}"
        ));
    }
    if free_before != free_after {
        return Err(format!(
            "TopoRAM benchmark leaked frames: free_before={free_before}, free_after={free_after}"
        ));
    }

    Ok(())
}

fn parse_manifold_teleport_benchmark(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[BENCH] manifold-teleport")?;
    let api = parse_field(line, "api=")?;
    let fs_mode = parse_field(line, "fs_mode=")?;
    let persistence = parse_field(line, "persistence=")?;
    let samples = parse_metric(line, "samples=")?;
    let ok = parse_metric(line, "ok=")?;
    let same_inode = parse_metric(line, "same_inode=")?;
    let src_gone = parse_metric(line, "src_gone=")?;
    let dst_present = parse_metric(line, "dst_present=")?;
    let entries_min = parse_metric(line, "entries_min=")?;
    let entries_max = parse_metric(line, "entries_max=")?;
    let payload_bytes = parse_metric(line, "payload_bytes=")?;
    let p50 = parse_metric(line, "p50_cycles=")?;
    let p95 = parse_metric(line, "p95_cycles=")?;
    let max = parse_metric(line, "max_cycles=")?;
    let _ticks_max = parse_metric(line, "ticks_max=")?;
    let metadata_ops_max = parse_metric(line, "metadata_ops_max=")?;
    let write_through_bytes = parse_metric(line, "write_through_bytes_per_move=")?;
    let payload_points = parse_metric(line, "payload_points=")?;

    if api != "teleport" || fs_mode != "ramfs" || persistence != "write_through" {
        return Err(format!(
            "ManifoldFS teleport benchmark identifies the wrong API/path: api={api}, fs_mode={fs_mode}, persistence={persistence}"
        ));
    }
    if samples < 3 {
        return Err(format!(
            "ManifoldFS teleport benchmark sample count too low: {samples}"
        ));
    }
    if ok != samples || same_inode != samples || src_gone != samples || dst_present != samples {
        return Err(format!(
            "ManifoldFS teleport benchmark failed move invariants: ok={ok}, same_inode={same_inode}, src_gone={src_gone}, dst_present={dst_present}, samples={samples}"
        ));
    }
    if entries_min == 0 || entries_max < 256 || entries_max <= entries_min {
        return Err(format!(
            "ManifoldFS teleport benchmark did not span enough directory population: min={entries_min}, max={entries_max}"
        ));
    }
    if payload_bytes == 0 || payload_points == 0 {
        return Err(format!(
            "ManifoldFS teleport benchmark payload not encoded: bytes={payload_bytes}, points={payload_points}"
        ));
    }
    if p50 == 0 || p95 == 0 || max == 0 || p50 > p95 || p95 > max {
        return Err(format!(
            "ManifoldFS teleport cycle metrics invalid: p50={p50}, p95={p95}, max={max}"
        ));
    }
    if metadata_ops_max > 7 {
        return Err(format!(
            "ManifoldFS teleport metadata operation bound regressed: metadata_ops_max={metadata_ops_max}"
        ));
    }
    if write_through_bytes != payload_bytes {
        return Err(format!(
            "ManifoldFS teleport write-through accounting mismatch: write_through_bytes_per_move={write_through_bytes}, payload_bytes={payload_bytes}"
        ));
    }
    Ok(())
}

fn parse_scheduler_select_benchmark(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[BENCH] scheduler-select-next")?;
    let selector = parse_field(line, "selector=")?;
    let mode = parse_field(line, "mode=")?;
    let clock = parse_field(line, "clock=")?;
    let iterations = parse_metric(line, "iterations=")?;
    let ok = parse_metric(line, "ok=")?;
    let ready_before = parse_metric(line, "ready_before=")?;
    let ready_after = parse_metric(line, "ready_after=")?;
    let cells = parse_metric(line, "cells=")?;
    let priority_buckets = parse_metric(line, "priority_buckets=")?;
    let voronoi_locate_probes = parse_metric(line, "voronoi_locate_probes=")?;
    let max_cell_bitmap_tests = parse_metric(line, "max_cell_bitmap_tests=")?;
    let max_priority_bucket_scan = parse_metric(line, "max_priority_bucket_scan=")?;
    let context_switches = parse_metric(line, "context_switches=")?;
    let selected_priority_max = parse_metric(line, "selected_priority_max=")?;
    let p50 = parse_metric(line, "p50_cycles=")?;
    let p95 = parse_metric(line, "p95_cycles=")?;
    let max = parse_metric(line, "max_cycles=")?;

    if selector != "select_next_task" || mode != "live_requeue" || clock != "rdtsc" {
        return Err(format!(
            "scheduler select benchmark identifies the wrong selector/path: selector={selector}, mode={mode}, clock={clock}"
        ));
    }
    if iterations < 64 {
        return Err(format!(
            "scheduler select benchmark iteration count too low: {iterations}"
        ));
    }
    if ok != iterations {
        return Err(format!(
            "scheduler select benchmark failed selections: ok={ok}, iterations={iterations}"
        ));
    }
    if ready_before < 1 {
        return Err(format!(
            "scheduler select benchmark had no ready tasks: ready_before={ready_before}"
        ));
    }
    if ready_after != ready_before {
        return Err(format!(
            "scheduler select benchmark did not requeue live state: ready_before={ready_before}, ready_after={ready_after}"
        ));
    }
    if cells != 8 || priority_buckets != 256 {
        return Err(format!(
            "scheduler select static bounds changed: cells={cells}, priority_buckets={priority_buckets}"
        ));
    }
    if voronoi_locate_probes != 8 || max_cell_bitmap_tests > 9 {
        return Err(format!(
            "scheduler select cell probe bounds regressed: voronoi_locate_probes={voronoi_locate_probes}, max_cell_bitmap_tests={max_cell_bitmap_tests}"
        ));
    }
    if max_priority_bucket_scan > 256 {
        return Err(format!(
            "scheduler select priority scan bound regressed: max_priority_bucket_scan={max_priority_bucket_scan}"
        ));
    }
    if context_switches != 0 {
        return Err(format!(
            "scheduler select benchmark performed context switches: context_switches={context_switches}"
        ));
    }
    if selected_priority_max == 0 {
        return Err(String::from(
            "scheduler select benchmark never selected a nonzero-priority task",
        ));
    }
    if p50 == 0 || p95 == 0 || max == 0 || p50 > p95 || p95 > max {
        return Err(format!(
            "scheduler select cycle metrics invalid: p50={p50}, p95={p95}, max={max}"
        ));
    }

    Ok(())
}

fn parse_ubuntu_alloc_benchmark(text: &str) -> Result<AllocBenchMetrics, String> {
    let line = find_marker_line(text, "[UBUNTU-BENCH] alloc-frame")?;
    let os = parse_field(line, "os=")?;
    if os != "ubuntu" {
        return Err(format!(
            "Ubuntu benchmark artifact must be produced on Ubuntu: os={os}"
        ));
    }
    let version_id = parse_field(line, "version_id=")?;
    if version_id != CURRENT_UBUNTU_BASELINE_VERSION {
        return Err(format!(
            "Ubuntu benchmark artifact must target current Ubuntu baseline {CURRENT_UBUNTU_BASELINE_VERSION}: version_id={version_id}"
        ));
    }
    let kernel = parse_field(line, "kernel=")?;
    let kernel_lower = kernel.to_ascii_lowercase();
    if kernel_lower.contains("microsoft") || kernel_lower.contains("wsl") {
        return Err(format!(
            "Ubuntu benchmark artifact must come from native VM/bare-metal Ubuntu, not WSL: kernel={kernel}"
        ));
    }
    let bytes = parse_metric(line, "bytes=")?;
    if bytes != 4096 {
        return Err(format!(
            "Ubuntu alloc-frame benchmark must use 4096-byte pages: bytes={bytes}"
        ));
    }
    parse_alloc_bench_metrics(line, "Ubuntu alloc-frame benchmark")
}

fn parse_alloc_bench_metrics(line: &str, label: &str) -> Result<AllocBenchMetrics, String> {
    let bench = AllocBenchMetrics {
        iterations: parse_metric(line, "iterations=")?,
        ok: parse_metric(line, "ok=")?,
        p50: parse_metric(line, "p50_cycles=")?,
        p95: parse_metric(line, "p95_cycles=")?,
        max: parse_metric(line, "max_cycles=")?,
    };
    if bench.iterations < 64 {
        return Err(format!(
            "{label} iteration count too low: {}",
            bench.iterations
        ));
    }
    if bench.ok != bench.iterations {
        return Err(format!(
            "{label} failed allocations: ok={}, iterations={}",
            bench.ok, bench.iterations
        ));
    }
    if bench.p50 == 0 || bench.p95 == 0 || bench.max == 0 {
        return Err(format!("{label} cycle counters must be nonzero"));
    }
    if bench.p50 > bench.p95 || bench.p95 > bench.max {
        return Err(format!(
            "{label} cycle percentiles are not monotonic: p50={}, p95={}, max={}",
            bench.p50, bench.p95, bench.max
        ));
    }
    Ok(bench)
}

fn find_marker_line<'a>(text: &'a str, marker: &str) -> Result<&'a str, String> {
    text.lines()
        .find(|line| line.starts_with(marker))
        .ok_or_else(|| format!("missing {marker} marker"))
}

fn check_desktop_soak(log_path: &Path) -> Result<(), String> {
    let text =
        fs::read_to_string(log_path).map_err(|e| format!("read {}: {e}", log_path.display()))?;
    check_desktop_soak_text(&text)
}

fn check_desktop_soak_text(text: &str) -> Result<(), String> {
    let line = text
        .lines()
        .find(|line| line.starts_with("[GFX] desktop-soak"))
        .ok_or_else(|| String::from("missing [GFX] desktop-soak marker"))?;
    let frames = parse_metric(line, "frames=")?;
    let p50 = parse_metric(line, "p50_cycles=")?;
    let p95 = parse_metric(line, "p95_cycles=")?;
    let max = parse_metric(line, "max_cycles=")?;
    let dirty_px_max = parse_metric(line, "dirty_px_max=")?;
    if frames < 16 {
        return Err(format!("desktop soak frames too low: {frames}"));
    }
    if p50 == 0 || p95 == 0 || max == 0 {
        return Err(String::from("desktop soak cycle counters must be nonzero"));
    }
    if p50 > p95 || p95 > max {
        return Err(format!(
            "desktop soak cycle percentiles are not monotonic: p50={p50}, p95={p95}, max={max}"
        ));
    }
    if dirty_px_max < 1024 * 768 {
        return Err(format!(
            "desktop soak dirty pixel coverage too small: {dirty_px_max}"
        ));
    }
    if !line.contains("input_events=") {
        return Err(String::from("desktop soak missing input_events field"));
    }
    Ok(())
}

fn check_proof_manifest(manifest_path: &Path) -> Result<(), String> {
    let text = fs::read_to_string(manifest_path)
        .map_err(|e| format!("read {}: {e}", manifest_path.display()))?;
    let fields = parse_manifest_fields(&text)?;
    let parent = manifest_path
        .parent()
        .ok_or_else(|| format!("manifest has no parent: {}", manifest_path.display()))?;

    require_manifest_field_eq(&fields, "seal_proof_manifest_version", "1")?;
    let vm_target = require_manifest_field(&fields, "vm_target")?;
    if vm_target != "qemu" && vm_target != "vbox" {
        return Err(format!("unexpected vm_target `{vm_target}`"));
    }
    require_manifest_field_eq(&fields, "proof_verdict", "PASS")?;
    require_manifest_field_eq(&fields, "gate_verify", "ok")?;
    require_manifest_field_eq(&fields, "gate_theorem_log", "ok")?;
    require_manifest_field_eq(&fields, "gate_aether_runtime", "ok")?;
    require_manifest_field_eq(&fields, "gate_desktop_soak", "ok")?;
    require_manifest_field_eq(&fields, "gate_benchmark_log", "ok")?;

    match vm_target {
        "qemu" => {
            require_manifest_field_eq(&fields, "gate_vm_proof", "ok")?;
            require_manifest_field_eq(&fields, "gate_proof_screen", "ok")?;
            let backend = require_manifest_field(&fields, "qemu_backend")?;
            if backend != "native" && backend != "wsl" && backend != "ci" {
                return Err(format!("unexpected qemu_backend `{backend}`"));
            }
        }
        "vbox" => {
            require_manifest_field_eq(&fields, "gate_vbox_proof", "ok")?;
            let backend = require_manifest_field(&fields, "virtualbox_backend")?;
            if backend != "headless" {
                return Err(format!("unexpected virtualbox_backend `{backend}`"));
            }
        }
        _ => unreachable!(),
    }

    let git_commit = require_manifest_field(&fields, "git_commit")?;
    if git_commit != "unknown" && (git_commit.len() < 7 || !is_lower_hex(git_commit)) {
        return Err(format!("invalid git_commit `{git_commit}`"));
    }

    let git_dirty = require_manifest_field(&fields, "git_dirty")?;
    if git_dirty != "true" && git_dirty != "false" {
        return Err(format!(
            "git_dirty must be true or false, got `{git_dirty}`"
        ));
    }

    let created_utc = require_manifest_field(&fields, "created_utc")?;
    if !created_utc.ends_with('Z') || !created_utc.contains('T') {
        return Err(format!(
            "created_utc must be an ISO-8601 UTC stamp: {created_utc}"
        ));
    }

    let proof_seconds = parse_manifest_u64(&fields, "proof_seconds")?;
    if proof_seconds == 0 {
        return Err(String::from("proof_seconds must be nonzero"));
    }

    check_manifest_artifact(&fields, parent, "serial_log", Some("serial.log"))?;
    match vm_target {
        "qemu" => {
            check_manifest_artifact(&fields, parent, "screen_ppm", Some("screen.ppm"))?;
            if fields.contains_key("screen_png_path") {
                check_manifest_artifact(&fields, parent, "screen_png", Some("screen.png"))?;
            }
        }
        "vbox" => {
            check_manifest_artifact(&fields, parent, "screenshot_png", Some("screenshot.png"))?;
            if fields.contains_key("screen_ppm_path") {
                return Err(String::from(
                    "VirtualBox manifest must not claim QEMU PPM proof-screen parity",
                ));
            }
        }
        _ => unreachable!(),
    }
    check_manifest_artifact(&fields, parent, "image", None)?;
    if fields.contains_key("efi_path") {
        check_manifest_artifact(&fields, parent, "efi", None)?;
    }

    Ok(())
}

fn parse_manifest_fields(text: &str) -> Result<BTreeMap<String, String>, String> {
    let mut fields = BTreeMap::new();
    for (idx, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("manifest line {} is not key=value: {line}", idx + 1))?;
        if key.is_empty()
            || !key
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        {
            return Err(format!("manifest line {} has invalid key `{key}`", idx + 1));
        }
        if value.trim().is_empty() {
            return Err(format!("manifest key `{key}` has an empty value"));
        }
        if fields
            .insert(key.to_string(), value.trim().to_string())
            .is_some()
        {
            return Err(format!("manifest key `{key}` is duplicated"));
        }
    }
    Ok(fields)
}

fn require_manifest_field<'a>(
    fields: &'a BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, String> {
    fields
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("proof manifest missing `{key}`"))
}

fn require_manifest_field_eq(
    fields: &BTreeMap<String, String>,
    key: &str,
    expected: &str,
) -> Result<(), String> {
    let actual = require_manifest_field(fields, key)?;
    if actual != expected {
        return Err(format!(
            "proof manifest `{key}` must be `{expected}`, got `{actual}`"
        ));
    }
    Ok(())
}

fn parse_manifest_u64(fields: &BTreeMap<String, String>, key: &str) -> Result<u64, String> {
    require_manifest_field(fields, key)?
        .parse::<u64>()
        .map_err(|e| format!("manifest `{key}` is not a u64: {e}"))
}

fn is_lower_hex(text: &str) -> bool {
    text.bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn check_manifest_artifact(
    fields: &BTreeMap<String, String>,
    manifest_parent: &Path,
    prefix: &str,
    bundled_name: Option<&str>,
) -> Result<(), String> {
    let path_key = format!("{prefix}_path");
    let bytes_key = format!("{prefix}_bytes");
    let crc_key = format!("{prefix}_crc32");
    let sha_key = format!("{prefix}_sha256");
    let manifest_path = PathBuf::from(require_manifest_field(fields, &path_key)?);
    let path = if let Some(name) = bundled_name {
        manifest_parent.join(name)
    } else if let Some(name) = manifest_path.file_name() {
        let bundled = manifest_parent.join(name);
        if bundled.exists() {
            bundled
        } else {
            manifest_path
        }
    } else {
        manifest_path
    };
    let expected_bytes = parse_manifest_u64(fields, &bytes_key)?;
    let expected_crc = require_manifest_field(fields, &crc_key)?;
    let expected_sha = require_manifest_field(fields, &sha_key)?;

    if expected_bytes == 0 {
        return Err(format!("manifest `{bytes_key}` must be nonzero"));
    }
    if expected_crc.len() != 8 || !is_lower_hex(expected_crc) {
        return Err(format!("manifest `{crc_key}` is not lowercase CRC32 hex"));
    }
    if expected_sha.len() != 64 || !is_lower_hex(expected_sha) {
        return Err(format!("manifest `{sha_key}` is not lowercase SHA-256 hex"));
    }

    let data = fs::read(&path).map_err(|e| format!("read {prefix} {}: {e}", path.display()))?;
    if data.is_empty() {
        return Err(format!("manifest artifact `{prefix}` is empty"));
    }
    if data.len() as u64 != expected_bytes {
        return Err(format!(
            "manifest `{bytes_key}` mismatch for {}: manifest={}, actual={}",
            path.display(),
            expected_bytes,
            data.len()
        ));
    }
    let actual_crc = format!("{:08x}", crc32(&data));
    if actual_crc != expected_crc {
        return Err(format!(
            "manifest `{crc_key}` mismatch for {}: manifest={}, actual={actual_crc}",
            path.display(),
            expected_crc
        ));
    }

    Ok(())
}

fn parse_metric(line: &str, key: &str) -> Result<u64, String> {
    let value = parse_field(line, key)?;
    value
        .parse::<u64>()
        .map_err(|e| format!("invalid metric `{key}{value}`: {e}"))
}

fn parse_field<'a>(line: &'a str, key: &str) -> Result<&'a str, String> {
    let value = line
        .split_whitespace()
        .find_map(|part| part.strip_prefix(key))
        .ok_or_else(|| format!("missing metric `{key}`"))?;
    Ok(value.trim_end_matches(','))
}

#[derive(Copy, Clone)]
enum VmProofTarget {
    Qemu,
    VirtualBox,
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALLOC_PROOF_LOG: &str = "[ALLOC] O(1) proof: topo_cells=8 l3_word_probes_per_cell=2 single_word_probes_per_cell=8192 contiguous_candidate_probes=128 contiguous_max_run_pages=64 toporam_max_run_pages=64 marking=bounded_by_contiguous_max_run_pages\n";
    const TOPORAM_ALLOC_LOG: &str = "[BENCH] toporam-alloc iterations=64 ok=64 p50_cycles=90 p95_cycles=130 max_cycles=150 target_cell_hits_delta=64 target_cell_fallbacks_delta=0 low_to_high_fallbacks_delta=0 high_to_low_fallbacks_delta=0 pcie_to_high_fallbacks_delta=0 pcie_to_low_fallbacks_delta=0 free_before=10 free_after=10\n";
    const SEAL_ALLOC_LOG: &str = "[BENCH] alloc-frame iterations=64 ok=64 p50_cycles=100 p95_cycles=140 max_cycles=160 fast_hits_delta=64 bounded_misses_delta=0 max_contiguous_probes_seen_delta=0 free_before=10 free_after=10\n";
    const MANIFOLD_TELEPORT_LOG: &str = "[BENCH] manifold-teleport api=teleport fs_mode=ramfs persistence=write_through samples=3 ok=3 same_inode=3 src_gone=3 dst_present=3 entries_min=8 entries_max=256 payload_bytes=64 p50_cycles=1000 p95_cycles=2000 max_cycles=2000 ticks_max=1 metadata_ops_max=7 write_through_bytes_per_move=64 payload_points=4\n";
    const SCHEDULER_SELECT_LOG: &str = "[BENCH] scheduler-select-next selector=select_next_task mode=live_requeue clock=rdtsc iterations=64 ok=64 ready_before=3 ready_after=3 cells=8 priority_buckets=256 voronoi_locate_probes=8 max_cell_bitmap_tests=9 max_priority_bucket_scan=256 context_switches=0 selected_priority_max=10 p50_cycles=80 p95_cycles=120 max_cycles=140\n";
    const AETHER_RUNTIME_LOG: &str = "[Aether-Lang] runtime proof: parser=ok interpreter=ok app_host=ok script=aether_boot_probe result=seal-topology-ok\n";
    const UBUNTU_ALLOC_LOG: &str = "[UBUNTU-BENCH] alloc-frame os=ubuntu version_id=26.04 kernel=6.14.0-native iterations=64 ok=64 bytes=4096 backend=rust-std-box-page-touch-drop clock=rdtsc p50_cycles=200 p95_cycles=300 max_cycles=400\n";

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("seal-mkimage-{label}-{nanos}"))
    }

    fn fake_sha256() -> &'static str {
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    }

    fn manifest_artifact(prefix: &str, path: &Path, data: &[u8]) -> String {
        format!(
            "{prefix}_path={}\n{prefix}_bytes={}\n{prefix}_crc32={:08x}\n{prefix}_sha256={}\n",
            path.display(),
            data.len(),
            crc32(data),
            fake_sha256()
        )
    }

    fn write_test_manifest(root: &Path) -> PathBuf {
        fs::create_dir_all(root).unwrap();
        let serial = b"serial proof log";
        let screen = b"P6\n1 1\n255\nrgb";
        let image = b"disk image bytes";
        let efi = b"efi image bytes";
        fs::write(root.join("serial.log"), serial).unwrap();
        fs::write(root.join("screen.ppm"), screen).unwrap();
        fs::write(root.join("seal-os.img"), image).unwrap();
        fs::write(root.join("seal-os.efi"), efi).unwrap();

        let mut manifest = String::from(
            "seal_proof_manifest_version=1\n\
vm_target=qemu\n\
qemu_backend=ci\n\
proof_verdict=PASS\n\
proof_seconds=300\n\
created_utc=2026-05-30T00:00:00Z\n\
git_commit=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\
git_dirty=false\n\
gate_verify=ok\n\
gate_vm_proof=ok\n\
gate_theorem_log=ok\n\
gate_aether_runtime=ok\n\
gate_desktop_soak=ok\n\
gate_benchmark_log=ok\n\
gate_proof_screen=ok\n",
        );
        manifest.push_str(&manifest_artifact(
            "image",
            &root.join("seal-os.img"),
            image,
        ));
        manifest.push_str(&manifest_artifact("efi", &root.join("seal-os.efi"), efi));
        manifest.push_str(&manifest_artifact(
            "serial_log",
            &root.join("serial.log"),
            serial,
        ));
        manifest.push_str(&manifest_artifact(
            "screen_ppm",
            &root.join("screen.ppm"),
            screen,
        ));

        let manifest_path = root.join("proof-manifest.txt");
        fs::write(&manifest_path, manifest).unwrap();
        manifest_path
    }

    fn write_test_vbox_manifest(root: &Path) -> PathBuf {
        fs::create_dir_all(root).unwrap();
        let serial = b"vbox serial proof log";
        let screenshot = b"fake png bytes";
        let image = b"disk image bytes";
        let efi = b"efi image bytes";
        fs::write(root.join("serial.log"), serial).unwrap();
        fs::write(root.join("screenshot.png"), screenshot).unwrap();
        fs::write(root.join("seal-os.img"), image).unwrap();
        fs::write(root.join("seal-os.efi"), efi).unwrap();

        let mut manifest = String::from(
            "seal_proof_manifest_version=1\n\
vm_target=vbox\n\
virtualbox_backend=headless\n\
proof_verdict=PASS\n\
proof_seconds=240\n\
created_utc=2026-05-30T00:00:00Z\n\
git_commit=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n\
git_dirty=false\n\
gate_verify=ok\n\
gate_vbox_proof=ok\n\
gate_theorem_log=ok\n\
gate_aether_runtime=ok\n\
gate_desktop_soak=ok\n\
gate_benchmark_log=ok\n",
        );
        manifest.push_str(&manifest_artifact(
            "image",
            &root.join("seal-os.img"),
            image,
        ));
        manifest.push_str(&manifest_artifact("efi", &root.join("seal-os.efi"), efi));
        manifest.push_str(&manifest_artifact(
            "serial_log",
            &root.join("serial.log"),
            serial,
        ));
        manifest.push_str(&manifest_artifact(
            "screenshot_png",
            &root.join("screenshot.png"),
            screenshot,
        ));

        let manifest_path = root.join("proof-manifest.txt");
        fs::write(&manifest_path, manifest).unwrap();
        manifest_path
    }

    #[test]
    fn proof_manifest_checks_bundle_artifacts() {
        let root = unique_temp_dir("manifest-ok");
        let manifest = write_test_manifest(&root);

        assert!(check_proof_manifest(&manifest).is_ok());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn proof_manifest_rejects_artifact_drift() {
        let root = unique_temp_dir("manifest-drift");
        let manifest = write_test_manifest(&root);
        fs::write(root.join("screen.ppm"), b"changed screen").unwrap();

        assert!(check_proof_manifest(&manifest).is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn proof_manifest_accepts_vbox_bundle_without_ppm_claim() {
        let root = unique_temp_dir("manifest-vbox");
        let manifest = write_test_vbox_manifest(&root);

        assert!(check_proof_manifest(&manifest).is_ok());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn proof_manifest_rejects_vbox_ppm_parity_claim() {
        let root = unique_temp_dir("manifest-vbox-ppm");
        let manifest = write_test_vbox_manifest(&root);
        fs::write(root.join("screen.ppm"), b"ppm").unwrap();
        let mut text = fs::read_to_string(&manifest).unwrap();
        text.push_str(&manifest_artifact(
            "screen_ppm",
            &root.join("screen.ppm"),
            b"ppm",
        ));
        fs::write(&manifest, text).unwrap();

        assert!(check_proof_manifest(&manifest).is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parses_seal_and_ubuntu_allocator_benchmarks() {
        assert!(parse_alloc_o1_proof(ALLOC_PROOF_LOG).is_ok());
        assert!(parse_toporam_alloc_benchmark(TOPORAM_ALLOC_LOG).is_ok());
        let seal = parse_seal_alloc_benchmark(SEAL_ALLOC_LOG).unwrap();
        let ubuntu = parse_ubuntu_alloc_benchmark(UBUNTU_ALLOC_LOG).unwrap();

        assert_eq!(seal.iterations, 64);
        assert_eq!(seal.p50, 100);
        assert_eq!(ubuntu.iterations, 64);
        assert_eq!(ubuntu.p95, 300);
    }

    #[test]
    fn allocator_o1_proof_requires_matching_toporam_run_cap() {
        let missing = ALLOC_PROOF_LOG.replace(" toporam_max_run_pages=64", "");
        assert!(parse_alloc_o1_proof(&missing).is_err());

        let mismatched =
            ALLOC_PROOF_LOG.replace("toporam_max_run_pages=64", "toporam_max_run_pages=65");
        assert!(parse_alloc_o1_proof(&mismatched).is_err());
    }

    #[test]
    fn rejects_toporam_target_cell_fallbacks() {
        let regressed = TOPORAM_ALLOC_LOG.replace(
            "target_cell_fallbacks_delta=0",
            "target_cell_fallbacks_delta=1",
        );

        assert!(parse_toporam_alloc_benchmark(&regressed).is_err());
    }

    #[test]
    fn parses_aether_runtime_boot_proof() {
        assert!(check_aether_runtime_text(AETHER_RUNTIME_LOG).is_ok());

        let missing_app_host = AETHER_RUNTIME_LOG.replace("app_host=ok", "app_host=missing");
        assert!(check_aether_runtime_text(&missing_app_host).is_err());
    }

    #[test]
    fn rejects_non_ubuntu_allocator_artifact() {
        let windows_log = UBUNTU_ALLOC_LOG.replace("os=ubuntu", "os=windows");

        assert!(parse_ubuntu_alloc_benchmark(&windows_log).is_err());
    }

    #[test]
    fn rejects_stale_ubuntu_allocator_artifact() {
        let stale_log = UBUNTU_ALLOC_LOG.replace("version_id=26.04", "version_id=24.04");

        assert!(parse_ubuntu_alloc_benchmark(&stale_log).is_err());
    }

    #[test]
    fn rejects_wsl_ubuntu_allocator_artifact() {
        let wsl_log = UBUNTU_ALLOC_LOG.replace(
            "kernel=6.14.0-native",
            "kernel=6.6.87.2-microsoft-standard-WSL2",
        );

        assert!(parse_ubuntu_alloc_benchmark(&wsl_log).is_err());
    }

    #[test]
    fn parses_manifold_teleport_benchmark() {
        assert!(parse_manifold_teleport_benchmark(MANIFOLD_TELEPORT_LOG).is_ok());

        let regressed = MANIFOLD_TELEPORT_LOG.replace("metadata_ops_max=7", "metadata_ops_max=9");
        assert!(parse_manifold_teleport_benchmark(&regressed).is_err());
    }

    #[test]
    fn parses_scheduler_select_benchmark() {
        assert!(parse_scheduler_select_benchmark(SCHEDULER_SELECT_LOG).is_ok());

        let regressed =
            SCHEDULER_SELECT_LOG.replace("max_cell_bitmap_tests=9", "max_cell_bitmap_tests=10");
        assert!(parse_scheduler_select_benchmark(&regressed).is_err());
    }

    #[test]
    fn doc_claim_contract_requires_ubuntu_and_benchmark_guards() {
        let readme = "\
not a blanket victory claim
Seal OS only claims a win over Ubuntu for a row after the same-machine benchmark exists
raw Ubuntu artifact pending
persistent write-through bytes still counted
Where Seal OS must still prove superiority
Minimal TLS 1.3 PSK record path
no X.509/PKI/ECDHE gate yet
remote registry fixture and signed package gate are pending
fixture gate pending before \"full filesystem parity\" claims
TopCrypt is topological encoding/obfuscation, not cryptographic protection
grid/value-height projection
`seal-mkimage --check-aether-runtime
`seal-mkimage --check-benchmark-log
";
        let benchmark = "\
That claim is not
global and not automatic
Ubuntu comparison numbers are still pending, so no global Ubuntu win is claimed
[UBUNTU-BENCH] alloc-frame os=ubuntu version_id=26.04 kernel=<native-kernel>
WSL is rejected for benchmark evidence
";
        let ci = "\
`seal-mkimage --check-aether-runtime /tmp/seal-os.log`
`seal-mkimage --check-benchmark-log /tmp/seal-os.log`
`seal-mkimage --compare-benchmark-logs /tmp/seal-os.log ubuntu-alloc.log`
";

        assert!(check_doc_claim_contract_text(readme, benchmark, ci).is_ok());

        let bad_readme = readme.replace("raw Ubuntu artifact pending", "Ubuntu artifact captured");
        assert!(check_doc_claim_contract_text(&bad_readme, benchmark, ci).is_err());
    }

    #[test]
    fn comparison_requires_seal_to_win_all_recorded_percentiles() {
        let seal = parse_seal_alloc_benchmark(SEAL_ALLOC_LOG).unwrap();
        let ubuntu = parse_ubuntu_alloc_benchmark(UBUNTU_ALLOC_LOG).unwrap();

        assert!(require_seal_alloc_bench_lead(seal, ubuntu).is_ok());

        let faster_ubuntu = AllocBenchMetrics { p95: 120, ..ubuntu };
        assert!(require_seal_alloc_bench_lead(seal, faster_ubuntu).is_err());
    }
}

impl VmProofTarget {
    fn disk_model(self) -> &'static str {
        match self {
            Self::Qemu => "[AHCI] Device model: QEMU HARDDISK",
            Self::VirtualBox => "[AHCI] Device model: VBOX HARDDISK",
        }
    }
}

fn check_vm_proof(log_path: &Path, target: VmProofTarget) -> Result<(), String> {
    check_theorem_log(log_path)?;
    let text =
        fs::read_to_string(log_path).map_err(|e| format!("read {}: {e}", log_path.display()))?;
    let required = [
        "[BOOT] GOP mode set to 1024x768",
        target.disk_model(),
        "[AHCI] Registered as block device 0x800",
        "[disk::ahci] First disk readable (sector 0 OK)",
        "[VFS] ManifoldFS mounted from disk",
    ];
    let banned = [
        "[VFS] No persistent disk found",
        "Falling back to ramfs",
        "[AHCI] No SATA device found",
        "[disk::ahci] No AHCI device present",
    ];

    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|pattern| !text.contains(pattern))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "missing VM proof patterns: {}",
            missing.join(" | ")
        ));
    }

    let contradictions: Vec<&str> = banned
        .iter()
        .copied()
        .filter(|pattern| text.contains(pattern))
        .collect();
    if !contradictions.is_empty() {
        return Err(format!(
            "contradictory VM proof patterns: {}",
            contradictions.join(" | ")
        ));
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
        let text =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        for (idx, line) in text.lines().enumerate() {
            let code = strip_allowed_comment(line);
            for &(needle, reason) in &banned {
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
        check_source_language_surface(root)
    } else {
        Err(findings.join("\n"))
    }
}

fn check_source_language_surface(root: &Path) -> Result<(), String> {
    let gitattributes = root.join(".gitattributes");
    let attrs = fs::read_to_string(&gitattributes)
        .map_err(|e| format!("read {}: {e}", gitattributes.display()))?;
    for required in [
        "* text=auto eol=lf",
        "*.bat text eol=crlf",
        "*.cmd text eol=crlf",
        "*.png binary",
        "*.ico binary",
        "*.icns binary",
        "*.wasm binary",
        "*.rs linguist-language=Rust",
        "*.s linguist-language=Assembly",
        "*.S linguist-language=Assembly",
        "*.asm linguist-language=Assembly",
        "*.py linguist-vendored",
        "*.pyw linguist-vendored",
        "*.aether linguist-language=Aether",
        "*.ag linguist-language=Aether",
        "*.aegis linguist-language=Aether",
        "*.lean linguist-documentation",
        "*.tex linguist-documentation",
        "*.bib linguist-documentation",
        "*.md linguist-documentation",
        "*.txt linguist-documentation",
        "LICENSE* linguist-documentation",
        "*.yml linguist-vendored",
        "*.yaml linguist-vendored",
        "*.toml linguist-vendored",
        "*.json linguist-vendored",
        "*.lock linguist-vendored",
        "*.sh linguist-vendored",
        "*.ps1 linguist-vendored",
        "*.bat linguist-vendored",
        "*.js linguist-vendored",
        "*.mjs linguist-vendored",
        "*.ts linguist-vendored",
        "*.tsx linguist-vendored",
        "*.css linguist-vendored",
        "*.html linguist-vendored",
        "*.svg linguist-vendored",
        "*.ico linguist-vendored",
        "*.png linguist-vendored",
        "*.icns linguist-vendored",
        "*.c linguist-vendored",
        "*.h linguist-vendored",
        "*.cpp linguist-vendored",
        "*.hpp linguist-vendored",
        "Dockerfile linguist-vendored",
        "**/Dockerfile linguist-vendored",
        ".dockerignore linguist-vendored",
        "**/.dockerignore linguist-vendored",
        ".gitignore linguist-vendored",
        "**/.gitignore linguist-vendored",
        ".gitattributes linguist-vendored",
        "**/.gitattributes linguist-vendored",
        "*.local linguist-vendored",
        "*.env linguist-vendored",
        "requirements*.txt linguist-vendored",
        "**/requirements*.txt linguist-vendored",
        ".githooks/* linguist-vendored",
        "Cargo.lock linguist-vendored",
        "package-lock.json linguist-vendored",
        "**/.lake/** linguist-generated",
        "*.olean linguist-generated",
        "*.olean.lock linguist-generated",
        "*.olean.trace linguist-generated",
        "*.wasm linguist-generated",
        "*.db linguist-generated",
        "*.sqlite* linguist-generated",
        "*.log linguist-generated",
        "*.profraw linguist-generated",
        "*.profdata linguist-generated",
        "*.db-shm linguist-generated",
        "*.db-wal linguist-generated",
        "*.trace linguist-generated",
        "*.dump linguist-generated",
        "*.dmp linguist-generated",
        "*.core linguist-generated",
        "*.ppm linguist-generated",
        "*.pcap linguist-generated",
        "*.pcapng linguist-generated",
        "perf.data* linguist-generated",
        "flamegraph*.svg linguist-generated",
        "*.dSYM/** linguist-generated",
        "**/lean-toolchain linguist-generated",
        "*.img linguist-generated",
        "*.iso linguist-generated",
        "*.vdi linguist-generated",
        "*.vmdk linguist-generated",
        "*.qcow2 linguist-generated",
        "*.vhd linguist-generated",
        "*.vhdx linguist-generated",
        "*.raw linguist-generated",
        "*.efi linguist-generated",
        "*.fd linguist-generated",
        "*.ovmf linguist-generated",
        "*.vbox linguist-generated",
        "*.vbox-prev linguist-generated",
        "*.nvram linguist-generated",
        "*.pdb linguist-generated",
        "*.map linguist-generated",
        "*.sym linguist-generated",
        "qemu-proof/** linguist-generated",
        "qemu-proof-runs/** linguist-generated",
        "vbox-smoke/** linguist-generated",
        "kernel/seal-os/target/**/qemu-proof/** linguist-generated",
        "kernel/seal-os/target/**/qemu-proof-runs/** linguist-generated",
        "kernel/seal-os/target/**/vbox-smoke/** linguist-generated",
    ] {
        if !attrs.lines().any(|line| line.trim() == required) {
            return Err(format!(
                "{} missing language quarantine rule `{required}`",
                gitattributes.display()
            ));
        }
    }

    check_artifact_ignore_rules(root)?;

    let quarantine_doc = root.join("docs").join("HOST_LANGUAGE_QUARANTINE.md");
    let quarantine_text = fs::read_to_string(&quarantine_doc)
        .map_err(|e| format!("read {}: {e}", quarantine_doc.display()))?;
    for needle in [
        "Legacy host script quarantine",
        "not part of the Seal OS runtime",
        "Rust/Aether replacement",
    ] {
        if !quarantine_text.contains(needle) {
            return Err(format!(
                "{} missing quarantine contract text `{needle}`",
                quarantine_doc.display()
            ));
        }
    }

    let production_roots = [
        root.join("kernel").join("seal-os"),
        root.join("kernel")
            .join("epsilon")
            .join("epsilon")
            .join("crates"),
        root.join("kernel")
            .join("aether")
            .join("Aether-Lang")
            .join("crates"),
    ];
    let mut python_hits = Vec::new();
    for production_root in production_roots {
        if !production_root.exists() {
            continue;
        }
        collect_files_with_extension(&production_root, "py", &mut python_hits)?;
        collect_files_with_extension(&production_root, "pyw", &mut python_hits)?;
    }
    if !python_hits.is_empty() {
        let hits = python_hits
            .iter()
            .map(|path| {
                path.strip_prefix(root)
                    .unwrap_or(path)
                    .display()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!(
            "production OS/Rust roots contain host-script files:\n{hits}"
        ));
    }

    Ok(())
}

fn check_artifact_ignore_rules(root: &Path) -> Result<(), String> {
    let gitignore = root.join(".gitignore");
    let ignore =
        fs::read_to_string(&gitignore).map_err(|e| format!("read {}: {e}", gitignore.display()))?;
    for required in [
        "target/",
        "**/target/",
        "*.img",
        "*.iso",
        "*.vdi",
        "*.efi",
        "*.fd",
        "*.ppm",
        "*.pcap",
        "*.pcapng",
        "*.trace",
        "*.dump",
        "*.core",
        "*.db",
        "*.sqlite*",
        "*.log",
        "*.profraw",
        "*.profdata",
        "*.dSYM/",
        "**/*.dSYM/",
        "**/lean-toolchain",
        "perf.data",
        "perf.data.*",
        "qemu-proof/",
        "qemu-proof-runs/",
        "vbox-smoke/",
        "kernel/seal-os/target/**/qemu-proof/",
        "kernel/seal-os/target/**/qemu-proof-runs/",
        "kernel/seal-os/target/**/vbox-smoke/",
        "ubuntu-alloc.log",
    ] {
        if !ignore.lines().any(|line| line.trim() == required) {
            return Err(format!(
                "{} missing artifact ignore rule `{required}`",
                gitignore.display()
            ));
        }
    }
    Ok(())
}

fn check_lean_proof_hygiene(root: &Path) -> Result<(), String> {
    let lean_root = root
        .join("kernel")
        .join("aether")
        .join("aether-verified")
        .join("lean");
    let mut files = Vec::new();
    collect_files_with_extension(&lean_root, "lean", &mut files)?;

    let mut findings = Vec::new();
    for path in files {
        if path_has_component(&path, ".lake") {
            continue;
        }
        let text =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let code = strip_lean_comments(&text);
        for keyword in ["sorry", "admit", "axiom"] {
            if contains_lean_keyword(&code, keyword) {
                findings.push(format!(
                    "{} contains forbidden Lean proof escape `{keyword}`",
                    path.display()
                ));
            }
        }

        let lines = text.lines().collect::<Vec<_>>();
        for (line_idx, line) in lines.iter().enumerate() {
            if !line.contains("True := trivial") {
                continue;
            }
            let start = line_idx.saturating_sub(8);
            let window = lines[start..=line_idx].join("\n").to_ascii_lowercase();
            let marked = ["placeholder", "skeleton", "deferred"]
                .iter()
                .any(|marker| window.contains(marker));
            if !marked {
                findings.push(format!(
                    "{}:{} uses `True := trivial` without a nearby placeholder/skeleton marker",
                    path.display(),
                    line_idx + 1
                ));
            }
        }
    }

    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn strip_lean_comments(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut idx = 0usize;
    let mut block_depth = 0usize;
    let mut line_comment = false;

    while idx < bytes.len() {
        if line_comment {
            if bytes[idx] == b'\n' {
                line_comment = false;
                out.push('\n');
            }
            idx += 1;
            continue;
        }

        if block_depth > 0 {
            if idx + 1 < bytes.len() && bytes[idx] == b'/' && bytes[idx + 1] == b'-' {
                block_depth += 1;
                idx += 2;
                continue;
            }
            if idx + 1 < bytes.len() && bytes[idx] == b'-' && bytes[idx + 1] == b'/' {
                block_depth -= 1;
                idx += 2;
                continue;
            }
            if bytes[idx] == b'\n' {
                out.push('\n');
            } else {
                out.push(' ');
            }
            idx += 1;
            continue;
        }

        if idx + 1 < bytes.len() && bytes[idx] == b'-' && bytes[idx + 1] == b'-' {
            line_comment = true;
            out.push(' ');
            out.push(' ');
            idx += 2;
            continue;
        }
        if idx + 1 < bytes.len() && bytes[idx] == b'/' && bytes[idx + 1] == b'-' {
            block_depth = 1;
            out.push(' ');
            out.push(' ');
            idx += 2;
            continue;
        }

        out.push(bytes[idx] as char);
        idx += 1;
    }

    out
}

fn contains_lean_keyword(code: &str, keyword: &str) -> bool {
    code.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .any(|token| token == keyword)
}

fn path_has_component(path: &Path, needle: &str) -> bool {
    path.components()
        .any(|component| component.as_os_str().to_string_lossy() == needle)
}

fn unsafe_inventory(root: &Path) -> Result<(), String> {
    let src = root.join("kernel").join("seal-os").join("src");
    let mut files = Vec::new();
    collect_rs_files(&src, &mut files)?;

    let mut total = 0usize;
    for path in files {
        let text =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let count = text.lines().filter(|line| is_unsafe_site(line)).count();
        if count > 0 {
            total += count;
            println!("[seal-audit] unsafe {count:>3} {}", path.display());
        }
    }
    println!("[seal-audit] unsafe total {total}");
    Ok(())
}

fn check_language_hygiene(root: &Path) -> Result<(), String> {
    let surfaces = [
        root.join("README.md"),
        root.join("kernel").join("seal-os").join("README.md"),
        root.join("docs"),
        root.join(".github").join("workflows"),
        root.join("scripts"),
    ];
    let mut files = Vec::new();
    for surface in &surfaces {
        collect_public_files(surface, &mut files)?;
    }

    let banned = [
        ("python", "host scripting language"),
        ("pytest", "host test runner"),
        ("compileall", "host bytecode check"),
        (".py", "host script extension"),
        ("node.js", "host runtime"),
        ("npm ", "host package runner"),
        ("yarn ", "host package runner"),
    ];

    let mut findings = Vec::new();
    for path in files {
        let rel = path.strip_prefix(root).unwrap_or(&path);
        if language_hygiene_exempt(rel) {
            continue;
        }
        let text =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        for (idx, line) in text.lines().enumerate() {
            let lower = line.to_ascii_lowercase();
            for &(needle, reason) in &banned {
                if lower.contains(needle) && !language_line_allowed(&lower) {
                    findings.push(format!(
                        "{}:{}: {reason}: {}",
                        rel.display(),
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

fn check_doc_claim_contract(root: &Path) -> Result<(), String> {
    let readme_path = root.join("README.md");
    let benchmark_path = root.join("docs").join("BENCHMARK_PLAN.md");
    let ci_path = root.join("docs").join("CI.md");
    let readme = fs::read_to_string(&readme_path)
        .map_err(|e| format!("read {}: {e}", readme_path.display()))?;
    let benchmark = fs::read_to_string(&benchmark_path)
        .map_err(|e| format!("read {}: {e}", benchmark_path.display()))?;
    let ci =
        fs::read_to_string(&ci_path).map_err(|e| format!("read {}: {e}", ci_path.display()))?;
    check_doc_claim_contract_text(&readme, &benchmark, &ci)
}

fn check_doc_claim_contract_text(readme: &str, benchmark: &str, ci: &str) -> Result<(), String> {
    let required = [
        (
            "README.md",
            readme,
            "not a blanket victory claim",
            "README must deny global Ubuntu victory until benchmark artifacts exist",
        ),
        (
            "README.md",
            readme,
            "Seal OS only claims a win over Ubuntu for a row after the same-machine benchmark exists",
            "README must bind every Ubuntu win to same-machine benchmark evidence",
        ),
        (
            "README.md",
            readme,
            "raw Ubuntu artifact pending",
            "README allocator rows must expose the missing Ubuntu artifact",
        ),
        (
            "README.md",
            readme,
            "persistent write-through bytes still counted",
            "README must not claim persistent ManifoldFS moves are byte-free",
        ),
        (
            "README.md",
            readme,
            "Where Seal OS must still prove superiority",
            "README must preserve the superiority gap statement",
        ),
        (
            "README.md",
            readme,
            "Minimal TLS 1.3 PSK record path",
            "README must scope TLS claims to the implemented PSK-only path",
        ),
        (
            "README.md",
            readme,
            "no X.509/PKI/ECDHE gate yet",
            "README must expose missing production TLS gates",
        ),
        (
            "README.md",
            readme,
            "remote registry fixture and signed package gate are pending",
            "README must expose missing ManifoldPkg remote proof",
        ),
        (
            "README.md",
            readme,
            "fixture gate pending before \"full filesystem parity\" claims",
            "README must expose missing filesystem fixture proof",
        ),
        (
            "README.md",
            readme,
            "TopCrypt is topological encoding/obfuscation, not cryptographic protection",
            "README must not market TopCrypt as encryption without AEAD/KDF proof",
        ),
        (
            "README.md",
            readme,
            "grid/value-height projection",
            "README must match the implemented tensor renderer instead of claiming SVD",
        ),
        (
            "README.md",
            readme,
            "`seal-mkimage --check-aether-runtime",
            "README must point Aether runtime claims at the audit gate",
        ),
        (
            "README.md",
            readme,
            "`seal-mkimage --check-benchmark-log",
            "README must point benchmark claims at the audit gate",
        ),
        (
            "docs/BENCHMARK_PLAN.md",
            benchmark,
            "That claim is not\nglobal and not automatic",
            "benchmark plan must reject broad Ubuntu wins",
        ),
        (
            "docs/BENCHMARK_PLAN.md",
            benchmark,
            "Ubuntu comparison numbers are still pending, so no global Ubuntu win is claimed",
            "benchmark plan must keep current comparison state honest",
        ),
        (
            "docs/BENCHMARK_PLAN.md",
            benchmark,
            "[UBUNTU-BENCH] alloc-frame os=ubuntu version_id=26.04 kernel=<native-kernel>",
            "benchmark plan must specify the accepted Ubuntu artifact marker",
        ),
        (
            "docs/BENCHMARK_PLAN.md",
            benchmark,
            "WSL is rejected for benchmark evidence",
            "benchmark plan must reject WSL artifacts",
        ),
        (
            "docs/CI.md",
            ci,
            "`seal-mkimage --check-aether-runtime /tmp/seal-os.log`",
            "CI docs must include the Aether runtime gate",
        ),
        (
            "docs/CI.md",
            ci,
            "`seal-mkimage --check-benchmark-log /tmp/seal-os.log`",
            "CI docs must include the runtime benchmark gate",
        ),
        (
            "docs/CI.md",
            ci,
            "`seal-mkimage --compare-benchmark-logs /tmp/seal-os.log ubuntu-alloc.log`",
            "CI docs must include the Seal-vs-Ubuntu comparison gate",
        ),
    ];

    let missing: Vec<String> = required
        .iter()
        .filter(|(_, text, needle, _)| !text.contains(needle))
        .map(|(file, _, needle, reason)| format!("{file}: missing `{needle}` ({reason})"))
        .collect();
    if !missing.is_empty() {
        return Err(missing.join("\n"));
    }

    let banned_readme_claims = [
        (
            "Seal OS is faster than Ubuntu",
            "global speed claim needs workload and evidence",
        ),
        (
            "Seal OS beats Ubuntu",
            "global Ubuntu win claim needs workload and evidence",
        ),
        (
            "proper TLS 1.3 PSK handshake",
            "TLS implementation is PSK-only and not a full production handshake",
        ),
        (
            "projected via SVD",
            "tensor renderer currently uses grid/value-height projection",
        ),
        (
            "indistinguishable from random noise",
            "TopCrypt is not proven cryptographic protection",
        ),
        (
            "Only Seal OS can wake locked files",
            "TopCrypt lock is obfuscation without AEAD/KDF proof",
        ),
        (
            "O(1) allocation for everything",
            "allocator proof is scoped to gated paths",
        ),
        (
            "persistent ManifoldFS moves are metadata-only",
            "current persistent path still counts write-through bytes",
        ),
    ];
    let banned_hits: Vec<String> = banned_readme_claims
        .iter()
        .filter(|(needle, _)| readme.contains(needle))
        .map(|(needle, reason)| format!("README.md: banned `{needle}` ({reason})"))
        .collect();
    if !banned_hits.is_empty() {
        return Err(banned_hits.join("\n"));
    }

    Ok(())
}

fn check_aether_migration(root: &Path) -> Result<(), String> {
    let rust_root = root
        .join("kernel")
        .join("epsilon")
        .join("epsilon")
        .join("crates")
        .join("aether-core")
        .join("src");
    let required_files = [
        ("governor.rs", "pub struct GovernorConvergenceAnalyzer"),
        ("governor.rs", "pub struct GovernorSimulationHistory"),
        ("governor.rs", "pub fn simulate_measurements"),
        ("governor.rs", "pub fn simulate_default_history"),
        (
            "cross_manifold_alignment.rs",
            "pub fn alignment_error_bound",
        ),
        (
            "cross_manifold_alignment.rs",
            "pub fn mutual_information_bound",
        ),
        (
            "cross_manifold_alignment.rs",
            "pub fn transitive_error_bound",
        ),
        (
            "cross_manifold_alignment.rs",
            "pub struct CrossManifoldAligner",
        ),
        ("cross_manifold_alignment.rs", "pub struct CmaVerification"),
        ("geodesic_consolidation.rs", "pub fn cluster_entropy"),
        (
            "geodesic_consolidation.rs",
            "pub fn entropy_change_on_merge",
        ),
        ("geodesic_consolidation.rs", "pub fn great_circle_distance"),
        (
            "geodesic_consolidation.rs",
            "pub struct GeodesicConsolidator",
        ),
        (
            "geodesic_consolidation.rs",
            "pub struct ConsolidationReport",
        ),
        (
            "hyperbolic_capacity.rs",
            "pub fn hyperbolic_distortion_bound",
        ),
        (
            "hyperbolic_capacity.rs",
            "pub fn euclidean_distortion_bound",
        ),
        ("hyperbolic_capacity.rs", "pub fn separation_ratio"),
        ("hyperbolic_capacity.rs", "pub fn h100_analysis"),
        ("hyperbolic_capacity.rs", "pub struct HcsVerifier"),
        ("hyperbolic_capacity.rs", "pub struct HcsVerification"),
        ("hyperbolic_geometry.rs", "pub struct PoincareBall"),
        (
            "hyperbolic_geometry.rs",
            "pub struct AngularMomentumTracker",
        ),
        ("hyperbolic_geometry.rs", "pub fn verify_plasticity_bound"),
        ("meta_controller.rs", "pub struct MetaController"),
        (
            "meta_controller.rs",
            "pub struct ConstitutionalSafetyFilter",
        ),
        ("meta_controller.rs", "pub const MAX_TOOL_COUNT"),
        ("meta_controller.rs", "pub struct ToolSpec"),
        ("meta_controller.rs", "pub struct ToolRegistry"),
        ("meta_controller.rs", "pub struct ExecutionPayloadMetadata"),
        ("meta_controller.rs", "pub struct ToolDecision"),
        ("meta_controller.rs", "pub fn decide_with_tools"),
        ("meta_controller.rs", "pub fn decide_from_state_with_tools"),
        ("meta_controller.rs", "NoCompliantTool"),
        ("meta_controller.rs", "ToolRegistryTooLarge"),
        ("meta_controller.rs", "pub fn check_compliance_text"),
        ("meta_controller.rs", "pub const fn entropy_threshold"),
        ("meta_controller.rs", "entropy < self.entropy_threshold"),
        ("parallel_riemannian.rs", "pub fn stiefel_project_tangent"),
        ("parallel_riemannian.rs", "pub fn tangent_space_deviation"),
        ("parallel_riemannian.rs", "pub fn coherence_bound"),
        ("parallel_riemannian.rs", "pub fn sync_frequency"),
        ("parallel_riemannian.rs", "pub fn nvlink_sync_cost_seconds"),
        (
            "parallel_riemannian.rs",
            "pub struct DistributedRiemannianSgd",
        ),
        ("parallel_riemannian.rs", "pub struct RgcsVerification"),
        ("persistent_kv_partition.rs", "pub enum MemoryTier"),
        ("persistent_kv_partition.rs", "pub fn kv_cache_size_gb"),
        ("persistent_kv_partition.rs", "pub fn topological_locality"),
        (
            "persistent_kv_partition.rs",
            "pub struct BettiGuidedPartitioner",
        ),
        ("persistent_kv_partition.rs", "pub struct PartitionReport"),
        ("persistent_kv_partition.rs", "pub fn partition"),
        ("persistent_kv_partition.rs", "pub fn sparse_latency"),
        ("scm.rs", "pub struct TelemetryOperator"),
        ("scm.rs", "pub struct LatentPredictor"),
        ("scm.rs", "pub struct SpectralContractionVerifier"),
        ("scm.rs", "pub fn verify_contraction"),
        ("scm.rs", "pub fn verify_convergence"),
        ("scm.rs", "pub fn full_verification"),
        ("spectral_entropy.rs", "pub struct HaarWaveletTransform"),
        ("spectral_entropy.rs", "pub fn haar_wavelet_transform"),
        ("spectral_entropy.rs", "pub struct SpectralCoherenceTracker"),
        (
            "thermodynamic_plasticity.rs",
            "pub fn landauer_energy_per_bit",
        ),
        (
            "thermodynamic_plasticity.rs",
            "pub fn min_energy_per_update",
        ),
        (
            "thermodynamic_plasticity.rs",
            "pub fn max_sustainable_hot_ratio",
        ),
        ("thermodynamic_plasticity.rs", "pub fn gibbs_entropy"),
        ("thermodynamic_plasticity.rs", "pub fn thermodynamic_lr"),
        (
            "thermodynamic_plasticity.rs",
            "pub fn helmholtz_free_energy_change",
        ),
        (
            "thermodynamic_plasticity.rs",
            "pub struct ThermodynamicAnalyzer",
        ),
        ("world_model_horizon.rs", "pub fn predictive_horizon"),
        ("world_model_horizon.rs", "pub fn flat_horizon"),
        ("world_model_horizon.rs", "pub fn topological_advantage"),
        ("world_model_horizon.rs", "pub fn multi_model_horizon"),
        ("world_model_horizon.rs", "pub struct WorldModelAnalyzer"),
        ("world_model_horizon.rs", "pub struct HorizonVerification"),
        ("tss.rs", "pub struct SphericalGridHashIndex"),
        ("tss.rs", "pub const EMPTY_LOCATE"),
        ("tss.rs", "pub fn auto_sized_dimensions"),
        ("tss.rs", "pub fn build"),
        ("tss.rs", "pub fn locate_legacy"),
        ("tss.rs", "pub struct TssVerifier"),
        ("tss.rs", "pub fn verify_separation"),
        ("tss.rs", "pub fn epsilon_from_chebyshev"),
        ("tss.rs", "pub fn tss_retrieval_bound"),
        ("tss.rs", "pub fn insert_centroid"),
        ("tss.rs", "pub fn locate_with_payload"),
        ("governor.rs", "pub struct GovernorTheoremVerification"),
        ("governor.rs", "pub fn verify_theorem"),
    ];

    let mut findings = Vec::new();
    for (file, symbol) in required_files {
        let path = rust_root.join(file);
        let text =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        if !text.contains(symbol) {
            findings.push(format!("{} missing `{symbol}`", path.display()));
        }
    }

    let legacy_root = root.join("kernel").join("epsilon").join("epsilon_core");
    let legacy_migrations = [
        ("cross_manifold_alignment.py", "cross_manifold_alignment.rs"),
        ("geodesic_consolidation.py", "geodesic_consolidation.rs"),
        ("governor_convergence.py", "governor.rs"),
        ("hyperbolic_capacity.py", "hyperbolic_capacity.rs"),
        ("hyperbolic_geometry.py", "hyperbolic_geometry.rs"),
        ("meta_controller.py", "meta_controller.rs"),
        ("parallel_riemannian.py", "parallel_riemannian.rs"),
        ("persistent_kv_partition.py", "persistent_kv_partition.rs"),
        ("spectral_contraction.py", "scm.rs"),
        ("spectral_entropy.py", "spectral_entropy.rs"),
        ("thermodynamic_plasticity.py", "thermodynamic_plasticity.rs"),
        ("topological_state_sync.py", "tss.rs"),
        ("world_model_horizon.py", "world_model_horizon.rs"),
    ];
    for (legacy_file, rust_file) in legacy_migrations {
        let legacy_path = legacy_root.join(legacy_file);
        if legacy_path.exists() {
            findings.push(format!(
                "{} still exists; migrated theorem code must live in {}",
                legacy_path.display(),
                rust_root.join(rust_file).display()
            ));
        }
    }

    let banned_imports = [
        "cross_manifold_alignment",
        "geodesic_consolidation",
        "governor_convergence",
        "hyperbolic_capacity",
        "hyperbolic_geometry",
        "meta_controller",
        "parallel_riemannian",
        "persistent_kv_partition",
        "spectral_contraction",
        "spectral_entropy",
        "thermodynamic_plasticity",
        "topological_state_sync",
        "world_model_horizon",
    ];
    let tests_root = root.join("tests");
    for import_root in [&legacy_root, &tests_root] {
        if !import_root.exists() {
            continue;
        }
        let mut legacy_files = Vec::new();
        collect_files_with_extension(import_root, "py", &mut legacy_files)?;
        for path in legacy_files {
            let text =
                fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            for (idx, line) in text.lines().enumerate() {
                let trimmed = line.trim_start();
                let is_import = trimmed.starts_with("from ") || trimmed.starts_with("import ");
                if is_import {
                    for banned in banned_imports {
                        if trimmed.contains(banned) {
                            findings.push(format!(
                                "{}:{} imports deleted legacy module `{banned}`",
                                path.display(),
                                idx + 1
                            ));
                        }
                    }
                }
            }
        }
    }

    let gate = root
        .join("kernel")
        .join("epsilon")
        .join("epsilon")
        .join("crates")
        .join("aether-core")
        .join("tests")
        .join("legacy_migration_gate.rs");
    let gate_text =
        fs::read_to_string(&gate).map_err(|e| format!("read {}: {e}", gate.display()))?;
    for needle in [
        "ConstitutionalSafetyFilter",
        "SphericalGridHashIndex",
        "TelemetryOperator",
        "SpectralContractionVerifier",
        "LatentPredictor",
        "GovernorConvergenceAnalyzer",
        "GovernorSimulationHistory",
        "simulate_measurements",
        "simulate_default_history",
        "GovernorTheoremVerification",
        "alignment_error_bound",
        "mutual_information_bound",
        "transitive_error_bound",
        "CrossManifoldAligner",
        "cluster_entropy",
        "entropy_change_on_merge",
        "great_circle_distance",
        "GeodesicConsolidator",
        "hyperbolic_distortion_bound",
        "euclidean_distortion_bound",
        "separation_ratio",
        "h100_analysis",
        "HcsVerifier",
        "ToolSpec",
        "ToolRegistry",
        "ToolDecision",
        "decide_with_tools",
        "NoCompliantTool",
        "stiefel_project_tangent",
        "tangent_space_deviation",
        "coherence_bound",
        "sync_frequency",
        "nvlink_sync_cost_seconds",
        "DistributedRiemannianSgd",
        "kv_cache_size_gb",
        "topological_locality",
        "BettiGuidedPartitioner",
        "MemoryTier",
        "TssVerifier",
        "epsilon_from_chebyshev",
        "tss_retrieval_bound",
        "EMPTY_LOCATE",
        "auto_sized_dimensions",
        "build",
        "locate_legacy",
        "SpectralCoherenceTracker",
        "HaarWaveletTransform",
        "haar_wavelet_transform",
        "spectral_entropy",
        "wavelet_entropy_per_scale",
        "predict_betti_survival",
        "landauer_energy_per_bit",
        "min_energy_per_update",
        "max_sustainable_hot_ratio",
        "gibbs_entropy",
        "thermodynamic_lr",
        "helmholtz_free_energy_change",
        "ThermodynamicAnalyzer",
        "predictive_horizon",
        "flat_horizon",
        "topological_advantage",
        "multi_model_horizon",
        "WorldModelAnalyzer",
        "PoincareBall",
        "AngularMomentumTracker",
        "verify_plasticity_bound",
    ] {
        if !gate_text.contains(needle) {
            findings.push(format!(
                "{} missing migration assertion for `{needle}`",
                gate.display()
            ));
        }
    }

    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn check_runtime_theorems(root: &Path) -> Result<(), String> {
    let seal = root.join("kernel").join("seal-os").join("src");
    let checks = [
        (
            "lib.rs",
            &[
                "pub static THEOREM_STATES",
                "THEOREM_STATES[idx].store",
                "[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths",
            ][..],
        ),
        (
            "memory/topo_ram.rs",
            &[
                "use crate::THEOREM_STATES",
                "fn reseed_voronoi",
                "THEOREM_STATES[0].load",
                "fn update_spectral",
                "THEOREM_STATES[1].load",
                "fn fragmentation_entropy_inner",
                "THEOREM_STATES[2].load",
                "fn update_pressure",
                "THEOREM_STATES[3].load",
                "fn hyperbolic_class",
                "THEOREM_STATES[4].load",
            ][..],
        ),
        (
            "process/scheduler.rs",
            &[
                "use aether_core::tss::SphericalVoronoiIndex",
                "use aether_core::scm::SpectralContractionOperator",
                "use aether_core::governor::GeometricGovernor",
                "predictor: SpectralContractionOperator<8>",
                "fn select_next_task",
                "self.voronoi.locate",
                ".apply(&self.predict_state, &next_task.manifold_embedding)",
                "self.governor.adapt",
                "process_tree",
                "hyperbolic process tree",
            ][..],
        ),
        (
            "fs/manifold_fs.rs",
            &[
                "use super::voronoi_cap::VoronoiCap",
                "use aether_core::scm::SpectralContractionOperator",
                "use aether_core::governor::GeometricGovernor",
                "self.voronoi.locate",
                "fn update_prefetch_state",
                "self.scm.apply",
                "self.check_entropy_and_merge",
                "self.governor.adapt",
                "fn update_hyperbolic_ratio",
                "theorem_status_text",
                "T1/TSS",
                "T3/GMC",
                "T5/HCS",
            ][..],
        ),
        (
            "wm/compositor.rs",
            &[
                "use aether_core::tss::SphericalVoronoiIndex",
                "use aether_core::governor::GeometricGovernor",
                "fn screen_point_cell",
                "self.voronoi.locate",
                "self.governor.adapt",
                "T1: screen-space Voronoi chooses",
                "T4: Adaptive FPS",
            ][..],
        ),
        (
            "drivers/acpi/topological_power.rs",
            &[
                "static ZONES",
                "pub fn thermal_governor_step",
                "fn betti_0",
                "fn update_hyperbolic_pstate",
                "pstate_from_epsilon",
                "zones[i].temp_c = temps[i]",
                "zones[i].predict",
                "apply_frequency_policy",
                "T1 + T2: Update zones",
                "T3: Betti-0",
                "T4: PD thermal governor",
                "T5: Hyperbolic P-state selection",
            ][..],
        ),
        (
            "wm/taskbar.rs",
            &[
                "use crate::{GOVERNOR_EPSILON, THEOREM_COUNT, THEOREM_STATES}",
                "THEOREM_STATES[i].load",
                "GOVERNOR_EPSILON.load",
            ][..],
        ),
    ];

    let mut findings = Vec::new();
    for (rel, needles) in checks {
        let path = seal.join(rel);
        let text =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        for needle in needles {
            if !text.contains(needle) {
                findings.push(format!("{} missing `{needle}`", path.display()));
            }
        }
    }

    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    collect_files_with_extension(dir, "rs", out)
}

fn collect_files_with_extension(
    dir: &Path,
    extension: &str,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir entry {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_files_with_extension(&path, extension, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some(extension) {
            out.push(path);
        }
    }
    Ok(())
}

fn check_o1_allocator(root: &Path) -> Result<(), String> {
    let phys = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("memory")
        .join("phys.rs");
    let text = fs::read_to_string(&phys).map_err(|e| format!("read {}: {e}", phys.display()))?;
    let topo = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("memory")
        .join("topo_ram.rs");
    let topo_text =
        fs::read_to_string(&topo).map_err(|e| format!("read {}: {e}", topo.display()))?;
    let required = [
        "pub struct AllocatorO1Proof",
        "pub const fn allocation_o1_proof",
        "pub fn topo_alloc_stats",
        "const MAX_ALLOC_L3_WORD_PROBES",
        "const SINGLE_ALLOC_WORD_CANDIDATE_PROBES",
        "const MAX_SINGLE_ALLOC_WORD_PROBES_PER_CELL",
        "const CONTIGUOUS_CANDIDATE_PROBES",
        "pub const MAX_CONTIGUOUS_RUN_PAGES",
        "SUMMARY_L3_U64S <= 2",
        "MAX_ALLOC_TOPO_CELL_PROBES == 8",
        "MAX_SINGLE_ALLOC_WORD_PROBES_PER_CELL <= 8192",
        "fn child_range_mask",
        "pub(crate) fn with_bitmap_read",
        "fn find_indexed_free_frame_from_cell",
        "fn contiguous_run_is_free",
        "pub fn alloc_frame_in_range",
        "pub fn alloc_frames_contiguous_in_range",
    ];
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|needle| !text.contains(needle))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "missing allocator proof hooks: {}",
            missing.join(" | ")
        ));
    }
    reject_patterns(
        "phys allocator module",
        &text,
        &["pub fn with_bitmap_mut", "FnOnce(&mut [u64])"],
    )?;

    let alloc_frame_body = extract_between(
        &text,
        "pub fn alloc_frame()",
        "/// Allocate a single 4 KiB physical frame from high memory",
    )?;
    reject_patterns(
        "alloc_frame",
        alloc_frame_body,
        &[
            "count_free_bits",
            "bitmap.iter()",
            "bitmap.iter_mut()",
            "for word in bitmap",
            "for word_idx in 0..BITMAP_U64S",
        ],
    )?;

    let single_body = extract_between(
        &text,
        "fn alloc_indexed_frame_from_cell",
        "fn find_indexed_free_frame_from_cell",
    )?;
    reject_patterns(
        "alloc_indexed_frame_from_cell",
        single_body,
        &[
            "bitmap.iter()",
            "bitmap.iter_mut()",
            "for word_idx in 0..BITMAP_U64S",
        ],
    )?;
    if !single_body.contains("SINGLE_ALLOC_WORD_CANDIDATE_PROBES")
        || !single_body.contains("word_probes >= SINGLE_ALLOC_WORD_CANDIDATE_PROBES")
        || !single_body.contains("child_range_mask")
    {
        return Err(String::from(
            "single-frame allocator summary walk lacks an explicit bitmap-word probe budget",
        ));
    }

    let find_body = extract_between(
        &text,
        "fn find_indexed_free_frame_from_cell",
        "fn contiguous_run_is_free",
    )?;
    reject_patterns(
        "find_indexed_free_frame_from_cell",
        find_body,
        &[
            "bitmap.iter()",
            "bitmap.iter_mut()",
            "for word_idx in 0..BITMAP_U64S",
        ],
    )?;
    if !find_body.contains("SINGLE_ALLOC_WORD_CANDIDATE_PROBES")
        || !find_body.contains("word_probes >= SINGLE_ALLOC_WORD_CANDIDATE_PROBES")
        || !find_body.contains("child_range_mask")
    {
        return Err(String::from(
            "contiguous candidate discovery lacks an explicit bitmap-word probe budget",
        ));
    }

    let contiguous_body = extract_between(
        &text,
        "pub fn alloc_frames_contiguous_in_range",
        "/// Allocate `count` contiguous 4 KiB frames.",
    )?;
    reject_patterns(
        "alloc_frames_contiguous_in_range",
        contiguous_body,
        &[
            "let first_word = min_frame / 64",
            "let last_word = (limit - 1) / 64",
            "for bit in 0..64",
            "for word_idx in 0..BITMAP_U64S",
            "bitmap.iter_mut().enumerate()",
            "count_free_bits",
        ],
    )?;

    if !contiguous_body.contains("CONTIGUOUS_CANDIDATE_PROBES")
        || !contiguous_body.contains("count > MAX_CONTIGUOUS_RUN_PAGES")
        || !contiguous_body.contains("find_indexed_free_frame_from_cell")
        || !contiguous_body.contains("contiguous_run_is_free")
    {
        return Err(String::from(
            "contiguous allocator is not using bounded topological candidate probes and bounded run size",
        ));
    }

    let topo_required = [
        "pub const MAX_TOPO_RAM_RUN_PAGES",
        "pub const fn max_run_pages",
        "const ENTROPY_RECHECK_INTERVAL",
        "const FRAGMENTATION_SAMPLE_LIMIT",
        "const RESEED_CELL_REFRESH_LIMIT",
        "const PREFETCH_SCAN_LIMIT",
        "const GLOBAL_ENTROPY_SAMPLE_LIMIT",
        "const SWAP_SCAN_LIMIT",
    ];
    let topo_missing: Vec<&str> = topo_required
        .iter()
        .copied()
        .filter(|needle| !topo_text.contains(needle))
        .collect();
    if !topo_missing.is_empty() {
        return Err(format!(
            "missing topo-ram O(1) proof hooks: {}",
            topo_missing.join(" | ")
        ));
    }

    let setup_claim_markers = [
        (
            "phys allocator module",
            &text,
            "SETUP_SCAN_OK: O(total bitmap words) boot/setup rebuild",
        ),
        (
            "phys allocator module",
            &text,
            "SETUP_SCAN_OK: O(memory-map pages) boot discovery",
        ),
        (
            "phys allocator module",
            &text,
            "SETUP_SCAN_OK: O(memory-map pages) high-memory discovery",
        ),
        (
            "phys allocator module",
            &text,
            "RUNTIME_BOUNDED_BY_REQUEST: caller rejects count > MAX_CONTIGUOUS_RUN_PAGES",
        ),
        (
            "topo_ram module",
            &topo_text,
            "SETUP_SCAN_OK: O(zone frame count) metadata construction",
        ),
        (
            "topo_ram module",
            &topo_text,
            "SETUP_SCAN_OK: zone metadata is built once",
        ),
        (
            "topo_ram module",
            &topo_text,
            "SETUP_SCAN_OK: O(registered BAR pages) device-registration metadata",
        ),
        (
            "topo_ram module",
            &topo_text,
            "RUNTIME_BOUNDED_BY_REQUEST: TopoRAM rejects count >",
        ),
        (
            "topo_ram module",
            &topo_text,
            "RUNTIME_BOUNDED_BY_REQUEST: public TopoRAM allocation uses the same",
        ),
        (
            "topo_ram module",
            &topo_text,
            "RUNTIME_BOUNDED_BY_REQUEST: free-side topology repair is bounded",
        ),
    ];
    for (name, module_text, marker) in setup_claim_markers {
        if !module_text.contains(marker) {
            return Err(format!(
                "{name} must mark setup/request-size scans explicitly: missing `{marker}`"
            ));
        }
    }

    let topo_alloc_body = extract_between(
        &topo_text,
        "fn alloc_frames_in_zone",
        "// ---------------------------------------------------------------------------\n// Public API",
    )?;
    reject_patterns(
        "topo_ram::alloc_frames_in_zone",
        topo_alloc_body,
        &[
            "for local_idx in 0..topo.frame_meta.len()",
            "for i in zone_start..zone_end",
            "with_bitmap_mut",
        ],
    )?;
    reject_patterns("topo_ram module", &topo_text, &["with_bitmap_mut"])?;
    if !topo_alloc_body.contains("ENTROPY_RECHECK_INTERVAL") {
        return Err(String::from(
            "topo allocation must gate entropy checks behind a bounded interval",
        ));
    }
    if !topo_alloc_body.contains("count > MAX_TOPO_RAM_RUN_PAGES") {
        return Err(String::from(
            "topo allocation must reject requests above MAX_TOPO_RAM_RUN_PAGES",
        ));
    }

    let topo_public_alloc_body = extract_between(
        &topo_text,
        "pub fn alloc_frames(count: usize",
        "/// Free `count` frames starting at `addr`.",
    )?;
    if !topo_public_alloc_body.contains("count > MAX_TOPO_RAM_RUN_PAGES") {
        return Err(String::from(
            "public TopoRAM allocation must reject unbounded run sizes",
        ));
    }

    let topo_free_body = extract_between(
        &topo_text,
        "pub fn free_frames(addr: PhysAddr, count: usize)",
        "/// Record an access tick for `addr`.",
    )?;
    if !topo_free_body.contains("count > MAX_TOPO_RAM_RUN_PAGES") {
        return Err(String::from(
            "TopoRAM free-side metadata repair must reject unbounded run sizes",
        ));
    }

    let prefetch_body = extract_between(
        &topo_text,
        "fn prefetch_hint_inner",
        "// ---------------------------------------------------------------------------\n// T3",
    )?;
    reject_patterns(
        "prefetch_hint_inner",
        prefetch_body,
        &[
            "for local_idx in 0..topo.frame_meta.len()",
            "for local_idx in 0..span",
        ],
    )?;
    if !prefetch_body.contains("PREFETCH_SCAN_LIMIT") || !prefetch_body.contains("RECENT_ALLOC_LEN")
    {
        return Err(String::from(
            "topological prefetch must use bounded samples and recent allocations",
        ));
    }

    let entropy_body = extract_between(
        &topo_text,
        "fn fragmentation_entropy_inner",
        "/// Global fragmentation entropy",
    )?;
    reject_patterns(
        "fragmentation_entropy_inner",
        entropy_body,
        &[
            "for i in zone_start..zone_end",
            "for local_idx in 0..topo.frame_meta.len()",
        ],
    )?;
    if !entropy_body.contains("FRAGMENTATION_SAMPLE_LIMIT") {
        return Err(String::from(
            "fragmentation entropy must use a bounded sample limit",
        ));
    }

    let global_entropy_body = extract_between(
        &topo_text,
        "pub fn fragmentation_entropy()",
        "// ---------------------------------------------------------------------------\n// T4",
    )?;
    reject_patterns(
        "fragmentation_entropy",
        global_entropy_body,
        &[
            "for i in 0..limit",
            "for i in 0..phys::total_usable_frames()",
            "for local_idx in 0..topo.frame_meta.len()",
        ],
    )?;
    if !global_entropy_body.contains("GLOBAL_ENTROPY_SAMPLE_LIMIT") {
        return Err(String::from(
            "global fragmentation entropy must use bounded sampling",
        ));
    }

    let reseed_body = extract_between(
        &topo_text,
        "fn reseed_voronoi",
        "// ---------------------------------------------------------------------------\n// T2",
    )?;
    reject_patterns(
        "reseed_voronoi",
        reseed_body,
        &["for i in 0..topo.frame_meta.len()"],
    )?;
    if !reseed_body.contains("RESEED_CELL_REFRESH_LIMIT") {
        return Err(String::from(
            "Voronoi reseed must refresh a bounded frame set",
        ));
    }

    Ok(())
}

fn extract_between<'a>(text: &'a str, start_pat: &str, end_pat: &str) -> Result<&'a str, String> {
    let start = text
        .find(start_pat)
        .ok_or_else(|| format!("{start_pat} not found"))?;
    let end = text[start..]
        .find(end_pat)
        .map(|offset| start + offset)
        .unwrap_or(text.len());
    Ok(&text[start..end])
}

fn reject_patterns(name: &str, body: &str, banned: &[&str]) -> Result<(), String> {
    let hits: Vec<&str> = banned
        .iter()
        .copied()
        .filter(|needle| body.contains(needle))
        .collect();
    if hits.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{name} reintroduced RAM-wide scan patterns: {}",
            hits.join(" | ")
        ))
    }
}

fn check_proof_screen(path: &Path) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if !bytes.starts_with(b"P6") {
        if path.extension().and_then(|ext| ext.to_str()) == Some("png") {
            let ppm = path.with_extension("ppm");
            if ppm.exists() {
                return check_proof_screen(&ppm);
            }
        }
        return Err(format!("{} is not a binary PPM (P6)", path.display()));
    }

    let mut idx = 2usize;
    let width = read_ppm_number(&bytes, &mut idx)?;
    let height = read_ppm_number(&bytes, &mut idx)?;
    let max = read_ppm_number(&bytes, &mut idx)?;
    if max != 255 {
        return Err(format!("unsupported PPM max value {max}"));
    }
    if idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }

    let expected = width
        .checked_mul(height)
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| String::from("PPM dimensions overflow"))?;
    if bytes.len().saturating_sub(idx) < expected {
        return Err(format!(
            "PPM pixel payload is truncated: expected {expected} bytes, got {}",
            bytes.len().saturating_sub(idx)
        ));
    }

    let pixels = &bytes[idx..idx + expected];
    let mut non_black = 0usize;
    let mut icon_region_signal = 0usize;
    let mut control_region_signal = 0usize;
    let mut primary_titlebar_signal = 0usize;

    for y in 0..height {
        for x in 0..width {
            let p = ((y * width + x) * 3) as usize;
            let r = pixels[p];
            let g = pixels[p + 1];
            let b = pixels[p + 2];
            if r > 8 || g > 8 || b > 8 {
                non_black += 1;
            }
            if x < 1_000 && (40..140).contains(&y) && (r > 32 || g > 32 || b > 32) {
                icon_region_signal += 1;
            }
            if (40..700).contains(&x) && (120..580).contains(&y) && (r > 28 || g > 28 || b > 28) {
                control_region_signal += 1;
            }
            if (48..660).contains(&x) && (132..158).contains(&y) && b > 96 && g > 32 && r < 96 {
                primary_titlebar_signal += 1;
            }
        }
    }

    if width < 1_024 || height < 768 {
        return Err(format!("proof screen is too small: {width}x{height}"));
    }
    if non_black < 20_000 {
        return Err(format!(
            "screen is mostly blank: {non_black} non-black pixels"
        ));
    }
    if icon_region_signal < 2_000 {
        return Err(format!(
            "desktop icons are not visible enough: {icon_region_signal} signal pixels"
        ));
    }
    if control_region_signal < 8_000 {
        return Err(format!(
            "primary terminal/control region is not visible enough: {control_region_signal} signal pixels"
        ));
    }
    if primary_titlebar_signal < 6_000 {
        return Err(format!(
            "primary terminal titlebar is not visible enough: {primary_titlebar_signal} signal pixels"
        ));
    }

    Ok(())
}

fn read_ppm_number(bytes: &[u8], idx: &mut usize) -> Result<usize, String> {
    while *idx < bytes.len() && bytes[*idx].is_ascii_whitespace() {
        *idx += 1;
    }
    if *idx < bytes.len() && bytes[*idx] == b'#' {
        while *idx < bytes.len() && bytes[*idx] != b'\n' {
            *idx += 1;
        }
        return read_ppm_number(bytes, idx);
    }

    let start = *idx;
    while *idx < bytes.len() && bytes[*idx].is_ascii_digit() {
        *idx += 1;
    }
    if start == *idx {
        return Err(String::from("expected PPM number"));
    }

    std::str::from_utf8(&bytes[start..*idx])
        .map_err(|e| format!("PPM number utf8: {e}"))?
        .parse::<usize>()
        .map_err(|e| format!("PPM number parse: {e}"))
}

fn collect_public_files(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_file() {
        out.push(path.to_path_buf());
        return Ok(());
    }

    let entries = fs::read_dir(path).map_err(|e| format!("read_dir {}: {e}", path.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir entry {}: {e}", path.display()))?;
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|s| s.to_str()) == Some("target") {
                continue;
            }
            collect_public_files(&path, out)?;
        } else if public_surface_extension(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn public_surface_extension(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()),
        Some("md" | "yml" | "yaml" | "sh" | "ps1" | "toml" | "txt")
    )
}

fn language_hygiene_exempt(rel: &Path) -> bool {
    let rel = rel.to_string_lossy().replace('\\', "/");
    rel.starts_with("docs/research/")
        || rel.starts_with("docs/superpowers/plans/")
        || rel == "docs/TOPOLOGICAL_OS_CONTRACT.md"
        || rel == "docs/BENCHMARK_PLAN.md"
        || rel == "docs/HOST_LANGUAGE_QUARANTINE.md"
}

fn language_line_allowed(line: &str) -> bool {
    line.contains("not linux")
        || line.contains("not unix")
        || line.contains("not posix")
        || line.contains("no linux")
        || line.contains("no unix")
        || line.contains("no posix")
        || line.contains("seal abi/no-posix")
        || line.contains("non-posix")
        || line.contains("host-runner")
        || line.contains("host runner")
        || line.contains("host ci")
}

fn strip_allowed_comment(line: &str) -> &str {
    line.split_once("//").map_or(line, |(code, _)| code)
}

fn is_unsafe_site(line: &str) -> bool {
    line.contains("unsafe {")
        || line.contains("unsafe{")
        || line.contains("unsafe fn")
        || line.contains("unsafe impl")
}
