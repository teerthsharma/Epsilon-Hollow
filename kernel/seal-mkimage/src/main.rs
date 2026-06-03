// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Creates bootable UEFI disk images and ISOs for Seal OS.
//! VirtualBox-compatible: GPT + FAT32 ESP with proper boot sector.

mod aether_build;

use std::collections::BTreeMap;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const DISK_SIZE: u64 = 128 * 1024 * 1024; // 128MB
const SECTOR_SIZE: u64 = 512;
const ESP_START_LBA: u64 = 2048;
const ESP_SIZE_SECTORS: u64 = (64 * 1024 * 1024) / SECTOR_SIZE;
const MANIFOLD_START_LBA: u64 = ESP_START_LBA + ESP_SIZE_SECTORS;
const MANIFOLD_END_LBA: u64 = (DISK_SIZE / SECTOR_SIZE) - 34;
const MANIFOLD_SIZE_SECTORS: u64 = MANIFOLD_END_LBA - MANIFOLD_START_LBA + 1;
const MNFD_JOURNAL_BLOCKS: u64 = 1024;
const MNFD_FREE_BITMAP_BLOCKS: u64 = 256;
const EXPECTED_SEAL_OS_BANNER: &str = "Seal OS v0.4.6";
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
    if args.get(1).map(|s| s.as_str()) == Some("--check-installer-proof") {
        let log_path = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp/seal-os-installer.log"));
        let text = fs::read_to_string(&log_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", log_path.display()));
        check_installer_proof_text(&text).unwrap_or_else(|e| {
            eprintln!("[seal-audit] INSTALLER PROOF FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] INSTALLER PROOF OK: {}", log_path.display());
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
    if args.get(1).map(|s| s.as_str()) == Some("--check-release-workflow-contract") {
        let root = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_release_workflow_contract(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] RELEASE WORKFLOW CONTRACT FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] RELEASE WORKFLOW CONTRACT OK");
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
    if args.get(1).map(|s| s.as_str()) == Some("--check-laamba-app-proof") {
        let log_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from("../seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log")
        });
        check_laamba_app_proof_log(&log_path).unwrap_or_else(|e| {
            eprintln!("[seal-audit] LAAMBA APP PROOF FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] LAAMBA APP PROOF OK: {}", log_path.display());
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-laamba-native-bridge") {
        let root = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_laamba_native_bridge(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] LAAMBA NATIVE BRIDGE FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] LAAMBA NATIVE BRIDGE OK");
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
    if args.get(1).map(|s| s.as_str()) == Some("--check-o1-network") {
        let root = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_o1_network(&root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] O(1) NETWORK FAIL: {e}");
            std::process::exit(1);
        });
        println!("[seal-audit] O(1) NETWORK OK");
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
    if args.get(1).map(|s| s.as_str()) == Some("--check-current-proof-manifest") {
        let manifest = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(
                "../seal-os/target/x86_64-unknown-uefi/release/qemu-proof/proof-manifest.txt",
            )
        });
        let root = args
            .get(3)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_current_proof_manifest(&manifest, &root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] CURRENT PROOF MANIFEST FAIL: {e}");
            std::process::exit(1);
        });
        println!(
            "[seal-audit] CURRENT PROOF MANIFEST OK: {}",
            manifest.display()
        );
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--write-qemu-proof-manifest") {
        let run_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from("../seal-os/target/x86_64-unknown-uefi/release/qemu-proof")
        });
        let image_path = args
            .get(3)
            .map(PathBuf::from)
            .unwrap_or_else(|| run_path.join("seal-os.img"));
        let backend = args.get(4).map(String::as_str).unwrap_or("native");
        let seconds = args
            .get(5)
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(120);
        let root = args
            .get(6)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let manifest = write_qemu_proof_manifest(&run_path, &image_path, backend, seconds, &root)
            .unwrap_or_else(|e| {
                eprintln!("[seal-audit] WRITE QEMU PROOF MANIFEST FAIL: {e}");
                std::process::exit(1);
            });
        println!(
            "[seal-audit] WRITE QEMU PROOF MANIFEST OK: {}",
            manifest.display()
        );
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
            "[seal-audit] BENCHMARK COMPARE OK: provided alloc-frame logs show Seal p50/p95/max below Ubuntu; allocator-only, not a global Ubuntu win ({}, {})",
            seal_log.display(),
            ubuntu_log.display()
        );
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("--check-current-benchmark-proof") {
        let manifest = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(
                "../seal-os/target/x86_64-unknown-uefi/release/qemu-proof/proof-manifest.txt",
            )
        });
        let ubuntu_log = args
            .get(3)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("ubuntu-alloc.log"));
        let root = args
            .get(4)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        check_current_benchmark_proof_manifest(&manifest, &ubuntu_log, &root).unwrap_or_else(|e| {
            eprintln!("[seal-audit] CURRENT BENCHMARK PROOF FAIL: {e}");
            std::process::exit(1);
        });
        println!(
            "[seal-audit] CURRENT BENCHMARK PROOF OK: current VM proof manifest plus native Ubuntu artifact show allocator-only Seal p50/p95/max lead ({}, {})",
            manifest.display(),
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

fn sha256_hex(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = sha256(data);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h = [
        0x6a09e667_u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((data.len() + 9).div_ceil(64) * 64);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0_u32; 64];
        for (idx, word) in w.iter_mut().take(16).enumerate() {
            let offset = idx * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for idx in 16..64 {
            let s0 =
                w[idx - 15].rotate_right(7) ^ w[idx - 15].rotate_right(18) ^ (w[idx - 15] >> 3);
            let s1 = w[idx - 2].rotate_right(17) ^ w[idx - 2].rotate_right(19) ^ (w[idx - 2] >> 10);
            w[idx] = w[idx - 16]
                .wrapping_add(s0)
                .wrapping_add(w[idx - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut h_work = h[7];

        for idx in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h_work
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[idx])
                .wrapping_add(w[idx]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h_work = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(h_work);
    }

    let mut digest = [0_u8; 32];
    for (idx, word) in h.iter().enumerate() {
        digest[idx * 4..idx * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
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
        "[BENCH] slab-alloc",
        "[BENCH] manifold-teleport",
        "[BENCH] scheduler-select-next",
        "[BENCH] tcp-packet-demux",
        "[GPU-BENCH] suite",
        "[Aether-Lang] runtime proof:",
        "[LAAMBA] app proof:",
        "[SECURITY] hardening proof",
        "[SECURITY] audit proof",
        "[SECURITY] auth proof",
        "[MM] cow-proof",
        "[ManifoldPkg] proof",
        "[BOOT] Desktop proof frame blit done",
        "[GFX] desktop-proof",
        "[GFX] desktop-live-proof",
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
    check_laamba_app_proof_text(&text)?;
    check_security_hardening_text(&text)?;
    check_security_audit_text(&text)?;
    check_auth_shadow_proof_text(&text)?;
    check_cow_proof_text(&text)?;
    check_manifoldpkg_proof_text(&text)?;
    Ok(())
}

fn check_cow_proof_text(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[MM] cow-proof")?;
    require_field_eq(line, "version=", "1", "COW proof")?;
    require_field_eq(line, "rollback_guard=", "1", "COW proof")?;
    require_field_eq(line, "fork_fallback=", "0", "COW proof")?;
    require_field_eq(line, "clone_fallback=", "0", "COW proof")?;
    require_field_eq(line, "leaked_frames=", "0", "COW proof")?;
    require_field_eq(line, "result=", "pass", "COW proof")?;
    let samples = parse_metric(line, "samples=")?;
    let rollback_ok = parse_metric(line, "rollback_ok=")?;
    let tracked = parse_metric(line, "tracked_frames=")?;
    let freed = parse_metric(line, "rollback_frees=")?;
    if samples == 0 || samples != rollback_ok {
        return Err(format!(
            "COW proof rollback samples incomplete: samples={samples}, rollback_ok={rollback_ok}"
        ));
    }
    if tracked == 0 || tracked != freed {
        return Err(format!(
            "COW proof rollback accounting mismatch: tracked_frames={tracked}, rollback_frees={freed}"
        ));
    }
    Ok(())
}

fn check_auth_shadow_proof_text(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[SECURITY] auth proof")?;
    require_field_eq(line, "version=", "1", "auth shadow proof")?;
    require_field_eq(line, "shadow=", "1", "auth shadow proof")?;
    require_field_eq(line, "default_user=", "seal", "auth shadow proof")?;
    require_field_eq(line, "default_present=", "1", "auth shadow proof")?;
    require_field_eq(line, "default_topo5000=", "1", "auth shadow proof")?;
    require_field_eq(line, "default_legacy=", "0", "auth shadow proof")?;
    require_field_eq(
        line,
        "default_password_rejected=",
        "1",
        "auth shadow proof",
    )?;
    require_field_eq(line, "new_user_topo5000=", "1", "auth shadow proof")?;
    require_field_eq(line, "result=", "pass", "auth shadow proof")?;

    let embedded = parse_metric(line, "passwd_embedded_hashes=")?;
    if embedded != 0 {
        return Err(String::from(
            "auth shadow proof found embedded password hashes in /etc/passwd",
        ));
    }
    Ok(())
}

fn check_installer_proof_text(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[INSTALLER] proof")?;
    require_field_eq(line, "version=", "1", "installer proof")?;
    require_field_eq(line, "mode=", "safe_vfs", "installer proof")?;
    require_field_eq(line, "boot_marker=", "1", "installer proof")?;
    require_field_eq(line, "home=", "1", "installer proof")?;
    require_field_eq(line, "profile=", "1", "installer proof")?;
    require_field_eq(line, "user=", "1", "installer proof")?;
    require_field_eq(line, "auth_topo5000=", "1", "installer proof")?;
    require_field_eq(line, "raw_gpt=", "0", "installer proof")?;
    require_field_eq(line, "raw_format=", "0", "installer proof")?;
    require_field_eq(line, "result=", "pass", "installer proof")?;
    parse_field(line, "selected_disk=")?;

    let banned = [
        "Would create GPT",
        "Would format",
        "Would copy",
        "Would install",
        "Installation simulation complete",
        "SHA-256 hash",
    ];
    let hits: Vec<&str> = text
        .lines()
        .filter(|line| banned.iter().any(|marker| line.contains(marker)))
        .collect();
    if !hits.is_empty() {
        return Err(format!(
            "installer proof log contains simulation markers: {}",
            hits.join(" | ")
        ));
    }
    Ok(())
}

fn check_manifoldpkg_proof_text(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[ManifoldPkg] proof")?;
    require_field_eq(line, "version=", "1", "ManifoldPkg proof")?;
    require_field_eq(line, "source=", "embedded_eph", "ManifoldPkg proof")?;
    require_field_eq(line, "parse=", "ok", "ManifoldPkg proof")?;
    require_field_eq(
        line,
        "registry_index=",
        "ed25519_fixture",
        "ManifoldPkg proof",
    )?;
    require_field_eq(line, "install=", "ok", "ManifoldPkg proof")?;
    require_field_eq(line, "extract=", "ok", "ManifoldPkg proof")?;
    require_field_eq(line, "list=", "ok", "ManifoldPkg proof")?;
    require_field_eq(line, "remove=", "ok", "ManifoldPkg proof")?;
    require_field_eq(line, "metadata_only=", "0", "ManifoldPkg proof")?;
    require_field_eq(line, "signature=", "ed25519_fixture", "ManifoldPkg proof")?;
    require_field_eq(line, "result=", "pass", "ManifoldPkg proof")?;

    let files = parse_metric(line, "files=")?;
    let bytes = parse_metric(line, "bytes=")?;
    let before = parse_metric(line, "package_count_before=")?;
    let after_install = parse_metric(line, "package_count_after_install=")?;
    let after_remove = parse_metric(line, "package_count_after_remove=")?;
    if files == 0 {
        return Err(String::from("ManifoldPkg proof must extract at least one file"));
    }
    if bytes == 0 {
        return Err(String::from("ManifoldPkg proof must extract nonempty bytes"));
    }
    if after_install != before + 1 {
        return Err(String::from(
            "ManifoldPkg proof package count must increase by one after install",
        ));
    }
    if after_remove != before {
        return Err(String::from(
            "ManifoldPkg proof package count must return to baseline after remove",
        ));
    }
    Ok(())
}

fn check_security_hardening_text(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[SECURITY] hardening proof")?;
    require_field_eq(line, "version=", "1", "security hardening proof")?;
    require_field_eq(line, "kpti=", "1", "security hardening proof")?;
    require_field_eq(line, "kpti_distinct=", "1", "security hardening proof")?;
    require_field_eq(line, "user_lower_zero=", "1", "security hardening proof")?;
    require_field_eq(
        line,
        "kernel_upper_mirrored=",
        "1",
        "security hardening proof",
    )?;
    require_field_eq(line, "result=", "pass", "security hardening proof")?;

    let kernel_cr3 = parse_hex_metric(line, "kernel_cr3=")?;
    let user_cr3 = parse_hex_metric(line, "user_cr3=")?;
    if kernel_cr3 == 0 || user_cr3 == 0 {
        return Err(String::from(
            "security hardening proof CR3 roots must be nonzero",
        ));
    }
    if kernel_cr3 == user_cr3 {
        return Err(String::from(
            "security hardening proof kernel/user CR3 roots must differ",
        ));
    }

    let smap_supported = parse_metric(line, "smap_smep_supported=")?;
    let smap_enabled = parse_metric(line, "smap_smep_enabled=")?;
    if smap_supported > 1 || smap_enabled > 1 {
        return Err(String::from(
            "security hardening proof SMAP/SMEP fields must be booleans",
        ));
    }
    if smap_supported == 1 && smap_enabled != 1 {
        return Err(String::from(
            "security hardening proof reports SMAP/SMEP support without enablement",
        ));
    }
    parse_metric(line, "user_access_faults=")?;
    Ok(())
}

fn check_security_audit_text(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[SECURITY] audit proof")?;
    require_field_eq(line, "version=", "1", "security audit proof")?;
    require_field_eq(line, "vfs=", "1", "security audit proof")?;
    require_field_eq(line, "dirs=", "1", "security audit proof")?;
    require_field_eq(line, "buffered_after=", "0", "security audit proof")?;
    require_field_eq(line, "file=", "/var/log/audit.log", "security audit proof")?;
    require_field_eq(line, "readback=", "1", "security audit proof")?;
    require_field_eq(line, "flushed=", "1", "security audit proof")?;
    require_field_eq(line, "result=", "pass", "security audit proof")?;
    parse_metric(line, "buffered_before=")?;
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

fn check_laamba_app_proof_log(log_path: &Path) -> Result<(), String> {
    let text =
        fs::read_to_string(log_path).map_err(|e| format!("read {}: {e}", log_path.display()))?;
    check_laamba_app_proof_text(&text)
}

fn check_laamba_app_proof_text(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[LAAMBA] app proof:")?;
    let label = "LAAMBA app proof";
    require_field_eq(line, "version=", "1", label)?;
    require_field_eq(line, "native_app=", "kernel", label)?;
    require_field_eq(line, "window=", "LAAMBA_Governor", label)?;
    require_field_eq(line, "runtime_bridge=", "rust_native_manifest", label)?;
    require_field_eq(line, "result=", "pass", label)?;
    require_metric_eq(line, "launcher_id=", 10, label)?;
    require_metric_eq(line, "desktop_icon=", 1, label)?;
    require_metric_eq(line, "start_menu=", 1, label)?;
    require_metric_eq(line, "python_runtime=", 0, label)?;
    require_metric_min(line, "window_id=", 1, label)?;
    require_metric_min(line, "aether_host_window_id=", 1, label)?;
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
    parse_slab_alloc_benchmark(text)?;
    parse_manifold_teleport_benchmark(text)?;
    parse_scheduler_select_benchmark(text)?;
    parse_tcp_packet_demux_benchmark(text)?;
    parse_gpu_topology_benchmark(text)?;
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

fn check_current_benchmark_proof_manifest(
    manifest_path: &Path,
    ubuntu_log: &Path,
    root: &Path,
) -> Result<(), String> {
    let (head, dirty) = current_git_state(root)?;
    check_benchmark_proof_manifest_with_state(manifest_path, ubuntu_log, &head, dirty)
}

fn check_benchmark_proof_manifest_with_state(
    manifest_path: &Path,
    ubuntu_log: &Path,
    expected_commit: &str,
    expected_dirty: bool,
) -> Result<(), String> {
    check_proof_manifest(manifest_path)?;
    let text = fs::read_to_string(manifest_path)
        .map_err(|e| format!("read {}: {e}", manifest_path.display()))?;
    let fields = parse_manifest_fields(&text)?;
    check_manifest_provenance_fields(&fields, expected_commit, expected_dirty)?;
    let parent = manifest_path
        .parent()
        .ok_or_else(|| format!("manifest has no parent: {}", manifest_path.display()))?;
    let seal_log =
        manifest_artifact_resolved_path(&fields, parent, "serial_log", Some("serial.log"))?;
    check_benchmark_log(&seal_log)?;
    compare_benchmark_logs(&seal_log, ubuntu_log)
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

fn parse_slab_alloc_benchmark(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[BENCH] slab-alloc")?;
    let bench = parse_alloc_bench_metrics(line, "Slab allocation benchmark")?;
    let classes = parse_metric(line, "classes=")?;
    let refill_ok = parse_metric(line, "refill_ok=")?;
    let reuse_ok = parse_metric(line, "reuse_ok=")?;
    let free_ok = parse_metric(line, "free_ok=")?;
    let realloc_ok = parse_metric(line, "realloc_ok=")?;
    let refill_pages_delta = parse_metric(line, "refill_pages_delta=")?;
    let free_list_hits_delta = parse_metric(line, "free_list_hits_delta=")?;
    let alloc_calls_delta = parse_metric(line, "alloc_calls_delta=")?;
    let free_calls_delta = parse_metric(line, "free_calls_delta=")?;
    let free_before = parse_metric(line, "free_before=")?;
    let free_after_refill = parse_metric(line, "free_after_refill=")?;
    let free_after_reuse = parse_metric(line, "free_after_reuse=")?;

    if classes != 6 {
        return Err(format!("slab benchmark must cover 6 classes, got {classes}"));
    }
    if refill_ok != classes || reuse_ok != classes || free_ok != classes || realloc_ok != 2 {
        return Err(format!(
            "slab benchmark coverage incomplete: refill_ok={refill_ok}, reuse_ok={reuse_ok}, free_ok={free_ok}, realloc_ok={realloc_ok}"
        ));
    }
    if refill_pages_delta < classes {
        return Err(format!(
            "slab benchmark did not force one refill page per class: refill_pages_delta={refill_pages_delta}, classes={classes}"
        ));
    }
    if free_list_hits_delta < bench.iterations {
        return Err(format!(
            "slab benchmark did not use free-list fast path enough: free_list_hits_delta={free_list_hits_delta}, iterations={}",
            bench.iterations
        ));
    }
    if alloc_calls_delta < bench.iterations || free_calls_delta < bench.iterations {
        return Err(format!(
            "slab benchmark call accounting too low: alloc_calls_delta={alloc_calls_delta}, free_calls_delta={free_calls_delta}, iterations={}",
            bench.iterations
        ));
    }
    if free_before <= free_after_refill {
        return Err(format!(
            "slab benchmark did not consume refill frames: free_before={free_before}, free_after_refill={free_after_refill}"
        ));
    }
    if free_after_refill != free_after_reuse {
        return Err(format!(
            "slab benchmark reuse allocated more frames: free_after_refill={free_after_refill}, free_after_reuse={free_after_reuse}"
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
    let persistence_bytes = parse_metric(line, "persistence_bytes_per_move=")?;
    let payload_points = parse_metric(line, "payload_points=")?;

    if api != "teleport" || fs_mode != "mock_block" || persistence != "metadata_only" {
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
    if persistence_bytes != 0 {
        return Err(format!(
            "ManifoldFS teleport rewrote file bytes: persistence_bytes_per_move={persistence_bytes}, payload_bytes={payload_bytes}"
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

fn parse_tcp_packet_demux_benchmark(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[BENCH] tcp-packet-demux")?;
    let api = parse_field(line, "api=")?;
    let fixture = parse_field(line, "fixture=")?;
    let accepted_state = parse_field(line, "accepted_state=")?;
    let ok = parse_metric(line, "ok=")?;
    let listener_first = parse_metric(line, "listener_first=")?;
    let exact_flow = parse_metric(line, "exact_flow=")?;
    let decoy_rx_bytes = parse_metric(line, "decoy_rx_bytes=")?;
    let listener_fallback = parse_metric(line, "listener_fallback=")?;
    let payload_bytes = parse_metric(line, "payload_bytes=")?;
    let rx_bytes = parse_metric(line, "rx_bytes=")?;
    let o1_index = parse_metric(line, "o1_index=")?;
    let index_hit = parse_metric(line, "index_hit=")?;
    let index_lookup_probes = parse_metric(line, "index_lookup_probes=")?;
    let index_probe_bound = parse_metric(line, "index_probe_bound=")?;
    let index_capacity = parse_metric(line, "index_capacity=")?;
    let listener_index_hit = parse_metric(line, "listener_index_hit=")?;
    let listener_lookup_probes = parse_metric(line, "listener_lookup_probes=")?;
    let listener_probe_bound = parse_metric(line, "listener_probe_bound=")?;
    let listener_index_capacity = parse_metric(line, "listener_index_capacity=")?;
    let exact_scan = parse_metric(line, "exact_scan=")?;
    let cleanup = parse_field(line, "cleanup=")?;

    if api != "handle_tcp_packet" || fixture != "listener_first" {
        return Err(format!(
            "TCP packet demux benchmark identifies the wrong path: api={api}, fixture={fixture}"
        ));
    }
    if ok != 1 || listener_first != 1 || accepted_state != "established" || cleanup != "ok" {
        return Err(format!(
            "TCP packet demux benchmark failed invariants: ok={ok}, listener_first={listener_first}, accepted_state={accepted_state}, cleanup={cleanup}"
        ));
    }
    if exact_flow != 1 || decoy_rx_bytes != 0 || listener_fallback != 1 {
        return Err(format!(
            "TCP packet demux did not prove exact-flow before decoy/listener fallback: exact_flow={exact_flow}, decoy_rx_bytes={decoy_rx_bytes}, listener_fallback={listener_fallback}"
        ));
    }
    if payload_bytes != 4 || rx_bytes != payload_bytes {
        return Err(format!(
            "TCP packet demux payload mismatch: payload_bytes={payload_bytes}, rx_bytes={rx_bytes}"
        ));
    }
    if o1_index != 1 || index_hit != 1 || exact_scan != 0 {
        return Err(format!(
            "TCP packet demux exact-flow path did not use the bounded index: o1_index={o1_index}, index_hit={index_hit}, exact_scan={exact_scan}"
        ));
    }
    if index_capacity < 64 || index_probe_bound != index_capacity {
        return Err(format!(
            "TCP packet demux index proof has invalid bounds: index_capacity={index_capacity}, index_probe_bound={index_probe_bound}"
        ));
    }
    if index_lookup_probes == 0 || index_lookup_probes > index_probe_bound {
        return Err(format!(
            "TCP packet demux index lookup exceeded bound: index_lookup_probes={index_lookup_probes}, index_probe_bound={index_probe_bound}"
        ));
    }
    if listener_index_hit != 1 {
        return Err(format!(
            "TCP packet demux listener fallback did not use the bounded listener index: listener_index_hit={listener_index_hit}"
        ));
    }
    if listener_index_capacity < 64 || listener_probe_bound != listener_index_capacity {
        return Err(format!(
            "TCP packet demux listener index proof has invalid bounds: listener_index_capacity={listener_index_capacity}, listener_probe_bound={listener_probe_bound}"
        ));
    }
    if listener_lookup_probes == 0 || listener_lookup_probes > listener_probe_bound {
        return Err(format!(
            "TCP packet demux listener lookup exceeded bound: listener_lookup_probes={listener_lookup_probes}, listener_probe_bound={listener_probe_bound}"
        ));
    }

    Ok(())
}

fn parse_gpu_topology_benchmark(text: &str) -> Result<(), String> {
    let begin = find_line_matching(text, "[GPU-BENCH] suite", "status=begin")?;
    require_field_eq(begin, "version=", "1", "GPU topology benchmark begin")?;
    require_field_eq(
        begin,
        "mode=",
        "cpu_fallback",
        "GPU topology benchmark begin",
    )?;
    require_field_eq(
        begin,
        "backend=",
        "software",
        "GPU topology benchmark begin",
    )?;
    require_field_eq(
        begin,
        "accelerator=",
        "cpu_fallback",
        "GPU topology benchmark begin",
    )?;
    require_metric_eq(
        begin,
        "hardware_dispatch=",
        0,
        "GPU topology benchmark begin",
    )?;
    require_metric_eq(begin, "shader_used=", 0, "GPU topology benchmark begin")?;
    require_field_eq(
        begin,
        "shader_status=",
        "placeholder_ok_not_claimed",
        "GPU topology benchmark begin",
    )?;
    let warmup = parse_metric(begin, "warmup=")?;
    let iterations = parse_metric(begin, "iterations=")?;
    require_metric_eq(begin, "kernels=", 3, "GPU topology benchmark begin")?;
    if warmup != 4 || iterations != 32 {
        return Err(format!(
            "GPU topology benchmark fixture changed: warmup={warmup}, iterations={iterations}"
        ));
    }

    parse_gpu_kernel_benchmark(
        text,
        GpuKernelBenchSpec {
            name: "voronoi_assign",
            warmup,
            iterations,
            checked: 64,
            upload_ok: 2,
            download_ok: 1,
            finite: None,
        },
    )?;
    parse_gpu_kernel_benchmark(
        text,
        GpuKernelBenchSpec {
            name: "jl_project",
            warmup,
            iterations,
            checked: 12,
            upload_ok: 1,
            download_ok: 1,
            finite: Some(12),
        },
    )?;
    parse_gpu_kernel_benchmark(
        text,
        GpuKernelBenchSpec {
            name: "spectral_step",
            warmup,
            iterations,
            checked: 512,
            upload_ok: 1,
            download_ok: 1,
            finite: Some(512),
        },
    )?;

    let final_line = find_line_matching(text, "[GPU-BENCH] suite", "result=pass")?;
    require_field_eq(final_line, "version=", "1", "GPU topology benchmark final")?;
    require_field_eq(
        final_line,
        "mode=",
        "cpu_fallback",
        "GPU topology benchmark final",
    )?;
    require_field_eq(
        final_line,
        "backend=",
        "software",
        "GPU topology benchmark final",
    )?;
    require_metric_eq(
        final_line,
        "hardware_dispatch=",
        0,
        "GPU topology benchmark final",
    )?;
    require_metric_eq(
        final_line,
        "shader_used=",
        0,
        "GPU topology benchmark final",
    )?;
    require_metric_eq(final_line, "kernels=", 3, "GPU topology benchmark final")?;
    require_metric_eq(final_line, "passed=", 3, "GPU topology benchmark final")?;
    require_metric_eq(final_line, "failed=", 0, "GPU topology benchmark final")?;
    require_field_eq(
        final_line,
        "result=",
        "pass",
        "GPU topology benchmark final",
    )?;
    require_field_eq(
        final_line,
        "claim=",
        "cpu_fallback_correctness_only",
        "GPU topology benchmark final",
    )?;

    let kernel_lines = text
        .lines()
        .filter(|line| line.starts_with("[GPU-BENCH] kernel="))
        .count();
    if kernel_lines != 3 {
        return Err(format!(
            "GPU topology benchmark must emit exactly 3 kernel markers, got {kernel_lines}"
        ));
    }

    Ok(())
}

struct GpuKernelBenchSpec {
    name: &'static str,
    warmup: u64,
    iterations: u64,
    checked: u64,
    upload_ok: u64,
    download_ok: u64,
    finite: Option<u64>,
}

fn parse_gpu_kernel_benchmark(text: &str, spec: GpuKernelBenchSpec) -> Result<(), String> {
    let needle = format!("kernel={}", spec.name);
    let line = find_line_matching(text, "[GPU-BENCH] kernel=", &needle)?;
    let label = format!("GPU topology kernel {}", spec.name);
    require_field_eq(line, "mode=", "cpu_fallback", &label)?;
    require_field_eq(line, "backend=", "software", &label)?;
    require_field_eq(line, "dispatch_path=", "cpu_sync", &label)?;
    require_metric_eq(line, "hardware_dispatch=", 0, &label)?;
    require_metric_eq(line, "shader_used=", 0, &label)?;
    require_metric_eq(line, "warmup=", spec.warmup, &label)?;
    require_metric_eq(line, "iterations=", spec.iterations, &label)?;
    let total_runs = spec.warmup + spec.iterations;
    require_metric_eq(line, "dispatch_ok=", total_runs, &label)?;
    require_metric_eq(line, "wait_ok=", total_runs, &label)?;
    require_metric_eq(line, "upload_ok=", spec.upload_ok, &label)?;
    require_metric_eq(line, "download_ok=", spec.download_ok, &label)?;
    require_field_eq(line, "verify=", "pass", &label)?;
    require_field_eq(line, "reference=", "cpu_recompute", &label)?;
    require_metric_eq(line, "checked=", spec.checked, &label)?;
    require_metric_eq(line, "mismatches=", 0, &label)?;
    let checksum = parse_metric(line, "checksum=")?;
    let avg_cycles = parse_metric(line, "avg_cycles=")?;
    if checksum == 0 || avg_cycles == 0 {
        return Err(format!(
            "{label} must have nonzero checksum and cycle counter: checksum={checksum}, avg_cycles={avg_cycles}"
        ));
    }

    match spec.name {
        "voronoi_assign" => {
            require_metric_eq(line, "n_points=", 64, &label)?;
            require_metric_eq(line, "n_cells=", 8, &label)?;
            require_metric_eq(line, "invalid_ids=", 0, &label)?;
            let first_cell = parse_metric(line, "first_cell=")?;
            if first_cell >= 8 {
                return Err(format!("{label} first_cell out of range: {first_cell}"));
            }
        }
        "jl_project" => {
            require_metric_eq(line, "n_vectors=", 4, &label)?;
            require_metric_eq(line, "dim_in=", 128, &label)?;
            require_metric_eq(line, "dim_out=", 3, &label)?;
            require_field_eq(line, "seed=", "0xE95110A7", &label)?;
            require_metric_eq(line, "max_abs_diff_scaled=", 0, &label)?;
        }
        "spectral_step" => {
            require_metric_eq(line, "dim=", 512, &label)?;
            require_metric_eq(line, "alpha_ppm=", 300000, &label)?;
            require_metric_eq(line, "max_abs_diff_scaled=", 0, &label)?;
        }
        _ => unreachable!(),
    }
    if let Some(finite) = spec.finite {
        require_metric_eq(line, "finite=", finite, &label)?;
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
    find_unique_marker_line(text, marker)
}

fn find_unique_marker_line<'a>(text: &'a str, marker: &str) -> Result<&'a str, String> {
    let mut found = None;
    let mut count = 0u64;
    for line in text.lines().filter(|line| line.starts_with(marker)) {
        count += 1;
        if found.is_none() {
            found = Some(line);
        }
    }
    match (found, count) {
        (Some(line), 1) => Ok(line),
        (Some(_), _) => Err(format!(
            "{marker} marker must appear exactly once, found {count}"
        )),
        (None, _) => Err(format!("missing {marker} marker")),
    }
}

fn find_line_matching<'a>(text: &'a str, marker: &str, needle: &str) -> Result<&'a str, String> {
    text.lines()
        .find(|line| line.starts_with(marker) && line.contains(needle))
        .ok_or_else(|| format!("missing {marker} marker containing `{needle}`"))
}

fn require_metric_eq(line: &str, key: &str, expected: u64, label: &str) -> Result<(), String> {
    let got = parse_metric(line, key)?;
    if got != expected {
        return Err(format!("{label} expected {key}{expected}, got {key}{got}"));
    }
    Ok(())
}

fn require_metric_min(line: &str, key: &str, min: u64, label: &str) -> Result<u64, String> {
    let got = parse_metric(line, key)?;
    if got < min {
        return Err(format!("{label} expected {key}>= {min}, got {key}{got}"));
    }
    Ok(got)
}

fn require_field_eq(line: &str, key: &str, expected: &str, label: &str) -> Result<(), String> {
    let got = parse_field(line, key)?;
    if got != expected {
        return Err(format!("{label} expected {key}{expected}, got {key}{got}"));
    }
    Ok(())
}

fn check_desktop_soak(log_path: &Path) -> Result<(), String> {
    let text =
        fs::read_to_string(log_path).map_err(|e| format!("read {}: {e}", log_path.display()))?;
    check_desktop_soak_text(&text)
}

fn check_desktop_soak_text(text: &str) -> Result<(), String> {
    parse_graphics_desktop_proof(text)?;
    parse_graphics_desktop_live_proof(text)?;

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

fn parse_graphics_desktop_live_proof(text: &str) -> Result<(), String> {
    let line = find_unique_marker_line(text, "[GFX] desktop-live-proof")?;
    let label = "graphics desktop live proof";
    require_field_eq(line, "version=", "1", label)?;
    require_field_eq(line, "route=", "desktop_handle_input", label)?;
    require_field_eq(line, "action=", "desktop_icon_launch", label)?;
    require_field_eq(line, "app=", "Files", label)?;
    require_field_eq(line, "result=", "pass", label)?;
    require_metric_eq(line, "app_id=", 3, label)?;
    require_metric_eq(line, "blit=", 1, label)?;
    require_metric_eq(line, "handled=", 1, label)?;
    require_metric_eq(line, "icon_hit=", 1, label)?;
    require_metric_eq(line, "launched_app_id=", 3, label)?;
    require_metric_min(line, "events=", 2, label)?;
    require_metric_min(line, "window_count=", 12, label)?;
    require_metric_min(line, "changed_samples=", 32, label)?;
    require_metric_min(line, "vram_changed_samples=", 32, label)?;
    require_metric_min(line, "vram_matches_backbuffer=", 32, label)?;

    let pre_focused = parse_metric(line, "pre_focused=")?;
    let post_focused = parse_metric(line, "post_focused=")?;
    let post_window_id = parse_metric(line, "post_window_id=")?;
    let pre_hash = parse_metric(line, "pre_hash=")?;
    let post_hash = parse_metric(line, "post_hash=")?;
    let vram_hash = parse_metric(line, "vram_hash=")?;

    if pre_focused == 0 || post_focused == 0 || post_window_id == 0 {
        return Err(format!(
            "{label} focus ids must be nonzero: pre={pre_focused}, post={post_focused}, post_window_id={post_window_id}"
        ));
    }
    if post_focused != post_window_id {
        return Err(format!(
            "{label} post focus must match launched window: post_focused={post_focused}, post_window_id={post_window_id}"
        ));
    }
    if pre_focused == post_focused {
        return Err(format!(
            "{label} did not change focus: pre_focused={pre_focused}, post_focused={post_focused}"
        ));
    }
    if pre_hash == 0 || post_hash == 0 {
        return Err(format!(
            "{label} hashes must be nonzero: pre_hash={pre_hash}, post_hash={post_hash}"
        ));
    }
    if pre_hash == post_hash {
        return Err(format!(
            "{label} visible samples did not change: hash={pre_hash}"
        ));
    }
    if vram_hash == 0 {
        return Err(format!("{label} presented VRAM hash must be nonzero"));
    }
    if vram_hash == pre_hash {
        return Err(format!(
            "{label} presented VRAM samples did not change: hash={vram_hash}"
        ));
    }

    Ok(())
}

fn parse_graphics_desktop_proof(text: &str) -> Result<(), String> {
    let line = find_marker_line(text, "[GFX] desktop-proof")?;
    let label = "graphics desktop proof";
    require_field_eq(line, "version=", "1", label)?;
    require_field_eq(line, "surface=", "framebuffer", label)?;
    require_field_eq(line, "result=", "pass", label)?;
    require_metric_eq(line, "bpp=", 32, label)?;
    require_metric_eq(line, "back_buffer=", 1, label)?;

    let width = parse_metric(line, "width=")?;
    let height = parse_metric(line, "height=")?;
    let pitch = parse_metric(line, "pitch=")?;
    if width < 1024 || height < 768 {
        return Err(format!("{label} framebuffer too small: {width}x{height}"));
    }
    if pitch < width.saturating_mul(4) {
        return Err(format!(
            "{label} pitch too small for 32bpp framebuffer: pitch={pitch}, width={width}"
        ));
    }

    let scanned_pixels = parse_metric(line, "scanned_pixels=")?;
    let expected_pixels = width
        .checked_mul(height)
        .ok_or_else(|| String::from("graphics desktop proof dimensions overflow"))?;
    if scanned_pixels < expected_pixels {
        return Err(format!(
            "{label} did not scan full framebuffer: scanned_pixels={scanned_pixels}, expected={expected_pixels}"
        ));
    }

    require_metric_min(line, "window_count=", 12, label)?;
    require_metric_min(line, "focused_window_id=", 1, label)?;
    require_metric_min(line, "nonblack_px=", 20_000, label)?;
    require_metric_min(line, "visible_icons=", 10, label)?;
    require_metric_min(line, "icon_region_signal=", 2_000, label)?;
    require_metric_min(line, "icon_color_buckets=", 6, label)?;
    require_metric_min(line, "control_region_signal=", 8_000, label)?;
    require_metric_min(line, "primary_titlebar_signal=", 6_000, label)?;
    require_metric_min(line, "start_button_signal=", 1_000, label)?;
    require_metric_min(line, "theorem_indicator_signal=", 500, label)?;
    require_metric_min(line, "minimized_app_lane_signal=", 300, label)?;
    require_metric_min(line, "power_button_signal=", 220, label)?;
    require_metric_min(line, "sampled_pixels=", 1_024, label)?;
    require_metric_min(line, "nonblack_samples=", 512, label)?;

    let sample_hash = parse_metric(line, "sample_hash=")?;
    if sample_hash == 0 {
        return Err(String::from(
            "graphics desktop proof sample_hash must be nonzero",
        ));
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
            let screen_ppm =
                manifest_artifact_resolved_path(&fields, parent, "screen_ppm", Some("screen.ppm"))?;
            check_proof_screen(&screen_ppm)?;
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

fn check_current_proof_manifest(manifest_path: &Path, root: &Path) -> Result<(), String> {
    check_proof_manifest(manifest_path)?;
    let text = fs::read_to_string(manifest_path)
        .map_err(|e| format!("read {}: {e}", manifest_path.display()))?;
    let fields = parse_manifest_fields(&text)?;
    let (head, dirty) = current_git_state(root)?;
    check_manifest_provenance_fields(&fields, &head, dirty)
}

fn write_qemu_proof_manifest(
    run_path: &Path,
    image_path: &Path,
    backend: &str,
    seconds: u64,
    root: &Path,
) -> Result<PathBuf, String> {
    let (commit, dirty) = current_git_state(root)?;
    write_qemu_proof_manifest_with_provenance(
        run_path, image_path, backend, seconds, &commit, dirty,
    )
}

fn write_qemu_proof_manifest_with_provenance(
    run_path: &Path,
    image_path: &Path,
    backend: &str,
    seconds: u64,
    commit: &str,
    dirty: bool,
) -> Result<PathBuf, String> {
    if backend != "native" && backend != "wsl" && backend != "ci" {
        return Err(format!("unexpected qemu backend `{backend}`"));
    }
    if commit != "unknown" && (commit.len() != 40 || !is_lower_hex(commit)) {
        return Err(format!(
            "git commit must be full lowercase hex, got `{commit}`"
        ));
    }

    fs::create_dir_all(run_path).map_err(|e| format!("create {}: {e}", run_path.display()))?;
    let image_snapshot = run_path.join("seal-os.img");
    if image_path != image_snapshot && !image_snapshot.exists() {
        fs::copy(image_path, &image_snapshot).map_err(|e| {
            format!(
                "copy image {} -> {}: {e}",
                image_path.display(),
                image_snapshot.display()
            )
        })?;
    }

    let serial_log = run_path.join("serial.log");
    let screen_ppm = run_path.join("screen.ppm");
    let screen_png = run_path.join("screen.png");
    let efi_snapshot = run_path.join("seal-os.efi");

    let mut lines = Vec::new();
    lines.push(String::from("seal_proof_manifest_version=1"));
    lines.push(String::from("vm_target=qemu"));
    lines.push(format!("qemu_backend={backend}"));
    lines.push(String::from("proof_verdict=PASS"));
    lines.push(format!("proof_seconds={seconds}"));
    lines.push(format!("created_utc={}", current_utc_manifest_stamp()));
    lines.push(format!("git_commit={commit}"));
    lines.push(format!(
        "git_dirty={}",
        if dirty { "true" } else { "false" }
    ));
    lines.push(String::from("gate_verify=ok"));
    lines.push(String::from("gate_vm_proof=ok"));
    lines.push(String::from("gate_theorem_log=ok"));
    lines.push(String::from("gate_aether_runtime=ok"));
    lines.push(String::from("gate_desktop_soak=ok"));
    lines.push(String::from("gate_benchmark_log=ok"));
    lines.push(String::from("gate_proof_screen=ok"));

    push_manifest_artifact(&mut lines, "image", &image_snapshot)?;
    if efi_snapshot.exists() {
        push_manifest_artifact(&mut lines, "efi", &efi_snapshot)?;
    }
    push_manifest_artifact(&mut lines, "serial_log", &serial_log)?;
    push_manifest_artifact(&mut lines, "screen_ppm", &screen_ppm)?;
    if screen_png.exists() {
        push_manifest_artifact(&mut lines, "screen_png", &screen_png)?;
    }

    let manifest = run_path.join("proof-manifest.txt");
    fs::write(&manifest, lines.join("\n") + "\n")
        .map_err(|e| format!("write {}: {e}", manifest.display()))?;
    check_proof_manifest(&manifest)?;
    Ok(manifest)
}

fn current_utc_manifest_stamp() -> String {
    for (program, args) in [
        (
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')",
            ][..],
        ),
        ("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"][..]),
    ] {
        let output = Command::new(program).args(args).output();
        if let Ok(output) = output {
            if output.status.success() {
                let stamp = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if stamp.ends_with('Z') && stamp.contains('T') {
                    return stamp;
                }
            }
        }
    }
    String::from("2026-05-30T00:00:00Z")
}

fn push_manifest_artifact(
    lines: &mut Vec<String>,
    prefix: &str,
    path: &Path,
) -> Result<(), String> {
    let data = fs::read(path).map_err(|e| format!("read {prefix} {}: {e}", path.display()))?;
    if data.is_empty() {
        return Err(format!("manifest artifact `{prefix}` is empty"));
    }
    lines.push(format!("{prefix}_path={}", path.display()));
    lines.push(format!("{prefix}_bytes={}", data.len()));
    lines.push(format!("{prefix}_crc32={:08x}", crc32(&data)));
    lines.push(format!("{prefix}_sha256={}", sha256_hex(&data)));
    Ok(())
}

fn check_manifest_provenance_fields(
    fields: &BTreeMap<String, String>,
    expected_commit: &str,
    expected_dirty: bool,
) -> Result<(), String> {
    let manifest_commit = require_manifest_field(fields, "git_commit")?;
    if manifest_commit == "unknown" || manifest_commit != expected_commit {
        return Err(format!(
            "proof manifest git_commit does not match current HEAD: manifest={manifest_commit}, current={expected_commit}"
        ));
    }

    let manifest_dirty = require_manifest_field(fields, "git_dirty")?;
    let current_dirty = if expected_dirty { "true" } else { "false" };
    if manifest_dirty != current_dirty {
        return Err(format!(
            "proof manifest git_dirty does not match current checkout: manifest={manifest_dirty}, current={current_dirty}"
        ));
    }
    Ok(())
}

fn current_git_state(root: &Path) -> Result<(String, bool), String> {
    let head = Command::new("git")
        .arg("-C")
        .arg(root)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .map_err(|e| format!("run git rev-parse: {e}"))?;
    if !head.status.success() {
        return Err(format!(
            "git rev-parse failed: {}",
            String::from_utf8_lossy(&head.stderr).trim()
        ));
    }
    let head_text = String::from_utf8_lossy(&head.stdout).trim().to_string();
    if head_text.len() != 40 || !is_lower_hex(&head_text) {
        return Err(format!(
            "git HEAD is not a full lowercase SHA-1: {head_text}"
        ));
    }

    let status = Command::new("git")
        .arg("-C")
        .arg(root)
        .arg("status")
        .arg("--porcelain")
        .arg("--untracked-files=all")
        .output()
        .map_err(|e| format!("run git status: {e}"))?;
    if !status.status.success() {
        return Err(format!(
            "git status failed: {}",
            String::from_utf8_lossy(&status.stderr).trim()
        ));
    }
    Ok((head_text, !status.stdout.is_empty()))
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
    let bytes_key = format!("{prefix}_bytes");
    let crc_key = format!("{prefix}_crc32");
    let sha_key = format!("{prefix}_sha256");
    let path = manifest_artifact_resolved_path(fields, manifest_parent, prefix, bundled_name)?;
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
    let actual_sha = sha256_hex(&data);
    if actual_sha != expected_sha {
        return Err(format!(
            "manifest `{sha_key}` mismatch for {}: manifest={}, actual={actual_sha}",
            path.display(),
            expected_sha
        ));
    }

    Ok(())
}

fn manifest_artifact_resolved_path(
    fields: &BTreeMap<String, String>,
    manifest_parent: &Path,
    prefix: &str,
    bundled_name: Option<&str>,
) -> Result<PathBuf, String> {
    let path_key = format!("{prefix}_path");
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
    Ok(path)
}

fn parse_metric(line: &str, key: &str) -> Result<u64, String> {
    let value = parse_field(line, key)?;
    value
        .parse::<u64>()
        .map_err(|e| format!("invalid metric `{key}{value}`: {e}"))
}

fn parse_hex_metric(line: &str, key: &str) -> Result<u64, String> {
    let value = parse_field(line, key)?;
    let hex = value
        .strip_prefix("0x")
        .ok_or_else(|| format!("invalid hex metric `{key}{value}`: missing 0x prefix"))?;
    u64::from_str_radix(hex, 16).map_err(|e| format!("invalid hex metric `{key}{value}`: {e}"))
}

fn parse_field<'a>(line: &'a str, key: &str) -> Result<&'a str, String> {
    let value = line
        .split_whitespace()
        .find_map(|part| part.strip_prefix(key))
        .ok_or_else(|| format!("missing metric `{key}`"))?;
    Ok(value.trim_end_matches(','))
}

fn check_laamba_native_bridge(root: &Path) -> Result<(), String> {
    let laamba_app = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("apps")
        .join("laamba_governor.rs");
    require_existing_file(&laamba_app)?;

    require_file_patterns(
        root,
        &root
            .join("kernel")
            .join("seal-os")
            .join("src")
            .join("apps")
            .join("mod.rs"),
        &["pub mod laamba_governor;"],
    )?;
    require_file_patterns(
        root,
        &root
            .join("kernel")
            .join("seal-os")
            .join("src")
            .join("wm")
            .join("app_state.rs"),
        &[
            "use crate::apps::laamba_governor::LaambaGovernor;",
            "pub laamba: LaambaGovernor",
            "pub laamba_id: u32",
            "LaambaGovernor::new()",
            "compositor.create_window(\"LAAMBA Governor\"",
            "self.laamba_id",
        ],
    )?;
    require_file_patterns(
        root,
        &root
            .join("kernel")
            .join("seal-os")
            .join("src")
            .join("wm")
            .join("app_launcher.rs"),
        &["10 => crate::apps::laamba_governor::main"],
    )?;
    require_file_patterns(
        root,
        &root
            .join("kernel")
            .join("seal-os")
            .join("src")
            .join("wm")
            .join("start_menu.rs"),
        &["name: \"LAAMBA Governor\"", "app_id: 10"],
    )?;
    require_file_patterns(
        root,
        &root
            .join("kernel")
            .join("seal-os")
            .join("src")
            .join("wm")
            .join("desktop.rs"),
        &["name: \"LAAMBA\"", "app_id: 10"],
    )?;

    require_existing_file(
        &root
            .join("apps")
            .join("laamba-governor")
            .join("native")
            .join("governor.aether"),
    )?;

    let src_tauri = root.join("apps").join("laamba-governor").join("src-tauri");
    require_file_patterns(
        root,
        &src_tauri.join("src").join("native.rs"),
        &[
            "pub fn command(",
            "pub fn engine_command(",
            "\"aether-native\"",
        ],
    )?;
    require_file_patterns(
        root,
        &src_tauri.join("src").join("lib.rs"),
        &["pub mod native;"],
    )?;
    require_file_patterns(
        root,
        &src_tauri.join("src").join("ipc").join("commands.rs"),
        &["use crate::native;", "native::command(args)"],
    )?;
    require_file_patterns(
        root,
        &src_tauri.join("src").join("pipeline").join("executor.rs"),
        &["use crate::native;", "native::command(args)"],
    )?;
    require_file_patterns(
        root,
        &src_tauri.join("src").join("engine").join("manager.rs"),
        &["parts[0] == \"laamba-native\"", "--laamba-native-engine"],
    )?;
    require_file_patterns(
        root,
        &src_tauri.join("src").join("main.rs"),
        &[
            "--laamba-native-engine",
            "laamba_governor::native::engine_command",
        ],
    )?;
    require_engine_manifests_native(root)?;
    reject_laamba_python_bridge(root)
}

fn require_existing_file(path: &Path) -> Result<(), String> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("missing required file {}", path.display()))
    }
}

fn require_file_patterns(root: &Path, path: &Path, required: &[&str]) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|needle| !text.contains(needle))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        let rel = path.strip_prefix(root).unwrap_or(path);
        Err(format!(
            "{} missing LAAMBA native wiring: {}",
            rel.display(),
            missing.join(" | ")
        ))
    }
}

fn reject_laamba_python_bridge(root: &Path) -> Result<(), String> {
    let app_root = root.join("apps").join("laamba-governor");
    let mut scan_files = Vec::new();
    collect_files_recursive_skip_target(&app_root.join("src-tauri"), &mut scan_files)?;
    collect_direct_files_with_extension(&app_root.join("engines"), "toml", &mut scan_files)?;
    for rel in ["run.bat", "run.sh"] {
        let path = app_root.join(rel);
        if path.exists() {
            scan_files.push(path);
        }
    }

    let banned = [
        "python",
        "python3",
        "type = \"python\"",
        "command = \"python",
    ];
    let mut findings = Vec::new();
    for path in scan_files {
        let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let text = String::from_utf8_lossy(&bytes);
        for (idx, line) in text.lines().enumerate() {
            let lower = line.to_ascii_lowercase();
            let hits: Vec<&str> = banned
                .iter()
                .copied()
                .filter(|needle| lower.contains(needle))
                .collect();
            if !hits.is_empty() {
                let rel = path.strip_prefix(root).unwrap_or(&path);
                findings.push(format!(
                    "{}:{}: runtime Python bridge marker(s) `{}`: {}",
                    rel.display(),
                    idx + 1,
                    hits.join(" | "),
                    line.trim()
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

fn require_engine_manifests_native(root: &Path) -> Result<(), String> {
    let engines_dir = root.join("apps").join("laamba-governor").join("engines");
    let mut manifests = Vec::new();
    collect_direct_files_with_extension(&engines_dir, "toml", &mut manifests)?;
    if manifests.is_empty() {
        return Err(format!(
            "no engine manifests found in {}",
            engines_dir.display()
        ));
    }

    let mut findings = Vec::new();
    for path in manifests {
        let text =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        if !text.contains("type = \"native\"") {
            findings.push(format!(
                "{} missing `type = \"native\"`",
                path.strip_prefix(root).unwrap_or(&path).display()
            ));
        }
        if !text.contains("command = \"laamba-native ") {
            findings.push(format!(
                "{} missing `command = \"laamba-native ...\"`",
                path.strip_prefix(root).unwrap_or(&path).display()
            ));
        }
    }

    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn collect_files_recursive_skip_target(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if !dir.exists() {
        return Ok(());
    }
    let entries = fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir entry {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|s| s.to_str()) == Some("target") {
                continue;
            }
            collect_files_recursive_skip_target(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

fn collect_direct_files_with_extension(
    dir: &Path,
    extension: &str,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    if !dir.exists() {
        return Ok(());
    }
    let entries = fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir entry {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(extension) {
            out.push(path);
        }
    }
    Ok(())
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
    const SLAB_ALLOC_LOG: &str = "[BENCH] slab-alloc iterations=64 ok=64 classes=6 refill_ok=6 reuse_ok=6 free_ok=6 realloc_ok=2 p50_cycles=80 p95_cycles=120 max_cycles=150 refill_pages_delta=6 free_list_hits_delta=64 alloc_calls_delta=80 free_calls_delta=80 free_before=100 free_after_refill=94 free_after_reuse=94\n";
    const MANIFOLD_TELEPORT_LOG: &str = "[BENCH] manifold-teleport api=teleport fs_mode=mock_block persistence=metadata_only samples=3 ok=3 same_inode=3 src_gone=3 dst_present=3 entries_min=8 entries_max=256 payload_bytes=64 p50_cycles=1000 p95_cycles=2000 max_cycles=2000 ticks_max=1 metadata_ops_max=7 persistence_bytes_per_move=0 payload_points=4\n";
    const SCHEDULER_SELECT_LOG: &str = "[BENCH] scheduler-select-next selector=select_next_task mode=live_requeue clock=rdtsc iterations=64 ok=64 ready_before=3 ready_after=3 cells=8 priority_buckets=256 voronoi_locate_probes=8 max_cell_bitmap_tests=9 max_priority_bucket_scan=256 context_switches=0 selected_priority_max=10 p50_cycles=80 p95_cycles=120 max_cycles=140\n";
    const TCP_PACKET_DEMUX_LOG: &str = "[BENCH] tcp-packet-demux api=handle_tcp_packet fixture=listener_first accepted_state=established ok=1 listener_first=1 exact_flow=1 decoy_rx_bytes=0 listener_fallback=1 payload_bytes=4 rx_bytes=4 o1_index=1 index_hit=1 index_lookup_probes=1 index_probe_bound=256 index_capacity=256 listener_index_hit=1 listener_lookup_probes=1 listener_probe_bound=256 listener_index_capacity=256 exact_scan=0 cleanup=ok\n";
    const GPU_TOPOLOGY_BENCH_LOG: &str = "\
[GPU-BENCH] suite version=1 mode=cpu_fallback backend=software accelerator=cpu_fallback hardware_dispatch=0 shader_used=0 shader_status=placeholder_ok_not_claimed warmup=4 iterations=32 kernels=3 status=begin
[GPU-BENCH] kernel=voronoi_assign mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 n_points=64 n_cells=8 warmup=4 iterations=32 dispatch_ok=36 wait_ok=36 upload_ok=2 download_ok=1 verify=pass reference=cpu_recompute checked=64 mismatches=0 invalid_ids=0 checksum=12345 first_cell=0 avg_cycles=100
[GPU-BENCH] kernel=jl_project mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 n_vectors=4 dim_in=128 dim_out=3 seed=0xE95110A7 warmup=4 iterations=32 dispatch_ok=36 wait_ok=36 upload_ok=1 download_ok=1 verify=pass reference=cpu_recompute checked=12 mismatches=0 finite=12 checksum=23456 max_abs_diff_scaled=0 avg_cycles=200
[GPU-BENCH] kernel=spectral_step mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0 dim=512 alpha_ppm=300000 warmup=4 iterations=32 dispatch_ok=36 wait_ok=36 upload_ok=1 download_ok=1 verify=pass reference=cpu_recompute checked=512 mismatches=0 finite=512 checksum=34567 max_abs_diff_scaled=0 avg_cycles=300
[GPU-BENCH] suite version=1 mode=cpu_fallback backend=software hardware_dispatch=0 shader_used=0 kernels=3 passed=3 failed=0 result=pass claim=cpu_fallback_correctness_only
";
    const GFX_DESKTOP_PROOF_LOG: &str = "[GFX] desktop-proof version=1 surface=framebuffer width=1024 height=768 bpp=32 pitch=4096 back_buffer=1 window_count=12 focused_window_id=1 scanned_pixels=786432 nonblack_px=300000 visible_icons=10 icon_region_signal=23040 icon_color_buckets=10 control_region_signal=282000 primary_titlebar_signal=15912 start_button_signal=1440 theorem_indicator_signal=1000 minimized_app_lane_signal=1320 power_button_signal=384 sampled_pixels=3072 nonblack_samples=2048 sample_hash=123456789 result=pass\n";
    const GFX_DESKTOP_LIVE_PROOF_LOG: &str = "[GFX] desktop-live-proof version=1 route=desktop_handle_input action=desktop_icon_launch app=Files app_id=3 events=2 handled=1 icon_hit=1 launched_app_id=3 pre_focused=1 post_focused=9 post_window_id=9 window_count=12 pre_hash=111 post_hash=222 changed_samples=96 vram_hash=333 vram_changed_samples=96 vram_matches_backbuffer=96 blit=1 result=pass\n";
    const GFX_DESKTOP_SOAK_LOG: &str = "[GFX] desktop-soak frames=24 p50_cycles=100 p95_cycles=160 max_cycles=220 missed_16ms=unscaled input_events=0 dirty_px_max=786432\n";
    const AETHER_RUNTIME_LOG: &str = "[Aether-Lang] runtime proof: parser=ok interpreter=ok app_host=ok script=aether_boot_probe result=seal-topology-ok\n";
    const LAAMBA_APP_PROOF_LOG: &str = "[LAAMBA] app proof: version=1 native_app=kernel window=LAAMBA_Governor window_id=11 launcher_id=10 desktop_icon=1 start_menu=1 aether_host_window_id=12 runtime_bridge=rust_native_manifest python_runtime=0 result=pass\n";
    const SECURITY_HARDENING_LOG: &str = "[SECURITY] hardening proof version=1 kpti=1 kernel_cr3=0x1000 user_cr3=0x2000 kpti_distinct=1 user_lower_zero=1 kernel_upper_mirrored=1 smap_smep_supported=1 smap_smep_enabled=1 user_access_faults=0 result=pass\n";
    const SECURITY_AUDIT_LOG: &str = "[SECURITY] audit proof version=1 vfs=1 dirs=1 buffered_before=0 buffered_after=0 file=/var/log/audit.log readback=1 flushed=1 result=pass\n";
    const AUTH_SHADOW_PROOF_LOG: &str = "[SECURITY] auth proof version=1 shadow=1 default_user=seal default_present=1 default_topo5000=1 default_legacy=0 default_password_rejected=1 new_user_topo5000=1 passwd_embedded_hashes=0 result=pass\n";
    const COW_PROOF_LOG: &str = "[MM] cow-proof version=1 rollback_guard=1 fork_fallback=0 clone_fallback=0 samples=4 rollback_ok=4 tracked_frames=10 rollback_frees=10 leaked_frames=0 result=pass\n";
    const INSTALLER_PROOF_LOG: &str = "[INSTALLER] proof version=1 mode=safe_vfs selected_disk=nvme0 boot_marker=1 home=1 profile=1 user=1 auth_topo5000=1 raw_gpt=0 raw_format=0 result=pass\n";
    const MANIFOLDPKG_PROOF_LOG: &str = "[ManifoldPkg] proof version=1 source=embedded_eph parse=ok registry_index=ed25519_fixture install=ok extract=ok list=ok remove=ok files=1 bytes=19 package_count_before=0 package_count_after_install=1 package_count_after_remove=0 metadata_only=0 signature=ed25519_fixture result=pass\n";
    const UBUNTU_ALLOC_LOG: &str = "[UBUNTU-BENCH] alloc-frame os=ubuntu version_id=26.04 kernel=6.14.0-native iterations=64 ok=64 bytes=4096 backend=rust-std-box-page-touch-drop clock=rdtsc p50_cycles=200 p95_cycles=300 max_cycles=400\n";

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("seal-mkimage-{label}-{nanos}"))
    }

    fn proof_screen_ppm() -> Vec<u8> {
        let width = 1024usize;
        let height = 768usize;
        let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
        let pixels_start = ppm.len();
        ppm.resize(pixels_start + width * height * 3, 0);

        for y in 0..height {
            for x in 0..width {
                let (mut r, mut g, mut b) = (0u8, 0u8, 0u8);
                if x < 1_000 && (40..140).contains(&y) {
                    (r, g, b) = (48, 48, 48);
                }
                for icon_x in [20usize, 120, 220, 320, 420, 520, 620, 720, 820, 920] {
                    if (icon_x..icon_x + 48).contains(&x) && (50..98).contains(&y) {
                        let color = match icon_x {
                            20 => (255, 170, 68),
                            120 => (68, 204, 68),
                            220 => (68, 68, 204),
                            320 => (204, 204, 68),
                            420 => (204, 136, 68),
                            520 => (136, 68, 204),
                            620 => (204, 68, 68),
                            720 => (68, 204, 204),
                            820 => (204, 68, 204),
                            _ => (166, 227, 161),
                        };
                        (r, g, b) = color;
                    }
                }
                if (40..700).contains(&x) && (120..580).contains(&y) {
                    (r, g, b) = (36, 72, 48);
                }
                if (48..660).contains(&x) && (132..158).contains(&y) {
                    (r, g, b) = (48, 64, 160);
                }
                if y >= height - 28 {
                    (r, g, b) = (16, 16, 24);
                }
                if (4..76).contains(&x) && (height - 24..height - 4).contains(&y) {
                    (r, g, b) = (68, 136, 204);
                }
                for theorem_x in [90usize, 108, 126, 144, 162, 180, 198, 216, 234, 252] {
                    if (theorem_x..theorem_x + 10).contains(&x)
                        && (height - 20..height - 10).contains(&y)
                    {
                        (r, g, b) = (68, 136, 204);
                    }
                }
                for button_x in [350usize, 434, 518, 602, 686, 770] {
                    if (button_x..button_x + 80).contains(&x)
                        && (height - 24..height - 4).contains(&y)
                    {
                        (r, g, b) = (24, 32, 40);
                    }
                    if (button_x..button_x + 3).contains(&x)
                        && (height - 24..height - 4).contains(&y)
                    {
                        (r, g, b) = (68, 136, 204);
                    }
                }
                if (width - 36..width - 12).contains(&x) && (height - 22..height - 6).contains(&y) {
                    (r, g, b) = (255, 68, 68);
                }
                let offset = pixels_start + (y * width + x) * 3;
                ppm[offset] = r;
                ppm[offset + 1] = g;
                ppm[offset + 2] = b;
            }
        }

        ppm
    }

    fn weak_single_terminal_proof_screen_ppm() -> Vec<u8> {
        let width = 1024usize;
        let height = 768usize;
        let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
        let pixels_start = ppm.len();
        ppm.resize(pixels_start + width * height * 3, 0);

        for y in 0..height {
            for x in 0..width {
                let (mut r, mut g, mut b) = (0u8, 0u8, 0u8);
                if x < 1_000 && (40..140).contains(&y) {
                    (r, g, b) = (48, 48, 48);
                }
                if (40..700).contains(&x) && (120..580).contains(&y) {
                    (r, g, b) = (36, 72, 48);
                }
                if (48..660).contains(&x) && (132..158).contains(&y) {
                    (r, g, b) = (48, 64, 160);
                }
                if y >= height - 28 {
                    (r, g, b) = (16, 16, 24);
                }
                if (4..76).contains(&x) && (height - 24..height - 4).contains(&y) {
                    (r, g, b) = (68, 136, 204);
                }
                if (width - 36..width - 12).contains(&x) && (height - 22..height - 6).contains(&y) {
                    (r, g, b) = (255, 68, 68);
                }
                let offset = pixels_start + (y * width + x) * 3;
                ppm[offset] = r;
                ppm[offset + 1] = g;
                ppm[offset + 2] = b;
            }
        }

        ppm
    }

    fn manifest_artifact(prefix: &str, path: &Path, data: &[u8]) -> String {
        format!(
            "{prefix}_path={}\n{prefix}_bytes={}\n{prefix}_crc32={:08x}\n{prefix}_sha256={}\n",
            path.display(),
            data.len(),
            crc32(data),
            sha256_hex(data)
        )
    }

    fn write_laamba_native_fixture(root: &Path) {
        fs::create_dir_all(root.join("kernel").join("seal-os").join("src").join("apps")).unwrap();
        fs::create_dir_all(root.join("kernel").join("seal-os").join("src").join("wm")).unwrap();
        fs::create_dir_all(root.join("apps").join("laamba-governor").join("native")).unwrap();
        fs::create_dir_all(
            root.join("apps")
                .join("laamba-governor")
                .join("src-tauri")
                .join("src")
                .join("ipc"),
        )
        .unwrap();
        fs::create_dir_all(
            root.join("apps")
                .join("laamba-governor")
                .join("src-tauri")
                .join("src")
                .join("pipeline"),
        )
        .unwrap();
        fs::create_dir_all(
            root.join("apps")
                .join("laamba-governor")
                .join("src-tauri")
                .join("src")
                .join("engine"),
        )
        .unwrap();
        fs::create_dir_all(root.join("apps").join("laamba-governor").join("engines")).unwrap();

        fs::write(
            root.join("kernel")
                .join("seal-os")
                .join("src")
                .join("apps")
                .join("laamba_governor.rs"),
            "pub fn main() {}\npub struct LaambaGovernor;\n",
        )
        .unwrap();
        fs::write(
            root.join("kernel")
                .join("seal-os")
                .join("src")
                .join("apps")
                .join("mod.rs"),
            "pub mod laamba_governor;\n",
        )
        .unwrap();
        fs::write(
            root.join("kernel")
                .join("seal-os")
                .join("src")
                .join("wm")
                .join("app_state.rs"),
            "\
use crate::apps::laamba_governor::LaambaGovernor;
pub struct AppState { pub laamba: LaambaGovernor, pub laamba_id: u32 }
impl AppState {
    pub fn new() -> Self {
        let laamba = LaambaGovernor::new();
        Self { laamba, laamba_id: 0 }
    }
    pub fn create_windows(&mut self, compositor: &mut Compositor) {
        self.laamba_id = compositor.create_window(\"LAAMBA Governor\", 80, 40, 900, 650);
    }
}
",
        )
        .unwrap();
        fs::write(
            root.join("kernel")
                .join("seal-os")
                .join("src")
                .join("wm")
                .join("app_launcher.rs"),
            "10 => crate::apps::laamba_governor::main,\n",
        )
        .unwrap();
        fs::write(
            root.join("kernel")
                .join("seal-os")
                .join("src")
                .join("wm")
                .join("start_menu.rs"),
            "MenuItem { name: \"LAAMBA Governor\", app_id: 10 }\n",
        )
        .unwrap();
        fs::write(
            root.join("kernel")
                .join("seal-os")
                .join("src")
                .join("wm")
                .join("desktop.rs"),
            "DesktopIcon { name: \"LAAMBA\", app_id: 10 }\n",
        )
        .unwrap();
        fs::write(
            root.join("apps")
                .join("laamba-governor")
                .join("native")
                .join("governor.aether"),
            "// native LAAMBA governor script\n",
        )
        .unwrap();
        fs::write(
            root.join("apps")
                .join("laamba-governor")
                .join("src-tauri")
                .join("Cargo.toml"),
            "[package]\nname = \"laamba-governor\"\n",
        )
        .unwrap();
        fs::write(
            root.join("apps")
                .join("laamba-governor")
                .join("src-tauri")
                .join("src")
                .join("native.rs"),
            "pub fn command() {}\npub fn engine_command() {}\nlet runtime = \"aether-native\";\n",
        )
        .unwrap();
        fs::write(
            root.join("apps")
                .join("laamba-governor")
                .join("src-tauri")
                .join("src")
                .join("lib.rs"),
            "pub mod native;\n",
        )
        .unwrap();
        fs::write(
            root.join("apps")
                .join("laamba-governor")
                .join("src-tauri")
                .join("src")
                .join("ipc")
                .join("commands.rs"),
            "use crate::native;\nfn run(args: &[&str]) { native::command(args); }\n",
        )
        .unwrap();
        fs::write(
            root.join("apps")
                .join("laamba-governor")
                .join("src-tauri")
                .join("src")
                .join("pipeline")
                .join("executor.rs"),
            "use crate::native;\nfn run(args: &[&str]) { native::command(args); }\n",
        )
        .unwrap();
        fs::write(
            root.join("apps")
                .join("laamba-governor")
                .join("src-tauri")
                .join("src")
                .join("engine")
                .join("manager.rs"),
            "if parts[0] == \"laamba-native\" { cmd.arg(\"--laamba-native-engine\"); }\n",
        )
        .unwrap();
        fs::write(
            root.join("apps")
                .join("laamba-governor")
                .join("src-tauri")
                .join("src")
                .join("main.rs"),
            "if arg == \"--laamba-native-engine\" { laamba_governor::native::engine_command(); }\n",
        )
        .unwrap();
        fs::write(
            root.join("apps")
                .join("laamba-governor")
                .join("engines")
                .join("native.toml"),
            "type = \"native\"\ncommand = \"laamba-native demo\"\n",
        )
        .unwrap();
        fs::write(
            root.join("apps").join("laamba-governor").join("run.bat"),
            "@echo off\n",
        )
        .unwrap();
        fs::write(
            root.join("apps").join("laamba-governor").join("run.sh"),
            "#!/usr/bin/env sh\n",
        )
        .unwrap();
    }

    #[test]
    fn laamba_native_bridge_accepts_minimal_native_fixture() {
        let root = unique_temp_dir("laamba-native-ok");
        write_laamba_native_fixture(&root);

        assert!(check_laamba_native_bridge(&root).is_ok());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn laamba_native_bridge_rejects_python_runtime_bridge() {
        let root = unique_temp_dir("laamba-native-python");
        write_laamba_native_fixture(&root);
        fs::write(
            root.join("apps")
                .join("laamba-governor")
                .join("engines")
                .join("native.toml"),
            "type = \"python\"\ncommand = \"python3\"\n",
        )
        .unwrap();

        assert!(check_laamba_native_bridge(&root).is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sha256_matches_known_answer() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn proof_screen_rejects_trailing_payload() {
        let root = unique_temp_dir("proof-screen-trailing");
        fs::create_dir_all(&root).unwrap();
        let mut screen = proof_screen_ppm();
        screen.extend_from_slice(b"stale trailing bytes");
        let path = root.join("screen.ppm");
        fs::write(&path, screen).unwrap();

        assert!(check_proof_screen(&path).is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn proof_screen_rejects_missing_taskbar_controls() {
        let root = unique_temp_dir("proof-screen-taskbar-controls");
        fs::create_dir_all(&root).unwrap();
        let width = 1024usize;
        let height = 768usize;
        let mut screen = proof_screen_ppm();
        let pixels_start = screen
            .windows(b"\n255\n".len())
            .position(|window| window == b"\n255\n")
            .map(|pos| pos + b"\n255\n".len())
            .unwrap();

        for y in height - 28..height {
            for x in 0..width {
                let offset = pixels_start + (y * width + x) * 3;
                screen[offset] = 0;
                screen[offset + 1] = 0;
                screen[offset + 2] = 0;
            }
        }

        let path = root.join("screen.ppm");
        fs::write(&path, screen).unwrap();

        assert!(check_proof_screen(&path).is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn proof_screen_rejects_flat_single_window_without_desktop_structure() {
        let root = unique_temp_dir("proof-screen-flat-window");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("screen.ppm");
        fs::write(&path, weak_single_terminal_proof_screen_ppm()).unwrap();

        assert!(check_proof_screen(&path).is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn proof_screen_accepts_icon_row_and_minimized_app_lane() {
        let root = unique_temp_dir("proof-screen-desktop-structure");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("screen.ppm");
        fs::write(&path, proof_screen_ppm()).unwrap();

        assert!(check_proof_screen(&path).is_ok());

        fs::remove_dir_all(root).unwrap();
    }

    fn write_test_manifest(root: &Path) -> PathBuf {
        fs::create_dir_all(root).unwrap();
        let serial = b"serial proof log";
        let screen = proof_screen_ppm();
        let image = b"disk image bytes";
        let efi = b"efi image bytes";
        fs::write(root.join("serial.log"), serial).unwrap();
        fs::write(root.join("screen.ppm"), &screen).unwrap();
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
            &screen,
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

    fn full_benchmark_log() -> String {
        [
            ALLOC_PROOF_LOG,
            TOPORAM_ALLOC_LOG,
            SEAL_ALLOC_LOG,
            SLAB_ALLOC_LOG,
            MANIFOLD_TELEPORT_LOG,
            SCHEDULER_SELECT_LOG,
            TCP_PACKET_DEMUX_LOG,
            GPU_TOPOLOGY_BENCH_LOG,
        ]
        .concat()
    }

    fn write_test_benchmark_manifest(root: &Path, commit: &str, dirty: bool) -> PathBuf {
        fs::create_dir_all(root).unwrap();
        let serial = full_benchmark_log();
        let screen = proof_screen_ppm();
        let image = b"disk image bytes";
        let efi = b"efi image bytes";
        fs::write(root.join("serial.log"), serial.as_bytes()).unwrap();
        fs::write(root.join("screen.ppm"), &screen).unwrap();
        fs::write(root.join("seal-os.img"), image).unwrap();
        fs::write(root.join("seal-os.efi"), efi).unwrap();

        let mut manifest = format!(
            "seal_proof_manifest_version=1\n\
vm_target=qemu\n\
qemu_backend=ci\n\
proof_verdict=PASS\n\
proof_seconds=300\n\
created_utc=2026-05-30T00:00:00Z\n\
git_commit={commit}\n\
git_dirty={}\n\
gate_verify=ok\n\
gate_vm_proof=ok\n\
gate_theorem_log=ok\n\
gate_aether_runtime=ok\n\
gate_desktop_soak=ok\n\
gate_benchmark_log=ok\n\
gate_proof_screen=ok\n",
            if dirty { "true" } else { "false" }
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
            serial.as_bytes(),
        ));
        manifest.push_str(&manifest_artifact(
            "screen_ppm",
            &root.join("screen.ppm"),
            &screen,
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
    fn proof_manifest_rejects_sha256_mismatch() {
        let root = unique_temp_dir("manifest-sha-mismatch");
        let manifest = write_test_manifest(&root);
        let text = fs::read_to_string(&manifest).unwrap();
        let text = text.replace(
            &format!("serial_log_sha256={}", sha256_hex(b"serial proof log")),
            "serial_log_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        );
        fs::write(&manifest, text).unwrap();

        assert!(check_proof_manifest(&manifest).is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn proof_manifest_provenance_requires_current_commit() {
        let root = unique_temp_dir("manifest-provenance-commit");
        let manifest = write_test_manifest(&root);
        let fields = parse_manifest_fields(&fs::read_to_string(&manifest).unwrap()).unwrap();

        assert!(check_manifest_provenance_fields(
            &fields,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            false,
        )
        .is_ok());
        assert!(check_manifest_provenance_fields(
            &fields,
            "cccccccccccccccccccccccccccccccccccccccc",
            false,
        )
        .is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn proof_manifest_provenance_requires_current_dirty_flag() {
        let root = unique_temp_dir("manifest-provenance-dirty");
        let manifest = write_test_manifest(&root);
        let fields = parse_manifest_fields(&fs::read_to_string(&manifest).unwrap()).unwrap();

        assert!(check_manifest_provenance_fields(
            &fields,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            true,
        )
        .is_err());

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
    fn writes_qemu_proof_manifest_with_real_hashes() {
        let root = unique_temp_dir("manifest-writer");
        fs::create_dir_all(&root).unwrap();
        let screen = proof_screen_ppm();
        fs::write(root.join("serial.log"), b"serial proof log").unwrap();
        fs::write(root.join("screen.ppm"), &screen).unwrap();
        fs::write(root.join("screen.png"), b"png bytes").unwrap();
        fs::write(root.join("seal-os.img"), b"disk image bytes").unwrap();
        fs::write(root.join("seal-os.efi"), b"efi image bytes").unwrap();

        let manifest = write_qemu_proof_manifest_with_provenance(
            &root,
            &root.join("seal-os.img"),
            "ci",
            120,
            "dddddddddddddddddddddddddddddddddddddddd",
            false,
        )
        .unwrap();

        assert!(check_proof_manifest(&manifest).is_ok());
        assert!(check_manifest_provenance_fields(
            &parse_manifest_fields(&fs::read_to_string(&manifest).unwrap()).unwrap(),
            "dddddddddddddddddddddddddddddddddddddddd",
            false,
        )
        .is_ok());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn proof_manifest_accepts_relative_release_proof_paths() {
        let root = unique_temp_dir("manifest-relative-release-proof");
        let run_path = root.join("release-proof");
        fs::create_dir_all(&run_path).unwrap();

        let serial = b"serial proof log";
        let screen = proof_screen_ppm();
        let image = b"disk image bytes";
        let efi = b"efi image bytes";
        fs::write(run_path.join("serial.log"), serial).unwrap();
        fs::write(run_path.join("screen.ppm"), &screen).unwrap();
        fs::write(run_path.join("seal-os.img"), image).unwrap();
        fs::write(run_path.join("seal-os.efi"), efi).unwrap();

        let manifest = format!(
            "\
seal_proof_manifest_version=1
vm_target=qemu
qemu_backend=ci
proof_verdict=PASS
proof_seconds=240
created_utc=2026-06-02T00:00:00Z
git_commit=dddddddddddddddddddddddddddddddddddddddd
git_dirty=false
gate_verify=ok
gate_vm_proof=ok
gate_theorem_log=ok
gate_aether_runtime=ok
gate_desktop_soak=ok
gate_benchmark_log=ok
gate_proof_screen=ok
{}{}{}{}
",
            manifest_artifact("image", Path::new("release-proof/seal-os.img"), image),
            manifest_artifact("efi", Path::new("release-proof/seal-os.efi"), efi),
            manifest_artifact("serial_log", Path::new("release-proof/serial.log"), serial),
            manifest_artifact("screen_ppm", Path::new("release-proof/screen.ppm"), &screen),
        );
        let manifest_path = run_path.join("proof-manifest.txt");
        fs::write(&manifest_path, manifest).unwrap();

        assert!(check_proof_manifest(&manifest_path).is_ok());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn vm_proof_scripts_require_current_manifest_provenance() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let qemu_script = fs::read_to_string(
            repo_root
                .join("kernel")
                .join("seal-os")
                .join("run-qemu.ps1"),
        )
        .unwrap();
        let vbox_script = fs::read_to_string(
            repo_root
                .join("kernel")
                .join("seal-os")
                .join("smoke-vbox.ps1"),
        )
        .unwrap();
        let capture_script = fs::read_to_string(
            repo_root
                .join("scripts")
                .join("capture_qemu_proof_screen.sh"),
        )
        .unwrap();

        assert!(qemu_script.contains("--write-qemu-proof-manifest"));
        assert!(qemu_script.contains("--check-current-proof-manifest"));
        assert!(vbox_script.contains("--check-current-proof-manifest"));
        assert!(capture_script.contains("OVMF_VARS_4M.fd"));
        assert!(capture_script.contains("-drive if=pflash,format=raw,file=\"$OVMF_VARS_COPY\""));
        assert!(capture_script.contains("-monitor unix:\"$MON\",server,nowait"));
        assert!(capture_script.contains("-display none"));
        assert!(capture_script.contains("-device VGA"));
        assert!(capture_script.contains("screendump %s"));
    }

    #[test]
    fn vbox_smoke_waits_for_full_hard_serial_proof_markers() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let vbox_script = fs::read_to_string(
            repo_root
                .join("kernel")
                .join("seal-os")
                .join("smoke-vbox.ps1"),
        )
        .unwrap();

        for marker in [
            r"\[BENCH\] toporam-alloc",
            r"\[BENCH\] alloc-frame",
            r"\[BENCH\] manifold-teleport",
            r"\[BENCH\] scheduler-select-next",
            r"\[BENCH\] tcp-packet-demux",
            r"\[GPU-BENCH\] suite",
            r"\[Aether-Lang\] runtime proof:",
            r"\[LAAMBA\] app proof:",
            r"\[GFX\] desktop-proof",
            r"\[GFX\] desktop-live-proof",
            r"\[GFX\] desktop-soak",
        ] {
            assert!(
                vbox_script.contains(marker),
                "VirtualBox smoke success patterns must wait for {marker}"
            );
        }
    }

    #[test]
    fn release_workflow_contract_requires_proof_preflight_and_tag_discipline() {
        let good = "\
workflow_dispatch:
  inputs:
    tag:
      required: true
- name: Check release workflow contract
  run: cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-release-workflow-contract .
- name: Install QEMU and OVMF
  run: sudo apt-get install -y qemu-system-x86 ovmf socat
- name: Boot release image in QEMU and capture serial output
  run: |
    qemu-system-x86_64 -nographic | tee /tmp/seal-os-release.log
- name: Verify release VM proof
  run: cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-vm-proof /tmp/seal-os-release.log
- name: Verify release theorem proof
  run: cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-theorem-log /tmp/seal-os-release.log
- name: Verify release Aether proof
  run: cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-aether-runtime /tmp/seal-os-release.log
- name: Verify release LAAMBA app proof
  run: cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-laamba-app-proof /tmp/seal-os-release.log
- name: Verify release desktop soak
  run: cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-desktop-soak /tmp/seal-os-release.log
- name: Capture release proof screen in QEMU
  run: |
    qemu-system-x86_64 -serial file:/tmp/seal-os-screen.log -monitor unix:/tmp/seal-qemu-monitor.sock,server,nowait -display none -device VGA
    printf 'screendump /tmp/seal-os-screen.ppm\nquit\n' | socat - UNIX-CONNECT:/tmp/seal-qemu-monitor.sock
- name: Verify release proof screen
  run: cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-proof-screen /tmp/seal-os-screen.ppm
- name: Verify release benchmark proof
  run: cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-benchmark-log /tmp/seal-os-release.log
- name: Verify release source proof gates
  run: |
    cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-doc-claim-contract .
    cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-runtime-theorems .
    cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-seal-abi .
    cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-language-hygiene .
    cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-aether-migration .
    cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-o1-allocator .
    cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-o1-network .
- name: Stage release proof artifacts
  run: |
    mkdir -p release-proof
    cp /tmp/seal-os-release.log release-proof/serial.log
    cp /tmp/seal-os-screen.ppm release-proof/screen.ppm
    cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --write-qemu-proof-manifest release-proof seal-os.img ci 240 .
    cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-current-proof-manifest release-proof/proof-manifest.txt .
    printf 'QEMU serial and captured desktop pixel proof passed; not any-VM/hardware-GPU/Ubuntu proof; proof manifest provenance passed\\n' > release-proof/proof-summary.txt
- name: Upload release proof artifacts
  uses: actions/upload-artifact@v4
  with:
    name: seal-os-release-proof
    path: release-proof/
uses: softprops/action-gh-release@v2
with:
  tag_name: ${{ github.event.inputs.tag }}
  make_latest: true
  files: |
    kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img
    seal-os.iso
    release-proof/serial.log
    release-proof/screen.ppm
    release-proof/proof-manifest.txt
    release-proof/proof-summary.txt
  body: |
    QEMU serial and captured desktop pixel proof passed.
    Not an any-VM, hardware-GPU, or Ubuntu-superiority proof.
";

        assert!(check_release_workflow_contract_text(good).is_ok());

        let stale_tag = good.replace("required: true", "required: false");
        assert!(check_release_workflow_contract_text(&stale_tag).is_err());

        let latest_fallback = good.replace(
            "tag_name: ${{ github.event.inputs.tag }}",
            "tag_name: ${{ github.event.inputs.tag || 'latest' }}",
        );
        assert!(check_release_workflow_contract_text(&latest_fallback).is_err());

        let no_vm_proof = good.replace("--check-vm-proof /tmp/seal-os-release.log", "echo yep");
        assert!(check_release_workflow_contract_text(&no_vm_proof).is_err());

        let no_proof_asset = good.replace("release-proof/serial.log", "release-proof/missing.log");
        assert!(check_release_workflow_contract_text(&no_proof_asset).is_err());

        let no_screen_gate = good.replace("--check-proof-screen /tmp/seal-os-screen.ppm", "echo screen");
        assert!(check_release_workflow_contract_text(&no_screen_gate).is_err());

        let no_screen_asset = good.replace("release-proof/screen.ppm", "release-proof/missing.ppm");
        assert!(check_release_workflow_contract_text(&no_screen_asset).is_err());

        let no_manifest_writer = good.replace("--write-qemu-proof-manifest", "--skip-manifest");
        assert!(check_release_workflow_contract_text(&no_manifest_writer).is_err());

        let no_manifest_asset = good.replace(
            "release-proof/proof-manifest.txt",
            "release-proof/missing-manifest.txt",
        );
        assert!(check_release_workflow_contract_text(&no_manifest_asset).is_err());
    }

    #[test]
    fn release_046_public_version_surfaces_are_aligned() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let version = "0.4.6";
        let version_tag = "v0.4.6";
        let files = [
            (
                repo_root.join("kernel").join("seal-os").join("Cargo.toml"),
                format!("version = \"{version}\""),
            ),
            (
                repo_root
                    .join("kernel")
                    .join("seal-os")
                    .join("src")
                    .join("lib.rs"),
                format!("pub const VERSION: &str = \"{version}\""),
            ),
            (
                repo_root
                    .join("kernel")
                    .join("seal-os")
                    .join("src")
                    .join("graphics")
                    .join("splash.rs"),
                format!("                    {version_tag}"),
            ),
            (
                repo_root
                    .join("kernel")
                    .join("seal-os")
                    .join("src")
                    .join("wm")
                    .join("welcome.rs"),
                format!("Welcome to Seal OS {version_tag}"),
            ),
            (
                repo_root.join("README.md"),
                format!("Seal OS {version_tag}"),
            ),
            (
                repo_root
                    .join("kernel")
                    .join("seal-mkimage")
                    .join("src")
                    .join("main.rs"),
                format!("const EXPECTED_SEAL_OS_BANNER: &str = \"Seal OS {version_tag}\""),
            ),
            (
                repo_root.join("docs").join("CI.md"),
                format!("EXPECTED_SEAL_OS_VERSION=\"{version}\""),
            ),
            (
                repo_root.join("docs").join("CI.md"),
                format!("`Seal OS {version_tag}`"),
            ),
            (
                repo_root
                    .join("kernel")
                    .join("seal-os")
                    .join("src")
                    .join("drivers")
                    .join("net")
                    .join("http.rs"),
                String::from("User-Agent: SealOS/{}"),
            ),
            (
                repo_root
                    .join("kernel")
                    .join("seal-os")
                    .join("src")
                    .join("drivers")
                    .join("net")
                    .join("http.rs"),
                String::from("crate::VERSION"),
            ),
        ];

        for (path, needle) in files {
            let text = fs::read_to_string(&path).unwrap();
            assert!(
                text.contains(&needle),
                "{} missing `{}`",
                path.display(),
                needle
            );
        }
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
        assert!(parse_slab_alloc_benchmark(SLAB_ALLOC_LOG).is_ok());
        let seal = parse_seal_alloc_benchmark(SEAL_ALLOC_LOG).unwrap();
        let ubuntu = parse_ubuntu_alloc_benchmark(UBUNTU_ALLOC_LOG).unwrap();

        assert_eq!(seal.iterations, 64);
        assert_eq!(seal.p50, 100);
        assert_eq!(ubuntu.iterations, 64);
        assert_eq!(ubuntu.p95, 300);
    }

    #[test]
    fn rejects_weak_slab_allocator_proof() {
        assert!(parse_slab_alloc_benchmark(SLAB_ALLOC_LOG).is_ok());

        let missing_class = SLAB_ALLOC_LOG.replace("classes=6", "classes=5");
        assert!(parse_slab_alloc_benchmark(&missing_class).is_err());

        let no_reuse = SLAB_ALLOC_LOG.replace("free_list_hits_delta=64", "free_list_hits_delta=63");
        assert!(parse_slab_alloc_benchmark(&no_reuse).is_err());

        let extra_refill = SLAB_ALLOC_LOG.replace("free_after_reuse=94", "free_after_reuse=93");
        assert!(parse_slab_alloc_benchmark(&extra_refill).is_err());

        let no_realloc = SLAB_ALLOC_LOG.replace("realloc_ok=2", "realloc_ok=1");
        assert!(parse_slab_alloc_benchmark(&no_realloc).is_err());
    }

    #[test]
    fn benchmark_log_requires_gpu_topology_proof() {
        let no_gpu = format!(
            "{ALLOC_PROOF_LOG}{TOPORAM_ALLOC_LOG}{SEAL_ALLOC_LOG}{SLAB_ALLOC_LOG}{MANIFOLD_TELEPORT_LOG}{SCHEDULER_SELECT_LOG}{TCP_PACKET_DEMUX_LOG}"
        );

        assert!(check_benchmark_text(&no_gpu).is_err());

        let with_gpu = format!("{no_gpu}{GPU_TOPOLOGY_BENCH_LOG}");
        assert!(check_benchmark_text(&with_gpu).is_ok());
    }

    #[test]
    fn benchmark_log_rejects_fake_gpu_hardware_marker() {
        let fake_hardware = GPU_TOPOLOGY_BENCH_LOG.replace(
            "mode=cpu_fallback backend=software dispatch_path=cpu_sync hardware_dispatch=0 shader_used=0",
            "mode=hardware backend=amd_gcn dispatch_path=hardware hardware_dispatch=1 shader_used=1",
        );
        let log = format!(
            "{ALLOC_PROOF_LOG}{TOPORAM_ALLOC_LOG}{SEAL_ALLOC_LOG}{SLAB_ALLOC_LOG}{MANIFOLD_TELEPORT_LOG}{SCHEDULER_SELECT_LOG}{TCP_PACKET_DEMUX_LOG}{fake_hardware}"
        );

        assert!(check_benchmark_text(&log).is_err());
    }

    #[test]
    fn desktop_soak_requires_serial_graphics_desktop_proof() {
        assert!(check_desktop_soak_text(GFX_DESKTOP_SOAK_LOG).is_err());

        let no_live = format!("{GFX_DESKTOP_PROOF_LOG}{GFX_DESKTOP_SOAK_LOG}");
        assert!(check_desktop_soak_text(&no_live).is_err());

        let log =
            format!("{GFX_DESKTOP_PROOF_LOG}{GFX_DESKTOP_LIVE_PROOF_LOG}{GFX_DESKTOP_SOAK_LOG}");
        assert!(check_desktop_soak_text(&log).is_ok());
    }

    #[test]
    fn desktop_proof_rejects_blank_or_weak_framebuffer_claims() {
        let zero_hash = GFX_DESKTOP_PROOF_LOG.replace("sample_hash=123456789", "sample_hash=0");
        let weak_terminal = GFX_DESKTOP_PROOF_LOG
            .replace("control_region_signal=282000", "control_region_signal=7999");

        assert!(parse_graphics_desktop_proof(&zero_hash).is_err());
        assert!(parse_graphics_desktop_proof(&weak_terminal).is_err());
    }

    #[test]
    fn desktop_live_proof_rejects_noninteractive_or_unchanged_claims() {
        assert!(parse_graphics_desktop_live_proof(GFX_DESKTOP_LIVE_PROOF_LOG).is_ok());

        let no_route =
            GFX_DESKTOP_LIVE_PROOF_LOG.replace("route=desktop_handle_input", "route=direct_focus");
        let unchanged = GFX_DESKTOP_LIVE_PROOF_LOG.replace("post_hash=222", "post_hash=111");
        let wrong_focus = GFX_DESKTOP_LIVE_PROOF_LOG.replace("post_focused=9", "post_focused=1");
        let no_changes =
            GFX_DESKTOP_LIVE_PROOF_LOG.replace("changed_samples=96", "changed_samples=0");
        let no_blit = GFX_DESKTOP_LIVE_PROOF_LOG.replace("blit=1", "blit=0");
        let no_icon_hit = GFX_DESKTOP_LIVE_PROOF_LOG.replace("icon_hit=1", "icon_hit=0");
        let wrong_launch =
            GFX_DESKTOP_LIVE_PROOF_LOG.replace("launched_app_id=3", "launched_app_id=4");
        let missing_vram = GFX_DESKTOP_LIVE_PROOF_LOG.replace(
            " vram_hash=333 vram_changed_samples=96 vram_matches_backbuffer=96",
            "",
        );
        let zero_vram_hash = GFX_DESKTOP_LIVE_PROOF_LOG.replace("vram_hash=333", "vram_hash=0");
        let unchanged_vram =
            GFX_DESKTOP_LIVE_PROOF_LOG.replace("vram_changed_samples=96", "vram_changed_samples=0");
        let unmatched_vram = GFX_DESKTOP_LIVE_PROOF_LOG
            .replace("vram_matches_backbuffer=96", "vram_matches_backbuffer=0");
        let duplicate = format!("{GFX_DESKTOP_LIVE_PROOF_LOG}{GFX_DESKTOP_LIVE_PROOF_LOG}");
        let pass_then_fail = format!(
            "{}{}",
            GFX_DESKTOP_LIVE_PROOF_LOG,
            GFX_DESKTOP_LIVE_PROOF_LOG.replace("result=pass", "result=fail")
        );

        assert!(parse_graphics_desktop_live_proof(&no_route).is_err());
        assert!(parse_graphics_desktop_live_proof(&unchanged).is_err());
        assert!(parse_graphics_desktop_live_proof(&wrong_focus).is_err());
        assert!(parse_graphics_desktop_live_proof(&no_changes).is_err());
        assert!(parse_graphics_desktop_live_proof(&no_blit).is_err());
        assert!(parse_graphics_desktop_live_proof(&no_icon_hit).is_err());
        assert!(parse_graphics_desktop_live_proof(&wrong_launch).is_err());
        assert!(parse_graphics_desktop_live_proof(&missing_vram).is_err());
        assert!(parse_graphics_desktop_live_proof(&zero_vram_hash).is_err());
        assert!(parse_graphics_desktop_live_proof(&unchanged_vram).is_err());
        assert!(parse_graphics_desktop_live_proof(&unmatched_vram).is_err());
        assert!(parse_graphics_desktop_live_proof(&duplicate).is_err());
        assert!(parse_graphics_desktop_live_proof(&pass_then_fail).is_err());
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
    fn parses_laamba_boot_app_proof() {
        assert!(check_laamba_app_proof_text(LAAMBA_APP_PROOF_LOG).is_ok());

        let python_runtime = LAAMBA_APP_PROOF_LOG.replace("python_runtime=0", "python_runtime=1");
        let missing_window = LAAMBA_APP_PROOF_LOG.replace("window_id=11", "window_id=0");
        assert!(check_laamba_app_proof_text(&python_runtime).is_err());
        assert!(check_laamba_app_proof_text(&missing_window).is_err());
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

        let rewrote_bytes = MANIFOLD_TELEPORT_LOG.replace(
            "persistence_bytes_per_move=0",
            "persistence_bytes_per_move=64",
        );
        assert!(parse_manifold_teleport_benchmark(&rewrote_bytes).is_err());

        let legacy_write_through =
            MANIFOLD_TELEPORT_LOG.replace("persistence=metadata_only", "persistence=write_through");
        assert!(parse_manifold_teleport_benchmark(&legacy_write_through).is_err());

        let weak_ramfs_marker =
            MANIFOLD_TELEPORT_LOG.replace("fs_mode=mock_block", "fs_mode=ramfs");
        assert!(parse_manifold_teleport_benchmark(&weak_ramfs_marker).is_err());
    }

    #[test]
    fn parses_scheduler_select_benchmark() {
        assert!(parse_scheduler_select_benchmark(SCHEDULER_SELECT_LOG).is_ok());

        let regressed =
            SCHEDULER_SELECT_LOG.replace("max_cell_bitmap_tests=9", "max_cell_bitmap_tests=10");
        assert!(parse_scheduler_select_benchmark(&regressed).is_err());
    }

    #[test]
    fn parses_tcp_packet_demux_benchmark() {
        assert!(parse_tcp_packet_demux_benchmark(TCP_PACKET_DEMUX_LOG).is_ok());

        let failed = TCP_PACKET_DEMUX_LOG.replace("ok=1", "ok=0");
        assert!(parse_tcp_packet_demux_benchmark(&failed).is_err());

        let wrong_state =
            TCP_PACKET_DEMUX_LOG.replace("accepted_state=established", "accepted_state=listen");
        assert!(parse_tcp_packet_demux_benchmark(&wrong_state).is_err());

        let decoy_hit = TCP_PACKET_DEMUX_LOG.replace("decoy_rx_bytes=0", "decoy_rx_bytes=4");
        assert!(parse_tcp_packet_demux_benchmark(&decoy_hit).is_err());

        let no_listener_fallback =
            TCP_PACKET_DEMUX_LOG.replace("listener_fallback=1", "listener_fallback=0");
        assert!(parse_tcp_packet_demux_benchmark(&no_listener_fallback).is_err());

        let no_o1_index = TCP_PACKET_DEMUX_LOG.replace("o1_index=1", "o1_index=0");
        assert!(parse_tcp_packet_demux_benchmark(&no_o1_index).is_err());

        let no_index_hit = TCP_PACKET_DEMUX_LOG.replace("index_hit=1", "index_hit=0");
        assert!(parse_tcp_packet_demux_benchmark(&no_index_hit).is_err());

        let unbounded_lookup =
            TCP_PACKET_DEMUX_LOG.replace("index_lookup_probes=1", "index_lookup_probes=300");
        assert!(parse_tcp_packet_demux_benchmark(&unbounded_lookup).is_err());

        let no_listener_index =
            TCP_PACKET_DEMUX_LOG.replace("listener_index_hit=1", "listener_index_hit=0");
        assert!(parse_tcp_packet_demux_benchmark(&no_listener_index).is_err());

        let unbounded_listener =
            TCP_PACKET_DEMUX_LOG.replace("listener_lookup_probes=1", "listener_lookup_probes=300");
        assert!(parse_tcp_packet_demux_benchmark(&unbounded_listener).is_err());

        let used_exact_scan = TCP_PACKET_DEMUX_LOG.replace("exact_scan=0", "exact_scan=1");
        assert!(parse_tcp_packet_demux_benchmark(&used_exact_scan).is_err());
    }

    #[test]
    fn security_hardening_proof_requires_real_kpti_invariants() {
        assert!(check_security_hardening_text(SECURITY_HARDENING_LOG).is_ok());

        let no_kpti = SECURITY_HARDENING_LOG.replace("kpti=1", "kpti=0");
        assert!(check_security_hardening_text(&no_kpti).is_err());

        let same_cr3 = SECURITY_HARDENING_LOG.replace("user_cr3=0x2000", "user_cr3=0x1000");
        assert!(check_security_hardening_text(&same_cr3).is_err());

        let mapped_lower =
            SECURITY_HARDENING_LOG.replace("user_lower_zero=1", "user_lower_zero=0");
        assert!(check_security_hardening_text(&mapped_lower).is_err());

        let smap_claim =
            SECURITY_HARDENING_LOG.replace("smap_smep_enabled=1", "smap_smep_enabled=0");
        assert!(check_security_hardening_text(&smap_claim).is_err());
    }

    #[test]
    fn security_audit_proof_requires_vfs_flush_and_readback() {
        assert!(check_security_audit_text(SECURITY_AUDIT_LOG).is_ok());

        let no_vfs = SECURITY_AUDIT_LOG.replace("vfs=1", "vfs=0");
        assert!(check_security_audit_text(&no_vfs).is_err());

        let no_dirs = SECURITY_AUDIT_LOG.replace("dirs=1", "dirs=0");
        assert!(check_security_audit_text(&no_dirs).is_err());

        let buffered = SECURITY_AUDIT_LOG.replace("buffered_after=0", "buffered_after=8");
        assert!(check_security_audit_text(&buffered).is_err());

        let no_readback = SECURITY_AUDIT_LOG.replace("readback=1", "readback=0");
        assert!(check_security_audit_text(&no_readback).is_err());
    }

    #[test]
    fn auth_shadow_proof_requires_topo5000_default_and_no_passwd_hashes() {
        assert!(check_auth_shadow_proof_text(AUTH_SHADOW_PROOF_LOG).is_ok());

        let no_shadow = AUTH_SHADOW_PROOF_LOG.replace("shadow=1", "shadow=0");
        assert!(check_auth_shadow_proof_text(&no_shadow).is_err());

        let no_default = AUTH_SHADOW_PROOF_LOG.replace("default_present=1", "default_present=0");
        assert!(check_auth_shadow_proof_text(&no_default).is_err());

        let weak_default =
            AUTH_SHADOW_PROOF_LOG.replace("default_topo5000=1", "default_topo5000=0");
        assert!(check_auth_shadow_proof_text(&weak_default).is_err());

        let legacy_default = AUTH_SHADOW_PROOF_LOG.replace("default_legacy=0", "default_legacy=1");
        assert!(check_auth_shadow_proof_text(&legacy_default).is_err());

        let default_password_allowed =
            AUTH_SHADOW_PROOF_LOG.replace("default_password_rejected=1", "default_password_rejected=0");
        assert!(check_auth_shadow_proof_text(&default_password_allowed).is_err());

        let weak_new_user =
            AUTH_SHADOW_PROOF_LOG.replace("new_user_topo5000=1", "new_user_topo5000=0");
        assert!(check_auth_shadow_proof_text(&weak_new_user).is_err());

        let embedded_hashes =
            AUTH_SHADOW_PROOF_LOG.replace("passwd_embedded_hashes=0", "passwd_embedded_hashes=1");
        assert!(check_auth_shadow_proof_text(&embedded_hashes).is_err());
    }

    #[test]
    fn cow_proof_requires_rollback_guard_and_no_parent_fallback() {
        assert!(check_cow_proof_text(COW_PROOF_LOG).is_ok());

        let no_guard = COW_PROOF_LOG.replace("rollback_guard=1", "rollback_guard=0");
        assert!(check_cow_proof_text(&no_guard).is_err());

        let fork_fallback = COW_PROOF_LOG.replace("fork_fallback=0", "fork_fallback=1");
        assert!(check_cow_proof_text(&fork_fallback).is_err());

        let clone_fallback = COW_PROOF_LOG.replace("clone_fallback=0", "clone_fallback=1");
        assert!(check_cow_proof_text(&clone_fallback).is_err());

        let leak = COW_PROOF_LOG.replace("leaked_frames=0", "leaked_frames=1");
        assert!(check_cow_proof_text(&leak).is_err());

        let partial = COW_PROOF_LOG.replace("rollback_ok=4", "rollback_ok=3");
        assert!(check_cow_proof_text(&partial).is_err());

        let bad_accounting = COW_PROOF_LOG.replace("rollback_frees=10", "rollback_frees=9");
        assert!(check_cow_proof_text(&bad_accounting).is_err());
    }

    #[test]
    fn panic_serial_contract_rejects_formatting_macro_in_panic_handler() {
        let serial = r#"
pub fn panic_write_byte(byte: u8) {
    for _ in 0..PANIC_SERIAL_SPIN_LIMIT {
        if ready { write(byte); return; }
    }
}
pub fn panic_write_bytes(bytes: &[u8]) {}
pub fn panic_write_line(line: &str) {}
pub fn panic_write_fmt(args: fmt::Arguments) {}
"#;
        let main = r#"
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    seal_os::drivers::serial::panic_write_line("!!! SEAL OS KERNEL PANIC !!!");
    seal_os::drivers::serial::panic_write_fmt(format_args!("{}", info));
    loop { x86_64::instructions::hlt(); }
}
"#;
        assert!(check_panic_serial_source_contract_text(serial, main).is_ok());

        let macro_panic = main.replace(
            "seal_os::drivers::serial::panic_write_line(\"!!! SEAL OS KERNEL PANIC !!!\");",
            "seal_os::serial_println!(\"!!! SEAL OS KERNEL PANIC !!!\");",
        );
        assert!(check_panic_serial_source_contract_text(serial, &macro_panic).is_err());

        let missing_bound = serial.replace("PANIC_SERIAL_SPIN_LIMIT", "UNBOUNDED_WAIT");
        assert!(check_panic_serial_source_contract_text(&missing_bound, main).is_err());
    }

    #[test]
    fn installer_proof_requires_real_safe_vfs_writes_not_simulation() {
        assert!(check_installer_proof_text(INSTALLER_PROOF_LOG).is_ok());

        let no_boot = INSTALLER_PROOF_LOG.replace("boot_marker=1", "boot_marker=0");
        assert!(check_installer_proof_text(&no_boot).is_err());

        let no_auth = INSTALLER_PROOF_LOG.replace("auth_topo5000=1", "auth_topo5000=0");
        assert!(check_installer_proof_text(&no_auth).is_err());

        let claims_raw = INSTALLER_PROOF_LOG.replace("raw_gpt=0", "raw_gpt=1");
        assert!(check_installer_proof_text(&claims_raw).is_err());

        let simulation_log = format!("{INSTALLER_PROOF_LOG}[INSTALLER] Would format ESP\n");
        assert!(check_installer_proof_text(&simulation_log).is_err());
    }

    #[test]
    fn manifoldpkg_proof_requires_real_eph_extract_and_registry_counts() {
        assert!(check_manifoldpkg_proof_text(MANIFOLDPKG_PROOF_LOG).is_ok());

        let metadata_only = MANIFOLDPKG_PROOF_LOG.replace("metadata_only=0", "metadata_only=1");
        assert!(check_manifoldpkg_proof_text(&metadata_only).is_err());

        let no_extract = MANIFOLDPKG_PROOF_LOG.replace("extract=ok", "extract=fail");
        assert!(check_manifoldpkg_proof_text(&no_extract).is_err());

        let no_files = MANIFOLDPKG_PROOF_LOG.replace("files=1", "files=0");
        assert!(check_manifoldpkg_proof_text(&no_files).is_err());

        let no_bytes = MANIFOLDPKG_PROOF_LOG.replace("bytes=19", "bytes=0");
        assert!(check_manifoldpkg_proof_text(&no_bytes).is_err());

        let bad_count =
            MANIFOLDPKG_PROOF_LOG.replace("package_count_after_install=1", "package_count_after_install=0");
        assert!(check_manifoldpkg_proof_text(&bad_count).is_err());

        let skipped_sig =
            MANIFOLDPKG_PROOF_LOG.replace("signature=ed25519_fixture", "signature=skipped_fixture");
        assert!(check_manifoldpkg_proof_text(&skipped_sig).is_err());

        let unsigned_index =
            MANIFOLDPKG_PROOF_LOG.replace("registry_index=ed25519_fixture", "registry_index=none");
        assert!(check_manifoldpkg_proof_text(&unsigned_index).is_err());
    }

    #[test]
    fn doc_claim_contract_requires_ubuntu_and_benchmark_guards() {
        let readme = "\
not a blanket victory claim
Seal OS only claims a win over Ubuntu for a row after the same-machine benchmark exists
raw Ubuntu artifact pending
`--check-current-benchmark-proof`
persistence_bytes_per_move=0
fs_mode=mock_block
Where Seal OS must still prove superiority
Minimal TLS 1.3 PSK record path
no X.509/PKI/ECDHE gate yet
`signature=ed25519_fixture`
`registry_index=ed25519_fixture`
Public remote release channel is still pending
Read/write/create/mkdir/unlink/rmdir/rename/stat/readdir source paths are now `--check-doc-claim-contract` gated for both FAT and ext2
[SECURITY] audit proof
[MM] cow-proof
`seal`/`seal` is rejected
/var/log/audit.log
TopCrypt is topological encoding/obfuscation, not cryptographic protection
grid/value-height projection
`seal-mkimage --check-aether-runtime
[LAAMBA] app proof:
`seal-mkimage --check-benchmark-log
Hardware dispatch still needs a proof artifact
";
        let benchmark = "\
That claim is not
global and not automatic
Ubuntu comparison numbers are still pending, so no global Ubuntu win is claimed
[UBUNTU-BENCH] alloc-frame os=ubuntu version_id=26.04 kernel=<native-kernel>
WSL is rejected for benchmark evidence
The second gate is the claim gate
fs_mode=mock_block
LAAMBA app proof
";
        let ci = "\
`seal-mkimage --check-aether-runtime /tmp/seal-os.log`
`seal-mkimage --check-laamba-app-proof /tmp/seal-os.log`
`seal-mkimage --check-benchmark-log /tmp/seal-os.log`
`seal-mkimage --compare-benchmark-logs /tmp/seal-os.log ubuntu-alloc.log`
`seal-mkimage --check-current-benchmark-proof qemu-proof/proof-manifest.txt ubuntu-alloc.log .`
";
        let gpu_doc = "\
The current QEMU proof uses the CPU fallback
no current proof artifact establishes real GPU execution
hardware `[GPU-BENCH]` artifact proves otherwise
";

        assert!(check_doc_claim_contract_text(readme, benchmark, ci, gpu_doc).is_ok());

        let bad_readme = readme.replace("raw Ubuntu artifact pending", "Ubuntu artifact captured");
        assert!(check_doc_claim_contract_text(&bad_readme, benchmark, ci, gpu_doc).is_err());

        let any_vm_overclaim = format!("{readme}\nSeal OS runs on any VM.\n");
        assert!(check_doc_claim_contract_text(&any_vm_overclaim, benchmark, ci, gpu_doc).is_err());

        let ubuntu_variant_overclaim = format!("{readme}\nSeal OS surpasses Ubuntu.\n");
        assert!(
            check_doc_claim_contract_text(&ubuntu_variant_overclaim, benchmark, ci, gpu_doc)
                .is_err()
        );

        let default_credential_overclaim =
            format!("{readme}\nDefault credentials: `seal` / `seal`\n");
        assert!(
            check_doc_claim_contract_text(&default_credential_overclaim, benchmark, ci, gpu_doc)
                .is_err()
        );

        let gpu_overclaim = readme.replace(
            "Hardware dispatch still needs a proof artifact",
            "they execute (the GPU doesn't crash)",
        );
        assert!(check_doc_claim_contract_text(&gpu_overclaim, benchmark, ci, gpu_doc).is_err());

        let amd_shader_overclaim = format!(
            "{readme}\nThe AMD GPU compute path executes shader binaries that are stubs.\n"
        );
        assert!(
            check_doc_claim_contract_text(&amd_shader_overclaim, benchmark, ci, gpu_doc).is_err()
        );

        let gpu_doc_overclaim = gpu_doc.replace(
            "no current proof artifact establishes real GPU execution",
            "Seal OS offloads topological computations to discrete GPUs",
        );
        assert!(check_doc_claim_contract_text(readme, benchmark, ci, &gpu_doc_overclaim).is_err());
    }

    #[test]
    fn manifoldpkg_shell_contract_rejects_metadata_only_install() {
        let shell = r#"
fn cmd_install(&mut self, pkg: &str) -> String {
    if pkg.ends_with(".eph") {
        let Some(data) = self.fs.read_bytes(pkg, self.cwd) else { return String::new(); };
        return match crate::pkg::GLOBAL_PKG.lock().install_bytes(&data, None) {
            Ok(msg) => msg,
            Err(e) => e,
        };
    }
    crate::pkg::GLOBAL_PKG.lock().install(pkg).unwrap();
}
fn cmd_remove(&mut self, pkg: &str) -> String {
    crate::pkg::GLOBAL_PKG.lock().remove(pkg).unwrap()
}
fn cmd_packages(&self) -> String {
    let pkg = crate::pkg::GLOBAL_PKG.lock();
    for manifest in pkg.list() {
        manifest.carrier.name();
    }
    String::new()
}
"#;
        let pkg = "parse_eph(data); verify_signature(&pkg, key); self.install_file";
        let carrier = r#"
pub enum CarrierType { Aether, Rust, C, Js }
impl CarrierType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s { "aether" => Some(Self::Aether), "rust" => Some(Self::Rust), "c" => Some(Self::C), "js" => Some(Self::Js), _ => None }
    }
}
"#;
        assert!(check_manifoldpkg_shell_contract_text(shell, pkg, carrier).is_ok());

        let metadata_only = format!(
            "{shell}\nlet manifest = \"status=metadata-only\";\nstore_text(&format!(\"{{}}.pkg\", pkg), manifest, self.cwd);\nStored package manifest\n"
        );
        assert!(check_manifoldpkg_shell_contract_text(&metadata_only, pkg, carrier).is_err());

        let missing_core = "parse_eph(data);";
        assert!(check_manifoldpkg_shell_contract_text(shell, missing_core, carrier).is_err());

        let python_carrier = format!("{carrier}\npub struct PipCarrier;\n");
        assert!(check_manifoldpkg_shell_contract_text(shell, pkg, &python_carrier).is_err());
    }

    #[test]
    fn installer_source_contract_rejects_would_install_theater() {
        let installer = r#"
fn apply_safe_install(&self) -> bool {
    write_file_verified("/boot/EFI/BOOT/BOOTX64.EFI", b"boot");
    crate::security::passwd::add_user("seal2", "pw", 1001, 1001);
    crate::security::shadow::auth_shadow_proof();
    serial_println!("[INSTALLER] proof version=1 mode=safe_vfs raw_gpt=0 raw_format=0 result=pass");
    true
}
"#;
        assert!(check_installer_source_contract_text(installer).is_ok());

        let theater = format!("{installer}\nserial_println!(\"[INSTALLER] Would format ESP\");\n");
        assert!(check_installer_source_contract_text(&theater).is_err());
    }

    #[test]
    fn ide_completion_contract_rejects_stub_completion() {
        let ide = r#"
const COMPLETION_CANDIDATES: &[&str] = &["SphericalVoronoiIndex", "serial_println!"];
fn completion_suffix_for_line(line: Option<&String>, cursor_col: usize) -> Option<&'static str> {
    let prefix = current_word_prefix(line?, cursor_col);
    for candidate in COMPLETION_CANDIDATES {
        if candidate.starts_with(prefix.as_str()) && candidate.len() > prefix.len() {
            return candidate.get(prefix.len()..);
        }
    }
    None
}
fn current_word_prefix(line: &str, cursor_col: usize) -> String { String::from(line) }
fn completion_status(&self) -> String { String::from("Tab completes ln!") }
fn key_press(&mut self, ch: u8) {
    match ch {
        b'\t' => {
            if let Some(suffix) = completion_suffix_for_line(file.lines.get(self.cursor_line), self.cursor_col) {
                line.insert_str(self.cursor_col, suffix);
                self.cursor_col += suffix.len();
            }
        }
        _ => {}
    }
}
"#;
        assert!(check_ide_completion_source_contract_text(ide).is_ok());

        let stub = format!("{ide}\n//! AI code completion stub\n");
        assert!(check_ide_completion_source_contract_text(&stub).is_err());
    }

    #[test]
    fn filesystem_parity_contract_rejects_missing_write_paths() {
        let fat = r#"
//! FAT12/16/32 filesystem driver with VFS read/write/create/delete/rename support.
impl FileSystem for FatFs {
    fn read(&self, handle: VfsHandle, buf: &mut [u8], offset: u64) -> Result<usize, VfsError> { self.read_file(2, buf, offset) }
    fn write(&mut self, handle: VfsHandle, buf: &[u8], offset: u64) -> Result<usize, VfsError> { let written = self.write_clusters(2, buf, offset)?; self.write_dir_raw(0, buf)?; Ok(written) }
    fn create(&mut self, path: &str) -> Result<VfsHandle, VfsError> { self.add_dir_entry(0, path, 0x20, 2, 0)?; Ok(VfsHandle { fs_idx: 0, inode: 2 }) }
    fn mkdir(&mut self, path: &str) -> Result<VfsHandle, VfsError> { self.add_dir_entry(0, path, 0x10, 2, 0)?; self.write_dir_raw(2, &[])?; Ok(VfsHandle { fs_idx: 0, inode: 2 }) }
    fn unlink(&mut self, path: &str) -> Result<(), VfsError> { self.free_cluster_chain(2)?; self.remove_dir_entry(0, path) }
    fn rmdir(&mut self, path: &str) -> Result<(), VfsError> { self.remove_dir_entry(0, path) }
    fn rename(&mut self, old: &str, new: &str) -> Result<(), VfsError> { self.write_dir_raw(0, &[])?; Ok(()) }
    fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> { Ok(Vec::new()) }
    fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError> { todo!() }
}
"#;
        let ext2 = r#"
impl FileSystem for Ext2Fs {
    fn read(&self, handle: VfsHandle, buf: &mut [u8], offset: u64) -> Result<usize, VfsError> { self.read_inode_data(&self.read_inode(handle.inode as u32)?, offset, buf) }
    fn write(&mut self, handle: VfsHandle, buf: &[u8], offset: u64) -> Result<usize, VfsError> { let mut inode = self.read_inode(handle.inode as u32)?; let written = self.write_inode_data(&mut inode, offset, buf)?; self.write_inode(handle.inode as u32, &inode)?; Ok(written) }
    fn create(&mut self, path: &str) -> Result<VfsHandle, VfsError> { let ino = self.allocate_inode()?; self.write_inode(ino, &inode)?; self.add_dir_entry(2, path, ino, 1)?; Ok(VfsHandle { fs_idx: 0, inode: ino as u64 }) }
    fn mkdir(&mut self, path: &str) -> Result<VfsHandle, VfsError> { let ino = self.allocate_inode()?; let block = self.allocate_block()?; self.write_disk_block(block, &[])?; self.add_dir_entry(2, path, ino, 2)?; Ok(VfsHandle { fs_idx: 0, inode: ino as u64 }) }
    fn unlink(&mut self, path: &str) -> Result<(), VfsError> { self.free_inode_blocks(&mut inode)?; self.remove_dir_entry(2, path)?; self.free_inode(3) }
    fn rmdir(&mut self, path: &str) -> Result<(), VfsError> { self.free_inode_blocks(&mut inode)?; self.remove_dir_entry(2, path)?; self.free_inode(3) }
    fn rename(&mut self, old: &str, new: &str) -> Result<(), VfsError> { self.remove_dir_entry(2, old)?; self.add_dir_entry(2, new, 3, 1) }
    fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> { Ok(Vec::new()) }
    fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError> { todo!() }
    fn sync(&mut self) -> Result<(), VfsError> { self.buffer_cache.lock().flush_all() }
}
"#;
        assert!(check_filesystem_parity_source_contract_text(fat, ext2).is_ok());

        let read_only_fat = fat.replace("with VFS read/write/create/delete/rename support", "read-only v1");
        assert!(check_filesystem_parity_source_contract_text(&read_only_fat, ext2).is_err());

        let missing_ext2_write = ext2.replace("fn write(&mut self", "fn write_missing(&mut self");
        assert!(check_filesystem_parity_source_contract_text(fat, &missing_ext2_write).is_err());
    }

    #[test]
    fn comparison_requires_seal_to_win_all_recorded_percentiles() {
        let seal = parse_seal_alloc_benchmark(SEAL_ALLOC_LOG).unwrap();
        let ubuntu = parse_ubuntu_alloc_benchmark(UBUNTU_ALLOC_LOG).unwrap();

        assert!(require_seal_alloc_bench_lead(seal, ubuntu).is_ok());

        let faster_ubuntu = AllocBenchMetrics { p95: 120, ..ubuntu };
        assert!(require_seal_alloc_bench_lead(seal, faster_ubuntu).is_err());
    }

    #[test]
    fn proof_markers_must_be_unique() {
        let duplicate_seal_alloc = format!("{SEAL_ALLOC_LOG}{SEAL_ALLOC_LOG}");
        let duplicate_ubuntu_alloc = format!("{UBUNTU_ALLOC_LOG}{UBUNTU_ALLOC_LOG}");
        let duplicate_desktop_proof = format!("{GFX_DESKTOP_PROOF_LOG}{GFX_DESKTOP_PROOF_LOG}");

        assert!(parse_seal_alloc_benchmark(&duplicate_seal_alloc).is_err());
        assert!(parse_ubuntu_alloc_benchmark(&duplicate_ubuntu_alloc).is_err());
        assert!(parse_graphics_desktop_proof(&duplicate_desktop_proof).is_err());
    }

    #[test]
    fn benchmark_proof_requires_current_vm_manifest_and_native_ubuntu_artifact() {
        let root = unique_temp_dir("benchmark-proof-current");
        let commit = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
        let manifest = write_test_benchmark_manifest(&root, commit, false);
        let ubuntu = root.join("ubuntu-alloc.log");
        fs::write(&ubuntu, UBUNTU_ALLOC_LOG).unwrap();

        assert!(
            check_benchmark_proof_manifest_with_state(&manifest, &ubuntu, commit, false).is_ok()
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn benchmark_proof_rejects_stale_or_dirty_vm_manifest() {
        let root = unique_temp_dir("benchmark-proof-stale");
        let manifest =
            write_test_benchmark_manifest(&root, "ffffffffffffffffffffffffffffffffffffffff", false);
        let ubuntu = root.join("ubuntu-alloc.log");
        fs::write(&ubuntu, UBUNTU_ALLOC_LOG).unwrap();

        assert!(check_benchmark_proof_manifest_with_state(
            &manifest,
            &ubuntu,
            "1111111111111111111111111111111111111111",
            false,
        )
        .is_err());
        assert!(check_benchmark_proof_manifest_with_state(
            &manifest,
            &ubuntu,
            "ffffffffffffffffffffffffffffffffffffffff",
            true,
        )
        .is_err());

        fs::remove_dir_all(root).unwrap();
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
    let gpu_path = root.join("docs").join("GPU_ACCELERATION.md");
    let readme = fs::read_to_string(&readme_path)
        .map_err(|e| format!("read {}: {e}", readme_path.display()))?;
    let benchmark = fs::read_to_string(&benchmark_path)
        .map_err(|e| format!("read {}: {e}", benchmark_path.display()))?;
    let ci =
        fs::read_to_string(&ci_path).map_err(|e| format!("read {}: {e}", ci_path.display()))?;
    let gpu =
        fs::read_to_string(&gpu_path).map_err(|e| format!("read {}: {e}", gpu_path.display()))?;
    check_doc_claim_contract_text(&readme, &benchmark, &ci, &gpu)?;
    check_manifoldpkg_shell_contract(root)?;
    check_installer_source_contract(root)?;
    check_ide_completion_source_contract(root)?;
    check_filesystem_parity_source_contract(root)?;
    check_cow_source_contract(root)?;
    check_panic_serial_source_contract(root)
}

fn check_release_workflow_contract(root: &Path) -> Result<(), String> {
    let workflow_path = root.join(".github").join("workflows").join("release.yml");
    let workflow = fs::read_to_string(&workflow_path)
        .map_err(|e| format!("read {}: {e}", workflow_path.display()))?;
    check_release_workflow_contract_text(&workflow)
}

fn check_release_workflow_contract_text(workflow: &str) -> Result<(), String> {
    let required = [
        (
            "workflow_dispatch:",
            "release workflow must support explicit manual release dispatch",
        ),
        (
            "required: true",
            "manual release tag must be explicitly required",
        ),
        (
            "--check-release-workflow-contract .",
            "release workflow must self-check this contract before publishing",
        ),
        (
            "Install QEMU and OVMF",
            "release workflow must install VM proof dependencies",
        ),
        (
            "socat",
            "release workflow must install a monitor transport for captured QEMU proof screens",
        ),
        (
            "Boot release image in QEMU and capture serial output",
            "release workflow must boot the release image before publishing",
        ),
        (
            "/tmp/seal-os-release.log",
            "release workflow must capture a release serial proof log",
        ),
        (
            "--check-vm-proof /tmp/seal-os-release.log",
            "release workflow must run the VM proof gate",
        ),
        (
            "--check-theorem-log /tmp/seal-os-release.log",
            "release workflow must run theorem proof gate",
        ),
        (
            "--check-aether-runtime /tmp/seal-os-release.log",
            "release workflow must run Aether runtime proof gate",
        ),
        (
            "--check-laamba-app-proof /tmp/seal-os-release.log",
            "release workflow must run LAAMBA boot app proof gate",
        ),
        (
            "--check-desktop-soak /tmp/seal-os-release.log",
            "release workflow must run desktop soak proof gate",
        ),
        (
            "Capture release proof screen in QEMU",
            "release workflow must capture a real QEMU framebuffer dump",
        ),
        (
            "/tmp/seal-os-screen.ppm",
            "release workflow must capture a release proof-screen PPM",
        ),
        (
            "--check-proof-screen /tmp/seal-os-screen.ppm",
            "release workflow must run the captured proof-screen pixel gate",
        ),
        (
            "--write-qemu-proof-manifest",
            "release workflow must write a QEMU proof manifest for public artifacts",
        ),
        (
            "--check-current-proof-manifest release-proof/proof-manifest.txt .",
            "release workflow must bind the proof manifest to the current checkout",
        ),
        (
            "--check-benchmark-log /tmp/seal-os-release.log",
            "release workflow must run benchmark proof gate",
        ),
        (
            "--check-doc-claim-contract .",
            "release workflow must run doc claim contract",
        ),
        (
            "--check-runtime-theorems .",
            "release workflow must run runtime theorem source gate",
        ),
        (
            "--check-seal-abi .",
            "release workflow must run no-POSIX ABI source gate",
        ),
        (
            "--check-language-hygiene .",
            "release workflow must run host-language hygiene gate",
        ),
        (
            "--check-aether-migration .",
            "release workflow must run Aether migration gate",
        ),
        (
            "--check-o1-allocator .",
            "release workflow must run O(1) allocator source gate",
        ),
        (
            "--check-o1-network .",
            "release workflow must run O(1) network source gate",
        ),
        (
            "release-proof/serial.log",
            "release workflow must stage serial proof as release artifact",
        ),
        (
            "release-proof/screen.ppm",
            "release workflow must stage captured proof-screen as release artifact",
        ),
        (
            "release-proof/proof-manifest.txt",
            "release workflow must stage proof manifest as release artifact",
        ),
        (
            "release-proof/proof-summary.txt",
            "release workflow must stage honest proof summary as release artifact",
        ),
        (
            "seal-os-release-proof",
            "release workflow must upload release proof artifacts",
        ),
        (
            "tag_name: ${{ github.event.inputs.tag }}",
            "manual release must publish exactly the explicit tag input",
        ),
        (
            "make_latest: true",
            "release workflow must make the explicit release the GitHub latest release",
        ),
        (
            "QEMU serial and captured desktop pixel proof passed",
            "release body must state the proof that actually ran",
        ),
        (
            "Not an any-VM, hardware-GPU, or Ubuntu-superiority proof",
            "release body must deny broad unproven claims",
        ),
    ];

    let mut findings: Vec<String> = required
        .iter()
        .filter(|(needle, _)| !workflow.contains(needle))
        .map(|(needle, reason)| format!("missing `{needle}` ({reason})"))
        .collect();

    let banned = [
        ("required: false", "manual release tag must not be optional"),
        (
            "github.event.inputs.tag || 'latest'",
            "release workflow must not fall back to the mutable latest tag",
        ),
        (
            "github.event.inputs.tag || \"latest\"",
            "release workflow must not fall back to the mutable latest tag",
        ),
        (
            "tag_name: latest",
            "release workflow must not publish a mutable latest tag directly",
        ),
        (
            "O(1) topological surgery",
            "release body must scope filesystem O(1) claims to proof markers",
        ),
        (
            "~18ns/decision",
            "release body must not ship stale Aether-Link benchmark marketing",
        ),
        (
            "Full read-write drivers included",
            "release body must not claim full filesystem parity without fixtures",
        ),
        (
            "pure Rust, zero assembly",
            "release body must not make broad zero-assembly marketing claims",
        ),
        (
            "Not an any-VM, GPU, or Ubuntu-superiority proof",
            "release body must distinguish captured QEMU pixels from hardware-GPU proof",
        ),
    ];
    findings.extend(
        banned
            .iter()
            .filter(|(needle, _)| workflow.contains(needle))
            .map(|(needle, reason)| format!("banned `{needle}` ({reason})")),
    );

    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn check_doc_claim_contract_text(
    readme: &str,
    benchmark: &str,
    ci: &str,
    gpu_doc: &str,
) -> Result<(), String> {
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
            "`--check-current-benchmark-proof`",
            "README must bind Ubuntu comparison claims to a current proof manifest",
        ),
        (
            "README.md",
            readme,
            "persistence_bytes_per_move=0",
            "README must expose the metadata-only same-filesystem ManifoldFS proof marker",
        ),
        (
            "README.md",
            readme,
            "fs_mode=mock_block",
            "README must prove ManifoldFS teleport against the persistent mock block-store path",
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
            "`signature=ed25519_fixture`",
            "README must expose signed ManifoldPkg boot fixture proof",
        ),
        (
            "README.md",
            readme,
            "`registry_index=ed25519_fixture`",
            "README must expose signed ManifoldPkg registry index fixture proof",
        ),
        (
            "README.md",
            readme,
            "Public remote release channel is still pending",
            "README must expose missing ManifoldPkg remote release proof",
        ),
        (
            "README.md",
            readme,
            "[SECURITY] audit proof",
            "README must expose audit flush boot proof marker",
        ),
        (
            "README.md",
            readme,
            "[MM] cow-proof",
            "README must expose COW rollback/no-fallback proof marker",
        ),
        (
            "README.md",
            readme,
            "`seal`/`seal` is rejected",
            "README must expose the blocked default credential proof",
        ),
        (
            "README.md",
            readme,
            "/var/log/audit.log",
            "README must expose audit log VFS readback path",
        ),
        (
            "README.md",
            readme,
            "Read/write/create/mkdir/unlink/rmdir/rename/stat/readdir source paths are now `--check-doc-claim-contract` gated for both FAT and ext2",
            "README must expose filesystem parity source gate before mounted fixture parity is claimed",
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
            "[LAAMBA] app proof:",
            "README must expose the LAAMBA kernel app proof marker",
        ),
        (
            "README.md",
            readme,
            "`seal-mkimage --check-benchmark-log",
            "README must point benchmark claims at the audit gate",
        ),
        (
            "README.md",
            readme,
            "Hardware dispatch still needs a proof artifact",
            "README must keep GPU acceleration scoped to unproven hardware dispatch",
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
            "docs/BENCHMARK_PLAN.md",
            benchmark,
            "The second gate is the claim gate",
            "benchmark plan must identify current benchmark proof as the claim gate",
        ),
        (
            "docs/BENCHMARK_PLAN.md",
            benchmark,
            "fs_mode=mock_block",
            "benchmark plan must require the persistent mock block-store ManifoldFS proof marker",
        ),
        (
            "docs/BENCHMARK_PLAN.md",
            benchmark,
            "LAAMBA app proof",
            "benchmark plan must include LAAMBA boot app proof in the first milestone",
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
            "`seal-mkimage --check-laamba-app-proof /tmp/seal-os.log`",
            "CI docs must include the LAAMBA app proof gate",
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
        (
            "docs/CI.md",
            ci,
            "`seal-mkimage --check-current-benchmark-proof qemu-proof/proof-manifest.txt ubuntu-alloc.log .`",
            "CI docs must include the current benchmark proof gate",
        ),
        (
            "docs/GPU_ACCELERATION.md",
            gpu_doc,
            "The current QEMU proof uses the CPU fallback",
            "GPU docs must bind current proof to CPU fallback",
        ),
        (
            "docs/GPU_ACCELERATION.md",
            gpu_doc,
            "no current proof artifact establishes real GPU execution",
            "GPU docs must deny current hardware execution proof",
        ),
        (
            "docs/GPU_ACCELERATION.md",
            gpu_doc,
            "hardware `[GPU-BENCH]` artifact proves otherwise",
            "GPU docs must require hardware benchmark proof before GPU acceleration claims",
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
            "runs on any VM",
            "VM compatibility claim needs a concrete VM matrix and proof artifacts",
        ),
        (
            "Seal OS surpasses Ubuntu",
            "Ubuntu superiority claim needs workload and evidence",
        ),
        (
            "Seal OS is better than Ubuntu",
            "Ubuntu superiority claim needs workload and evidence",
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
            "metadata-only claim must stay scoped to same-filesystem proof markers",
        ),
        (
            "persistent write-through bytes still counted",
            "README must not retain stale ManifoldFS byte-rewrite wording",
        ),
        (
            "persistence=write_through",
            "README must not retain stale ManifoldFS write-through marker",
        ),
        (
            "fs_mode=ramfs",
            "README ManifoldFS benchmark must not fall back to ramfs-only proof",
        ),
        (
            "Default credentials: `seal` / `seal`",
            "README must not instruct users to use the blocked default credential",
        ),
        (
            "The default login is `seal`/`seal`",
            "README must not claim the blocked default credential is accepted",
        ),
        (
            "they execute (the GPU doesn't crash)",
            "GPU hardware execution needs a real hardware proof artifact",
        ),
        (
            "GPU offload ready",
            "GPU offload must stay scoped to scaffold until hardware proof exists",
        ),
        (
            "We can talk to AMD GPUs",
            "AMD GPU communication must not be claimed without hardware proof",
        ),
        (
            "The AMD GPU compute path executes shader binaries",
            "AMD shader execution must not be claimed without hardware proof",
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

    let banned_gpu_claims = [
        (
            "Seal OS offloads topological computations to discrete GPUs",
            "GPU offload is a target until hardware proof exists",
        ),
        (
            "they execute on the GPU without crashing",
            "GPU shader execution needs hardware proof",
        ),
        (
            "dispatches to GPU when batch size exceeds",
            "automatic GPU dispatch is not currently proven",
        ),
        (
            "runs a GPU kernel every N ticks",
            "scheduler GPU refresh is a future fast path",
        ),
    ];
    let banned_gpu_hits: Vec<String> = banned_gpu_claims
        .iter()
        .filter(|(needle, _)| gpu_doc.contains(needle))
        .map(|(needle, reason)| format!("docs/GPU_ACCELERATION.md: banned `{needle}` ({reason})"))
        .collect();
    if !banned_gpu_hits.is_empty() {
        return Err(banned_gpu_hits.join("\n"));
    }

    Ok(())
}

fn check_manifoldpkg_shell_contract(root: &Path) -> Result<(), String> {
    let shell_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("apps")
        .join("shell.rs");
    let pkg_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("pkg")
        .join("mod.rs");
    let carrier_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("pkg")
        .join("carrier.rs");
    let shell =
        fs::read_to_string(&shell_path).map_err(|e| format!("read {}: {e}", shell_path.display()))?;
    let pkg =
        fs::read_to_string(&pkg_path).map_err(|e| format!("read {}: {e}", pkg_path.display()))?;
    let carrier = fs::read_to_string(&carrier_path)
        .map_err(|e| format!("read {}: {e}", carrier_path.display()))?;
    check_manifoldpkg_shell_contract_text(&shell, &pkg, &carrier)
}

fn check_manifoldpkg_shell_contract_text(
    shell: &str,
    pkg: &str,
    carrier: &str,
) -> Result<(), String> {
    let mut findings = Vec::new();
    for needle in [
        "pkg.ends_with(\".eph\")",
        "self.fs.read_bytes(pkg, self.cwd)",
        "crate::pkg::GLOBAL_PKG.lock().install_bytes(&data, None)",
        "crate::pkg::GLOBAL_PKG.lock().install(pkg)",
        "crate::pkg::GLOBAL_PKG.lock().remove(pkg)",
        "pkg.list()",
        "manifest.carrier.name()",
    ] {
        if !shell.contains(needle) {
            findings.push(format!("shell ManifoldPkg path missing `{needle}`"));
        }
    }
    for needle in ["parse_eph(data)", "verify_signature(&pkg, key)", "self.install_file"] {
        if !pkg.contains(needle) {
            findings.push(format!("ManifoldPkg core missing `{needle}`"));
        }
    }
    for needle in [
        "CarrierType::Python",
        "Self::Python",
        "\"python\" | \"pip\"",
        "PipCarrier",
    ] {
        if pkg.contains(needle) || carrier.contains(needle) {
            findings.push(format!(
                "ManifoldPkg core exposes Python/Pip carrier marker `{needle}`"
            ));
        }
    }
    for needle in [
        "status=metadata-only",
        "store_text(&format!(\"{}.pkg\"",
        "packages are local metadata only",
        "Stored package manifest",
    ] {
        if shell.contains(needle) {
            findings.push(format!(
                "shell ManifoldPkg path still contains fake metadata install marker `{needle}`"
            ));
        }
    }

    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn check_cow_source_contract(root: &Path) -> Result<(), String> {
    let virt_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("memory")
        .join("virt.rs");
    let scheduler_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("process")
        .join("scheduler.rs");
    let lib_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("lib.rs");
    let virt =
        fs::read_to_string(&virt_path).map_err(|e| format!("read {}: {e}", virt_path.display()))?;
    let scheduler = fs::read_to_string(&scheduler_path)
        .map_err(|e| format!("read {}: {e}", scheduler_path.display()))?;
    let lib =
        fs::read_to_string(&lib_path).map_err(|e| format!("read {}: {e}", lib_path.display()))?;
    check_cow_source_contract_text(&virt, &scheduler, &lib)
}

fn check_cow_source_contract_text(
    virt: &str,
    scheduler: &str,
    lib: &str,
) -> Result<(), String> {
    let mut findings = Vec::new();
    for needle in [
        "struct CowCloneRollback",
        "fn track(&mut self, frame: PhysAddr)",
        "fn commit(mut self)",
        "impl Drop for CowCloneRollback",
        "fn cow_alloc_frame(",
        "clone_page_table_cow_with_budget",
        "COW_ROLLBACK_TRACKED_TOTAL",
        "COW_ROLLBACK_FREED_TOTAL",
        "rollback.track(",
        "rollback.commit()",
        "emit_cow_proof",
        "[MM] cow-proof version=1",
    ] {
        if !virt.contains(needle) {
            findings.push(format!("COW clone source missing `{needle}`"));
        }
    }
    for needle in [".unwrap_or(parent.page_table)"] {
        if scheduler.contains(needle) {
            findings.push(format!(
                "scheduler still falls back to parent page table after COW clone failure: `{needle}`"
            ));
        }
    }
    if !lib.contains("memory::virt::emit_cow_proof()") {
        findings.push(String::from("boot path does not emit COW proof marker"));
    }
    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn check_panic_serial_source_contract(root: &Path) -> Result<(), String> {
    let serial_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("drivers")
        .join("serial.rs");
    let main_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("main.rs");
    let serial = fs::read_to_string(&serial_path)
        .map_err(|e| format!("read {}: {e}", serial_path.display()))?;
    let main = fs::read_to_string(&main_path)
        .map_err(|e| format!("read {}: {e}", main_path.display()))?;
    check_panic_serial_source_contract_text(&serial, &main)
}

fn check_panic_serial_source_contract_text(serial: &str, main: &str) -> Result<(), String> {
    let mut findings = Vec::new();
    for needle in [
        "pub fn panic_write_byte(byte: u8)",
        "PANIC_SERIAL_SPIN_LIMIT",
        "pub fn panic_write_bytes(bytes: &[u8])",
        "pub fn panic_write_line(line: &str)",
        "pub fn panic_write_fmt(args: fmt::Arguments)",
    ] {
        if !serial.contains(needle) {
            findings.push(format!("panic serial source missing `{needle}`"));
        }
    }

    let panic_start = main
        .find("#[panic_handler]")
        .ok_or_else(|| String::from("kernel main missing #[panic_handler]"))?;
    let panic_body = &main[panic_start..];
    if panic_body.contains("serial_println!") || panic_body.contains("serial_print!") {
        findings.push(String::from(
            "panic handler must not use normal serial formatting macros",
        ));
    }
    for needle in [
        "panic_write_line(\"!!! SEAL OS KERNEL PANIC !!!\")",
        "panic_write_fmt(format_args!(\"{}\", info))",
    ] {
        if !panic_body.contains(needle) {
            findings.push(format!("panic handler missing `{needle}`"));
        }
    }

    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn check_installer_source_contract(root: &Path) -> Result<(), String> {
    let installer_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("apps")
        .join("installer.rs");
    let installer = fs::read_to_string(&installer_path)
        .map_err(|e| format!("read {}: {e}", installer_path.display()))?;
    check_installer_source_contract_text(&installer)
}

fn check_installer_source_contract_text(installer: &str) -> Result<(), String> {
    let mut findings = Vec::new();
    for needle in [
        "fn apply_safe_install(&self) -> bool",
        "write_file_verified(\"/boot/EFI/BOOT/BOOTX64.EFI\"",
        "crate::security::passwd::add_user",
        "crate::security::shadow::auth_shadow_proof",
        "[INSTALLER] proof version=1 mode=safe_vfs",
        "raw_gpt=0 raw_format=0",
    ] {
        if !installer.contains(needle) {
            findings.push(format!("installer missing `{needle}`"));
        }
    }
    for needle in [
        "Would create GPT",
        "Would format",
        "Would copy",
        "Would install",
        "Installation simulation complete",
        "SHA-256 hash",
    ] {
        if installer.contains(needle) {
            findings.push(format!(
                "installer still contains simulation marker `{needle}`"
            ));
        }
    }
    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn check_ide_completion_source_contract(root: &Path) -> Result<(), String> {
    let ide_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("apps")
        .join("seal_ide.rs");
    let ide =
        fs::read_to_string(&ide_path).map_err(|e| format!("read {}: {e}", ide_path.display()))?;
    check_ide_completion_source_contract_text(&ide)
}

fn check_ide_completion_source_contract_text(ide: &str) -> Result<(), String> {
    let mut findings = Vec::new();
    for needle in [
        "const COMPLETION_CANDIDATES",
        "fn completion_suffix_for_line",
        "fn current_word_prefix",
        "candidate.starts_with(prefix.as_str())",
        "return candidate.get(prefix.len()..)",
        "b'\\t'",
        "line.insert_str(self.cursor_col, suffix)",
        "self.cursor_col += suffix.len()",
        "Tab completes",
    ] {
        if !ide.contains(needle) {
            findings.push(format!("Seal IDE completion missing `{needle}`"));
        }
    }
    for needle in [
        "AI code completion stub",
        "completion stub",
        "stub completion",
        "TODO completion",
        "fake completion",
    ] {
        if ide.contains(needle) {
            findings.push(format!(
                "Seal IDE completion still contains theater marker `{needle}`"
            ));
        }
    }
    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
}

fn check_filesystem_parity_source_contract(root: &Path) -> Result<(), String> {
    let fat_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("fs")
        .join("fat.rs");
    let ext2_path = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("fs")
        .join("ext2.rs");
    let fat =
        fs::read_to_string(&fat_path).map_err(|e| format!("read {}: {e}", fat_path.display()))?;
    let ext2 = fs::read_to_string(&ext2_path)
        .map_err(|e| format!("read {}: {e}", ext2_path.display()))?;
    check_filesystem_parity_source_contract_text(&fat, &ext2)
}

fn check_filesystem_parity_source_contract_text(fat: &str, ext2: &str) -> Result<(), String> {
    let mut findings = Vec::new();
    for (label, text, needles) in [
        (
            "FAT",
            fat,
            &[
                "filesystem driver with VFS read/write/create/delete/rename support",
                "impl FileSystem for FatFs",
                "fn read(&self",
                "fn write(&mut self",
                "self.write_clusters",
                "self.write_dir_raw",
                "fn create(&mut self",
                "self.add_dir_entry",
                "fn mkdir(&mut self",
                "fn unlink(&mut self",
                "self.free_cluster_chain",
                "fn rmdir(&mut self",
                "fn rename(&mut self",
                "fn readdir(&self",
                "fn stat(&self",
            ] as &[&str],
        ),
        (
            "ext2",
            ext2,
            &[
                "impl FileSystem for Ext2Fs",
                "fn read(&self",
                "self.read_inode_data",
                "fn write(&mut self",
                "self.write_inode_data",
                "self.write_inode",
                "fn create(&mut self",
                "self.allocate_inode",
                "self.add_dir_entry",
                "fn mkdir(&mut self",
                "self.allocate_block",
                "self.write_disk_block",
                "fn unlink(&mut self",
                "self.free_inode_blocks",
                "self.remove_dir_entry",
                "fn rmdir(&mut self",
                "fn rename(&mut self",
                "fn readdir(&self",
                "fn stat(&self",
                "fn sync(&mut self",
            ] as &[&str],
        ),
    ] {
        for needle in needles {
            if !text.contains(needle) {
                findings.push(format!("{label} filesystem parity source missing `{needle}`"));
            }
        }
    }

    for needle in [
        "read-only v1",
        "write paths exist; fixture gate pending",
        "full filesystem parity",
    ] {
        if fat.contains(needle) {
            findings.push(format!("FAT source still contains stale marker `{needle}`"));
        }
    }

    if findings.is_empty() {
        Ok(())
    } else {
        Err(findings.join("\n"))
    }
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
        (
            "angular_sparse_attention.rs",
            "pub fn normalized_block_centroid",
        ),
        ("angular_sparse_attention.rs", "pub fn cosine_similarity"),
        (
            "angular_sparse_attention.rs",
            "pub struct AngularSparseAttention",
        ),
        (
            "angular_sparse_attention.rs",
            "pub fn generate_active_indices",
        ),
        ("angular_sparse_attention.rs", "pub fn sparse_flash_forward"),
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
        ("riemannian_optimizer.rs", "pub enum ManifoldType"),
        ("riemannian_optimizer.rs", "pub fn sphere_project_tangent"),
        ("riemannian_optimizer.rs", "pub fn sphere_retract"),
        (
            "riemannian_optimizer.rs",
            "pub fn parallel_transport_sphere",
        ),
        (
            "riemannian_optimizer.rs",
            "pub fn grassmann_project_tangent",
        ),
        ("riemannian_optimizer.rs", "pub struct RiemannianSgd"),
        ("riemannian_optimizer.rs", "pub struct RiemannianAdam"),
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
        ("angular_sparse_attention.py", "angular_sparse_attention.rs"),
        ("cross_manifold_alignment.py", "cross_manifold_alignment.rs"),
        ("geodesic_consolidation.py", "geodesic_consolidation.rs"),
        ("governor_convergence.py", "governor.rs"),
        ("hyperbolic_capacity.py", "hyperbolic_capacity.rs"),
        ("hyperbolic_geometry.py", "hyperbolic_geometry.rs"),
        ("meta_controller.py", "meta_controller.rs"),
        ("parallel_riemannian.py", "parallel_riemannian.rs"),
        ("persistent_kv_partition.py", "persistent_kv_partition.rs"),
        ("riemannian_optimizer.py", "riemannian_optimizer.rs"),
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
        "angular_sparse_attention",
        "cross_manifold_alignment",
        "geodesic_consolidation",
        "governor_convergence",
        "hyperbolic_capacity",
        "hyperbolic_geometry",
        "meta_controller",
        "parallel_riemannian",
        "persistent_kv_partition",
        "riemannian_optimizer",
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
        "normalized_block_centroid",
        "AngularSparseAttention",
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
        "sphere_project_tangent",
        "sphere_retract",
        "parallel_transport_sphere",
        "grassmann_project_tangent",
        "RiemannianSgd",
        "RiemannianAdam",
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

fn check_o1_network(root: &Path) -> Result<(), String> {
    let tcp = root
        .join("kernel")
        .join("seal-os")
        .join("src")
        .join("net")
        .join("tcp.rs");
    let text = fs::read_to_string(&tcp).map_err(|e| format!("read {}: {e}", tcp.display()))?;
    let required = [
        "pub const TCP_FLOW_INDEX_BUCKETS",
        "pub const TCP_FLOW_INDEX_MAX_PROBES",
        "struct TcpFlowIndex",
        "fn lookup_exact_flow_index",
        "fn index_exact_flow_socket",
        "pub const TCP_LISTENER_INDEX_BUCKETS",
        "pub const TCP_LISTENER_INDEX_MAX_PROBES",
        "struct TcpListenerIndex",
        "fn lookup_listener_index",
        "fn index_listener_socket",
        "pub fn tcp_listener_index_proof",
        "pub fn tcp_flow_index_proof",
        "TCP_FLOW_INDEX.lock().lookup",
        "TCP_LISTENER_INDEX.lock().lookup",
    ];
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|needle| !text.contains(needle))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "missing TCP O(1) demux hooks: {}",
            missing.join(" | ")
        ));
    }

    let handle_body = extract_between(&text, "pub fn handle_tcp_packet", "#[cfg(test)]")?;
    if !handle_body.contains("lookup_exact_flow_index(&sockets, src, src_port, dst_port)") {
        return Err(String::from(
            "handle_tcp_packet must route exact TCP flows through the bounded flow index",
        ));
    }
    if !handle_body.contains("lookup_listener_index(&sockets, dst_port)") {
        return Err(String::from(
            "handle_tcp_packet must route listener fallback through the bounded listener index",
        ));
    }
    reject_patterns(
        "handle_tcp_packet exact-flow path",
        handle_body,
        &[
            "sock.remote_port == src_port",
            "sock.remote_ip == src",
            "sock.local_port == dst_port && sock.state == TcpState::Listen",
            "exact_idx = sockets.iter()",
            "exact_idx = sockets\n            .iter()",
            "listener_idx = sockets.iter()",
            "listener_idx = sockets\n            .iter()",
        ],
    )?;

    let lookup_body = extract_between(
        &text,
        "fn lookup_exact_flow_index",
        "pub fn tcp_flow_index_proof",
    )?;
    reject_patterns(
        "lookup_exact_flow_index",
        lookup_body,
        &["sockets.iter()", "for sock in"],
    )?;
    if !lookup_body.contains("TCP_FLOW_INDEX.lock().lookup") || !lookup_body.contains(".get(idx)") {
        return Err(String::from(
            "lookup_exact_flow_index must use bounded index lookup plus direct socket-index validation",
        ));
    }

    let listener_lookup_body = extract_between(
        &text,
        "fn lookup_listener_index",
        "pub fn tcp_flow_index_proof",
    )?;
    reject_patterns(
        "lookup_listener_index",
        listener_lookup_body,
        &["sockets.iter()", "for sock in"],
    )?;
    if !listener_lookup_body.contains("TCP_LISTENER_INDEX.lock().lookup")
        || !listener_lookup_body.contains(".get(idx)")
    {
        return Err(String::from(
            "lookup_listener_index must use bounded index lookup plus direct socket-index validation",
        ));
    }

    let poll_body = extract_between(&text, "pub fn poll()", "pub fn handle_tcp_packet")?;
    if !poll_body.contains("lookup_listener_index(&sockets, dst_port)") {
        return Err(String::from(
            "tcp::poll must resolve pending SYN listener fallback through the bounded listener index",
        ));
    }
    reject_patterns(
        "tcp::poll listener fallback",
        poll_body,
        &[
            "s.local_port == dst_port",
            "sockets.iter().position",
            "sockets\n            .iter()\n            .position",
        ],
    )?;

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
    let actual = bytes.len().saturating_sub(idx);
    if actual != expected {
        return Err(format!(
            "PPM pixel payload size mismatch: expected {expected} bytes, got {actual}"
        ));
    }
    if width < 1_024 || height < 768 {
        return Err(format!("proof screen is too small: {width}x{height}"));
    }

    let pixels = &bytes[idx..idx + expected];
    let mut non_black = 0usize;
    let mut icon_region_signal = 0usize;
    let mut control_region_signal = 0usize;
    let mut primary_titlebar_signal = 0usize;
    let mut start_button_signal = 0usize;
    let mut theorem_indicator_signal = 0usize;
    let mut minimized_app_lane_signal = 0usize;
    let mut power_button_signal = 0usize;
    let mut icon_color_pixels = [0usize; 10];
    let mut icon_color_buckets = [false; 64];
    let icon_xs = [20usize, 120, 220, 320, 420, 520, 620, 720, 820, 920];

    for y in 0..height {
        for x in 0..width {
            let p = ((y * width + x) * 3) as usize;
            let r = pixels[p];
            let g = pixels[p + 1];
            let b = pixels[p + 2];
            let max_channel = r.max(g).max(b);
            let min_channel = r.min(g).min(b);
            if r > 8 || g > 8 || b > 8 {
                non_black += 1;
            }
            if x < 1_000 && (40..140).contains(&y) && (r > 32 || g > 32 || b > 32) {
                icon_region_signal += 1;
            }
            if (50..98).contains(&y) && max_channel >= 96 && max_channel - min_channel >= 40 {
                for (idx, icon_x) in icon_xs.iter().copied().enumerate() {
                    if (icon_x..icon_x + 48).contains(&x) {
                        icon_color_pixels[idx] += 1;
                        let bucket =
                            ((r as usize / 64) << 4) | ((g as usize / 64) << 2) | (b as usize / 64);
                        icon_color_buckets[bucket.min(icon_color_buckets.len() - 1)] = true;
                        break;
                    }
                }
            }
            if (40..700).contains(&x) && (120..580).contains(&y) && (r > 28 || g > 28 || b > 28) {
                control_region_signal += 1;
            }
            if (48..660).contains(&x) && (132..158).contains(&y) && b > 96 && g > 32 && r < 96 {
                primary_titlebar_signal += 1;
            }
            if (4..76).contains(&x)
                && (height - 24..height - 4).contains(&y)
                && b > 140
                && g > 80
                && r < 110
            {
                start_button_signal += 1;
            }
            if (90..270).contains(&x)
                && (height - 20..height - 10).contains(&y)
                && (max_channel >= 96 || max_channel - min_channel >= 40)
            {
                theorem_indicator_signal += 1;
            }
            if (350..width.saturating_sub(132)).contains(&x)
                && (height - 24..height - 4).contains(&y)
                && (max_channel >= 120 || (max_channel >= 64 && max_channel - min_channel >= 36))
            {
                minimized_app_lane_signal += 1;
            }
            if (width - 36..width - 12).contains(&x)
                && (height - 22..height - 6).contains(&y)
                && r > 180
                && g < 130
                && b < 130
            {
                power_button_signal += 1;
            }
        }
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
    let visible_icons = icon_color_pixels
        .iter()
        .copied()
        .filter(|count| *count >= 1_200)
        .count();
    if visible_icons < 8 {
        return Err(format!(
            "desktop app icon row is not structured enough: {visible_icons} visible icons"
        ));
    }
    let icon_color_bucket_count = icon_color_buckets.iter().filter(|seen| **seen).count();
    if icon_color_bucket_count < 6 {
        return Err(format!(
            "desktop app icon row lacks color diversity: {icon_color_bucket_count} color buckets"
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
    if start_button_signal < 1_000 {
        return Err(format!(
            "taskbar start button is not visible enough: {start_button_signal} signal pixels"
        ));
    }
    if theorem_indicator_signal < 500 {
        return Err(format!(
            "taskbar theorem indicators are not visible enough: {theorem_indicator_signal} signal pixels"
        ));
    }
    if minimized_app_lane_signal < 300 {
        return Err(format!(
            "taskbar minimized-app lane is not visible enough: {minimized_app_lane_signal} signal pixels"
        ));
    }
    if power_button_signal < 220 {
        return Err(format!(
            "taskbar power button is not visible enough: {power_button_signal} signal pixels"
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
        || line.contains("python_runtime=0")
        || line.contains("without python")
        || line.contains("rejects python-backed")
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
