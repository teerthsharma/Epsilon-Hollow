# Plan: Seal OS 0.4.5 — Bootable Personal OS

## Objective
Transform Seal OS from a research kernel into a **bootable personal operating system** that can install to disk, log in users with passwords, launch applications from a desktop, and persist data across reboots. Version bump: 0.4.0 → 0.4.5.

## Target User
ML researchers and HFT professionals who want a geometry-native, mathematically rigorous personal OS.

---

## Phase 1: Bootable ISO + Disk Install

### 1.1 ISO Build Verification
- Fix `scripts/build_iso.sh` to work cross-platform (bash on Windows Git Bash)
- Update `kernel/seal-mkimage` to produce proper FAT12 EFI boot image
- Ensure `cargo +nightly build --release` produces bootable EFI binary

### 1.2 Disk Installer (`apps/installer.rs`)
- Create a full-screen installer app that runs from the live ISO
- Steps:
  1. Welcome screen with Seal OS branding
  2. Disk selection — enumerate block devices (NVMe, AHCI, USB MSC)
  3. Partitioning — create GPT with EFI System Partition (FAT32) and Root Partition
  4. Format ESP as FAT32, format root as ManifoldFS
  5. Copy kernel EFI binary to ESP/EFI/BOOT/BOOTX64.EFI
  6. Write GRUB config for disk boot
  7. Create user account (username + password)
  8. Install progress bar with topological animation
  9. Reboot prompt

### 1.3 GRUB Disk Boot Config
- `grub.cfg` for installed disk

---

## Phase 2: Password Login + User System

### 2.1 User Database
- File `/etc/passwd` on ManifoldFS: `username:uid:gid:home_dir:shell:password_hash`
- Password hashed with SHA-256 (v1)
- Default user: `seal` uid=1000 gid=1000

### 2.2 Login Screen (`graphics/login.rs`)
- Real password entry with username + password fields
- SHA-256 hash comparison against `/etc/passwd`
- Failed login shows red error

---

## Phase 3: Desktop App Launcher

### 3.1 Desktop Icons
- Grid layout of app icons on desktop
- Click launches app in new window

### 3.2 Start Menu
- Taskbar "Seal" button opens app menu

### 3.3 Window Management
- Multiple windows tracked by compositor
- Close button kills app task

---

## Phase 4: Persistent Settings + Power Management

### 4.1 Persistent Settings
- `/etc/sealos/settings.conf`
- Load on boot, save on change

### 4.2 Power Button
- Shutdown, Reboot, Logout in taskbar

---

## Phase 5: Version Bump + Polish

### 5.1 Version Bump
- `lib.rs`: `VERSION = "0.4.5"`
- `Cargo.toml`: update version
- `README.md`: update all version references
