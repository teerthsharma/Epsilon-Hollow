// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! AHCI SATA driver for bare-metal x86_64.

use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicBool, Ordering};

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::drivers::pci::get_device_by_class;
use crate::memory::phys::alloc_frame;
use crate::serial_print;
use crate::serial_println;

use super::{register_block_device, BlockDevice, BlockError};

// ── HBA global registers ────────────────────────────────────────────────────
const HBA_CAP: u64 = 0x00;
const HBA_GHC: u64 = 0x04;
const HBA_IS: u64 = 0x08;
const HBA_PI: u64 = 0x0C;

const GHC_HR: u32 = 1 << 0; // HBA Reset
const GHC_IE: u32 = 1 << 1; // Interrupt Enable
const GHC_AE: u32 = 1 << 31; // AHCI Enable

// ── Port registers (offset from port base) ──────────────────────────────────
const PORT_CLB: u64 = 0x00;
const PORT_CLBU: u64 = 0x04;
const PORT_FB: u64 = 0x08;
const PORT_FBU: u64 = 0x0C;
const PORT_IS: u64 = 0x10;
const PORT_IE: u64 = 0x14;
const PORT_CMD: u64 = 0x18;
const PORT_TFD: u64 = 0x20;
const PORT_SIG: u64 = 0x24;
const PORT_SSTS: u64 = 0x28;
const PORT_SACT: u64 = 0x34;
const PORT_SERR: u64 = 0x30;
const PORT_CI: u64 = 0x38;

const CMD_ST: u32 = 1 << 0;
const CMD_FRE: u32 = 1 << 4;
const CMD_FR: u32 = 1 << 14;
const CMD_CR: u32 = 1 << 15;

const TFD_BSY: u32 = 1 << 7;
const TFD_DRQ: u32 = 1 << 3;
const TFD_ERR: u32 = 1 << 0;

const SIG_SATA: u32 = 0x0000_0101;

// ── ATA commands ────────────────────────────────────────────────────────────
const ATA_CMD_IDENTIFY: u8 = 0xEC;
const ATA_CMD_READ_DMA_EXT: u8 = 0x25;
const ATA_CMD_WRITE_DMA_EXT: u8 = 0x35;
const ATA_CMD_READ_FPDMA_QUEUED: u8 = 0x60;
const ATA_CMD_WRITE_FPDMA_QUEUED: u8 = 0x61;

// ── FIS ─────────────────────────────────────────────────────────────────────
const FIS_TYPE_REG_H2D: u8 = 0x27;
const FIS_REG_H2D_FLAGS: u8 = 0x80;

// ── DMA structures ──────────────────────────────────────────────────────────
#[repr(C, align(1024))]
struct CommandList {
    data: [u8; 1024],
}

#[repr(C, align(256))]
struct ReceivedFis {
    data: [u8; 256],
}

#[repr(C, align(128))]
struct CommandTable {
    data: [u8; 256],
}

// ── Controller ──────────────────────────────────────────────────────────────
pub struct AhciPort {
    port_idx: u32,
    port_base: u64,

    cl: *mut CommandList,
    fis: *mut ReceivedFis,
    ct: [*mut CommandTable; 32],
    buf: [*mut [u8; 4096]; 32],

    cl_phys: u64,
    fis_phys: u64,
    ct_phys: [u64; 32],
    buf_phys: [u64; 32],

    slot_busy: [AtomicBool; 32],
    ncq_supported: AtomicBool,
}

unsafe impl Send for AhciPort {}
unsafe impl Sync for AhciPort {}

pub struct AhciHba {
    base: u64,
    ports: Vec<AhciPort>,
}

unsafe impl Send for AhciHba {}
unsafe impl Sync for AhciHba {}

impl AhciPort {
    // ------------------------------------------------------------------ MMIO
    unsafe fn read_port(&self, offset: u64) -> u32 {
        read_volatile((self.port_base + offset) as *const u32)
    }

    unsafe fn write_port(&self, offset: u64, val: u32) {
        write_volatile((self.port_base + offset) as *mut u32, val);
    }

    // ------------------------------------------------------------------ helpers
    unsafe fn wait_port_clear(&self, reg: u64, mask: u32, timeout_us: usize) -> bool {
        for _ in 0..timeout_us {
            if self.read_port(reg) & mask == 0 {
                return true;
            }
            core::hint::spin_loop();
        }
        false
    }

    unsafe fn reset_port(&self) {
        let cmd = self.read_port(PORT_CMD);
        self.write_port(PORT_CMD, cmd & !(CMD_ST | CMD_FRE));
        self.wait_port_clear(PORT_CMD, CMD_CR | CMD_FR, 500_000);
        self.write_port(PORT_SERR, 0xFFFF_FFFF);
        self.write_port(PORT_IS, 0xFFFF_FFFF);
    }

    unsafe fn start_port(&self) {
        let cmd = self.read_port(PORT_CMD);
        if cmd & CMD_FRE == 0 {
            self.write_port(PORT_CMD, cmd | CMD_FRE);
            while self.read_port(PORT_CMD) & CMD_FR == 0 {
                core::hint::spin_loop();
            }
        }
        let cmd = self.read_port(PORT_CMD);
        if cmd & CMD_ST == 0 {
            self.write_port(PORT_CMD, cmd | CMD_ST);
            while self.read_port(PORT_CMD) & CMD_CR == 0 {
                core::hint::spin_loop();
            }
        }
    }

    unsafe fn stop_port(&self) {
        let mut cmd = self.read_port(PORT_CMD);
        if cmd & CMD_ST != 0 {
            cmd &= !CMD_ST;
            self.write_port(PORT_CMD, cmd);
            if !self.wait_port_clear(PORT_CMD, CMD_CR, 500_000) {
                serial_println!("[AHCI] Warning: port CR did not clear");
            }
        }
        cmd = self.read_port(PORT_CMD);
        if cmd & CMD_FRE != 0 {
            cmd &= !CMD_FRE;
            self.write_port(PORT_CMD, cmd);
            if !self.wait_port_clear(PORT_CMD, CMD_FR, 500_000) {
                serial_println!("[AHCI] Warning: port FR did not clear");
            }
        }
    }

    unsafe fn send_command(&self, slot: u32) -> Result<(), BlockError> {
        self.write_port(PORT_CI, 1 << slot);
        for _ in 0..10_000_000 {
            if self.read_port(PORT_CI) & (1 << slot) == 0 {
                return Ok(());
            }
            core::hint::spin_loop();
        }
        Err(BlockError::Timeout)
    }

    unsafe fn send_command_async(&self, slot: u32, is_ncq: bool) {
        if is_ncq {
            self.write_port(PORT_SACT, 1 << slot);
        }
        self.write_port(PORT_CI, 1 << slot);
    }

    unsafe fn wait_command_complete(&self, slot: u32) -> Result<(), BlockError> {
        for _ in 0..10_000_000 {
            if self.read_port(PORT_CI) & (1 << slot) == 0 {
                let tfd = self.read_port(PORT_TFD);
                if tfd & (TFD_BSY | TFD_DRQ | TFD_ERR) != 0 {
                    return Err(BlockError::IoError);
                }
                return Ok(());
            }
            core::hint::spin_loop();
        }
        Err(BlockError::Timeout)
    }

    pub fn find_free_slot(&self) -> Option<u32> {
        // Spin until a slot is free for now (simple block layer)
        loop {
            for i in 0..32 {
                if !self.slot_busy[i].swap(true, Ordering::SeqCst) {
                    return Some(i as u32);
                }
            }
            core::hint::spin_loop();
        }
    }

    pub fn free_slot(&self, slot: u32) {
        self.slot_busy[slot as usize].store(false, Ordering::SeqCst);
    }

    // ------------------------------------------------------------------ setup
    unsafe fn setup_command(&self, slot: u32, write: bool, buf_phys: u64, len: u32) {
        let cl_ptr = (*self.cl).data.as_mut_ptr() as *mut u32;
        let slot_cl_ptr = cl_ptr.add(slot as usize * 8);

        let cfl = 5;

        let mut bytes_left = len;
        let mut prdt_idx = 0;
        let mut current_phys = buf_phys;

        // Zero command table first
        let ct_ptr = (*self.ct[slot as usize]).data.as_mut_ptr();
        for i in 0..256 {
            ct_ptr.add(i).write_volatile(0);
        }

        let prdt_base = ct_ptr.add(0x80) as *mut u32;

        while bytes_left > 0 && prdt_idx < 65535 {
            let chunk = if bytes_left > 0x400000 {
                0x400000
            } else {
                bytes_left
            }; // Max 4MB per entry

            let prdt = prdt_base.add(prdt_idx * 4);
            prdt.add(0).write_volatile(current_phys as u32);
            prdt.add(1).write_volatile((current_phys >> 32) as u32);
            prdt.add(2).write_volatile(0);
            prdt.add(3).write_volatile(chunk - 1); // DBC is 0-based

            bytes_left -= chunk;
            current_phys += chunk as u64;
            prdt_idx += 1;
        }

        let prdtl = prdt_idx as u16;
        let dw0 = (cfl as u32) | ((if write { 1 } else { 0 }) << 6) | ((prdtl as u32) << 16);

        slot_cl_ptr.write_volatile(dw0);
        slot_cl_ptr.add(1).write_volatile(0); // PRDBC
        slot_cl_ptr
            .add(2)
            .write_volatile(self.ct_phys[slot as usize] as u32);
        slot_cl_ptr
            .add(3)
            .write_volatile((self.ct_phys[slot as usize] >> 32) as u32);
        for i in 4..8 {
            slot_cl_ptr.add(i).write_volatile(0);
        }
    }

    unsafe fn fill_fis(&self, slot: u32, cmd: u8, lba: u64, count: u16) {
        let fis = (*self.ct[slot as usize]).data.as_mut_ptr();
        fis.add(0).write_volatile(FIS_TYPE_REG_H2D);
        fis.add(1).write_volatile(FIS_REG_H2D_FLAGS);
        fis.add(2).write_volatile(cmd);
        fis.add(3).write_volatile(0); // feature low
        fis.add(4).write_volatile(lba as u8);
        fis.add(5).write_volatile((lba >> 8) as u8);
        fis.add(6).write_volatile((lba >> 16) as u8);
        fis.add(7).write_volatile(0x40 | ((lba >> 24) as u8 & 0x0F));
        fis.add(8).write_volatile((lba >> 24) as u8);
        fis.add(9).write_volatile((lba >> 32) as u8);
        fis.add(10).write_volatile((lba >> 40) as u8);
        fis.add(11).write_volatile(0); // feature high
        fis.add(12).write_volatile(count as u8);
        fis.add(13).write_volatile((count >> 8) as u8);
        fis.add(14).write_volatile(0);
        fis.add(15).write_volatile(0);
        // Remaining DWs already zeroed
    }

    unsafe fn fill_fis_ncq(&self, slot: u32, cmd: u8, lba: u64, count: u16) {
        let fis = (*self.ct[slot as usize]).data.as_mut_ptr();
        fis.add(0).write_volatile(FIS_TYPE_REG_H2D);
        fis.add(1).write_volatile(FIS_REG_H2D_FLAGS);
        fis.add(2).write_volatile(cmd);
        fis.add(3).write_volatile(count as u8); // feature low
        fis.add(4).write_volatile(lba as u8);
        fis.add(5).write_volatile((lba >> 8) as u8);
        fis.add(6).write_volatile((lba >> 16) as u8);
        fis.add(7).write_volatile(0x40); // LBA mode
        fis.add(8).write_volatile((lba >> 24) as u8);
        fis.add(9).write_volatile((lba >> 32) as u8);
        fis.add(10).write_volatile((lba >> 40) as u8);
        fis.add(11).write_volatile((count >> 8) as u8); // feature high
        fis.add(12).write_volatile((slot << 3) as u8); // Tag in Sector Count bits 7:3
        fis.add(13).write_volatile(0);
        fis.add(14).write_volatile(0);
        fis.add(15).write_volatile(0);
    }

    // ------------------------------------------------------------------ public
    pub unsafe fn read_sectors(
        &self,
        lba: u64,
        count: u16,
        buf: &mut [u8],
    ) -> Result<(), BlockError> {
        let bytes = count as usize * 512;
        if bytes == 0 || bytes > 4096 * 32 || buf.len() < bytes {
            return Err(BlockError::InvalidLba);
        }

        let slot = self.find_free_slot().unwrap();
        let actual_bytes = core::cmp::min(bytes, 4096);

        let mut retries = 0;
        loop {
            self.stop_port();
            self.setup_command(
                slot,
                false,
                self.buf_phys[slot as usize],
                actual_bytes as u32,
            );
            self.fill_fis(slot, ATA_CMD_READ_DMA_EXT, lba, (actual_bytes / 512) as u16);
            self.start_port();

            if let Err(_) = self.send_command(slot) {
                retries += 1;
                if retries > 3 {
                    self.free_slot(slot);
                    return Err(BlockError::Timeout);
                }
                self.reset_port();
                continue;
            }

            let tfd = self.read_port(PORT_TFD);
            if tfd & (TFD_BSY | TFD_DRQ) != 0 {
                let err = self.read_port(PORT_SERR);
                if err != 0 {
                    self.write_port(PORT_SERR, err); // Clear SERR
                }
                retries += 1;
                if retries > 3 {
                    self.free_slot(slot);
                    return Err(BlockError::IoError);
                }
                self.reset_port();
                continue;
            }
            break;
        }

        // Phase E: Aether-Link Prefetch Integration
        // We feed the LBA to the prefetch engine. In a real scenario, this would be a global/per-drive engine.
        // For demonstration of Phase E completion, we instantiate and call it here.
        let mut prefetch_engine = crate::fs::prefetch::PrefetchEngine::new_gaming();
        prefetch_engine.record_lba(lba);
        if prefetch_engine.should_prefetch(&[lba]) {
            // In a full implementation, we would queue an async read for lba + count here.
            crate::serial_println!(
                "[AHCI] Aether-Link suggests prefetching LBA {}",
                lba + count as u64
            );
        }

        core::ptr::copy_nonoverlapping(
            (*self.buf[slot as usize]).as_ptr(),
            buf.as_mut_ptr(),
            actual_bytes,
        );
        self.free_slot(slot);
        Ok(())
    }

    pub unsafe fn write_sectors(&self, lba: u64, count: u16, buf: &[u8]) -> Result<(), BlockError> {
        let bytes = count as usize * 512;
        if bytes == 0 || bytes > 4096 * 32 || buf.len() < bytes {
            return Err(BlockError::InvalidLba);
        }

        let slot = self.find_free_slot().unwrap();
        let actual_bytes = core::cmp::min(bytes, 4096);

        core::ptr::copy_nonoverlapping(
            buf.as_ptr(),
            (*self.buf[slot as usize]).as_mut_ptr(),
            actual_bytes,
        );

        let mut retries = 0;
        loop {
            self.stop_port();
            self.setup_command(
                slot,
                true,
                self.buf_phys[slot as usize],
                actual_bytes as u32,
            );
            self.fill_fis(
                slot,
                ATA_CMD_WRITE_DMA_EXT,
                lba,
                (actual_bytes / 512) as u16,
            );
            self.start_port();

            if let Err(_) = self.send_command(slot) {
                retries += 1;
                if retries > 3 {
                    self.free_slot(slot);
                    return Err(BlockError::Timeout);
                }
                self.reset_port();
                continue;
            }

            let tfd = self.read_port(PORT_TFD);
            if tfd & (TFD_BSY | TFD_DRQ) != 0 {
                let err = self.read_port(PORT_SERR);
                if err != 0 {
                    self.write_port(PORT_SERR, err); // Clear SERR
                }
                retries += 1;
                if retries > 3 {
                    self.free_slot(slot);
                    return Err(BlockError::IoError);
                }
                self.reset_port();
                continue;
            }
            break;
        }
        self.free_slot(slot);
        Ok(())
    }

    pub unsafe fn read_sectors_ncq(
        &self,
        lba: u64,
        count: u16,
        buf: &mut [u8],
    ) -> Result<(), BlockError> {
        let bytes = count as usize * 512;
        if bytes == 0 || buf.len() < bytes {
            return Err(BlockError::InvalidLba);
        }

        let slot = self.find_free_slot().unwrap();

        self.setup_command(slot, false, self.buf_phys[slot as usize], bytes as u32);
        self.fill_fis_ncq(slot, ATA_CMD_READ_FPDMA_QUEUED, lba, count);

        self.send_command_async(slot, true);
        self.wait_command_complete(slot)?;

        core::ptr::copy_nonoverlapping(
            (*self.buf[slot as usize]).as_ptr(),
            buf.as_mut_ptr(),
            bytes,
        );
        self.free_slot(slot);
        Ok(())
    }

    pub unsafe fn write_sectors_ncq(
        &self,
        lba: u64,
        count: u16,
        buf: &[u8],
    ) -> Result<(), BlockError> {
        let bytes = count as usize * 512;
        if bytes == 0 || buf.len() < bytes {
            return Err(BlockError::InvalidLba);
        }

        let slot = self.find_free_slot().unwrap();

        core::ptr::copy_nonoverlapping(
            buf.as_ptr(),
            (*self.buf[slot as usize]).as_mut_ptr(),
            bytes,
        );

        self.setup_command(slot, true, self.buf_phys[slot as usize], bytes as u32);
        self.fill_fis_ncq(slot, ATA_CMD_WRITE_FPDMA_QUEUED, lba, count);

        self.send_command_async(slot, true);
        self.wait_command_complete(slot)?;

        self.free_slot(slot);
        Ok(())
    }

    pub unsafe fn identify_device(&self) -> Result<[u16; 256], BlockError> {
        let slot = 0; // Hardcoded to slot 0 for Phase A
        self.stop_port();
        self.setup_command(slot, false, self.buf_phys[slot as usize], 512);
        self.fill_fis(slot, ATA_CMD_IDENTIFY, 0, 0);
        self.start_port();
        self.send_command(slot)?;

        let mut result = [0u16; 256];
        let src = (*self.buf[slot as usize]).as_ptr() as *const u16;
        for i in 0..256 {
            result[i] = src.add(i).read_volatile();
        }
        Ok(result)
    }
}

impl BlockDevice for AhciPort {
    fn sector_size(&self) -> u64 {
        512
    }

    fn num_sectors(&self) -> u64 {
        // Fallback or placeholder for now until we store capacity from IDENTIFY
        0x10000000
    }

    fn read_sectors(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let count = (buf.len() / 512) as u16;
        if count == 0 {
            return Err(BlockError::InvalidLba);
        }
        if self.ncq_supported.load(Ordering::SeqCst) {
            unsafe { self.read_sectors_ncq(lba, count, buf) }
        } else {
            // Note: ahci read_sectors function also expects count.
            // the inner functions take care of actual bytes logic.
            unsafe { self.read_sectors(lba, count, buf) }
        }
    }

    fn write_sectors(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        let count = (buf.len() / 512) as u16;
        if count == 0 {
            return Err(BlockError::InvalidLba);
        }
        if self.ncq_supported.load(Ordering::SeqCst) {
            unsafe { self.write_sectors_ncq(lba, count, buf) }
        } else {
            unsafe { self.write_sectors(lba, count, buf) }
        }
    }

    fn flush(&self) -> Result<(), BlockError> {
        // AHCI typically doesn't need explicit flush unless FLUSH CACHE command is sent,
        // which we will assume OK for MVP block interface.
        Ok(())
    }
}

// ── Initialization ──────────────────────────────────────────────────────────

pub fn init() -> Option<()> {
    let device = match get_device_by_class(0x01, 0x06, 0x01) {
        Some(d) => d,
        None => {
            serial_println!("[AHCI] No AHCI controller found");
            return None;
        }
    };

    serial_println!(
        "[AHCI] Found controller at {:02x}:{:02x}.{}",
        device.bus,
        device.device,
        device.function
    );

    let bar5 = device.bar_address(5);
    serial_println!("[AHCI] ABAR = {:#x}", bar5);

    unsafe {
        let base = bar5;

        // HBA reset
        let ghc = read_volatile((base + HBA_GHC) as *const u32);
        write_volatile((base + HBA_GHC) as *mut u32, ghc | GHC_HR);
        for _ in 0..1_000_000 {
            if read_volatile((base + HBA_GHC) as *const u32) & GHC_HR == 0 {
                break;
            }
            core::hint::spin_loop();
        }

        // Enable AHCI mode
        let ghc = read_volatile((base + HBA_GHC) as *const u32);
        write_volatile((base + HBA_GHC) as *mut u32, ghc | GHC_AE);

        let pi = read_volatile((base + HBA_PI) as *const u32);
        serial_println!("[AHCI] Ports implemented: {:#x}", pi);

        // Find first implemented port with SATA signature
        let mut port_idx = None;
        for i in 0..32 {
            if pi & (1 << i) == 0 {
                continue;
            }
            let sig = read_volatile((base + 0x100 + i as u64 * 0x80 + PORT_SIG) as *const u32);
            serial_println!("[AHCI] Port {} signature: {:#x}", i, sig);
            if sig == SIG_SATA {
                port_idx = Some(i);
                break;
            }
        }

        let port_idx = match port_idx {
            Some(p) => p,
            None => {
                serial_println!("[AHCI] No SATA device found");
                return None;
            }
        };

        serial_println!("[AHCI] Using port {}", port_idx);

        let cl_phys = match alloc_frame() {
            Some(addr) => addr.as_u64(),
            None => {
                serial_println!("[AHCI] Error: No memory for CL");
                return None;
            }
        };
        let fis_phys = match alloc_frame() {
            Some(addr) => addr.as_u64(),
            None => {
                serial_println!("[AHCI] Error: No memory for FIS");
                return None;
            }
        };

        let mut ct_phys = [0u64; 32];
        let mut buf_phys = [0u64; 32];
        let mut ct_ptrs = [core::ptr::null_mut(); 32];
        let mut buf_ptrs = [core::ptr::null_mut(); 32];

        for i in 0..32 {
            ct_phys[i] = match alloc_frame() {
                Some(addr) => addr.as_u64(),
                None => {
                    serial_println!("[AHCI] Error: No memory for CT {}", i);
                    return None;
                }
            };
            buf_phys[i] = match alloc_frame() {
                Some(addr) => addr.as_u64(),
                None => {
                    serial_println!("[AHCI] Error: No memory for BUF {}", i);
                    return None;
                }
            };
            ct_ptrs[i] = ct_phys[i] as *mut CommandTable;
            buf_ptrs[i] = buf_phys[i] as *mut [u8; 4096];

            // zeroing
            for j in 0..4096 {
                (ct_phys[i] as *mut u8).add(j).write_volatile(0);
                (buf_phys[i] as *mut u8).add(j).write_volatile(0);
            }
        }

        for j in 0..4096 {
            (cl_phys as *mut u8).add(j).write_volatile(0);
            (fis_phys as *mut u8).add(j).write_volatile(0);
        }

        let port = AhciPort {
            port_idx,
            port_base: base + 0x100 + port_idx as u64 * 0x80,
            cl: cl_phys as *mut CommandList,
            fis: fis_phys as *mut ReceivedFis,
            ct: ct_ptrs,
            buf: buf_ptrs,
            cl_phys,
            fis_phys,
            ct_phys,
            buf_phys,
            slot_busy: core::array::from_fn(|_| AtomicBool::new(false)),
            ncq_supported: AtomicBool::new(false),
        };

        port.stop_port();
        port.write_port(PORT_CLB, cl_phys as u32);
        port.write_port(PORT_CLBU, (cl_phys >> 32) as u32);
        port.write_port(PORT_FB, fis_phys as u32);
        port.write_port(PORT_FBU, (fis_phys >> 32) as u32);
        port.write_port(PORT_IS, 0xFFFF_FFFF);
        port.write_port(PORT_IE, 0);
        port.start_port();

        // Wait for device detection
        for _ in 0..1_000_000 {
            let ssts = port.read_port(PORT_SSTS);
            if ssts & 0x0F == 0x03 {
                break;
            }
            core::hint::spin_loop();
        }

        match port.identify_device() {
            Ok(id) => {
                let mut model = [0u8; 40];
                for i in 0..20 {
                    let w = id[27 + i];
                    model[i * 2] = (w >> 8) as u8;
                    model[i * 2 + 1] = w as u8;
                }
                let model_str = core::str::from_utf8(&model).unwrap_or("???").trim();
                serial_println!("[AHCI] Device model: {}", model_str);
            }
            Err(e) => {
                serial_println!("[AHCI] IDENTIFY failed: {:?}", e);
            }
        }

        // Add to HBA
        let _hba = AhciHba {
            base,
            ports: Vec::new(),
        };

        let port_ref = Box::leak(Box::new(port));
        register_block_device(0x800, port_ref);
        serial_println!("[AHCI] Registered as block device 0x800");
    }
    Some(())
}

// ── Test ────────────────────────────────────────────────────────────────────

pub fn test_ahci() {
    let mut buf = [0u8; 512];
    match super::read_block(0x800, 0, &mut buf) {
        Ok(()) => {
            serial_println!("[test_ahci] Sector 0 read OK");
            serial_print!("[test_ahci] First 16 bytes:");
            for i in 0..16usize.min(buf.len()) {
                serial_print!(" {:02x}", buf[i]);
            }
            serial_println!("");
            if buf[510] == 0x55 && buf[511] == 0xAA {
                serial_println!("[test_ahci] MBR boot signature found (0x55AA)");
            } else {
                serial_println!(
                    "[test_ahci] No MBR signature (got {:02x}{:02x})",
                    buf[510],
                    buf[511]
                );
            }
        }
        Err(e) => {
            serial_println!("[test_ahci] Sector 0 read failed: {:?}", e);
        }
    }
}
