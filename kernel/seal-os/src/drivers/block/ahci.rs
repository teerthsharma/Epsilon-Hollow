// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! AHCI SATA driver for bare-metal x86_64.

use core::ptr::{read_volatile, write_volatile};

use alloc::boxed::Box;

use crate::drivers::pci::get_device_by_class;
use crate::memory::phys::alloc_frame;
use crate::serial_print;
use crate::serial_println;

use super::{BlockDevice, BlockError, register_block_device};

// ── HBA global registers ────────────────────────────────────────────────────
const HBA_CAP: u64 = 0x00;
const HBA_GHC: u64 = 0x04;
const HBA_IS: u64 = 0x08;
const HBA_PI: u64 = 0x0C;

const GHC_HR: u32 = 1 << 0;   // HBA Reset
const GHC_IE: u32 = 1 << 1;   // Interrupt Enable
const GHC_AE: u32 = 1 << 31;  // AHCI Enable

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
const PORT_SERR: u64 = 0x30;
const PORT_CI: u64 = 0x38;

const CMD_ST: u32 = 1 << 0;
const CMD_FRE: u32 = 1 << 4;
const CMD_FR: u32 = 1 << 14;
const CMD_CR: u32 = 1 << 15;

const TFD_BSY: u32 = 1 << 7;
const TFD_DRQ: u32 = 1 << 3;

const SIG_SATA: u32 = 0x0000_0101;

// ── ATA commands ────────────────────────────────────────────────────────────
const ATA_CMD_IDENTIFY: u8 = 0xEC;
const ATA_CMD_READ_DMA_EXT: u8 = 0x25;
const ATA_CMD_WRITE_DMA_EXT: u8 = 0x35;

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
pub struct AhciController {
    base: u64,
    port: u32,
    cl: *mut CommandList,
    fis: *mut ReceivedFis,
    ct: *mut CommandTable,
    buf: *mut [u8; 4096],
    cl_phys: u64,
    fis_phys: u64,
    ct_phys: u64,
    buf_phys: u64,
}

unsafe impl Send for AhciController {}
unsafe impl Sync for AhciController {}

impl AhciController {
    // ------------------------------------------------------------------ MMIO
    unsafe fn read_hba(&self, offset: u64) -> u32 {
        read_volatile((self.base + offset) as *const u32)
    }

    unsafe fn write_hba(&self, offset: u64, val: u32) {
        write_volatile((self.base + offset) as *mut u32, val);
    }

    unsafe fn port_base(&self) -> u64 {
        self.base + 0x100 + self.port as u64 * 0x80
    }

    unsafe fn read_port(&self, offset: u64) -> u32 {
        read_volatile((self.port_base() + offset) as *const u32)
    }

    unsafe fn write_port(&self, offset: u64, val: u32) {
        write_volatile((self.port_base() + offset) as *mut u32, val);
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

    // ------------------------------------------------------------------ setup
    unsafe fn setup_command(&self, write: bool, buf_phys: u64, len: u32) {
        let cl_ptr = (*self.cl).data.as_mut_ptr() as *mut u32;

        let cfl = 5; // 5 DWs = 20 bytes for H2D register FIS
        let prdtl = 1u16;
        let dw0 = (cfl as u32)
            | ((if write { 1 } else { 0 }) << 6)
            | ((prdtl as u32) << 16);

        cl_ptr.write_volatile(dw0);
        cl_ptr.add(1).write_volatile(0); // PRDBC
        cl_ptr.add(2).write_volatile(self.ct_phys as u32);
        cl_ptr.add(3).write_volatile((self.ct_phys >> 32) as u32);
        for i in 4..8 {
            cl_ptr.add(i).write_volatile(0);
        }

        // Zero command table
        let ct_ptr = (*self.ct).data.as_mut_ptr();
        for i in 0..256 {
            ct_ptr.add(i).write_volatile(0);
        }

        // PRDT entry at offset 0x80
        let prdt = ct_ptr.add(0x80) as *mut u32;
        prdt.add(0).write_volatile(buf_phys as u32);
        prdt.add(1).write_volatile((buf_phys >> 32) as u32);
        prdt.add(2).write_volatile(0);
        prdt.add(3).write_volatile(len - 1); // DBC is 0-based, bit 31 = I = 0
    }

    unsafe fn fill_fis(&self, cmd: u8, lba: u64, count: u16) {
        let fis = (*self.ct).data.as_mut_ptr();
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

    // ------------------------------------------------------------------ public
    pub unsafe fn read_sectors(
        &self,
        lba: u64,
        count: u16,
        buf: &mut [u8],
    ) -> Result<(), BlockError> {
        let bytes = count as usize * 512;
        if bytes == 0 || bytes > 4096 || buf.len() < bytes {
            return Err(BlockError::InvalidLba);
        }

        self.stop_port();
        self.setup_command(false, self.buf_phys, bytes as u32);
        self.fill_fis(ATA_CMD_READ_DMA_EXT, lba, count);
        self.start_port();
        self.send_command(0)?;

        let tfd = self.read_port(PORT_TFD);
        if tfd & (TFD_BSY | TFD_DRQ) != 0 {
            return Err(BlockError::IoError);
        }

        core::ptr::copy_nonoverlapping((*self.buf).as_ptr(), buf.as_mut_ptr(), bytes);
        Ok(())
    }

    pub unsafe fn write_sectors(
        &self,
        lba: u64,
        count: u16,
        buf: &[u8],
    ) -> Result<(), BlockError> {
        let bytes = count as usize * 512;
        if bytes == 0 || bytes > 4096 || buf.len() < bytes {
            return Err(BlockError::InvalidLba);
        }

        core::ptr::copy_nonoverlapping(buf.as_ptr(), (*self.buf).as_mut_ptr(), bytes);

        self.stop_port();
        self.setup_command(true, self.buf_phys, bytes as u32);
        self.fill_fis(ATA_CMD_WRITE_DMA_EXT, lba, count);
        self.start_port();
        self.send_command(0)?;

        let tfd = self.read_port(PORT_TFD);
        if tfd & (TFD_BSY | TFD_DRQ) != 0 {
            return Err(BlockError::IoError);
        }
        Ok(())
    }

    pub unsafe fn identify_device(&self) -> Result<[u16; 256], BlockError> {
        self.stop_port();
        self.setup_command(false, self.buf_phys, 512);
        self.fill_fis(ATA_CMD_IDENTIFY, 0, 0);
        self.start_port();
        self.send_command(0)?;

        let mut result = [0u16; 256];
        let src = (*self.buf).as_ptr() as *const u16;
        for i in 0..256 {
            result[i] = src.add(i).read_volatile();
        }
        Ok(result)
    }
}

impl BlockDevice for AhciController {
    fn read_sector(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() != 512 {
            return Err(BlockError::InvalidLba);
        }
        unsafe { self.read_sectors(lba, 1, buf) }
    }

    fn write_sector(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != 512 {
            return Err(BlockError::InvalidLba);
        }
        unsafe { self.write_sectors(lba, 1, buf) }
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
            let sig = read_volatile(
                (base + 0x100 + i as u64 * 0x80 + PORT_SIG) as *const u32,
            );
            serial_println!("[AHCI] Port {} signature: {:#x}", i, sig);
            if sig == SIG_SATA {
                port_idx = Some(i);
                break;
            }
        }

        let port = match port_idx {
            Some(p) => p,
            None => {
                serial_println!("[AHCI] No SATA device found");
                return None;
            }
        };

        serial_println!("[AHCI] Using port {}", port);

        // Allocate DMA structures (identity-mapped below 4 GiB)
        let cl_phys = match alloc_frame() {
            Some(addr) => addr.as_u64(),
            None => {
                serial_println!("[AHCI] Warning: no memory for CL");
                return None;
            }
        };
        let fis_phys = match alloc_frame() {
            Some(addr) => addr.as_u64(),
            None => {
                serial_println!("[AHCI] Warning: no memory for FIS");
                return None;
            }
        };
        let ct_phys = match alloc_frame() {
            Some(addr) => addr.as_u64(),
            None => {
                serial_println!("[AHCI] Warning: no memory for CT");
                return None;
            }
        };
        let buf_phys = match alloc_frame() {
            Some(addr) => addr.as_u64(),
            None => {
                serial_println!("[AHCI] Warning: no memory for DMA buf");
                return None;
            }
        };

        for addr in [cl_phys, fis_phys, ct_phys, buf_phys] {
            let ptr = addr as *mut u8;
            for i in 0..4096 {
                ptr.add(i).write_volatile(0);
            }
        }

        let cl = cl_phys as *mut CommandList;
        let fis = fis_phys as *mut ReceivedFis;
        let ct = ct_phys as *mut CommandTable;
        let buf = buf_phys as *mut [u8; 4096];

        let controller = AhciController {
            base,
            port,
            cl,
            fis,
            ct,
            buf,
            cl_phys,
            fis_phys,
            ct_phys,
            buf_phys,
        };

        controller.stop_port();
        controller.write_port(PORT_CLB, cl_phys as u32);
        controller.write_port(PORT_CLBU, (cl_phys >> 32) as u32);
        controller.write_port(PORT_FB, fis_phys as u32);
        controller.write_port(PORT_FBU, (fis_phys >> 32) as u32);
        controller.write_port(PORT_IS, 0xFFFF_FFFF);
        controller.write_port(PORT_IE, 0);
        controller.start_port();

        // Wait for device detection
        for _ in 0..1_000_000 {
            let ssts = controller.read_port(PORT_SSTS);
            if ssts & 0x0F == 0x03 {
                break;
            }
            core::hint::spin_loop();
        }

        match controller.identify_device() {
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

        let controller = Box::leak(Box::new(controller));
        register_block_device(0x800, controller);
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
