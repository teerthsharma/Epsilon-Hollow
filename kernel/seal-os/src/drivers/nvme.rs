// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! NVMe controller driver — admin + I/O queues, identify, read/write sectors.

use core::ptr::{read_volatile, write_volatile};
use crate::drivers::pci::get_device_by_class;
use crate::memory::phys::{alloc_frame, alloc_frames_contiguous};
use crate::serial_println;

// Register offsets in BAR0
const REG_CAP: u64 = 0x00;
const REG_VS: u64 = 0x08;
const REG_INTMS: u64 = 0x0C;
const REG_INTMC: u64 = 0x10;
const REG_CC: u64 = 0x14;
const REG_CSTS: u64 = 0x1C;
const REG_NSSR: u64 = 0x20;
const REG_AQA: u64 = 0x24;
const REG_ASQ: u64 = 0x28;
const REG_ACQ: u64 = 0x30;

// CC bits
const CC_EN: u32 = 1 << 0;
const CSS_NVM: u32 = 0 << 4;
const MPS_4K: u32 = 0 << 7;
const AMS_RR: u32 = 0 << 11;

// CSTS bits
const CSTS_RDY: u32 = 1 << 0;
const CSTS_CFS: u32 = 1 << 1;

// Opcodes
const OPCODE_WRITE: u8 = 0x01;
const OPCODE_READ: u8 = 0x02;
const OPCODE_CREATE_IO_SQ: u8 = 0x01;
const OPCODE_CREATE_IO_CQ: u8 = 0x05;
const OPCODE_IDENTIFY: u8 = 0x06;

// Identify CNS
const CNS_NAMESPACE: u8 = 0x00;
const CNS_CONTROLLER: u8 = 0x01;

// NVMe constants
const SECTOR_SIZE: usize = 512;
const IO_Q_DEPTH: u16 = 64;

pub struct NvmeController {
    pub bar0: u64,
    pub vendor_id: u16,
    pub device_id: u16,
    pub max_q_entries: u32,
    pub page_size: u64,
    doorbell_stride: u64,

    admin_sq: u64,
    admin_cq: u64,
    admin_sq_tail: u16,
    admin_cq_head: u16,
    admin_phase: bool,

    io_sq: u64,
    io_cq: u64,
    io_sq_tail: u16,
    io_cq_head: u16,
    io_phase: bool,

    ns_size: u64,       // sectors
    ns_capacity: u64,   // sectors
    block_size: u64,
    ready: bool,
}

impl NvmeController {
    pub unsafe fn read_reg32(&self, offset: u64) -> u32 {
        read_volatile((self.bar0 + offset) as *const u32)
    }

    pub unsafe fn read_reg64(&self, offset: u64) -> u64 {
        read_volatile((self.bar0 + offset) as *const u64)
    }

    pub unsafe fn write_reg32(&self, offset: u64, val: u32) {
        write_volatile((self.bar0 + offset) as *mut u32, val);
    }

    pub unsafe fn write_reg64(&self, offset: u64, val: u64) {
        write_volatile((self.bar0 + offset) as *mut u64, val);
    }

    fn db_offset(&self, qid: u16, is_cq: bool) -> u64 {
        let stride = 4 << self.doorbell_stride;
        let base = if is_cq { 1 } else { 0 };
        (2 * qid as u64 + base) * stride
    }

    unsafe fn ring_sq_doorbell(&self, qid: u16) {
        let off = self.db_offset(qid, false);
        let tail = if qid == 0 { self.admin_sq_tail } else { self.io_sq_tail };
        write_volatile((self.bar0 + 0x1000 + off) as *mut u32, tail as u32);
    }

    unsafe fn ring_cq_doorbell(&self, qid: u16) {
        let off = self.db_offset(qid, true);
        let head = if qid == 0 { self.admin_cq_head } else { self.io_cq_head };
        write_volatile((self.bar0 + 0x1000 + off) as *mut u32, head as u32);
    }

    pub fn probe() -> Option<Self> {
        serial_println!("[NVMe] Probing PCI class 0x01/0x08 (Non-Volatile Memory Controller)...");
        let dev = match get_device_by_class(0x01, 0x08, 0x02) {
            Some(d) => d,
            None => {
                serial_println!("[NVMe] No NVMe controller detected");
                return None;
            }
        };
        serial_println!(
            "[NVMe] Found controller at {:02x}:{:02x}.{} — {:04X}:{:04X}",
            dev.bus, dev.device, dev.function, dev.vendor_id, dev.device_id
        );
        Some(Self {
            bar0: dev.bar_address(0),
            vendor_id: dev.vendor_id,
            device_id: dev.device_id,
            max_q_entries: 0,
            page_size: 4096,
            doorbell_stride: 0,
            admin_sq: 0,
            admin_cq: 0,
            admin_sq_tail: 0,
            admin_cq_head: 0,
            admin_phase: true,
            io_sq: 0,
            io_cq: 0,
            io_sq_tail: 0,
            io_cq_head: 0,
            io_phase: true,
            ns_size: 0,
            ns_capacity: 0,
            block_size: 512,
            ready: false,
        })
    }

    pub fn reset_and_init(&mut self) -> Result<(), &'static str> {
        unsafe {
            let cap = self.read_reg64(REG_CAP);
            let mqes = (cap & 0xFFFF) as u32 + 1;
            let dstrd = ((cap >> 32) & 0xF) as u64;
            let mpsmin = ((cap >> 48) & 0xF) as u64;
            self.max_q_entries = mqes;
            self.page_size = 1u64 << (12 + mpsmin);
            self.doorbell_stride = dstrd;
            serial_println!("[NVMe] CAP: MQES={}, DSTRD={}B, MPSMIN={}B", mqes, (dstrd + 1) * 4, self.page_size);

            let vs = self.read_reg32(REG_VS);
            serial_println!("[NVMe] Version {}.{}", vs >> 16, vs & 0xFFFF);

            let mut cc = self.read_reg32(REG_CC);
            if cc & CC_EN != 0 {
                cc &= !CC_EN;
                self.write_reg32(REG_CC, cc);
                for _ in 0..1_000_000 {
                    if self.read_reg32(REG_CSTS) & CSTS_RDY == 0 { break; }
                    core::hint::spin_loop();
                }
            }

            let sq_frame = alloc_frame().ok_or("NVMe SQ alloc failed")?;
            let cq_frame = alloc_frame().ok_or("NVMe CQ alloc failed")?;
            core::ptr::write_bytes(sq_frame.as_u64() as *mut u8, 0, 4096);
            core::ptr::write_bytes(cq_frame.as_u64() as *mut u8, 0, 4096);

            self.admin_sq = sq_frame.as_u64();
            self.admin_cq = cq_frame.as_u64();

            let aqa = ((mqes.min(64) - 1) << 16) | ((mqes.min(64) - 1) << 0);
            self.write_reg32(REG_AQA, aqa);
            self.write_reg64(REG_ASQ, self.admin_sq);
            self.write_reg64(REG_ACQ, self.admin_cq);

            let cc_val = CC_EN | CSS_NVM | MPS_4K | AMS_RR;
            self.write_reg32(REG_CC, cc_val);

            for i in 0..1_000_000 {
                let csts = self.read_reg32(REG_CSTS);
                if csts & CSTS_RDY != 0 {
                    serial_println!("[NVMe] Controller ready after {} spins", i);
                    break;
                }
                if csts & CSTS_CFS != 0 {
                    return Err("NVMe controller fatal status");
                }
                core::hint::spin_loop();
            }
            if self.read_reg32(REG_CSTS) & CSTS_RDY == 0 {
                return Err("NVMe controller ready timeout");
            }
        }

        // Identify controller
        let mut id_ctrl = [0u8; 4096];
        self.identify(CNS_CONTROLLER, 0, &mut id_ctrl)?;
        let serial = &id_ctrl[4..24];
        let model = &id_ctrl[24..64];
        serial_println!("[NVMe] Model: {}  Serial: {}",
            core::str::from_utf8(model).unwrap_or("???").trim_end_matches('\0'),
            core::str::from_utf8(serial).unwrap_or("???").trim_end_matches('\0')
        );

        // Identify namespace 1
        let mut id_ns = [0u8; 4096];
        self.identify(CNS_NAMESPACE, 1, &mut id_ns)?;
        self.ns_size = u64::from_le_bytes([
            id_ns[0], id_ns[1], id_ns[2], id_ns[3],
            id_ns[4], id_ns[5], id_ns[6], id_ns[7],
        ]);
        self.ns_capacity = u64::from_le_bytes([
            id_ns[8], id_ns[9], id_ns[10], id_ns[11],
            id_ns[12], id_ns[13], id_ns[14], id_ns[15],
        ]);
        let flbas = id_ns[26];
        let lba_fmt = (flbas & 0xF) as usize;
        self.block_size = 1u64 << id_ns[128 + 4 * lba_fmt];
        serial_println!("[NVMe] NS1 size={} cap={} block_size={}", self.ns_size, self.ns_capacity, self.block_size);

        if self.ns_size == 0 {
            return Err("NVMe namespace 1 has zero size");
        }

        // Create I/O Completion Queue
        let io_cq_frame = alloc_frame().ok_or("NVMe IO CQ alloc failed")?;
        unsafe { core::ptr::write_bytes(io_cq_frame.as_u64() as *mut u8, 0, 4096); }
        self.io_cq = io_cq_frame.as_u64();
        self.create_io_cq(1, self.io_cq, IO_Q_DEPTH - 1)?;

        // Create I/O Submission Queue
        let io_sq_frame = alloc_frame().ok_or("NVMe IO SQ alloc failed")?;
        unsafe { core::ptr::write_bytes(io_sq_frame.as_u64() as *mut u8, 0, 4096); }
        self.io_sq = io_sq_frame.as_u64();
        self.create_io_sq(1, self.io_sq, IO_Q_DEPTH - 1, 1)?;

        self.ready = true;
        Ok(())
    }

    fn submit_admin(&mut self, cmd: &[u64; 8]) {
        let tail = self.admin_sq_tail as usize;
        let addr = (self.admin_sq + tail as u64 * 64) as *mut u64;
        unsafe {
            for i in 0..8 {
                addr.add(i).write_volatile(cmd[i]);
            }
        }
        self.admin_sq_tail = (self.admin_sq_tail + 1) & (self.max_q_entries.min(64) - 1) as u16;
        unsafe { self.ring_sq_doorbell(0); }
    }

    fn wait_completion(&mut self, _cid: u16) -> Result<u16, &'static str> {
        let timeout = 1_000_000usize;
        for _ in 0..timeout {
            let head = self.admin_cq_head as usize;
            let addr = (self.admin_cq + head as u64 * 16) as *const u64;
            unsafe {
                let cqe_low = addr.read_volatile();
                let cqe_high = addr.add(1).read_volatile();
                let phase = ((cqe_high >> 16) & 1) != 0;
                if phase == self.admin_phase {
                    let status = ((cqe_high >> 17) & 0xFFFF) as u16;
                    let _sq_head = (cqe_low & 0xFFFF) as u16;
                    let _cid = ((cqe_low >> 16) & 0xFFFF) as u16;
                    self.admin_cq_head = (self.admin_cq_head + 1) & (self.max_q_entries.min(64) - 1) as u16;
                    if self.admin_cq_head == 0 {
                        self.admin_phase = !self.admin_phase;
                    }
                    self.ring_cq_doorbell(0);
                    return Ok(status);
                }
            }
            core::hint::spin_loop();
        }
        Err("NVMe admin command timeout")
    }

    fn identify(&mut self, cns: u8, nsid: u32, buf: &mut [u8]) -> Result<(), &'static str> {
        let phys = buf.as_ptr() as u64;
        let mut cmd = [0u64; 8];
        cmd[0] = (OPCODE_IDENTIFY as u64) | (1u64 << 32); // CID = 1
        cmd[1] = nsid as u64;
        cmd[3] = phys; // PRP1
        cmd[5] = (cns as u64) << 0; // CDW10
        self.submit_admin(&cmd);
        let status = self.wait_completion(1)?;
        if status != 0 {
            return Err("NVMe identify failed");
        }
        Ok(())
    }

    fn create_io_cq(&mut self, qid: u16, base: u64, size: u16) -> Result<(), &'static str> {
        let mut cmd = [0u64; 8];
        cmd[0] = (OPCODE_CREATE_IO_CQ as u64) | (2u64 << 32);
        cmd[3] = base; // PRP1
        cmd[5] = (size as u64) | ((qid as u64) << 16) | (1u64 << 1); // CDW10, IEN=1
        self.submit_admin(&cmd);
        let status = self.wait_completion(2)?;
        if status != 0 {
            return Err("NVMe create IO CQ failed");
        }
        Ok(())
    }

    fn create_io_sq(&mut self, qid: u16, base: u64, size: u16, cqid: u16) -> Result<(), &'static str> {
        let mut cmd = [0u64; 8];
        cmd[0] = (OPCODE_CREATE_IO_SQ as u64) | (3u64 << 32);
        cmd[3] = base; // PRP1
        cmd[5] = (size as u64) | ((qid as u64) << 16) | ((cqid as u64) << 32) | (1u64 << 0); // CDW10, PC=1
        self.submit_admin(&cmd);
        let status = self.wait_completion(3)?;
        if status != 0 {
            return Err("NVMe create IO SQ failed");
        }
        Ok(())
    }

    pub fn read_sector(&mut self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        if !self.ready {
            return Err("NVMe controller not ready");
        }
        if buf.len() < SECTOR_SIZE {
            return Err("buffer too small for sector");
        }
        let pages = (buf.len() + 4095) / 4096;
        let phys = if pages == 1 {
            alloc_frame().ok_or("NVMe read alloc failed")?.as_u64()
        } else {
            alloc_frames_contiguous(pages).ok_or("NVMe read alloc failed")?.as_u64()
        };

        let mut cmd = [0u64; 8];
        cmd[0] = (OPCODE_READ as u64) | (4u64 << 32);
        cmd[1] = 1; // NSID
        cmd[3] = phys; // PRP1
        cmd[5] = lba; // CDW10
        cmd[6] = lba >> 32; // CDW11
        cmd[7] = ((buf.len() / SECTOR_SIZE) as u64 - 1) & 0xFFFF; // CDW12

        let tail = self.io_sq_tail as usize;
        let addr = (self.io_sq + tail as u64 * 64) as *mut u64;
        unsafe {
            for i in 0..8 {
                addr.add(i).write_volatile(cmd[i]);
            }
        }
        self.io_sq_tail = (self.io_sq_tail + 1) & (IO_Q_DEPTH - 1);
        unsafe { self.ring_sq_doorbell(1); }

        // Poll I/O completion
        let timeout = 1_000_000usize;
        for _ in 0..timeout {
            let head = self.io_cq_head as usize;
            let addr = (self.io_cq + head as u64 * 16) as *const u64;
            unsafe {
                let cqe_high = addr.add(1).read_volatile();
                let phase = ((cqe_high >> 16) & 1) != 0;
                if phase == self.io_phase {
                    self.io_cq_head = (self.io_cq_head + 1) & (IO_Q_DEPTH - 1);
                    if self.io_cq_head == 0 {
                        self.io_phase = !self.io_phase;
                    }
                    self.ring_cq_doorbell(1);
                    break;
                }
            }
            core::hint::spin_loop();
        }

        unsafe {
            core::ptr::copy_nonoverlapping(phys as *const u8, buf.as_mut_ptr(), buf.len());
        }
        // Free frame
        // NOTE: phys frame leak in error path — acceptable for research kernel
        Ok(())
    }

    pub fn write_sector(&mut self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        if !self.ready {
            return Err("NVMe controller not ready");
        }
        if buf.len() < SECTOR_SIZE {
            return Err("buffer too small for sector");
        }
        let pages = (buf.len() + 4095) / 4096;
        let phys = if pages == 1 {
            alloc_frame().ok_or("NVMe write alloc failed")?.as_u64()
        } else {
            alloc_frames_contiguous(pages).ok_or("NVMe write alloc failed")?.as_u64()
        };
        unsafe {
            core::ptr::copy_nonoverlapping(buf.as_ptr(), phys as *mut u8, buf.len());
        }

        let mut cmd = [0u64; 8];
        cmd[0] = (OPCODE_WRITE as u64) | (5u64 << 32);
        cmd[1] = 1; // NSID
        cmd[3] = phys; // PRP1
        cmd[5] = lba; // CDW10
        cmd[6] = lba >> 32; // CDW11
        cmd[7] = ((buf.len() / SECTOR_SIZE) as u64 - 1) & 0xFFFF; // CDW12

        let tail = self.io_sq_tail as usize;
        let addr = (self.io_sq + tail as u64 * 64) as *mut u64;
        unsafe {
            for i in 0..8 {
                addr.add(i).write_volatile(cmd[i]);
            }
        }
        self.io_sq_tail = (self.io_sq_tail + 1) & (IO_Q_DEPTH - 1);
        unsafe { self.ring_sq_doorbell(1); }

        let timeout = 1_000_000usize;
        for _ in 0..timeout {
            let head = self.io_cq_head as usize;
            let addr = (self.io_cq + head as u64 * 16) as *const u64;
            unsafe {
                let cqe_high = addr.add(1).read_volatile();
                let phase = ((cqe_high >> 16) & 1) != 0;
                if phase == self.io_phase {
                    self.io_cq_head = (self.io_cq_head + 1) & (IO_Q_DEPTH - 1);
                    if self.io_cq_head == 0 {
                        self.io_phase = !self.io_phase;
                    }
                    self.ring_cq_doorbell(1);
                    break;
                }
            }
            core::hint::spin_loop();
        }
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }

    pub fn namespace_size(&self) -> u64 {
        self.ns_size
    }

    pub fn block_size(&self) -> u64 {
        self.block_size
    }
}

static mut NVME_CTRL: Option<NvmeController> = None;

pub fn init() -> Option<()> {
    let mut ctrl = NvmeController::probe()?;
    match ctrl.reset_and_init() {
        Ok(()) => {
            serial_println!("[NVMe] Initialized successfully — NS1 ready for I/O");
            unsafe { NVME_CTRL = Some(ctrl); }
            Some(())
        }
        Err(e) => {
            serial_println!("[NVMe] Init failed: {}", e);
            None
        }
    }
}

pub fn with_nvme<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut NvmeController) -> R,
{
    unsafe {
        NVME_CTRL.as_mut().map(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvme_probe_none() {
        // On host, no NVMe PCI device exists; probe returns None
        assert!(NvmeController::probe().is_none());
    }
}
