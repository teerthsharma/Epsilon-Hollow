// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![allow(dead_code)] // REASON: SCSI command constants and endpoint fields for future MSC completion

//! USB Mass Storage Class (MSC) driver — Bulk-Only Transport (BBB) with SCSI.
//! Implements the kernel BlockDevice trait for xHCI bulk endpoints.
#![allow(static_mut_refs)]

use core::ptr::read_unaligned;
use spin::Mutex;

use crate::drivers::block::{BlockDevice, BlockError};
use crate::drivers::usb::xhci;
use crate::serial_println;

pub const MASS_STORAGE_CLASS: u8 = 8;
pub const SCSI_SUBCLASS: u8 = 6;
pub const BBB_PROTOCOL: u8 = 0x50;

// SCSI opcodes
const SCSI_INQUIRY: u8 = 0x12;
const SCSI_READ_CAPACITY_10: u8 = 0x25;
const SCSI_READ_10: u8 = 0x28;
const SCSI_WRITE_10: u8 = 0x2A;
const SCSI_REQUEST_SENSE: u8 = 0x03;
const SCSI_TEST_UNIT_READY: u8 = 0x00;

const CBW_SIGNATURE: u32 = 0x43425355;
const CSW_SIGNATURE: u32 = 0x53425355;

/// Command Block Wrapper — 31 bytes.
#[repr(C, packed)]
struct Cbw {
    signature: u32,
    tag: u32,
    data_len: u32,
    flags: u8,
    lun: u8,
    cmd_len: u8,
    cmd: [u8; 16],
}

/// Command Status Wrapper — 13 bytes.
#[repr(C, packed)]
struct Csw {
    signature: u32,
    tag: u32,
    residue: u32,
    status: u8,
}

impl Cbw {
    fn new(tag: u32, data_len: u32, flags: u8, cmd: &[u8]) -> Self {
        let mut c = Self {
            signature: CBW_SIGNATURE,
            tag,
            data_len,
            flags,
            lun: 0,
            cmd_len: cmd.len() as u8,
            cmd: [0; 16],
        };
        let len = cmd.len().min(16);
        c.cmd[..len].copy_from_slice(&cmd[..len]);
        c
    }
}

impl Csw {
    fn parse(raw: &[u8; 13]) -> Result<Self, &'static str> {
        let csw: Self = unsafe { read_unaligned(raw.as_ptr() as *const Self) };
        if csw.signature != CSW_SIGNATURE {
            return Err("invalid CSW signature");
        }
        Ok(csw)
    }
}

/// Maximum sectors per SCSI READ/WRITE 10 command (limited by Normal TRB length).
const MAX_SECTORS_PER_CMD: usize = 128;

struct UsbMassStorageInner {
    slot_id: u8,
    bulk_in: u8,
    bulk_out: u8,
    max_packet: u16,
    sector_size: u32,
    num_sectors: u64,
    initialized: bool,
    tag_counter: u32,
}

/// USB Mass Storage device implementing BlockDevice.
pub struct UsbMassStorage {
    inner: Mutex<UsbMassStorageInner>,
}

impl UsbMassStorage {
    pub fn new(slot_id: u8, bulk_in: u8, bulk_out: u8, max_packet: u16) -> Self {
        Self {
            inner: Mutex::new(UsbMassStorageInner {
                slot_id,
                bulk_in,
                bulk_out,
                max_packet,
                sector_size: 512,
                num_sectors: 0,
                initialized: false,
                tag_counter: 1,
            }),
        }
    }

    pub fn initialized(&self) -> bool {
        self.inner.lock().initialized
    }

    pub fn sector_size(&self) -> u32 {
        self.inner.lock().sector_size
    }

    pub fn num_sectors(&self) -> u64 {
        self.inner.lock().num_sectors
    }

    pub fn init(&self) -> Result<(), &'static str> {
        let mut inner = self.inner.lock();
        if inner.initialized {
            return Ok(());
        }

        let _inq = Self::inquiry_raw(&mut inner)?;
        let (num_sectors, sector_size) = Self::read_capacity_raw(&mut inner)?;
        inner.num_sectors = num_sectors;
        inner.sector_size = sector_size;
        inner.initialized = true;

        serial_println!(
            "[MSC] Ready — {} sectors × {} bytes ({} MB)",
            num_sectors,
            sector_size,
            (num_sectors * sector_size as u64) / (1024 * 1024)
        );
        Ok(())
    }

    fn inquiry_raw(inner: &mut UsbMassStorageInner) -> Result<[u8; 36], &'static str> {
        let cmd = [SCSI_INQUIRY, 0, 0, 0, 36, 0];
        let cbw = Cbw::new(inner.tag_counter, 36, 0x80, &cmd);
        inner.tag_counter = inner.tag_counter.wrapping_add(1);
        Self::send_cbw(inner, &cbw)?;
        let mut data = [0u8; 36];
        Self::recv_bulk_in(inner, &mut data)?;
        Self::recv_csw(inner)?;
        Ok(data)
    }

    fn read_capacity_raw(inner: &mut UsbMassStorageInner) -> Result<(u64, u32), &'static str> {
        let cmd = [SCSI_READ_CAPACITY_10, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let cbw = Cbw::new(inner.tag_counter, 8, 0x80, &cmd);
        inner.tag_counter = inner.tag_counter.wrapping_add(1);
        Self::send_cbw(inner, &cbw)?;
        let mut data = [0u8; 8];
        Self::recv_bulk_in(inner, &mut data)?;
        Self::recv_csw(inner)?;

        let last_lba = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let block_size = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        Ok(((last_lba as u64) + 1, block_size))
    }

    pub fn read_sectors(&self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        let mut inner = self.inner.lock();
        let sector_size = inner.sector_size as usize;
        if sector_size == 0 {
            return Err("sector size not initialized");
        }
        if buf.len() % sector_size != 0 {
            return Err("buffer not sector-aligned");
        }
        let total_sectors = buf.len() / sector_size;
        let mut offset = 0;
        while offset < total_sectors {
            let chunk = (total_sectors - offset).min(MAX_SECTORS_PER_CMD);
            let chunk_bytes = chunk * sector_size;
            let chunk_buf = &mut buf[offset * sector_size..offset * sector_size + chunk_bytes];
            Self::read_sectors_raw(&mut inner, lba + offset as u64, chunk_buf)?;
            offset += chunk;
        }
        Ok(())
    }

    fn read_sectors_raw(
        inner: &mut UsbMassStorageInner,
        lba: u64,
        buf: &mut [u8],
    ) -> Result<(), &'static str> {
        let sector_count = (buf.len() / inner.sector_size as usize) as u16;
        let mut cmd = [0u8; 10];
        cmd[0] = SCSI_READ_10;
        cmd[2..6].copy_from_slice(&(lba as u32).to_be_bytes());
        cmd[7..9].copy_from_slice(&sector_count.to_be_bytes());

        let cbw = Cbw::new(inner.tag_counter, buf.len() as u32, 0x80, &cmd);
        inner.tag_counter = inner.tag_counter.wrapping_add(1);
        Self::send_cbw(inner, &cbw)?;
        Self::recv_bulk_in(inner, buf)?;
        Self::recv_csw(inner)?;
        Ok(())
    }

    pub fn write_sectors(&self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        let mut inner = self.inner.lock();
        let sector_size = inner.sector_size as usize;
        if sector_size == 0 {
            return Err("sector size not initialized");
        }
        if buf.len() % sector_size != 0 {
            return Err("buffer not sector-aligned");
        }
        let total_sectors = buf.len() / sector_size;
        let mut offset = 0;
        while offset < total_sectors {
            let chunk = (total_sectors - offset).min(MAX_SECTORS_PER_CMD);
            let chunk_bytes = chunk * sector_size;
            let chunk_buf = &buf[offset * sector_size..offset * sector_size + chunk_bytes];
            Self::write_sectors_raw(&mut inner, lba + offset as u64, chunk_buf)?;
            offset += chunk;
        }
        Ok(())
    }

    fn write_sectors_raw(
        inner: &mut UsbMassStorageInner,
        lba: u64,
        buf: &[u8],
    ) -> Result<(), &'static str> {
        let sector_count = (buf.len() / inner.sector_size as usize) as u16;
        let mut cmd = [0u8; 10];
        cmd[0] = SCSI_WRITE_10;
        cmd[2..6].copy_from_slice(&(lba as u32).to_be_bytes());
        cmd[7..9].copy_from_slice(&sector_count.to_be_bytes());

        let cbw = Cbw::new(inner.tag_counter, buf.len() as u32, 0x00, &cmd);
        inner.tag_counter = inner.tag_counter.wrapping_add(1);
        Self::send_cbw(inner, &cbw)?;
        Self::send_bulk_out(inner, buf)?;
        Self::recv_csw(inner)?;
        Ok(())
    }

    fn send_cbw(inner: &UsbMassStorageInner, cbw: &Cbw) -> Result<(), &'static str> {
        let raw = unsafe { core::slice::from_raw_parts(cbw as *const _ as *const u8, 31) };
        Self::send_bulk_out(inner, raw)
    }

    fn recv_csw(inner: &UsbMassStorageInner) -> Result<Csw, &'static str> {
        let mut raw = [0u8; 13];
        Self::recv_bulk_in(inner, &mut raw)?;
        let csw = Csw::parse(&raw)?;
        if csw.status != 0 {
            return Err("CSW reports command failure");
        }
        Ok(csw)
    }

    fn send_bulk_out(inner: &UsbMassStorageInner, data: &[u8]) -> Result<(), &'static str> {
        xhci::with_xhci(|ctrl| ctrl.send_bulk_out(inner.slot_id, inner.bulk_out, data))
            .ok_or("xHCI unavailable")?
    }

    fn recv_bulk_in(inner: &UsbMassStorageInner, buf: &mut [u8]) -> Result<(), &'static str> {
        xhci::with_xhci(|ctrl| ctrl.recv_bulk_in(inner.slot_id, inner.bulk_in, buf))
            .ok_or("xHCI unavailable")?
    }
}

impl BlockDevice for UsbMassStorage {
    fn sector_size(&self) -> u64 {
        self.inner.lock().sector_size as u64
    }

    fn num_sectors(&self) -> u64 {
        self.inner.lock().num_sectors
    }

    fn read_sectors(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        UsbMassStorage::read_sectors(self, lba, buf).map_err(|_| BlockError::IoError)
    }

    fn write_sectors(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        UsbMassStorage::write_sectors(self, lba, buf).map_err(|_| BlockError::IoError)
    }

    fn flush(&self) -> Result<(), BlockError> {
        Ok(())
    }
}

// Global MSC device storage
static USB_MSC: spin::Mutex<Option<UsbMassStorage>> = spin::Mutex::new(None);

/// Register a newly enumerated USB mass storage device.
pub fn register_device(slot_id: u8, bulk_in: u8, bulk_out: u8, max_packet: u16) {
    let mut guard = USB_MSC.lock();
    if guard.is_some() {
        serial_println!("[MSC] Device already registered; ignoring new device");
        return;
    }
    serial_println!(
        "[MSC] Registering slot={} in={:02X} out={:02X}",
        slot_id,
        bulk_in,
        bulk_out
    );
    *guard = Some(UsbMassStorage::new(slot_id, bulk_in, bulk_out, max_packet));
}

/// Access the global MSC device.
pub fn with_usb_msc<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut UsbMassStorage) -> R,
{
    USB_MSC.lock().as_mut().map(f)
}

/// Initialize the MSC subsystem (call after xHCI is ready).
pub fn init() {
    serial_println!("[MSC] Subsystem initialized");
}

/// Poll: initialize any pending MSC device and register it as a block device.
pub fn poll() {
    let mut guard = USB_MSC.lock();
    if let Some(msc) = guard.as_mut() {
        if msc.initialized() {
            return;
        }
        if let Err(e) = msc.init() {
            serial_println!("[MSC] Init failed: {}", e);
            return;
        }
        // Register as block device by leaking a reference (block device registry copies data)
        let msc_ptr: *mut UsbMassStorage = msc as *mut _;
        let msc_ref: &'static UsbMassStorage = unsafe { &*msc_ptr };
        crate::drivers::block::register_block_device(1, msc_ref);
        serial_println!("[MSC] Registered as block device 1");
    }
}
