// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Intel HDA driver — real CORB/RIRB init, codec probe, widget discovery, PCM stream output.

use crate::drivers::pci::get_device_by_class;
use crate::memory::phys::alloc_frame;
use crate::serial_println;
use core::ptr::{read_volatile, write_volatile};

// Global Capabilities
const REG_GCAP: u64 = 0x00;
const REG_GCTL: u64 = 0x08;
const REG_WAKEEN: u64 = 0x0C;
const REG_STATESTS: u64 = 0x0E;
const REG_GSTS: u64 = 0x10;

// CORB/RIRB
const REG_CORBLBASE: u64 = 0x40;
const REG_CORBUBASE: u64 = 0x44;
const REG_CORBWP: u64 = 0x48;
const REG_CORBRP: u64 = 0x4A;
const REG_CORBCTL: u64 = 0x4C;
const REG_CORBSTS: u64 = 0x4D;
const REG_CORBSIZE: u64 = 0x4E;
const REG_RIRBLBASE: u64 = 0x50;
const REG_RIRBUBASE: u64 = 0x54;
const REG_RIRBWP: u64 = 0x58;
const REG_RINTCNT: u64 = 0x5A;
const REG_RIRBCTL: u64 = 0x5C;
const REG_RIRBSTS: u64 = 0x5D;
const REG_RIRBSIZE: u64 = 0x5E;

// Immediate Command / Response
const REG_ICW: u64 = 0x60;
const REG_IRR: u64 = 0x64;
const REG_ICS: u64 = 0x68;

const GCTL_CRST: u32 = 1 << 0;
const CORBCTL_CORBRUN: u8 = 1 << 1;
const RIRBCTL_RIRBDMAEN: u8 = 1 << 1;
const ICS_ICB: u16 = 1 << 0;
const ICS_IRV: u16 = 1 << 1;

// Stream Descriptor registers (output stream 0)
fn sd_base(bar0: u64, iss: u8, bss: u8, stream: u8) -> u64 {
    bar0 + 0x80 + (iss + bss + stream) as u64 * 0x20
}

const SD_CTL: u64 = 0x00;
const SD_STS: u64 = 0x03;
const SD_CBL: u64 = 0x08;
const SD_LVI: u64 = 0x0C;
const SD_FMT: u64 = 0x12;
const SD_BDPL: u64 = 0x18;
const SD_BDPU: u64 = 0x1C;

const SD_CTL_RUN: u32 = 1 << 1;
const SD_CTL_IOCE: u32 = 1 << 2;
const SD_CTL_STRM: u32 = 1 << 16;

// BDL Entry
#[repr(C, align(16))]
struct BdlEntry {
    addr: u64,
    len: u32,
    ioc: u32,
}

pub struct HdaController {
    pub bar0: u64,
    pub vendor_id: u16,
    pub device_id: u16,
    pub corb_entries: u16,
    pub rirb_entries: u16,
    oss: u8,
    iss: u8,
    bss: u8,
    dac_nid: u8,
    stream_ready: bool,
}

impl HdaController {
    unsafe fn read_reg16(&self, offset: u64) -> u16 {
        read_volatile((self.bar0 + offset) as *const u16)
    }
    unsafe fn read_reg32(&self, offset: u64) -> u32 {
        read_volatile((self.bar0 + offset) as *const u32)
    }
    unsafe fn write_reg32(&self, offset: u64, val: u32) {
        write_volatile((self.bar0 + offset) as *mut u32, val);
    }
    unsafe fn write_reg8(&self, offset: u64, val: u8) {
        write_volatile((self.bar0 + offset) as *mut u8, val);
    }
    unsafe fn write_reg16(&self, offset: u64, val: u16) {
        write_volatile((self.bar0 + offset) as *mut u16, val);
    }

    pub fn probe() -> Option<Self> {
        serial_println!("[HDA] Probing PCI class 0x04/0x03 (High Definition Audio)...");
        let dev = match get_device_by_class(0x04, 0x03, 0x00) {
            Some(d) => d,
            None => {
                serial_println!("[HDA] No HDA controller detected");
                return None;
            }
        };
        serial_println!(
            "[HDA] Found controller at {:02x}:{:02x}.{} — {:04X}:{:04X}",
            dev.bus,
            dev.device,
            dev.function,
            dev.vendor_id,
            dev.device_id
        );
        Some(Self {
            bar0: dev.bar_address(0),
            vendor_id: dev.vendor_id,
            device_id: dev.device_id,
            corb_entries: 256,
            rirb_entries: 256,
            oss: 0,
            iss: 0,
            bss: 0,
            dac_nid: 0,
            stream_ready: false,
        })
    }

    pub fn reset_and_init(&mut self) -> Result<(), &'static str> {
        unsafe {
            let gcap = self.read_reg16(REG_GCAP);
            self.oss = ((gcap >> 12) & 0xF) as u8;
            self.iss = ((gcap >> 8) & 0xF) as u8;
            self.bss = ((gcap >> 3) & 0x1F) as u8;
            serial_println!(
                "[HDA] GCAP: OSS={}, ISS={}, BSS={}",
                self.oss,
                self.iss,
                self.bss
            );

            let mut gctl = self.read_reg32(REG_GCTL);
            gctl &= !GCTL_CRST;
            self.write_reg32(REG_GCTL, gctl);
            for _ in 0..100_000 {
                core::hint::spin_loop();
            }
            gctl |= GCTL_CRST;
            self.write_reg32(REG_GCTL, gctl);
            for _ in 0..100_000 {
                core::hint::spin_loop();
            }

            let statests = self.read_reg16(REG_STATESTS);
            serial_println!("[HDA] STATESTS: {:#06x} (codec bits)", statests);
            if statests == 0 {
                return Err("No codecs reported present");
            }
            self.write_reg16(REG_STATESTS, statests);

            let corb_frame = alloc_frame().ok_or("HDA CORB alloc failed")?;
            core::ptr::write_bytes(corb_frame.as_u64() as *mut u8, 0, 4096);
            self.write_reg32(REG_CORBLBASE, corb_frame.as_u64() as u32);
            self.write_reg32(REG_CORBUBASE, (corb_frame.as_u64() >> 32) as u32);
            self.write_reg16(REG_CORBRP, 1 << 15);
            self.write_reg8(REG_CORBCTL, 0);
            self.write_reg16(REG_CORBWP, 0);
            self.write_reg8(REG_CORBCTL, CORBCTL_CORBRUN);

            let rirb_frame = alloc_frame().ok_or("HDA RIRB alloc failed")?;
            core::ptr::write_bytes(rirb_frame.as_u64() as *mut u8, 0, 4096);
            self.write_reg32(REG_RIRBLBASE, rirb_frame.as_u64() as u32);
            self.write_reg32(REG_RIRBUBASE, (rirb_frame.as_u64() >> 32) as u32);
            self.write_reg16(REG_RIRBWP, 1 << 15);
            self.write_reg8(REG_RIRBCTL, 0);
            self.write_reg16(REG_RINTCNT, 1);
            self.write_reg8(REG_RIRBCTL, RIRBCTL_RIRBDMAEN);
        }

        // Probe codec 0
        let vid_did = unsafe { self.send_immediate_command(0, 0x00, 0xF0000) };
        serial_println!("[HDA] Codec 0 VendorID/DeviceID: {:#010x}", vid_did);

        // Discover widgets
        let fg_count = self.get_param(0, 0, 0x04);
        let start_nid = (fg_count & 0xFF) as u8;
        let count = ((fg_count >> 8) & 0xFF) as u8;
        serial_println!(
            "[HDA] Codec 0 function groups: start={}, count={}",
            start_nid,
            count
        );

        for fg in start_nid..start_nid + count {
            let fg_type = self.get_param(0, fg, 0x05);
            if (fg_type & 0x7F) == 0x01 {
                // Audio Function Group
                self.discover_dac(0, fg)?;
                break;
            }
        }

        if self.dac_nid == 0 {
            return Err("No DAC widget found");
        }

        self.setup_stream(0)?;
        Ok(())
    }

    fn get_param(&self, codec: u8, nid: u8, param: u8) -> u32 {
        unsafe { self.send_immediate_command(codec, nid, 0xF0000 | (param as u32)) }
    }

    fn discover_dac(&mut self, codec: u8, afg: u8) -> Result<(), &'static str> {
        let count = self.get_param(codec, afg, 0x04);
        let start_nid = (count & 0xFF) as u8;
        let ncount = ((count >> 8) & 0xFF) as u8;

        for nid in start_nid..start_nid + ncount {
            let cap = self.get_param(codec, nid, 0x09);
            let wtype = (cap >> 20) & 0xF;
            if wtype == 0 {
                // Audio Output
                serial_println!("[HDA] Found DAC widget {}", nid);
                self.dac_nid = nid;
                // Power on
                self.send_verb(codec, nid, 0x705, 0);
                // Set stream/channel
                self.send_verb(codec, nid, 0x706, 0x10); // stream 1, channel 0
                return Ok(());
            }
        }
        Err("No DAC found")
    }

    fn setup_stream(&mut self, _stream: u8) -> Result<(), &'static str> {
        let sd = sd_base(self.bar0, self.iss, self.bss, 0);
        unsafe {
            // Stop stream
            let mut ctl = read_volatile((sd + SD_CTL) as *const u32);
            ctl &= !SD_CTL_RUN;
            write_volatile((sd + SD_CTL) as *mut u32, ctl);
            for _ in 0..10_000 {
                core::hint::spin_loop();
            }

            // Format: 48kHz, 16-bit, stereo = 0x0011
            write_volatile((sd + SD_FMT) as *mut u16, 0x0011);

            // Allocate BDL + DMA buffer
            let bdl_frame = alloc_frame().ok_or("HDA BDL alloc failed")?;
            let buf_frame = alloc_frame().ok_or("HDA PCM buffer alloc failed")?;
            core::ptr::write_bytes(bdl_frame.as_u64() as *mut u8, 0, 4096);
            core::ptr::write_bytes(buf_frame.as_u64() as *mut u8, 0, 4096);

            let bdl = bdl_frame.as_u64() as *mut BdlEntry;
            (*bdl).addr = buf_frame.as_u64();
            (*bdl).len = 4096;
            (*bdl).ioc = 0;

            write_volatile((sd + SD_CBL) as *mut u32, 4096);
            write_volatile((sd + SD_LVI) as *mut u16, 0);
            write_volatile((sd + SD_BDPL) as *mut u32, bdl_frame.as_u64() as u32);
            write_volatile(
                (sd + SD_BDPU) as *mut u32,
                (bdl_frame.as_u64() >> 32) as u32,
            );

            // Set stream number and start
            ctl = (1 << 16) | SD_CTL_IOCE;
            write_volatile((sd + SD_CTL) as *mut u32, ctl);
        }
        self.stream_ready = true;
        serial_println!("[HDA] Output stream 0 ready (48kHz 16-bit stereo)");
        Ok(())
    }

    pub fn play_pcm(&mut self, samples: &[i16]) -> Result<(), &'static str> {
        if !self.stream_ready {
            return Err("HDA stream not ready");
        }
        let sd = sd_base(self.bar0, self.iss, self.bss, 0);
        unsafe {
            // Copy samples to DMA buffer
            let bdl = read_volatile((sd + SD_BDPL) as *const u32) as u64;
            let bdl_hi = read_volatile((sd + SD_BDPU) as *const u32) as u64;
            let bdl_addr = bdl | (bdl_hi << 32);
            let entry = (bdl_addr as *const BdlEntry).read_volatile();
            let buf = entry.addr as *mut i16;
            let len = samples.len().min(2048);
            core::ptr::copy_nonoverlapping(samples.as_ptr(), buf, len);

            // Start DMA
            let mut ctl = read_volatile((sd + SD_CTL) as *const u32);
            if (ctl & SD_CTL_RUN) == 0 {
                ctl |= SD_CTL_RUN;
                write_volatile((sd + SD_CTL) as *mut u32, ctl);
                serial_println!("[HDA] Playback started ({} samples)", len);
            }
        }
        Ok(())
    }

    pub fn stop_playback(&mut self) {
        if !self.stream_ready {
            return;
        }
        let sd = sd_base(self.bar0, self.iss, self.bss, 0);
        unsafe {
            let mut ctl = read_volatile((sd + SD_CTL) as *const u32);
            ctl &= !SD_CTL_RUN;
            write_volatile((sd + SD_CTL) as *mut u32, ctl);
        }
    }

    unsafe fn send_immediate_command(&self, codec: u8, nid: u8, verb: u32) -> u32 {
        let cmd = ((codec as u32) << 28) | ((nid as u32) << 20) | (verb & 0xFFFFF);
        for _ in 0..1_000_000 {
            if self.read_reg16(REG_ICS) & ICS_ICB == 0 {
                break;
            }
            core::hint::spin_loop();
        }
        self.write_reg32(REG_ICW, cmd);
        self.write_reg16(REG_ICS, ICS_ICB);
        for _ in 0..1_000_000 {
            let ics = self.read_reg16(REG_ICS);
            if ics & ICS_IRV != 0 {
                return self.read_reg32(REG_IRR);
            }
            core::hint::spin_loop();
        }
        0
    }

    fn send_verb(&self, codec: u8, nid: u8, verb: u16, payload: u16) {
        let full = ((verb as u32) << 8) | (payload as u32);
        unsafe {
            self.send_immediate_command(codec, nid, full);
        }
    }
}

static mut HDA_CTRL: Option<HdaController> = None;

pub fn init() -> Option<()> {
    let mut ctrl = HdaController::probe()?;
    match ctrl.reset_and_init() {
        Ok(()) => {
            serial_println!("[HDA] Initialized and stream ready");
            unsafe {
                HDA_CTRL = Some(ctrl);
            }
            Some(())
        }
        Err(e) => {
            serial_println!("[HDA] Init failed: {}", e);
            None
        }
    }
}

pub fn with_hda<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut HdaController) -> R,
{
    unsafe {
        let ptr = core::ptr::addr_of_mut!(HDA_CTRL);
        (*ptr).as_mut().map(f)
    }
}
