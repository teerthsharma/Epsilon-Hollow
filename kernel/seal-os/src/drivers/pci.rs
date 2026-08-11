// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! PCI bus enumeration — config space at 0xCF8/0xCFC.

use alloc::vec::Vec;
use spin::Mutex;
use x86_64::instructions::port::Port;

const PCI_CONFIG_ADDR: u16 = 0xCF8;
const PCI_CONFIG_DATA: u16 = 0xCFC;

static PCI_DEVICES: Mutex<Vec<PciDevice>> = Mutex::new(Vec::new());

#[derive(Debug, Clone)]
pub struct PciDevice {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub revision: u8,
    pub class: u8,
    pub subclass: u8,
    pub prog_if: u8,
    pub bar0: u32,
}

impl PciDevice {
    pub fn is_xhci(&self) -> bool {
        self.class == 0x0C && self.subclass == 0x03 && self.prog_if == 0x30
    }

    pub fn is_network(&self) -> bool {
        self.class == 0x02
    }

    pub fn is_display(&self) -> bool {
        self.class == 0x03
    }

    pub fn is_wifi(&self) -> bool {
        self.class == 0x02 && self.subclass == 0x80
    }

    pub fn is_ahci(&self) -> bool {
        self.class == 0x01 && self.subclass == 0x06 && self.prog_if == 0x01
    }

    pub fn bar_address(&self, bar: u8) -> u64 {
        let val = pci_read32(self.bus, self.device, self.function, 0x10 + bar * 4);
        if val & 1 == 0 {
            let typ = (val >> 1) & 0x03;
            if typ == 0x02 && bar < 5 {
                let high = pci_read32(self.bus, self.device, self.function, 0x10 + (bar + 1) * 4);
                ((val & !0x0F) as u64) | ((high as u64) << 32)
            } else {
                (val & !0x0F) as u64
            }
        } else {
            (val & !0x03) as u64
        }
    }

    pub fn enable_bus_mastering(&self) {
        let cmd = pci_read32(self.bus, self.device, self.function, 0x04);
        // MMIO-backed devices such as AHCI need Memory Space (bit 1) before
        // BAR reads are meaningful; DMA-capable devices also need Bus Master.
        pci_write32(
            self.bus,
            self.device,
            self.function,
            0x04,
            cmd | (1 << 1) | (1 << 2),
        );
    }
}

pub fn pci_read32(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
    let addr: u32 = 0x80000000
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);

    unsafe {
        let mut addr_port = Port::<u32>::new(PCI_CONFIG_ADDR);
        let mut data_port = Port::<u32>::new(PCI_CONFIG_DATA);
        addr_port.write(addr);
        data_port.read()
    }
}

pub fn pci_write32(bus: u8, device: u8, function: u8, offset: u8, value: u32) {
    let addr: u32 = 0x80000000
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);

    unsafe {
        let mut addr_port = Port::<u32>::new(PCI_CONFIG_ADDR);
        let mut data_port = Port::<u32>::new(PCI_CONFIG_DATA);
        addr_port.write(addr);
        data_port.write(value);
    }
}

/// Read one `(bus, device, function)` triple out of config space through
/// `read`, or `None` if nothing answers there.
///
/// An unimplemented function returns all-ones on every offset, so a vendor ID
/// of `0xFFFF` is the absence signal and never a device: this fails closed and
/// records nothing rather than pushing a phantom entry.
fn probe_function(
    read: fn(u8, u8, u8, u8) -> u32,
    bus: u8,
    device: u8,
    function: u8,
) -> Option<PciDevice> {
    let vendor_device = read(bus, device, function, 0);
    let vendor_id = (vendor_device & 0xFFFF) as u16;
    if vendor_id == 0xFFFF {
        return None;
    }
    let class_reg = read(bus, device, function, 0x08);

    Some(PciDevice {
        bus,
        device,
        function,
        vendor_id,
        device_id: ((vendor_device >> 16) & 0xFFFF) as u16,
        revision: (class_reg & 0xFF) as u8,
        class: ((class_reg >> 24) & 0xFF) as u8,
        subclass: ((class_reg >> 16) & 0xFF) as u8,
        prog_if: ((class_reg >> 8) & 0xFF) as u8,
        bar0: read(bus, device, function, 0x10),
    })
}

/// The enumeration loop, with config-space access injected so it can be driven
/// against a synthetic bus in `tests`. `enumerate` passes `pci_read32`.
///
/// Order is bus-major then device, and each device's function 0 is pushed
/// before any of its higher functions, so every device this returned before
/// still appears at the same relative position: callers that take the first
/// match by class keep the device they already had.
fn scan(read: fn(u8, u8, u8, u8) -> u32) -> Vec<PciDevice> {
    let mut devices = Vec::new();

    for bus in 0..=255u16 {
        for device in 0..32u8 {
            let Some(f0) = probe_function(read, bus as u8, device, 0) else {
                continue;
            };
            // Header type lives at offset 0x0E, the third byte of the dword at
            // 0x0C. Bit 7 set means the device implements functions 1..=7 as
            // well; the remaining bits are the layout (0 = device, 1 = bridge)
            // and are not consulted here, since nothing in this scan branches
            // on layout. Probing all eight unconditionally would be legal but
            // would cost 7 extra reads on every single-function device.
            let multifunction = probe_multifunction(read(bus as u8, device, 0, 0x0C));
            devices.push(f0);
            if multifunction {
                for function in 1..8u8 {
                    if let Some(dev) = probe_function(read, bus as u8, device, function) {
                        devices.push(dev);
                    }
                }
            }
        }
    }

    devices
}

/// Whether the multifunction bit is set in the header-type byte of the dword
/// at config offset 0x0C.
fn probe_multifunction(header_dword: u32) -> bool {
    ((header_dword >> 16) & 0x80) != 0
}

pub fn enumerate() -> Vec<PciDevice> {
    scan(pci_read32)
}

pub fn init() {
    let devices = enumerate();
    *PCI_DEVICES.lock() = devices;
}

pub fn get_devices() -> Vec<PciDevice> {
    PCI_DEVICES.lock().clone()
}

pub fn get_device_by_class(class: u8, subclass: u8, prog_if: u8) -> Option<PciDevice> {
    PCI_DEVICES
        .lock()
        .iter()
        .find(|d| d.class == class && d.subclass == subclass && d.prog_if == prog_if)
        .cloned()
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// `(bus, device, function, vendor, device_id, class, subclass, prog_if,
    /// header_type)` — one function of a synthetic config space.
    type CfgEntry = (u8, u8, u8, u16, u16, u8, u8, u8, u8);

    /// A QEMU `-machine q35 -device ahci,id=seal_sata` topology — the one
    /// `ci.yml` boots.
    const Q35: &[CfgEntry] = &[
        (0, 0x00, 0, 0x8086, 0x29C0, 0x06, 0x00, 0x00, 0x00), // host bridge
        (0, 0x01, 0, 0x1234, 0x1111, 0x03, 0x00, 0x00, 0x00), // VGA
        (0, 0x02, 0, 0x8086, 0x100E, 0x02, 0x00, 0x00, 0x00), // e1000 NIC
        (0, 0x03, 0, 0x8086, 0x2922, 0x01, 0x06, 0x01, 0x00), // -device ahci,id=seal_sata
        (0, 0x1F, 0, 0x8086, 0x2918, 0x06, 0x01, 0x00, 0x80), // ICH9 LPC, multifunction
        (0, 0x1F, 2, 0x8086, 0x2922, 0x01, 0x06, 0x01, 0x00), // ICH9 AHCI
        (0, 0x1F, 3, 0x8086, 0x2930, 0x0C, 0x05, 0x00, 0x00), // ICH9 SMBus
    ];

    /// Stands in for `pci_read32`. Every offset of an unpopulated function
    /// reads back all-ones, exactly as a real host bridge answers.
    fn fake_read(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
        let Some(e) = Q35
            .iter()
            .find(|e| e.0 == bus && e.1 == device && e.2 == function)
        else {
            return 0xFFFF_FFFF;
        };
        match offset & 0xFC {
            0x00 => ((e.4 as u32) << 16) | e.3 as u32,
            0x08 => ((e.5 as u32) << 24) | ((e.6 as u32) << 16) | ((e.7 as u32) << 8) | 0x02,
            0x0C => (e.8 as u32) << 16,
            0x10 => 0xFEB0_0000 | ((device as u32) << 8) | function as u32,
            _ => 0,
        }
    }

    /// `enumerate` used to read function 0 of each device and stop, so every
    /// device at a non-zero function number was invisible kernel-wide. Under
    /// `-machine q35` that hides the ICH9 AHCI controller at 00:1f.2 — the
    /// controller a bare `-drive` attaches to — and the ICH9 SMBus at 00:1f.3;
    /// on real boards it also hides xHCI, which commonly sits at function 3.
    ///
    /// This drives the real `scan` against the topology above and requires
    /// that every function of the multifunction device at 00:1f is present.
    /// Against the function-0-only loop the same body reports
    /// `Fail("00:1f.2 missing from enumeration")`.
    ///
    /// It also pins the two properties the fix had to preserve. Fail closed:
    /// a function that does not respond reads `0xFFFF` and must be dropped,
    /// never recorded. Purely additive: order is bus-major then device with
    /// function 0 first, so the first class match — what `get_device_by_class`
    /// returns — is still 00:03.0 for AHCI and not the newly visible 00:1f.2.
    fn multifunction_functions_are_enumerated() -> TestResult {
        let devs = scan(fake_read);

        for e in Q35 {
            test_assert!(
                devs.iter()
                    .any(|d| d.bus == e.0 && d.device == e.1 && d.function == e.2),
                "a function of the multifunction device is missing from enumeration"
            );
        }
        test_assert!(
            devs.iter().any(|d| (d.device, d.function) == (0x1F, 2)),
            "00:1f.2 missing from enumeration"
        );
        test_assert_eq!(devs.len(), Q35.len());

        // Fail closed.
        test_assert!(
            !devs.iter().any(|d| d.vendor_id == 0xFFFF),
            "a non-responding function was recorded as a device"
        );

        // Purely additive: unchanged winner for every first-match caller.
        let ahci = devs
            .iter()
            .find(|d| d.class == 0x01 && d.subclass == 0x06 && d.prog_if == 0x01);
        match ahci {
            Some(d) => {
                test_assert_eq!((d.bus, d.device, d.function), (0, 0x03, 0));
            }
            None => return TestResult::Fail("no AHCI controller enumerated"),
        }

        // Fields come from the probed function, not from function 0.
        let smbus = devs.iter().find(|d| (d.device, d.function) == (0x1F, 3));
        match smbus {
            Some(d) => {
                test_assert_eq!((d.vendor_id, d.device_id), (0x8086u16, 0x2930u16));
                test_assert_eq!((d.class, d.subclass), (0x0Cu8, 0x05u8));
                test_assert_eq!(d.bar0, fake_read(0, 0x1F, 3, 0x10));
            }
            None => return TestResult::Fail("00:1f.3 missing from enumeration"),
        }

        TestResult::Pass
    }

    /// Only bit 7 of the header-type byte at offset 0x0E decides whether
    /// functions 1..=7 are probed; the layout bits below it must not.
    fn header_type_selects_functions() -> TestResult {
        for (byte, want) in [
            (0x00u8, false),
            (0x01, false),
            (0x7F, false),
            (0x80, true),
            (0x81, true),
            (0xFF, true),
        ] {
            test_assert_eq!(probe_multifunction((byte as u32) << 16), want);
        }
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "pci::multifunction_functions_are_enumerated",
            multifunction_functions_are_enumerated,
        );
        crate::testing::register_test(
            "pci::header_type_selects_functions",
            header_type_selects_functions,
        );
    }
}
