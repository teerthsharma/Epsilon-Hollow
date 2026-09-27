// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Network driver core -- e1000 probe, raw frame TX/RX.

use spin::Mutex;

pub mod certs;
pub mod dhcp;
pub mod dns;
pub mod e1000;
pub mod ecdhe;
pub mod http;
pub mod icmp;
pub mod tcp;
pub mod tls;
pub mod tls_socket;
pub mod udp;
pub mod virtio_net;
pub mod x509;

static NET_DEVICE: Mutex<Option<e1000::E1000>> = Mutex::new(None);

pub fn init() {
    for dev in crate::drivers::pci::get_devices() {
        // Intel e1000 family — include 0x100E (VirtualBox) and 0x100F (QEMU)
        let is_intel_e1000 =
            dev.vendor_id == 0x8086 && (dev.device_id == 0x100E || dev.device_id == 0x100F);
        if dev.class == 0x02 && dev.subclass == 0x00 && is_intel_e1000 {
            crate::serial_println!(
                "[e1000] Found NIC at {}:{}.{} BAR0={:08X}",
                dev.bus,
                dev.device,
                dev.function,
                dev.bar0
            );
            dev.enable_bus_mastering();
            let bar0 = (dev.bar0 & 0xFFFFFFF0) as usize;
            unsafe {
                if let Some(mut nic) = e1000::E1000::new(bar0) {
                    if nic.init() {
                        let mac = nic.mac_address();
                        crate::serial_println!(
                            "[e1000] MAC: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
                            mac[0],
                            mac[1],
                            mac[2],
                            mac[3],
                            mac[4],
                            mac[5]
                        );
                        *NET_DEVICE.lock() = Some(nic);
                        return;
                    }
                }
            }
        }
    }

    // Fallback: Virtio-Net.
    //
    // NOTE: `_net` is dropped immediately and `NET_DEVICE` is never set, because
    // that static is typed for E1000 and there is no device trait yet. So this
    // is a probe only — `poll()`, `transmit()` and `get_mac_address()` remain
    // no-ops afterwards. The log line says "probed", not "initialized", to match.
    if let Ok(_net) = virtio_net::VirtioNet::discover_and_init() {
        crate::serial_println!(
            "[virtio-net] Probed NIC (not wired into the stack — TX/RX unavailable)"
        );
    }
    // Reached whether or not the virtio probe above succeeded; the stack really
    // does have no usable NIC at this point in either case.
    crate::serial_println!("[NET] No usable NIC bound to the stack");
}

pub fn poll() {
    let mut buf = [0u8; 2048];
    loop {
        let len = {
            let mut dev = NET_DEVICE.lock();
            if let Some(ref mut nic) = *dev {
                nic.recv_packet(&mut buf)
            } else {
                None
            }
        };
        if let Some(len) = len {
            crate::net::process_packet(&buf[..len]);
        } else {
            break;
        }
    }
}

pub fn transmit(buf: &[u8]) {
    let mut dev = NET_DEVICE.lock();
    if let Some(ref mut nic) = *dev {
        if !nic.send_packet(buf) {
            crate::serial_println!("[e1000] TX drop");
        }
    }
}

pub fn has_nic() -> bool {
    NET_DEVICE.lock().is_some()
}

pub fn get_mac_address() -> [u8; 6] {
    let dev = NET_DEVICE.lock();
    if let Some(ref nic) = *dev {
        nic.mac_address()
    } else {
        [0; 6]
    }
}

#[cfg(feature = "test-mode")]
pub mod tests {
    use crate::drivers::pci::{pci_read32, pci_write32};
    use crate::test_assert;
    use crate::testing::TestResult;

    const MEMORY_SPACE: u32 = 1 << 1;
    const BUS_MASTER: u32 = 1 << 2;

    /// The e1000 reads its descriptor rings and frames by DMA, so its PCI
    /// command register needs Memory Space and Bus Master set before `init`
    /// can use it. Firmware leaves those bits in whatever state its own
    /// drivers wanted; this clears both first, as a firmware that never bound
    /// the NIC would, and requires `init` to set them itself. Passes without
    /// checking when no e1000 is attached (QEMU `-nic user,model=e1000` adds one).
    fn test_e1000_init_enables_memory_space_and_bus_mastering() -> TestResult {
        crate::drivers::pci::init();
        let Some(dev) = crate::drivers::pci::get_devices()
            .into_iter()
            .find(|d| d.vendor_id == 0x8086 && (d.device_id == 0x100E || d.device_id == 0x100F))
        else {
            return TestResult::Pass;
        };
        let (b, s, f) = (dev.bus, dev.device, dev.function);
        let firmware = pci_read32(b, s, f, 0x04);
        crate::serial_println!(
            "[e1000-test] command register left by firmware: {:#06x}",
            firmware & 0xFFFF
        );
        pci_write32(b, s, f, 0x04, firmware & !(MEMORY_SPACE | BUS_MASTER));

        super::init();

        let cmd = pci_read32(b, s, f, 0x04);
        test_assert!(
            cmd & (MEMORY_SPACE | BUS_MASTER) == MEMORY_SPACE | BUS_MASTER,
            "e1000 init left Memory Space or Bus Master clear"
        );
        test_assert!(super::has_nic(), "e1000 init did not bind the NIC");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "net::e1000_init_enables_memory_space_and_bus_mastering",
            test_e1000_init_enables_memory_space_and_bus_mastering,
        );
    }
}
