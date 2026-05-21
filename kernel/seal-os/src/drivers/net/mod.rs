// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Network driver core -- e1000 probe, raw frame TX/RX.

use spin::Mutex;

pub mod e1000;
pub mod tcp;
pub mod udp;
pub mod dhcp;
pub mod dns;
pub mod icmp;
pub mod http;
pub mod tls;

static NET_DEVICE: Mutex<Option<e1000::E1000>> = Mutex::new(None);

pub fn init() {
    for dev in crate::drivers::pci::get_devices() {
        if dev.class == 0x02 && dev.subclass == 0x00 {
            crate::serial_println!(
                "[e1000] Found NIC at {}:{}.{} BAR0={:08X}",
                dev.bus, dev.device, dev.function, dev.bar0
            );
            let bar0 = (dev.bar0 & 0xFFFFFFF0) as usize;
            unsafe {
                if let Some(mut nic) = e1000::E1000::new(bar0) {
                    if nic.init() {
                        let mac = nic.mac_address();
                        crate::serial_println!(
                            "[e1000] MAC: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
                            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]
                        );
                        *NET_DEVICE.lock() = Some(nic);
                        return;
                    }
                }
            }
        }
    }
    crate::serial_println!("[e1000] No NIC found");
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

pub fn get_mac_address() -> [u8; 6] {
    let dev = NET_DEVICE.lock();
    if let Some(ref nic) = *dev {
        nic.mac_address()
    } else {
        [0; 6]
    }
}
