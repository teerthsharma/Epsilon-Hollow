// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Protocol stack -- ARP, IPv4, ICMP, UDP, TCP, DHCP, DNS.

pub mod arp;
pub mod dhcp;
pub mod dns;
pub mod icmp;
pub mod ipv4;
pub mod tcp;
pub mod udp;

use spin::Mutex;

static LOCAL_MAC: Mutex<[u8; 6]> = Mutex::new([0; 6]);
static LOCAL_IP: Mutex<[u8; 4]> = Mutex::new([0, 0, 0, 0]);
static GATEWAY: Mutex<[u8; 4]> = Mutex::new([0, 0, 0, 0]);
static SUBNET: Mutex<[u8; 4]> = Mutex::new([255, 255, 255, 0]);

/// Returns true only if a real NIC driver has been loaded and initialized.
pub fn has_nic() -> bool {
    crate::drivers::net::get_mac_address() != [0; 6]
}

pub fn init() {
    crate::drivers::net::init();
    *LOCAL_MAC.lock() = crate::drivers::net::get_mac_address();

    if has_nic() {
        crate::serial_println!("[NET] NIC detected — network stack initialized");
    } else {
        crate::serial_println!("[NET] No NIC detected — network stack initialized (TX disabled)");
    }

    arp::init();
    dhcp::init();
}

pub fn poll() {
    crate::drivers::net::poll();
    dhcp::poll();
    tcp::poll();
}

pub fn transmit(buf: &[u8]) {
    crate::drivers::net::transmit(buf);
}

pub fn process_packet(frame: &[u8]) {
    if frame.len() < 14 {
        return;
    }
    let ethertype = u16::from_be_bytes([frame[12], frame[13]]);
    match ethertype {
        0x0806 => arp::handle_arp_packet(&frame[14..]),
        0x0800 => ipv4::handle_ipv4_packet(&frame[14..]),
        _ => {}
    }
}

pub fn local_mac() -> [u8; 6] {
    *LOCAL_MAC.lock()
}

pub fn local_ip() -> [u8; 4] {
    *LOCAL_IP.lock()
}

pub fn set_local_ip(ip: [u8; 4]) {
    *LOCAL_IP.lock() = ip;
}

pub fn gateway() -> [u8; 4] {
    *GATEWAY.lock()
}

pub fn set_gateway(ip: [u8; 4]) {
    *GATEWAY.lock() = ip;
}

pub fn subnet() -> [u8; 4] {
    *SUBNET.lock()
}

pub fn set_subnet(ip: [u8; 4]) {
    *SUBNET.lock() = ip;
}
