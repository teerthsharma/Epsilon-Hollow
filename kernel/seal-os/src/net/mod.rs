// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Protocol stack -- ARP, IPv4, IPv6, ICMP, ICMPv6, UDP, TCP, DHCP, DNS.
//! Topological routing (T1-T5) and firewall integrated.

pub mod arp;
pub mod dhcp;
pub mod dns;
pub mod firewall;
pub mod icmp;
pub mod ipv4;
pub mod ipv6;
pub mod tcp;
pub mod topological;
pub mod udp;

use spin::Mutex;

/// Dual-stack IP address.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IpAddr {
    V4([u8; 4]),
    V6([u8; 16]),
}

impl IpAddr {
    pub fn is_v4(&self) -> bool {
        matches!(self, IpAddr::V4(_))
    }
    pub fn is_v6(&self) -> bool {
        matches!(self, IpAddr::V6(_))
    }
    pub fn to_v4(&self) -> Option<[u8; 4]> {
        match self {
            IpAddr::V4(a) => Some(*a),
            _ => None,
        }
    }
    pub fn to_v6(&self) -> Option<[u8; 16]> {
        match self {
            IpAddr::V6(a) => Some(*a),
            _ => None,
        }
    }

    /// T5: Map IP address to spherical coordinates (theta, phi) on S².
    /// IPv4: hash 32 bits onto sphere.
    /// IPv6: first 64 bits -> theta [0, pi], last 64 bits -> phi [0, 2pi].
    pub fn to_sphere(&self) -> (f64, f64) {
        match self {
            IpAddr::V4(addr) => {
                let bits_u32 = u32::from_be_bytes(*addr);
                let bits = bits_u32 as f64;
                let max = u32::MAX as f64;
                let theta = (bits / max) * core::f64::consts::PI;
                let phi_bits = bits_u32.wrapping_mul(2654435761u32) as f64;
                let phi = (phi_bits / max) * 2.0 * core::f64::consts::PI;
                (theta, phi)
            }
            IpAddr::V6(addr) => {
                let hi = u64::from_be_bytes([
                    addr[0], addr[1], addr[2], addr[3], addr[4], addr[5], addr[6], addr[7],
                ]) as f64;
                let lo = u64::from_be_bytes([
                    addr[8], addr[9], addr[10], addr[11], addr[12], addr[13], addr[14], addr[15],
                ]) as f64;
                let max = u64::MAX as f64;
                let theta = (hi / max) * core::f64::consts::PI;
                let phi = (lo / max) * 2.0 * core::f64::consts::PI;
                (theta, phi)
            }
        }
    }

    /// Convert to 3D cartesian point on unit sphere.
    pub fn to_manifold_point(&self) -> aether_core::manifold::ManifoldPoint<3> {
        let (theta, phi) = self.to_sphere();
        let x = libm::sin(theta) * libm::cos(phi);
        let y = libm::sin(theta) * libm::sin(phi);
        let z = libm::cos(theta);
        aether_core::manifold::ManifoldPoint::new([x, y, z])
    }
}

/// Address families.
pub const AF_INET: u16 = 2;
pub const AF_INET6: u16 = 10;

static LOCAL_MAC: Mutex<[u8; 6]> = Mutex::new([0; 6]);
static LOCAL_IP: Mutex<[u8; 4]> = Mutex::new([0, 0, 0, 0]);
static LOCAL_IP_V6: Mutex<[u8; 16]> = Mutex::new([0; 16]);
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
    tcp::init();
    dns::init();
    topological::init();
    firewall::init();
}

pub fn poll() {
    crate::drivers::net::poll();
    dhcp::poll();
    tcp::poll();
    firewall::poll();
    topological::poll();
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
        0x0800 => {
            let pkt = &frame[14..];
            if let Some(info) = firewall::packet_info_from_ipv4(pkt) {
                match firewall::firewall_eval(&info) {
                    firewall::FirewallAction::Allow => ipv4::handle_ipv4_packet(pkt),
                    firewall::FirewallAction::Drop => {}
                    firewall::FirewallAction::Log => ipv4::handle_ipv4_packet(pkt),
                }
            }
        }
        0x86DD => {
            let pkt = &frame[14..];
            if let Some(info) = firewall::packet_info_from_ipv6(pkt) {
                match firewall::firewall_eval(&info) {
                    firewall::FirewallAction::Allow => ipv6::handle_ipv6_packet(pkt),
                    firewall::FirewallAction::Drop => {}
                    firewall::FirewallAction::Log => ipv6::handle_ipv6_packet(pkt),
                }
            }
        }
        _ => {
            // Unknown ethertype (not ARP, IPv4, or IPv6); drop silently
        }
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

pub fn local_ip_v6() -> [u8; 16] {
    *LOCAL_IP_V6.lock()
}

pub fn set_local_ip_v6(ip: [u8; 16]) {
    *LOCAL_IP_V6.lock() = ip;
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
