// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! DHCP client -- DISCOVER, REQUEST, ACK state machine.

use alloc::vec::Vec;
use spin::Mutex;

pub const DHCP_SERVER_PORT: u16 = 67;
pub const DHCP_CLIENT_PORT: u16 = 68;
const DHCP_MAGIC: [u8; 4] = [0x63, 0x82, 0x53, 0x63];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DhcpState {
    Init,
    Discover,
    Request,
    Bound,
}

static DHCP_STATE: Mutex<DhcpState> = Mutex::new(DhcpState::Init);
static OFFERED_IP: Mutex<[u8; 4]> = Mutex::new([0; 4]);
static SERVER_IP: Mutex<[u8; 4]> = Mutex::new([0; 4]);
static XID: Mutex<u32> = Mutex::new(0x12345678);

pub fn init() {
    *DHCP_STATE.lock() = DhcpState::Init;
}

pub fn poll() {
    let state = *DHCP_STATE.lock();
    match state {
        DhcpState::Init => {
            dhcp_discover();
            *DHCP_STATE.lock() = DhcpState::Discover;
        }
        _ => {}
    }
}

fn build_dhcp_packet(msg_type: u8, requested_ip: [u8; 4], server_ip: [u8; 4]) -> Vec<u8> {
    let xid = *XID.lock();
    let mac = crate::net::local_mac();
    let mut pkt = Vec::with_capacity(300);
    pkt.push(1); // op: BOOTREQUEST
    pkt.push(1); // htype: Ethernet
    pkt.push(6); // hlen
    pkt.push(0); // hops
    pkt.extend_from_slice(&xid.to_be_bytes());
    pkt.extend_from_slice(&[0u8; 2]); // secs
    pkt.extend_from_slice(&[0x80, 0x00]); // flags: broadcast
    pkt.extend_from_slice(&[0u8; 4]); // ciaddr
    pkt.extend_from_slice(&requested_ip); // yiaddr (for request)
    pkt.extend_from_slice(&[0u8; 4]); // siaddr
    pkt.extend_from_slice(&[0u8; 4]); // giaddr
    pkt.extend_from_slice(&mac);
    pkt.extend_from_slice(&[0u8; 10]); // chaddr padding
    pkt.extend_from_slice(&[0u8; 64]); // sname
    pkt.extend_from_slice(&[0u8; 128]); // file
    pkt.extend_from_slice(&DHCP_MAGIC);
    pkt.push(53); // DHCP message type option
    pkt.push(1);
    pkt.push(msg_type);
    if msg_type == 3 {
        // DHCP Request
        pkt.push(50);
        pkt.push(4);
        pkt.extend_from_slice(&requested_ip);
        pkt.push(54);
        pkt.push(4);
        pkt.extend_from_slice(&server_ip);
    }
    pkt.push(55); // Parameter request list
    pkt.push(4);
    pkt.push(1);  // subnet mask
    pkt.push(3);  // router
    pkt.push(6);  // DNS
    pkt.push(15); // domain name
    pkt.push(255); // End
    // Pad to 300 bytes minimum
    while pkt.len() < 300 {
        pkt.push(0);
    }
    pkt
}

pub fn dhcp_discover() {
    let pkt = build_dhcp_packet(1, [0; 4], [0; 4]);
    let sock = crate::net::udp::socket();
    crate::net::udp::bind(sock, DHCP_CLIENT_PORT);
    crate::net::udp::sendto(sock, &pkt, [255, 255, 255, 255], DHCP_SERVER_PORT);
}

pub fn handle_dhcp_packet(pkt: &[u8]) {
    if pkt.len() < 240 {
        return;
    }
    let op = pkt[0];
    if op != 2 {
        return;
    }
    let xid = u32::from_be_bytes([pkt[4], pkt[5], pkt[6], pkt[7]]);
    if xid != *XID.lock() {
        return;
    }
    let yiaddr = [pkt[16], pkt[17], pkt[18], pkt[19]];
    let _siaddr = [pkt[20], pkt[21], pkt[22], pkt[23]];

    let mut msg_type = 0;
    let mut server_id = [0u8; 4];
    let mut subnet = [255u8; 4];
    let mut router = [0u8; 4];
    let mut dns = [0u8; 4];

    let mut i = 240;
    while i + 1 < pkt.len() {
        let opt = pkt[i];
        if opt == 255 {
            break;
        }
        if opt == 0 {
            i += 1;
            continue;
        }
        let len = pkt[i + 1] as usize;
        i += 2;
        if i + len > pkt.len() {
            break;
        }
        match opt {
            53 => msg_type = pkt[i],
            54 => server_id.copy_from_slice(&pkt[i..i + 4]),
            1 => subnet.copy_from_slice(&pkt[i..i + 4]),
            3 => router.copy_from_slice(&pkt[i..i + 4]),
            6 => dns.copy_from_slice(&pkt[i..i + 4]),
            _ => {}
        }
        i += len;
    }

    let state = DHCP_STATE.lock();
    match *state {
        DhcpState::Discover => {
            if msg_type == 2 {
                // DHCPOFFER
                *OFFERED_IP.lock() = yiaddr;
                *SERVER_IP.lock() = server_id;
                drop(state);
                let req = build_dhcp_packet(3, yiaddr, server_id);
                let sock = crate::net::udp::socket();
                crate::net::udp::bind(sock, DHCP_CLIENT_PORT);
                crate::net::udp::sendto(sock, &req, [255, 255, 255, 255], DHCP_SERVER_PORT);
                *DHCP_STATE.lock() = DhcpState::Request;
            }
        }
        DhcpState::Request => {
            if msg_type == 5 {
                // DHCPACK
                crate::net::set_local_ip(yiaddr);
                crate::net::set_subnet(subnet);
                crate::net::set_gateway(router);
                crate::net::dns::set_server(dns);
                *DHCP_STATE.lock() = DhcpState::Bound;
                crate::serial_println!(
                    "[DHCP] Bound to {}.{}.{}.{}",
                    yiaddr[0], yiaddr[1], yiaddr[2], yiaddr[3]
                );
            }
        }
        _ => {}
    }
}

pub fn is_bound() -> bool {
    *DHCP_STATE.lock() == DhcpState::Bound
}
