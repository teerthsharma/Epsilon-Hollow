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

// ---------------------------------------------------------------------------
// NEW: DhcpClient with explicit state machine and honest error handling.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DhcpClientState {
    Init,
    Selecting,
    Requesting,
    Bound,
}

/// Fixed-layout DHCP packet header (first 240 bytes before options).
#[derive(Debug, Clone)]
pub struct DhcpPacket {
    pub op: u8,
    pub htype: u8,
    pub hlen: u8,
    pub hops: u8,
    pub xid: u32,
    pub secs: u16,
    pub flags: u16,
    pub ciaddr: [u8; 4],
    pub yiaddr: [u8; 4],
    pub siaddr: [u8; 4],
    pub giaddr: [u8; 4],
    pub chaddr: [u8; 16],
    pub sname: [u8; 64],
    pub file: [u8; 128],
    pub magic: [u8; 4],
    pub options: Vec<u8>,
}

impl DhcpPacket {
    /// Parse a DHCP packet from raw bytes.
    /// Returns `None` if the slice is too short to contain the fixed header.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 240 {
            return None;
        }
        let mut chaddr = [0u8; 16];
        chaddr.copy_from_slice(&bytes[28..44]);
        let mut sname = [0u8; 64];
        sname.copy_from_slice(&bytes[44..108]);
        let mut file = [0u8; 128];
        file.copy_from_slice(&bytes[108..236]);
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[236..240]);
        Some(Self {
            op: bytes[0],
            htype: bytes[1],
            hlen: bytes[2],
            hops: bytes[3],
            xid: u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            secs: u16::from_be_bytes([bytes[8], bytes[9]]),
            flags: u16::from_be_bytes([bytes[10], bytes[11]]),
            ciaddr: [bytes[12], bytes[13], bytes[14], bytes[15]],
            yiaddr: [bytes[16], bytes[17], bytes[18], bytes[19]],
            siaddr: [bytes[20], bytes[21], bytes[22], bytes[23]],
            giaddr: [bytes[24], bytes[25], bytes[26], bytes[27]],
            chaddr,
            sname,
            file,
            magic,
            options: bytes[240..].to_vec(),
        })
    }
}

/// Real DHCP client state machine.
/// All transmit methods return an honest error when no NIC is present.
pub struct DhcpClient {
    pub state: DhcpClientState,
    pub xid: u32,
    pub offered_ip: [u8; 4],
    pub server_ip: [u8; 4],
}

impl DhcpClient {
    pub fn new() -> Self {
        Self {
            state: DhcpClientState::Init,
            xid: 0x12345678,
            offered_ip: [0; 4],
            server_ip: [0; 4],
        }
    }

    /// Transition to Selecting and send DHCPDISCOVER.
    /// Returns `Err("no NIC")` if no network interface is available.
    pub fn discover(&mut self) -> Result<(), &'static str> {
        if !crate::net::has_nic() {
            return Err("no NIC");
        }
        self.state = DhcpClientState::Selecting;
        let pkt = build_dhcp_packet(1, [0; 4], [0; 4], self.xid);
        let sock = crate::net::udp::socket();
        crate::net::udp::bind(sock, DHCP_CLIENT_PORT);
        crate::net::udp::sendto(
            sock,
            &pkt,
            crate::net::IpAddr::V4([255, 255, 255, 255]),
            DHCP_SERVER_PORT,
        );
        Ok(())
    }

    /// Send DHCPREQUEST for the offered IP.
    /// Returns `Err("no NIC")` if no network interface is available.
    pub fn request(&mut self, offered: [u8; 4], server: [u8; 4]) -> Result<(), &'static str> {
        if !crate::net::has_nic() {
            return Err("no NIC");
        }
        self.offered_ip = offered;
        self.server_ip = server;
        self.state = DhcpClientState::Requesting;
        let pkt = build_dhcp_packet(3, offered, server, self.xid);
        let sock = crate::net::udp::socket();
        crate::net::udp::bind(sock, DHCP_CLIENT_PORT);
        crate::net::udp::sendto(
            sock,
            &pkt,
            crate::net::IpAddr::V4([255, 255, 255, 255]),
            DHCP_SERVER_PORT,
        );
        Ok(())
    }

    /// Mark the lease as bound.
    pub fn bind(&mut self, ip: [u8; 4]) {
        crate::net::set_local_ip(ip);
        self.state = DhcpClientState::Bound;
    }
}

// ---------------------------------------------------------------------------
// Legacy module-level DHCP state machine (preserved for compatibility).
// ---------------------------------------------------------------------------

static DHCP_STATE: Mutex<DhcpState> = Mutex::new(DhcpState::Init);
static OFFERED_IP: Mutex<[u8; 4]> = Mutex::new([0; 4]);
static SERVER_IP: Mutex<[u8; 4]> = Mutex::new([0; 4]);
static XID: Mutex<u32> = Mutex::new(0x12345678);

pub fn init() {
    *DHCP_STATE.lock() = DhcpState::Init;
    dhcp_discover();
    *DHCP_STATE.lock() = DhcpState::Discover;
}

pub fn poll() {
    let state = *DHCP_STATE.lock();
    match state {
        DhcpState::Init => {
            dhcp_discover();
            *DHCP_STATE.lock() = DhcpState::Discover;
        }
        DhcpState::Discover | DhcpState::Request | DhcpState::Bound => {
            // No action required in these states during poll
        }
    }
}

fn build_dhcp_packet(msg_type: u8, requested_ip: [u8; 4], server_ip: [u8; 4], xid: u32) -> Vec<u8> {
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
    pkt.push(1); // subnet mask
    pkt.push(3); // router
    pkt.push(6); // DNS
    pkt.push(15); // domain name
    pkt.push(255); // End
                   // Pad to 300 bytes minimum
    while pkt.len() < 300 {
        pkt.push(0);
    }
    pkt
}

pub fn dhcp_discover() {
    let pkt = build_dhcp_packet(1, [0; 4], [0; 4], *XID.lock());
    let sock = crate::net::udp::socket();
    crate::net::udp::bind(sock, DHCP_CLIENT_PORT);
    crate::net::udp::sendto(
        sock,
        &pkt,
        crate::net::IpAddr::V4([255, 255, 255, 255]),
        DHCP_SERVER_PORT,
    );
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
            _ => {
                // Unknown DHCP option; skip per RFC 2131
            }
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
                let req = build_dhcp_packet(3, yiaddr, server_id, *XID.lock());
                let sock = crate::net::udp::socket();
                crate::net::udp::bind(sock, DHCP_CLIENT_PORT);
                crate::net::udp::sendto(
                    sock,
                    &req,
                    crate::net::IpAddr::V4([255, 255, 255, 255]),
                    DHCP_SERVER_PORT,
                );
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
                    yiaddr[0],
                    yiaddr[1],
                    yiaddr[2],
                    yiaddr[3]
                );
            }
        }
        DhcpState::Init => {
            // Unexpected response in Init state; ignore
        }
        DhcpState::Bound => {
            // Already bound; ignore duplicate ACKs
        }
    }
}

pub fn is_bound() -> bool {
    *DHCP_STATE.lock() == DhcpState::Bound
}

pub fn has_lease() -> bool {
    is_bound() || crate::net::local_ip() != [0, 0, 0, 0]
}

pub fn state() -> DhcpState {
    *DHCP_STATE.lock()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_dhcp_offer(yiaddr: [u8; 4], server_id: [u8; 4]) -> Vec<u8> {
        let mut pkt = vec![0u8; 300];
        pkt[0] = 2; // BOOTREPLY
        pkt[1] = 1; // htype Ethernet
        pkt[2] = 6; // hlen
        let xid = *XID.lock();
        pkt[4..8].copy_from_slice(&xid.to_be_bytes());
        pkt[16..20].copy_from_slice(&yiaddr);
        pkt[236..240].copy_from_slice(&DHCP_MAGIC);
        // option 53 = DHCPOFFER
        pkt[240] = 53;
        pkt[241] = 1;
        pkt[242] = 2;
        // option 54 = server id
        pkt[243] = 54;
        pkt[244] = 4;
        pkt[245..249].copy_from_slice(&server_id);
        pkt[249] = 255;
        pkt
    }

    fn build_dhcp_ack(yiaddr: [u8; 4], subnet: [u8; 4], router: [u8; 4], dns: [u8; 4]) -> Vec<u8> {
        let mut pkt = vec![0u8; 300];
        pkt[0] = 2; // BOOTREPLY
        pkt[1] = 1;
        pkt[2] = 6;
        let xid = *XID.lock();
        pkt[4..8].copy_from_slice(&xid.to_be_bytes());
        pkt[16..20].copy_from_slice(&yiaddr);
        pkt[236..240].copy_from_slice(&DHCP_MAGIC);
        // option 53 = DHCPACK
        pkt[240] = 53;
        pkt[241] = 1;
        pkt[242] = 5;
        // option 1 = subnet
        pkt[243] = 1;
        pkt[244] = 4;
        pkt[245..249].copy_from_slice(&subnet);
        // option 3 = router
        pkt[249] = 3;
        pkt[250] = 4;
        pkt[251..255].copy_from_slice(&router);
        // option 6 = dns
        pkt[255] = 6;
        pkt[256] = 4;
        pkt[257..261].copy_from_slice(&dns);
        pkt[261] = 255;
        pkt
    }

    #[test]
    fn test_dhcp_state_transitions() {
        init();
        assert_eq!(state(), DhcpState::Discover);

        poll(); // no-op when already in Discover
        assert_eq!(state(), DhcpState::Discover);

        let offer = build_dhcp_offer([192, 168, 1, 100], [192, 168, 1, 1]);
        handle_dhcp_packet(&offer);
        assert_eq!(state(), DhcpState::Request);

        let ack = build_dhcp_ack(
            [192, 168, 1, 100],
            [255, 255, 255, 0],
            [192, 168, 1, 1],
            [8, 8, 8, 8],
        );
        handle_dhcp_packet(&ack);
        assert_eq!(state(), DhcpState::Bound);
        assert!(is_bound());
        assert_eq!(crate::net::local_ip(), [192, 168, 1, 100]);
        assert_eq!(crate::net::subnet(), [255, 255, 255, 0]);
        assert_eq!(crate::net::gateway(), [192, 168, 1, 1]);
    }

    #[test]
    fn test_dhcp_packet_parser() {
        let mut raw = vec![0u8; 300];
        raw[0] = 2;
        raw[1] = 1;
        raw[2] = 6;
        raw[3] = 0;
        raw[4..8].copy_from_slice(&0xDEADBEEFu32.to_be_bytes());
        raw[16..20].copy_from_slice(&[192, 168, 1, 100]);
        raw[236..240].copy_from_slice(&DHCP_MAGIC);
        let pkt = DhcpPacket::from_bytes(&raw).unwrap();
        assert_eq!(pkt.op, 2);
        assert_eq!(pkt.htype, 1);
        assert_eq!(pkt.hlen, 6);
        assert_eq!(pkt.xid, 0xDEADBEEF);
        assert_eq!(pkt.yiaddr, [192, 168, 1, 100]);
        assert_eq!(pkt.magic, DHCP_MAGIC);
    }

    #[test]
    fn test_dhcp_client_no_nic() {
        let mut client = DhcpClient::new();
        // In unit tests there is no real NIC, so discover must fail honestly.
        assert_eq!(client.discover(), Err("no NIC"));
        assert_eq!(client.state, DhcpClientState::Init);
    }
}
