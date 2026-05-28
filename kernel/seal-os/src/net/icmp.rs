// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ICMP echo (ping) support.

use core::sync::atomic::{AtomicBool, Ordering};

pub static ECHO_REPLY_RECEIVED: AtomicBool = AtomicBool::new(false);

#[repr(C, packed)]
struct IcmpEcho {
    icmptype: u8,
    code: u8,
    checksum: u16,
    id: u16,
    seq: u16,
}

pub fn send_echo_request(dst: [u8; 4], seq: u16) {
    let mut pkt = IcmpEcho {
        icmptype: 8,
        code: 0,
        checksum: 0,
        id: 0x1234_u16.to_be(),
        seq: seq.to_be(),
    };
    let bytes = unsafe {
        core::slice::from_raw_parts(
            &pkt as *const _ as *const u8,
            core::mem::size_of::<IcmpEcho>(),
        )
    };
    pkt.checksum = crate::net::ipv4::internet_checksum(bytes);
    let bytes = unsafe {
        core::slice::from_raw_parts(
            &pkt as *const _ as *const u8,
            core::mem::size_of::<IcmpEcho>(),
        )
    };
    crate::net::ipv4::send_ipv4_packet(dst, 1, bytes);
}

fn send_echo_reply(dst: [u8; 4], id: u16, seq: u16, data: &[u8]) {
    let mut buf = alloc::vec::Vec::with_capacity(8 + data.len());
    buf.push(0); // type
    buf.push(0); // code
    buf.push(0);
    buf.push(0); // checksum placeholder
    buf.extend_from_slice(&id.to_be_bytes());
    buf.extend_from_slice(&seq.to_be_bytes());
    buf.extend_from_slice(data);
    let checksum = crate::net::ipv4::internet_checksum(&buf);
    buf[2] = (checksum >> 8) as u8;
    buf[3] = (checksum & 0xFF) as u8;
    crate::net::ipv4::send_ipv4_packet(dst, 1, &buf);
}

pub fn handle_icmp_packet(src: [u8; 4], pkt: &[u8]) {
    if pkt.len() < 8 {
        return;
    }
    let icmptype = pkt[0];
    let code = pkt[1];
    if code != 0 {
        return;
    }
    match icmptype {
        8 => {
            // Echo request
            let id = u16::from_be_bytes([pkt[4], pkt[5]]);
            let seq = u16::from_be_bytes([pkt[6], pkt[7]]);
            send_echo_reply(src, id, seq, &pkt[8..]);
        }
        0 => {
            // Echo reply
            ECHO_REPLY_RECEIVED.store(true, Ordering::SeqCst);
            crate::serial_println!(
                "[ICMP] Echo reply from {}.{}.{}.{}",
                src[0],
                src[1],
                src[2],
                src[3]
            );
        }
        _ => {}
    }
}
