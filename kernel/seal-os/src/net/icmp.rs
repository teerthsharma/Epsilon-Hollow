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
    // Network order, like `id` and `seq` above: this struct is blitted raw.
    pkt.checksum = crate::net::ipv4::internet_checksum(bytes).to_be();
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
    // RFC 792: the ICMP checksum covers the message and nothing else. There is
    // no pseudo-header here, unlike ICMPv6 (`ipv6::icmpv6_checksum_input`),
    // which is why this handler can verify with only the bytes it was handed
    // and needs no destination address.
    //
    // The field is mandatory, so there is no "not computed" encoding to admit
    // the way a zero UDP checksum must be admitted over IPv4: a message whose
    // checksum reads zero is verified like any other, and passes only if the
    // rest of the message sums to zero as well.
    //
    // `handle_ipv4_packet` hands over `&pkt[ihl..total_len]`, so Ethernet
    // padding on a short frame is already outside this slice and cannot make a
    // sound message fail.
    if crate::net::ipv4::internet_checksum(pkt) != 0 {
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

// ---------------------------------------------------------------------------
// Tests -- run by the in-kernel harness (crate::testing), not `cargo test`.
// `testing/runner.rs` does not name this group; `net::ipv6::tests::register_all`
// chains it, the way `drivers::net::tls` chains `x509` and `ecdhe`.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;

    /// An echo message of `icmptype` carrying a correct RFC 792 checksum.
    fn echo(icmptype: u8, seq: u16) -> alloc::vec::Vec<u8> {
        let mut msg = alloc::vec![icmptype, 0, 0, 0, 0x12, 0x34, 0, 0];
        msg[6..8].copy_from_slice(&seq.to_be_bytes());
        let checksum = crate::net::ipv4::internet_checksum(&msg);
        msg[2..4].copy_from_slice(&checksum.to_be_bytes());
        msg
    }

    /// An echo request is the one ICMP message this stack answers, so a
    /// corrupted one that is processed anyway becomes a reply built from
    /// whatever the corruption produced -- an identifier and sequence the peer
    /// never sent, and the corrupted payload echoed back verbatim.
    ///
    /// The reply is counted through the loopback dispatch counter rather than a
    /// transmit hook, so this needs no NIC: a request sourced from 127.0.0.1 is
    /// answered to 127.0.0.1, which `ipv4::send_ipv4_packet` delivers in-process
    /// and counts. The second half is the control -- the same request with a
    /// checksum that matches is still answered -- so a verifier that refuses
    /// every message cannot pass this.
    fn test_corrupt_echo_request_not_answered() -> TestResult {
        let src = [127, 0, 0, 1];

        let mut corrupt = echo(8, 1);
        corrupt[5] ^= 0x01; // one bit of the identifier, checksum left stale
        let before = crate::net::ipv4::loopback_dispatched();
        handle_icmp_packet(src, &corrupt);
        test_assert!(
            crate::net::ipv4::loopback_dispatched() == before,
            "an ICMP echo request whose checksum does not verify was answered"
        );

        handle_icmp_packet(src, &echo(8, 2));
        test_assert!(
            crate::net::ipv4::loopback_dispatched() > before,
            "an ICMP echo request whose checksum verifies was not answered"
        );
        TestResult::Pass
    }

    /// The echo reply arm records that the peer answered. A corrupted reply is
    /// not evidence of anything: the source address it appears to come from is
    /// as unverified as the rest of the message.
    fn test_corrupt_echo_reply_not_recorded() -> TestResult {
        let src = [192, 0, 2, 1];

        let mut corrupt = echo(0, 3);
        corrupt[7] ^= 0x01; // one bit of the sequence, checksum left stale
        ECHO_REPLY_RECEIVED.store(false, Ordering::SeqCst);
        handle_icmp_packet(src, &corrupt);
        test_assert!(
            !ECHO_REPLY_RECEIVED.load(Ordering::SeqCst),
            "an ICMP echo reply whose checksum does not verify was recorded as an answer"
        );

        handle_icmp_packet(src, &echo(0, 4));
        test_assert!(
            ECHO_REPLY_RECEIVED.load(Ordering::SeqCst),
            "an ICMP echo reply whose checksum verifies was not recorded"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "icmp::corrupt_echo_request_not_answered",
            test_corrupt_echo_request_not_answered,
        );
        crate::testing::register_test(
            "icmp::corrupt_echo_reply_not_recorded",
            test_corrupt_echo_reply_not_recorded,
        );
    }
}
