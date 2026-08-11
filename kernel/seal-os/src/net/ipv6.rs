// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! IPv6 protocol -- fixed 40-byte header, ICMPv6, NDP (ARP replacement).

use alloc::vec::Vec;
use spin::Mutex;

#[repr(C, packed)]
pub struct Ipv6Header {
    pub ver_tc_fl: [u8; 4], // version(4), traffic class(8), flow label(20)
    pub payload_len: u16,
    pub next_header: u8,
    pub hop_limit: u8,
    pub src: [u8; 16],
    pub dst: [u8; 16],
}

impl Ipv6Header {
    pub fn version(&self) -> u8 {
        self.ver_tc_fl[0] >> 4
    }
    pub fn traffic_class(&self) -> u8 {
        ((self.ver_tc_fl[0] & 0x0F) << 4) | ((self.ver_tc_fl[1] & 0xF0) >> 4)
    }
    pub fn flow_label(&self) -> u32 {
        ((self.ver_tc_fl[1] as u32 & 0x0F) << 16)
            | ((self.ver_tc_fl[2] as u32) << 8)
            | (self.ver_tc_fl[3] as u32)
    }

    pub fn new(next_header: u8, src: [u8; 16], dst: [u8; 16], payload_len: usize) -> Self {
        Self {
            ver_tc_fl: [0x60, 0, 0, 0], // version 6
            payload_len: (payload_len as u16).to_be(),
            next_header,
            hop_limit: 64,
            src,
            dst,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 40 {
            return None;
        }
        if bytes[0] >> 4 != 6 {
            return None;
        }
        Some(Self {
            ver_tc_fl: [bytes[0], bytes[1], bytes[2], bytes[3]],
            payload_len: u16::from_be_bytes([bytes[4], bytes[5]]),
            next_header: bytes[6],
            hop_limit: bytes[7],
            src: [
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15], bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21],
                bytes[22], bytes[23],
            ],
            dst: [
                bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30],
                bytes[31], bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37],
                bytes[38], bytes[39],
            ],
        })
    }
}

pub fn send_ipv6_packet(dst: [u8; 16], next_header: u8, payload: &[u8]) {
    if dst == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] {
        // Loopback
        let mut pkt = Vec::with_capacity(40 + payload.len());
        let src = crate::net::local_ip_v6();
        let hdr = Ipv6Header::new(next_header, src, dst, payload.len());
        let hdr_bytes = unsafe {
            core::slice::from_raw_parts(
                &hdr as *const _ as *const u8,
                core::mem::size_of::<Ipv6Header>(),
            )
        };
        pkt.extend_from_slice(hdr_bytes);
        pkt.extend_from_slice(payload);
        handle_ipv6_packet(&pkt);
        return;
    }

    let src = crate::net::local_ip_v6();
    let src_mac = crate::net::local_mac();
    let dst_mac = ndp_lookup(dst).unwrap_or([0x33, 0x33, 0x00, 0x00, 0x00, 0x01]);

    let mut frame = Vec::with_capacity(14 + 40 + payload.len());
    frame.extend_from_slice(&dst_mac);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&0x86DD_u16.to_be_bytes());

    let hdr = Ipv6Header::new(next_header, src, dst, payload.len());
    let hdr_bytes = unsafe {
        core::slice::from_raw_parts(
            &hdr as *const _ as *const u8,
            core::mem::size_of::<Ipv6Header>(),
        )
    };
    frame.extend_from_slice(hdr_bytes);
    frame.extend_from_slice(payload);
    crate::net::transmit(&frame);
}

pub fn handle_ipv6_packet(pkt: &[u8]) {
    if pkt.len() < core::mem::size_of::<Ipv6Header>() {
        return;
    }
    let hdr = unsafe { &*(pkt.as_ptr() as *const Ipv6Header) };
    if hdr.version() != 6 {
        return;
    }
    let payload_len = u16::from_be(hdr.payload_len) as usize;
    if payload_len > pkt.len() - 40 {
        return;
    }
    let payload = &pkt[40..40 + payload_len];
    let next_header = hdr.next_header;
    let src = hdr.src;
    let dst = hdr.dst;

    match next_header {
        58 => handle_icmpv6_packet(src, dst, payload),
        // Both addresses travel on for the same reason ICMPv6 needs them: RFC
        // 8200 section 8.1 sums them into the pseudo-header, and IPv6 has no
        // header checksum, so this is the only integrity check either transport
        // gets.
        6 => crate::net::tcp::handle_tcp_packet(
            crate::net::IpAddr::V6(src),
            crate::net::IpAddr::V6(dst),
            payload,
        ),
        17 => crate::net::udp::handle_udp_packet(
            crate::net::IpAddr::V6(src),
            crate::net::IpAddr::V6(dst),
            payload,
        ),
        _ => {
            // Unknown IPv6 next-header; drop silently
        }
    }
}

// ---------------------------------------------------------------------------
// ICMPv6
// ---------------------------------------------------------------------------

const ICMPV6_ECHO_REQUEST: u8 = 128;
const ICMPV6_ECHO_REPLY: u8 = 129;
const ICMPV6_NEIGHBOR_SOLICIT: u8 = 135;
const ICMPV6_NEIGHBOR_ADVERT: u8 = 136;

#[repr(C, packed)]
struct Icmpv6Echo {
    icmptype: u8,
    code: u8,
    checksum: u16,
    id: u16,
    seq: u16,
}

/// The RFC 8200 section 8.1 pseudo-header for an ICMPv6 message, followed by
/// the message itself: the byte string whose one's-complement sum is the
/// checksum RFC 4443 section 2.3 requires.
///
/// Unlike ICMPv4 (`icmp::handle_icmp_packet`) the addresses are part of the
/// sum, so a receiver needs both ends of the IPv6 header to check it -- which
/// is why `handle_icmpv6_packet` takes a destination as well as a source.
///
/// Every site that produces or checks an ICMPv6 checksum builds its input
/// here. Four senders each assembled their own copy of these 40 bytes, and a
/// verifier that assembled a fifth could disagree with any of them and reject
/// this stack's own traffic.
fn icmpv6_checksum_input(src: [u8; 16], dst: [u8; 16], msg: &[u8]) -> Vec<u8> {
    let mut pseudo = Vec::with_capacity(40 + msg.len());
    pseudo.extend_from_slice(&src);
    pseudo.extend_from_slice(&dst);
    pseudo.extend_from_slice(&(msg.len() as u32).to_be_bytes());
    pseudo.push(0);
    pseudo.push(0);
    pseudo.push(0);
    pseudo.push(58);
    pseudo.extend_from_slice(msg);
    pseudo
}

pub fn send_icmpv6_echo_request(dst: [u8; 16], seq: u16) {
    let src = crate::net::local_ip_v6();
    let mut pkt = Icmpv6Echo {
        icmptype: ICMPV6_ECHO_REQUEST,
        code: 0,
        checksum: 0,
        id: 0x1234_u16.to_be(),
        seq: seq.to_be(),
    };
    let bytes = unsafe { core::slice::from_raw_parts(&pkt as *const _ as *const u8, 8) };
    let cksum = crate::net::ipv4::internet_checksum(&icmpv6_checksum_input(src, dst, bytes));
    // Network order, like `id` and `seq` above: this struct is blitted raw.
    // The three explicit-shift sites below already write network order.
    pkt.checksum = cksum.to_be();
    let bytes = unsafe { core::slice::from_raw_parts(&pkt as *const _ as *const u8, 8) };
    send_ipv6_packet(dst, 58, bytes);
}

fn send_icmpv6_echo_reply(dst: [u8; 16], id: u16, seq: u16, data: &[u8]) {
    let src = crate::net::local_ip_v6();
    let mut buf = Vec::with_capacity(8 + data.len());
    buf.push(ICMPV6_ECHO_REPLY);
    buf.push(0);
    buf.push(0);
    buf.push(0);
    buf.extend_from_slice(&id.to_be_bytes());
    buf.extend_from_slice(&seq.to_be_bytes());
    buf.extend_from_slice(data);
    let checksum = crate::net::ipv4::internet_checksum(&icmpv6_checksum_input(src, dst, &buf));
    buf[2] = (checksum >> 8) as u8;
    buf[3] = (checksum & 0xFF) as u8;
    send_ipv6_packet(dst, 58, &buf);
}

fn send_neighbor_advertisement(target: [u8; 16], dst: [u8; 16]) {
    let src = crate::net::local_ip_v6();
    let mut buf = Vec::with_capacity(32);
    buf.push(ICMPV6_NEIGHBOR_ADVERT);
    buf.push(0);
    buf.push(0);
    buf.push(0); // checksum placeholder
                 // Flags: Solicited + Override
    buf.extend_from_slice(&0x6000_0000u32.to_be_bytes());
    buf.extend_from_slice(&target);
    // Source link-layer address option
    buf.push(2); // type
    buf.push(1); // length (1 * 8 bytes)
    let mac = crate::net::local_mac();
    buf.extend_from_slice(&mac);
    while buf.len() % 8 != 0 {
        buf.push(0);
    }

    let checksum = crate::net::ipv4::internet_checksum(&icmpv6_checksum_input(src, dst, &buf));
    buf[2] = (checksum >> 8) as u8;
    buf[3] = (checksum & 0xFF) as u8;
    send_ipv6_packet(dst, 58, &buf);
}

struct NdCacheEntry {
    mac: [u8; 6],
    expires_at: u64,
}

static ND_CACHE: Mutex<Vec<([u8; 16], NdCacheEntry)>> = Mutex::new(Vec::new());

fn ticks() -> u64 {
    crate::drivers::interrupts::ticks()
}

fn is_expired(e: &NdCacheEntry) -> bool {
    ticks().wrapping_sub(e.expires_at) < 0x8000_0000_0000_0000 && ticks() >= e.expires_at
}

pub fn ndp_lookup(ip: [u8; 16]) -> Option<[u8; 6]> {
    let cache = ND_CACHE.lock();
    for (cached_ip, entry) in cache.iter() {
        if *cached_ip == ip && !is_expired(entry) {
            return Some(entry.mac);
        }
    }
    drop(cache);
    let _ = send_neighbor_solicitation(ip);
    None
}

/// Records `mac` as the link-layer address of `ip`, replacing whatever was
/// cached for it. Callers that hold an unauthenticated address -- everything
/// arriving off the wire -- go through `ndp_record` instead.
pub fn insert_ndp(ip: [u8; 16], mac: [u8; 6]) {
    ndp_record(ip, mac, true);
}

/// With `overwrite` false the entry is only created, never replaced: a live
/// mapping keeps the address it already resolved to. Expiry still applies, so
/// an entry that has aged out is replaced like any empty slot.
fn ndp_record(ip: [u8; 16], mac: [u8; 6], overwrite: bool) {
    let mut cache = ND_CACHE.lock();
    for (cached_ip, entry) in cache.iter_mut() {
        if *cached_ip == ip {
            if !overwrite && !is_expired(entry) {
                return;
            }
            entry.mac = mac;
            entry.expires_at = ticks().wrapping_add(300_000);
            return;
        }
    }
    cache.push((
        ip,
        NdCacheEntry {
            mac,
            expires_at: ticks().wrapping_add(300_000),
        },
    ));
}

fn send_neighbor_solicitation(target: [u8; 16]) -> Result<(), &'static str> {
    if !crate::net::has_nic() {
        return Err("no NIC");
    }
    let src = crate::net::local_ip_v6();
    let src_mac = crate::net::local_mac();
    // Solicited-node multicast address
    let dst = [
        0xFF, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xFF, target[13],
        target[14], target[15],
    ];
    let dst_mac = [0x33, 0x33, dst[12], dst[13], dst[14], dst[15]];

    let mut buf = Vec::with_capacity(32);
    buf.push(ICMPV6_NEIGHBOR_SOLICIT);
    buf.push(0);
    buf.push(0);
    buf.push(0);
    buf.extend_from_slice(&0u32.to_be_bytes()); // reserved
    buf.extend_from_slice(&target);
    // Source link-layer address option
    buf.push(1); // type
    buf.push(1); // length
    buf.extend_from_slice(&src_mac);
    while buf.len() % 8 != 0 {
        buf.push(0);
    }

    let checksum = crate::net::ipv4::internet_checksum(&icmpv6_checksum_input(src, dst, &buf));
    buf[2] = (checksum >> 8) as u8;
    buf[3] = (checksum & 0xFF) as u8;

    let mut frame = Vec::with_capacity(14 + 40 + buf.len());
    frame.extend_from_slice(&dst_mac);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&0x86DD_u16.to_be_bytes());
    let hdr = Ipv6Header::new(58, src, dst, buf.len());
    let hdr_bytes = unsafe { core::slice::from_raw_parts(&hdr as *const _ as *const u8, 40) };
    frame.extend_from_slice(hdr_bytes);
    frame.extend_from_slice(&buf);
    crate::net::transmit(&frame);
    Ok(())
}

/// `dst` is the destination of the IPv6 header this message arrived in, not
/// necessarily this host's address: a Neighbor Solicitation arrives at a
/// solicited-node multicast address, and the sender summed that address, so
/// substituting `local_ip_v6()` here would reject every solicitation.
pub fn handle_icmpv6_packet(src: [u8; 16], dst: [u8; 16], pkt: &[u8]) {
    if pkt.len() < 8 {
        return;
    }
    // RFC 4443 section 2.3: the checksum covers the IPv6 pseudo-header as well
    // as the message, and it is mandatory -- IPv6 has no header checksum of its
    // own, so this is the only integrity check anything in this file gets.
    // Unlike UDP over IPv4 there is no "not computed" encoding: a zero field is
    // summed like any other value and passes only if the message sums to zero
    // with it.
    if crate::net::ipv4::internet_checksum(&icmpv6_checksum_input(src, dst, pkt)) != 0 {
        return;
    }
    let icmptype = pkt[0];
    let code = pkt[1];
    if code != 0 {
        return;
    }
    match icmptype {
        ICMPV6_ECHO_REQUEST => {
            let id = u16::from_be_bytes([pkt[4], pkt[5]]);
            let seq = u16::from_be_bytes([pkt[6], pkt[7]]);
            send_icmpv6_echo_reply(src, id, seq, &pkt[8..]);
        }
        ICMPV6_ECHO_REPLY => {
            crate::serial_println!("[ICMPv6] Echo reply from {:02x}{:02x}:...", src[0], src[1]);
        }
        ICMPV6_NEIGHBOR_SOLICIT => {
            if pkt.len() < 24 {
                return;
            }
            let target = [
                pkt[8], pkt[9], pkt[10], pkt[11], pkt[12], pkt[13], pkt[14], pkt[15], pkt[16],
                pkt[17], pkt[18], pkt[19], pkt[20], pkt[21], pkt[22], pkt[23],
            ];
            let local = crate::net::local_ip_v6();
            if target == local {
                send_neighbor_advertisement(target, src);
            }
        }
        ICMPV6_NEIGHBOR_ADVERT => {
            // RFC 4861 4.4: flags octet at 4, target address at 8..24, then
            // options. A target link-layer address option is type 2 with
            // length 1 (one 8-octet unit): two header bytes and a 6-byte MAC.
            // Anything shorter carries no address, and caching a placeholder
            // for it blackholes the target -- `ndp_lookup` would then hand
            // `send_ipv6_packet` an all-zero destination MAC for every frame
            // until the entry expires.
            //
            // ponytail: only the option at 24 is read, so an advertisement
            // that puts another option ahead of the link-layer address is
            // dropped rather than parsed. That fails closed. Walk the option
            // chain by its length fields if a peer is seen to send one.
            if pkt.len() < 32 || pkt[24] != 2 || pkt[25] != 1 {
                return;
            }
            let target = [
                pkt[8], pkt[9], pkt[10], pkt[11], pkt[12], pkt[13], pkt[14], pkt[15], pkt[16],
                pkt[17], pkt[18], pkt[19], pkt[20], pkt[21], pkt[22], pkt[23],
            ];
            let mut mac = [0u8; 6];
            mac.copy_from_slice(&pkt[26..32]);
            // The all-zero address blackholes the target and a group address
            // (low bit of the first octet, which covers ff:ff:ff:ff:ff:ff)
            // floods the link; neither is a unicast link-layer address. A
            // multicast or unspecified target is not a neighbour at all.
            if mac == [0u8; 6] || mac[0] & 0x01 != 0 || target[0] == 0xFF || target == [0u8; 16] {
                return;
            }
            // Nothing records which solicitations went out, so an
            // advertisement cannot be matched against one and "unsolicited"
            // cannot be detected. What is left is to refuse the takeover: an
            // advertisement may fill an entry that does not resolve yet, and
            // only one that claims Solicited and Override -- the pair this
            // stack's own replies set -- may replace one that does.
            ndp_record(target, mac, pkt[4] & 0x60 == 0x60);
        }
        _ => {
            // Unknown ICMPv6 type; drop silently
        }
    }
}

// ---------------------------------------------------------------------------
// Tests -- run by the in-kernel harness (crate::testing), not `cargo test`.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;

    const MAC_A: [u8; 6] = [0x52, 0x54, 0x00, 0xAA, 0xAA, 0x01];
    const MAC_B: [u8; 6] = [0x52, 0x54, 0x00, 0xBB, 0xBB, 0x02];

    /// RFC 4861 4.4 flags octet.
    const F_SOLICITED: u8 = 0x40;
    const F_OVERRIDE: u8 = 0x20;

    /// Builds a Neighbor Advertisement as it arrives at `handle_icmpv6_packet`:
    /// type, code, two checksum bytes, the flags octet, three reserved bytes
    /// and the 16-byte target. With `opt` it carries a target link-layer
    /// address option (type 2, length 1); without it the message is the
    /// 24-byte minimum, which is what the length guard admits.
    ///
    /// The checksum is left zero here and stamped by `checksummed`, which is
    /// the only thing that knows the addresses it has to be summed against.
    fn advert(flags: u8, target: [u8; 16], opt: Option<[u8; 6]>) -> Vec<u8> {
        let mut pkt = Vec::with_capacity(32);
        pkt.push(ICMPV6_NEIGHBOR_ADVERT);
        pkt.push(0); // code
        pkt.push(0); // checksum, stamped by `checksummed`
        pkt.push(0);
        pkt.push(flags);
        pkt.push(0); // reserved
        pkt.push(0);
        pkt.push(0);
        pkt.extend_from_slice(&target);
        if let Some(mac) = opt {
            pkt.push(2); // target link-layer address
            pkt.push(1); // one 8-octet unit
            pkt.extend_from_slice(&mac);
        }
        pkt
    }

    /// A distinct target per test, so the shared cache carries no state from
    /// one case into the next.
    fn target(n: u8) -> [u8; 16] {
        [0xFD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n]
    }

    /// Stamp the RFC 4443 checksum a real sender would have computed, then hand
    /// the message to the receive path addressed to this host, the way
    /// `handle_ipv6_packet` does. Every case below goes through here so that
    /// what they assert stays a statement about the NDP rules and not about the
    /// checksum -- except `advert_with_bad_checksum_not_cached`, which corrupts
    /// the message afterwards on purpose.
    fn deliver(src: [u8; 16], pkt: &[u8]) {
        let dst = crate::net::local_ip_v6();
        handle_icmpv6_packet(src, dst, &checksummed(src, dst, pkt));
    }

    /// `pkt` with the checksum field filled in, exactly as
    /// `send_neighbor_advertisement` fills it. The field is cleared first, so a
    /// message that already carries one -- a corrupted message being repaired --
    /// is summed the way a sender would sum it rather than over its own stale
    /// checksum.
    fn checksummed(src: [u8; 16], dst: [u8; 16], pkt: &[u8]) -> Vec<u8> {
        let mut pkt = pkt.to_vec();
        pkt[2] = 0;
        pkt[3] = 0;
        let checksum = crate::net::ipv4::internet_checksum(&icmpv6_checksum_input(src, dst, &pkt));
        pkt[2] = (checksum >> 8) as u8;
        pkt[3] = (checksum & 0xFF) as u8;
        pkt
    }

    /// Reads the cache directly rather than through `ndp_lookup`, which emits a
    /// solicitation on a miss.
    fn cached(ip: [u8; 16]) -> Option<[u8; 6]> {
        ND_CACHE
            .lock()
            .iter()
            .find(|(cached_ip, _)| *cached_ip == ip)
            .map(|(_, entry)| entry.mac)
    }

    /// An advertisement with no link-layer address option carries no address to
    /// cache. Installing a placeholder makes `ndp_lookup` return
    /// `00:00:00:00:00:00`, which `send_ipv6_packet` writes as the destination
    /// of every frame to that address for the entry's lifetime.
    fn test_advert_without_option_not_cached() -> TestResult {
        let t = target(0x11);
        let src = target(0xEE);
        deliver(src, &advert(F_SOLICITED | F_OVERRIDE, t, None));
        test_assert!(
            cached(t).is_none(),
            "a 24-byte Neighbor Advertisement with no link-layer option was cached"
        );
        TestResult::Pass
    }

    /// Control: an advertisement that does carry an option still resolves, so
    /// the guard above does not disable neighbour discovery.
    fn test_advert_with_option_cached() -> TestResult {
        let t = target(0x12);
        let src = target(0xEE);
        deliver(src, &advert(F_SOLICITED | F_OVERRIDE, t, Some(MAC_A)));
        test_assert!(
            cached(t) == Some(MAC_A),
            "a Neighbor Advertisement carrying a link-layer option did not resolve"
        );
        TestResult::Pass
    }

    /// Nothing records which solicitations went out, so an advertisement cannot
    /// be matched against one. What is left is to refuse the takeover: an
    /// advertisement may fill an empty entry, and only a solicited one that
    /// also sets Override may replace an entry that already resolves.
    fn test_unsolicited_advert_does_not_overwrite() -> TestResult {
        let t = target(0x13);
        let src = target(0xEE);
        deliver(src, &advert(F_SOLICITED | F_OVERRIDE, t, Some(MAC_A)));
        test_assert!(cached(t) == Some(MAC_A), "setup advertisement did not cache");

        deliver(src, &advert(F_OVERRIDE, t, Some(MAC_B)));
        test_assert!(
            cached(t) == Some(MAC_A),
            "an unsolicited Neighbor Advertisement replaced a resolved entry"
        );

        // RFC 4861 7.2.5: with Override clear, a cached address that differs
        // is kept whatever else the advertisement claims.
        deliver(src, &advert(F_SOLICITED, t, Some(MAC_B)));
        test_assert!(
            cached(t) == Some(MAC_A),
            "an advertisement without Override replaced a resolved entry"
        );

        deliver(src, &advert(F_SOLICITED | F_OVERRIDE, t, Some(MAC_B)));
        test_assert!(
            cached(t) == Some(MAC_B),
            "a solicited Override advertisement failed to update a resolved entry"
        );
        TestResult::Pass
    }

    /// An option can carry an address that is unusable as a unicast
    /// destination: all-zero blackholes the target and a group address floods
    /// it. A multicast or unspecified target is not a neighbour at all.
    fn test_unusable_addresses_not_cached() -> TestResult {
        let src = target(0xEE);

        let zero = target(0x14);
        deliver(
            src,
            &advert(F_SOLICITED | F_OVERRIDE, zero, Some([0u8; 6])),
        );
        test_assert!(
            cached(zero).is_none(),
            "an all-zero link-layer address was cached"
        );

        let bcast = target(0x15);
        deliver(
            src,
            &advert(F_SOLICITED | F_OVERRIDE, bcast, Some([0xFF; 6])),
        );
        test_assert!(
            cached(bcast).is_none(),
            "a broadcast link-layer address was cached"
        );

        let mcast = [
            0xFF, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01,
        ];
        deliver(src, &advert(F_SOLICITED | F_OVERRIDE, mcast, Some(MAC_A)));
        test_assert!(
            cached(mcast).is_none(),
            "a multicast target address was cached as a neighbour"
        );

        let unspec = [0u8; 16];
        deliver(src, &advert(F_SOLICITED | F_OVERRIDE, unspec, Some(MAC_A)));
        test_assert!(
            cached(unspec).is_none(),
            "the unspecified target address was cached as a neighbour"
        );
        TestResult::Pass
    }

    /// The option header has to say what it holds. An option that is not a
    /// target link-layer address, or one that declares a length other than a
    /// single 8-octet unit, does not carry six bytes of Ethernet address at
    /// 26..32 -- reading them anyway caches whatever happened to be there.
    fn test_advert_with_malformed_option_not_cached() -> TestResult {
        let src = target(0xEE);

        let wrong_type = target(0x16);
        let mut pkt = advert(F_SOLICITED | F_OVERRIDE, wrong_type, Some(MAC_A));
        pkt[24] = 1; // source link-layer address, not the target's
        deliver(src, &pkt);
        test_assert!(
            cached(wrong_type).is_none(),
            "an option that is not a target link-layer address was cached as one"
        );

        let wrong_len = target(0x17);
        let mut pkt = advert(F_SOLICITED | F_OVERRIDE, wrong_len, Some(MAC_A));
        pkt[25] = 2; // claims 16 octets, of which only 8 arrived
        deliver(src, &pkt);
        test_assert!(
            cached(wrong_len).is_none(),
            "a truncated link-layer option was cached from its first six bytes"
        );
        TestResult::Pass
    }

    /// IPv6 carries no header checksum, so RFC 4443's is the only integrity
    /// check an ICMPv6 message gets. Without it a single flipped bit anywhere
    /// in a Neighbor Advertisement -- including in the target address or the
    /// link-layer option this stack then caches -- is taken at face value, and
    /// `ndp_lookup` addresses every later frame with whatever it produced.
    ///
    /// The message is checksummed first and corrupted afterwards, so it fails
    /// for exactly one reason: every other guard in the Neighbor Advertisement
    /// arm still passes. The second half is the control -- the same message
    /// checksummed after the corruption resolves normally -- so this cannot be
    /// satisfied by a verifier that rejects everything.
    fn test_advert_with_bad_checksum_not_cached() -> TestResult {
        let src = target(0xEE);
        let dst = crate::net::local_ip_v6();

        let t = target(0x18);
        let mut pkt = checksummed(src, dst, &advert(F_SOLICITED | F_OVERRIDE, t, Some(MAC_A)));
        pkt[23] ^= 0x01; // last bit of the target address, checksum left stale
        handle_icmpv6_packet(src, dst, &pkt);
        test_assert!(
            cached(target(0x19)).is_none(),
            "a corrupted Neighbor Advertisement was cached under the address the flipped bit produced"
        );
        test_assert!(
            cached(t).is_none(),
            "a corrupted Neighbor Advertisement was cached under its original address"
        );

        deliver(src, &pkt);
        test_assert!(
            cached(target(0x19)) == Some(MAC_A),
            "the same message with a checksum matching its contents did not resolve"
        );
        TestResult::Pass
    }

    /// The destination summed into the pseudo-header is the one the datagram
    /// carried, not this host's address. They differ for every message sent to
    /// a multicast group -- which is every Neighbor Solicitation, addressed to
    /// a solicited-node address -- so a verifier that substituted
    /// `local_ip_v6()` would verify this stack's loopback traffic and its own
    /// unit tests and then refuse the whole of neighbour discovery on a real
    /// link.
    fn test_advert_verified_against_the_delivered_destination() -> TestResult {
        let src = target(0xEE);
        let t = target(0x1A);
        let mcast = [
            0xFF, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0xFF, t[13], t[14], t[15],
        ];
        test_assert!(
            mcast != crate::net::local_ip_v6(),
            "the fixture destination equals the local address, so this proves nothing"
        );
        let pkt = checksummed(src, mcast, &advert(F_SOLICITED | F_OVERRIDE, t, Some(MAC_A)));
        handle_icmpv6_packet(src, mcast, &pkt);
        test_assert!(
            cached(t) == Some(MAC_A),
            "a message checksummed against the address it was actually sent to was rejected"
        );
        TestResult::Pass
    }

    /// The IPv6 header a transport segment arrives in. There is no header
    /// checksum to compute: that absence is exactly why RFC 8200 section 8.1
    /// makes the transport checksum mandatory here.
    fn ipv6_frame(src: [u8; 16], dst: [u8; 16], next_header: u8, segment: &[u8]) -> Vec<u8> {
        let mut pkt = Vec::with_capacity(40 + segment.len());
        pkt.extend_from_slice(&[0x60, 0, 0, 0]);
        pkt.extend_from_slice(&(segment.len() as u16).to_be_bytes());
        pkt.push(next_header);
        pkt.push(64); // hop limit
        pkt.extend_from_slice(&src);
        pkt.extend_from_slice(&dst);
        pkt.extend_from_slice(segment);
        pkt
    }

    /// `segment` with its checksum field recomputed at `offset` over the RFC
    /// 8200 pseudo-header, the way a sender computes it. Built here rather than
    /// through `icmpv6_checksum_input`, which is fixed to next-header 58.
    fn stamped_v6(
        src: [u8; 16],
        dst: [u8; 16],
        next_header: u8,
        offset: usize,
        segment: &[u8],
    ) -> Vec<u8> {
        let mut seg = segment.to_vec();
        seg[offset] = 0;
        seg[offset + 1] = 0;
        let mut p = Vec::with_capacity(40 + seg.len());
        p.extend_from_slice(&src);
        p.extend_from_slice(&dst);
        p.extend_from_slice(&(seg.len() as u32).to_be_bytes());
        p.push(0);
        p.push(0);
        p.push(0);
        p.push(next_header);
        p.extend_from_slice(&seg);
        let ck = crate::net::ipv4::internet_checksum(&p);
        seg[offset..offset + 2].copy_from_slice(&ck.to_be_bytes());
        seg
    }

    fn udp_segment_v6(
        src: [u8; 16],
        dst: [u8; 16],
        src_port: u16,
        dst_port: u16,
        payload: &[u8],
    ) -> Vec<u8> {
        let mut seg = Vec::with_capacity(8 + payload.len());
        seg.extend_from_slice(&src_port.to_be_bytes());
        seg.extend_from_slice(&dst_port.to_be_bytes());
        seg.extend_from_slice(&((8 + payload.len()) as u16).to_be_bytes());
        seg.extend_from_slice(&[0, 0]);
        seg.extend_from_slice(payload);
        stamped_v6(src, dst, 17, 6, &seg)
    }

    /// IPv6 has no header checksum, so the transport checksum is the only
    /// integrity check a UDP datagram or a TCP segment gets on this path --
    /// and nothing read either field. A corrupted datagram was buffered and
    /// handed to a reader verbatim.
    ///
    /// The datagram is checksummed first and corrupted afterwards; the control
    /// is the same corrupted bytes with a checksum recomputed over them.
    fn test_udp_checksum_verified_on_receive_v6() -> TestResult {
        const PORT: u16 = 9110;
        let src = target(0xE1);
        let dst = crate::net::local_ip_v6();
        let idx = crate::net::udp::socket();
        crate::net::udp::bind(idx, PORT);

        let mut corrupt = udp_segment_v6(src, dst, 40_010, PORT, b"payload");
        corrupt[9] ^= 0x01; // one bit of the payload, checksum left stale
        handle_ipv6_packet(&ipv6_frame(src, dst, 17, &corrupt));
        let mut buf = [0u8; 32];
        test_assert!(
            crate::net::udp::recvfrom(idx, &mut buf).is_none(),
            "a UDP datagram over IPv6 whose checksum does not verify was delivered to a socket"
        );

        let repaired = stamped_v6(src, dst, 17, 6, &corrupt);
        handle_ipv6_packet(&ipv6_frame(src, dst, 17, &repaired));
        test_assert!(
            crate::net::udp::recvfrom(idx, &mut buf).is_some(),
            "the same datagram with a checksum matching its contents was not delivered"
        );
        TestResult::Pass
    }

    /// RFC 8200 section 8.1: unlike IPv4, a UDP datagram over IPv6 has no
    /// "not computed" encoding -- a zero checksum field is illegal and the
    /// datagram must be discarded. A receiver that shared one code path with
    /// the IPv4 rule accepts these, and every unchecked datagram then arrives
    /// with no integrity check anywhere in the stack behind it.
    ///
    /// The control is the same datagram with a real checksum, so this cannot be
    /// satisfied by refusing IPv6 UDP altogether.
    fn test_udp_zero_checksum_refused_over_ipv6() -> TestResult {
        const PORT: u16 = 9111;
        let src = target(0xE2);
        let dst = crate::net::local_ip_v6();
        let idx = crate::net::udp::socket();
        crate::net::udp::bind(idx, PORT);

        let mut seg = udp_segment_v6(src, dst, 40_011, PORT, b"unchecked");
        seg[6] = 0;
        seg[7] = 0;
        handle_ipv6_packet(&ipv6_frame(src, dst, 17, &seg));
        let mut buf = [0u8; 32];
        test_assert!(
            crate::net::udp::recvfrom(idx, &mut buf).is_none(),
            "a UDP datagram over IPv6 with a zero checksum was accepted -- RFC 8200 8.1 forbids the encoding"
        );

        // The case the arithmetic alone lets through: a zero field contributes
        // nothing to the sum, so a datagram whose remaining words add to all
        // ones verifies with the field left at zero. One datagram in 65536 has
        // that shape and an attacker picks which one, so the refusal has to be
        // a rule and not a by-product of the sum.
        let mut coincidental = None;
        for candidate in 0..=u16::MAX {
            let payload = candidate.to_be_bytes();
            let mut seg = Vec::with_capacity(10);
            seg.extend_from_slice(&40_011u16.to_be_bytes());
            seg.extend_from_slice(&PORT.to_be_bytes());
            seg.extend_from_slice(&10u16.to_be_bytes());
            seg.extend_from_slice(&[0, 0]);
            seg.extend_from_slice(&payload);
            if crate::net::ipv4::transport_checksum(
                crate::net::IpAddr::V6(src),
                crate::net::IpAddr::V6(dst),
                17,
                &seg,
            ) == Some(0)
            {
                coincidental = Some(seg);
                break;
            }
        }
        let Some(coincidental) = coincidental else {
            return TestResult::Fail(
                "no datagram with a zero field sums to all ones, so the case this asserts on was never built",
            );
        };
        handle_ipv6_packet(&ipv6_frame(src, dst, 17, &coincidental));
        test_assert!(
            crate::net::udp::recvfrom(idx, &mut buf).is_none(),
            "a zero-checksum datagram whose words happen to sum to all ones was accepted -- the IPv6 refusal is falling out of the arithmetic instead of being a rule"
        );

        let good = stamped_v6(src, dst, 17, 6, &seg);
        handle_ipv6_packet(&ipv6_frame(src, dst, 17, &good));
        test_assert!(
            crate::net::udp::recvfrom(idx, &mut buf).is_some(),
            "the same datagram carrying a real checksum was refused"
        );
        TestResult::Pass
    }

    /// The TCP checksum is mandatory in both families, and the IPv6
    /// pseudo-header differs from the IPv4 one in every field, so a verifier
    /// that reached for the wrong shape here would refuse all IPv6 TCP. A SYN to
    /// a listener is the observable: it is answered by creating a socket the
    /// listener hands out.
    fn test_tcp_checksum_verified_on_receive_v6() -> TestResult {
        const PORT: u16 = 49_620;
        let src = target(0xE3);
        let dst = crate::net::local_ip_v6();
        let listener = crate::net::tcp::socket();
        crate::net::tcp::bind(listener, PORT);
        crate::net::tcp::listen(listener);

        let mut seg = Vec::with_capacity(20);
        seg.extend_from_slice(&40_012u16.to_be_bytes());
        seg.extend_from_slice(&PORT.to_be_bytes());
        seg.extend_from_slice(&910u32.to_be_bytes()); // seq
        seg.extend_from_slice(&0u32.to_be_bytes()); // ack
        seg.extend_from_slice(&((5u16 << 12) | 0x02).to_be_bytes()); // SYN
        seg.extend_from_slice(&65_535u16.to_be_bytes());
        seg.extend_from_slice(&[0, 0]); // checksum
        seg.extend_from_slice(&[0, 0]); // urgent
        let mut corrupt = stamped_v6(src, dst, 6, 16, &seg);
        corrupt[7] ^= 0x01; // one bit of the sequence, checksum left stale
        handle_ipv6_packet(&ipv6_frame(src, dst, 6, &corrupt));
        crate::net::tcp::poll();
        test_assert!(
            crate::net::tcp::accept(listener).is_none(),
            "a TCP SYN over IPv6 whose checksum does not verify opened a connection"
        );

        let repaired = stamped_v6(src, dst, 6, 16, &corrupt);
        handle_ipv6_packet(&ipv6_frame(src, dst, 6, &repaired));
        crate::net::tcp::poll();
        test_assert!(
            crate::net::tcp::accept(listener).is_some(),
            "the same SYN with a checksum matching its contents did not open a connection"
        );
        TestResult::Pass
    }

    /// The destination summed into the pseudo-header is the one the datagram
    /// carried. A multicast destination is the case that separates it from
    /// `local_ip_v6()`, and it is not hypothetical: it is where every Neighbor
    /// Solicitation and every DHCPv6 message arrives.
    fn test_transport_checksum_uses_the_delivered_destination_v6() -> TestResult {
        const PORT: u16 = 9112;
        let src = target(0xE4);
        let dst = [
            0xFF, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x00, 0x02,
        ];
        test_assert!(
            dst != crate::net::local_ip_v6(),
            "the fixture destination equals the local address, so this proves nothing"
        );
        let idx = crate::net::udp::socket();
        crate::net::udp::bind(idx, PORT);

        let seg = udp_segment_v6(src, dst, 40_013, PORT, b"multicast");
        handle_ipv6_packet(&ipv6_frame(src, dst, 17, &seg));
        let mut buf = [0u8; 32];
        test_assert!(
            crate::net::udp::recvfrom(idx, &mut buf).is_some(),
            "a datagram checksummed against the address it was actually sent to was refused"
        );

        let wrong = udp_segment_v6(src, crate::net::local_ip_v6(), 40_013, PORT, b"multicast");
        handle_ipv6_packet(&ipv6_frame(src, dst, 17, &wrong));
        test_assert!(
            crate::net::udp::recvfrom(idx, &mut buf).is_none(),
            "a datagram summed against an address other than the one it was sent to was accepted"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        // ICMPv4's own group is chained here rather than registered from
        // `testing::runner`, next to the ICMPv6 verification it mirrors.
        crate::net::icmp::tests::register_all();
        crate::testing::register_test(
            "ipv6::udp_checksum_verified_on_receive",
            test_udp_checksum_verified_on_receive_v6,
        );
        crate::testing::register_test(
            "ipv6::udp_zero_checksum_refused",
            test_udp_zero_checksum_refused_over_ipv6,
        );
        crate::testing::register_test(
            "ipv6::tcp_checksum_verified_on_receive",
            test_tcp_checksum_verified_on_receive_v6,
        );
        crate::testing::register_test(
            "ipv6::transport_checksum_uses_the_delivered_destination",
            test_transport_checksum_uses_the_delivered_destination_v6,
        );
        crate::testing::register_test(
            "ipv6::advert_verified_against_the_delivered_destination",
            test_advert_verified_against_the_delivered_destination,
        );
        crate::testing::register_test(
            "ipv6::advert_with_bad_checksum_not_cached",
            test_advert_with_bad_checksum_not_cached,
        );
        crate::testing::register_test(
            "ipv6::advert_without_option_not_cached",
            test_advert_without_option_not_cached,
        );
        crate::testing::register_test(
            "ipv6::advert_with_malformed_option_not_cached",
            test_advert_with_malformed_option_not_cached,
        );
        crate::testing::register_test(
            "ipv6::advert_with_option_cached",
            test_advert_with_option_cached,
        );
        crate::testing::register_test(
            "ipv6::unsolicited_advert_does_not_overwrite",
            test_unsolicited_advert_does_not_overwrite,
        );
        crate::testing::register_test(
            "ipv6::unusable_addresses_not_cached",
            test_unusable_addresses_not_cached,
        );
    }
}
