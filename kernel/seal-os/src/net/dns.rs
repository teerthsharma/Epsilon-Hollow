// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Simple DNS resolver over UDP with BTreeMap TTL cache.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

struct DnsCacheEntry {
    ip: [u8; 4],
    expires_at: u64,
}

static DNS_SERVER: Mutex<[u8; 4]> = Mutex::new([10, 0, 2, 3]);
static CACHE: Mutex<BTreeMap<String, DnsCacheEntry>> = Mutex::new(BTreeMap::new());
// ponytail: each query allocates a fresh `udp::socket()` rather than reusing
// one shared socket. A shared socket was tried and reverted: `rebind` and
// `sendto` are separate UDP_SOCKETS lock acquisitions, and a second task's
// concurrent query could rebind the shared socket in the gap between them,
// changing the on-wire source port of a query that had not sent yet and
// failing the dst_port check on its legitimate response. Per-query sockets
// mean UDP_SOCKETS grows without bound from DNS traffic again; that is
// accepted because `udp::allocate_ephemeral_port`'s terminal case now
// returns a randomized draw, not a predictable counter (see net/udp.rs) --
// upgrade to a per-query-checked-out-and-returned socket (or a real
// close/free-list on UDP_SOCKETS) if the growth itself becomes the problem.

pub fn init() {}

pub fn set_server(ip: [u8; 4]) {
    *DNS_SERVER.lock() = ip;
}

pub fn server() -> [u8; 4] {
    *DNS_SERVER.lock()
}

fn ticks() -> u64 {
    crate::drivers::interrupts::ticks()
}

static PENDING_QUERY: Mutex<Option<String>> = Mutex::new(None);
// Transaction ID of the single outstanding query, matched against the
// response's ID in `handle_dns_response` to reject off-path spoofed answers.
static PENDING_QUERY_ID: Mutex<Option<u16>> = Mutex::new(None);
// (address, port) the outstanding query was actually sent to -- recorded from
// the same local values used for the `sendto` call, not re-read from
// DNS_SERVER at receive time, so a `set_server` call racing with an in-flight
// query can't shift what the response is checked against.
static PENDING_QUERY_SERVER: Mutex<Option<(crate::net::IpAddr, u16)>> = Mutex::new(None);
// Local ephemeral UDP port the outstanding query went out from (from the
// same socket handle used for `sendto`). A response must be addressed back
// to this exact port -- see DST-PORT check in `handle_dns_response`.
static PENDING_QUERY_LOCAL_PORT: Mutex<Option<u16>> = Mutex::new(None);

// ---------------------------------------------------------------------------
// Transaction-ID entropy.
//
// IDs used to come from a plain incrementing counter, making the "next" ID
// fully predictable to anyone who observed -- or merely induced, by making
// the kernel resolve any unrelated hostname -- one prior lookup. Draw from
// hardware entropy instead (`drivers::entropy::getrandom`), with the shared
// boot-seeded PRNG fallback (`drivers::entropy::fallback_random_u64`) for
// the rare CPU with neither RDSEED nor RDRAND. That fallback used to be a
// private copy here; it is now the one shared copy also used by
// `net::udp`'s ephemeral-port allocator, so the two can't drift apart.
// ---------------------------------------------------------------------------

/// Draw a random 16-bit DNS transaction ID. Prefers hardware entropy
/// (`drivers::entropy::getrandom`, which itself prefers RDSEED and falls
/// back to RDRAND, and checks CPUID before using either); falls back to the
/// shared boot-seeded PRNG only if the CPU offers neither.
fn random_query_id() -> u16 {
    let mut buf = [0u8; 2];
    if crate::drivers::entropy::getrandom(&mut buf) {
        u16::from_ne_bytes(buf)
    } else {
        crate::drivers::entropy::fallback_random_u64() as u16
    }
}

// ---------------------------------------------------------------------------
// NEW: Real DNS header and query structs with honest error handling.
// ---------------------------------------------------------------------------

/// DNS packet header (12 bytes, RFC 1035).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DnsHeader {
    pub id: u16,
    pub flags: u16,
    pub qdcount: u16,
    pub ancount: u16,
    pub nscount: u16,
    pub arcount: u16,
}

impl DnsHeader {
    /// Parse a DNS header from exactly 12 bytes.
    /// Returns `None` if the slice is too short.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 {
            return None;
        }
        Some(Self {
            id: u16::from_be_bytes([bytes[0], bytes[1]]),
            flags: u16::from_be_bytes([bytes[2], bytes[3]]),
            qdcount: u16::from_be_bytes([bytes[4], bytes[5]]),
            ancount: u16::from_be_bytes([bytes[6], bytes[7]]),
            nscount: u16::from_be_bytes([bytes[8], bytes[9]]),
            arcount: u16::from_be_bytes([bytes[10], bytes[11]]),
        })
    }

    /// Serialize the header to a fixed 12-byte array.
    pub fn to_bytes(&self) -> [u8; 12] {
        let mut b = [0u8; 12];
        b[0..2].copy_from_slice(&self.id.to_be_bytes());
        b[2..4].copy_from_slice(&self.flags.to_be_bytes());
        b[4..6].copy_from_slice(&self.qdcount.to_be_bytes());
        b[6..8].copy_from_slice(&self.ancount.to_be_bytes());
        b[8..10].copy_from_slice(&self.nscount.to_be_bytes());
        b[10..12].copy_from_slice(&self.arcount.to_be_bytes());
        b
    }

    /// True if the QR bit indicates a response.
    pub fn is_response(&self) -> bool {
        self.flags & 0x8000 != 0
    }
}

/// Represents a single DNS question.
#[derive(Debug, Clone, PartialEq)]
pub struct DnsQuery {
    pub name: String,
    pub qtype: u16,
    pub qclass: u16,
}

/// Honest DNS resolver.
/// Returns `Err("no network")` if no NIC is present.
/// Otherwise sends a query and returns `None` until a response arrives.
pub fn resolve(name: &str) -> Result<Option<[u8; 4]>, &'static str> {
    if !crate::net::has_nic() {
        return Err("no network");
    }

    let cache = CACHE.lock();
    if let Some(entry) = cache.get(name) {
        if ticks() < entry.expires_at {
            return Ok(Some(entry.ip));
        }
    }
    drop(cache);

    *PENDING_QUERY.lock() = Some(String::from(name));

    let id = random_query_id();
    *PENDING_QUERY_ID.lock() = Some(id);

    let hdr = DnsHeader {
        id,
        flags: 0x0100, // standard query
        qdcount: 1,
        ancount: 0,
        nscount: 0,
        arcount: 0,
    };

    let mut pkt = Vec::new();
    pkt.extend_from_slice(&hdr.to_bytes());

    for label in name.split('.') {
        pkt.push(label.len() as u8);
        pkt.extend_from_slice(label.as_bytes());
    }
    pkt.push(0); // end of name
    pkt.extend_from_slice(&[0x00, 0x01]); // type A
    pkt.extend_from_slice(&[0x00, 0x01]); // class IN

    let server = *DNS_SERVER.lock();
    let dst = crate::net::IpAddr::V4(server);
    let dst_port = 53;
    *PENDING_QUERY_SERVER.lock() = Some((dst, dst_port));
    let sock = crate::net::udp::socket();
    *PENDING_QUERY_LOCAL_PORT.lock() = Some(crate::net::udp::port(sock));
    crate::net::udp::sendto(sock, &pkt, dst, dst_port);
    Ok(None)
}

// ---------------------------------------------------------------------------
// Legacy query function (preserved for compatibility).
// ---------------------------------------------------------------------------

pub fn query(name: &str) -> Option<[u8; 4]> {
    let cache = CACHE.lock();
    if let Some(entry) = cache.get(name) {
        if ticks() < entry.expires_at {
            return Some(entry.ip);
        }
    }
    drop(cache);

    *PENDING_QUERY.lock() = Some(String::from(name));

    let id = random_query_id();
    *PENDING_QUERY_ID.lock() = Some(id);

    let mut pkt = Vec::new();
    pkt.extend_from_slice(&id.to_be_bytes());
    pkt.extend_from_slice(&[0x01, 0x00]); // flags: standard query
    pkt.extend_from_slice(&[0x00, 0x01]); // questions: 1
    pkt.extend_from_slice(&[0x00, 0x00]); // answer RRs
    pkt.extend_from_slice(&[0x00, 0x00]); // authority RRs
    pkt.extend_from_slice(&[0x00, 0x00]); // additional RRs

    for label in name.split('.') {
        pkt.push(label.len() as u8);
        pkt.extend_from_slice(label.as_bytes());
    }
    pkt.push(0); // end of name
    pkt.extend_from_slice(&[0x00, 0x01]); // type A
    pkt.extend_from_slice(&[0x00, 0x01]); // class IN

    let server = *DNS_SERVER.lock();
    let dst = crate::net::IpAddr::V4(server);
    let dst_port = 53;
    *PENDING_QUERY_SERVER.lock() = Some((dst, dst_port));
    let sock = crate::net::udp::socket();
    *PENDING_QUERY_LOCAL_PORT.lock() = Some(crate::net::udp::port(sock));
    crate::net::udp::sendto(sock, &pkt, dst, dst_port);
    None
}

/// Decode a (possibly compressed) DNS name starting at `offset`.
///
/// Returns `(name, offset_immediately_after_the_name_in_this_message)`, or
/// `None` on malformed input. Compression pointers (RFC 1035 SS4.1.4) are
/// followed, but every pointer target must land strictly before the pointer
/// itself; combined with a hard cap on the number of jumps, a pointer that
/// targets itself or forms a cycle is rejected outright rather than merely
/// bounded, so no packet can drive this into a long or infinite loop. Also
/// enforces RFC 1035 SS3.1's 255-octet cap on the encoded name length --
/// belt-and-suspenders alongside the jump cap and `pkt.len()` bound, not
/// load-bearing for termination by itself, but a malformed/oversized name is
/// still a malformed name and should be rejected as one.
fn read_name(pkt: &[u8], offset: usize) -> Option<(String, usize)> {
    let mut name = String::new();
    let mut pos = offset;
    let mut end_pos: Option<usize> = None;
    let mut jumps = 0u32;
    let mut wire_len = 0usize;
    loop {
        if pos >= pkt.len() {
            return None;
        }
        let len = pkt[pos];
        if len == 0 {
            wire_len += 1; // terminating zero octet
            if wire_len > 255 {
                return None;
            }
            if end_pos.is_none() {
                end_pos = Some(pos + 1);
            }
            break;
        }
        if len & 0xC0 == 0xC0 {
            if pos + 1 >= pkt.len() {
                return None;
            }
            let target = (((len & 0x3F) as usize) << 8) | pkt[pos + 1] as usize;
            if end_pos.is_none() {
                end_pos = Some(pos + 2);
            }
            jumps += 1;
            if jumps > 128 || target >= pos {
                return None; // must strictly decrease -- rules out any loop
            }
            pos = target;
            continue;
        }
        let label_len = len as usize;
        let start = pos + 1;
        let stop = start + label_len;
        if stop > pkt.len() {
            return None;
        }
        wire_len += 1 + label_len; // length octet + label bytes
        if wire_len > 255 {
            return None;
        }
        if !name.is_empty() {
            name.push('.');
        }
        name.push_str(core::str::from_utf8(&pkt[start..stop]).ok()?);
        pos = stop;
    }
    Some((name, end_pos.unwrap_or(pos)))
}

/// Handle an inbound UDP datagram from `src:src_port`, addressed to our
/// local `dst_port`, that claims to be a DNS response. All three come
/// straight from the IP/UDP headers `udp.rs` already parsed for this packet
/// -- see `handle_udp_packet` (udp.rs:207).
pub fn handle_dns_response(src: crate::net::IpAddr, src_port: u16, dst_port: u16, pkt: &[u8]) {
    if pkt.len() < 12 {
        return;
    }
    let flags = u16::from_be_bytes([pkt[2], pkt[3]]);
    if flags & 0x8000 == 0 {
        return; // not a response
    }

    // Reject any response whose transaction ID doesn't match the single
    // outstanding query. A mismatch must not touch any PENDING_QUERY_* state
    // -- dropping silently is the fail-closed behavior; clearing state on an
    // unverified response would hand an attacker a denial-of-service
    // primitive instead of a poisoning one.
    let id = u16::from_be_bytes([pkt[0], pkt[1]]);
    if *PENDING_QUERY_ID.lock() != Some(id) {
        return;
    }

    // Reject any response not addressed to the local port the query went
    // out from (the classic second 16 bits of entropy post-Kaminsky).
    if *PENDING_QUERY_LOCAL_PORT.lock() != Some(dst_port) {
        return;
    }

    // Reject any response that didn't come from the server the query was
    // actually sent to, on the port it was sent to. Compared against the
    // value `resolve`/`query` recorded at send time (PENDING_QUERY_SERVER),
    // never re-derived from the current DNS_SERVER here.
    if *PENDING_QUERY_SERVER.lock() != Some((src, src_port)) {
        return;
    }

    // Reject any response whose echoed question doesn't name the host this
    // ID's query actually asked about (cross-association guard for the
    // single-slot PENDING_QUERY, and a fourth independent check beyond ID,
    // destination port, and source).
    let matches_pending = match read_name(pkt, 12) {
        Some((qname, _)) => PENDING_QUERY
            .lock()
            .as_deref()
            .is_some_and(|expected| qname.eq_ignore_ascii_case(expected)),
        None => false,
    };
    if !matches_pending {
        return;
    }

    let qdcount = u16::from_be_bytes([pkt[4], pkt[5]]);
    let ancount = u16::from_be_bytes([pkt[6], pkt[7]]);

    // Walk the record sections with `read_name`, which reports the offset
    // just past a name: past the two pointer octets when the name ends in a
    // compression pointer (RFC 1035 SS4.1.4), past the single zero octet when
    // it ends in a root label. The two cases must not be conflated. The
    // hand-rolled skip loops that used to live here consumed the pointer's
    // two octets and *then* also tested for a trailing zero, so an answer
    // whose owner name is the usual back-pointer to the question -- the form
    // every real resolver emits -- had its TYPE field's high byte (0x00 for
    // type A) swallowed as if it were a name terminator. The one-byte slip
    // made TYPE read as 0x0100, so the A record was skipped and no legitimate
    // answer was ever cached. A name `read_name` rejects fails closed here:
    // the whole response is dropped rather than parsed from a guessed offset.
    let mut offset = 12;
    // Skip questions
    for _ in 0..qdcount {
        let Some((_, after_name)) = read_name(pkt, offset) else {
            return;
        };
        offset = after_name + 4; // type + class
    }

    // Parse answers
    for _ in 0..ancount {
        // Skip name
        let Some((_, after_name)) = read_name(pkt, offset) else {
            return;
        };
        offset = after_name;
        if offset + 10 > pkt.len() {
            break;
        }
        let rtype = u16::from_be_bytes([pkt[offset], pkt[offset + 1]]);
        offset += 4; // type + class
        let ttl = u32::from_be_bytes([
            pkt[offset],
            pkt[offset + 1],
            pkt[offset + 2],
            pkt[offset + 3],
        ]);
        offset += 4;
        let rdlen = u16::from_be_bytes([pkt[offset], pkt[offset + 1]]) as usize;
        offset += 2;
        if rtype == 1 && rdlen == 4 && offset + 4 <= pkt.len() {
            let ip = [
                pkt[offset],
                pkt[offset + 1],
                pkt[offset + 2],
                pkt[offset + 3],
            ];
            let hostname = PENDING_QUERY
                .lock()
                .take()
                .unwrap_or_else(|| String::from("unknown"));
            *PENDING_QUERY_ID.lock() = None;
            *PENDING_QUERY_SERVER.lock() = None;
            *PENDING_QUERY_LOCAL_PORT.lock() = None;
            let mut cache = CACHE.lock();
            if cache.len() >= 16 {
                let first = cache.keys().next().cloned();
                if let Some(k) = first {
                    cache.remove(&k);
                }
            }
            cache.insert(
                hostname,
                DnsCacheEntry {
                    ip,
                    expires_at: ticks().wrapping_add(ttl as u64 * 100),
                },
            );
            return;
        }
        offset += rdlen;
    }
}

// ---------------------------------------------------------------------------
// Tests -- run by the in-kernel harness (crate::testing), not `cargo test`.
// `kernel/seal-os` is excluded from the workspace, so `cargo test --workspace`
// never builds this crate; these register into crate::testing::TEST_REGISTRY
// and execute under QEMU via testing::runner::test_main(). See WIRING note on
// `register_all` below -- runner.rs does not call it yet.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// Build a well-formed DNS response: given `id`, answering `hostname`
    /// with `ip`. Used both for legitimate-response acceptance and, with a
    /// mismatched `id` or `hostname`, as the spoofed/off-path packet.
    fn build_dns_response(id: u16, hostname: &str, ip: [u8; 4], ttl: u32) -> Vec<u8> {
        let mut pkt = Vec::new();
        pkt.extend_from_slice(&id.to_be_bytes());
        pkt.extend_from_slice(&0x8180u16.to_be_bytes()); // flags: response
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // qdcount
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // ancount
        pkt.extend_from_slice(&0x0000u16.to_be_bytes()); // nscount
        pkt.extend_from_slice(&0x0000u16.to_be_bytes()); // arcount
                                                          // question
        for label in hostname.split('.') {
            pkt.push(label.len() as u8);
            pkt.extend_from_slice(label.as_bytes());
        }
        pkt.push(0);
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // type A
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // class IN
                                                          // answer
        pkt.push(0xC0);
        pkt.push(0x0C); // pointer to question name
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // type A
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // class IN
        pkt.extend_from_slice(&ttl.to_be_bytes());
        pkt.extend_from_slice(&0x0004u16.to_be_bytes()); // rdlen
        pkt.extend_from_slice(&ip);
        pkt
    }

    /// The address/port `arm_pending` records as "where the query went",
    /// matching `DNS_SERVER`'s default so tests don't depend on `set_server`.
    const TEST_SERVER: crate::net::IpAddr = crate::net::IpAddr::V4([10, 0, 2, 3]);
    const TEST_PORT: u16 = 53;
    /// The local ephemeral port `arm_pending` records as "where the query
    /// went out from" -- a response must be addressed back to this port.
    const TEST_LOCAL_PORT: u16 = 51234;

    /// Arm a pending query the way `resolve`/`query` would on the send path,
    /// without touching the network.
    fn arm_pending(hostname: &str, id: u16) {
        *PENDING_QUERY.lock() = Some(String::from(hostname));
        *PENDING_QUERY_ID.lock() = Some(id);
        *PENDING_QUERY_SERVER.lock() = Some((TEST_SERVER, TEST_PORT));
        *PENDING_QUERY_LOCAL_PORT.lock() = Some(TEST_LOCAL_PORT);
    }

    fn reset_state() {
        *PENDING_QUERY.lock() = None;
        *PENDING_QUERY_ID.lock() = None;
        *PENDING_QUERY_SERVER.lock() = None;
        *PENDING_QUERY_LOCAL_PORT.lock() = None;
        CACHE.lock().clear();
    }

    /// RED: a response with the right hostname but the wrong transaction ID
    /// (the off-path spoof: an attacker who cannot see the query ID) must be
    /// dropped, and must not disturb the still-pending query.
    fn test_wrong_id_rejected() -> TestResult {
        reset_state();
        arm_pending("example.com", 0x1234);
        let spoofed = build_dns_response(0xBEEF, "example.com", [6, 6, 6, 6], 300);
        handle_dns_response(TEST_SERVER, TEST_PORT, TEST_LOCAL_PORT, &spoofed);
        test_assert!(
            CACHE.lock().get("example.com").is_none(),
            "response with mismatched transaction ID was accepted into the cache"
        );
        test_assert_eq!(*PENDING_QUERY.lock(), Some(String::from("example.com")));
        test_assert_eq!(*PENDING_QUERY_ID.lock(), Some(0x1234));
        test_assert_eq!(*PENDING_QUERY_SERVER.lock(), Some((TEST_SERVER, TEST_PORT)));
        TestResult::Pass
    }

    /// RED: an attacker who induced a prior lookup (or simply guessed the
    /// old counter's successor) tries the "next" ID under the pre-fix
    /// sequential scheme. With random IDs that guess is just another wrong
    /// ID and must be rejected the same way -- this documents that the
    /// specific sequential-guess strategy the old counter enabled no longer
    /// works, not just that arbitrary wrong IDs are rejected.
    fn test_predicted_id_rejected() -> TestResult {
        reset_state();
        let observed_prior_id: u16 = 0x1234; // an ID seen/induced on an earlier lookup
        arm_pending("example.com", 0x9ABC); // the real (random) ID for this query
        let predicted = observed_prior_id.wrapping_add(1); // old counter's guess
        test_assert!(
            predicted != 0x9ABC,
            "test fixture invalid: predicted guess accidentally equals the armed ID"
        );
        let spoofed = build_dns_response(predicted, "example.com", [6, 6, 6, 6], 300);
        handle_dns_response(TEST_SERVER, TEST_PORT, TEST_LOCAL_PORT, &spoofed);
        test_assert!(
            CACHE.lock().get("example.com").is_none(),
            "response using a sequentially-predicted ID was accepted"
        );
        test_assert_eq!(*PENDING_QUERY_ID.lock(), Some(0x9ABC));
        TestResult::Pass
    }

    /// Sanity check that `random_query_id` is not the old monotonic counter
    /// -- the smallest thing that fails if the entropy plumbing regresses to
    /// `n, n+1, n+2, ...`. Not a statistical randomness test, just a tripwire.
    fn test_query_id_not_sequential() -> TestResult {
        let mut ids: Vec<u16> = Vec::new();
        for _ in 0..8 {
            ids.push(random_query_id());
        }
        let looks_sequential = ids.windows(2).all(|w| w[1] == w[0].wrapping_add(1));
        test_assert!(
            !looks_sequential,
            "random_query_id looks like a monotonic counter, not randomized"
        );
        TestResult::Pass
    }

    /// RED: a response with the right ID but a question section naming a
    /// different host (the cross-association case: a second, unrelated
    /// resolution's answer, or an off-path guess of the ID) must be dropped
    /// without disturbing the pending query.
    fn test_wrong_question_rejected() -> TestResult {
        reset_state();
        arm_pending("bank.example", 0x1234);
        let spoofed = build_dns_response(0x1234, "evil.attacker.example", [6, 6, 6, 6], 300);
        handle_dns_response(TEST_SERVER, TEST_PORT, TEST_LOCAL_PORT, &spoofed);
        test_assert!(
            CACHE.lock().get("bank.example").is_none(),
            "response naming a different host than the pending query was accepted"
        );
        test_assert!(
            CACHE.lock().get("evil.attacker.example").is_none(),
            "response was cached under the attacker-supplied name"
        );
        test_assert_eq!(*PENDING_QUERY.lock(), Some(String::from("bank.example")));
        TestResult::Pass
    }

    /// RED: a response with the right ID and the right question, but from a
    /// source address that isn't the configured DNS server, must be dropped
    /// without disturbing the pending query -- the off-path spoof scenario
    /// where the attacker also happens to guess the ID.
    fn test_wrong_source_address_rejected() -> TestResult {
        reset_state();
        arm_pending("example.com", 0x1234);
        let spoofed = build_dns_response(0x1234, "example.com", [6, 6, 6, 6], 300);
        let attacker = crate::net::IpAddr::V4([203, 0, 113, 66]); // not TEST_SERVER
        handle_dns_response(attacker, TEST_PORT, TEST_LOCAL_PORT, &spoofed);
        test_assert!(
            CACHE.lock().get("example.com").is_none(),
            "response from an unexpected source address was accepted into the cache"
        );
        test_assert_eq!(*PENDING_QUERY.lock(), Some(String::from("example.com")));
        test_assert_eq!(*PENDING_QUERY_ID.lock(), Some(0x1234));
        test_assert_eq!(*PENDING_QUERY_SERVER.lock(), Some((TEST_SERVER, TEST_PORT)));
        TestResult::Pass
    }

    /// RED: same as above but for the source port -- the right server IP
    /// answering from an unexpected port must also be dropped.
    fn test_wrong_source_port_rejected() -> TestResult {
        reset_state();
        arm_pending("example.com", 0x1234);
        let spoofed = build_dns_response(0x1234, "example.com", [6, 6, 6, 6], 300);
        handle_dns_response(TEST_SERVER, 9999, TEST_LOCAL_PORT, &spoofed); // right IP, wrong port
        test_assert!(
            CACHE.lock().get("example.com").is_none(),
            "response from an unexpected source port was accepted into the cache"
        );
        test_assert_eq!(*PENDING_QUERY.lock(), Some(String::from("example.com")));
        test_assert_eq!(*PENDING_QUERY_ID.lock(), Some(0x1234));
        test_assert_eq!(*PENDING_QUERY_SERVER.lock(), Some((TEST_SERVER, TEST_PORT)));
        TestResult::Pass
    }

    /// RED: a response from the right server on the right server-side port,
    /// but addressed to a local port other than the one the query actually
    /// went out from, must be dropped without disturbing the pending query.
    fn test_wrong_dst_port_rejected() -> TestResult {
        reset_state();
        arm_pending("example.com", 0x1234);
        let spoofed = build_dns_response(0x1234, "example.com", [6, 6, 6, 6], 300);
        handle_dns_response(TEST_SERVER, TEST_PORT, 9999, &spoofed); // wrong local port
        test_assert!(
            CACHE.lock().get("example.com").is_none(),
            "response addressed to an unexpected local port was accepted into the cache"
        );
        test_assert_eq!(*PENDING_QUERY.lock(), Some(String::from("example.com")));
        test_assert_eq!(*PENDING_QUERY_ID.lock(), Some(0x1234));
        test_assert_eq!(*PENDING_QUERY_LOCAL_PORT.lock(), Some(TEST_LOCAL_PORT));
        TestResult::Pass
    }

    /// GREEN: a response matching the outstanding ID, the destination port,
    /// the source the query was sent to, and the pending hostname is still
    /// accepted and cached, and the pending state is cleared on acceptance.
    fn test_legit_response_accepted() -> TestResult {
        reset_state();
        arm_pending("example.com", 0x1234);
        let good = build_dns_response(0x1234, "example.com", [93, 184, 216, 34], 300);
        handle_dns_response(TEST_SERVER, TEST_PORT, TEST_LOCAL_PORT, &good);
        test_assert_eq!(
            CACHE.lock().get("example.com").map(|e| e.ip),
            Some([93, 184, 216, 34])
        );
        test_assert!(
            PENDING_QUERY.lock().is_none(),
            "pending query was not cleared on a validated accept"
        );
        test_assert!(
            PENDING_QUERY_ID.lock().is_none(),
            "pending query ID was not cleared on a validated accept"
        );
        test_assert!(
            PENDING_QUERY_SERVER.lock().is_none(),
            "pending query server was not cleared on a validated accept"
        );
        test_assert!(
            PENDING_QUERY_LOCAL_PORT.lock().is_none(),
            "pending query local port was not cleared on a validated accept"
        );
        TestResult::Pass
    }

    /// Append `name` to `pkt` as literal length-prefixed labels ending in the
    /// root label -- the uncompressed wire form, no compression pointer.
    fn push_uncompressed_name(pkt: &mut Vec<u8>, name: &str) {
        for label in name.split('.') {
            pkt.push(label.len() as u8);
            pkt.extend_from_slice(label.as_bytes());
        }
        pkt.push(0);
    }

    /// GREEN: an answer's owner name is not required to be a compression
    /// pointer -- a server may repeat the name literally. The two forms end a
    /// name with a different number of octets (two for a pointer, one for the
    /// root label), so `build_dns_response`'s pointer fixture alone does not
    /// cover this path. The A record must still be found at the right offset.
    fn test_uncompressed_answer_name_accepted() -> TestResult {
        reset_state();
        arm_pending("example.com", 0x1234);
        let mut pkt = Vec::new();
        pkt.extend_from_slice(&0x1234u16.to_be_bytes()); // id
        pkt.extend_from_slice(&0x8180u16.to_be_bytes()); // flags: response
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // qdcount
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // ancount
        pkt.extend_from_slice(&0x0000u16.to_be_bytes()); // nscount
        pkt.extend_from_slice(&0x0000u16.to_be_bytes()); // arcount
        push_uncompressed_name(&mut pkt, "example.com"); // question name
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // qtype A
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // qclass IN
        push_uncompressed_name(&mut pkt, "example.com"); // answer owner name
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // type A
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // class IN
        pkt.extend_from_slice(&300u32.to_be_bytes()); // ttl
        pkt.extend_from_slice(&0x0004u16.to_be_bytes()); // rdlen
        pkt.extend_from_slice(&[93, 184, 216, 34]);
        handle_dns_response(TEST_SERVER, TEST_PORT, TEST_LOCAL_PORT, &pkt);
        test_assert_eq!(
            CACHE.lock().get("example.com").map(|e| e.ip),
            Some([93, 184, 216, 34])
        );
        TestResult::Pass
    }

    /// DECOMPRESSION: a name whose only label is a compression pointer aimed
    /// at itself. Before the strict-decrease rule, a decoder that followed
    /// the pointer blindly would loop forever; `read_name` must reject it
    /// and `handle_dns_response` must return instead of hanging the harness.
    fn test_decompression_self_pointer_rejected() -> TestResult {
        let mut pkt = Vec::new();
        pkt.extend_from_slice(&0x1234u16.to_be_bytes()); // id
        pkt.extend_from_slice(&0x8180u16.to_be_bytes()); // flags: response
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // qdcount
        pkt.extend_from_slice(&0x0000u16.to_be_bytes()); // ancount
        pkt.extend_from_slice(&0x0000u16.to_be_bytes()); // nscount
        pkt.extend_from_slice(&0x0000u16.to_be_bytes()); // arcount
        pkt.push(0xC0);
        pkt.push(0x0C); // question name: pointer at offset 12, targeting offset 12 -- itself
        pkt.extend_from_slice(&0x0001u16.to_be_bytes());
        pkt.extend_from_slice(&0x0001u16.to_be_bytes());
        test_assert!(
            read_name(&pkt, 12).is_none(),
            "self-referential compression pointer was decoded instead of rejected"
        );

        reset_state();
        arm_pending("example.com", 0x1234);
        handle_dns_response(TEST_SERVER, TEST_PORT, TEST_LOCAL_PORT, &pkt); // must return promptly, not loop
        test_assert!(
            CACHE.lock().get("example.com").is_none(),
            "malformed name was accepted as a match"
        );
        TestResult::Pass
    }

    /// RFC 1035 SS3.1: a name whose wire-encoded length exceeds 255 octets
    /// must be rejected, not truncated or accepted.
    fn test_oversized_name_rejected() -> TestResult {
        // 4 labels of 63 bytes each (the max single-label length) plus
        // separators comfortably exceeds the 255-octet total cap.
        let mut pkt = Vec::new();
        pkt.extend_from_slice(&0x1234u16.to_be_bytes());
        pkt.extend_from_slice(&0x8180u16.to_be_bytes());
        pkt.extend_from_slice(&0x0001u16.to_be_bytes());
        pkt.extend_from_slice(&0x0000u16.to_be_bytes());
        pkt.extend_from_slice(&0x0000u16.to_be_bytes());
        pkt.extend_from_slice(&0x0000u16.to_be_bytes());
        for _ in 0..5 {
            pkt.push(63);
            pkt.extend_from_slice(&[b'a'; 63]);
        }
        pkt.push(0);
        pkt.extend_from_slice(&0x0001u16.to_be_bytes());
        pkt.extend_from_slice(&0x0001u16.to_be_bytes());
        test_assert!(
            read_name(&pkt, 12).is_none(),
            "name over 255 wire-encoded octets was accepted"
        );
        TestResult::Pass
    }

    fn test_dns_cache_expiration() -> TestResult {
        reset_state();
        CACHE.lock().insert(
            String::from("old.com"),
            DnsCacheEntry {
                ip: [1, 2, 3, 4],
                expires_at: 0, // expired
            },
        );
        test_assert!(query("old.com").is_none());
        TestResult::Pass
    }

    fn test_dns_cache_hit() -> TestResult {
        reset_state();
        CACHE.lock().insert(
            String::from("fresh.com"),
            DnsCacheEntry {
                ip: [5, 6, 7, 8],
                expires_at: u64::MAX,
            },
        );
        test_assert_eq!(query("fresh.com"), Some([5, 6, 7, 8]));
        TestResult::Pass
    }

    fn test_dns_header_roundtrip() -> TestResult {
        let hdr = DnsHeader {
            id: 0x1234,
            flags: 0x8180,
            qdcount: 1,
            ancount: 1,
            nscount: 0,
            arcount: 0,
        };
        let bytes = hdr.to_bytes();
        let Some(parsed) = DnsHeader::from_bytes(&bytes) else {
            return TestResult::Fail("header failed to round-trip through from_bytes");
        };
        test_assert_eq!(parsed, hdr);
        test_assert!(parsed.is_response(), "QR bit lost in round-trip");
        TestResult::Pass
    }

    fn test_dns_resolve_no_network() -> TestResult {
        // The QEMU harness image boots with no NIC device attached, so
        // `resolve` must fail honestly rather than silently hang.
        test_assert_eq!(resolve("example.com"), Err("no network"));
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("dns::wrong_id_rejected", test_wrong_id_rejected);
        crate::testing::register_test("dns::predicted_id_rejected", test_predicted_id_rejected);
        crate::testing::register_test(
            "dns::query_id_not_sequential",
            test_query_id_not_sequential,
        );
        crate::testing::register_test(
            "dns::wrong_question_rejected",
            test_wrong_question_rejected,
        );
        crate::testing::register_test(
            "dns::wrong_source_address_rejected",
            test_wrong_source_address_rejected,
        );
        crate::testing::register_test(
            "dns::wrong_source_port_rejected",
            test_wrong_source_port_rejected,
        );
        crate::testing::register_test(
            "dns::wrong_dst_port_rejected",
            test_wrong_dst_port_rejected,
        );
        crate::testing::register_test(
            "dns::legit_response_accepted",
            test_legit_response_accepted,
        );
        crate::testing::register_test(
            "dns::uncompressed_answer_name_accepted",
            test_uncompressed_answer_name_accepted,
        );
        crate::testing::register_test(
            "dns::decompression_self_pointer_rejected",
            test_decompression_self_pointer_rejected,
        );
        crate::testing::register_test(
            "dns::oversized_name_rejected",
            test_oversized_name_rejected,
        );
        crate::testing::register_test("dns::cache_expiration", test_dns_cache_expiration);
        crate::testing::register_test("dns::cache_hit", test_dns_cache_hit);
        crate::testing::register_test("dns::header_roundtrip", test_dns_header_roundtrip);
        crate::testing::register_test("dns::resolve_no_network", test_dns_resolve_no_network);
    }
}
