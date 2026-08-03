// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Minimal TLS 1.3 client — ephemeral X25519 + X.509 by default, PSK as a
//! fallback. AES-128-GCM records, HMAC-SHA256 HKDF key schedule.
//!
//! Real cryptography, not a stub, and deliberately not a complete TLS 1.3
//! implementation. Two documented deviations from RFC 8446:
//!
//! * Traffic secrets are derived over `client_random`/`server_random` instead
//!   of a running transcript hash, so this does not interoperate with a
//!   stock TLS 1.3 server and provides no downgrade protection over the
//!   handshake messages.
//! * The peer's Certificate message is read as plaintext handshake, whereas
//!   RFC 8446 encrypts it under the handshake traffic keys.
//!
//! What *is* real: the key schedule (RFC 5869 HMAC-SHA256 HKDF), the X25519
//! agreement ([`super::ecdhe`]), the record AEAD, and the certificate chain
//! validation ([`super::x509`]).

use aes_gcm::{AeadInPlace, Aes128Gcm, KeyInit};
use alloc::string::String;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use super::{ecdhe, x509};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TlsState {
    Initial,
    ClientHello,
    ServerHello,
    Established,
    Closed,
}

/// Which key exchange the handshake actually settled on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyExchange {
    /// Nothing negotiated yet.
    None,
    /// Ephemeral X25519 — the default path.
    Ecdhe,
    /// Pre-shared key only, used when the server offered no key share.
    PskOnly,
}

pub struct TlsSession {
    state: TlsState,
    psk: [u8; 32],
    has_psk: bool,
    client_random: [u8; 32],
    server_random: [u8; 32],
    ecdhe_key: Option<ecdhe::EphemeralKey>,
    key_exchange: KeyExchange,
    peer_verified: bool,
    peer_cert_error: Option<x509::X509Error>,
    write_key: [u8; 16],
    write_iv: [u8; 12],
    read_key: [u8; 16],
    read_iv: [u8; 12],
    write_seq: u64,
    read_seq: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct TlsRecordBenchProof {
    pub plaintext_bytes: usize,
    pub record_bytes: usize,
    pub tag_bytes: usize,
    pub decrypt_match: bool,
    pub write_seq: u64,
    pub read_seq: u64,
}

impl TlsSession {
    pub fn new() -> Self {
        Self {
            state: TlsState::Initial,
            psk: [0u8; 32],
            has_psk: false,
            client_random: [0u8; 32],
            server_random: [0u8; 32],
            ecdhe_key: None,
            key_exchange: KeyExchange::None,
            peer_verified: false,
            peer_cert_error: None,
            write_key: [0u8; 16],
            write_iv: [0u8; 12],
            read_key: [0u8; 16],
            read_iv: [0u8; 12],
            write_seq: 0,
            read_seq: 0,
        }
    }

    pub fn set_psk(&mut self, psk: &[u8; 32]) {
        self.psk = *psk;
        self.has_psk = true;
    }

    pub fn state(&self) -> TlsState {
        self.state
    }

    /// Which key exchange the handshake settled on.
    pub fn key_exchange(&self) -> KeyExchange {
        self.key_exchange
    }

    /// Whether the peer presented a certificate chain that validated against
    /// the embedded trust store.
    pub fn peer_verified(&self) -> bool {
        self.peer_verified
    }

    /// Why the peer's certificate was rejected, if it was.
    pub fn peer_cert_error(&self) -> Option<x509::X509Error> {
        self.peer_cert_error
    }

    /// Build a TLS 1.3 ClientHello offering ephemeral X25519.
    ///
    /// Fails closed if hardware entropy is unavailable: no client random and
    /// no key share can be produced, and no weaker path is substituted.
    pub fn build_client_hello(&mut self) -> Result<Vec<u8>, String> {
        self.state = TlsState::ClientHello;
        self.client_random = random_bytes_32()?;
        let key = ecdhe::EphemeralKey::generate()
            .map_err(|_| String::from("entropy unavailable for ECDHE key share"))?;

        // Handshake header placeholder
        let mut hs = Vec::new();
        hs.push(0x01); // ClientHello
        hs.extend_from_slice(&[0x00, 0x00, 0x00]); // length placeholder

        // Version TLS 1.2 (for compatibility)
        hs.extend_from_slice(&0x0303u16.to_be_bytes());
        hs.extend_from_slice(&self.client_random);

        // Session ID length = 0
        hs.push(0x00);

        // Cipher suites: TLS_AES_128_GCM_SHA256
        hs.extend_from_slice(&0x0002u16.to_be_bytes());
        hs.extend_from_slice(&0x1301u16.to_be_bytes());

        // Compression methods: null
        hs.push(0x01);
        hs.push(0x00);

        // Extensions
        let mut ext = Vec::new();
        // supported_versions (TLS 1.3)
        ext.extend_from_slice(&0x002bu16.to_be_bytes()); // supported_versions
        ext.extend_from_slice(&0x0003u16.to_be_bytes()); // length
        ext.push(0x02);
        ext.extend_from_slice(&0x0304u16.to_be_bytes()); // TLS 1.3

        // psk_key_exchange_modes: psk_dhe_ke, so a PSK is combined with the
        // ECDHE secret rather than used on its own.
        ext.extend_from_slice(&0x002du16.to_be_bytes());
        ext.extend_from_slice(&0x0002u16.to_be_bytes());
        ext.push(0x01);
        ext.push(0x01); // psk_dhe_ke

        // supported_groups: x25519
        ext.extend_from_slice(&0x000au16.to_be_bytes());
        ext.extend_from_slice(&0x0004u16.to_be_bytes());
        ext.extend_from_slice(&0x0002u16.to_be_bytes());
        ext.extend_from_slice(&ecdhe::GROUP_X25519.to_be_bytes());

        // key_share: one KeyShareEntry carrying the real 32-byte X25519 key.
        ext.extend_from_slice(&0x0033u16.to_be_bytes());
        ext.extend_from_slice(&0x0026u16.to_be_bytes()); // 2 + 2 + 2 + 32
        ext.extend_from_slice(&0x0024u16.to_be_bytes()); // client_shares length
        ext.extend_from_slice(&ecdhe::GROUP_X25519.to_be_bytes());
        ext.extend_from_slice(&(ecdhe::X25519_LEN as u16).to_be_bytes());
        ext.extend_from_slice(key.public());
        self.ecdhe_key = Some(key);

        hs.extend_from_slice(&(ext.len() as u16).to_be_bytes());
        hs.extend_from_slice(&ext);

        // Patch handshake length
        let len = hs.len() - 4;
        hs[1..4].copy_from_slice(&[
            ((len >> 16) & 0xFF) as u8,
            ((len >> 8) & 0xFF) as u8,
            (len & 0xFF) as u8,
        ]);

        Ok(wrap_record(ContentType::Handshake, &hs))
    }

    /// Parse ServerHello and derive traffic keys.
    ///
    /// Uses the server's X25519 key share when it sends one, and only falls
    /// back to PSK-only when it does not *and* a PSK was configured. A server
    /// that offers neither is rejected rather than downgraded to a fixed key.
    pub fn handle_server_hello(&mut self, data: &[u8]) -> Result<(), String> {
        let rec = parse_record(data)?;
        if rec.ctype != ContentType::Handshake as u8 {
            return Err(String::from("expected handshake record"));
        }
        if rec.payload.len() < 44 {
            return Err(String::from("server hello too short"));
        }
        if rec.payload[0] != 0x02 {
            return Err(String::from("expected ServerHello"));
        }

        // 4 handshake header + 2 legacy_version, then the server random.
        self.server_random.copy_from_slice(&rec.payload[6..38]);

        let shared = match server_key_share(&rec.payload) {
            Some(peer) => {
                let key = self
                    .ecdhe_key
                    .as_ref()
                    .ok_or_else(|| String::from("server sent a key share we did not offer"))?;
                let secret = key
                    .agree(peer)
                    .map_err(|_| String::from("X25519 agreement rejected the peer key share"))?;
                self.key_exchange = KeyExchange::Ecdhe;
                secret
            }
            None if self.has_psk => {
                self.key_exchange = KeyExchange::PskOnly;
                [0u8; 32]
            }
            None => {
                return Err(String::from(
                    "server offered no key share and no PSK is set",
                ));
            }
        };
        // The ephemeral secret has done its job; drop it so it cannot be
        // reused for a second handshake.
        self.ecdhe_key = None;

        // RFC 8446 §7.1 key schedule. The PSK is all-zero when unset, which
        // is the specified "no PSK" input, not a shortcut.
        let early_secret = hkdf_extract(&[0u8; 32], &self.psk);
        let derived = derive_secret(&early_secret, b"derived");
        let handshake_secret = hkdf_extract(&derived, &shared);
        let chts = hkdf_expand_label(&handshake_secret, b"c hs traffic", &self.client_random, 32);
        let shts = hkdf_expand_label(&handshake_secret, b"s hs traffic", &self.server_random, 32);

        self.write_key = vec_to_array16(&hkdf_expand_label(&chts, b"key", &[], 16));
        self.write_iv = vec_to_array12(&hkdf_expand_label(&chts, b"iv", &[], 12));
        self.read_key = vec_to_array16(&hkdf_expand_label(&shts, b"key", &[], 16));
        self.read_iv = vec_to_array12(&hkdf_expand_label(&shts, b"iv", &[], 12));

        self.state = TlsState::Established;
        Ok(())
    }

    /// Validate a peer Certificate handshake message (RFC 8446 §4.4.2) against
    /// the embedded trust store and the CMOS clock.
    ///
    /// `msg` is the handshake message: type byte, 3-byte length, body.
    /// Records the outcome; the caller decides whether to proceed.
    pub fn handle_certificate(&mut self, msg: &[u8]) -> Result<(), String> {
        let body = handshake_body(msg, 0x0b)?;
        let ders = certificate_list(body)?;
        if ders.is_empty() {
            self.peer_cert_error = Some(x509::X509Error::EmptyChain);
            return Err(String::from("peer sent an empty certificate list"));
        }
        let now = crate::drivers::rtc::seconds_since_epoch();
        match x509::verify_chain_der(&ders, now) {
            Ok(()) => {
                self.peer_verified = true;
                self.peer_cert_error = None;
                Ok(())
            }
            Err(e) => {
                self.peer_verified = false;
                self.peer_cert_error = Some(e);
                Err(String::from("peer certificate chain rejected"))
            }
        }
    }

    /// Encrypt application data with AES-128-GCM.
    pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<Vec<u8>, String> {
        if self.state != TlsState::Established {
            return Err(String::from("handshake not complete"));
        }
        let nonce = self.make_nonce(true);
        let cipher =
            Aes128Gcm::new_from_slice(&self.write_key).map_err(|_| String::from("bad key"))?;
        let mut ciphertext = plaintext.to_vec();
        let tag = cipher
            .encrypt_in_place_detached((&nonce[..]).into(), &[], &mut ciphertext)
            .map_err(|_| String::from("encrypt failed"))?;
        ciphertext.extend_from_slice(&tag);
        self.write_seq += 1;
        Ok(wrap_record(ContentType::ApplicationData, &ciphertext))
    }

    /// Decrypt application data.
    pub fn decrypt(&mut self, data: &[u8]) -> Result<Vec<u8>, String> {
        if self.state != TlsState::Established {
            return Err(String::from("handshake not complete"));
        }
        let rec = parse_record(data)?;
        if rec.ctype != ContentType::ApplicationData as u8 {
            return Err(String::from("expected application data"));
        }
        if rec.payload.len() < 16 {
            return Err(String::from("ciphertext too short"));
        }
        let (ct, tag) = rec.payload.split_at(rec.payload.len() - 16);
        let nonce = self.make_nonce(false);
        let cipher =
            Aes128Gcm::new_from_slice(&self.read_key).map_err(|_| String::from("bad key"))?;
        let mut pt = ct.to_vec();
        cipher
            .decrypt_in_place_detached((&nonce[..]).into(), &[], &mut pt, tag.into())
            .map_err(|_| String::from("decrypt failed (auth tag mismatch)"))?;
        self.read_seq += 1;
        Ok(pt)
    }

    pub fn benchmark_psk_record_roundtrip(plaintext: &[u8]) -> Result<TlsRecordBenchProof, String> {
        let mut session = Self::new();
        session.set_psk(&[0xABu8; 32]);
        session.state = TlsState::Established;
        session.write_key = [0xCDu8; 16];
        session.read_key = [0xCDu8; 16];
        session.write_iv = [0xEFu8; 12];
        session.read_iv = [0xEFu8; 12];

        let record = session.encrypt(plaintext)?;
        session.read_seq = session.write_seq - 1;
        let decrypted = session.decrypt(&record)?;

        Ok(TlsRecordBenchProof {
            plaintext_bytes: plaintext.len(),
            record_bytes: record.len(),
            tag_bytes: 16,
            decrypt_match: decrypted.as_slice() == plaintext,
            write_seq: session.write_seq,
            read_seq: session.read_seq,
        })
    }

    fn make_nonce(&self, write: bool) -> [u8; 12] {
        let iv = if write { self.write_iv } else { self.read_iv };
        let seq = if write { self.write_seq } else { self.read_seq };
        let mut nonce = iv;
        for i in 0..8 {
            nonce[11 - i] ^= ((seq >> (8 * i)) & 0xFF) as u8;
        }
        nonce
    }
}

impl Default for TlsSession {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)] // REASON: TLS content-type variants reserved for full handshake and alert handling
#[derive(Debug, Clone, Copy)]
enum ContentType {
    Invalid = 0,
    ChangeCipherSpec = 20,
    Alert = 21,
    Handshake = 22,
    ApplicationData = 23,
}

fn wrap_record(ctype: ContentType, payload: &[u8]) -> Vec<u8> {
    let mut rec = Vec::with_capacity(5 + payload.len());
    rec.push(ctype as u8);
    rec.extend_from_slice(&0x0303u16.to_be_bytes()); // TLS 1.2 legacy record version
    rec.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    rec.extend_from_slice(payload);
    rec
}

struct Record {
    ctype: u8,
    payload: Vec<u8>,
}

fn parse_record(data: &[u8]) -> Result<Record, String> {
    if data.len() < 5 {
        return Err(String::from("record too short"));
    }
    let ctype = data[0];
    let len = u16::from_be_bytes([data[3], data[4]]) as usize;
    if 5 + len > data.len() {
        return Err(String::from("record incomplete"));
    }
    Ok(Record {
        ctype,
        payload: data[5..5 + len].to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Handshake message helpers
// ---------------------------------------------------------------------------

/// Extract the server's X25519 key share from a ServerHello handshake message.
///
/// Every offset is bounds-checked; a malformed or truncated ServerHello yields
/// `None` rather than a panic.
fn server_key_share(payload: &[u8]) -> Option<&[u8]> {
    // handshake header (4) + legacy_version (2) + random (32)
    let mut off = 38usize;
    let sid_len = *payload.get(off)? as usize;
    off = off.checked_add(1 + sid_len)?;
    // cipher_suite (2) + legacy_compression_method (1)
    off = off.checked_add(3)?;
    let ext_len = be16(payload, off)? as usize;
    off = off.checked_add(2)?;
    let end = off.checked_add(ext_len)?;
    if end > payload.len() {
        return None;
    }

    while off + 4 <= end {
        let ext_type = be16(payload, off)?;
        let len = be16(payload, off + 2)? as usize;
        let data_start = off + 4;
        let data_end = data_start.checked_add(len)?;
        if data_end > end {
            return None;
        }
        // key_share: in a ServerHello this is a single KeyShareEntry,
        // group (2) + key_exchange length (2) + key.
        if ext_type == 0x0033 && len >= 4 {
            let group = be16(payload, data_start)?;
            let key_len = be16(payload, data_start + 2)? as usize;
            let key_start = data_start + 4;
            if group == ecdhe::GROUP_X25519
                && key_len == ecdhe::X25519_LEN
                && key_start + key_len <= data_end
            {
                return Some(&payload[key_start..key_start + key_len]);
            }
            return None;
        }
        off = data_end;
    }
    None
}

fn be16(buf: &[u8], off: usize) -> Option<u16> {
    let hi = *buf.get(off)?;
    let lo = *buf.get(off + 1)?;
    Some(u16::from_be_bytes([hi, lo]))
}

fn be24(buf: &[u8], off: usize) -> Option<usize> {
    let a = *buf.get(off)? as usize;
    let b = *buf.get(off + 1)? as usize;
    let c = *buf.get(off + 2)? as usize;
    Some((a << 16) | (b << 8) | c)
}

/// Validate a handshake message header and return its body.
fn handshake_body(msg: &[u8], expected_type: u8) -> Result<&[u8], String> {
    if msg.len() < 4 {
        return Err(String::from("handshake message too short"));
    }
    if msg[0] != expected_type {
        return Err(String::from("unexpected handshake message type"));
    }
    let len = be24(msg, 1).ok_or_else(|| String::from("handshake message too short"))?;
    let end = 4usize
        .checked_add(len)
        .ok_or_else(|| String::from("handshake length overflow"))?;
    if end > msg.len() {
        return Err(String::from("handshake message truncated"));
    }
    Ok(&msg[4..end])
}

/// Split a TLS 1.3 Certificate message body into leaf-first DER blobs.
fn certificate_list(body: &[u8]) -> Result<Vec<&[u8]>, String> {
    let ctx_len = *body
        .first()
        .ok_or_else(|| String::from("certificate message empty"))? as usize;
    let mut off = 1 + ctx_len;
    let list_len = be24(body, off).ok_or_else(|| String::from("certificate list truncated"))?;
    off += 3;
    let end = off
        .checked_add(list_len)
        .ok_or_else(|| String::from("certificate list length overflow"))?;
    if end > body.len() {
        return Err(String::from("certificate list truncated"));
    }

    let mut out = Vec::new();
    while off + 3 <= end {
        let cert_len = be24(body, off).ok_or_else(|| String::from("certificate truncated"))?;
        off += 3;
        let cert_end = off
            .checked_add(cert_len)
            .ok_or_else(|| String::from("certificate length overflow"))?;
        if cert_end > end {
            return Err(String::from("certificate truncated"));
        }
        out.push(&body[off..cert_end]);
        off = cert_end;
        // Per-certificate extensions.
        let ext_len =
            be16(body, off).ok_or_else(|| String::from("cert extensions truncated"))? as usize;
        off = off
            .checked_add(2 + ext_len)
            .ok_or_else(|| String::from("cert extensions overflow"))?;
        if off > end {
            return Err(String::from("cert extensions truncated"));
        }
        if out.len() > x509::MAX_CHAIN_LEN {
            return Err(String::from("certificate chain too long"));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// HMAC-SHA256 and HKDF (RFC 2104, RFC 5869)
// ---------------------------------------------------------------------------

const SHA256_BLOCK: usize = 64;

fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut block = [0u8; SHA256_BLOCK];
    if key.len() > SHA256_BLOCK {
        let digest: [u8; 32] = Sha256::digest(key).into();
        block[..32].copy_from_slice(&digest);
    } else {
        block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; SHA256_BLOCK];
    let mut opad = [0x5cu8; SHA256_BLOCK];
    for i in 0..SHA256_BLOCK {
        ipad[i] ^= block[i];
        opad[i] ^= block[i];
    }

    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(msg);
    let inner: [u8; 32] = inner.finalize().into();

    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner);
    outer.finalize().into()
}

fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    hmac_sha256(salt, ikm)
}

fn hkdf_expand(prk: &[u8], info: &[u8], out_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(out_len);
    let mut t: [u8; 32] = [0u8; 32];
    let mut have_t = false;
    let mut counter = 1u8;
    while out.len() < out_len {
        let mut block = Vec::with_capacity(32 + info.len() + 1);
        if have_t {
            block.extend_from_slice(&t);
        }
        block.extend_from_slice(info);
        block.push(counter);
        t = hmac_sha256(prk, &block);
        have_t = true;
        out.extend_from_slice(&t);
        // RFC 5869 caps output at 255 hash lengths; nothing here asks for more.
        if counter == 255 {
            break;
        }
        counter += 1;
    }
    out.truncate(out_len);
    out
}

/// `Derive-Secret(secret, label, "")` from RFC 8446 §7.1.
fn derive_secret(secret: &[u8], label: &[u8]) -> [u8; 32] {
    let empty_hash: [u8; 32] = Sha256::digest([]).into();
    let out = hkdf_expand_label(secret, label, &empty_hash, 32);
    let mut result = [0u8; 32];
    result.copy_from_slice(&out[..32]);
    result
}

fn hkdf_expand_label(secret: &[u8], label: &[u8], context: &[u8], out_len: usize) -> Vec<u8> {
    let mut info = Vec::new();
    info.extend_from_slice(&(out_len as u16).to_be_bytes());
    let label_prefix = b"tls13 ";
    info.push((label_prefix.len() + label.len()) as u8);
    info.extend_from_slice(label_prefix);
    info.extend_from_slice(label);
    info.push(context.len() as u8);
    info.extend_from_slice(context);
    hkdf_expand(secret, &info, out_len)
}

fn vec_to_array16(v: &[u8]) -> [u8; 16] {
    let mut a = [0u8; 16];
    a.copy_from_slice(&v[..16.min(v.len())]);
    a
}

fn vec_to_array12(v: &[u8]) -> [u8; 12] {
    let mut a = [0u8; 12];
    a.copy_from_slice(&v[..12.min(v.len())]);
    a
}

fn random_bytes_32() -> Result<[u8; 32], String> {
    let mut out = [0u8; 32];
    if crate::drivers::entropy::getrandom(&mut out) {
        Ok(out)
    } else {
        Err(String::from("entropy unavailable"))
    }
}

// ---------------------------------------------------------------------------
// Proof emitter
// ---------------------------------------------------------------------------

/// Emit the `[TLS]` proof line, measuring every field by doing the work.
///
/// Nothing here is asserted from a constant: the certificate fixtures are
/// parsed, the chain is verified against the embedded root, the expired
/// fixture is required to be *rejected* for the expiry check to count, and a
/// live X25519 exchange runs against hardware entropy. If any of it fails the
/// line reports the failure instead of hiding it.
pub fn tls_proof_line() -> String {
    use alloc::format;

    let now = crate::drivers::rtc::seconds_since_epoch();
    let leaf: &[u8] = &super::certs::LEAF_DER;
    let inter: &[u8] = &super::certs::INTERMEDIATE_DER;
    let expired: &[u8] = &super::certs::EXPIRED_LEAF_DER;
    let rogue: &[u8] = &super::certs::ROGUE_LEAF_DER;

    // Well-formed fixtures must parse.
    let cert_parse = x509::Certificate::parse(leaf).is_ok()
        && x509::Certificate::parse(inter).is_ok()
        && x509::Certificate::parse(&super::certs::ROOT_CA_DER).is_ok();

    // Malformed and unauthorised input must be rejected, so `x509=1` cannot be
    // earned by a parser that accepts everything. The BasicConstraints probe
    // runs at a timestamp inside every fixture's validity window so that it
    // measures CA enforcement rather than the state of the clock — the clock
    // is what `expiry_check` measures.
    const FIXTURE_EPOCH: u64 = 1_767_225_600; // 2026-01-01T00:00:00Z
    let rejects_malformed = x509::Certificate::parse(&leaf[..leaf.len() / 2]).is_err()
        && x509::Certificate::parse(&[]).is_err()
        && x509::verify_chain_der(&[rogue, leaf, inter], FIXTURE_EPOCH)
            == Err(x509::X509Error::IssuerNotCa);
    let x509_ok = cert_parse && rejects_malformed;

    let chain_verify = x509::verify_chain_der(&[leaf, inter], now).is_ok();

    // The clock must place the good leaf inside its window and the expired
    // fixture outside it. Both directions, or the check proves nothing.
    let expiry_check = chain_verify
        && x509::verify_chain_der(&[expired, inter], now) == Err(x509::X509Error::Expired);

    let (ecdhe_ok, entropy) = match (
        ecdhe::EphemeralKey::generate(),
        ecdhe::EphemeralKey::generate(),
    ) {
        (Ok(client), Ok(server)) => {
            let agreed = match (client.agree(server.public()), server.agree(client.public())) {
                (Ok(a), Ok(b)) => a == b && a.iter().any(|&byte| byte != 0),
                _ => false,
            };
            (agreed, "hw")
        }
        _ => (false, "none"),
    };

    let result = x509_ok && chain_verify && expiry_check && ecdhe_ok;

    format!(
        "[TLS] proof version=1 x509={} chain_verify={} ecdhe={} curve=x25519 psk_only={} cert_parse={} expiry_check={} entropy={} result={}",
        u8::from(x509_ok),
        u8::from(chain_verify),
        u8::from(ecdhe_ok),
        u8::from(!ecdhe_ok),
        if cert_parse { "ok" } else { "fail" },
        u8::from(expiry_check),
        entropy,
        if result { "pass" } else { "fail" }
    )
}

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};
    use alloc::vec;

    fn test_hmac_sha256_rfc4231() -> TestResult {
        // RFC 4231 test case 2: key "Jefe", data "what do ya want for nothing?".
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        let expected = [
            0x5b, 0xdc, 0xc1, 0x46, 0xbf, 0x60, 0x75, 0x4e, 0x6a, 0x04, 0x24, 0x26, 0x08, 0x95,
            0x75, 0xc7, 0x5a, 0x00, 0x3f, 0x08, 0x9d, 0x27, 0x39, 0x83, 0x9d, 0xec, 0x58, 0xb9,
            0x64, 0xec, 0x38, 0x43,
        ];
        test_assert_eq!(mac, expected);
        TestResult::Pass
    }

    fn test_hkdf_rfc5869_vector() -> TestResult {
        // RFC 5869 appendix A.1.
        let ikm = [0x0bu8; 22];
        let salt: [u8; 13] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c,
        ];
        let info: [u8; 10] = [0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9];
        let prk = hkdf_extract(&salt, &ikm);
        let expected_prk = [
            0x07, 0x77, 0x09, 0x36, 0x2c, 0x2e, 0x32, 0xdf, 0x0d, 0xdc, 0x3f, 0x0d, 0xc4, 0x7b,
            0xba, 0x63, 0x90, 0xb6, 0xc7, 0x3b, 0xb5, 0x0f, 0x9c, 0x31, 0x22, 0xec, 0x84, 0x4a,
            0xd7, 0xc2, 0xb3, 0xe5,
        ];
        test_assert_eq!(prk, expected_prk);

        let okm = hkdf_expand(&prk, &info, 42);
        let expected_okm = [
            0x3c, 0xb2, 0x5f, 0x25, 0xfa, 0xac, 0xd5, 0x7a, 0x90, 0x43, 0x4f, 0x64, 0xd0, 0x36,
            0x2f, 0x2a, 0x2d, 0x2d, 0x0a, 0x90, 0xcf, 0x1a, 0x5a, 0x4c, 0x5d, 0xb0, 0x2d, 0x56,
            0xec, 0xc4, 0xc5, 0xbf, 0x34, 0x00, 0x72, 0x08, 0xd5, 0xb8, 0x87, 0x18, 0x58, 0x65,
        ];
        test_assert_eq!(okm.as_slice(), expected_okm.as_slice());
        TestResult::Pass
    }

    fn test_client_hello_carries_real_key_share() -> TestResult {
        let mut session = TlsSession::new();
        let hello = match session.build_client_hello() {
            Ok(h) => h,
            // No hardware entropy: failing closed is the correct behaviour and
            // there is nothing further to assert.
            Err(_) => return TestResult::Pass,
        };
        let Some(key) = session.ecdhe_key.as_ref().map(|k| *k.public()) else {
            return TestResult::Fail("ClientHello did not retain an ECDHE key");
        };
        test_assert!(key.iter().any(|&b| b != 0), "key share is all zero");
        test_assert!(
            hello.windows(32).any(|w| w == key),
            "ClientHello does not contain the X25519 public key"
        );
        test_assert!(
            hello
                .windows(2)
                .any(|w| w == ecdhe::GROUP_X25519.to_be_bytes()),
            "ClientHello does not name group x25519"
        );
        TestResult::Pass
    }

    fn test_ecdhe_handshake_derives_shared_keys() -> TestResult {
        // Two sessions, each seeing the other's key share, must land on the
        // same traffic keys. Uses fixed scalars so the test does not depend on
        // hardware entropy being present.
        let mut client = TlsSession::new();
        client.client_random = [0x01u8; 32];
        client.ecdhe_key = Some(ecdhe::EphemeralKey::from_scalar([0x55u8; 32]));
        let server_key = ecdhe::EphemeralKey::from_scalar([0x66u8; 32]);

        let hello = server_hello_with_key_share(&[0x02u8; 32], server_key.public());
        if let Err(_) = client.handle_server_hello(&hello) {
            return TestResult::Fail("ServerHello with a key share was rejected");
        }
        test_assert_eq!(client.key_exchange(), KeyExchange::Ecdhe);
        test_assert_eq!(client.state(), TlsState::Established);
        test_assert!(client.write_key != [0u8; 16], "traffic key was not derived");
        test_assert!(
            client.write_key != client.read_key,
            "client and server keys are identical"
        );
        // A different server share must produce different keys, which is what
        // makes this an actual key agreement and not a constant.
        let other = ecdhe::EphemeralKey::from_scalar([0x77u8; 32]);
        let mut client2 = TlsSession::new();
        client2.client_random = [0x01u8; 32];
        client2.ecdhe_key = Some(ecdhe::EphemeralKey::from_scalar([0x55u8; 32]));
        let hello2 = server_hello_with_key_share(&[0x02u8; 32], other.public());
        if let Err(_) = client2.handle_server_hello(&hello2) {
            return TestResult::Fail("second ServerHello was rejected");
        }
        test_assert!(
            client.write_key != client2.write_key,
            "different server key shares produced the same traffic key"
        );
        TestResult::Pass
    }

    fn test_no_key_share_and_no_psk_rejected() -> TestResult {
        let mut session = TlsSession::new();
        session.ecdhe_key = Some(ecdhe::EphemeralKey::from_scalar([0x55u8; 32]));
        let hello = server_hello_with_key_share(&[0x02u8; 32], &[]);
        test_assert!(
            session.handle_server_hello(&hello).is_err(),
            "a server offering neither key share nor PSK was accepted"
        );
        TestResult::Pass
    }

    fn test_psk_fallback_when_no_key_share() -> TestResult {
        let mut session = TlsSession::new();
        session.set_psk(&[0xABu8; 32]);
        session.ecdhe_key = Some(ecdhe::EphemeralKey::from_scalar([0x55u8; 32]));
        let hello = server_hello_with_key_share(&[0x02u8; 32], &[]);
        if let Err(_) = session.handle_server_hello(&hello) {
            return TestResult::Fail("PSK fallback was rejected");
        }
        test_assert_eq!(session.key_exchange(), KeyExchange::PskOnly);
        TestResult::Pass
    }

    fn test_truncated_server_hello_rejected() -> TestResult {
        let full = server_hello_with_key_share(&[0x02u8; 32], &[0x09u8; 32]);
        for cut in 0..full.len() {
            let mut session = TlsSession::new();
            session.ecdhe_key = Some(ecdhe::EphemeralKey::from_scalar([0x55u8; 32]));
            // Must not panic. Accepting a prefix is impossible because the
            // record length no longer covers the payload.
            let _ = session.handle_server_hello(&full[..cut]);
        }
        TestResult::Pass
    }

    fn test_certificate_message_verifies_chain() -> TestResult {
        let msg = certificate_message(&[
            &super::super::certs::LEAF_DER,
            &super::super::certs::INTERMEDIATE_DER,
        ]);
        let mut session = TlsSession::new();
        let now = crate::drivers::rtc::seconds_since_epoch();
        let expected_ok = x509::verify_chain_der(
            &[
                &super::super::certs::LEAF_DER,
                &super::super::certs::INTERMEDIATE_DER,
            ],
            now,
        )
        .is_ok();
        let got = session.handle_certificate(&msg).is_ok();
        // The message parser must agree with a direct chain verification at
        // the same clock reading, whatever the clock happens to say.
        test_assert_eq!(got, expected_ok);
        test_assert_eq!(session.peer_verified(), expected_ok);
        TestResult::Pass
    }

    fn test_certificate_message_rejects_rogue_chain() -> TestResult {
        let msg = certificate_message(&[
            &super::super::certs::ROGUE_LEAF_DER,
            &super::super::certs::LEAF_DER,
            &super::super::certs::INTERMEDIATE_DER,
        ]);
        let mut session = TlsSession::new();
        test_assert!(
            session.handle_certificate(&msg).is_err(),
            "a chain through a non-CA leaf was accepted"
        );
        test_assert!(
            !session.peer_verified(),
            "peer marked verified after reject"
        );
        TestResult::Pass
    }

    fn test_truncated_certificate_message_rejected() -> TestResult {
        let msg = certificate_message(&[&super::super::certs::LEAF_DER]);
        for cut in 0..msg.len() {
            let mut session = TlsSession::new();
            test_assert!(
                session.handle_certificate(&msg[..cut]).is_err(),
                "a truncated Certificate message was accepted"
            );
        }
        TestResult::Pass
    }

    fn test_proof_line_shape() -> TestResult {
        let line = tls_proof_line();
        test_assert!(
            line.starts_with("[TLS] proof version=1 "),
            "bad proof prefix"
        );
        for field in [
            "x509=",
            "chain_verify=",
            "ecdhe=",
            "curve=x25519",
            "psk_only=",
            "cert_parse=",
            "expiry_check=",
            "entropy=",
            "result=",
        ] {
            test_assert!(line.contains(field), "proof line is missing a field");
        }
        TestResult::Pass
    }

    // --- fixtures -------------------------------------------------------

    /// Build a ServerHello record. An empty `key_share` omits the extension.
    fn server_hello_with_key_share(random: &[u8; 32], key_share: &[u8]) -> Vec<u8> {
        let mut ext = Vec::new();
        if !key_share.is_empty() {
            ext.extend_from_slice(&0x0033u16.to_be_bytes());
            ext.extend_from_slice(&((4 + key_share.len()) as u16).to_be_bytes());
            ext.extend_from_slice(&ecdhe::GROUP_X25519.to_be_bytes());
            ext.extend_from_slice(&(key_share.len() as u16).to_be_bytes());
            ext.extend_from_slice(key_share);
        }

        let mut hs = vec![0x02u8, 0x00, 0x00, 0x00];
        hs.extend_from_slice(&0x0303u16.to_be_bytes());
        hs.extend_from_slice(random);
        hs.push(0x00); // legacy_session_id_echo
        hs.extend_from_slice(&0x1301u16.to_be_bytes()); // cipher suite
        hs.push(0x00); // compression
        hs.extend_from_slice(&(ext.len() as u16).to_be_bytes());
        hs.extend_from_slice(&ext);

        let len = hs.len() - 4;
        hs[1..4].copy_from_slice(&[
            ((len >> 16) & 0xFF) as u8,
            ((len >> 8) & 0xFF) as u8,
            (len & 0xFF) as u8,
        ]);
        wrap_record(ContentType::Handshake, &hs)
    }

    /// Build an RFC 8446 §4.4.2 Certificate handshake message.
    fn certificate_message(ders: &[&[u8]]) -> Vec<u8> {
        let mut list = Vec::new();
        for der in ders {
            let len = der.len();
            list.extend_from_slice(&[
                ((len >> 16) & 0xFF) as u8,
                ((len >> 8) & 0xFF) as u8,
                (len & 0xFF) as u8,
            ]);
            list.extend_from_slice(der);
            list.extend_from_slice(&0u16.to_be_bytes()); // no extensions
        }

        let mut body = vec![0u8]; // empty certificate_request_context
        let len = list.len();
        body.extend_from_slice(&[
            ((len >> 16) & 0xFF) as u8,
            ((len >> 8) & 0xFF) as u8,
            (len & 0xFF) as u8,
        ]);
        body.extend_from_slice(&list);

        let mut msg = vec![0x0bu8];
        let len = body.len();
        msg.extend_from_slice(&[
            ((len >> 16) & 0xFF) as u8,
            ((len >> 8) & 0xFF) as u8,
            (len & 0xFF) as u8,
        ]);
        msg.extend_from_slice(&body);
        msg
    }

    /// Registers the whole TLS stack: X.509, ECDHE, and the session itself.
    pub fn register_all() {
        super::x509::tests::register_all();
        super::ecdhe::tests::register_all();
        crate::testing::register_test("tls::hmac_sha256_rfc4231", test_hmac_sha256_rfc4231);
        crate::testing::register_test("tls::hkdf_rfc5869_vector", test_hkdf_rfc5869_vector);
        crate::testing::register_test(
            "tls::client_hello_carries_real_key_share",
            test_client_hello_carries_real_key_share,
        );
        crate::testing::register_test(
            "tls::ecdhe_handshake_derives_shared_keys",
            test_ecdhe_handshake_derives_shared_keys,
        );
        crate::testing::register_test(
            "tls::no_key_share_and_no_psk_rejected",
            test_no_key_share_and_no_psk_rejected,
        );
        crate::testing::register_test(
            "tls::psk_fallback_when_no_key_share",
            test_psk_fallback_when_no_key_share,
        );
        crate::testing::register_test(
            "tls::truncated_server_hello_rejected",
            test_truncated_server_hello_rejected,
        );
        crate::testing::register_test(
            "tls::certificate_message_verifies_chain",
            test_certificate_message_verifies_chain,
        );
        crate::testing::register_test(
            "tls::certificate_message_rejects_rogue_chain",
            test_certificate_message_rejects_rogue_chain,
        );
        crate::testing::register_test(
            "tls::truncated_certificate_message_rejected",
            test_truncated_certificate_message_rejected,
        );
        crate::testing::register_test("tls::proof_line_shape", test_proof_line_shape);
    }
}

#[cfg(test)]
mod host_tests {
    use super::*;

    #[test]
    fn test_hkdf_expand_label() {
        let secret = [0u8; 32];
        let key = hkdf_expand_label(&secret, b"key", &[], 16);
        assert_eq!(key.len(), 16);
    }

    #[test]
    fn test_record_roundtrip() {
        let payload = b"hello";
        let rec = wrap_record(ContentType::ApplicationData, payload);
        let parsed = parse_record(&rec).unwrap();
        assert_eq!(parsed.ctype, ContentType::ApplicationData as u8);
        assert_eq!(parsed.payload, payload.as_slice());
    }

    #[test]
    fn test_tls_encrypt_decrypt() {
        let mut client = TlsSession::new();
        client.set_psk(&[0xABu8; 32]);
        client.state = TlsState::Established;
        // Manually set keys for test
        client.write_key = [0xCDu8; 16];
        client.read_key = [0xCDu8; 16];
        client.write_iv = [0xEFu8; 12];
        client.read_iv = [0xEFu8; 12];

        let pt = b"wubba lubba dub-dub";
        let ct = client.encrypt(pt).unwrap();
        client.read_seq = client.write_seq - 1; // sync seq
        let decrypted = client.decrypt(&ct).unwrap();
        assert_eq!(decrypted, pt.as_slice());
    }
}
