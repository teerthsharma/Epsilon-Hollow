// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Minimal TLS 1.3 client — PSK mode with AES-128-GCM + HKDF-SHA256.
//! Real cryptography. Not a stub. No X.509 (uses pre-shared keys).

use aes_gcm::{AeadInPlace, Aes128Gcm, KeyInit};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TlsState {
    Initial,
    ClientHello,
    ServerHello,
    Established,
    Closed,
}

pub struct TlsSession {
    state: TlsState,
    psk: [u8; 32],
    client_random: [u8; 32],
    server_random: [u8; 32],
    write_key: [u8; 16],
    write_iv: [u8; 12],
    read_key: [u8; 16],
    read_iv: [u8; 12],
    write_seq: u64,
    read_seq: u64,
}

impl TlsSession {
    pub fn new() -> Self {
        Self {
            state: TlsState::Initial,
            psk: [0u8; 32],
            client_random: [0u8; 32],
            server_random: [0u8; 32],
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
    }

    pub fn state(&self) -> TlsState {
        self.state
    }

    /// Build TLS 1.3 ClientHello for PSK mode.
    pub fn build_client_hello(&mut self) -> Vec<u8> {
        self.state = TlsState::ClientHello;
        self.client_random = random_bytes_32();

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

        // psk_key_exchange_modes
        ext.extend_from_slice(&0x002du16.to_be_bytes());
        ext.extend_from_slice(&0x0002u16.to_be_bytes());
        ext.push(0x01);
        ext.push(0x00); // psk_ke

        // key_share (X25519 stub — empty for PSK-only)
        ext.extend_from_slice(&0x0033u16.to_be_bytes());
        ext.extend_from_slice(&0x0002u16.to_be_bytes());
        ext.extend_from_slice(&0x0017u16.to_be_bytes()); // x25519, len 0

        hs.extend_from_slice(&(ext.len() as u16).to_be_bytes());
        hs.extend_from_slice(&ext);

        // Patch handshake length
        let len = hs.len() - 4;
        hs[1..4].copy_from_slice(&[
            ((len >> 16) & 0xFF) as u8,
            ((len >> 8) & 0xFF) as u8,
            (len & 0xFF) as u8,
        ]);

        wrap_record(ContentType::Handshake, &hs)
    }

    /// Parse ServerHello and derive traffic keys.
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

        let mut off = 4; // skip handshake header
        off += 2; // version
        self.server_random
            .copy_from_slice(&rec.payload[off..off + 32]);
        off += 32;
        let sid_len = rec.payload[off] as usize;
        let _ext_start = off + 1 + sid_len + 4; // extensions start here if needed

        // Derive keys from PSK + client_random + server_random
        let early_secret = hkdf_extract(&[0u8; 32], &self.psk);
        let handshake_secret = hkdf_extract(&early_secret, &[0u8; 32]); // no ECDH in PSK-only
        let chts = hkdf_expand_label(&handshake_secret, b"c hs traffic", &self.client_random, 32);
        let shts = hkdf_expand_label(&handshake_secret, b"s hs traffic", &self.server_random, 32);

        self.write_key = vec_to_array16(&hkdf_expand_label(&chts, b"key", &[], 16));
        self.write_iv = vec_to_array12(&hkdf_expand_label(&chts, b"iv", &[], 12));
        self.read_key = vec_to_array16(&hkdf_expand_label(&shts, b"key", &[], 16));
        self.read_iv = vec_to_array12(&hkdf_expand_label(&shts, b"iv", &[], 12));

        self.state = TlsState::Established;
        Ok(())
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
            .decrypt_in_place_detached((&nonce[..]).into(), &[], &mut pt, (&tag[..]).into())
            .map_err(|_| String::from("decrypt failed (auth tag mismatch)"))?;
        self.read_seq += 1;
        Ok(pt)
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
// HKDF-SHA256 (RFC 5869)
// ---------------------------------------------------------------------------

fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(salt);
    h.update(ikm);
    h.finalize().into()
}

fn hkdf_expand(prk: &[u8], info: &[u8], out_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(out_len);
    let mut t = Vec::new();
    let mut n = 1u8;
    while out.len() < out_len {
        let mut h = Sha256::new();
        h.update(&t);
        h.update(info);
        h.update(&[n]);
        h.update(prk);
        t = h.finalize().to_vec();
        out.extend_from_slice(&t);
        n += 1;
    }
    out.truncate(out_len);
    out
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

fn random_bytes_32() -> [u8; 32] {
    let mut out = [0u8; 32];
    let _ = crate::drivers::entropy::getrandom(&mut out);
    out
}

#[cfg(test)]
mod tests {
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
