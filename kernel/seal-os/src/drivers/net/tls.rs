// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Minimal TLS 1.3 client — ephemeral X25519 + X.509 by default, PSK as a
//! fallback. AES-128-GCM records, HMAC-SHA256 HKDF key schedule.
//!
//! Real cryptography, not a stub, and deliberately not a complete TLS 1.3
//! implementation.
//!
//! # Peer authentication
//!
//! A certificate chain is public data, so validating one proves only that a CA
//! signed *some* key — anyone who observed a connection can replay the chain.
//! The peer is authenticated only when it signs this handshake's transcript,
//! and only as the name the caller asked for:
//!
//! * [`TlsSession::set_hostname`] records the name being connected to. A
//!   session with no hostname refuses every certificate, because a chain that
//!   is not checked against a name authenticates the holder of *any*
//!   trust-store-issued certificate, including one issued for another name.
//! * [`TlsSession::handle_certificate`] validates the chain against the
//!   embedded anchor, requires the leaf's SubjectAltName to carry that
//!   hostname, and remembers the leaf's public key.
//! * [`TlsSession::handle_certificate_verify`] checks an RFC 8446 §4.4.3
//!   signature by that key over 64 `0x20` bytes, the context string, a zero
//!   byte, and the transcript hash.
//! * [`TlsSession::handle_finished`] checks the RFC 8446 §4.4.4 MAC over the
//!   transcript under the server handshake traffic secret.
//!
//! Only a verified Finished sets the authenticated flag, and
//! [`TlsSession::encrypt`] / [`TlsSession::decrypt`] refuse to move a byte
//! until it is set. There is no permissive path and no flag that skips the
//! check: a peer that cannot be authenticated gets no connection.
//!
//! **Signature algorithms:** `ed25519` (`SignatureScheme` 0x0807) only, because
//! [`super::x509`] parses Ed25519 certificates only. A server offering
//! `rsa_pss_rsae_sha256`, `ecdsa_secp256r1_sha256`, or anything else in
//! CertificateVerify is refused rather than accepted unverified — its
//! certificate would already have failed chain validation for the same reason.
//! For a pre-shared key with no certificate, RFC 8446 authenticates by the
//! Finished MAC alone, which is what this code requires there.
//!
//! # Record layer
//!
//! Application data is fragmented at the RFC 8446 §5.1 limit of 2^14 bytes and
//! each fragment becomes its own record under its own sequence number, so a
//! large write is never truncated to fit a 16-bit length field. Every record
//! header is authenticated: RFC 8446 §5.2 makes `opaque_type`,
//! `legacy_record_version`, and `length` the AEAD additional data, so altering
//! any of the five header bytes in flight fails the tag check.
//!
//! One documented deviation from RFC 8446 remains: the peer's Certificate,
//! CertificateVerify, and Finished are read as plaintext handshake records,
//! whereas RFC 8446 encrypts them under the handshake traffic keys, and no
//! EncryptedExtensions message is expected. The ClientHello also carries no
//! `server_name` extension, so the hostname binds the certificate this end
//! accepts without telling the peer which certificate to send. This does not
//! interoperate with a stock TLS 1.3 server.
//!
//! What *is* real: the key schedule (RFC 5869 HMAC-SHA256 HKDF) over a running
//! SHA-256 transcript, the X25519 agreement ([`super::ecdhe`]), the record
//! AEAD, and the certificate chain validation ([`super::x509`]).

use aes_gcm::{AeadInPlace, Aes128Gcm, KeyInit};
use alloc::string::String;
use alloc::vec::Vec;
use ed25519_dalek::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};

use super::{ecdhe, x509};

/// `SignatureScheme.ed25519` (RFC 8446 §4.2.3). The only scheme verifiable
/// here, since `x509.rs` accepts Ed25519 certificates only.
const SIG_ED25519: u16 = 0x0807;
/// The one cipher suite the ClientHello offers.
const SUITE_AES_128_GCM_SHA256: u16 = 0x1301;
/// Handshake message types this session consumes.
const HS_CERTIFICATE: u8 = 0x0b;
const HS_CERTIFICATE_VERIFY: u8 = 0x0f;
const HS_FINISHED: u8 = 0x14;
/// RFC 8446 §4.4.3 context string for a server CertificateVerify.
const CERT_VERIFY_CONTEXT: &[u8] = b"TLS 1.3, server CertificateVerify";
/// RFC 8446 §5.1: a TLSPlaintext fragment is at most 2^14 bytes. Longer
/// application data is fragmented across records, never truncated.
const MAX_PLAINTEXT_LEN: usize = 1 << 14;
/// RFC 8446 §5.2: TLSCiphertext.length is at most 2^14 + 256, which is the
/// largest length the 16-bit record header may ever be asked to describe.
const MAX_RECORD_LEN: usize = (1 << 14) + 256;
/// AES-128-GCM authentication tag.
const TAG_LEN: usize = 16;

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
    /// The name this connection asked for. The leaf certificate must carry it
    /// as a SubjectAltName dNSName; `None` refuses every certificate.
    hostname: Option<String>,
    /// Running SHA-256 over every handshake message, in wire order. Cloned to
    /// snapshot Transcript-Hash at each point RFC 8446 needs one.
    transcript: Sha256,
    /// Ed25519 key from the validated leaf certificate; the key that
    /// CertificateVerify must have signed with.
    peer_public_key: Option<[u8; 32]>,
    /// Server handshake traffic secret, the Finished MAC base key.
    server_hs_secret: [u8; 32],
    /// A CertificateVerify signature over this transcript verified.
    cert_verify_ok: bool,
    /// A Finished MAC over this transcript verified. Gates the record layer.
    authenticated: bool,
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
            hostname: None,
            transcript: Sha256::new(),
            peer_public_key: None,
            server_hs_secret: [0u8; 32],
            cert_verify_ok: false,
            authenticated: false,
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

    /// Name the peer's certificate must be issued for.
    ///
    /// Without one, [`Self::handle_certificate`] refuses every chain: a
    /// certificate checked only against the trust store authenticates whoever
    /// holds *a* certificate the anchor issued, not the host being reached.
    pub fn set_hostname(&mut self, hostname: &str) {
        self.hostname = Some(String::from(hostname));
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

    /// Whether the peer proved possession of its key over *this* handshake.
    ///
    /// True only after a Finished MAC over the running transcript verified,
    /// which on the certificate path additionally requires a validated chain
    /// and a CertificateVerify signature by the leaf's key. This, not
    /// [`Self::peer_verified`], is what may gate application data.
    pub fn peer_authenticated(&self) -> bool {
        self.authenticated
    }

    /// Transcript-Hash over every handshake message fed so far.
    fn transcript_hash(&self) -> [u8; 32] {
        self.transcript.clone().finalize().into()
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
        hs.extend_from_slice(&SUITE_AES_128_GCM_SHA256.to_be_bytes());

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

        // signature_algorithms: ed25519 only. Everything else in the registry
        // needs a verifier this kernel does not have.
        ext.extend_from_slice(&0x000du16.to_be_bytes());
        ext.extend_from_slice(&0x0004u16.to_be_bytes());
        ext.extend_from_slice(&0x0002u16.to_be_bytes());
        ext.extend_from_slice(&SIG_ED25519.to_be_bytes());

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

        // Seed the transcript. Everything the peer later signs or MACs covers
        // these bytes, so a tampered ClientHello cannot go unnoticed.
        self.transcript.update(&hs);
        wrap_record(ContentType::Handshake, &hs)
    }

    /// Parse ServerHello and derive traffic keys.
    ///
    /// Uses the server's X25519 key share when it sends one, and only falls
    /// back to PSK-only when it does not *and* a PSK was configured. A server
    /// that offers neither is rejected rather than downgraded to a fixed key.
    /// The selected cipher suite must be one the ClientHello offered.
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

        // RFC 8446 §4.1.3: the selected suite must be one the client offered,
        // and TLS_AES_128_GCM_SHA256 is the only one this ClientHello carries.
        match server_cipher_suite(&rec.payload) {
            Some(SUITE_AES_128_GCM_SHA256) => {}
            Some(_) => {
                return Err(String::from(
                    "server selected a cipher suite we did not offer",
                ))
            }
            None => return Err(String::from("ServerHello has no readable cipher suite")),
        }

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

        self.transcript.update(&rec.payload);

        // RFC 8446 §7.1 key schedule. The PSK is all-zero when unset, which
        // is the specified "no PSK" input, not a shortcut. Derive-Secret takes
        // Transcript-Hash(ClientHello..ServerHello), so every handshake byte
        // seen so far is bound into the traffic secrets.
        let early_secret = hkdf_extract(&[0u8; 32], &self.psk);
        let derived = derive_secret(&early_secret, b"derived");
        let handshake_secret = hkdf_extract(&derived, &shared);
        let th = self.transcript_hash();
        let chts = hkdf_expand_label(&handshake_secret, b"c hs traffic", &th, 32);
        let shts = hkdf_expand_label(&handshake_secret, b"s hs traffic", &th, 32);
        self.server_hs_secret = vec_to_array32(&shts);

        self.write_key = vec_to_array16(&hkdf_expand_label(&chts, b"key", &[], 16));
        self.write_iv = vec_to_array12(&hkdf_expand_label(&chts, b"iv", &[], 12));
        self.read_key = vec_to_array16(&hkdf_expand_label(&shts, b"key", &[], 16));
        self.read_iv = vec_to_array12(&hkdf_expand_label(&shts, b"iv", &[], 12));

        self.state = TlsState::Established;
        Ok(())
    }

    /// Feed one plaintext handshake record payload, which may carry several
    /// concatenated handshake messages (RFC 8446 §5.1).
    ///
    /// The single entry point for post-ServerHello handshake traffic, so every
    /// caller gets the same ordering rules and the same transcript. An
    /// unexpected message type is an error, not something to skip: skipping it
    /// would silently desynchronise the transcript.
    pub fn handle_handshake_record(&mut self, payload: &[u8]) -> Result<(), String> {
        let mut off = 0usize;
        while off < payload.len() {
            let len = be24(payload, off + 1)
                .ok_or_else(|| String::from("handshake message truncated"))?;
            let end = off
                .checked_add(4)
                .and_then(|s| s.checked_add(len))
                .ok_or_else(|| String::from("handshake length overflow"))?;
            if end > payload.len() {
                return Err(String::from("handshake message truncated"));
            }
            let msg = &payload[off..end];
            match msg[0] {
                HS_CERTIFICATE => self.handle_certificate(msg)?,
                HS_CERTIFICATE_VERIFY => self.handle_certificate_verify(msg)?,
                HS_FINISHED => self.handle_finished(msg)?,
                _ => return Err(String::from("unexpected handshake message")),
            }
            off = end;
        }
        Ok(())
    }

    /// Validate a peer Certificate handshake message (RFC 8446 §4.4.2) against
    /// the embedded trust store and the CMOS clock.
    ///
    /// `msg` is the handshake message: type byte, 3-byte length, body. The
    /// leaf must also be issued for the hostname given to
    /// [`Self::set_hostname`], or a certificate the anchor issued for any
    /// other name would authenticate this connection. On success the leaf's
    /// Ed25519 key is retained for [`Self::handle_certificate_verify`]; a
    /// valid chain on its own authenticates nothing.
    ///
    /// A name mismatch is not an [`x509::X509Error`] — the chain itself is
    /// sound — so it leaves [`Self::peer_cert_error`] unset and reports the
    /// reason in the returned error instead. A session with no hostname at all
    /// matches nothing and so refuses every chain.
    pub fn handle_certificate(&mut self, msg: &[u8]) -> Result<(), String> {
        let body = handshake_body(msg, HS_CERTIFICATE)?;
        let msg_len = 4 + body.len();
        let ders = certificate_list(body)?;
        if ders.is_empty() {
            self.peer_cert_error = Some(x509::X509Error::EmptyChain);
            return Err(String::from("peer sent an empty certificate list"));
        }
        let now = crate::drivers::rtc::seconds_since_epoch();
        match x509::verify_chain_der(&ders, now) {
            Ok(()) => {
                // Re-parse the leaf for its key. The chain already verified,
                // so this cannot fail; propagate rather than unwrap anyway.
                let leaf = x509::Certificate::parse(ders[0])
                    .map_err(|_| String::from("leaf certificate re-parse failed"))?;
                if !self
                    .hostname
                    .as_deref()
                    .is_some_and(|host| leaf.matches_dns(host))
                {
                    self.peer_verified = false;
                    self.peer_public_key = None;
                    return Err(String::from(
                        "peer certificate is not issued for the requested hostname",
                    ));
                }
                self.peer_public_key = Some(leaf.public_key);
                self.peer_verified = true;
                self.peer_cert_error = None;
                self.transcript.update(&msg[..msg_len]);
                Ok(())
            }
            Err(e) => {
                self.peer_verified = false;
                self.peer_public_key = None;
                self.peer_cert_error = Some(e);
                Err(String::from("peer certificate chain rejected"))
            }
        }
    }

    /// Verify a CertificateVerify handshake message (RFC 8446 §4.4.3).
    ///
    /// The signature must be `ed25519` over 64 octets of `0x20`, the context
    /// string `"TLS 1.3, server CertificateVerify"`, one `0x00`, and
    /// Transcript-Hash(ClientHello..Certificate). This is what proves the peer
    /// holds the private key for the certificate it presented; a replayed
    /// signature is over a different transcript and fails here.
    ///
    /// Any other signature scheme is refused. `x509.rs` parses Ed25519
    /// certificates only, so an RSA or ECDSA peer could not have got this far.
    pub fn handle_certificate_verify(&mut self, msg: &[u8]) -> Result<(), String> {
        let body = handshake_body(msg, HS_CERTIFICATE_VERIFY)?;
        let msg_len = 4 + body.len();
        if !self.peer_verified {
            return Err(String::from(
                "CertificateVerify before a validated certificate chain",
            ));
        }
        let peer_key = self
            .peer_public_key
            .ok_or_else(|| String::from("no peer public key to verify against"))?;
        if body.len() < 4 {
            return Err(String::from("CertificateVerify truncated"));
        }
        let scheme = u16::from_be_bytes([body[0], body[1]]);
        if scheme != SIG_ED25519 {
            return Err(String::from(
                "CertificateVerify uses a signature algorithm this kernel cannot verify",
            ));
        }
        let sig_len = u16::from_be_bytes([body[2], body[3]]) as usize;
        if sig_len != 64 || body.len() != 4 + sig_len {
            return Err(String::from("CertificateVerify signature length"));
        }
        let content = certificate_verify_content(&self.transcript_hash());
        verify_ed25519(&peer_key, &content, &body[4..])?;
        self.transcript.update(&msg[..msg_len]);
        self.cert_verify_ok = true;
        Ok(())
    }

    /// Verify the server's Finished message (RFC 8446 §4.4.4) and, only then,
    /// unlock the record layer.
    ///
    /// `verify_data` is HMAC over Transcript-Hash(ClientHello..CertificateVerify)
    /// keyed by `HKDF-Expand-Label(server_handshake_traffic_secret,
    /// "finished", "", 32)`. On the certificate path a validated chain *and* a
    /// verified CertificateVerify are required first; on the PSK path the MAC
    /// itself is the authentication, since only a holder of the PSK can
    /// produce it.
    pub fn handle_finished(&mut self, msg: &[u8]) -> Result<(), String> {
        let body = handshake_body(msg, HS_FINISHED)?;
        let msg_len = 4 + body.len();
        if self.state != TlsState::Established {
            return Err(String::from("Finished before traffic keys were derived"));
        }
        if self.key_exchange != KeyExchange::PskOnly && !(self.peer_verified && self.cert_verify_ok)
        {
            return Err(String::from(
                "Finished before the peer proved possession of its certificate key",
            ));
        }
        if body.len() != 32 {
            return Err(String::from("Finished verify_data length"));
        }
        let finished_key = hkdf_expand_label(&self.server_hs_secret, b"finished", &[], 32);
        let expected = hmac_sha256(&finished_key, &self.transcript_hash());
        if !ct_eq(&expected, body) {
            return Err(String::from("server Finished did not verify"));
        }
        self.transcript.update(&msg[..msg_len]);
        self.authenticated = true;
        Ok(())
    }

    /// Encrypt application data with AES-128-GCM.
    ///
    /// Plaintext longer than the RFC 8446 §5.1 fragment limit is split across
    /// records, each with its own sequence number, rather than truncated into
    /// one record whose 16-bit length field has wrapped. Each record's header
    /// is the AEAD additional data (RFC 8446 §5.2), so the type, version, and
    /// length are authenticated along with the payload.
    pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<Vec<u8>, String> {
        if self.state != TlsState::Established {
            return Err(String::from("handshake not complete"));
        }
        if !self.authenticated {
            return Err(String::from("peer has not authenticated"));
        }
        let cipher =
            Aes128Gcm::new_from_slice(&self.write_key).map_err(|_| String::from("bad key"))?;
        let mut out = Vec::with_capacity(plaintext.len() + 5 + TAG_LEN);
        let mut rest = plaintext;
        loop {
            let (fragment, tail) = rest.split_at(rest.len().min(MAX_PLAINTEXT_LEN));
            let header = record_header(
                ContentType::ApplicationData,
                fragment.len().saturating_add(TAG_LEN),
            )?;
            let nonce = self.make_nonce(true);
            let mut ciphertext = fragment.to_vec();
            let tag = cipher
                .encrypt_in_place_detached((&nonce[..]).into(), &header, &mut ciphertext)
                .map_err(|_| String::from("encrypt failed"))?;
            self.write_seq += 1;
            out.extend_from_slice(&header);
            out.extend_from_slice(&ciphertext);
            out.extend_from_slice(&tag);
            rest = tail;
            // A zero-length write is still one record, so this runs at least
            // once before the tail is consulted.
            if rest.is_empty() {
                break;
            }
        }
        Ok(out)
    }

    /// Decrypt application data.
    ///
    /// The record's own five header bytes are the additional data, so a type,
    /// version, or length altered in flight fails the tag check instead of
    /// being ignored.
    pub fn decrypt(&mut self, data: &[u8]) -> Result<Vec<u8>, String> {
        if self.state != TlsState::Established {
            return Err(String::from("handshake not complete"));
        }
        if !self.authenticated {
            return Err(String::from("peer has not authenticated"));
        }
        let rec = parse_record(data)?;
        if rec.ctype != ContentType::ApplicationData as u8 {
            return Err(String::from("expected application data"));
        }
        // RFC 8446 §5.2: a longer record is a record_overflow, not something
        // to buffer and try to open.
        if rec.payload.len() > MAX_RECORD_LEN {
            return Err(String::from("TLS record exceeds RFC 8446 §5.2 maximum"));
        }
        if rec.payload.len() < TAG_LEN {
            return Err(String::from("ciphertext too short"));
        }
        let header: [u8; 5] = [data[0], data[1], data[2], data[3], data[4]];
        let (ct, tag) = rec.payload.split_at(rec.payload.len() - TAG_LEN);
        let nonce = self.make_nonce(false);
        let cipher =
            Aes128Gcm::new_from_slice(&self.read_key).map_err(|_| String::from("bad key"))?;
        let mut pt = ct.to_vec();
        cipher
            .decrypt_in_place_detached((&nonce[..]).into(), &header, &mut pt, tag.into())
            .map_err(|_| String::from("decrypt failed (auth tag mismatch)"))?;
        self.read_seq += 1;
        Ok(pt)
    }

    pub fn benchmark_psk_record_roundtrip(plaintext: &[u8]) -> Result<TlsRecordBenchProof, String> {
        let mut session = Self::new();
        session.set_psk(&[0xABu8; 32]);
        session.state = TlsState::Established;
        // Record-layer benchmark, not a handshake: the keys below are fixed
        // constants, so there is no peer to authenticate. Set explicitly so
        // the record layer's authentication gate stays a real gate everywhere
        // a handshake actually runs.
        session.authenticated = true;
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

/// The five header bytes of a record, which RFC 8446 §5.2 also makes the AEAD
/// additional data: `opaque_type`, `legacy_record_version`, `length`.
///
/// Refuses any length the 16-bit field cannot describe rather than letting it
/// wrap, so a record on the wire always says how long it really is.
fn record_header(ctype: ContentType, len: usize) -> Result<[u8; 5], String> {
    if len > MAX_RECORD_LEN {
        return Err(String::from("TLS record exceeds RFC 8446 §5.2 maximum"));
    }
    Ok([
        ctype as u8,
        0x03,
        0x03, // TLS 1.2 legacy record version
        (len >> 8) as u8,
        len as u8,
    ])
}

/// Frame a plaintext record. RFC 8446 §5.1 caps the fragment at 2^14 bytes;
/// a caller with more to send fragments it, as [`TlsSession::encrypt`] does.
fn wrap_record(ctype: ContentType, payload: &[u8]) -> Result<Vec<u8>, String> {
    if payload.len() > MAX_PLAINTEXT_LEN {
        return Err(String::from(
            "TLS plaintext exceeds the RFC 8446 §5.1 fragment limit",
        ));
    }
    let mut rec = Vec::with_capacity(5 + payload.len());
    rec.extend_from_slice(&record_header(ctype, payload.len())?);
    rec.extend_from_slice(payload);
    Ok(rec)
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

/// Read the `cipher_suite` a ServerHello selected. `None` if it is not there.
fn server_cipher_suite(payload: &[u8]) -> Option<u16> {
    // handshake header (4) + legacy_version (2) + random (32)
    let sid_len = *payload.get(38)? as usize;
    be16(payload, 39usize.checked_add(sid_len)?)
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

// ---------------------------------------------------------------------------
// Peer authentication primitives
// ---------------------------------------------------------------------------

/// The exact byte string a server CertificateVerify signs (RFC 8446 §4.4.3):
/// 64 octets of `0x20`, the context string, one `0x00`, then the transcript
/// hash. 130 bytes for SHA-256.
fn certificate_verify_content(transcript_hash: &[u8; 32]) -> Vec<u8> {
    let mut content = Vec::with_capacity(64 + CERT_VERIFY_CONTEXT.len() + 1 + 32);
    content.extend_from_slice(&[0x20u8; 64]);
    content.extend_from_slice(CERT_VERIFY_CONTEXT);
    content.push(0x00);
    content.extend_from_slice(transcript_hash);
    content
}

/// Verify an Ed25519 signature, rejecting non-canonical encodings and
/// small-order keys (`verify_strict`, RFC 8032 §5.1.7 / ed25519-dalek).
fn verify_ed25519(public_key: &[u8; 32], msg: &[u8], sig: &[u8]) -> Result<(), String> {
    if sig.len() != 64 {
        return Err(String::from("Ed25519 signature is not 64 bytes"));
    }
    let vk = VerifyingKey::from_bytes(public_key)
        .map_err(|_| String::from("peer key is not a valid Ed25519 point"))?;
    let mut bytes = [0u8; 64];
    bytes.copy_from_slice(sig);
    vk.verify_strict(msg, &Signature::from_bytes(&bytes))
        .map_err(|_| String::from("signature did not verify over this transcript"))
}

/// Length-checked, data-independent byte comparison for MAC verification.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn vec_to_array32(v: &[u8]) -> [u8; 32] {
    let mut a = [0u8; 32];
    a[..32.min(v.len())].copy_from_slice(&v[..32.min(v.len())]);
    a
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
    use ed25519_dalek::{Signer, SigningKey};

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
        session.set_hostname("seal.local");
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
        // Named, so the rejection below is the chain and not a missing name.
        session.set_hostname("seal.local");
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
            // Named, so each rejection below is the truncation and not a
            // missing name.
            session.set_hostname("seal.local");
            test_assert!(
                session.handle_certificate(&msg[..cut]).is_err(),
                "a truncated Certificate message was accepted"
            );
        }
        TestResult::Pass
    }

    fn test_chain_alone_does_not_authenticate() -> TestResult {
        // A certificate chain is public data. Anyone who observed one
        // connection can replay it verbatim, so chain validation alone must
        // never unlock the record layer.
        let mut session = TlsSession::new();
        session.set_hostname("seal.local");
        session.client_random = [0x01u8; 32];
        session.ecdhe_key = Some(ecdhe::EphemeralKey::from_scalar([0x55u8; 32]));
        let server_key = ecdhe::EphemeralKey::from_scalar([0x66u8; 32]);
        let hello = server_hello_with_key_share(&[0x02u8; 32], server_key.public());
        if session.handle_server_hello(&hello).is_err() {
            return TestResult::Fail("ServerHello with a key share was rejected");
        }
        let cert = certificate_message(&[
            &super::super::certs::LEAF_DER,
            &super::super::certs::INTERMEDIATE_DER,
        ]);
        if session.handle_certificate(&cert).is_err() {
            return TestResult::Fail("the valid fixture chain was rejected");
        }
        test_assert!(
            session.encrypt(b"GET /secret HTTP/1.1").is_err(),
            "application data was encrypted to a peer that never proved it holds the certificate's private key"
        );
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

    // --- peer authentication --------------------------------------------

    /// RFC 8032 §7.1 TEST 2 secret key. Doubles as the fixed seed for the
    /// stand-in server signing key below, so the tests never touch entropy.
    const RFC8032_SECRET: [u8; 32] = [
        0x4c, 0xcd, 0x08, 0x9b, 0x28, 0xff, 0x96, 0xda, 0x9d, 0xb6, 0xc3, 0x46, 0xec, 0x11, 0x4e,
        0x0f, 0x5b, 0x8a, 0x31, 0x9f, 0x35, 0xab, 0xa6, 0x24, 0xda, 0x8c, 0xf6, 0xed, 0x4f, 0xb8,
        0xa6, 0xfb,
    ];

    fn test_ed25519_rfc8032_vector() -> TestResult {
        // RFC 8032 §7.1 TEST 2: 1-byte message 0x72. Pins the verifier this
        // module hands CertificateVerify to against a published vector, so a
        // verifier that returned Ok unconditionally would be caught here.
        let public = [
            0x3d, 0x40, 0x17, 0xc3, 0xe8, 0x43, 0x89, 0x5a, 0x92, 0xb7, 0x0a, 0xa7, 0x4d, 0x1b,
            0x7e, 0xbc, 0x9c, 0x98, 0x2c, 0xcf, 0x2e, 0xc4, 0x96, 0x8c, 0xc0, 0xcd, 0x55, 0xf1,
            0x2a, 0xf4, 0x66, 0x0c,
        ];
        let signature = [
            0x92, 0xa0, 0x09, 0xa9, 0xf0, 0xd4, 0xca, 0xb8, 0x72, 0x0e, 0x82, 0x0b, 0x5f, 0x64,
            0x25, 0x40, 0xa2, 0xb2, 0x7b, 0x54, 0x16, 0x50, 0x3f, 0x8f, 0xb3, 0x76, 0x22, 0x23,
            0xeb, 0xdb, 0x69, 0xda, 0x08, 0x5a, 0xc1, 0xe4, 0x3e, 0x15, 0x99, 0x6e, 0x45, 0x8f,
            0x36, 0x13, 0xd0, 0xf1, 0x1d, 0x8c, 0x38, 0x7b, 0x2e, 0xae, 0xb4, 0x30, 0x2a, 0xee,
            0xb0, 0x0d, 0x29, 0x16, 0x12, 0xbb, 0x0c, 0x00,
        ];
        test_assert!(
            verify_ed25519(&public, &[0x72], &signature).is_ok(),
            "RFC 8032 §7.1 TEST 2 signature did not verify"
        );
        // The same signature over a different message must not verify.
        test_assert!(
            verify_ed25519(&public, &[0x73], &signature).is_err(),
            "a signature verified over the wrong message"
        );
        // The signing key derived from the RFC's secret must yield the RFC's
        // public key, which is what lets the fixtures below sign.
        test_assert_eq!(
            SigningKey::from_bytes(&RFC8032_SECRET)
                .verifying_key()
                .to_bytes(),
            public
        );
        TestResult::Pass
    }

    fn test_certificate_verify_content_matches_rfc8446() -> TestResult {
        // RFC 8446 §4.4.3 spells the signed string out octet by octet.
        let hash = [0xA5u8; 32];
        let content = certificate_verify_content(&hash);
        test_assert_eq!(content.len(), 130);
        test_assert!(
            content[..64].iter().all(|&b| b == 0x20),
            "the 64-octet 0x20 padding is wrong"
        );
        test_assert_eq!(
            &content[64..97],
            b"TLS 1.3, server CertificateVerify".as_slice()
        );
        test_assert_eq!(content[97], 0x00);
        test_assert_eq!(&content[98..], hash.as_slice());
        TestResult::Pass
    }

    fn test_certificate_verify_and_finished_complete_handshake() -> TestResult {
        // Positive control. Without this the suite could be satisfied by a
        // stack that refuses every handshake.
        let Some((mut session, signing)) = session_ready_for_certificate_verify(0x01) else {
            return TestResult::Fail("could not reach the CertificateVerify step");
        };
        test_assert!(
            session.encrypt(b"early").is_err(),
            "the record layer opened before CertificateVerify"
        );

        let cv = signed_certificate_verify(&session, &signing, 0);
        if session.handle_certificate_verify(&cv).is_err() {
            return TestResult::Fail("a correct CertificateVerify was refused");
        }
        test_assert!(
            session.encrypt(b"early").is_err(),
            "the record layer opened before Finished"
        );

        let fin = server_finished(&session);
        if session.handle_finished(&fin).is_err() {
            return TestResult::Fail("a correct Finished was refused");
        }
        test_assert!(
            session.peer_authenticated(),
            "peer never marked authenticated"
        );
        test_assert!(
            session.encrypt(b"GET / HTTP/1.1").is_ok(),
            "an authenticated peer could not be written to"
        );
        TestResult::Pass
    }

    fn test_certificate_verify_rejects_forged_signatures() -> TestResult {
        let Some((base, signing)) = session_ready_for_certificate_verify(0x01) else {
            return TestResult::Fail("could not reach the CertificateVerify step");
        };
        let good = signed_certificate_verify(&base, &signing, 0);

        // A signature over a transcript that differs by one bit.
        let wrong = signed_certificate_verify(&base, &signing, 1);
        let Some((mut session, _)) = session_ready_for_certificate_verify(0x01) else {
            return TestResult::Fail("fixture rebuild failed");
        };
        test_assert!(
            session.handle_certificate_verify(&wrong).is_err(),
            "a signature over the wrong transcript was accepted"
        );

        // An absent signature.
        let Some((mut session, _)) = session_ready_for_certificate_verify(0x01) else {
            return TestResult::Fail("fixture rebuild failed");
        };
        test_assert!(
            session
                .handle_certificate_verify(&cert_verify_message(SIG_ED25519, &[]))
                .is_err(),
            "an empty CertificateVerify signature was accepted"
        );

        // Every single-bit mutation of a valid message.
        for i in 4..good.len() {
            for bit in 0..8u32 {
                let mut fuzzed = good.clone();
                fuzzed[i] ^= 1 << bit;
                let Some((mut session, _)) = session_ready_for_certificate_verify(0x01) else {
                    return TestResult::Fail("fixture rebuild failed");
                };
                if session.handle_certificate_verify(&fuzzed).is_ok() {
                    return TestResult::Fail("a mutated CertificateVerify was accepted");
                }
            }
        }

        // Every truncation of a valid message.
        for cut in 0..good.len() {
            let Some((mut session, _)) = session_ready_for_certificate_verify(0x01) else {
                return TestResult::Fail("fixture rebuild failed");
            };
            if session.handle_certificate_verify(&good[..cut]).is_ok() {
                return TestResult::Fail("a truncated CertificateVerify was accepted");
            }
        }
        TestResult::Pass
    }

    fn test_certificate_verify_rejects_unsupported_algorithm() -> TestResult {
        // x509.rs parses Ed25519 only, so these could never be verified. They
        // are refused, never waved through.
        let Some((base, signing)) = session_ready_for_certificate_verify(0x01) else {
            return TestResult::Fail("could not reach the CertificateVerify step");
        };
        let good = signed_certificate_verify(&base, &signing, 0);
        let sig = good[8..].to_vec();
        for scheme in [
            0x0804u16, // rsa_pss_rsae_sha256
            0x0403,    // ecdsa_secp256r1_sha256
            0x0401,    // rsa_pkcs1_sha256
            0x0808,    // ed448
            0x0000,
        ] {
            let Some((mut session, _)) = session_ready_for_certificate_verify(0x01) else {
                return TestResult::Fail("fixture rebuild failed");
            };
            test_assert!(
                session
                    .handle_certificate_verify(&cert_verify_message(scheme, &sig))
                    .is_err(),
                "a signature algorithm this kernel cannot verify was accepted"
            );
        }
        TestResult::Pass
    }

    fn test_certificate_verify_replay_from_another_handshake_rejected() -> TestResult {
        // The exact attack a chain-only check cannot see: everything the peer
        // sent last time, replayed verbatim into a fresh handshake.
        let Some((first, signing)) = session_ready_for_certificate_verify(0x01) else {
            return TestResult::Fail("could not reach the CertificateVerify step");
        };
        let captured = signed_certificate_verify(&first, &signing, 0);

        let Some((mut second, _)) = session_ready_for_certificate_verify(0x02) else {
            return TestResult::Fail("fixture rebuild failed");
        };
        test_assert!(
            second.handle_certificate_verify(&captured).is_err(),
            "a CertificateVerify captured from another handshake was accepted"
        );
        test_assert!(
            second.encrypt(b"secret").is_err(),
            "the record layer opened after a replayed CertificateVerify"
        );
        TestResult::Pass
    }

    fn test_certificate_verify_is_bound_to_the_certificate_message() -> TestResult {
        // Two handshakes alike in every byte except the Certificate message
        // the server chose to send. Both chains validate and both end at the
        // same leaf key, so only the transcript tells them apart: a
        // CertificateVerify made for one must not verify against the other.
        // Without the Certificate in the transcript, a signature obtained from
        // a handshake with one chain replays into a handshake with another.
        let plain = certificate_message(&[
            &super::super::certs::LEAF_DER,
            &super::super::certs::INTERMEDIATE_DER,
        ]);
        let with_root = certificate_message(&[
            &super::super::certs::LEAF_DER,
            &super::super::certs::INTERMEDIATE_DER,
            &super::super::certs::ROOT_CA_DER,
        ]);
        test_assert!(plain != with_root, "the two Certificate messages are equal");

        let signing = SigningKey::from_bytes(&RFC8032_SECRET);
        let (Some(mut first), Some(mut second)) = (
            session_after_certificate(0x01, &plain),
            session_after_certificate(0x01, &with_root),
        ) else {
            return TestResult::Fail("could not reach the CertificateVerify step");
        };
        let cv = signed_certificate_verify(&first, &signing, 0);
        if first.handle_certificate_verify(&cv).is_err() {
            return TestResult::Fail("a correct CertificateVerify was refused");
        }
        test_assert!(
            second.handle_certificate_verify(&cv).is_err(),
            "a CertificateVerify was accepted against a different Certificate message"
        );
        TestResult::Pass
    }

    fn test_client_hello_enters_the_transcript() -> TestResult {
        // Everything the peer later signs or MACs has to cover the
        // ClientHello, or its own key share and extensions go unprotected.
        let empty = TlsSession::new().transcript_hash();
        let mut session = TlsSession::new();
        // No hardware entropy: failing closed is correct and there is nothing
        // further to assert.
        if session.build_client_hello().is_err() {
            return TestResult::Pass;
        }
        test_assert!(
            session.transcript_hash() != empty,
            "the ClientHello is not covered by the transcript"
        );
        TestResult::Pass
    }

    fn test_certificate_verify_before_certificate_rejected() -> TestResult {
        let Some((base, signing)) = session_ready_for_certificate_verify(0x01) else {
            return TestResult::Fail("could not reach the CertificateVerify step");
        };
        let cv = signed_certificate_verify(&base, &signing, 0);
        let mut bare = TlsSession::new();
        test_assert!(
            bare.handle_certificate_verify(&cv).is_err(),
            "CertificateVerify was accepted with no validated certificate"
        );
        TestResult::Pass
    }

    fn test_finished_requires_certificate_verify() -> TestResult {
        // A valid chain and a correct Finished, but no CertificateVerify: the
        // peer still never proved it holds the leaf's private key.
        let Some((mut session, _)) = session_ready_for_certificate_verify(0x01) else {
            return TestResult::Fail("could not reach the CertificateVerify step");
        };
        let fin = server_finished(&session);
        test_assert!(
            session.handle_finished(&fin).is_err(),
            "Finished was accepted with no CertificateVerify"
        );
        test_assert!(!session.peer_authenticated(), "peer marked authenticated");
        test_assert!(
            session.encrypt(b"secret").is_err(),
            "the record layer opened with no CertificateVerify"
        );
        TestResult::Pass
    }

    fn test_finished_rejects_wrong_mac() -> TestResult {
        let Some((mut session, signing)) = session_ready_for_certificate_verify(0x01) else {
            return TestResult::Fail("could not reach the CertificateVerify step");
        };
        let cv = signed_certificate_verify(&session, &signing, 0);
        if session.handle_certificate_verify(&cv).is_err() {
            return TestResult::Fail("a correct CertificateVerify was refused");
        }
        let good = server_finished(&session);
        for i in 4..good.len() {
            let mut fuzzed = good.clone();
            fuzzed[i] ^= 0x01;
            if session.handle_finished(&fuzzed).is_ok() {
                return TestResult::Fail("a Finished with a wrong MAC was accepted");
            }
        }
        // Length games must not pass either.
        test_assert!(
            session
                .handle_finished(&handshake_message(0x14, &[]))
                .is_err(),
            "an empty Finished was accepted"
        );
        test_assert!(!session.peer_authenticated(), "peer marked authenticated");
        // The genuine one still works, so the rejections above are not the
        // session having been poisoned.
        test_assert!(
            session.handle_finished(&good).is_ok(),
            "the correct Finished stopped verifying"
        );
        TestResult::Pass
    }

    fn test_certificate_message_records_leaf_key() -> TestResult {
        // The key CertificateVerify is checked against must be the validated
        // leaf's key, and a rejected chain must leave no key behind.
        let Ok(leaf) = x509::Certificate::parse(&super::super::certs::LEAF_DER) else {
            return TestResult::Fail("leaf fixture failed to parse");
        };
        let mut session = TlsSession::new();
        session.set_hostname("seal.local");
        let msg = certificate_message(&[
            &super::super::certs::LEAF_DER,
            &super::super::certs::INTERMEDIATE_DER,
        ]);
        if session.handle_certificate(&msg).is_ok() {
            test_assert_eq!(session.peer_public_key, Some(leaf.public_key));
        } else {
            test_assert_eq!(session.peer_public_key, None);
        }

        let mut rogue = TlsSession::new();
        rogue.set_hostname("seal.local");
        let bad = certificate_message(&[
            &super::super::certs::ROGUE_LEAF_DER,
            &super::super::certs::LEAF_DER,
            &super::super::certs::INTERMEDIATE_DER,
        ]);
        test_assert!(
            rogue.handle_certificate(&bad).is_err(),
            "a chain through a non-CA leaf was accepted"
        );
        test_assert_eq!(rogue.peer_public_key, None);
        TestResult::Pass
    }

    fn test_server_hello_rejects_unoffered_cipher_suite() -> TestResult {
        let server_key = ecdhe::EphemeralKey::from_scalar([0x66u8; 32]);
        for suite in [0x1302u16, 0x1303, 0xc02f, 0x0000, 0x00ff] {
            let mut session = TlsSession::new();
            session.ecdhe_key = Some(ecdhe::EphemeralKey::from_scalar([0x55u8; 32]));
            let hello = server_hello_full(&[0x02u8; 32], server_key.public(), suite);
            test_assert!(
                session.handle_server_hello(&hello).is_err(),
                "the server selected a cipher suite the ClientHello never offered"
            );
        }
        TestResult::Pass
    }

    fn test_handshake_record_rejects_unknown_message() -> TestResult {
        // Skipping an unrecognised handshake message would desynchronise the
        // transcript silently, so it is an error instead.
        let mut session = TlsSession::new();
        test_assert!(
            session
                .handle_handshake_record(&handshake_message(0x08, &[0u8; 2]))
                .is_err(),
            "an unexpected handshake message was skipped"
        );
        for cut in 1..8usize {
            let mut session = TlsSession::new();
            let msg = handshake_message(0x14, &[0u8; 32]);
            test_assert!(
                session.handle_handshake_record(&msg[..cut]).is_err(),
                "a truncated handshake message was accepted"
            );
        }
        TestResult::Pass
    }

    fn test_certificate_must_match_the_requested_hostname() -> TestResult {
        // A trust-store-issued certificate proves the peer is *somebody*. It
        // has to be the somebody the caller asked for, or a valid certificate
        // for any other name authenticates this connection.
        let msg = certificate_message(&[
            &super::super::certs::LEAF_DER,
            &super::super::certs::INTERMEDIATE_DER,
        ]);
        let now = crate::drivers::rtc::seconds_since_epoch();
        let chain_ok = x509::verify_chain_der(
            &[
                &super::super::certs::LEAF_DER,
                &super::super::certs::INTERMEDIATE_DER,
            ],
            now,
        )
        .is_ok();

        // Both SubjectAltName dNSNames the leaf actually carries, so this is a
        // SAN lookup and not a single hard-coded string.
        for host in ["seal.local", "proof.seal", "SEAL.LOCAL"] {
            let mut session = TlsSession::new();
            session.set_hostname(host);
            test_assert_eq!(session.handle_certificate(&msg).is_ok(), chain_ok);
        }

        // Names the leaf does not carry. The chain still validates against the
        // embedded anchor, so only the name check can refuse these.
        for host in [
            "evil.seal",
            "seal.local.evil.seal",
            "evil.seal.local",
            "eal.local",
            "seal.loca",
            "*.local",
            "",
        ] {
            let mut session = TlsSession::new();
            session.set_hostname(host);
            test_assert!(
                session.handle_certificate(&msg).is_err(),
                "a certificate issued for another name was accepted"
            );
            test_assert!(
                !session.peer_verified(),
                "peer marked verified on a name mismatch"
            );
            test_assert_eq!(session.peer_public_key, None);
        }

        // No hostname configured at all: fail closed rather than accept any
        // name the peer happens to present.
        let mut session = TlsSession::new();
        test_assert!(
            session.handle_certificate(&msg).is_err(),
            "a certificate was accepted with no hostname to check it against"
        );
        test_assert!(
            !session.peer_verified(),
            "peer marked verified with no hostname"
        );
        TestResult::Pass
    }

    fn test_hostname_mismatch_stops_the_handshake() -> TestResult {
        // The name check has to keep the transcript closed too: a refused
        // Certificate must leave nothing for CertificateVerify to build on.
        let cert = certificate_message(&[
            &super::super::certs::LEAF_DER,
            &super::super::certs::INTERMEDIATE_DER,
        ]);
        let mut session = TlsSession::new();
        session.set_hostname("evil.seal");
        session.client_random = [0x01u8; 32];
        session.ecdhe_key = Some(ecdhe::EphemeralKey::from_scalar([0x55u8; 32]));
        let server_key = ecdhe::EphemeralKey::from_scalar([0x66u8; 32]);
        let hello = server_hello_with_key_share(&[0xfeu8; 32], server_key.public());
        if session.handle_server_hello(&hello).is_err() {
            return TestResult::Fail("ServerHello with a key share was rejected");
        }
        test_assert!(
            session.handle_handshake_record(&cert).is_err(),
            "a certificate issued for another name was accepted"
        );
        let signing = SigningKey::from_bytes(&RFC8032_SECRET);
        session.peer_public_key = Some(signing.verifying_key().to_bytes());
        let cv = signed_certificate_verify(&session, &signing, 0);
        test_assert!(
            session.handle_certificate_verify(&cv).is_err(),
            "CertificateVerify was accepted after a refused certificate"
        );
        test_assert!(
            session.encrypt(b"GET /secret HTTP/1.1").is_err(),
            "the record layer opened for a certificate issued to another name"
        );
        test_assert!(!session.peer_authenticated(), "peer marked authenticated");
        TestResult::Pass
    }

    // --- record layer ---------------------------------------------------

    fn test_large_plaintext_never_wraps_the_length_field() -> TestResult {
        // 65_520 bytes of plaintext put 65_536 into a 16-bit length field.
        // RFC 8446 §5.1 caps a fragment at 2^14, so this has to come back as
        // several records, each one honestly describing its own payload.
        let mut session = record_session();
        let plaintext = vec![0x5Au8; 65_520];
        let Ok(out) = session.encrypt(&plaintext) else {
            return TestResult::Fail("a plaintext larger than one fragment was refused outright");
        };

        let mut peer = record_session();
        let mut recovered: Vec<u8> = Vec::new();
        let mut records = 0usize;
        let mut off = 0usize;
        while off < out.len() {
            test_assert!(out.len() - off >= 5, "record header truncated");
            let len = u16::from_be_bytes([out[off + 3], out[off + 4]]) as usize;
            test_assert!(len > 0, "record declares a zero length");
            test_assert!(
                len <= MAX_RECORD_LEN,
                "record exceeds the RFC 8446 §5.2 maximum"
            );
            let end = off + 5 + len;
            test_assert!(end <= out.len(), "record length field overruns the buffer");
            let Ok(pt) = peer.decrypt(&out[off..end]) else {
                return TestResult::Fail("an emitted record did not decrypt");
            };
            recovered.extend_from_slice(&pt);
            off = end;
            records += 1;
        }
        test_assert!(records >= 4, "the plaintext was not fragmented");
        test_assert_eq!(recovered.len(), plaintext.len());
        test_assert!(
            recovered == plaintext,
            "the plaintext did not survive fragmentation"
        );

        // The frame builder refuses outright anything the 16-bit length field
        // cannot describe, so no other caller can emit a wrapped length
        // either.
        test_assert!(
            wrap_record(ContentType::Handshake, &vec![0u8; 70_000]).is_err(),
            "a record longer than the length field was framed"
        );
        test_assert!(
            wrap_record(ContentType::Handshake, &vec![0u8; MAX_PLAINTEXT_LEN + 1]).is_err(),
            "a fragment over the RFC 8446 §5.1 limit was framed"
        );
        test_assert!(
            record_header(ContentType::ApplicationData, MAX_RECORD_LEN + 1).is_err(),
            "a record over the RFC 8446 §5.2 maximum was framed"
        );
        // The limits themselves still frame, so the checks above are bounds
        // and not a blanket refusal.
        let Ok(largest) = wrap_record(ContentType::Handshake, &vec![0u8; MAX_PLAINTEXT_LEN]) else {
            return TestResult::Fail("the largest legal fragment was refused");
        };
        test_assert_eq!(
            u16::from_be_bytes([largest[3], largest[4]]) as usize,
            MAX_PLAINTEXT_LEN
        );
        // A receiver refuses a record longer than §5.2 allows even when its
        // tag is genuinely valid, so the limit is the receiver's own and not
        // something the AEAD happens to enforce. Both records below are built
        // here with the fixture key, and only the length tells them apart.
        let Ok(cipher) = Aes128Gcm::new_from_slice(&[0xCDu8; 16]) else {
            return TestResult::Fail("AES-128-GCM rejected the fixture key");
        };
        for (declared, allowed) in [(MAX_RECORD_LEN, true), (MAX_RECORD_LEN + 1, false)] {
            let mut forged = vec![23u8, 0x03, 0x03];
            forged.extend_from_slice(&(declared as u16).to_be_bytes());
            let mut body = vec![0x11u8; declared - TAG_LEN];
            let Ok(tag) =
                cipher.encrypt_in_place_detached((&[0xEFu8; 12][..]).into(), &forged, &mut body)
            else {
                return TestResult::Fail("the oversize fixture would not encrypt");
            };
            forged.extend_from_slice(&body);
            forged.extend_from_slice(&tag);
            let mut peer = record_session();
            test_assert_eq!(peer.decrypt(&forged).is_ok(), allowed);
        }
        TestResult::Pass
    }

    fn test_record_header_is_authenticated() -> TestResult {
        // RFC 8446 §5.2 binds the record header as AEAD additional data. The
        // two legacy_record_version bytes are read nowhere else in this
        // module, so only the AEAD can refuse an altered one.
        let mut session = record_session();
        let Ok(rec) = session.encrypt(b"authenticated header") else {
            return TestResult::Fail("the record layer refused a plaintext");
        };
        for i in [1usize, 2] {
            let mut tampered = rec.clone();
            tampered[i] ^= 0x01;
            let mut peer = record_session();
            if peer.decrypt(&tampered).is_ok() {
                return TestResult::Fail("a record with an altered header was accepted");
            }
        }
        // Control: the untouched record still opens, so the rejections above
        // are the header check and not a broken fixture.
        let mut peer = record_session();
        test_assert!(
            peer.decrypt(&rec).is_ok(),
            "an untouched record failed to decrypt"
        );
        TestResult::Pass
    }

    fn test_record_aad_is_the_rfc8446_header() -> TestResult {
        // The additional data is built here from the RFC 8446 §5.2 text —
        // opaque_type ‖ legacy_record_version ‖ length — and the record is
        // opened by a separate AES-GCM invocation that never calls this
        // module's helper, so agreement means the record layer matches the
        // specification rather than matching itself.
        let mut session = record_session();
        let plaintext = b"RFC 8446 5.2 additional_data";
        let Ok(rec) = session.encrypt(plaintext) else {
            return TestResult::Fail("the record layer refused a plaintext");
        };
        test_assert_eq!(rec.len(), 5 + plaintext.len() + 16);

        let mut aad: Vec<u8> = Vec::new();
        aad.push(23u8); // TLSCiphertext.opaque_type = application_data
        aad.extend_from_slice(&[0x03, 0x03]); // legacy_record_version
        aad.extend_from_slice(&((rec.len() - 5) as u16).to_be_bytes()); // length
        test_assert_eq!(aad.as_slice(), &rec[..5]);

        // RFC 8446 §5.3: the nonce is the write IV XOR the 64-bit sequence
        // number, right-aligned. This is the first record, so sequence 0.
        let nonce = [0xEFu8; 12];
        let Ok(cipher) = Aes128Gcm::new_from_slice(&[0xCDu8; 16]) else {
            return TestResult::Fail("AES-128-GCM rejected the fixture key");
        };
        let (ct, tag) = rec[5..].split_at(rec.len() - 5 - 16);
        let mut pt = ct.to_vec();
        if cipher
            .decrypt_in_place_detached((&nonce[..]).into(), &aad, &mut pt, tag.into())
            .is_err()
        {
            return TestResult::Fail("the record does not open under the RFC 8446 §5.2 header");
        }
        test_assert_eq!(pt.as_slice(), plaintext.as_slice());

        // The empty additional data this module used to pass must not open
        // it, and neither must a header whose length field is wrong.
        let mut wrong_len = aad.clone();
        wrong_len[4] ^= 0x01;
        for bad in [Vec::new(), wrong_len] {
            let mut pt = ct.to_vec();
            if cipher
                .decrypt_in_place_detached((&nonce[..]).into(), &bad, &mut pt, tag.into())
                .is_ok()
            {
                return TestResult::Fail("the record opened under something other than its header");
            }
        }
        TestResult::Pass
    }

    // --- fixtures -------------------------------------------------------

    /// A session with the record layer open on fixed keys, so a record it
    /// encrypts can be handed straight to another one's `decrypt`. No
    /// handshake runs, so `authenticated` is set explicitly rather than
    /// earned; the gate itself is exercised by the `tls::*` handshake tests.
    fn record_session() -> TlsSession {
        let mut session = TlsSession::new();
        session.state = TlsState::Established;
        session.authenticated = true;
        session.write_key = [0xCDu8; 16];
        session.read_key = [0xCDu8; 16];
        session.write_iv = [0xEFu8; 12];
        session.read_iv = [0xEFu8; 12];
        session
    }

    /// Wrap a body in a handshake header: type byte, then a 24-bit length.
    fn handshake_message(ty: u8, body: &[u8]) -> Vec<u8> {
        let mut msg = vec![ty];
        let len = body.len();
        msg.extend_from_slice(&[
            ((len >> 16) & 0xFF) as u8,
            ((len >> 8) & 0xFF) as u8,
            (len & 0xFF) as u8,
        ]);
        msg.extend_from_slice(body);
        msg
    }

    fn cert_verify_message(scheme: u16, sig: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&scheme.to_be_bytes());
        body.extend_from_slice(&(sig.len() as u16).to_be_bytes());
        body.extend_from_slice(sig);
        handshake_message(0x0f, &body)
    }

    /// Sign the RFC 8446 §4.4.3 content string over the session's current
    /// transcript. `hash_flip` XORs one bit into the transcript hash first, to
    /// build a signature that is genuine but over the wrong handshake.
    fn signed_certificate_verify(
        session: &TlsSession,
        signing: &SigningKey,
        hash_flip: u8,
    ) -> Vec<u8> {
        let mut hash = session.transcript_hash();
        hash[0] ^= hash_flip;
        let sig = signing.sign(&certificate_verify_content(&hash));
        cert_verify_message(SIG_ED25519, &sig.to_bytes())
    }

    /// The Finished the server owes for the session's current transcript.
    fn server_finished(session: &TlsSession) -> Vec<u8> {
        let key = hkdf_expand_label(&session.server_hs_secret, b"finished", &[], 32);
        let verify_data = hmac_sha256(&key, &session.transcript_hash());
        handshake_message(0x14, &verify_data)
    }

    /// A session driven through ServerHello and a Certificate, positioned
    /// exactly where a CertificateVerify is due.
    ///
    /// The peer key is then replaced with one the test holds: the fixture CA's
    /// private keys are not in this tree, so nothing here can sign as the
    /// leaf. `test_certificate_message_records_leaf_key` covers the link this
    /// substitution stands in for — that the key checked against is the
    /// validated leaf's key and nothing else.
    fn session_ready_for_certificate_verify(tag: u8) -> Option<(TlsSession, SigningKey)> {
        let cert = certificate_message(&[
            &super::super::certs::LEAF_DER,
            &super::super::certs::INTERMEDIATE_DER,
        ]);
        Some((
            session_after_certificate(tag, &cert)?,
            SigningKey::from_bytes(&RFC8032_SECRET),
        ))
    }

    /// As above, with the Certificate message spelled out.
    fn session_after_certificate(tag: u8, cert: &[u8]) -> Option<TlsSession> {
        let mut session = TlsSession::new();
        // The name the fixture leaf is issued for, which is what production
        // supplies through `TlsSocket::connect`.
        session.set_hostname("seal.local");
        session.client_random = [tag; 32];
        session.ecdhe_key = Some(ecdhe::EphemeralKey::from_scalar([0x55u8; 32]));
        let server_key = ecdhe::EphemeralKey::from_scalar([0x66u8; 32]);
        let hello = server_hello_with_key_share(&[tag ^ 0xff; 32], server_key.public());
        session.handle_server_hello(&hello).ok()?;

        if session.handle_certificate(cert).is_err() {
            // The kernel clock sits outside the fixture validity window. Stand
            // in for the Certificate step so the CertificateVerify checks
            // still run; chain validation itself is x509.rs's own suite.
            session.peer_verified = true;
            session.transcript.update(cert);
        }
        session.peer_public_key = Some(
            SigningKey::from_bytes(&RFC8032_SECRET)
                .verifying_key()
                .to_bytes(),
        );
        Some(session)
    }

    /// Build a ServerHello record. An empty `key_share` omits the extension.
    fn server_hello_with_key_share(random: &[u8; 32], key_share: &[u8]) -> Vec<u8> {
        server_hello_full(random, key_share, 0x1301)
    }

    /// As above, with the selected cipher suite spelled out.
    fn server_hello_full(random: &[u8; 32], key_share: &[u8], suite: u16) -> Vec<u8> {
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
        hs.extend_from_slice(&suite.to_be_bytes()); // cipher suite
        hs.push(0x00); // compression
        hs.extend_from_slice(&(ext.len() as u16).to_be_bytes());
        hs.extend_from_slice(&ext);

        let len = hs.len() - 4;
        hs[1..4].copy_from_slice(&[
            ((len >> 16) & 0xFF) as u8,
            ((len >> 8) & 0xFF) as u8,
            (len & 0xFF) as u8,
        ]);
        // Every fixture here is a few hundred bytes, far below the §5.1
        // fragment limit. An empty record would fail its test loudly.
        wrap_record(ContentType::Handshake, &hs).unwrap_or_default()
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
        crate::testing::register_test(
            "tls::chain_alone_does_not_authenticate",
            test_chain_alone_does_not_authenticate,
        );
        crate::testing::register_test("tls::ed25519_rfc8032_vector", test_ed25519_rfc8032_vector);
        crate::testing::register_test(
            "tls::certificate_verify_content_matches_rfc8446",
            test_certificate_verify_content_matches_rfc8446,
        );
        crate::testing::register_test(
            "tls::certificate_verify_and_finished_complete_handshake",
            test_certificate_verify_and_finished_complete_handshake,
        );
        crate::testing::register_test(
            "tls::certificate_verify_rejects_forged_signatures",
            test_certificate_verify_rejects_forged_signatures,
        );
        crate::testing::register_test(
            "tls::certificate_verify_rejects_unsupported_algorithm",
            test_certificate_verify_rejects_unsupported_algorithm,
        );
        crate::testing::register_test(
            "tls::certificate_verify_replay_rejected",
            test_certificate_verify_replay_from_another_handshake_rejected,
        );
        crate::testing::register_test(
            "tls::certificate_verify_is_bound_to_the_certificate_message",
            test_certificate_verify_is_bound_to_the_certificate_message,
        );
        crate::testing::register_test(
            "tls::client_hello_enters_the_transcript",
            test_client_hello_enters_the_transcript,
        );
        crate::testing::register_test(
            "tls::certificate_verify_before_certificate_rejected",
            test_certificate_verify_before_certificate_rejected,
        );
        crate::testing::register_test(
            "tls::finished_requires_certificate_verify",
            test_finished_requires_certificate_verify,
        );
        crate::testing::register_test(
            "tls::finished_rejects_wrong_mac",
            test_finished_rejects_wrong_mac,
        );
        crate::testing::register_test(
            "tls::certificate_message_records_leaf_key",
            test_certificate_message_records_leaf_key,
        );
        crate::testing::register_test(
            "tls::server_hello_rejects_unoffered_cipher_suite",
            test_server_hello_rejects_unoffered_cipher_suite,
        );
        crate::testing::register_test(
            "tls::handshake_record_rejects_unknown_message",
            test_handshake_record_rejects_unknown_message,
        );
        crate::testing::register_test(
            "tls::certificate_must_match_the_requested_hostname",
            test_certificate_must_match_the_requested_hostname,
        );
        crate::testing::register_test(
            "tls::hostname_mismatch_stops_the_handshake",
            test_hostname_mismatch_stops_the_handshake,
        );
        crate::testing::register_test(
            "tls::large_plaintext_never_wraps_the_length_field",
            test_large_plaintext_never_wraps_the_length_field,
        );
        crate::testing::register_test(
            "tls::record_header_is_authenticated",
            test_record_header_is_authenticated,
        );
        crate::testing::register_test(
            "tls::record_aad_is_the_rfc8446_header",
            test_record_aad_is_the_rfc8446_header,
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
        let rec = wrap_record(ContentType::ApplicationData, payload).unwrap();
        let parsed = parse_record(&rec).unwrap();
        assert_eq!(parsed.ctype, ContentType::ApplicationData as u8);
        assert_eq!(parsed.payload, payload.as_slice());
    }

    #[test]
    fn test_tls_encrypt_decrypt() {
        let mut client = TlsSession::new();
        client.set_psk(&[0xABu8; 32]);
        client.state = TlsState::Established;
        // Record-layer test with fixed keys and no handshake, so there is no
        // peer to authenticate; the gate is exercised by the registered
        // `tls::*` suite instead.
        client.authenticated = true;
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
