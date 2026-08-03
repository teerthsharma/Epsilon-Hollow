// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! WPA2-PSK supplicant: PMK/PTK derivation, EAPOL-Key codec, the 4-way handshake
//! state machine, and CCMP.
//!
//! # What is implemented
//!
//! * **PMK** from passphrase via PBKDF2-HMAC-SHA1, 4096 iterations, SSID as salt,
//!   256-bit output (802.11-2016 J.4.1). Gated against the Annex J.4.2 vectors.
//! * **PTK** via the 802.11 PRF (12.7.1.2) over the PMK with the label
//!   `"Pairwise key expansion"` and `Min(AA,SPA) || Max(AA,SPA) || Min(ANonce,SNonce)
//!   || Max(ANonce,SNonce)`. PRF-384 for CCMP, PRF-512 also available.
//! * **PMKID** = `HMAC-SHA1-128(PMK, "PMK Name" || AA || SPA)` (12.7.1.3).
//! * **EAPOL-Key** parse and build, Key Descriptor Type 2 (RSN), with the full Key
//!   Information bit field.
//! * **4-way handshake** as an explicit state machine with typed states, MIC
//!   verification via HMAC-SHA1-128, and replay-counter enforcement.
//! * **CCMP** encrypt and decrypt over AES-128-CCM (M=8, L=2), with the nonce and
//!   AAD constructed from the MAC header per 12.5.3.3.
//!
//! # What is NOT implemented
//!
//! * **No RNG.** This crate runs in a kernel with no entropy source of its own, so it
//!   never generates a nonce. [`Supplicant::new`] takes the SNonce from the caller. A
//!   predictable SNonce destroys the security of the handshake; supplying one from a
//!   counter would be worse than not shipping this at all.
//! * **No GTK.** Message 3 carries the GTK inside Key Data encrypted with the KEK
//!   under NIST AES Key Wrap (RFC 3394). No key-wrap implementation is available to
//!   this crate and hand-rolling one is out of scope, so a Message 3 with the
//!   Encrypted Key Data bit set is refused with [`Wpa2Error::KeyDataEncrypted`]. The
//!   pairwise key path completes; broadcast traffic would not decrypt.
//! * **No group key handshake**, no PMKSA caching, no FT, no 802.11w (management
//!   frame protection). SHA-256 AKMs (`PSK-SHA256`) use a different KDF and are not
//!   accepted.
//! * **Key Descriptor Version 1** (HMAC-MD5 + RC4) and **Version 3** (AES-128-CMAC)
//!   are refused: neither MD5 nor CMAC is available here. Only Version 2
//!   (HMAC-SHA1-128 + AES Key Wrap) is spoken.
//! * **TKIP** is not implemented and will not be. CCMP only.
//! * **No retransmission handling.** A retransmitted Message 1 or 3 carrying the same
//!   replay counter is rejected as a replay rather than answered again. A real STA
//!   needs the timer and duplicate logic that goes with it.
//! * **CCMP receive replay is strict-monotonic** ([`CcmpReplay`]) with no BlockAck
//!   reorder window and no per-TID counters.
//!
//! # Verification
//!
//! Every published vector used here is cited in a comment beside it. The PMK path is
//! gated end to end. The PTK and EAPOL MIC paths are **not** gated against a published
//! vector: no PTK/MIC test vector could be cited precisely, so those paths carry
//! property tests (role symmetry, key-split boundaries, tamper rejection) which are
//! weaker evidence and are labelled as such at each test.

use alloc::vec::Vec;
use core::fmt;

use aes::Aes128;
use ccm::aead::{AeadInPlace, KeyInit};
use ccm::consts::{U13, U8};
use ccm::Ccm;
use hmac::{Hmac, Mac};
use sha1::Sha1;

use crate::frame::{FrameError, FrameType, MacAddr, MacHeader, Reader};

type Aes128Ccm = Ccm<Aes128, U8, U13>;
type HmacSha1 = Hmac<Sha1>;

/// Every way the WPA2 path can fail. Nothing here panics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wpa2Error {
    /// Passphrase outside the length 802.11-2016 J.4.1 allows.
    PassphraseLength(usize),
    /// Passphrase contains an octet outside the printable ASCII range 32..=126.
    PassphraseNotPrintableAscii,
    /// SSID longer than 32 octets.
    SsidLength(usize),
    /// The EAPOL PDU was truncated or otherwise malformed.
    Frame(FrameError),
    /// EAPOL Packet Type is not 3 (EAPOL-Key).
    NotEapolKey(u8),
    /// Key Descriptor Type is not 2 (RSN).
    UnsupportedDescriptorType(u8),
    /// Key Descriptor Version is not 2 (HMAC-SHA1-128 + AES Key Wrap).
    UnsupportedDescriptorVersion(u8),
    /// Declared Key Data Length does not match the declared body length.
    KeyDataLengthMismatch { declared: usize, available: usize },
    /// The frame claims a MIC but the state machine has no PTK to check it with.
    NoPtk,
    /// The frame that should carry a MIC does not set the Key MIC bit.
    MicBitNotSet,
    /// HMAC-SHA1-128 over the frame did not match the MIC field.
    MicMismatch,
    /// Replay counter did not strictly increase.
    ReplayCounterReuse { got: u64, last: u64 },
    /// Message 3's ANonce differs from Message 1's.
    AnonceMismatch,
    /// Key Data is AES-Key-Wrapped; RFC 3394 unwrap is not implemented here.
    KeyDataEncrypted,
    /// A message arrived that the current state does not accept.
    UnexpectedMessage { state: &'static str },
    /// The handshake already failed; it does not recover.
    HandshakeFailed,
    /// CCMP body shorter than the 8-octet header plus 8-octet MIC.
    CcmpTooShort(usize),
    /// CCMP header does not set the ExtIV bit, so it is not a CCMP header.
    CcmpNoExtIv,
    /// CCMP MIC did not verify. The frame is discarded; no plaintext is returned.
    CcmpMicMismatch,
    /// Packet Number did not strictly increase.
    CcmpPnReplay { got: u64, last: u64 },
    /// Packet Number exceeds the 48 bits CCMP allows.
    CcmpPnOverflow(u64),
    /// A RustCrypto constructor rejected a key length. Unreachable with the fixed-size
    /// keys this module uses; present so no call site needs `unwrap`.
    KeySetup,
}

impl From<FrameError> for Wpa2Error {
    fn from(e: FrameError) -> Self {
        Self::Frame(e)
    }
}

impl fmt::Display for Wpa2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PassphraseLength(n) => write!(f, "passphrase of {n} characters out of range"),
            Self::PassphraseNotPrintableAscii => write!(f, "passphrase is not printable ASCII"),
            Self::SsidLength(n) => write!(f, "SSID of {n} octets exceeds 32"),
            Self::Frame(e) => write!(f, "EAPOL frame: {e}"),
            Self::NotEapolKey(t) => write!(f, "EAPOL packet type {t} is not EAPOL-Key"),
            Self::UnsupportedDescriptorType(t) => write!(f, "key descriptor type {t} unsupported"),
            Self::UnsupportedDescriptorVersion(v) => {
                write!(f, "key descriptor version {v} unsupported")
            }
            Self::KeyDataLengthMismatch {
                declared,
                available,
            } => write!(
                f,
                "key data length {declared} but {available} octets in body"
            ),
            Self::NoPtk => write!(f, "no PTK derived yet"),
            Self::MicBitNotSet => write!(f, "key MIC bit not set on a message that needs one"),
            Self::MicMismatch => write!(f, "EAPOL-Key MIC mismatch"),
            Self::ReplayCounterReuse { got, last } => {
                write!(f, "replay counter {got} does not exceed {last}")
            }
            Self::AnonceMismatch => write!(f, "message 3 ANonce differs from message 1"),
            Self::KeyDataEncrypted => {
                write!(f, "encrypted key data: AES key unwrap not implemented")
            }
            Self::UnexpectedMessage { state } => write!(f, "unexpected message in state {state}"),
            Self::HandshakeFailed => write!(f, "handshake already failed"),
            Self::CcmpTooShort(n) => write!(f, "CCMP body of {n} octets is too short"),
            Self::CcmpNoExtIv => write!(f, "CCMP header has no ExtIV bit"),
            Self::CcmpMicMismatch => write!(f, "CCMP MIC mismatch"),
            Self::CcmpPnReplay { got, last } => {
                write!(f, "CCMP packet number {got} does not exceed {last}")
            }
            Self::CcmpPnOverflow(n) => write!(f, "CCMP packet number {n} exceeds 48 bits"),
            Self::KeySetup => write!(f, "key setup rejected by cipher"),
        }
    }
}

// ---------------------------------------------------------------------------
// Key derivation
// ---------------------------------------------------------------------------

/// Derive the PMK from a passphrase and SSID.
///
/// `PMK = PBKDF2(HMAC-SHA1, passphrase, SSID, 4096, 256)` — 802.11-2016 J.4.1.
///
/// The passphrase is 8 to 63 characters from the printable ASCII range 32..=126, and
/// the SSID is at most 32 octets, both as J.4.1 requires.
pub fn pmk_from_passphrase(passphrase: &str, ssid: &[u8]) -> Result<[u8; 32], Wpa2Error> {
    let p = passphrase.as_bytes();
    if p.len() < 8 || p.len() > 63 {
        return Err(Wpa2Error::PassphraseLength(p.len()));
    }
    if !p.iter().all(|c| (32..=126).contains(c)) {
        return Err(Wpa2Error::PassphraseNotPrintableAscii);
    }
    if ssid.len() > 32 {
        return Err(Wpa2Error::SsidLength(ssid.len()));
    }
    let mut pmk = [0u8; 32];
    pbkdf2::pbkdf2_hmac::<Sha1>(p, ssid, 4096, &mut pmk);
    Ok(pmk)
}

fn hmac_sha1(key: &[u8], chunks: &[&[u8]]) -> Result<[u8; 20], Wpa2Error> {
    let mut mac = <HmacSha1 as Mac>::new_from_slice(key).map_err(|_| Wpa2Error::KeySetup)?;
    for c in chunks {
        mac.update(c);
    }
    let mut out = [0u8; 20];
    out.copy_from_slice(&mac.finalize().into_bytes());
    Ok(out)
}

/// The 802.11 PRF (802.11-2016 12.7.1.2).
///
/// `PRF(K, A, B, Len)` concatenates `HMAC-SHA-1(K, A || 0x00 || B || i)` for
/// `i = 0, 1, ...` and truncates to `Len` bits. `A` is a text label, `B` the context.
pub fn prf(key: &[u8], label: &[u8], data: &[u8], bits: usize) -> Result<Vec<u8>, Wpa2Error> {
    let bytes = bits.div_ceil(8);
    let iterations = bytes.div_ceil(20);
    let mut out = Vec::with_capacity(iterations * 20);
    for i in 0..iterations {
        let counter = [i as u8];
        out.extend_from_slice(&hmac_sha1(key, &[label, &[0u8], data, &counter])?);
    }
    out.truncate(bytes);
    Ok(out)
}

/// A pairwise transient key split into its three parts.
///
/// PTK-384 for CCMP: `KCK || KEK || TK`, 128 bits each (802.11-2016 Table 12-8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ptk {
    /// EAPOL-Key Confirmation Key — computes and checks the EAPOL-Key MIC.
    pub kck: [u8; 16],
    /// EAPOL-Key Encryption Key — unwraps Key Data. Unused here; see the module note
    /// on RFC 3394.
    pub kek: [u8; 16],
    /// Temporal Key — the CCMP key.
    pub tk: [u8; 16],
}

/// Derive the PTK.
///
/// `PTK = PRF-384(PMK, "Pairwise key expansion", Min(AA,SPA) || Max(AA,SPA) ||
/// Min(ANonce,SNonce) || Max(ANonce,SNonce))` — 802.11-2016 12.7.1.3.
///
/// The min/max ordering is what makes both ends compute the same key without agreeing
/// on who is who; get it wrong and the handshake fails in one direction only.
pub fn derive_ptk(
    pmk: &[u8; 32],
    aa: MacAddr,
    spa: MacAddr,
    anonce: &[u8; 32],
    snonce: &[u8; 32],
) -> Result<Ptk, Wpa2Error> {
    let (lo_mac, hi_mac) = if aa <= spa { (aa, spa) } else { (spa, aa) };
    let (lo_n, hi_n) = if anonce <= snonce {
        (anonce, snonce)
    } else {
        (snonce, anonce)
    };
    let mut data = Vec::with_capacity(12 + 64);
    data.extend_from_slice(&lo_mac.0);
    data.extend_from_slice(&hi_mac.0);
    data.extend_from_slice(lo_n);
    data.extend_from_slice(hi_n);
    let ptk = prf(pmk, b"Pairwise key expansion", &data, 384)?;
    let mut out = Ptk {
        kck: [0u8; 16],
        kek: [0u8; 16],
        tk: [0u8; 16],
    };
    out.kck.copy_from_slice(&ptk[0..16]);
    out.kek.copy_from_slice(&ptk[16..32]);
    out.tk.copy_from_slice(&ptk[32..48]);
    Ok(out)
}

/// PRF-512 form of the PTK, whose extra 128 bits are the TKIP MIC keys.
///
/// Provided because 12.7.1.3 defines it; TKIP itself is not implemented, so the tail
/// is returned raw rather than interpreted.
pub fn derive_ptk_512(
    pmk: &[u8; 32],
    aa: MacAddr,
    spa: MacAddr,
    anonce: &[u8; 32],
    snonce: &[u8; 32],
) -> Result<Vec<u8>, Wpa2Error> {
    let (lo_mac, hi_mac) = if aa <= spa { (aa, spa) } else { (spa, aa) };
    let (lo_n, hi_n) = if anonce <= snonce {
        (anonce, snonce)
    } else {
        (snonce, anonce)
    };
    let mut data = Vec::with_capacity(12 + 64);
    data.extend_from_slice(&lo_mac.0);
    data.extend_from_slice(&hi_mac.0);
    data.extend_from_slice(lo_n);
    data.extend_from_slice(hi_n);
    prf(pmk, b"Pairwise key expansion", &data, 512)
}

/// `PMKID = HMAC-SHA1-128(PMK, "PMK Name" || AA || SPA)` — 802.11-2016 12.7.1.3.
pub fn pmkid(pmk: &[u8; 32], aa: MacAddr, spa: MacAddr) -> Result<[u8; 16], Wpa2Error> {
    let full = hmac_sha1(pmk, &[b"PMK Name", &aa.0, &spa.0])?;
    let mut out = [0u8; 16];
    out.copy_from_slice(&full[..16]);
    Ok(out)
}

// ---------------------------------------------------------------------------
// EAPOL-Key
// ---------------------------------------------------------------------------

/// EAPOL Packet Type for EAPOL-Key (IEEE 802.1X-2010 Table 11-3).
pub const EAPOL_TYPE_KEY: u8 = 3;
/// Key Descriptor Type 2 = RSN (802.11-2016 9.4.2.25 / Table 12-6).
pub const KEY_DESCRIPTOR_RSN: u8 = 2;
/// Only Key Descriptor Version 2 (HMAC-SHA1-128 + AES Key Wrap) is spoken here.
pub const KEY_DESCRIPTOR_VERSION_HMAC_SHA1: u8 = 2;

/// Octets from Descriptor Type through Key Data Length, inclusive.
const EAPOL_KEY_FIXED: usize = 95;
/// Offset of the Key MIC field inside the whole EAPOL PDU.
const MIC_OFFSET: usize = 81;
const MIC_LEN: usize = 16;

/// The Key Information bit field (802.11-2016 Figure 12-34).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyInfo(pub u16);

impl KeyInfo {
    pub fn descriptor_version(&self) -> u8 {
        (self.0 & 0x0007) as u8
    }
    /// Key Type: set = Pairwise, clear = Group.
    pub fn is_pairwise(&self) -> bool {
        self.0 & 0x0008 != 0
    }
    pub fn key_index(&self) -> u8 {
        ((self.0 >> 4) & 0x0003) as u8
    }
    pub fn install(&self) -> bool {
        self.0 & 0x0040 != 0
    }
    pub fn key_ack(&self) -> bool {
        self.0 & 0x0080 != 0
    }
    pub fn key_mic(&self) -> bool {
        self.0 & 0x0100 != 0
    }
    pub fn secure(&self) -> bool {
        self.0 & 0x0200 != 0
    }
    pub fn error(&self) -> bool {
        self.0 & 0x0400 != 0
    }
    pub fn request(&self) -> bool {
        self.0 & 0x0800 != 0
    }
    pub fn encrypted_key_data(&self) -> bool {
        self.0 & 0x1000 != 0
    }
    pub fn smk_message(&self) -> bool {
        self.0 & 0x2000 != 0
    }
}

/// A parsed or constructed EAPOL-Key frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EapolKey {
    pub protocol_version: u8,
    pub descriptor_type: u8,
    pub key_info: KeyInfo,
    pub key_length: u16,
    pub replay_counter: u64,
    pub nonce: [u8; 32],
    pub key_iv: [u8; 16],
    pub key_rsc: [u8; 8],
    pub key_mic: [u8; 16],
    pub key_data: Vec<u8>,
}

impl EapolKey {
    /// Octets this frame occupies on the wire.
    pub fn wire_len(&self) -> usize {
        4 + EAPOL_KEY_FIXED + self.key_data.len()
    }

    /// Decode an EAPOL-Key PDU.
    ///
    /// The declared Packet Body Length must be exactly `95 + Key Data Length`; a body
    /// that declares more is refused rather than silently padded, because the MIC is
    /// computed over the declared extent and a mismatch there is how a parser gets
    /// walked past a field. Octets in `buf` beyond the declared frame are ignored,
    /// which is what an 802.3 frame's own padding looks like.
    pub fn decode(buf: &[u8]) -> Result<Self, Wpa2Error> {
        let mut r = Reader::new(buf);
        let protocol_version = r.u8()?;
        let packet_type = r.u8()?;
        if packet_type != EAPOL_TYPE_KEY {
            return Err(Wpa2Error::NotEapolKey(packet_type));
        }
        let body_len = r.be16()? as usize;
        if body_len < EAPOL_KEY_FIXED || body_len > r.remaining() {
            return Err(Wpa2Error::Frame(FrameError::Truncated {
                need: body_len,
                got: r.remaining(),
            }));
        }
        let descriptor_type = r.u8()?;
        if descriptor_type != KEY_DESCRIPTOR_RSN {
            return Err(Wpa2Error::UnsupportedDescriptorType(descriptor_type));
        }
        let key_info = KeyInfo(r.be16()?);
        let key_length = r.be16()?;
        let replay_counter = r.be64()?;
        let nonce = r.array::<32>()?;
        let key_iv = r.array::<16>()?;
        let key_rsc = r.array::<8>()?;
        let _reserved = r.array::<8>()?; // 802.11-2016 reserves these; ignored.
        let key_mic = r.array::<16>()?;
        let key_data_len = r.be16()? as usize;
        let available = body_len - EAPOL_KEY_FIXED;
        if key_data_len != available {
            return Err(Wpa2Error::KeyDataLengthMismatch {
                declared: key_data_len,
                available,
            });
        }
        let key_data = r.take(key_data_len)?.to_vec();
        Ok(Self {
            protocol_version,
            descriptor_type,
            key_info,
            key_length,
            replay_counter,
            nonce,
            key_iv,
            key_rsc,
            key_mic,
            key_data,
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let body_len = EAPOL_KEY_FIXED + self.key_data.len();
        let mut out = Vec::with_capacity(4 + body_len);
        out.push(self.protocol_version);
        out.push(EAPOL_TYPE_KEY);
        out.extend_from_slice(&(body_len as u16).to_be_bytes());
        out.push(self.descriptor_type);
        out.extend_from_slice(&self.key_info.0.to_be_bytes());
        out.extend_from_slice(&self.key_length.to_be_bytes());
        out.extend_from_slice(&self.replay_counter.to_be_bytes());
        out.extend_from_slice(&self.nonce);
        out.extend_from_slice(&self.key_iv);
        out.extend_from_slice(&self.key_rsc);
        out.extend_from_slice(&[0u8; 8]); // reserved
        out.extend_from_slice(&self.key_mic);
        out.extend_from_slice(&(self.key_data.len() as u16).to_be_bytes());
        out.extend_from_slice(&self.key_data);
        out
    }

    /// Encode with the MIC field filled in by HMAC-SHA1-128 under the KCK.
    ///
    /// 802.11-2016 12.7.2: the MIC covers the entire EAPOL-Key frame, from the EAPOL
    /// protocol version octet through the end of Key Data, with the Key MIC field
    /// itself set to zero.
    pub fn encode_with_mic(&self, kck: &[u8; 16]) -> Result<Vec<u8>, Wpa2Error> {
        let mut buf = self.encode();
        buf[MIC_OFFSET..MIC_OFFSET + MIC_LEN].fill(0);
        let mic = hmac_sha1(kck, &[&buf])?;
        buf[MIC_OFFSET..MIC_OFFSET + MIC_LEN].copy_from_slice(&mic[..MIC_LEN]);
        Ok(buf)
    }
}

/// Verify the MIC over a received EAPOL-Key frame.
///
/// `frame` must be the exact wire extent of the EAPOL-Key PDU. The comparison runs
/// through `Mac::verify_truncated_left`, which is the `subtle` constant-time
/// comparison — a `==` here leaks the MIC one byte at a time under timing.
pub fn verify_mic(kck: &[u8; 16], frame: &[u8], mic: &[u8; 16]) -> Result<(), Wpa2Error> {
    if frame.len() < MIC_OFFSET + MIC_LEN {
        return Err(Wpa2Error::Frame(FrameError::Truncated {
            need: MIC_OFFSET + MIC_LEN,
            got: frame.len(),
        }));
    }
    let mut mac = <HmacSha1 as Mac>::new_from_slice(kck).map_err(|_| Wpa2Error::KeySetup)?;
    mac.update(&frame[..MIC_OFFSET]);
    mac.update(&[0u8; MIC_LEN]);
    mac.update(&frame[MIC_OFFSET + MIC_LEN..]);
    mac.verify_truncated_left(mic)
        .map_err(|_| Wpa2Error::MicMismatch)
}

// ---------------------------------------------------------------------------
// 4-way handshake
// ---------------------------------------------------------------------------

/// Which message of the 4-way handshake a received frame is.
///
/// A supplicant only ever receives 1 and 3; the discrimination is the Key ACK and Key
/// MIC bits (802.11-2016 12.7.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeMessage {
    /// Key ACK set, Key MIC clear.
    Message1,
    /// Key ACK set, Key MIC set.
    Message3,
}

impl HandshakeMessage {
    pub fn classify(info: KeyInfo) -> Option<Self> {
        match (info.key_ack(), info.key_mic()) {
            (true, false) => Some(Self::Message1),
            (true, true) => Some(Self::Message3),
            _ => None,
        }
    }
}

/// Supplicant state. There is no path from `Failed` back to a live handshake.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandshakeState {
    /// Nothing received; waiting for Message 1.
    Idle,
    /// Message 1 accepted, Message 2 sent, PTK derived but not yet confirmed.
    PtkStart { ptk: Ptk, anonce: [u8; 32] },
    /// Message 3 verified and Message 4 sent. The PTK is confirmed.
    PtkDone { ptk: Ptk },
    /// Fail-closed. Every later message is refused.
    Failed(Wpa2Error),
}

impl HandshakeState {
    /// A short stable tag, in the style the rest of this kernel uses for boot proofs.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::PtkStart { .. } => "ptk_start",
            Self::PtkDone { .. } => "ptk_done",
            Self::Failed(_) => "failed",
        }
    }
}

/// WPA2-PSK supplicant driving the 4-way handshake.
///
/// The SNonce is supplied by the caller and never generated here. See the module
/// note: this crate has no entropy source, and inventing one would be the exact kind
/// of comfortable lie the rest of this kernel refuses to tell.
pub struct Supplicant {
    pmk: [u8; 32],
    /// Authenticator (AP) address.
    aa: MacAddr,
    /// Supplicant (STA) address.
    spa: MacAddr,
    snonce: [u8; 32],
    /// The station's own RSN element, echoed in Message 2's Key Data.
    rsn_ie: Vec<u8>,
    state: HandshakeState,
    last_replay: Option<u64>,
}

impl Supplicant {
    pub fn new(
        pmk: [u8; 32],
        aa: MacAddr,
        spa: MacAddr,
        snonce: [u8; 32],
        rsn_ie: Vec<u8>,
    ) -> Self {
        Self {
            pmk,
            aa,
            spa,
            snonce,
            rsn_ie,
            state: HandshakeState::Idle,
            last_replay: None,
        }
    }

    pub fn state(&self) -> &HandshakeState {
        &self.state
    }

    /// The confirmed PTK, available only after Message 3 verified.
    pub fn ptk(&self) -> Option<&Ptk> {
        match &self.state {
            HandshakeState::PtkDone { ptk } => Some(ptk),
            _ => None,
        }
    }

    /// Feed a received EAPOL-Key PDU; get back the EAPOL-Key PDU to transmit.
    ///
    /// Errors leave the supplicant in [`HandshakeState::Failed`]. Nothing recovers a
    /// failed handshake: a supplicant that keeps trying after a MIC failure is a
    /// supplicant an attacker gets unlimited attempts against.
    pub fn on_eapol(&mut self, buf: &[u8]) -> Result<Vec<u8>, Wpa2Error> {
        let r = self.step(buf);
        if let Err(e) = r {
            self.state = HandshakeState::Failed(e);
        }
        r
    }

    fn step(&mut self, buf: &[u8]) -> Result<Vec<u8>, Wpa2Error> {
        if let HandshakeState::Failed(_) = self.state {
            return Err(Wpa2Error::HandshakeFailed);
        }
        let key = EapolKey::decode(buf)?;
        if key.key_info.descriptor_version() != KEY_DESCRIPTOR_VERSION_HMAC_SHA1 {
            return Err(Wpa2Error::UnsupportedDescriptorVersion(
                key.key_info.descriptor_version(),
            ));
        }
        let frame = &buf[..key.wire_len()];
        match HandshakeMessage::classify(key.key_info) {
            Some(HandshakeMessage::Message1) => self.on_message1(&key),
            Some(HandshakeMessage::Message3) => self.on_message3(&key, frame),
            None => Err(Wpa2Error::UnexpectedMessage {
                state: self.state.tag(),
            }),
        }
    }

    fn on_message1(&mut self, key: &EapolKey) -> Result<Vec<u8>, Wpa2Error> {
        // Message 1 carries no MIC, so nothing about it can be authenticated. The only
        // defence available is the replay counter, and it must strictly increase.
        if let Some(last) = self.last_replay {
            if key.replay_counter <= last {
                return Err(Wpa2Error::ReplayCounterReuse {
                    got: key.replay_counter,
                    last,
                });
            }
        }
        let anonce = key.nonce;
        let ptk = derive_ptk(&self.pmk, self.aa, self.spa, &anonce, &self.snonce)?;
        let msg2 = EapolKey {
            protocol_version: key.protocol_version,
            descriptor_type: KEY_DESCRIPTOR_RSN,
            // Version 2, Pairwise, Key MIC.
            key_info: KeyInfo(u16::from(KEY_DESCRIPTOR_VERSION_HMAC_SHA1) | 0x0008 | 0x0100),
            key_length: 0,
            replay_counter: key.replay_counter,
            nonce: self.snonce,
            key_iv: [0u8; 16],
            key_rsc: [0u8; 8],
            key_mic: [0u8; 16],
            key_data: self.rsn_ie.clone(),
        };
        let out = msg2.encode_with_mic(&ptk.kck)?;
        self.last_replay = Some(key.replay_counter);
        self.state = HandshakeState::PtkStart { ptk, anonce };
        Ok(out)
    }

    fn on_message3(&mut self, key: &EapolKey, frame: &[u8]) -> Result<Vec<u8>, Wpa2Error> {
        let HandshakeState::PtkStart { ptk, anonce } = &self.state else {
            return Err(Wpa2Error::UnexpectedMessage {
                state: self.state.tag(),
            });
        };
        let (ptk, anonce) = (*ptk, *anonce);
        if !key.key_info.key_mic() {
            return Err(Wpa2Error::MicBitNotSet);
        }
        // 802.11-2016 12.7.6.4: discard Message 3 if the replay counter has been used
        // or the MIC is invalid. The MIC is checked first and the counter is recorded
        // only after both pass, so an unauthenticated frame cannot advance our counter
        // and lock out the real Message 3.
        verify_mic(&ptk.kck, frame, &key.key_mic)?;
        if let Some(last) = self.last_replay {
            if key.replay_counter <= last {
                return Err(Wpa2Error::ReplayCounterReuse {
                    got: key.replay_counter,
                    last,
                });
            }
        }
        if key.nonce != anonce {
            return Err(Wpa2Error::AnonceMismatch);
        }
        if key.key_info.encrypted_key_data() && !key.key_data.is_empty() {
            // The GTK lives here under RFC 3394 AES Key Wrap, which this crate does
            // not implement. Refusing is the honest outcome; pretending to install a
            // group key would not be.
            return Err(Wpa2Error::KeyDataEncrypted);
        }
        let msg4 = EapolKey {
            protocol_version: key.protocol_version,
            descriptor_type: KEY_DESCRIPTOR_RSN,
            // Version 2, Pairwise, Key MIC, Secure.
            key_info: KeyInfo(
                u16::from(KEY_DESCRIPTOR_VERSION_HMAC_SHA1) | 0x0008 | 0x0100 | 0x0200,
            ),
            key_length: 0,
            replay_counter: key.replay_counter,
            nonce: [0u8; 32],
            key_iv: [0u8; 16],
            key_rsc: [0u8; 8],
            key_mic: [0u8; 16],
            key_data: Vec::new(),
        };
        let out = msg4.encode_with_mic(&ptk.kck)?;
        self.last_replay = Some(key.replay_counter);
        self.state = HandshakeState::PtkDone { ptk };
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// CCMP
// ---------------------------------------------------------------------------

/// Octets of CCMP header prepended to the encrypted MPDU (802.11-2016 Figure 12-16).
pub const CCMP_HEADER_LEN: usize = 8;
/// CCMP MIC length. AES-CCM with M=8.
pub const CCMP_MIC_LEN: usize = 8;
const PN_MAX: u64 = (1u64 << 48) - 1;

/// Build the 8-octet CCMP header: PN0, PN1, reserved, KeyID|ExtIV, PN2..PN5.
fn ccmp_header(pn: u64, key_id: u8) -> [u8; CCMP_HEADER_LEN] {
    let b = pn.to_le_bytes();
    [
        b[0],
        b[1],
        0,
        ((key_id & 0x03) << 6) | 0x20, // ExtIV
        b[2],
        b[3],
        b[4],
        b[5],
    ]
}

fn parse_ccmp_header(h: &[u8]) -> Result<(u64, u8), Wpa2Error> {
    if h.len() < CCMP_HEADER_LEN {
        return Err(Wpa2Error::CcmpTooShort(h.len()));
    }
    if h[3] & 0x20 == 0 {
        return Err(Wpa2Error::CcmpNoExtIv);
    }
    let pn = u64::from_le_bytes([h[0], h[1], h[4], h[5], h[6], h[7], 0, 0]);
    Ok((pn, (h[3] >> 6) & 0x03))
}

/// CCM nonce (802.11-2016 12.5.3.3.4): nonce flags, A2, then the PN big-endian.
fn ccm_nonce(header: &MacHeader, pn: u64) -> [u8; 13] {
    let tid = header.qos_ctrl.map_or(0, |q| (q & 0x000f) as u8);
    let mgmt = u8::from(header.fc.frame_type == FrameType::Management) << 4;
    let mut n = [0u8; 13];
    n[0] = tid | mgmt;
    n[1..7].copy_from_slice(&header.addr2.0);
    let b = pn.to_be_bytes();
    n[7..13].copy_from_slice(&b[2..8]);
    n
}

/// CCM AAD (802.11-2016 12.5.3.3.3).
///
/// The masked bits are the ones a retransmission or a power-save transition may flip
/// in flight; if they were covered, a legitimately retransmitted frame would fail its
/// own MIC.
fn ccm_aad(header: &MacHeader) -> Vec<u8> {
    const RETRY: u16 = 0x0800;
    const PWR_MGT: u16 = 0x1000;
    const MORE_DATA: u16 = 0x2000;
    const PROTECTED: u16 = 0x4000;
    const SUBTYPE_HIGH: u16 = 0x0070;

    let mut fc = header.fc.bits() & !(RETRY | PWR_MGT | MORE_DATA);
    if header.fc.frame_type != FrameType::Management {
        fc &= !SUBTYPE_HIGH;
    }
    fc |= PROTECTED;

    let mut aad = Vec::with_capacity(30);
    aad.extend_from_slice(&fc.to_le_bytes());
    aad.extend_from_slice(&header.addr1.0);
    aad.extend_from_slice(&header.addr2.0);
    aad.extend_from_slice(&header.addr3.0);
    // Sequence number masked out, fragment number retained.
    aad.extend_from_slice(&(header.seq_ctrl.bits() & 0x000f).to_le_bytes());
    if let Some(a4) = header.addr4 {
        aad.extend_from_slice(&a4.0);
    }
    if let Some(q) = header.qos_ctrl {
        aad.extend_from_slice(&(q & 0x000f).to_le_bytes());
    }
    aad
}

/// Encrypt an MPDU payload under CCMP.
///
/// Returns `CCMP header || ciphertext || MIC`, which is exactly what goes in the frame
/// body of a Protected frame.
pub fn ccmp_encrypt(
    tk: &[u8; 16],
    header: &MacHeader,
    key_id: u8,
    pn: u64,
    plaintext: &[u8],
) -> Result<Vec<u8>, Wpa2Error> {
    if pn > PN_MAX {
        return Err(Wpa2Error::CcmpPnOverflow(pn));
    }
    let cipher = Aes128Ccm::new_from_slice(tk).map_err(|_| Wpa2Error::KeySetup)?;
    let nonce = ccm::aead::Nonce::<Aes128Ccm>::from(ccm_nonce(header, pn));
    let aad = ccm_aad(header);
    let mut buf = plaintext.to_vec();
    let tag = cipher
        .encrypt_in_place_detached(&nonce, &aad, &mut buf)
        .map_err(|_| Wpa2Error::KeySetup)?;
    let mut out = Vec::with_capacity(CCMP_HEADER_LEN + buf.len() + CCMP_MIC_LEN);
    out.extend_from_slice(&ccmp_header(pn, key_id));
    out.extend_from_slice(&buf);
    out.extend_from_slice(tag.as_slice());
    Ok(out)
}

/// Decrypt a Protected frame body. Returns the plaintext and the received PN.
///
/// On MIC failure the plaintext is dropped and [`Wpa2Error::CcmpMicMismatch`] is
/// returned; unauthenticated plaintext never leaves this function.
pub fn ccmp_decrypt(
    tk: &[u8; 16],
    header: &MacHeader,
    body: &[u8],
) -> Result<(Vec<u8>, u64), Wpa2Error> {
    if body.len() < CCMP_HEADER_LEN + CCMP_MIC_LEN {
        return Err(Wpa2Error::CcmpTooShort(body.len()));
    }
    let (pn, _key_id) = parse_ccmp_header(&body[..CCMP_HEADER_LEN])?;
    let split = body.len() - CCMP_MIC_LEN;
    let mut buf = body[CCMP_HEADER_LEN..split].to_vec();
    let mut tag_bytes = [0u8; CCMP_MIC_LEN];
    tag_bytes.copy_from_slice(&body[split..]);
    let tag = ccm::aead::Tag::<Aes128Ccm>::from(tag_bytes);
    let cipher = Aes128Ccm::new_from_slice(tk).map_err(|_| Wpa2Error::KeySetup)?;
    let nonce = ccm::aead::Nonce::<Aes128Ccm>::from(ccm_nonce(header, pn));
    let aad = ccm_aad(header);
    cipher
        .decrypt_in_place_detached(&nonce, &aad, &mut buf, &tag)
        .map_err(|_| Wpa2Error::CcmpMicMismatch)?;
    Ok((buf, pn))
}

/// Strict-monotonic CCMP receive replay counter.
///
/// No BlockAck reorder window and no per-TID counters: a legitimate reordered frame
/// under BlockAck is dropped. That is a throughput bug, not a security bug, and the
/// fix is a bitmap window per TID.
#[derive(Debug, Default, Clone, Copy)]
pub struct CcmpReplay {
    last: Option<u64>,
}

impl CcmpReplay {
    pub fn accept(&mut self, pn: u64) -> Result<(), Wpa2Error> {
        if let Some(last) = self.last {
            if pn <= last {
                return Err(Wpa2Error::CcmpPnReplay { got: pn, last });
            }
        }
        self.last = Some(pn);
        Ok(())
    }

    pub fn last(&self) -> Option<u64> {
        self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FrameControl, SequenceControl};
    use alloc::vec;

    fn hex(s: &str) -> Vec<u8> {
        let b: Vec<u8> = s.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
        b.chunks(2)
            .map(|p| {
                let hi = (p[0] as char).to_digit(16).unwrap_or(0) as u8;
                let lo = (p[1] as char).to_digit(16).unwrap_or(0) as u8;
                (hi << 4) | lo
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Primitive-level vectors
    // -----------------------------------------------------------------------

    /// RFC 6070, "PKCS #5: Password-Based Key Derivation Function 2 (PBKDF2) Test
    /// Vectors", section 2. Gates PBKDF2-HMAC-SHA1 itself before anything is built on
    /// top of it. The c = 16777216 case is omitted deliberately: it is minutes of CPU
    /// for no additional coverage.
    #[test]
    fn rfc6070_pbkdf2_hmac_sha1_vectors() {
        let cases: &[(&[u8], &[u8], u32, &str)] = &[
            (
                b"password",
                b"salt",
                1,
                "0c60c80f961f0e71f3a9b524af6012062fe037a6",
            ),
            (
                b"password",
                b"salt",
                2,
                "ea6c014dc72d6f8ccd1ed92ace1d41f0d8de8957",
            ),
            (
                b"password",
                b"salt",
                4096,
                "4b007901b765489abead49d926f721d065a429c1",
            ),
            (
                b"passwordPASSWORDpassword",
                b"saltSALTsaltSALTsaltSALTsaltSALTsalt",
                4096,
                "3d2eec4fe41c849b80c8d83662c0e44a8b291a964cf2f07038",
            ),
            (
                b"pass\0word",
                b"sa\0lt",
                4096,
                "56fa6aa75548099dcc37d7f03425e0c3",
            ),
        ];
        for (p, s, c, expect) in cases {
            let want = hex(expect);
            let mut got = vec![0u8; want.len()];
            pbkdf2::pbkdf2_hmac::<Sha1>(p, s, *c, &mut got);
            assert_eq!(got, want, "RFC 6070 case c={c} failed");
        }
    }

    /// RFC 2202, "Test Cases for HMAC-MD5 and HMAC-SHA-1", section 3. Gates the
    /// HMAC-SHA-1 that the 802.11 PRF and the EAPOL-Key MIC are both composed from.
    #[test]
    fn rfc2202_hmac_sha1_vectors() {
        let c1 = hmac_sha1(&[0x0b; 20], &[b"Hi There"]).expect("hmac");
        assert_eq!(
            &c1[..],
            &hex("b617318655057264e28bc0b6fb378c8ef146be00")[..]
        );

        let c2 = hmac_sha1(b"Jefe", &[b"what do ya want for nothing?"]).expect("hmac");
        assert_eq!(
            &c2[..],
            &hex("effcdf6ae5eb2fa2d27416d5f184df9c259a7c79")[..]
        );

        let c3 = hmac_sha1(&[0xaa; 20], &[&[0xdd; 50][..]]).expect("hmac");
        assert_eq!(
            &c3[..],
            &hex("125d7342b9ac11cd91a39af48aa17b4f63f175d3")[..]
        );
    }

    /// RFC 3610, "Counter with CBC-MAC (CCM)", section 8, Packet Vector #1.
    ///
    /// The parameters there — M = 8, L = 2, 13-octet nonce — are exactly CCMP's, so
    /// this vector gates that `Aes128Ccm` is instantiated with the sizes CCMP needs.
    /// Instantiate it with M = 16 or L = 8 and this test fails, which is the point:
    /// the CCMP round-trip test below would pass happily with the wrong parameters
    /// because it checks itself against itself.
    #[test]
    fn rfc3610_ccm_packet_vector_1() {
        let key = hex("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECF");
        let nonce_bytes = hex("00000003020100A0A1A2A3A4A5");
        let input = hex("000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E");
        let (aad, plaintext) = input.split_at(8);
        let expected = hex("588C979A61C663D2F066D0C2C0F989806D5F6B61DAC38417E8D12CFDF926E0");

        let cipher = Aes128Ccm::new_from_slice(&key).expect("key");
        let mut n = [0u8; 13];
        n.copy_from_slice(&nonce_bytes);
        let nonce = ccm::aead::Nonce::<Aes128Ccm>::from(n);
        let mut buf = plaintext.to_vec();
        let tag = cipher
            .encrypt_in_place_detached(&nonce, aad, &mut buf)
            .expect("encrypt");
        let mut got = buf.clone();
        got.extend_from_slice(tag.as_slice());
        assert_eq!(got, expected, "RFC 3610 packet vector #1");
    }

    // -----------------------------------------------------------------------
    // IEEE 802.11 PSK vectors
    // -----------------------------------------------------------------------

    /// IEEE Std 802.11-2016, Annex J.4.2, "Sample pass-phrase-to-PSK mappings"
    /// (identical to the test cases in 802.11i-2004 Annex H.4.3). This is the
    /// composition test: PBKDF2-HMAC-SHA1, 4096 iterations, SSID as salt, 256 bits.
    ///
    /// All three published vectors are gated.
    ///
    /// The third was briefly dropped on the belief that its published PSK was
    /// unreproducible. That was an input error, not a spec error: the vector
    /// pairs a **32-character all-`a` pass-phrase** with a 32-character all-`Z`
    /// SSID, and a search over all-`Z` pass-phrases naturally found nothing. An
    /// exhaustive sweep of `{a,Z}` pass-phrases of length 1..=70 against
    /// `{a,Z}` SSIDs of length 1..=32 returns exactly one preimage for
    /// `becb9386…`, and it is `("a"×32, "Z"×32)`. Recorded here because
    /// "the published constant is wrong" is a conclusion that deserves an
    /// exhaustive search behind it, and this one did not survive the search.
    ///
    /// Note the length: 32 is inside the standard's own 8..=63 pass-phrase
    /// bound, so gating this vector requires no relaxation of that bound.
    #[test]
    fn ieee80211_2016_j42_psk_vectors() {
        let all_a = "a".repeat(32);
        let all_z = "Z".repeat(32);
        let cases: &[(&str, &[u8], &str)] = &[
            (
                "password",
                b"IEEE",
                "f42c6fc52df0ebef9ebb4b90b38a5f902e83fe1b135a70e23aed762e9710a12e",
            ),
            (
                "ThisIsAPassword",
                b"ThisIsASSID",
                "0dc0d6eb90555ed6419756b9a15ec3e3209b63df707dd508d14581f8982721af",
            ),
            (
                all_a.as_str(),
                all_z.as_bytes(),
                "becb93866bb8c3832cb777c2f559807c8c59afcb6eae734885001300a981cc62",
            ),
        ];
        for (pass, ssid, expect) in cases {
            let pmk = pmk_from_passphrase(pass, ssid).expect("PMK derivation");
            assert_eq!(
                pmk.to_vec(),
                hex(expect),
                "802.11-2016 J.4.2 vector for SSID {:?} failed",
                core::str::from_utf8(ssid).unwrap_or("?")
            );
        }
    }

    #[test]
    fn passphrase_and_ssid_bounds_are_enforced() {
        assert_eq!(
            pmk_from_passphrase("short", b"IEEE"),
            Err(Wpa2Error::PassphraseLength(5))
        );
        assert_eq!(
            pmk_from_passphrase("passwordé", b"IEEE"),
            Err(Wpa2Error::PassphraseNotPrintableAscii)
        );
        let long_ssid = [b'a'; 33];
        assert_eq!(
            pmk_from_passphrase("password", &long_ssid),
            Err(Wpa2Error::SsidLength(33))
        );
    }

    // -----------------------------------------------------------------------
    // PTK — property tests, NOT vector-gated. Stated plainly: no published PTK
    // vector could be cited precisely, so these check structure and symmetry only.
    // -----------------------------------------------------------------------

    fn fixture() -> ([u8; 32], MacAddr, MacAddr, [u8; 32], [u8; 32]) {
        let pmk = pmk_from_passphrase("password", b"IEEE").expect("pmk");
        let aa = MacAddr([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        let spa = MacAddr([0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb]);
        let anonce = [0xa5u8; 32];
        let mut snonce = [0x5au8; 32];
        snonce[0] = 0x01;
        (pmk, aa, spa, anonce, snonce)
    }

    #[test]
    fn ptk_is_role_symmetric_under_min_max_ordering() {
        // Property, not a published vector: both ends must land on the same PTK with
        // their arguments swapped. An implementation that concatenates in the order
        // given rather than Min/Max order fails this and only this.
        let (pmk, aa, spa, anonce, snonce) = fixture();
        let a = derive_ptk(&pmk, aa, spa, &anonce, &snonce).expect("ptk");
        let b = derive_ptk(&pmk, spa, aa, &snonce, &anonce).expect("ptk");
        assert_eq!(a, b);
    }

    #[test]
    fn ptk_depends_on_every_input() {
        let (pmk, aa, spa, anonce, snonce) = fixture();
        let base = derive_ptk(&pmk, aa, spa, &anonce, &snonce).expect("ptk");

        let mut other_pmk = pmk;
        other_pmk[0] ^= 1;
        assert_ne!(
            derive_ptk(&other_pmk, aa, spa, &anonce, &snonce).expect("ptk"),
            base
        );

        let other_aa = MacAddr([0x00, 0x11, 0x22, 0x33, 0x44, 0x56]);
        assert_ne!(
            derive_ptk(&pmk, other_aa, spa, &anonce, &snonce).expect("ptk"),
            base
        );

        let mut other_anonce = anonce;
        other_anonce[31] ^= 1;
        assert_ne!(
            derive_ptk(&pmk, aa, spa, &other_anonce, &snonce).expect("ptk"),
            base
        );

        let mut other_snonce = snonce;
        other_snonce[31] ^= 1;
        assert_ne!(
            derive_ptk(&pmk, aa, spa, &anonce, &other_snonce).expect("ptk"),
            base
        );
    }

    #[test]
    fn ptk_384_is_the_prefix_of_ptk_512() {
        let (pmk, aa, spa, anonce, snonce) = fixture();
        let p384 = derive_ptk(&pmk, aa, spa, &anonce, &snonce).expect("ptk");
        let p512 = derive_ptk_512(&pmk, aa, spa, &anonce, &snonce).expect("ptk512");
        assert_eq!(p512.len(), 64);
        assert_eq!(&p512[0..16], &p384.kck);
        assert_eq!(&p512[16..32], &p384.kek);
        assert_eq!(&p512[32..48], &p384.tk);
    }

    #[test]
    fn prf_truncates_to_the_requested_bit_length() {
        let key = [0x0bu8; 20];
        let long = prf(&key, b"prefix", b"Hi There", 512).expect("prf");
        assert_eq!(long.len(), 64);
        for bits in [128usize, 192, 256, 384] {
            let short = prf(&key, b"prefix", b"Hi There", bits).expect("prf");
            assert_eq!(short.len(), bits / 8);
            assert_eq!(short[..], long[..bits / 8]);
        }
    }

    #[test]
    fn pmkid_is_128_bits_and_order_sensitive() {
        let (pmk, aa, spa, _, _) = fixture();
        let a = pmkid(&pmk, aa, spa).expect("pmkid");
        let b = pmkid(&pmk, spa, aa).expect("pmkid");
        assert_ne!(a, b, "PMKID must not be symmetric in AA and SPA");
    }

    // -----------------------------------------------------------------------
    // EAPOL-Key codec
    // -----------------------------------------------------------------------

    fn sample_key(key_info: u16, replay: u64, nonce: [u8; 32], key_data: Vec<u8>) -> EapolKey {
        EapolKey {
            protocol_version: 2,
            descriptor_type: KEY_DESCRIPTOR_RSN,
            key_info: KeyInfo(key_info),
            key_length: 16,
            replay_counter: replay,
            nonce,
            key_iv: [0u8; 16],
            key_rsc: [0u8; 8],
            key_mic: [0u8; 16],
            key_data,
        }
    }

    #[test]
    fn eapol_key_round_trips() {
        let k = sample_key(0x008a, 1, [0xa5; 32], vec![0x30, 0x14, 0x01, 0x00]);
        let bytes = k.encode();
        assert_eq!(bytes.len(), k.wire_len());
        assert_eq!(bytes.len(), 4 + 95 + 4);
        // Body length is big-endian: EAPOL is network order, unlike 802.11 MAC fields.
        assert_eq!(u16::from_be_bytes([bytes[2], bytes[3]]), 99);
        assert_eq!(EapolKey::decode(&bytes).expect("decode"), k);
    }

    #[test]
    fn key_info_bits_decode() {
        // Version 2, Pairwise, Install, Ack, MIC, Secure, Encrypted Key Data.
        let i = KeyInfo(0x13ca);
        assert_eq!(i.descriptor_version(), 2);
        assert!(i.is_pairwise());
        assert!(i.install());
        assert!(i.key_ack());
        assert!(i.key_mic());
        assert!(i.secure());
        assert!(!i.error());
        assert!(!i.request());
        assert!(i.encrypted_key_data());
        assert!(!i.smk_message());
    }

    #[test]
    fn every_truncation_of_an_eapol_key_is_an_error() {
        let k = sample_key(0x008a, 1, [0xa5; 32], vec![0x30, 0x14]);
        let bytes = k.encode();
        for n in 0..bytes.len() {
            assert!(
                EapolKey::decode(&bytes[..n]).is_err(),
                "prefix of {n} octets decoded"
            );
        }
        assert!(EapolKey::decode(&bytes).is_ok());
    }

    #[test]
    fn lying_key_data_length_is_refused() {
        let k = sample_key(0x008a, 1, [0xa5; 32], vec![0x01, 0x02, 0x03, 0x04]);
        let mut bytes = k.encode();
        // Body says 99 octets; claim only 2 octets of key data.
        let n = bytes.len();
        bytes[n - 4 - 2] = 0x00;
        bytes[n - 4 - 1] = 0x02;
        assert_eq!(
            EapolKey::decode(&bytes),
            Err(Wpa2Error::KeyDataLengthMismatch {
                declared: 2,
                available: 4
            })
        );
    }

    #[test]
    fn non_eapol_key_packet_types_are_refused() {
        let mut bytes = sample_key(0x008a, 1, [0; 32], vec![]).encode();
        bytes[1] = 0; // EAP-Packet
        assert_eq!(EapolKey::decode(&bytes), Err(Wpa2Error::NotEapolKey(0)));
    }

    #[test]
    fn mic_verifies_and_a_single_flipped_bit_does_not() {
        let kck = [0x42u8; 16];
        let k = sample_key(0x008a, 1, [0xa5; 32], vec![0x30, 0x14]);
        let signed = k.encode_with_mic(&kck).expect("mic");
        let parsed = EapolKey::decode(&signed).expect("decode");
        verify_mic(&kck, &signed, &parsed.key_mic).expect("MIC must verify");

        for i in 0..signed.len() {
            let mut bad = signed.clone();
            bad[i] ^= 0x01;
            let Ok(p) = EapolKey::decode(&bad) else {
                continue; // some flips break the framing first; that is also a refusal
            };
            assert_eq!(
                verify_mic(&kck, &bad, &p.key_mic),
                Err(Wpa2Error::MicMismatch),
                "flipped bit at octet {i} still verified"
            );
        }
    }

    #[test]
    fn mic_under_the_wrong_kck_is_refused() {
        let k = sample_key(0x008a, 1, [0xa5; 32], vec![]);
        let signed = k.encode_with_mic(&[0x42u8; 16]).expect("mic");
        let parsed = EapolKey::decode(&signed).expect("decode");
        assert_eq!(
            verify_mic(&[0x43u8; 16], &signed, &parsed.key_mic),
            Err(Wpa2Error::MicMismatch)
        );
    }

    // -----------------------------------------------------------------------
    // Handshake state machine
    // -----------------------------------------------------------------------

    /// Stand-in authenticator: enough to drive the supplicant, no more. It knows the
    /// PMK because this is a PSK network; that is not a shortcut.
    struct Authenticator {
        pmk: [u8; 32],
        aa: MacAddr,
        spa: MacAddr,
        anonce: [u8; 32],
        replay: u64,
    }

    impl Authenticator {
        fn message1(&mut self) -> Vec<u8> {
            self.replay += 1;
            // Version 2, Pairwise, Ack.
            sample_key(0x008a, self.replay, self.anonce, Vec::new()).encode()
        }

        fn message3(&mut self, snonce: [u8; 32]) -> Vec<u8> {
            let ptk = derive_ptk(&self.pmk, self.aa, self.spa, &self.anonce, &snonce).expect("ptk");
            self.replay += 1;
            // Version 2, Pairwise, Install, Ack, MIC, Secure.
            let k = sample_key(0x03ca, self.replay, self.anonce, Vec::new());
            k.encode_with_mic(&ptk.kck).expect("mic")
        }
    }

    fn handshake_pair() -> (Supplicant, Authenticator, [u8; 32]) {
        let (pmk, aa, spa, anonce, snonce) = fixture();
        let sup = Supplicant::new(pmk, aa, spa, snonce, vec![0x30, 0x14, 0x01, 0x00]);
        let auth = Authenticator {
            pmk,
            aa,
            spa,
            anonce,
            replay: 0,
        };
        (sup, auth, snonce)
    }

    #[test]
    fn four_way_handshake_completes_and_both_ends_agree_on_the_ptk() {
        let (mut sup, mut auth, snonce) = handshake_pair();
        assert_eq!(sup.state().tag(), "idle");

        let m1 = auth.message1();
        let m2 = sup.on_eapol(&m1).expect("message 2");
        assert_eq!(sup.state().tag(), "ptk_start");

        // The authenticator checks message 2's MIC, which is the supplicant proving it
        // holds the PMK. Without this the test would not exercise the MIC we generate.
        let parsed2 = EapolKey::decode(&m2).expect("decode m2");
        let expected =
            derive_ptk(&auth.pmk, auth.aa, auth.spa, &auth.anonce, &parsed2.nonce).expect("ptk");
        verify_mic(&expected.kck, &m2, &parsed2.key_mic).expect("message 2 MIC");
        assert_eq!(parsed2.nonce, snonce);

        let m3 = auth.message3(parsed2.nonce);
        let m4 = sup.on_eapol(&m3).expect("message 4");
        assert_eq!(sup.state().tag(), "ptk_done");

        let parsed4 = EapolKey::decode(&m4).expect("decode m4");
        assert!(parsed4.key_info.secure());
        assert!(parsed4.key_data.is_empty());
        verify_mic(&expected.kck, &m4, &parsed4.key_mic).expect("message 4 MIC");
        assert_eq!(sup.ptk(), Some(&expected));
    }

    #[test]
    fn replayed_message1_is_refused() {
        let (mut sup, mut auth, _) = handshake_pair();
        let m1 = auth.message1();
        sup.on_eapol(&m1).expect("message 2");
        assert_eq!(
            sup.on_eapol(&m1),
            Err(Wpa2Error::ReplayCounterReuse { got: 1, last: 1 })
        );
        assert_eq!(sup.state().tag(), "failed");
    }

    #[test]
    fn message3_replaying_message1s_counter_is_refused() {
        let (mut sup, mut auth, _) = handshake_pair();
        let m1 = auth.message1();
        let m2 = sup.on_eapol(&m1).expect("message 2");
        let snonce = EapolKey::decode(&m2).expect("m2").nonce;
        // Roll the authenticator's counter back so message 3 reuses message 1's.
        auth.replay = 0;
        let m3 = auth.message3(snonce);
        assert_eq!(
            sup.on_eapol(&m3),
            Err(Wpa2Error::ReplayCounterReuse { got: 1, last: 1 })
        );
        assert_eq!(sup.state().tag(), "failed");
    }

    #[test]
    fn message3_with_a_bad_mic_is_refused_and_the_counter_does_not_advance() {
        let (mut sup, mut auth, _) = handshake_pair();
        let m1 = auth.message1();
        let m2 = sup.on_eapol(&m1).expect("message 2");
        let snonce = EapolKey::decode(&m2).expect("m2").nonce;
        let mut m3 = auth.message3(snonce);
        m3[MIC_OFFSET] ^= 0x80;
        assert_eq!(sup.on_eapol(&m3), Err(Wpa2Error::MicMismatch));
        assert_eq!(sup.state().tag(), "failed");
        // Fail-closed: a genuine message 3 afterwards is still refused.
        let good = auth.message3(snonce);
        assert_eq!(sup.on_eapol(&good), Err(Wpa2Error::HandshakeFailed));
        assert!(sup.ptk().is_none());
    }

    #[test]
    fn message3_with_a_different_anonce_is_refused() {
        let (mut sup, mut auth, _) = handshake_pair();
        let m1 = auth.message1();
        let m2 = sup.on_eapol(&m1).expect("message 2");
        let snonce = EapolKey::decode(&m2).expect("m2").nonce;
        // A new ANonce yields a different PTK, so the MIC fails before the ANonce check
        // ever runs — which is itself the security property worth asserting.
        auth.anonce = [0x77; 32];
        let m3 = auth.message3(snonce);
        assert_eq!(sup.on_eapol(&m3), Err(Wpa2Error::MicMismatch));
    }

    #[test]
    fn message3_before_message1_is_refused() {
        let (mut sup, mut auth, _) = handshake_pair();
        let m3 = auth.message3([0u8; 32]);
        assert_eq!(
            sup.on_eapol(&m3),
            Err(Wpa2Error::UnexpectedMessage { state: "idle" })
        );
    }

    #[test]
    fn encrypted_key_data_is_refused_not_silently_dropped() {
        let (mut sup, mut auth, _) = handshake_pair();
        let m1 = auth.message1();
        let m2 = sup.on_eapol(&m1).expect("message 2");
        let snonce = EapolKey::decode(&m2).expect("m2").nonce;
        let ptk = derive_ptk(&auth.pmk, auth.aa, auth.spa, &auth.anonce, &snonce).expect("ptk");
        auth.replay += 1;
        // Version 2, Pairwise, Install, Ack, MIC, Secure, Encrypted Key Data.
        let k = sample_key(0x13ca, auth.replay, auth.anonce, vec![0xde; 56]);
        let m3 = k.encode_with_mic(&ptk.kck).expect("mic");
        assert_eq!(sup.on_eapol(&m3), Err(Wpa2Error::KeyDataEncrypted));
    }

    #[test]
    fn unsupported_key_descriptor_versions_are_refused() {
        let (mut sup, mut auth, _) = handshake_pair();
        let mut m1 = auth.message1();
        // Key Information is at offset 5, big-endian; drop the version to 1 (MD5/RC4).
        m1[6] = (m1[6] & 0xf8) | 1;
        assert_eq!(
            sup.on_eapol(&m1),
            Err(Wpa2Error::UnsupportedDescriptorVersion(1))
        );
        let (mut sup, mut auth, _) = handshake_pair();
        let mut m1 = auth.message1();
        m1[6] = (m1[6] & 0xf8) | 3; // AES-128-CMAC
        assert_eq!(
            sup.on_eapol(&m1),
            Err(Wpa2Error::UnsupportedDescriptorVersion(3))
        );
    }

    #[test]
    fn garbage_never_panics_the_supplicant() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut buf = [0u8; 128];
        for _ in 0..20_000 {
            for b in buf.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *b = state as u8;
            }
            let (mut sup, _, _) = handshake_pair();
            for len in [0usize, 1, 4, 50, 99, 128] {
                let _ = sup.on_eapol(&buf[..len]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // CCMP
    // -----------------------------------------------------------------------

    fn data_header(qos: bool) -> MacHeader {
        let mut fc = FrameControl::new(FrameType::Data, if qos { 8 } else { 0 });
        fc.to_ds = true;
        fc.protected = true;
        MacHeader {
            fc,
            duration: 0x002c,
            addr1: MacAddr([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]),
            addr2: MacAddr([0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb]),
            addr3: MacAddr([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]),
            seq_ctrl: SequenceControl {
                fragment: 0,
                sequence: 42,
            },
            addr4: None,
            qos_ctrl: if qos { Some(0x0006) } else { None },
            ht_ctrl: None,
        }
    }

    #[test]
    fn ccmp_round_trips() {
        let tk = [0x11u8; 16];
        let h = data_header(false);
        let plaintext = b"the section is missing, and that is the honest answer";
        let body = ccmp_encrypt(&tk, &h, 0, 1, plaintext).expect("encrypt");
        assert_eq!(body.len(), 8 + plaintext.len() + 8);
        let (got, pn) = ccmp_decrypt(&tk, &h, &body).expect("decrypt");
        assert_eq!(got, plaintext);
        assert_eq!(pn, 1);
    }

    #[test]
    fn ccmp_header_encodes_the_packet_number_in_the_split_layout() {
        // 802.11-2016 Figure 12-16: PN0, PN1, Rsvd, KeyID|ExtIV, PN2, PN3, PN4, PN5.
        let h = ccmp_header(0x0605_0403_0201, 2);
        assert_eq!(h, [0x01, 0x02, 0x00, 0xa0, 0x03, 0x04, 0x05, 0x06]);
        let (pn, key_id) = parse_ccmp_header(&h).expect("parse");
        assert_eq!(pn, 0x0605_0403_0201);
        assert_eq!(key_id, 2);
    }

    #[test]
    fn ccmp_nonce_and_aad_have_the_documented_shape() {
        let h = data_header(true);
        let n = ccm_nonce(&h, 0x0605_0403_0201);
        assert_eq!(n[0], 6, "nonce flags carry the TID from QoS Control");
        assert_eq!(&n[1..7], &h.addr2.0);
        assert_eq!(&n[7..13], &[0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);

        let aad = ccm_aad(&h);
        assert_eq!(aad.len(), 24, "3 addresses + QoS Control");
        let fc = u16::from_le_bytes([aad[0], aad[1]]);
        assert_eq!(fc & 0x4000, 0x4000, "protected bit forced on");
        assert_eq!(fc & 0x0070, 0, "high subtype bits masked on data frames");
        assert_eq!(&aad[20..22], &[0x00, 0x00], "sequence number masked out");
        assert_eq!(&aad[22..24], &[0x06, 0x00], "QoS masked to the TID");

        let no_qos = ccm_aad(&data_header(false));
        assert_eq!(no_qos.len(), 22);
    }

    #[test]
    fn ccmp_masked_header_bits_do_not_change_the_mic() {
        // Retry, Power Management and More Data flip in flight on legitimate frames.
        let tk = [0x11u8; 16];
        let h = data_header(false);
        let body = ccmp_encrypt(&tk, &h, 0, 7, b"payload").expect("encrypt");
        let mut retried = h.clone();
        retried.fc.retry = true;
        retried.fc.power_management = true;
        retried.fc.more_data = true;
        let (got, _) = ccmp_decrypt(&tk, &retried, &body).expect("retransmission must verify");
        assert_eq!(got, b"payload");
    }

    #[test]
    fn ccmp_covered_header_bits_do_change_the_mic() {
        let tk = [0x11u8; 16];
        let h = data_header(false);
        let body = ccmp_encrypt(&tk, &h, 0, 7, b"payload").expect("encrypt");
        let mut moved = h.clone();
        moved.addr1 = MacAddr([0xde, 0xad, 0xbe, 0xef, 0x00, 0x01]);
        assert_eq!(
            ccmp_decrypt(&tk, &moved, &body),
            Err(Wpa2Error::CcmpMicMismatch),
            "redirecting addr1 must break the MIC"
        );
        let mut refragmented = h.clone();
        refragmented.seq_ctrl.fragment = 1;
        assert_eq!(
            ccmp_decrypt(&tk, &refragmented, &body),
            Err(Wpa2Error::CcmpMicMismatch),
            "fragment number is covered by the AAD"
        );
    }

    #[test]
    fn ccmp_tampering_with_authenticated_octets_yields_no_plaintext() {
        let tk = [0x11u8; 16];
        let h = data_header(false);
        let body = ccmp_encrypt(&tk, &h, 0, 3, b"payload").expect("encrypt");
        // Octets 2 and 3 of the CCMP header are Reserved and Key ID; neither enters
        // the nonce nor the AAD, so neither is authenticated. That is CCMP, not a
        // gap here, and the next test pins it down. Every other octet is covered.
        for i in (0..body.len()).filter(|i| *i != 2 && *i != 3) {
            let mut bad = body.clone();
            bad[i] ^= 0x01;
            match ccmp_decrypt(&tk, &h, &bad) {
                Err(_) => {}
                Ok((pt, _)) => panic!("tampered octet {i} decrypted to {pt:?}"),
            }
        }
    }

    /// Pinned down so that nobody later "hardens" it and breaks interoperability:
    /// 802.11-2016 Figure 12-16 leaves CCMP header octet 2 and bits 0–4 of octet 3
    /// Reserved, and 12.5.3.3 puts neither in the nonce nor the AAD. They are
    /// genuinely unauthenticated and a receiver must not reject on them.
    #[test]
    fn ccmp_reserved_header_octets_are_not_authenticated() {
        let tk = [0x11u8; 16];
        let h = data_header(false);
        let mut body = ccmp_encrypt(&tk, &h, 0, 3, b"payload").expect("encrypt");
        body[2] ^= 0xff;
        body[3] ^= 0x01;
        let (pt, pn) = ccmp_decrypt(&tk, &h, &body).expect("reserved octets are not covered");
        assert_eq!(pt, b"payload");
        assert_eq!(pn, 3);
    }

    #[test]
    fn ccmp_rejects_short_bodies_and_missing_extiv() {
        let tk = [0x11u8; 16];
        let h = data_header(false);
        for n in 0..16usize {
            assert_eq!(
                ccmp_decrypt(&tk, &h, &vec![0u8; n]),
                Err(Wpa2Error::CcmpTooShort(n))
            );
        }
        let mut body = ccmp_encrypt(&tk, &h, 0, 1, b"x").expect("encrypt");
        body[3] &= !0x20;
        assert_eq!(ccmp_decrypt(&tk, &h, &body), Err(Wpa2Error::CcmpNoExtIv));
    }

    #[test]
    fn ccmp_pn_beyond_48_bits_is_refused() {
        let tk = [0x11u8; 16];
        let h = data_header(false);
        assert_eq!(
            ccmp_encrypt(&tk, &h, 0, 1 << 48, b"x"),
            Err(Wpa2Error::CcmpPnOverflow(1 << 48))
        );
    }

    #[test]
    fn ccmp_replay_counter_is_strict_monotonic() {
        let mut r = CcmpReplay::default();
        r.accept(1).expect("first");
        r.accept(2).expect("second");
        assert_eq!(
            r.accept(2),
            Err(Wpa2Error::CcmpPnReplay { got: 2, last: 2 })
        );
        assert_eq!(
            r.accept(1),
            Err(Wpa2Error::CcmpPnReplay { got: 1, last: 2 })
        );
        assert_eq!(r.last(), Some(2));
        r.accept(3).expect("third");
    }
}
