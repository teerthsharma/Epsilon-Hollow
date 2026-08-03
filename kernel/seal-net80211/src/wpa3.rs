// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! WPA3-SAE — message codec, key derivation, and handshake state machine.
//!
//! # Read this before believing anything else in this file
//!
//! **The group arithmetic is not implemented and this handshake cannot complete.**
//! SAE is a dragonfly PAKE over a finite cyclic group. Two steps need arithmetic in
//! that group:
//!
//! 1. Deriving the password element PWE from the password and the two MAC addresses.
//! 2. Computing the shared secret `K = scalar-op(rand, element-op(peer-Element,
//!    scalar-op(peer-scalar, PWE)))`.
//!
//! Both need elliptic-curve point arithmetic over NIST P-256, which needs a
//! big-integer and curve implementation. No such dependency is available to this
//! crate and hand-rolling one would be exactly the kind of thing this kernel's README
//! spends a chapter refusing to do. So:
//!
//! * [`derive_pwe`] and [`shared_secret`] exist, are the only two places the
//!   arithmetic would go, and both **unconditionally return**
//!   [`SaeError::GroupArithmeticNotImplemented`]. They contain no arithmetic, fake or
//!   otherwise.
//! * [`Sae::install_shared_secret`] is the seam. A caller that has a real P-256
//!   implementation supplies `K` and `(commit-scalar + peer-commit-scalar) mod r`, and
//!   everything downstream — keyseed, KCK, PMK, PMKID, Confirm — is real.
//! * With no such caller, [`Sae`] gets as far as `Committed` and stops. It never
//!   reaches `Accepted`, never produces a PMK, and there is no code path that pretends
//!   otherwise.
//!
//! Neither hash-to-element (RFC 9380 / 802.11-2020 12.4.4.3.3) nor hunting-and-pecking
//! (802.11-2016 12.4.4.2.2) is implemented. The question of which to prefer does not
//! arise, because implementing either requires the field arithmetic named above. If
//! one is ever added it should be hash-to-element: hunting-and-pecking's loop count
//! depends on the password, which is the timing side channel Dragonblood exploited.
//!
//! # What IS implemented, and is real
//!
//! * SAE Commit and Confirm message codecs, encode and decode, bounds-checked
//!   (802.11-2016 9.4.1.35–9.4.1.39, field order per 12.4.7.4/12.4.7.5).
//! * `KDF-Hash-Length` as 802.11-2016 12.7.1.6.2 defines it. This is **not** HKDF: it
//!   has no extract step and no `T(i-1)` chaining, so RFC 5869 vectors do not apply
//!   to it and none are claimed here.
//! * keyseed, KCK, PMK and PMKID derivation from a supplied `K` (12.4.5.4).
//! * Confirm computation and verification, with the operand order swapped for the
//!   peer's confirm, compared in constant time via `subtle` through
//!   `Mac::verify_slice`.
//! * The four-state machine of 12.4.8.1, with the reflection check of 12.4.5.4 and
//!   the scalar range check `1 < scalar < r`.
//!
//! # Test vector status: none
//!
//! **No published SAE test vector is gated here.** 802.11-2020 Annex J.10 contains
//! SAE vectors; they could not be quoted with confidence and inventing one would be
//! worse than admitting the gap. Everything in this module is covered by round-trip,
//! bounds and property tests only, which is weaker evidence than the WPA2 path has and
//! is stated here rather than buried.
//!
//! # Other limits
//!
//! * Only group 19 (NIST P-256, SHA-256) is accepted. Groups 20 and 21 need SHA-384
//!   and SHA-512 keyed differently, and the MODP groups need prime lengths this module
//!   does not carry. Everything else is [`SaeError::UnsupportedGroup`].
//! * **Element validation is not performed.** A peer's Element is accepted on length
//!   alone: not checked to be on the curve, not checked against the identity, not
//!   checked that its coordinates are below the field prime. All three need field
//!   arithmetic. A caller that adds the arithmetic must add these checks with it —
//!   an unvalidated point is an invalid-curve attack.
//! * The optional Password Identifier element is not supported; a Commit carrying one
//!   decodes to [`SaeError::TrailingOctets`] rather than being misread.
//! * No SAE-PK, no confirm-counter/retransmission logic, no Sync counter, no
//!   dueling-commit resolution.

use alloc::vec::Vec;
use core::fmt;

use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::frame::Reader;

type HmacSha256 = Hmac<Sha256>;

/// Status code 76: the AP demands an anti-clogging token before it will do the work
/// of a Commit (802.11-2016 Table 9-46, `ANTI_CLOGGING_TOKEN_REQUIRED`).
pub const STATUS_ANTI_CLOGGING_TOKEN_REQUIRED: u16 = 76;
/// Status code 77: the group in the Commit is not one the peer supports.
pub const STATUS_UNSUPPORTED_FINITE_CYCLIC_GROUP: u16 = 77;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeError {
    /// Group is not 19, or is one whose parameters this module does not carry.
    UnsupportedGroup(u16),
    Truncated {
        need: usize,
        got: usize,
    },
    /// Octets remained after the last field the codec understands. Most likely a
    /// Password Identifier element, which is not supported.
    TrailingOctets(usize),
    /// Scalar is not in `1 < scalar < r`, where r is the group order (12.4.5.4).
    ScalarOutOfRange,
    /// The peer echoed our own scalar and element back at us — a reflection attack.
    ReflectedCommit,
    /// Peer's Confirm did not match the value computed from the KCK.
    ConfirmMismatch,
    ConfirmLength {
        expected: usize,
        got: usize,
    },
    /// A message arrived that the current state does not accept.
    UnexpectedState(&'static str),
    /// Keys have not been derived, because no shared secret has been installed.
    NoKeys,
    /// The elliptic-curve arithmetic SAE needs is not implemented. See the module
    /// documentation; this is not a transient failure.
    GroupArithmeticNotImplemented,
    /// A RustCrypto constructor rejected a key length. Unreachable with fixed-size
    /// keys; present so no call site needs `unwrap`.
    KeySetup,
}

impl fmt::Display for SaeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedGroup(g) => write!(f, "finite cyclic group {g} unsupported"),
            Self::Truncated { need, got } => {
                write!(f, "truncated: need {need} octets, got {got}")
            }
            Self::TrailingOctets(n) => write!(f, "{n} unparsed octets after the last field"),
            Self::ScalarOutOfRange => write!(f, "commit scalar outside 1 < scalar < r"),
            Self::ReflectedCommit => write!(f, "peer reflected our own commit"),
            Self::ConfirmMismatch => write!(f, "SAE confirm mismatch"),
            Self::ConfirmLength { expected, got } => {
                write!(f, "confirm of {got} octets, expected {expected}")
            }
            Self::UnexpectedState(s) => write!(f, "unexpected SAE message in state {s}"),
            Self::NoKeys => write!(f, "no SAE keys: shared secret not installed"),
            Self::GroupArithmeticNotImplemented => {
                write!(f, "SAE group arithmetic is not implemented in this crate")
            }
            Self::KeySetup => write!(f, "key setup rejected by MAC"),
        }
    }
}

impl From<crate::frame::FrameError> for SaeError {
    fn from(e: crate::frame::FrameError) -> Self {
        match e {
            crate::frame::FrameError::Truncated { need, got } => Self::Truncated { need, got },
            _ => Self::Truncated { need: 0, got: 0 },
        }
    }
}

// ---------------------------------------------------------------------------
// Group parameters
// ---------------------------------------------------------------------------

/// A finite cyclic group identifier from the IANA "Group Description" registry, which
/// is what 802.11-2016 9.4.1.41 points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Group(pub u16);

/// Order r of NIST P-256 (SEC 2 / FIPS 186-4), big-endian.
const P256_ORDER: [u8; 32] = [
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xbc, 0xe6, 0xfa, 0xad, 0xa7, 0x17, 0x9e, 0x84, 0xf3, 0xb9, 0xca, 0xc2, 0xfc, 0x63, 0x25, 0x51,
];

impl Group {
    /// Group 19: 256-bit random ECP group, NIST P-256 (RFC 5903).
    pub const P256: Self = Self(19);

    /// Octets in a scalar — the length of the group order.
    pub fn scalar_len(self) -> Option<usize> {
        match self.0 {
            19 => Some(32),
            _ => None,
        }
    }

    /// Octets in an element. For an ECP group this is an uncompressed affine point
    /// without the leading format octet: `x || y`.
    pub fn element_len(self) -> Option<usize> {
        self.scalar_len().map(|n| n * 2)
    }

    /// Octets of hash output. SHA-256 for group 19.
    pub fn hash_len(self) -> Option<usize> {
        match self.0 {
            19 => Some(32),
            _ => None,
        }
    }

    fn order(self) -> Option<&'static [u8; 32]> {
        match self.0 {
            19 => Some(&P256_ORDER),
            _ => None,
        }
    }

    fn require(self) -> Result<(usize, usize), SaeError> {
        match (self.scalar_len(), self.element_len()) {
            (Some(s), Some(e)) => Ok((s, e)),
            _ => Err(SaeError::UnsupportedGroup(self.0)),
        }
    }
}

/// `1 < scalar < r`, per 802.11-2016 12.4.5.4. Byte-wise comparison of a big-endian
/// integer against the group order — a comparison, not arithmetic.
fn scalar_in_range(group: Group, scalar: &[u8]) -> bool {
    let Some(order) = group.order() else {
        return false;
    };
    if scalar.len() != order.len() {
        return false;
    }
    // Reject 0 and 1.
    let leading_zero = scalar[..scalar.len() - 1].iter().all(|b| *b == 0);
    if leading_zero && scalar[scalar.len() - 1] <= 1 {
        return false;
    }
    scalar < &order[..]
}

// ---------------------------------------------------------------------------
// Message codec
// ---------------------------------------------------------------------------

/// SAE Commit. These octets sit in an Authentication frame body with Algorithm 3 and
/// Transaction Sequence 1, after the three fixed Authentication fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaeCommit {
    pub group: Group,
    /// Present only when replying to a `STATUS_ANTI_CLOGGING_TOKEN_REQUIRED`.
    pub anti_clogging_token: Vec<u8>,
    pub scalar: Vec<u8>,
    pub element: Vec<u8>,
}

impl SaeCommit {
    /// Decode a Commit body.
    ///
    /// `expected_token_len` must be supplied by the caller because the Anti-Clogging
    /// Token has no length prefix and sits between two fields of known size; only the
    /// station that asked for a token knows how long it is. Pass 0 when none was
    /// requested. Anything left over after the Element is
    /// [`SaeError::TrailingOctets`] — a Password Identifier lands here rather than
    /// being silently absorbed into a field.
    pub fn decode(body: &[u8], expected_token_len: usize) -> Result<Self, SaeError> {
        let mut r = Reader::new(body);
        let group = Group(r.le16()?);
        let (scalar_len, element_len) = group.require()?;
        let anti_clogging_token = r.take(expected_token_len)?.to_vec();
        let scalar = r.take(scalar_len)?.to_vec();
        let element = r.take(element_len)?.to_vec();
        if !r.is_empty() {
            return Err(SaeError::TrailingOctets(r.remaining()));
        }
        Ok(Self {
            group,
            anti_clogging_token,
            scalar,
            element,
        })
    }

    pub fn encode(&self) -> Result<Vec<u8>, SaeError> {
        let (scalar_len, element_len) = self.group.require()?;
        if self.scalar.len() != scalar_len {
            return Err(SaeError::Truncated {
                need: scalar_len,
                got: self.scalar.len(),
            });
        }
        if self.element.len() != element_len {
            return Err(SaeError::Truncated {
                need: element_len,
                got: self.element.len(),
            });
        }
        let mut out =
            Vec::with_capacity(2 + self.anti_clogging_token.len() + scalar_len + element_len);
        out.extend_from_slice(&self.group.0.to_le_bytes());
        out.extend_from_slice(&self.anti_clogging_token);
        out.extend_from_slice(&self.scalar);
        out.extend_from_slice(&self.element);
        Ok(out)
    }
}

/// SAE Confirm: Authentication frame with Algorithm 3 and Transaction Sequence 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaeConfirm {
    pub send_confirm: u16,
    pub confirm: Vec<u8>,
}

impl SaeConfirm {
    pub fn decode(body: &[u8], group: Group) -> Result<Self, SaeError> {
        let hash_len = group
            .hash_len()
            .ok_or(SaeError::UnsupportedGroup(group.0))?;
        let mut r = Reader::new(body);
        let send_confirm = r.le16()?;
        let confirm = r.take(hash_len)?.to_vec();
        if !r.is_empty() {
            return Err(SaeError::TrailingOctets(r.remaining()));
        }
        Ok(Self {
            send_confirm,
            confirm,
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + self.confirm.len());
        out.extend_from_slice(&self.send_confirm.to_le_bytes());
        out.extend_from_slice(&self.confirm);
        out
    }
}

// ---------------------------------------------------------------------------
// Key derivation
// ---------------------------------------------------------------------------

fn hmac_sha256(key: &[u8], chunks: &[&[u8]]) -> Result<[u8; 32], SaeError> {
    let mut mac = HmacSha256::new_from_slice(key).map_err(|_| SaeError::KeySetup)?;
    for c in chunks {
        mac.update(c);
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&mac.finalize().into_bytes());
    Ok(out)
}

/// `KDF-Hash-Length` of 802.11-2016 12.7.1.6.2, with Hash = SHA-256.
///
/// `result = HMAC-SHA-256(K, i || label || context || Length)` concatenated for
/// `i = 1, 2, ...`, truncated to `bits`. Both `i` and `Length` are 16-bit
/// little-endian. This is not HKDF-Expand and RFC 5869 vectors do not apply.
pub fn kdf_sha256(
    key: &[u8],
    label: &[u8],
    context: &[u8],
    bits: u16,
) -> Result<Vec<u8>, SaeError> {
    let bytes = (bits as usize).div_ceil(8);
    let iterations = bytes.div_ceil(32);
    let mut out = Vec::with_capacity(iterations * 32);
    for i in 1..=iterations {
        let counter = (i as u16).to_le_bytes();
        let length = bits.to_le_bytes();
        out.extend_from_slice(&hmac_sha256(key, &[&counter, label, context, &length])?);
    }
    out.truncate(bytes);
    Ok(out)
}

/// The keys SAE produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeKeys {
    /// Key Confirmation Key — keys the Confirm exchange.
    pub kck: [u8; 32],
    /// The PMK handed to the 4-way handshake in `crate::wpa2`.
    pub pmk: [u8; 32],
    pub pmkid: [u8; 16],
}

/// Derive KCK, PMK and PMKID from the SAE shared secret (802.11-2016 12.4.5.4).
///
/// ```text
/// keyseed        = HMAC-SHA-256(<0>32, k)
/// KCK || PMK     = KDF-Hash-512(keyseed, "SAE KCK and PMK", (r1 + r2) mod r)
/// PMKID          = L((r1 + r2) mod r, 0, 128)
/// ```
///
/// `k` and `scalar_sum` both come from group arithmetic this crate does not perform;
/// see [`shared_secret`]. Everything from here down is real and testable.
pub fn derive_keys(k: &[u8], scalar_sum: &[u8]) -> Result<SaeKeys, SaeError> {
    let keyseed = hmac_sha256(&[0u8; 32], &[k])?;
    let out = kdf_sha256(&keyseed, b"SAE KCK and PMK", scalar_sum, 512)?;
    if out.len() < 64 || scalar_sum.len() < 16 {
        return Err(SaeError::NoKeys);
    }
    let mut keys = SaeKeys {
        kck: [0u8; 32],
        pmk: [0u8; 32],
        pmkid: [0u8; 16],
    };
    keys.kck.copy_from_slice(&out[0..32]);
    keys.pmk.copy_from_slice(&out[32..64]);
    keys.pmkid.copy_from_slice(&scalar_sum[..16]);
    Ok(keys)
}

/// `CN(key, X, Y, Z, ...) = HMAC-SHA-256(key, X || Y || Z || ...)` — 802.11-2016
/// 12.4.5.5. `send_confirm` is 16-bit little-endian.
pub fn confirm_value(
    kck: &[u8; 32],
    send_confirm: u16,
    scalar1: &[u8],
    element1: &[u8],
    scalar2: &[u8],
    element2: &[u8],
) -> Result<[u8; 32], SaeError> {
    hmac_sha256(
        kck,
        &[
            &send_confirm.to_le_bytes(),
            scalar1,
            element1,
            scalar2,
            element2,
        ],
    )
}

/// Verify a peer's Confirm in constant time.
///
/// `Mac::verify_slice` is the `subtle` comparison. Comparing a confirm value with `==`
/// hands an attacker a byte-at-a-time oracle against the KCK.
#[allow(clippy::too_many_arguments)]
pub fn verify_confirm(
    kck: &[u8; 32],
    send_confirm: u16,
    peer_scalar: &[u8],
    peer_element: &[u8],
    own_scalar: &[u8],
    own_element: &[u8],
    peer_confirm: &[u8],
) -> Result<(), SaeError> {
    if peer_confirm.len() != 32 {
        return Err(SaeError::ConfirmLength {
            expected: 32,
            got: peer_confirm.len(),
        });
    }
    let mut mac = HmacSha256::new_from_slice(kck).map_err(|_| SaeError::KeySetup)?;
    mac.update(&send_confirm.to_le_bytes());
    mac.update(peer_scalar);
    mac.update(peer_element);
    mac.update(own_scalar);
    mac.update(own_element);
    mac.verify_slice(peer_confirm)
        .map_err(|_| SaeError::ConfirmMismatch)
}

// ---------------------------------------------------------------------------
// The two stubs. Neither contains arithmetic, fake or otherwise.
// ---------------------------------------------------------------------------

/// Derive the password element PWE.
///
/// **Not implemented.** Always returns [`SaeError::GroupArithmeticNotImplemented`].
/// Both routes to PWE — hash-to-element (802.11-2020 12.4.4.3.3, RFC 9380 style) and
/// hunting-and-pecking (802.11-2016 12.4.4.2.2) — require arithmetic in the group.
pub fn derive_pwe(
    _group: Group,
    _password: &[u8],
    _mac_a: &[u8; 6],
    _mac_b: &[u8; 6],
) -> Result<Vec<u8>, SaeError> {
    Err(SaeError::GroupArithmeticNotImplemented)
}

/// Compute the SAE shared secret `K`.
///
/// **Not implemented.** Always returns [`SaeError::GroupArithmeticNotImplemented`].
/// `K = scalar-op(rand, element-op(peer-Element, scalar-op(peer-scalar, PWE)))`, which
/// is two P-256 scalar multiplications and a point addition.
pub fn shared_secret(
    _group: Group,
    _rand: &[u8],
    _pwe: &[u8],
    _peer_scalar: &[u8],
    _peer_element: &[u8],
) -> Result<Vec<u8>, SaeError> {
    Err(SaeError::GroupArithmeticNotImplemented)
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// SAE protocol states (802.11-2016 12.4.8.1). `Nothing` and `Accepted` are the two
/// ends; nothing shipped in this crate can reach `Accepted` on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeState {
    Nothing,
    Committed,
    Confirmed,
    Accepted,
    Failed,
}

impl SaeState {
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Nothing => "nothing",
            Self::Committed => "committed",
            Self::Confirmed => "confirmed",
            Self::Accepted => "accepted",
            Self::Failed => "failed",
        }
    }
}

/// SAE handshake driver.
///
/// The caller supplies its own commit scalar and element, and — at
/// [`Sae::install_shared_secret`] — the products of the group arithmetic. Without
/// that call the machine stops at `Committed` and no PMK exists.
pub struct Sae {
    group: Group,
    state: SaeState,
    own: Option<(Vec<u8>, Vec<u8>)>,
    peer: Option<(Vec<u8>, Vec<u8>)>,
    keys: Option<SaeKeys>,
    send_confirm: u16,
}

impl Sae {
    pub fn new(group: Group) -> Result<Self, SaeError> {
        group.require()?;
        Ok(Self {
            group,
            state: SaeState::Nothing,
            own: None,
            peer: None,
            keys: None,
            send_confirm: 0,
        })
    }

    pub fn state(&self) -> SaeState {
        self.state
    }

    pub fn group(&self) -> Group {
        self.group
    }

    /// Record our own commit values and produce the Commit body to transmit.
    ///
    /// `scalar` and `element` come from group arithmetic the caller performs; this
    /// crate cannot produce them (see [`derive_pwe`]).
    pub fn commit(
        &mut self,
        scalar: Vec<u8>,
        element: Vec<u8>,
        anti_clogging_token: Vec<u8>,
    ) -> Result<Vec<u8>, SaeError> {
        if self.state != SaeState::Nothing {
            return Err(SaeError::UnexpectedState(self.state.tag()));
        }
        if !scalar_in_range(self.group, &scalar) {
            return Err(SaeError::ScalarOutOfRange);
        }
        let msg = SaeCommit {
            group: self.group,
            anti_clogging_token,
            scalar: scalar.clone(),
            element: element.clone(),
        };
        let bytes = msg.encode()?;
        self.own = Some((scalar, element));
        self.state = SaeState::Committed;
        Ok(bytes)
    }

    /// Accept the peer's Commit.
    pub fn on_commit(&mut self, peer: &SaeCommit) -> Result<(), SaeError> {
        if self.state != SaeState::Committed {
            return Err(SaeError::UnexpectedState(self.state.tag()));
        }
        if peer.group != self.group {
            return Err(SaeError::UnsupportedGroup(peer.group.0));
        }
        if !scalar_in_range(self.group, &peer.scalar) {
            return Err(SaeError::ScalarOutOfRange);
        }
        let (_, element_len) = self.group.require()?;
        if peer.element.len() != element_len {
            return Err(SaeError::Truncated {
                need: element_len,
                got: peer.element.len(),
            });
        }
        // 802.11-2016 12.4.5.4: a peer echoing our own scalar and element is a
        // reflection attack, and the frame is discarded.
        if let Some((s, e)) = &self.own {
            if *s == peer.scalar && *e == peer.element {
                self.state = SaeState::Failed;
                return Err(SaeError::ReflectedCommit);
            }
        }
        self.peer = Some((peer.scalar.clone(), peer.element.clone()));
        Ok(())
    }

    /// Install the products of the group arithmetic and derive KCK, PMK and PMKID.
    ///
    /// This is the seam. `k` is the shared secret; `scalar_sum` is
    /// `(commit-scalar + peer-commit-scalar) mod r`. Both would come from
    /// [`shared_secret`] and a modular addition, neither of which this crate has.
    pub fn install_shared_secret(&mut self, k: &[u8], scalar_sum: &[u8]) -> Result<(), SaeError> {
        if self.state != SaeState::Committed || self.peer.is_none() {
            return Err(SaeError::UnexpectedState(self.state.tag()));
        }
        self.keys = Some(derive_keys(k, scalar_sum)?);
        Ok(())
    }

    pub fn keys(&self) -> Option<&SaeKeys> {
        // Only surfaced once the peer's Confirm has been verified. Before that the
        // PMK is unauthenticated and handing it out would defeat the exchange.
        match self.state {
            SaeState::Accepted => self.keys.as_ref(),
            _ => None,
        }
    }

    /// Build our Confirm message. Requires that the shared secret has been installed.
    pub fn confirm(&mut self) -> Result<Vec<u8>, SaeError> {
        if self.state != SaeState::Committed {
            return Err(SaeError::UnexpectedState(self.state.tag()));
        }
        let keys = self.keys.as_ref().ok_or(SaeError::NoKeys)?;
        let (own_s, own_e) = self.own.as_ref().ok_or(SaeError::NoKeys)?;
        let (peer_s, peer_e) = self.peer.as_ref().ok_or(SaeError::NoKeys)?;
        let value = confirm_value(&keys.kck, self.send_confirm, own_s, own_e, peer_s, peer_e)?;
        let msg = SaeConfirm {
            send_confirm: self.send_confirm,
            confirm: value.to_vec(),
        };
        self.state = SaeState::Confirmed;
        Ok(msg.encode())
    }

    /// Verify the peer's Confirm. Success is the only path to `Accepted`.
    pub fn on_confirm(&mut self, peer: &SaeConfirm) -> Result<(), SaeError> {
        if self.state != SaeState::Confirmed {
            return Err(SaeError::UnexpectedState(self.state.tag()));
        }
        let keys = self.keys.as_ref().ok_or(SaeError::NoKeys)?;
        let (own_s, own_e) = self.own.as_ref().ok_or(SaeError::NoKeys)?;
        let (peer_s, peer_e) = self.peer.as_ref().ok_or(SaeError::NoKeys)?;
        // Operand order is swapped relative to our own confirm (12.4.5.5).
        let r = verify_confirm(
            &keys.kck,
            peer.send_confirm,
            peer_s,
            peer_e,
            own_s,
            own_e,
            &peer.confirm,
        );
        match r {
            Ok(()) => {
                self.state = SaeState::Accepted;
                Ok(())
            }
            Err(e) => {
                self.state = SaeState::Failed;
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn scalar(seed: u8) -> Vec<u8> {
        let mut s = vec![0u8; 32];
        s[31] = seed;
        s[0] = 0x11;
        s
    }

    fn element(seed: u8) -> Vec<u8> {
        vec![seed; 64]
    }

    #[test]
    fn the_group_arithmetic_stubs_never_succeed() {
        // If either of these ever returns Ok, this module has started lying.
        assert_eq!(
            derive_pwe(Group::P256, b"password", &[0; 6], &[1; 6]),
            Err(SaeError::GroupArithmeticNotImplemented)
        );
        assert_eq!(
            shared_secret(Group::P256, &[1; 32], &[2; 64], &[3; 32], &[4; 64]),
            Err(SaeError::GroupArithmeticNotImplemented)
        );
    }

    #[test]
    fn a_handshake_driven_only_by_this_crate_cannot_reach_accepted() {
        // The whole honesty claim of this module in one assertion: with no external
        // group implementation there is no route to a PMK.
        let mut sae = Sae::new(Group::P256).expect("group 19");
        sae.commit(scalar(2), element(0xaa), Vec::new())
            .expect("commit");
        let peer = SaeCommit {
            group: Group::P256,
            anti_clogging_token: Vec::new(),
            scalar: scalar(3),
            element: element(0xbb),
        };
        sae.on_commit(&peer).expect("peer commit");
        assert_eq!(sae.confirm(), Err(SaeError::NoKeys));
        assert_eq!(sae.state(), SaeState::Committed);
        assert!(sae.keys().is_none());
    }

    #[test]
    fn commit_round_trips() {
        let c = SaeCommit {
            group: Group::P256,
            anti_clogging_token: Vec::new(),
            scalar: scalar(7),
            element: element(0x5a),
        };
        let bytes = c.encode().expect("encode");
        assert_eq!(bytes.len(), 2 + 32 + 64);
        assert_eq!(&bytes[0..2], &[19, 0], "group id is little-endian");
        assert_eq!(SaeCommit::decode(&bytes, 0).expect("decode"), c);
    }

    #[test]
    fn commit_with_anti_clogging_token_round_trips() {
        let c = SaeCommit {
            group: Group::P256,
            anti_clogging_token: vec![0xde, 0xad, 0xbe, 0xef],
            scalar: scalar(7),
            element: element(0x5a),
        };
        let bytes = c.encode().expect("encode");
        assert_eq!(bytes.len(), 2 + 4 + 32 + 64);
        assert_eq!(SaeCommit::decode(&bytes, 4).expect("decode"), c);
        // Decoded with the wrong token length the fields shift and the frame no longer
        // fits; it must be refused rather than silently misaligned.
        assert!(SaeCommit::decode(&bytes, 0).is_err());
    }

    #[test]
    fn confirm_round_trips() {
        let c = SaeConfirm {
            send_confirm: 1,
            confirm: vec![0x42; 32],
        };
        let bytes = c.encode();
        assert_eq!(bytes.len(), 34);
        assert_eq!(SaeConfirm::decode(&bytes, Group::P256).expect("decode"), c);
    }

    #[test]
    fn every_truncation_of_a_commit_is_an_error_not_a_panic() {
        let c = SaeCommit {
            group: Group::P256,
            anti_clogging_token: Vec::new(),
            scalar: scalar(7),
            element: element(0x5a),
        };
        let bytes = c.encode().expect("encode");
        for n in 0..bytes.len() {
            assert!(
                SaeCommit::decode(&bytes[..n], 0).is_err(),
                "prefix {n} decoded"
            );
        }
        assert!(SaeCommit::decode(&bytes, 0).is_ok());
    }

    #[test]
    fn trailing_octets_are_refused_rather_than_absorbed() {
        let c = SaeCommit {
            group: Group::P256,
            anti_clogging_token: Vec::new(),
            scalar: scalar(7),
            element: element(0x5a),
        };
        let mut bytes = c.encode().expect("encode");
        bytes.extend_from_slice(&[0xff, 0x21, 0x03]); // a Password Identifier would look like this
        assert_eq!(
            SaeCommit::decode(&bytes, 0),
            Err(SaeError::TrailingOctets(3))
        );
    }

    #[test]
    fn unsupported_groups_are_refused() {
        assert_eq!(
            Sae::new(Group(20)).err(),
            Some(SaeError::UnsupportedGroup(20))
        );
        assert_eq!(
            Sae::new(Group(21)).err(),
            Some(SaeError::UnsupportedGroup(21))
        );
        assert_eq!(
            Sae::new(Group(14)).err(),
            Some(SaeError::UnsupportedGroup(14))
        );
        let mut body = vec![20u8, 0];
        body.extend_from_slice(&[0u8; 96]);
        assert_eq!(
            SaeCommit::decode(&body, 0),
            Err(SaeError::UnsupportedGroup(20))
        );
    }

    #[test]
    fn scalar_range_is_enforced_against_the_p256_order() {
        let mut zero = vec![0u8; 32];
        assert!(!scalar_in_range(Group::P256, &zero));
        zero[31] = 1;
        assert!(
            !scalar_in_range(Group::P256, &zero),
            "scalar 1 is out of range"
        );
        zero[31] = 2;
        assert!(scalar_in_range(Group::P256, &zero));
        // r itself and r+1 are both out of range.
        assert!(!scalar_in_range(Group::P256, &P256_ORDER));
        let mut over = P256_ORDER;
        over[31] = 0x52;
        assert!(!scalar_in_range(Group::P256, &over));
        let mut under = P256_ORDER;
        under[31] = 0x50;
        assert!(scalar_in_range(Group::P256, &under));
        assert!(!scalar_in_range(Group::P256, &[0xff; 31]), "wrong length");
    }

    #[test]
    fn a_reflected_commit_is_refused() {
        let mut sae = Sae::new(Group::P256).expect("group 19");
        sae.commit(scalar(9), element(0xcc), Vec::new())
            .expect("commit");
        let mirror = SaeCommit {
            group: Group::P256,
            anti_clogging_token: Vec::new(),
            scalar: scalar(9),
            element: element(0xcc),
        };
        assert_eq!(sae.on_commit(&mirror), Err(SaeError::ReflectedCommit));
        assert_eq!(sae.state(), SaeState::Failed);
    }

    #[test]
    fn out_of_range_peer_scalars_are_refused() {
        let mut sae = Sae::new(Group::P256).expect("group 19");
        sae.commit(scalar(9), element(0xcc), Vec::new())
            .expect("commit");
        let bad = SaeCommit {
            group: Group::P256,
            anti_clogging_token: Vec::new(),
            scalar: vec![0u8; 32],
            element: element(0xbb),
        };
        assert_eq!(sae.on_commit(&bad), Err(SaeError::ScalarOutOfRange));
    }

    /// The full exchange, with the group arithmetic replaced by a fixed `k` that both
    /// sides are simply handed. This proves the KDF, Confirm and state transitions
    /// agree between two peers. It proves **nothing** about SAE's security, because
    /// `k` did not come from a PAKE — it came from this test.
    #[test]
    fn confirm_exchange_agrees_when_a_shared_secret_is_supplied() {
        let k = [0x5au8; 32];
        let scalar_sum = [0x3cu8; 32];
        let (sa, ea) = (scalar(2), element(0xa1));
        let (sb, eb) = (scalar(3), element(0xb2));

        let mut a = Sae::new(Group::P256).expect("group");
        let mut b = Sae::new(Group::P256).expect("group");
        let a_commit = a
            .commit(sa.clone(), ea.clone(), Vec::new())
            .expect("a commit");
        let b_commit = b
            .commit(sb.clone(), eb.clone(), Vec::new())
            .expect("b commit");

        a.on_commit(&SaeCommit::decode(&b_commit, 0).expect("decode b"))
            .expect("a accepts b");
        b.on_commit(&SaeCommit::decode(&a_commit, 0).expect("decode a"))
            .expect("b accepts a");

        a.install_shared_secret(&k, &scalar_sum).expect("a keys");
        b.install_shared_secret(&k, &scalar_sum).expect("b keys");

        let a_conf = a.confirm().expect("a confirm");
        let b_conf = b.confirm().expect("b confirm");
        assert_ne!(
            a_conf, b_conf,
            "confirms differ because the operand order does"
        );

        a.on_confirm(&SaeConfirm::decode(&b_conf, Group::P256).expect("decode"))
            .expect("a verifies b");
        b.on_confirm(&SaeConfirm::decode(&a_conf, Group::P256).expect("decode"))
            .expect("b verifies a");

        assert_eq!(a.state(), SaeState::Accepted);
        assert_eq!(b.state(), SaeState::Accepted);
        let ka = a.keys().expect("a pmk");
        let kb = b.keys().expect("b pmk");
        assert_eq!(ka, kb);
        assert_eq!(&ka.pmkid[..], &scalar_sum[..16]);
        assert_ne!(ka.kck, ka.pmk);
    }

    #[test]
    fn a_tampered_confirm_is_refused_and_the_state_machine_fails_closed() {
        let k = [0x5au8; 32];
        let scalar_sum = [0x3cu8; 32];
        let mut a = Sae::new(Group::P256).expect("group");
        let mut b = Sae::new(Group::P256).expect("group");
        let a_commit = a.commit(scalar(2), element(0xa1), Vec::new()).expect("a");
        let b_commit = b.commit(scalar(3), element(0xb2), Vec::new()).expect("b");
        a.on_commit(&SaeCommit::decode(&b_commit, 0).expect("decode"))
            .expect("a");
        b.on_commit(&SaeCommit::decode(&a_commit, 0).expect("decode"))
            .expect("b");
        a.install_shared_secret(&k, &scalar_sum).expect("a keys");
        b.install_shared_secret(&k, &scalar_sum).expect("b keys");
        let _ = a.confirm().expect("a confirm");
        let mut b_conf = b.confirm().expect("b confirm");
        b_conf[10] ^= 0x01;
        assert_eq!(
            a.on_confirm(&SaeConfirm::decode(&b_conf, Group::P256).expect("decode")),
            Err(SaeError::ConfirmMismatch)
        );
        assert_eq!(a.state(), SaeState::Failed);
        assert!(a.keys().is_none());
    }

    #[test]
    fn state_machine_refuses_out_of_order_messages() {
        let mut sae = Sae::new(Group::P256).expect("group");
        let peer = SaeCommit {
            group: Group::P256,
            anti_clogging_token: Vec::new(),
            scalar: scalar(3),
            element: element(0xbb),
        };
        assert_eq!(
            sae.on_commit(&peer),
            Err(SaeError::UnexpectedState("nothing"))
        );
        assert_eq!(sae.confirm(), Err(SaeError::UnexpectedState("nothing")));
        sae.commit(scalar(2), element(0xaa), Vec::new())
            .expect("commit");
        assert_eq!(
            sae.commit(scalar(2), element(0xaa), Vec::new()),
            Err(SaeError::UnexpectedState("committed"))
        );
        assert_eq!(
            sae.on_confirm(&SaeConfirm {
                send_confirm: 0,
                confirm: vec![0; 32]
            }),
            Err(SaeError::UnexpectedState("committed"))
        );
    }

    #[test]
    fn kdf_truncates_and_chains_by_counter_not_by_feedback() {
        // Structural check of 12.7.1.6.2: block i is HMAC(K, i || label || context ||
        // Length), so a shorter request is a prefix of a longer one only when Length
        // matches. It is not, because Length is inside the MAC input — which is the
        // difference from HKDF-Expand and the reason RFC 5869 vectors do not apply.
        let k = [0x01u8; 32];
        let a = kdf_sha256(&k, b"label", b"ctx", 256).expect("kdf");
        let b = kdf_sha256(&k, b"label", b"ctx", 512).expect("kdf");
        assert_eq!(a.len(), 32);
        assert_eq!(b.len(), 64);
        assert_ne!(&a[..], &b[..32], "Length is bound into every block");
        let c = kdf_sha256(&k, b"label", b"ctx", 512).expect("kdf");
        assert_eq!(b, c, "deterministic");
        let d = kdf_sha256(&k, b"label", b"ctY", 512).expect("kdf");
        assert_ne!(b, d, "context is bound in");
    }

    #[test]
    fn garbage_never_panics_the_codec() {
        let mut state = 0xdead_beef_cafe_f00du64;
        let mut buf = [0u8; 160];
        for _ in 0..20_000 {
            for b in buf.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *b = state as u8;
            }
            for len in [0usize, 1, 2, 34, 98, 102, 160] {
                for token in [0usize, 4, 32] {
                    let _ = SaeCommit::decode(&buf[..len], token);
                }
                let _ = SaeConfirm::decode(&buf[..len], Group::P256);
            }
        }
    }
}
