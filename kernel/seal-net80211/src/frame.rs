// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! IEEE 802.11 MAC frame codec — decode and encode, bounds-checked throughout.
//!
//! # What is implemented
//!
//! * Frame Control: protocol version, type, subtype, ToDS, FromDS, More Fragments,
//!   Retry, Power Management, More Data, Protected Frame, Order (+HTC).
//! * MAC header: Duration/ID, Address 1–4, Sequence Control (fragment + sequence
//!   number), QoS Control, HT Control. Address 4 is decoded if and only if the frame
//!   is a Data frame with `ToDS && FromDS` (802.11-2016 Table 9-26). QoS Control is
//!   decoded if and only if the Data subtype has bit 3 set. HT Control is decoded if
//!   and only if the Order bit is set on a Management or QoS Data frame — on a
//!   non-QoS Data frame the Order bit means StrictlyOrdered and carries no field.
//! * Management bodies: Beacon, Probe Request, Probe Response, Authentication,
//!   Association Request, Association Response, Deauthentication, Disassociation.
//! * Information elements: SSID (0), Supported Rates (1), DS Parameter Set (3),
//!   RSN (48) with full cipher-suite and AKM-suite selector parsing. Every other
//!   element ID round-trips through [`InfoElement::Unknown`] — unknown elements are
//!   skipped, never rejected, because rejecting them would fail on ordinary APs.
//!
//! # What is NOT implemented
//!
//! * Control frames (RTS/CTS/ACK/BlockAck). [`decode`] returns
//!   [`FrameError::UnsupportedFrameType`] for them; the supplicant path needs none.
//! * Reassociation Request/Response, ATIM, Action, Timing Advertisement. Their
//!   subtypes return [`FrameError::UnsupportedSubtype`] rather than a partial parse.
//! * The FCS. Frames handed to [`decode`] must already have the 4-octet trailing FCS
//!   stripped, which is what every driver hands up. Nothing here computes or checks
//!   a CRC-32.
//! * Fragment reassembly, A-MSDU and A-MPDU deaggregation.
//! * Extended Supported Rates (50), HT/VHT/HE capability elements, and vendor-specific
//!   elements (221) are preserved byte-exactly as [`InfoElement::Unknown`] but are not
//!   interpreted.
//!
//! # Hostile input
//!
//! These are frames from the air. Every read goes through [`Reader`], which returns
//! [`FrameError::Truncated`] instead of indexing out of bounds. There is no `unsafe`,
//! no `unwrap`, no `expect`, and no slicing by a length taken from the frame without
//! a preceding bounds check. The malformed-input tests at the bottom of this file are
//! the point of the module, not an afterthought.

use alloc::vec::Vec;
use core::fmt;

/// Every way a frame can fail to decode. No decode path panics; it returns one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    /// Fewer octets remained than the field being read requires.
    Truncated { need: usize, got: usize },
    /// Control (1) and Extension (3) frames are not decoded by this module.
    UnsupportedFrameType(u8),
    /// A Management subtype outside the implemented set.
    UnsupportedSubtype { frame_type: u8, subtype: u8 },
    /// An element's Length octet reaches past the end of the element buffer.
    ElementOverrun { id: u8, len: u8, remaining: usize },
    /// An element's Length octet is legal as a number but wrong for that element ID
    /// (for example a DS Parameter Set that is not exactly one octet).
    ElementBadLength { id: u8, len: u8 },
    /// An element body exceeds the 255-octet Length field during encoding.
    ElementTooLong { id: u8, len: usize },
    /// SSID element longer than the 32-octet maximum of 802.11-2016 9.4.2.2.
    SsidTooLong(usize),
    /// RSN element Version field is not 1 (802.11-2016 9.4.2.25.1).
    RsnVersion(u16),
    /// RSN element ends in the middle of a field rather than on a field boundary.
    RsnTruncated,
    /// An RSN suite/PMKID count multiplied out past the end of the element.
    RsnCountOverflow,
    /// Frame Control carries a protocol version this codec does not speak.
    ProtocolVersion(u8),
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { need, got } => write!(f, "truncated: need {need} octets, got {got}"),
            Self::UnsupportedFrameType(t) => write!(f, "unsupported frame type {t}"),
            Self::UnsupportedSubtype {
                frame_type,
                subtype,
            } => write!(f, "unsupported subtype {subtype} of type {frame_type}"),
            Self::ElementOverrun { id, len, remaining } => write!(
                f,
                "element {id} declares {len} octets, {remaining} remain in buffer"
            ),
            Self::ElementBadLength { id, len } => {
                write!(f, "element {id} has illegal length {len}")
            }
            Self::ElementTooLong { id, len } => {
                write!(f, "element {id} body of {len} octets exceeds 255")
            }
            Self::SsidTooLong(n) => write!(f, "SSID of {n} octets exceeds 32"),
            Self::RsnVersion(v) => write!(f, "RSN version {v} is not 1"),
            Self::RsnTruncated => write!(f, "RSN element ends mid-field"),
            Self::RsnCountOverflow => write!(f, "RSN suite count reaches past element end"),
            Self::ProtocolVersion(v) => write!(f, "802.11 protocol version {v} is not 0"),
        }
    }
}

/// A 48-bit MAC address in wire order.
///
/// `Ord` is the lexicographic order of the six octets, which is the numeric order of
/// the address as a big-endian 48-bit integer. That is exactly the comparison
/// 802.11-2016 12.7.1.3 means by `Min(AA, SPA)`, so `wpa2` orders addresses with the
/// derived `Ord` rather than a bespoke comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MacAddr(pub [u8; 6]);

impl MacAddr {
    pub const BROADCAST: Self = Self([0xff; 6]);

    /// Build from a slice; `None` if the slice is not exactly six octets.
    pub fn from_slice(s: &[u8]) -> Option<Self> {
        let mut a = [0u8; 6];
        if s.len() != 6 {
            return None;
        }
        a.copy_from_slice(s);
        Some(Self(a))
    }

    pub fn as_bytes(&self) -> &[u8; 6] {
        &self.0
    }

    /// Group bit of the first octet — set for broadcast and multicast.
    pub fn is_multicast(&self) -> bool {
        self.0[0] & 0x01 != 0
    }

    pub fn is_broadcast(&self) -> bool {
        self.0 == [0xff; 6]
    }
}

impl fmt::Display for MacAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.0;
        write!(
            f,
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            b[0], b[1], b[2], b[3], b[4], b[5]
        )
    }
}

/// Bounds-checked forward reader. Every accessor returns [`FrameError::Truncated`]
/// rather than panicking, so no decode path in this crate can index out of range.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8], FrameError> {
        let end = self.pos.checked_add(n).ok_or(FrameError::Truncated {
            need: n,
            got: self.remaining(),
        })?;
        if end > self.buf.len() {
            return Err(FrameError::Truncated {
                need: n,
                got: self.remaining(),
            });
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    pub fn u8(&mut self) -> Result<u8, FrameError> {
        Ok(self.take(1)?[0])
    }

    /// Little-endian, the byte order of every 802.11 MAC field.
    pub fn le16(&mut self) -> Result<u16, FrameError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn le32(&mut self) -> Result<u32, FrameError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn le64(&mut self) -> Result<u64, FrameError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    /// Big-endian — used by EAPOL-Key, which is network order, not 802.11 order.
    pub fn be16(&mut self) -> Result<u16, FrameError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    pub fn be64(&mut self) -> Result<u64, FrameError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_be_bytes(a))
    }

    pub fn mac(&mut self) -> Result<MacAddr, FrameError> {
        let b = self.take(6)?;
        MacAddr::from_slice(b).ok_or(FrameError::Truncated { need: 6, got: 0 })
    }

    pub fn array<const N: usize>(&mut self) -> Result<[u8; N], FrameError> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    /// Consume and return everything left.
    pub fn rest(&mut self) -> &'a [u8] {
        let s = &self.buf[self.pos.min(self.buf.len())..];
        self.pos = self.buf.len();
        s
    }
}

// ---------------------------------------------------------------------------
// Frame Control
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Management,
    Control,
    Data,
    Extension,
}

impl FrameType {
    fn from_bits(b: u8) -> Self {
        match b & 0x03 {
            0 => Self::Management,
            1 => Self::Control,
            2 => Self::Data,
            _ => Self::Extension,
        }
    }

    pub fn bits(self) -> u8 {
        match self {
            Self::Management => 0,
            Self::Control => 1,
            Self::Data => 2,
            Self::Extension => 3,
        }
    }
}

/// Management frame subtypes (802.11-2016 Table 9-1).
pub mod mgmt_subtype {
    pub const ASSOC_REQ: u8 = 0;
    pub const ASSOC_RESP: u8 = 1;
    pub const PROBE_REQ: u8 = 4;
    pub const PROBE_RESP: u8 = 5;
    pub const BEACON: u8 = 8;
    pub const DISASSOC: u8 = 10;
    pub const AUTH: u8 = 11;
    pub const DEAUTH: u8 = 12;
}

/// The 2-octet Frame Control field, decomposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameControl {
    pub protocol_version: u8,
    pub frame_type: FrameType,
    pub subtype: u8,
    pub to_ds: bool,
    pub from_ds: bool,
    pub more_fragments: bool,
    pub retry: bool,
    pub power_management: bool,
    pub more_data: bool,
    pub protected: bool,
    /// Order / +HTC.
    pub order: bool,
}

impl FrameControl {
    pub fn new(frame_type: FrameType, subtype: u8) -> Self {
        Self {
            protocol_version: 0,
            frame_type,
            subtype: subtype & 0x0f,
            to_ds: false,
            from_ds: false,
            more_fragments: false,
            retry: false,
            power_management: false,
            more_data: false,
            protected: false,
            order: false,
        }
    }

    pub fn from_bits(v: u16) -> Self {
        let b0 = (v & 0xff) as u8;
        let b1 = (v >> 8) as u8;
        Self {
            protocol_version: b0 & 0x03,
            frame_type: FrameType::from_bits(b0 >> 2),
            subtype: (b0 >> 4) & 0x0f,
            to_ds: b1 & 0x01 != 0,
            from_ds: b1 & 0x02 != 0,
            more_fragments: b1 & 0x04 != 0,
            retry: b1 & 0x08 != 0,
            power_management: b1 & 0x10 != 0,
            more_data: b1 & 0x20 != 0,
            protected: b1 & 0x40 != 0,
            order: b1 & 0x80 != 0,
        }
    }

    pub fn bits(&self) -> u16 {
        let b0 = (self.protocol_version & 0x03)
            | (self.frame_type.bits() << 2)
            | ((self.subtype & 0x0f) << 4);
        let b1 = (self.to_ds as u8)
            | ((self.from_ds as u8) << 1)
            | ((self.more_fragments as u8) << 2)
            | ((self.retry as u8) << 3)
            | ((self.power_management as u8) << 4)
            | ((self.more_data as u8) << 5)
            | ((self.protected as u8) << 6)
            | ((self.order as u8) << 7);
        u16::from_le_bytes([b0, b1])
    }

    /// True for Data subtypes 8–15, which carry a QoS Control field.
    pub fn has_qos(&self) -> bool {
        self.frame_type == FrameType::Data && (self.subtype & 0x08) != 0
    }

    /// Address 4 is present only on a Data frame bridged between two DSs.
    pub fn has_addr4(&self) -> bool {
        self.frame_type == FrameType::Data && self.to_ds && self.from_ds
    }

    /// HT Control rides along only on Management and QoS Data frames. On a non-QoS
    /// Data frame the same bit means StrictlyOrdered and adds no field.
    pub fn has_ht_control(&self) -> bool {
        self.order && (self.frame_type == FrameType::Management || self.has_qos())
    }
}

/// Sequence Control: a 4-bit fragment number and a 12-bit sequence number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SequenceControl {
    pub fragment: u8,
    pub sequence: u16,
}

impl SequenceControl {
    pub fn from_bits(v: u16) -> Self {
        Self {
            fragment: (v & 0x000f) as u8,
            sequence: (v >> 4) & 0x0fff,
        }
    }

    pub fn bits(&self) -> u16 {
        ((self.sequence & 0x0fff) << 4) | (self.fragment as u16 & 0x000f)
    }
}

/// A decoded MAC header. Optional fields are `Some` exactly when the presence rules
/// on [`FrameControl`] say they are on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacHeader {
    pub fc: FrameControl,
    pub duration: u16,
    pub addr1: MacAddr,
    pub addr2: MacAddr,
    pub addr3: MacAddr,
    pub seq_ctrl: SequenceControl,
    pub addr4: Option<MacAddr>,
    pub qos_ctrl: Option<u16>,
    pub ht_ctrl: Option<u32>,
}

impl MacHeader {
    pub fn decode(r: &mut Reader<'_>) -> Result<Self, FrameError> {
        let fc = FrameControl::from_bits(r.le16()?);
        if fc.protocol_version != 0 {
            return Err(FrameError::ProtocolVersion(fc.protocol_version));
        }
        let duration = r.le16()?;
        let addr1 = r.mac()?;
        let addr2 = r.mac()?;
        let addr3 = r.mac()?;
        let seq_ctrl = SequenceControl::from_bits(r.le16()?);
        let addr4 = if fc.has_addr4() { Some(r.mac()?) } else { None };
        let qos_ctrl = if fc.has_qos() { Some(r.le16()?) } else { None };
        let ht_ctrl = if fc.has_ht_control() {
            Some(r.le32()?)
        } else {
            None
        };
        Ok(Self {
            fc,
            duration,
            addr1,
            addr2,
            addr3,
            seq_ctrl,
            addr4,
            qos_ctrl,
            ht_ctrl,
        })
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.fc.bits().to_le_bytes());
        out.extend_from_slice(&self.duration.to_le_bytes());
        out.extend_from_slice(&self.addr1.0);
        out.extend_from_slice(&self.addr2.0);
        out.extend_from_slice(&self.addr3.0);
        out.extend_from_slice(&self.seq_ctrl.bits().to_le_bytes());
        if let Some(a4) = self.addr4 {
            out.extend_from_slice(&a4.0);
        }
        if let Some(q) = self.qos_ctrl {
            out.extend_from_slice(&q.to_le_bytes());
        }
        if let Some(h) = self.ht_ctrl {
            out.extend_from_slice(&h.to_le_bytes());
        }
    }

    /// Octet length of this header on the wire.
    pub fn len(&self) -> usize {
        24 + self.addr4.map_or(0, |_| 6)
            + self.qos_ctrl.map_or(0, |_| 2)
            + self.ht_ctrl.map_or(0, |_| 4)
    }

    /// A MAC header is never zero-length; present so `len` does not trip clippy.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Build a management header addressed AP-ward: addr1 = BSSID, addr2 = STA,
    /// addr3 = BSSID.
    pub fn management(subtype: u8, sta: MacAddr, bssid: MacAddr) -> Self {
        Self {
            fc: FrameControl::new(FrameType::Management, subtype),
            duration: 0,
            addr1: bssid,
            addr2: sta,
            addr3: bssid,
            seq_ctrl: SequenceControl::default(),
            addr4: None,
            qos_ctrl: None,
            ht_ctrl: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Information elements
// ---------------------------------------------------------------------------

pub mod element_id {
    pub const SSID: u8 = 0;
    pub const SUPPORTED_RATES: u8 = 1;
    pub const DS_PARAMETER_SET: u8 = 3;
    pub const RSN: u8 = 48;
}

/// A cipher or AKM suite selector: 3-octet OUI followed by a 1-octet suite type,
/// stored in wire order. Unrecognised selectors are preserved, not rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuiteSelector(pub [u8; 4]);

/// The 00-0F-AC OUI that 802.11-2016 Table 9-131 assigns to RSN suite selectors.
pub const OUI_RSN: [u8; 3] = [0x00, 0x0f, 0xac];

impl SuiteSelector {
    pub const fn rsn(suite_type: u8) -> Self {
        Self([OUI_RSN[0], OUI_RSN[1], OUI_RSN[2], suite_type])
    }

    pub fn oui(&self) -> [u8; 3] {
        [self.0[0], self.0[1], self.0[2]]
    }

    pub fn suite_type(&self) -> u8 {
        self.0[3]
    }

    pub fn is_rsn_oui(&self) -> bool {
        self.oui() == OUI_RSN
    }
}

/// Cipher suite types under the 00-0F-AC OUI (802.11-2016 Table 9-131).
pub mod cipher {
    use super::SuiteSelector;
    pub const USE_GROUP: SuiteSelector = SuiteSelector::rsn(0);
    pub const WEP40: SuiteSelector = SuiteSelector::rsn(1);
    pub const TKIP: SuiteSelector = SuiteSelector::rsn(2);
    pub const CCMP_128: SuiteSelector = SuiteSelector::rsn(4);
    pub const WEP104: SuiteSelector = SuiteSelector::rsn(5);
    pub const BIP_CMAC_128: SuiteSelector = SuiteSelector::rsn(6);
    pub const GCMP_128: SuiteSelector = SuiteSelector::rsn(8);
    pub const GCMP_256: SuiteSelector = SuiteSelector::rsn(9);
    pub const CCMP_256: SuiteSelector = SuiteSelector::rsn(10);
}

/// AKM suite types under the 00-0F-AC OUI (802.11-2016 Table 9-133, plus SAE/OWE).
pub mod akm {
    use super::SuiteSelector;
    pub const IEEE8021X: SuiteSelector = SuiteSelector::rsn(1);
    pub const PSK: SuiteSelector = SuiteSelector::rsn(2);
    pub const FT_IEEE8021X: SuiteSelector = SuiteSelector::rsn(3);
    pub const FT_PSK: SuiteSelector = SuiteSelector::rsn(4);
    pub const IEEE8021X_SHA256: SuiteSelector = SuiteSelector::rsn(5);
    pub const PSK_SHA256: SuiteSelector = SuiteSelector::rsn(6);
    pub const SAE: SuiteSelector = SuiteSelector::rsn(8);
    pub const FT_SAE: SuiteSelector = SuiteSelector::rsn(9);
    pub const OWE: SuiteSelector = SuiteSelector::rsn(18);
}

/// A decoded RSN element (ID 48, 802.11-2016 9.4.2.25).
///
/// Trailing fields are optional on the wire and each one may be omitted along with
/// everything after it. Omission is modelled as an empty list or `None`. On encode
/// the same rule runs backwards: emission stops at the first absent field, except
/// that a PMKID count of zero is still emitted when a Group Management Cipher Suite
/// follows it, because that field cannot be reached otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsnElement {
    pub version: u16,
    pub group_cipher: SuiteSelector,
    pub pairwise_ciphers: Vec<SuiteSelector>,
    pub akm_suites: Vec<SuiteSelector>,
    pub rsn_capabilities: Option<u16>,
    pub pmkids: Vec<[u8; 16]>,
    pub group_management_cipher: Option<SuiteSelector>,
}

impl RsnElement {
    /// The element a WPA2-PSK station offers: CCMP-128 pairwise and group, PSK AKM.
    pub fn wpa2_psk() -> Self {
        Self {
            version: 1,
            group_cipher: cipher::CCMP_128,
            pairwise_ciphers: alloc::vec![cipher::CCMP_128],
            akm_suites: alloc::vec![akm::PSK],
            rsn_capabilities: Some(0),
            pmkids: Vec::new(),
            group_management_cipher: None,
        }
    }

    fn decode(body: &[u8]) -> Result<Self, FrameError> {
        let mut r = Reader::new(body);
        let version = r.le16()?;
        if version != 1 {
            return Err(FrameError::RsnVersion(version));
        }
        let group_cipher = SuiteSelector(r.array::<4>()?);
        let mut out = Self {
            version,
            group_cipher,
            pairwise_ciphers: Vec::new(),
            akm_suites: Vec::new(),
            rsn_capabilities: None,
            pmkids: Vec::new(),
            group_management_cipher: None,
        };

        // Each block below is optional, but a block that starts must finish.
        if r.is_empty() {
            return Ok(out);
        }
        out.pairwise_ciphers = read_suites(&mut r)?;
        if r.is_empty() {
            return Ok(out);
        }
        out.akm_suites = read_suites(&mut r)?;
        if r.is_empty() {
            return Ok(out);
        }
        out.rsn_capabilities = Some(r.le16().map_err(|_| FrameError::RsnTruncated)?);
        if r.is_empty() {
            return Ok(out);
        }
        let pmkid_count = r.le16().map_err(|_| FrameError::RsnTruncated)? as usize;
        let need = pmkid_count
            .checked_mul(16)
            .ok_or(FrameError::RsnCountOverflow)?;
        if need > r.remaining() {
            return Err(FrameError::RsnCountOverflow);
        }
        for _ in 0..pmkid_count {
            out.pmkids.push(r.array::<16>()?);
        }
        if r.is_empty() {
            return Ok(out);
        }
        out.group_management_cipher = Some(SuiteSelector(
            r.array::<4>().map_err(|_| FrameError::RsnTruncated)?,
        ));
        // Anything left over is not a field this revision defines. Refuse rather than
        // silently accept a frame we did not fully understand.
        if !r.is_empty() {
            return Err(FrameError::RsnTruncated);
        }
        Ok(out)
    }

    fn encode_body(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.group_cipher.0);
        if self.pairwise_ciphers.is_empty() {
            return;
        }
        out.extend_from_slice(&(self.pairwise_ciphers.len() as u16).to_le_bytes());
        for s in &self.pairwise_ciphers {
            out.extend_from_slice(&s.0);
        }
        if self.akm_suites.is_empty() {
            return;
        }
        out.extend_from_slice(&(self.akm_suites.len() as u16).to_le_bytes());
        for s in &self.akm_suites {
            out.extend_from_slice(&s.0);
        }
        let Some(caps) = self.rsn_capabilities else {
            return;
        };
        out.extend_from_slice(&caps.to_le_bytes());
        if self.pmkids.is_empty() && self.group_management_cipher.is_none() {
            return;
        }
        out.extend_from_slice(&(self.pmkids.len() as u16).to_le_bytes());
        for p in &self.pmkids {
            out.extend_from_slice(p);
        }
        if let Some(gmc) = self.group_management_cipher {
            out.extend_from_slice(&gmc.0);
        }
    }
}

fn read_suites(r: &mut Reader<'_>) -> Result<Vec<SuiteSelector>, FrameError> {
    let count = r.le16().map_err(|_| FrameError::RsnTruncated)? as usize;
    let need = count.checked_mul(4).ok_or(FrameError::RsnCountOverflow)?;
    if need > r.remaining() {
        return Err(FrameError::RsnCountOverflow);
    }
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(SuiteSelector(r.array::<4>()?));
    }
    Ok(v)
}

/// One information element. Anything this codec does not model is kept verbatim so
/// that decode/encode is byte-exact and so that an unfamiliar element from a real AP
/// is skipped rather than treated as an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InfoElement {
    /// Zero-length SSID is the wildcard used by Probe Requests, and is legal.
    Ssid(Vec<u8>),
    SupportedRates(Vec<u8>),
    DsParameterSet(u8),
    Rsn(RsnElement),
    Unknown {
        id: u8,
        data: Vec<u8>,
    },
}

impl InfoElement {
    pub fn id(&self) -> u8 {
        match self {
            Self::Ssid(_) => element_id::SSID,
            Self::SupportedRates(_) => element_id::SUPPORTED_RATES,
            Self::DsParameterSet(_) => element_id::DS_PARAMETER_SET,
            Self::Rsn(_) => element_id::RSN,
            Self::Unknown { id, .. } => *id,
        }
    }

    fn from_parts(id: u8, body: &[u8]) -> Result<Self, FrameError> {
        match id {
            element_id::SSID => {
                if body.len() > 32 {
                    return Err(FrameError::SsidTooLong(body.len()));
                }
                Ok(Self::Ssid(body.to_vec()))
            }
            element_id::SUPPORTED_RATES => Ok(Self::SupportedRates(body.to_vec())),
            element_id::DS_PARAMETER_SET => {
                if body.len() != 1 {
                    return Err(FrameError::ElementBadLength {
                        id,
                        len: body.len() as u8,
                    });
                }
                Ok(Self::DsParameterSet(body[0]))
            }
            element_id::RSN => Ok(Self::Rsn(RsnElement::decode(body)?)),
            _ => Ok(Self::Unknown {
                id,
                data: body.to_vec(),
            }),
        }
    }

    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), FrameError> {
        let id = self.id();
        let start = out.len();
        out.push(id);
        out.push(0); // length placeholder
        match self {
            Self::Ssid(s) => {
                if s.len() > 32 {
                    return Err(FrameError::SsidTooLong(s.len()));
                }
                out.extend_from_slice(s);
            }
            Self::SupportedRates(r) => out.extend_from_slice(r),
            Self::DsParameterSet(c) => out.push(*c),
            Self::Rsn(rsn) => rsn.encode_body(out),
            Self::Unknown { data, .. } => out.extend_from_slice(data),
        }
        let body_len = out.len() - start - 2;
        if body_len > 255 {
            out.truncate(start);
            return Err(FrameError::ElementTooLong { id, len: body_len });
        }
        out[start + 1] = body_len as u8;
        Ok(())
    }
}

/// Parse a run of information elements.
///
/// An element whose Length octet reaches past the end of the buffer is a hard error:
/// silently truncating it would let a crafted frame hide fields from the parser.
pub fn parse_elements(buf: &[u8]) -> Result<Vec<InfoElement>, FrameError> {
    let mut r = Reader::new(buf);
    let mut out = Vec::new();
    while !r.is_empty() {
        let id = r.u8()?;
        let len = r.u8()? as usize;
        if len > r.remaining() {
            return Err(FrameError::ElementOverrun {
                id,
                len: len as u8,
                remaining: r.remaining(),
            });
        }
        let body = r.take(len)?;
        out.push(InfoElement::from_parts(id, body)?);
    }
    Ok(out)
}

pub fn encode_elements(elements: &[InfoElement], out: &mut Vec<u8>) -> Result<(), FrameError> {
    for e in elements {
        e.encode(out)?;
    }
    Ok(())
}

/// First SSID element, if any.
pub fn find_ssid(elements: &[InfoElement]) -> Option<&[u8]> {
    elements.iter().find_map(|e| match e {
        InfoElement::Ssid(s) => Some(s.as_slice()),
        _ => None,
    })
}

/// First RSN element, if any.
pub fn find_rsn(elements: &[InfoElement]) -> Option<&RsnElement> {
    elements.iter().find_map(|e| match e {
        InfoElement::Rsn(r) => Some(r),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// Management bodies
// ---------------------------------------------------------------------------

/// Authentication Algorithm Number (802.11-2016 9.4.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthAlgorithm(pub u16);

impl AuthAlgorithm {
    pub const OPEN_SYSTEM: Self = Self(0);
    pub const SHARED_KEY: Self = Self(1);
    pub const FAST_BSS_TRANSITION: Self = Self(2);
    pub const SAE: Self = Self(3);
}

/// Beacon and Probe Response share a body: Timestamp, Beacon Interval, Capability,
/// then elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeaconBody {
    pub timestamp: u64,
    pub beacon_interval: u16,
    pub capability: u16,
    pub elements: Vec<InfoElement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthBody {
    pub algorithm: AuthAlgorithm,
    pub transaction_seq: u16,
    pub status: u16,
    /// Everything after the three fixed fields, verbatim.
    ///
    /// For Open System these octets are information elements; parse with
    /// [`parse_elements`]. For SAE they are the SAE fixed fields and are parsed by
    /// `crate::wpa3`, which is why this codec does not presume a format.
    pub body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssocRequestBody {
    pub capability: u16,
    pub listen_interval: u16,
    pub elements: Vec<InfoElement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssocResponseBody {
    pub capability: u16,
    pub status: u16,
    pub association_id: u16,
    pub elements: Vec<InfoElement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasonBody {
    pub reason: u16,
    pub elements: Vec<InfoElement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManagementBody {
    Beacon(BeaconBody),
    ProbeRequest { elements: Vec<InfoElement> },
    ProbeResponse(BeaconBody),
    Authentication(AuthBody),
    AssocRequest(AssocRequestBody),
    AssocResponse(AssocResponseBody),
    Disassociation(ReasonBody),
    Deauthentication(ReasonBody),
}

impl ManagementBody {
    pub fn subtype(&self) -> u8 {
        match self {
            Self::Beacon(_) => mgmt_subtype::BEACON,
            Self::ProbeRequest { .. } => mgmt_subtype::PROBE_REQ,
            Self::ProbeResponse(_) => mgmt_subtype::PROBE_RESP,
            Self::Authentication(_) => mgmt_subtype::AUTH,
            Self::AssocRequest(_) => mgmt_subtype::ASSOC_REQ,
            Self::AssocResponse(_) => mgmt_subtype::ASSOC_RESP,
            Self::Disassociation(_) => mgmt_subtype::DISASSOC,
            Self::Deauthentication(_) => mgmt_subtype::DEAUTH,
        }
    }

    fn decode(subtype: u8, r: &mut Reader<'_>) -> Result<Self, FrameError> {
        match subtype {
            mgmt_subtype::BEACON | mgmt_subtype::PROBE_RESP => {
                let b = BeaconBody {
                    timestamp: r.le64()?,
                    beacon_interval: r.le16()?,
                    capability: r.le16()?,
                    elements: parse_elements(r.rest())?,
                };
                Ok(if subtype == mgmt_subtype::BEACON {
                    Self::Beacon(b)
                } else {
                    Self::ProbeResponse(b)
                })
            }
            mgmt_subtype::PROBE_REQ => Ok(Self::ProbeRequest {
                elements: parse_elements(r.rest())?,
            }),
            mgmt_subtype::AUTH => Ok(Self::Authentication(AuthBody {
                algorithm: AuthAlgorithm(r.le16()?),
                transaction_seq: r.le16()?,
                status: r.le16()?,
                body: r.rest().to_vec(),
            })),
            mgmt_subtype::ASSOC_REQ => Ok(Self::AssocRequest(AssocRequestBody {
                capability: r.le16()?,
                listen_interval: r.le16()?,
                elements: parse_elements(r.rest())?,
            })),
            mgmt_subtype::ASSOC_RESP => Ok(Self::AssocResponse(AssocResponseBody {
                capability: r.le16()?,
                status: r.le16()?,
                association_id: r.le16()?,
                elements: parse_elements(r.rest())?,
            })),
            mgmt_subtype::DISASSOC | mgmt_subtype::DEAUTH => {
                let b = ReasonBody {
                    reason: r.le16()?,
                    elements: parse_elements(r.rest())?,
                };
                Ok(if subtype == mgmt_subtype::DISASSOC {
                    Self::Disassociation(b)
                } else {
                    Self::Deauthentication(b)
                })
            }
            other => Err(FrameError::UnsupportedSubtype {
                frame_type: 0,
                subtype: other,
            }),
        }
    }

    fn encode(&self, out: &mut Vec<u8>) -> Result<(), FrameError> {
        match self {
            Self::Beacon(b) | Self::ProbeResponse(b) => {
                out.extend_from_slice(&b.timestamp.to_le_bytes());
                out.extend_from_slice(&b.beacon_interval.to_le_bytes());
                out.extend_from_slice(&b.capability.to_le_bytes());
                encode_elements(&b.elements, out)
            }
            Self::ProbeRequest { elements } => encode_elements(elements, out),
            Self::Authentication(a) => {
                out.extend_from_slice(&a.algorithm.0.to_le_bytes());
                out.extend_from_slice(&a.transaction_seq.to_le_bytes());
                out.extend_from_slice(&a.status.to_le_bytes());
                out.extend_from_slice(&a.body);
                Ok(())
            }
            Self::AssocRequest(a) => {
                out.extend_from_slice(&a.capability.to_le_bytes());
                out.extend_from_slice(&a.listen_interval.to_le_bytes());
                encode_elements(&a.elements, out)
            }
            Self::AssocResponse(a) => {
                out.extend_from_slice(&a.capability.to_le_bytes());
                out.extend_from_slice(&a.status.to_le_bytes());
                out.extend_from_slice(&a.association_id.to_le_bytes());
                encode_elements(&a.elements, out)
            }
            Self::Disassociation(b) | Self::Deauthentication(b) => {
                out.extend_from_slice(&b.reason.to_le_bytes());
                encode_elements(&b.elements, out)
            }
        }
    }
}

/// A decoded frame: header plus body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub header: MacHeader,
    pub body: FrameBody,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameBody {
    Management(ManagementBody),
    /// Data frame payload, verbatim. When `header.fc.protected` is set these octets
    /// are the CCMP header, ciphertext and MIC — decrypt with `crate::wpa2::ccmp`.
    Data(Vec<u8>),
}

/// Decode one 802.11 frame. The FCS, if the driver left one on, must be stripped
/// first: nothing here validates a CRC-32.
pub fn decode(buf: &[u8]) -> Result<Frame, FrameError> {
    let mut r = Reader::new(buf);
    let header = MacHeader::decode(&mut r)?;
    let body = match header.fc.frame_type {
        FrameType::Management => {
            FrameBody::Management(ManagementBody::decode(header.fc.subtype, &mut r)?)
        }
        FrameType::Data => FrameBody::Data(r.rest().to_vec()),
        FrameType::Control => return Err(FrameError::UnsupportedFrameType(1)),
        FrameType::Extension => return Err(FrameError::UnsupportedFrameType(3)),
    };
    Ok(Frame { header, body })
}

pub fn encode(frame: &Frame) -> Result<Vec<u8>, FrameError> {
    let mut out = Vec::new();
    frame.header.encode(&mut out);
    match &frame.body {
        FrameBody::Management(m) => m.encode(&mut out)?,
        FrameBody::Data(d) => out.extend_from_slice(d),
    }
    Ok(out)
}

/// Build a management frame whose Frame Control subtype matches its body.
pub fn management_frame(sta: MacAddr, bssid: MacAddr, body: ManagementBody) -> Frame {
    Frame {
        header: MacHeader::management(body.subtype(), sta, bssid),
        body: FrameBody::Management(body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn mac(n: u8) -> MacAddr {
        MacAddr([n, n + 1, n + 2, n + 3, n + 4, n + 5])
    }

    #[test]
    fn frame_control_bits_round_trip() {
        for raw in 0u16..=0xffff {
            let fc = FrameControl::from_bits(raw);
            assert_eq!(
                fc.bits(),
                raw,
                "frame control {raw:#06x} did not round-trip"
            );
        }
    }

    #[test]
    fn sequence_control_splits_fragment_and_sequence() {
        // 802.11-2016 9.2.4.4: bits 0-3 fragment number, bits 4-15 sequence number.
        let sc = SequenceControl::from_bits(0xABC5);
        assert_eq!(sc.fragment, 5);
        assert_eq!(sc.sequence, 0x0ABC);
        assert_eq!(sc.bits(), 0xABC5);
    }

    #[test]
    fn addr4_present_only_when_to_ds_and_from_ds() {
        let mut fc = FrameControl::new(FrameType::Data, 0);
        assert!(!fc.has_addr4());
        fc.to_ds = true;
        assert!(!fc.has_addr4());
        fc.from_ds = true;
        assert!(fc.has_addr4());
        // The same bit pattern on a management frame carries no address 4.
        let mut m = FrameControl::new(FrameType::Management, mgmt_subtype::BEACON);
        m.to_ds = true;
        m.from_ds = true;
        assert!(!m.has_addr4());
    }

    #[test]
    fn qos_and_ht_control_presence() {
        let qos = FrameControl::new(FrameType::Data, 8);
        assert!(qos.has_qos());
        let nonqos = FrameControl::new(FrameType::Data, 0);
        assert!(!nonqos.has_qos());
        // Order on a non-QoS data frame means StrictlyOrdered, not +HTC.
        let mut ordered_data = FrameControl::new(FrameType::Data, 0);
        ordered_data.order = true;
        assert!(!ordered_data.has_ht_control());
        let mut ordered_qos = FrameControl::new(FrameType::Data, 8);
        ordered_qos.order = true;
        assert!(ordered_qos.has_ht_control());
    }

    #[test]
    fn four_address_qos_header_round_trips() {
        let mut fc = FrameControl::new(FrameType::Data, 8);
        fc.to_ds = true;
        fc.from_ds = true;
        fc.order = true;
        let h = MacHeader {
            fc,
            duration: 0x1234,
            addr1: mac(1),
            addr2: mac(0x11),
            addr3: mac(0x21),
            seq_ctrl: SequenceControl {
                fragment: 3,
                sequence: 100,
            },
            addr4: Some(mac(0x31)),
            qos_ctrl: Some(0x0007),
            ht_ctrl: Some(0xdeadbeef),
        };
        let mut buf = Vec::new();
        h.encode(&mut buf);
        assert_eq!(buf.len(), h.len());
        assert_eq!(buf.len(), 24 + 6 + 2 + 4);
        let decoded = MacHeader::decode(&mut Reader::new(&buf)).expect("header decode");
        assert_eq!(decoded, h);
    }

    #[test]
    fn beacon_round_trips() {
        let body = ManagementBody::Beacon(BeaconBody {
            timestamp: 0x0102030405060708,
            beacon_interval: 100,
            capability: 0x0431,
            elements: vec![
                InfoElement::Ssid(b"seal-test".to_vec()),
                InfoElement::SupportedRates(vec![0x82, 0x84, 0x8b, 0x96]),
                InfoElement::DsParameterSet(6),
                InfoElement::Rsn(RsnElement::wpa2_psk()),
                InfoElement::Unknown {
                    id: 221,
                    data: vec![0x00, 0x50, 0xf2, 0x02, 0x01, 0x01],
                },
            ],
        });
        let f = management_frame(mac(1), mac(0x40), body);
        let bytes = encode(&f).expect("encode");
        let back = decode(&bytes).expect("decode");
        assert_eq!(back, f);
        assert_eq!(encode(&back).expect("re-encode"), bytes);
    }

    #[test]
    fn every_management_body_round_trips() {
        let elems = vec![InfoElement::Ssid(b"x".to_vec())];
        let bodies = vec![
            ManagementBody::ProbeRequest {
                elements: elems.clone(),
            },
            ManagementBody::ProbeResponse(BeaconBody {
                timestamp: 7,
                beacon_interval: 100,
                capability: 1,
                elements: elems.clone(),
            }),
            ManagementBody::Authentication(AuthBody {
                algorithm: AuthAlgorithm::OPEN_SYSTEM,
                transaction_seq: 1,
                status: 0,
                body: vec![],
            }),
            ManagementBody::Authentication(AuthBody {
                algorithm: AuthAlgorithm::SAE,
                transaction_seq: 1,
                status: 0,
                body: vec![0x13, 0x00, 0xaa, 0xbb],
            }),
            ManagementBody::AssocRequest(AssocRequestBody {
                capability: 0x0431,
                listen_interval: 10,
                elements: elems.clone(),
            }),
            ManagementBody::AssocResponse(AssocResponseBody {
                capability: 0x0431,
                status: 0,
                association_id: 0xc001,
                elements: elems.clone(),
            }),
            ManagementBody::Disassociation(ReasonBody {
                reason: 8,
                elements: vec![],
            }),
            ManagementBody::Deauthentication(ReasonBody {
                reason: 3,
                elements: elems,
            }),
        ];
        for b in bodies {
            let f = management_frame(mac(1), mac(0x40), b);
            let bytes = encode(&f).expect("encode");
            assert_eq!(decode(&bytes).expect("decode"), f);
        }
    }

    #[test]
    fn rsn_element_selectors_decode() {
        let rsn = RsnElement::wpa2_psk();
        let mut buf = Vec::new();
        InfoElement::Rsn(rsn.clone())
            .encode(&mut buf)
            .expect("encode");
        assert_eq!(buf[0], element_id::RSN);
        let parsed = parse_elements(&buf).expect("parse");
        let InfoElement::Rsn(got) = &parsed[0] else {
            panic!("not an RSN element")
        };
        assert_eq!(got, &rsn);
        assert_eq!(got.group_cipher, cipher::CCMP_128);
        assert_eq!(got.pairwise_ciphers, vec![cipher::CCMP_128]);
        assert_eq!(got.akm_suites, vec![akm::PSK]);
        assert!(got.akm_suites[0].is_rsn_oui());
        assert_eq!(got.akm_suites[0].suite_type(), 2);
    }

    #[test]
    fn rsn_element_full_form_round_trips() {
        let rsn = RsnElement {
            version: 1,
            group_cipher: cipher::CCMP_128,
            pairwise_ciphers: vec![cipher::CCMP_128, cipher::TKIP],
            akm_suites: vec![akm::PSK, akm::SAE, SuiteSelector([0x00, 0x50, 0xf2, 0x01])],
            rsn_capabilities: Some(0x00c0),
            pmkids: vec![[0xaa; 16], [0xbb; 16]],
            group_management_cipher: Some(cipher::BIP_CMAC_128),
        };
        let mut buf = Vec::new();
        InfoElement::Rsn(rsn.clone())
            .encode(&mut buf)
            .expect("encode");
        let parsed = parse_elements(&buf).expect("parse");
        assert_eq!(parsed, vec![InfoElement::Rsn(rsn)]);
    }

    #[test]
    fn rsn_truncated_at_field_boundaries_uses_defaults() {
        // Version + group cipher only: legal, everything after defaults away.
        let body = [0x01, 0x00, 0x00, 0x0f, 0xac, 0x04];
        let rsn = RsnElement::decode(&body).expect("short RSN is legal");
        assert_eq!(rsn.group_cipher, cipher::CCMP_128);
        assert!(rsn.pairwise_ciphers.is_empty());
        assert!(rsn.akm_suites.is_empty());
        assert_eq!(rsn.rsn_capabilities, None);
    }

    #[test]
    fn rsn_lying_pairwise_count_is_refused() {
        // Claims 0x0fff pairwise suites, supplies one.
        let body = [
            0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0xff, 0x0f, 0x00, 0x0f, 0xac, 0x04,
        ];
        assert_eq!(RsnElement::decode(&body), Err(FrameError::RsnCountOverflow));
    }

    #[test]
    fn rsn_mid_field_truncation_is_refused() {
        // Version, group cipher, then a single stray octet where a count belongs.
        let body = [0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0x01];
        assert_eq!(RsnElement::decode(&body), Err(FrameError::RsnTruncated));
    }

    #[test]
    fn rsn_bad_version_is_refused() {
        let body = [0x02, 0x00, 0x00, 0x0f, 0xac, 0x04];
        assert_eq!(RsnElement::decode(&body), Err(FrameError::RsnVersion(2)));
    }

    #[test]
    fn element_length_past_end_is_refused() {
        // SSID element claiming 40 octets with 4 present.
        let buf = [0x00, 0x28, b'a', b'b', b'c', b'd'];
        assert_eq!(
            parse_elements(&buf),
            Err(FrameError::ElementOverrun {
                id: 0,
                len: 0x28,
                remaining: 4
            })
        );
    }

    #[test]
    fn oversized_ssid_is_refused() {
        let mut buf = vec![0x00, 33];
        buf.extend(core::iter::repeat_n(b'z', 33));
        assert_eq!(parse_elements(&buf), Err(FrameError::SsidTooLong(33)));
    }

    #[test]
    fn ds_parameter_set_must_be_one_octet() {
        let buf = [0x03, 0x02, 0x06, 0x00];
        assert_eq!(
            parse_elements(&buf),
            Err(FrameError::ElementBadLength { id: 3, len: 2 })
        );
    }

    #[test]
    fn unknown_elements_are_preserved_not_rejected() {
        let buf = [
            0xdd, 0x03, 0x01, 0x02, 0x03, // vendor specific
            0x32, 0x02, 0x0c, 0x12, // extended supported rates
            0x00, 0x02, b'h', b'i',
        ];
        let e = parse_elements(&buf).expect("unknown elements must be skipped, not fatal");
        assert_eq!(e.len(), 3);
        assert_eq!(
            e[0],
            InfoElement::Unknown {
                id: 0xdd,
                data: vec![1, 2, 3]
            }
        );
        assert_eq!(e[2], InfoElement::Ssid(b"hi".to_vec()));
        let mut out = Vec::new();
        encode_elements(&e, &mut out).expect("re-encode");
        assert_eq!(out, buf);
    }

    #[test]
    fn zero_length_ssid_is_the_wildcard_and_is_legal() {
        let e = parse_elements(&[0x00, 0x00]).expect("wildcard SSID");
        assert_eq!(e, vec![InfoElement::Ssid(Vec::new())]);
    }

    #[test]
    fn every_truncation_of_a_beacon_either_errors_or_is_a_whole_shorter_frame() {
        let f = management_frame(
            mac(1),
            mac(0x40),
            ManagementBody::Beacon(BeaconBody {
                timestamp: 1,
                beacon_interval: 100,
                capability: 0x0431,
                elements: vec![
                    InfoElement::Ssid(b"seal".to_vec()),
                    InfoElement::Rsn(RsnElement::wpa2_psk()),
                ],
            }),
        );
        let bytes = encode(&f).expect("encode");
        for n in 0..bytes.len() {
            match decode(&bytes[..n]) {
                // Anything shorter than the fixed header cannot decode at all.
                Err(_) => assert!(n < bytes.len()),
                // A prefix that does decode must be a whole frame in its own right —
                // a 36-octet beacon carrying no elements is legal. What must never
                // happen is a decode that silently swallows a partial field, so the
                // decoded frame has to re-encode to exactly the octets it was given.
                Ok(partial) => assert_eq!(
                    encode(&partial).expect("re-encode"),
                    &bytes[..n],
                    "prefix of {n} octets decoded without round-tripping"
                ),
            }
        }
        // Nothing shorter than the 24-octet MAC header can decode under any subtype.
        for n in 0..24 {
            assert!(decode(&bytes[..n]).is_err(), "{n}-octet prefix decoded");
        }
        assert!(decode(&bytes).is_ok());
    }

    #[test]
    fn random_looking_garbage_never_panics() {
        // Deterministic xorshift so a failure is reproducible without a fixture file.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut buf = [0u8; 96];
        for _ in 0..20_000 {
            for b in buf.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *b = state as u8;
            }
            for len in [0usize, 1, 2, 10, 24, 30, 64, 96] {
                let _ = decode(&buf[..len]);
                let _ = parse_elements(&buf[..len]);
            }
        }
    }

    #[test]
    fn control_frames_are_refused_rather_than_half_parsed() {
        let mut buf = Vec::new();
        let fc = FrameControl::new(FrameType::Control, 13); // ACK
        buf.extend_from_slice(&fc.bits().to_le_bytes());
        buf.extend_from_slice(&[0u8; 22]);
        assert_eq!(decode(&buf), Err(FrameError::UnsupportedFrameType(1)));
    }

    #[test]
    fn unsupported_management_subtype_is_refused() {
        let f = MacHeader::management(2, mac(1), mac(0x40)); // Reassociation Request
        let mut buf = Vec::new();
        f.encode(&mut buf);
        buf.extend_from_slice(&[0u8; 8]);
        assert_eq!(
            decode(&buf),
            Err(FrameError::UnsupportedSubtype {
                frame_type: 0,
                subtype: 2
            })
        );
    }

    #[test]
    fn nonzero_protocol_version_is_refused() {
        let buf = [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(decode(&buf), Err(FrameError::ProtocolVersion(1)));
    }
}
