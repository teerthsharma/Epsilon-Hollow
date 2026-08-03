// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! X.509 v3 certificate parsing and chain validation.
//!
//! Trust boundary: every byte handled here is attacker-controlled. The parser
//! never indexes without a prior length check, never panics on malformed
//! input, and reports a typed [`X509Error`] instead of falling back to a
//! default. There is no `unwrap` on parsed data anywhere in this module.
//!
//! Scope: **Ed25519 only** (RFC 8410, OID 1.3.101.112). RSA and ECDSA
//! certificates are rejected with [`X509Error::UnsupportedAlgorithm`] rather
//! than accepted unverified. Extensions understood: BasicConstraints,
//! KeyUsage, SubjectAltName. Any *critical* extension outside that set is a
//! hard reject, per RFC 5280 §4.2.

use alloc::vec::Vec;
use ed25519_dalek::{Signature, VerifyingKey};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Every failure mode of parsing or validating a certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X509Error {
    /// Ran off the end of the buffer.
    Truncated,
    /// Length octets are malformed, indefinite, non-minimal, or absurd.
    BadLength,
    /// A DER tag that is not the one the grammar requires at this position.
    UnexpectedTag,
    /// Bytes left over after a structure that should have consumed them all.
    TrailingData,
    /// Not an X.509 v3 certificate.
    UnsupportedVersion,
    /// Signature or key algorithm other than Ed25519.
    UnsupportedAlgorithm,
    /// Malformed UTCTime / GeneralizedTime.
    BadTime,
    /// SubjectPublicKeyInfo is not a well-formed 32-byte Ed25519 key.
    BadPublicKey,
    /// Signature is absent, wrong length, or does not verify.
    BadSignature,
    /// A string field is not valid UTF-8.
    BadEncoding,
    /// A critical extension this parser does not understand.
    UnknownCriticalExtension,
    /// `notBefore` is in the future relative to the kernel clock.
    NotYetValid,
    /// `notAfter` is in the past relative to the kernel clock.
    Expired,
    /// Issuer DN of the child does not match the subject DN of the parent.
    IssuerMismatch,
    /// The issuing certificate does not carry BasicConstraints CA:TRUE.
    IssuerNotCa,
    /// The issuing certificate has KeyUsage without `keyCertSign`.
    KeyUsageViolation,
    /// More intermediates below an issuer than its `pathLenConstraint` allows.
    PathLenExceeded,
    /// No trust anchor issued the top of the chain.
    UnknownIssuer,
    /// Chain had no certificates.
    EmptyChain,
    /// Chain longer than [`MAX_CHAIN_LEN`].
    ChainTooLong,
}

// ---------------------------------------------------------------------------
// DER reader
// ---------------------------------------------------------------------------

const TAG_BOOL: u8 = 0x01;
const TAG_INT: u8 = 0x02;
const TAG_BITSTR: u8 = 0x03;
const TAG_OCTETSTR: u8 = 0x04;
const TAG_OID: u8 = 0x06;
const TAG_SEQ: u8 = 0x30;
const TAG_SET: u8 = 0x31;
const TAG_UTCTIME: u8 = 0x17;
const TAG_GENTIME: u8 = 0x18;
/// `[0] EXPLICIT` — TBSCertificate version.
const TAG_CTX0: u8 = 0xa0;
/// `[1]`/`[2]` IMPLICIT — deprecated unique identifiers, skipped.
const TAG_CTX1: u8 = 0xa1;
const TAG_CTX2: u8 = 0xa2;
/// `[3] EXPLICIT` — extensions.
const TAG_CTX3: u8 = 0xa3;
/// `[2] IMPLICIT IA5String` — GeneralName dNSName.
const TAG_DNS_NAME: u8 = 0x82;

/// OID 1.3.101.112 (Ed25519), content octets only.
const OID_ED25519: &[u8] = &[0x2b, 0x65, 0x70];
/// OID 2.5.29.19 (basicConstraints).
const OID_BASIC_CONSTRAINTS: &[u8] = &[0x55, 0x1d, 0x13];
/// OID 2.5.29.15 (keyUsage).
const OID_KEY_USAGE: &[u8] = &[0x55, 0x1d, 0x0f];
/// OID 2.5.29.17 (subjectAltName).
const OID_SAN: &[u8] = &[0x55, 0x1d, 0x11];
/// OID 2.5.4.3 (commonName).
const OID_COMMON_NAME: &[u8] = &[0x55, 0x04, 0x03];

/// KeyUsage bit 5 — `keyCertSign`.
pub const KU_KEY_CERT_SIGN: u16 = 1 << 5;
/// KeyUsage bit 0 — `digitalSignature`.
pub const KU_DIGITAL_SIGNATURE: u16 = 1 << 0;

/// A cursor over a DER byte string. Every read is bounds-checked.
struct Der<'a> {
    rest: &'a [u8],
}

impl<'a> Der<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { rest: bytes }
    }

    fn is_empty(&self) -> bool {
        self.rest.is_empty()
    }

    fn peek(&self) -> Option<u8> {
        self.rest.first().copied()
    }

    /// Read one TLV. Returns `(tag, full_encoding, value)`.
    ///
    /// `full_encoding` includes the header, which is what a signature covers.
    fn next(&mut self) -> Result<(u8, &'a [u8], &'a [u8]), X509Error> {
        let b = self.rest;
        if b.len() < 2 {
            return Err(X509Error::Truncated);
        }
        let tag = b[0];
        if tag & 0x1f == 0x1f {
            // Multi-byte tag numbers do not occur in X.509.
            return Err(X509Error::UnexpectedTag);
        }
        let first = b[1];
        let (hdr, len) = if first & 0x80 == 0 {
            (2usize, first as usize)
        } else {
            let n = (first & 0x7f) as usize;
            // n == 0 is BER indefinite length, forbidden in DER.
            if n == 0 || n > 4 {
                return Err(X509Error::BadLength);
            }
            if b.len() < 2 + n {
                return Err(X509Error::Truncated);
            }
            let mut v = 0usize;
            for &x in &b[2..2 + n] {
                v = (v << 8) | x as usize;
            }
            // DER requires the shortest possible length encoding.
            if v < 0x80 {
                return Err(X509Error::BadLength);
            }
            (2 + n, v)
        };
        let end = hdr.checked_add(len).ok_or(X509Error::BadLength)?;
        if b.len() < end {
            return Err(X509Error::Truncated);
        }
        self.rest = &b[end..];
        Ok((tag, &b[..end], &b[hdr..end]))
    }

    /// Read one TLV, requiring a specific tag. Returns the value octets.
    fn expect(&mut self, tag: u8) -> Result<&'a [u8], X509Error> {
        let (t, _, v) = self.next()?;
        if t != tag {
            return Err(X509Error::UnexpectedTag);
        }
        Ok(v)
    }

    fn finish(&self) -> Result<(), X509Error> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(X509Error::TrailingData)
        }
    }
}

/// Pull the algorithm OID out of an `AlgorithmIdentifier` body.
///
/// Ed25519 carries absent parameters (RFC 8410 §3); anything else is a
/// different algorithm and is rejected upstream.
fn algorithm_oid(body: &[u8]) -> Result<&[u8], X509Error> {
    let mut d = Der::new(body);
    let oid = d.expect(TAG_OID)?;
    d.finish()?;
    Ok(oid)
}

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------

fn digits(b: &[u8]) -> Result<i64, X509Error> {
    let mut v = 0i64;
    for &c in b {
        if !c.is_ascii_digit() {
            return Err(X509Error::BadTime);
        }
        v = v * 10 + (c - b'0') as i64;
    }
    Ok(v)
}

/// Days since 1970-01-01 (Howard Hinnant's civil-from-days inverse).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Decode UTCTime (`YYMMDDHHMMSSZ`) or GeneralizedTime (`YYYYMMDDHHMMSSZ`)
/// to seconds since the Unix epoch, saturating at 0 for pre-1970 dates.
fn parse_time(tag: u8, v: &[u8]) -> Result<u64, X509Error> {
    let (year, rest) = match tag {
        TAG_UTCTIME => {
            if v.len() != 13 {
                return Err(X509Error::BadTime);
            }
            let yy = digits(&v[..2])?;
            // RFC 5280 §4.1.2.5.1: YY >= 50 means 19YY.
            (if yy < 50 { 2000 + yy } else { 1900 + yy }, &v[2..])
        }
        TAG_GENTIME => {
            if v.len() != 15 {
                return Err(X509Error::BadTime);
            }
            (digits(&v[..4])?, &v[4..])
        }
        _ => return Err(X509Error::BadTime),
    };
    // `rest` is exactly 11 bytes in both branches: MMDDHHMMSS + 'Z'.
    if rest[10] != b'Z' {
        return Err(X509Error::BadTime);
    }
    let month = digits(&rest[0..2])?;
    let day = digits(&rest[2..4])?;
    let hour = digits(&rest[4..6])?;
    let min = digits(&rest[6..8])?;
    let sec = digits(&rest[8..10])?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || min > 59 || sec > 60 {
        return Err(X509Error::BadTime);
    }
    let secs = days_from_civil(year, month, day) * 86400 + hour * 3600 + min * 60 + sec;
    Ok(if secs < 0 { 0 } else { secs as u64 })
}

// ---------------------------------------------------------------------------
// Certificate
// ---------------------------------------------------------------------------

/// Longest chain accepted, leaf included, trust anchor excluded.
pub const MAX_CHAIN_LEN: usize = 4;

/// Embedded trust store. One anchor: the Seal OS root CA.
pub static TRUST_ANCHORS: [&[u8]; 1] = [&super::certs::ROOT_CA_DER];

/// A parsed X.509 v3 certificate. All fields borrow the original DER.
#[derive(Debug)]
pub struct Certificate<'a> {
    /// The exact `TBSCertificate` encoding, header included — the signed bytes.
    pub tbs_der: &'a [u8],
    /// `serialNumber` content octets, unsigned-big-endian with a sign byte.
    pub serial: &'a [u8],
    /// Full `issuer` Name encoding, compared byte-for-byte against a subject.
    pub issuer_der: &'a [u8],
    /// Full `subject` Name encoding.
    pub subject_der: &'a [u8],
    /// `notBefore` as seconds since the Unix epoch.
    pub not_before: u64,
    /// `notAfter` as seconds since the Unix epoch.
    pub not_after: u64,
    /// Ed25519 public key from SubjectPublicKeyInfo.
    pub public_key: [u8; 32],
    /// Ed25519 signature over `tbs_der`.
    pub signature: [u8; 64],
    /// BasicConstraints `cA`.
    pub is_ca: bool,
    /// BasicConstraints `pathLenConstraint`, if present.
    pub path_len: Option<u8>,
    /// KeyUsage bits, bit *n* of the DER BIT STRING at bit *n* here. Zero
    /// means the extension was absent.
    pub key_usage: u16,
    /// dNSName entries from SubjectAltName.
    pub san_dns: Vec<&'a str>,
    /// Subject commonName, if the DN carries one.
    pub common_name: Option<&'a str>,
}

impl<'a> Certificate<'a> {
    /// Parse a DER-encoded certificate.
    ///
    /// Rejects trailing bytes after the outer SEQUENCE, so a certificate
    /// cannot smuggle a payload past the parser.
    pub fn parse(der: &'a [u8]) -> Result<Self, X509Error> {
        let mut top = Der::new(der);
        let cert_body = top.expect(TAG_SEQ)?;
        top.finish()?;

        let mut body = Der::new(cert_body);
        let (tag, tbs_der, tbs_body) = body.next()?;
        if tag != TAG_SEQ {
            return Err(X509Error::UnexpectedTag);
        }
        let outer_alg = body.expect(TAG_SEQ)?;
        let sig_bits = body.expect(TAG_BITSTR)?;
        body.finish()?;

        if algorithm_oid(outer_alg)? != OID_ED25519 {
            return Err(X509Error::UnsupportedAlgorithm);
        }
        if sig_bits.len() != 65 || sig_bits[0] != 0 {
            return Err(X509Error::BadSignature);
        }
        let mut signature = [0u8; 64];
        signature.copy_from_slice(&sig_bits[1..]);

        let mut tbs = Der::new(tbs_body);

        // version [0] EXPLICIT INTEGER — v3 (value 2) is the only version
        // that can carry extensions, and the only one accepted.
        let version_body = tbs.expect(TAG_CTX0)?;
        let mut vd = Der::new(version_body);
        let version = vd.expect(TAG_INT)?;
        vd.finish()?;
        if version != [0x02] {
            return Err(X509Error::UnsupportedVersion);
        }

        let serial = tbs.expect(TAG_INT)?;

        let inner_alg = tbs.expect(TAG_SEQ)?;
        // RFC 5280 §4.1.1.2: the two algorithm identifiers must agree, or an
        // attacker can advertise one algorithm and have another verified.
        if algorithm_oid(inner_alg)? != OID_ED25519 {
            return Err(X509Error::UnsupportedAlgorithm);
        }

        let (tag, issuer_der, _) = tbs.next()?;
        if tag != TAG_SEQ {
            return Err(X509Error::UnexpectedTag);
        }

        let validity = tbs.expect(TAG_SEQ)?;
        let mut vd = Der::new(validity);
        let (nb_tag, _, nb_val) = vd.next()?;
        let not_before = parse_time(nb_tag, nb_val)?;
        let (na_tag, _, na_val) = vd.next()?;
        let not_after = parse_time(na_tag, na_val)?;
        vd.finish()?;

        let (tag, subject_der, subject_body) = tbs.next()?;
        if tag != TAG_SEQ {
            return Err(X509Error::UnexpectedTag);
        }

        let spki = tbs.expect(TAG_SEQ)?;
        let mut sd = Der::new(spki);
        let spki_alg = sd.expect(TAG_SEQ)?;
        let spki_bits = sd.expect(TAG_BITSTR)?;
        sd.finish()?;
        if algorithm_oid(spki_alg)? != OID_ED25519 {
            return Err(X509Error::UnsupportedAlgorithm);
        }
        if spki_bits.len() != 33 || spki_bits[0] != 0 {
            return Err(X509Error::BadPublicKey);
        }
        let mut public_key = [0u8; 32];
        public_key.copy_from_slice(&spki_bits[1..]);

        let mut extensions = None;
        while !tbs.is_empty() {
            let (tag, _, body) = tbs.next()?;
            match tag {
                // issuerUniqueID / subjectUniqueID — deprecated, ignored.
                TAG_CTX1 | TAG_CTX2 => {}
                TAG_CTX3 => extensions = Some(body),
                _ => return Err(X509Error::UnexpectedTag),
            }
        }

        let mut cert = Certificate {
            tbs_der,
            serial,
            issuer_der,
            subject_der,
            not_before,
            not_after,
            public_key,
            signature,
            is_ca: false,
            path_len: None,
            key_usage: 0,
            san_dns: Vec::new(),
            common_name: common_name(subject_body)?,
        };
        if let Some(body) = extensions {
            cert.parse_extensions(body)?;
        }
        Ok(cert)
    }

    fn parse_extensions(&mut self, body: &'a [u8]) -> Result<(), X509Error> {
        let mut outer = Der::new(body);
        let list = outer.expect(TAG_SEQ)?;
        outer.finish()?;

        let mut items = Der::new(list);
        while !items.is_empty() {
            let ext = items.expect(TAG_SEQ)?;
            let mut e = Der::new(ext);
            let oid = e.expect(TAG_OID)?;
            let (tag, _, first) = e.next()?;
            let (critical, value) = if tag == TAG_BOOL {
                if first.len() != 1 {
                    return Err(X509Error::BadLength);
                }
                (first[0] != 0, e.expect(TAG_OCTETSTR)?)
            } else if tag == TAG_OCTETSTR {
                (false, first)
            } else {
                return Err(X509Error::UnexpectedTag);
            };
            e.finish()?;

            if oid == OID_BASIC_CONSTRAINTS {
                self.parse_basic_constraints(value)?;
            } else if oid == OID_KEY_USAGE {
                self.key_usage = parse_key_usage(value)?;
            } else if oid == OID_SAN {
                self.parse_san(value)?;
            } else if critical {
                // RFC 5280 §4.2: an unrecognised critical extension means the
                // certificate must be rejected, not used with the extension
                // silently ignored.
                return Err(X509Error::UnknownCriticalExtension);
            }
        }
        Ok(())
    }

    fn parse_basic_constraints(&mut self, value: &[u8]) -> Result<(), X509Error> {
        let mut outer = Der::new(value);
        let seq = outer.expect(TAG_SEQ)?;
        outer.finish()?;
        let mut d = Der::new(seq);
        if d.peek() == Some(TAG_BOOL) {
            let v = d.expect(TAG_BOOL)?;
            if v.len() != 1 {
                return Err(X509Error::BadLength);
            }
            self.is_ca = v[0] != 0;
        }
        if d.peek() == Some(TAG_INT) {
            let v = d.expect(TAG_INT)?;
            // A path length over 127 is meaningless at any realistic depth and
            // would need multi-byte handling; reject rather than truncate.
            if v.len() != 1 || v[0] > 0x7f {
                return Err(X509Error::BadLength);
            }
            self.path_len = Some(v[0]);
        }
        d.finish()
    }

    fn parse_san(&mut self, value: &'a [u8]) -> Result<(), X509Error> {
        let mut outer = Der::new(value);
        let seq = outer.expect(TAG_SEQ)?;
        outer.finish()?;
        let mut names = Der::new(seq);
        while !names.is_empty() {
            let (tag, _, v) = names.next()?;
            if tag == TAG_DNS_NAME {
                let name = core::str::from_utf8(v).map_err(|_| X509Error::BadEncoding)?;
                self.san_dns.push(name);
            }
            // Other GeneralName variants (rfc822Name, iPAddress, ...) are
            // parsed past but not surfaced.
        }
        Ok(())
    }

    /// Check the validity window against a Unix timestamp.
    pub fn check_validity(&self, now: u64) -> Result<(), X509Error> {
        if now < self.not_before {
            return Err(X509Error::NotYetValid);
        }
        if now > self.not_after {
            return Err(X509Error::Expired);
        }
        Ok(())
    }

    /// Exact (non-wildcard) match of `host` against SubjectAltName dNSNames.
    pub fn matches_dns(&self, host: &str) -> bool {
        self.san_dns.iter().any(|n| n.eq_ignore_ascii_case(host))
    }

    /// Check that `self` is allowed to have issued `child`, and that it did.
    ///
    /// `intermediates_below` is the number of CA certificates between `self`
    /// and the leaf, which is what `pathLenConstraint` bounds.
    pub fn authorizes(
        &self,
        child: &Certificate<'_>,
        intermediates_below: usize,
    ) -> Result<(), X509Error> {
        if self.subject_der != child.issuer_der {
            return Err(X509Error::IssuerMismatch);
        }
        if !self.is_ca {
            return Err(X509Error::IssuerNotCa);
        }
        if self.key_usage != 0 && self.key_usage & KU_KEY_CERT_SIGN == 0 {
            return Err(X509Error::KeyUsageViolation);
        }
        if let Some(limit) = self.path_len {
            if intermediates_below > limit as usize {
                return Err(X509Error::PathLenExceeded);
            }
        }
        verify_ed25519(&self.public_key, child.tbs_der, &child.signature)
    }
}

fn parse_key_usage(value: &[u8]) -> Result<u16, X509Error> {
    let mut outer = Der::new(value);
    let bits = outer.expect(TAG_BITSTR)?;
    outer.finish()?;
    if bits.is_empty() || bits[0] > 7 {
        return Err(X509Error::BadLength);
    }
    let mut usage = 0u16;
    for (i, &byte) in bits[1..].iter().take(2).enumerate() {
        for bit in 0..8 {
            if byte & (0x80 >> bit) != 0 {
                usage |= 1 << (i * 8 + bit);
            }
        }
    }
    Ok(usage)
}

/// Best-effort commonName lookup in an RDNSequence.
fn common_name(dn_body: &[u8]) -> Result<Option<&str>, X509Error> {
    let mut rdns = Der::new(dn_body);
    while !rdns.is_empty() {
        let rdn = rdns.expect(TAG_SET)?;
        let mut attrs = Der::new(rdn);
        while !attrs.is_empty() {
            let attr = attrs.expect(TAG_SEQ)?;
            let mut a = Der::new(attr);
            let oid = a.expect(TAG_OID)?;
            let (_, _, value) = a.next()?;
            if oid == OID_COMMON_NAME {
                return core::str::from_utf8(value)
                    .map(Some)
                    .map_err(|_| X509Error::BadEncoding);
            }
        }
    }
    Ok(None)
}

fn verify_ed25519(public_key: &[u8; 32], msg: &[u8], sig: &[u8; 64]) -> Result<(), X509Error> {
    let vk = VerifyingKey::from_bytes(public_key).map_err(|_| X509Error::BadPublicKey)?;
    let signature = Signature::from_bytes(sig);
    vk.verify_strict(msg, &signature)
        .map_err(|_| X509Error::BadSignature)
}

// ---------------------------------------------------------------------------
// Chain validation
// ---------------------------------------------------------------------------

/// Verify a leaf-first certificate chain against [`TRUST_ANCHORS`].
///
/// `chain[0]` is the leaf, each following entry issues the one before it. A
/// trailing self-signed certificate is dropped: trust comes from the embedded
/// anchor, never from a root the peer supplied.
pub fn verify_chain(chain: &[Certificate<'_>], now: u64) -> Result<(), X509Error> {
    let chain = match chain.last() {
        Some(last) if chain.len() > 1 && last.subject_der == last.issuer_der => {
            &chain[..chain.len() - 1]
        }
        _ => chain,
    };
    if chain.is_empty() {
        return Err(X509Error::EmptyChain);
    }
    if chain.len() > MAX_CHAIN_LEN {
        return Err(X509Error::ChainTooLong);
    }

    for cert in chain {
        cert.check_validity(now)?;
    }
    for i in 0..chain.len() - 1 {
        chain[i + 1].authorizes(&chain[i], i)?;
    }

    let top = &chain[chain.len() - 1];
    for der in TRUST_ANCHORS {
        let anchor = Certificate::parse(der)?;
        if anchor.subject_der != top.issuer_der {
            continue;
        }
        anchor.check_validity(now)?;
        anchor.authorizes(top, chain.len() - 1)?;
        return Ok(());
    }
    Err(X509Error::UnknownIssuer)
}

/// Parse then verify a leaf-first chain of DER blobs.
pub fn verify_chain_der(ders: &[&[u8]], now: u64) -> Result<(), X509Error> {
    let mut chain = Vec::with_capacity(ders.len());
    for der in ders {
        chain.push(Certificate::parse(der)?);
    }
    verify_chain(&chain, now)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::super::certs;
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// Inside every fixture's validity window except the expired one.
    const NOW: u64 = 1_767_225_600; // 2026-01-01T00:00:00Z

    fn test_parse_leaf() -> TestResult {
        let Ok(cert) = Certificate::parse(&certs::LEAF_DER) else {
            return TestResult::Fail("leaf fixture failed to parse");
        };
        test_assert_eq!(cert.common_name, Some("seal.local"));
        test_assert!(!cert.is_ca, "leaf must not be a CA");
        test_assert!(cert.matches_dns("seal.local"), "SAN dNSName missing");
        test_assert!(cert.matches_dns("proof.seal"), "second SAN missing");
        test_assert!(!cert.matches_dns("evil.seal"), "SAN matched a foreign name");
        test_assert!(
            cert.key_usage & KU_DIGITAL_SIGNATURE != 0,
            "leaf KeyUsage digitalSignature missing"
        );
        TestResult::Pass
    }

    fn test_parse_root_generalized_time() -> TestResult {
        let Ok(root) = Certificate::parse(&certs::ROOT_CA_DER) else {
            return TestResult::Fail("root fixture failed to parse");
        };
        test_assert!(root.is_ca, "root must be a CA");
        test_assert_eq!(root.path_len, Some(1));
        // notAfter is 2060-01-01, encoded as GeneralizedTime.
        test_assert_eq!(root.not_after, 2_840_140_800u64);
        test_assert_eq!(root.not_before, 1_577_836_800u64);
        TestResult::Pass
    }

    fn test_valid_chain_accepts() -> TestResult {
        let ders: [&[u8]; 2] = [&certs::LEAF_DER, &certs::INTERMEDIATE_DER];
        test_assert_eq!(verify_chain_der(&ders, NOW), Ok(()));
        TestResult::Pass
    }

    fn test_chain_with_root_supplied_accepts() -> TestResult {
        let ders: [&[u8]; 3] = [
            &certs::LEAF_DER,
            &certs::INTERMEDIATE_DER,
            &certs::ROOT_CA_DER,
        ];
        test_assert_eq!(verify_chain_der(&ders, NOW), Ok(()));
        TestResult::Pass
    }

    fn test_expired_cert_rejects() -> TestResult {
        let ders: [&[u8]; 2] = [&certs::EXPIRED_LEAF_DER, &certs::INTERMEDIATE_DER];
        test_assert_eq!(verify_chain_der(&ders, NOW), Err(X509Error::Expired));
        TestResult::Pass
    }

    fn test_not_yet_valid_rejects() -> TestResult {
        // 2010-01-01, before every fixture's notBefore.
        let ders: [&[u8]; 2] = [&certs::LEAF_DER, &certs::INTERMEDIATE_DER];
        test_assert_eq!(
            verify_chain_der(&ders, 1_262_304_000),
            Err(X509Error::NotYetValid)
        );
        TestResult::Pass
    }

    fn test_wrong_signature_rejects() -> TestResult {
        // Corrupt only the signature BIT STRING (the trailing 64 bytes). The
        // DER stays structurally valid, isolating signature verification.
        let mut tampered = certs::LEAF_DER;
        let last = tampered.len() - 1;
        tampered[last] ^= 0xff;
        let Ok(leaf) = Certificate::parse(&tampered) else {
            return TestResult::Fail("signature tampering broke the DER structure");
        };
        let Ok(inter) = Certificate::parse(&certs::INTERMEDIATE_DER) else {
            return TestResult::Fail("intermediate fixture failed to parse");
        };
        test_assert_eq!(
            verify_chain(&[leaf, inter], NOW),
            Err(X509Error::BadSignature)
        );
        TestResult::Pass
    }

    fn test_any_single_bit_flip_rejects() -> TestResult {
        // Offset-independent statement of the same property: no single-bit
        // mutation anywhere in the leaf can produce an accepted chain. Covers
        // tampering with the serial, DN, validity, key, extensions, and
        // signature alike, and doubles as a crash sweep over the parser.
        let inter: &[u8] = &certs::INTERMEDIATE_DER;
        for i in 0..certs::LEAF_DER.len() {
            let mut fuzzed = certs::LEAF_DER;
            fuzzed[i] ^= 0x01;
            let leaf: &[u8] = &fuzzed;
            if verify_chain_der(&[leaf, inter], NOW).is_ok() {
                return TestResult::Fail("a mutated certificate verified");
            }
        }
        TestResult::Pass
    }

    fn test_truncated_der_rejects() -> TestResult {
        // Every prefix of a valid certificate must produce an error, not a
        // panic and not a partially-populated certificate.
        for cut in 0..certs::LEAF_DER.len() {
            if Certificate::parse(&certs::LEAF_DER[..cut]).is_ok() {
                return TestResult::Fail("a truncated certificate parsed");
            }
        }
        TestResult::Pass
    }

    fn test_trailing_garbage_rejects() -> TestResult {
        let mut extended = certs::LEAF_DER.to_vec();
        extended.push(0x00);
        test_assert_eq!(
            Certificate::parse(&extended).err(),
            Some(X509Error::TrailingData)
        );
        TestResult::Pass
    }

    fn test_corrupt_bytes_never_panic() -> TestResult {
        // Heavier single-byte corruption across the whole certificate: the
        // parser may accept or reject, but must never panic or index out of
        // bounds.
        for i in 0..certs::LEAF_DER.len() {
            let mut fuzzed = certs::LEAF_DER;
            fuzzed[i] ^= 0xa5;
            let _ = Certificate::parse(&fuzzed);
        }
        TestResult::Pass
    }

    fn test_ca_flag_violation_rejects() -> TestResult {
        // ROGUE_LEAF_DER is validly signed by the leaf's key, but the leaf
        // carries BasicConstraints CA:FALSE.
        let ders: [&[u8]; 3] = [
            &certs::ROGUE_LEAF_DER,
            &certs::LEAF_DER,
            &certs::INTERMEDIATE_DER,
        ];
        test_assert_eq!(verify_chain_der(&ders, NOW), Err(X509Error::IssuerNotCa));
        TestResult::Pass
    }

    fn test_untrusted_chain_rejects() -> TestResult {
        // The leaf alone: its issuer is the intermediate, which is not an
        // anchor, so there is nothing to root the chain in.
        let ders: [&[u8]; 1] = [&certs::LEAF_DER];
        test_assert_eq!(verify_chain_der(&ders, NOW), Err(X509Error::UnknownIssuer));
        TestResult::Pass
    }

    fn test_empty_chain_rejects() -> TestResult {
        test_assert_eq!(verify_chain(&[], NOW), Err(X509Error::EmptyChain));
        TestResult::Pass
    }

    fn test_chain_too_long_rejects() -> TestResult {
        let mut chain: Vec<Certificate<'_>> = Vec::new();
        for _ in 0..MAX_CHAIN_LEN + 1 {
            match Certificate::parse(&certs::LEAF_DER) {
                Ok(c) => chain.push(c),
                Err(_) => return TestResult::Fail("leaf fixture failed to parse"),
            }
        }
        test_assert_eq!(verify_chain(&chain, NOW), Err(X509Error::ChainTooLong));
        TestResult::Pass
    }

    fn test_bad_length_encodings_reject() -> TestResult {
        // BER indefinite length, non-minimal long form, and a length that
        // overruns the buffer.
        test_assert_eq!(
            Certificate::parse(&[0x30, 0x80, 0x00, 0x00]).err(),
            Some(X509Error::BadLength)
        );
        test_assert_eq!(
            Certificate::parse(&[0x30, 0x81, 0x02, 0x00, 0x00]).err(),
            Some(X509Error::BadLength)
        );
        test_assert_eq!(
            Certificate::parse(&[0x30, 0x84, 0xff, 0xff, 0xff, 0xff, 0x00]).err(),
            Some(X509Error::Truncated)
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("x509::parse_leaf", test_parse_leaf);
        crate::testing::register_test(
            "x509::parse_root_generalized_time",
            test_parse_root_generalized_time,
        );
        crate::testing::register_test("x509::valid_chain_accepts", test_valid_chain_accepts);
        crate::testing::register_test(
            "x509::chain_with_root_supplied_accepts",
            test_chain_with_root_supplied_accepts,
        );
        crate::testing::register_test("x509::expired_cert_rejects", test_expired_cert_rejects);
        crate::testing::register_test("x509::not_yet_valid_rejects", test_not_yet_valid_rejects);
        crate::testing::register_test(
            "x509::wrong_signature_rejects",
            test_wrong_signature_rejects,
        );
        crate::testing::register_test(
            "x509::any_single_bit_flip_rejects",
            test_any_single_bit_flip_rejects,
        );
        crate::testing::register_test("x509::truncated_der_rejects", test_truncated_der_rejects);
        crate::testing::register_test(
            "x509::trailing_garbage_rejects",
            test_trailing_garbage_rejects,
        );
        crate::testing::register_test(
            "x509::corrupt_bytes_never_panic",
            test_corrupt_bytes_never_panic,
        );
        crate::testing::register_test(
            "x509::ca_flag_violation_rejects",
            test_ca_flag_violation_rejects,
        );
        crate::testing::register_test(
            "x509::untrusted_chain_rejects",
            test_untrusted_chain_rejects,
        );
        crate::testing::register_test("x509::empty_chain_rejects", test_empty_chain_rejects);
        crate::testing::register_test("x509::chain_too_long_rejects", test_chain_too_long_rejects);
        crate::testing::register_test(
            "x509::bad_length_encodings_reject",
            test_bad_length_encodings_reject,
        );
    }
}
