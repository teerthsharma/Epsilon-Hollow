// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Remote release channel for ManifoldPkg.
//!
//! A release channel is an endpoint that serves a signed index plus the `.eph`
//! packages the index names. Everything fetched over the channel is
//! attacker-controlled until proven otherwise, so the install path applies three
//! independent checks in order and refuses on the first failure:
//!
//! 1. ed25519 over the index body (`EPHIDX2` framing, same shape as the local
//!    `EPHIDX1` boot fixture, extended with `index_version` and repeated
//!    `entry=` lines).
//! 2. monotonic `index_version` — an index older than the last accepted one is
//!    a replay and is refused, which blocks the downgrade attack that
//!    reintroduces a fixed package.
//! 3. SHA-256 of the fetched package bytes against the digest the signed index
//!    carries for that entry, then the package's own ed25519 signature via the
//!    existing [`super::ManifoldPkg::install_bytes`] path.
//!
//! There is no unverified fallback. Every failure — no NIC, no DNS, bad TLS,
//! unreachable endpoint, bad signature, stale index, wrong digest — surfaces as
//! a typed [`ChannelError`] and installs nothing.
//!
//! Index wire format (signed body is everything between the magic line and
//! `signature=`):
//!
//! ```text
//! EPHIDX2
//! index_version=<u64>
//! entry=<name>,<version>,<bytes>,<sha256-hex>
//! signature=<ed25519-hex>
//! ```

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};

use super::{hex32, hex64, parse_hex64, ManifoldPkg};

const INDEX_MAGIC: &str = "EPHIDX2\n";

/// Configured release channel endpoint. The fixture transport serves URLs under
/// this prefix; the live probe attempts a real HTTPS fetch against it.
pub const CHANNEL_ENDPOINT: &str = "https://releases.seal-os.local/channel/stable/";

/// Fixture signing key for the channel index. Distinct from the package signing
/// key so index verification and package verification are provably separate.
const CHANNEL_INDEX_SIGNING_KEY: [u8; 32] = [
    0x7e, 0x11, 0xc4, 0x08, 0x9a, 0x35, 0xd6, 0x62, 0x4f, 0x80, 0x2b, 0xe7, 0x13, 0xa9, 0x5c, 0x0d,
    0x46, 0xf2, 0x88, 0x31, 0xbb, 0x0a, 0x67, 0xd4, 0x9e, 0x25, 0x70, 0xc3, 0x1f, 0xa6, 0x54, 0x39,
];

/// Typed refusal reasons. Every remote install failure names exactly one.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelError {
    /// No NIC — nothing to fetch over.
    NoNetwork,
    /// Endpoint is not `https://`. The channel refuses plaintext transports.
    InsecureScheme,
    /// Hostname could not be resolved.
    DnsFailed,
    /// TCP/TLS could not carry the request.
    TransportFailed,
    /// Endpoint answered with a non-200 status.
    HttpStatus(u16),
    /// Index framing, encoding, or field layout is not `EPHIDX2`.
    IndexMalformed,
    /// ed25519 over the index body did not verify.
    IndexSignatureInvalid,
    /// Offered index is not newer than the last accepted one — replay refused.
    IndexRollback { accepted: u64, offered: u64 },
    /// Index carries no entry for the requested name/version.
    EntryNotFound,
    /// Fetched package length disagrees with the signed index.
    SizeMismatch,
    /// SHA-256 of the fetched package disagrees with the signed index.
    DigestMismatch,
    /// Package parsed and digested clean but its own signature was rejected.
    PackageRejected,
}

impl ChannelError {
    /// Stable single-token reason, safe to embed in a proof line.
    pub fn reason(&self) -> &'static str {
        match self {
            ChannelError::NoNetwork => "no_network",
            ChannelError::InsecureScheme => "insecure_scheme",
            ChannelError::DnsFailed => "dns_failed",
            ChannelError::TransportFailed => "transport_failed",
            ChannelError::HttpStatus(_) => "http_status",
            ChannelError::IndexMalformed => "index_malformed",
            ChannelError::IndexSignatureInvalid => "index_signature_invalid",
            ChannelError::IndexRollback { .. } => "index_rollback",
            ChannelError::EntryNotFound => "entry_not_found",
            ChannelError::SizeMismatch => "size_mismatch",
            ChannelError::DigestMismatch => "digest_mismatch",
            ChannelError::PackageRejected => "package_rejected",
        }
    }
}

/// Byte source for a channel URL. Implemented by the real HTTPS transport and
/// by the checked-in loopback fixture; the verification, rollback, and install
/// logic above it is identical for both.
pub trait ChannelTransport {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, ChannelError>;
}

/// Real transport: DNS + TCP + TLS via the kernel HTTP client. Refuses before
/// touching the wire when there is no NIC or the endpoint is not HTTPS.
pub struct HttpsTransport;

impl ChannelTransport for HttpsTransport {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, ChannelError> {
        if !url.starts_with("https://") {
            return Err(ChannelError::InsecureScheme);
        }
        if !crate::net::has_nic() {
            return Err(ChannelError::NoNetwork);
        }
        let response = crate::drivers::net::http::HttpClient::new()
            .get(url)
            .map_err(|e| {
                if e.starts_with("DNS") {
                    ChannelError::DnsFailed
                } else {
                    ChannelError::TransportFailed
                }
            })?;
        if response.status != 200 {
            return Err(ChannelError::HttpStatus(response.status));
        }
        Ok(response.body)
    }
}

/// In-memory loopback transport used by the checked-in release fixture and by
/// the negative controls. Only the byte transport is simulated — the caller
/// still drives the real parse, signature, rollback, digest, and install code.
pub struct FixtureTransport {
    routes: Vec<(String, Vec<u8>)>,
}

impl FixtureTransport {
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    /// Publish (or replace) the body served at `url`.
    pub fn serve(&mut self, url: &str, body: &[u8]) {
        for route in self.routes.iter_mut() {
            if route.0 == url {
                route.1 = body.to_vec();
                return;
            }
        }
        self.routes.push((String::from(url), body.to_vec()));
    }
}

impl Default for FixtureTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelTransport for FixtureTransport {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, ChannelError> {
        for (route, body) in &self.routes {
            if route == url {
                return Ok(body.clone());
            }
        }
        Err(ChannelError::HttpStatus(404))
    }
}

/// One package the signed index vouches for.
#[derive(Debug, Clone)]
pub struct ReleaseEntry {
    pub name: String,
    pub version: String,
    pub bytes: usize,
    pub sha256: String,
}

/// A parsed, signature-verified index.
#[derive(Debug, Clone)]
pub struct ReleaseIndex {
    pub index_version: u64,
    pub entries: Vec<ReleaseEntry>,
}

/// A configured channel plus the rollback floor accepted so far.
pub struct ReleaseChannel {
    endpoint: String,
    index_key: [u8; 32],
    package_key: [u8; 32],
    accepted_index_version: u64,
}

impl ReleaseChannel {
    pub fn new(endpoint: &str, index_key: [u8; 32], package_key: [u8; 32]) -> Self {
        let mut endpoint = String::from(endpoint);
        if !endpoint.ends_with('/') {
            endpoint.push('/');
        }
        Self {
            endpoint,
            index_key,
            package_key,
            accepted_index_version: 0,
        }
    }

    pub fn index_url(&self) -> String {
        format!("{}index", self.endpoint)
    }

    pub fn package_url(&self, name: &str, version: &str) -> String {
        format!("{}{}-{}.eph", self.endpoint, name, version)
    }

    pub fn accepted_index_version(&self) -> u64 {
        self.accepted_index_version
    }

    /// Fetch, verify, and accept the channel index. Advances the rollback floor
    /// only after both the signature and the monotonicity check pass.
    pub fn fetch_index(
        &mut self,
        transport: &dyn ChannelTransport,
    ) -> Result<ReleaseIndex, ChannelError> {
        let raw = transport.fetch(&self.index_url())?;
        let index = parse_index(&raw, &self.index_key)?;
        if index.index_version <= self.accepted_index_version {
            return Err(ChannelError::IndexRollback {
                accepted: self.accepted_index_version,
                offered: index.index_version,
            });
        }
        self.accepted_index_version = index.index_version;
        Ok(index)
    }

    /// Full remote install: signed index, monotonic version, per-entry digest,
    /// then the package's own signature. Installs nothing unless all four hold.
    pub fn install_into(
        &mut self,
        pkg: &mut ManifoldPkg,
        transport: &dyn ChannelTransport,
        name: &str,
        version: &str,
    ) -> Result<String, ChannelError> {
        let index = self.fetch_index(transport)?;
        let entry = index
            .entries
            .iter()
            .find(|e| e.name == name && e.version == version)
            .ok_or(ChannelError::EntryNotFound)?;

        let body = transport.fetch(&self.package_url(name, version))?;
        if body.len() != entry.bytes {
            return Err(ChannelError::SizeMismatch);
        }
        let mut hasher = Sha256::new();
        hasher.update(&body);
        let digest: [u8; 32] = hasher.finalize().into();
        if hex32(&digest) != entry.sha256 {
            return Err(ChannelError::DigestMismatch);
        }

        pkg.install_bytes(&body, Some(&self.package_key))
            .map_err(|_| ChannelError::PackageRejected)
    }
}

/// Parse and signature-verify an `EPHIDX2` index body.
pub fn parse_index(raw: &[u8], index_key: &[u8; 32]) -> Result<ReleaseIndex, ChannelError> {
    let text = core::str::from_utf8(raw).map_err(|_| ChannelError::IndexMalformed)?;
    if !text.starts_with(INDEX_MAGIC) {
        return Err(ChannelError::IndexMalformed);
    }
    let sig_pos = text
        .find("signature=")
        .ok_or(ChannelError::IndexMalformed)?;
    let signed = &text[INDEX_MAGIC.len()..sig_pos];
    let sig_hex = text[sig_pos + "signature=".len()..].trim();
    let sig_bytes = parse_hex64(sig_hex).ok_or(ChannelError::IndexMalformed)?;
    let vk = VerifyingKey::from_bytes(index_key).map_err(|_| ChannelError::IndexMalformed)?;
    vk.verify_strict(signed.as_bytes(), &Signature::from_bytes(&sig_bytes))
        .map_err(|_| ChannelError::IndexSignatureInvalid)?;

    let mut index_version = None;
    let mut entries = Vec::new();
    for line in signed.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("index_version=") {
            index_version = Some(rest.parse::<u64>().map_err(|_| ChannelError::IndexMalformed)?);
        } else if let Some(rest) = line.strip_prefix("entry=") {
            let mut parts = rest.split(',');
            let name = parts.next().ok_or(ChannelError::IndexMalformed)?;
            let version = parts.next().ok_or(ChannelError::IndexMalformed)?;
            let bytes = parts.next().ok_or(ChannelError::IndexMalformed)?;
            let sha256 = parts.next().ok_or(ChannelError::IndexMalformed)?;
            if parts.next().is_some() || sha256.len() != 64 {
                return Err(ChannelError::IndexMalformed);
            }
            entries.push(ReleaseEntry {
                name: String::from(name),
                version: String::from(version),
                bytes: bytes.parse::<usize>().map_err(|_| ChannelError::IndexMalformed)?,
                sha256: String::from(sha256),
            });
        }
    }
    let index_version = index_version.ok_or(ChannelError::IndexMalformed)?;
    if entries.is_empty() {
        return Err(ChannelError::IndexMalformed);
    }
    Ok(ReleaseIndex {
        index_version,
        entries,
    })
}

fn channel_index_public_key() -> [u8; 32] {
    SigningKey::from_bytes(&CHANNEL_INDEX_SIGNING_KEY)
        .verifying_key()
        .to_bytes()
}

/// Build a signed index vouching for `package` at `name`/`version`.
pub fn build_index(index_version: u64, name: &str, version: &str, package: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(package);
    let digest: [u8; 32] = hasher.finalize().into();
    let body = format!(
        "index_version={}\nentry={},{},{},{}\n",
        index_version,
        name,
        version,
        package.len(),
        hex32(&digest)
    );
    let signature = SigningKey::from_bytes(&CHANNEL_INDEX_SIGNING_KEY).sign(body.as_bytes());
    let mut out = Vec::new();
    out.extend_from_slice(INDEX_MAGIC.as_bytes());
    out.extend_from_slice(body.as_bytes());
    out.extend_from_slice(b"signature=");
    out.extend_from_slice(hex64(&signature.to_bytes()).as_bytes());
    out.push(b'\n');
    out
}

/// Checked-in hosted release fixture: a signed index and the package it names,
/// served over the loopback transport at [`CHANNEL_ENDPOINT`].
pub fn fixture_transport(
    channel: &ReleaseChannel,
    index_version: u64,
    name: &str,
    version: &str,
    package: &[u8],
) -> FixtureTransport {
    let mut transport = FixtureTransport::new();
    transport.serve(
        &channel.index_url(),
        &build_index(index_version, name, version, package),
    );
    transport.serve(&channel.package_url(name, version), package);
    transport
}

/// A channel wired to the fixture index key and the boot proof package key.
pub fn fixture_channel() -> ReleaseChannel {
    ReleaseChannel::new(
        CHANNEL_ENDPOINT,
        channel_index_public_key(),
        super::proof_pkg_public_key(),
    )
}

/// Strip a `.eph` signature in place, leaving a structurally valid package that
/// must still be refused by the package signature check.
fn unsign_eph(eph: &[u8]) -> Vec<u8> {
    let mut out = eph.to_vec();
    if out.len() >= 8 {
        let manifest_len = u32::from_be_bytes([out[4], out[5], out[6], out[7]]) as usize;
        let start = 8 + manifest_len;
        if start + 64 <= out.len() {
            for byte in &mut out[start..start + 64] {
                *byte = 0;
            }
        }
    }
    out
}

/// Measured facts for the `[ManifoldPkg] proof` remote fields. Every field is
/// produced by driving the real channel code at boot; nothing is asserted.
pub struct ChannelProof {
    pub transport: &'static str,
    pub index_signature_ok: bool,
    pub index_version: u64,
    pub packages_fetched: usize,
    pub digest_ok: usize,
    pub rollback_refused: bool,
    pub tamper_refused: bool,
    pub digest_mismatch_refused: bool,
    pub package_signature_enforced: bool,
    pub live_probe: &'static str,
    pub fail_closed: bool,
}

impl ChannelProof {
    pub fn ok(&self) -> bool {
        self.index_signature_ok
            && self.packages_fetched == 1
            && self.digest_ok == 1
            && self.rollback_refused
            && self.tamper_refused
            && self.digest_mismatch_refused
            && self.package_signature_enforced
            && self.fail_closed
    }
}

/// Drive the remote release channel end to end and report what was measured.
/// `pkg` is mutated (one install, then removed) and left as it was found.
pub fn measure(pkg: &mut ManifoldPkg, eph: &[u8]) -> ChannelProof {
    const NAME: &str = super::PROOF_PKG_NAME;
    const VERSION: &str = super::PROOF_PKG_VERSION;

    // Positive control: signed index v3, digest match, valid package signature.
    let mut channel = fixture_channel();
    let good = fixture_transport(&channel, 3, NAME, VERSION, eph);
    let installed = channel.install_into(pkg, &good, NAME, VERSION).is_ok();
    let index_version = channel.accepted_index_version();
    let index_signature_ok = index_version == 3;
    let digest_ok = usize::from(installed);
    let packages_fetched = usize::from(installed);
    if installed {
        let _ = pkg.remove(NAME);
    }

    // Rollback: replay index v2 against the same channel that accepted v3.
    let stale = fixture_transport(&channel, 2, NAME, VERSION, eph);
    let rollback_refused = matches!(
        channel.install_into(pkg, &stale, NAME, VERSION),
        Err(ChannelError::IndexRollback { .. })
    );

    // Tampered index: flip a byte of the signed body, leave the signature.
    let mut fresh = fixture_channel();
    let mut tampered_bytes = build_index(4, NAME, VERSION, eph);
    if let Some(byte) = tampered_bytes.get_mut(INDEX_MAGIC.len()) {
        *byte ^= 0x20;
    }
    let mut tampered = fixture_transport(&fresh, 4, NAME, VERSION, eph);
    tampered.serve(&fresh.index_url(), &tampered_bytes);
    let tamper_refused = matches!(
        fresh.install_into(pkg, &tampered, NAME, VERSION),
        Err(ChannelError::IndexSignatureInvalid) | Err(ChannelError::IndexMalformed)
    );

    // Corrupted package body: index still valid, fetched bytes mutated.
    let mut corrupt_channel = fixture_channel();
    let mut corrupt = fixture_transport(&corrupt_channel, 5, NAME, VERSION, eph);
    let mut corrupt_body = eph.to_vec();
    if let Some(byte) = corrupt_body.last_mut() {
        *byte ^= 0xff;
    }
    corrupt.serve(&corrupt_channel.package_url(NAME, VERSION), &corrupt_body);
    let digest_mismatch_refused = matches!(
        corrupt_channel.install_into(pkg, &corrupt, NAME, VERSION),
        Err(ChannelError::DigestMismatch) | Err(ChannelError::SizeMismatch)
    );

    // Package signature still enforced after a valid fetch: the index vouches
    // for the unsigned bytes, so the digest matches and only the package
    // signature can refuse it.
    let mut unsigned_channel = fixture_channel();
    let unsigned = unsign_eph(eph);
    let unsigned_transport = fixture_transport(&unsigned_channel, 6, NAME, VERSION, &unsigned);
    let package_signature_enforced = matches!(
        unsigned_channel.install_into(pkg, &unsigned_transport, NAME, VERSION),
        Err(ChannelError::PackageRejected)
    );

    // Live probe: the real HTTPS transport against the configured endpoint.
    // Must refuse with a typed reason and install nothing.
    let before = pkg.package_count();
    let mut live = fixture_channel();
    let live_probe = match live.install_into(pkg, &HttpsTransport, NAME, VERSION) {
        Ok(_) => "reachable",
        Err(e) => e.reason(),
    };
    let fail_closed = live_probe != "reachable" && pkg.package_count() == before;

    ChannelProof {
        transport: "fixture_loopback",
        index_signature_ok,
        index_version,
        packages_fetched,
        digest_ok,
        rollback_refused,
        tamper_refused,
        digest_mismatch_refused,
        package_signature_enforced,
        live_probe,
        fail_closed,
    }
}

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    const NAME: &str = super::super::PROOF_PKG_NAME;
    const VERSION: &str = super::super::PROOF_PKG_VERSION;

    fn eph() -> Vec<u8> {
        super::super::build_proof_eph()
    }

    fn test_good_index_accepts() -> TestResult {
        let package = eph();
        let mut channel = fixture_channel();
        let transport = fixture_transport(&channel, 7, NAME, VERSION, &package);
        let index = match channel.fetch_index(&transport) {
            Ok(index) => index,
            Err(_) => return TestResult::Fail("signed fixture index must verify"),
        };
        test_assert_eq!(index.index_version, 7);
        test_assert_eq!(index.entries.len(), 1);
        test_assert_eq!(index.entries[0].bytes, package.len());
        test_assert!(index.entries[0].name == NAME, "entry names the package");
        TestResult::Pass
    }

    fn test_tampered_index_refused() -> TestResult {
        let package = eph();
        let mut channel = fixture_channel();
        // Mutate the signed body, leave the signature untouched.
        let mut raw = build_index(7, NAME, VERSION, &package);
        raw[INDEX_MAGIC.len() + 2] ^= 0x40;
        let mut transport = fixture_transport(&channel, 7, NAME, VERSION, &package);
        transport.serve(&channel.index_url(), &raw);
        match channel.fetch_index(&transport) {
            Err(ChannelError::IndexSignatureInvalid) | Err(ChannelError::IndexMalformed) => {}
            _ => return TestResult::Fail("tampered index body must be refused"),
        }
        test_assert_eq!(channel.accepted_index_version(), 0);
        TestResult::Pass
    }

    fn test_stale_version_refused() -> TestResult {
        let package = eph();
        let mut channel = fixture_channel();
        let newer = fixture_transport(&channel, 9, NAME, VERSION, &package);
        test_assert!(channel.fetch_index(&newer).is_ok(), "v9 index must accept");
        let older = fixture_transport(&channel, 8, NAME, VERSION, &package);
        match channel.fetch_index(&older) {
            Err(ChannelError::IndexRollback { accepted, offered }) => {
                test_assert_eq!(accepted, 9);
                test_assert_eq!(offered, 8);
            }
            _ => return TestResult::Fail("replayed older index must be refused"),
        }
        test_assert_eq!(channel.accepted_index_version(), 9);
        TestResult::Pass
    }

    fn test_digest_mismatch_refused() -> TestResult {
        let package = eph();
        let mut channel = fixture_channel();
        let mut transport = fixture_transport(&channel, 10, NAME, VERSION, &package);
        let mut corrupted = package.clone();
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0xff;
        transport.serve(&channel.package_url(NAME, VERSION), &corrupted);
        let mut pkg = ManifoldPkg::new();
        let before = pkg.package_count();
        match channel.install_into(&mut pkg, &transport, NAME, VERSION) {
            Err(ChannelError::DigestMismatch) => {}
            _ => return TestResult::Fail("corrupted package body must be refused"),
        }
        test_assert_eq!(pkg.package_count(), before);
        TestResult::Pass
    }

    fn test_unreachable_channel_fails_closed() -> TestResult {
        let mut channel = fixture_channel();
        let mut pkg = ManifoldPkg::new();
        let before = pkg.package_count();
        let err = channel
            .install_into(&mut pkg, &HttpsTransport, NAME, VERSION)
            .err();
        test_assert!(err.is_some(), "live channel must not resolve under test");
        test_assert_eq!(pkg.package_count(), before);
        test_assert_eq!(channel.accepted_index_version(), 0);
        TestResult::Pass
    }

    fn test_package_signature_still_enforced() -> TestResult {
        // The index vouches for the unsigned bytes, so index signature, version
        // and digest all pass; only the package signature can refuse this.
        let package = unsign_eph(&eph());
        let mut channel = fixture_channel();
        let transport = fixture_transport(&channel, 11, NAME, VERSION, &package);
        let mut pkg = ManifoldPkg::new();
        match channel.install_into(&mut pkg, &transport, NAME, VERSION) {
            Err(ChannelError::PackageRejected) => {}
            _ => return TestResult::Fail("unsigned package must be refused after fetch"),
        }
        test_assert_eq!(channel.accepted_index_version(), 11);
        TestResult::Pass
    }

    fn test_insecure_scheme_refused() -> TestResult {
        let result = HttpsTransport.fetch("http://releases.seal-os.local/channel/stable/index");
        test_assert_eq!(result.err(), Some(ChannelError::InsecureScheme));
        TestResult::Pass
    }

    fn test_entry_not_found_refused() -> TestResult {
        let package = eph();
        let mut channel = fixture_channel();
        let transport = fixture_transport(&channel, 12, NAME, VERSION, &package);
        let mut pkg = ManifoldPkg::new();
        test_assert_eq!(
            channel
                .install_into(&mut pkg, &transport, "not-in-index", VERSION)
                .err(),
            Some(ChannelError::EntryNotFound)
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("pkg::channel_good_index_accepts", test_good_index_accepts);
        crate::testing::register_test(
            "pkg::channel_tampered_index_refused",
            test_tampered_index_refused,
        );
        crate::testing::register_test(
            "pkg::channel_stale_version_refused",
            test_stale_version_refused,
        );
        crate::testing::register_test(
            "pkg::channel_digest_mismatch_refused",
            test_digest_mismatch_refused,
        );
        crate::testing::register_test(
            "pkg::channel_unreachable_fails_closed",
            test_unreachable_channel_fails_closed,
        );
        crate::testing::register_test(
            "pkg::channel_package_signature_enforced",
            test_package_signature_still_enforced,
        );
        crate::testing::register_test(
            "pkg::channel_insecure_scheme_refused",
            test_insecure_scheme_refused,
        );
        crate::testing::register_test(
            "pkg::channel_entry_not_found_refused",
            test_entry_not_found_refused,
        );
    }
}

