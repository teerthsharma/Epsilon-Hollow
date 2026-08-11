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
//!    reintroduces a fixed package. The floor survives reboots: it is written
//!    to [`FLOOR_PATH`] as a signed [`FLOOR_MAGIC`] record (see `load_floor`
//!    for what a missing or unreadable one does).
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

/// Where the rollback floor is kept between boots.
const FLOOR_PATH: &str = "/packages/.channel_floor";

/// Framing of the on-disk floor record: `FLOOR_MAGIC` (8) then the floor as a
/// big-endian `u64` (8), then ed25519 over those first 16 bytes (64). Fixed
/// length on purpose — a short read is then a refusal rather than a parse, and
/// rewriting the record can never leave a longer stale one behind it.
const FLOOR_MAGIC: &[u8; 8] = b"EPHFLR1\0";
const FLOOR_RECORD_LEN: usize = 80;

/// Signing key for the floor record, distinct from both the channel index key
/// and the package key because it authenticates something else entirely: the
/// kernel's own note to its next boot. Nothing off the network is ever checked
/// against it, and no floor record is ever checked against a channel key, so
/// neither signature can be replayed as the other.
///
/// ponytail: the key ships in the boot image, so this stops an attacker who
/// can write the data partition (`/packages` on ext2) without reading the
/// image on the ESP — it does not stop one who has both. Closing that, and
/// closing the delete-and-replay hole `load_floor` documents, needs monotonic
/// storage the kernel does not have yet: a TPM NV counter or a UEFI
/// authenticated variable.
const FLOOR_RECORD_KEY: [u8; 32] = [
    0xc2, 0x5b, 0x90, 0x3d, 0x17, 0xe8, 0x4a, 0x71, 0x06, 0xbd, 0xf3, 0x28, 0x5e, 0x94, 0x0c, 0xa7,
    0x33, 0x6f, 0xd1, 0x82, 0x49, 0xab, 0x7c, 0x15, 0xe0, 0x38, 0x62, 0xcf, 0x9b, 0x24, 0x50, 0xd6,
];

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
    /// The persisted rollback floor could not be read back or could not be
    /// advanced. Nothing is installed while the floor is in doubt.
    FloorUnavailable,
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
            ChannelError::FloorUnavailable => "floor_unavailable",
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
    /// Where this channel's floor is persisted, or `None` for a channel whose
    /// floor must not outlive it — see [`ReleaseChannel::ephemeral`].
    floor_path: Option<&'static str>,
}

impl ReleaseChannel {
    /// A channel whose rollback floor persists across boots. The floor is read
    /// lazily on the first fetch, not here, so a channel can be constructed
    /// before the VFS is up.
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
            floor_path: Some(FLOOR_PATH),
        }
    }

    /// A channel whose floor lives and dies with the struct.
    ///
    /// This is for the fixture and self-proof channels only. They drive
    /// deliberately arbitrary index versions (the boot proof replays v2 after
    /// v3 on purpose), so persisting their floor would both wreck the proof on
    /// the second boot and leave the shared floor sitting at a fixture's
    /// version. Every channel that faces the network uses [`Self::new`].
    pub fn ephemeral(endpoint: &str, index_key: [u8; 32], package_key: [u8; 32]) -> Self {
        Self {
            floor_path: None,
            ..Self::new(endpoint, index_key, package_key)
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
    ///
    /// The floor compared against is the higher of this channel's own accepted
    /// version and the one persisted on disk, so an index that is stale for
    /// either reason is refused. The new floor is written *before* the index is
    /// returned: an index that cannot have its floor recorded is refused
    /// outright, because accepting it and losing the floor would let the very
    /// package it replaces be served again on the next boot.
    pub fn fetch_index(
        &mut self,
        transport: &dyn ChannelTransport,
    ) -> Result<ReleaseIndex, ChannelError> {
        let raw = transport.fetch(&self.index_url())?;
        let index = parse_index(&raw, &self.index_key)?;
        let floor = match self.floor_path {
            Some(path) => self.accepted_index_version.max(load_floor(path)?),
            None => self.accepted_index_version,
        };
        if index.index_version <= floor {
            return Err(ChannelError::IndexRollback {
                accepted: floor,
                offered: index.index_version,
            });
        }
        if let Some(path) = self.floor_path {
            persist_floor(path, index.index_version)?;
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

/// Read the floor that survived the last boot.
///
/// Which way each failure falls, and why:
///
/// * **No record at all** — `Ok(0)`. No floor was ever established, so there is
///   nothing to roll back below; refusing here would mean a freshly installed
///   system could never accept a first index and the channel would be dead on
///   arrival. This is also the state an attacker who can *delete* the file
///   reaches, and no amount of signing fixes that — see [`FLOOR_RECORD_KEY`].
/// * **A record that is short, misframed, or not signed by this kernel** —
///   [`ChannelError::FloorUnavailable`], which refuses every install until the
///   file is removed by hand. Treating a damaged record as zero is exactly the
///   attack: it turns "I can flip one byte" into "the floor is gone".
fn load_floor(path: &str) -> Result<u64, ChannelError> {
    let mut record = [0u8; FLOOR_RECORD_LEN];
    let read = crate::fs::vfs::with_vfs(|vfs| {
        let handle = match vfs.lookup_follow(path) {
            Ok(handle) => handle,
            Err(crate::fs::vfs::VfsError::NotFound) => return Ok(None),
            Err(_) => return Err(()),
        };
        vfs.read(handle, &mut record, 0).map(Some).map_err(|_| ())
    })
    .map_err(|_| ChannelError::FloorUnavailable)?;
    let Some(read) = read else {
        return Ok(0);
    };
    if read != FLOOR_RECORD_LEN || &record[..FLOOR_MAGIC.len()] != FLOOR_MAGIC {
        return Err(ChannelError::FloorUnavailable);
    }
    let mut signature = [0u8; 64];
    signature.copy_from_slice(&record[16..]);
    SigningKey::from_bytes(&FLOOR_RECORD_KEY)
        .verifying_key()
        .verify_strict(&record[..16], &Signature::from_bytes(&signature))
        .map_err(|_| ChannelError::FloorUnavailable)?;
    let mut floor = [0u8; 8];
    floor.copy_from_slice(&record[8..16]);
    Ok(u64::from_be_bytes(floor))
}

/// Move the persisted floor to `floor`.
///
/// The stored value is re-read and anything not strictly higher is a no-op, so
/// the floor is monotonic as a property of the store itself rather than of
/// whoever calls it. A record that fails to read back is *not* overwritten:
/// `load_floor`'s error propagates, so a damaged floor is never silently
/// replaced by a lower one.
fn persist_floor(path: &str, floor: u64) -> Result<(), ChannelError> {
    if floor <= load_floor(path)? {
        return Ok(());
    }
    let mut record = [0u8; FLOOR_RECORD_LEN];
    record[..FLOOR_MAGIC.len()].copy_from_slice(FLOOR_MAGIC);
    record[8..16].copy_from_slice(&floor.to_be_bytes());
    let signature = SigningKey::from_bytes(&FLOOR_RECORD_KEY).sign(&record[..16]);
    record[16..].copy_from_slice(&signature.to_bytes());
    crate::fs::vfs::with_vfs(|vfs| {
        if let Some(slash) = path.rfind('/').filter(|slash| *slash > 0) {
            let _ = vfs.mkdir(&path[..slash]);
        }
        let handle = match vfs.create(path) {
            Ok(handle) => handle,
            Err(crate::fs::vfs::VfsError::AlreadyExists) => {
                vfs.lookup_follow(path).map_err(|_| ())?
            }
            Err(_) => return Err(()),
        };
        match vfs.write(handle, &record, 0) {
            Ok(written) if written == FLOOR_RECORD_LEN => Ok(()),
            _ => Err(()),
        }
    })
    .map_err(|_| ChannelError::FloorUnavailable)
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
            index_version = Some(
                rest.parse::<u64>()
                    .map_err(|_| ChannelError::IndexMalformed)?,
            );
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
                bytes: bytes
                    .parse::<usize>()
                    .map_err(|_| ChannelError::IndexMalformed)?,
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
/// Deliberately [`ReleaseChannel::ephemeral`]: the fixtures replay old index
/// versions on purpose, which must never touch the floor a real channel keeps.
pub fn fixture_channel() -> ReleaseChannel {
    ReleaseChannel::ephemeral(
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

    /// A channel wired exactly like [`fixture_channel`] but keeping its floor
    /// where a real channel keeps it: on disk.
    fn persistent_channel() -> ReleaseChannel {
        ReleaseChannel::new(
            CHANNEL_ENDPOINT,
            channel_index_public_key(),
            super::super::proof_pkg_public_key(),
        )
    }

    /// Return the persisted floor to "never established". This is also the one
    /// move an attacker with delete access to the data partition has, so every
    /// floor test starts from it and none depend on registration order.
    fn clear_floor() {
        let _ = crate::fs::vfs::with_vfs(|vfs| vfs.unlink(FLOOR_PATH));
    }

    /// The floor record exactly as it sits on disk. Empty when there is none.
    fn read_floor_record() -> Vec<u8> {
        crate::fs::vfs::with_vfs(|vfs| {
            let handle = vfs.lookup_follow(FLOOR_PATH).ok()?;
            let mut buf = alloc::vec![0u8; FLOOR_RECORD_LEN];
            let read = vfs.read(handle, &mut buf, 0).ok()?;
            buf.truncate(read);
            Some(buf)
        })
        .unwrap_or_default()
    }

    /// Put attacker-chosen bytes where the floor record lives.
    fn write_floor_record(bytes: &[u8]) {
        crate::fs::vfs::with_vfs(|vfs| {
            if let Ok(handle) = vfs.create(FLOOR_PATH).or_else(|_| vfs.lookup_follow(FLOOR_PATH)) {
                let _ = vfs.write(handle, bytes, 0);
            }
        });
    }

    /// The floor exists to stop an old, validly signed release from being
    /// served again. Keeping it in memory means a reboot re-opens the window,
    /// so the attack is: accept v9, reboot, offer v8.
    fn test_floor_survives_reboot() -> TestResult {
        clear_floor();
        let package = eph();
        let mut before_reboot = persistent_channel();
        let newer = fixture_transport(&before_reboot, 9, NAME, VERSION, &package);
        test_assert!(
            before_reboot.fetch_index(&newer).is_ok(),
            "v9 index must accept on a clean floor"
        );
        drop(before_reboot);

        // Reboot: every field of the channel is rebuilt from nothing.
        let mut after_reboot = persistent_channel();
        let mut pkg = ManifoldPkg::new();
        let before = pkg.package_count();
        let older = fixture_transport(&after_reboot, 8, NAME, VERSION, &package);
        match after_reboot.install_into(&mut pkg, &older, NAME, VERSION) {
            Err(ChannelError::IndexRollback { accepted, offered }) => {
                test_assert_eq!(accepted, 9);
                test_assert_eq!(offered, 8);
            }
            _ => {
                return TestResult::Fail(
                    "a package below the persisted floor must be refused after a reboot",
                )
            }
        }
        test_assert_eq!(pkg.package_count(), before);
        TestResult::Pass
    }

    /// Pin the boundary in both directions across a reboot: the version the
    /// floor already accepted is refused, the next one up is not.
    fn test_floor_boundary_exact() -> TestResult {
        clear_floor();
        let package = eph();
        let mut first = persistent_channel();
        let at = fixture_transport(&first, 20, NAME, VERSION, &package);
        test_assert!(
            first.fetch_index(&at).is_ok(),
            "v20 must accept on a clean floor"
        );
        drop(first);

        let mut equal = persistent_channel();
        let same = fixture_transport(&equal, 20, NAME, VERSION, &package);
        match equal.fetch_index(&same) {
            Err(ChannelError::IndexRollback { accepted, offered }) => {
                test_assert_eq!(accepted, 20);
                test_assert_eq!(offered, 20);
            }
            _ => {
                return TestResult::Fail("an index exactly at the persisted floor must be refused")
            }
        }
        drop(equal);

        let mut above = persistent_channel();
        let next = fixture_transport(&above, 21, NAME, VERSION, &package);
        test_assert!(
            above.fetch_index(&next).is_ok(),
            "the first version above the persisted floor must still accept"
        );
        TestResult::Pass
    }

    /// A floor that was never established must not brick the channel: a fresh
    /// system accepts its first index, and that acceptance is what closes the
    /// window from the next boot on.
    fn test_absent_floor_bootstraps_then_holds() -> TestResult {
        clear_floor();
        let package = eph();
        let mut fresh = persistent_channel();
        let first = fixture_transport(&fresh, 1, NAME, VERSION, &package);
        test_assert!(
            fresh.fetch_index(&first).is_ok(),
            "a system with no floor on disk must accept its first index"
        );
        drop(fresh);

        let mut rebooted = persistent_channel();
        let replay = fixture_transport(&rebooted, 1, NAME, VERSION, &package);
        test_assert!(
            matches!(
                rebooted.fetch_index(&replay),
                Err(ChannelError::IndexRollback { .. })
            ),
            "the bootstrapped floor must hold across the reboot"
        );
        TestResult::Pass
    }

    /// The floor lands on a filesystem an attacker may own. A record that does
    /// not verify must refuse installs rather than reset the floor to zero —
    /// the second is worth one byte flip to anyone holding an old release.
    fn test_forged_floor_refused() -> TestResult {
        clear_floor();
        let package = eph();
        let mut established = persistent_channel();
        let good = fixture_transport(&established, 30, NAME, VERSION, &package);
        test_assert!(
            established.fetch_index(&good).is_ok(),
            "v30 must accept on a clean floor"
        );
        drop(established);

        // Lower the floor in place, keeping the signature the kernel wrote.
        let mut record = read_floor_record();
        test_assert_eq!(record.len(), FLOOR_RECORD_LEN);
        record[8..16].copy_from_slice(&5u64.to_be_bytes());
        write_floor_record(&record);
        let mut attacked = persistent_channel();
        let mut pkg = ManifoldPkg::new();
        let before = pkg.package_count();
        let stale = fixture_transport(&attacked, 6, NAME, VERSION, &package);
        test_assert_eq!(
            attacked.install_into(&mut pkg, &stale, NAME, VERSION).err(),
            Some(ChannelError::FloorUnavailable)
        );
        test_assert_eq!(pkg.package_count(), before);

        // A record that is merely damaged fails the same way — and refuses a
        // *newer* index too, which is what "fail closed" means here.
        clear_floor();
        write_floor_record(&[0xa5; FLOOR_RECORD_LEN]);
        let mut damaged = persistent_channel();
        let newer = fixture_transport(&damaged, 31, NAME, VERSION, &package);
        test_assert_eq!(
            damaged.fetch_index(&newer).err(),
            Some(ChannelError::FloorUnavailable)
        );

        // So does one too short to be a record at all.
        clear_floor();
        write_floor_record(&record[..FLOOR_RECORD_LEN - 1]);
        let mut truncated = persistent_channel();
        let newer = fixture_transport(&truncated, 32, NAME, VERSION, &package);
        test_assert_eq!(
            truncated.fetch_index(&newer).err(),
            Some(ChannelError::FloorUnavailable)
        );
        clear_floor();
        TestResult::Pass
    }

    /// Nothing may lower the floor. The store enforces that itself, so a
    /// caller asking to go backward is a no-op and not a downgrade.
    fn test_floor_never_moves_backward() -> TestResult {
        clear_floor();
        let package = eph();
        let mut channel = persistent_channel();
        let transport = fixture_transport(&channel, 40, NAME, VERSION, &package);
        test_assert!(
            channel.fetch_index(&transport).is_ok(),
            "v40 must accept on a clean floor"
        );
        test_assert_eq!(persist_floor(FLOOR_PATH, 4), Ok(()));
        test_assert_eq!(load_floor(FLOOR_PATH), Ok(40));
        test_assert_eq!(persist_floor(FLOOR_PATH, 40), Ok(()));
        test_assert_eq!(load_floor(FLOOR_PATH), Ok(40));
        TestResult::Pass
    }

    /// The boot proof replays old index versions on purpose. If its fixture
    /// channels shared the persisted floor they would leave it at a fixture's
    /// version and then fail their own positive control on the next boot.
    fn test_fixture_channel_leaves_floor_untouched() -> TestResult {
        clear_floor();
        let package = eph();
        let mut fixture = fixture_channel();
        let transport = fixture_transport(&fixture, 50, NAME, VERSION, &package);
        test_assert!(
            fixture.fetch_index(&transport).is_ok(),
            "the fixture channel must still accept"
        );
        test_assert!(
            read_floor_record().is_empty(),
            "a fixture channel must not write the persisted floor"
        );
        test_assert_eq!(load_floor(FLOOR_PATH), Ok(0));
        TestResult::Pass
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
        crate::testing::register_test("pkg::channel_floor_survives_reboot", test_floor_survives_reboot);
        crate::testing::register_test("pkg::channel_floor_boundary_exact", test_floor_boundary_exact);
        crate::testing::register_test(
            "pkg::channel_absent_floor_bootstraps",
            test_absent_floor_bootstraps_then_holds,
        );
        crate::testing::register_test("pkg::channel_forged_floor_refused", test_forged_floor_refused);
        crate::testing::register_test(
            "pkg::channel_floor_never_moves_backward",
            test_floor_never_moves_backward,
        );
        crate::testing::register_test(
            "pkg::channel_fixture_leaves_floor_untouched",
            test_fixture_channel_leaves_floor_untouched,
        );
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
