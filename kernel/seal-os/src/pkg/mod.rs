// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ManifoldPkg — native package manager.

pub mod carrier;
pub mod channel;
pub mod format;
pub mod manifest;
pub mod registry;
pub mod resolver;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};

use self::format::{parse_eph, EphPackage};
use self::manifest::PackageManifest;
use self::registry::PackageRegistry;
use self::resolver::DependencyResolver;
use spin::Mutex;

pub static GLOBAL_PKG: Mutex<ManifoldPkg> = Mutex::new(ManifoldPkg::new());
const SEAL_PKG_PUBLIC_KEY: [u8; 32] = [
    0x3b, 0x6a, 0x27, 0xbc, 0xce, 0xb6, 0xa4, 0x2d, 0x62, 0xa3, 0xa8, 0xd0, 0x2a, 0x6f, 0x0d, 0x73,
    0x63, 0x2e, 0x3e, 0x77, 0xe3, 0xe9, 0xdf, 0x15, 0xe2, 0xda, 0x4c, 0x64, 0x3a, 0x53, 0x97, 0x43,
];
const PROOF_PKG_NAME: &str = "seal-proof-pkg";
const PROOF_PKG_VERSION: &str = "0.0.1";
const PROOF_FILE_PATH: &str = "/packages/seal-proof.txt";
const PROOF_FILE_BYTES: &[u8] = b"seal package proof\n";
const PROOF_PKG_SIGNING_KEY: [u8; 32] = [
    0x51, 0x9d, 0x7a, 0x12, 0xe3, 0x04, 0x42, 0xb7, 0x28, 0x6f, 0xaa, 0xc1, 0x09, 0x5b, 0x73, 0xd0,
    0x18, 0x8c, 0xf5, 0x36, 0x21, 0xee, 0x90, 0x44, 0x67, 0xa3, 0xd2, 0x0f, 0xb9, 0x5c, 0x61, 0x2a,
];

/// Domain tag opening every package signature preimage. It exists to make the
/// scheme self-identifying and to keep a package signature from ever being
/// valid over some other structure signed with the same key.
const PKG_SIGNATURE_DOMAIN: &[u8; 8] = b"EPHSIG1\0";

/// Directory trees a package may write into. Anything outside them is refused:
/// a package declares absolute paths (`Vfs::canonicalize_path`, fs/vfs.rs:173,
/// refuses every relative one), and without this list an unsigned `.eph` could
/// name `/etc/passwd` and the VFS would happily create it.
///
/// ponytail: a flat two-entry allowlist, because two roots exist. If packages
/// ever need to own more of the tree, this becomes a per-package prefix
/// granted at install time rather than a constant.
const INSTALL_ROOTS: [&str; 2] = ["/packages", crate::bundle::STORE_DIR];

pub struct ManifoldPkg {
    registry: PackageRegistry,
    resolver: DependencyResolver,
    registry_url: String,
}

pub fn emit_boot_proof() {
    let mut pkg = GLOBAL_PKG.lock();
    let _ = pkg.remove(PROOF_PKG_NAME);
    let before = pkg.package_count();
    let eph = build_proof_eph();
    let proof_public_key = proof_pkg_public_key();
    let registry_index = build_proof_registry_index(&eph);
    let registry_index_ok = verify_proof_registry_index(&registry_index, &eph, &proof_public_key);
    let parse_ok = parse_eph(&eph)
        .map(|parsed| {
            parsed.manifest.name == PROOF_PKG_NAME
                && parsed.manifest.version == PROOF_PKG_VERSION
                && parsed.files.len() == 1
                && parsed.files[0].path == PROOF_FILE_PATH
                && parsed.files[0].data == PROOF_FILE_BYTES
        })
        .unwrap_or(false);
    let signature_ok = parse_eph(&eph)
        .map(|parsed| verify_package_signature(&parsed, &proof_public_key))
        .unwrap_or(false);
    let install_ok = pkg.install_bytes(&eph, Some(&proof_public_key)).is_ok();
    let after_install = pkg.package_count();
    let list_ok = pkg
        .list()
        .iter()
        .any(|manifest| manifest.name == PROOF_PKG_NAME && manifest.version == PROOF_PKG_VERSION);
    let extract_ok = proof_file_matches();
    let remove_ok = pkg.remove(PROOF_PKG_NAME).is_ok();
    let after_remove = pkg.package_count();
    let counts_ok = after_install == before + 1 && after_remove == before;
    // Remote release channel: measured after the local counts are captured, so
    // the channel's own install/remove cycle cannot perturb them.
    let ch = self::channel::measure(&mut pkg, &eph);
    let result = if parse_ok
        && registry_index_ok
        && signature_ok
        && install_ok
        && list_ok
        && extract_ok
        && remove_ok
        && counts_ok
        && ch.ok()
    {
        "pass"
    } else {
        "fail"
    };
    crate::serial_println!(
        "[ManifoldPkg] proof version=1 source=embedded_eph parse={} registry_index={} install={} extract={} list={} remove={} files=1 bytes={} package_count_before={} package_count_after_install={} package_count_after_remove={} metadata_only=0 signature={} channel_endpoint={} channel_transport={} channel_index_signature={} channel_index_version={} channel_packages_fetched={} channel_digest_ok={} channel_rollback_refused={} channel_tamper_refused={} channel_digest_mismatch_refused={} channel_package_signature_enforced={} channel_live_probe={} channel_fail_closed={} channel_unverified_fallback=0 result={}",
        if parse_ok { "ok" } else { "fail" },
        if registry_index_ok { "ed25519_fixture" } else { "fail" },
        if install_ok { "ok" } else { "fail" },
        if extract_ok { "ok" } else { "fail" },
        if list_ok { "ok" } else { "fail" },
        if remove_ok { "ok" } else { "fail" },
        PROOF_FILE_BYTES.len(),
        before,
        after_install,
        after_remove,
        if signature_ok { "ed25519_fixture" } else { "fail" },
        self::channel::CHANNEL_ENDPOINT,
        ch.transport,
        if ch.index_signature_ok { "ed25519_fixture" } else { "fail" },
        ch.index_version,
        ch.packages_fetched,
        ch.digest_ok,
        u8::from(ch.rollback_refused),
        u8::from(ch.tamper_refused),
        u8::from(ch.digest_mismatch_refused),
        u8::from(ch.package_signature_enforced),
        ch.live_probe,
        u8::from(ch.fail_closed),
        result
    );
}

fn build_proof_eph() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"EPH\0");
    let manifest = format!(
        "name=\"{}\"\nversion=\"{}\"\ndescription=\"boot proof\"",
        PROOF_PKG_NAME, PROOF_PKG_VERSION
    );
    data.extend_from_slice(&(manifest.len() as u32).to_be_bytes());
    data.extend_from_slice(manifest.as_bytes());
    data.extend_from_slice(&[0u8; 64]);
    data.extend_from_slice(&(PROOF_FILE_PATH.len() as u16).to_be_bytes());
    data.extend_from_slice(PROOF_FILE_PATH.as_bytes());
    data.extend_from_slice(&(PROOF_FILE_BYTES.len() as u32).to_be_bytes());
    data.extend_from_slice(PROOF_FILE_BYTES);
    data.extend_from_slice(b"END\0");
    let signature_offset = 8 + manifest.len();
    if let Ok(parsed) = parse_eph(&data) {
        let sig = sign_proof_package(&parsed);
        data[signature_offset..signature_offset + 64].copy_from_slice(&sig);
    }
    data
}

fn build_proof_registry_index(eph: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(eph);
    let hash: [u8; 32] = hasher.finalize().into();
    let body = format!(
        "name={}\nversion={}\nbytes={}\nsha256={}\n",
        PROOF_PKG_NAME,
        PROOF_PKG_VERSION,
        eph.len(),
        hex32(&hash)
    );
    let signature = SigningKey::from_bytes(&PROOF_PKG_SIGNING_KEY).sign(body.as_bytes());
    let mut index = Vec::new();
    index.extend_from_slice(b"EPHIDX1\n");
    index.extend_from_slice(body.as_bytes());
    index.extend_from_slice(b"signature=");
    index.extend_from_slice(hex64(&signature.to_bytes()).as_bytes());
    index.extend_from_slice(b"\n");
    index
}

fn verify_proof_registry_index(index: &[u8], eph: &[u8], public_key: &[u8; 32]) -> bool {
    let text = match core::str::from_utf8(index) {
        Ok(text) => text,
        Err(_) => return false,
    };
    if !text.starts_with("EPHIDX1\n") {
        return false;
    }
    let Some(sig_pos) = text.find("signature=") else {
        return false;
    };
    let signed = &text["EPHIDX1\n".len()..sig_pos];
    let expected_hash = {
        let mut hasher = Sha256::new();
        hasher.update(eph);
        let hash: [u8; 32] = hasher.finalize().into();
        hex32(&hash)
    };
    let expected_bytes = format!("bytes={}\n", eph.len());
    let expected_hash_line = format!("sha256={}\n", expected_hash);
    if !signed.contains("name=seal-proof-pkg\n")
        || !signed.contains("version=0.0.1\n")
        || !signed.contains(expected_bytes.as_str())
        || !signed.contains(expected_hash_line.as_str())
    {
        return false;
    }
    let sig_hex = text[sig_pos + "signature=".len()..].trim();
    let Some(sig_bytes) = parse_hex64(sig_hex) else {
        return false;
    };
    let Ok(vk) = VerifyingKey::from_bytes(public_key) else {
        return false;
    };
    let sig = Signature::from_bytes(&sig_bytes);
    vk.verify_strict(signed.as_bytes(), &sig).is_ok()
}

fn hex32(bytes: &[u8; 32]) -> String {
    let mut out = String::new();
    for byte in bytes {
        out.push(nibble_hex(byte >> 4));
        out.push(nibble_hex(byte & 0x0f));
    }
    out
}

fn hex64(bytes: &[u8; 64]) -> String {
    let mut out = String::new();
    for byte in bytes {
        out.push(nibble_hex(byte >> 4));
        out.push(nibble_hex(byte & 0x0f));
    }
    out
}

fn parse_hex64(text: &str) -> Option<[u8; 64]> {
    if text.len() != 128 {
        return None;
    }
    let mut out = [0u8; 64];
    let bytes = text.as_bytes();
    for i in 0..64 {
        out[i] = (hex_nibble(bytes[i * 2])? << 4) | hex_nibble(bytes[i * 2 + 1])?;
    }
    Some(out)
}

fn nibble_hex(n: u8) -> char {
    match n & 0x0f {
        0..=9 => (b'0' + (n & 0x0f)) as char,
        v => (b'a' + (v - 10)) as char,
    }
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(10 + b - b'a'),
        b'A'..=b'F' => Some(10 + b - b'A'),
        _ => None,
    }
}

fn proof_pkg_public_key() -> [u8; 32] {
    SigningKey::from_bytes(&PROOF_PKG_SIGNING_KEY)
        .verifying_key()
        .to_bytes()
}

/// The bytes a package signature covers.
///
/// Every field the wire format lets a package vary is inside, and every field
/// carries its own big-endian `u32` length or count, so no two distinct
/// packages share a preimage:
///
/// ```text
/// "EPHSIG1\0"
/// u32(len name)          name
/// u32(len version)       version
/// u32(len description)   description
/// u32(count deps)        then per dependency: u32(len) bytes
/// u32(count files)       then per file:       u32(len path) path, sha256[32]
/// ```
///
/// The length prefixes are the point of the encoding, not decoration. The
/// previous scheme concatenated `name || version || (path || hash)*` raw, so
/// `name="ab" version="c"` and `name="a" version="bc"` hashed to the same
/// bytes and one signature covered both. It also omitted `description` and
/// `dependencies` entirely, which let an attacker rewrite the dependency list
/// of a validly signed package — the list `install_bytes` hands straight to
/// the resolver.
///
/// `carrier` and `voronoi_cell` are deliberately absent: `format::parse_eph`
/// never reads them off the wire (`parse_manifest` hardcodes `Aether` and 0),
/// so they carry no attacker-controlled bits. Add them here the moment the
/// parser starts reading them.
fn signature_preimage(pkg: &EphPackage) -> Vec<u8> {
    fn push(out: &mut Vec<u8>, bytes: &[u8]) {
        out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        out.extend_from_slice(bytes);
    }

    let mut out = Vec::new();
    out.extend_from_slice(PKG_SIGNATURE_DOMAIN);
    push(&mut out, pkg.manifest.name.as_bytes());
    push(&mut out, pkg.manifest.version.as_bytes());
    push(&mut out, pkg.manifest.description.as_bytes());
    out.extend_from_slice(&(pkg.manifest.dependencies.len() as u32).to_be_bytes());
    for dep in &pkg.manifest.dependencies {
        push(&mut out, dep.as_bytes());
    }
    out.extend_from_slice(&(pkg.files.len() as u32).to_be_bytes());
    for f in &pkg.files {
        push(&mut out, f.path.as_bytes());
        out.extend_from_slice(&f.hash);
    }
    out
}

/// Verify a package against `public_key` over [`signature_preimage`].
///
/// This replaces `format::verify_signature`, whose preimage covers neither the
/// description nor the dependency list. Both install paths route through here.
fn verify_package_signature(pkg: &EphPackage, public_key: &[u8; 32]) -> bool {
    let Ok(vk) = VerifyingKey::from_bytes(public_key) else {
        return false;
    };
    vk.verify_strict(
        &signature_preimage(pkg),
        &Signature::from_bytes(&pkg.signature),
    )
    .is_ok()
}

/// Reject any package path that is not a plain, canonical location inside an
/// install root. Rejected, never rewritten: silently normalizing `..` away is
/// how traversal bugs come back.
///
/// A path must be absolute, must contain no empty, `.` or `..` component, and
/// must sit strictly inside one of [`INSTALL_ROOTS`] at a component boundary.
/// Absoluteness and `..` are also handled downstream by
/// `Vfs::canonicalize_path`, but downstream *resolves* them where this
/// refuses: `/packages/../etc/passwd` reaches the VFS as `/etc/passwd` and is
/// created without complaint.
///
/// U+FFFD stands in for "not valid UTF-8": `format::parse_eph` decodes path
/// bytes with `from_utf8_lossy`, so by the time a path arrives here the
/// invalid bytes have already become replacement characters and the original
/// bytes are gone. Refusing the replacement character refuses both cases.
fn install_path_ok(path: &str) -> bool {
    if path.contains('\u{fffd}') {
        return false;
    }
    // Matching an install root at a component boundary is also what makes the
    // path absolute — no relative path can start with one of these roots — so
    // absoluteness is not checked a second time.
    let Some(rest) = INSTALL_ROOTS
        .iter()
        .find_map(|root| path.strip_prefix(*root).filter(|r| r.starts_with('/')))
    else {
        return false;
    };
    // `rest` opens with the separator, so the first split yields "" and the
    // rest are the components below the root. Each must be a plain name.
    rest.split('/')
        .skip(1)
        .all(|component| !component.is_empty() && component != "." && component != "..")
}

fn sign_proof_package(pkg: &EphPackage) -> [u8; 64] {
    SigningKey::from_bytes(&PROOF_PKG_SIGNING_KEY)
        .sign(&signature_preimage(pkg))
        .to_bytes()
}

fn proof_file_matches() -> bool {
    crate::fs::vfs::with_vfs(|vfs| {
        let handle = vfs.lookup_follow(PROOF_FILE_PATH).ok()?;
        let mut buf = alloc::vec![0u8; PROOF_FILE_BYTES.len()];
        let read = vfs.read(handle, &mut buf, 0).ok()?;
        Some(read == PROOF_FILE_BYTES.len() && buf == PROOF_FILE_BYTES)
    })
    .unwrap_or(false)
}

impl ManifoldPkg {
    pub const fn new() -> Self {
        Self {
            registry: PackageRegistry::new(),
            resolver: DependencyResolver::new(),
            registry_url: String::new(),
        }
    }

    pub fn init_defaults(&mut self) {
        self.registry_url = String::from("https://repo.seal-os.local/packages/");
    }

    pub fn set_registry_url(&mut self, url: &str) {
        self.registry_url = String::from(url);
        if !self.registry_url.ends_with('/') {
            self.registry_url.push('/');
        }
    }

    /// Install a package from raw `.eph` bytes.
    pub fn install_bytes(
        &mut self,
        data: &[u8],
        public_key: Option<&[u8; 32]>,
    ) -> Result<String, String> {
        let pkg = parse_eph(data).map_err(|e| format!("parse error: {:?}", e))?;

        if let Some(key) = public_key {
            if !verify_package_signature(&pkg, key) {
                return Err(String::from("signature: rejected"));
            }
        }

        // Every path is checked before any file is written, so a package that
        // names one unsafe path installs none of its files.
        for file in &pkg.files {
            if !install_path_ok(&file.path) {
                return Err(format!("unsafe path '{}'", file.path));
            }
        }

        // Register deps in resolver graph
        self.resolver
            .register(&pkg.manifest.name, &pkg.manifest.dependencies);

        // Resolve and install dependencies first
        let dep_order = self
            .resolver
            .resolve(&pkg.manifest.name)
            .map_err(|e| format!("deps: {}", e))?;
        for dep in dep_order {
            if !self.registry.is_installed(&dep) {
                return Err(format!("missing dependency '{}'", dep));
            }
        }

        // Extract files to ManifoldFS via VFS
        for file in &pkg.files {
            if let Err(e) = self.install_file(&file.path, &file.data) {
                return Err(format!("extract '{}': {:?}", file.path, e));
            }
        }

        self.registry.install(pkg.manifest.clone());
        Ok(format!(
            "Installed '{}' v{} ({} files)",
            pkg.manifest.name,
            pkg.manifest.version,
            pkg.files.len()
        ))
    }

    /// Install by name — downloads .eph from registry and installs.
    pub fn install(&mut self, name: &str) -> Result<String, String> {
        if !crate::net::has_nic() {
            return Err(String::from("no network — cannot download package"));
        }
        let url = alloc::format!("{}{}.eph", self.registry_url, name);
        let client = crate::drivers::net::http::HttpClient::new();
        let response = client
            .get(&url)
            .map_err(|e| alloc::format!("download failed: {}", e))?;
        if response.status != 200 {
            return Err(alloc::format!(
                "package '{}' not found on registry (status {})",
                name,
                response.status
            ));
        }
        self.install_bytes(&response.body, Some(&SEAL_PKG_PUBLIC_KEY))
    }

    pub fn remove(&mut self, name: &str) -> Result<String, String> {
        if !self.registry.is_installed(name) {
            return Err(format!("'{}' is not installed", name));
        }
        self.registry.remove(name);
        Ok(format!("Removed '{}'", name))
    }

    pub fn list(&self) -> Vec<&PackageManifest> {
        self.registry.list()
    }

    pub fn package_count(&self) -> usize {
        self.registry.count()
    }

    fn install_file(&self, path: &str, data: &[u8]) -> Result<(), VfsInstallError> {
        use crate::fs::vfs::{with_vfs, VfsError};
        // Ensure parent directory exists
        if let Some(last_slash) = path.rfind('/') {
            let dir = &path[..last_slash];
            if !dir.is_empty() {
                let _ = with_vfs(|vfs| vfs.mkdir(dir));
            }
        }
        match with_vfs(|vfs| vfs.create(path)) {
            Ok(handle) => {
                with_vfs(|vfs| vfs.write(handle, data, 0)).map_err(VfsInstallError::Vfs)?;
                Ok(())
            }
            Err(VfsError::AlreadyExists) => {
                // Overwrite
                let handle =
                    with_vfs(|vfs| vfs.lookup_follow(path)).map_err(VfsInstallError::Vfs)?;
                with_vfs(|vfs| vfs.write(handle, data, 0)).map_err(VfsInstallError::Vfs)?;
                Ok(())
            }
            Err(e) => Err(VfsInstallError::Vfs(e)),
        }
    }
}

impl Default for ManifoldPkg {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
enum VfsInstallError {
    #[allow(dead_code)] // REASON: VFS error payload preserved for future install error diagnostics
    Vfs(crate::fs::vfs::VfsError),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;

    /// Every unsafe path the installer must refuse, paired with the failure
    /// message naming the property that broke.
    const UNSAFE_PATHS: [(&[u8], &str); 11] = [
        (b"../etc/passwd", "relative traversal must be refused"),
        (b"/etc/passwd", "a path outside every install root must be refused"),
        (b"a/../../b", "relative path with .. must be refused"),
        (b"a//b", "relative path with an empty component must be refused"),
        (
            b"/packages/../etc/passwd",
            "absolute path escaping its root via .. must be refused",
        ),
        (
            b"/packages//x",
            "empty path component must be refused, not collapsed",
        ),
        (
            b"/packages/./x",
            "a `.` component must be refused, not dropped",
        ),
        (b"/packages/", "a trailing slash must be refused"),
        (b"", "an empty path must be refused"),
        (
            b"/packages/\xff\xfe.txt",
            "a path that was not valid UTF-8 must be refused, not replaced",
        ),
        (
            b"/packagesfoo/x",
            "a byte-prefix of an install root is not inside it",
        ),
    ];

    /// Build an `.eph` carrying an all-zero signature. File paths are raw bytes
    /// so a test can hand the parser one that is not valid UTF-8.
    fn build_eph(
        name: &str,
        version: &str,
        description: &str,
        deps: &[&str],
        files: &[(&[u8], &[u8])],
    ) -> Vec<u8> {
        let mut manifest = format!(
            "name=\"{}\"\nversion=\"{}\"\ndescription=\"{}\"",
            name, version, description
        );
        for dep in deps {
            manifest.push_str(&format!("\ndep=\"{}\"", dep));
        }
        let mut data = Vec::new();
        data.extend_from_slice(b"EPH\0");
        data.extend_from_slice(&(manifest.len() as u32).to_be_bytes());
        data.extend_from_slice(manifest.as_bytes());
        data.extend_from_slice(&[0u8; 64]);
        for (path, bytes) in files {
            data.extend_from_slice(&(path.len() as u16).to_be_bytes());
            data.extend_from_slice(path);
            data.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
            data.extend_from_slice(bytes);
        }
        data.extend_from_slice(b"END\0");
        data
    }

    fn signature_offset(eph: &[u8]) -> usize {
        8 + u32::from_be_bytes([eph[4], eph[5], eph[6], eph[7]]) as usize
    }

    /// Sign `eph` in place with the proof key — the only package signing key
    /// in the tree.
    fn sign(eph: &mut [u8]) {
        if let Ok(parsed) = parse_eph(eph) {
            let at = signature_offset(eph);
            eph[at..at + 64].copy_from_slice(&sign_proof_package(&parsed));
        }
    }

    /// Move `from`'s signature onto `to` without re-signing: the forgery every
    /// signature test below performs.
    fn copy_signature(from: &[u8], to: &mut [u8]) {
        let src = signature_offset(from);
        let dst = signature_offset(to);
        let mut sig = [0u8; 64];
        sig.copy_from_slice(&from[src..src + 64]);
        to[dst..dst + 64].copy_from_slice(&sig);
    }

    /// True when the installer refused `eph` *for its signature*. Every other
    /// refusal — a missing dependency, a bad path, a parse error — is not a
    /// pass: a test that accepted them would go green with the signature check
    /// deleted.
    fn signature_refused(eph: &[u8]) -> bool {
        match ManifoldPkg::new().install_bytes(eph, Some(&proof_pkg_public_key())) {
            Err(e) => e.starts_with("signature:"),
            Ok(_) => false,
        }
    }

    fn signature_accepted(eph: &[u8]) -> bool {
        ManifoldPkg::new()
            .install_bytes(eph, Some(&proof_pkg_public_key()))
            .is_ok()
    }

    fn file_exists(path: &str) -> bool {
        crate::fs::vfs::with_vfs(|vfs| vfs.lookup_follow(path)).is_ok()
    }

    /// The dependency list drives the resolver (`install_bytes`), so a
    /// signature that does not cover it lets an attacker rewrite what a
    /// validly signed package pulls in. Rewriting one dependency must break
    /// the signature, and must break it *as a signature failure* — the
    /// resolver refusing an uninstalled dependency is not the check under
    /// test.
    fn test_signature_covers_dependencies() -> TestResult {
        let path = b"/packages/pkg-mod-deps.txt".as_slice();
        let mut good = build_eph("pkg-mod-deps", "1.0.0", "", &[], &[(path, b"x")]);
        sign(&mut good);
        test_assert!(
            signature_accepted(&good),
            "control: the signed package must verify"
        );

        let mut tampered = build_eph("pkg-mod-deps", "1.0.0", "", &["evil"], &[(path, b"x")]);
        copy_signature(&good, &mut tampered);
        let Ok(parsed) = parse_eph(&tampered) else {
            return TestResult::Fail("tampered fixture must still parse");
        };
        test_assert!(
            parsed.manifest.dependencies.len() == 1,
            "the tamper must actually land in the parsed manifest"
        );
        test_assert!(
            signature_refused(&tampered),
            "a rewritten dependency list must break the signature"
        );
        TestResult::Pass
    }

    /// Same hole, second field: the description rides in the manifest and is
    /// kept in the installed registry, so it must be signed too.
    fn test_signature_covers_description() -> TestResult {
        let path = b"/packages/pkg-mod-desc.txt".as_slice();
        let mut good = build_eph("pkg-mod-desc", "1.0.0", "honest", &[], &[(path, b"x")]);
        sign(&mut good);
        test_assert!(
            signature_accepted(&good),
            "control: the signed package must verify"
        );

        let mut tampered = build_eph("pkg-mod-desc", "1.0.0", "forged", &[], &[(path, b"x")]);
        copy_signature(&good, &mut tampered);
        test_assert!(
            signature_refused(&tampered),
            "a rewritten description must break the signature"
        );
        TestResult::Pass
    }

    /// Fields concatenated without length separators are interchangeable:
    /// `name="ab" version="c"` and `name="a" version="bc"` produce the same
    /// bytes, so one signature covers both packages. Same for a dependency
    /// list that concatenates identically.
    fn test_signature_field_split_unambiguous() -> TestResult {
        let mut ab_c = build_eph("ab", "c", "", &[], &[]);
        sign(&mut ab_c);
        test_assert!(
            signature_accepted(&ab_c),
            "control: the signed package must verify"
        );
        let mut a_bc = build_eph("a", "bc", "", &[], &[]);
        copy_signature(&ab_c, &mut a_bc);
        test_assert!(
            signature_refused(&a_bc),
            "a different name/version split must not share a signature"
        );

        // Same dependency count, same concatenation, different split — so this
        // stays red if the per-entry length prefix is dropped and only the
        // count survives.
        let mut ab_c_deps = build_eph("pkg-mod-split", "1.0.0", "", &["ab", "c"], &[]);
        sign(&mut ab_c_deps);
        let mut a_bc_deps = build_eph("pkg-mod-split", "1.0.0", "", &["a", "bc"], &[]);
        copy_signature(&ab_c_deps, &mut a_bc_deps);
        test_assert!(
            signature_refused(&a_bc_deps),
            "a different dependency split must not share a signature"
        );
        TestResult::Pass
    }

    /// One signature must not carry over to any package that differs in any
    /// signed field. Each mutant below changes exactly one thing and keeps the
    /// original signature; the field lengths are held equal where possible so
    /// a mutant cannot be caught by an incidental length change.
    fn test_signature_covers_every_field() -> TestResult {
        const PATH: &[u8] = b"/packages/sig-base.txt";
        let mut base = build_eph("sig-base", "1.0.0", "d", &[], &[(PATH, b"a")]);
        sign(&mut base);
        test_assert!(
            signature_accepted(&base),
            "control: the signed base package must verify"
        );

        let mutants: [(Vec<u8>, &str); 7] = [
            (
                build_eph("sig-bass", "1.0.0", "d", &[], &[(PATH, b"a")]),
                "a renamed package must not keep its signature",
            ),
            (
                build_eph("sig-base", "1.0.1", "d", &[], &[(PATH, b"a")]),
                "a re-versioned package must not keep its signature",
            ),
            (
                build_eph("sig-base", "1.0.0", "e", &[], &[(PATH, b"a")]),
                "a re-described package must not keep its signature",
            ),
            (
                build_eph("sig-base", "1.0.0", "d", &["x"], &[(PATH, b"a")]),
                "an added dependency must not keep its signature",
            ),
            (
                build_eph(
                    "sig-base",
                    "1.0.0",
                    "d",
                    &[],
                    &[(b"/packages/sig-bass.txt", b"a")],
                ),
                "a moved file must not keep its signature",
            ),
            (
                build_eph("sig-base", "1.0.0", "d", &[], &[(PATH, b"b")]),
                "rewritten file content must not keep its signature",
            ),
            (
                build_eph(
                    "sig-base",
                    "1.0.0",
                    "d",
                    &[],
                    &[(PATH, b"a"), (b"/packages/sig-extra.txt", b"a")],
                ),
                "an added file must not keep its signature",
            ),
        ];

        for (mut mutant, what) in mutants {
            copy_signature(&base, &mut mutant);
            if !signature_refused(&mutant) {
                return TestResult::Fail(what);
            }
        }
        TestResult::Pass
    }

    /// Package file paths are attacker-controlled. Every one of these must be
    /// refused by the installer itself — not normalized, and not left to
    /// whatever the filesystem underneath happens to reject.
    fn test_unsafe_paths_refused() -> TestResult {
        for (path, what) in UNSAFE_PATHS {
            let eph = build_eph("pkg-mod-unsafe", "1.0.0", "", &[], &[(path, b"x")]);
            // Either layer may refuse: `parse_eph` rejects a path that is not
            // valid UTF-8 before the installer ever sees it, and the installer
            // refuses the rest. What matters is that nothing is written, not
            // which layer says so.
            match ManifoldPkg::new().install_bytes(&eph, None) {
                Err(e)
                    if e.starts_with("unsafe path") || e == "parse error: BadPath" => {}
                _ => return TestResult::Fail(what),
            }
        }
        TestResult::Pass
    }

    /// One bad path poisons the whole package: nothing is written, not even
    /// the files listed before it.
    fn test_unsafe_path_installs_nothing() -> TestResult {
        const SAFE: &str = "/packages/pkg-mod-atomic.txt";
        const ESCAPE: &[u8] = b"/etc/pkg-mod-escape.txt";
        let eph = build_eph(
            "pkg-mod-atomic",
            "1.0.0",
            "",
            &[],
            &[(SAFE.as_bytes(), b"first"), (ESCAPE, b"owned")],
        );
        test_assert!(
            ManifoldPkg::new().install_bytes(&eph, None).is_err(),
            "a package naming an unsafe path must be refused"
        );
        test_assert!(
            !file_exists("/etc/pkg-mod-escape.txt"),
            "the escaping file must not be written"
        );
        test_assert!(
            !file_exists(SAFE),
            "a refused package must write none of its files"
        );
        TestResult::Pass
    }

    /// Positive control for the path check. Note the ordinary case here is an
    /// *absolute* path under an install root: `Vfs::canonicalize_path`
    /// (fs/vfs.rs:173) refuses every relative path, so a relative package path
    /// could never install in this kernel.
    fn test_ordinary_path_installs() -> TestResult {
        const PATH: &str = "/packages/pkg-mod-ordinary.txt";
        let eph = build_eph(
            "pkg-mod-ordinary",
            "1.0.0",
            "",
            &[],
            &[(PATH.as_bytes(), b"ordinary")],
        );
        test_assert!(
            ManifoldPkg::new().install_bytes(&eph, None).is_ok(),
            "an ordinary path under an install root must install"
        );
        test_assert!(file_exists(PATH), "the installed file must reach the disk");
        TestResult::Pass
    }

    fn test_install_bytes_ok() -> TestResult {
        let eph = build_eph("foo", "1.0.0", "", &[], &[]);
        test_assert!(
            ManifoldPkg::new().install_bytes(&eph, None).is_ok(),
            "a metadata-only package must install"
        );
        TestResult::Pass
    }

    fn test_remove_existing() -> TestResult {
        let mut pkg = ManifoldPkg::new();
        let eph = build_eph("bar", "2.0.0", "", &[], &[]);
        test_assert!(
            pkg.install_bytes(&eph, None).is_ok(),
            "a metadata-only package must install"
        );
        test_assert!(pkg.remove("bar").is_ok(), "installed package must remove");
        test_assert!(
            !pkg.registry.is_installed("bar"),
            "removed package must leave the registry"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "pkg::signature_covers_dependencies",
            test_signature_covers_dependencies,
        );
        crate::testing::register_test(
            "pkg::signature_covers_description",
            test_signature_covers_description,
        );
        crate::testing::register_test(
            "pkg::signature_field_split_unambiguous",
            test_signature_field_split_unambiguous,
        );
        crate::testing::register_test(
            "pkg::signature_covers_every_field",
            test_signature_covers_every_field,
        );
        crate::testing::register_test("pkg::unsafe_paths_refused", test_unsafe_paths_refused);
        crate::testing::register_test(
            "pkg::unsafe_path_installs_nothing",
            test_unsafe_path_installs_nothing,
        );
        crate::testing::register_test("pkg::ordinary_path_installs", test_ordinary_path_installs);
        crate::testing::register_test("pkg::install_bytes_ok", test_install_bytes_ok);
        crate::testing::register_test("pkg::remove_existing", test_remove_existing);
    }
}
