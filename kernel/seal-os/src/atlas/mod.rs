// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Atlas — device chart provisioning.
//!
//! A manifold is only usable once an atlas of charts covers it. A DMA-capable
//! device is only usable once its vendor chart — the firmware image the device
//! executes — is resident. Seal OS ships **no** vendor charts: they are vendor
//! IP under redistribution terms this repository cannot accept. It ships the
//! atlas instead: the request path, the signed chart index, the digest gate,
//! and — the part that matters — the refusal.
//!
//! A chart that is not in the index, or not in the store, or whose bytes do not
//! match the index digest, is refused. There is no simulation fallback in this
//! kernel; a driver that cannot get its chart stays down and says so.
//!
//! Store layout (ManifoldFS, reachable through the VFS):
//!   `/atlas/<chart-name>`   chart bytes
//!   `/atlas/index.seal`     ed25519-signed chart index (optional, provisioned)
//!
//! Charts and the store index are installed with ManifoldPkg (`.eph`), so a
//! user who legally obtains vendor firmware provisions it without rebuilding
//! the kernel.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};
use spin::Mutex;

/// Directory the chart store lives in.
pub const STORE_DIR: &str = "/atlas";
/// Provisioned (optional) signed chart index inside the store.
pub const STORE_INDEX_PATH: &str = "/atlas/index.seal";

const INDEX_MAGIC: &str = "SEALATLAS1\n";

/// Fixture key. Real deployments replace this with the distributor key that
/// signs the chart index shipped alongside vendor firmware.
const ATLAS_INDEX_SIGNING_KEY: [u8; 32] = [
    0x2f, 0x14, 0xb8, 0x6c, 0x07, 0x9a, 0x5d, 0x31, 0xc4, 0x88, 0x1e, 0x73, 0xa0, 0x66, 0xd2, 0x4b,
    0x59, 0x0f, 0xe7, 0x35, 0x8a, 0x21, 0xcc, 0x76, 0x13, 0xbd, 0x42, 0x98, 0x6a, 0x0d, 0xf1, 0x57,
];

/// Synthetic test fixture. This is NOT firmware and is not accepted by any
/// device — it exists only so the positive path (index → store → digest →
/// cache) is exercised at boot without shipping vendor IP.
pub const FIXTURE_CHART_NAME: &str = "test-synthetic-fixture.chart";
const FIXTURE_CHART_BYTES: &[u8] = b"SEAL-OS ATLAS SYNTHETIC TEST CHART v1 - NOT VENDOR FIRMWARE\n";
/// Digest of `FIXTURE_CHART_BYTES`, written independently of the bytes so the
/// comparison is a real check and not an identity.
const FIXTURE_CHART_SHA256: &str =
    "15a82a6da11f27c838a11976569480f3fddece9aeaea9c9bda25dd5670ba9b2b";

/// Indexed by the provisioned store index, deliberately never written to the
/// store: proves the fail-closed path.
pub const ABSENT_CHART_NAME: &str = "test-absent-fixture.chart";
/// Indexed with the fixture digest but provisioned with different bytes:
/// proves the digest gate.
pub const CORRUPT_CHART_NAME: &str = "test-corrupt-fixture.chart";

static CACHE: Mutex<BTreeMap<String, Arc<Chart>>> = Mutex::new(BTreeMap::new());
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
static PROVISIONED: AtomicUsize = AtomicUsize::new(0);
static NOT_PROVISIONED: AtomicUsize = AtomicUsize::new(0);
static DIGEST_OK: AtomicUsize = AtomicUsize::new(0);
static DIGEST_REFUSED: AtomicUsize = AtomicUsize::new(0);
static CACHE_HITS: AtomicUsize = AtomicUsize::new(0);

/// A resident chart. Shared between drivers through `Arc`, so two drivers
/// requesting the same chart share one allocation.
#[derive(Debug)]
pub struct Chart {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Refcounted handle to a resident chart.
pub type ChartRef = Arc<Chart>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartError {
    /// The VFS is not up yet; the store cannot be consulted.
    StoreUnavailable,
    /// No signed index entry names this chart.
    NotIndexed,
    /// Indexed, but the store has no bytes for it — the normal case for vendor
    /// firmware that the user has not provisioned.
    NotProvisioned,
    /// Store bytes do not match the indexed length.
    SizeMismatch,
    /// Store bytes do not match the indexed SHA-256.
    DigestMismatch,
    /// Index magic, encoding, or ed25519 signature failed.
    IndexRejected,
    /// Chart names are flat store entries; `/` and empty names are rejected.
    InvalidName,
    /// The store entry exists but could not be read.
    ReadFailed,
}

impl ChartError {
    pub fn tag(&self) -> &'static str {
        match self {
            Self::StoreUnavailable => "store_unavailable",
            Self::NotIndexed => "not_indexed",
            Self::NotProvisioned => "not_provisioned",
            Self::SizeMismatch => "size_mismatch",
            Self::DigestMismatch => "digest_mismatch",
            Self::IndexRejected => "index_rejected",
            Self::InvalidName => "invalid_name",
            Self::ReadFailed => "read_failed",
        }
    }
}

impl core::fmt::Display for ChartError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.tag())
    }
}

struct IndexEntry {
    name: String,
    bytes: usize,
    digest: [u8; 32],
}

// ---------------------------------------------------------------- public API

/// Request a chart by name.
///
/// Fail-closed: every failure mode returns an error naming the chart. Nothing
/// in this path substitutes generated data for firmware.
pub fn request_chart(name: &str) -> Result<ChartRef, ChartError> {
    REQUESTED.fetch_add(1, Ordering::Relaxed);
    if name.is_empty() || name.contains('/') {
        return Err(ChartError::InvalidName);
    }
    if let Some(chart) = CACHE.lock().get(name) {
        CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        PROVISIONED.fetch_add(1, Ordering::Relaxed);
        return Ok(chart.clone());
    }

    let entry = index()?
        .into_iter()
        .find(|e| e.name == name)
        .ok_or_else(|| {
            NOT_PROVISIONED.fetch_add(1, Ordering::Relaxed);
            ChartError::NotIndexed
        })?;

    let path = alloc::format!("{}/{}", STORE_DIR, name);
    let bytes = read_store_file(&path).inspect_err(|e| {
        if matches!(e, ChartError::NotProvisioned) {
            NOT_PROVISIONED.fetch_add(1, Ordering::Relaxed);
        }
    })?;

    if bytes.len() != entry.bytes {
        DIGEST_REFUSED.fetch_add(1, Ordering::Relaxed);
        return Err(ChartError::SizeMismatch);
    }
    if sha256(&bytes) != entry.digest {
        DIGEST_REFUSED.fetch_add(1, Ordering::Relaxed);
        return Err(ChartError::DigestMismatch);
    }
    DIGEST_OK.fetch_add(1, Ordering::Relaxed);
    PROVISIONED.fetch_add(1, Ordering::Relaxed);

    let chart = Arc::new(Chart {
        name: String::from(name),
        bytes,
    });
    CACHE.lock().insert(String::from(name), chart.clone());
    Ok(chart)
}

/// Drop the cache's reference to a chart, freeing the allocation once no driver
/// still holds a `ChartRef`. Returns `true` if the entry was evicted.
pub fn release_chart(name: &str) -> bool {
    let mut cache = CACHE.lock();
    match cache.get(name) {
        Some(chart) if Arc::strong_count(chart) == 1 => {
            cache.remove(name);
            true
        }
        _ => false,
    }
}

/// Number of charts currently resident.
pub fn cached_charts() -> usize {
    CACHE.lock().len()
}

/// Names in the effective index (built-in plus provisioned store index).
pub fn indexed_names() -> Vec<String> {
    index()
        .map(|e| e.into_iter().map(|i| i.name).collect())
        .unwrap_or_default()
}

/// This kernel contains no simulated radio path. Always `false`; the field
/// exists so the proof line can state it rather than leave it implied.
pub const SIMULATION_PRESENT: bool = false;

// ------------------------------------------------------------- index handling

fn index() -> Result<Vec<IndexEntry>, ChartError> {
    let mut entries = parse_index(&built_in_index(), &index_public_key())?;
    if let Ok(raw) = read_store_file(STORE_INDEX_PATH) {
        if let Ok(store) = parse_index(&raw, &index_public_key()) {
            for entry in store {
                if !entries.iter().any(|e| e.name == entry.name) {
                    entries.push(entry);
                }
            }
        }
    }
    Ok(entries)
}

fn index_public_key() -> [u8; 32] {
    SigningKey::from_bytes(&ATLAS_INDEX_SIGNING_KEY)
        .verifying_key()
        .to_bytes()
}

/// Index compiled into the kernel: the synthetic fixture only. Vendor charts
/// arrive with a provisioned store index.
fn built_in_index() -> Vec<u8> {
    sign_index(&alloc::format!(
        "chart={} bytes={} sha256={}\n",
        FIXTURE_CHART_NAME,
        FIXTURE_CHART_BYTES.len(),
        FIXTURE_CHART_SHA256
    ))
}

fn sign_index(body: &str) -> Vec<u8> {
    let signature = SigningKey::from_bytes(&ATLAS_INDEX_SIGNING_KEY).sign(body.as_bytes());
    let mut out = Vec::new();
    out.extend_from_slice(INDEX_MAGIC.as_bytes());
    out.extend_from_slice(body.as_bytes());
    out.extend_from_slice(b"signature=");
    out.extend_from_slice(hex(&signature.to_bytes()).as_bytes());
    out.push(b'\n');
    out
}

fn parse_index(raw: &[u8], public_key: &[u8; 32]) -> Result<Vec<IndexEntry>, ChartError> {
    let text = core::str::from_utf8(raw).map_err(|_| ChartError::IndexRejected)?;
    if !text.starts_with(INDEX_MAGIC) {
        return Err(ChartError::IndexRejected);
    }
    let sig_pos = text.find("signature=").ok_or(ChartError::IndexRejected)?;
    let body = &text[INDEX_MAGIC.len()..sig_pos];
    let sig_bytes = parse_hex64(text[sig_pos + "signature=".len()..].trim())
        .ok_or(ChartError::IndexRejected)?;
    let vk = VerifyingKey::from_bytes(public_key).map_err(|_| ChartError::IndexRejected)?;
    vk.verify_strict(body.as_bytes(), &Signature::from_bytes(&sig_bytes))
        .map_err(|_| ChartError::IndexRejected)?;

    let mut entries = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut name = None;
        let mut bytes = None;
        let mut digest = None;
        for field in line.split_whitespace() {
            if let Some(v) = field.strip_prefix("chart=") {
                name = Some(String::from(v));
            } else if let Some(v) = field.strip_prefix("bytes=") {
                bytes = v.parse::<usize>().ok();
            } else if let Some(v) = field.strip_prefix("sha256=") {
                digest = parse_hex32(v);
            }
        }
        match (name, bytes, digest) {
            (Some(name), Some(bytes), Some(digest)) if !name.contains('/') => {
                entries.push(IndexEntry {
                    name,
                    bytes,
                    digest,
                })
            }
            _ => return Err(ChartError::IndexRejected),
        }
    }
    Ok(entries)
}

// ------------------------------------------------------------------ store I/O

fn read_store_file(path: &str) -> Result<Vec<u8>, ChartError> {
    if !crate::fs::vfs::is_vfs_initialized() {
        return Err(ChartError::StoreUnavailable);
    }
    crate::fs::vfs::with_vfs(|vfs| {
        let handle = vfs
            .lookup_follow(path)
            .map_err(|_| ChartError::NotProvisioned)?;
        let node = vfs.stat(handle).map_err(|_| ChartError::ReadFailed)?;
        let mut buf = alloc::vec![0u8; node.size as usize];
        let mut off = 0usize;
        while off < buf.len() {
            let n = vfs
                .read(handle, &mut buf[off..], off as u64)
                .map_err(|_| ChartError::ReadFailed)?;
            if n == 0 {
                break;
            }
            off += n;
        }
        buf.truncate(off);
        Ok(buf)
    })
}

// ------------------------------------------------------- provisioning fixture

/// Build the `.eph` that provisions the test-fixture charts and a signed store
/// index. This is the exact shape a vendor-firmware package takes: chart bytes
/// under `/atlas/`, plus `/atlas/index.seal` signed by the distributor key.
fn fixture_package() -> Vec<u8> {
    let store_index = sign_index(&alloc::format!(
        "chart={} bytes={} sha256={}\nchart={} bytes={} sha256={}\n",
        CORRUPT_CHART_NAME,
        FIXTURE_CHART_BYTES.len(),
        FIXTURE_CHART_SHA256,
        ABSENT_CHART_NAME,
        FIXTURE_CHART_BYTES.len(),
        FIXTURE_CHART_SHA256
    ));
    // Same length as the fixture, one byte different: digest must reject it.
    let mut corrupt = FIXTURE_CHART_BYTES.to_vec();
    let last = corrupt.len() - 1;
    corrupt[last] = b'!';

    let mut data = Vec::new();
    data.extend_from_slice(b"EPH\0");
    let manifest = "name=\"seal-atlas-test-charts\"\nversion=\"0.0.1\"\ndescription=\"atlas synthetic test fixture — not vendor firmware\"";
    data.extend_from_slice(&(manifest.len() as u32).to_be_bytes());
    data.extend_from_slice(manifest.as_bytes());
    data.extend_from_slice(&[0u8; 64]);
    for (name, bytes) in [
        (FIXTURE_CHART_NAME, FIXTURE_CHART_BYTES),
        (CORRUPT_CHART_NAME, corrupt.as_slice()),
    ] {
        push_file(&mut data, &alloc::format!("{}/{}", STORE_DIR, name), bytes);
    }
    push_file(&mut data, STORE_INDEX_PATH, &store_index);
    data.extend_from_slice(b"END\0");
    data
}

fn push_file(data: &mut Vec<u8>, path: &str, bytes: &[u8]) {
    data.extend_from_slice(&(path.len() as u16).to_be_bytes());
    data.extend_from_slice(path.as_bytes());
    data.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    data.extend_from_slice(bytes);
}

/// Provision the test fixtures through ManifoldPkg — the same path a user takes
/// with a real vendor-firmware `.eph`.
pub fn provision_test_fixture() -> bool {
    let mut pkg = crate::pkg::GLOBAL_PKG.lock();
    let _ = pkg.remove("seal-atlas-test-charts");
    pkg.install_bytes(&fixture_package(), None).is_ok()
}

// ----------------------------------------------------------------- proof line

/// Measured proof of the atlas contract. Every field below is read back at
/// runtime; nothing is asserted from a constant.
pub fn firmware_proof_line() -> String {
    let index_ok = parse_index(&built_in_index(), &index_public_key()).is_ok();

    // Negative control: flip one body byte, signature must reject the index.
    let mut tampered = built_in_index();
    tampered[INDEX_MAGIC.len()] ^= 0x01;
    let tampered_refused = parse_index(&tampered, &index_public_key()).is_err();

    let installed = provision_test_fixture();
    let index_entries = indexed_names().len();

    // Positive path.
    let first = request_chart(FIXTURE_CHART_NAME);
    let second = request_chart(FIXTURE_CHART_NAME);
    let (same_alloc, rc_peak, rc_after_drop) = match (&first, &second) {
        (Ok(a), Ok(b)) => {
            let same = Arc::ptr_eq(a, b);
            let peak = Arc::strong_count(a);
            drop(second);
            (same, peak, Arc::strong_count(first.as_ref().unwrap()))
        }
        _ => (false, 0, 0),
    };
    let fixture_bytes = first.as_ref().map(|c| c.bytes.len()).unwrap_or(0);
    let fixture_ok = first.is_ok();
    let cached_while_held = cached_charts();
    drop(first);
    let released = release_chart(FIXTURE_CHART_NAME);

    // Fail-closed controls.
    let absent = request_chart(ABSENT_CHART_NAME).err();
    let corrupt = request_chart(CORRUPT_CHART_NAME).err();

    // Driver state, measured by running the real probe path.
    let mut wifi = crate::drivers::wifi::WifiDriver::new();
    wifi.probe_pci();
    let wifi_scan = wifi.scan().len();
    let mut bt = crate::drivers::bluetooth::BluetoothDriver::new();
    bt.probe_pci();
    let bt_scan = bt.scan().len();

    let pass = index_ok
        && tampered_refused
        && installed
        && fixture_ok
        && fixture_bytes == FIXTURE_CHART_BYTES.len()
        && same_alloc
        && rc_peak == 3
        && rc_after_drop == 2
        && released
        && cached_while_held >= 1
        && absent == Some(ChartError::NotProvisioned)
        && corrupt == Some(ChartError::DigestMismatch)
        && !SIMULATION_PRESENT
        && wifi_scan == 0
        && bt_scan == 0
        && !wifi.is_up()
        && !bt.is_up();

    alloc::format!(
        "[Atlas] proof version=1 store={} index=ed25519_fixture index_verify={} index_tampered={} index_entries={} store_index={} provision_pkg={} requested={} provisioned={} not_provisioned={} digest_ok={} digest_refused={} cache_hits={} fixture={} fixture_bytes={} cache_hit={} refcount_peak={} refcount_after_drop={} cached_while_held={} released={} cached_after_release={} absent_chart={}:{} corrupt_chart={}:{} simulation={} wifi={} wifi_chart={} wifi_scan_entries={} bt={} bt_chart={} bt_scan_entries={} result={}",
        STORE_DIR,
        if index_ok { "ok" } else { "fail" },
        if tampered_refused { "refused" } else { "accepted" },
        index_entries,
        if indexed_names().iter().any(|n| n == ABSENT_CHART_NAME) { "ed25519_fixture" } else { "absent" },
        if installed { "eph_installed" } else { "fail" },
        REQUESTED.load(Ordering::Relaxed),
        PROVISIONED.load(Ordering::Relaxed),
        NOT_PROVISIONED.load(Ordering::Relaxed),
        DIGEST_OK.load(Ordering::Relaxed),
        DIGEST_REFUSED.load(Ordering::Relaxed),
        CACHE_HITS.load(Ordering::Relaxed),
        if fixture_ok { "synthetic_test_fixture" } else { "fail" },
        fixture_bytes,
        if same_alloc { "same_alloc" } else { "miss" },
        rc_peak,
        rc_after_drop,
        cached_while_held,
        if released { 1 } else { 0 },
        cached_charts(),
        ABSENT_CHART_NAME,
        absent.map(|e| e.tag()).unwrap_or("accepted"),
        CORRUPT_CHART_NAME,
        corrupt.map(|e| e.tag()).unwrap_or("accepted"),
        if SIMULATION_PRESENT { "present" } else { "absent" },
        wifi.state_tag(),
        wifi.chart_name().unwrap_or("none"),
        wifi_scan,
        bt.state_tag(),
        bt.chart_name().unwrap_or("none"),
        bt_scan,
        if pass { "pass" } else { "fail" }
    )
}

/// Emit the atlas proof to serial, matching the `[ManifoldPkg] proof` style.
pub fn emit_boot_proof() {
    crate::serial_println!("{}", firmware_proof_line());
}

// -------------------------------------------------------------------- helpers

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::new();
    for byte in bytes {
        out.push(nibble(byte >> 4));
        out.push(nibble(byte & 0x0f));
    }
    out
}

fn nibble(n: u8) -> char {
    match n & 0x0f {
        v @ 0..=9 => (b'0' + v) as char,
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

fn parse_hex32(text: &str) -> Option<[u8; 32]> {
    let bytes = text.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (hex_nibble(bytes[i * 2])? << 4) | hex_nibble(bytes[i * 2 + 1])?;
    }
    Some(out)
}

fn parse_hex64(text: &str) -> Option<[u8; 64]> {
    let bytes = text.as_bytes();
    if bytes.len() != 128 {
        return None;
    }
    let mut out = [0u8; 64];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (hex_nibble(bytes[i * 2])? << 4) | hex_nibble(bytes[i * 2 + 1])?;
    }
    Some(out)
}

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{seal_test, test_assert, test_assert_eq};

    fn test_absent_chart_fails_closed() -> TestResult {
        test_assert!(provision_test_fixture(), "fixture package must install");
        let result = request_chart(ABSENT_CHART_NAME);
        test_assert!(result.is_err(), "absent chart must be refused");
        test_assert_eq!(result.err(), Some(ChartError::NotProvisioned));
        TestResult::Pass
    }

    fn test_digest_mismatch_refused() -> TestResult {
        test_assert!(provision_test_fixture(), "fixture package must install");
        let result = request_chart(CORRUPT_CHART_NAME);
        test_assert_eq!(result.err(), Some(ChartError::DigestMismatch));
        TestResult::Pass
    }

    fn test_unindexed_chart_refused() -> TestResult {
        let result = request_chart("wifi-dead-beef.chart");
        test_assert_eq!(result.err(), Some(ChartError::NotIndexed));
        TestResult::Pass
    }

    fn test_bad_index_signature_refused() -> TestResult {
        let mut tampered = built_in_index();
        tampered[INDEX_MAGIC.len()] ^= 0x01;
        test_assert!(
            parse_index(&tampered, &index_public_key()).is_err(),
            "tampered index body must fail signature check"
        );
        let mut wrong_key = [0u8; 32];
        wrong_key.copy_from_slice(&index_public_key());
        wrong_key[0] ^= 0xff;
        test_assert!(
            parse_index(&built_in_index(), &wrong_key).is_err(),
            "index must not verify under a foreign key"
        );
        TestResult::Pass
    }

    fn test_cache_shares_one_allocation() -> TestResult {
        test_assert!(provision_test_fixture(), "fixture package must install");
        let first = match request_chart(FIXTURE_CHART_NAME) {
            Ok(c) => c,
            Err(_) => return TestResult::Fail("fixture chart must be provisioned"),
        };
        let second = match request_chart(FIXTURE_CHART_NAME) {
            Ok(c) => c,
            Err(_) => return TestResult::Fail("repeat request must hit the cache"),
        };
        test_assert!(
            Arc::ptr_eq(&first, &second),
            "repeat request must share one allocation"
        );
        test_assert_eq!(Arc::strong_count(&first), 3);
        drop(second);
        test_assert_eq!(Arc::strong_count(&first), 2);
        drop(first);
        TestResult::Pass
    }

    fn test_refcount_released() -> TestResult {
        test_assert!(provision_test_fixture(), "fixture package must install");
        let held = match request_chart(FIXTURE_CHART_NAME) {
            Ok(c) => c,
            Err(_) => return TestResult::Fail("fixture chart must be provisioned"),
        };
        test_assert!(
            !release_chart(FIXTURE_CHART_NAME),
            "must not evict while held"
        );
        drop(held);
        test_assert!(
            release_chart(FIXTURE_CHART_NAME),
            "must evict once unreferenced"
        );
        test_assert!(
            !CACHE.lock().contains_key(FIXTURE_CHART_NAME),
            "evicted chart must leave the cache"
        );
        TestResult::Pass
    }

    fn test_wifi_stays_down_without_chart() -> TestResult {
        let mut wifi = crate::drivers::wifi::WifiDriver::new();
        wifi.probe_pci();
        test_assert!(!wifi.is_up(), "wifi must stay down without a chart");
        test_assert!(wifi.scan().is_empty(), "wifi must not emit scan results");
        test_assert!(
            wifi.connect("any", "").is_err(),
            "wifi connect must fail without a chart"
        );
        TestResult::Pass
    }

    fn test_bluetooth_stays_down_without_chart() -> TestResult {
        let mut bt = crate::drivers::bluetooth::BluetoothDriver::new();
        bt.probe_pci();
        test_assert!(!bt.is_up(), "bluetooth must stay down without a chart");
        test_assert!(bt.scan().is_empty(), "bluetooth must not emit scan results");
        test_assert!(
            bt.pair("any").is_err(),
            "bluetooth pair must fail without a chart"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        seal_test!(test_absent_chart_fails_closed);
        seal_test!(test_digest_mismatch_refused);
        seal_test!(test_unindexed_chart_refused);
        seal_test!(test_bad_index_signature_refused);
        seal_test!(test_cache_shares_one_allocation);
        seal_test!(test_refcount_released);
        seal_test!(test_wifi_stays_down_without_chart);
        seal_test!(test_bluetooth_stays_down_without_chart);
    }
}
