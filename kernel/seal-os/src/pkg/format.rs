// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! `.eph` package format parser and verifier.
//!
//! Layout:
//!   [4]   magic "EPH\0"
//!   [4]   manifest_len (big-endian u32)
//!   [N]   manifest bytes (UTF-8 JSON-like)
//!   [64]  ed25519 signature
//!   [file sections...]
//!     [2]   path_len
//!     [N]   path bytes
//!     [4]   file_data_len (big-endian u32)
//!     [N]   file data
//!   [4]   trailer "END\0"
//!
//! Manifest bytes and path bytes must both be valid UTF-8; a package carrying
//! anything else is refused here rather than repaired into a lossy string.
//!
//! This module parses and does not verify. Signature verification lives in
//! `super::verify_package_signature`, over the length-separated preimage in
//! `super::signature_preimage`. The weak verifier that used to sit here signed
//! `name || version || (path || hash)*` — no length separators, no description,
//! no dependency list — and was deleted rather than left as the name a caller
//! would reach for first.

use alloc::string::String;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use super::manifest::PackageManifest;

#[derive(Debug, Clone)]
pub struct EphFileEntry {
    pub path: String,
    pub data: Vec<u8>,
    pub hash: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct EphPackage {
    pub manifest: PackageManifest,
    pub signature: [u8; 64],
    pub files: Vec<EphFileEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EphError {
    BadMagic,
    BadTrailer,
    BadManifest,
    /// A file path in the package was not valid UTF-8.
    BadPath,
    BadSignature,
    HashMismatch { path: String },
    ShortRead,
}

pub fn parse_eph(data: &[u8]) -> Result<EphPackage, EphError> {
    if data.len() < 8 || &data[0..4] != b"EPH\0" {
        return Err(EphError::BadMagic);
    }
    let manifest_len = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let mut offset = 8;
    if offset + manifest_len + 64 + 4 > data.len() {
        return Err(EphError::ShortRead);
    }
    let manifest_bytes = &data[offset..offset + manifest_len];
    offset += manifest_len;

    let manifest = parse_manifest(manifest_bytes)?;

    let mut signature = [0u8; 64];
    signature.copy_from_slice(&data[offset..offset + 64]);
    offset += 64;

    let mut files = Vec::new();
    let mut found_trailer = false;
    while offset + 4 <= data.len() {
        if data[offset..offset + 4] == *b"END\0" {
            offset += 4;
            found_trailer = true;
            break;
        }
        if offset + 2 > data.len() {
            return Err(EphError::ShortRead);
        }
        let path_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        if offset + path_len + 4 > data.len() {
            return Err(EphError::ShortRead);
        }
        // Refused, never repaired: a lossy decode turns invalid bytes into
        // U+FFFD, so the path every later layer checks and writes is no longer
        // the path the package declared and the original bytes are gone.
        let path = match core::str::from_utf8(&data[offset..offset + path_len]) {
            Ok(path) => String::from(path),
            Err(_) => return Err(EphError::BadPath),
        };
        offset += path_len;
        let file_len = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        if offset + file_len > data.len() {
            return Err(EphError::ShortRead);
        }
        let file_data = data[offset..offset + file_len].to_vec();
        offset += file_len;

        let mut hasher = Sha256::new();
        hasher.update(&file_data);
        let hash: [u8; 32] = hasher.finalize().into();

        files.push(EphFileEntry {
            path,
            data: file_data,
            hash,
        });
    }

    // The trailer is mandatory: the loop must have matched `END\0` exactly
    // (not merely run out of remaining bytes), and nothing may follow it.
    if !found_trailer || offset != data.len() {
        return Err(EphError::BadTrailer);
    }

    Ok(EphPackage {
        manifest,
        signature,
        files,
    })
}

fn parse_manifest(bytes: &[u8]) -> Result<PackageManifest, EphError> {
    // Every field parsed out of here is covered by the package signature
    // (`super::signature_preimage`). Decoding lossily would let two different
    // manifests on the wire collapse to one preimage, so one signature would
    // cover both.
    let text = match core::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => return Err(EphError::BadManifest),
    };
    let mut name = String::new();
    let mut version = String::new();
    let mut description = String::new();
    let mut deps = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("name=") {
            name = String::from(v.trim_matches('"'));
        } else if let Some(v) = line.strip_prefix("version=") {
            version = String::from(v.trim_matches('"'));
        } else if let Some(v) = line.strip_prefix("description=") {
            description = String::from(v.trim_matches('"'));
        } else if let Some(v) = line.strip_prefix("dep=") {
            deps.push(String::from(v.trim_matches('"')));
        }
    }
    if name.is_empty() || version.is_empty() {
        return Err(EphError::BadManifest);
    }
    let mut manifest = PackageManifest::new(&name, &version, super::carrier::CarrierType::Aether);
    manifest.description = description;
    manifest.dependencies = deps;
    Ok(manifest)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert_eq;
    use crate::testing::TestResult;

    fn build_test_eph() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"EPH\0");
        let manifest = b"name=\"testpkg\"\nversion=\"1.0.0\"\ndep=\"libc\"";
        data.extend_from_slice(&(manifest.len() as u32).to_be_bytes());
        data.extend_from_slice(manifest);
        data.extend_from_slice(&[0u8; 64]); // dummy signature
                                            // file
        data.extend_from_slice(&(5u16).to_be_bytes());
        data.extend_from_slice(b"hello");
        data.extend_from_slice(&(5u32).to_be_bytes());
        data.extend_from_slice(b"world");
        data.extend_from_slice(b"END\0");
        data
    }

    fn test_parse_eph_basic() -> TestResult {
        let raw = build_test_eph();
        let Ok(pkg) = parse_eph(&raw) else {
            return TestResult::Fail("well-formed .eph failed to parse");
        };
        test_assert_eq!(pkg.manifest.name, "testpkg");
        test_assert_eq!(pkg.manifest.version, "1.0.0");
        test_assert_eq!(pkg.manifest.dependencies.len(), 1);
        test_assert_eq!(pkg.files.len(), 1);
        test_assert_eq!(pkg.files[0].path, "hello");
        test_assert_eq!(pkg.files[0].data, b"world");
        TestResult::Pass
    }

    fn test_bad_magic() -> TestResult {
        let mut raw = build_test_eph();
        raw[0] = b'X';
        test_assert_eq!(parse_eph(&raw).err(), Some(EphError::BadMagic));
        TestResult::Pass
    }

    fn test_sha256_hash() -> TestResult {
        let raw = build_test_eph();
        let Ok(pkg) = parse_eph(&raw) else {
            return TestResult::Fail("well-formed .eph failed to parse");
        };
        let mut hasher = Sha256::new();
        hasher.update(b"world");
        let expected: [u8; 32] = hasher.finalize().into();
        test_assert_eq!(pkg.files[0].hash, expected);
        TestResult::Pass
    }

    /// Regression for the trailer bypass: the parse loop can exit because
    /// fewer than 4 bytes remain without ever matching `END\0`. If that
    /// happens to land exactly on `offset == data.len()` (as it does here,
    /// with the trailer omitted entirely), a package with no trailer at all
    /// must still be rejected — the trailer is mandatory, not merely
    /// checked-when-present.
    fn test_missing_trailer_rejects() -> TestResult {
        let mut raw = build_test_eph();
        let end = raw.len();
        raw.truncate(end - 4); // drop the trailing "END\0"
        test_assert_eq!(parse_eph(&raw).err(), Some(EphError::BadTrailer));
        TestResult::Pass
    }

    /// Regression for the second half of the same bug: once `END\0` is
    /// matched mid-stream, trailing bytes after it must be rejected rather
    /// than accepted (the old check re-inspected the just-consumed `END\0`
    /// itself instead of the bytes that actually follow it).
    fn test_trailing_garbage_after_trailer_rejects() -> TestResult {
        let mut raw = build_test_eph();
        raw.extend_from_slice(b"\xff\xff\xff\xff");
        test_assert_eq!(parse_eph(&raw).err(), Some(EphError::BadTrailer));
        TestResult::Pass
    }

    /// A package cut short must be refused, not partially accepted. Two cut
    /// points, one per bounds check: inside the final section's data (the
    /// declared `file_data_len` runs past the buffer) and inside the fixed
    /// header (the declared `manifest_len` plus signature plus trailer runs
    /// past the buffer).
    fn test_truncated_rejects() -> TestResult {
        let mut mid_section = build_test_eph();
        let end = mid_section.len();
        mid_section.truncate(end - 7); // drops "END\0" and 3 bytes of "world"
        test_assert_eq!(parse_eph(&mid_section).err(), Some(EphError::ShortRead));

        let mut mid_header = build_test_eph();
        mid_header.truncate(20); // 8-byte header parsed, manifest runs past it
        test_assert_eq!(parse_eph(&mid_header).err(), Some(EphError::ShortRead));
        TestResult::Pass
    }

    /// Every length in a `.eph` is attacker-controlled. A package that claims
    /// a manifest, a path or a file longer than the buffer holding it must be
    /// refused by a bounds check, never turned into an out-of-range slice or
    /// a wrapped offset.
    fn test_hostile_lengths_rejected() -> TestResult {
        let pristine = build_test_eph();
        // The first section header sits after the 8-byte fixed header, the
        // manifest and the 64-byte signature.
        let section = 8
            + u32::from_be_bytes([pristine[4], pristine[5], pristine[6], pristine[7]]) as usize
            + 64;

        let mut huge_manifest = build_test_eph();
        huge_manifest[4..8].copy_from_slice(&u32::MAX.to_be_bytes());
        test_assert_eq!(parse_eph(&huge_manifest).err(), Some(EphError::ShortRead));

        let mut huge_path = build_test_eph();
        huge_path[section..section + 2].copy_from_slice(&u16::MAX.to_be_bytes());
        test_assert_eq!(parse_eph(&huge_path).err(), Some(EphError::ShortRead));

        let mut huge_file = build_test_eph();
        let len_at = section + 2 + 5; // past path_len and the 5-byte path
        huge_file[len_at..len_at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        test_assert_eq!(parse_eph(&huge_file).err(), Some(EphError::ShortRead));
        TestResult::Pass
    }

    /// A package path that is not valid UTF-8 must be refused where it is
    /// read, not silently repaired. Decoding it lossily turns the invalid
    /// bytes into U+FFFD and hands every later layer a plausible-looking
    /// string whose original bytes are gone.
    ///
    /// The second half pins *what* is refused: the invalid bytes, not the
    /// replacement character. A path whose bytes are the valid encoding of
    /// U+FFFD is valid UTF-8 and must parse — refusing it here would be the
    /// proxy check standing in for the real one, and it would leave the real
    /// hole open for every other invalid sequence.
    fn test_non_utf8_path_rejected() -> TestResult {
        let section = |raw: &[u8]| {
            8 + u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize + 64
        };

        let mut invalid = build_test_eph();
        let at = section(&invalid) + 2;
        invalid[at..at + 5].copy_from_slice(b"h\xffl\xfeo");
        test_assert_eq!(parse_eph(&invalid).err(), Some(EphError::BadPath));

        let mut replacement = build_test_eph();
        let at = section(&replacement) + 2;
        replacement[at..at + 5].copy_from_slice("\u{fffd}xy".as_bytes());
        let Ok(pkg) = parse_eph(&replacement) else {
            return TestResult::Fail("a path of valid UTF-8 must parse, U+FFFD included");
        };
        test_assert_eq!(pkg.files[0].path, "\u{fffd}xy");
        TestResult::Pass
    }

    /// Same rule one field up: the manifest carries the name, version,
    /// description and dependency list that the package signature covers, so
    /// decoding it lossily lets two different wire manifests collapse to one
    /// preimage. Refuse the bytes instead.
    fn test_non_utf8_manifest_rejected() -> TestResult {
        let mut raw = build_test_eph();
        raw[8 + 10] = 0xff; // inside name="testpkg"
        test_assert_eq!(parse_eph(&raw).err(), Some(EphError::BadManifest));
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("pkg::format::parse_eph_basic", test_parse_eph_basic);
        crate::testing::register_test("pkg::format::bad_magic", test_bad_magic);
        crate::testing::register_test("pkg::format::sha256_hash", test_sha256_hash);
        crate::testing::register_test(
            "pkg::format::missing_trailer_rejects",
            test_missing_trailer_rejects,
        );
        crate::testing::register_test(
            "pkg::format::trailing_garbage_after_trailer_rejects",
            test_trailing_garbage_after_trailer_rejects,
        );
        crate::testing::register_test("pkg::format::truncated_rejects", test_truncated_rejects);
        crate::testing::register_test(
            "pkg::format::hostile_lengths_rejected",
            test_hostile_lengths_rejected,
        );
        crate::testing::register_test(
            "pkg::format::non_utf8_path_rejected",
            test_non_utf8_path_rejected,
        );
        crate::testing::register_test(
            "pkg::format::non_utf8_manifest_rejected",
            test_non_utf8_manifest_rejected,
        );
    }
}
