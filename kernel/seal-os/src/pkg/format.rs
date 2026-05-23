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

use alloc::format;
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
    while offset + 4 <= data.len() {
        if data[offset..offset + 4] == *b"END\0" {
            offset += 4;
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
        let path = String::from_utf8_lossy(&data[offset..offset + path_len]).into_owned();
        offset += path_len;
        let file_len = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
        offset += 4;
        if offset + file_len > data.len() {
            return Err(EphError::ShortRead);
        }
        let file_data = data[offset..offset + file_len].to_vec();
        offset += file_len;

        let mut hasher = Sha256::new();
        hasher.update(&file_data);
        let hash: [u8; 32] = hasher.finalize().into();

        files.push(EphFileEntry { path, data: file_data, hash });
    }

    if offset != data.len() && (offset < 4 || data[offset - 4..offset] != *b"END\0") {
        return Err(EphError::BadTrailer);
    }

    Ok(EphPackage { manifest, signature, files })
}

pub fn verify_signature(pkg: &EphPackage, public_key: &[u8; 32]) -> Result<(), EphError> {
    use ed25519_dalek::{VerifyingKey, Signature};
    let vk = VerifyingKey::from_bytes(public_key).map_err(|_| EphError::BadSignature)?;
    let sig = Signature::from_bytes(&pkg.signature);
    // Build signed data = manifest bytes concatenated with file hashes
    let mut signed = Vec::new();
    signed.extend_from_slice(pkg.manifest.name.as_bytes());
    signed.extend_from_slice(pkg.manifest.version.as_bytes());
    for f in &pkg.files {
        signed.extend_from_slice(f.path.as_bytes());
        signed.extend_from_slice(&f.hash);
    }
    vk.verify_strict(&signed, &sig).map_err(|_| EphError::BadSignature)
}

fn parse_manifest(bytes: &[u8]) -> Result<PackageManifest, EphError> {
    let text = String::from_utf8_lossy(bytes);
    let mut name = String::new();
    let mut version = String::new();
    let mut description = String::new();
    let mut deps = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("name=") {
            name = String::from(line[5..].trim_matches('"'));
        } else if line.starts_with("version=") {
            version = String::from(line[8..].trim_matches('"'));
        } else if line.starts_with("description=") {
            description = String::from(line[12..].trim_matches('"'));
        } else if line.starts_with("dep=") {
            deps.push(String::from(line[4..].trim_matches('"')));
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_parse_eph_basic() {
        let raw = build_test_eph();
        let pkg = parse_eph(&raw).unwrap();
        assert_eq!(pkg.manifest.name, "testpkg");
        assert_eq!(pkg.manifest.version, "1.0.0");
        assert_eq!(pkg.manifest.dependencies.len(), 1);
        assert_eq!(pkg.files.len(), 1);
        assert_eq!(pkg.files[0].path, "hello");
        assert_eq!(pkg.files[0].data, b"world");
    }

    #[test]
    fn test_bad_magic() {
        let mut raw = build_test_eph();
        raw[0] = b'X';
        assert_eq!(parse_eph(&raw), Err(EphError::BadMagic));
    }

    #[test]
    fn test_sha256_hash() {
        let raw = build_test_eph();
        let pkg = parse_eph(&raw).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(b"world");
        let expected: [u8; 32] = hasher.finalize().into();
        assert_eq!(pkg.files[0].hash, expected);
    }
}
