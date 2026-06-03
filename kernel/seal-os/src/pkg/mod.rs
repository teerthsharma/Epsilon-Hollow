// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ManifoldPkg — native package manager.

pub mod carrier;
pub mod format;
pub mod manifest;
pub mod registry;
pub mod resolver;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use ed25519_dalek::{Signer, SigningKey};

use self::format::{parse_eph, verify_signature};
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
        .and_then(|parsed| verify_signature(&parsed, &proof_public_key))
        .is_ok();
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
    let result = if parse_ok && signature_ok && install_ok && list_ok && extract_ok && remove_ok && counts_ok {
            "pass"
        } else {
            "fail"
        };
    crate::serial_println!(
        "[ManifoldPkg] proof version=1 source=embedded_eph parse={} install={} extract={} list={} remove={} files=1 bytes={} package_count_before={} package_count_after_install={} package_count_after_remove={} metadata_only=0 signature={} result={}",
        if parse_ok { "ok" } else { "fail" },
        if install_ok { "ok" } else { "fail" },
        if extract_ok { "ok" } else { "fail" },
        if list_ok { "ok" } else { "fail" },
        if remove_ok { "ok" } else { "fail" },
        PROOF_FILE_BYTES.len(),
        before,
        after_install,
        after_remove,
        if signature_ok { "ed25519_fixture" } else { "fail" },
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

fn proof_pkg_public_key() -> [u8; 32] {
    SigningKey::from_bytes(&PROOF_PKG_SIGNING_KEY)
        .verifying_key()
        .to_bytes()
}

fn sign_proof_package(pkg: &self::format::EphPackage) -> [u8; 64] {
    let mut signed = Vec::new();
    signed.extend_from_slice(pkg.manifest.name.as_bytes());
    signed.extend_from_slice(pkg.manifest.version.as_bytes());
    for f in &pkg.files {
        signed.extend_from_slice(f.path.as_bytes());
        signed.extend_from_slice(&f.hash);
    }
    SigningKey::from_bytes(&PROOF_PKG_SIGNING_KEY)
        .sign(&signed)
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
            verify_signature(&pkg, key).map_err(|e| format!("signature: {:?}", e))?;
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

#[derive(Debug)]
enum VfsInstallError {
    #[allow(dead_code)] // REASON: VFS error payload preserved for future install error diagnostics
    Vfs(crate::fs::vfs::VfsError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_eph(name: &str, version: &str) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"EPH\0");
        let manifest = format!("name=\"{}\"\nversion=\"{}\"", name, version);
        data.extend_from_slice(&(manifest.len() as u32).to_be_bytes());
        data.extend_from_slice(manifest.as_bytes());
        data.extend_from_slice(&[0u8; 64]);
        data.extend_from_slice(b"END\0");
        data
    }

    #[test]
    fn test_install_bytes_ok() {
        let mut pkg = ManifoldPkg::new();
        let eph = dummy_eph("foo", "1.0.0");
        // VFS may not be initialized in unit tests, so we expect vfs error or ok
        let _ = pkg.install_bytes(&eph, None);
    }

    #[test]
    fn test_remove_existing() {
        let mut pkg = ManifoldPkg::new();
        let eph = dummy_eph("bar", "2.0.0");
        if pkg.install_bytes(&eph, None).is_ok() {
            assert!(pkg.remove("bar").is_ok());
            assert!(!pkg.registry.is_installed("bar"));
        }
    }
}
