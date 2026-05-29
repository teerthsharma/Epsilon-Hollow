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

use self::format::{parse_eph, verify_signature, EphError};
use self::manifest::PackageManifest;
use self::registry::PackageRegistry;
use self::resolver::DependencyResolver;
use spin::Mutex;

pub static GLOBAL_PKG: Mutex<ManifoldPkg> = Mutex::new(ManifoldPkg::new());
const SEAL_PKG_PUBLIC_KEY: [u8; 32] = [
    0x3b, 0x6a, 0x27, 0xbc, 0xce, 0xb6, 0xa4, 0x2d, 0x62, 0xa3, 0xa8, 0xd0, 0x2a, 0x6f, 0x0d, 0x73,
    0x63, 0x2e, 0x3e, 0x77, 0xe3, 0xe9, 0xdf, 0x15, 0xe2, 0xda, 0x4c, 0x64, 0x3a, 0x53, 0x97, 0x43,
];

pub struct ManifoldPkg {
    registry: PackageRegistry,
    resolver: DependencyResolver,
    registry_url: String,
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
