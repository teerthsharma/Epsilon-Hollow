// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ManifoldPkg — native package manager. Packages = ManifoldFS inodes, deps = Voronoi cells.

pub mod carrier;
pub mod manifest;
pub mod registry;
pub mod resolver;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use self::manifest::PackageManifest;
use self::registry::PackageRegistry;
use self::resolver::DependencyResolver;

pub struct ManifoldPkg {
    registry: PackageRegistry,
    resolver: DependencyResolver,
}

impl ManifoldPkg {
    pub fn new() -> Self {
        Self {
            registry: PackageRegistry::new(),
            resolver: DependencyResolver::new(),
        }
    }

    pub fn install(&mut self, name: &str) -> Result<String, String> {
        if self.registry.is_installed(name) {
            return Err(format!("'{}' is already installed", name));
        }

        let deps = self.resolver.resolve(name);
        for dep in &deps {
            if !self.registry.is_installed(dep) {
                let manifest = PackageManifest::new(dep, "1.0.0", carrier::CarrierType::Aether);
                self.registry.install(manifest);
            }
        }

        let manifest = PackageManifest::new(name, "1.0.0", carrier::CarrierType::Aether);
        self.registry.install(manifest);

        Ok(format!(
            "Installed '{}' v1.0.0 — metadata stored ({} deps resolved)",
            name,
            deps.len()
        ))
    }

    pub fn remove(&mut self, name: &str) -> Result<String, String> {
        if !self.registry.is_installed(name) {
            return Err(format!("'{}' is not installed", name));
        }
        self.registry.remove(name);
        Ok(format!("Removed '{}' — metadata cleared", name))
    }

    pub fn list(&self) -> Vec<&PackageManifest> {
        self.registry.list()
    }

    pub fn package_count(&self) -> usize {
        self.registry.count()
    }
}
