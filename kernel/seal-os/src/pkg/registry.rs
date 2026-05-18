// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Local package registry backed by ManifoldFS inodes.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::manifest::PackageManifest;

pub struct PackageRegistry {
    packages: BTreeMap<String, PackageManifest>,
}

impl PackageRegistry {
    pub fn new() -> Self {
        Self {
            packages: BTreeMap::new(),
        }
    }

    pub fn install(&mut self, manifest: PackageManifest) {
        self.packages.insert(manifest.name.clone(), manifest);
    }

    pub fn remove(&mut self, name: &str) {
        self.packages.remove(name);
    }

    pub fn is_installed(&self, name: &str) -> bool {
        self.packages.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Option<&PackageManifest> {
        self.packages.get(name)
    }

    pub fn list(&self) -> Vec<&PackageManifest> {
        self.packages.values().collect()
    }

    pub fn count(&self) -> usize {
        self.packages.len()
    }
}
