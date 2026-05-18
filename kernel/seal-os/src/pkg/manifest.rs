// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Package manifest — name, version, carrier, dependencies.

use alloc::string::String;
use alloc::vec::Vec;

use super::carrier::CarrierType;

pub struct PackageManifest {
    pub name: String,
    pub version: String,
    pub carrier: CarrierType,
    pub description: String,
    pub dependencies: Vec<String>,
    pub voronoi_cell: usize,
}

impl PackageManifest {
    pub fn new(name: &str, version: &str, carrier: CarrierType) -> Self {
        Self {
            name: String::from(name),
            version: String::from(version),
            carrier,
            description: String::new(),
            dependencies: Vec::new(),
            voronoi_cell: 0,
        }
    }
}
