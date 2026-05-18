// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Dependency resolution via Voronoi cell lookup.

use alloc::string::String;
use alloc::vec::Vec;

pub struct DependencyResolver {
    _cell_count: usize,
}

impl DependencyResolver {
    pub fn new() -> Self {
        Self { _cell_count: 8 }
    }

    /// [Sim] No real dependency graph — returns empty deps list.
    pub fn resolve(&self, _name: &str) -> Vec<String> {
        Vec::new()
    }
}
