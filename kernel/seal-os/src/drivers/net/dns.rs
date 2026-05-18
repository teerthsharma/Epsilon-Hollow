// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! DNS resolver — name → IP address lookup.

use alloc::collections::BTreeMap;
use alloc::string::String;

pub struct DnsResolver {
    cache: BTreeMap<String, [u8; 4]>,
    server: [u8; 4],
}

impl DnsResolver {
    pub fn new(server: [u8; 4]) -> Self {
        let mut cache = BTreeMap::new();
        // [Sim] No NIC — these are hardcoded entries, not real DNS lookups
        cache.insert(String::from("pypi.org"), [151, 101, 0, 223]);
        cache.insert(String::from("github.com"), [140, 82, 121, 3]);
        cache.insert(String::from("crates.io"), [108, 138, 64, 68]);
        Self { cache, server }
    }

    /// [Sim] Returns hardcoded cache entry — no real DNS query is performed.
    pub fn resolve(&self, hostname: &str) -> Option<[u8; 4]> {
        self.cache.get(hostname).copied()
    }

    pub fn server(&self) -> [u8; 4] {
        self.server
    }
}
