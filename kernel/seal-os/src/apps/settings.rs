// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Key-value settings store backed by ManifoldFS.

use alloc::collections::BTreeMap;
use alloc::string::String;

pub struct Settings {
    store: BTreeMap<String, String>,
}

impl Settings {
    pub fn new() -> Self {
        let mut store = BTreeMap::new();
        store.insert(String::from("theme"), String::from("dark"));
        store.insert(String::from("font-size"), String::from("14"));
        store.insert(String::from("wallpaper"), String::from("default"));
        store.insert(String::from("shell-prompt"), String::from("seal"));
        Self { store }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.store.get(key).map(|s| s.as_str())
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.store.insert(String::from(key), String::from(value));
    }

    pub fn list(&self) -> &BTreeMap<String, String> {
        &self.store
    }
}
