// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Key-value settings store backed by ManifoldFS.

use alloc::collections::BTreeMap;
use alloc::string::String;
use spin::Mutex;

pub static GLOBAL_SETTINGS: Mutex<Settings> = Mutex::new(Settings::new());

pub struct Settings {
    store: BTreeMap<String, String>,
}

impl Settings {
    pub const fn new() -> Self {
        Self { store: BTreeMap::new() }
    }

    pub fn init_defaults(&mut self) {
        self.store.insert(String::from("theme"), String::from("dark"));
        self.store.insert(String::from("font-size"), String::from("14"));
        self.store.insert(String::from("wallpaper"), String::from("default"));
        self.store.insert(String::from("shell-prompt"), String::from("seal"));
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
