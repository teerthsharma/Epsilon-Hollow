// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Copy/paste buffer for ManifoldFS files.

use alloc::string::String;

#[derive(Clone)]
pub struct ClipboardEntry {
    pub name: String,
    pub source_dir: u64,
}

pub struct Clipboard {
    entry: Option<ClipboardEntry>,
}

impl Clipboard {
    pub fn new() -> Self {
        Self { entry: None }
    }

    pub fn copy(&mut self, name: &str, source_dir: u64) {
        self.entry = Some(ClipboardEntry {
            name: String::from(name),
            source_dir,
        });
    }

    pub fn peek(&self) -> Option<&ClipboardEntry> {
        self.entry.as_ref()
    }

    pub fn take(&mut self) -> Option<ClipboardEntry> {
        self.entry.take()
    }

    pub fn is_empty(&self) -> bool {
        self.entry.is_none()
    }
}
