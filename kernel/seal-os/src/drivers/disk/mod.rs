// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

pub mod ahci;

pub fn init() {
    let _ = ahci::probe();
}
