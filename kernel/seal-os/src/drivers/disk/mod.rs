// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

pub mod ahci;

/// Probe the disk controllers that can carry the root filesystem.
pub fn init() {
    let _ = ahci::probe();
    crate::drivers::block::virtio_blk::init();
}
