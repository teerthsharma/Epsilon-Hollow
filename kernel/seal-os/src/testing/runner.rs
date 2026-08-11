// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! QEMU integration test runner.
//!
//! When the `test-mode` feature is enabled, `kernel_main` calls `test_main()`
//! which runs all registered tests and then exits QEMU via the isa-debug-exit
//! device (port 0x501).

use super::harness::run_all_tests_and_exit;

/// Entry point for the in-kernel test runner.
pub fn test_main() -> ! {
    // Register all in-kernel tests from each module
    crate::apps::installer::tests::register_all();
    crate::apps::shell::tests::register_all();
    crate::wm::window::tests::register_all();
    crate::wm::welcome::tests::register_all();
    crate::wm::taskbar::tests::register_all();
    crate::graphics::htek::tests::register_all();
    crate::wm::compositor::tests::register_all();
    crate::graphics::topo_render::tests::register_all();
    crate::memory::tests::register_all();
    crate::fs::block_store::tests::register_all();
    crate::fs::manifold_fs::tests::register_all();
    crate::fs::vfs::tests::register_all();
    crate::fs::dir_hash::tests::register_all();
    crate::fs::ext2::tests::register_all();
    crate::fs::parity::tests::register_all();
    crate::process::scheduler::tests::register_all();
    crate::process::elf::tests::register_all();
    crate::security::tests::register_all();
    crate::syscall::table::tests::register_all();
    crate::atlas::tests::register_all();
    crate::bundle::tests::register_all();
    crate::lang::tests::register_all();
    crate::net::dns::tests::register_all();
    crate::net::udp::tests::register_all();
    crate::pkg::channel::tests::register_all();
    crate::drivers::apic::tests::register_all();
    crate::drivers::block::ahci::tests::register_all();
    crate::drivers::block::virtio_blk::tests::register_all();
    crate::drivers::block::ramdisk::tests::register_all();
    crate::drivers::acpi::rsdp::tests::register_all();
    crate::drivers::acpi::madt::tests::register_all();
    crate::drivers::acpi::fadt::tests::register_all();
    // Chains xhci::tests — see drivers/usb/mass_storage.rs.
    crate::drivers::usb::mass_storage::tests::register_all();
    crate::drivers::gpu::gpu_bench::tests::register_all();
    // Chains x509::tests and ecdhe::tests — see drivers/net/tls.rs.
    crate::drivers::net::tls::tests::register_all();
    crate::cpu::tests::register_all();
    crate::sync::tests::register_all();
    crate::ml_engine::stratum::tests::register_all();
    crate::ml_engine::foliation::tests::register_all();
    crate::ml_engine::topo_asm::tests::register_all();
    crate::ml_engine::tests::register_all();

    run_all_tests_and_exit()
}
