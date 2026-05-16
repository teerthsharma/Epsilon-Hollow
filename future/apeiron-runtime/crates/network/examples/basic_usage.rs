// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Basic usage example for AETHER-Link kernel.
//!
//! Run with: cargo run --example basic_usage

use aether_link::AetherLinkKernel;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     🚀 AETHER-Link: Adaptive I/O Super-Kernel Demo 🚀      ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Paper: DOI 10.13140/RG.2.2.22443.91687                    ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize with default parameters
    let mut kernel = AetherLinkKernel::default();
    println!("📊 Kernel initialized with default parameters:");
    println!("   ε (epsilon): {:.3}", kernel.epsilon);
    println!("   φ (phi):     {:.3}", kernel.phi);
    println!("   λ (lambda):  {:?}", kernel.lambda);
    println!("   bias:        {:.3}", kernel.bias);
    println!();

    // Simulate an LBA stream (Logical Block Addresses)
    let lba_stream: Vec<u64> = vec![
        1000, 1001, 1002, 1003, 1004,  // Sequential reads
        1010, 1011, 1012,              // Small jump
        2000, 2001, 2002,              // Large jump (new file?)
        2003, 2004, 2005, 2006,        // Continue sequential
    ];

    println!("📥 Processing LBA stream ({} addresses):", lba_stream.len());
    println!("   {:?}", lba_stream);
    println!();

    // Process the I/O cycle
    let should_prefetch = kernel.process_io_cycle(&lba_stream);

    println!("⚡ Decision: {}", if should_prefetch {
        "PREFETCH → Bypass OS, use DirectStorage!"
    } else {
        "STANDARD → Use OS page cache"
    });
    println!();

    // Show updated state
    println!("📈 Updated kernel state after learning:");
    println!("   ε (epsilon): {:.3} ({})", kernel.epsilon, 
        if kernel.epsilon > 0.5 { "conservative" } else { "aggressive" });
    println!("   φ (phi):     {:.3} rad", kernel.phi);
    println!();

    // Run multiple cycles to show adaptation
    println!("🔄 Running 100 I/O cycles to demonstrate adaptation...");
    for i in 0..100 {
        let dynamic_stream: Vec<u64> = (i * 10..i * 10 + 15).map(|x| x as u64).collect();
        kernel.process_io_cycle(&dynamic_stream);
    }

    println!("   Cycles processed: {}", kernel.cycles);
    println!("   Prefetches triggered: {}", kernel.prefetches);
    println!("   Prefetch ratio: {:.1}%", kernel.prefetch_ratio() * 100.0);
    println!();

    // Compare presets
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📋 Preset Comparison:");
    println!();

    let hft = AetherLinkKernel::new_hft();
    println!("   🏦 HFT Preset (Conservative):");
    println!("      ε = {:.3}, φ = {:.3}", hft.epsilon, hft.phi);

    let gaming = AetherLinkKernel::new_gaming();
    println!("   🎮 Gaming Preset (Aggressive):");
    println!("      ε = {:.3}, φ = {:.3}", gaming.epsilon, gaming.phi);
    println!();

    println!("✅ Demo complete! Run `cargo bench` for performance metrics.");
}
