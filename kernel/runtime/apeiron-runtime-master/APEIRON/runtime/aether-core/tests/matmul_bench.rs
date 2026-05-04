// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

use aether_core::ml::tensor::Tensor;
use std::time::Instant;

#[test]
fn bench_matmul_large() {
    let size = 400; // 400x400 should be enough to show difference (O(n^3) -> 64M ops)
    println!("Initializing matrices of size {}x{}", size, size);
    // Use kaiming_uniform to populate with random data
    let a = Tensor::kaiming_uniform((size, size), Some(1));
    let b = Tensor::kaiming_uniform((size, size), Some(2));

    println!("Starting multiplication...");
    let start = Instant::now();
    let _c = a.matmul(&b);
    let duration = start.elapsed();

    println!("Matmul {}x{} took: {:?}", size, size, duration);
}
