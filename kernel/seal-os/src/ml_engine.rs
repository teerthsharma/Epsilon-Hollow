// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ML Engine — wraps aether-core tensor and neural operations for kernel runtime.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use aether_core::ml::{
    Activation, DenseLayer, MLP, OptimizerConfig, Tensor, TrainingResult,
};

/// Status of the ML runtime.
pub struct MlStatus {
    pub tensor_ops_available: bool,
    pub neural_net_available: bool,
    pub avx2_detected: bool,
    pub avx512_detected: bool,
}

impl MlStatus {
    pub fn detect() -> Self {
        Self {
            tensor_ops_available: true,
            neural_net_available: true,
            avx2_detected: detect_avx2(),
            avx512_detected: detect_avx512(),
        }
    }
}

/// Detect AVX2 support via CPUID.
fn detect_avx2() -> bool {
    unsafe {
        // CPUID leaf 1: check ECX bit 28 (AVX) and bit 27 (OSXSAVE)
        let leaf1 = core::arch::x86_64::__cpuid(1);
        let avx_available = (leaf1.ecx & (1 << 28)) != 0;
        let osxsave = (leaf1.ecx & (1 << 27)) != 0;
        if !avx_available || !osxsave {
            return false;
        }
        // CPUID leaf 7 subleaf 0: check EBX bit 5 (AVX2)
        let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
        (leaf7.ebx & (1 << 5)) != 0
    }
}

/// Detect AVX-512 support via CPUID.
fn detect_avx512() -> bool {
    unsafe {
        // CPUID leaf 1: check ECX bit 28 (AVX)
        let leaf1 = core::arch::x86_64::__cpuid(1);
        if (leaf1.ecx & (1 << 28)) == 0 {
            return false;
        }
        // CPUID leaf 7 subleaf 0: check EBX bit 16 (AVX-512F)
        let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
        (leaf7.ebx & (1 << 16)) != 0
    }
}

/// Create a tensor from raw data.
pub fn tensor_from_data(data: Vec<f64>, shape: Vec<usize>) -> Result<Tensor, String> {
    if data.len() != shape.iter().product::<usize>() {
        return Err(format!(
            "Data length {} does not match shape product {}",
            data.len(),
            shape.iter().product::<usize>()
        ));
    }
    Ok(Tensor::from_vec(data, shape))
}

/// Matrix multiply two tensors.
pub fn tensor_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.shape.len() < 2 || b.shape.len() < 2 {
        return Err(String::from("Both tensors must have at least 2 dimensions for matmul"));
    }
    let a_cols = a.shape.last().unwrap();
    let b_rows = b.shape[b.shape.len().saturating_sub(2)];
    if a_cols != &b_rows {
        return Err(format!(
            "Incompatible shapes for matmul: {:?} x {:?}",
            a.shape, b.shape
        ));
    }
    Ok(a.matmul(b))
}

/// Train a simple MLP on synthetic XOR-like data.
/// Returns a human-readable training report.
pub fn demo_train_mlp(epochs: usize) -> String {
    let mut mlp = MLP::new(
        OptimizerConfig::Adam { learning_rate: 0.01 },
        aether_core::ml::LossConfig::MSE,
    );

    // 2 -> 4 -> 1 network for XOR
    mlp.add_layer(2, 4, Activation::ReLU);
    mlp.add_layer(4, 1, Activation::Sigmoid);

    // Synthetic XOR dataset
    let x = vec![
        Tensor::new(&[0.0, 0.0], &[2]),
        Tensor::new(&[0.0, 1.0], &[2]),
        Tensor::new(&[1.0, 0.0], &[2]),
        Tensor::new(&[1.0, 1.0], &[2]),
    ];
    let y = vec![
        Tensor::new(&[0.0], &[1]),
        Tensor::new(&[1.0], &[1]),
        Tensor::new(&[1.0], &[1]),
        Tensor::new(&[0.0], &[1]),
    ];

    let result = mlp.fit(&x, &y, epochs);

    // Test predictions
    let mut out = format!(
        "MLP Training Report\n\
         ═══════════════════\n\
         Architecture: 2 -> 4 -> 1 (ReLU -> Sigmoid)\n\
         Dataset: XOR (4 samples)\n\
         Optimizer: Adam (lr=0.01)\n\
         Loss: MSE\n\
         Epochs: {}\n\
         Final loss: {:.6}\n\
         \n\
         Predictions:\n",
        epochs, result.final_loss
    );

    for (i, input) in x.iter().enumerate() {
        let pred = mlp.predict(input);
        let val = pred.get(&[0]);
        out.push_str(&format!(
            "  Input [{:.0}, {:.0}] -> Output {:.4} (target: {:.0})\n",
            input.get(&[0]),
            input.get(&[1]),
            val,
            y[i].get(&[0])
        ));
    }

    out
}

/// Format a tensor for display.
pub fn format_tensor(t: &Tensor) -> String {
    if t.shape.len() == 1 {
        let vals: Vec<String> = (0..t.shape[0])
            .map(|i| format!("{:.4}", t.get(&[i])))
            .collect();
        format!("Tensor(shape={:?}) = [{}]", t.shape, vals.join(", "))
    } else if t.shape.len() == 2 {
        let mut out = format!("Tensor(shape={:?})\n", t.shape);
        for r in 0..t.shape[0] {
            let vals: Vec<String> = (0..t.shape[1])
                .map(|c| format!("{:.4}", t.get(&[r, c])))
                .collect();
            out.push_str(&format!("  [{}]\n", vals.join(", ")));
        }
        out
    } else {
        format!("Tensor(shape={:?}) — {} elements", t.shape, t.data.len())
    }
}
