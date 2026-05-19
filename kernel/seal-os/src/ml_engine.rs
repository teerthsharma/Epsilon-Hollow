// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ML Engine — wraps aether-core tensor and neural operations for kernel runtime.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use aether_core::ml::{
    Activation, DenseLayer, MLP, OptimizerConfig, Tensor,
};
use aether_core::ml::linalg::LossConfig;

/// Status of the ML runtime.
pub struct MlStatus {
    pub tensor_ops_available: bool,
    pub neural_net_available: bool,
    pub avx2_detected: bool,
    pub avx512_detected: bool,
    pub gpu_detected: Option<(String, u16, u16)>,
}

impl MlStatus {
    pub fn detect() -> Self {
        Self {
            tensor_ops_available: true,
            neural_net_available: true,
            avx2_detected: detect_avx2(),
            avx512_detected: detect_avx512(),
            gpu_detected: detect_gpu(),
        }
    }
}

/// Detect AVX2 support via CPUID.
fn detect_avx2() -> bool {
    unsafe {
        let leaf1 = core::arch::x86_64::__cpuid(1);
        let avx_available = (leaf1.ecx & (1 << 28)) != 0;
        let osxsave = (leaf1.ecx & (1 << 27)) != 0;
        if !avx_available || !osxsave {
            return false;
        }
        let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
        (leaf7.ebx & (1 << 5)) != 0
    }
}

/// Detect AVX-512 support via CPUID.
fn detect_avx512() -> bool {
    unsafe {
        let leaf1 = core::arch::x86_64::__cpuid(1);
        if (leaf1.ecx & (1 << 28)) == 0 {
            return false;
        }
        let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
        (leaf7.ebx & (1 << 16)) != 0
    }
}

/// Probe PCI for GPU presence.
fn detect_gpu() -> Option<(String, u16, u16)> {
    let devices = crate::drivers::pci::enumerate();
    for dev in &devices {
        if dev.class == 0x03 {
            let vendor_name = match dev.vendor_id {
                0x10DE => "NVIDIA",
                0x1002 => "AMD",
                0x8086 => "Intel",
                0x1AF4 => "VirtIO",
                _ => "Unknown",
            };
            return Some((
                format!("{} {:04X}:{:04X}", vendor_name, dev.vendor_id, dev.device_id),
                dev.vendor_id,
                dev.device_id,
            ));
        }
    }
    None
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
/// Returns (human-readable report, serialized model bytes).
pub fn demo_train_mlp(epochs: usize) -> (String, Vec<u8>) {
    let mut mlp = MLP::new(
        OptimizerConfig::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        LossConfig::MSE,
    );

    // 2 -> 4 -> 1 network for XOR
    mlp.add_layer(2, 4, Activation::ReLU, None);
    mlp.add_layer(4, 1, Activation::Sigmoid, None);

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

    let bytes = serialize_mlp(&mlp);
    (out, bytes)
}

/// Format a tensor for display.
pub fn format_tensor(t: &Tensor) -> String {
    let data_len = t.data.lock().len();
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
        format!("Tensor(shape={:?}) — {} elements", t.shape, data_len)
    }
}

// ── Model Serialization ─────────────────────────────────────────────────────

fn push_u32(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

fn push_u8(buf: &mut Vec<u8>, val: u8) {
    buf.push(val);
}

fn push_f64(buf: &mut Vec<u8>, val: f64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Option<u32> {
    if *offset + 4 > bytes.len() {
        return None;
    }
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&bytes[*offset..*offset + 4]);
    *offset += 4;
    Some(u32::from_le_bytes(arr))
}

fn read_u8(bytes: &[u8], offset: &mut usize) -> Option<u8> {
    if *offset >= bytes.len() {
        return None;
    }
    let val = bytes[*offset];
    *offset += 1;
    Some(val)
}

fn read_f64(bytes: &[u8], offset: &mut usize) -> Option<f64> {
    if *offset + 8 > bytes.len() {
        return None;
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[*offset..*offset + 8]);
    *offset += 8;
    Some(f64::from_le_bytes(arr))
}

fn activation_to_u8(a: Activation) -> u8 {
    match a {
        Activation::ReLU => 0,
        Activation::Sigmoid => 1,
        Activation::Tanh => 2,
        Activation::Linear => 3,
        Activation::LeakyReLU => 4,
        Activation::Softmax => 5,
    }
}

fn activation_from_u8(v: u8) -> Option<Activation> {
    match v {
        0 => Some(Activation::ReLU),
        1 => Some(Activation::Sigmoid),
        2 => Some(Activation::Tanh),
        3 => Some(Activation::Linear),
        4 => Some(Activation::LeakyReLU),
        5 => Some(Activation::Softmax),
        _ => None,
    }
}

/// Serialize an MLP to bytes.
pub fn serialize_mlp(mlp: &MLP) -> Vec<u8> {
    let mut buf = Vec::new();
    // Magic
    buf.extend_from_slice(b"SEALML01");
    // Layer count
    push_u32(&mut buf, mlp.layers.len() as u32);
    for layer in &mlp.layers {
        push_u32(&mut buf, layer.input_size as u32);
        push_u32(&mut buf, layer.output_size as u32);
        push_u8(&mut buf, activation_to_u8(layer.activation));
        // Weights
        let w_data = layer.weights.data.lock();
        push_u32(&mut buf, w_data.len() as u32);
        for &v in w_data.iter() {
            push_f64(&mut buf, v);
        }
        // Biases
        let b_data = layer.biases.data.lock();
        push_u32(&mut buf, b_data.len() as u32);
        for &v in b_data.iter() {
            push_f64(&mut buf, v);
        }
    }
    buf
}

/// Deserialize an MLP from bytes.
pub fn deserialize_mlp(bytes: &[u8]) -> Result<MLP, String> {
    if &bytes[..8] != b"SEALML01" {
        return Err(String::from("Invalid model magic bytes"));
    }
    let mut off = 8;
    let n_layers = read_u32(bytes, &mut off).ok_or("Missing layer count")? as usize;

    let mut mlp = MLP::new(
        OptimizerConfig::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        LossConfig::MSE,
    );

    for _ in 0..n_layers {
        let input_size = read_u32(bytes, &mut off).ok_or("Missing input_size")? as usize;
        let output_size = read_u32(bytes, &mut off).ok_or("Missing output_size")? as usize;
        let act_u8 = read_u8(bytes, &mut off).ok_or("Missing activation")?;
        let activation = activation_from_u8(act_u8).ok_or("Invalid activation")?;

        // Weights
        let w_len = read_u32(bytes, &mut off).ok_or("Missing weights len")? as usize;
        let mut w_data = Vec::with_capacity(w_len);
        for _ in 0..w_len {
            w_data.push(read_f64(bytes, &mut off).ok_or("Missing weight")?);
        }
        // Biases
        let b_len = read_u32(bytes, &mut off).ok_or("Missing biases len")? as usize;
        let mut b_data = Vec::with_capacity(b_len);
        for _ in 0..b_len {
            b_data.push(read_f64(bytes, &mut off).ok_or("Missing bias")?);
        }

        let mut layer = DenseLayer::new(input_size, output_size, activation, None);
        layer.weights = Tensor::from_vec(w_data, vec![output_size, input_size]);
        layer.biases = Tensor::from_vec(b_data, vec![output_size, 1]);
        // Initialize optimizer state
        layer.init_optimizer(&OptimizerConfig::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        });
        mlp.layers.push(layer);
    }

    Ok(mlp)
}

/// Save model bytes to ManifoldFS.
pub fn save_model_bytes(name: &str, bytes: &[u8]) -> Result<String, String> {
    let mut fs = crate::fs::manifold_fs::ManifoldFS::new();
    let root = 0u64;
    fs.store(name, bytes, root)
        .map(|_| format!("Model '{}' saved ({} bytes)", name, bytes.len()))
        .map_err(|e| format!("Save failed: {:?}", e))
}

/// Load an MLP from ManifoldFS.
pub fn load_model(name: &str) -> Result<MLP, String> {
    let mut fs = crate::fs::manifold_fs::ManifoldFS::new();
    let root = 0u64;
    let inode_id = fs.resolve_path_from(name, root)
        .map_err(|_| format!("Model '{}' not found", name))?;
    let inode = fs.inode(inode_id).ok_or("Inode missing")?;
    let bytes = &inode.data;
    deserialize_mlp(bytes)
}

// ── Simple Markov Text Generator ────────────────────────────────────────────

use alloc::collections::BTreeMap;

/// A simple character-level Markov chain for text generation.
pub struct MarkovChain {
    order: usize,
    transitions: BTreeMap<String, BTreeMap<char, usize>>,
    total_counts: BTreeMap<String, usize>,
}

impl MarkovChain {
    pub fn new(order: usize) -> Self {
        Self {
            order,
            transitions: BTreeMap::new(),
            total_counts: BTreeMap::new(),
        }
    }

    /// Train on a text corpus.
    pub fn train(&mut self, text: &str) {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() <= self.order {
            return;
        }
        for i in 0..chars.len() - self.order {
            let key: String = chars[i..i + self.order].iter().collect();
            let next = chars[i + self.order];
            *self.transitions.entry(key.clone()).or_default().entry(next).or_insert(0) += 1;
            *self.total_counts.entry(key).or_insert(0) += 1;
        }
    }

    /// Generate text of given length from a seed.
    pub fn generate(&self, seed: &str, length: usize) -> String {
        let mut result = String::from(seed);
        for _ in 0..length {
            let window = if result.len() >= self.order {
                &result[result.len() - self.order..]
            } else {
                &result
            };
            match self.sample_next(window) {
                Some(ch) => result.push(ch),
                None => break,
            }
        }
        result
    }

    fn sample_next(&self, key: &str) -> Option<char> {
        let counts = self.transitions.get(key)?;
        let total = self.total_counts.get(key)?;
        // Deterministic: pick the most likely next char
        let mut best_char = ' ';
        let mut best_count = 0;
        for (&ch, &count) in counts.iter() {
            if count > best_count {
                best_count = count;
                best_char = ch;
            }
        }
        Some(best_char)
    }
}

/// Built-in corpus for Markov training.
const DEFAULT_CORPUS: &str =
    "Seal OS is the geometrical operating system. \
     All data is geometry on the unit sphere. \
     File moves are O(1) topological surgery. \
     The governor controls epsilon with a PID controller. \
     Voronoi cells partition tasks across CPUs. \
     ManifoldFS stores files as point clouds. \
     Aether-Lang is the language of topology. \
     The scheduler uses work stealing across cells. \
     Teleportation is instant and lossless. \
     Seal OS runs on bare metal with no libc.";

/// Train a Markov chain and generate text.
pub fn demo_generate_text(seed: &str, length: usize) -> String {
    let mut chain = MarkovChain::new(3);
    chain.train(DEFAULT_CORPUS);
    chain.generate(seed, length)
}
