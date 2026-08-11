// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ML Engine — wraps aether-core tensor and neural operations for kernel runtime.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use aether_core::ml::linalg::LossConfig;
use aether_core::ml::{Activation, DenseLayer, OptimizerConfig, Tensor, MLP};

pub mod foliation;
pub mod stratum;
pub mod tensor_viz;
pub mod topo_asm;

/// Status of the ML runtime.
pub struct MlStatus {
    pub tensor_ops_available: bool,
    pub neural_net_available: bool,
    pub xsave_detected: bool,
    pub avx2_detected: bool,
    pub avx512_detected: bool,
    pub gpu_detected: Option<(String, u16, u16)>,
}

impl MlStatus {
    pub fn detect() -> Self {
        let xsave = detect_xsave();
        Self {
            tensor_ops_available: true,
            neural_net_available: true,
            xsave_detected: xsave,
            avx2_detected: xsave && detect_avx2(),
            avx512_detected: xsave && detect_avx512(),
            gpu_detected: detect_gpu(),
        }
    }
}

/// Returns true if the CPU supports XSAVE and OSXSAVE.
pub fn xsave_available() -> bool {
    detect_xsave()
}

/// Detect XSAVE support via CPUID.
fn detect_xsave() -> bool {
    let leaf1 = core::arch::x86_64::__cpuid(1);
    let has_xsave = (leaf1.ecx & (1 << 26)) != 0;
    let osxsave = (leaf1.ecx & (1 << 27)) != 0;
    has_xsave && osxsave
}

/// Detect AVX2 support via CPUID.
fn detect_avx2() -> bool {
    let leaf1 = core::arch::x86_64::__cpuid(1);
    let avx_available = (leaf1.ecx & (1 << 28)) != 0;
    let osxsave = (leaf1.ecx & (1 << 27)) != 0;
    if !avx_available || !osxsave {
        return false;
    }
    let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
    (leaf7.ebx & (1 << 5)) != 0
}

/// Detect AVX-512 support via CPUID.
fn detect_avx512() -> bool {
    let leaf1 = core::arch::x86_64::__cpuid(1);
    let avx_available = (leaf1.ecx & (1 << 28)) != 0;
    if !avx_available {
        return false;
    }
    let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
    (leaf7.ebx & (1 << 16)) != 0
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
                format!(
                    "{} {:04X}:{:04X}",
                    vendor_name, dev.vendor_id, dev.device_id
                ),
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
///
/// `Tensor::matmul` handles rank exactly 2 and `assert_eq!`s on anything
/// else, so this rank check must be exact rather than a lower bound: the
/// kernel builds with `panic = "abort"`, which makes that assertion a machine
/// stop instead of an error the shell can print. Accepting rank 3 here and
/// letting it reach `matmul` is not a laxer contract, it is a halt.
/// `seal-graph`'s `matmul_shape` guards the same call the same way.
pub fn tensor_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(format!(
            "Matmul requires 2D tensors, got {:?} x {:?}",
            a.shape, b.shape
        ));
    }
    if a.shape[1] != b.shape[0] {
        return Err(format!(
            "Incompatible shapes for matmul: {:?} x {:?}",
            a.shape, b.shape
        ));
    }
    Ok(a.matmul(b))
}

/// Train a simple MLP on synthetic XOR-like data.
/// Returns (human-readable report, serialized model bytes).
///
/// Samples are column vectors — `[features, 1]`, not `[features]`.
/// `DenseLayer::forward` computes `weights.matmul(input)` with weights shaped
/// `[out, in]`, and `Tensor::matmul` `assert_eq!`s that both operands are rank
/// 2. A rank-1 sample therefore does not train badly, it aborts the kernel:
/// the crate builds with `panic = "abort"`. aether-core's own `xor_problem`
/// fixture uses the same `[2, 1]` / `[1, 1]` pair. Every read below indexes
/// with both axes for the same reason — `Tensor::compute_offset` asserts that
/// the index rank matches the shape rank.
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
        Tensor::new(&[0.0, 0.0], &[2, 1]),
        Tensor::new(&[0.0, 1.0], &[2, 1]),
        Tensor::new(&[1.0, 0.0], &[2, 1]),
        Tensor::new(&[1.0, 1.0], &[2, 1]),
    ];
    let y = vec![
        Tensor::new(&[0.0], &[1, 1]),
        Tensor::new(&[1.0], &[1, 1]),
        Tensor::new(&[1.0], &[1, 1]),
        Tensor::new(&[0.0], &[1, 1]),
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
        let val = pred.get(&[0, 0]);
        out.push_str(&format!(
            "  Input [{:.0}, {:.0}] -> Output {:.4} (target: {:.0})\n",
            input.get(&[0, 0]),
            input.get(&[1, 0]),
            val,
            y[i].get(&[0, 0])
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
///
/// Every count read out of the file (layer count, weight length, bias
/// length) is untrusted — it comes straight from whatever `ml load <name>`
/// read off disk. Each one is checked against the bytes actually remaining
/// in `bytes` *before* it sizes a `Vec::with_capacity` or is used to index,
/// mirroring the `at()`-style bounds checks in `atlas/relobj.rs`. This crate
/// builds with `panic = "abort"` and defines no `#[alloc_error_handler]`, so
/// an unchecked huge allocation or an out-of-bounds index would halt the
/// kernel rather than just fail the load.
pub fn deserialize_mlp(bytes: &[u8]) -> Result<MLP, String> {
    if bytes.len() < 8 || &bytes[..8] != b"SEALML01" {
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

        // Weights. `w_len` is a raw u32 off the file (up to ~4e9); each
        // element is an 8-byte f64, so the buffer itself gives a hard,
        // non-invented ceiling: it cannot hold more than
        // `(bytes.len() - off) / 8` of them. Reject before `with_capacity`
        // ever sees the untrusted count.
        let w_len = read_u32(bytes, &mut off).ok_or("Missing weights len")? as usize;
        if w_len > bytes.len().saturating_sub(off) / 8 {
            return Err(String::from("Weight count exceeds remaining model bytes"));
        }
        // The declared shape must exactly account for the weight buffer.
        // This is also what `Tensor::from_vec` asserts internally (it
        // panics — not `Result` — on a mismatch), and, since `w_len` is now
        // bounded above, it caps `input_size * output_size` before
        // `DenseLayer::new` does its own Xavier-init allocation of that size.
        if input_size.checked_mul(output_size) != Some(w_len) {
            return Err(String::from("Weight length does not match layer shape"));
        }
        let mut w_data = Vec::with_capacity(w_len);
        for _ in 0..w_len {
            w_data.push(read_f64(bytes, &mut off).ok_or("Missing weight")?);
        }
        // Biases — same buffer-derived cap, and count must equal output_size
        // (bias tensor shape is [output_size, 1]) for the same reason.
        let b_len = read_u32(bytes, &mut off).ok_or("Missing biases len")? as usize;
        if b_len > bytes.len().saturating_sub(off) / 8 {
            return Err(String::from("Bias count exceeds remaining model bytes"));
        }
        if b_len != output_size {
            return Err(String::from("Bias length does not match layer shape"));
        }
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
    let fs = crate::fs::manifold_fs::ManifoldFS::new();
    let root = 0u64;
    let inode_id = fs
        .resolve_path_from(name, root)
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
            *self
                .transitions
                .entry(key.clone())
                .or_default()
                .entry(next)
                .or_insert(0) += 1;
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
        let _total = self.total_counts.get(key)?;
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
const DEFAULT_CORPUS: &str = "Seal OS is the geometrical operating system. \
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

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// Defect A: the old code did `&bytes[..8]` with no length check first.
    /// A 7-byte buffer indexed that way panics (abort, under `panic = "abort"`).
    fn test_truncated_buffer_rejected() -> TestResult {
        let bytes = [0x53u8, 0x45, 0x41, 0x4C, 0x4D, 0x4C, 0x30]; // "SEALML0", 7 bytes
        test_assert!(bytes.len() < 8);
        test_assert!(
            deserialize_mlp(&bytes).is_err(),
            "a buffer shorter than the magic must be rejected, not panic"
        );
        TestResult::Pass
    }

    fn test_empty_buffer_rejected() -> TestResult {
        test_assert!(
            deserialize_mlp(&[]).is_err(),
            "an empty buffer must be rejected, not panic"
        );
        TestResult::Pass
    }

    fn test_bad_magic_rejected() -> TestResult {
        let bytes = *b"NOTSEAL!";
        test_assert!(deserialize_mlp(&bytes).is_err(), "wrong magic must be rejected");
        TestResult::Pass
    }

    /// Defect B: `w_len` came straight off the file into
    /// `Vec::with_capacity(w_len)` with no check against the buffer. A
    /// header claiming ~4e9 weights with zero bytes left to back them must
    /// be rejected before that allocation is attempted.
    fn test_oversized_weight_len_rejected() -> TestResult {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"SEALML01");
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_layers
        bytes.extend_from_slice(&2u32.to_le_bytes()); // input_size
        bytes.extend_from_slice(&2u32.to_le_bytes()); // output_size
        bytes.push(0); // activation = ReLU
        bytes.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // w_len ~ 4e9, no data follows
        test_assert!(
            deserialize_mlp(&bytes).is_err(),
            "a weight length the buffer cannot hold must be rejected before allocating"
        );
        TestResult::Pass
    }

    /// Same class as Defect B, on the bias length.
    fn test_oversized_bias_len_rejected() -> TestResult {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"SEALML01");
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_layers
        bytes.extend_from_slice(&1u32.to_le_bytes()); // input_size
        bytes.extend_from_slice(&1u32.to_le_bytes()); // output_size
        bytes.push(0); // activation = ReLU
        bytes.extend_from_slice(&1u32.to_le_bytes()); // w_len = 1, matches shape
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // b_len ~ 4e9, no data follows
        test_assert!(
            deserialize_mlp(&bytes).is_err(),
            "a bias length the buffer cannot hold must be rejected before allocating"
        );
        TestResult::Pass
    }

    /// Third site: `w_len` fitting the buffer is not enough — it must also
    /// match `input_size * output_size`, or `Tensor::from_vec`'s internal
    /// `assert_eq!` panics instead of this function returning `Err`.
    fn test_mismatched_shape_rejected() -> TestResult {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"SEALML01");
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_layers
        bytes.extend_from_slice(&2u32.to_le_bytes()); // input_size
        bytes.extend_from_slice(&2u32.to_le_bytes()); // output_size (needs w_len == 4)
        bytes.push(0); // activation = ReLU
        bytes.extend_from_slice(&1u32.to_le_bytes()); // w_len = 1, fits buffer but wrong shape
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        test_assert!(
            deserialize_mlp(&bytes).is_err(),
            "a weight length that fits the buffer but not the declared shape must be rejected"
        );
        TestResult::Pass
    }

    /// `Tensor::matmul` handles rank exactly 2 and `assert_eq!`s otherwise.
    /// The pre-fix guard here only demanded rank >= 2, so a rank-3 pair
    /// cleared it and reached that assertion — which, under `panic = "abort"`,
    /// halts the machine instead of returning the `Err` the shell prints.
    /// Both directions are checked: a guard that refused everything would
    /// also stop the abort and would also be wrong.
    fn test_matmul_rejects_non_2d() -> TestResult {
        let cube = match tensor_from_data(vec![1.0; 8], vec![2, 2, 2]) {
            Ok(t) => t,
            Err(_) => return TestResult::Fail("rank-3 tensor must be constructible"),
        };
        let vector = match tensor_from_data(vec![1.0, 2.0], vec![2]) {
            Ok(t) => t,
            Err(_) => return TestResult::Fail("rank-1 tensor must be constructible"),
        };
        let square = match tensor_from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]) {
            Ok(t) => t,
            Err(_) => return TestResult::Fail("2x2 tensor must be constructible"),
        };

        test_assert!(
            tensor_matmul(&cube, &square).is_err(),
            "a rank-3 left operand must be rejected, not handed to matmul's assertion"
        );
        test_assert!(
            tensor_matmul(&square, &cube).is_err(),
            "a rank-3 right operand must be rejected, not handed to matmul's assertion"
        );
        test_assert!(
            tensor_matmul(&vector, &square).is_err(),
            "a rank-1 left operand must be rejected, not handed to matmul's assertion"
        );

        // The shell's `ml matmul` still has to compute. [[1,2],[3,4]] squared
        // is [[7,10],[15,22]]; both values are exact in f64.
        let product = match tensor_matmul(&square, &square) {
            Ok(t) => t,
            Err(_) => return TestResult::Fail("a 2x2 by 2x2 multiply must still be accepted"),
        };
        test_assert_eq!(product.shape, vec![2, 2]);
        test_assert_eq!(product.get(&[0, 0]), 7.0);
        test_assert_eq!(product.get(&[1, 1]), 22.0);
        TestResult::Pass
    }

    /// The fix must fail closed on garbage without breaking a real model.
    /// Reaches `demo_train_mlp`, so it is also the case that the rank-1
    /// training samples aborted in `Tensor::matmul` before the fix.
    fn test_valid_roundtrip_still_loads() -> TestResult {
        let (_, bytes) = demo_train_mlp(1);
        let mlp = deserialize_mlp(&bytes);
        test_assert!(
            mlp.is_ok(),
            "a genuinely serialized model must still deserialize"
        );
        test_assert_eq!(mlp.unwrap().layers.len(), 2);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "ml_engine::deserialize_truncated_buffer_rejected",
            test_truncated_buffer_rejected,
        );
        crate::testing::register_test(
            "ml_engine::deserialize_empty_buffer_rejected",
            test_empty_buffer_rejected,
        );
        crate::testing::register_test(
            "ml_engine::deserialize_bad_magic_rejected",
            test_bad_magic_rejected,
        );
        crate::testing::register_test(
            "ml_engine::deserialize_oversized_weight_len_rejected",
            test_oversized_weight_len_rejected,
        );
        crate::testing::register_test(
            "ml_engine::deserialize_oversized_bias_len_rejected",
            test_oversized_bias_len_rejected,
        );
        crate::testing::register_test(
            "ml_engine::deserialize_mismatched_shape_rejected",
            test_mismatched_shape_rejected,
        );
        crate::testing::register_test(
            "ml_engine::matmul_rejects_non_2d",
            test_matmul_rejects_non_2d,
        );
        crate::testing::register_test(
            "ml_engine::deserialize_valid_roundtrip_still_loads",
            test_valid_roundtrip_still_loads,
        );
    }
}
