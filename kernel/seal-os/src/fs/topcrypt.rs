// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! TopCrypt — topological file encryption engine for Seal OS Lypnos Guard Edition.
//! Files don't exist as bytes. They exist as geometry on S².

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;
use crate::fs::vfs::with_vfs;

// ─── Global Lock Registry ────────────────────────────────────────────────────

static LOCKED_REGISTRY: Mutex<BTreeMap<String, bool>> = Mutex::new(BTreeMap::new());

pub fn mark_locked(path: &str) {
    LOCKED_REGISTRY.lock().insert(String::from(path), true);
}

pub fn mark_unlocked(path: &str) {
    LOCKED_REGISTRY.lock().remove(path);
}

pub fn is_locked(path: &str) -> bool {
    LOCKED_REGISTRY.lock().get(path).copied().unwrap_or(false)
}

/// A file represented as a walk on S².
#[derive(Debug, Clone)]
pub struct TopologicalFile {
    pub name_hash: [u8; 32],
    pub block_count: u64,
    pub embedding_seed: u64,
    pub blocks: Vec<TopoBlock>,
    pub locked: bool,
    pub lock_entropy: f64,
}

/// One 64-byte chunk encoded as 16 points on S² (32 quantized angles).
#[derive(Debug, Clone, Copy)]
pub struct TopoBlock {
    pub embedding: [u16; 32], // 16 (θ,φ) pairs
    pub checksum: u32,        // CRC32 of original 64 bytes
}

// ─── CRC32 ───────────────────────────────────────────────────────────────────

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB88320 } else { crc >> 1 };
        }
    }
    !crc
}

// ─── Angle Conversion ────────────────────────────────────────────────────────

fn bytes_to_angles(bytes: &[u8; 64]) -> [u16; 32] {
    let mut out = [0u16; 32];
    for i in 0..32 {
        out[i] = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
    }
    out
}

fn angles_to_bytes(angles: &[u16; 32]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for i in 0..32 {
        let b = angles[i].to_le_bytes();
        out[i * 2] = b[0];
        out[i * 2 + 1] = b[1];
    }
    out
}

// ─── LCG PRNG ────────────────────────────────────────────────────────────────

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    const A: u64 = 6364136223846793005;
    const C: u64 = 1442695040888963407;

    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(Self::A).wrapping_add(Self::C);
        self.state
    }
}

// ─── Mask Derivation ─────────────────────────────────────────────────────────

fn derive_mask(key: u64) -> [u16; 32] {
    let mut rng = Lcg64::new(key);
    let mut mask = [0u16; 32];
    for i in 0..32 {
        mask[i] = rng.next() as u16;
    }
    mask
}

// ─── Entropy ─────────────────────────────────────────────────────────────────

fn compute_entropy(topo: &TopologicalFile) -> f64 {
    let mut counts = [0u32; 256];
    let total = topo.blocks.len() * 64;
    if total == 0 {
        return 0.0;
    }
    for block in &topo.blocks {
        for &angle in &block.embedding {
            let [b0, b1] = angle.to_le_bytes();
            counts[b0 as usize] += 1;
            counts[b1 as usize] += 1;
        }
    }
    let mut entropy = 0.0;
    for &count in &counts {
        if count == 0 {
            continue;
        }
        let p = count as f64 / total as f64;
        entropy -= p * libm::log2(p);
    }
    entropy
}

// ─── Shuffling ───────────────────────────────────────────────────────────────

fn generate_permutation(len: usize, seed: u64) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..len).collect();
    let mut rng = Lcg64::new(seed);
    for i in (1..len).rev() {
        let j = (rng.next() as usize) % (i + 1);
        perm.swap(i, j);
    }
    perm
}

fn shuffle_blocks(blocks: &mut [TopoBlock], seed: u64) {
    let perm = generate_permutation(blocks.len(), seed);
    let mut shuffled = Vec::with_capacity(blocks.len());
    for &i in &perm {
        shuffled.push(blocks[i].clone());
    }
    blocks.copy_from_slice(&shuffled);
}

fn unshuffle_blocks(blocks: &mut [TopoBlock], seed: u64) {
    let perm = generate_permutation(blocks.len(), seed);
    let mut unshuffled = blocks.to_vec();
    for (new_pos, &old_pos) in perm.iter().enumerate() {
        unshuffled[old_pos] = blocks[new_pos].clone();
    }
    blocks.copy_from_slice(&unshuffled);
}

// ─── Core API ────────────────────────────────────────────────────────────────

/// Encode raw bytes into a topological file.
pub fn encode_bytes(data: &[u8], seed: u64) -> TopologicalFile {
    let len = data.len();
    let block_count = (len + 63) / 64;
    let mut blocks = Vec::with_capacity(block_count);
    for i in 0..block_count {
        let start = i * 64;
        let end = (start + 64).min(len);
        let mut chunk = [0u8; 64];
        chunk[..end - start].copy_from_slice(&data[start..end]);
        let angles = bytes_to_angles(&chunk);
        let checksum = crc32(&chunk);
        blocks.push(TopoBlock { embedding: angles, checksum });
    }
    TopologicalFile {
        name_hash: [0u8; 32],
        block_count: block_count as u64,
        embedding_seed: seed,
        blocks,
        locked: false,
        lock_entropy: 0.0,
    }
}

/// Decode a topological file back to raw bytes.
pub fn decode_bytes(topo: &TopologicalFile) -> Vec<u8> {
    let mut out = Vec::with_capacity(topo.blocks.len() * 64);
    for block in &topo.blocks {
        let bytes = angles_to_bytes(&block.embedding);
        out.extend_from_slice(&bytes);
    }
    out
}

/// Lock a topological file with a key.
pub fn lock_file(topo: &mut TopologicalFile, key: u64) {
    if topo.locked {
        return;
    }
    let seed = topo.embedding_seed ^ key;
    let mask = derive_mask(key);
    for block in &mut topo.blocks {
        for (i, angle) in block.embedding.iter_mut().enumerate() {
            let m = mask[i];
            let [b0, b1] = angle.to_le_bytes();
            let new_bytes = [b0 ^ (m as u8), b1 ^ ((m >> 8) as u8)];
            *angle = u16::from_le_bytes(new_bytes);
        }
    }
    shuffle_blocks(&mut topo.blocks, seed);
    topo.locked = true;
    topo.lock_entropy = compute_entropy(topo);
}

/// Unlock a topological file with a key.
pub fn unlock_file(topo: &mut TopologicalFile, key: u64) -> bool {
    if !topo.locked {
        return true;
    }
    let seed = topo.embedding_seed ^ key;
    unshuffle_blocks(&mut topo.blocks, seed);
    let mask = derive_mask(key);
    for block in &mut topo.blocks {
        for (i, angle) in block.embedding.iter_mut().enumerate() {
            let m = mask[i];
            let [b0, b1] = angle.to_le_bytes();
            let new_bytes = [b0 ^ (m as u8), b1 ^ ((m >> 8) as u8)];
            *angle = u16::from_le_bytes(new_bytes);
        }
    }
    for block in &topo.blocks {
        let bytes = angles_to_bytes(&block.embedding);
        if crc32(&bytes) != block.checksum {
            topo.locked = true;
            return false;
        }
    }
    topo.locked = false;
    topo.lock_entropy = 0.0;
    true
}

/// Serialize a topological file to bytes (`.topo` wire format).
pub fn export_to_bytes(topo: &TopologicalFile) -> Vec<u8> {
    let mut out = Vec::with_capacity(62 + topo.blocks.len() * 68);
    out.extend_from_slice(b"TOPC");
    out.push(1u8); // version
    out.extend_from_slice(&topo.name_hash);
    out.extend_from_slice(&topo.block_count.to_le_bytes());
    out.extend_from_slice(&topo.embedding_seed.to_le_bytes());
    out.push(if topo.locked { 1 } else { 0 });
    out.extend_from_slice(&topo.lock_entropy.to_le_bytes());
    for block in &topo.blocks {
        for &angle in &block.embedding {
            out.extend_from_slice(&angle.to_le_bytes());
        }
        out.extend_from_slice(&block.checksum.to_le_bytes());
    }
    out
}

/// Deserialize a topological file from bytes (`.topo` wire format).
/// The `seed` parameter is retained for API compatibility but ignored
/// because the seed is embedded in the serialized data.
pub fn import_from_bytes(data: &[u8], _seed: u64) -> TopologicalFile {
    assert!(data.len() >= 62, "topo file too small");
    assert_eq!(&data[0..4], b"TOPC", "invalid magic");
    assert_eq!(data[4], 1, "unsupported version");

    let mut name_hash = [0u8; 32];
    name_hash.copy_from_slice(&data[5..37]);

    let block_count = u64::from_le_bytes([
        data[37], data[38], data[39], data[40],
        data[41], data[42], data[43], data[44],
    ]);

    let embedding_seed = u64::from_le_bytes([
        data[45], data[46], data[47], data[48],
        data[49], data[50], data[51], data[52],
    ]);

    let locked = data[53] != 0;

    let lock_entropy = f64::from_le_bytes([
        data[54], data[55], data[56], data[57],
        data[58], data[59], data[60], data[61],
    ]);

    let mut blocks = Vec::with_capacity(block_count as usize);
    let mut offset = 62;
    for _ in 0..block_count {
        let mut embedding = [0u16; 32];
        for i in 0..32 {
            embedding[i] = u16::from_le_bytes([data[offset], data[offset + 1]]);
            offset += 2;
        }
        let checksum = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        ]);
        offset += 4;
        blocks.push(TopoBlock { embedding, checksum });
    }

    TopologicalFile {
        name_hash,
        block_count,
        embedding_seed,
        blocks,
        locked,
        lock_entropy,
    }
}

/// Read a file via VFS and encode it topologically.
pub fn file_to_topological(path: &str) -> Option<TopologicalFile> {
    let buf = with_vfs(|vfs| {
        let handle = vfs.lookup(path).ok()?;
        let node = vfs.stat(handle).ok()?;
        let size = node.size as usize;
        let mut buf = Vec::with_capacity(size);
        let mut offset = 0u64;
        while offset < size as u64 {
            let mut chunk = [0u8; 4096];
            let n = vfs.read(handle, &mut chunk, offset).ok()?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
            offset += n as u64;
        }
        Some(buf)
    })?;
    let seed = crc32(&buf) as u64;
    let mut topo = encode_bytes(&buf, seed);
    topo.name_hash = sha256(path.as_bytes());
    Some(topo)
}

/// Decode a topological file and write it via VFS.
pub fn topological_to_file(topo: &TopologicalFile, path: &str) -> bool {
    let data = decode_bytes(topo);
    with_vfs(|vfs| {
        let handle = vfs.create(path).ok()?;
        vfs.write(handle, &data, 0).ok()?;
        Some(())
    })
    .is_some()
}

// ─── SHA-256 helper ──────────────────────────────────────────────────────────

fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}
