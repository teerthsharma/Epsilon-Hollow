# Plan: Lypnos Guard Tensor-Topological Rendering Engine + TopCrypt File System

## Version
Seal OS 0.4.5 → Lypnos Guard Edition

## Philosophy
Files are illusions. Bytes are flat. Reality is curved. In Seal OS:
- **All files are topological manifolds** — stored as point clouds on S²
- **Ctrl+L = Lypnos Lock** — file dissolves into its native topological sleep state
- **Ctrl+E = Export** — manifold reconstitutes into boring flat bytes for outsiders
- **Ctrl+I = Import** — external file gets absorbed into the manifold
- **3D Tensor Renderer** — tensors render as hyperbolic 3-manifolds, not spreadsheets
- **Trading CSVs become geometry** — profit/loss as curvature, time as hyperbolic depth

---

## Part 1: TopCrypt — Topological File Encryption

### Scientific Basis
A file of N bytes is chunked into blocks of 64 bytes. Each block is interpreted as:
- 32 × u16 values (quantized angles)
- 16 point-pairs = 16 points on the unit sphere S²
- The file becomes a **walk on S²** — a sequence of points
- Without the topological decoder ( Seal OS kernel ), the walk is indistinguishable from random noise
- The decoder uses the inverse: angles → bytes

### Security Model
- **Native storage**: File exists ONLY as manifold embedding in ManifoldFS
- **External sharing**: User presses Ctrl+E → file is "flattened" to normal bytes → saved to FAT/ext2/USB
- **External import**: User presses Ctrl+I → normal bytes → encoded to manifold → stored in ManifoldFS
- **Lock (Ctrl+L)**: File is re-encoded with a one-time topological permutation (T3 entropy shuffle)
- **Unlock**: Reverse permutation using kernel-held seed

### Data Structures
```rust
pub struct TopologicalFile {
    pub name_hash: [u8; 32],      // SHA-256 of filename (for lookup)
    pub block_count: u64,
    pub embedding_seed: u64,       // LCG seed for this file's encoding
    pub blocks: Vec<TopoBlock>,
    pub locked: bool,
    pub lock_entropy: f64,         // T3 Betti-0 / total blocks
}

pub struct TopoBlock {
    pub embedding: [u16; 32],      // 16 points on S²
    pub checksum: u32,             // CRC32 of original bytes
}
```

### Encoding Pipeline
```
Input: Vec<u8> (raw file bytes)
  ↓
Chunk into 64-byte blocks (pad last block with zeros)
  ↓
For each block:
  - Interpret as 32 × u16 little-endian
  - Normalize: u16 → angle [0, 2π) via (val / 65536) * 2π
  - Pair angles: (θ₁, φ₁), (θ₂, φ₂), ... = 16 points on S²
  - Compute CRC32 checksum
  ↓
Store as TopologicalFile in ManifoldFS
```

### Decoding Pipeline
```
Input: TopologicalFile
  ↓
For each TopoBlock:
  - Read 16 (θ, φ) pairs
  - Convert back to u16: val = (θ / 2π) * 65536
  - Reconstruct 64-byte block
  - Verify CRC32
  ↓
Concatenate blocks, strip padding
  ↓
Output: Vec<u8> (original file bytes)
```

### Ctrl+L — Lypnos Lock
- Generate random topological permutation using T1 Voronoi cell re-seeding
- Shuffle block order based on Voronoi cell assignments
- XOR each block embedding with spectral eigenvector (T2)
- Store lock_entropy = Betti-0 of shuffled blocks
- Set `locked = true`
- File is now "asleep" — readable only by Seal OS with the kernel-held seed

### API
```rust
pub fn encode_file(name: &str, data: &[u8]) -> TopologicalFile;
pub fn decode_file(topo: &TopologicalFile) -> Vec<u8>;
pub fn lock_file(topo: &mut TopologicalFile);
pub fn unlock_file(topo: &mut TopologicalFile) -> bool;
pub fn export_file(topo: &TopologicalFile, path: &str); // writes flat bytes to FAT/USB
pub fn import_file(path: &str) -> TopologicalFile;      // reads flat bytes, encodes
```

---

## Part 2: 3D Tensor Renderer — Lypnos Guard Engine

### Scientific Basis
Tensors are multi-dimensional arrays. A tensor T ∈ ℝ^(n×m×k) is visualized as:
1. **Flatten** tensor to point cloud P ∈ ℝ^(nmk×3) via dimensionality reduction
2. **PCA/Spectral embedding** (T2): find top 3 principal components → (x, y, z)
3. **Color mapping** (T1 Voronoi): partition value space into cells, assign colors
4. **Hyperbolic projection** (T5): map Euclidean (x,y,z) to hyperboloid for 3D view
5. **LOD** (T2 spectral): simplify distant points, preserve local structure
6. **Integrity** (T3 Betti): ensure no holes created during simplification
7. **Governor** (T4): adapt mesh density based on frame time

### XZ Mapping
For 2D tensors (matrices, CSVs):
- X axis = row index
- Z axis = column index  
- Y axis (height/color) = cell value
- This creates a **terrain** where trading data becomes a landscape
- Profit = green peaks, Loss = red valleys

For 3D tensors:
- X, Y, Z = first 3 principal components from SVD
- Color = 4th component (alpha/brightness)

### Trading CSV Visualization
```
CSV columns: [timestamp, price, volume, signal, pnl]
  ↓
Parse to matrix M ∈ ℝ^(rows × 5)
  ↓
Normalize each column to [-1, 1]
  ↓
SVD: M = U Σ V^T
  ↓
Project onto top 3 singular vectors: (x, y, z) = U[:,0:3] * Σ[0:3]
  ↓
Color by PnL: green = positive, red = negative
  ↓
Render as 3D point cloud / mesh via topo_render.rs
```

### LLM Integration
- LLM reads the topological representation directly
- Instead of parsing CSV bytes, the LLM traverses the manifold
- Neighboring points on S² = semantically similar trading moments
- Voronoi cells = trading regimes (bull/bear/volatile)
- The LLM "feels" the market curvature

### GPU Acceleration (Software GPU)
Since we don't have real GPU drivers (Intel/NVIDIA stubs):
- Use AVX2/SSE SIMD in the topological renderer for parallel vertex processing
- Batch process 8 vertices at a time with 256-bit vectors
- Use the existing framebuffer back buffer for output
- This is a "software GPU" — CPU pretending to be GPU via SIMD

---

## Part 3: Integration

### File Manager
- Shows files as manifold icons (rotating S² with point cloud)
- Locked files have "sleeping" animation (slow rotation, dim colors)
- Double-click opens tensor viewer for .csv, .tensor, .topo files

### Keyboard Shortcuts
- **Ctrl+L**: Lypnos Lock — selected file dissolves into topology
- **Ctrl+E**: Export — flatten to bytes for external sharing
- **Ctrl+I**: Import — absorb external file into manifold
- **Ctrl+T**: Tensor View — render selected file as 3D manifold

### Shell Commands
```
seal> topcrypt encode secrets.txt
[TopCrypt] Encoded 1024 bytes → 16 blocks on S²

seal> topcrypt lock secrets.topo
[Lypnos Guard] File locked. Betti-0 = 0.42. Sweet dreams.

seal> topcrypt render trades.csv
[Lypnos Guard] Rendering 5000×5 tensor... peaks: 12, valleys: 8

seal> topcrypt export secrets.topo /usb/secrets.txt
[TopCrypt] Flattening manifold → 1024 bytes
```

---

## Implementation Order

1. **Agent A**: TopCrypt encode/decode engine (`fs/topcrypt.rs`)
2. **Agent B**: 3D Tensor Renderer extensions (`graphics/topo_render.rs` + `ml_engine/tensor_viz.rs`)
3. **Agent C**: Integration — keyboard shortcuts, shell commands, file manager icons
