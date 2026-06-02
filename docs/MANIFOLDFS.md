# ManifoldFS Technical Reference

> Source: `kernel/seal-os/src/fs/manifold_fs.rs`, `kernel/seal-os/src/fs/encoder.rs`

---

## Design Philosophy

ManifoldFS layers a geometry-first model over the current block-store path. File content keeps faithful raw bytes for reads/writes and also carries point-cloud metadata on the 2-sphere S^2. File operations use that geometric metadata where it is proven:

- **Store** = encode content to manifold points, assign to Voronoi cell
- **Move (teleport)** = update directory/inode metadata in O(1) without rewriting file bytes on the same filesystem
- **Search (find)** = encode query, locate Voronoi bucket, rank by cosine similarity

The key insight: by projecting file content into a fixed-dimensional geometric space, content-addressable search can narrow to a Voronoi bucket, and same-filesystem file moves can rewire metadata independently of file size.

---

## Encoding Pipeline

Source: `kernel/seal-os/src/fs/encoder.rs`

### Constants

| Constant | Value | Location |
|----------|-------|----------|
| `HASH_DIM` | 128 | line 13 |
| `PROJECTION_DIM` | 3 | line 12 |
| `BLOCK_SIZE` | 4096 bytes | line 14 |
| `MAX_PAYLOAD_POINTS` | 64 | line 11 |
| Projection seed | `0xE951_10A7` | line 134 |
| Union-Find epsilon^2 | 0.25 (epsilon = 0.5) | line 177 |

### Pipeline Steps

**1. Block Segmentation** (lines 135-147)  
Input data is divided into 4096-byte blocks. If there are more blocks than `MAX_PAYLOAD_POINTS` (64), a stride is computed: `stride = num_blocks.div_ceil(MAX_PAYLOAD_POINTS)`.

**2. Trigram Hashing** -- `block_to_hash_vector()` (lines 221-239)  
Each block is converted to a 128-dimensional sparse vector:
- For each trigram at position i: `h = block[i]*31 XOR block[i+1]*37 XOR block[i+2]*41`
- Index: `h % 128`, sign: `+1.0` if `(h / 128) % 2 == 0`, else `-1.0`
- Additionally, per-byte contribution: index `(b * 7 + 3) % 128`, weight `+0.5`

**3. Random Projection** -- `ProjectionMatrix` (lines 46-75)  
A 128x3 Gaussian random matrix is generated from the seed `0xE951_10A7` using Xoshiro256 PRNG (lines 77-118). Box-Muller transform generates Gaussian samples (line 113-117). Each row is scaled by `1 / sqrt(3)`. The 128-dim vector is projected to 3D via matrix-vector multiply.

**4. L2 Normalization** -- `l2_normalize_3d()` (lines 242-248)  
The 3D vector is normalized to unit length (projected onto S^2). If norm < 1e-12, returns `[1.0, 0.0, 0.0]`.

**5. Betti-0 Computation** -- `compute_betti_0()` (lines 173-195)  
Connected components are counted via Union-Find. Two points are connected if their squared Euclidean distance < 0.25 (epsilon = 0.5). Uses path compression with halving (line 199) and union-by-rank (lines 205-219).

**6. Content Hash** -- `hash_bytes()` (lines 250-257)  
FNV-1a hash: init `0xcbf29ce484222325`, for each byte: XOR then multiply by `0x100000001b3`.

### Output: ManifoldPayload

```
struct ManifoldPayload {
    points: Vec<SpherePoint>,    // Up to 64 points on S^2
    point_count: usize,
    betti_0: usize,              // Connected components count
    original_size: u64,
    content_hash: u64,           // FNV-1a
}
```
(Source: `encoder.rs` lines 38-44)

---

## Inode Structure

Source: `manifold_fs.rs` lines 29-47

```
struct Inode {
    id: u64,                     // Unique inode number
    name: String,                // Filename
    kind: InodeKind,             // File or Directory
    payload: ManifoldPayload,    // Encoded content on S^2
    metadata: InodeMetadata {
        created_ms: u64,         // Timestamp (from PIT ticks)
        modified_ms: u64,
        original_size: u64,      // Original byte count
        permissions: u16,        // Unix-style (0o644 for files, 0o755 for dirs)
    },
    voronoi_cell: usize,         // Assigned Voronoi cell index (0..7)
    cluster_id: i32,             // Current cluster assignment
    parent: u64,                 // Parent directory inode ID
}
```

The filesystem uses `BTreeMap<u64, Inode>` for inode storage and `BTreeMap<u64, BTreeMap<String, u64>>` for directory entries (lines 57-58).

---

## Metadata Teleport Mechanism

Source: `manifold_fs.rs` lines 187-255, function `teleport()`

Traditional filesystems use metadata-only rename on the same filesystem and copy+delete across filesystems. The fair comparison for the current Seal marker is same-filesystem metadata movement.

ManifoldFS teleport has an O(1) metadata core:

1. **Record start time**: `t0 = interrupts::ticks()` (line 193)
2. **Lookup inode ID**: `dir_entries[src_dir].get(name)` -- BTreeMap lookup (line 195)
3. **Pre-governor tick**: `governor_tick(0.0)` -- T4 adaptation (line 209)
4. **Remove from source**: `dir_entries[src_dir].remove(name)` -- pointer removal (line 211)
5. **Insert into destination**: `dir_entries[dst_dir].insert(name, inode_id)` -- pointer insertion (line 212)
6. **Update inode parent**: `inode.parent = dst_dir` (line 218)
7. **Update modified timestamp**: `inode.metadata.modified_ms = now_ms()` (line 219)
8. **Post-governor tick**: `governor_tick(1.0)` -- T4 adaptation (line 221)
9. **Log event**: Tagged `"T1/TSS+T4/AGCR"` (line 234)
10. **Entropy check**: `check_entropy_and_merge()` -- T3/GMC rebalancing (line 242)

The current boot benchmark runs this surgery through the persistent mock block-store path (`fs_mode=mock_block`). It proves same inode, source removal, destination presence, bounded metadata operations, and `persistence_bytes_per_move=0` for that same-filesystem move. It does not prove AHCI device latency, cross-filesystem moves, or all VM backends.

---

## Content-Addressable Search

Source: `manifold_fs.rs` lines 257-279, function `find()`

1. Encode query string as ManifoldPayload: `encoder::encode_text(query)` (line 259)
2. Assign query to Voronoi cell: `assign_voronoi_cell(&query_payload)` (line 260) -- T1/TSS
3. Update prefetch prediction state via T2/SCM: `update_prefetch_state(&query_payload)` (line 262)
4. Scan only files in the matching cell: iterate `self.cell_files[cell]` (line 265)
5. Compute similarity via dot product of first sphere points: `payload_similarity()` (lines 549-556) computes `pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2]` (cosine similarity on unit vectors)
6. Sort results by descending similarity (line 277)

Search is narrowed to a single Voronoi cell (1/8 of all files by default), then ranked by geometric similarity.

---

## Theorem Integration Table

| Operation | Theorem(s) | Function | What It Does |
|-----------|-----------|----------|-------------|
| `store()` | T1/TSS, T4/AGCR | `assign_voronoi_cell()`, `governor_tick()` | Places file in nearest Voronoi cell; adapts governor |
| `teleport()` | T1/TSS, T4/AGCR, T3/GMC | `governor_tick()`, `check_entropy_and_merge()` | O(1) metadata rewiring on the same filesystem; mock block-store gate requires `persistence_bytes_per_move=0` |
| `find()` | T1/TSS, T2/SCM | `assign_voronoi_cell()`, `update_prefetch_state()` | Cell-scoped search; SCM predicts next access |
| `mkdir()` | T5/HCS | `update_hyperbolic_ratio()` | Tracks hyperbolic capacity growth with depth |
| prefetch | T2/SCM | `scm.apply()` | Predicts next file access from access history |
| entropy merge | T3/GMC | `check_entropy_and_merge()` | Merges smallest clusters when entropy > 2.0 |
| governor | T4/AGCR | `governor.adapt()` | PD controller adjusts epsilon threshold |

---

## Comparison with Traditional Filesystems

| Operation | ext4 / NTFS | ManifoldFS |
|-----------|-------------|------------|
| File move (same filesystem) | Rename: O(1) metadata update | O(1) metadata rewiring; mock block-store gate requires `persistence_bytes_per_move=0` |
| File move (cross filesystem) | Copy+delete: O(file bytes) | Not covered by the teleport marker |
| Search by content | Full scan: O(N * n) | Voronoi bucket scan plus geometric ranking |
| Content deduplication | External tools (hash comparison) | Built-in via content_hash (FNV-1a) on ManifoldPayload |
| Predictive prefetch | OS page cache heuristics | T2/SCM spectral contraction on access state |
| Adaptive scheduling | None (fixed I/O scheduler) | T4/AGCR governor adapts to workload |

Key architectural difference: ManifoldFS is geometry-first even though the persistent path uses a block-store backend. The ManifoldPayload drives Voronoi locality while raw bytes remain available for faithful reads and writes. The Voronoi index with K=8 cells (`VORONOI_CELLS` constant, line 20) provides spatial locality. The 8 centroids are evenly distributed at theta = pi/2, phi = i * 2*pi/8 (lines 129-137).
