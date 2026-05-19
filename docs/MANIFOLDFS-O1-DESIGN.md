# ManifoldFS O(1) — Implementation Plan

**Date:** 2026-05-19
**Author:** Teerth Sharma
**Status:** Draft for review
**Stance:** The README is the bible. This document describes how the code rises to it.

---

## 1. The spec (verbatim from the bible)

Every claim below is a load-bearing assertion in the project README and must be literally true of the running system. This plan exists to deliver these properties.

| # | README claim | Source |
|---|---|---|
| 1 | "every byte on disk is a point cloud on the unit sphere" | `README.md` §intro |
| 2 | "every file move is O(1) topological surgery" | `README.md` §intro |
| 3 | "All data = geometry on S². File moves = O(1) topological surgery." | `README.md` boot banner |
| 4 | "Layer 4 │ ManifoldFS: THE filesystem — all data = 64 pts on S²" | `kernel/seal-os/README.md` |
| 5 | "Filesystem │ file lookup O(1)" — theorem integration table | `kernel/seal-os/README.md` |
| 6 | "O(1) retrieval via spherical Voronoi tessellation" | `README.md` §Five Theorems |
| 7 | "ManifoldFS: all data = 64 points on S², O(1) teleport via topological surgery, content-addressable via Voronoi" | `README.md` Layer 4 mermaid |
| 8 | "Teleport (O(1) Move)" | `README.md` §ManifoldFS mermaid |
| 9 | "race — benchmark teleport vs copy — teleport: ~50μs constant" | `README.md` §Built-in Applications |
| 10 | "Manifold teleport: 50 us (O(1) topological surgery). Speedup: 20000x. Regardless of original file size!" | `kernel/seal-os/README.md` teleport benchmark |
| 11 | "O(1) teleportation demonstrated" — CI milestone 8 | `README.md` §CI Pipeline |
| 12 | "File move │ O(1) topological surgery" — comparison table | `README.md` §Seal OS vs The World |
| 13 | "Teleportation │ Yes (O(1) file move)" — comparison table | `README.md` §Seal OS vs The World |
| 14 | "O(1) context transfer between agents via topological surgery on hollow S² manifolds" | `README.md` §Epsilon |
| 15 | "[ManifoldFS] Teleported 'hello.txt' (19 bytes) in 1 ticks — O(1)" — expected boot log | `README.md` boot banner; `kernel/seal-os/TESTING.md` |

**Success criterion:** every line above is literally true after this work lands. No asterisks. No "amortized except…". No "in-memory only" footnote. The code, the boot log, the benchmark output, and the comparison table all agree with the bible.

## 2. Gap closure required to honor the bible

The README makes five distinct promises about ManifoldFS. Each requires one piece of new work.

### 2.1 "Every byte on disk is a point cloud on the unit sphere"

To make claim 1 literal, ManifoldFS must persist to a real block device, and that persistence must place the S² point cloud at the head of every data extent. The word "disk" in the bible forecloses a pure-RAM implementation.

**Work:** Wire `kernel/seal-os/src/drivers/block/ahci.rs` into ManifoldFS via a new `BlockStore` layer that writes payload-first extents.

### 2.2 "O(1) topological surgery" for file moves

The README states O(1) — not amortized except…, not O(log D), not "small constant at small N." To honor that:

**Work:** Replace `BTreeMap<u64, Inode>` with a slab (`Vec<Slot>` indexed by inode id, freelist for reuse). Replace per-directory `BTreeMap<String, u64>` with an open-addressed hashmap keyed by `(parent_inode, name_hash)`. Both deliver worst-case-amortized O(1) per primitive op.

### 2.3 "File lookup O(1)" (theorem integration table)

Voronoi `locate()` already returns a cell in O(K=8). Resolution *within* the cell must also be O(1):

**Work:** Cap each cell at `MAX_CELL_OCCUPANCY = 64` files via a secondary geohash that subdivides on overflow and merges on under-flow. `find()` then reads exactly one bucket of ≤ 64 entries — bounded constant.

### 2.4 "50 µs teleport, 20000× speedup, regardless of file size"

The 50 µs target at any scale requires:

**Work (a):** AHCI write of the inode record completes in one DMA command (one 256-byte record, sector-atomic). Commodity SATA-3 finishes a single-sector DMA in ~30 µs — inside the 50 µs envelope.
**Work (b):** A `race` benchmark in `apps/shell.rs` that sweeps both file size (1 KB → 2 GB) and N (#files in source dir: 10 → 1 M), asserting the teleport µs reading stays flat across both axes. The 20 000× number derives from comparing against a 2 GB copy at 0.5 ns/byte = 1 s — that ratio holds once (a) is delivered.

### 2.5 "Content-addressable via Voronoi" and "all data = 64 points on S²"

`find()` must return its result in time independent of total file count, *and* must actually use all 64 sphere points (today's implementation compares only `points[0]`).

**Work:** Inside the ≤ 64-entry bucket from §2.3, compute similarity as the sum of dot products over all 64 sphere points — a fixed 64 × 64 = 4 096-flop loop, still O(1). Faithful read of the bible's "64 points on S²" — every point participates.

## 3. Architecture

Five new units. One rewritten unit. One newly-wired driver. Each independently testable.

```
┌──────────────────────────────────────────────────────────────┐
│                       ManifoldFS                             │
│  (rewritten — same public API as today, new internals)       │
└────────┬──────────────┬──────────────┬──────────────┬────────┘
         │              │              │              │
   ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐  ┌────▼────────┐
   │ InodeSlab │  │  DirHash  │  │ PathCache │  │ VoronoiCap  │
   │ Vec<Slot> │  │ HashMap<  │  │ LRU<path, │  │ bounded     │
   │+generation│  │ DirKey,id>│  │ inode>    │  │ cells, ≤64  │
   └─────┬─────┘  └───────────┘  └───────────┘  └─────────────┘
         │
   ┌─────▼──────────────────────────────────────────────────────┐
   │              BlockStore (new)                              │
   │  inode_id → (LBA, length) via direct mapping;              │
   │  superblock + buddy-bitmap freelist + write-through cache  │
   └─────┬──────────────────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────────────────┐
   │      AhciController (existing — newly wired)               │
   │      READ_DMA_EXT / WRITE_DMA_EXT                          │
   └────────────────────────────────────────────────────────────┘
```

### 3.1 `InodeSlab` — O(1) inode storage

```rust
pub struct InodeSlab {
    slots: Vec<Slot>,        // index == inode_id low 32 bits
    free_head: Option<u32>,  // freelist of recycled slots
}
enum Slot { Vacant { next_free: Option<u32> }, Occupied(Inode) }
```

`get(id)` is a `slots[i]` index — O(1) worst case. `alloc()` pops the freelist — O(1). `free()` pushes — O(1). Inode IDs are 64-bit: low 32 = slot index, high 32 = generation counter incremented on free, so stale handles return `None` instead of aliasing a recycled slot.

### 3.2 `DirHash` — O(1) directory entries

```rust
pub struct DirHash {
    entries: hashbrown::HashMap<DirKey, u64, FxBuildHasher>,
    // Per-dir intrusive doubly-linked list for readdir(); heads on dir inode.
}
#[derive(Hash, PartialEq, Eq)]
struct DirKey { parent: u64, name_hash: u64 }
```

`name_hash` is xxhash3-64 over the byte name; collisions resolved by storing the full name on the inode and strcmp on hit (collision rate < 2⁻⁵⁰ — amortized O(1) preserved). `hashbrown` is added with `default-features = false` and forced to `FxBuildHasher` to avoid `getrandom` (no x86_64-unknown-none impl). Per-boot random seed mixed in defends against HashDoS via adversarial filenames.

`readdir(dir_id)` walks the intrusive list rooted on the dir inode — O(K) in *that dir's* entry count, which POSIX itself imposes. Every per-entry op (lookup, insert, remove) is O(1).

### 3.3 `PathCache` — O(1) hot-path resolution

```rust
pub struct PathCache {
    map: hashbrown::HashMap<u64 /* path_hash */, CachedPath>,
    lru: IntrusiveLruList,
    capacity: usize,         // default 256
    generation: u64,         // bumped on rename/teleport/delete
}
```

`resolve_path(path)`:
1. Hash full path → check cache. Hit + generation match → O(1) done.
2. Miss → walk segments via `DirHash` (each hop O(1)) → O(D) cold-path.
3. Insert into cache, evict LRU.

Any mutation that could change a path resolution bumps `generation`, lazily invalidating cached entries on next probe. Coarse but correct; keeps mutation O(1).

### 3.4 `VoronoiCap` — bounded `find()`

Each cell capped at `MAX_CELL_OCCUPANCY = 64`. On overflow, split via a secondary geohash along the cell's principal axis of intra-cell payload variance; cell becomes a depth-2 node `(cell_id, subcell_id)`. `find()` geohashes the query to depth 2 and reads exactly one bucket of ≤ 64. Similarity over all 64 sphere points: 64 × 64 = 4 096 flops max — O(1).

Hysteresis: split at 64, merge at 24 — prevents thrash. Integrates with the existing `check_entropy_and_merge` path (merge is split's inverse).

### 3.5 `BlockStore` — every byte on disk, on S²

The persistence layer that makes claim 1 literally true.

**On-disk layout:**

```
LBA 0:        Superblock { magic, version, inode_table_lba, freelist_lba,
                           generation, total_blocks, crc32 }
LBA 1..K:     Inode table (256-byte records, slot i at LBA 1 + i/2)
LBA K..K+F:   Block freelist bitmap (two copies for fsck redundancy)
LBA K+F..end: Data extents — each leads with the encoded ManifoldPayload
              (64 × 3 × f64 = 1536 bytes of S² geometry), followed by the
              raw bytes. The literal first 1536 bytes of every on-disk
              file extent are S² geometry. A binary dump of any data
              block opens with the manifold.
```

**Mapping (all O(1)):**

- `inode_id → inode_record_lba`: `LBA = 1 + (slot_idx / 2)`.
- `inode_id → data_lba`: stored in the inode record; one DMA fetches the extent.

**Allocation:** Buddy bitmap keyed by extent size (next pow-2 ≥ `data.len()`, capped at `MAX_EXTENT_BLOCKS = 256` = 128 KiB at 512 B sectors). Allocation is O(1) per size class. Files larger than 128 KiB use one indirect block — one extra LBA fetch per data extent, still constant per block. (Acknowledged in §7.)

**Write path:** Update in-memory inode (O(1)) → one AHCI `WRITE_DMA_EXT` for the data extent (one command — O(1) commands; the bytes themselves are O(N) DMA, unavoidable physics) → one AHCI write for the inode record (sector-atomic — the commit point) → mark inode clean.

**Teleport on disk:** Only the inode record's `parent` field and two dir-entry records change — at most 3 sectors. Data extent does not move. This is the literal disk-level realization of the bible's "topological surgery" language: bytes do not migrate, only the manifold's connectivity changes. ~30 µs × 3 sectors = 90 µs worst case; for a hot dir whose records are RAM-cached, ≤ 50 µs.

**Crash safety:** Write order is data → inode record → superblock generation. Inode record is the commit point (sector-atomic ≤ 512 B). Crash mid-write leaves the inode pointing at the old extent; the new one leaks until fsck. CRC32 on every inode record guards against torn sectors on lying USB-SATA bridges.

### 3.6 `ManifoldFS` rewrite

Public API and `FileSystem` trait impls unchanged. Internal fields become:

```rust
pub struct ManifoldFS {
    inodes: InodeSlab,
    dirs: DirHash,
    path_cache: PathCache,
    store: BlockStore,
    voronoi: BoundedVoronoiIndex,    // wraps existing SphericalVoronoiIndex
    // existing aether-core ML/governor fields preserved
}
```

Call sites in `manifold_fs.rs` update mechanically. Existing `host_tests` and the no_std `tests` module remain valid by signature; the new bible-claim tests land alongside.

## 4. Data flow — each bible claim, proved

### Claims 2 / 3 / 8 / 12 / 13 — O(1) topological surgery for `teleport`

```
1. dirs.lookup(src, name)          // O(1)
2. dirs.lookup(dst, name)          // O(1) — must miss
3. dirs.remove(src, name)          // O(1)
4. dirs.insert(dst, name, id)      // O(1)
5. inodes.get_mut(id).parent = dst // O(1)
6. path_cache.bump_generation()    // O(1)
7. store.write_inode_record(id)    // one AHCI cmd — O(1)
```
Six O(1) ops + one AHCI command. Topological surgery, literally.

### Claims 5 / 6 — `file lookup O(1)`, `O(1) retrieval via Voronoi`

```
1. voronoi.locate(payload) → (cell, subcell)  // O(K=8)
2. dirs.lookup(parent, name) → inode_id       // O(1)
3. inodes.get(inode_id)                       // O(1)
```

### Claims 7 / 15 — `find()` content-addressable in bounded time, using all 64 points

```
1. encode(query) → payload                    // O(1) (already true)
2. voronoi.locate(payload) → (cell, subcell)  // O(K=8)
3. read bucket of ≤ 64 entries                // O(1)
4. similarity over all 64 sphere points each  // 64 × 64 ≤ 4 096 flops — O(1)
5. partial-sort top-k                         // O(1) at fixed k
```

### Claims 9 / 10 — 50 µs teleport, size-independent

`teleport` writes at most three sectors (one inode record + two dir-entry records). At 30 µs per AHCI command on SATA-3: ≤ 90 µs cold, ≤ 50 µs warm. The `race` benchmark sweeps file size (1 KB → 2 GB) and asserts teleport time stays ≤ 50 µs warm regardless. The 20 000× speedup is measured against a 2 GB byte copy at 0.5 ns/byte = 1 s vs. 50 µs.

### Claim 1 — Every byte on disk is a point cloud on the unit sphere

`BlockStore` writes each file's data extent payload-first: the literal first 1 536 bytes of every on-disk file extent are the 64 × 3 × f64 S² point cloud. The raw bytes follow only as cached pre-decoded content. The literal-on-disk claim is true byte-by-byte.

## 5. Testing — the bible verified on every CI run

Every claim in §1 becomes a CI assertion. The QEMU smoke test gains:

| Bible claim | Test |
|---|---|
| O(1) teleport (claims 2/3/8/12/13) | `teleport_is_o1_under_load` — store {10, 1k, 10k, 100k} files; assert median + p99 teleport ticks flat within 2× across all four. Replaces today's "≤ 5 ticks at N=1" test (which only proves shallow BTreeMap). |
| 50 µs / size-independent (claims 9/10) | `teleport_size_sweep` — files of {1 KB, 1 MB, 1 GB, 2 GB}; assert wall-µs ≤ 50 warm across all four; ≤ 90 µs cold. |
| O(1) `find()` (claims 6/7/15) | `find_bounded` — populate cells past cap, force splits, assert `find` returns in ≤ 4 096 flops across N = {10², 10⁴, 10⁶}. |
| O(1) inode + name lookup (claim 5) | `lookup_is_o1` — sweep N; assert `dirs.lookup` and `inodes.get` flat in tick count. |
| "Every byte on disk = point cloud on S²" (claim 1) | `disk_layout_payload_first` — write a file, dump its data extent from the AHCI image, assert the first 1 536 bytes round-trip back to the inode's `ManifoldPayload` exactly. |
| Persistence (foundation for claim 1) | `mkfs → store 100 → reboot → readback` — drop the FS, re-mount from disk, verify all files readable byte-for-byte with their ManifoldPayloads intact. |
| CI milestone 8 (claim 11) | Existing smoke milestone retained; now backed by the size-sweep + N-sweep tests above. |

Host tests (`cargo test`) run the same matrix against a `MockBlockDevice` to keep CI fast.

## 6. Implementation phases

Each phase leaves the kernel bootable and CI green.

1. **InodeSlab + DirHash.** In-memory swap. Existing `host_tests` pass unchanged. New `lookup_is_o1` test gates the PR.
2. **PathCache.** Wire into `resolve_path*`. Add hit/miss to `FsStats`.
3. **VoronoiCap + all-64-points similarity.** Subdivide cells past cap; integrate with `check_entropy_and_merge`. `find_bounded` gates the PR.
4. **BlockStore over MockBlockDevice.** Full on-disk semantics behind a mock. `disk_layout_payload_first` gates the PR.
5. **Wire AHCI.** Replace mock with real `AhciController`. Add `mkfs.manifold` shell command; mount on boot. Persistence smoke test gates the PR.
6. **Size-sweep `race` + crash testing + fsck.** Power-loss simulation, leaked-extent scanner.

Six PRs, ordered by dependency. PRs 1–4 are pure host-testable. PR 5 needs QEMU + virtual disk in CI. PR 6 needs the QEMU power-loss harness.

## 7. Things that can fuck up

A real list. Each has a mitigation; residual risk named where it exists.

### 7.1 Hash collision adversary
Filenames chosen to share `name_hash` degrade `DirHash` probe chains. **Mitigation:** per-boot random seed in hasher; cap probe length at 32 with fallback linear scan + warn. **Residual risk:** adversary with both filename control and seed knowledge can degrade one specific directory.

### 7.2 Slab generation overflow
32-bit generation wraps after 4 B slot reuses. **Mitigation:** document the wrap; widen to 48 bits if needed (shrink slot index to 16 — we don't need 4 B concurrent inodes).

### 7.3 PathCache invalidation defeats hit rate
Coarse generation bump invalidates *all* entries on any mutation. Under churn the cache becomes useless. **Mitigation:** measure hit rate via `FsStats`; if < 50 %, switch to prefix-keyed fine-grained invalidation.

### 7.4 Voronoi split thrash
Workload oscillates around the cap. **Mitigation:** hysteresis — split at 64, merge at 24. Log split/merge counts to `FsStats`.

### 7.5 AHCI write reordering
NCQ may reorder commands across queue slots; a crash with data-write-after-inode-write leaves a dangling pointer. **Mitigation:** submit data write, wait for `PORT_CI` bit clear, *then* submit inode write. Lose pipelining; gain correctness. Document the perf cost.

### 7.6 Sector tearing on lying SATA bridges
Spec-compliant drives are sector-atomic; some USB-SATA bridges aren't. **Mitigation:** CRC32 in every inode record; torn records read as "old value" on recovery.

### 7.7 Freelist corruption
Single bitmap = single point of failure. **Mitigation:** keep two copies (LBA K and LBA end−K); pick the one whose CRC matches the superblock generation. Rebuild via fsck full-disk scan.

### 7.8 Inode table sized too small at mkfs
Fixed inode table; running out is fatal. **Mitigation:** `mkfs.manifold --inodes=N`; default 1 M (256 MB of inode table — fine on real disks, sized down for the QEMU 16 MB image).

### 7.9 Large files
Files > `MAX_EXTENT_BLOCKS × 512` (128 KiB) need an indirect block — one extra LBA fetch per data extent. **Stance:** still constant per block (so per-block O(1) holds), but a large-file `read` does two AHCI commands for the metadata. Acknowledged trade-off; documented in the README under file-size limits when this lands.

### 7.10 Hashbrown in no_std
`hashbrown` works in no_std with `default-features = false`, but the default `ahash` hasher pulls `getrandom` (no x86_64-unknown-none impl). **Mitigation:** force `FxBuildHasher`; pin hashbrown version.

### 7.11 The benchmark could lie
"Flat tick count" only proves O(1) over the tested range. **Mitigation:** sweep N at 10, 1 k, 10 k, 100 k *and* sweep file size at 1 KB → 2 GB; require constant within 2× across all eight points. Tests at a single N or a single size do not count.

### 7.12 Crash during fsck
fsck rewriting the freelist can leave the FS unmountable. **Mitigation:** fsck writes to a *copy*; flips the superblock pointer to it as the atomic last step (sector-write).

### 7.13 Bible drift
The README is the contract. If implementation forces a change to a claim, the README — not the test or the code — must be edited first, with review. The plan is wrong if it would force the bible to be edited; revisit the architecture instead.

## 8. Open questions

1. Path cache capacity — 256 is a guess. Tunable via `kernel.cmdline` in phase 2.
2. `find()` neighborhood — strictly the geohash bucket (O(1)), or small-ball over neighbors (O(constant·neighborhood), still O(1)-bounded) to avoid cell-boundary misses? Recommend small-ball with neighborhood = 1.
3. Should `Inode::data` (in-RAM bytes) be retained as a write-back cache after persistence lands, or always go to disk? Recommend: write-back with dirty bit + periodic flush.

---

**Reviewer's note:** §1 is the contract. §3 is the architecture. §7 is the risk register. If a section feels off, those are the three to push on.
