# Plan: Topological Drivers for RAM and 3D Rendering

## Objective
Implement two topological drivers that embed T1–T5 theorems into kernel subsystems:
1. **Topological RAM Driver** — memory allocation driven by manifold geometry
2. **Topological 3D Render Driver** — software rasterizer driven by manifold geometry

---

## Phase 1: Topological RAM Driver (`kernel/seal-os/src/memory/topo_ram.rs`)

### Design
The topological RAM driver wraps the existing physical frame allocator (`memory/phys.rs`) and slab allocator (`memory/slab.rs`) with topology-aware decisions. Each allocation request is embedded into a 64-point cloud on S², and the T-theorem substrate guides frame selection, prefetching, defragmentation, and pressure adaptation.

### Data Structures
- `TopoFrame` — per-frame metadata:
  - `embedding: [f32; 64]` — 64D point cloud on S² (normalized to unit sphere)
  - `access_history: u64` — bitfield of last 64 access ticks (for T2 spectral prediction)
  - `cell_id: u16` — Voronoi cell assignment (T1)
  - `lifetime_class: u8` — 0=ephemeral, 1=medium, 2=permanent (T5 hyperbolic classification)
  - `entropy_score: f32` — local Betti-0 density estimate (T3)
- `TopoRam` — global singleton behind `spin::Mutex`:
  - `frame_meta: [TopoFrame; USABLE_FRAMES]` — metadata for every allocatable frame
  - `voronoi_seeds: Vec<[f32; 64]>` — seed points for Voronoi cells
  - `pressure: f32` — system pressure gauge for governor (T4)
  - `tick: u64` — monotonic access tick counter

### Theorem Integration

**T1 — Voronoi Partitioning**
- On `alloc_frames(count, locality_hint)`, compute the Voronoi cell of the hint embedding.
- Search for free frames within the same cell first (locality).
- If cell is exhausted, search adjacent cells by seed distance.
- Re-seed Voronoi cells every 1024 allocations or when fragmentation > threshold.

**T2 — Spectral Contraction**
- Maintain a 64×64 covariance matrix over recent allocation embeddings.
- Eigen-decompose (power iteration, 4 iterations per tick is enough) to find principal axes.
- When a frame is allocated near a principal axis, prefetch N nearby frames (where N = eigenvalue rank).
- Update access_history bitfield on every `alloc`/`free`/`access` call.

**T3 — Entropy/Fragmentation Governor**
- Compute Betti-0 (connected components of free frames) by scanning the bitmap.
- Entropy = log(components) / log(total_free). When entropy > 0.7, trigger compaction.
- Compaction: walk allocated frames, identify least-fragmented contiguous run, swap frames.
- This is advisory; actual compaction deferred to idle time.

**T4 — PD Governor**
- `pressure = (alloc_rate - free_rate) / (alloc_rate + free_rate + 1.0)` clamped to [0, 1].
- High pressure (>0.6): prefer large contiguous allocations, disable prefetch.
- Low pressure (<0.3): enable fine-grained per-frame metadata, aggressive prefetch.
- Adaptive allocation granularity: pressure > 0.8 switches to page-cluster (16-frame) mode.

**T5 — Hyperbolic Curvature**
- Map allocation lifetimes to hyperbolic disk Poincaré model.
- Short-lived allocations cluster near boundary (high curvature = fast turnover).
- Long-lived allocations cluster near center (low curvature = stable).
- When freeing a frame, propagate its curvature class to neighbors to predict their lifetimes.

### API
```rust
pub fn init();                              // called once from kernel_main after phys::init
pub fn alloc_frames(count: usize, hint: &[f32; 64]) -> Option<PhysAddr>;
pub fn free_frames(addr: PhysAddr, count: usize);
pub fn access_frame(addr: PhysAddr);        // mark frame as accessed this tick
pub fn prefetch_hint(addr: PhysAddr) -> Vec<PhysAddr>; // T2 predicted frames
pub fn fragmentation_entropy() -> f32;      // T3 metric
pub fn set_pressure(p: f32);                // T4 external input (from scheduler)
```

### Integration
- `memory/mod.rs` adds `pub mod topo_ram;`
- `kernel_main` calls `memory::topo_ram::init()` after `memory::phys::init()`
- Slab allocator can optionally call `topo_ram::alloc_frames(1, zero_hint)` when no topological hint is available

---

## Phase 2: Topological 3D Render Driver (`kernel/seal-os/src/graphics/topo_render.rs`)

### Design
A software rasterizer that accepts 3D triangle meshes and renders them to a `Window` framebuffer. All geometry is represented as point clouds on S², and rendering decisions (culling, LOD, quality) are driven by the T-theorem substrate.

### Data Structures
- `TopoMesh` — mesh data:
  - `vertices: Vec<[f32; 3]>` — model-space coordinates
  - `triangles: Vec<[u32; 3]>` — indices
  - `normals: Vec<[f32; 3]>` — per-vertex normals
  - `spherical_embedding: Vec<[f32; 64]>` — 64-point cloud on S² per vertex
  - `bbox: BoundingBox` — AABB for coarse culling
- `RenderState` — global renderer state:
  - `camera: Camera` — position, look_at, up, fov
  - `clip_planes: [f32; 4]` — near, far, left, right
  - `last_frame_ms: u32` — frame time for T4 governor
  - `quality_level: u8` — 0–4, adapted by T4
  - `voronoi_cells: Vec<VoronoiCell>` — screen-space spatial partition (T1)
- `VoronoiCell` — screen tile:
  - `x, y, w, h: u16` — pixel bounds
  - `triangle_indices: Vec<usize>` — triangles that overlap this cell

### Theorem Integration

**T1 — Voronoi Spatial Partition**
- Partition screen into 8×8 Voronoi cells (64 tiles).
- During rasterization, for each triangle, determine overlapping cells via bounding box.
- Only rasterize the triangle within those cells (trivial rejection).
- Cells are rebalanced every frame based on triangle density.

**T2 — Spectral LOD**
- Compute screen-space area of each triangle.
- If area < threshold, collapse to point or skip.
- The threshold is adapted by spectral prediction: if a mesh had low detail last frame and camera moved slowly, keep low detail (temporal coherence).
- Build a Laplacian over mesh connectivity; eigenvalues determine importance of vertices.

**T3 — Betti Mesh Integrity**
- Compute Betti-0 (connected components) and Betti-1 (cycles) of mesh graph.
- If Betti-1 drops (hole appears) after simplification, reject that LOD level.
- Use this to guarantee no topology-breaking simplification.

**T4 — Governor (Adaptive Quality)**
- Target frame time: 16 ms (60 FPS).
- If last_frame_ms > 20 ms, reduce quality_level (disable AA, reduce resolution, coarser LOD).
- If last_frame_ms < 12 ms for 5 frames, increase quality_level.
- Quality levels:
  - 0: wireframe, no culling
  - 1: flat shading, 4×4 tiles
  - 2: gouraud, 8×8 tiles, simple texture
  - 3: phong, 8×8 tiles, bilinear texture
  - 4: phong + 2×MSAA, adaptive tiles

**T5 — Hyperbolic Projection**
- Map camera frustum to hyperboloid model (Lorentzian inner product).
- Cull objects by hyperbolic distance: objects with negative distance are behind camera.
- Use hyperbolic law of cosines for backface culling in curved space.
- This gives a consistent "fish-eye" or wide-FOV effect and naturally handles extreme angles.

### Pipeline
```
Input: TopoMesh + Camera
  ↓
T5 Hyperbolic Cull & Transform (model → world → clip)
  ↓
T2 Spectral LOD Simplification
  ↓
T3 Betti Integrity Check
  ↓
T1 Voronoi Cell Assignment (screen tiles)
  ↓
Rasterize triangles per cell (scanline or barycentric)
  ↓
T4 Quality Adaptation (update state for next frame)
  ↓
Blit to Window framebuffer
```

### API
```rust
pub fn init();                              // called once from kernel_main
pub fn render_mesh(mesh: &TopoMesh, camera: &Camera, target: &mut Window);
pub fn set_camera(cam: Camera);
pub fn adaptive_quality(frame_ms: u32);     // T4 feedback
pub fn project_vertex(v: &[f32; 3], cam: &Camera) -> [f32; 4]; // T5 hyperbolic projection
```

### Integration
- `graphics/mod.rs` adds `pub mod topo_render;`
- `kernel_main` calls `graphics::topo_render::init()` after framebuffer init
- `apps::desktop` can call `topo_render::render_mesh()` for 3D wallpapers or widgets

---

## Build & Test Plan

1. Add new files to `kernel/seal-os/src/memory/topo_ram.rs` and `kernel/seal-os/src/graphics/topo_render.rs`
2. Update `memory/mod.rs` and `graphics/mod.rs`
3. Update `lib.rs` boot sequence to call `topo_ram::init()` and `topo_render::init()`
4. Run `cargo +nightly check` in `kernel/seal-os`
5. If clean, run `cargo +nightly test --features test-mode`
6. Update README.md "Just Activated" section with the two new drivers

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| 64D embedding too heavy for per-frame metadata | Store as 16× `u16` quantized angles (32 bytes per frame) |
| Voronoi re-seeding too slow | Only re-seed on fragmentation events, not every allocation |
| Hyperbolic math overflows | Use `libm` for `sinh`/`cosh`, clamp inputs |
| Software rasterizer too slow | Start with wireframe/flat, add features incrementally |
| `no_std`/`no_alloc` in kernel | Use `Vec` from `alloc` crate (already available in kernel) |
