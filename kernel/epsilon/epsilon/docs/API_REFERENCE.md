# Epsilon API Reference

## epsilon-os (Topological OS)

### `ManifoldMemory`

Episodic memory on S^(D-1) with Voronoi-accelerated retrieval.

```rust
let mut mem = ManifoldMemory::new(128, 100_000); // dim=128, capacity=100k

// Store
let vec = mem.encode_text("the cat sat on the mat");
let idx = mem.store(vec, "the cat sat on the mat".into(), HashMap::new());

// Retrieve (O(1) cell lookup when TSS active, O(n) otherwise)
let query = mem.encode_text("cat on mat");
let results = mem.retrieve(&query, 5); // top-5 by cosine similarity
// returns Vec<(index, score, &Episode)>

// Topology
let components = mem.betti_0(); // connected components at cluster_radius threshold

// Stats
let stats = mem.stats();
// stats.tss_active, stats.largest_cell_size, stats.betti_0, ...
```

TSS activates automatically when episode count reaches 16. The first 3 dimensions of
each L2-normalized vector are projected to spherical coordinates (theta, phi) and
assigned to one of 8 Voronoi cells. Retrieval scans only the query's cell.

### `LatentPredictor`

Forward dynamics via spectral contraction (T2/SCM).

```rust
let pred = LatentPredictor::new(128, 32, 42); // state_dim, action_dim, seed
let action = pred.hash_action("explore north");
let next_state = pred.step(&state, &action, &attractor);
// state contracts toward attractor by alpha per step
```

### `World`

Combines memory, predictor, and LLM bridge.

```rust
let mut w = World::new(api_key, 128, 100_000);
w.update("observation text");           // ingest
let q = w.query("question", 5);         // retrieve + optional LLM
let d = w.dream("scenario", 5);         // counterfactual rollout
let t = w.verify_theorems();            // T1-T10 pass/fail
```

### Result Types

All result types derive `serde::Serialize` for JSON output.

- `UpdateResult` — index, ingest_step, memory_size, betti_0
- `QueryResult` — retrieval_ms, top_k, llm_response
- `DreamResult` — horizon, total_reward, trace (per-step norms and rewards)
- `MemoryStats` — size, dim, capacity, betti_0, tss_active, tss_cells, largest_cell_size
- `WorldStatus` — dim, capacity, ingest_steps, memory stats, api_key_set

---

## epsilon (Context Teleportation)

### `EpsilonPoint<const D: usize>` -- manifold.rs

Point in D-dimensional space.

```rust
let p = EpsilonPoint::<3>::new([1.0, 2.0, 3.0]);
p.distance(&other)         // Euclidean
p.is_neighbor(&other, 0.5) // epsilon-test
```

### `SparseGraph<const D: usize>` -- manifold.rs

Sparse attention graph with Betti number computation.

```rust
let mut g = SparseGraph::<3>::new(1.0);
g.add_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
g.shape() // (b0, b1)
```

### `HollowCubeManifold<const D: usize>` -- manifold.rs

S^2 shell with b2=1 void for topological surgery.

```rust
let mut m = HollowCubeManifold::<3>::new(1.0);
m.add_shell_point(point);
m.inject_into_void(payload)?; // topological surgery
m.assimilate();               // merge into shell
```

### `SurgeryGovernor` -- governor.rs

PD controller: epsilon(t+1) = epsilon(t) + alpha*e(t) + beta*de/dt

```rust
let mut gov = SurgeryGovernor::new();
gov.adapt(deviation, dt);
let permit = gov.prepare_for_surgery(); // beta -> 0
gov.complete_surgery(permit);           // beta -> saved
```

---

## aether-core (Math Foundation)

### `SphericalVoronoiIndex<const K: usize>` -- tss.rs

O(1) amortized nearest-cell lookup on S^2.

```rust
let centroids = [(0.0, 0.0), (1.0, 0.5), (0.5, 1.0)];
let idx = SphericalVoronoiIndex::<3>::new(centroids);
let cell = idx.locate((0.1, 0.2)); // nearest centroid index
```

### `SpectralContractionOperator<const D: usize>` -- scm.rs

Banach contraction mapping. Lipschitz constant = 1 - alpha.

```rust
let op = SpectralContractionOperator::<4>::new(0.1); // alpha = 0.1
// state contracts toward attractor at rate 0.9^n
```

### `GeometricGovernor` -- governor.rs

Adaptive threshold controller with spectral stability bounds.

```rust
let mut gov = GeometricGovernor::with_gains(0.01, 0.05);
let new_eps = gov.adapt(deviation, dt);
```
