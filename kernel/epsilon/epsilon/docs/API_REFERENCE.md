# Epsilon API Reference

---

## Core Types

### `EpsilonPoint<const D: usize>` â€” manifold.rs

Point in D-dimensional space. Public `coords: [f64; D]`.

```rust
let p = EpsilonPoint::<3>::new([1.0, 2.0, 3.0]);
p.distance(&other)         // Euclidean
p.is_neighbor(&other, 0.5) // Îµ-test
```

### `SparseGraph<const D: usize>` â€” manifold.rs

Sparse attention graph. Public `points`, `point_count`.

```rust
let mut g = SparseGraph::<3>::new(1.0);
g.add_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
g.shape() // (Î²â‚€, Î²â‚)
```

### `ManifoldPayload<const D: usize>` â€” manifold.rs

Geometric payload with Betti signature and liveness anchor.

```rust
let payload = ManifoldPayload::from_graph(&graph, 5.0);
// payload.signature_b0, payload.liveness_anchor
```

### `HollowCubeManifold<const D: usize>` â€” manifold.rs

SÂ² shell with Î²â‚‚=1 void.

```rust
let mut m = HollowCubeManifold::<3>::new(1.0);
m.add_shell_point(point);
m.inject_into_void(payload)?; // topological surgery
m.assimilate();               // merge into shell
```

### `SurgeryError` â€” manifold.rs

```rust
VoidOccupied | TopologyMismatch { expected_b0, actual_b0 }
EmptyPayload | DegenerateShell { shell_b0 }
```

---

## Governor

### `SurgeryGovernor` â€” governor.rs

PD controller: `Îµ(t+1) = Îµ(t) + Î±Â·e(t) + Î²Â·de/dt`

```rust
let mut gov = SurgeryGovernor::new();
gov.adapt(deviation, dt);
let permit = gov.prepare_for_surgery(); // Î² â†’ 0
gov.complete_surgery(permit);           // Î² â†’ saved
```

### `SurgeryPermit` â€” governor.rs

One-shot. Non-Clone. Non-Copy. Consumed on restore.

---

## Memory

### `LivenessAnchor` â€” memory.rs

```rust
let anchor = LivenessAnchor::from_samples(&data, 2.0);
// anchor.anchor_liveness = Î¼ + Ïƒ
```

### `ChebyshevGuard` â€” memory.rs

```rust
let guard = ChebyshevGuard::with_inherited_anchor(&anchor);
guard.is_safe(liveness) // true if above Î¼ - kÂ·Ïƒ
```

---

## Orchestration

### `sys_context_inject()` â€” inject.rs

```rust
let result = sys_context_inject(&mut manifold, payload, &mut gov);
// injectResult::Success { points_assimilated }
```

Full pipeline: clutch â†’ inject â†’ assimilate â†’ restore.
