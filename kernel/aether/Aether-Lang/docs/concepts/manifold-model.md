# Manifold Model

Aether-Lang exposes data as manifold-oriented objects so scripts can describe
shape, locality, and convergence without directly managing every numeric
operation.

## Time-Delay Embedding

For scalar samples \(x(t)\), the documented embedding model is:

\[
\Phi(t) = [x(t), x(t-\tau), x(t-2\tau), \ldots, x(t-(d-1)\tau)]
\]

In the current interpreter surface this appears as:

```aegis
manifold M = embed(data, dim=3, tau=5)
```

The active implementation maintains manifold workspaces and derives block
metadata from embedded points. It is a runtime data model for scripts, not a
general persistent-homology engine.

## Block Metadata

A block is a contiguous region of a manifold workspace.

```aegis
block B = M.cluster(0:64)
center = B.center
spread = B.spread
```

The interpreter tracks metadata such as centroid, radius, variance, angular
concentration, and point count. That metadata is enough for local inspection and
script decisions.

## Regression Surface

Regression statements describe a model family and a convergence condition:

```aegis
regress {
  model: "polynomial",
  degree: 3,
  escalate: true,
  until: convergence(1e-6)
}
```

The documented contract is structural: scripts can express escalating model
selection and convergence thresholds. Stronger claims about accuracy, model
quality, or runtime speed require benchmark evidence.

## Failure Modes

- Sparse or low-variance input can produce uninformative block metadata.
- A small embedding dimension can hide structure that needs a larger state
  vector.
- A large `tau` can decorrelate samples enough to distort local geometry.
- Regression convergence can reflect the configured threshold rather than a
  useful model.

The runtime should expose these as diagnostics before the docs claim reliability
for a domain workload.
