# V3 — Formula Playgrounds

> Interactive math playgrounds where seal can manipulate topological/geometric objects in real-time.
> Each playground is a self-contained panel that can be added to the workstation layout.

## 1. Riemannian Metric Playground

**What**: Edit metric tensor components g_ij, see geodesics update live on a surface.

**Math**:
```
Metric tensor: ds² = g_ij dx^i dx^j

Geodesic equation:
d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0

Christoffel symbols:
Γ^k_ij = (1/2) g^{kl} (∂g_{li}/∂x^j + ∂g_{lj}/∂x^i - ∂g_{ij}/∂x^l)
```

**Implementation**:
- Create `src/playgrounds/RiemannianPlayground.tsx`
- 2x2 or 3x3 matrix input for g_ij (symmetric, positive definite)
- Compute Christoffel symbols numerically
- Integrate geodesic ODE with RK4
- Render surface + geodesic curves in Three.js
- Preset surfaces: flat, sphere, hyperbolic, torus

**Python backend**: Add `cli/riemannian.py`:
```python
def compute_geodesic(metric_fn, start, velocity, n_steps=1000, dt=0.01):
    """Integrate geodesic equation using RK4."""
    # Returns array of (n_steps, dim) positions
```

---

## 2. Homology Calculator

**What**: Compute H_0, H_1, H_2 of a simplicial complex built from point cloud data.

**Math**:
```
Chain complex: C_2 →^{∂_2} C_1 →^{∂_1} C_0

Boundary maps:
∂_1([v_0, v_1]) = v_1 - v_0
∂_2([v_0, v_1, v_2]) = [v_1, v_2] - [v_0, v_2] + [v_0, v_1]

Homology groups:
H_k = ker(∂_k) / im(∂_{k+1})

Betti numbers:
β_k = rank(H_k) = dim(ker(∂_k)) - dim(im(∂_{k+1}))
```

**Implementation**:
- CLI: `python cli/governor_cli.py homology <path> --epsilon=0.5 --max_dim=2`
- Build Vietoris-Rips complex at given epsilon
- Compute boundary matrices
- Smith normal form → Betti numbers
- Return generators (representative cycles)
- Frontend: show complex as wireframe, highlight cycle generators

---

## 3. Betti Number Explorer

**What**: Slider for epsilon, watch β_0, β_1, β_2 change in real-time.

**Math**:
```
Vietoris-Rips complex at scale ε:
VR_ε = { σ ⊆ X : diam(σ) ≤ ε }

Betti number function:
β_k(ε) = rank(H_k(VR_ε))

Persistent Betti:
β_k^{s,t} = rank(im(H_k(VR_s) → H_k(VR_t)))
```

**Implementation**:
- Precompute persistence intervals via ripser
- For any epsilon, count intervals containing that value
- Frontend: animated counter + bar chart that updates as slider moves

---

## 4. Euler Characteristic Tracker

**What**: χ = β_0 - β_1 + β_2 over the filtration.

**Math**:
```
Euler characteristic:
χ(K) = Σ_{k=0}^n (-1)^k β_k = Σ_{k=0}^n (-1)^k |K_k|

where |K_k| = number of k-simplices

Euler-Poincaré formula:
χ = V - E + F (vertices - edges + faces)

Gauss-Bonnet (smooth):
∫_M K dA = 2π χ(M)
```

**Implementation**: Compute from Betti curves, plot χ(ε) as function of filtration parameter.

---

## 5. Wasserstein Distance

**What**: Compare persistence diagrams between two datasets.

**Math**:
```
p-Wasserstein distance:
W_p(D₁, D₂) = (inf_{γ: D₁ → D₂} Σ ||x - γ(x)||_∞^p)^{1/p}

where γ ranges over all bijections (including diagonal projections)

Bottleneck distance (p = ∞):
W_∞(D₁, D₂) = inf_{γ} sup_x ||x - γ(x)||_∞
```

**Implementation**:
- Python: `pip install persim` for Wasserstein/bottleneck computation
- CLI: `python cli/governor_cli.py compare <path1> <path2>`
- Frontend: side-by-side persistence diagrams with matching lines

---

## 6. Vietoris-Rips Complex Builder

**What**: Step-by-step visualization of simplicial complex construction.

**Math**:
```
Definition: σ = [v_0, ..., v_k] ∈ VR_ε iff d(v_i, v_j) ≤ ε for all i,j

Inclusion: ε₁ ≤ ε₂ ⟹ VR_{ε₁} ⊆ VR_{ε₂}

This gives a filtration:
∅ ⊆ VR_{ε₀} ⊆ VR_{ε₁} ⊆ ... ⊆ VR_{ε_n}
```

**Implementation**:
- Show points, then edges, then triangles as epsilon slider increases
- Three.js wireframe + filled triangles
- Count simplices at each dimension

---

## 7. Mapper Algorithm

**What**: Interactive Mapper (topological data analysis summarization tool).

**Math**:
```
Mapper algorithm:
1. Choose lens function f: X → R (e.g., eccentricity, density, PCA component)
2. Cover f(X) with overlapping intervals U_1, ..., U_n
3. For each U_i, cluster f^{-1}(U_i) ∩ X
4. Build nerve complex: nodes = clusters, edges = shared points

Parameters:
- Lens function f
- Number of intervals n
- Overlap percentage p
- Clustering algorithm (DBSCAN, single linkage, etc.)
```

**Implementation**:
- Python: `pip install kmapper`
- CLI: `python cli/governor_cli.py mapper <path> --lens=eccentricity --intervals=10 --overlap=0.3`
- Frontend: Force-directed graph layout of Mapper output, colored by lens function value

---

## 8. Morse Theory Viewer

**What**: Critical points and gradient flow on surfaces.

**Math**:
```
Morse function f: M → R
Critical points: ∇f(p) = 0

Index of critical point = number of negative eigenvalues of Hessian

Index 0: minimum (source)
Index 1: saddle (1D unstable manifold)  
Index 2: maximum (sink)

Morse inequality:
β_k ≤ c_k (number of index-k critical points)

Strong Morse inequality:
Σ_{k=0}^n (-1)^{n-k} c_k ≥ Σ_{k=0}^n (-1)^{n-k} β_k
```

**Implementation**:
- Compute discrete Morse function on triangulation
- Find critical simplices
- Trace gradient paths (V-paths)
- Frontend: surface colored by Morse function value, critical points marked, gradient flow lines

---

## 9. Hodge Decomposition

**What**: Decompose a vector field on a simplicial complex into harmonic, exact, and coexact components.

**Math**:
```
Hodge decomposition theorem:
Ω^k = im(d_{k-1}) ⊕ im(δ_{k+1}) ⊕ H^k

where:
- im(d_{k-1}) = exact forms (gradients)
- im(δ_{k+1}) = coexact forms (curls)  
- H^k = harmonic forms (topology)

Discrete Hodge Laplacian:
Δ_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k

Harmonic forms: Δ_k ω = 0
dim(ker(Δ_k)) = β_k (Betti number)
```

**Implementation**:
- Build boundary matrices ∂_1, ∂_2
- Compute Hodge Laplacian Δ = ∂∂^T + ∂^T∂
- Eigendecomposition → harmonic, exact, coexact
- Frontend: 3D vector field visualization with component toggle
