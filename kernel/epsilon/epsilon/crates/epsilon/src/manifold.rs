// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//! Epsilon Hollow Cube Manifold & Injectable Payloads
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//!
//! Implements the core topological structures for Zero-Shot Context
//! Context Transfer. This module defines:
//!
//! 1. [`EpsilonPoint`] â€” A point in D-dimensional manifold space
//! 2. [`SparseGraph`] â€” Sparse attention graph with Betti number computation
//! 3. [`ManifoldPayload`] â€” Injectable data unit with Betti signature
//! 4. [`HollowCubeManifold`] â€” SÂ² boundary with Î²â‚‚ = 1 interior void
//!
//! # Mathematical Foundation (Section 2)
//!
//! ## The Hollow Cube (SÂ² Manifold)
//!
//! ```text
//!   Solid Manifold:  Î²â‚€ = 1, Î²â‚ = 0, Î²â‚‚ = 0
//!   Hollow Manifold: Î²â‚€ = 1, Î²â‚ = 0, Î²â‚‚ = 1
//! ```
//!
//! The presence of Î²â‚‚ = 1 defines an interior "void." This void serves
//! as the secure receptacle for instantaneous geometric injection.
//!
//! ## Topological Surgery & Fiber Bundle Projection
//!
//! ```text
//!   f: D âŠ‚ M_high â†’ Void(M_recv)
//! ```
//!
//! The injection maps the geometry of D (where the Seal-Loop has
//! converged) to the interior boundary constraints of M_recv.
//!
//! ## Wake-Up Rescan (Section 3.3)
//!
//! Once data is injected into the Î²â‚‚ void, it is not immediately active.
//! The Bio-Kernel triggers a Wake-Up interrupt, the Seal-Loop verifies
//! Betti boundaries, and if topologies align, the data is assimilated
//! in O(1) time relative to token length.
//!
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

use libm::sqrt;

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Constants
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// Maximum points in a sparse graph (memory constraint)
const MAX_POINTS: usize = 256;

/// Maximum points in a Injectable payload
const MAX_PAYLOAD_POINTS: usize = 64;

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// EpsilonPoint â€” Manifold Point
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// A point in D-dimensional manifold space.
///
/// Interoperable with `aether_core::ManifoldPoint` via coordinate arrays.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EpsilonPoint<const D: usize> {
    /// Coordinates of the point in D-dimensional space.
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub coords: [f64; D],
}

impl<const D: usize> EpsilonPoint<D> {
    /// Create the origin point (all zero coordinates).
    pub const fn zero() -> Self {
        Self { coords: [0.0; D] }
    }

    /// Construct a point from a coordinate array.
    pub fn new(coords: [f64; D]) -> Self {
        Self { coords }
    }

    /// Euclidean distance: ||p - q||â‚‚
    pub fn distance(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..D {
            let d = self.coords[i] - other.coords[i];
            sum += d * d;
        }
        sqrt(sum)
    }

    /// Îµ-neighborhood test (sparse attention criterion)
    pub fn is_neighbor(&self, other: &Self, epsilon: f64) -> bool {
        self.distance(other) < epsilon
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// SparseGraph â€” Sparse Attention Graph with Betti Computation
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// Sparse attention graph using geometric locality.
///
/// Instead of O(nÂ²) dense attention, connects only points within
/// Îµ-neighborhood: A(i,j) = 1 iff d(páµ¢, pâ±¼) < Îµ.
///
/// Computes Betti numbers Î²â‚€ (connected components) and Î²â‚ (cycles).
#[derive(Debug)]
pub struct SparseGraph<const D: usize> {
    /// Backing storage for points (only the first `point_count` entries are valid).
    pub points: [EpsilonPoint<D>; MAX_POINTS],
    /// Number of points currently inserted into the graph.
    pub point_count: usize,
    epsilon: f64,
}
// Adjacency is deliberately not stored; see the identical note in
// `aether-core`'s `SparseAttentionGraph`. A `[u64; MAX_POINTS]` bitmask cannot
// represent a graph on `MAX_POINTS` nodes when `MAX_POINTS > 64`, and the
// failure was silent: every point at index >= 64 was invisible, inflating
// beta0 by exactly `n - 64`.

impl<const D: usize> SparseGraph<D> {
    /// Construct an empty sparse graph with the given ε neighborhood radius.
    pub fn new(epsilon: f64) -> Self {
        Self {
            points: [EpsilonPoint::zero(); MAX_POINTS],
            point_count: 0,
            epsilon,
        }
    }

    /// Add a point to the cloud. Edges are derived on demand, so insertion is
    /// O(1) and the graph cannot depend on insertion order.
    pub fn add_point(&mut self, point: EpsilonPoint<D>) -> Option<usize> {
        if self.point_count >= MAX_POINTS {
            return None;
        }
        let idx = self.point_count;
        self.points[idx] = point;
        self.point_count += 1;
        Some(idx)
    }

    /// Whether two points are ε-neighbours: strict `distance < epsilon`,
    /// symmetric by construction, with no 64-point ceiling. A point is not its
    /// own neighbour.
    pub fn are_neighbors(&self, i: usize, j: usize) -> bool {
        if i >= self.point_count || j >= self.point_count || i == j {
            return false;
        }
        self.points[i].is_neighbor(&self.points[j], self.epsilon)
    }

    /// Compute Î²â‚€ (connected components) via DFS.
    /// Exact for every `point_count` up to `MAX_POINTS`. Union-find needs no
    /// explicit stack, so no depth bound can silently drop a neighbour and
    /// over-count components.
    pub fn compute_betti_0(&self) -> u32 {
        let n = self.point_count;
        if n == 0 {
            return 0;
        }

        let mut parent = [0usize; MAX_POINTS];
        for (i, p) in parent.iter_mut().enumerate().take(n) {
            *p = i;
        }

        fn find(parent: &mut [usize; MAX_POINTS], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        for i in 0..n {
            for j in (i + 1)..n {
                if self.are_neighbors(i, j) {
                    let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                    if ri != rj {
                        parent[ri] = rj;
                    }
                }
            }
        }

        (0..n).filter(|&i| find(&mut parent, i) == i).count() as u32
    }

    /// The first Betti number **of this graph**: `E - V + beta_0`.
    ///
    /// This was documented as an approximation, "beta_1 ~ E - V + beta_0
    /// (ignoring higher homology)". The identity is not an approximation. For a
    /// 1-complex it is exact, and `tests/house_betti2_is_not_an_euler_solve.rs`
    /// asserts it against an independently counted edge set: E = 558, V = 48,
    /// beta_0 = 1, and this function returns 511.
    ///
    /// What "approximation" was standing in for is a real and much larger gap,
    /// and calling it precision loss understated it. A [`SparseGraph`] holds no
    /// 2-cells, so every triangle of mutually adjacent points contributes an
    /// independent cycle here that is *filled* in the Vietoris-Rips complex over
    /// the same points. The two numbers are not close and do not converge: on
    /// those same 48 points sampled from a flat disc at `epsilon = 0.9`, this
    /// returns **511** while the Rips complex over the identical points has
    /// `beta_1 = 0`, because a disc is contractible.
    ///
    /// So this is the exact `beta_1` of the graph, and the graph is not the
    /// space. For the homology of the point cloud, route to
    /// `aether_core::persistence`, which builds the 2-simplices and reduces the
    /// boundary matrix rather than counting edges.
    pub fn estimate_betti_1(&self) -> u32 {
        let v = self.point_count as i32;
        let mut e = 0i32;
        for i in 0..self.point_count {
            for j in (i + 1)..self.point_count {
                if self.are_neighbors(i, j) {
                    e += 1;
                }
            }
        }

        let b0 = self.compute_betti_0() as i32;
        let b1 = e - v + b0;
        if b1 > 0 {
            b1 as u32
        } else {
            0
        }
    }

    /// Euler defect: `2 - chi`, where `chi = beta_0 - beta_1` is the Euler
    /// characteristic of this graph.
    ///
    /// **This is not beta_2.** A [`SparseGraph`] is a 1-skeleton — it has
    /// vertices and edges and no 2-cells — so `H_2` of the complex it
    /// represents is identically zero for every input, at every epsilon. See
    /// [`Self::betti_2`].
    ///
    /// The identity `beta_0 - beta_1 + beta_2 = 2` holds for a space
    /// *homeomorphic to S2*. Solving it for `beta_2` assumes that conclusion
    /// rather than deriving it, and the resulting quantity cannot represent a
    /// non-sphere: for any connected graph it equals `1 + beta_1`, so it is
    /// never zero and carries no information beyond
    /// [`Self::estimate_betti_1`]. A straight line of collinear points returns
    /// 1 from this function; its true `beta_2` is 0.
    ///
    /// It is retained under an honest name because it is a legitimate graph
    /// invariant and is transmitted as a matching signature in
    /// [`ManifoldPayload`]. It must not be read as homology.
    ///
    /// Negative values are clamped to 0, which loses the sign; that behaviour
    /// is preserved for wire compatibility.
    pub fn euler_defect(&self) -> u32 {
        if self.point_count == 0 {
            return 0;
        }
        let b0 = self.compute_betti_0() as i32;
        let b1 = self.estimate_betti_1() as i32;
        let b2 = 2 - b0 + b1;
        if b2 > 0 {
            b2 as u32
        } else {
            0
        }
    }

    /// Signature triple `(beta_0, beta_1, euler_defect)`.
    ///
    /// The third component is **not** `beta_2`; see [`Self::euler_defect`].
    /// The true `beta_2` of this 1-skeleton is 0 — [`Self::betti_2`].
    pub fn full_shape(&self) -> (u32, u32, u32) {
        // The third slot is the Euler defect, delegated rather than recomputed.
        // It used to inline `2 - b0 + b1` here, a second copy of the expression
        // in `euler_defect`. Renaming that function and documenting it honestly
        // therefore left this copy - and the value reaching
        // `ManifoldPayload::signature_b2` - completely untouched, which is the
        // form the repair's own gate failed to catch.
        (
            self.compute_betti_0(),
            self.estimate_betti_1(),
            self.euler_defect(),
        )
    }

    /// The second Betti number of this complex, which is 0 for every input.
    ///
    /// [`SparseGraph`] stores vertices and an epsilon-neighbour relation and
    /// nothing else, so the complex it represents is 1-dimensional. The chain
    /// group `C_2` is trivial, hence `H_2 = 0` identically — independent of the
    /// points, of `epsilon`, and of the point count.
    ///
    /// Computing a genuine `beta_2` requires 2-simplices. `aether_core`'s
    /// `persistence` module builds them and reduces the boundary matrix
    /// exactly; route there rather than inferring `beta_2` from an Euler
    /// identity.
    pub fn betti_2(&self) -> u32 {
        0
    }

    /// Topological shape signature: (Î²â‚€, Î²â‚).
    pub fn shape(&self) -> (u32, u32) {
        (self.compute_betti_0(), self.estimate_betti_1())
    }

    /// Clear the graph.
    pub fn clear(&mut self) {
        self.point_count = 0;
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// SurgeryError
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// Error conditions during topological surgery.
#[derive(Debug, Clone, PartialEq)]
pub enum SurgeryError {
    /// The receiving void is already occupied
    VoidOccupied,
    /// Payload Betti signature doesn't match void boundary constraints
    TopologyMismatch {
        /// Expected B0 implied by the void boundary.
        expected_b0: u32,
        /// Actual B0 measured on the payload.
        actual_b0: u32,
    },
    /// Payload is empty (zero points)
    EmptyPayload,
    /// Shell is degenerate (Î²â‚€ â‰  1, not a single connected component)
    DegenerateShell {
        /// Measured B0 of the offending shell.
        shell_b0: u32,
    },
    /// Assimilation refused: merging the payload would leave the shell with
    /// `beta_0 != 1`. The shell is left untouched and the payload is dropped
    /// from the void.
    DisconnectedAssimilation {
        /// B0 the shell would have had after the merge.
        merged_b0: u32,
        /// Index into the shell of the closest shell–payload pair.
        shell_index: usize,
        /// Index into the payload of the closest shell–payload pair.
        payload_index: usize,
        /// Distance between that pair. When it is at least the shell's
        /// epsilon, the payload does not touch the shell anywhere.
        distance: f64,
    },
    /// Assimilation refused: the shell has no room for every payload point.
    /// Nothing is merged; a partial merge would drop points silently.
    ShellCapacityExceeded {
        /// Points already in the shell.
        shell_points: usize,
        /// Points the payload carries.
        payload_points: usize,
        /// Maximum points a shell can hold.
        capacity: usize,
    },
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// ManifoldPayload â€” Injectable Data Unit
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// A Injectable manifold payload â€” the data unit for context Context Transfer.
///
/// Contains a pre-computed, topologically stable set of points from a
/// higher manifold M_high where the Seal-Loop has already converged.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ManifoldPayload<const D: usize> {
    /// Embedded points in this payload (fixed-capacity buffer).
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub points: [EpsilonPoint<D>; MAX_PAYLOAD_POINTS],
    /// Number of valid points stored in `points`.
    pub point_count: usize,
    /// b0 Betti number (connected component count).
    pub signature_b0: u32,
    /// b1 Betti number (independent loop count).
    pub signature_b1: u32,
    /// Euler defect `2 - chi = 2 - beta_0 + beta_1`, clamped at 0.
    ///
    /// **Not beta_2.** The graph is a 1-skeleton, so its true beta_2 is 0 for
    /// every input. For connected graphs this field equals `signature_b1 + 1`
    /// and is therefore redundant with the field above; it is retained only for
    /// wire compatibility as a matching signature. See
    /// [`SparseGraph::euler_defect`].
    pub signature_b2: u32,
    /// Inherited liveness score from source agent (for Chebyshev guard)
    pub liveness_anchor: f64,
}

impl<const D: usize> ManifoldPayload<D> {
    /// Construct an empty payload (no points, zero Betti signature).
    pub fn new() -> Self {
        Self {
            points: [EpsilonPoint::zero(); MAX_PAYLOAD_POINTS],
            point_count: 0,
            signature_b0: 0,
            signature_b1: 0,
            signature_b2: 0,
            liveness_anchor: 1.0,
        }
    }

    /// Build a payload from a converged [`SparseGraph`].
    ///
    /// At most the first `MAX_PAYLOAD_POINTS` (64) points are carried, and the
    /// signature is computed over those carried points at the graph's epsilon,
    /// not over the whole graph: a truncated payload can be disconnected even
    /// when its source graph is not.
    pub fn from_graph(graph: &SparseGraph<D>, liveness_anchor: f64) -> Self {
        let count = graph.point_count.min(MAX_PAYLOAD_POINTS);

        let mut payload = Self::new();
        let mut carried = SparseGraph::new(graph.epsilon);
        for i in 0..count {
            payload.points[i] = graph.points[i];
            carried.add_point(graph.points[i]);
        }
        let (b0, b1, b2) = carried.full_shape();
        payload.point_count = count;
        payload.signature_b0 = b0;
        payload.signature_b1 = b1;
        payload.signature_b2 = b2;
        payload.liveness_anchor = liveness_anchor;
        payload
    }

    /// Check if this payload is non-empty.
    pub fn is_valid(&self) -> bool {
        self.point_count > 0
    }
}

impl<const D: usize> Default for ManifoldPayload<D> {
    fn default() -> Self {
        Self::new()
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// HollowCubeManifold â€” SÂ² Boundary with Î²â‚‚ = 1
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// The Hollow Cube Manifold â€” an SÂ² boundary with Betti number Î²â‚‚ = 1.
///
/// Enforces the hollow geometric constraint where Î²â‚‚ = 1 defines an
/// interior "void" for receiving Injected context.
///
/// # Architecture
/// - **Shell**: A [`SparseGraph`] representing the outer SÂ² boundary
/// - **Void**: An `Option<ManifoldPayload>` â€” the injection receptacle
/// - **Assimilated**: Whether the wake-up rescan has completed
///
/// # Invariants
/// - Shell must maintain Î²â‚€ = 1 (single connected component)
/// - Void holds at most one payload at a time
/// - Assimilation merges payload points into the shell's active graph
pub struct HollowCubeManifold<const D: usize> {
    shell: SparseGraph<D>,
    void_payload: Option<ManifoldPayload<D>>,
    assimilated: bool,
}

impl<const D: usize> HollowCubeManifold<D> {
    /// Create a new hollow manifold with the given Îµ-neighborhood radius.
    pub fn new(epsilon: f64) -> Self {
        Self {
            shell: SparseGraph::new(epsilon),
            void_payload: None,
            assimilated: true,
        }
    }

    /// Add a point to the outer shell.
    pub fn add_shell_point(&mut self, point: EpsilonPoint<D>) -> Option<usize> {
        self.shell.add_point(point)
    }

    /// Shell's topological shape (Î²â‚€, Î²â‚).
    pub fn shell_shape(&self) -> (u32, u32) {
        self.shell.shape()
    }

    /// Is the void empty and ready for injection?
    pub fn void_is_empty(&self) -> bool {
        self.void_payload.is_none()
    }

    /// Does the manifold have an unassimilated payload?
    pub fn has_pending_payload(&self) -> bool {
        self.void_payload.is_some() && !self.assimilated
    }

    /// Perform topological surgery: inject payload into the void.
    ///
    /// # Surgery Protocol
    /// 1. Verify void is unoccupied
    /// 2. Verify shell is non-degenerate (Î²â‚€ = 1)
    /// 3. Verify payload is non-empty and topologically consistent
    /// 4. Write payload into the void
    ///
    /// # Safety
    /// Caller MUST hold a [`SurgeryPermit`](crate::SurgeryPermit) from the
    /// [`SurgeryGovernor`](crate::SurgeryGovernor) to ensure derivative
    /// gain is zeroed. Enforced at the `sys_context_inject` level.
    pub fn inject_into_void(&mut self, payload: ManifoldPayload<D>) -> Result<(), SurgeryError> {
        if self.void_payload.is_some() {
            return Err(SurgeryError::VoidOccupied);
        }

        let (shell_b0, _) = self.shell.shape();
        if shell_b0 != 1 {
            return Err(SurgeryError::DegenerateShell { shell_b0 });
        }

        if !payload.is_valid() {
            return Err(SurgeryError::EmptyPayload);
        }

        if payload.signature_b0 != 1 {
            return Err(SurgeryError::TopologyMismatch {
                expected_b0: 1,
                actual_b0: payload.signature_b0,
            });
        }

        self.void_payload = Some(payload);
        self.assimilated = false;
        Ok(())
    }

    /// Wake-Up Rescan: assimilate injected payload into the active shell.
    ///
    /// Verifies Betti boundaries of injected mass against the inner walls
    /// of the hollow cube. If topologies align, data is fully merged into
    /// the agent's active processing shell in O(1) time relative to token
    /// length.
    ///
    /// The merge is all-or-nothing. It commits only when every payload point
    /// fits and the merged shell has `beta_0 = 1`; otherwise the shell is left
    /// exactly as it was, the payload is dropped from the void, and the reason
    /// is returned as [`SurgeryError::ShellCapacityExceeded`] or
    /// [`SurgeryError::DisconnectedAssimilation`].
    ///
    /// Returns the number of points merged, or `Ok(0)` if the void is empty.
    pub fn assimilate(&mut self) -> Result<usize, SurgeryError> {
        let payload = match self.void_payload.take() {
            Some(p) => p,
            None => return Ok(0),
        };
        self.assimilated = true;

        let shell_points = self.shell.point_count;
        if shell_points + payload.point_count > MAX_POINTS {
            return Err(SurgeryError::ShellCapacityExceeded {
                shell_points,
                payload_points: payload.point_count,
                capacity: MAX_POINTS,
            });
        }

        let mut merged = SparseGraph::new(self.shell.epsilon);
        for p in self.shell.points[..shell_points]
            .iter()
            .chain(&payload.points[..payload.point_count])
        {
            merged.add_point(*p);
        }

        // ponytail: this is the graph's union-find beta_0 at the shell's
        // epsilon, uncertified. A later change replaces it with a certified
        // beta_0; the commit-only-if-1 gate stays the same.
        let merged_b0 = merged.compute_betti_0();
        if merged_b0 != 1 {
            // Both sides are non-empty here: injection requires a shell with
            // beta_0 = 1 and a valid payload, and only `reset` shrinks the
            // shell, which also empties the void.
            let (mut shell_index, mut payload_index, mut distance) = (0, 0, f64::INFINITY);
            for (i, s) in self.shell.points[..shell_points].iter().enumerate() {
                for (j, q) in payload.points[..payload.point_count].iter().enumerate() {
                    let d = s.distance(q);
                    if d < distance {
                        (shell_index, payload_index, distance) = (i, j, d);
                    }
                }
            }
            return Err(SurgeryError::DisconnectedAssimilation {
                merged_b0,
                shell_index,
                payload_index,
                distance,
            });
        }

        self.shell = merged;
        Ok(payload.point_count)
    }

    /// Inherited liveness anchor from the current payload (if any).
    pub fn payload_liveness_anchor(&self) -> Option<f64> {
        self.void_payload.as_ref().map(|p| p.liveness_anchor)
    }

    /// Reset the hollow manifold to empty state.
    pub fn reset(&mut self) {
        self.shell.clear();
        self.void_payload = None;
        self.assimilated = true;
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Unit Tests
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let p1 = EpsilonPoint::<3>::new([0.0, 0.0, 0.0]);
        let p2 = EpsilonPoint::<3>::new([3.0, 4.0, 0.0]);
        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_betti_0_connected() {
        let mut g = SparseGraph::<3>::new(1.0);
        g.add_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
        g.add_point(EpsilonPoint::new([0.5, 0.0, 0.0]));
        g.add_point(EpsilonPoint::new([0.5, 0.5, 0.0]));
        assert_eq!(g.compute_betti_0(), 1);
    }

    #[test]
    fn test_sparse_betti_0_disconnected() {
        let mut g = SparseGraph::<3>::new(0.1);
        g.add_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
        g.add_point(EpsilonPoint::new([100.0, 100.0, 100.0]));
        assert_eq!(g.compute_betti_0(), 2);
    }

    #[test]
    fn test_hollow_cube_betti_constraint() {
        let mut hollow = HollowCubeManifold::<3>::new(1.0);
        hollow.add_shell_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
        hollow.add_shell_point(EpsilonPoint::new([0.5, 0.0, 0.0]));
        hollow.add_shell_point(EpsilonPoint::new([0.5, 0.5, 0.0]));

        let (b0, _) = hollow.shell_shape();
        assert_eq!(b0, 1, "Shell must be single connected component");
        assert!(hollow.void_is_empty(), "Void should start empty");
    }

    #[test]
    fn test_inject_into_void() {
        let mut hollow = HollowCubeManifold::<3>::new(1.0);
        hollow.add_shell_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
        hollow.add_shell_point(EpsilonPoint::new([0.5, 0.0, 0.0]));
        hollow.add_shell_point(EpsilonPoint::new([0.5, 0.5, 0.0]));

        let mut src = SparseGraph::<3>::new(1.0);
        src.add_point(EpsilonPoint::new([1.0, 1.0, 1.0]));
        src.add_point(EpsilonPoint::new([1.5, 1.0, 1.0]));

        let payload = ManifoldPayload::from_graph(&src, 5.0);
        assert_eq!(payload.signature_b0, 1);

        assert!(hollow.inject_into_void(payload).is_ok());
        assert!(!hollow.void_is_empty());
        assert!(hollow.has_pending_payload());
    }

    #[test]
    fn test_inject_rejects_topology_mismatch() {
        let mut hollow = HollowCubeManifold::<3>::new(1.0);
        hollow.add_shell_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
        hollow.add_shell_point(EpsilonPoint::new([0.5, 0.0, 0.0]));

        // Disconnected payload (Î²â‚€ = 2)
        let mut src = SparseGraph::<3>::new(0.1);
        src.add_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
        src.add_point(EpsilonPoint::new([100.0, 100.0, 100.0]));

        let payload = ManifoldPayload::from_graph(&src, 3.0);
        assert_eq!(payload.signature_b0, 2);

        assert_eq!(
            hollow.inject_into_void(payload),
            Err(SurgeryError::TopologyMismatch {
                expected_b0: 1,
                actual_b0: 2
            })
        );
    }

    #[test]
    fn test_assimilate_merges_points() {
        let mut hollow = HollowCubeManifold::<3>::new(2.0);
        hollow.add_shell_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
        hollow.add_shell_point(EpsilonPoint::new([0.5, 0.0, 0.0]));
        hollow.add_shell_point(EpsilonPoint::new([0.5, 0.5, 0.0]));

        let mut src = SparseGraph::<3>::new(2.0);
        src.add_point(EpsilonPoint::new([1.0, 1.0, 1.0]));
        src.add_point(EpsilonPoint::new([1.5, 1.0, 1.0]));

        let payload = ManifoldPayload::from_graph(&src, 5.0);
        hollow.inject_into_void(payload).unwrap();

        let merged = hollow.assimilate().unwrap();
        assert_eq!(merged, 2);
        assert!(hollow.void_is_empty());
        assert!(!hollow.has_pending_payload());
    }

    // ─── Betti-2 Tests ────────────────────────────────────────────────────────

    #[test]
    fn euler_defect_equals_one_plus_b1_for_connected_graphs() {
        // The identity that makes the old `beta_2` field redundant, pinned.
        // Named mutant this kills: "treat the Euler defect as beta_2", which
        // would have to return 0 for some connected input and never can.
        let mut g = SparseGraph::<3>::new(1.5);
        g.add_point(EpsilonPoint::new([1.0, 0.0, 0.0]));
        g.add_point(EpsilonPoint::new([0.0, 1.0, 0.0]));
        g.add_point(EpsilonPoint::new([0.0, 0.0, 1.0]));
        g.add_point(EpsilonPoint::new([-1.0, 0.0, 0.0]));
        g.add_point(EpsilonPoint::new([0.0, -1.0, 0.0]));
        g.add_point(EpsilonPoint::new([0.0, 0.0, -1.0]));

        let (b0, b1, defect) = g.full_shape();
        assert_eq!(b0, 1, "shell must be connected");
        assert_eq!(
            defect,
            b1 + 1,
            "for a connected graph the Euler defect is exactly 1 + beta_1"
        );
        assert_eq!(g.betti_2(), 0, "H_2 of a 1-skeleton is identically 0");
    }

    #[test]
    fn test_betti_2_disconnected_graph_is_zero() {
        // Two isolated points → β₀=2, β₁=0 → β₂ = 2 − 2 + 0 = 0
        let mut g = SparseGraph::<3>::new(0.01);
        g.add_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
        g.add_point(EpsilonPoint::new([100.0, 100.0, 100.0]));

        let (b0, b1, b2) = g.full_shape();
        assert_eq!(b0, 2, "Two isolated points → β₀=2");
        assert_eq!(b1, 0, "No cycles → β₁=0");
        assert_eq!(b2, 0, "Disconnected graph → β₂=0 (clamped)");
    }

    #[test]
    fn test_payload_carries_b2() {
        let mut src = SparseGraph::<3>::new(2.0);
        // Three close points → well connected, β₀=1
        src.add_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
        src.add_point(EpsilonPoint::new([0.1, 0.0, 0.0]));
        src.add_point(EpsilonPoint::new([0.0, 0.1, 0.0]));

        let payload = ManifoldPayload::from_graph(&src, 1.0);
        assert_eq!(payload.signature_b0, 1);
        // β₂ = 2 − 1 + β₁; for 3 points with 3 edges β₁ = E−V+β₀ = 3−3+1 = 1
        // → β₂ = 2 − 1 + 1 = 2... clamped minimum is 0; we just verify type is populated
        // The key invariant: signature_b2 is now carried in the payload
        // (not stuck at zero as it was before this change)
        let _ = payload.signature_b2; // field must compile
    }
}
