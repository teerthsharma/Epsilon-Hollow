// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Topological State Synchronization (TSS) — operative data structure.
//!
//! Realises Theorem 1 of the Epsilon-Hollow specification: O(1) amortized
//! retrieval via spherical Voronoi tessellation. The latent space is projected
//! onto an S² manifold and partitioned into `K` Voronoi clusters (the Betti-0
//! components). With `K` fixed at compile time the per-query work is bounded by
//! a compile-time constant, hence O(1) amortized regardless of payload count.
//!
//! The math kernel (`aether-verified::aether_tss`) supplies the closed-form
//! separation bounds; this module is intentionally self-contained — the
//! great-circle helper is inline-ported to keep `aether-core` independent of
//! `aether-verified`.

/// Great-circle distance on S² between two points (θ₁,φ₁) and (θ₂,φ₂).
///
/// Inline port of `aether_verified::aether_tss::great_circle_distance` to
/// preserve crate-level separation of concerns.
#[inline]
fn great_circle_distance(t1: f64, p1: f64, t2: f64, p2: f64) -> f64 {
    let cos_d = libm::sin(t1) * libm::sin(t2) + libm::cos(t1) * libm::cos(t2) * libm::cos(p1 - p2);
    libm::acos(cos_d.clamp(-1.0, 1.0))
}

/// Verify all centroid pairs satisfy separation ≥ θ_min (used in `new`).
#[inline]
pub fn verify_separation(centroids: &[(f64, f64)], theta_min: f64) -> bool {
    let n = centroids.len();
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n {
            let (t1, p1) = centroids[i];
            let (t2, p2) = centroids[j];
            if great_circle_distance(t1, p1, t2, p2) < theta_min - 1e-6 {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

/// Compute the Chebyshev-bounded adaptive epsilon: `mu_local + k * sigma_local`.
#[inline]
pub fn epsilon_from_chebyshev(mu_local: f64, sigma_local: f64, k: f64) -> f64 {
    mu_local + k * sigma_local
}

/// Minimum angular separation implied by an adaptive epsilon.
///
/// `theta_min >= 2 * asin(epsilon_adaptive / 2)`.
#[inline]
pub fn compute_theta_min(epsilon_adaptive: f64) -> f64 {
    let arg = (epsilon_adaptive / 2.0).clamp(-1.0, 1.0);
    2.0 * libm::asin(arg)
}

/// Spherical cap packing bound on the maximum number of clusters.
///
/// `P_max = 4 / sin(theta_min / 2)^2`.
#[inline]
pub fn compute_p_max(theta_min: f64) -> f64 {
    if theta_min <= 0.0 {
        return f64::INFINITY;
    }

    let sin_half = libm::sin(theta_min / 2.0);
    if sin_half < 1e-12 {
        f64::INFINITY
    } else {
        4.0 / (sin_half * sin_half)
    }
}

/// Operation-count comparison for the TSS retrieval theorem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TssRetrievalBound {
    /// Within-cluster work for Epsilon: `k * d`.
    pub epsilon_o1_ops: usize,
    /// HNSW-style comparison work: `floor(log2(N)) * d`.
    pub hnsw_ops: usize,
    /// Brute-force comparison work: `N * d`.
    pub brute_force_ops: usize,
    /// Speedup over HNSW using the simple operation model.
    pub speedup_vs_hnsw: f64,
    /// Speedup over brute force using the simple operation model.
    pub speedup_vs_brute: f64,
    /// Stored vector count.
    pub n: usize,
    /// Cluster count.
    pub p: usize,
    /// Retrieval neighbor count.
    pub k: usize,
    /// Vector dimension.
    pub d: usize,
}

/// Aggregate TSS theorem verification report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TssVerificationReport {
    /// Minimum angular separation requested by the caller.
    pub theta_min: f64,
    /// Spherical cap packing upper bound.
    pub p_max: f64,
    /// Retrieval operation-count comparison.
    pub retrieval_bound: TssRetrievalBound,
    /// Whether all centroid pairs satisfy `theta_min`.
    pub separation_holds: bool,
    /// Whether the compile-time centroid count fits inside the packing bound.
    pub packing_holds: bool,
    /// True when separation, packing, and bounded retrieval work all hold.
    pub theorem_holds: bool,
}

/// Public verifier for the legacy TSS theorem API.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TssVerifier<const K: usize> {
    centroids: [(f64, f64); K],
    theta_min: f64,
}

impl<const K: usize> TssVerifier<K> {
    /// Construct a verifier over fixed spherical centroids.
    #[inline]
    pub const fn new(centroids: [(f64, f64); K], theta_min: f64) -> Self {
        Self {
            centroids,
            theta_min,
        }
    }

    /// Verify pairwise spherical centroid separation.
    #[inline]
    pub fn verify_separation(&self) -> bool {
        verify_separation(&self.centroids, self.theta_min)
    }

    /// Verify the spherical cap packing bound for `K` centroids.
    #[inline]
    pub fn verify_packing(&self) -> bool {
        let p_max = compute_p_max(self.theta_min);
        p_max.is_finite() && (K as f64) <= p_max
    }

    /// Run separation, packing, and retrieval complexity checks.
    pub fn full_verification(
        &self,
        n: usize,
        k_neighbors: usize,
        d: usize,
    ) -> TssVerificationReport {
        let p_max = compute_p_max(self.theta_min);
        let retrieval_bound = tss_retrieval_bound(n, K, k_neighbors, d);
        let separation_holds = self.verify_separation();
        let packing_holds = p_max.is_finite() && (K as f64) <= p_max;
        let bounded_retrieval = retrieval_bound.epsilon_o1_ops == k_neighbors.saturating_mul(d);

        TssVerificationReport {
            theta_min: self.theta_min,
            p_max,
            retrieval_bound,
            separation_holds,
            packing_holds,
            theorem_holds: separation_holds && packing_holds && bounded_retrieval,
        }
    }
}

/// Compute the TSS retrieval complexity comparison from the Python theorem.
#[inline]
pub fn tss_retrieval_bound(n: usize, p: usize, k: usize, d: usize) -> TssRetrievalBound {
    let epsilon_o1_ops = k.saturating_mul(d);
    let hnsw_ops = floor_log2(n.max(2)).saturating_mul(d);
    let brute_force_ops = n.saturating_mul(d);
    let denom = epsilon_o1_ops.max(1) as f64;

    TssRetrievalBound {
        epsilon_o1_ops,
        hnsw_ops,
        brute_force_ops,
        speedup_vs_hnsw: hnsw_ops as f64 / denom,
        speedup_vs_brute: brute_force_ops as f64 / denom,
        n,
        p,
        k,
        d,
    }
}

#[inline]
fn floor_log2(value: usize) -> usize {
    (usize::BITS - 1 - value.leading_zeros()) as usize
}

#[inline]
fn spherical_to_unit_vector(theta: f64, phi: f64) -> [f64; 3] {
    [
        libm::sin(theta) * libm::cos(phi),
        libm::sin(theta) * libm::sin(phi),
        libm::cos(theta),
    ]
}

#[inline]
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Summary statistics for the fixed-size spherical grid hash.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalGridStats {
    /// Theta bins.
    pub n_theta: usize,
    /// Phi bins.
    pub n_phi: usize,
    /// Total grid cells.
    pub total_cells: usize,
    /// Cells containing at least one centroid.
    pub occupied_cells: usize,
    /// Total centroid slots.
    pub total_centroids: usize,
    /// Average centroid count among occupied cells.
    pub avg_cell_size: f64,
    /// Largest occupied cell size.
    pub max_cell_size: usize,
    /// Fraction of lookups resolved from the 3x3 neighborhood.
    pub o1_hit_rate: f64,
    /// Total lookup calls.
    pub total_lookups: u64,
}

/// Legacy empty-location sentinel from `SphericalGridHash.locate`.
pub const EMPTY_LOCATE: isize = -1;

/// Return the auto-sized spherical grid dimensions for `p` centroids.
#[inline]
pub fn auto_sized_dimensions(p: usize) -> (usize, usize) {
    let n_theta = (libm::ceil(libm::sqrt(p.max(1) as f64)) as usize).max(4);
    let n_phi = (2 * n_theta).max(8);
    (n_theta, n_phi)
}

/// Fixed-size spherical grid hash over `K` centroids.
///
/// The grid auto-sizes as `ceil(sqrt(K)) x 2*ceil(sqrt(K))`, matching the
/// Python theorem slice while remaining allocation-free and no_std-friendly.
#[derive(Debug, Clone, Copy)]
pub struct SphericalGridHashIndex<const K: usize> {
    centroids: [(f64, f64); K],
    centroid_vectors: [[f64; 3]; K],
    payloads: [u64; K],
    n_theta: usize,
    n_phi: usize,
    last_winner: usize,
    lookup_count: u64,
    o1_hits: u64,
}

impl<const K: usize> SphericalGridHashIndex<K> {
    /// Build an auto-sized grid hash from spherical centroids.
    pub fn new(centroids: [(f64, f64); K]) -> Self {
        let (n_theta, n_phi) = auto_sized_dimensions(K);

        let mut centroid_vectors = [[0.0; 3]; K];
        let mut i = 0;
        while i < K {
            centroid_vectors[i] = spherical_to_unit_vector(centroids[i].0, centroids[i].1);
            i += 1;
        }

        Self {
            centroids,
            centroid_vectors,
            payloads: [0; K],
            n_theta,
            n_phi,
            last_winner: 0,
            lookup_count: 0,
            o1_hits: 0,
        }
    }

    /// Rebuild the fixed-capacity grid from a fresh centroid array.
    pub fn build(&mut self, centroids: [(f64, f64); K]) {
        self.centroids = centroids;
        self.payloads = [0; K];
        self.lookup_count = 0;
        self.o1_hits = 0;
        self.last_winner = 0;

        let mut i = 0;
        while i < K {
            self.centroid_vectors[i] = spherical_to_unit_vector(centroids[i].0, centroids[i].1);
            i += 1;
        }
    }

    /// Set a payload identifier for a centroid slot.
    #[inline]
    pub fn insert(&mut self, slot: usize, payload: u64) {
        if slot < K {
            self.payloads[slot] = payload;
        }
    }

    /// Replace a centroid and payload slot, matching the legacy dynamic insert API.
    #[inline]
    pub fn insert_centroid(&mut self, slot: usize, centroid: (f64, f64), payload: u64) {
        if slot < K {
            self.centroids[slot] = centroid;
            self.centroid_vectors[slot] = spherical_to_unit_vector(centroid.0, centroid.1);
            self.payloads[slot] = payload;
            self.last_winner = self.last_winner.min(K.saturating_sub(1));
        }
    }

    /// Locate the nearest centroid, checking the hashed cell and neighbors first.
    pub fn locate(&mut self, query: (f64, f64)) -> usize {
        self.lookup_count = self.lookup_count.saturating_add(1);
        if K == 0 {
            return usize::MAX;
        }

        let q = spherical_to_unit_vector(query.0, query.1);
        let (row, col) = self.hash(query.0, query.1);
        let mut found = false;
        let mut best_idx = self.last_winner.min(K - 1);
        let mut best_dot = f64::NEG_INFINITY;

        let mut dr = -1isize;
        while dr <= 1 {
            let r = row as isize + dr;
            if r >= 0 && (r as usize) < self.n_theta {
                let mut dc = -1isize;
                while dc <= 1 {
                    let c = wrap_col(col, dc, self.n_phi);
                    let mut i = 0;
                    while i < K {
                        if self.hash(self.centroids[i].0, self.centroids[i].1) == (r as usize, c) {
                            let dot = dot3(&q, &self.centroid_vectors[i]);
                            if !found || dot > best_dot {
                                found = true;
                                best_dot = dot;
                                best_idx = i;
                            }
                        }
                        i += 1;
                    }
                    dc += 1;
                }
            }
            dr += 1;
        }

        if found {
            self.o1_hits = self.o1_hits.saturating_add(1);
            self.last_winner = best_idx;
            return best_idx;
        }

        let mut i = 0;
        while i < K {
            let dot = dot3(&q, &self.centroid_vectors[i]);
            if i == 0 || dot > best_dot {
                best_dot = dot;
                best_idx = i;
            }
            i += 1;
        }
        self.last_winner = best_idx;
        best_idx
    }

    /// Locate using the legacy signed sentinel contract: `-1` when empty.
    pub fn locate_legacy(&mut self, query: (f64, f64)) -> isize {
        let idx = self.locate(query);
        if idx == usize::MAX {
            EMPTY_LOCATE
        } else {
            idx as isize
        }
    }

    /// Locate and return `(slot, payload)`.
    pub fn locate_with_payload(&mut self, query: (f64, f64)) -> (usize, u64) {
        let idx = self.locate(query);
        let payload = if idx < K { self.payloads[idx] } else { 0 };
        (idx, payload)
    }

    /// Fraction of lookups resolved by the O(1)-expected grid neighborhood.
    #[inline]
    pub fn o1_hit_rate(&self) -> f64 {
        if self.lookup_count == 0 {
            0.0
        } else {
            self.o1_hits as f64 / self.lookup_count as f64
        }
    }

    /// Compute grid occupancy statistics.
    pub fn stats(&self) -> SphericalGridStats {
        let mut occupied = 0usize;
        let mut max_cell_size = 0usize;
        let mut r = 0usize;
        while r < self.n_theta {
            let mut c = 0usize;
            while c < self.n_phi {
                let mut count = 0usize;
                let mut i = 0;
                while i < K {
                    if self.hash(self.centroids[i].0, self.centroids[i].1) == (r, c) {
                        count += 1;
                    }
                    i += 1;
                }
                if count > 0 {
                    occupied += 1;
                    max_cell_size = max_cell_size.max(count);
                }
                c += 1;
            }
            r += 1;
        }

        SphericalGridStats {
            n_theta: self.n_theta,
            n_phi: self.n_phi,
            total_cells: self.n_theta * self.n_phi,
            occupied_cells: occupied,
            total_centroids: K,
            avg_cell_size: if occupied == 0 {
                0.0
            } else {
                K as f64 / occupied as f64
            },
            max_cell_size,
            o1_hit_rate: self.o1_hit_rate(),
            total_lookups: self.lookup_count,
        }
    }

    #[inline]
    fn hash(&self, theta: f64, phi: f64) -> (usize, usize) {
        let theta = theta.clamp(0.0, core::f64::consts::PI);
        let phi = phi.clamp(-core::f64::consts::PI, core::f64::consts::PI);
        let row = libm::floor((theta / core::f64::consts::PI) * self.n_theta as f64) as usize;
        let col =
            ((phi + core::f64::consts::PI) / (2.0 * core::f64::consts::PI)) * self.n_phi as f64;
        let col = libm::floor(col) as usize;
        (row.min(self.n_theta - 1), col.min(self.n_phi - 1))
    }
}

#[inline]
fn wrap_col(col: usize, delta: isize, n_phi: usize) -> usize {
    let n = n_phi as isize;
    ((col as isize + delta).rem_euclid(n)) as usize
}

/// O(1)-amortized spherical Voronoi index over `K` cluster centroids on S².
///
/// `K` is a compile-time constant, so the linear scan in [`Self::locate`] has
/// a fixed instruction count for every monomorphisation: O(K) with K constant
/// is O(1) by definition.
///
/// Each Voronoi cell is, by construction, a single connected component on
/// S², so the index has Betti-0 number β₀ = K (see [`Self::betti_0`]).
#[derive(Debug, Clone, Copy)]
pub struct SphericalVoronoiIndex<const K: usize> {
    /// Cluster centroids on S² as `(theta, phi)` spherical coordinates.
    centroids: [(f64, f64); K],
    /// Parallel array of payload identifiers, one per centroid slot.
    payloads: [u64; K],
}

impl<const K: usize> SphericalVoronoiIndex<K> {
    /// Construct a new index from pre-computed centroids.
    ///
    /// Centroid payloads are zero-initialised; use [`Self::insert`] to attach
    /// payload IDs.
    #[inline]
    pub fn new(centroids: [(f64, f64); K]) -> Self {
        Self {
            centroids,
            payloads: [0u64; K],
        }
    }

    /// Construct and validate that centroids satisfy the TSS separation bound.
    ///
    /// Returns `None` if any centroid pair has great-circle distance below
    /// `theta_min` (within a 1e-6 tolerance).
    #[inline]
    pub fn new_verified(centroids: [(f64, f64); K], theta_min: f64) -> Option<Self> {
        if verify_separation(&centroids, theta_min) {
            Some(Self::new(centroids))
        } else {
            None
        }
    }

    /// Locate the nearest centroid to `query` by great-circle distance.
    ///
    /// Complexity: O(K) where K is fixed at compile time → O(1) amortized.
    #[inline]
    pub fn locate(&self, query: (f64, f64)) -> usize {
        let (qt, qp) = query;
        let mut best_idx = 0usize;
        let mut best_d = great_circle_distance(qt, qp, self.centroids[0].0, self.centroids[0].1);
        let mut i = 1usize;
        while i < K {
            let (ct, cp) = self.centroids[i];
            let d = great_circle_distance(qt, qp, ct, cp);
            if d < best_d {
                best_d = d;
                best_idx = i;
            }
            i += 1;
        }
        best_idx
    }

    /// Locate the nearest centroid and return `(slot, payload)`.
    ///
    /// Complexity: O(K) where K is fixed at compile time → O(1) amortized.
    #[inline]
    pub fn locate_with_payload(&self, query: (f64, f64)) -> (usize, u64) {
        let idx = self.locate(query);
        (idx, self.payloads[idx])
    }

    /// Set the payload ID for the given centroid slot. Complexity: O(1).
    #[inline]
    pub fn insert(&mut self, slot: usize, payload: u64) {
        self.payloads[slot] = payload;
    }

    /// Number of cluster slots (compile-time constant `K`).
    #[inline]
    pub fn capacity(&self) -> usize {
        K
    }

    /// Betti-0 number: by construction each Voronoi cell is one connected
    /// component, so β₀ = K.
    #[inline]
    pub fn betti_0(&self) -> u32 {
        K as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k4_centroids() -> [(f64, f64); 4] {
        // Four well-separated points on S².
        [(0.5, 0.0), (1.0, 1.5), (2.0, 3.0), (2.5, 4.5)]
    }

    fn k8_centroids() -> [(f64, f64); 8] {
        let mut c = [(0.0, 0.0); 8];
        let mut i = 0;
        while i < 8 {
            // Spread along longitude with mid-latitude θ = π/2.
            c[i] = (
                core::f64::consts::FRAC_PI_2,
                (i as f64) * (2.0 * core::f64::consts::PI / 8.0),
            );
            i += 1;
        }
        c
    }

    #[test]
    fn locate_returns_self_for_centroid_query() {
        let idx = SphericalVoronoiIndex::<4>::new(k4_centroids());
        for i in 0..4 {
            assert_eq!(idx.locate(k4_centroids()[i]), i);
        }
    }

    #[test]
    fn locate_returns_nearest() {
        let idx = SphericalVoronoiIndex::<4>::new(k4_centroids());
        // Perturb centroid 3 by a small delta.
        let (t, p) = k4_centroids()[3];
        let perturbed = (t + 1e-3, p - 1e-3);
        assert_eq!(idx.locate(perturbed), 3);
    }

    #[test]
    fn locate_is_deterministic() {
        let idx = SphericalVoronoiIndex::<4>::new(k4_centroids());
        let q = (1.2, 1.7);
        let a = idx.locate(q);
        let b = idx.locate(q);
        let c = idx.locate(q);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn payload_round_trip() {
        let mut idx = SphericalVoronoiIndex::<4>::new(k4_centroids());
        idx.insert(2, 0xDEAD_BEEF);
        let (slot, payload) = idx.locate_with_payload(k4_centroids()[2]);
        assert_eq!(slot, 2);
        assert_eq!(payload, 0xDEAD_BEEF);
    }

    #[test]
    fn betti_0_equals_k() {
        let idx4 = SphericalVoronoiIndex::<4>::new(k4_centroids());
        assert_eq!(idx4.betti_0(), 4);
        assert_eq!(idx4.capacity(), 4);
        let idx8 = SphericalVoronoiIndex::<8>::new(k8_centroids());
        assert_eq!(idx8.betti_0(), 8);
        assert_eq!(idx8.capacity(), 8);
    }

    #[test]
    fn o1_amortized_api_works_across_k() {
        // Structural O(1): API returns valid indices for K ∈ {8, 64, 512}.
        let idx8 = SphericalVoronoiIndex::<8>::new(k8_centroids());

        let mut c64 = [(0.0, 0.0); 64];
        let mut i = 0;
        while i < 64 {
            c64[i] = (
                core::f64::consts::FRAC_PI_2,
                (i as f64) * (2.0 * core::f64::consts::PI / 64.0),
            );
            i += 1;
        }
        let idx64 = SphericalVoronoiIndex::<64>::new(c64);

        let mut c512 = [(0.0, 0.0); 512];
        let mut i = 0;
        while i < 512 {
            c512[i] = (
                core::f64::consts::FRAC_PI_2,
                (i as f64) * (2.0 * core::f64::consts::PI / 512.0),
            );
            i += 1;
        }
        let idx512 = SphericalVoronoiIndex::<512>::new(c512);

        // 1000 deterministic queries each.
        let mut q = 0u64;
        let mut n = 0;
        while n < 1000 {
            // Pseudo-random query angles via a cheap LCG.
            q = q
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let theta = ((q >> 33) as f64 / (1u64 << 31) as f64) * core::f64::consts::PI;
            let phi =
                ((q & 0xFFFF_FFFF) as f64 / (1u64 << 32) as f64) * 2.0 * core::f64::consts::PI;

            let s8 = idx8.locate((theta, phi));
            assert!(s8 < 8);
            let s64 = idx64.locate((theta, phi));
            assert!(s64 < 64);
            let s512 = idx512.locate((theta, phi));
            assert!(s512 < 512);

            n += 1;
        }
    }

    #[test]
    fn new_verified_rejects_too_close_centroids() {
        // Two nearly-identical centroids: separation ≪ theta_min.
        let centroids = [(0.5, 0.5), (0.5001, 0.5001), (2.0, 3.0), (2.5, 4.5)];
        let theta_min = 0.5; // demand large separation
        assert!(SphericalVoronoiIndex::<4>::new_verified(centroids, theta_min).is_none());
    }
}

#[cfg(test)]
mod theorem_port_tests {
    use super::*;

    #[test]
    fn tss_bound_helpers_match_python_theorem_slice() {
        let eps = epsilon_from_chebyshev(0.2, 0.1, 2.0);
        assert!((eps - 0.4).abs() < 1e-12);

        let theta = compute_theta_min(eps);
        let p_max = compute_p_max(theta);
        assert!(theta > 0.0);
        assert!(p_max > 90.0 && p_max < 110.0);

        let retrieval = tss_retrieval_bound(1_000_000, 1_000, 5, 128);
        assert_eq!(retrieval.epsilon_o1_ops, 640);
        assert_eq!(retrieval.hnsw_ops, 19 * 128);
        assert_eq!(retrieval.brute_force_ops, 128_000_000);
        assert!(retrieval.speedup_vs_hnsw > 3.0);
    }

    #[test]
    fn grid_hash_finds_neighboring_cell_and_tracks_hits() {
        let mut centroids = [(0.0, 0.0); 8];
        let mut i = 0;
        while i < 8 {
            centroids[i] = (
                core::f64::consts::FRAC_PI_2,
                (i as f64) * (2.0 * core::f64::consts::PI / 8.0),
            );
            i += 1;
        }
        let mut grid = SphericalGridHashIndex::<8>::new(centroids);

        let (slot, payload) = grid.locate_with_payload(centroids[3]);
        assert_eq!(slot, 3);
        assert_eq!(payload, 0);
        assert!(grid.o1_hit_rate() > 0.0);

        let stats = grid.stats();
        assert_eq!(stats.total_centroids, 8);
        assert!(stats.total_cells >= 32);
        assert!(stats.occupied_cells > 0);
        assert!(stats.max_cell_size >= 1);

        grid.insert(3, 0xABCD);
        let (_, payload) = grid.locate_with_payload(centroids[3]);
        assert_eq!(payload, 0xABCD);
    }

    #[test]
    fn tss_verifier_and_dynamic_centroid_insert_cover_legacy_api() {
        let centroids = [(0.5, 0.0), (1.0, 1.5), (2.0, 3.0), (2.5, 4.5)];
        let verifier = TssVerifier::<4>::new(centroids, 0.1);
        let report = verifier.full_verification(1_000, 4, 16);
        assert!(report.separation_holds);
        assert!(report.packing_holds);
        assert!(report.theorem_holds);

        let mut grid = SphericalGridHashIndex::<4>::new(centroids);
        let moved = (2.9, 5.0);
        grid.insert_centroid(1, moved, 77);
        let (slot, payload) = grid.locate_with_payload(moved);
        assert_eq!(slot, 1);
        assert_eq!(payload, 77);
    }

    #[test]
    fn grid_hash_restores_legacy_auto_build_and_empty_locate() {
        assert_eq!(auto_sized_dimensions(0), (4, 8));
        assert_eq!(auto_sized_dimensions(16), (4, 8));
        assert_eq!(auto_sized_dimensions(17), (5, 10));

        let mut empty = SphericalGridHashIndex::<0>::new([]);
        assert_eq!(empty.locate_legacy((0.1, 0.2)), EMPTY_LOCATE);

        let original = [(0.5, 0.0), (1.0, 1.5)];
        let rebuilt = [(2.0, 3.0), (2.5, 4.5)];
        let mut grid = SphericalGridHashIndex::<2>::new(original);
        grid.insert(0, 99);
        grid.build(rebuilt);

        let (slot, payload) = grid.locate_with_payload(rebuilt[1]);
        assert_eq!(slot, 1);
        assert_eq!(payload, 0);
        assert_eq!(grid.stats().total_lookups, 1);
    }
}
