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
fn verify_separation(centroids: &[(f64, f64)], theta_min: f64) -> bool {
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
