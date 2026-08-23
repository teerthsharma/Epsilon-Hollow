//! `SphericalVoronoiIndex::betti_0` returns the type parameter `K`, justified
//! as "each Voronoi cell is one connected component, so beta_0 = K".
//!
//! A Voronoi decomposition of S^2 partitions the sphere; the UNION of the cells
//! is S^2, which is connected. beta_0 of the covered space is 1, not K.
//! Adjacent cells share boundaries — the decomposition is not a disjoint union.
use aether_core::SphericalVoronoiIndex;

#[test]
fn betti_0_of_a_decomposition_of_a_connected_sphere_is_one() {
    let idx = SphericalVoronoiIndex::<4>::new(
        [(0.5, 0.0), (1.0, 1.5), (2.0, 3.0), (2.5, 4.5)]);
    let b0 = idx.betti_0();
    println!("K=4 well-separated centroids -> betti_0 = {b0}, capacity = {}", idx.capacity());
    assert_eq!(b0, 1,
        "the cells tile S^2 and S^2 is connected, so beta_0 = 1, got {b0}");
}

#[test]
fn betti_0_does_not_grow_with_the_number_of_cells() {
    // A topological invariant of the covered space cannot depend on how finely
    // that space was subdivided. Refining a tiling does not create components.
    let a = SphericalVoronoiIndex::<2>::new([(1.0, 0.0), (2.0, 3.0)]).betti_0();
    let b = SphericalVoronoiIndex::<4>::new(
        [(0.5, 0.0), (1.0, 1.5), (2.0, 3.0), (2.5, 4.5)]).betti_0();
    let c = SphericalVoronoiIndex::<8>::new(
        [(0.3, 0.0), (0.6, 0.8), (1.0, 1.5), (1.4, 2.2),
         (1.8, 2.9), (2.2, 3.6), (2.6, 4.3), (3.0, 5.0)]).betti_0();
    println!("betti_0 at K=2,4,8 -> {a}, {b}, {c}");
    assert_eq!((a, b, c), (1, 1, 1),
        "refining the tiling changed beta_0: {a}, {b}, {c} — it is counting cells, not components");
}

#[test]
fn duplicate_centroids_leave_empty_cells_so_k_is_not_a_count_of_anything_real() {
    // Two coincident centroids: one of the two cells receives nothing, because
    // `locate` breaks the tie in favour of a single slot. K = 4 but at most 3
    // cells are ever occupied.
    let dup = [(1.0, 1.0), (1.0, 1.0), (2.0, 3.0), (2.5, 4.5)];
    let idx = SphericalVoronoiIndex::<4>::new(dup);
    let mut hit = [false; 4];
    let mut s: u64 = 0x1357_9BDF_2468_ACE0;
    for _ in 0..4000 {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let t = core::f64::consts::PI * (((s >> 11) as f64) / ((1u64 << 53) as f64));
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let p = 2.0 * core::f64::consts::PI * (((s >> 11) as f64) / ((1u64 << 53) as f64));
        hit[idx.locate((t, p))] = true;
    }
    let occupied = hit.iter().filter(|&&h| h).count();
    println!("duplicate centroids: K=4, cells actually reachable = {occupied}");
    assert!(occupied < 4,
        "expected a duplicate centroid to leave an unreachable cell; got {occupied} of 4");
    assert_eq!(idx.betti_0(), 1, "beta_0 is still 1: the tiling still covers a connected S^2");
}
