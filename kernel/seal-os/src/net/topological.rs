// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Topological packet routing and connection tracking.
//! T1 Voronoi classification, T2 spectral routing prediction, T5 spherical embedding.

use alloc::vec::Vec;
use spin::Mutex;

use aether_core::manifold::SparseAttentionGraph;
use aether_core::scm::SpectralContractionOperator;
use aether_core::tss::SphericalVoronoiIndex;

use crate::fs::encoder::SpherePoint;
use crate::net::IpAddr;

/// T1: Voronoi routing table entry.
pub struct RouteEntry {
    pub dst_embedding: SpherePoint,
    pub gateway: IpAddr,
    pub interface: u8,
}

const VORONOI_K: usize = 8;

static ROUTE_VORONOI: Mutex<Option<SphericalVoronoiIndex<VORONOI_K>>> = Mutex::new(None);

// T1: Each Voronoi cell has its own routing table slice.
static ROUTE_CELLS: Mutex<[Vec<RouteEntry>; VORONOI_K]> = Mutex::new([
    Vec::new(),
    Vec::new(),
    Vec::new(),
    Vec::new(),
    Vec::new(),
    Vec::new(),
    Vec::new(),
    Vec::new(),
]);

// T2: Spectral routing prediction -- traffic history per destination cell.
static TRAFFIC_HISTORY: Mutex<[f64; VORONOI_K]> = Mutex::new([0.0; VORONOI_K]);
/// Gain of every spectral contraction operator this module builds.
const SCM_ALPHA: f64 = 0.3;
static SPECTRAL_OP: Mutex<SpectralContractionOperator<VORONOI_K>> =
    Mutex::new(SpectralContractionOperator { alpha: SCM_ALPHA });

/// `alpha` of the running routing predictor, as `theorems` certifies T2 from.
pub fn scm_alpha() -> f64 {
    SPECTRAL_OP.lock().alpha
}

// T5: Connection tracking as sparse attention graph on S².
static CONN_GRAPH: Mutex<Option<SparseAttentionGraph<3>>> = Mutex::new(None);
static CONNECTIONS: Mutex<Vec<ConnectionState>> = Mutex::new(Vec::new());

/// Per-connection state with T2 spectral lifetime prediction.
pub struct ConnectionState {
    pub src: IpAddr,
    pub dst: IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: u8,
    pub bandwidth: f64,
    pub rate_vector: [f64; 8],
    pub established_at: u64,
    pub last_seen: u64,
    pub predicted_lifetime: f64,
}

/// Initialize topological routing infrastructure.
pub fn init() {
    // T1: Eight distinct cells. The {0, pi/2, pi} lattice this used put cells
    // 0, 3 and 6 on one pole, so routes never landed in cells 3 or 6.
    *ROUTE_VORONOI.lock() = Some(SphericalVoronoiIndex::new(aether_core::tss::CUBE_CENTROIDS));
}

fn ip_to_sphere_point(ip: &IpAddr) -> SpherePoint {
    let (theta, phi) = ip.to_sphere();
    let x = libm::sin(theta) * libm::cos(phi);
    let y = libm::sin(theta) * libm::sin(phi);
    let z = libm::cos(theta);
    SpherePoint { coords: [x, y, z] }
}

fn cartesian_to_spherical(p: &SpherePoint) -> (f64, f64) {
    let r = libm::sqrt(
        p.coords[0] * p.coords[0] + p.coords[1] * p.coords[1] + p.coords[2] * p.coords[2],
    );
    if r < 1e-12 {
        return (0.0, 0.0);
    }
    let theta = libm::acos((p.coords[2] / r).clamp(-1.0, 1.0));
    let mut phi = libm::atan2(p.coords[1], p.coords[0]);
    if phi < 0.0 {
        phi += 2.0 * core::f64::consts::PI;
    }
    (theta, phi)
}

/// T1: Add a route to the Voronoi cell nearest to `dst`.
pub fn route_add(dst: IpAddr, gateway: IpAddr, iface: u8) {
    let embedding = ip_to_sphere_point(&dst);
    let (theta, phi) = cartesian_to_spherical(&embedding);
    let cell = {
        let voronoi = ROUTE_VORONOI.lock();
        match voronoi.as_ref() {
            Some(v) => v.locate((theta, phi)),
            None => 0,
        }
    };
    let entry = RouteEntry {
        dst_embedding: embedding,
        gateway,
        interface: iface,
    };
    let mut cells = ROUTE_CELLS.lock();
    cells[cell].push(entry);
}

/// T1 + T2: Look up the route whose destination embedding is nearest `dst`.
///
/// Every cell is searched, not only `dst`'s: near a cell boundary the nearest
/// route can lie across it, and `dst`'s own cell can be empty while others
/// are not.
pub fn route_lookup(dst: IpAddr) -> Option<(IpAddr, u8)> {
    let embedding = ip_to_sphere_point(&dst);
    let (theta, phi) = cartesian_to_spherical(&embedding);

    let cell = {
        let voronoi = ROUTE_VORONOI.lock();
        match voronoi.as_ref() {
            Some(v) => v.locate((theta, phi)),
            None => 0,
        }
    };

    // T2: Update traffic history for this cell.
    {
        let mut history = TRAFFIC_HISTORY.lock();
        history[cell] += 1.0;
    }

    // ponytail: scans every route, O(routes); skip a cell when the query's
    // distance to its bisector with `cell` exceeds the best so far, once
    // tables grow.
    let cells = ROUTE_CELLS.lock();
    cells
        .iter()
        .flatten()
        .map(|e| (embedding.distance_sq(&e.dst_embedding), e))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, e)| (e.gateway, e.interface))
}

/// T2: Predict next hop via spectral contraction on traffic history.
pub fn predict_next_hop() -> Option<usize> {
    let history = TRAFFIC_HISTORY.lock();
    let op = SPECTRAL_OP.lock();
    // Predicted traffic = spectral contraction toward uniform mean.
    let mean = history.iter().copied().sum::<f64>() / VORONOI_K as f64;
    let pred = [mean; VORONOI_K];
    let predicted = op.apply(&history, &pred);
    // Return cell with highest predicted traffic.
    let mut best = 0usize;
    let mut best_val = predicted[0];
    for i in 1..VORONOI_K {
        if predicted[i] > best_val {
            best = i;
            best_val = predicted[i];
        }
    }
    Some(best)
}

/// T2 + T5: Track a TCP connection in the sparse attention graph.
pub fn track_connection(src: IpAddr, dst: IpAddr, src_port: u16, dst_port: u16, bytes: usize) {
    let now = crate::drivers::interrupts::ticks();
    let src_point = src.to_manifold_point();
    let dst_point = dst.to_manifold_point();

    let mut graph_opt = CONN_GRAPH.lock();
    if graph_opt.is_none() {
        *graph_opt = Some(SparseAttentionGraph::new(0.5));
    }
    if let Some(ref mut graph) = *graph_opt {
        let _ = graph.add_point(src_point);
        let _ = graph.add_point(dst_point);
    }
    drop(graph_opt);

    let mut conns = CONNECTIONS.lock();
    for c in conns.iter_mut() {
        if c.src == src && c.dst == dst && c.src_port == src_port && c.dst_port == dst_port {
            c.bandwidth += bytes as f64;
            c.last_seen = now;
            // T2: Update rate vector (sliding window approx)
            c.rate_vector[0] += bytes as f64;
            // Predict lifetime
            let op = SpectralContractionOperator::<8>::new(SCM_ALPHA);
            let pred = [0.0; 8];
            let next = op.apply(&c.rate_vector, &pred);
            c.predicted_lifetime = next.iter().sum();
            return;
        }
    }
    // New connection.
    let op = SpectralContractionOperator::<8>::new(SCM_ALPHA);
    let pred = [0.0; 8];
    let next = op.apply(&[bytes as f64; 8], &pred);
    conns.push(ConnectionState {
        src,
        dst,
        src_port,
        dst_port,
        protocol: 6,
        bandwidth: bytes as f64,
        rate_vector: [bytes as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        established_at: now,
        last_seen: now,
        predicted_lifetime: next.iter().sum(),
    });
}

/// T2: Prune short-lived connections aggressively, keep long-lived ones.
pub fn prune_connections() {
    let now = crate::drivers::interrupts::ticks();
    let mut conns = CONNECTIONS.lock();
    conns.retain(|c| {
        let age = now.wrapping_sub(c.established_at);
        // Short-lived = aggressive timeout; long-lived = keepalive
        let timeout = if c.predicted_lifetime < 100.0 {
            5000u64
        } else {
            60000u64
        };
        age < timeout
    });
}

/// Periodic housekeeping.
pub fn poll() {
    prune_connections();
}

// ---------------------------------------------------------------------------
// Tests -- run by the in-kernel harness (crate::testing), not `cargo test`.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;
    use aether_core::tss::CUBE_CENTROIDS;

    /// The index `init` builds sends each cube-vertex centroid to its own
    /// slot, so all eight route cells can receive a route.
    fn test_every_route_cell_is_reachable() -> TestResult {
        let saved = ROUTE_VORONOI.lock().take();
        init();
        let located: Vec<usize> = match ROUTE_VORONOI.lock().as_ref() {
            Some(v) => CUBE_CENTROIDS.iter().map(|&c| v.locate(c)).collect(),
            None => Vec::new(),
        };
        *ROUTE_VORONOI.lock() = saved;
        let mut distinct = located.clone();
        distinct.sort_unstable();
        distinct.dedup();
        crate::serial_println!(
            "[net::topological] locate(CUBE_CENTROIDS[k]) = {:?}, {} distinct",
            located,
            distinct.len()
        );
        test_assert!(
            located == (0..VORONOI_K).collect::<Vec<_>>(),
            "a route cell is unreachable"
        );
        TestResult::Pass
    }

    /// The IPv6 address `to_sphere` maps to `(theta, phi)`.
    fn v6_at(theta: f64, phi: f64) -> IpAddr {
        let hi = (theta / core::f64::consts::PI * u64::MAX as f64) as u64;
        let lo = (phi / core::f64::consts::TAU * u64::MAX as f64) as u64;
        let mut a = [0u8; 16];
        a[..8].copy_from_slice(&hi.to_be_bytes());
        a[8..].copy_from_slice(&lo.to_be_bytes());
        IpAddr::V6(a)
    }

    /// A query 0.05 rad inside cell 0 of the cell 0 / cell 1 boundary, a route
    /// 0.05 rad across it in cell 1 (0.08 rad away), and a route deep in cell
    /// 0 (0.75 rad away): the lookup must return the route across.
    fn test_lookup_finds_the_nearest_route_across_a_cell_boundary() -> TestResult {
        use core::f64::consts::{FRAC_PI_2, FRAC_PI_4};
        let north = CUBE_CENTROIDS[0].0;
        let query = v6_at(north, FRAC_PI_2 - 0.05);
        let near = v6_at(north, FRAC_PI_2 + 0.05);
        let far = v6_at(0.3, FRAC_PI_4);

        let saved_index = ROUTE_VORONOI
            .lock()
            .replace(SphericalVoronoiIndex::new(CUBE_CENTROIDS));
        let saved_cells = core::mem::take(&mut *ROUTE_CELLS.lock());
        route_add(far, IpAddr::V4([10, 0, 0, 1]), 1);
        route_add(near, IpAddr::V4([10, 0, 0, 2]), 2);
        let got = route_lookup(query);
        let cell = |ip: &IpAddr| {
            let (theta, phi) = cartesian_to_spherical(&ip_to_sphere_point(ip));
            ROUTE_VORONOI
                .lock()
                .as_ref()
                .map(|v| v.locate((theta, phi)))
        };
        let cells = (cell(&query), cell(&near), cell(&far));
        let q = ip_to_sphere_point(&query);
        let nearer =
            q.distance_sq(&ip_to_sphere_point(&near)) < q.distance_sq(&ip_to_sphere_point(&far));
        *ROUTE_CELLS.lock() = saved_cells;
        *ROUTE_VORONOI.lock() = saved_index;

        test_assert!(
            cells == (Some(0), Some(1), Some(0)) && nearer,
            "fixture premise: query and far route in cell 0, near route in cell 1"
        );
        test_assert!(
            got == Some((IpAddr::V4([10, 0, 0, 2]), 2)),
            "lookup returned a farther route from the query's own cell"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "net::topological::every_route_cell_is_reachable",
            test_every_route_cell_is_reachable,
        );
        crate::testing::register_test(
            "net::topological::lookup_finds_the_nearest_route_across_a_cell_boundary",
            test_lookup_finds_the_nearest_route_across_a_cell_boundary,
        );
    }
}
