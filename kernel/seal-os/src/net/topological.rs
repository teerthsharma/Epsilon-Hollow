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
static SPECTRAL_OP: Mutex<SpectralContractionOperator<VORONOI_K>> =
    Mutex::new(SpectralContractionOperator { alpha: 0.3 });

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
    // T1: Place 8 well-separated centroids on S².
    let centroids: [(f64, f64); VORONOI_K] = [
        (0.0, 0.0),
        (core::f64::consts::FRAC_PI_2, 0.0),
        (core::f64::consts::PI, 0.0),
        (0.0, core::f64::consts::FRAC_PI_2),
        (core::f64::consts::FRAC_PI_2, core::f64::consts::FRAC_PI_2),
        (core::f64::consts::PI, core::f64::consts::FRAC_PI_2),
        (0.0, core::f64::consts::PI),
        (core::f64::consts::FRAC_PI_2, core::f64::consts::PI),
    ];
    *ROUTE_VORONOI.lock() = Some(SphericalVoronoiIndex::new(centroids));
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

/// T1 + T2: Look up a route by finding the Voronoi cell and searching its entries.
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

    let cells = ROUTE_CELLS.lock();
    let entries = &cells[cell];
    if entries.is_empty() {
        return None;
    }
    // Return the entry with smallest embedding distance to dst.
    let mut best = &entries[0];
    let mut best_dist = embedding.distance_sq(&entries[0].dst_embedding);
    for e in entries.iter().skip(1) {
        let d = embedding.distance_sq(&e.dst_embedding);
        if d < best_dist {
            best = e;
            best_dist = d;
        }
    }
    Some((best.gateway, best.interface))
}

/// T2: Predict next hop via spectral contraction on traffic history.
pub fn predict_next_hop() -> Option<usize> {
    let history = TRAFFIC_HISTORY.lock();
    let op = SPECTRAL_OP.lock();
    // Predicted traffic = spectral contraction toward uniform mean.
    let mean = history.iter().copied().sum::<f64>() / VORONOI_K as f64;
    let pred = [mean; VORONOI_K];
    let predicted = op.apply(&*history, &pred);
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
            let op = SpectralContractionOperator::<8>::new(0.3);
            let pred = [0.0; 8];
            let next = op.apply(&c.rate_vector, &pred);
            c.predicted_lifetime = next.iter().sum();
            return;
        }
    }
    // New connection.
    let op = SpectralContractionOperator::<8>::new(0.3);
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
