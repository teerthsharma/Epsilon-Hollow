// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Boot theorem lines, computed from the running kernel.
//!
//! Every line T1-T10 carries one of three verdicts and the evidence
//! `seal-mkimage --check-theorem-log` recomputes it from:
//!
//! - `CERTIFIED`: the hypothesis holds for the parameters the running
//!   instances were built from, read from the definitions those instances
//!   read (never a copy), at the live scheduler governor's epsilon.
//! - `NOT CERTIFIED`: it fails for a running instance; the line names the
//!   value that fails.
//! - `NOT CHECKED`: no running instance carries the theorem's parameters.
//!
//! `THEOREM_STATES[i]` is true exactly when the theorem is certified, so the
//! runtime paths gated on it take their fallback otherwise. A verdict is
//! reported, never panicked on.

use alloc::format;
use alloc::string::String;
use core::sync::atomic::{AtomicU8, Ordering};

use aether_core::scm::SpectralContractionOperator;
use aether_verified::{aether_agcr, aether_tss};

use crate::{
    serial_println, GOVERNOR_ALPHA, GOVERNOR_BETA, GOVERNOR_DT, GOVERNOR_EPSILON, THEOREM_COUNT,
    THEOREM_NAMES, THEOREM_STATES,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    NotChecked = 0,
    Certified = 1,
    NotCertified = 2,
}

impl Status {
    pub const fn text(self) -> &'static str {
        match self {
            Status::Certified => "CERTIFIED",
            Status::NotCertified => "NOT CERTIFIED",
            Status::NotChecked => "NOT CHECKED",
        }
    }
}

/// Verdict per theorem; `NotChecked` until `init` runs.
static STATUS: [AtomicU8; THEOREM_COUNT] = [const { AtomicU8::new(0) }; THEOREM_COUNT];

pub fn status(idx: usize) -> Status {
    match STATUS[idx].load(Ordering::Relaxed) {
        1 => Status::Certified,
        2 => Status::NotCertified,
        _ => Status::NotChecked,
    }
}

/// The one rendering of a theorem's state every consumer prints.
pub fn status_text(idx: usize) -> &'static str {
    status(idx).text()
}

/// Centroid table of a running spherical Voronoi index.
pub struct Table {
    pub users: &'static str,
    pub centroids: [(f64, f64); 8],
}

/// Everything a verdict is computed from.
pub struct LiveState {
    /// Epsilon of the scheduler's running governor.
    pub epsilon: f64,
    /// Every S^2 centroid table a running index was built from.
    pub tables: [Table; 2],
    /// `alpha` of every spectral contraction operator this module can read.
    pub operators: [(&'static str, f64); 3],
    /// Running operators whose `alpha` is private to their module.
    pub unread_operators: &'static str,
    /// `(alpha, beta, dt)` every runtime governor is built and stepped with.
    pub gains: (f64, f64, f64),
}

impl LiveState {
    pub fn read() -> Self {
        Self {
            epsilon: crate::process::scheduler::governor_epsilon(),
            tables: [
                Table {
                    users: "scheduler+compositor+firewall+route",
                    centroids: aether_core::tss::CUBE_CENTROIDS,
                },
                Table {
                    users: "manifoldfs",
                    centroids: crate::fs::voronoi_cap::VoronoiCap::default_centroids(),
                },
            ],
            operators: [
                ("manifoldfs", crate::fs::manifold_fs::SCM_ALPHA),
                ("firewall", crate::net::firewall::scm_alpha()),
                ("route", crate::net::topological::scm_alpha()),
            ],
            // `ManifoldScheduler::new` builds its predictor from a literal.
            unread_operators: "scheduler",
            gains: (GOVERNOR_ALPHA, GOVERNOR_BETA, GOVERNOR_DT),
        }
    }
}

pub struct Verdict {
    pub status: Status,
    /// Everything after `[THEOREM] <name> <STATUS>: `.
    pub detail: String,
}

fn verdict(certified: bool, detail: String) -> Verdict {
    let status = if certified {
        Status::Certified
    } else {
        Status::NotCertified
    };
    Verdict { status, detail }
}

fn not_checked(reason: &str) -> Verdict {
    Verdict {
        status: Status::NotChecked,
        detail: String::from(reason),
    }
}

/// T1/TSS: every table packs within `P_max` and is separated by more than
/// `theta_min = 2 asin(eps / 2)` at the live epsilon.
fn t1(state: &LiveState) -> Verdict {
    let theta = aether_tss::theta_min_from_epsilon(state.epsilon);
    let p_max = aether_tss::p_max(theta);
    let mut min_sep = f64::INFINITY;
    let mut failing: Option<&'static str> = None;
    for table in &state.tables {
        let c = &table.centroids;
        for i in 0..c.len() {
            for j in (i + 1)..c.len() {
                let d = aether_tss::great_circle_distance(c[i].0, c[i].1, c[j].0, c[j].1);
                min_sep = min_sep.min(d);
            }
        }
        let holds = aether_tss::verify_packing_bound(c.len(), theta)
            && aether_tss::verify_separation(c, theta);
        if !holds && failing.is_none() {
            failing = Some(table.users);
        }
    }
    let users: String = state
        .tables
        .iter()
        .map(|t| t.users)
        .collect::<alloc::vec::Vec<_>>()
        .join(",");
    let cells = state
        .tables
        .iter()
        .map(|t| t.centroids.len())
        .max()
        .unwrap_or(0);
    let evidence = format!(
        "eps={:.4} theta_min={:.4} cells={} p_max={:.1} min_sep={:.4} covers={}",
        state.epsilon, theta, cells, p_max, min_sep, users
    );
    match failing {
        None => verdict(true, evidence),
        Some(users) => verdict(false, format!("table of {} fails; {}", users, evidence)),
    }
}

/// T2/SCM: each operator, run through aether-core's own `apply`, moves two
/// states no further apart than its claimed Lipschitz constant `1 - alpha`,
/// and that constant is below 1.
fn t2(state: &LiveState) -> Verdict {
    let (s1, s2, pred) = ([1.0, -2.0], [-3.0, 0.5], [0.25, 0.75]);
    let before = libm::hypot(s1[0] - s2[0], s1[1] - s2[1]);
    let mut certified = true;
    let mut max_lip = f64::NEG_INFINITY;
    let mut max_ratio = 0.0f64;
    let mut listed = String::new();
    for (idx, (users, alpha)) in state.operators.iter().enumerate() {
        let op = SpectralContractionOperator::<2> { alpha: *alpha };
        let (a, b) = (op.apply(&s1, &pred), op.apply(&s2, &pred));
        let ratio = libm::hypot(a[0] - b[0], a[1] - b[1]) / before;
        let lip = op.lipschitz_constant();
        certified &= lip < 1.0 && ratio <= lip + 1e-12;
        max_lip = max_lip.max(lip);
        max_ratio = max_ratio.max(ratio);
        if idx > 0 {
            listed.push(',');
        }
        listed.push_str(&format!("{}:{:.2}", users, alpha));
    }
    verdict(
        certified,
        format!(
            "operators={} max_lip={:.2} max_ratio={:.4} unread={}",
            listed, max_lip, max_ratio, state.unread_operators
        ),
    )
}

/// T4/AGCR: `alpha + beta/dt < 1` at the gains and step every runtime
/// governor uses.
fn t4(state: &LiveState) -> Verdict {
    let (alpha, beta, dt) = state.gains;
    let margin = alpha + beta / dt;
    let rho = aether_agcr::contraction_rate(alpha, beta, dt);
    let certified = rho > 0.0
        && rho < 1.0
        && aether_agcr::half_life(rho).is_finite()
        && aether_agcr::gain_margin_stable(alpha, beta, dt);
    let relation = if certified { "<" } else { ">=" };
    verdict(
        certified,
        format!("alpha+beta/dt={:.2} {} 1 at dt={}", margin, relation, dt),
    )
}

const T3_UNCHECKED: &str = "no running instance merges clusters here: TopoRAM's T3 path is a run-count ratio and ManifoldFS mounts after this check";
const T5_UNCHECKED: &str = "no running instance embeds a tree with curvature, dimension and depth: TopoRAM's T5 path is an access-density threshold and the scheduler's process tree is a parent/child map";
const UNUSED: &str = "no kernel subsystem runs it";

/// Verdicts for T1-T10, in `THEOREM_NAMES` order.
pub fn evaluate(state: &LiveState) -> [Verdict; THEOREM_COUNT] {
    [
        t1(state),
        t2(state),
        not_checked(T3_UNCHECKED),
        t4(state),
        not_checked(T5_UNCHECKED),
        not_checked(UNUSED),
        not_checked(UNUSED),
        not_checked(UNUSED),
        not_checked(UNUSED),
        not_checked(UNUSED),
    ]
}

/// Evaluate every theorem against the running kernel, publish the verdicts,
/// and print one line each plus the tally.
pub fn init() {
    let state = LiveState::read();
    GOVERNOR_EPSILON.store(state.epsilon.to_bits(), Ordering::Relaxed);
    serial_println!(
        "[T4/AGCR] Governor online: epsilon = {:.4} alpha={} beta={} dt={}",
        state.epsilon,
        GOVERNOR_ALPHA,
        GOVERNOR_BETA,
        GOVERNOR_DT
    );
    let voronoi =
        aether_core::tss::SphericalVoronoiIndex::<8>::new(aether_core::tss::CUBE_CENTROIDS);
    serial_println!(
        "[T1/TSS]  Voronoi index: {} cells, test lookup -> cell {}",
        voronoi.capacity(),
        voronoi.locate((0.5, 0.5))
    );

    let verdicts = evaluate(&state);
    let mut tally: [String; 3] = [String::new(), String::new(), String::new()];
    let mut counts = [0usize; 3];
    for (idx, v) in verdicts.iter().enumerate() {
        STATUS[idx].store(v.status as u8, Ordering::Relaxed);
        THEOREM_STATES[idx].store(v.status == Status::Certified, Ordering::Relaxed);
        serial_println!(
            "[THEOREM] {} {}: {}",
            THEOREM_NAMES[idx],
            v.status.text(),
            v.detail
        );
        let slot = match v.status {
            Status::Certified => 0,
            Status::NotCertified => 1,
            Status::NotChecked => 2,
        };
        counts[slot] += 1;
        if !tally[slot].is_empty() {
            tally[slot].push(' ');
        }
        tally[slot].push_str(THEOREM_NAMES[idx]);
    }
    serial_println!(
        "[BOOT] Theorems: {} certified ({}), {} not certified ({}), {} not checked ({})",
        counts[0],
        tally[0],
        counts[1],
        tally[1],
        counts[2],
        tally[2]
    );
}
