// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Topological firewall -- T1 Voronoi trust zones, T2 intrusion detection,
//! T3 entropy attack detection, T4 adaptive strictness, T5 hyperbolic trust.

use alloc::vec::Vec;
use core::sync::atomic::Ordering;
use spin::Mutex;

use aether_core::scm::SpectralContractionOperator;
use aether_core::tss::SphericalVoronoiIndex;

use crate::net::IpAddr;

/// Firewall action for a packet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FirewallAction {
    Allow,
    Drop,
    Log,
}

/// Parsed packet info for firewall evaluation.
pub struct PacketInfo {
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub src_port: Option<u16>,
    pub dst_port: Option<u16>,
    pub protocol: u8,
    pub len: usize,
}

/// Firewall rule.
#[derive(Debug, Clone)]
pub enum FirewallRule {
    Allow {
        zone: usize,
        ports: Vec<u16>,
        protocols: Vec<u8>,
    },
    Deny {
        zone: usize,
        ports: Vec<u16>,
        protocols: Vec<u8>,
    },
}

const VORONOI_K: usize = 8;
const RATE_VEC_DIM: usize = 8;
const ENTROPY_WINDOW: usize = 64;

// T1: Voronoi trust zones.
static TRUST_ZONE_VORONOI: Mutex<Option<SphericalVoronoiIndex<VORONOI_K>>> = Mutex::new(None);
static RULES: Mutex<Vec<FirewallRule>> = Mutex::new(Vec::new());

// T2: Per-source IP packet rate vectors for intrusion detection.
struct SourceRate {
    ip: IpAddr,
    rates: [f64; RATE_VEC_DIM],
    last_update: u64,
    packet_count: u64,
}

static SOURCE_RATES: Mutex<Vec<SourceRate>> = Mutex::new(Vec::new());
static RATE_SCM: Mutex<SpectralContractionOperator<RATE_VEC_DIM>> =
    Mutex::new(SpectralContractionOperator { alpha: 0.3 });

// T3: Destination port entropy window.
static PORT_HISTORY: Mutex<Vec<u16>> = Mutex::new(Vec::new());

/// Initialize firewall state.
pub fn init() {
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
    *TRUST_ZONE_VORONOI.lock() = Some(SphericalVoronoiIndex::new(centroids));
}

/// Add a firewall rule.
pub fn add_rule(rule: FirewallRule) {
    RULES.lock().push(rule);
}

fn ip_to_zone(ip: &IpAddr) -> usize {
    let (theta, phi) = ip.to_sphere();
    let voronoi = TRUST_ZONE_VORONOI.lock();
    match voronoi.as_ref() {
        Some(v) => v.locate((theta, phi)),
        None => 0,
    }
}

fn update_source_rate(src: &IpAddr) {
    let now = crate::drivers::interrupts::ticks();
    let mut rates = SOURCE_RATES.lock();
    for r in rates.iter_mut() {
        if r.ip == *src {
            r.packet_count += 1;
            if now.wrapping_sub(r.last_update) > 1000 {
                r.rates.copy_within(0..RATE_VEC_DIM - 1, 1);
                r.rates[RATE_VEC_DIM - 1] = r.packet_count as f64;
                r.packet_count = 0;
                r.last_update = now;
            }
            return;
        }
    }
    rates.push(SourceRate {
        ip: *src,
        rates: [0.0; RATE_VEC_DIM],
        last_update: now,
        packet_count: 1,
    });
}

fn detect_intrusion(src: &IpAddr) -> bool {
    let rates = SOURCE_RATES.lock();
    let r = match rates.iter().find(|r| r.ip == *src) {
        Some(r) => r,
        None => return false,
    };
    let scm = RATE_SCM.lock();
    let pred = [0.0; RATE_VEC_DIM];
    let contracted = scm.apply(&r.rates, &pred);
    let sum: f64 = r.rates.iter().sum();
    let csum: f64 = contracted.iter().sum();
    if sum > 0.0 && csum > 0.0 {
        let ratio = sum / csum;
        ratio > 5.0
    } else {
        false
    }
}

fn shannon_entropy_ports() -> f64 {
    let history = PORT_HISTORY.lock();
    let n = history.len();
    if n == 0 {
        return 0.0;
    }
    let mut counts: [(u16, u32); ENTROPY_WINDOW] = [(0, 0); ENTROPY_WINDOW];
    let mut unique = 0usize;
    for &port in history.iter() {
        let mut found = false;
        for i in 0..unique {
            if counts[i].0 == port {
                counts[i].1 += 1;
                found = true;
                break;
            }
        }
        if !found && unique < ENTROPY_WINDOW {
            counts[unique] = (port, 1);
            unique += 1;
        }
    }
    let mut entropy = 0.0;
    for i in 0..unique {
        let p = counts[i].1 as f64 / n as f64;
        if p > 0.0 {
            entropy -= p * libm::log2(p);
        }
    }
    entropy
}

// T4: Adaptive strictness from governor epsilon.
fn adaptive_strictness() -> f64 {
    let epsilon_bits = crate::GOVERNOR_EPSILON.load(Ordering::Relaxed);
    let epsilon = f64::from_bits(epsilon_bits);
    // T4: epsilon > 0.7 after attack -> tighten rules. epsilon < 0.2 normal -> relax.
    if epsilon > 0.7 {
        1.0 // strict
    } else if epsilon < 0.2 {
        0.0 // relaxed
    } else {
        epsilon
    }
}

// T5: Hyperbolic trust -- trusted near origin, untrusted near boundary.
fn ip_to_poincare(ip: &IpAddr) -> (f64, f64) {
    let (theta, phi) = ip.to_sphere();
    // Azimuthal equidistant projection centered at north pole.
    let r = theta / core::f64::consts::PI;
    let x = r * libm::cos(phi);
    let y = r * libm::sin(phi);
    (x, y)
}

fn poincare_distance_from_origin(z: (f64, f64)) -> f64 {
    let norm_sq = z.0 * z.0 + z.1 * z.1;
    let norm = libm::sqrt(norm_sq).min(0.9999);
    2.0 * libm::atanh(norm)
}

fn hyperbolic_trust(ip: &IpAddr) -> f64 {
    let z = ip_to_poincare(ip);
    let d = poincare_distance_from_origin(z);
    // Near origin = trusted (distance ~ 0), near boundary = untrusted.
    1.0 / (1.0 + d)
}

/// Evaluate a packet against the topological firewall.
pub fn firewall_eval(pkt: &PacketInfo) -> FirewallAction {
    // T1: Voronoi zone classification.
    let zone = ip_to_zone(&pkt.src_ip);

    // T2: Spectral intrusion detection.
    update_source_rate(&pkt.src_ip);
    let anomaly = detect_intrusion(&pkt.src_ip);

    // T3: Entropy of destination ports.
    if let Some(dst_port) = pkt.dst_port {
        let mut history = PORT_HISTORY.lock();
        history.push(dst_port);
        if history.len() > ENTROPY_WINDOW {
            history.remove(0);
        }
    }
    let entropy = shannon_entropy_ports();

    // T4: Adaptive strictness from governor epsilon.
    let strictness = adaptive_strictness();

    // T5: Hyperbolic trust hierarchy.
    let trust = hyperbolic_trust(&pkt.src_ip);

    // Combine factors into risk score.
    let mut score = 0.0;
    score += (1.0 - trust) * 0.3;
    if anomaly {
        score += 0.4;
    }
    if entropy > 5.0 {
        // T3: DDoS = high entropy
        score += 0.3;
    } else if entropy < 2.0 && entropy > 0.0 {
        // T3: Port scan = low entropy
        score += 0.2;
    }
    score += strictness * 0.2;

    // Evaluate explicit rules.
    let rules = RULES.lock();
    for rule in rules.iter() {
        match rule {
            FirewallRule::Allow {
                zone: rz,
                ports,
                protocols,
            } => {
                if *rz == zone
                    && (ports.is_empty() || ports.contains(&pkt.dst_port.unwrap_or(0)))
                    && (protocols.is_empty() || protocols.contains(&pkt.protocol))
                {
                    return FirewallAction::Allow;
                }
            }
            FirewallRule::Deny {
                zone: rz,
                ports,
                protocols,
            } => {
                if *rz == zone
                    && (ports.is_empty() || ports.contains(&pkt.dst_port.unwrap_or(0)))
                    && (protocols.is_empty() || protocols.contains(&pkt.protocol))
                {
                    return FirewallAction::Drop;
                }
            }
        }
    }

    if score > 0.8 {
        return FirewallAction::Drop;
    } else if score > 0.5 {
        return FirewallAction::Log;
    }
    FirewallAction::Allow
}

/// Periodic housekeeping for the firewall.
pub fn poll() {
    let now = crate::drivers::interrupts::ticks();
    let mut rates = SOURCE_RATES.lock();
    rates.retain(|r| now.wrapping_sub(r.last_update) < 30000);
    let mut history = PORT_HISTORY.lock();
    if history.len() > ENTROPY_WINDOW {
        let excess = history.len() - ENTROPY_WINDOW;
        history.drain(..excess);
    }
}

/// Build PacketInfo from an IPv4 raw packet.
pub fn packet_info_from_ipv4(pkt: &[u8]) -> Option<PacketInfo> {
    if pkt.len() < 20 {
        return None;
    }
    let ihl = (pkt[0] & 0x0F) as usize * 4;
    let total_len = u16::from_be_bytes([pkt[2], pkt[3]]) as usize;
    let protocol = pkt[9];
    let src = [pkt[12], pkt[13], pkt[14], pkt[15]];
    let dst = [pkt[16], pkt[17], pkt[18], pkt[19]];

    let (src_port, dst_port) = match protocol {
        6 | 17 => {
            if pkt.len() >= ihl + 4 {
                let sp = u16::from_be_bytes([pkt[ihl], pkt[ihl + 1]]);
                let dp = u16::from_be_bytes([pkt[ihl + 2], pkt[ihl + 3]]);
                (Some(sp), Some(dp))
            } else {
                (None, None)
            }
        }
        _ => (None, None),
    };

    Some(PacketInfo {
        src_ip: IpAddr::V4(src),
        dst_ip: IpAddr::V4(dst),
        src_port,
        dst_port,
        protocol,
        len: total_len.min(pkt.len()),
    })
}

/// Build PacketInfo from an IPv6 raw packet.
pub fn packet_info_from_ipv6(pkt: &[u8]) -> Option<PacketInfo> {
    if pkt.len() < 40 {
        return None;
    }
    let payload_len = u16::from_be_bytes([pkt[4], pkt[5]]) as usize;
    let next_header = pkt[6];
    let mut src = [0u8; 16];
    let mut dst = [0u8; 16];
    src.copy_from_slice(&pkt[8..24]);
    dst.copy_from_slice(&pkt[24..40]);

    let (src_port, dst_port) = match next_header {
        6 | 17 => {
            if pkt.len() >= 44 {
                let sp = u16::from_be_bytes([pkt[40], pkt[41]]);
                let dp = u16::from_be_bytes([pkt[42], pkt[43]]);
                (Some(sp), Some(dp))
            } else {
                (None, None)
            }
        }
        _ => (None, None),
    };

    Some(PacketInfo {
        src_ip: IpAddr::V6(src),
        dst_ip: IpAddr::V6(dst),
        src_port,
        dst_port,
        protocol: next_header,
        len: 40 + payload_len.min(pkt.len() - 40),
    })
}
