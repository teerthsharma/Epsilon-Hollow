// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Aether-Link prefetch engine — adaptive I/O prediction.
//!
//! Ported from `kernel/aether/aether-link/` for no_std. Uses 6D telemetry extraction
//! and a POVM-inspired trigonometric heuristic for prefetch decisions.
//!
//! No per-decision latency is measured in-kernel: this module contains no
//! `rdtsc` or cycle capture, and the `cycles` field is a decision *counter*
//! incremented once per call, not a timing. Any ns/decision figure for this code
//! would have to come from the aether-link host benchmark, not from here.

use core::f32::consts::{FRAC_PI_2, PI};

#[derive(Debug, Clone, Copy)]
pub enum PrefetchPreset {
    Gaming,
    Hft,
    ModelTraining,
}

pub struct PrefetchEngine {
    epsilon: f32,
    phi: f32,
    lambda: [f32; 3],
    bias: f32,
    cycles: u64,
    prefetch_hits: u64,
    prefetch_misses: u64,
    preset: PrefetchPreset,
    /// The most recent [`PrefetchEngine::record_lba`] arguments, oldest first.
    /// Only `history_len` entries are live.
    lba_history: [u64; 64],
    history_len: usize,
}

impl PrefetchEngine {
    pub fn new_gaming() -> Self {
        Self::with_params(0.4, 0.2, [0.15, 0.25, 0.35], 0.05, PrefetchPreset::Gaming)
    }

    pub fn new_hft() -> Self {
        Self::with_params(0.65, 0.05, [0.03, 0.08, 0.15], -0.02, PrefetchPreset::Hft)
    }

    /// Model-training preset. Adopts the `stratum` fit controller's published
    /// prefetch threshold when a registered training workload has produced one.
    ///
    /// **Nothing in this tree calls this.** Every engine actually constructed is
    /// [`PrefetchEngine::new_gaming`] — one per `read_sectors` in the AHCI driver
    /// and one per `prefetch status` in the shell. So the threshold the
    /// controller publishes reaches no I/O decision until a model-training read
    /// path builds this preset, and `stratum`'s `prefetch_epsilon` is a
    /// recommendation rather than kernel control until then.
    pub fn new_model_training() -> Self {
        let mut engine = Self::with_params(
            0.30,
            0.15,
            [0.20, 0.30, 0.40],
            0.10,
            PrefetchPreset::ModelTraining,
        );
        if let Some(eps) = crate::ml_engine::stratum::training_prefetch_epsilon() {
            engine.epsilon = eps.clamp(0.1, 0.9);
        }
        engine
    }

    fn with_params(
        epsilon: f32,
        phi: f32,
        lambda: [f32; 3],
        bias: f32,
        preset: PrefetchPreset,
    ) -> Self {
        Self {
            epsilon,
            phi,
            lambda,
            bias,
            cycles: 0,
            prefetch_hits: 0,
            prefetch_misses: 0,
            preset,
            lba_history: [0; 64],
            history_len: 0,
        }
    }

    /// Append `lba` to the history, dropping the oldest entry once 64 are held.
    ///
    /// This shifts rather than writing into a ring, because everything
    /// downstream of [`PrefetchEngine::current_stream`] reads the history as a
    /// time series: `extract_telemetry` takes `delta` as `last - first` and
    /// differences adjacent pairs. A ring hands back its backing array in slot
    /// order, so past the 64th record a sequential scan arrived with a
    /// fabricated backwards seam at the write position.
    //
    // ponytail: 512-byte shift per record once the history is full, which is
    // fine while nothing calls this more than once per block request. A caller
    // on a hot path wants the ring back plus a copy-out that rotates it into
    // arrival order.
    pub fn record_lba(&mut self, lba: u64) {
        if self.history_len == 64 {
            self.lba_history.copy_within(1.., 0);
            self.lba_history[63] = lba;
        } else {
            self.lba_history[self.history_len] = lba;
            self.history_len += 1;
        }
    }

    fn current_stream(&self) -> &[u64] {
        &self.lba_history[..self.history_len]
    }

    /// Decide whether the block after `lba_stream` is worth fetching early.
    ///
    /// Two LBAs are the floor, on either branch: a caller either passes a
    /// stream of at least two, or passes fewer and falls back to at least two
    /// it has recorded through [`PrefetchEngine::record_lba`]. Fewer than that
    /// on both is not "no" — there is nothing to extrapolate from — but it is
    /// counted as a miss and returned as `false`, so an engine asked too early
    /// depresses its own [`PrefetchEngine::hit_ratio`].
    ///
    /// **No I/O path in this tree calls this.** The AHCI read path did, with a
    /// fresh engine and a one-element slice, so it took neither branch and its
    /// verdict was false on every call; that call is gone and the reasoning is
    /// on `AhciPort::read_sectors`. The remaining constructor call, the shell's
    /// `prefetch status`, builds an engine and reports its counters without
    /// ever deciding, which is why that display always reads 0 decisions and a
    /// 0% hit ratio. Reaching a real verdict needs an engine that outlives one
    /// request.
    pub fn should_prefetch(&mut self, lba_stream: &[u64]) -> bool {
        self.cycles += 1;

        let stream = if lba_stream.len() >= 2 {
            lba_stream
        } else {
            let s = self.current_stream();
            if s.len() < 2 {
                self.prefetch_misses += 1;
                return false;
            }
            s
        };

        let telemetry = extract_telemetry(stream);
        let q_angles = prepare_angles(telemetry);

        let (a1, a2, a3) = simulate_qpu_eval(&q_angles, self.phi);

        self.phi = (self.phi + self.lambda[1] * a2) % (2.0 * PI);
        self.epsilon += self.lambda[0] * a1;
        self.epsilon = self.epsilon.clamp(0.1, 0.9);

        let exponent = -(self.lambda[2] * a3 + self.bias);
        let p_fetch = fast_sigmoid(exponent);

        let decision = p_fetch > self.epsilon;
        if decision {
            self.prefetch_hits += 1;
        } else {
            self.prefetch_misses += 1;
        }
        decision
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon as f64
    }

    pub fn phi(&self) -> f64 {
        self.phi as f64
    }

    pub fn hit_ratio(&self) -> f64 {
        let total = self.prefetch_hits + self.prefetch_misses;
        if total == 0 {
            0.0
        } else {
            self.prefetch_hits as f64 / total as f64
        }
    }

    pub fn total_decisions(&self) -> u64 {
        self.cycles
    }

    pub fn preset(&self) -> PrefetchPreset {
        self.preset
    }

    /// Bias towards prefetching for files over 100 MiB.
    ///
    /// Nothing calls this, so no file size has ever moved `epsilon`. Same
    /// standing as [`PrefetchEngine::new_model_training`]: kept because it is
    /// the ported aether-link surface, not because it is wired.
    pub fn large_file_hint(&mut self, size_bytes: u64) {
        if size_bytes > 100 * 1024 * 1024 {
            self.epsilon = self.epsilon.min(0.35);
        }
    }
}

fn extract_telemetry(lba_stream: &[u64]) -> [f32; 6] {
    let len = lba_stream.len();
    if len < 2 {
        return [0.0; 6];
    }

    let last = lba_stream[len - 1];
    let first = lba_stream[0];
    let delta = last.wrapping_sub(first) as f32;
    let n_steps = (len - 1) as f32;
    let velocity = delta / n_steps;

    let mut mean = 0.0f32;
    let mut m2 = 0.0f32;
    for i in 1..len {
        let d = lba_stream[i].wrapping_sub(lba_stream[i - 1]) as f32;
        let count = i as f32;
        let delta_m = d - mean;
        mean += delta_m / count;
        let delta_m2 = d - mean;
        m2 += delta_m * delta_m2;
    }
    let variance = m2 / n_steps;

    let k = len as f32 / 4.0;
    let w = 2.0 * PI * k / len as f32;
    let coeff = 2.0 * libm::cosf(w);
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2;
    for i in 1..len {
        let d = lba_stream[i].wrapping_sub(lba_stream[i - 1]) as f32;
        s2 = s1;
        s1 = s0;
        s0 = d + coeff * s1 - s2;
    }
    let power = s0 * s0 + s1 * s1 - coeff * s0 * s1;
    let spectrum = if power > 0.0 {
        libm::sqrtf(power) / n_steps
    } else {
        0.0
    };

    let decay = 2.0 / len as f32;
    let mut w_sum = 0.0f32;
    let mut w_recent = 0.0f32;
    let half = len / 2;
    for i in 0..len {
        let w_i = libm::expf(-(i as f32) * decay);
        w_sum += w_i;
        if i >= half {
            w_recent += w_i;
        }
    }
    let history = if w_sum > 0.0 { w_recent / w_sum } else { 0.5 };

    let mut unique_count = 0u32;
    let mut prev_delta = u64::MAX;
    for i in 1..len {
        let d = lba_stream[i].wrapping_sub(lba_stream[i - 1]);
        if d != prev_delta {
            unique_count += 1;
            prev_delta = d;
        }
    }
    let context = unique_count as f32 / n_steps;

    [delta, velocity, variance, spectrum, history, context]
}

fn prepare_angles(features: [f32; 6]) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    for (i, &f) in features.iter().enumerate() {
        out[i] = fast_atan(f) * 2.0;
    }
    out
}

#[inline(always)]
fn simulate_qpu_eval(angles: &[f32], phi: f32) -> (f32, f32, f32) {
    let s = angles[0] + angles[1];
    let a1 = libm::cosf(s + phi);
    let a2 = libm::sinf(s * 0.5 - phi);
    let a3 = libm::cosf(angles[2] * phi);
    (a1, a2, a3)
}

#[inline(always)]
fn fast_atan(x: f32) -> f32 {
    if x.abs() > 1e6 {
        return if x > 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 };
    }
    x / (1.0 + 0.28125 * x * x)
}

#[inline(always)]
fn fast_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

pub fn init() {}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert_eq;
    use crate::testing::TestResult;
    use alloc::vec::Vec;

    /// `should_prefetch`'s two branches must decide on the same stream: passing
    /// the LBAs explicitly and letting the engine fall back to the ones it
    /// recorded are documented as the same thing, and the fallback is the only
    /// branch a block driver can reach without keeping its own history.
    ///
    /// Past the 64th recorded LBA they were not the same thing.
    /// `current_stream` returned the whole backing array once `history_len`
    /// hit 64, and the array is in slot order, not arrival order: after 73
    /// records it began at the 65th LBA, ran to the 73rd, then jumped back to
    /// the 9th. Every consumer downstream reads it as a time series —
    /// `extract_telemetry` takes `delta` as `last - first` and differences
    /// adjacent pairs — so a strictly ascending sequential scan arrived as a
    /// stream with one fabricated backwards seam, a negative `delta`, and a
    /// variance dominated by a step the drive never made.
    ///
    /// 73 records rather than 65: it puts the seam in the middle, where a
    /// rotation is distinguishable from an off-by-one at either end.
    ///
    /// This is the fallback branch only. No LBA stream reaches this engine from
    /// any I/O path in the tree — see `should_prefetch`'s note — so the check
    /// runs against the type directly, the same constraint that made
    /// `ahci::large_read_matches_chunked_small_reads` test the chunk plan
    /// instead of a device.
    fn test_recorded_history_matches_explicit_stream() -> TestResult {
        let scan: Vec<u64> = (0..73).map(|i| 100 + i * 8).collect();
        let mut recorded = PrefetchEngine::new_gaming();
        for &lba in &scan {
            recorded.record_lba(lba);
        }

        // The newest 64, in the order the drive asked for them.
        let expected = &scan[scan.len() - 64..];
        test_assert_eq!(recorded.current_stream(), expected);

        // Same engine parameters, same stream, so the fallback branch must
        // reach the verdict the explicit branch reaches — and must move
        // `epsilon` by the same amount getting there. Epsilon is the sharper of
        // the two: it carries the whole telemetry vector, where the verdict is
        // one threshold comparison that a wrong stream can still land on the
        // right side of. Both are computed by the same code from the same
        // input, so bitwise equality is the correct assertion.
        let mut explicit = PrefetchEngine::new_gaming();
        test_assert_eq!(
            recorded.should_prefetch(&[]),
            explicit.should_prefetch(expected)
        );
        test_assert_eq!(recorded.epsilon(), explicit.epsilon());
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "prefetch::recorded_history_matches_explicit_stream",
            test_recorded_history_matches_explicit_stream,
        );
    }
}
