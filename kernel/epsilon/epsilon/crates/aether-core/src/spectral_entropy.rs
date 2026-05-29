// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! no_std spectral coherence, entropy, Haar wavelet entropy, and decay bounds.

use alloc::vec::Vec;

const EPS: f64 = 1e-24;
const LOG2: f64 = core::f64::consts::LN_2;

/// Compute spectral coherence from the non-DC DFT power spectrum.
///
/// This mirrors the Python formula `sum(|F[k]|^2) / (N * variance)`.
pub fn spectral_coherence(signal: &[f64]) -> f64 {
    let n = signal.len();
    if n < 2 {
        return 0.0;
    }

    let var = variance(signal);
    if var < EPS {
        return n as f64;
    }

    dft_non_dc_power(signal).into_iter().sum::<f64>() / (n as f64 * var)
}

/// Compute Shannon entropy of the normalized non-DC DFT power spectrum.
pub fn spectral_entropy(signal: &[f64]) -> f64 {
    let power = dft_non_dc_power(signal);
    entropy_from_energy(&power)
}

/// Coefficients returned by [`haar_wavelet_transform`].
#[derive(Debug, Clone, PartialEq)]
pub struct HaarWaveletTransform {
    /// Final approximation coefficients after all Haar levels.
    pub approximation: Vec<f64>,
    /// Detail coefficients at each level, finest to coarsest.
    pub details: Vec<Vec<f64>>,
}

/// Compute the discrete Haar wavelet transform.
///
/// The input is zero-padded to the next power of two. The returned details
/// match the legacy ordering: level 1 first, then progressively coarser levels.
pub fn haar_wavelet_transform(signal: &[f64]) -> HaarWaveletTransform {
    let n = signal.len();
    let mut power = 1usize;
    while power < n {
        power *= 2;
    }

    let mut current = Vec::with_capacity(power);
    current.resize(power, 0.0);
    let mut i = 0;
    while i < n {
        current[i] = signal[i];
        i += 1;
    }

    let mut details = Vec::new();
    while current.len() > 1 {
        let half = current.len() / 2;
        let mut approx = Vec::with_capacity(half);
        approx.resize(half, 0.0);
        let mut detail = Vec::with_capacity(half);
        detail.resize(half, 0.0);

        let mut j = 0;
        while j < half {
            let a = current[2 * j];
            let b = current[2 * j + 1];
            approx[j] = (a + b) / libm::sqrt(2.0);
            detail[j] = (a - b) / libm::sqrt(2.0);
            j += 1;
        }

        details.push(detail);
        current = approx;
    }

    HaarWaveletTransform {
        approximation: current,
        details,
    }
}

/// Compute Haar wavelet entropy for each scale, finest to coarsest.
pub fn wavelet_entropy_per_scale(signal: &[f64]) -> Vec<f64> {
    let transform = haar_wavelet_transform(signal);
    let mut entropies = Vec::new();

    for detail in transform.details {
        let mut energy = Vec::with_capacity(detail.len());
        energy.resize(detail.len(), 0.0);
        let mut i = 0;
        while i < detail.len() {
            energy[i] = detail[i] * detail[i];
            i += 1;
        }

        entropies.push(entropy_from_energy(&energy));
    }

    entropies
}

/// KL divergence between normalized non-DC power spectra.
pub fn spectral_kl_divergence(signal_a: &[f64], signal_b: &[f64]) -> f64 {
    let n = signal_a.len().min(signal_b.len());
    if n < 2 {
        return 0.0;
    }

    let p_power = dft_non_dc_power(&signal_a[..n]);
    let q_power = dft_non_dc_power(&signal_b[..n]);
    let p = normalized_distribution(&p_power);
    let q = normalized_distribution(&q_power);

    let mut kl = 0.0;
    let mut i = 0;
    while i < p.len().min(q.len()) {
        kl += p[i] * libm::log((p[i] + 1e-10) / (q[i] + 1e-10));
        i += 1;
    }
    kl.max(0.0)
}

/// Result returned by [`SpectralCoherenceTracker::update`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralUpdate {
    /// Current spectral coherence.
    pub coherence: f64,
    /// Current spectral entropy.
    pub spectral_entropy: f64,
    /// Whether the spectral coherence invariant held for this update.
    pub invariant_holds: bool,
    /// Running invariant violation rate.
    pub violation_rate: f64,
    /// Number of samples currently retained in the rolling window.
    pub window_size: usize,
}

impl Default for SpectralUpdate {
    fn default() -> Self {
        Self {
            coherence: 0.0,
            spectral_entropy: 0.0,
            invariant_holds: true,
            violation_rate: 0.0,
            window_size: 0,
        }
    }
}

/// Rolling tracker for the Spectral Coherence Invariant theorem.
#[derive(Debug, Clone)]
pub struct SpectralCoherenceTracker {
    window_size: usize,
    epsilon_max: f64,
    history: Vec<f64>,
    coherence_history: Vec<f64>,
    invariant_violations: usize,
    total_checks: usize,
}

impl SpectralCoherenceTracker {
    /// Construct a tracker with a rolling window and maximum governor epsilon.
    pub fn new(window_size: usize, epsilon_max: f64) -> Self {
        Self {
            window_size: window_size.max(1),
            epsilon_max,
            history: Vec::new(),
            coherence_history: Vec::new(),
            invariant_violations: 0,
            total_checks: 0,
        }
    }

    /// Feed one I/O delta and check the spectral coherence invariant.
    pub fn update(&mut self, delta: f64, epsilon_t: f64) -> SpectralUpdate {
        self.history.push(delta);
        while self.history.len() > self.window_size {
            self.history.remove(0);
        }

        if self.history.len() < 4 {
            return SpectralUpdate {
                window_size: self.history.len(),
                ..SpectralUpdate::default()
            };
        }

        let coherence = spectral_coherence(&self.history);
        let spectral_entropy = spectral_entropy(&self.history);
        let mut invariant_holds = true;

        if let Some(&previous) = self.coherence_history.last() {
            let factor = if self.epsilon_max <= 0.0 {
                0.0
            } else {
                1.0 - epsilon_t / self.epsilon_max
            };
            let bound = previous * factor;
            self.total_checks += 1;
            if coherence < bound - 1e-6 {
                self.invariant_violations += 1;
                invariant_holds = false;
            }
        }
        self.coherence_history.push(coherence);

        SpectralUpdate {
            coherence,
            spectral_entropy,
            invariant_holds,
            violation_rate: self.violation_rate(),
            window_size: self.history.len(),
        }
    }

    /// Predict the Manifold Decay Theorem Betti-0 survival bound.
    pub fn predict_betti_survival(
        &self,
        betti_0_initial: usize,
        time_elapsed: f64,
        decay_lambda: f64,
        reinforcement_density: f64,
    ) -> f64 {
        let rho = reinforcement_density.clamp(0.0, 1.0);
        let exponent = -decay_lambda * time_elapsed * (1.0 - rho);
        betti_0_initial as f64 * libm::exp(exponent)
    }

    /// Return the running invariant violation rate.
    #[inline]
    pub fn violation_rate(&self) -> f64 {
        if self.total_checks == 0 {
            0.0
        } else {
            self.invariant_violations as f64 / self.total_checks as f64
        }
    }
}

fn variance(signal: &[f64]) -> f64 {
    let mean = signal.iter().copied().sum::<f64>() / signal.len() as f64;
    let mut acc = 0.0;
    for &x in signal {
        let d = x - mean;
        acc += d * d;
    }
    acc / signal.len() as f64
}

fn dft_non_dc_power(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n < 2 {
        return Vec::new();
    }

    let bins = n / 2;
    let mut power = Vec::with_capacity(bins);
    let mut k = 1usize;
    while k <= bins {
        let mut re = 0.0;
        let mut im = 0.0;
        let mut t = 0usize;
        while t < n {
            let angle = -2.0 * core::f64::consts::PI * k as f64 * t as f64 / n as f64;
            re += signal[t] * libm::cos(angle);
            im += signal[t] * libm::sin(angle);
            t += 1;
        }
        power.push(re * re + im * im);
        k += 1;
    }
    power
}

fn entropy_from_energy(energy: &[f64]) -> f64 {
    let total = energy.iter().copied().sum::<f64>();
    if total < EPS {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &e in energy {
        let p = e / total;
        if p > 0.0 {
            entropy -= p * (libm::log(p) / LOG2);
        }
    }
    entropy
}

fn normalized_distribution(power: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(power.len());
    let total = power.iter().copied().sum::<f64>();
    if power.is_empty() {
        return out;
    }

    if total < EPS {
        let uniform = 1.0 / power.len() as f64;
        let mut i = 0;
        while i < power.len() {
            out.push(uniform);
            i += 1;
        }
    } else {
        for &p in power {
            out.push(p / total);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_metrics_identify_periodic_and_constant_signals() {
        let periodic = [1.0, -1.0, 1.0, -1.0];
        assert!((spectral_coherence(&periodic) - 4.0).abs() < 1e-12);
        assert!(spectral_entropy(&periodic) < 1e-9);

        let constant = [3.0, 3.0, 3.0, 3.0];
        assert_eq!(spectral_coherence(&constant), 4.0);
        assert_eq!(spectral_entropy(&constant), 0.0);
    }

    #[test]
    fn haar_entropy_and_tracker_cover_sci_and_decay() {
        let signal = [1.0, 1.0, -1.0, -1.0];
        let transform = haar_wavelet_transform(&signal);
        assert_eq!(transform.approximation.len(), 1);
        assert!((transform.approximation[0] - 0.0).abs() < 1e-12);
        assert_eq!(transform.details.len(), 2);
        assert_eq!(transform.details[0].len(), 2);
        assert!((transform.details[1][0] - 2.0).abs() < 1e-12);

        let entropies = wavelet_entropy_per_scale(&signal);
        assert_eq!(entropies.len(), 2);
        assert!(entropies[0] <= 1.0);

        let mut tracker = SpectralCoherenceTracker::new(8, 0.9);
        let mut last = SpectralUpdate::default();
        for delta in [1.0, -1.0, 1.0, -1.0, 1.0] {
            last = tracker.update(delta, 0.1);
        }
        assert!(last.window_size >= 4);
        assert!(last.coherence > 0.0);
        assert!(last.violation_rate >= 0.0);

        let survival = tracker.predict_betti_survival(10, 2.0, 0.5, 0.5);
        assert!(survival > 6.0 && survival < 7.0);
    }

    #[test]
    fn spectral_kl_divergence_is_zero_for_identical_signals() {
        let signal = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        assert!(spectral_kl_divergence(&signal, &signal) < 1e-12);
    }
}
