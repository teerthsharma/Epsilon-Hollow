// Epsilon — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Regression: `ChebyshevGuard::from_samples` must not lose the variance to
//! cancellation when the samples share a large offset.
//!
//! It computed `sum_sq / n - mean^2`. For [1e8+1, 1e8+2, 1e8+3] both terms are
//! about 1e16, their difference is below one ulp, and sigma came out 0. The
//! boundary collapsed onto the mean, so `is_safe(1e8 + 1)` was false and a live
//! object was evicted. The exact sigma is sqrt(2/3) = 0.8165, boundary
//! 1e8 + 0.367, and 1e8 + 1 is safe.

use epsilon::{ChebyshevGuard, LivenessAnchor};

#[test]
fn large_offset_samples_keep_their_spread() {
    let samples = [1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0];
    let guard = ChebyshevGuard::from_samples(&samples);
    let exact = libm::sqrt(2.0 / 3.0);

    assert!(
        (guard.safe_boundary() - (1e8 + 2.0 - 2.0 * exact)).abs() < 1e-6,
        "boundary = {}",
        guard.safe_boundary()
    );
    assert!(
        guard.is_safe(1e8 + 1.0),
        "a sample inside 2 sigma must be safe"
    );

    // Same statistics as the anchor computed from the same samples.
    let anchor = LivenessAnchor::from_samples(&samples, 2.0);
    assert_eq!(
        guard.safe_boundary().to_bits(),
        anchor.safe_boundary().to_bits()
    );
}
