// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Ephemeral X25519 key agreement (RFC 7748) for the TLS handshake.
//!
//! Secrets come straight from [`crate::drivers::entropy::getrandom`], which is
//! RDSEED/RDRAND backed. There is no software PRNG fallback: without hardware
//! entropy [`EphemeralKey::generate`] returns
//! [`EcdheError::EntropyUnavailable`] and the handshake fails closed, matching
//! the property the PSK path already had.

use x25519_dalek::{x25519, X25519_BASEPOINT_BYTES};

/// Named group value for x25519 in the TLS `supported_groups` /
/// `key_share` registries (RFC 8446 §4.2.7).
pub const GROUP_X25519: u16 = 0x001d;

/// Size of an X25519 public key, private key, and shared secret.
pub const X25519_LEN: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcdheError {
    /// Hardware entropy is absent or RDSEED/RDRAND kept failing.
    EntropyUnavailable,
    /// The peer sent a point of small order, so the shared secret is the
    /// all-zero value and carries no secrecy (RFC 7748 §6.1).
    LowOrderPoint,
    /// Peer key share was not 32 bytes.
    BadPeerKeyLength,
}

/// One ephemeral X25519 key pair. Dropped after the handshake derives keys.
pub struct EphemeralKey {
    secret: [u8; X25519_LEN],
    public: [u8; X25519_LEN],
}

impl EphemeralKey {
    /// Draw a fresh private scalar from hardware entropy and derive its
    /// public key. Fails closed if entropy is unavailable.
    pub fn generate() -> Result<Self, EcdheError> {
        let mut secret = [0u8; X25519_LEN];
        if !crate::drivers::entropy::getrandom(&mut secret) {
            return Err(EcdheError::EntropyUnavailable);
        }
        // `x25519` clamps the scalar internally per RFC 7748, so the raw
        // random bytes are a valid private key as-is.
        let public = x25519(secret, X25519_BASEPOINT_BYTES);
        Ok(Self { secret, public })
    }

    /// Build a key pair from a caller-supplied scalar. Test and fixture use
    /// only — it bypasses the entropy check.
    #[cfg(feature = "test-mode")]
    pub fn from_scalar(secret: [u8; X25519_LEN]) -> Self {
        let public = x25519(secret, X25519_BASEPOINT_BYTES);
        Self { secret, public }
    }

    /// The public key to put in the TLS `key_share` extension.
    pub fn public(&self) -> &[u8; X25519_LEN] {
        &self.public
    }

    /// Compute the shared secret from the peer's public key.
    ///
    /// An all-zero result means the peer sent a small-order point; that is
    /// rejected rather than fed into key derivation.
    pub fn agree(&self, peer_public: &[u8]) -> Result<[u8; X25519_LEN], EcdheError> {
        if peer_public.len() != X25519_LEN {
            return Err(EcdheError::BadPeerKeyLength);
        }
        let mut peer = [0u8; X25519_LEN];
        peer.copy_from_slice(peer_public);
        let shared = x25519(self.secret, peer);
        if shared.iter().all(|&b| b == 0) {
            return Err(EcdheError::LowOrderPoint);
        }
        Ok(shared)
    }
}

/// Report whether hardware entropy can currently produce a key.
///
/// Used by the proof emitter to state `entropy=hw` or `entropy=none` from a
/// real draw rather than from a CPUID flag alone.
pub fn hardware_entropy_available() -> bool {
    let mut probe = [0u8; X25519_LEN];
    crate::drivers::entropy::getrandom(&mut probe)
}

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    fn test_shared_secrets_agree() -> TestResult {
        let alice = EphemeralKey::from_scalar([0x11u8; X25519_LEN]);
        let bob = EphemeralKey::from_scalar([0x22u8; X25519_LEN]);
        let Ok(a) = alice.agree(bob.public()) else {
            return TestResult::Fail("alice agreement failed");
        };
        let Ok(b) = bob.agree(alice.public()) else {
            return TestResult::Fail("bob agreement failed");
        };
        test_assert_eq!(a, b);
        test_assert!(a.iter().any(|&x| x != 0), "shared secret is all zero");
        test_assert!(
            alice.public() != bob.public(),
            "distinct scalars produced the same public key"
        );
        TestResult::Pass
    }

    fn test_rfc7748_vector() -> TestResult {
        // RFC 7748 §6.1 test vector: Alice's private key and the resulting
        // public key. Checks the curve itself, not just self-consistency.
        let alice_secret = [
            0x77, 0x07, 0x6d, 0x0a, 0x73, 0x18, 0xa5, 0x7d, 0x3c, 0x16, 0xc1, 0x72, 0x51, 0xb2,
            0x66, 0x45, 0xdf, 0x4c, 0x2f, 0x87, 0xeb, 0xc0, 0x99, 0x2a, 0xb1, 0x77, 0xfb, 0xa5,
            0x1d, 0xb9, 0x2c, 0x2a,
        ];
        let alice_public = [
            0x85, 0x20, 0xf0, 0x09, 0x89, 0x30, 0xa7, 0x54, 0x74, 0x8b, 0x7d, 0xdc, 0xb4, 0x3e,
            0xf7, 0x5a, 0x0d, 0xbf, 0x3a, 0x0d, 0x26, 0x38, 0x1a, 0xf4, 0xeb, 0xa4, 0xa9, 0x8e,
            0xaa, 0x9b, 0x4e, 0x6a,
        ];
        let bob_secret = [
            0x5d, 0xab, 0x08, 0x7e, 0x62, 0x4a, 0x8a, 0x4b, 0x79, 0xe1, 0x7f, 0x8b, 0x83, 0x80,
            0x0e, 0xe6, 0x6f, 0x3b, 0xb1, 0x29, 0x26, 0x18, 0xb6, 0xfd, 0x1c, 0x2f, 0x8b, 0x27,
            0xff, 0x88, 0xe0, 0xeb,
        ];
        let expected_shared = [
            0x4a, 0x5d, 0x9d, 0x5b, 0xa4, 0xce, 0x2d, 0xe1, 0x72, 0x8e, 0x3b, 0xf4, 0x80, 0x35,
            0x0f, 0x25, 0xe0, 0x7e, 0x21, 0xc9, 0x47, 0xd1, 0x9e, 0x33, 0x76, 0xf0, 0x9b, 0x3c,
            0x1e, 0x16, 0x17, 0x42,
        ];
        let alice = EphemeralKey::from_scalar(alice_secret);
        test_assert_eq!(*alice.public(), alice_public);
        let bob = EphemeralKey::from_scalar(bob_secret);
        let Ok(shared) = alice.agree(bob.public()) else {
            return TestResult::Fail("agreement failed");
        };
        test_assert_eq!(shared, expected_shared);
        TestResult::Pass
    }

    fn test_low_order_point_rejected() -> TestResult {
        let key = EphemeralKey::from_scalar([0x33u8; X25519_LEN]);
        // The identity element: any scalar multiple is the all-zero secret.
        test_assert_eq!(
            key.agree(&[0u8; X25519_LEN]),
            Err(EcdheError::LowOrderPoint)
        );
        TestResult::Pass
    }

    fn test_short_peer_key_rejected() -> TestResult {
        let key = EphemeralKey::from_scalar([0x44u8; X25519_LEN]);
        test_assert_eq!(key.agree(&[0u8; 16]), Err(EcdheError::BadPeerKeyLength));
        test_assert_eq!(key.agree(&[]), Err(EcdheError::BadPeerKeyLength));
        TestResult::Pass
    }

    fn test_generate_fails_closed_or_produces_key() -> TestResult {
        // On hardware without RDRAND/RDSEED this must be an error, never a
        // predictable key. Either outcome is acceptable; a zero key is not.
        match EphemeralKey::generate() {
            Ok(key) => {
                test_assert!(
                    key.public().iter().any(|&b| b != 0),
                    "generated public key is all zero"
                );
                TestResult::Pass
            }
            Err(EcdheError::EntropyUnavailable) => TestResult::Pass,
            Err(_) => TestResult::Fail("generate returned an unexpected error"),
        }
    }

    pub fn register_all() {
        crate::testing::register_test("ecdhe::shared_secrets_agree", test_shared_secrets_agree);
        crate::testing::register_test("ecdhe::rfc7748_vector", test_rfc7748_vector);
        crate::testing::register_test(
            "ecdhe::low_order_point_rejected",
            test_low_order_point_rejected,
        );
        crate::testing::register_test(
            "ecdhe::short_peer_key_rejected",
            test_short_peer_key_rejected,
        );
        crate::testing::register_test(
            "ecdhe::generate_fails_closed",
            test_generate_fails_closed_or_produces_key,
        );
    }
}
