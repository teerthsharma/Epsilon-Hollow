// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! IEEE 802.11 frame codec and WPA2/WPA3 supplicant, as a pure library.
//!
//! # Why this is not in the driver
//!
//! `seal-os`'s WiFi driver probes PCI, asks `bundle` for a signed vendor section, and
//! — since Seal OS ships no vendor sections — reports `section_missing` and refuses
//! every operation. `scan()` returns an empty `Vec` and the boot proof requires the
//! literal field `simulation=absent`. None of that changes here.
//!
//! What was missing was the protocol layer, and it lives here rather than in the
//! driver for one reason: **QEMU emulates no 802.11 NIC**, so an end-to-end
//! association can never be proven in a VM. A library can be proven. Every claim in
//! this crate is gated against a published test vector or is labelled as a property
//! test, and the boundary between the two is drawn explicitly in each module.
//!
//! Nothing in this crate produces a scan result, an SSID, a signal strength, or a
//! network entry. It transforms octets that a caller supplies into other octets. If it
//! is ever wired to a radio, that wiring will be somewhere else and it will be
//! obvious.
//!
//! # Modules
//!
//! * [`frame`] — 802.11 MAC frame codec. Management frames, information elements, RSN.
//! * [`wpa2`] — PMK/PTK derivation, EAPOL-Key, the 4-way handshake, CCMP.
//! * [`wpa3`] — SAE message codec, key schedule and state machine. **The SAE group
//!   arithmetic is not implemented and the SAE handshake cannot complete.** That is
//!   stated at the top of [`wpa3`] in more detail; read it before assuming otherwise.
//!
//! # Verification summary
//!
//! | Path | Gated against |
//! |---|---|
//! | PBKDF2-HMAC-SHA1 | RFC 6070 §2, five vectors |
//! | HMAC-SHA1 | RFC 2202 §3, test cases 1–3 |
//! | AES-CCM (M=8, L=2) | RFC 3610 §8, Packet Vector #1 |
//! | Passphrase to PMK | IEEE 802.11-2016 Annex J.4.2, three vectors |
//! | PTK, EAPOL MIC | **no published vector** — property tests only |
//! | CCMP nonce/AAD | **no published vector** — structural and tamper tests only |
//! | SAE, everything | **no published vector** — round-trip and property tests only |
//!
//! The three "no published vector" rows are the honest state of this crate, not an
//! oversight waiting to be filled in silently. A test whose expected value came from
//! this implementation would prove that the implementation agrees with itself.
//!
//! # Properties held throughout
//!
//! * No `unsafe`.
//! * No panic on parsed input: no `unwrap`, no `expect`, no indexing by a
//!   frame-supplied length without a bounds check first.
//! * Every MIC, tag and confirm comparison runs through `subtle`, via
//!   `Mac::verify_slice` / `Mac::verify_truncated_left` or the `ccm` crate's own tag
//!   check. There is no `==` on a MIC anywhere in this crate.
//! * No cryptographic primitive is implemented here. SHA-1, SHA-256, HMAC, PBKDF2,
//!   AES and CCM all come from RustCrypto; this crate composes them.
//! * `no_std + alloc` clean, because it compiles into a UEFI kernel.
//! * **No entropy source.** Nonces are supplied by the caller, never generated.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod frame;
pub mod wpa2;
pub mod wpa3;

pub use frame::{Frame, FrameError, MacAddr};
pub use wpa2::{Ptk, Supplicant, Wpa2Error};
pub use wpa3::{Sae, SaeError};
