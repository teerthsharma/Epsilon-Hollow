// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! TLS 1.3 — software implementation using aether-core math for AES/ChaCha20.

use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TlsState {
    Initial,
    ClientHello,
    ServerHello,
    Handshake,
    Established,
    Closed,
}

pub struct TlsSession {
    state: TlsState,
    cipher_suite: &'static str,
}

impl TlsSession {
    pub fn new() -> Self {
        Self {
            state: TlsState::Initial,
            cipher_suite: "TLS_AES_256_GCM_SHA384",
        }
    }

    pub fn handshake(&mut self) -> Result<(), &'static str> {
        Err("TLS not implemented")
    }

    pub fn encrypt(&self, _plaintext: &[u8]) -> Result<Vec<u8>, &'static str> {
        Err("TLS not implemented")
    }

    pub fn decrypt(&self, _ciphertext: &[u8]) -> Result<Vec<u8>, &'static str> {
        Err("TLS not implemented")
    }

    pub fn state(&self) -> TlsState {
        self.state
    }

    pub fn cipher_suite(&self) -> &str {
        self.cipher_suite
    }
}
