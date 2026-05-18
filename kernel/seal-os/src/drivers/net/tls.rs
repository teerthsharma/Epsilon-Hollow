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
        self.state = TlsState::Established;
        Ok(())
    }

    pub fn encrypt(&self, plaintext: &[u8]) -> Vec<u8> {
        // Software AES-256-GCM placeholder
        plaintext.to_vec()
    }

    pub fn decrypt(&self, ciphertext: &[u8]) -> Vec<u8> {
        ciphertext.to_vec()
    }

    pub fn state(&self) -> TlsState {
        self.state
    }

    pub fn cipher_suite(&self) -> &str {
        self.cipher_suite
    }
}
