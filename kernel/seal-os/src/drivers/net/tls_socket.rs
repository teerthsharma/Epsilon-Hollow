// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! TLS-over-TCP socket — wraps TcpSocket with TlsSession encryption.
//!
//! [`TlsSocket::connect`] fails closed: it returns an error unless the peer
//! authenticated over this handshake's transcript. There is no switch to skip
//! that, because a chain replayed from an observed connection would pass any
//! check that stops at chain validation.

use crate::drivers::net::tcp::TcpSocket;
use crate::drivers::net::tls::{TlsSession, TlsState};
use alloc::vec::Vec;

/// Ticks spent waiting for the peer's Certificate, CertificateVerify, and
/// Finished messages after ServerHello.
const CERT_WAIT_TICKS: u64 = 1000;

pub struct TlsSocket {
    tcp: TcpSocket,
    tls: TlsSession,
    rx_encrypted: Vec<u8>,
    rx_plaintext: Vec<u8>,
    connected: bool,
}

impl TlsSocket {
    pub fn new() -> Self {
        Self {
            tcp: TcpSocket::new(),
            tls: TlsSession::new(),
            rx_encrypted: Vec::new(),
            rx_plaintext: Vec::new(),
            connected: false,
        }
    }

    pub fn set_psk(&mut self, psk: &[u8; 32]) {
        self.tls.set_psk(psk);
    }

    /// Whether the peer's certificate chain validated against the embedded
    /// trust store. Chain validity alone is *not* authentication — see
    /// [`Self::peer_authenticated`].
    pub fn peer_verified(&self) -> bool {
        self.tls.peer_verified()
    }

    /// Whether the peer proved possession of its key over this handshake.
    pub fn peer_authenticated(&self) -> bool {
        self.tls.peer_authenticated()
    }

    pub fn connect(&mut self, ip: crate::net::IpAddr, port: u16) -> Result<(), &'static str> {
        self.tcp.connect(ip, port);

        let start = crate::drivers::interrupts::ticks();
        while self.tcp.state() != crate::net::tcp::TcpState::Established {
            if crate::drivers::interrupts::ticks().wrapping_sub(start) > 3000 {
                return Err("TCP connect timeout");
            }
            crate::net::tcp::poll();
            crate::net::poll();
        }

        let client_hello = self
            .tls
            .build_client_hello()
            .map_err(|_| "TLS entropy unavailable")?;
        self.tcp.send(&client_hello);

        let mut buf = [0u8; 4096];
        let handshake_start = crate::drivers::interrupts::ticks();
        loop {
            let n = self.tcp.recv(&mut buf);
            if n > 0 {
                self.rx_encrypted.extend_from_slice(&buf[..n]);
                if let Some(record) = Self::pop_record(&mut self.rx_encrypted) {
                    self.tls
                        .handle_server_hello(&record)
                        .map_err(|_| "TLS handshake failed")?;
                    break;
                }
            }
            if crate::drivers::interrupts::ticks().wrapping_sub(handshake_start) > 5000 {
                return Err("TLS handshake timeout");
            }
            crate::net::tcp::poll();
            crate::net::poll();
        }

        // Read the peer's Certificate, CertificateVerify, and Finished, and
        // require all three to check out before any application data moves.
        let cert_start = crate::drivers::interrupts::ticks();
        while !self.tls.peer_authenticated()
            && crate::drivers::interrupts::ticks().wrapping_sub(cert_start) < CERT_WAIT_TICKS
        {
            let n = self.tcp.recv(&mut buf);
            if n > 0 {
                self.rx_encrypted.extend_from_slice(&buf[..n]);
            }
            // Consume handshake records only. Anything else stays buffered
            // for `recv` rather than being dropped on the floor.
            while self.rx_encrypted.first() == Some(&22) {
                let Some(record) = Self::pop_record(&mut self.rx_encrypted) else {
                    break;
                };
                if self.tls.handle_handshake_record(&record[5..]).is_err() {
                    return Err("TLS peer authentication failed");
                }
            }
            crate::net::tcp::poll();
            crate::net::poll();
        }

        // Fail closed. A validated chain is public data and proves nothing on
        // its own; without a CertificateVerify and Finished over this
        // transcript there is no authenticated peer to talk to.
        if !self.tls.peer_authenticated() {
            return Err("TLS peer did not authenticate");
        }

        // ponytail: the peer is authenticated as *some* holder of a
        // trust-store-issued certificate, not as a particular name. `connect`
        // takes an IpAddr and never sees a hostname, so there is nothing to
        // match. Upgrade path: take the hostname alongside the address and
        // require `x509::Certificate::matches_dns` on the leaf, which
        // `x509.rs` already implements and tests.
        self.connected = true;
        Ok(())
    }

    pub fn send(&mut self, data: &[u8]) -> Result<(), &'static str> {
        if !self.connected {
            return Err("TLS socket not connected");
        }
        let encrypted = self.tls.encrypt(data).map_err(|_| "TLS encrypt failed")?;
        self.tcp.send(&encrypted);
        Ok(())
    }

    pub fn recv(&mut self, buf: &mut [u8]) -> usize {
        if !self.rx_plaintext.is_empty() {
            let len = self.rx_plaintext.len().min(buf.len());
            buf[..len].copy_from_slice(&self.rx_plaintext[..len]);
            self.rx_plaintext.drain(..len);
            return len;
        }

        let mut temp = [0u8; 4096];
        let n = self.tcp.recv(&mut temp);
        if n > 0 {
            self.rx_encrypted.extend_from_slice(&temp[..n]);
        }

        while let Some(record) = Self::pop_record(&mut self.rx_encrypted) {
            match self.tls.decrypt(&record) {
                Ok(pt) => self.rx_plaintext.extend_from_slice(&pt),
                Err(_) => break,
            }
        }

        if !self.rx_plaintext.is_empty() {
            let len = self.rx_plaintext.len().min(buf.len());
            buf[..len].copy_from_slice(&self.rx_plaintext[..len]);
            self.rx_plaintext.drain(..len);
            return len;
        }

        0
    }

    pub fn close(&mut self) {
        self.tcp.close();
        self.connected = false;
    }

    pub fn state(&self) -> TlsState {
        self.tls.state()
    }

    pub fn tcp_state(&self) -> crate::net::tcp::TcpState {
        self.tcp.state()
    }

    fn pop_record(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
        if buf.len() < 5 {
            return None;
        }
        let len = u16::from_be_bytes([buf[3], buf[4]]) as usize;
        let total = 5 + len;
        if buf.len() < total {
            return None;
        }
        let record = buf[..total].to_vec();
        buf.drain(..total);
        Some(record)
    }
}

impl Default for TlsSocket {
    fn default() -> Self {
        Self::new()
    }
}
