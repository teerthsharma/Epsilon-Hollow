// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! TLS-over-TCP socket — wraps TcpSocket with TlsSession encryption.

use alloc::vec::Vec;
use crate::drivers::net::tcp::TcpSocket;
use crate::drivers::net::tls::{TlsSession, TlsState};

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

        let client_hello = self.tls.build_client_hello();
        self.tcp.send(&client_hello);

        let mut buf = [0u8; 4096];
        let handshake_start = crate::drivers::interrupts::ticks();
        loop {
            let n = self.tcp.recv(&mut buf);
            if n > 0 {
                self.rx_encrypted.extend_from_slice(&buf[..n]);
                if let Some(record) = Self::pop_record(&mut self.rx_encrypted) {
                    self.tls.handle_server_hello(&record)
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

        self.connected = true;
        Ok(())
    }

    pub fn send(&mut self, data: &[u8]) -> Result<(), &'static str> {
        if !self.connected {
            return Err("TLS socket not connected");
        }
        let encrypted = self.tls.encrypt(data)
            .map_err(|_| "TLS encrypt failed")?;
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
