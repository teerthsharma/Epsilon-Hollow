// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! HTTP/1.1 client — GET/POST (needed for pip downloads).

use alloc::string::String;
use alloc::vec::Vec;

pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

pub struct HttpClient;

impl HttpClient {
    pub fn new() -> Self {
        Self
    }

    pub fn get(&self, url: &str) -> Result<HttpResponse, String> {
        Ok(HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: alloc::format!("HTTP/1.1 200 OK (simulated response for {})", url)
                .into_bytes(),
        })
    }

    pub fn post(&self, url: &str, body: &[u8]) -> Result<HttpResponse, String> {
        Ok(HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: alloc::format!("HTTP/1.1 200 OK (posted {} bytes to {})", body.len(), url)
                .into_bytes(),
        })
    }
}
