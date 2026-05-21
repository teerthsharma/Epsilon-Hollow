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

    pub fn get(&self, _url: &str) -> Result<HttpResponse, String> {
        Err(String::from("HTTP not implemented"))
    }

    pub fn post(&self, _url: &str, _body: &[u8]) -> Result<HttpResponse, String> {
        Err(String::from("HTTP not implemented"))
    }
}
