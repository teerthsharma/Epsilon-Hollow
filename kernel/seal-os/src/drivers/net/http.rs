// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! HTTP/1.1 client — GET/POST with real TCP socket backend + TLS for HTTPS.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Parsed URL components.
struct ParsedUrl {
    host: String,
    port: u16,
    path: String,
    secure: bool,
}

fn parse_url(url: &str) -> Result<ParsedUrl, String> {
    let rest = if url.starts_with("https://") {
        &url[8..]
    } else if url.starts_with("http://") {
        &url[7..]
    } else {
        return Err(String::from("URL must start with http:// or https://"));
    };
    let secure = url.starts_with("https://");

    let (host_port, path) = match rest.find('/') {
        Some(idx) => (&rest[..idx], &rest[idx..]),
        None => (rest, "/"),
    };

    let (host, port) = match host_port.find(':') {
        Some(idx) => {
            let h = &host_port[..idx];
            let p = host_port[idx + 1..]
                .parse::<u16>()
                .map_err(|_| String::from("bad port"))?;
            (h, p)
        }
        None => (host_port, if secure { 443 } else { 80 }),
    };

    Ok(ParsedUrl {
        host: String::from(host),
        port,
        path: String::from(path),
        secure,
    })
}

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

    /// Perform HTTP GET. Returns Err if TCP/TLS fails or response malformed.
    pub fn get(&self, url: &str) -> Result<HttpResponse, String> {
        let parsed = parse_url(url)?;
        let ip = Self::resolve_host(&parsed.host)?;

        if parsed.secure {
            self.get_https(&parsed, ip)
        } else {
            self.get_http(&parsed, ip)
        }
    }

    /// Perform HTTP POST.
    pub fn post(&self, url: &str, body: &[u8]) -> Result<HttpResponse, String> {
        let parsed = parse_url(url)?;
        let ip = Self::resolve_host(&parsed.host)?;

        if parsed.secure {
            self.post_https(&parsed, ip, body)
        } else {
            self.post_http(&parsed, ip, body)
        }
    }

    fn get_http(&self, parsed: &ParsedUrl, ip: [u8; 4]) -> Result<HttpResponse, String> {
        let mut tcp = crate::drivers::net::tcp::TcpSocket::new();
        tcp.connect(crate::net::IpAddr::V4(ip), parsed.port);

        let start = crate::drivers::interrupts::ticks();
        while tcp.state() != crate::net::tcp::TcpState::Established {
            if crate::drivers::interrupts::ticks().wrapping_sub(start) > 3000 {
                return Err(String::from("TCP connect timeout"));
            }
            crate::net::tcp::poll();
            crate::net::poll();
        }

        let req = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nUser-Agent: SealOS/{}\r\n\r\n",
            parsed.path,
            parsed.host,
            crate::VERSION
        );
        tcp.send(req.as_bytes());

        let mut buf = [0u8; 4096];
        let mut response = Vec::new();
        let read_start = crate::drivers::interrupts::ticks();
        loop {
            let n = tcp.recv(&mut buf);
            if n > 0 {
                response.extend_from_slice(&buf[..n]);
            }
            if tcp.state() == crate::net::tcp::TcpState::Closed
                || tcp.state() == crate::net::tcp::TcpState::CloseWait
            {
                loop {
                    let n = tcp.recv(&mut buf);
                    if n == 0 {
                        break;
                    }
                    response.extend_from_slice(&buf[..n]);
                }
                break;
            }
            if crate::drivers::interrupts::ticks().wrapping_sub(read_start) > 5000 {
                return Err(String::from("HTTP read timeout"));
            }
            crate::net::tcp::poll();
            crate::net::poll();
        }

        tcp.close();
        Self::parse_response(&response)
    }

    fn post_http(
        &self,
        parsed: &ParsedUrl,
        ip: [u8; 4],
        body: &[u8],
    ) -> Result<HttpResponse, String> {
        let mut tcp = crate::drivers::net::tcp::TcpSocket::new();
        tcp.connect(crate::net::IpAddr::V4(ip), parsed.port);

        let start = crate::drivers::interrupts::ticks();
        while tcp.state() != crate::net::tcp::TcpState::Established {
            if crate::drivers::interrupts::ticks().wrapping_sub(start) > 3000 {
                return Err(String::from("TCP connect timeout"));
            }
            crate::net::tcp::poll();
            crate::net::poll();
        }

        let req = format!(
            "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Length: {}\r\nConnection: close\r\nUser-Agent: SealOS/{}\r\n\r\n",
            parsed.path, parsed.host, body.len(), crate::VERSION
        );
        tcp.send(req.as_bytes());
        tcp.send(body);

        let mut buf = [0u8; 4096];
        let mut response = Vec::new();
        let read_start = crate::drivers::interrupts::ticks();
        loop {
            let n = tcp.recv(&mut buf);
            if n > 0 {
                response.extend_from_slice(&buf[..n]);
            }
            if tcp.state() == crate::net::tcp::TcpState::Closed
                || tcp.state() == crate::net::tcp::TcpState::CloseWait
            {
                loop {
                    let n = tcp.recv(&mut buf);
                    if n == 0 {
                        break;
                    }
                    response.extend_from_slice(&buf[..n]);
                }
                break;
            }
            if crate::drivers::interrupts::ticks().wrapping_sub(read_start) > 5000 {
                return Err(String::from("HTTP read timeout"));
            }
            crate::net::tcp::poll();
            crate::net::poll();
        }

        tcp.close();
        Self::parse_response(&response)
    }

    fn get_https(&self, parsed: &ParsedUrl, ip: [u8; 4]) -> Result<HttpResponse, String> {
        let mut tls = crate::drivers::net::tls_socket::TlsSocket::new();
        if let Err(e) = tls.connect(crate::net::IpAddr::V4(ip), parsed.port) {
            return Err(format!("TLS connect failed: {}", e));
        }

        let req = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nUser-Agent: SealOS/{}\r\n\r\n",
            parsed.path,
            parsed.host,
            crate::VERSION
        );
        if let Err(e) = tls.send(req.as_bytes()) {
            return Err(format!("TLS send failed: {}", e));
        }

        let mut buf = [0u8; 4096];
        let mut response = Vec::new();
        let read_start = crate::drivers::interrupts::ticks();
        loop {
            let n = tls.recv(&mut buf);
            if n > 0 {
                response.extend_from_slice(&buf[..n]);
            }
            if tls.tcp_state() == crate::net::tcp::TcpState::Closed
                || tls.tcp_state() == crate::net::tcp::TcpState::CloseWait
            {
                loop {
                    let n = tls.recv(&mut buf);
                    if n == 0 {
                        break;
                    }
                    response.extend_from_slice(&buf[..n]);
                }
                break;
            }
            if crate::drivers::interrupts::ticks().wrapping_sub(read_start) > 5000 {
                return Err(String::from("HTTPS read timeout"));
            }
            crate::net::tcp::poll();
            crate::net::poll();
        }

        tls.close();
        Self::parse_response(&response)
    }

    fn post_https(
        &self,
        parsed: &ParsedUrl,
        ip: [u8; 4],
        body: &[u8],
    ) -> Result<HttpResponse, String> {
        let mut tls = crate::drivers::net::tls_socket::TlsSocket::new();
        if let Err(e) = tls.connect(crate::net::IpAddr::V4(ip), parsed.port) {
            return Err(format!("TLS connect failed: {}", e));
        }

        let req = format!(
            "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Length: {}\r\nConnection: close\r\nUser-Agent: SealOS/{}\r\n\r\n",
            parsed.path, parsed.host, body.len(), crate::VERSION
        );
        if let Err(e) = tls.send(req.as_bytes()) {
            return Err(format!("TLS send failed: {}", e));
        }
        if let Err(e) = tls.send(body) {
            return Err(format!("TLS send body failed: {}", e));
        }

        let mut buf = [0u8; 4096];
        let mut response = Vec::new();
        let read_start = crate::drivers::interrupts::ticks();
        loop {
            let n = tls.recv(&mut buf);
            if n > 0 {
                response.extend_from_slice(&buf[..n]);
            }
            if tls.tcp_state() == crate::net::tcp::TcpState::Closed
                || tls.tcp_state() == crate::net::tcp::TcpState::CloseWait
            {
                loop {
                    let n = tls.recv(&mut buf);
                    if n == 0 {
                        break;
                    }
                    response.extend_from_slice(&buf[..n]);
                }
                break;
            }
            if crate::drivers::interrupts::ticks().wrapping_sub(read_start) > 5000 {
                return Err(String::from("HTTPS read timeout"));
            }
            crate::net::tcp::poll();
            crate::net::poll();
        }

        tls.close();
        Self::parse_response(&response)
    }

    fn resolve_host(host: &str) -> Result<[u8; 4], String> {
        if let Some(ip) = Self::parse_ipv4(host) {
            return Ok(ip);
        }
        match crate::net::dns::resolve(host) {
            Ok(Some(ip)) => Ok(ip),
            Ok(None) => Err(String::from("DNS resolve pending — try again")),
            Err(e) => Err(format!("DNS error: {}", e)),
        }
    }

    fn parse_ipv4(s: &str) -> Option<[u8; 4]> {
        let mut parts = s.split('.');
        let mut ip = [0u8; 4];
        for i in 0..4 {
            let part = parts.next()?;
            ip[i] = part.parse::<u8>().ok()?;
        }
        if parts.next().is_some() {
            return None;
        }
        Some(ip)
    }

    fn parse_response(data: &[u8]) -> Result<HttpResponse, String> {
        let text = String::from_utf8_lossy(data);
        let mut lines = text.lines();

        let status_line = lines.next().ok_or("empty response")?;
        let mut status_parts = status_line.split_whitespace();
        let _http_version = status_parts.next().ok_or("missing HTTP version")?;
        let status_str = status_parts.next().ok_or("missing status code")?;
        let status = status_str.parse::<u16>().map_err(|_| "bad status code")?;

        let mut headers = Vec::new();
        for line in lines.by_ref() {
            if line.is_empty() {
                break;
            }
            if let Some(idx) = line.find(':') {
                let key = String::from(line[..idx].trim());
                let val = String::from(line[idx + 1..].trim());
                headers.push((key, val));
            }
        }

        let header_end = text
            .find("\r\n\r\n")
            .or_else(|| text.find("\n\n"))
            .ok_or("no header/body boundary")?;
        let body_start = header_end
            + if text[header_end..].starts_with("\r\n\r\n") {
                4
            } else {
                2
            };
        let body = data[body_start..].to_vec();

        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_url_http() {
        let url = parse_url("http://example.com/path").unwrap();
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, 80);
        assert_eq!(url.path, "/path");
        assert!(!url.secure);
    }

    #[test]
    fn test_parse_url_https() {
        let url = parse_url("https://example.com/").unwrap();
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, 443);
        assert_eq!(url.path, "/");
        assert!(url.secure);
    }

    #[test]
    fn test_parse_url_with_port() {
        let url = parse_url("http://localhost:8080/api").unwrap();
        assert_eq!(url.host, "localhost");
        assert_eq!(url.port, 8080);
        assert_eq!(url.path, "/api");
    }

    #[test]
    fn test_parse_ipv4_literal() {
        assert_eq!(HttpClient::parse_ipv4("10.0.2.2"), Some([10, 0, 2, 2]));
        assert_eq!(HttpClient::parse_ipv4("256.1.1.1"), None);
        assert_eq!(HttpClient::parse_ipv4("not-an-ip"), None);
    }

    #[test]
    fn test_parse_response_ok() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<body>hello</body>";
        let resp = HttpClient::parse_response(raw).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.headers.len(), 1);
        assert_eq!(resp.headers[0].0, "Content-Type");
        assert_eq!(resp.body, b"<body>hello</body>");
    }

    #[test]
    fn test_parse_response_404() {
        let raw = b"HTTP/1.1 404 Not Found\r\n\r\n";
        let resp = HttpClient::parse_response(raw).unwrap();
        assert_eq!(resp.status, 404);
        assert!(resp.body.is_empty());
    }
}
