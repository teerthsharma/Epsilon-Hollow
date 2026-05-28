// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! DNS resolver wrapper -- delegates to net::dns.

pub struct DnsResolver {
    server: [u8; 4],
}

impl DnsResolver {
    pub fn new(server: [u8; 4]) -> Self {
        crate::net::dns::set_server(server);
        Self { server }
    }

    pub fn resolve(&self, hostname: &str) -> Option<[u8; 4]> {
        crate::net::dns::query(hostname)
    }

    pub fn server(&self) -> [u8; 4] {
        self.server
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dns_resolver_new() {
        let resolver = DnsResolver::new([8, 8, 8, 8]);
        assert_eq!(resolver.server(), [8, 8, 8, 8]);
    }
}
