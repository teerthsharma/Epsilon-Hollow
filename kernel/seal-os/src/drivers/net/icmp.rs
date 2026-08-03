// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ICMP ping driver wrapper -- Echo Request/Reply wired to net::transmit via net::icmp.

use core::sync::atomic::Ordering;

/// Send an ICMP Echo Request (ping) to `dst` with sequence number `seq`.
pub fn ping(dst: [u8; 4], seq: u16) {
    crate::net::icmp::send_echo_request(dst, seq);
}

/// Returns true if an echo reply has been received since last checked.
pub fn pong() -> bool {
    crate::net::icmp::ECHO_REPLY_RECEIVED.swap(false, Ordering::SeqCst)
}

/// Returns true if an echo reply has arrived and `pong` has not consumed it yet.
///
/// NOT sticky: this reads the same flag that `pong` clears with `swap(false)`,
/// so a single `pong()` call makes this return false from then on.
pub fn received() -> bool {
    crate::net::icmp::ECHO_REPLY_RECEIVED.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ping_starts_false() {
        assert!(!received());
    }
}
