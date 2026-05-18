# 07 — Epsilon Remote: Cross-Machine Teleportation

## Goal

Implement `RemoteVoid` — the ability to teleport epsilon context (point clouds on S²) between machines over a network, completing the "data teleportation" vision where transfer latency is bounded by hardware, not software.

## Current State

`kernel/epsilon/epsilon/crates/epsilon/src/teleport.rs`:
- `RemoteVoid` variant exists in the enum
- `teleport_remote()` explicitly returns `Err(TeleportError::RemoteUnimplemented)`
- `SECURITY.md` documents this as a known limitation

The local teleport pipeline (clutch → inject → assimilate → restore) is complete and tested. The remote version needs to serialize the payload, transmit it, and execute the same pipeline on the remote machine.

## Implementation Steps

1. **Define the wire protocol**
   - Serialize `TeleportPayload` to a compact binary format:
     - Header: magic (`b"EPSN"`), version (u8), payload_size (u64)
     - Governor state: ε (f64), α (f64), β (f64)
     - Surgery permit token (u64 nonce — single-use, verified on receipt)
     - Point cloud: N × 2 × f64 (theta, phi coordinates on S²)
     - Embedding graph adjacency: compressed sparse row format
     - HMAC-SHA256 over entire payload (integrity + authenticity)
   - Use `#[repr(C)]` structs for zero-copy deserialization where possible

2. **Implement serialization (`no_std` compatible)**
   - Manual byte-level serialization (no serde in kernel)
   - `TeleportPayload::to_bytes(&self) -> Vec<u8>`
   - `TeleportPayload::from_bytes(data: &[u8]) -> Result<Self, TeleportError>`
   - Validate all fields on deserialization (no NaN, bounds checks on ε)

3. **Implement transport layer abstraction**
   ```rust
   pub trait TeleportTransport {
       fn send(&mut self, dest: &RemoteAddr, payload: &[u8]) -> Result<(), TransportError>;
       fn recv(&mut self, buf: &mut [u8], timeout_ms: u64) -> Result<usize, TransportError>;
   }
   ```
   - `RemoteAddr`: IP + port, or named endpoint
   - Implementations: `TcpTransport` (for real network), `LoopbackTransport` (for testing)

4. **Implement the remote teleport protocol**
   - **Sender side**:
     1. Acquire surgery permit from local governor
     2. Build payload via bridge (same as local)
     3. Serialize payload
     4. Send via transport
     5. Wait for ACK (with timeout)
     6. Release permit on ACK or timeout
   - **Receiver side**:
     1. Receive payload
     2. Verify HMAC
     3. Deserialize and validate
     4. Acquire surgery permit from local governor
     5. Execute inject → assimilate → restore (same as local pipeline)
     6. Send ACK

5. **Implement permit delegation**
   - The receiver needs its own surgery permit (can't transfer permits — they're one-shot)
   - Sender includes a "delegation token" — a signed request for the receiver's governor to issue a permit
   - Receiver governor validates the token and issues a local permit
   - This prevents unauthorized remote injections

6. **Add handshake and capability negotiation**
   - Before first teleport: exchange version, supported features, max payload size
   - Verify both sides have compatible governor parameters
   - Establish shared HMAC key (via Diffie-Hellman or pre-shared key)

7. **Implement `LoopbackTransport` for testing**
   - In-memory channel between two `TeleportEndpoint` instances
   - No network required — tests run entirely in-process
   - Simulates latency and packet loss for robustness testing

8. **Error handling**
   - `TransportError::Timeout` — receiver didn't ACK in time
   - `TransportError::IntegrityFail` — HMAC mismatch
   - `TransportError::VersionMismatch` — incompatible protocol versions
   - `TransportError::PermitDenied` — receiver governor refused delegation
   - All errors are recoverable (retry after backoff)

## Dependencies

- **06-epsilon-local** (hardened local pipeline)
- **09-network-stack** (TCP transport for real network — but LoopbackTransport enables testing without it)

## Acceptance Criteria

- [ ] `RemoteVoid` no longer returns `RemoteUnimplemented`
- [ ] Loopback test: teleport point cloud from endpoint A to endpoint B, verify identical topology
- [ ] Serialization round-trip: `from_bytes(to_bytes(payload)) == payload`
- [ ] HMAC validation: tampered payload is rejected
- [ ] Timeout handling: receiver offline → sender gets `Timeout` error, permit released
- [ ] Permit delegation: unauthorized token → receiver rejects with `PermitDenied`
- [ ] `cargo test -p epsilon` passes with ≥5 new remote tests
- [ ] No panics on malformed input (fuzz-tested)

## Files to Modify

- `kernel/epsilon/epsilon/crates/epsilon/src/teleport.rs` (implement remote path)

## Files to Create

- `kernel/epsilon/epsilon/crates/epsilon/src/wire.rs` (serialization protocol)
- `kernel/epsilon/epsilon/crates/epsilon/src/transport.rs` (transport trait + impls)
- `kernel/epsilon/epsilon/crates/epsilon/src/permit_delegation.rs` (remote permit protocol)
