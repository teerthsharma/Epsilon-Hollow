# Cryptographic Audit — Seal OS

**Version:** 0.4.6
**Scope:** In-kernel TLS 1.3 client, hardware entropy driver, and package signature verification.  
**Auditor:** Agent 28-32 (Security Hardening & Documentation)  
**Date:** 2026-06-01

---

## 1. TLS 1.3 PSK Path

The TLS implementation lives in `kernel/seal-os/src/drivers/net/tls.rs` and
`tls_socket.rs`. It is a **real, non-stub** client that performs AES-128-GCM
encryption and HKDF key derivation. It is also **severely constrained**:
PSK-only, no X.509, no ECDHE.

### 1.1 Cipher Suite

- **AES-128-GCM** via the `aes-gcm` crate (RustCrypto).
- **Record size limit:** None enforced beyond 16 KiB implied by the TLS spec.
- **AEAD tag:** 16 bytes appended to ciphertext.

### 1.2 Nonce Construction

```rust
fn make_nonce(&self, write: bool) -> [u8; 12] {
    let iv = if write { self.write_iv } else { self.read_iv };
    let seq = if write { self.write_seq } else { self.read_seq };
    let mut nonce = iv;
    for i in 0..8 {
        nonce[11 - i] ^= ((seq >> (8 * i)) & 0xFF) as u8;
    }
    nonce
}
```

- **Mechanism:** 96-bit IV derived at handshake time is XORed with the 64-bit
  record sequence number (big-endian style, least-significant byte at index 11).
- **Correctness:** This matches TLS 1.3 AEAD nonce construction (RFC 8446 §5.3).
  Because the sequence number is monotonic and the IV is unique per connection,
  nonce reuse is prevented **as long as** the same `(key, iv)` pair is not
  instantiated twice.
- **Risk:** No random component per record; if IV derivation is flawed or
  duplicated across sessions, nonce reuse breaks GCM confidentiality and
  integrity.

### 1.3 Key Derivation (HKDF)

The file contains a *simplified* HKDF-like construction. **It is not RFC 5869
compliant.**

```rust
fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(salt);
    h.update(ikm);
    h.finalize().into()
}
```

- **Deviation:** RFC 5869 `HKDF-Extract` is `HMAC-SHA256(salt, ikm)`. The code
  above is raw SHA-256(`salt || ikm`). This changes the security properties
  when `salt` is not uniform (e.g., an empty salt becomes a simple hash of the
  PSK).

```rust
fn hkdf_expand(prk: &[u8], info: &[u8], out_len: usize) -> Vec<u8> {
    // ...
    h.update(&t);
    h.update(info);
    h.update(&[n]);
    h.update(prk);
    // ...
}
```

- **Deviation:** RFC 5869 `HKDF-Expand` is `HMAC-SHA256(prk, T(n-1) || info || 0xN)`.
  The code above uses raw SHA-256 and includes `prk` in the message instead of
  as the HMAC key.

**Impact:** The construction may be vulnerable to length-extension or algebraic
attacks that standard HKDF is not. It must be treated as a **custom KDF** until
replaced with a proper HMAC-based implementation.

### 1.4 Key Lifecycle

- `TlsSession` stores keys in plain `[u8; N]` arrays on the heap.
- **No secure zeroization** on drop. Keys remain in memory after session close.
- **No key update** (post-handshake key update from RFC 8446 is not implemented).

### 1.5 Handshake Overview (PSK-only)

```
ClientHello (with PSK binder)  ──►
                               ◄──  ServerHello
Derive early_secret            ──►  Derive handshake_secret
(no ECDHE shared secret)           Derive traffic keys
```

- Because there is no ECDHE, the handshake secret is derived solely from the
  PSK. This means **no forward secrecy**: compromise of the PSK reveals all
  traffic.
- The ServerHello parser is minimal and does not validate extensions or
  downgrade signals.

---

## 2. `getrandom` Path

Source: `kernel/seal-os/src/drivers/entropy.rs`

### 2.1 Hardware Probing

At boot, `entropy::init()` probes CPUID:

- **RDRAND** (CPUID.1:ECX[30]) — deterministic random bit generator.
- **RDSEED** (CPUID.7:EBX[18]) — non-deterministic entropy source.

### 2.2 Fallback Chain

```
RDSEED (preferred)
   └─► retries up to 10
   └─► if fail → RDRAND
        └─► retries up to 10
        └─► if fail → return false (fail-closed)
```

- `getrandom(buf: &mut [u8]) -> bool` fills the buffer 8 bytes at a time.
- If hardware entropy is unavailable, it returns `false`; callers must handle
  the error (e.g., `tls.rs` returns `Err("entropy unavailable")`).

### 2.3 Limitations

- No software entropy pool (e.g., Fortuna, ChaCha20 CSPRNG). Every call hits
  hardware directly.
- No mixing of multiple entropy sources (TSC, IRQ timing, etc.).
- On virtualised or broken hardware, 10 retries may still fail.
- ASLR uses a separate xorshift PRNG seeded from TSC+CPUID, not `getrandom`.

---

## 3. Ed25519 Package Signature Verification

Source: `kernel/seal-os/src/pkg/format.rs`

### 3.1 Format

`.eph` packages contain:

| Offset | Size | Content |
|---|---|---|
| 0 | 4 | Magic `"EPH\0"` |
| 4 | 4 | Manifest length (big-endian u32) |
| 8 | N | Manifest bytes (UTF-8, key-value lines) |
| 8+N | 64 | Ed25519 signature |
| … | … | File sections |
| end-4 | 4 | Trailer `"END\0"` |

### 3.2 Verification

```rust
pub fn verify_signature(pkg: &EphPackage, public_key: &[u8; 32]) -> Result<(), EphError> {
    use ed25519_dalek::{Signature, VerifyingKey};
    let vk = VerifyingKey::from_bytes(public_key).map_err(|_| EphError::BadSignature)?;
    let sig = Signature::from_bytes(&pkg.signature);
    let mut signed = Vec::new();
    signed.extend_from_slice(pkg.manifest.name.as_bytes());
    signed.extend_from_slice(pkg.manifest.version.as_bytes());
    for f in &pkg.files {
        signed.extend_from_slice(f.path.as_bytes());
        signed.extend_from_slice(&f.hash);
    }
    vk.verify_strict(&signed, &sig)
        .map_err(|_| EphError::BadSignature)
}
```

- **Algorithm:** Ed25519 (`ed25519_dalek` crate, strict verification).
- **Signed payload:** `name || version || (path || sha256_hash)*`.
- **Hash function:** SHA-256 per file (computed at parse time).

### 3.3 Trust Model

- The verifying public key must be provided **out-of-band**.
- There is no PKI, no certificate chain, and no revocation mechanism.
- Key rotation requires re-signing all packages and updating the embedded key.

---

## 4. Known Limitations

| # | Limitation | Risk | Planned Mitigation |
|---|---|---|---|
| L1 | **PSK-only TLS** — no X.509, no ECDHE. | No forward secrecy; no identity authentication beyond the PSK. | Add X25519 ECDHE + X.509 path (future). |
| L2 | **Non-standard HKDF** — raw SHA-256 instead of HMAC-SHA256. | Potential KDF weakness; not interoperable with standard TLS 1.3 implementations. | Replace with `hkdf` crate or HMAC-based implementation. |
| L3 | **No secure zeroization** of key material. | Keys may linger in freed heap memory. | Add `zeroize` crate to `TlsSession` and `EphPackage`. |
| L4 | **No entropy pool** — `getrandom` hits hardware on every call. | Performance penalty; single point of failure if RDRAND/RDSEED fail. | Implement a ChaCha20 CSPRNG seeded from hardware + jitter. |
| L5 | **No certificate validation** in TLS. | MITM possible if PSK is stolen or guessed. | PSK provisioning must occur over a separate secure channel. |
| L6 | **Minimal handshake parser** — no downgrade protection, no extension validation. | Version rollback or extension confusion attacks. | Harden parser; add supported_versions validation. |
| L7 | **Package signing** has no key distribution or revocation. | Compromised key allows arbitrary package installation. | Document key rotation procedure; consider key pinning. |

---

## 5. Audit Checklist for Future Crypto Changes

Before any cryptographic code is modified or replaced, verify:

- [ ] **KDF compliance:** Does the new KDF follow RFC 5869 (HKDF) or an approved NIST standard?
- [ ] **Test vectors:** Are there unit tests against known-good vectors (e.g., RFC 8448 for TLS 1.3)?
- [ ] **Nonce uniqueness:** Can you prove that the same `(key, nonce)` pair is never reused?
- [ ] **Forward secrecy:** If adding ECDHE, is the ephemeral private key discarded after use?
- [ ] **Secure zeroization:** Are all key buffers cleared on drop or error paths?
- [ ] **Side-channel resistance:** Is the implementation constant-time with respect to secret data?
- [ ] **Entropy quality:** Is the random source seeded from ≥128 bits of hardware entropy?
- [ ] **Fail-closed behavior:** If entropy or verification fails, does the system refuse to proceed (not fall back to weak defaults)?
- [ ] **Fuzzing:** Have record parsers and signature verification been fuzzed with `cargo fuzz` or equivalent?
- [ ] **Dependency audit:** Are crypto crates (`aes-gcm`, `sha2`, `ed25519_dalek`) pinned to audited versions with `cargo audit` clean?

---

*This document is a living audit. Update it after every crypto-related PR.*
