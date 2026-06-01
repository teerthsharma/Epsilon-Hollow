# Seal OS (Epsilon-Hollow) Security & Repository Hygiene Audit
**Version:** 0.4.5  
**Date:** 2026-06-01  
**Scope:** Read-only audit of sensitive data exposure, `.gitignore`, unsafe code, cryptography, supply chain, access control, and dependency hygiene.

---

## Executive Summary

The repository demonstrates strong baseline hygiene (comprehensive `.gitignore`, committed `Cargo.lock`, CI-based `cargo audit`, and a detailed `CRYPTO_AUDIT.md`). However, **several critical and high-severity security issues exist**, primarily in memory management (COW/double-free), cryptography (non-standard HKDF, weak password hashing, timing side-channels), and unsafe code patterns (missing SAFETY comments, GPU memory races). These require immediate attention before any production deployment.

---

## 1. Sensitive File Exposure

| Severity | Finding | File / Line | Details |
|----------|---------|-------------|---------|
| **Low** | Hardcoded default password hash | `kernel/seal-os/src/security/passwd.rs:144-155` | `init_passwd()` creates a default user with password `"seal"` and stores `SHA-256("seal")` in `/etc/passwd`. Documented as temporary, but still a shipped credential. |
| **Info** | Hardcoded package trust anchor | `kernel/seal-os/src/pkg/mod.rs:31-34` | `SEAL_PKG_PUBLIC_KEY` is a fixed Ed25519 public key. Not a secret, but key rotation requires kernel recompilation. |
| **Info** | No tracked `.env` or key files | — | `git ls-files` shows no `.env`, `.pem`, `.key`, or `.crt` files. Only `.env.example` is tracked, which is acceptable. |

### Remediation
- **Immediate:** Remove the default password initialization or force a mandatory password-change prompt on first boot.
- **Long-term:** Document a key-rotation procedure for `SEAL_PKG_PUBLIC_KEY` (e.g., allow multiple pinned keys or a signed key-list).

---

## 2. `.gitignore` Completeness

| Severity | Finding | Details |
|----------|---------|---------|
| **Info** | Excellent coverage | The root `.gitignore` covers `target/`, `**/*.rmeta`, `.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db`, swap files, OS files, build artifacts, `kernel/seal-os/target/`, `kernel/seal-os/target-*/`, and agent/runtime artifacts. No tracked IDE/OS/build artifacts were found. |

**Status:** No issues.

---

## 3. Unsafe Code Security

### 3.1 Critical Memory Management Bugs

| Severity | Finding | File / Line | Details |
|----------|---------|-------------|---------|
| **Critical** | COW page table double-free / use-after-free | `kernel/seal-os/src/memory/virt.rs:360` (`clone_page_table_cow`)<br>`kernel/seal-os/src/memory/swap.rs:296-298` (`swap_out_page`) | `clone_page_table_cow` clears `WRITABLE` but shares the **same physical frame** across processes without reference counting. When `swap_out_page` later unmaps and calls `phys::free_frame(old_phys)`, any sibling process still referencing that frame will experience a use-after-free or double-free. |

**Remediation:** Implement reference counting for physical frames, or at minimum mark shared/COW frames as non-freeable in `swap_out_page` until all references are dropped.

### 3.2 High-Severity Unsafe Issues

| Severity | Finding | File / Line | Details |
|----------|---------|-------------|---------|
| **High** | COW fault TLB coherency gap | `kernel/seal-os/src/memory/virt.rs:505` (`handle_cow_fault`) | After copying the page and updating the PTE, only `tlb::shootdown(fault_addr)` is called. In an SMP system, other CPUs may retain stale TLB entries for the old (now-freed) frame, leading to data corruption. |
| **High** | `this_cpu()` returns aliasing `&mut` | `kernel/seal-os/src/cpu/mod.rs:218` (`this_cpu`) | Returns `&'static mut PerCpu` derived from GS base. This mutable reference can alias if the function is invoked re-entrantly or on a CPU where GS is not yet initialized, violating Rust’s aliasing rules and risking data races. |

**Remediation:**
- **TLB:** Ensure `tlb::shootdown` sends IPIs to all CPUs and waits for acknowledgements before returning from the page-fault handler.
- **Per-CPU:** Return `&'static PerCpu` (immutable) and wrap mutable fields in `Mutex` or atomics, or use raw pointers internally with explicit locking.

### 3.3 Medium-Severity Unsafe Issues

| Severity | Finding | File / Line | Details |
|----------|---------|-------------|---------|
| **Medium** | GPU buffer freed while GPU may still access it | `kernel/seal-os/src/drivers/gpu/gpu_mem.rs:42-49` (`GpuBuffer::drop`) | `drop` calls `phys::free_frame` without waiting for a GPU fence. The GPU may still be reading/writing the memory, causing corruption or information leakage. |
| **Medium** | GPU fence busy-wait lacks device memory barriers | `kernel/seal-os/src/drivers/gpu/compute_queue.rs:326-335` (`wait_fence`) | Polls on GPU-mapped memory without ensuring CPU cache coherency with device writes. Also uses `_rdtsc` for timeouts, which is not guaranteed constant across frequency changes. |
| **Medium** | Slices created over active GPU memory | `kernel/seal-os/src/drivers/gpu/topology_accel.rs:412-414`, `446-447`, `468-469`, `486-488` | `core::slice::from_raw_parts(_mut)` is used on GPU physical addresses while the GPU may be writing. This constitutes a data race and is UB in Rust. |
| **Medium** | Slab allocator lacks double-free / size-mismatch protection | `kernel/seal-os/src/memory/slab.rs:80-86` (`free`) | No validation that `ptr` belongs to the requested `size` class or that it was not already freed. A double-free corrupts the free list. |
| **Medium** | Missing SAFETY comments | Hundreds of locations across `kernel/seal-os/src/` | Only ~18 explicit `// SAFETY:` comments exist, while there are hundreds of `unsafe` blocks (drivers, ACPI, APIC, memory, boot). Many are low-level hardware accesses, but without comments soundness is hard to verify. |

**Remediation:**
- GPU: Add explicit fence waits before dropping buffers; use `fence(Ordering::SeqCst)` + device-specific cache flushes; replace slice abstractions over device memory with volatile pointer abstractions.
- Slab: Add size-class validation and debug-build poisoning/cookies.
- Documentation: Add `// SAFETY:` rationale to every `unsafe` block.

---

## 4. Cryptography Audit

### 4.1 Critical / High Crypto Weaknesses

| Severity | Finding | File / Line | Details |
|----------|---------|-------------|---------|
| **Critical** | Single-iteration SHA-256 password hashing | `kernel/seal-os/src/security/shadow.rs:77-83` (`compute_hash`) | Uses `SHA-256(password \|\| salt \|\| seed)` with **no iteration count**, **no memory hardness**, and **no CPU hardness**. Extremely vulnerable to brute-force, dictionary, and GPU/ASIC attacks. |
| **High** | Timing side-channel in password verification | `kernel/seal-os/src/security/shadow.rs:163`, `169` | Uses `==` on `[u8; 32]` to compare password hashes. Rust’s array equality is not constant-time; an attacker can measure byte-by-byte differences to forge or recover passwords. |
| **High** | Non-standard, length-extension-vulnerable HKDF | `kernel/seal-os/src/drivers/net/tls.rs:247-270` (`hkdf_extract` / `hkdf_expand`) | `hkdf_extract` is `SHA-256(salt \|\| ikm)` instead of `HMAC-SHA256(salt, ikm)`. `hkdf_expand` puts `prk` in the message instead of as the HMAC key. Vulnerable to length-extension and lacks the security proofs of RFC 5869. |
| **High** | Hardcoded topological seed adds zero entropy | `kernel/seal-os/src/security/shadow.rs:21` | `TOPO_SEED` is hardcoded to `b"EpsilonHollowT5!"`. Because it is public and static, it provides no additional security beyond the salt. |

### 4.2 Medium / Low Crypto Weaknesses

| Severity | Finding | File / Line | Details |
|----------|---------|-------------|---------|
| **Medium** | No secure zeroization of TLS keys | `kernel/seal-os/src/drivers/net/tls.rs` (`TlsSession`) | Keys (`psk`, `write_key`, `read_key`, `write_iv`, `read_iv`) are stored in plain `[u8; N]` arrays. On session drop or error, they remain in memory. |
| **Medium** | No software CSPRNG pool | `kernel/seal-os/src/drivers/entropy.rs` | `getrandom` hits `RDRAND`/`RDSEED` hardware directly on every call. No Fortuna/ChaCha20 CSPRNG to mix entropy or survive hardware failures. |
| **Medium** | Misleading "rounds" field in shadow file | `kernel/seal-os/src/security/shadow.rs:183` | `make_shadow_line` embeds `$topo$5000$...` but the hash function does **not** iterate 5000 times. This is misleading for anyone auditing the shadow file format. |
| **Low** | Legacy unsalted password hash migration | `kernel/seal-os/src/security/shadow.rs:161-166` | Legacy entries compare `SHA-256(password)` directly, without salt. Should be rejected or force-reset on first boot. |
| **Low** | Deterministic 8-byte salt | `kernel/seal-os/src/security/shadow.rs:64-71` (`compute_salt`) | Salt is only 8 bytes and derived deterministically from `username + uid`. Low entropy and predictable per-user across installations. |
| **Low** | Minimal TLS handshake parser | `kernel/seal-os/src/drivers/net/tls.rs:117-151` | `handle_server_hello` does not validate extensions, supported_versions, or downgrade signals. Documented in `CRYPTO_AUDIT.md` but still a risk. |

**Remediation:**
- **Password hashing:** Replace `compute_hash` with **Argon2id** (preferred) or **PBKDF2-HMAC-SHA256** with ≥600,000 iterations (OWASP 2023 minimum). Increase salt to 16+ random bytes from `getrandom`. Replace `==` comparison with `subtle::ConstantTimeEq`.
- **HKDF:** Replace custom implementation with the `hkdf` crate or implement RFC 5869 using the `hmac` crate.
- **Zeroization:** Add `zeroize` crate and derive `ZeroizeOnDrop` for `TlsSession` and any other key-bearing structs.
- **Entropy:** Implement a ChaCha20 CSPRNG seeded from hardware entropy at boot; fall back to `getrandom` failure as fatal.
- **TOPO_SEED:** Generate a unique 16-byte seed at first boot from hardware entropy and store it in a secure root file.

---

## 5. Supply Chain Security

| Severity | Finding | File / Line | Details |
|----------|---------|-------------|---------|
| **Info** | `Cargo.lock` committed | Root + sub-crates | Good practice for reproducible binary builds. |
| **Info** | `cargo audit` in CI | `.github/workflows/ci.yml:131-138` | Runs on every push/PR. |
| **Low** | `deny.toml` disables advisory DB + ignores multiple RUSTSEC IDs | `deny.toml:12-28` | Advisory scanning is deferred to `cargo audit` because of CVSS 4.0 parsing issues. Several advisories are ignored with documented rationale (transitive optional deps). This is acceptable but must be revisited quarterly. |
| **Low** | Unused dependencies increase attack surface | `kernel/seal-os/Cargo.toml:20`, `27` | `volatile` and `x25519-dalek` are declared but never used in the kernel source code. |
| **Info** | No git dependencies | — | All crates are from crates.io. No un-pinned git deps found. |

**Remediation:**
- Remove `volatile` and `x25519-dalek` from `kernel/seal-os/Cargo.toml`.
- Schedule a quarterly review of ignored RUSTSEC advisories in `deny.toml` and `.cargo/audit.toml`.

---

## 6. Access Control

| Severity | Finding | File / Line | Details |
|----------|---------|-------------|---------|
| **Medium** | CODEOWNERS has single owner, no domain separation | `.github/CODEOWNERS` | Every file is owned by `@teerthsharma`. No separation of duties for crypto, memory management, or drivers. |
| **Info** | No branch protection rules documented | — | No `settings.yml`, `CONTRIBUTING.md`, or workflow files mention required reviews, signed commits, or merge requirements. |

**Remediation:**
- Add domain-specific CODEOWNERS entries (e.g., `@teerthsharma` + `@security-reviewer` for `src/security/` and `src/drivers/net/`).
- Document branch protection requirements (required PR reviews, CI pass before merge, signed commits).

---

## 7. Dependency Hygiene

| Severity | Finding | File / Line | Details |
|----------|---------|-------------|---------|
| **Low** | Unused crate `volatile` | `kernel/seal-os/Cargo.toml:20` | Not referenced in any `use volatile::` or `extern crate volatile` statement. All volatile accesses use `core::ptr::read_volatile` / `write_volatile`. |
| **Low** | Unused crate `x25519-dalek` | `kernel/seal-os/Cargo.toml:27` | Only referenced in a comment (`// x25519, len 0`) in `tls.rs`. The actual TLS code is PSK-only and does not perform ECDHE. |

**Remediation:** Remove both unused dependencies from `kernel/seal-os/Cargo.toml` and run `cargo check` to verify.

---

## Additional Observations

### Package Manager Signature Bypass
`kernel/seal-os/src/pkg/mod.rs:55-64` (`install_bytes`) accepts `public_key: Option<&[u8; 32]>`. When `None`, signature verification is **skipped entirely**. While the `install` wrapper always passes `Some(&SEAL_PKG_PUBLIC_KEY)`, any internal or future caller could pass `None` and install an arbitrary, unsigned package. **Remediation:** Make signature verification mandatory (non-optional) or provide an explicitly unsafe/unchecked variant.

### `docs/CRYPTO_AUDIT.md` Quality
The document is **accurate and well-maintained**. It correctly identifies the non-standard HKDF, lack of forward secrecy, missing zeroization, and entropy limitations. It should be updated as each issue in this report is resolved.

---

## Risk Matrix Summary

| Severity | Count | Top Files |
|----------|-------|-----------|
| **Critical** | 2 | `memory/virt.rs`, `security/shadow.rs` |
| **High** | 4 | `memory/virt.rs`, `cpu/mod.rs`, `drivers/net/tls.rs`, `security/shadow.rs` |
| **Medium** | 9 | `drivers/gpu/gpu_mem.rs`, `drivers/gpu/compute_queue.rs`, `drivers/gpu/topology_accel.rs`, `memory/slab.rs`, `pkg/mod.rs`, `security/passwd.rs`, `drivers/entropy.rs`, `security/shadow.rs` |
| **Low** | 7 | `Cargo.toml`, `deny.toml`, `.github/CODEOWNERS`, `security/shadow.rs`, `security/passwd.rs` |
| **Info** | 5 | `.gitignore`, `CRYPTO_AUDIT.md`, CI config, `pkg/mod.rs` |

---

*End of Audit Report*
