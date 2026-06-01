# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately. Do **not** open a public issue
for security-sensitive reports.

- Preferred: GitHub Security Advisories ("Report a vulnerability" on the repo's
  Security tab).
- Email fallback: `teerths57@gmail.com`.

Include reproduction steps, affected version/commit, and impact. Expect an
acknowledgement within a reasonable window; coordinated disclosure is
appreciated.

---

## Security Architecture

Seal OS implements a layered defense model. The table below maps each hardening
feature to its source file and current maturity.

| Defense | Source File(s) | Status | Notes |
|---|---|---|---|
| **KPTI** | `kernel/seal-os/src/security/kpti.rs`, `kernel/seal-os/src/memory/pgtable_asm.rs` | Scaffolding | Kernel CR3 captured, trampoline shadow PT allocated, real `swap_cr3` assembly present. Not yet wired into per-CPU syscall entry/exit. |
| **ASLR** | `kernel/seal-os/src/security/aslr.rs`, `kernel/seal-os/src/memory/virt.rs` (mmap integration) | Implemented | xorshift64 PRNG seeded from RDTSC+CPUID. Randomises mmap base, stack top, and heap base. Not a CSPRNG; ELF loader integration pending. |
| **Seccomp** | `kernel/seal-os/src/security/seccomp.rs` | Implemented | Simplified classic BPF evaluator (LD_W_ABS, JMP_JEQ, RET). Per-task filters. No default policy or user-space loader yet. |
| **SMAP / SMEP** | `kernel/seal-os/src/security/smap_smep.rs` | Implemented (hardware-dependent) | CR4 bits set if CPUID confirms support. `copy_from_user` / `copy_to_user` use `stac`/`clac` with range validation. Mapped-page permission checks and #PF recovery are pending. |
| **Retpoline** | `kernel/seal-os/src/security/retpoline.rs`, `kernel/seal-os/.cargo/config.toml` | Partial | Naked thunks for all GPRs exist. Compiler `-C target-feature=+retpoline` is **commented out** in `.cargo/config.toml`; automatic redirection is not active. |
| **Audit** | `kernel/seal-os/src/security/audit.rs` | Implemented | JSON-like event buffering, lazy VFS flush to `/var/log/audit.log`. No rotation, tamper protection, or remote forwarding. |

### KPTI — detailed status

`security/kpti.rs` is **not a stub**, but it is incomplete for a production
hard gate:

- ✅ `init()` captures the active kernel CR3 at boot.
- ✅ `switch_to_kernel_pt()` and `switch_to_user_pt()` call real x86_64 assembly
  (`pgtable_asm::swap_cr3`) to move CR3.
- ✅ `init_trampoline_pt()` allocates a single global PML4, copies kernel
  upper-half entries (256..512), zeros the lower half, and installs it as the
  user CR3.
- ❌ Shadow page tables are **global**, not per-CPU or per-process.
- ❌ The syscall entry/exit path does **not yet** invoke the CR3 swap.
- ❌ No boot selftest proves that a synthetic syscall actually runs with the
  shadow page table active.

**What remains for the hard gate:**
1. Allocate a per-CPU (or per-process) shadow PML4 so that user tasks cannot
   see each other's mappings.
2. Hook `switch_to_kernel_pt()` into the syscall entry stub and IRQ handler
   before dispatching to kernel code.
3. Hook `switch_to_user_pt()` into `sysret` / `iret` exit paths.
4. Add a boot-time integration test that verifies `has_kpti()` is true and that
   a user-mode read of a kernel symbol triggers a page fault.

### Retpoline — detailed status

`security/retpoline.rs` provides hand-written naked thunks for every general-
purpose register (`rax`–`r15`) plus an `lfence` barrier. These thunks are real
and can be called manually or linked into assembly stubs. However, the Rust
compiler is **not** currently configured to emit indirect branches through them:
`kernel/seal-os/.cargo/config.toml` contains `--cfg retpoline` but the LLVM
`-C target-feature=+retpoline` flag is commented out. Until that flag is enabled
and verified on the target nightly, compiler-generated indirect calls/jumps may
still use naive prediction.

---

## Verified vs Pending

| Feature | Verified | Pending (Hard Gate) |
|---|---|---|
| KPTI | CR3 swap assembly + trampoline allocation | Per-CPU shadow PT, syscall hook, boot selftest |
| ASLR | PRNG + randomisation functions | CSPRNG seed source, ELF loader integration |
| Seccomp | BPF evaluator + unit tests | Default deny policy, user-space loader |
| SMAP/SMEP | CR4 enable + `stac`/`clac` copy helpers | Mapped-page validation, #PF recovery stub |
| Retpoline | Naked thunks for all GPRs | Compiler flag enabled, full indirect-branch coverage |
| Audit | Event formatting + VFS flush | Log rotation, tamper-evidence, remote forwarding |

---

## Hardening Checklist for Maintainers

Before claiming a feature production-ready, the following must be true:

- [ ] **KPTI**
  - [ ] Boot selftest proves CR3 swap during a synthetic syscall.
  - [ ] Shadow page tables are per-CPU or per-process (not global).
  - [ ] Syscall entry and interrupt dispatch paths call `switch_to_kernel_pt()`.
  - [ ] `sysret` / `iret` exit paths call `switch_to_user_pt()`.
- [ ] **ASLR**
  - [ ] PRNG is seeded from a CSPRNG (not xorshift + TSC).
  - [ ] ELF loader and `mmap` paths consume the randomised bases.
  - [ ] 16-bit or higher entropy is measurable in a histogram test.
- [ ] **Seccomp**
  - [ ] A default-deny policy exists for the init task.
  - [ ] User-space can load filters via a stable ABI.
  - [ ] Filter JIT or bounds-checked interpreter fuzzed.
- [ ] **SMAP / SMEP**
  - [ ] `copy_from_user` / `copy_to_user` validate that pages are mapped and
        have correct permissions (not just address-range checks).
  - [ ] #PF handler invokes `handle_user_access_fault()` on SMAP violations.
  - [ ] Boot log confirms SMEP and SMAP bits are set in CR4.
- [ ] **Retpoline**
  - [ ] `.cargo/config.toml` enables `-C target-feature=+retpoline` (or equivalent
        LLVM arg) and the kernel still boots.
  - [ ] All indirect branches in kernel assembly stubs are routed through thunks.
  - [ ] Spectre-v1 mitigations (lfence after bounds checks) audited.
- [ ] **Audit**
  - [ ] Log file size is capped and rotated.
  - [ ] Tamper-evidence (append-only hash chain or similar) is implemented.
  - [ ] Remote forwarding can be configured for SIEM integration.
- [ ] **Cryptography**
  - [ ] TLS PSK keys are provisioned out-of-band via a secure channel.
  - [ ] Package signing keys are rotated on a documented schedule.
  - [ ] `getrandom` failure triggers a kernel panic or safe fallback (no weak
        randomness is silently used).

---

## Application Security — FastAPI Backend

The Epsilon-Hollow FastAPI backend
(`kernel/epsilon/epsilon-ide/pentesting/backend/main.py`) is designed for
**local-only** operation by default:

- Default bind host is `127.0.0.1`. Non-loopback binds require
  `EPSILON_API_TOKEN`.
- Mutating endpoints require a bearer token (constant-time compared) unless the
  process is in dev mode **and** bound to loopback.
- Shell-level endpoints (`/api/v1/claw/execute`, `/ws/terminal`) are
  triple-gated: `EPSILON_DEV_MODE=1` **and** `EPSILON_DEV_TERMINAL=1` **and**
  the same bearer-token check.
- File APIs are jailed to `workspace_root` (path traversal is rejected).
- CORS defaults to the documented frontend origin only.

### Environment variables

| Variable | Default | Effect | Security implication |
| --- | --- | --- | --- |
| `EPSILON_API_TOKEN` | unset | Bearer token required on mutating endpoints. | **Required** unless `EPSILON_DEV_MODE=1` AND bind is loopback. |
| `EPSILON_DEV_MODE` | `0` | Enables dev convenience features. | Permits empty token only in combination with loopback bind. |
| `EPSILON_DEV_TERMINAL` | `0` | Mounts `/api/v1/claw/execute` and `/ws/terminal`. | Required on top of `EPSILON_DEV_MODE=1` to expose shell access. |
| `EPSILON_BIND_HOST` | `127.0.0.1` | uvicorn bind host. | Any non-loopback value forces `EPSILON_API_TOKEN` to be set. |
| `EPSILON_BIND_PORT` | `8742` | uvicorn bind port. | Informational. |
| `EPSILON_CORS_ORIGINS` | `http://127.0.0.1:8742,http://localhost:8742` | Comma-separated allowed origins. | Avoid `*`; that mode is no longer the default. |
| `EPSILON_WORKSPACE_ROOT` | `~` | Initial workspace jail root. | Determines what the file APIs can see. |

---

## Known Limitations

- `TeleportTarget::RemoteVoid` (see
  `kernel/epsilon/epsilon/crates/epsilon/src/teleport.rs`) is an unauthenticated
  stub returning `RemoteUnimplemented`. It is not safe for use over untrusted
  networks and must not be exposed beyond a trusted host.
- The dev terminal, even when triple-gated, runs commands with the host user's
  privileges. Treat it like SSH — only enable it on a machine and network you
  control.
- Native C++ extensions are optional; Python fallbacks have not been audited
  for parity.
- The TLS 1.3 stack is **PSK-only** (no X.509, no ECDHE). See
  [`docs/CRYPTO_AUDIT.md`](docs/CRYPTO_AUDIT.md) for the full cryptographic
  audit and limitations.

---

## References

- [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) — Seal OS threat model
- [`docs/CRYPTO_AUDIT.md`](docs/CRYPTO_AUDIT.md) — Cryptographic audit
- [`.github/SECURITY_ADVISORIES.md`](.github/SECURITY_ADVISORIES.md) — Advisory template and disclosure policy
