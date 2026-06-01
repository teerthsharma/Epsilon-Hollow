# Seal OS Threat Model

**Version:** 0.4.5  
**Scope:** Bare-metal x86_64 kernel, drivers, network stack, and FastAPI backend.  
**Assumption:** The machine is physically controlled by the operator unless noted otherwise.

---

## 1. Physical Access Threats

| Threat | Mitigation | Status |
|---|---|---|
| **DMA attacks** (malicious PCIe device accessing physical RAM) | None — no IOMMU (VT-d) support yet. | Out of scope for v0.4.x. |
| **Cold boot** (extracting RAM contents after power loss) | None — memory encryption (TME/SME) not implemented. | Accepted risk; operate in a physically secured environment. |
| **Hardware implants** (modified firmware, bus sniffers) | None. | Out of scope; rely on supply-chain trust. |
| **Evil maid** (bootloader tampering) | UEFI Secure Boot not integrated. | Out of scope; seal the chassis. |

> **Note:** Because Seal OS targets ML/HFT research on dedicated bare-metal nodes,
> physical security is treated as an environmental precondition rather than a
> software-mitigated threat.

---

## 2. Software Threats

### 2.1 Kernel Exploits

Seal OS is written in Rust, which eliminates large classes of memory-safety
bugs (use-after-free, buffer overflows, double-free). However, `unsafe` blocks
exist in:

- Assembly stubs (IDT, GDT, syscall entry, page-table walks)
- Physical memory allocators (`memory/phys.rs`, `memory/slab.rs`)
- Device drivers (MMIO, PCI config space, DMA descriptors)
- Page-table manipulation (`memory/virt.rs`, `memory/pgtable_asm.rs`)

**Current mitigations:**
- SMAP / SMEP prevent most kernel-space execution of user pages and accidental
  user-space dereferences.
- KPTI scaffolding exists but is **not active** on syscall paths yet.
- No stack canary or KASLR for the kernel image itself (higher-half base is
  fixed at `0xffffffff80000000`).

**Gaps:**
- Kernel ASLR is not implemented.
- Stack overflow into guard pages is not enforced on all kernel stacks.
- No Control-Flow Integrity (CFI) or shadow stack.

### 2.2 Information Leaks

| Vector | Mitigation | Status |
|---|---|---|
| Userspace kernel pointer leaks | KPTI (pending) + SMAP copy helpers | Partial |
| Side-channel (cache timing) | No constant-time primitives for secret handling | Not mitigated |
| `/proc` or `/sys` info leaks | No procfs/sysfs equivalent yet | N/A |
| Kernel log leaks (`dmesg`) | Serial output is unrestricted | Audit needed |

### 2.3 Side Channels

- **Spectre-v2 (BTI):** Retpoline thunks exist but compiler auto-redirect is
  disabled. Mitigation is partial.
- **Spectre-v1 (bounds-check bypass):** No explicit `lfence` after array bounds
  checks in kernel code.
- **Meltdown:** Not mitigated. KPTI, once active, will address this.
- **PortSmash / SMT:** No SMT disabling logic.

---

## 3. Network Threats

### 3.1 TLS 1.3 PSK-Only Scope

Seal OS ships a minimal TLS 1.3 client in `kernel/seal-os/src/drivers/net/tls.rs`.

**What is implemented:**
- AES-128-GCM encryption/decryption.
- HKDF-SHA256 key derivation (simplified construction; see `CRYPTO_AUDIT.md`).
- PSK (pre-shared key) mode.

**What is NOT implemented:**
- X.509 certificate parsing or validation.
- ECDHE key exchange (no forward secrecy).
- Certificate revocation (OCSP / CRL).
- ALPN, SNI, or session resumption.

**Implications:**
- Identity is bound to the pre-shared key, not to a certificate chain.
- Compromise of the PSK reveals all past and future traffic encrypted under that
  key (no forward secrecy).
- Man-in-the-middle is prevented **only** if the PSK is distributed over an
  out-of-band channel that the attacker cannot observe or modify.

### 3.2 TCP/IP Stack

The in-kernel TCP/IP stack is research-grade. It does not implement:
- SYN cookies (SYN flood mitigation).
- Ingress/egress firewall or packet filter.
- TCP MD5 or AO (authentication option).

---

## 4. Trust Boundaries

```
┌─────────────────────────────────────────┐
│  Hardware (CPU, RAM, PCIe, NIC)         │
├─────────────────────────────────────────┤
│  Kernel (Ring 0)                        │
│  ├── Memory manager                     │
│  ├── Scheduler                          │
│  ├── Drivers (NIC, GPU, disk)           │
│  ├── TLS stack (PSK-only)               │
│  └── Security modules (KPTI, SMEP…)     │
├─────────────────────────────────────────┤
│  Userspace (Ring 3)                     │
│  ├── Aether-Lang interpreter            │
│  └── Native Seal ABI programs           │
├─────────────────────────────────────────┤
│  FastAPI Backend (host OS, if running)  │
│  └── Local-only by default              │
└─────────────────────────────────────────┘
```

### 4.1 Kernel ↔ Userspace

- **SMAP/SMEP** enforce the boundary at the hardware level.
- **Seccomp** can restrict the syscall surface per-task.
- **KPTI** will remove kernel mappings from user page tables once completed.
- **ASLR** randomises userspace layout.

### 4.2 Device Drivers

All drivers run in Ring 0. There is no user-space driver framework (no UIO,
no DPDK-style offload). A compromised driver has full kernel privilege.

### 4.3 FastAPI Backend ↔ Host

The FastAPI backend is a host-side Python process, not part of the bare-metal
kernel. It is bound to loopback by default and gated by bearer tokens. See
[`SECURITY.md`](../SECURITY.md) for the full backend threat model.

---

## 5. Out-of-Scope Items

The following are explicitly not addressed in Seal OS v0.4.x and should not be
assumed without additional documentation:

1. **Multi-tenant security** — No containers, cgroups, or VM isolation.
2. **Internet exposure** — The kernel TCP stack and TLS client are designed for
   trusted-lab or point-to-point links.
3. **Untrusted operators on the host** — Root / physical access equals total
   compromise by design.
4. **Formal verification of `unsafe` blocks** — Lean 4 proofs cover T1–T10
   theorems, not Rust memory safety.
5. **TPM / secure boot / measured launch** — No hardware root-of-trust integration.
6. **Side-channel resistance** — No constant-time crypto primitives, no cache
   partitioning, no SMT disable.
7. **X.509 / PKI** — TLS is PSK-only.
8. **IOMMU / DMA protection** — No VT-d support.

---

## 6. Risk Register

| ID | Risk | Likelihood | Impact | Owner |
|---|---|---|---|---|
| R1 | Kernel exploit via unsafe MMIO driver | Medium | Critical | Kernel team |
| R2 | PSK compromise (no forward secrecy) | Medium | High | Crypto team |
| R3 | KPTI bypass due to incomplete syscall hook | Low | High | Kernel team |
| R4 | FastAPI backend exposed to non-loopback | Low | Medium | Epsilon team |
| R5 | DMA attack by malicious PCIe device | Low | High | Hardware / future |

---

*Last updated: 2026-06-01*
