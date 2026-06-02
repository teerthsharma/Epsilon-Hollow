# Seal OS Guidebook

The useful, sarcastic, Linux-inspired manual for an operating system that looked at normal kernel design and said: what if the filesystem had coordinates?

This is a book-scale Markdown artifact, not a completed 300-page monolith yet. The structure below is sized for a serious guidebook: roughly 18 parts, 72 chapters, appendices, proof ledgers, runbooks, and contributor paths. The starter chapters are written now; the empty chapters are named so future work has somewhere to land instead of sprawled across the repo like a stack trace with ambition.

## Proof Status At A Glance

| Claim | Current status | Evidence boundary |
| --- | --- | --- |
| QEMU headless boot proof | Proven for the current QEMU proof gate | Requires theorem, AHCI, ManifoldFS mount, desktop/event-loop, benchmark, manifest, and freshness gates |
| ManifoldFS metadata teleport | Proven for the QEMU `fs_mode=mock_block` benchmark path | Same-inode move, source gone, destination present, bounded metadata ops, and `persistence_bytes_per_move=0` |
| TCP exact-flow demux | Proven for the listener-first fixture | `exact_flow=1`, decoy receives zero bytes, listener fallback still accepts a fresh SYN |
| LAAMBA kernel app proof | Proven in QEMU and Oracle VM VirtualBox serial logs | `[LAAMBA] app proof:` shows `window=LAAMBA_Governor`, launcher/start-menu evidence, Aether host window id, Rust native-manifest bridge, `python_runtime=0`, and `result=pass` |
| Host-script-free production runtime | Partially proven | Production roots are quarantined; legacy host wrappers may remain outside runtime |
| Ubuntu parity | Not proven | Target milestone only; same-machine Ubuntu artifacts are required |
| Any-VM compatibility | Not proven | QEMU proof is real; other hypervisors need their own proof bundles |
| Full HFT/ML/GPU superiority | Not proven | Design and partial infrastructure exist; benchmark and GPU proof gates are pending |
| Complete daily-driver OS | Not proven | It is a research OS. Pretending otherwise would be marketing with a keyboard |

## Table Of Contents

### Part I: Read This Before You Hurt Yourself

1. Chapter 1: What Seal OS Is, And What It Is Not
2. Chapter 2: The Proof Ledger
3. Chapter 3: The Linux Inspiration Without The Linux Costume
4. Chapter 4: How To Read A Claim Without Swallowing The Hype

### Part II: Getting A Boot

5. Chapter 5: Host Requirements
6. Chapter 6: Building The Workspace
7. Chapter 7: Building The UEFI Image
8. Chapter 8: QEMU Headless Proof
9. Chapter 9: VirtualBox And Other VM Targets
10. Chapter 10: Real Hardware, Also Known As Consequences

### Part III: The Kernel Map

11. Chapter 11: UEFI Entry
12. Chapter 12: Memory Map And Page Tables
13. Chapter 13: Interrupts, APIC, And The Part Where CPUs Get Opinions
14. Chapter 14: Scheduler Overview
15. Chapter 15: Seal ABI, Not POSIX In A Fake Mustache

### Part IV: Filesystems

16. Chapter 16: VFS
17. Chapter 17: ManifoldFS Concepts
18. Chapter 18: Metadata Teleport
19. Chapter 19: FAT And ext2 Compatibility Work
20. Chapter 20: Persistence Boundaries

### Part V: Network Stack

21. Chapter 21: IPv4
22. Chapter 22: TCP
23. Chapter 23: Exact Flow Before Listener Fallback
24. Chapter 24: UDP, DHCP, DNS
25. Chapter 25: TLS PSK And Why Public HTTPS Is Still Mostly Rude

### Part VI: Aether-Lang And Runtime

26. Chapter 26: Aether-Lang Basics
27. Chapter 27: Parser, Interpreter, Titan VM
28. Chapter 28: Kernel Bridge
29. Chapter 29: Native Manifest Commands
30. Chapter 30: LAAMBA Native Bridge

### Part VII: Theorem Gates

31. Chapter 31: T1 Through T5 Runtime Gates
32. Chapter 32: T6 Through T10 Boot Gates
33. Chapter 33: Lean Artifacts
34. Chapter 34: Rust `no_std` Bridge Checks
35. Chapter 35: How Proofs Fail Usefully

### Part VIII: Graphics And Desktop

36. Chapter 36: Framebuffer
37. Chapter 37: Compositor
38. Chapter 38: Desktop Proof Frame
39. Chapter 39: Applications
40. Chapter 40: GPU Reality Check

### Part IX: Benchmarks

41. Chapter 41: What Counts As Evidence
42. Chapter 42: Boot Markers
43. Chapter 43: Allocator Benchmarks
44. Chapter 44: ManifoldFS Teleport Benchmarks
45. Chapter 45: TCP Demux Benchmarks
46. Chapter 46: Ubuntu Comparison Rules
47. Chapter 47: HFT/ML Workloads

### Part X: Security

48. Chapter 48: Threat Model
49. Chapter 49: ASLR, Seccomp, And KPTI Status
50. Chapter 50: Crypto Status
51. Chapter 51: Known Weaknesses
52. Chapter 52: Reporting Vulnerabilities

### Part XI: Host-Language Quarantine

53. Chapter 53: Why Legacy Host Scripts Still Exist
54. Chapter 54: Runtime Versus Research Baggage
55. Chapter 55: Deleting Legacy Wrappers Without Lying To Yourself

### Part XII: Contributor Guide

56. Chapter 56: Picking A Task
57. Chapter 57: Writing A Proof-Gated Feature
58. Chapter 58: Updating Docs Without Accidentally Founding A Cult
59. Chapter 59: Review Checklist

### Appendices

60. Appendix A: Glossary
61. Appendix B: Status Matrix
62. Appendix C: Command Reference
63. Appendix D: Mermaid Diagram Rules
64. Appendix E: Claim Contract
65. Appendix F: Release Checklist
66. Appendix G: Benchmark Artifact Template
67. Appendix H: VM Proof Template
68. Appendix I: Troubleshooting
69. Appendix J: Reading The Repo
70. Appendix K: Future 300-Page Expansion Plan

---

# Part I: Read This Before You Hurt Yourself

## Chapter 1: What Seal OS Is, And What It Is Not

Seal OS is a bare-metal x86_64 research operating system written around Rust, UEFI, a native Seal ABI, Aether-Lang, theorem gates, and topological data structures. Its central trick is not "Linux but with more stickers." Its central trick is that memory, files, scheduling, and several prediction paths are modeled through geometric/topological primitives.

That is interesting. It is also not a permission slip to claim Ubuntu parity, any-VM reliability, GPU dominance, HFT supremacy, or a completed consumer OS. The current strongest boot claim is narrower and better: the QEMU headless proof gate passes when the expected proof bundle is current. That gate proves a lot, but it does not prove everything. One test passing in QEMU does not baptize every hypervisor on Earth. Hypervisors are petty.

Seal OS borrows the useful spirit of Linux: buildable source, direct command-line workflows, docs that tell you how the machine fits together, and a culture of proof-by-artifact instead of proof-by-vibes. It does not borrow POSIX as a runtime contract. Familiar words may appear, but the ABI is Seal-defined.

### Current Truth

Seal OS can honestly say:

- The QEMU headless proof path passes with hard boot, theorem, desktop, filesystem, networking, and benchmark markers.
- ManifoldFS same-filesystem metadata teleport is proven in the mock block-store benchmark path with `fs_mode=mock_block` and zero file-byte persistence per move.
- TCP demux has a fixture proving exact accepted-flow delivery before listener fallback, while keeping a same-port decoy empty.
- LAAMBA has a native bridge gate after the runtime bridge moved to Rust/native manifest commands.
- Production OS roots are guarded against host-language drift.

Seal OS cannot honestly say:

- It is Ubuntu-compatible.
- It beats Ubuntu broadly.
- It runs on any VM.
- It has production GPU compute.
- It is a secure daily-driver OS.
- It has erased every legacy host script from history. It has not. Some wrappers are legacy baggage. The baggage carousel is documented.

## Chapter 2: The Proof Ledger

The proof ledger is the habit that keeps this project from becoming a brochure with a kernel attached. Every important claim must land in one of four buckets:

| Bucket | Meaning | Example |
| --- | --- | --- |
| Proven | A local gate or artifact verifies it | QEMU headless proof reaches event loop and benchmark sentinels |
| Partial | Code/design exists but coverage is incomplete | GPU PM4 infrastructure with stub shaders |
| Target | Desired milestone with a named gate | Ubuntu HFT/ML comparison |
| Rejected | Not a goal or contradicted by evidence | POSIX compatibility as a runtime promise |

The rule is simple: do not upgrade a claim from target to proven because it sounds cool in a README. This project already has enough unusual math. It does not need unusual accounting.

### Current High-Value Gates

| Gate | What It Protects |
| --- | --- |
| `--check-vm-proof` | QEMU boot proof markers |
| `--check-proof-manifest` | Artifact freshness, hashes, backend, commit, and dirty flag |
| `--check-benchmark-log` | Allocator, TopoRAM, ManifoldFS teleport, scheduler select, and TCP demux markers |
| `--check-aether-runtime` | Aether runtime proof |
| `--check-language-hygiene` | Host-language quarantine |
| `--check-current-benchmark-proof` | Prevents stale benchmark comparisons from wearing a fake mustache |

## Chapter 3: The Linux Inspiration Without The Linux Costume

Linux is the obvious comparison because it is the operating system everyone reaches for when they want a serious baseline. Seal OS should learn from Linux's build discipline, hardware seriousness, command-line ergonomics, and refusal to be embarrassed by plain text logs.

Seal OS should not claim it is Linux-compatible unless the compatibility exists. It should not call Seal ABI syscalls POSIX. It should not call a PSK-only TLS path "the web." It should not claim driver parity with Linux, because Linux has spent decades accumulating hardware support and Seal OS has spent its time teaching files to think in spheres.

The inspiration is cultural and operational:

- Logs should be useful.
- Builds should be repeatable.
- Claims should have artifacts.
- Interfaces should be documented.
- Regressions should fail loudly.

The imitation stops before the costume party:

- No POSIX parity claim.
- No Ubuntu parity claim.
- No "runs everywhere Linux runs" claim.
- No pretending a research kernel is a production distribution.

## Chapter 4: How To Read A Claim Without Swallowing The Hype

When reading Seal OS docs, look for the proof noun.

Bad claim:

```text
Seal OS beats Linux for HFT.
```

Useful claim:

```text
The allocator comparison harness exists, but full HFT/ML Ubuntu comparison
requires same-machine Ubuntu artifacts and current Seal proof manifests.
```

The first sentence sells you fireworks. The second sentence gives you a place to plug in evidence. Prefer the second. Fireworks are how CI catches fire.

---

# Part II: Getting A Boot

## Chapter 5: Host Requirements

The practical path is still host-built. Seal OS is not self-hosting. You need a modern host toolchain, a UEFI-capable VM path, and enough patience to let proof gates finish.

Minimum mental requirements:

- Accept that QEMU proof is not any-VM proof.
- Accept that "runs in a VM" is not "daily driver."
- Accept that logs are evidence, not decoration.

## Chapter 6: Building The Workspace

The workspace has host crates, kernel crates, Aether components, theorem/proof artifacts, app surfaces, and documentation. The kernel uses nightly Rust for bare-metal pieces. Host tooling may use stable Rust.

Do not treat every top-level directory as runtime OS code. The repo contains research history, host tools, plans, and legacy wrappers. This is normal for a research codebase and inconvenient for marketing departments, which is another point in its favor.

## Chapter 7: Building The UEFI Image

The UEFI image path produces a bootable disk image with an EFI System Partition and a ManifoldFS root partition. The image must pass verification before anyone should talk about boot proof.

The important distinction:

- Image verification says the artifact shape is valid.
- VM proof says the artifact actually reached required runtime markers.
- Benchmark proof says specific runtime paths emitted specific bounded markers.

Those are different claims. Mix them together and you get soup. Soup is not a proof format.

## Chapter 8: QEMU Headless Proof

This is the current strongest operational baseline. The QEMU headless proof is expected to show:

- UEFI entry and Seal OS banner.
- Heap, IDT, PIC/APIC, theorem, and scheduler markers.
- AHCI disk identity and block device registration.
- Persistent ManifoldFS root mount.
- Aether runtime proof marker.
- Desktop proof frame, desktop soak, desktop ready, and event loop entry.
- ManifoldFS teleport benchmark with `fs_mode=mock_block`.
- TCP demux benchmark with exact-flow and fallback markers.
- A proof manifest tied to the current checkout.

This proves the QEMU headless path. It does not prove VirtualBox, VMware, Hyper-V, real laptops, or your cousin's haunted NUC.

## Chapter 9: VirtualBox And Other VM Targets

Other VM targets are useful. They are not free. Each target needs its own proof bundle, checker, and freshness rule. A QEMU proof can inspire confidence, but it cannot impersonate a VirtualBox proof.

The milestone format should be:

| Target | Required evidence |
| --- | --- |
| VirtualBox | Dedicated smoke run, serial/screenshot proof, manifest freshness |
| VMware | Dedicated boot proof and device matrix |
| Hyper-V | Dedicated boot proof and synthetic device notes |
| Real hardware | Hardware model, firmware version, logs, failure modes |

Until then, the honest phrase is "target" or "gated milestone." The dishonest phrase is "any VM." The docs should choose honesty because the kernel already has enough chaos.

---

# Part IV: Filesystems

## Chapter 18: Metadata Teleport

ManifoldFS metadata teleport is one of the project names that sounds fake until you read the log marker. The plain version: moving a file inside the same filesystem rewires metadata rather than copying file bytes.

The current proof is bounded:

- It is a same-filesystem metadata move.
- It is exercised by the QEMU benchmark marker.
- The benchmark path reports `fs_mode=mock_block`.
- It requires same inode, source gone, destination present, bounded metadata ops, and `persistence_bytes_per_move=0`.

This is not a claim that every filesystem operation is O(1). It is not a claim that cross-mount moves teleport arbitrary bytes through a geometry portal. It is a specific claim with a specific gate. Miraculously, this is how engineering works.

---

# Part V: Network Stack

## Chapter 23: Exact Flow Before Listener Fallback

The TCP demux proof protects a subtle bug class: a listener-first socket table can accidentally deliver payloads to the listener path before checking an established accepted flow.

The proof marker requires:

- Listener-first fixture ordering.
- Established accepted flow receives the payload.
- Same-port decoy receives zero bytes.
- Listener fallback still accepts a fresh SYN.
- Fixture cleanup succeeds.

That is a useful networking claim. It is not a claim of production internet readiness. TLS is still PSK-only, certificate validation is not production-ready, and public HTTPS compatibility remains a separate gate. The stack can do real work; it cannot yet browse the modern web without the modern web filing a complaint.

---

# Part VI: Aether-Lang And Runtime

## Chapter 30: LAAMBA Native Bridge

LAAMBA is a native-app/control-surface workload with a history of host prototypes. The current status is:

- The runtime bridge moved to Rust/native manifest commands.
- A native bridge gate exists and passes.
- Legacy host wrappers may remain outside the runtime path.
- Those wrappers are baggage, not boot-critical runtime.

The deletion rule is deliberately boring:

1. Read the legacy wrapper.
2. Replace the behavior in Rust or Aether.
3. Add a gate proving the replacement.
4. Delete the wrapper only after the gate passes.

This prevents the classic open-source maneuver where someone deletes code, declares victory, and leaves a crater where behavior used to live.

---

# Part IX: Benchmarks

## Chapter 46: Ubuntu Comparison Rules

Ubuntu is the baseline because it is boring, available, well-supported, and likely to beat a research OS at most things until proven otherwise. That is not defeatism. That is calibration.

Seal OS may claim a win only when:

- Seal OS has a current proof manifest.
- The benchmark log is inside that proof bundle.
- Ubuntu ran on the same machine class with declared CPU, RAM, disk, and VM/hardware settings.
- Raw Ubuntu artifacts are preserved.
- The comparison gate passes for the specific workload.

A win in one row is a win in one row. It is not Ubuntu parity. It is not "Linux is obsolete." It is a measured result. The world will survive the nuance.

## Chapter 47: HFT/ML Workloads

HFT and ML are target domains because topological locality, bounded allocation, prefetching, and data movement matter there. They are not trophies already mounted on the wall.

Current honest position:

- Some infrastructure and benchmark harnesses exist.
- Allocator and I/O microbenchmarks are useful signals.
- Full HFT/ML superiority over Ubuntu is pending.
- GPU superiority is pending on real shader/driver/data-movement proof.

The guidebook should eventually grow detailed workload chapters:

- HFT message ring latency.
- Tick-to-decision simulation.
- ML tensor page movement.
- Sequential training data prefetch.
- GPU/VRAM fast path once real compute proof exists.

Until then, any sentence that says "Seal OS beats Ubuntu for HFT/ML" needs to be escorted out of the docs by security.

---

# Part XI: Host-Language Quarantine

## Chapter 53: Why Legacy Host Scripts Still Exist

Legacy host scripts exist because research code has history. Some files are old scaffolding, examples, wrappers, fixtures, or migration references. Their existence does not mean a host scripting runtime is part of the Seal OS production runtime.

The important boundary:

- Production OS roots must stay Rust/Aether/assembly as gated.
- Legacy host scripts must stay outside runtime proof paths.
- Deletion requires Rust/Aether replacement and tests.

Language percentage theater is not engineering. If a legacy wrapper still documents behavior, keep it quarantined until the replacement is real.

## Chapter 54: Runtime Versus Research Baggage

Runtime code is what the OS proof path depends on. Research baggage is what remains around the project because it explains, prototypes, or tests old ideas.

The docs should never say "host-script-free repo" while legacy wrappers remain. They can say:

```text
Production OS roots are guarded against host-script runtime drift. Remaining
legacy host scaffolding stays outside the runtime path and is scheduled for
replacement only when Rust/Aether gates cover the behavior.
```

That sentence is less glamorous than "pure Rust everywhere," but it has the advantage of being true.

---

# Appendix D: Mermaid Diagram Rules

Mermaid diagrams in this repo should be boringly renderable:

- Quote node labels that contain punctuation, HTML breaks, or symbols.
- Escape generic angle brackets as `&lt;` and `&gt;`.
- Prefer plain pipe labels on edges, such as `-->|label|`, without nested quotes.
- Keep subgraph IDs alphanumeric.
- Do not rely on one renderer accepting syntax another renderer rejects.

Diagrams are documentation, not a renderer compatibility experiment. We already have an OS experiment. One is enough.

---

# Appendix K: Future 300-Page Expansion Plan

This file should grow by replacing chapter stubs with proof-tied material, not by inflating adjectives until the Markdown groans.

Suggested expansion budget:

| Part | Target pages | Notes |
| --- | ---: | --- |
| I: Orientation | 20 | Proof policy, claim language, status reading |
| II: Boot | 35 | Host setup, image build, QEMU, VM targets |
| III: Kernel Map | 45 | UEFI, memory, interrupts, scheduler, ABI |
| IV: Filesystems | 35 | VFS, ManifoldFS, teleport, persistence |
| V: Network | 25 | TCP, UDP, DHCP, DNS, TLS limits |
| VI: Aether/Runtime | 25 | Language, bridge, native manifest commands, LAAMBA |
| VII: Theorems | 30 | T1-T10, Lean, Rust gates |
| VIII: Graphics/Desktop | 20 | Compositor, desktop proof, apps |
| IX: Benchmarks | 30 | Artifact templates and Ubuntu comparison |
| X-XII plus appendices | 35 | Security, quarantine, contribution, glossary |

Total target: about 300 pages.

Current status: scaffold plus starter chapters. Anyone claiming the book is done should be assigned Appendix D until morale improves.
