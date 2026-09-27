<!-- Seal OS v0.4.7.5 README. The pre-2026-09-27 development record is docs/RECORD.md. -->

# Seal OS: a `no_std` Rust x86_64 kernel whose topological answers come with a certificate or a refusal

**Seal OS is a monolithic UEFI kernel that runs two machine-learning workload services in kernel space, a loss-trajectory fit detector and a prefix-tree KV cache, and returns each topological quantity it computes either with a certificate that floating-point rounding cannot change it or with a refusal that names the input responsible.**

`https://github.com/teerthsharma/Epsilon-Hollow` · Project page with every certify-or-refuse result drawn as a live figure: <https://teerthsharma.github.io/Epsilon-Hollow/> · Release: Seal OS v0.4.7.5

[![DOI 10.5281/zenodo.20264206](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20264206-1682D4?style=flat-square)](https://doi.org/10.5281/zenodo.20264206)
[![License: MIT](https://img.shields.io/github/license/teerthsharma/epsilon-hollow?style=flat-square&color=00aaff)](LICENSE)

Invented by Teerth Sharma (https://teerthsharma.vercel.app/), teerths57@gmail.com

Every number below carries its provenance: a command that reproduces it, a CI run id with its commit, or the words "stated, not re-measured". Numbers attributed to 2026-09-27 were measured at commit `9ebbe2e` by a verification agent; CI numbers come from run [36165748105](https://github.com/teerthsharma/Epsilon-Hollow/actions/runs/36165748105) on the same commit. The development history that earlier versions of this file carried (verifier findings, red-gate investigations, per-round logs) is preserved verbatim in [docs/RECORD.md](docs/RECORD.md).

---

## Abstract

Seal OS is a research kernel for x86_64, written in `no_std` Rust, booted only through UEFI, with its own 69-call system-call ABI. It tests one proposition: a kernel that serves machine-learning workloads can compute their structure itself, the shape of a training run's validation curve and the prefix structure of an inference server's KV cache, and it should refuse to return a discrete answer whose value was chosen by floating-point rounding rather than by the data. Component counts are certified by an empty band around the scale in the minimum-spanning-tree merge heights, attention top-k by Higham error intervals, and nearest-centroid lookup on S² by a distance-to-boundary test; before that test, the grid-hash index answered 1,215 of 5,000 seeded queries wrongly while reporting a hit rate of 1.0000 (commit `cdb4a4e`, as stated on the project page). Under QEMU in CI run 36165748105, the `foliation` KV policy matches the Belady hit rate on its synthetic trace (952 bp against 952 bp) where LRU scores 0 bp, and beats a random policy on 32 of 32 seeds; `stratum` classifies 7 of 7 synthetic runs, including a negative control that a train/validation gap threshold misclassifies. What does not work is stated once, in Limits.

## Background

### Why workload structure belongs in the kernel under test

**The kernel sees every step and owns the memory.** On a conventional kernel a training job is a process with a large heap, and its validation curve exists only in user space. In Seal OS a trainer pushes two scalars per step, `(train_loss, val_loss)`, through `SYS_FIT_OBSERVE` (121); the kernel keeps a fixed-size stream per run (4,792 bytes, measured at boot) and returns a regime. The kernel observes nothing else about the model: not weights, activations or gradients (`kernel/seal-os/src/ml_engine/stratum.rs`, module documentation).

**KV-cache residency is a physical-memory decision.** vLLM's PagedAttention manages KV blocks in user space and is deployed in production against real models; `foliation` does not compete with it on those terms. What it tests is placement and constraint: the block table of a sequence is its path down a prefix tree held by the kernel, each block is backed by a physical frame from the kernel allocator, and eviction may remove only a free face of that tree.

**Near a decision boundary, rounding picks the answer.** A component count at threshold `s` changes at each merge height; when a height lies within rounding error of `s`, the integer returned is decided by the arithmetic. The same holds for a top-k boundary between two nearly equal attention scores and for the nearest centroid of a query near a Voronoi edge. The rule applied throughout is: compute an enclosure; if it clears the decision boundary, return the answer; if it touches the boundary, return the witness and let the caller widen, scan or refuse.

### Prior Art

| System | Language | Kernel type | Linux ABI | Verification | Relative to Seal OS |
|---|---|---|---|---|---|
| seL4 | C | Microkernel, L4 family | No; Linux runs as a guest in a user-level VMM | Machine-checked functional correctness of the C implementation in Isabelle/HOL [1] | seL4 proves its kernel correct. Seal OS proves algebraic lemmas in Lean 4 and evaluates conditions at boot; it has no refinement proof of any kernel code. |
| Redox | Rust | Microkernel | No; POSIX-like interface through its own C library, relibc [2] | No published kernel proof | Both are Rust. Redox runs drivers and filesystems in user space; Seal OS compiles drivers, filesystems, the network stack and its applications into the kernel image. |
| Theseus | Rust | Single address space, single privilege level, runtime-composable cells [3] | No | Relies on Rust's type and ownership checks ("intralingual" design); no machine-checked kernel proof | The earlier precedent for a non-POSIX Rust OS structure. Seal OS instead keeps hardware privilege separation in its design: a ring-3 entry path and KPTI page tables. |
| Asterinas | Rust | Framekernel: one address space, `unsafe` confined to a small framework, OSTD [4] | Yes; targets the Linux system-call ABI | Soundness argument rests on the small `unsafe` framework | Asterinas runs Linux binaries. Seal OS has its own 69-call ABI, and 612 of its 624 `unsafe` blocks carry no safety comment (run 36165748105). |
| Fuchsia (Zircon) | C++ | Microkernel with capability handles [5] | Not native; Starnix runs Linux binaries as a component | No published kernel proof | Production-deployed; Seal OS has no deployment. |
| Linux | C, with Rust for some drivers since 6.1 | Monolithic with loadable modules | Native; 386 x86_64 system calls [6] | No whole-kernel proof; KUnit, kselftest and syzkaller fuzzing | Same monolithic shape at a far smaller scale: 62 driver files, exercised in one QEMU configuration. The two ML services, the S² placement of frames, files and tasks, and the certify-or-refuse rule have no Linux in-kernel counterpart. |

[1] Klein et al., "seL4: Formal Verification of an OS Kernel", SOSP 2009; <https://sel4.systems>. [2] <https://www.redox-os.org>. [3] Boos et al., "Theseus: an Experiment in Operating System Structure and State Management", OSDI 2020; <https://github.com/theseus-os/Theseus>. [4] "Asterinas: A Linux ABI-Compatible, Rust-Based Framekernel OS with a Small and Sound TCB", USENIX ATC 2025; <https://github.com/asterinas/asterinas>. [5] <https://fuchsia.dev>. [6] `arch/x86/entry/syscalls/syscall_64.tbl`, Linux master 7.3.0-rc4, fetched 2026-09-27 by the verification agent (stated, not re-fetched). Rows [1] to [5] are written from published documentation, not from runs on this project's hardware.

## Theoretical Foundation

### 1) Certified β₀

Let $h_1 \le \dots \le h_{n-1}$ be the edge weights of the Euclidean minimum spanning tree of a point set $X = \{x_1, \dots, x_n\}$; these are exactly the single-linkage merge heights. For a threshold $t$,

$$
\beta_0(X, t) = n - \#\{\,k : h_k < t\,\}. \tag{1}
$$

Given a scale $s$ and a band ratio $r \ge 1$, the count is **certified** when no merge height lies in the band:

$$
h_k \notin \left[\, s/\sqrt{r},\; s\sqrt{r} \,\right] \quad \text{for all } k. \tag{2}
$$

Every threshold in that band, under either `<` or `<=`, then gives the same integer. Otherwise the result is a refusal `Refused { i, j, height }` naming the in-band tree edge whose height is nearest $s$. Heights are computed as $m\sqrt{\sum_d (\delta_d/m)^2}$ with $m = \max_d |\delta_d|$, so separations near $10^{-170}$ or $10^{170}$ neither underflow nor overflow. Implemented by `certified_beta0` in `kernel/epsilon/epsilon/crates/aether-core/src/certified_betti.rs` (all-pairs Prim, $O(n^2)$); the rule is ported from planimeter's gap rule.

### 2) Certified attention top-k

Each score $\hat{s}_j = \mathrm{fl}(q \cdot k_j)$ of head dimension $n$ carries Higham's a-priori bound (Accuracy and Stability of Numerical Algorithms, 2nd ed., Theorem 3.1):

$$
\left|\hat{s}_j - q \cdot k_j\right| \le \gamma_n \sum_{d} \left|q_d\, k_{j,d}\right|, \qquad \gamma_n = \frac{n u}{1 - n u}, \quad u = 2^{-53}. \tag{3}
$$

The computed radius $r_j$ inflates (3) by $(1 + 2\gamma_n + 4u)$ and adds $n$ times the smallest subnormal, so it bounds rather than estimates. The top set $T$ of size $k$ is certified when

$$
\min_{i \in T}\left(\hat{s}_i - r_i\right) \;>\; \max_{j \notin T}\left(\hat{s}_j + r_j\right). \tag{4}
$$

Otherwise the row widens to every key whose upper end reaches the lowest selected lower end, at no extra dot products; a NaN or infinite score refuses and the row falls back to dense. Implemented by `certified_top_k` and `enclosed_dot` in `aether-core/src/attention.rs`; the separation test is ported from separatrix's error intervals.

### 3) Fold score of a loss trajectory (`stratum`)

From the validation losses $v_t$ the kernel keeps the last 64 delay points $p_t = (v_t, v_{t-1}, v_{t-2}) \in \mathbb{R}^3$ and resamples them to uniform arc length. Let $\varepsilon^\ast$ be the largest edge of its minimum spanning tree and $m$ the number of resampled points. The loop score is

$$
\ell =
\begin{cases}
0 & \text{if the window is monotone,} \\
\min\!\left(1,\; c(\kappa\,\varepsilon^\ast)/m\right) & \text{otherwise,}
\end{cases}
\qquad c(\varepsilon) = E_\varepsilon - V + \beta_0, \tag{5}
$$

where $E_\varepsilon$ counts Vietoris–Rips 1-skeleton edges at scale $\varepsilon$ except two-step chords already filled by a triangle, so $c$ upper-bounds Rips $\beta_1$. The one free constant is bounded by a derivation, not tuned:

$$
\sqrt{8/3} \approx 1.633 \;<\; \kappa = 1.68 \;<\; \sqrt{3} \approx 1.732. \tag{6}
$$

The floor comes from a symmetric V, whose descending and ascending arms first meet at $\sqrt{8}\,s = \sqrt{8/3}\,\varepsilon^\ast$; the ceiling comes from a monotone stretch, where the first chord able to close a cycle spans three steps and is at least $\sqrt{3}\,\varepsilon^\ast$ long. Underfit is read from the participation ratio of the training-loss autocovariances $c_0, c_1, c_2$:

$$
\mathrm{PR} = \frac{3}{3 + 4(c_1/c_0)^2 + 2(c_2/c_0)^2} \in \left[\tfrac{1}{3}, 1\right], \tag{7}
$$

which is returned as NaN, a refusal, when the variation is below its own rounding bound. `classify` then decides in a fixed order: non-finite input or unmeasurable signal gives Collapsing; fewer samples than the warm-up gives WellFit; training-loss drift above threshold, or shatter with any rise, gives Collapsing; $\ell$ above threshold together with residual drift gives Overfit; a finite PR at or below the trend threshold gives Underfit; anything else is WellFit (`aether-core/src/trajectory_shape.rs`, `fold_score`, `cycle_rank`, `classify`).

### 4) Eviction as a free-face collapse (`foliation`)

A leaf of the prefix tree may be evicted only if it is resident, its reference count is zero, and it has no resident children. Removing such a leaf is an elementary collapse, so the resident set stays a connected rooted subtree under every policy. Policies differ only in which candidate they pick: the foliation policy orders the frontier lexicographically by (number of distinct sequences that ever entered the leaf, negative depth, last use); LRU by last use; the Belady oracle by next use in a supplied future trace. Two sequences share a block only when their tokens are equal; the 64-bit key narrows the search and decides nothing (`kernel/seal-os/src/ml_engine/foliation.rs`).

### 5) The T4 governor and its gain margin

The governor adapts a wake-up threshold $\varepsilon_t$ from an observed deviation $\Delta_t$ with a PD step:

$$
e_t = R^\ast - \frac{\Delta_t}{\varepsilon_t}, \qquad
\varepsilon_{t+1} = \operatorname{clamp}\!\left(\varepsilon_t - \alpha\, e_t - \beta\,\frac{e_t - e_{t-1}}{\Delta t}\right). \tag{8}
$$

T4 (AGCR) certifies geometric convergence when the gain margin is below one:

$$
\alpha + \frac{\beta}{\Delta t} < 1 \;\Longrightarrow\; \rho = 1 - \frac{\alpha}{1 + \beta/\Delta t} \in (0, 1). \tag{9}
$$

With the shipped gains $\alpha = 0.01$, $\beta = 0.05$, the margin is 0.06 at $\Delta t = 1$ and 5.01 at $\Delta t = 0.01$, the tick every runtime caller passes. Condition (9) also treats the map from $\varepsilon$ to $e$ as unit gain; linearising (8), $\partial e/\partial\varepsilon = \Delta/\varepsilon^2$, which is about $10^6$ at equilibrium (stated by the 2026-09-27 verification, not re-derived here). Folding that gain $K$ into (9) multiplies the margin by it; $\alpha K \approx 10^4$ alone exceeds one, so no choice of $\Delta t$ satisfies the condition. Earning T4 requires redesigning the governor, not retuning it.

## Implementation

### Kernel

`kernel/seal-os` is a single `no_std` crate built for `x86_64-unknown-uefi` with `build-std = ["core", "alloc"]` (`kernel/seal-os/.cargo/config.toml`). It boots only as a UEFI application: there is no Linux boot protocol, no initramfs and no kernel command line. Drivers, filesystems, the network stack, the window manager and the applications are compiled into one EFI image. At `9ebbe2e` the crate has 223 `.rs` files and 96,110 lines under `src/` (`find kernel/seal-os/src -name '*.rs' -exec cat {} + | wc -l`). The crate is in the workspace `exclude` list, so `cargo test --workspace` never compiles it; its tests run only inside QEMU.

| Subsystem | Path under `kernel/seal-os/src` | Lines | Contents |
|---|---|---:|---|
| Drivers | `drivers/` (62 files) | 20,415 | ACPI (RSDP/XSDT, MADT, FADT), Local and IO APIC, AHCI, NVMe, virtio-blk, virtio-net, e1000, xHCI with HID and mass storage, Intel HDA, virtio-gpu 2D, AMD GCN PM4 queue, RDRAND, RTC, serial. WiFi and Bluetooth are PCI probes only. PCI configuration through ports 0xCF8/0xCFC. |
| Filesystems | `fs/` | 14,875 | VFS, ext2, FAT12/16/32 (used by the installer and a parity proof, not mounted in the VFS), ManifoldFS (persisted through ext2), procfs (`version`, `uptime`, `cpuinfo`, `meminfo`, `self`, `1`), sysfs (`bus/pci/devices` only), devtmpfs (`null`, `zero`, `random`, `console`), pipefs. |
| Applications | `apps/` (20 files) | 9,798 | Shell, terminal, IDE, calculator, media player, tensor viewer, games; kernel code, not user processes. |
| Network | `net/`, `drivers/net/` | 8,946 | ARP, IPv4, IPv6 with NDP, ICMP, UDP, TCP, DHCP, DNS. TLS 1.3 client (`drivers/net/tls.rs`): X25519, X.509 with Ed25519 certificates only, AES-128-GCM. |
| Security | `security/` | 6,406 | KPTI, SMEP/SMAP enablement, KASLR of mappings, seccomp, audit log, shadow passwords, MAC, `unsafe` census. |
| Processes | `process/` | 4,637 | Scheduler, ELF loader, `syscall` entry, signals, context switch. |
| Memory | `memory/` | 4,512 | Frame allocator, TopoRAM (three Voronoi zones with eight seeds each), slab, page tables, swap. |
| ML services | `ml_engine/`, `ml_engine.rs` | 3,927 + 706 | `stratum`, `foliation`. |
| Desktop | `wm/`, `graphics/` | 3,665 + 3,117 | Compositor, desktop, software rasteriser. |
| Packages, modules | `pkg/`, `atlas/` | 2,635 + 1,899 | ManifoldPkg `.eph` packages and loadable ELF64 relocatable "charts", both Ed25519-signed. |
| System calls | `syscall/` | 1,972 | Dispatch table. |
| Other | `lib.rs`, `sandbox.rs`, `tuner.rs` | 2,520 + 1,290 + 717 | Boot sequence and theorem gate, sandbox, tuner. |

The surrounding workspace supplies the mathematics and the tooling: `aether-core` (certified β₀, certified top-k, trajectory shape, spherical Voronoi indices, SCM, governor), `aether-verified` (Rust theorem kernels and their Lean 4 sources), `epsilon-os` (a host-side model of ManifoldFS that runs the T4 check at runtime constants), Aether-Lang (a scripting language whose `no_std` runtime the kernel embeds), and `seal-mkimage` (the disk-image builder and every boot-log gate). Repository-wide line count, rewritten on each push to `main` by `.github/workflows/loc.yml`; the assembly figure counts only `.S`, `.s` and `.asm` files, so the kernel's `global_asm!` and `asm!` blocks are not in it:

<!-- RUST_LINE_COUNT_START -->
**187541 lines of Rust** across 504 files | 0 lines of x86 assembly | 1823 lines of Aether-Lang DSL | **189364 total**
<!-- RUST_LINE_COUNT_END -->

### Seal ABI

User code enters through `syscall` and leaves through `sysretq` (`process/userspace.rs:129-166`). Three argument registers are used, `rdi`, `rsi` and `rdx`; the seccomp filter runs first on every call (`syscall/table.rs:683`); 69 numbers are dispatched (`syscall/table.rs:680`). The numbering is Seal OS's own: of these numbers only `write` = 1 means the same call as on Linux x86_64. There are no socket system calls.

| Numbers | Calls |
|---|---|
| 0–11, 14–45 (44 calls) | exit, write, read, open, close, exec, fork, waitpid, mmap, getpid, stat, mkdir; chdir, getcwd, setuid, setgid, reboot, lseek, unlink, rmdir, rename, getrandom, kmsg_read, kill, sigaction, sigreturn, pipe, dup, dup2, brk, gettimeofday, settimeofday, watchdog, ioctl, sleep, sync, getppid, nanosleep, seteuid, setegid, clone, setrlimit, getrlimit, sigaltstack |
| 100–111 | manifold query, teleport, theorem status, package install / remove / list, WiFi scan / connect, Bluetooth scan / pair, setting get / set |
| 112–114 | chart graft / prune / list (`atlas`) |
| 120–124 | fit register / observe / regime / calibrate / unregister (`stratum`) |
| 130–134 | KV sequence create / append / release / stats, policy stats (`foliation`); a task that did not open a sequence gets `NoSuchSeq` |

### Boot proofs and gates

`seal-mkimage` builds a GPT disk image with a FAT EFI System Partition holding `EFI/BOOT/BOOTX64.EFI` and a ManifoldFS partition, and it parses the kernel's serial log. CI boots the image under QEMU, requires 25 milestone strings, then runs one `--check-*` gate per serial proof line. A missing or malformed line fails the job. The gates, and the claims this README may make about them:

| Serial marker | Gate | What the line must show (run 36165748105) |
|---|---|---|
| `[THEOREM] T1/TSS` … `T10/WPHB` | `--check-theorem-log` | Ten `VERIFIED` lines, the T4 line included; see the theorem table below for what each one checks. |
| `[BENCH] manifold-teleport` | `--check-benchmark-log` | Same-inode move with `fs_mode=mock_block` and `persistence_bytes_per_move=0`, at most 7 metadata operations. |
| `[ManifoldPkg] proof` | `--check-theorem-log` | Parse, install, extract, list and remove of an embedded package with `signature=ed25519_fixture` and `registry_index=ed25519_fixture`; rollback, tamper and digest-mismatch refusals. The channel transport is `fixture_loopback`: Public remote release channel is still pending. |
| `[SECURITY] audit proof` | `--check-theorem-log` | Audit buffer flushed and read back from `/var/log/audit.log`. |
| `[SECURITY] auth proof` | `--check-theorem-log` | `/etc/shadow` present, `$topo$5000` hashes, and `seal`/`seal` is rejected. |
| `[MM] cow-proof` | `--check-theorem-log` | 4 of 4 rollback samples succeed, 10 of 10 tracked frames freed, no fork or clone fallback to the parent page table. |
| `[AHCI]`, `[VFS]` | `--check-vm-proof` | 1024×768 GOP mode, AHCI disk registered and readable, ManifoldFS mounted from disk, no ramfs fallback. |
| `[LAAMBA] app proof:` | `--check-laamba-app-proof` | Native kernel window with launcher and start-menu entries. |
| `[Aether-Lang] runtime proof` | `seal-mkimage --check-aether-runtime /tmp/seal-os.log` | The embedded Aether-Lang runtime evaluates its boot probe. |
| `[FSPARITY] proof` | `--check-fs-parity` | FAT16 and ext2 fixtures driven through the same operations and compared byte for byte, with a corrupt-a-byte negative control. Read/write/create/mkdir/unlink/rmdir/rename/stat/readdir source paths are now `--check-doc-claim-contract` gated for both FAT and ext2. |
| `[KVPOLICY] proof`, `[MLFIT] proof` | `--check-kv-policy`, `--check-mlfit-proof` | See Results. |
| `[GPU-BENCH]` | `--check-gpu-bench` | CPU fallback correctness only. Hardware dispatch still needs a proof artifact. |
| `[TLS] proof` | `--check-tls-proof` | `x509=1 chain_verify=1 ecdhe=1 curve=x25519 psk_only=0`. |
| `[BENCH] tensor-render` | `seal-mkimage --check-benchmark-log /tmp/seal-os.log` | 100×100 CSV rendered by grid/value-height projection into 10,000 points and 19,602 triangles. |

TopCrypt is topological encoding/obfuscation, not cryptographic protection: `fs/topcrypt.rs` stores 64-byte blocks as 16-point clouds on S² with CRC32, a shuffle and XOR masks, and has no AEAD or key derivation.

### Theorem gates T1–T10

`init_theorems` in `kernel/seal-os/src/lib.rs` evaluates `verify_topology_theorems` (`lib.rs:2154-2208`) at boot and panics if any entry is false; the boot log of run 36165748105 prints all ten as `VERIFIED`. What that line establishes differs per theorem. "Certified" below means the condition holds at the inputs the running kernel uses; "refused" means it was evaluated there and fails; "not checked" means it was evaluated only on fixed boot constants, or nothing at runtime consumes it. Lean sources are in `kernel/aether/aether-verified/lean/EpsilonTheorems.lean`; the Lean 4 CI job built them in run 36165748105, and `--check-lean-proof-hygiene` rejects `sorry`, `admit` and `axiom`.

| ID | Name | Lean artifact | Boot evaluation | Status at runtime inputs, and where decided |
|---|---|---|---|---|
| T1 | TSS | `tss_packing_bound`, layered on a named cap-area hypothesis; `tss_separation_guarantee` is a `True` placeholder | Packing bound and pairwise separation of the 8 boot centroids | **Not checked.** The runtime indices are built from other centroid sets (`process/scheduler.rs:279`, `fs/voronoi_cap.rs:74`, `net/firewall.rs:84`, `net/topological.rs:76`). |
| T2 | SCM | `scm_contraction` proves only the inequality $1 - \alpha < 1$ for $\alpha \in (0,1)$ | One pair contracted at $\alpha = 0.1$ | **Certified**, elementarily: the operator $(1-\alpha)S + \alpha P$ contracts by $1-\alpha < 1$ at the runtime gains 0.7 (`process/scheduler.rs:292`, `fs/manifold_fs.rs:277`) and 0.3 (`net/firewall.rs:67`). |
| T3 | GMC | `gmc_bounded_termination` proved; `gmc_entropy_nonincreasing` is a `True` placeholder | Fixed constants (100, 50, 1000) and `max_merges(8) == 7` | **Not checked.** |
| T4 | AGCR | `agcr_gain_margin_stable`, conditional on (9) | Evaluated at $\Delta t = 1.0$ (`lib.rs:2180`), margin 0.06 | **Refused.** Every runtime call passes $\Delta t = 0.01$ (`process/scheduler.rs:560`, `fs/manifold_fs.rs:810`, `wm/compositor.rs:534`), margin 5.01. `epsilon-os` refuses it (`epsilon-os/src/manifold_fs.rs:31-39`, `world.rs:596-604`); the seal-os boot line `T4/AGCR VERIFIED` is unearned (see Limits). |
| T5 | HCS | `hcs_separation`, an exact identity | Fixed constants | **Not checked.** |
| T6 | RGCS | `rgcs_coherence_bound` (non-negativity) | Fixed constants | **Not checked**; no runtime consumer. |
| T7 | PHKP | `phkp_perfect_locality` is a `True` placeholder | Fixed constants | **Not checked**; no runtime consumer. |
| T8 | TEB | `teb_energy_nonneg` | Landauer bound at 300 K lies in (2.8e-21, 2.9e-21) J | **Not checked**; no runtime consumer. |
| T9 | CMA | `cma_linear_accumulation` | Fixed constants | **Not checked**; no runtime consumer. |
| T10 | WPHB | `wphb_topological_advantage`, `wphb_multi_model` | Fixed constants | **Not checked**; no runtime consumer. |

The per-theorem proof strengths and closure criteria are in [docs/THEOREMS.md](docs/THEOREMS.md) and [kernel/aether/aether-verified/lean/README.md](kernel/aether/aether-verified/lean/README.md).

## Results

### Test suites and boot

| Measurement | Value | Condition | Provenance |
|---|---|---|---|
| Host test suite | 861 passed, 0 failed, 2 ignored, across 94 test targets | `cargo test --workspace`; does not compile `kernel/seal-os` | CI run 36165748105, commit `9ebbe2e` |
| In-kernel harness, last CI run | 514 / 514 passed | QEMU, `--features test-mode` image | Kernel Tests run 31487106604, 2026-08-11, commit `f913f32` |
| In-kernel harness, local | 563 / 563 passed | QEMU, local session, 2026-09-25 | Stated, not a CI result |
| In-kernel tests registered | 563 `register_test` call sites | commit `9ebbe2e` | `grep -rn "register_test(" kernel/seal-os/src`, less the definition |
| QEMU boot milestones | 25 / 25 | q35, OVMF, AHCI disk, no NIC, 4 GiB | CI run 36165748105 |

### ML services, from QEMU serial proof lines (CI run 36165748105)

| Measurement | Value | Condition |
|---|---|---|
| `foliation` hit rate | 952 bp; Belady 952 bp, LRU 0 bp, random 619 bp | Synthetic trace of 30 requests and 1,680 tokens; 24-block pool, 8 tokens per block |
| `foliation` against random | Wins on 32 / 32 seeds; random spans 238 to 857 bp | Same trace |
| `foliation` sharing and safety | 20 shared descents, 81,920 bytes saved; 190 frames backed and 190 freed, 0 failed; 0 referenced evictions, 0 collapse violations | Same trace |
| `stratum` classification | 7 / 7 correct: underfit, wellfit, overfit, collapsing, negative control, monotone line, monotone exponential | Synthetic streams of 128 steps, window 64, $\kappa = 1.68$ |
| `stratum` negative control | Detector: not flagged. Gap-threshold baseline: flagged | Healthy run with an irreducible validation gap |
| `stratum` stream size | 4,792 bytes per stream, bounded over a 4,096-step stream | |

The ABI default policy is LRU, not the foliation ranking (`ml_engine/foliation.rs:1290-1294`); the boot proof selects each policy explicitly. Reproduce: boot as in Quick Start, then `--check-kv-policy /tmp/seal-os.log` and `--check-mlfit-proof /tmp/seal-os.log`.

### Certify-or-refuse, before and after

| Quantity | Before | After | Provenance |
|---|---|---|---|
| Nearest centroid on S² | 1,215 of 5,000 seeded queries wrong, reported hit rate 1.0000 | 0 wrong; 4,909 certified in the 3×3 block, 91 full scans, reported hit rate 0.9818 | Project page, commits `cdb4a4e` and `0f040e0` (stated, not re-measured). `cargo test -p aether-core --test house_tss_grid_locate` asserts 0 wrong over 25,000 queries at K = 2, 8, 20, 64 and 200. |
| β₀ at scale | Two points at 0.5 ± 1e-9 gave two different integers | Both refused; 500 seeded clouds agree with all-pairs union-find | Project page, commit `8678c97`. `cargo test -p aether-core --test certified_betti` |
| Attention top-k | $k_0 = [10^{17}, 1, -10^{17}]$, $q = [1,1,1]$: float score 0.0 against exact 1, and key 1 (0.5) taken over key 0 | Row widened | Project page, commits `afd0969`, `eb4af16`. `cargo test -p aether-core --test attention_contracts` |
| Loop score of a monotone staircase | 0.969, verdict Overfit | Certified 0 before any complex is built | `cargo test -p aether-core --test trajectory_shape` (`monotone_staircase_scores_no_fold`) |
| T4 in `epsilon-os` | Certified at $\Delta t = 1$ while the governor ticks at 0.01 | Refused at $\Delta t = 0.01$, margin 5.01 | Project page, commits `c5b5853`, `6c450d4`, `2605f70`. `cargo test -p epsilon-os` (`test_verify_theorems_pass_except_uncertified_runtime_t4`) |

### Filesystem, GPU and security lines (CI run 36165748105)

| Measurement | Value |
|---|---|
| FAT16 against ext2 parity | 19 operations each; 4 files and 4,388 bytes equal byte for byte; 28 / 28 stat fields; 8 / 8 error cases; 17 expected divergences (mode, mtime, directory size, case, 8.3 names, cross-directory); negative control detected |
| GPU | CPU fallback only: 3 / 3 kernels agree with a CPU recompute; `hardware_dispatch=0`. One of four kernels has GFX9 machine code (96 bytes; 24 / 24 words round-trip, 17 / 17 instructions decode); no AMD GPU was present |
| KASLR | 30 bits (8 kernel-alias, 22 heap-window) from RDRAND; the image base is not randomised (firmware base `0x140000000`) |
| `unsafe` census | 624 blocks in 84 files; 12 carry a safety comment, 612 do not |
| W^X | 4,311 of 4,311 scanned kernel-alias pages are writable and executable; not enforced |
| SMEP / SMAP | Not supported by `-cpu qemu64`; enablement code not exercised |

### Microbenchmarks

| Benchmark | p50 / p95 cycles | Notes |
|---|---|---|
| `alloc-frame` | 1,958 / 2,130 | 64 iterations, 64 fast-path hits, no frame leak |
| `slab-alloc` | 272 / 354 | 6 size classes |
| `toporam-alloc` | 6,126 / 12,446 | 64 / 64 target-cell hits |
| `manifold-lookup` | 3,060 / 4,508 | 4-component paths, at most 6 directory-hash probes against a bound of 256 |
| `scheduler-select-next` | 304,184 / 386,796 | 64 selections, 0 context switches |
| `tcp-roundtrip` | not timed | 8 / 8 loopback connections, 512 bytes echoed |

Substrate: GitHub-hosted `ubuntu-latest` runner, QEMU `-machine q35 -cpu qemu64,+rdrand -m 4G` without `-accel kvm`, so QEMU runs its TCG emulator and every cycle count above is an emulated TSC delta, not a hardware measurement. Toolchains: rustc 1.100.0-nightly (f7575a9da 2026-09-24) for the kernel, rustc 1.98.1 for host crates. Reproduce with `seal-mkimage --check-benchmark-log /tmp/seal-os.log` after a boot.

The comparison table in [docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md) is not a blanket victory claim. Seal OS only claims a win over Ubuntu for a row after the same-machine benchmark exists, bound by `--check-current-benchmark-proof`; the native Ubuntu 26.04 allocator job was skipped in run 36165748105, so every row is raw Ubuntu artifact pending. Where Seal OS must still prove superiority: every row of that plan, since none has been measured on the same machine.

## Results imported from related work

Techniques ported from the author's other repositories, each with the failing test that showed the defect and the passing state after the port.

| Source | Result ported | Site in this repository | Evidence, RED to GREEN | Landed |
|---|---|---|---|---|
| github.com/teerthsharma/planimeter | Gap rule: a count is certified only when no merge height lies in a band around the scale | `aether-core/src/certified_betti.rs`, `certified_beta0` | A chord at 0.5 ± 1e-9 returned two different integers; both are refused now, and 500 seeded clouds agree with all-pairs union-find (commit `8678c97`, project page) | Before 2026-09-27 |
| github.com/teerthsharma/separatrix | Error-interval separation test for a decision boundary | `aether-core/src/attention.rs`, `certified_top_k` | Budget-1 top-k took key 1 (score 0.5) over key 0 (exact score 1); the row now widens (commits `afd0969`, `eb4af16`, project page) | Before 2026-09-27 |
| github.com/teerthsharma/cleave | Union-find join "younger dies, elder absorbs" (`persist` module, line 181) | `aether-core/src/ml/clustering.rs`, `cut_tree` | Merge ids n+m were never resolved, so a merge joining an already-merged cluster was dropped. RED: `k = 1 is one cluster left: [0, 0, 1] right: [0, 0, 0]`. GREEN: 433 / 433 aether-core tests | Side branch, 2026-09-27; stated by the integration agent, not re-measured here |
| github.com/teerthsharma/cleave | Same join | `epsilon-os/src/manifold_fs.rs`, `check_entropy_and_merge` | After an entropy merge, new files were still placed in the emptied cell. RED: `same content, same cell left: 2 right: 4`. GREEN: 48 / 48 epsilon-os tests | Side branch, 2026-09-27; stated, not re-measured here |
<!-- INTEGRATIONS: filled after integration round -->

Two further changes landed on side branches in the same round and are not imports. `fs/fat.rs` `write_fat_entry` now writes every live FAT copy and honours the FAT32 ExtFlags mirroring field; two new in-kernel tests were RED (563 / 565) and the suite is 565 / 565 under QEMU after the fix. `epsilon-os` gained a guard test that a T4 certificate must be backed by the loop actually settling (48 / 48); T4 remains refused at $\Delta t = 0.01$. Both are stated by the integration agents and not re-measured here.

## Quick Start

```bash
git clone https://github.com/teerthsharma/Epsilon-Hollow
cd Epsilon-Hollow

# Host crates (stable toolchain)
cargo test --workspace

# Kernel: nightly, x86_64-unknown-uefi, build-std
(cd kernel/seal-os && cargo +nightly build --release)

# GPT disk image: FAT ESP with EFI/BOOT/BOOTX64.EFI plus a ManifoldFS partition
(cd kernel/seal-mkimage && cargo +stable run --release)

# Boot exactly as CI does (OVMF path as packaged by Debian and Ubuntu)
timeout 240 qemu-system-x86_64 -machine q35 -cpu qemu64,+rdrand \
  -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
  -device ahci,id=seal_sata \
  -drive if=none,id=seal_disk,file=kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img,format=raw,media=disk \
  -device ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0 \
  -nographic -m 4G -no-reboot -no-shutdown | tee /tmp/seal-os.log

# Check the serial log with the gates CI runs
gate() { cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- "$@"; }
gate --check-theorem-log /tmp/seal-os.log
gate --check-kv-policy /tmp/seal-os.log
gate --check-mlfit-proof /tmp/seal-os.log
gate --check-fs-parity /tmp/seal-os.log
gate --check-benchmark-log /tmp/seal-os.log
```

A successful boot reaches `[BOOT] Seal OS desktop ready.` and `[EVENT] Entering real event loop` on the serial console; QEMU keeps running until `timeout` stops it, which CI treats as success. `scripts/test_kernel.sh` builds the `test-mode` kernel, boots it and prints `ALL TESTS PASSED` when every registered in-kernel test passes. On Windows, `kernel/seal-os/run-qemu.ps1` boots the image; `kernel/seal-os/build-vbox.ps1` and `smoke-vbox.ps1` convert and smoke-test it under VirtualBox. The full CI job list is in [docs/CI.md](docs/CI.md).

## Requirements

| Component | Requirement | Where it is set |
|---|---|---|
| Host crates | Rust stable. `rust-version = "1.85"` is declared by `aether-link` and `ubuntu-alloc-bench`; CI builds with 1.98.1 and does not test 1.85. | `rust-toolchain.toml`, `.github/workflows/ci.yml` |
| Kernel | Rust nightly with `rust-src` and `llvm-tools-preview` (CI: 1.100.0-nightly f7575a9da, 2026-09-24). Unstable features: `abi_x86_interrupt`, `build-std`. | `kernel/seal-os/rust-toolchain.toml`, `kernel/seal-os/.cargo/config.toml` |
| Machine | x86_64 with long mode and UEFI firmware. Boots only as a UEFI application. RDRAND supplies KASLR entropy; CI enables it with `-cpu qemu64,+rdrand`. | `boot/uefi_entry.rs`, `security/kaslr.rs` |
| Emulator | `qemu-system-x86_64` and OVMF (CI installs `qemu-system-x86 ovmf socat` on Ubuntu). Proven configuration: q35, AHCI disk, 4 GiB, no NIC. | `.github/workflows/ci.yml` |
| Display | 1024×768 framebuffer for the desktop; the serial console carries every proof line. | `graphics/`, checked by the `[GFX] desktop-proof` line |
| Proofs (optional) | Lean 4 v4.7.0 with mathlib v4.7.0; `lake build` in `kernel/aether/aether-verified/lean`. | `lean-toolchain` |

## Repository layout

| Path | Contents |
|---|---|
| `kernel/seal-os/` | The kernel; see [kernel/seal-os/ARCHITECTURE.md](kernel/seal-os/ARCHITECTURE.md) and [kernel/seal-os/TESTING.md](kernel/seal-os/TESTING.md) |
| `kernel/seal-mkimage/` | Disk-image builder and every `--check-*` gate |
| `kernel/epsilon/epsilon/crates/` | `aether-core` (mathematics), `epsilon` and `epsilon-os` (host-side ManifoldFS and world model) |
| `kernel/aether/` | `aether-verified` (Rust and Lean 4), Aether-Lang crates, `aether-link` (IO scheduling, with the `io_cycle_8_lbas` bench regression gate) |
| `kernel/seal-graph/`, `kernel/seal-jit/`, `kernel/seal-net80211/` | ML graph artifact format and executor, execution-plan autotune memo, IEEE 802.11 frame codec and WPA2/WPA3 supplicant state machine |
| `tools/ubuntu-alloc-bench/` | Ubuntu allocator baseline for the comparison gate |
| `apps/laamba-governor/` | Tauri desktop application (host side) |
| `docs/` | Design documents ([THEOREMS](docs/THEOREMS.md), [MANIFOLDFS](docs/MANIFOLDFS.md), [THREAT_MODEL](docs/THREAT_MODEL.md), [UNSAFE_INVENTORY](docs/UNSAFE_INVENTORY.md), [GPU_ACCELERATION](docs/GPU_ACCELERATION.md)), the project page `index.html`, and [RECORD.md](docs/RECORD.md) |
| `infrastructure/`, `tests/`, `future/` | Legacy host tooling and excluded experiments, outside the build, boot and proof paths per [docs/HOST_LANGUAGE_QUARANTINE.md](docs/HOST_LANGUAGE_QUARANTINE.md) |

## Direction

<!-- DIRECTION: filled after the Linux-parity review -->

## Limits

Items marked "code reading" were found by reading the source at `9ebbe2e` on 2026-09-27 and have not yet been reproduced at runtime.

1. **Syscall entry is unsafe to use from ring 3 (code reading).** `syscall` does not change `rsp`, and `syscall_entry` pushes the register frame and calls into Rust on that user-controlled stack: there is no switch to a kernel stack and no `swapgs` (`process/userspace.rs:129-166`). The kernel never sets `EFER.SCE`, so `syscall` from ring 3 works only if firmware left it set; the `[SECURITY-FEATURES]` line of run 36165748105 reads `efer=0xd00`, with SCE clear.
2. **No task is ever scheduled (code reading).** `PerCpu::current_task` is assigned only inside `schedule()`, and every non-test caller of `schedule()` returns early while it is null (`process/scheduler.rs:546`, `1394`, `1408`; `cpu/smp.rs:160-166`); the comment at `scheduler.rs:1496-1503` states the same. The three tasks spawned at boot never run.
3. **No user space runs.** `/bin/init` is absent from the image, and the kernel falls back to its in-kernel desktop (`[execve] '/bin/init' not found` in run 36165748105). The ELF loader builds no argc, argv, envp or auxiliary vector and applies only `R_X86_64_RELATIVE` relocations, skipping every other type (`process/elf.rs:716`). `SYS_WAITPID` returns its first argument without waiting (`syscall/table.rs:1031`, code reading). All applications are kernel code.
4. **T4 is certified at boot without being earned.** seal-os evaluates the gain margin at $\Delta t = 1.0$ (`lib.rs:2180`) while every runtime caller passes 0.01, where the margin is 5.01; the boot line `T4/AGCR VERIFIED` is therefore unearned. A fix is queued, and it must change the gate as well: `--check-theorem-log` requires that exact line (`kernel/seal-mkimage/src/main.rs:1131`). Because the plant gain is about $10^6$, no step size earns T4; the governor needs redesign. Of the other nine lines, only T2 holds at runtime inputs.
5. **The kernel's own unit tests are not run by CI.** `kernel/seal-os` is excluded from the workspace, and the Kernel Tests workflow runs only after a fully green CI run; its last 100 runs were skipped, and the last one to execute passed 514 / 514 on 2026-08-11.
6. **CI is red, and some milestones are weak.** The QEMU job of run 36165748105 fails at the language-hygiene gate, on line 7930 of the previous README and on `scripts/ci_parity.sh:163-164`; the gates after it in that job did not run. This README and `docs/` pass the gate; `scripts/ci_parity.sh` still fails it. Two of the 25 milestones are string matches that prove little: "Syscalls verified" matches `[BOOT] SYSCALL/SYSRET MSRs programmed`, and "Scheduler started" matches any line containing `Scheduler`.
7. **The ML services have never seen a real model.** Every `stratum` and `foliation` number comes from a synthetic fixture. `FitAction` is advisory and enforced nowhere (`ml_engine/stratum.rs:170-178`). The foliation ranking beats LRU only at the capacity cliff built into its trace and ties or loses elsewhere. The band for $\kappa$ in (6) is proved only for a symmetric fold; an asymmetric fold closes later. Whether kernel placement of the KV cache buys anything over user-space PagedAttention has not been measured.
8. **Hardware coverage is one QEMU configuration.** In CI, AHCI works and NVMe, HDA, xHCI and every NIC are reported absent. There is no driver binding framework, no PCIe ECAM and no MSI; WiFi and Bluetooth are PCI probes only. The GPU path has executed only on the CPU fallback.
9. **Filesystems and networking are partial.** There is no ext4; FAT is not mounted in the VFS; sysfs exposes only PCI devices. TLS accepts Ed25519 certificates only, so a server presenting an RSA or ECDSA certificate is refused, and no socket system call exposes the network stack. The doc-claim contract in `kernel/seal-mkimage/src/main.rs` (`check_doc_claim_contract_text`) still requires this README to contain the phrases "Minimal TLS 1.3 PSK record path" and "no X.509/PKI/ECDHE gate yet". The second is out of date: the `[TLS]` line of run 36165748105 reports `x509=1 chain_verify=1 ecdhe=1`, and only the `[BENCH] tls-encrypt` fixture (`psk_aes_128_gcm_record`) is PSK-only. Both phrases are quoted here because the gate requires them.
10. **Security mitigations are measured, not complete.** W^X is violated on every scanned kernel-alias page and not enforced; KASLR randomises mappings, not the image base; SMEP and SMAP were not exercised because the CI CPU model lacks them; 612 of 624 `unsafe` blocks carry no safety comment.
11. **No performance comparison exists.** Every cycle count is from QEMU TCG. No Ubuntu or Linux comparison has been run.
12. **Formal verification covers side lemmas.** The Lean files prove algebraic facts about constants and bounds; three theorem statements and one pruning bound are `True` placeholders, and no Lean statement is connected to kernel code by refinement.

## Citation and license

Cite with [CITATION.cff](CITATION.cff) or DOI [10.5281/zenodo.20264206](https://doi.org/10.5281/zenodo.20264206). Released under the MIT License ([LICENSE](LICENSE)). The tree contains no GPL code; `deny.toml` bans copyleft licenses except LGPL-3.0 for the transitive `wav` crate. Security reports: [SECURITY.md](SECURITY.md). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).
