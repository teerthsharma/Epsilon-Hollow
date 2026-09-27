# Seal OS: measured results, provenance and imported work

Measured results, provenance and imported work for Seal OS; the README links here.

Until 2026-09-27 this material opened the README. It moved here unchanged except for links made relative to `docs/`, the repository line count (which stays in the README, where CI rewrites it), the per-theorem runtime status table (now in [THEOREMS.md](THEOREMS.md#status-at-runtime-inputs)), a new section recounting Linux coverage for the README's charts, and dated updates where a later commit on `main` changed a finding. Nothing measured was dropped.

Every number below carries its provenance: a command that reproduces it, a CI run id with its commit, a commit whose message records the run, or the words "stated, not re-measured". The verification pass of 2026-09-27 measured at commit `9ebbe2e`; CI numbers come from run [36165748105](https://github.com/teerthsharma/Epsilon-Hollow/actions/runs/36165748105) on the same commit. Results that landed later that day, on branch `revamp/certify-or-refuse`, are quoted from the commit that landed each one. File and line references are to the tree at `c055594`. The development history that earlier versions of the README carried (verifier findings, red-gate investigations, per-round logs) is preserved verbatim in [docs/RECORD.md](RECORD.md).

---

## Summary

Seal OS is a research kernel for x86_64, written in `no_std` Rust, booted only through UEFI, with its own 69-call system-call ABI. It tests one proposition: a kernel that serves machine-learning workloads can compute their structure itself, the shape of a training run's validation curve and the prefix structure of an inference server's KV cache, and it should refuse to return a discrete answer whose value was chosen by floating-point rounding rather than by the data. Component counts are certified by an empty band around the scale in the minimum-spanning-tree merge heights, attention top-k by Higham error intervals, and nearest-centroid lookup on S² by a distance-to-boundary test; before that test, the grid-hash index answered 1,215 of 5,000 seeded queries wrongly while reporting a hit rate of 1.0000 (commit `cdb4a4e`, as stated on the project page). Under QEMU, the `foliation` KV policy matches the Belady hit rate on a synthetic trace built so that recency always evicts the shared prefix (952 bp, LRU 0 bp; CI run 36165748105) and loses to LRU on a multi-turn chat trace (5,284 bp against 8,068; commit `0ab2377`), so the system-call-facing cache defaults to LRU; `stratum` classifies 7 of 7 synthetic runs, including a negative control that a train/validation gap threshold misclassifies. The accepted direction is for Seal OS to replace the Linux kernel under an unmodified distribution; no ring-3 instruction has yet executed (see Direction). What does not work is stated once, in Limits.

## Linux coverage, counted

The README draws these counts as charts. Each is read from the failure message of a check that is RED by design, because it is the gate for open work; recounted at `a50b8d6` on 2026-09-27 on Windows 11 with the host runner, over the checks under [tests/linux_parity/](../tests/linux_parity/). The Linux side was captured once by `tests/linux_parity/chase_measure_linux.sh` on Ubuntu 26.04 LTS (glibc 2.43, gcc 15.2.0, running on kernel 6.6.87.2-microsoft-standard-WSL2) and is checked in as `tests/linux_parity/chase_linux_measured.json`.

| Quantity | Count at `a50b8d6` | Measured by |
|---|---|---|
| Linux x86_64 system calls that reach the same call in Seal dispatch | 1 of 386 (`write`, number 1) | Linux: `arch/x86/entry/syscalls/syscall_64.tbl` at Linux master 7.3.0-rc4, 339 `common` plus 47 `64` entries, fetched and counted on 2026-09-27. Seal: the 69 dispatched numbers in `kernel/seal-os/src/syscall/table.rs`, compared by name; no other Seal number carries the Linux call of the same number. The Ubuntu 26.04 `asm/unistd_64.h` behind the glibc fixture defines 385 numbers; the count of 1 holds against either table |
| System calls made by five Ubuntu programs (`cat`, a shell pipeline, a static hello, `systemd --version`, `true`) | 49 distinct: 1 reaches the same call, 22 reach a different Seal call, 26 reach no handler | host runner: `test_chase_syscall_abi`, `test_linux_syscall_numbers_reach_the_same_call_in_seal_dispatch`. Example misroutes: `access` reaches rmdir, `getgid` reaches package removal, `munmap` reaches mkdir |
| Distinct system calls of a static glibc `hello` (`gcc -static -O2`) | 1 of 12 reaches the same call; 11 are misrouted or unhandled | host runner: `test_cameron_linux_abi_surface` |
| Kernel symbols imported by the 480 driver modules under `drivers/` of that kernel | Seal OS exports 4 of 7,243 | host runner: `test_chase_linux_drivers`. Binary reuse is pinned as well: 924 of 924 modules carry one vermagic string and per-symbol CRCs, and the licence gate refuses 853 of 924 |
| M0 items whose gate passes | 4 of 11 | Ticked boxes in [FUTURE_PLAN.md, Phase 0](../FUTURE_PLAN.md#phase-0-linux-kernel-replacement) at `a50b8d6` |
| Milestones M0 to M8 passed | 0 of 9 | Same |
| Distributions that boot on Seal OS | 0 | The M3 and later gates are not yet written |

The driver-symbol count is not a target. Decision D3 of the accepted plan runs Linux drivers unmodified inside separate driver servers built from LKL, the Linux kernel as a library, so Seal OS never has to export those symbols itself; the count records why that decision was taken.

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
| Asterinas | Rust | Framekernel: one address space, `unsafe` confined to a small framework, OSTD [4] | Yes; targets the Linux system-call ABI | Soundness argument rests on the small `unsafe` framework | Asterinas runs Linux binaries. Seal OS has its own 69-call ABI, which decision D1 of the accepted plan replaces with Linux's (see Direction), and 611 of its 627 `unsafe` blocks carry no safety comment (`kernel/seal-os/tests/unsafe-audit.fixture` at commit `b3cf934`). |
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

A leaf of the prefix tree may be evicted only if it is resident, its reference count is zero, and it has no resident children. Removing such a leaf is an elementary collapse, so the resident set stays a connected rooted subtree under every policy. Policies differ only in which candidate they pick: the foliation policy orders the frontier lexicographically by (number of distinct sequences that ever entered the leaf, negative depth, last use); a locality-only null by (negative depth, last use), the foliation order with the entrant count removed; LRU by last use; the Belady oracle by next use in a supplied future trace. Two sequences share a block only when their tokens are equal; the 64-bit key narrows the search and decides nothing (`kernel/seal-os/src/ml_engine/foliation.rs`).

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

With the shipped gains $\alpha = 0.01$, $\beta = 0.05$, the margin is 0.06 at $\Delta t = 1$ and 5.01 at $\Delta t = 0.01$, the tick every runtime caller passes (`GOVERNOR_ALPHA`, `GOVERNOR_BETA` and `GOVERNOR_DT`, `kernel/seal-os/src/lib.rs:161-163`); since commit `3c14df0` the boot gate evaluates (9) at exactly these values and refuses T4. Condition (9) also treats the map from $\varepsilon$ to $e$ as unit gain; linearising (8), $\partial e/\partial\varepsilon = \Delta/\varepsilon^2$. In the loop the host-side ManifoldFS model runs, `store()` feeds `adapt(1.0, dt)`, whose error $e = 1000 - 1/\varepsilon$ has gain $1/\varepsilon^2 = 10^6$ at the equilibrium $\varepsilon^\ast = 0.001$ (commit `dcc35b6`). Folding that gain $K$ into (9) multiplies the margin by it; $\alpha K \approx 10^4$ alone exceeds one, so no choice of $\Delta t$ satisfies the condition. Measured over 2,000 ticks from $\varepsilon = 0.1$, the loop 2-cycles between $\varepsilon = 0.001$ and $10$ at $\Delta t = 0.01$, $0.0506$ and $1$ alike (same commit). Earning T4 requires redesigning the governor, not retuning it.

## Implementation

### Kernel

`kernel/seal-os` is a single `no_std` crate built for `x86_64-unknown-uefi` with `build-std = ["core", "alloc"]` (`kernel/seal-os/.cargo/config.toml`). It boots only as a UEFI application: there is no Linux boot protocol, no initramfs and no kernel command line. Drivers, filesystems, the network stack, the window manager and the applications are compiled into one EFI image. At `9ebbe2e` the crate has 223 `.rs` files and 96,110 lines under `src/` (`find kernel/seal-os/src -name '*.rs' -exec cat {} + | wc -l`). The crate is in the workspace `exclude` list, so `cargo test --workspace` never compiles it; its tests run only inside QEMU.

| Subsystem | Path under `kernel/seal-os/src` | Lines | Contents |
|---|---|---:|---|
| Drivers | `drivers/` (62 files) | 20,415 | ACPI (RSDP/XSDT, MADT, FADT), Local and IO APIC, AHCI, NVMe, virtio-blk, virtio-net, e1000, xHCI with HID and mass storage, Intel HDA, virtio-gpu 2D, AMD GCN PM4 queue, RDRAND, RTC, serial. WiFi and Bluetooth are PCI probes only. PCI configuration through ports 0xCF8/0xCFC. |
| Filesystems | `fs/` | 14,875 | VFS, ext2, FAT12/16/32 (used by the installer and a parity proof, not mounted in the VFS; each FAT write reaches every live FAT copy, honouring FAT32 ExtFlags, since commit `1c24631`), ManifoldFS (persisted through ext2), procfs (`version`, `uptime`, `cpuinfo`, `meminfo`, `self`, `1`), sysfs (`bus/pci/devices` only), devtmpfs (`null`, `zero`, `random`, `console`), pipefs. |
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

The surrounding workspace supplies the mathematics and the tooling: `aether-core` (certified β₀, certified top-k, trajectory shape, spherical Voronoi indices, SCM, governor), `aether-verified` (Rust theorem kernels and their Lean 4 sources), `epsilon-os` (a host-side model of ManifoldFS that runs the T4 check at runtime constants), Aether-Lang (a scripting language whose `no_std` runtime the kernel embeds), and `seal-mkimage` (the disk-image builder and every boot-log gate). The repository-wide line count is in the README, where `.github/workflows/loc.yml` rewrites it on each push to `main`; its assembly figure counts only `.S`, `.s` and `.asm` files, so the kernel's `global_asm!` and `asm!` blocks are not in it.

### Seal ABI

User code enters through `syscall` and leaves through `sysretq` (`process/userspace.rs:129-166`). Three argument registers are used, `rdi`, `rsi` and `rdx`; the seccomp filter runs first on every call (`syscall/table.rs:683`); 69 numbers are dispatched (`syscall/table.rs:680`). The numbering is Seal OS's own: of these numbers only `write` = 1 means the same call as on Linux x86_64. There are no socket system calls. Decision D1 of the accepted plan replaces this table with the Linux x86_64 one and moves the Seal-specific calls to `/dev/seal` and `/sys/kernel/seal/` (see Direction).

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
| `[THEOREM] T1/TSS` … `T10/WPHB` | `--check-theorem-log` | Run 36165748105 required ten `VERIFIED` lines. Since commit `3c14df0` the gate requires nine (T1–T3, T5–T10), reads `alpha`, `beta` and `dt` from the `[T4/AGCR] Governor online` line, requires `[THEOREM] T4/AGCR NOT CERTIFIED: alpha+beta/dt=5.01 >= 1 at dt=0.01` and the summary `9 of 10 theorems VERIFIED`, and rejects a `T4/AGCR VERIFIED` line while the margin is at least 1. The theorem table in [THEOREMS.md](THEOREMS.md#status-at-runtime-inputs) says what each line checks. |
| `[BENCH] manifold-teleport` | `--check-benchmark-log` | Same-inode move with `fs_mode=mock_block` and `persistence_bytes_per_move=0`, at most 7 metadata operations. |
| `[ManifoldPkg] proof` | `--check-theorem-log` | Parse, install, extract, list and remove of an embedded package with `signature=ed25519_fixture` and `registry_index=ed25519_fixture`; rollback, tamper and digest-mismatch refusals. The channel transport is `fixture_loopback`: Public remote release channel is still pending. |
| `[SECURITY] audit proof` | `--check-theorem-log` | Audit buffer flushed and read back from `/var/log/audit.log`. |
| `[SECURITY] auth proof` | `--check-theorem-log` | `/etc/shadow` present, `$topo$5000` hashes, and `seal`/`seal` is rejected. |
| `[MM] cow-proof` | `--check-theorem-log` | 4 of 4 rollback samples succeed, 10 of 10 tracked frames freed, no fork or clone fallback to the parent page table. |
| `[AHCI]`, `[VFS]` | `--check-vm-proof` | 1024×768 GOP mode, AHCI disk registered and readable, ManifoldFS mounted from disk, no ramfs fallback. |
| `[LAAMBA] app proof:` | `--check-laamba-app-proof` | Native kernel window with launcher and start-menu entries. |
| `[Aether-Lang] runtime proof` | `seal-mkimage --check-aether-runtime /tmp/seal-os.log` | The embedded Aether-Lang runtime evaluates its boot probe. |
| `[FSPARITY] proof` | `--check-fs-parity` | FAT16 and ext2 fixtures driven through the same operations and compared byte for byte, with a corrupt-a-byte negative control. Read/write/create/mkdir/unlink/rmdir/rename/stat/readdir source paths are now `--check-doc-claim-contract` gated for both FAT and ext2. |
| `[KVPOLICY] proof`, `[MLFIT] proof` | `--check-kv-policy`, `--check-mlfit-proof` | See Results. Since commit `40d3568` the KV gate also requires the locality-null and chat-trace fields. |
| `[GPU-BENCH]` | `--check-gpu-bench` | CPU fallback correctness only. Hardware dispatch still needs a proof artifact. |
| `[TLS] proof` | `--check-tls-proof` | `x509=1 chain_verify=1 ecdhe=1 curve=x25519 psk_only=0`. |
| `[BENCH] tensor-render` | `seal-mkimage --check-benchmark-log /tmp/seal-os.log` | 100×100 CSV rendered by grid/value-height projection into 10,000 points and 19,602 triangles. |

TopCrypt is topological encoding/obfuscation, not cryptographic protection: `fs/topcrypt.rs` stores 64-byte blocks as 16-point clouds on S² with CRC32, a shuffle and XOR masks, and has no AEAD or key derivation.

### Theorem gates T1–T10

The status of each theorem line at the kernel's runtime inputs (certified, refused or not checked, and where each is decided) is in [THEOREMS.md, Status at runtime inputs](THEOREMS.md#status-at-runtime-inputs).

## Results

### Test suites and boot

| Measurement | Value | Condition | Provenance |
|---|---|---|---|
| Host test suite | 861 passed, 0 failed, 2 ignored, across 94 test targets | `cargo test --workspace`; does not compile `kernel/seal-os` | CI run 36165748105, commit `9ebbe2e` |
| Host, `aether-core` and `epsilon-os` | 490 passed, 0 failed, 0 ignored, across 58 test targets (`aether-core` 441, `epsilon-os` 49); 488 at `e2f1fb9` | `cargo +stable test -p aether-core -p epsilon-os`, commit `c055594`, Windows 11, rustc 1.97.1 | Measured for this README, not a CI result |
| In-kernel harness, last CI run | 514 / 514 passed | QEMU, `--features test-mode` image | Kernel Tests run 31487106604, 2026-08-11, commit `f913f32` |
| In-kernel harness, local | 563 / 563 on 2026-09-25; on 2026-09-27, per branch: 564 / 564 (`3c14df0`), 565 / 565 (`1c24631`), 563 / 563 (`87d7b10`), 564 / 564 (`b3cf934`), 570 / 570 (`e887a79`, with `9702061`), and 563 / 564 (`264235c`) and 564 / 565 (`0ab2377`), where the one failure was `tcp::time_wait_is_held_for_2msl_and_then_returns_the_port` | QEMU q35, OVMF, 1 GiB, local sessions | Stated in each commit message; no run of the merged tree at `c055594` as one test image is recorded |
| `tcp::time_wait_is_held_for_2msl_and_then_returns_the_port` | Failed 58 of 234 completed full-suite boots at `9ebbe2e`; 505 of 505 passed after commit `1735b2c` | 8 boots in parallel, QEMU TCG, 1 GiB, 2 CPUs | Commit `1735b2c`: the reap takes an injected clock, so a timer tick between the FIN and the reap no longer ends the 2MSL hold a tick early |
| In-kernel tests registered | 573 `register_test` call sites (563 at `9ebbe2e`) | commit `c055594` | `grep -rn "register_test(" kernel/seal-os/src`, less the definition |
| QEMU boot milestones | 25 / 25 | q35, OVMF, AHCI disk, no NIC, 4 GiB | CI run 36165748105 |

### ML services, from QEMU serial proof lines

| Measurement | Value | Condition | Provenance |
|---|---|---|---|
| `foliation`, boot trace | 952 bp; Belady 952, random 619, LRU 0; locality-only null 476 | 30 requests, 1,680 tokens; 24-block pool, 8 tokens per block. The hot prefix returns only after 31 other blocks, more than the pool holds, so LRU's 0 is a property of the trace (a `const` assertion states the construction) | CI run 36165748105; locality null and assertion: commits `264235c`, `0ab2377`, local QEMU |
| `foliation` against random, boot trace | Wins on 32 / 32 seeds; random spans 238 to 857 bp | Same trace | CI run 36165748105 |
| `foliation`, chat trace | 5,284 bp; Belady 8,143, LRU 8,068, locality-only null 6,818, random 7,026 to 7,443; beats random on 0 / 32 seeds | 16 conversations, 4 live at a time, 6 turns each, every turn resending a shared 2-block system prompt and the conversation so far; 96 requests, 528 descents; same pool | Commit `0ab2377`, local QEMU; the random range from a host replay of the same module |
| `foliation`, chat trace with 16 live conversations | 5,113 bp; LRU 3,731, Belady 5,378; beats random on 32 / 32 seeds | Mutation build (`CHAT_LIVE` 4 to 16), not the shipped proof | Commit `0ab2377`, one build |
| `foliation` sharing and safety | 20 shared descents, 81,920 bytes saved; 190 frames backed and 190 freed, 0 failed; 0 referenced evictions, 0 collapse violations | Boot trace; every replay of either trace must also show 0 referenced evictions and 0 collapse violations | CI run 36165748105; replays: commit `0ab2377` |
| `stratum` classification | 7 / 7 correct: underfit, wellfit, overfit, collapsing, negative control, monotone line, monotone exponential | Synthetic streams of 128 steps, window 64, $\kappa = 1.68$ | CI run 36165748105 |
| `stratum` negative control | Detector: not flagged. Gap-threshold baseline: flagged | Healthy run with an irreducible validation gap | CI run 36165748105 |
| `stratum` stream size | 4,792 bytes per stream, bounded over a 4,096-step stream | | CI run 36165748105 |

Which policy wins follows whether reuse distance exceeds the pool, not the name of the request shape: the foliation ranking wins where recency is adversarial and loses where reuse follows recency. The system-call-facing cache therefore defaults to LRU, not the foliation ranking (`ml_engine/foliation.rs:1407-1411`); the boot proof selects each policy explicitly. Since commit `40d3568`, `--check-kv-policy` requires the locality-null and chat-trace fields and refuses any policy whose hit rate exceeds Belady on either trace; margins between policies are recorded, not gated. Reproduce: boot as in Quick Start, then `--check-kv-policy /tmp/seal-os.log` and `--check-mlfit-proof /tmp/seal-os.log`.

### Certify-or-refuse, before and after

| Quantity | Before | After | Provenance |
|---|---|---|---|
| Nearest centroid on S² | 1,215 of 5,000 seeded queries wrong, reported hit rate 1.0000 | 0 wrong; 4,909 certified in the 3×3 block, 91 full scans, reported hit rate 0.9818 | Project page, commits `cdb4a4e` and `0f040e0` (stated, not re-measured). `cargo test -p aether-core --test house_tss_grid_locate` asserts 0 wrong over 25,000 queries at K = 2, 8, 20, 64 and 200. |
| β₀ at scale | Two points at 0.5 ± 1e-9 gave two different integers | Both refused; 500 seeded clouds agree with all-pairs union-find | Project page, commit `8678c97`. `cargo test -p aether-core --test certified_betti` |
| Attention top-k | $k_0 = [10^{17}, 1, -10^{17}]$, $q = [1,1,1]$: float score 0.0 against exact 1, and key 1 (0.5) taken over key 0 | Row widened | Project page, commits `afd0969`, `eb4af16`. `cargo test -p aether-core --test attention_contracts` |
| Loop score of a monotone staircase | 0.969, verdict Overfit | Certified 0 before any complex is built | `cargo test -p aether-core --test trajectory_shape` (`monotone_staircase_scores_no_fold`) |
| T4 in `epsilon-os` | Certified at $\Delta t = 1$ while the governor ticks at 0.01 | Refused at $\Delta t = 0.01$, margin 5.01 | Project page, commits `c5b5853`, `6c450d4`, `2605f70`. `cargo test -p epsilon-os` (`test_verify_theorems_pass_except_uncertified_runtime_t4`); commit `dcc35b6` adds `test_t4_certificate_holds_on_the_loop_store_runs`, which fails if the gate certifies a loop that does not settle |
| T4 at the seal-os boot | `[THEOREM] T4/AGCR VERIFIED`, evaluated at $\Delta t = 1$ while every runtime caller passes 0.01 | `[THEOREM] T4/AGCR NOT CERTIFIED: alpha+beta/dt=5.01 >= 1 at dt=0.01`; `9 of 10 theorems VERIFIED` | Commit `3c14df0`: in-kernel `kernel_foundation::t4_gate_matches_runtime_governor_dt`, 563 / 564 before and 564 / 564 after, local QEMU; `seal-mkimage` tests 80 / 80 |

Further certify-or-refuse results from the same round were ported from related repositories and are listed with their evidence in the next section: an ε-edge certificate for Rips β₁, a rounding-radius certificate for centroid separation, a refusal of non-finite attention scores, and an injective slot table for the 8-cell indices.

### Filesystem, GPU and security lines

Values are from CI run 36165748105 unless a later commit is named in the row.

| Measurement | Value |
|---|---|
| FAT16 against ext2 parity | 19 operations each; 4 files and 4,388 bytes equal byte for byte; 28 / 28 stat fields; 8 / 8 error cases; 17 expected divergences (mode, mtime, directory size, case, 8.3 names, cross-directory); negative control detected |
| GPU | CPU fallback only: 3 / 3 kernels agree with a CPU recompute; `hardware_dispatch=0`. One of four kernels has GFX9 machine code (96 bytes; 24 / 24 words round-trip, 17 / 17 instructions decode); no AMD GPU was present |
| KASLR | 30 bits (8 kernel-alias, 22 heap-window) from RDRAND; the image base is not randomised (firmware base `0x140000000`) |
| `unsafe` census | 624 blocks in 84 files; 12 carry a safety comment, 612 do not. At commit `b3cf934` the audit fixture reads 627 blocks, 611 without one (`kernel/seal-os/tests/unsafe-audit.fixture`) |
| W^X | 4,311 of 4,311 scanned kernel-alias pages writable and executable; not enforced. Since commit `b3cf934`: `wx=1 wx_violations=0 wx_pages_scanned=24004 wx_scope=kernel-root` under `tests/linux_parity/chase_boot.sh wx` (local QEMU, stated in the commit), with `.text` RX and every other kernel page NX |
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

The comparison table in [docs/BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) is not a blanket victory claim. Seal OS only claims a win over Ubuntu for a row after the same-machine benchmark exists, bound by `--check-current-benchmark-proof`; the native Ubuntu 26.04 allocator job was skipped in run 36165748105, so every row is raw Ubuntu artifact pending. Where Seal OS must still prove superiority: every row of that plan, since none has been measured on the same machine.

## Results imported from related work

Techniques ported from the author's other repositories and from upstream pull requests, each with the failing test that showed the defect and the passing state after the port. Rows that landed on 2026-09-27 are quoted from their commit messages; the `aether-core` and `epsilon-os` totals at `c055594` are in the test table above.

| Source | Result ported | Site in this repository | Evidence, RED to GREEN | Landed |
|---|---|---|---|---|
| github.com/teerthsharma/planimeter | Gap rule: a count is certified only when no merge height lies in a band around the scale | `aether-core/src/certified_betti.rs`, `certified_beta0` | A chord at 0.5 ± 1e-9 returned two different integers; both are refused now, and 500 seeded clouds agree with all-pairs union-find (commit `8678c97`, project page) | Before 2026-09-27 |
| github.com/teerthsharma/separatrix | Error-interval separation test for a decision boundary | `aether-core/src/attention.rs`, `certified_top_k` | Budget-1 top-k took key 1 (score 0.5) over key 0 (exact score 1); the row now widens (commits `afd0969`, `eb4af16`, project page) | Before 2026-09-27 |
| github.com/teerthsharma/cleave | Union-find join "younger dies, elder absorbs" (`persist` module, line 181) | `aether-core/src/ml/clustering.rs`, `cut_tree` | Merge ids n+m were never resolved, so a merge joining an already-merged cluster was dropped. RED: `k = 1 is one cluster left: [0, 0, 1] right: [0, 0, 0]`. GREEN: 433 / 433 aether-core tests | Commit `14a3bb1`, 2026-09-27 |
| github.com/teerthsharma/cleave | Same join | `epsilon-os/src/manifold_fs.rs`, `check_entropy_and_merge` | After an entropy merge, new files were still placed in the emptied cell. RED: `same content, same cell left: 2 right: 4`. GREEN: 48 / 48 epsilon-os tests | Commit `82e034f`, 2026-09-27 |
| github.com/teerthsharma/resolvent | Refuse non-finite logits before the softmax, because the row normaliser is positive only when every logit is finite (`ceqjepa/operator` module, line 194) | `aether-core/src/scheduled.rs`, `scheduled_attention` | Finite `q = [1e200]`, `k = [-1e200]` overflow to a score of −∞, and the row read 0/0: `Ok([NaN])`. RED: 18 / 19 `scheduled_attention` tests. GREEN: `ScheduleError::NonFiniteScore { row: 0, col: 0 }`; 433 / 433 aether-core tests | Commit `c26d88a`, 2026-09-27 |
| github.com/teerthsharma/resolvent | Same refusal, extended to the reference kernels | `aether-core/src/attention.rs`, `sparse_attention`; `scheduled::dense_masked_attention` | The dense reference answered `Ok([NaN])` on the same `q` and `k` while `scheduled_attention` refused them. RED: `[NaN]` from both. GREEN: both return `NonFiniteScore`; 38 / 38 and 19 / 19 in the two test binaries; 573 passed across `aether-core`, `aether_verified` and `epsilon-os`. The return type becomes a `Result`, a breaking change for any caller outside the tree | Commit `af416fa`, 2026-09-27 |
| github.com/teerthsharma/sigmoid | Chebyshev keep rule: σ is taken from the scores being judged, and nothing is pruned when they have no spread (`telemetry` module, lines 92-93) | `aether-core/src/memory.rs`, `ManifoldHeap::regulate_entropy` | 64 unmarked objects at liveness 1.0 were all pruned in one pass against the ceiling $n/k^2 = 16$ at $k = 2$ that `AetherVerified.Chebyshev` proves. RED: `uniform, pass 0: pruned 64 of 64`. GREEN: 434 / 434 aether-core tests; mutants that judge the decayed value, use a one-pass variance or drop the σ guard each fail | Commit `f20f6b2`, 2026-09-27 |
| github.com/teerthsharma/separatrix | Threshold rule: decide `d < ε` only where the distance's error interval clears ε (`api` module, line 365), here with a Higham radius | `aether-core/src/manifold.rs`, `SparseAttentionGraph::rips_betti_1` | The unit square at ε = fl(√2): the exact Rips complex is K4 with β₁ = 0, while the rounded diagonal equals ε and the function returned `Ok(1)`. RED: 1 / 2. GREEN: 2 / 2, `PersistenceError::UndecidedEdge` naming pair (0, 2); controls at ε = 1.2 and 1.5 still return 1 and 0; 434 / 434 aether-core tests | Commit `82e4a8f`, 2026-09-27 |
| github.com/teerthsharma/separatrix | Threshold trit: certify a pair only where $d - R > \theta_{\min}$, with $R$ the rounding radius of the haversine distance | `aether-core/src/tss.rs` and `aether-verified/src/aether_tss.rs`, `verify_separation` | Against `theta_min - 1e-6`, a pair 0.4999995 apart at θ_min = 0.5, a pair with a NaN coordinate, and two equator points $\pi - 10^{-9}$ apart at θ_min = $\pi - 5 \cdot 10^{-10}$ were all certified. RED: 0 / 1 in each crate. GREEN: 1 / 1 each; 572 passed across `aether-core`, `aether_verified` and `epsilon-os` | Commit `2014ddb`, 2026-09-27 |
| github.com/teerthsharma/branchcut | Theorem 1: a map meant to be injective onto m values from n inputs misassigns at least n − m (`partition` module, line 197), evaluated as n − #distinct(locate(centroid_k)) = 0 | `aether-core/src/tss.rs`, `CUBE_CENTROIDS`, used by the seal-os scheduler, compositor and firewall | The {0, π/2, π}² lattice put slots 0, 3 and 6 on the north pole and 2 and 5 on the south: `locate(centroid[k])` = [0, 1, 2, 0, 4, 5, 0, 7], so slots 3 and 6 were unreachable and firewall rules on zones 3 or 6 could never match. RED: 0 / 2. GREEN: 2 / 2; in-kernel 563 / 563 | Commit `87d7b10`, 2026-09-27 |
| github.com/teerthsharma/branchcut | Same certificate, for the router | `kernel/seal-os/src/net/topological.rs`, router index and `route_lookup` | Built from the same lattice, the router answered [0, 0, 0, 0, 1, 4, 7, 2] at the eight cube centroids, 5 distinct slots, so no route was ever filed in cell 3 or 6; `route_lookup` also searched only the query's own cell and returned a route 0.75 rad away over one 0.08 rad away across the boundary. RED: both in-kernel tests fail. GREEN: [0 … 7], the nearest route over every cell; in-kernel 570 / 570 | Commit `9702061`, 2026-09-27 |
| triton-lang/kernels#22, a topology-derived sparse-attention schedule | Sink plus local-window scaffold, used as a null: rank free faces by (negative depth, last use), the foliation order without its entrant count | `kernel/seal-os/src/ml_engine/foliation.rs`, `Policy::Locality` | No replay separated the entrant term from the depth term. RED: `foliation::locality_null_is_measured` failed, 563 / 565. GREEN: `hit_bp_locality=476` against foliation's 952 on the boot trace, so removing the entrant count halves the hit rate there | Commit `264235c`, 2026-09-27 |
| NVIDIA/NeMo-Relay#481 | Request shape: reuse keyed on a stable scaffold under varying turns, a shared system prompt plus the conversation so far | `kernel/seal-os/src/ml_engine/foliation.rs`, `build_chat_trace` | The proof replayed only a trace LRU loses by construction. RED: `foliation::proof_replays_a_recency_trace` failed, 563 / 565. GREEN: foliation 5,284 bp against LRU 8,068 and Belady 8,143, and beats random on 0 / 32 seeds | Commit `0ab2377`, 2026-09-27 |

Seven further changes landed in the same round and are not imports. Kernel W^X is enforced: `.text` and the AP trampoline code page are RX and every other kernel page NX, the `[SECURITY-FEATURES]` probe walks every present leaf from the kernel root, and `tests/linux_parity/chase_boot.sh wx` turns from RED (4,310 of 4,310 pages W+X) to GREEN (`wx_violations=0` of 24,004 scanned); the AP bring-up defects on the same path (two far-jump pointers, a GDTR write over the trampoline's `call rax`, and `ltr` against the temporary GDT) are fixed; in-kernel 564 / 564 (commit `b3cf934`, local QEMU). The TCP 2MSL test no longer fails when a timer tick lands between the FIN and the reap (commit `1735b2c`, results in the test table). `manifold_acl`'s `hash_path` takes the first 8 bytes of SHA-256: the old polynomial hash sent a 1,024-symbol Thue–Morse path and its complement to one value, so the second access skipped the anomaly check (commit `e887a79`; in-kernel 570 / 570). `fs/fat.rs` `write_fat_entry` writes every live FAT copy and honours the FAT32 ExtFlags mirroring field: two new in-kernel tests were RED (563 / 565) and the suite is 565 / 565 under QEMU after the fix (commit `1c24631`). The seal-os boot gate refuses T4 at the runtime step (commit `3c14df0`, above). `epsilon-os` gained a guard test that a T4 certificate must be backed by the loop settling; it passes while the gate refuses, and fails under a `GOVERNOR_DT = 1.0` mutant, 45 / 48 (commit `dcc35b6`). `seal-mkimage --check-kv-policy` now requires the locality-null and chat-trace fields, so a kernel that stopped reporting either comparison fails the boot gate (commit `40d3568`; `seal-mkimage` tests 80 / 80).

## Build requirements

| Component | Requirement | Where it is set |
|---|---|---|
| Host crates | Rust stable. `rust-version = "1.85"` is declared by `aether-link` and `ubuntu-alloc-bench`; CI builds with 1.98.1 and does not test 1.85. | `rust-toolchain.toml`, `.github/workflows/ci.yml` |
| Kernel | Rust nightly with `rust-src` and `llvm-tools-preview` (CI: 1.100.0-nightly f7575a9da, 2026-09-24). Unstable features: `abi_x86_interrupt`, `build-std`. | `kernel/seal-os/rust-toolchain.toml`, `kernel/seal-os/.cargo/config.toml` |
| Machine | x86_64 with long mode and UEFI firmware. Boots only as a UEFI application. RDRAND supplies KASLR entropy; CI enables it with `-cpu qemu64,+rdrand`. | `boot/uefi_entry.rs`, `security/kaslr.rs` |
| Emulator | `qemu-system-x86_64` and OVMF (CI installs `qemu-system-x86 ovmf socat` on Ubuntu). Proven configuration: q35, AHCI disk, 4 GiB, no NIC. | `.github/workflows/ci.yml` |
| Display | 1024×768 framebuffer for the desktop; the serial console carries every proof line. | `graphics/`, checked by the `[GFX] desktop-proof` line |
| Proofs (optional) | Lean 4 v4.7.0 with mathlib v4.7.0; `lake build` in `kernel/aether/aether-verified/lean`. | `lean-toolchain` |

## Direction

**Goal.** Seal OS boots under any Linux distribution as that distribution's kernel. The distribution's userland runs unmodified, its bootloader and initrd tooling work unchanged, and Linux device drivers are usable; components Seal OS does not implement natively are brought in as ports of existing open-source kernel code. This is an owner decision, recorded as the accepted plan in [docs/design/LINUX-REPLACEMENT.md](design/LINUX-REPLACEMENT.md) (commit `4c5abfd`); it supersedes the earlier policy that rejected POSIX, Linux, libc and GRUB compatibility.

**Starting point.** Every finding in the plan's starting-point table is bound to a test that fails at `9ebbe2e`: 25 host-runner test functions under [tests/linux_parity/](../tests/linux_parity/), of which 21 fail there and 3 controls and 1 cost measurement pass, and the QEMU modes of `chase_boot.sh` and `cameron_qemu_milestone.sh` (commits `811b82e`, `3c26040`, `91f857e`, `c2d4ba3`). They show that no ring-3 instruction has ever executed; that syscall entry stores onto the caller's stack; that 1 of the 49 distinct syscalls made by five Ubuntu programs reaches the same call in Seal dispatch; that the ELF loader builds no auxiliary vector; that the image carries no Linux boot protocol; that the ext2 driver mounts a distribution's ext4 volume as ext2; and that 4,310 of 4,310 scanned kernel pages are writable and executable, the one finding since turned GREEN (commit `b3cf934`). An audit pass re-ran the findings: of 105 claims, 4 were struck, and none of the four is cited in the design document (stated by the audit pass, not re-measured here; commit `e2f1fb9` replaced the one the table had cited).

**Decisions.**

- **D1.** The Linux x86_64 syscall ABI becomes the one native ABI. Seal-specific functions move to ioctls on `/dev/seal` and attributes under `/sys/kernel/seal/`, not private syscall numbers; every unimplemented call returns `-ENOSYS`.
- **D2.** The kernel boots and installs like a Linux kernel: boot-protocol setup header and EFI stub, command line, initramfs, `uname -r` reporting `<LTS>-seal`, and a `seal-kernel` package installed by the distribution's own tools.
- **D3.** Linux drivers run unmodified inside LKL driver servers, one user-mode process per server confined by the IOMMU; the native spec drivers stay in the kernel.
- **D4.** Filesystems refuse unknown ext2 features first, then implement ext4 with jbd2, then pass a crash-consistency gate in which a stock Linux kernel replays the journal.
- **D5.** The geometric subsystems keep running under Linux semantics: Linux permissions are authoritative, `manifold_acl` becomes an audit-only restriction layer, and every theorem line reports certified, refused or not checked.
- **D6.** Progress is the Linux Test Project `syscalls` pass count plus a negative security suite, and every milestone gate executes in QEMU; source-inspection tests are pre-checks only.
- **D7.** Components not built natively are ported under `ports/`, pinned by hash and licence-gated; a native implementation replaces a port only by passing the same gate.

**Milestones.** Each gate is a test that fails at `9ebbe2e`. A box in [FUTURE_PLAN.md, Phase 0](../FUTURE_PLAN.md#phase-0-linux-kernel-replacement) is ticked only when its gate passes, with the passing run cited.

| # | Scope | Gate | Status at `c055594` |
|---|---|---|---|
| M0 | Substrate and safety: ring 3 executes; syscall entry with `swapgs` and a kernel stack; user faults kill the process; user-copy fixups; kernel W^X; ext2 feature refusal; package-removal privilege check; lock order; theorem lines from live state; CI green; `ports/` | QEMU: `chase_boot.sh usermode-seal`, `cameron_qemu_milestone.sh ring3-seal`, `chase_boot.sh wx`, `chase_boot.sh ext4`; `write(fd, 0x1000, 1)` returns `-EFAULT` | Open; 0 of 11 items ticked. Landed: kernel W^X, with `chase_boot.sh wx` passing locally (`b3cf934`); the boot T4 refusal (`3c14df0`; the other lines still come from fixed inputs); and `ports/`, [PORTING.md](../PORTING.md) and the rewritten [CONTRIBUTING.md](../CONTRIBUTING.md), with the port licence gate at 12 passed (`03ba455`) |
| M1 | Process model: per-process address spaces with VMAs, page cache, per-process fd tables, wait queues, SIGSEGV delivery | One boot-executed test per item | Open; gate not yet written |
| M2 | Linux ABI core: D1 renumbering, six-argument dispatch, auxv, TLS, `execve`, `futex`, `clone` threads, `/dev/seal` | A static glibc `hello` prints on serial; static busybox `sh` runs a script | Open; `cameron_qemu_milestone.sh ring3-glibc` exists and fails, the busybox gate is not yet written |
| M3 | Linux boot: setup header, EFI stub, command line, initramfs, `seal-kernel` package layout | GRUB loads Seal OS as `linux` with an Alpine initramfs and reaches a busybox prompt | Open; gate not yet written |
| M4 | Dynamic userland without systemd | Alpine (musl, OpenRC) boots to a login prompt; LTP `syscalls` count recorded | Open; gate not yet written |
| M5 | ext4 read and write with jbd2 | The D4 crash-consistency gate | Open; gate not yet written |
| M6 | systemd distributions: cgroup2, netlink, device model with uevents, namespaces, seccomp on Linux numbers | Debian, Ubuntu, Fedora and Arch cloud images install `seal-kernel` with their own tools and boot to login | Open; gate not yet written |
| M7 | Hardware: ACPICA port, ECAM, MSI/MSI-X, IOMMU, LKL driver server | An unmodified Linux driver in a driver server passes its QEMU device model behind a virtual IOMMU | Open; ACPICA recorded as a planned port, its gate not yet written |
| M8 | Bare metal | One reference machine boots a distribution to login | Open; no machine chosen before M7 passes |

Update at `a50b8d6`: four M0 items now pass their gates and are ticked in [FUTURE_PLAN.md, Phase 0](../FUTURE_PLAN.md#phase-0-linux-kernel-replacement): kernel W^X (commit `b3cf934`), ext2 feature refusal (commit `d42e815`; `chase_boot.sh ext4` refuses a distribution ext4 volume and leaves it byte-identical), the privilege check on package removal (commit `667a1c8`; in-kernel 575 / 575), and `ports/` with PORTING.md and CONTRIBUTING.md (commit `03ba455`). No milestone has passed.

**Reversibility.** Every milestone lands behind its own gate, and the existing Seal image keeps booting. A distribution keeps its Linux kernel as the default boot entry until M6 passes on that distribution. No write to a foreign filesystem is enabled before the D4 crash-consistency gate passes. Contributions start from [CONTRIBUTING.md](../CONTRIBUTING.md) (a RED test first; theorem lines certified, refused or not checked) and, for ports, [PORTING.md](../PORTING.md) and [ports/README.md](../ports/README.md).

## Limits

Items marked "code reading" were found by reading the source at `9ebbe2e` on 2026-09-27 and have not yet been reproduced at runtime; their line references are at `c055594`. Where a later commit on `main` changed a finding, the item carries a dated update.

1. **Syscall entry is unsafe to use from ring 3 (code reading).** `syscall` does not change `rsp`, and `syscall_entry` pushes the register frame and calls into Rust on that user-controlled stack: there is no switch to a kernel stack and no `swapgs` (`process/userspace.rs:129-166`). IA32_FMASK = 0x200 leaves TF, DF and AC set in ring 0. The kernel never sets `EFER.SCE`, so `syscall` from ring 3 works only if firmware left it set; the `[SECURITY-FEATURES]` line of run 36165748105 reads `efer=0xd00`, with SCE clear. The source pre-checks `test_cameron_syscall_entry` and `test_foreman_abi_substrate` under `tests/linux_parity/` pin these findings (host runner).
2. **No task is ever scheduled (code reading).** `PerCpu::current_task` is assigned only inside `schedule()` (`process/scheduler.rs:534`), and every non-test caller of `schedule()` returns early while it is null (`scheduler.rs:1382`, `1396`; `cpu/smp.rs:170-176`); the comment at `scheduler.rs:1486-1491` states the same. The three tasks spawned at boot never run.
3. **No user space runs.** `/bin/init` is absent from the image, and the kernel falls back to its in-kernel desktop (`[execve] '/bin/init' not found` in run 36165748105). With a `/bin/init` present on an ext2 root, whether a static Linux ELF, a static glibc program or the kernel's own 208-byte `EMERGENCY_SHELL_ELF`, boot stops after `[execve] Loading '/bin/init'` and no user-mode instruction executes (`chase_boot.sh usermode` and `usermode-seal`, commit `3c26040`; `cameron_qemu_milestone.sh ring3-seal`, `ring3-linux` and `ring3-glibc`, commit `91f857e`). The ELF loader builds no argc, argv, envp or auxiliary vector and applies only `R_X86_64_RELATIVE` relocations, skipping every other type (`process/elf.rs:716`). `SYS_WAITPID` returns its first argument without waiting (`syscall/table.rs:1031`, code reading). All applications are kernel code.
4. **T4 is refused at boot, and no step size earns it.** The unearned boot certificate this item used to report is fixed: since commit `3c14df0` the boot gate evaluates the gain margin at the runtime step and prints `[THEOREM] T4/AGCR NOT CERTIFIED: alpha+beta/dt=5.01 >= 1 at dt=0.01`, and `--check-theorem-log` rejects a `T4/AGCR VERIFIED` line while the margin is at least 1. The governor itself remains the limit: (9) treats the plant as unit gain, while the loop the code runs has a gain of about $10^6$ and 2-cycles between $\varepsilon = 0.001$ and $10$ at $\Delta t = 0.01$, $0.0506$ and $1$ alike, so earning T4 requires a redesign (section 5). The ManifoldFS status, system calls 100 and 102, and the Aether-Lang theorem views still print `FAILED`, not `NOT CERTIFIED`, for T4 (commit `3c14df0`, stated in its message). The other nine lines are still computed from fixed boot constants rather than the running kernel's state; of those, T2 holds at runtime inputs, and T1 at four of the five indices that consume it.
5. **The kernel's own unit tests are not run by CI.** `kernel/seal-os` is excluded from the workspace, and the Kernel Tests workflow runs only after a fully green CI run; its last 100 runs were skipped, and the last one to execute passed 514 / 514 on 2026-08-11.
6. **CI is red, and some milestones are weak.** The QEMU job of run 36165748105 fails at the language-hygiene gate, on line 7930 of the previous README and on `scripts/ci_parity.sh:163-164`; the gates after it in that job did not run. This README and `docs/` pass the gate; `scripts/ci_parity.sh` still fails it. Two of the 25 milestones are string matches that prove little: "Syscalls verified" matches `[BOOT] SYSCALL/SYSRET MSRs programmed`, and "Scheduler started" matches any line containing `Scheduler`.
7. **The ML services have never seen a real model.** Every `stratum` and `foliation` number comes from a synthetic fixture. `FitAction` is advisory and enforced nowhere (`ml_engine/stratum.rs:170-178`). Which KV policy wins depends on the trace. On the boot trace, where the hot prefix returns only after 31 other blocks against a 24-block pool, foliation reaches Belady's 952 bp and LRU scores 0. On the chat trace, where reuse follows recency, foliation loses to LRU (5,284 bp against 8,068), to the locality-only null (6,818) and to all 32 random seeds; with 16 live conversations instead of 4 the chat result reverses again (5,113 against 3,731, one mutation build) (commits `264235c`, `0ab2377`). The sign follows whether reuse distance exceeds the pool, both traces are synthetic, and both run at one pool size. The band for $\kappa$ in (6) is proved only for a symmetric fold; an asymmetric fold closes later. Whether kernel placement of the KV cache buys anything over user-space PagedAttention has not been measured.
8. **Hardware coverage is one QEMU configuration.** In CI, AHCI works and NVMe, HDA, xHCI and every NIC are reported absent. `virtio_blk::init` has no caller, so the same image attached as virtio-blk boots and falls back to ramfs (`cameron_qemu_milestone.sh virtio-root`, commit `91f857e`). There is no driver binding framework, no PCIe ECAM and no MSI; WiFi and Bluetooth are PCI probes only. The GPU path has executed only on the CPU fallback.
9. **Filesystems and networking are partial.** There is no ext4, and the ext2 driver never reads `s_feature_incompat` (`fs/ext2.rs:308`): it mounts a distribution's `mkfs.ext4` volume as ext2 (`chase_boot.sh ext4`, commit `3c26040`, where the volume's bytes stayed identical) and then attempts to create `/swap.topo` on it (design document, starting point). FAT is not mounted in the VFS, and on this branch FAT32 reads its root cluster from byte offset 40 instead of 44, so a FAT32 root walk fails; the fix, commit `5534767`, is on `main` and not merged here. (Update at `a50b8d6`: both findings are fixed on `main`. ext2 refuses unknown INCOMPAT features and mounts unknown RO_COMPAT features read-only, and `chase_boot.sh ext4` leaves a distribution ext4 volume byte-identical, commit `d42e815`; FAT32 reads its root cluster from offset 44, commit `5534767`.) sysfs exposes only PCI devices. TLS accepts Ed25519 certificates only, so a server presenting an RSA or ECDSA certificate is refused, and no socket system call exposes the network stack. The doc-claim contract in `kernel/seal-mkimage/src/main.rs` (`check_doc_claim_contract_text`) still requires this README to contain the phrases "Minimal TLS 1.3 PSK record path" and "no X.509/PKI/ECDHE gate yet". The second is out of date: the `[TLS]` line of run 36165748105 reports `x509=1 chain_verify=1 ecdhe=1`, and only the `[BENCH] tls-encrypt` fixture (`psk_aes_128_gcm_record`) is PSK-only. Both phrases are quoted here because the gate requires them.
10. **Security mitigations are measured, not complete.** Kernel W^X holds since commit `b3cf934`, with two stated gaps: the probe classifies leaf flags per mapping, so a frame writable through one mapping and executable through another is caught only by construction, and the AP still runs with `CR0.CD` and `NW` set while `smp_start_aps` stays disabled; under `-cpu qemu64` without RDRAND the probe line still reads `result=fail` because `kaslr=0`. KASLR randomises mappings, not the image base; SMEP and SMAP were not exercised because the CI CPU model lacks them; 611 of 627 `unsafe` blocks carry no safety comment (audit fixture at `b3cf934`). `manifold_acl::check_access` runs on every lookup and exec and refuses access Linux permits: T5 denies uid 27 and above on root-owned files unless group and other bits are all set, so uid 1000 cannot read a 0644 file (`fs/vfs.rs:276`, `321`, `355`; code reading, from the design document's starting point).
11. **No performance comparison exists.** Every cycle count is from QEMU TCG. No Ubuntu or Linux comparison has been run.
12. **Formal verification covers side lemmas.** The Lean files prove algebraic facts about constants and bounds; three theorem statements and one pruning bound are `True` placeholders, and no Lean statement is connected to kernel code by refinement.
