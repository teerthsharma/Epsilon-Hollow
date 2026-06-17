# Seal OS vs Ubuntu Benchmark Plan

Seal OS aims to surpass Ubuntu for selected HFT/ML workloads. That claim is not
global and not automatic. It becomes true only for benchmark rows where fresh
Seal OS measurements beat fresh Ubuntu measurements under the same constraints.
I am building this as an independent researcher, so the plan is deliberately
artifact-heavy: no team-size aura, no blanket victory claim, only reproducible
rows.

## Baseline Rule

Every comparison must record:

- date and commit hash
- host CPU, RAM, disk, firmware, and OS
- hypervisor and version
- guest CPU count, RAM, disk size, disk controller, network mode
- Seal OS image path and verifier output
- Ubuntu version and kernel version
- exact commands
- raw logs
- summary table

## Initial Ubuntu Baseline

Use Ubuntu 26.04 LTS as the current baseline. On 2026-05-29 the official
Ubuntu release list names Ubuntu 26.04 LTS as the latest LTS release; Ubuntu
24.04 LTS remains a legacy comparison row only. Record:

```bash
lsb_release -a
uname -a
lscpu
free -h
lsblk
```

Capture the first allocator baseline with the Rust-only Ubuntu harness on the
same machine. Use a native Ubuntu 26.04 VM, bare-metal Ubuntu 26.04 install, or
self-hosted Ubuntu 26.04 CI runner. WSL is rejected for benchmark evidence
because its kernel and hypervisor path are not the Ubuntu VM/bare-metal baseline:

```bash
grep -E '^(ID|VERSION_ID)=' /etc/os-release
uname -r
command -v cargo && cargo --version
cargo +stable run --manifest-path tools/ubuntu-alloc-bench/Cargo.toml --release -- 64 > tools/ubuntu-alloc-bench/ubuntu-alloc.log
cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-ubuntu-benchmark-log tools/ubuntu-alloc-bench/ubuntu-alloc.log
```

The audit tool accepts this artifact only when the log says `os=ubuntu`, the
current-baseline gate says `version_id=26.04`, and `kernel=` does not contain
`microsoft` or `wsl`. Windows, macOS, WSL, WSL-missing, or unknown-host output is
useful for smoke testing the tool, but it is not Ubuntu evidence.

CI has a manual `ubuntu-alloc-baseline` lane for this proof. It only runs from
`workflow_dispatch` on a self-hosted runner labeled `ubuntu-26.04`, boots the
Seal image on that runner, captures the Ubuntu allocator artifact, compares both
logs, and uploads the manifest bundle.

## Guest Configuration

| Field | Seal OS | Ubuntu |
| --- | --- | --- |
| CPU | same vCPU count | same vCPU count |
| RAM | same RAM | same RAM |
| Disk | same virtual disk size/class | same virtual disk size/class |
| Storage | same controller where possible | same controller where possible |
| Network | same adapter class | same adapter class |
| Display | framebuffer/serial as needed | serial/console as needed |

## Workload Set

### 1. Boot To Theorem Gate

Measures how fast each OS reaches its trust gate.

Seal OS endpoint:

```text
[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths
```

Ubuntu endpoint:

```text
systemd reached basic.target or multi-user.target
```

Output:

- milliseconds from VM start to endpoint
- serial log
- boot command

### 2. Boot To Desktop Ready

Seal OS endpoint:

```text
[BOOT] Desktop proof frame blit done
[GFX] desktop-live-proof version=1 route=desktop_handle_input action=desktop_icon_launch app=Files app_id=3 events=2 handled=1 icon_hit=1 launched_app_id=3 pre_focused=<n> post_focused=<n> post_window_id=<n> window_count=12 pre_hash=<n> post_hash=<n> changed_samples=<n> vram_hash=<n> vram_changed_samples=<n> vram_matches_backbuffer=<n> blit=1 result=pass
[BOOT] Seal OS desktop ready.
```

Ubuntu endpoint:

```text
display manager or graphical.target ready
```

Output:

- milliseconds from VM start to endpoint
- screenshot proof from `qemu-proof/screen.ppm`
- `seal-mkimage --check-proof-screen qemu-proof/screen.ppm` output
- serial or journal evidence

### 3. Seal ABI Syscall Latency

Seal OS measures native Seal ABI calls:

- `getpid`
- `gettimeofday`
- `getrandom`
- `theorem_status`
- `manifold_query`

Ubuntu measures comparable Linux syscalls where they exist:

- `getpid`
- `clock_gettime`
- `getrandom`

Output:

- p50, p90, p99, p999 latency
- loop count
- timer source
- warmup count

### 4. Scheduler Wake Latency

Seal OS target:

- timer tick to runnable task selection
- scheduler yield return time
- T4 governor epsilon at selection
- current boot marker:

```text
[BENCH] scheduler-select-next selector=select_next_task mode=live_requeue clock=rdtsc iterations=64 ok=64 ready_before=<n> ready_after=<n> cells=8 priority_buckets=256 voronoi_locate_probes=8 max_cell_bitmap_tests=9 max_priority_bucket_scan=256 context_switches=0 selected_priority_max=<n> p50_cycles=<n> p95_cycles=<n> max_cycles=<n>
```

The current marker proves the bounded selector shape on the live scheduler
without switching context. End-to-end wake latency is still pending because it
must include timer interrupt delivery, runnable transition, and return to the
selected task.

### 4.1 TCP Packet Demux Fixture

Seal OS:

- listener and accepted socket share one local TCP port
- packet enters `handle_tcp_packet`
- current boot marker:

```text
[BENCH] tcp-packet-demux api=handle_tcp_packet fixture=listener_first accepted_state=established ok=1 listener_first=1 exact_flow=1 decoy_rx_bytes=0 listener_fallback=1 payload_bytes=4 rx_bytes=4 o1_index=1 index_hit=1 index_lookup_probes=<n> index_probe_bound=256 index_capacity=256 listener_index_hit=1 listener_lookup_probes=<n> listener_probe_bound=256 listener_index_capacity=256 exact_scan=0 cleanup=ok
```

The marker proves same-port packet demux chooses the exact accepted flow through
the bounded flow index before listener fallback, leaves a same-port decoy socket
empty, and routes a fresh SYN through the bounded listener-port index. It is not
a network latency benchmark; DHCP, external RX/TX, and HFT packet latency remain
separate artifacts.

### 4.2 TLS PSK Record Encrypt Fixture

Seal OS:

- a PSK-only `TlsSession` is placed in established state
- a 1024-byte record is encrypted with AES-128-GCM and then decrypted with the
  matching read key/IV
- current boot marker:

```text
[BENCH] tls-encrypt api=TlsSession::encrypt fixture=psk_aes_128_gcm_record plaintext_bytes=1024 record_bytes=1045 tag_bytes=16 decrypt_match=1 write_seq=1 read_seq=1 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> result=pass
```

The marker proves the kernel TLS record path wraps a 1024-byte payload into a
1045-byte TLS record, authenticates with a 16-byte GCM tag, advances read/write
sequence numbers, decrypts back to the source payload, and emits monotonic cycle
samples. It does not claim X.509, ECDHE, certificate validation, socket I/O, or
public HTTPS compatibility.

Ubuntu target:

- `perf sched` or cyclictest wake latency

Output:

- p50, p99, max wake latency
- CPU count
- idle/load condition

### 5. ManifoldFS Teleport vs Byte Copy

Seal OS:

- same-filesystem `teleport` over the persistent mock block-store path
- move 1 KB, 1 MB, 100 MB, 1 GB logical files where image size allows
- current boot marker:

```text
[BENCH] manifold-teleport api=teleport fs_mode=mock_block persistence=metadata_only samples=3 ok=3 same_inode=3 src_gone=3 dst_present=3 entries_min=8 entries_max=256 payload_bytes=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> ticks_max=<n> metadata_ops_max=7 persistence_bytes_per_move=0 payload_points=<n>
[BENCH] manifold-lookup api=resolve_path_with_proof fs_mode=mock_block fixture=dirhash_path_walk samples=64 ok=64 entries=64 path_depth=4 components_max=4 payload_bytes=64 dirhash_probes_total_max=<n> dirhash_probes_max=<n> dirhash_probe_bound=<n> p50_cycles=<n> p95_cycles=<n> max_cycles=<n> result=pass
```

- this marker proves bounded metadata surgery through the persistent mock
  block-store path, same-inode movement, and `persistence_bytes_per_move=0`
  for same-filesystem moves in `fs_mode=mock_block`; it is not an AHCI latency
  or any-VM proof
- the lookup marker proves 64 four-component `resolve_path_with_proof` walks
  in the same mock-block ManifoldFS fixture and requires bounded DirHash probes;
  it is not a content-similarity Voronoi search proof

Ubuntu:

- `rename` same filesystem
- `cp` plus `unlink` for byte-copy baseline

Output:

- operation latency by file size
- whether operation copies bytes or edits metadata
- filesystem type
- disk cache state

### 6. Block I/O

Seal OS:

- AHCI/NVMe read and write sector latency
- sequential and random patterns

Ubuntu:

- `fio` with comparable block sizes and queue depth

Output:

- IOPS
- bandwidth
- p99 latency
- controller type

### 7. ManifoldVRAM Data Movement

Run only when the VM or bare-metal host exposes a usable GPU path. This row is
optional for VirtualBox/Oracle smoke runs and required for GPU-native claims.

Seal OS:

- ManifoldFS descriptor teleport with VRAM-hot ManifoldPayload
- tensor batch load into ManifoldVRAM cache
- NVMe or NIC peer-DMA path where hardware supports it

Ubuntu:

- comparable GPU upload through the normal userspace stack
- CPU memory copy plus GPU upload baseline

Output:

- descriptor update latency
- bytes copied by CPU
- GPU buffer residency proof
- p50, p99, p999 transfer latency

### 8. Network Stack

Run only when a real or emulated NIC is active.

Seal OS:

- DHCP lease time
- ping/ICMP if implemented
- TCP connect and echo throughput

Ubuntu:

- `ping`
- `iperf3`
- TCP connect latency

Output:

- latency
- throughput
- packet loss
- adapter model

### 9. Aether-Lang Startup and Command Latency

Seal OS:

- `.aether` script parse and run time
- `theorem.status`
- `fs.teleport`
- `net.status`

Ubuntu:

- Rust or shell baseline for comparable tasks

Output:

- cold start
- warm start
- command p50/p99 latency

### 9. HFT Synthetic Tick-To-Action

Seal OS:

- synthetic market tick in memory
- Aether or Rust strategy evaluates
- action emitted through Seal ABI or kernel app path

Ubuntu:

- equivalent userspace strategy loop
- pinned CPU
- isolated process where possible

Output:

- tick-to-action latency p50/p99/p999
- jitter
- CPU isolation settings

### 10. ML Tensor Locality Microbench

Seal OS:

- tensor chunks allocated through TopoRAM
- access pattern follows correlated dimensions
- prefetch hit/miss proxy recorded

Ubuntu:

- same tensor access pattern in userspace
- page faults/cache misses recorded where possible

Output:

- latency per tensor slice
- throughput
- cache miss proxy
- allocation layout notes

### 11. Topological Allocator Hot Path

Seal OS:

- single-frame `alloc_frame()` via eight-cell topological free index
- hint-biased TopoRAM `alloc_frames(1, ZoneHint::Low, Some(seed))` via target
  Voronoi cell
- multi-page `alloc_frames_contiguous_in_range()` via 128 bounded candidate probes
- TopoRAM public allocation/free-side metadata repair capped at 64 pages per
  call; larger transfers must be chunked or scatter-gather
- boot markers:

```text
[BENCH] toporam-alloc iterations=64 ok=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> target_cell_hits_delta=64 target_cell_fallbacks_delta=0 low_to_high_fallbacks_delta=0 high_to_low_fallbacks_delta=0 pcie_to_high_fallbacks_delta=0 pcie_to_low_fallbacks_delta=0 free_before=<n> free_after=<n>
[BENCH] alloc-frame iterations=64 ok=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> fast_hits_delta=64 bounded_misses_delta=0 max_contiguous_probes_seen_delta=0 free_before=<n> free_after=<n>
```

- report allocation cycles, failed-probe count, free-frame conservation, and
  request size

Ubuntu:

- `mmap`/page fault path for equivalent page counts
- optional hugepage path as separate row, not mixed with normal allocation
- first same-machine artifact:

```text
[UBUNTU-BENCH] alloc-frame os=ubuntu version_id=26.04 kernel=<native-kernel> iterations=64 ok=64 bytes=4096 backend=rust-std-box-page-touch-drop clock=rdtsc p50_cycles=<n> p95_cycles=<n> max_cycles=<n>
```

- comparison gate:

```bash
cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --compare-benchmark-logs kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log tools/ubuntu-alloc-bench/ubuntu-alloc.log
cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-current-benchmark-proof kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/proof-manifest.txt tools/ubuntu-alloc-bench/ubuntu-alloc.log .
```

The second gate is the claim gate. It refuses to compare Ubuntu against an
arbitrary Seal serial log: the Seal benchmark log must be the serial artifact
inside a current VM proof manifest whose commit and dirty flag match the working
checkout.

Output:

- p50/p99 allocation latency by page count
- maximum probe count observed
- proof that latency does not scale with installed RAM size
- notes where chunked/scatter-gather I/O is required beyond the 64-page
  contiguous-run cap

## Result Table Format

| Workload | Seal OS result | Ubuntu result | Winner | Evidence |
| --- | --- | --- | --- | --- |
| Boot to theorem gate | measured value | measured value | Seal/Ubuntu/Tie | log path |
| Boot to desktop | measured value | measured value | Seal/Ubuntu/Tie | log path |
| Syscall latency | measured value | measured value | Seal/Ubuntu/Tie | report path |

## Current Benchmark Status

Seal OS now has local VM proof for theorem-gated boot, desktop-ready serial
markers, allocator O(1) proof markers, a Seal-side single-frame allocation
benchmark marker, a ManifoldFS same-inode teleport marker, a desktop compositor
soak marker, first-frame screenshot pixels, and a Rust-only Ubuntu allocator
baseline harness with a comparison gate.
Ubuntu comparison numbers are still pending, so no global Ubuntu win is claimed.
The allocator-only Ubuntu comparison becomes claimable only after
`--check-current-benchmark-proof` passes against a current Seal proof manifest
and a native Ubuntu 26.04 artifact from the same benchmark environment.

## Claim Policy

Allowed:

- "Seal OS is designed to beat Ubuntu on the HFT/ML benchmark set."
- "Seal OS beat Ubuntu on workload X in run Y."
- "Seal OS has not yet been measured for workload X."

Not allowed:

- "Seal OS is faster than Ubuntu" without workload and evidence.
- "Seal OS surpasses Ubuntu in all specs" without full benchmark table.
- "The theorem gate is formally complete" while Lean placeholders remain.

## First Milestone

The first benchmark milestone is modest and measurable:

1. Build Seal OS image.
2. Verify image with `seal-mkimage --verify`.
3. Boot Seal OS in QEMU or VirtualBox.
4. Capture serial log to theorem gate, Aether runtime proof, LAAMBA app proof,
   serial desktop pixel proof, live desktop input proof, desktop proof frame,
   desktop-ready sentinel, `[ALLOC] O(1) proof:`, `[BENCH] toporam-alloc`,
   `[BENCH] alloc-frame`, `[BENCH] slab-alloc`, `[BENCH] manifold-teleport`,
   `[BENCH] manifold-lookup`, `[BENCH] scheduler-select-next`,
   `[BENCH] tcp-packet-demux`, `[BENCH] tcp-roundtrip`,
   `[BENCH] tls-encrypt`, `[BENCH] topo-render-3d`,
   `[BENCH] tensor-render`, and
   `[GFX] desktop-proof` / `[GFX] desktop-live-proof` /
   `[GFX] desktop-soak` markers.
5. Run `seal-mkimage --check-proof-screen` against the captured `screen.ppm`.
6. Run `seal-mkimage --check-benchmark-log` and
   `seal-mkimage --check-laamba-app-proof` and
   `seal-mkimage --check-desktop-soak` against the serial log.
7. Boot Ubuntu 26.04 LTS with same vCPU/RAM/disk class.
8. Run `tools/ubuntu-alloc-bench` on Ubuntu and validate it with
   `seal-mkimage --check-ubuntu-benchmark-log`.
9. Run `seal-mkimage --compare-benchmark-logs` against the Seal OS serial log
   and Ubuntu allocator log.
10. Run `seal-mkimage --check-current-benchmark-proof` against the Seal proof
    manifest and Ubuntu allocator log so the comparison is bound to current VM
    proof provenance.
11. Capture boot-to-target timings and the remaining comparable microbenchmarks.
12. Publish raw logs and summary table.
