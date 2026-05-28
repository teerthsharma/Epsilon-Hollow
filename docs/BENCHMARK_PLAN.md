# Seal OS vs Ubuntu Benchmark Plan

Seal OS aims to surpass Ubuntu for selected HFT/ML workloads. That claim is not
global and not automatic. It becomes true only for benchmark rows where fresh
Seal OS measurements beat fresh Ubuntu measurements under the same constraints.

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

Use Ubuntu 24.04 LTS unless a newer baseline is explicitly selected. Record:

```bash
lsb_release -a
uname -a
lscpu
free -h
lsblk
```

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
[BOOT] Seal OS desktop ready.
```

Ubuntu endpoint:

```text
display manager or graphical.target ready
```

Output:

- milliseconds from VM start to endpoint
- screenshot if GUI is enabled
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

Ubuntu target:

- `perf sched` or cyclictest wake latency

Output:

- p50, p99, max wake latency
- CPU count
- idle/load condition

### 5. ManifoldFS Teleport vs Byte Copy

Seal OS:

- same-filesystem `teleport`
- move 1 KB, 1 MB, 100 MB, 1 GB logical files where image size allows

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

### 7. Network Stack

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

### 8. Aether-Lang Startup and Command Latency

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

## Result Table Format

| Workload | Seal OS result | Ubuntu result | Winner | Evidence |
| --- | --- | --- | --- | --- |
| Boot to theorem gate | measured value | measured value | Seal/Ubuntu/Tie | log path |
| Boot to desktop | measured value | measured value | Seal/Ubuntu/Tie | log path |
| Syscall latency | measured value | measured value | Seal/Ubuntu/Tie | report path |

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
4. Capture serial log to theorem gate and desktop-ready sentinel.
5. Boot Ubuntu 24.04 with same vCPU/RAM/disk class.
6. Capture boot-to-target timings.
7. Publish raw logs and summary table.
