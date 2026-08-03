# Native ML Framework Integration — Design

Status: **in progress.** This document records decisions and contracts. It is not
a claim that any of it works yet. Sections marked LANDED are backed by passing
tests; everything else is a plan.

## The constraint that shapes everything

PyTorch, TensorFlow and Triton cannot execute inside Seal OS. They require a
libc, a CPython interpreter, and a vendor userspace driver stack. That is not a
scoping preference, it is a dependency closure measured in tens of millions of
lines, and no amount of kernel work changes it.

So "native integration" is defined here as: **Seal OS executes the model, and
the frameworks become exporters and clients.** The kernel already owns the parts
that matter — `stratum` (topological fit control, syscalls 120-124) and
`foliation` (paged KV cache, syscalls 130-134). What it lacked was the ability
to run a graph at all. That is the gap being closed.

## What this closes, stated against the README's own invoice

`README.md`, "The Honest ML Limitations", enumerates ten. Two are directly in
scope here, quoted from that section:

- **#6** — "All fixtures are synthetic. Nothing has been validated against a
  real model. [...] zero of them produced by PyTorch, JAX, or anything with a
  GPU behind it." A host exporter that ships a real trained graph into the
  kernel is the first time `stratum` sees a loss curve it did not generate.
- **#7** — "`reg_scale`, `lr_scale` and `batch_scale` are numbers in a struct
  that your trainer is free to ignore [...] because nobody has written the
  client." The framework adapters are that client.

Nothing here touches limitations #1-#5 or #8-#10. Those remain exactly as
written.

## Crate topology, and why

The decisive constraint was discovered before any code was written:
`kernel/seal-os/.cargo/config.toml` pins `target = "x86_64-unknown-uefi"` with
`build-std`, so `cargo test` inside the kernel crate does not compile — it fails
with `duplicate lang item in crate core`. All 72,000 lines of kernel are
verified only by in-kernel harnesses whose serial output `seal-mkimage` parses
after a QEMU boot.

That is a packaging accident, not a design constraint. `aether-core` already
demonstrates the escape: it is a member of the **root** workspace (host-tested)
*and* a path dependency of `seal-os` (UEFI), because it is `no_std + alloc` with
a `std` feature.

New crates are therefore placed in the root workspace from birth, which makes
them host-testable by construction and avoids a refactor of working code:

| crate | responsibility | depends on |
|---|---|---|
| `kernel/seal-graph` | `.sealg` artifact format, wire codec, graph executor | `aether-core` |
| `kernel/seal-jit` | autotune memo, specialisation ladder, shrinking search budget | nothing |
| `kernel/seal-net80211` | 802.11 MAC frames, WPA2 supplicant, WPA3 SAE | RustCrypto |

`seal-jit` is deliberately generic over opaque `PlanKey`/`PlanChoice` so it does
not couple to `seal-graph`. `seal-graph` reuses `aether_core::ml::Tensor` and
its autograd rather than introducing a second tensor type.

## Transport

Decision: **serial first, TCP second, one codec.**

The wire codec is generic over a `Transport` trait and knows nothing about
either. Serial needs no new driver (`drivers/serial.rs` exists) and no network
configuration, so it produces a working result with the fewest failure surfaces
between the host and the kernel. TCP over `net/tcp.rs` follows as a second
implementation of the same trait — which also serves as the proof that the codec
is genuinely transport-independent rather than accidentally serial-shaped.

## The booster, described accurately

The request was "a self involving compiler [...] if a ML have to be trained 1000
loops every loop will be a little faster."

What is being built is a **persistent autotune memo with a monotonically
shrinking search budget and a specialisation ladder**. Repetition gets cheaper
because the system stops re-searching a space it has already measured, and
because it commits to more specialised plans as confidence accumulates. It is
not a compiler that rewrites itself, and no code or doc comment in this tree
describes it as one.

The claim is falsifiable and carries a control:

> Over N repetitions of an identical workload, cumulative cost is monotone
> non-increasing per repetition; a control run with the memo disabled shows no
> such trend.

Demotion on measured regression is required, not optional. A ladder that only
promotes will lock in a winner that has gone stale under workload drift, and
report the resulting slowdown as success.

## WiFi

QEMU emulates no 802.11 NIC. An end-to-end association therefore cannot be
proven in a VM, on this machine or any other, and no amount of kernel code
changes that either.

The deliverable is consequently the layer that *can* be verified: MAC frame
codec, WPA2 four-way handshake, WPA3 SAE message codec — as a host-tested
library gated against published IEEE 802.11-2016 Annex J and RFC 6070 vectors.
The driver state machine in `drivers/wifi.rs` is completed up to the
section-upload boundary and no further.

The existing driver's honesty is a hard invariant: `scan()` fabricates nothing,
the boot proof requires `simulation=absent`, and no change may introduce a code
path that can produce an SSID that did not come off the air.

## Round 2 — blocked until the contracts above are real

1. Kernel glue: new syscall range for graph load/run, a serial listener task, and
   the executor wired to `stratum` so a running graph is fit-controlled.
2. `bindings/seal-ml-py`: `torch.export` and TF SavedModel → `.sealg`; Triton
   kernel capture; the host side of the wire codec.
3. `drivers/wifi.rs` wired to `seal-net80211`, state machine completed to the
   section-upload boundary.

All three touch `kernel/seal-os/src/`, so they are sequenced after the parallel
crate work rather than run alongside it.
