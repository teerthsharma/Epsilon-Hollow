# Porting upstream kernel code to Seal OS

Seal OS brings in existing open-source kernel code for components it does not implement natively (D7 of [`docs/design/LINUX-REPLACEMENT.md`](docs/design/LINUX-REPLACEMENT.md)). Each port lives in [`ports/`](ports/README.md) as a `PORT.toml` that pins the upstream source by commit and hash, records the licence Seal takes it under, where it runs, how many patches Seal carries, and the QEMU gate it must pass. This file states what may be ported where, how a port is pinned and gated, and when a native implementation replaces it. The field reference is [`ports/README.md`](ports/README.md); the licence gate is [`tests/ports/test_port_licenses.py`](tests/ports/test_port_licenses.py).

## 1. Placement and licence

A port has one of two placements.

- **`kernel`**: compiled into the Seal OS kernel image, which ships under MIT ([`LICENSE`](LICENSE)).
- **`driver-server`**: a separate user-mode program (D3), confined to its own process and IOMMU domain, reaching the kernel only over virtio-style rings. It is its own program under its own licence.

| Upstream licence | `kernel` | `driver-server` |
|---|---|---|
| In deny.toml `[licenses].allow` | yes | yes |
| GPL-2.0-only, GPL-2.0-or-later | no | yes |
| BSD-4-Clause and its variants | no | no |
| Any other licence | no | reviewed per port; the gate does not decide it |

**Kernel placement** takes only licences in the allow list of [`deny.toml`](deny.toml) (lines 25-40), the same list cargo-deny applies to Rust dependencies. At `4c5abfd` that list is: MIT, Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, Unicode-DFS-2016, Unicode-3.0, ISC, Zlib, MPL-2.0, CC0-1.0, BSL-1.0, CDLA-Permissive-2.0. The gate reads deny.toml when it runs, so a licence is added for ports and dependencies alike by one reviewed change to deny.toml. The crate-scoped `wav` exception for LGPL-3.0 (deny.toml line 46) does not extend to ports. MPL-2.0 is file-level copyleft: a modified MPL-2.0 file stays under MPL-2.0, and its source is published with any image that contains it.

**Driver-server placement** is the only place GPL-2.0 code runs. Linux drivers run there, unmodified, inside LKL (the Linux kernel built as a library), one process per server (D3). The source offer for a server is its pinned upstream tree, configuration and patches, published by the project. Whether GPL-2.0 server binaries ship in project images is reserved for the owner; the default is that users build them from the published source.

**BSD-4-Clause is refused in both placements.** Its advertising clause requires every advertisement that mentions features or use of the software to acknowledge the original authors. In the kernel it would attach that obligation to every downstream advertisement, which the MIT licence does not carry; in a driver server GPL-2.0 does not permit the added restriction. The gate refuses every SPDX identifier beginning `BSD-4-Clause` (including `-UC` and `-Shortened`) in `license` and in `upstream_license`.

**Dual-licensed code** records the choice explicitly: `license` names the one option Seal takes, and `upstream_license` records the whole upstream expression. An `OR` in `license` fails the gate.

## 2. Pinning

- `pinned_rev` is the full 40-hex upstream commit. A tag or branch name is not a pin.
- `sha256` is the SHA-256 of the uncompressed tar that `git archive` produces for that commit, with the port name as prefix:

```bash
git clone <upstream_url> upstream-<name>
git -C upstream-<name> archive --format=tar --prefix=<name>/ <pinned_rev> | sha256sum
```

`git archive` stamps every entry with the commit's time, so the tar is reproducible from the commit. The compressed form is not: compressed bytes depend on the compressor implementation, which is what changed the checksums of GitHub-generated archives in January 2023.

- Moving a port to a newer upstream is one commit: new `pinned_rev` and `sha256`, patches rebased, `patch_count` updated, and the gate result quoted.

## 3. Bringing a port in

Each step is its own commit.

1. **Record.** Add `ports/<name>/PORT.toml` with `status = "planned"`, empty `pinned_rev` and `sha256`, the placement, the elected licence, and a licence record in comments: which upstream files were read, at which commit. A repository-level licence summary is not enough on its own. GitHub's licence API reports one file; for ACPICA it reports BSD-3-Clause alone, while the source files are dual-licensed.
2. **Gate, RED.** Write the QEMU test at the `gate` path and run it on the current tree. It fails, for the reason the port addresses.
3. **Fetch and pin.** Set `pinned_rev` and `sha256` (section 2). Read the licence of every fetched file; a file outside the elected licence is dropped from the port or blocks it.
   - `kernel`: commit the unmodified upstream files the build uses under `ports/<name>/upstream/`. Nothing under `upstream/` is edited in place.
   - `driver-server`: nothing enters the tree. The server build fetches `pinned_rev`, checks `sha256` and applies the patches.
4. **Patch.** Seal changes go in `ports/<name>/patches/` as `git format-patch` output against `pinned_rev`, numbered in apply order (`0001-...`, `0002-...`). `patch_count` equals the number of files there.
5. **Land.** Set `status = "ported"`. The gate passes with the port in place, the licence gate passes, and the commit quotes the gate's RED result from step 2 and its GREEN result now.

Glue that connects a port to the kernel is Seal code under MIT in `kernel/seal-os/`, not part of the port. For ACPICA that is the OS services layer (`AcpiOs*`) declared in upstream `source/include/acpiosxf.h`.

## 4. Writing the gate

- It tests the property the port provides, never the port's internals, so a native implementation can be held to the same gate unchanged (section 5).
- It executes in QEMU (D6). A source-inspection check can be a pre-check, never the gate.
- It carries a control in the same run proving the VM booted and the serial log was captured, so a missing marker is a property failure and not a dead VM.
- It exits 0 when the property holds, 1 when it is violated, and 2 on a setup failure, the convention of [`tests/linux_parity/chase_boot.sh`](tests/linux_parity/chase_boot.sh).
- It is seen failing before the port lands and passing after, unmodified ([`CONTRIBUTING.md`](CONTRIBUTING.md), RED test first).
- It lives at `tests/ports/<name>_<property>.sh`, or `gate` points at an existing `tests/linux_parity/` mode when a milestone gate already covers the property.

## 5. When a native implementation supersedes a port

A native implementation replaces a port only when it passes the same gate, unmodified (D7). The replacing commit adds the native code, quotes the gate passing without the port, sets `status = "superseded"` and `superseded_by` to the native implementation's path, and deletes `upstream/` and `patches/`. `PORT.toml` stays as the record of what was ported, from where, and under which licence. A gate edited in the same change does not count: a gate change lands first, in its own commit, passing with the port in place.

## 6. Worked example: ACPICA (planned)

[`ports/acpica/PORT.toml`](ports/acpica/PORT.toml) is the first entry. It is at step 1 of section 3: nothing fetched, nothing vendored, no gate file yet.

- **Purpose.** ACPI AML evaluation at M7. The kernel parses the FADT but has no AML interpreter; the S3 and S5 sleep types are hard-coded constants (`kernel/seal-os/src/drivers/acpi/fadt.rs:125-126`).
- **Licence.** Checked on 2026-09-27 against upstream master `c71849d476bb5f97ca1867849e893368e12a69e6` (committed 2026-09-23). `gh api repos/acpica/acpica/license` reports BSD-3-Clause, from `LICENSE.BSD-3-Clause`. The repository root holds `LICENSE.BSD-3-Clause` and `LICENSE.GPL-2.0-only`. `source/include/acpi.h`, `source/include/platform/acenv.h`, `source/components/dispatcher/dsfield.c` and `source/components/executer/exfield.c` each carry `SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0-only`. So `upstream_license = "BSD-3-Clause OR GPL-2.0-only"`, and Seal takes `license = "BSD-3-Clause"`, the option D7 records FreeBSD, NetBSD, Haiku and Fuchsia as using. BSD-3-Clause is in the allow list, so `placement = "kernel"` passes the gate. Files other than those four are unverified until the step 3 scan.
- **Location.** `github.com/acpica/acpica` redirects to `github.com/open-acpica/acpica` (GitHub API `full_name`, 2026-09-27). The canonical URL is recorded, because GitHub drops a redirect when a new repository takes the old name.
- **Pin.** Empty until step 3.
- **Gate.** `tests/ports/acpica_s5.sh`, to be written at step 2. Property: under QEMU q35 the S5 sleep type comes from ACPICA's evaluation of `\_S5` in the firmware DSDT, not from the `fadt.rs` constants, and the guest powers off with it.

```toml
name = "acpica"
status = "planned"
upstream_url = "https://github.com/open-acpica/acpica"
pinned_rev = ""
sha256 = ""
license = "BSD-3-Clause"
upstream_license = "BSD-3-Clause OR GPL-2.0-only"
placement = "kernel"
patch_count = 0
gate = "tests/ports/acpica_s5.sh"
```
