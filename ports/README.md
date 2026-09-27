# ports/

Existing open-source kernel code that Seal OS uses where it has no native implementation (D7 of [`docs/design/LINUX-REPLACEMENT.md`](../docs/design/LINUX-REPLACEMENT.md)). What may be ported where, and the order in which a port is brought in, is in [`PORTING.md`](../PORTING.md). This file is the layout and field reference.

## Layout

```
ports/
  README.md          this file
  <name>/            one directory per port; <name> equals PORT.toml `name`
    PORT.toml        pin, licence, placement, gate
    patches/         Seal changes as numbered `git format-patch` files, applied in order
    upstream/        kernel placement only: unmodified upstream files at pinned_rev
```

Nothing is vendored until a port is brought in. A `planned` port is a `PORT.toml` alone: no `patches/`, no `upstream/`, no fetched source. A `driver-server` port never gains `upstream/`; its build fetches `pinned_rev` and checks `sha256` ([`PORTING.md`](../PORTING.md), section 3).

## PORT.toml fields

| Field | Value | Meaning | Checked by the licence gate |
|---|---|---|---|
| `name` | string | port name | equals the directory name |
| `status` | `"planned"`, `"ported"` or `"superseded"` | planned: recorded, nothing fetched. ported: source in use. superseded: a native implementation passed the same gate | one of the three |
| `upstream_url` | URL | canonical upstream repository | present |
| `pinned_rev` | 40-hex commit, or `""` while planned | the upstream commit the port is taken from | 40 hex digits when ported |
| `sha256` | 64-hex, or `""` while planned | SHA-256 of `git archive --format=tar <pinned_rev>` | 64 hex digits when ported |
| `license` | SPDX expression without `OR` | the licence Seal takes the code under | kernel placement: every `AND` term in deny.toml `[licenses].allow`; never BSD-4-Clause |
| `upstream_license` | SPDX expression (optional) | everything upstream offers, for dual-licensed code | never BSD-4-Clause |
| `placement` | `"kernel"` or `"driver-server"` | linked into the MIT kernel image, or run as a separate user-mode program (D3) | one of the two |
| `patch_count` | integer | number of files in `patches/` | equals that count when ported |
| `gate` | repository-relative path | the QEMU test that must pass with the port in place | exists when ported |
| `superseded_by` | repository-relative path (superseded only) | the native implementation that passed the gate | not checked |

The licence gate is [`tests/ports/test_port_licenses.py`](../tests/ports/test_port_licenses.py):

```bash
python -m pytest tests/ports -q
```

## Ports

| Port | Status | Placement | Licence taken (upstream) | Gate | Milestone |
|---|---|---|---|---|---|
| [`acpica`](acpica/PORT.toml) | planned | kernel | BSD-3-Clause (BSD-3-Clause OR GPL-2.0-only) | `tests/ports/acpica_s5.sh`, not yet written | M7 |

D7 also names the LKL driver server (M7) and a NetBSD rump server (after the first LKL driver passes). Neither has a `PORT.toml` yet.
