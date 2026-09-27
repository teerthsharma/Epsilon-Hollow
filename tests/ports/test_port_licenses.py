"""Licence gate for ports/: every ports/<name>/PORT.toml is checked.

Rules (PORTING.md; decision D7 of docs/design/LINUX-REPLACEMENT.md):
- kernel placement: every term of `license` is in deny.toml [licenses].allow;
- every placement: no BSD-4-Clause variant in `license` or `upstream_license`;
- `license` names the one option Seal takes (no OR), and is not empty;
- required fields present, `status` and `placement` from the fixed sets,
  `name` equals the directory name, every directory under ports/ has a PORT.toml;
- status "ported": 40-hex pinned_rev, 64-hex sha256, patch_count equals the
  files in patches/, and the gate path exists.

Run: python -m pytest tests/ports -q
"""
import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PORTS = ROOT / "ports"
ALLOW = frozenset(tomllib.loads((ROOT / "deny.toml").read_text(encoding="utf-8"))["licenses"]["allow"])
REQUIRED = ("name", "status", "upstream_url", "pinned_rev", "sha256", "license", "placement", "patch_count", "gate")
STATUSES = ("planned", "ported", "superseded")
PLACEMENTS = ("kernel", "driver-server")


def problems(port_dir: Path, allow=ALLOW) -> list[str]:
    """Every rule the port at port_dir breaks; empty when it passes."""
    try:
        port = tomllib.loads((port_dir / "PORT.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        return [f"PORT.toml unreadable: {e}"]
    missing = [k for k in REQUIRED if k not in port]
    if missing:
        return [f"missing fields {missing}"]
    out = []
    lic = port["license"]
    if port["name"] != port_dir.name:
        out.append(f"name {port['name']!r} differs from directory {port_dir.name!r}")
    if port["status"] not in STATUSES:
        out.append(f"status {port['status']!r} not in {STATUSES}")
    if port["placement"] not in PLACEMENTS:
        out.append(f"placement {port['placement']!r} not in {PLACEMENTS}")
    if not lic.strip():
        out.append("license is empty")
    if re.search(r"\bOR\b", lic):
        out.append(f"license {lic!r} must name the one option Seal takes; put the full expression in upstream_license")
    for expr in (lic, port.get("upstream_license", "")):
        if re.search(r"BSD-4-Clause", expr, re.IGNORECASE):
            out.append(f"{expr!r}: BSD-4-Clause is refused in every placement")
    if port["placement"] == "kernel":
        terms = [t.strip(" ()") for t in re.split(r"\bAND\b", lic)]
        refused = [t for t in terms if t not in allow]
        if refused:
            out.append(f"kernel placement needs licences in deny.toml [licenses].allow; not allowed: {refused}")
    if port["status"] == "ported":
        if not re.fullmatch(r"[0-9a-f]{40}", port["pinned_rev"]):
            out.append("ported: pinned_rev must be a full 40-hex commit")
        if not re.fullmatch(r"[0-9a-f]{64}", port["sha256"]):
            out.append("ported: sha256 must be 64 hex digits")
        patches = port_dir / "patches"
        count = sum(p.is_file() for p in patches.iterdir()) if patches.is_dir() else 0
        if port["patch_count"] != count:
            out.append(f"ported: patch_count {port['patch_count']} but patches/ holds {count} files")
        if not (ROOT / port["gate"]).is_file():
            out.append(f"ported: gate {port['gate']!r} does not exist")
    return out


PORT_DIRS = sorted(d for d in PORTS.iterdir() if d.is_dir()) if PORTS.is_dir() else []


def test_ports_tree_is_populated_and_every_port_has_a_port_toml():
    # Vacuity control: with no ports the per-port test below would pass by collecting nothing.
    assert PORT_DIRS, "no directories under ports/; the per-port licence gate would pass vacuously"
    bare = [d.name for d in PORT_DIRS if not (d / "PORT.toml").is_file()]
    assert not bare, f"directories under ports/ without PORT.toml escape the gate: {bare}"


@pytest.mark.parametrize("port_dir", PORT_DIRS, ids=lambda d: d.name)
def test_port_passes_the_licence_gate(port_dir):
    found = problems(port_dir)
    assert not found, f"ports/{port_dir.name}/PORT.toml: " + "; ".join(found)


# RED demonstration: bad PORT.toml fixtures written to a temporary tree must be refused.
GOOD = {
    "name": "fixture",
    "status": "planned",
    "upstream_url": "https://example.invalid/fixture",
    "pinned_rev": "",
    "sha256": "",
    "license": "MIT",
    "placement": "kernel",
    "patch_count": 0,
    "gate": "tests/ports/fixture.sh",
}


def write_port(tmp_path: Path, **override) -> Path:
    fields = {**GOOD, **override}
    port_dir = tmp_path / fields["name"]
    port_dir.mkdir()
    # json.dumps renders str and int as valid TOML values.
    (port_dir / "PORT.toml").write_text("".join(f"{k} = {json.dumps(v)}\n" for k, v in fields.items()))
    return port_dir


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"license": "GPL-2.0-only"}, "not allowed: ['GPL-2.0-only']"),
        ({"license": "MIT AND LGPL-3.0-only"}, "not allowed: ['LGPL-3.0-only']"),
        ({"license": "BSD-3-Clause OR GPL-2.0-only"}, "must name the one option"),
        ({"license": ""}, "license is empty"),
        ({"placement": "driver-server", "license": "BSD-4-Clause"}, "BSD-4-Clause is refused"),
        ({"upstream_license": "BSD-4-Clause-UC OR MIT"}, "BSD-4-Clause is refused"),
        ({"placement": "kernal"}, "placement 'kernal'"),
        ({"status": "ported"}, "pinned_rev must be a full 40-hex commit"),
    ],
    ids=["kernel-gpl", "kernel-and-lgpl", "kernel-dual-unelected", "empty", "server-bsd4", "upstream-bsd4-uc",
         "placement-typo", "ported-unpinned"],
)
def test_gate_refuses_a_bad_port(tmp_path, override, expected):
    found = problems(write_port(tmp_path, **override))
    assert any(expected in p for p in found), f"gate did not refuse {override}: {found}"


def test_gate_refuses_a_port_toml_missing_its_license(tmp_path):
    port_dir = write_port(tmp_path)
    toml = port_dir / "PORT.toml"
    toml.write_text("".join(line for line in toml.read_text().splitlines(True) if not line.startswith("license")))
    assert problems(port_dir) == ["missing fields ['license']"]


def test_gate_admits_good_ports(tmp_path):
    # Control: the refusals above come from the rules, not from a gate that refuses everything.
    assert problems(write_port(tmp_path)) == []
    assert problems(write_port(tmp_path, name="lkl", placement="driver-server", license="GPL-2.0-only")) == []
    assert problems(write_port(tmp_path, name="llvm", license="Apache-2.0 WITH LLVM-exception")) == []
