"""Whether the distinctive subsystems reach anyone who does not boot Seal OS.

stratum's pure f64 shape kernels already moved to aether-core
(src/trajectory_shape.rs says so, because kernel unit tests never run), and
`cargo test --workspace` covers them. The classifier that turns those numbers
into underfit / wellfit / overfit / collapsing, and its 7-case proof, stay in
kernel/seal-os/src/ml_engine/stratum.rs, which the root Cargo.toml excludes;
foliation (KV-cache eviction) and ManifoldFS likewise. Doc comments that name
those files do not compile them, so comments are stripped before matching.

The second test is the cost side: stratum.rs, unmodified, built as a host
program with a one-line `serial_println` shim, runs its own boot proof.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
STRATUM = REPO / "kernel/seal-os/src/ml_engine/stratum.rs"
AETHER_CORE = REPO / "kernel/epsilon/epsilon/crates/aether-core"
PILLAR_MARKERS = ("stratum_proof_line", "foliation_proof_line", "ml_engine/stratum.rs",
                  "ml_engine/foliation.rs", "fs/manifold_fs.rs")


def workspace_members():
    text = (REPO / "Cargo.toml").read_text(encoding="utf-8")
    block = re.search(r"members\s*=\s*\[(.*?)\]", text, re.S).group(1)
    return re.findall(r'"([^"]+)"', block)


def test_host_workspace_exercises_a_distinctive_pillar():
    members = workspace_members()
    hits = []
    for member in members:
        for path in (REPO / member).rglob("*.rs"):
            if "target" in path.parts:
                continue
            code = re.sub(r"//.*", "", path.read_text(encoding="utf-8", errors="replace"))
            hits += [f"{path.relative_to(REPO)}: {m}" for m in PILLAR_MARKERS if m in code]
    assert hits, (
        f"0 of {len(members)} host workspace members compile the stratum classifier, foliation or ManifoldFS; "
        f"they build only inside kernel/seal-os, which Cargo.toml excludes"
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo")
def test_stratum_runs_its_proof_on_the_host_unmodified(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "stratum_host_probe"\nversion = "0.0.0"\nedition = "2021"\n[workspace]\n'
        "[dependencies]\n"
        f'aether-core = {{ path = "{AETHER_CORE.as_posix()}", default-features = false, features = ["no_std"] }}\n'
        'spin = "0.9"\nlibm = "0.2"\n', encoding="utf-8")
    (tmp_path / "src/main.rs").write_text(
        "extern crate alloc;\n"
        "#[macro_export] macro_rules! serial_println { ($($t:tt)*) => { std::println!($($t)*) }; }\n"
        f'#[path = "{STRATUM.as_posix()}"] #[allow(dead_code, unused)] mod stratum;\n'
        'fn main() { println!("{}", stratum::stratum_proof_line()); }\n', encoding="utf-8")
    run = subprocess.run(["cargo", "run", "--offline", "--quiet", "--manifest-path", str(tmp_path / "Cargo.toml")],
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "subsystem=stratum" in run.stdout and "result=pass" in run.stdout, run.stdout[-1000:]
