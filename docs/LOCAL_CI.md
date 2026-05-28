# Local CI System

This repository includes a **local CI runner** that mirrors the checks run in `.github/workflows/ci.yml` so you can catch failures before pushing.

---

## Quick Start

### 1. Enable Git hooks

```bash
git config core.hooksPath .githooks
```

Once enabled, every `git push` will automatically run the full local CI suite. If any check fails, the push is blocked.

### 2. Push with CI (alias)

If you prefer not to use hooks, use the provided alias:

```bash
git config --local include.path ../.gitconfig.local
git push-ci
```

This runs the same checks and only pushes on success.

### 3. Run CI manually

```bash
./scripts/local_ci.sh
```

---

## Bypassing Checks

If you need to push urgently without running CI (e.g., to share a WIP branch):

```bash
git push --no-verify
```

> ⚠️ Only use `--no-verify` for temporary branches. All code merged to `main` must pass CI.

---

## What Checks Run

| Step | Command |
|------|---------|
| Format | `cargo fmt --all -- --check` |
| Lints | `cargo clippy --workspace --all-targets -- -D warnings` |
| Tests | `cargo test --workspace` |
| Kernel build | `cd kernel/seal-os && cargo +nightly build --release` |
| Python tests | `python -m pytest tests/ -v` *(if pytest installed)* |
| Python compile | `python -m compileall -q infrastructure scripts tests kernel/epsilon/epsilon_core` *(if python installed)* |
| BOM check | `./scripts/check_no_bom.sh` |
| Docs | `cargo doc --workspace --no-deps` |

Each step is timed and color-coded:
- 🟢 **PASS** — green
- 🔴 **FAIL** — red (stops the pipeline immediately)

---

## Files

| File | Purpose |
|------|---------|
| `scripts/local_ci.sh` | Main CI runner |
| `scripts/push_with_ci.sh` | Wrapper that runs CI then `git push` |
| `.githooks/pre-push` | Hook that blocks push on CI failure |
| `.githooks/post-push` | Hook that confirms successful push |
| `.gitconfig.local` | Git alias `push-ci` |
| `docs/LOCAL_CI.md` | This documentation |
