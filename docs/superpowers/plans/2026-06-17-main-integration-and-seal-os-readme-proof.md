# Main Integration And Seal OS README Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan in an isolated worktree. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `main` forward by integrating viable PR/branch work, then make Seal OS satisfy the README's proof-gated v0.4.6 operating-system contract.

**Architecture:** Integration happens on `codex/integrate-main-2026-06-17`, based on `origin/main`, so the dirty local Seal OS completion branch remains preserved. PRs are merged in viability groups, with verification after each group. Seal OS work is judged against README and `docs/SEAL_OS_GUIDE.md` proof gates, not against aspirational prose alone.

**Tech Stack:** Git/GitHub CLI, Rust stable workspace, Rust nightly UEFI kernel build, `kernel/seal-mkimage` proof/audit gates, QEMU/VirtualBox proof scripts when available.

## Global Constraints

- Never overwrite the dirty checkout on `feat/seal-os-complete-2026-06-03`.
- Use the isolated worktree at `.claude/worktrees/codex-integrate-main-2026-06-17`.
- Do not merge PRs that are known-conflicted or failing without inspecting and fixing the reason.
- Preserve the README contract: Seal OS is a bare-metal Rust-first x86_64 UEFI kernel with proof-gated theorem/runtime/desktop/image artifacts.
- Verification must include source gates before any claim of OS readiness: `cargo +stable test --workspace`, `seal-mkimage --check-doc-claim-contract .`, `--check-seal-abi .`, `--check-language-hygiene .`, `--check-o1-allocator .`, and `--check-runtime-theorems .`.
- Full proof readiness additionally requires `cargo +nightly build --release` in `kernel/seal-os`, image creation/verification, and QEMU or VirtualBox proof artifacts.

---

### Task 1: Integrate Clean Open PRs

**Files:**
- Modify: files changed by PRs #35, #36, #37, and #38.

**Interfaces:**
- Consumes: `origin/main`, open PR heads from GitHub.
- Produces: integration branch containing clean PR changes with conflicts resolved.

- [ ] **Step 1: Merge PR #35**

Run: `gh pr merge 35 --repo teerthsharma/Epsilon-Hollow --merge --auto=false` only if repository permissions allow it; otherwise merge locally with `git merge --no-ff origin/bolt-optimization-react-memo-15495305594044364360`.

- [ ] **Step 2: Merge PR #36**

Run: `git merge --no-ff origin/palette-ux-chat-input-disabled-state-11961104562174843818`.

- [ ] **Step 3: Merge PR #37**

Run: `git merge --no-ff origin/bolt/optimize-chat-render-12662202546210248822`.

- [ ] **Step 4: Merge PR #38**

Run: `git merge --no-ff origin/palette/chat-input-disabled-ux-10524282739181406374`.

- [ ] **Step 5: Verify frontend merge surface**

Run: `git diff --check origin/main...HEAD`.

Expected: no whitespace/conflict marker errors.

### Task 2: Evaluate Latest Forward-Moving PR Representatives

**Files:**
- Inspect: `future/apeiron-runtime/APEIRON/frontend/components/ChatInterface.tsx`
- Inspect/modify as needed: files touched by PRs #65 and #66.

**Interfaces:**
- Consumes: latest Palette PR #65 and latest Bolt PR #66.
- Produces: either integrated representative changes or documented rejection with evidence.

- [ ] **Step 1: Inspect PR #65 changed files and checks**

Run: `gh pr diff 65 --repo teerthsharma/Epsilon-Hollow --name-only` and `gh pr checks 65 --repo teerthsharma/Epsilon-Hollow`.

- [ ] **Step 2: Inspect PR #66 changed files and checks**

Run: `gh pr diff 66 --repo teerthsharma/Epsilon-Hollow --name-only` and `gh pr checks 66 --repo teerthsharma/Epsilon-Hollow`.

- [ ] **Step 3: Merge viable representative changes locally**

Use local merge or cherry-pick only after identifying whether the failures are stale CI, unrelated baseline failures, or actual code regressions.

- [ ] **Step 4: Fix conflicts and build failures created by representative merges**

Run the smallest failing command from checks locally, fix the root cause, and rerun it.

### Task 3: Integrate Seal OS Completion Branch

**Files:**
- Modify: Seal OS files changed by `origin/feat/seal-os-complete-2026-06-03`.
- Preserve: uncommitted local edits in the original checkout remain untouched.

**Interfaces:**
- Consumes: `origin/feat/seal-os-complete-2026-06-03`.
- Produces: integration branch containing committed Seal OS proof-gate work.

- [ ] **Step 1: Merge branch**

Run: `git merge --no-ff origin/feat/seal-os-complete-2026-06-03`.

- [ ] **Step 2: Resolve conflicts by keeping README/proof gates honest**

Prefer code that strengthens proof gates and removes fake/stub claims. Do not resolve conflicts by deleting tests or weakening README guardrails.

- [ ] **Step 3: Compare original dirty branch edits**

Run: `git diff --name-status origin/feat/seal-os-complete-2026-06-03 -- C:\Users\seal\Documents\GitHub\Epsilon-Hollow` is not valid across worktrees; instead inspect the original checkout status and selectively port the uncommitted OS work after reviewing each file.

### Task 4: README Proof-Gate Implementation

**Files:**
- Modify as needed: `kernel/seal-mkimage/src/main.rs`
- Modify as needed: `kernel/seal-os/src/**`
- Modify as needed: `docs/CI.md`, `docs/SEAL_OS_GUIDE.md`, `docs/VM_RUNBOOK.md`, `README.md`

**Interfaces:**
- Consumes: README/guide proof contract.
- Produces: stronger OS proof behavior and documentation that matches code.

- [ ] **Step 1: Run source/audit gates**

Run:

```powershell
cargo +stable test --workspace
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-doc-claim-contract .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-seal-abi .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-language-hygiene .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-o1-allocator .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-runtime-theorems .
```

- [ ] **Step 2: Fix every source/audit gate failure**

Make code or doc changes that make the README more true. Do not weaken gates unless the README claim is narrowed at the same time.

- [ ] **Step 3: Build kernel**

Run:

```powershell
Set-Location kernel\seal-os
cargo +nightly build --release
```

- [ ] **Step 4: Build and verify image**

Run:

```powershell
Set-Location ..\seal-mkimage
cargo +stable run --release
cargo +stable run --release -- --verify ..\seal-os\target\x86_64-unknown-uefi\release\seal-os.img ..\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
```

- [ ] **Step 5: Run VM proof if local tooling exists**

Run QEMU proof first. If native QEMU is unavailable, use the documented WSL2 fallback. If QEMU is unavailable but VirtualBox exists, run the VirtualBox smoke proof.

### Task 5: Completion Evidence

**Files:**
- Produce: command output evidence in final report.
- Optional produce: proof manifests under ignored target directories.

**Interfaces:**
- Consumes: verification results from Tasks 1-4.
- Produces: a clear readiness report and any remaining blockers.

- [ ] **Step 1: Summarize merged PRs and rejected PRs**

List PR numbers, merge decisions, and evidence.

- [ ] **Step 2: Summarize branch integration**

Report whether `origin/feat/seal-os-complete-2026-06-03` is merged and whether original dirty checkout changes were ported.

- [ ] **Step 3: Summarize proof status**

Report exact commands run, pass/fail status, and remaining gaps.
