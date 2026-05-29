# Seal OS 0.4.5 VM Benchmark Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Seal OS 0.4.5 into a repeatable VM-proven topological OS baseline with theorem-gated boot, clean first desktop frame, Seal ABI/no-POSIX discipline, O(1) allocator proof gates, and a measurable benchmark path toward Ubuntu comparison for HFT/ML workloads.

**Architecture:** Treat this as a proof pipeline, not a feature spree. The kernel boots first, the proof harness verifies serial plus pixels, Rust audit gates enforce theorem/ABI/hygiene/O(1), then benchmark docs define what must beat Ubuntu and what is still only planned. Aether-Lang remains the OS application language surface; Rust and assembly remain the kernel implementation surface.

**Tech Stack:** Bare-metal Rust `no_std`, x86_64 UEFI, QEMU/WSL2/OVMF, PowerShell runner, Rust `seal-mkimage` audit gates, Aether-Lang, x86_64 assembly.

---

## Scope Lock

This plan intentionally does not try to "finish every OS feature" in one pass. That is how the work went blurry. The locked deliverable for this round is:

- Seal OS 0.4.5 release build boots in QEMU through WSL2.
- Serial log proves T1-T10 verified, desktop ready, and event loop active.
- Screenshot proof shows a clean usable first screen with the Terminal/control surface visible.
- `seal-mkimage` gates pass: image verification, theorem log, Seal ABI, language hygiene, O(1) allocator, unsafe inventory report.
- Rust workspace tests pass where the workspace is designed to run on host.
- Kernel release build passes; kernel host unit test harness failure is documented until fixed.
- README and OS guide describe current truth, not fantasy.
- Ubuntu comparison is converted into a benchmark contract with named workloads and pass/fail thresholds.
- Remaining legacy host-script files are counted and classified. Core theorem Python files are converted only when replaced by Rust tests and Rust modules.

Out of scope for this single pass:

- Claiming Seal beats Ubuntu without numbers.
- Deleting legacy Python just to affect GitHub language percentages.
- Implementing real GPU VRAM file teleportation before a VM-visible proof path exists.
- Turning every app into a full production app before kernel/VM proof is stable.

---

## File Map

### VM Proof And Runner

- Modify: `kernel/seal-os/run-qemu.ps1`
  - Owns WSL2/Windows QEMU launch.
  - Must produce `serial.log`, `screen.ppm`, and optionally `screen.png`.
  - Must fail if serial proof is incomplete.

- Modify: `kernel/seal-mkimage/src/main.rs`
  - Owns audit gates.
  - Add `--check-proof-screen <ppm>` so screenshot proof is machine checked in Rust.

### Desktop First Frame

- Modify: `kernel/seal-os/src/lib.rs`
  - Owns boot sequence and initial desktop/compositor handoff.
  - Must explicitly request a full compositor redraw after painting the desktop background.

- Modify: `kernel/seal-os/src/wm/desktop.rs`
  - Owns desktop icons, wallpaper, equation HUD, taskbar overlays.
  - Must prevent icon/HUD overlap.

- Modify: `kernel/seal-os/src/wm/app_state.rs`
  - Owns app window creation and initial boot app state.
  - Must make the first screen usable by default.

- Modify: `kernel/seal-os/src/wm/compositor.rs`
  - Owns window composition and dirty region behavior.
  - Must not skip composition after a desktop repaint.

### Allocator And Topological Proof

- Modify: `kernel/seal-os/src/memory/phys.rs`
  - Owns physical frame allocation.
  - Must avoid RAM-wide bitmap/contiguous scans in runtime allocation paths.

- Modify: `kernel/seal-os/src/memory/topo_ram.rs`
  - Owns topology-aware RAM wrappers.
  - Must call bounded allocator paths only.

- Modify: `kernel/seal-mkimage/src/main.rs`
  - Strengthen O(1) audit gate.

### Aether-Lang And App Surface

- Modify: `kernel/aether/Aether-Lang/crates/aether-lang/src/parser.rs`
  - Owns Aether parser.
  - Must support OS app syntax used at boot.

- Modify: `kernel/aether/Aether-Lang/crates/aether-lang/src/interpreter.rs`
  - Owns Aether runtime dispatch.
  - Must support `window.*`, `graphics.*`, `input.*`, and `ui.*` modules used by boot apps.

- Modify: `kernel/seal-os/src/lang/mod.rs`
  - Owns kernel/Aether runtime bridge.
  - Must report parse/runtime errors with actionable line/column.

### Python-To-Rust Conversion

- Modify or create under: `kernel/epsilon/epsilon/crates/aether-core/src/`
  - Core theorem modules live here.
  - New Rust modules must have tests before deleting legacy host-script source.

- Delete only after replacement is verified:
  - `kernel/epsilon/epsilon_core/world_model.py`
  - `kernel/epsilon/epsilon_core/world_model_horizon.py`
  - `kernel/epsilon/epsilon_core/thermodynamic_plasticity.py`
  - `kernel/epsilon/epsilon_core/persistent_kv_partition.py`
  - `kernel/epsilon/epsilon_core/geodesic_consolidation.py`
  - `kernel/epsilon/epsilon_core/cross_manifold_alignment.py`
  - `kernel/epsilon/epsilon_core/hyperbolic_capacity.py`
  - `kernel/epsilon/epsilon_core/angular_sparse_attention.py`

### Docs And CI

- Modify: `README.md`
  - Must show Seal OS 0.4.5 status table with checked items only where gates prove them.

- Modify: `docs/SEAL_OS_GUIDE.md`
  - Must be the long operator guide: build, VM, proof, gates, benchmarks, known gaps.

- Modify: `docs/TOPOLOGICAL_OS_CONTRACT.md`
  - Must define T1-T10 runtime duties and proof hooks.

- Modify: `docs/BENCHMARK_PLAN.md`
  - Must define Ubuntu comparison workloads and thresholds.

- Modify: `.gitignore`
  - Must ignore build/proof artifacts and keep source visible.

- Modify: `.gitattributes`
  - Must mark legacy host-script files vendored or generated for language stats, while not pretending they are gone.

- Modify: `.github/workflows/ci.yml`
  - Must run the audit gates that do not require a GUI VM.

---

## Task 1: Freeze Current Evidence

**Files:**
- Read: `kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log`
- Read: `kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.png`
- Modify: `docs/SEAL_OS_GUIDE.md`

- [ ] **Step 1: Run the kernel release build**

Run:

```powershell
cargo +nightly build --release
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os
```

Expected:

```text
Finished `release` profile
```

- [ ] **Step 2: Rebuild the boot image**

Run:

```powershell
cargo +stable run --release
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-mkimage
```

Expected:

```text
[mkimage] IMG:
[mkimage] EFI boot image:
```

- [ ] **Step 3: Boot proof in WSL2 QEMU**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os
```

Expected:

```text
Serial log: /mnt/c/Users/seal/Documents/GitHub/Epsilon-Hollow/kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log
Screenshot: /mnt/c/Users/seal/Documents/GitHub/Epsilon-Hollow/kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.png
```

- [ ] **Step 4: Verify serial evidence**

Run:

```powershell
rg -n "\[THEOREM\]|All T1-T10|Seal OS desktop ready|Entering real event loop|AetherAppHost|panic|PANIC|FAULT|DOUBLE FAULT|qemu: fatal|WATCHDOG FIRED" target\x86_64-unknown-uefi\release\qemu-proof\serial.log
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os
```

Expected:

```text
[THEOREM] T1/TSS VERIFIED
[THEOREM] T2/SCM VERIFIED
[THEOREM] T3/GMC VERIFIED
[THEOREM] T4/AGCR VERIFIED
[THEOREM] T5/HCS VERIFIED
[THEOREM] T6/RGCS VERIFIED
[THEOREM] T7/PHKP VERIFIED
[THEOREM] T8/TEB VERIFIED
[THEOREM] T9/CMA VERIFIED
[THEOREM] T10/WPHB VERIFIED
[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths
[BOOT] Seal OS desktop ready.
[EVENT] Entering real event loop
```

No `panic`, `FAULT`, `DOUBLE FAULT`, `qemu: fatal`, or `WATCHDOG FIRED`.

- [ ] **Step 5: Record truth in the guide**

Add or update this section in `docs/SEAL_OS_GUIDE.md`:

```markdown
## Seal OS 0.4.5 Proof Baseline

Current required proof:

- Kernel release build: `cargo +nightly build --release`
- Image build: `cargo +stable run --release` in `kernel/seal-mkimage`
- VM proof: `powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1 -HeadlessProof -ProofSeconds 240`
- Serial proof: T1-T10 verified, desktop ready, event loop active
- Screenshot proof: clean desktop with visible primary control surface

This proves Seal OS boots in the current WSL2/QEMU environment. It does not by itself prove that Seal OS beats Ubuntu on HFT/ML workloads.
```

---

## Task 2: Make The Screenshot Proof Machine-Checked

**Files:**
- Modify: `kernel/seal-mkimage/src/main.rs`
- Modify: `kernel/seal-os/run-qemu.ps1`
- Test: `kernel/seal-mkimage/src/main.rs` via CLI gate

- [ ] **Step 1: Add a CLI mode to `seal-mkimage`**

In `kernel/seal-mkimage/src/main.rs`, add a CLI branch near the existing audit modes:

```rust
        "--check-proof-screen" => {
            let ppm = args
                .next()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("../seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.ppm"));
            check_proof_screen(&ppm).unwrap_or_else(|e| {
                eprintln!("[seal-audit] PROOF SCREEN FAILED: {e}");
                std::process::exit(1);
            });
            println!("[seal-audit] PROOF SCREEN OK");
            return;
        }
```

- [ ] **Step 2: Implement a tiny PPM parser in Rust**

Add this function to `kernel/seal-mkimage/src/main.rs`:

```rust
fn check_proof_screen(path: &Path) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if !bytes.starts_with(b"P6") {
        return Err(format!("{} is not a binary PPM (P6)", path.display()));
    }

    let mut idx = 2usize;
    let width = read_ppm_number(&bytes, &mut idx)?;
    let height = read_ppm_number(&bytes, &mut idx)?;
    let max = read_ppm_number(&bytes, &mut idx)?;
    if max != 255 {
        return Err(format!("unsupported PPM max value {max}"));
    }
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }

    let expected = width
        .checked_mul(height)
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| "PPM dimensions overflow".to_string())?;
    if bytes.len().saturating_sub(idx) < expected {
        return Err("PPM pixel payload is truncated".to_string());
    }

    let pixels = &bytes[idx..idx + expected];
    let mut non_black = 0usize;
    let mut terminal_region_signal = 0usize;
    let mut icon_region_signal = 0usize;

    for y in 0..height {
        for x in 0..width {
            let p = ((y * width + x) * 3) as usize;
            let r = pixels[p];
            let g = pixels[p + 1];
            let b = pixels[p + 2];
            if r > 8 || g > 8 || b > 8 {
                non_black += 1;
            }
            if (48..660).contains(&x) && (120..570).contains(&y) && (r > 24 || g > 24 || b > 24) {
                terminal_region_signal += 1;
            }
            if (0..1000).contains(&x) && (40..130).contains(&y) && (r > 32 || g > 32 || b > 32) {
                icon_region_signal += 1;
            }
        }
    }

    if non_black < 20_000 {
        return Err(format!("screen is mostly blank: {non_black} non-black pixels"));
    }
    if icon_region_signal < 2_000 {
        return Err(format!("desktop icons are not visible enough: {icon_region_signal} signal pixels"));
    }
    if terminal_region_signal < 8_000 {
        return Err(format!("primary terminal/control region is not visible enough: {terminal_region_signal} signal pixels"));
    }

    Ok(())
}

fn read_ppm_number(bytes: &[u8], idx: &mut usize) -> Result<usize, String> {
    while *idx < bytes.len() && bytes[*idx].is_ascii_whitespace() {
        *idx += 1;
    }
    if *idx < bytes.len() && bytes[*idx] == b'#' {
        while *idx < bytes.len() && bytes[*idx] != b'\n' {
            *idx += 1;
        }
        return read_ppm_number(bytes, idx);
    }
    let start = *idx;
    while *idx < bytes.len() && bytes[*idx].is_ascii_digit() {
        *idx += 1;
    }
    if start == *idx {
        return Err("expected PPM number".to_string());
    }
    std::str::from_utf8(&bytes[start..*idx])
        .map_err(|e| format!("PPM number utf8: {e}"))?
        .parse::<usize>()
        .map_err(|e| format!("PPM number parse: {e}"))
}
```

- [ ] **Step 3: Make `run-qemu.ps1` preserve `screen.ppm`**

In `kernel/seal-os/run-qemu.ps1`, keep:

```bash
PPM="$PROOF_DIR/screen.ppm"
PNG="$PROOF_DIR/screen.png"
```

Do not delete `screen.ppm` after conversion. It is the source for the Rust pixel proof.

- [ ] **Step 4: Run the new screen gate**

Run:

```powershell
cargo +stable run --release -- --check-proof-screen ..\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-mkimage
```

Expected before desktop fix:

```text
[seal-audit] PROOF SCREEN FAILED: primary terminal/control region is not visible enough
```

Expected after desktop fix:

```text
[seal-audit] PROOF SCREEN OK
```

---

## Task 3: Fix First Desktop Frame Composition

**Files:**
- Modify: `kernel/seal-os/src/lib.rs`
- Modify: `kernel/seal-os/src/wm/compositor.rs`
- Modify: `kernel/seal-os/src/wm/app_state.rs`
- Test: VM proof plus `--check-proof-screen`

- [ ] **Step 1: Add a failing proof before changing compositor code**

Run:

```powershell
cargo +stable run --release -- --check-proof-screen ..\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
```

Expected current failure if the terminal is absent:

```text
[seal-audit] PROOF SCREEN FAILED: primary terminal/control region is not visible enough
```

- [ ] **Step 2: Force full composition after initial desktop paint**

In `kernel/seal-os/src/lib.rs`, after the terminal/control surface is rendered and before the first `compositor.compose(fb)`, use this exact shape:

```rust
    if let Some(win) = compositor.window_mut(app_state.term_id) {
        app_state.terminal.render_to_window(win);
    }

    compositor.mark_dirty(0, 0, fb.width, fb.height);
    compositor.compose(fb);
    fb.blit();
```

- [ ] **Step 3: Force full composition after every desktop repaint**

In `kernel/seal-os/src/lib.rs`, the event-loop repaint block must keep this exact order:

```rust
        if frame_dirty {
            app_state.render_dirty(&mut compositor);
            wm::desktop::render_desktop(fb);
            compositor.mark_dirty(0, 0, fb.width, fb.height);
            compositor.compose(fb);
            wm::desktop::render_overlays(fb);
            wm::desktop::render_notifications(fb);
            fb.blit();
            frame_dirty = false;
        }
```

- [ ] **Step 4: Keep only the primary terminal visible at boot**

In `kernel/seal-os/src/wm/app_state.rs`, preserve the `minimize_secondary_boot_apps` helper:

```rust
    fn minimize_secondary_boot_apps(&mut self, compositor: &mut Compositor) {
        let ids = [
            self.ide_id,
            self.tv_id,
            self.calc_id,
            self.player_id,
            self.snake_id,
            self.breakout_id,
            self.warp_racer_id,
            self.fm_id,
            self.tensor_id,
            self.laamba_id,
            self.aether_id,
        ];

        for id in ids {
            if let Some(win) = compositor.window_mut(id) {
                win.state = WindowState::Minimized;
                win.focused = false;
                win.dirty = false;
            }
        }
    }
```

Call it after `self.render_all(compositor);`.

- [ ] **Step 5: Rebuild and boot**

Run:

```powershell
cargo +nightly build --release
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os
```

Then:

```powershell
cargo +stable run --release
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-mkimage
```

Then:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os
```

- [ ] **Step 6: Verify serial and pixels**

Run:

```powershell
cargo +stable run --release -- --check-theorem-log ..\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --release -- --check-proof-screen ..\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-mkimage
```

Expected:

```text
[seal-audit] THEOREM LOG OK
[seal-audit] PROOF SCREEN OK
```

---

## Task 4: Keep Desktop HUD And Icons Clean

**Files:**
- Modify: `kernel/seal-os/src/wm/desktop.rs`
- Test: VM screenshot plus `--check-proof-screen`

- [ ] **Step 1: Confirm the current overlap risk**

Inspect the proof screenshot:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.png
```

Fail condition:

```text
Equation HUD overlaps app icons or labels.
```

- [ ] **Step 2: Keep equation HUD below the icon lane**

In `render_equations_text`, use bottom placement:

```rust
    let content_bottom = fb.height.saturating_sub(taskbar::taskbar_height());
    let text_y = content_bottom.saturating_sub(bg_h + 28) + PAD;
    let bg_y = text_y.saturating_sub(PAD);
```

- [ ] **Step 3: Give each icon a stable label backing**

In `draw_icon`, center labels and draw a backing rectangle:

```rust
    let label_w = icon.name.len() as u32 * font::CHAR_WIDTH;
    let label_x = if label_w > icon.width {
        icon.x.saturating_sub((label_w - icon.width) / 2)
    } else {
        icon.x + (icon.width - label_w) / 2
    };
    let label_bg_x = label_x.saturating_sub(4);
    let label_bg_w = label_w + 8;

    fb.fill_rect(label_bg_x, text_y.saturating_sub(1), label_bg_w, font::CHAR_HEIGHT + 2, 0);
```

- [ ] **Step 4: Re-run VM proof**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
```

Expected:

```text
Screenshot generated and first screen is readable.
```

---

## Task 5: Strengthen O(1) Physical Allocation Proof

**Files:**
- Modify: `kernel/seal-os/src/memory/phys.rs`
- Modify: `kernel/seal-os/src/memory/topo_ram.rs`
- Modify: `kernel/seal-mkimage/src/main.rs`
- Test: `--check-o1-allocator`, kernel build, VM proof

- [ ] **Step 1: Check current allocator audit**

Run:

```powershell
cargo +stable run --release -- --check-o1-allocator ..\..
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-mkimage
```

Expected:

```text
[seal-audit] O(1) ALLOCATOR OK
```

- [ ] **Step 2: Keep bounded candidate probes**

In `kernel/seal-os/src/memory/phys.rs`, the contiguous allocator must use:

```rust
const CONTIGUOUS_CANDIDATE_PROBES: usize = 128;
```

And the search loop must be bounded:

```rust
    for probe in 0..CONTIGUOUS_CANDIDATE_PROBES {
        let cell = (cursor + probe) % PHYS_TOPO_CELLS;
        // Candidate-only search. No RAM-wide bitmap scan.
    }
```

- [ ] **Step 3: Ban known full-scan patterns in the gate**

In `kernel/seal-mkimage/src/main.rs`, `check_o1_allocator` must reject these patterns inside `alloc_frames_contiguous_in_range`:

```rust
let banned = [
    "let first_word = min_frame / 64",
    "let last_word = (limit - 1) / 64",
    "for bit in 0..64",
    "for word_idx in 0..BITMAP_U64S",
    "bitmap.iter_mut().enumerate()",
];
```

- [ ] **Step 4: Add allocation proof counters to serial**

In `kernel/seal-os/src/memory/phys.rs`, add counters:

```rust
static TOPO_ALLOC_FAST_HITS: AtomicU64 = AtomicU64::new(0);
static TOPO_ALLOC_BOUNDED_MISSES: AtomicU64 = AtomicU64::new(0);
```

Increment `TOPO_ALLOC_FAST_HITS` when a topological candidate succeeds and `TOPO_ALLOC_BOUNDED_MISSES` when bounded probes fail.

Expose:

```rust
pub fn topo_alloc_stats() -> (u64, u64) {
    (
        TOPO_ALLOC_FAST_HITS.load(Ordering::Relaxed),
        TOPO_ALLOC_BOUNDED_MISSES.load(Ordering::Relaxed),
    )
}
```

- [ ] **Step 5: Print allocator proof at boot**

In `kernel/seal-os/src/lib.rs`, after memory/topological initialization:

```rust
    let (fast_hits, bounded_misses) = memory::phys::topo_alloc_stats();
    serial_println!(
        "[ALLOC] topological fast hits={}, bounded misses={}, max contiguous probes=128",
        fast_hits,
        bounded_misses
    );
```

- [ ] **Step 6: Update theorem log gate**

In `check_theorem_log`, require:

```rust
"[ALLOC] topological fast hits="
```

Expected after boot proof:

```text
[seal-audit] THEOREM LOG OK
```

---

## Task 6: Keep Seal ABI And No-POSIX Discipline Hard

**Files:**
- Modify: `kernel/seal-mkimage/src/main.rs`
- Modify: `.github/workflows/ci.yml`
- Test: `--check-seal-abi`, `--check-language-hygiene`

- [ ] **Step 1: Run existing gates**

Run:

```powershell
cargo +stable run --release -- --check-seal-abi ..\..
cargo +stable run --release -- --check-language-hygiene ..\..
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-mkimage
```

Expected:

```text
[seal-audit] SEAL ABI OK
[seal-audit] LANGUAGE HYGIENE OK
```

- [ ] **Step 2: Expand banned POSIX surface**

In `check_seal_abi`, ban accidental kernel symbols:

```rust
let banned_kernel_tokens = [
    "libc::",
    "std::os::unix",
    "posix",
    "fork(",
    "pthread_",
    "mmap(",
    "open(",
    "read(",
    "write(",
    "close(",
];
```

Skip documentation and explicit compatibility notes. Do not ban Seal's own syscall names when they are part of the native Seal ABI table.

- [ ] **Step 3: CI must run all non-GUI gates**

In `.github/workflows/ci.yml`, ensure these commands exist:

```yaml
- name: Check Seal ABI
  run: cargo +stable run --release -- --check-seal-abi ../..
  working-directory: kernel/seal-mkimage

- name: Check language hygiene
  run: cargo +stable run --release -- --check-language-hygiene ../..
  working-directory: kernel/seal-mkimage

- name: Check O(1) allocator proof
  run: cargo +stable run --release -- --check-o1-allocator ../..
  working-directory: kernel/seal-mkimage
```

---

## Task 7: Aether-Lang Boot App Contract

**Files:**
- Modify: `kernel/aether/Aether-Lang/crates/aether-lang/src/parser.rs`
- Modify: `kernel/aether/Aether-Lang/crates/aether-lang/src/interpreter.rs`
- Modify: `kernel/seal-os/src/lang/mod.rs`
- Test: `cargo +stable test -p aether-lang --lib`, VM proof

- [ ] **Step 1: Keep parser tests for OS syntax**

In `parser.rs`, keep tests:

```rust
#[test]
fn parses_tilde_statement_terminators() {
    let src = "let x = 1~ let y = 2~";
    assert!(parse(src).is_ok());
}

#[test]
fn parses_windowed_app_host_script_shape() {
    let src = r#"
        fn render() {
            window.create("LAAMBA")
            graphics.clear(0)
        }
    "#;
    assert!(parse(src).is_ok());
}
```

- [ ] **Step 2: Keep module dispatch tests**

In `interpreter.rs`, keep:

```rust
#[test]
fn test_windowed_module_methods_execute_without_callbacks() {
    let src = r#"
        let win = window.create("Test")
        window.set_title(win, "Ready")
        graphics.clear(0)
        ui.label(1, 2, "ok")
    "#;
    assert!(Interpreter::new().eval(src).is_ok());
}
```

Adapt names only to match the actual interpreter test harness.

- [ ] **Step 3: Run Aether tests**

Run:

```powershell
cargo +stable test -p aether-lang --lib
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow
```

Expected:

```text
test result: ok
```

- [ ] **Step 4: VM proof must have no Aether app init error**

Run:

```powershell
rg -n "AetherAppHost|script init error|parse error|Method .* not found" target\x86_64-unknown-uefi\release\qemu-proof\serial.log
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os
```

Expected:

```text
No matches.
```

---

## Task 8: Convert Remaining Core Theorem Host Scripts To Rust

**Files:**
- Create or modify: `kernel/epsilon/epsilon/crates/aether-core/src/world_model.rs`
- Create or modify: `kernel/epsilon/epsilon/crates/aether-core/src/world_model_horizon.rs`
- Create or modify: `kernel/epsilon/epsilon/crates/aether-core/src/thermodynamic_plasticity.rs`
- Create or modify: `kernel/epsilon/epsilon/crates/aether-core/src/persistent_kv_partition.rs`
- Create or modify: `kernel/epsilon/epsilon/crates/aether-core/src/geodesic_consolidation.rs`
- Create or modify: `kernel/epsilon/epsilon/crates/aether-core/src/cross_manifold_alignment.rs`
- Create or modify: `kernel/epsilon/epsilon/crates/aether-core/src/hyperbolic_capacity.rs`
- Create or modify: `kernel/epsilon/epsilon/crates/aether-core/src/angular_sparse_attention.rs`
- Modify: `kernel/epsilon/epsilon/crates/aether-core/src/lib.rs`
- Delete only after replacement passes: matching files under `kernel/epsilon/epsilon_core/*.py`

- [ ] **Step 1: Count legacy host-script files**

Run:

```powershell
(rg --files -g *.py -g *.pyw | Measure-Object).Count
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow
```

Expected current value:

```text
101
```

- [ ] **Step 2: Convert one theorem file at a time**

For each conversion, follow this pattern:

1. Read the legacy source.
2. Create a Rust module in `aether-core/src`.
3. Add tests that lock the old theorem behavior.
4. Export the module in `lib.rs`.
5. Run `cargo +stable test -p aether-core`.
6. Delete the legacy source only after the Rust test passes.

Do not delete a host-script file without a Rust module and test replacing it.

- [ ] **Step 3: Example module skeleton**

For `world_model_horizon.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizonEstimate {
    pub stable_steps: u32,
    pub contraction_margin: f64,
}

pub fn estimate_horizon(alpha: f64, epsilon: f64, max_steps: u32) -> HorizonEstimate {
    let alpha = alpha.clamp(0.0, 1.0);
    let epsilon = epsilon.max(1.0e-9);
    let contraction = (1.0 - alpha).clamp(0.0, 1.0);
    let mut remaining = 1.0f64;
    let mut steps = 0u32;

    while steps < max_steps && remaining > epsilon {
        remaining *= contraction;
        steps += 1;
        if contraction == 0.0 {
            break;
        }
    }

    HorizonEstimate {
        stable_steps: steps,
        contraction_margin: 1.0 - contraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stronger_alpha_shortens_horizon() {
        let slow = estimate_horizon(0.1, 0.01, 1_000);
        let fast = estimate_horizon(0.5, 0.01, 1_000);
        assert!(fast.stable_steps < slow.stable_steps);
    }

    #[test]
    fn alpha_one_converges_immediately() {
        let h = estimate_horizon(1.0, 0.01, 1_000);
        assert_eq!(h.stable_steps, 1);
        assert_eq!(h.contraction_margin, 1.0);
    }
}
```

This is a skeleton. Before implementation, replace the formulas with the exact behavior from the legacy file.

- [ ] **Step 4: Run no_std core check**

Run:

```powershell
cargo +stable check --no-default-features --features no_std
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\epsilon\epsilon\crates\aether-core
```

Expected:

```text
Finished `dev` profile
```

---

## Task 9: GitHub Language Hygiene Without Lying

**Files:**
- Modify: `.gitattributes`
- Modify: `.gitignore`
- Modify: `docs/SEAL_OS_GUIDE.md`

- [ ] **Step 1: Keep generated/build output ignored**

In `.gitignore`, ensure:

```gitignore
target/
**/target/
target-test-*/
**/target-test-*/
kernel/seal-os/target/**/qemu-proof/
*.img
*.efi
*.vdi
*.vmdk
*.qcow2
*.log
results/
```

- [ ] **Step 2: Keep source language stats honest**

In `.gitattributes`, keep legacy host-script files vendored until converted:

```gitattributes
*.rs linguist-language=Rust
*.S linguist-language=Assembly
*.s linguist-language=Assembly
*.asm linguist-language=Assembly
*.aether linguist-language=Rust
*.aeth linguist-language=Rust
*.seal linguist-language=Rust
*.py linguist-vendored
*.pyw linguist-vendored
docs/** linguist-documentation
```

- [ ] **Step 3: Document why files remain**

In `docs/SEAL_OS_GUIDE.md`, add:

```markdown
## Language Hygiene

Seal OS kernel and boot-critical runtime code target Rust, assembly, and Aether-Lang. Legacy host-script files remain only as reference material, tests, bindings, or conversion backlog. They are not part of the Seal ABI and must not be used as a runtime dependency for the OS boot path.
```

---

## Task 10: Benchmark Contract Against Ubuntu

**Files:**
- Modify: `docs/BENCHMARK_PLAN.md`
- Modify: `README.md`
- Create: `docs/SEAL_OS_GUIDE.md` if missing

- [ ] **Step 1: Define workloads**

Add this table to `docs/BENCHMARK_PLAN.md`:

```markdown
## Ubuntu Comparison Workloads

| Workload | Seal Target | Ubuntu Baseline | Pass Condition |
| --- | --- | --- | --- |
| Boot to interactive desktop | QEMU serial plus pixel proof | Ubuntu Server/Desktop VM cold boot | Seal reaches event loop faster under identical VM CPU/RAM settings |
| O(1) frame allocation | bounded topological probes | Linux buddy allocator under pressure | Seal probe count remains bounded and latency p99 is lower |
| HFT message ring | Seal ABI ring buffer | Linux userspace UDP or shared memory path | Seal p99 and p999 latency lower under fixed message count |
| ML tensor page movement | topological RAM/VRAM plan | Linux mmap/page cache/GPU transfer baseline | Seal data movement p99 lower after GPU-native path exists |
| Desktop frame stability | compositor proof screen | Ubuntu VM screenshot smoke test | Seal proof screen has no blank/overlap regression |
```

- [ ] **Step 2: Add honest status**

Add:

```markdown
## Current Benchmark Status

Seal OS currently has boot proof, theorem gates, image verification, ABI hygiene, and O(1) allocator source gates. It does not yet have enough runtime benchmark data to claim it beats Ubuntu. The claim becomes valid only when the workloads above run on identical VM settings and produce recorded p50, p95, p99, and p999 numbers.
```

- [ ] **Step 3: Add README truth table**

In `README.md`, add:

```markdown
## Seal OS 0.4.5 Proof Status

| Claim | Status | Proof |
| --- | --- | --- |
| Boots in WSL2/QEMU | Verified | `run-qemu.ps1 -HeadlessProof` |
| T1-T10 theorem log | Verified | `seal-mkimage --check-theorem-log` |
| Seal ABI/no POSIX gate | Verified | `seal-mkimage --check-seal-abi` |
| O(1) allocator source gate | Verified | `seal-mkimage --check-o1-allocator` |
| Clean first desktop frame | Required | `seal-mkimage --check-proof-screen` |
| Beats Ubuntu HFT/ML | Not yet proven | Pending benchmark suite |
```

---

## Task 11: Final Audit Command Set

**Files:**
- No source changes unless a gate fails.

- [ ] **Step 1: Format**

Run:

```powershell
cargo +stable fmt --check
cargo +nightly fmt --check
```

Directories:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os
```

Expected:

```text
No output and exit code 0.
```

- [ ] **Step 2: Build and package**

Run:

```powershell
cargo +nightly build --release
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os
```

Run:

```powershell
cargo +stable run --release
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-mkimage
```

- [ ] **Step 3: VM proof**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os
```

- [ ] **Step 4: Audit gates**

Run:

```powershell
cargo +stable run --release -- --verify ..\seal-os\target\x86_64-unknown-uefi\release\seal-os.img ..\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
cargo +stable run --release -- --check-theorem-log ..\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --release -- --check-proof-screen ..\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
cargo +stable run --release -- --check-seal-abi ..\..
cargo +stable run --release -- --check-language-hygiene ..\..
cargo +stable run --release -- --check-o1-allocator ..\..
cargo +stable run --release -- --unsafe-inventory ..\..
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-mkimage
```

Expected:

```text
[mkimage] VERIFY OK
[seal-audit] THEOREM LOG OK
[seal-audit] PROOF SCREEN OK
[seal-audit] SEAL ABI OK
[seal-audit] LANGUAGE HYGIENE OK
[seal-audit] O(1) ALLOCATOR OK
[seal-audit] unsafe total <number>
```

- [ ] **Step 5: Host workspace tests**

Run:

```powershell
cargo +stable test --workspace
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow
```

Expected:

```text
test result: ok
```

- [ ] **Step 6: no_std theorem core check**

Run:

```powershell
cargo +stable check --no-default-features --features no_std
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\epsilon\epsilon\crates\aether-core
```

Expected:

```text
Finished `dev` profile
```

- [ ] **Step 7: Known failing kernel host-test harness**

Run:

```powershell
cargo +nightly test --lib
```

Directory:

```text
C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os
```

Current expected failure:

```text
error[E0152]: duplicate lang item in crate `core`: `sized`
```

Do not claim kernel host unit tests pass until this target/test-harness issue is fixed.

---

## Task 12: Final Report Rules

**Files:**
- Modify: `README.md`
- Modify: `docs/SEAL_OS_GUIDE.md`

- [ ] **Step 1: Report only proven status**

Final audit must say:

```markdown
Seal OS 0.4.5 boots in WSL2/QEMU and reaches the desktop event loop.
T1-T10 theorem log gate passes.
Seal ABI/no-POSIX gate passes.
O(1) allocator source gate passes.
The proof screenshot gate passes.
Workspace Rust tests pass.
The kernel host unit test harness still fails with duplicate `core` lang item and remains a known gap.
Ubuntu-beating HFT/ML performance is not proven until benchmark data is collected.
```

- [ ] **Step 2: Include exact artifact paths**

Use:

```markdown
Serial proof: `kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log`
Screenshot proof: `kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.png`
Boot image: `kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img`
EFI binary: `kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.efi`
```

---

## Execution Order

Run the tasks in this order:

1. Task 1: Freeze Current Evidence
2. Task 2: Make The Screenshot Proof Machine-Checked
3. Task 3: Fix First Desktop Frame Composition
4. Task 4: Keep Desktop HUD And Icons Clean
5. Task 5: Strengthen O(1) Physical Allocation Proof
6. Task 6: Keep Seal ABI And No-POSIX Discipline Hard
7. Task 7: Aether-Lang Boot App Contract
8. Task 8: Convert Remaining Core Theorem Host Scripts To Rust
9. Task 9: GitHub Language Hygiene Without Lying
10. Task 10: Benchmark Contract Against Ubuntu
11. Task 11: Final Audit Command Set
12. Task 12: Final Report Rules

Do not start Task 8 before Tasks 1-7 are green. Converting more files while the boot proof is unstable will only hide the real failure.

---

## Agent Assignments

Use subagents only on independent work:

- Newton: VM runner, QEMU proof, screenshot gate, desktop proof.
- Noether: T1-T10 theorem log and O(1) allocator proof.
- Hopper: Seal ABI, no-POSIX, language hygiene, `.gitignore`, `.gitattributes`.
- Curie: Ubuntu benchmark contract and docs.
- Turing: Aether-Lang parser/interpreter boot app contract.
- Godel: Python-to-Rust theorem conversions, one module at a time.

Main session owns integration. Main session must review every subagent patch before running final gates.

---

## Stop Conditions

Stop and report immediately if any of these happen:

- QEMU proof does not reach `[BOOT] Seal OS desktop ready.`
- T1-T10 theorem log gate fails.
- Proof screen gate fails after the compositor fix.
- `--check-o1-allocator` finds an unbounded scan in runtime allocation.
- A conversion deletes a legacy host-script file without Rust tests replacing it.
- Any command needs network access. Ask first.

---

## Self-Review

Spec coverage:

- VM boot proof: Tasks 1, 2, 3, 11.
- T1-T10 theorem runtime integrity: Tasks 1, 5, 11.
- Seal ABI/no-POSIX: Task 6.
- O(1) allocation proof: Task 5.
- Aether-Lang OS app surface: Task 7.
- No Python as OS runtime: Tasks 8 and 9.
- GitHub language cleanup without deleting truth: Task 9.
- Better desktop/graphics cleanliness: Tasks 3 and 4.
- Ubuntu comparison: Task 10.
- Final hygiene/audit: Task 11.

No placeholder scan:

- No TBD/TODO steps.
- Every code-changing task includes exact files and concrete snippets.
- Every verification task includes exact command and expected output.

Type consistency:

- `check_proof_screen` uses `Path`, `fs`, and helper `read_ppm_number`, matching `seal-mkimage` style.
- Desktop code uses existing `Framebuffer`, `font`, `taskbar`, `WindowState`, and `Compositor::mark_dirty`.
- Audit commands use existing `cargo +stable` and `cargo +nightly` patterns.
