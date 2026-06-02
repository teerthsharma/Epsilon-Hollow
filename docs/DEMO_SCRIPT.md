# Seal OS 3-Minute Demo Script

> Target: investors, press, conference audiences  
> Tone: confident, technical but accessible, fast-paced  
> Total runtime: ~3:00

---

## 0:00–0:15 — Boot Sequence: UEFI → Theorem Verification → Desktop

**VOICEOVER:**
> "Seal OS boots in under a second from pure UEFI. No GRUB. No Multiboot. No 32-bit trampoline. Watch the theorem gate — ten formal theorems verified before the first pixel hits the screen."

**ON-SCREEN TEXT:**
- `Seal OS v0.4.6` (top-left, fade in)
- `T1/TSS … T10/WPHB VERIFIED` (center, green checkmarks)
- `Desktop ready.` (bottom)

**ACTION:**
- Start QEMU recording.
- Serial log scrolls theorem verification lines.
- Desktop frame blits at 1024×768.

**TECHNICAL NOTE:**
- Use `run-qemu.ps1 -HeadlessProof` to get deterministic boot timing.
- If boot is too fast for narration, insert a 0.5× slow-mo on the theorem scroll.

---

## 0:15–0:45 — Shell Demo: `look`, `peek`, `move`, `search`

**VOICEOVER:**
> "This is SealShell — plain English, no cryptic flags. `look` lists files. `peek` shows contents. `move` doesn't copy bytes — it teleports metadata. And `search` is content-addressable by geometry, not filename."

**ON-SCREEN TEXT (side panel):**
```
> look
  documents/
  trades.csv
  notes.txt

> peek trades.csv
  date,price,volume
  2026-05-30,104.2,1.4M

> move trades.csv documents/
  [ManifoldFS] Teleported metadata in 1 tick

> search "price"
  trades.csv  (sim=0.94)
```

**ACTION:**
- Type commands at ~1.5× natural speed.
- Highlight the `Teleported metadata in 1 tick` line with a subtle glow.

**TECHNICAL NOTE:**
- Pre-populate a small `trades.csv` in the ramfs root before recording.
- Use `seal` command after to show theorem stats in background.

---

## 0:45–1:15 — Apps: Calculator, Theorem Viewer, File Manager

**VOICEOVER:**
> "Seal OS ships with a full desktop environment — not a toy. Calculator with expression history and glow effects. A live theorem dashboard showing T1 through T10 status in real time. And a topological file manager where every file is a point cloud on the sphere."

**ON-SCREEN TEXT:**
- `Calculator` — `sin(π/4) = 0.7071`
- `Theorem Viewer` — green bars for T1–T5, blue bars for T6–T10
- `File Manager` — Voronoi cell icons (`Cell 0` … `Cell 7`)

**ACTION:**
- Launch calculator, compute `sin(π/4)`.
- Switch to theorem viewer; pan across T1–T10 bars.
- Open file manager; hover over `trades.csv` to show Voronoi cell assignment.

**TECHNICAL NOTE:**
- Click the start-menu icon, then each app. Keep mouse movements smooth.
- If theorem viewer data looks static, run `info trades.csv` in shell first to populate counters.

---

## 1:15–1:45 — Unique Features: Tensor Render & Topological 3D

**VOICEOVER:**
> "Here is where it gets weird — in a good way. `tensor render` loads a CSV and projects it as a hyperbolic manifold in 3D. The camera auto-rotates around the tensor embedding. This is not a screensaver. This is how Seal OS sees data — as geometry you can navigate."

**ON-SCREEN TEXT:**
- `tensor render trades.csv`
- `Shape: [14, 3]  Min: 98.1  Max: 112.4`
- `Manifold: hyperbolic  |  Camera: auto-orbit`

**ACTION:**
- Type `tensor render trades.csv` in shell.
- Full-screen tensor viewer opens; wireframe rotates slowly.
- Brief overlay: cosine-similarity formula `a·b / (|a||b|)` fades in/out.

**TECHNICAL NOTE:**
- The tensor viewer uses `tensor_viz::tensor_to_point_cloud` and `topo_render`.
- If CSV parsing is slow, pre-load with `PENDING_CSV` mutex via a tiny Aether script before recording.

---

## 1:45–2:15 — Games: Snake, Breakout, Warp Racer

**VOICEOVER:**
> "Yes, it plays games. Snake. Breakout. And Warp Racer — our own topology-inspired arcade game. Because an operating system isn't real until you can procrastinate on it."

**ON-SCREEN TEXT:**
- `snake` — score counter top-right
- `breakout` — lives + level
- `warp` — speed gauge, hyperbolic tunnel

**ACTION:**
- Launch Snake, play ~5 seconds, score 2–3 apples.
- Switch to Breakout, clear one row.
- Switch to Warp Racer, show 3 seconds of tunnel flight.
- Use Alt-Tab style window switcher or click taskbar.

**TECHNICAL NOTE:**
- Snake uses arrow keys; Breakout uses left/right.
- Warp Racer may need a brief key-hold to show motion.
- Keep each game segment under 8 seconds to stay within the 3:00 window.

---

## 2:15–2:45 — Technical Depth: T1–T5 Status & Governor Epsilon

**VOICEOVER:**
> "Under the hood, T1 through T5 are not just verified at boot — they are active in every hot path. Voronoi placement. Spectral prefetch. Geodesic memory consolidation. Adaptive governor convergence. Hyperbolic capacity separation. The epsilon value you see here is the live PD controller gain derived from workload geometry."

**ON-SCREEN TEXT:**
```
[T1/TSS]  Voronoi cells: 8  |  Lookups: 1,247
[T2/SCM]  Prefetch accuracy: 78%  |  Predictions: 312
[T3/GMC]  Entropy: 1.18  |  Merges: 4
[T4/AGCR] Epsilon: 0.1000  |  Governor: online
[T5/HCS]  Hyperbolic ratio: 4.2  |  Depth: 3
```

**ACTION:**
- Open theorem viewer again, now with live counters.
- Open a second terminal; run `seal` to print system info.
- Split-screen or quick cuts between theorem viewer and shell output.

**TECHNICAL NOTE:**
- Run a few `search` and `move` commands beforehand to pump the counters.
- The `seal` command prints `T1/TSS Voronoi cells: 8, Betti-0: 8` and scheduler epsilon.

---

## 2:45–3:00 — Closing

**VOICEOVER:**
> "Seal OS. Bare metal. Pure Rust. Ten theorem-gated subsystems. One unified geometry. Where every byte is topology — and every kernel decision is a proof."

**ON-SCREEN TEXT:**
- `Seal OS — where every byte is geometry.`
- `github.com/teerthsharma/epsilon-hollow  |  MIT License`
- `Contributors & research collaborators welcome.`

**ACTION:**
- Desktop fades to dark gradient.
- Text centers. Logo (if available) appears.
- Hold final frame for 3 seconds.

**TECHNICAL NOTE:**
- Fade can be done in post-production; raw capture ends on the desktop.
- Keep serial log running until fade-out for proof continuity.

---

## Post-Production Checklist

- [ ] Stabilize boot scroll with 0.5× slow-mo if needed
- [ ] Add theorem overlay graphics (T1–T10 icons) during technical depth segment
- [ ] Highlight `Teleported metadata in 1 tick` with subtle glow
- [ ] Add cosine-similarity formula overlay during tensor render
- [ ] Background music: ambient/electronic, low volume, no lyrics
- [ ] Export at 1080p (upscale from 1024×768 center-crop or add dark border)
- [ ] Include serial log as PDF appendix for technical audiences
