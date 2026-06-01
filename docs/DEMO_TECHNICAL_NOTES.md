# Demo Recording — Technical Notes

This document covers how to capture, record, and post-produce a Seal OS demo video from QEMU or Oracle VM VirtualBox.

---

## 1. Recording QEMU Output

### Display Capture

QEMU offers several display backends. For demo recording, use **SDL** or **GTK** with standard display scaling:

```powershell
cd kernel\seal-os
qemu-system-x86_64 `
  -drive if=virtio,format=raw,file=target\x86_64-unknown-uefi\release\seal-os.img `
  -bios "C:\Program Files\qemu\share\edk2-x86_64\code.fd" `
  -display sdl `
  -serial file:serial.log `
  -m 4096
```

Alternatively, capture the framebuffer directly to video with `-display dbus` or use OBS Studio window-capture on the QEMU SDL window.

### SPICE Protocol (Advanced)

```powershell
qemu-system-x86_64 `
  -drive if=virtio,format=raw,file=target\x86_64-unknown-uefi\release\seal-os.img `
  -bios "C:\Program Files\qemu\share\edk2-x86_64\code.fd" `
  -spice port=5924,disable-ticketing `
  -display spice-app `
  -serial file:serial.log
```

Use **Virt-Viewer** (`remote-viewer spice://localhost:5924`) and capture that window for cleaner compositing.

### Headless Frame Capture (For Still Proofs)

```powershell
.\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
```

This writes:
- `qemu-proof\screen.ppm` — raw framebuffer
- `qemu-proof\screen.png` — converted screenshot
- `qemu-proof\serial.log` — full serial output

These are **audit artifacts**, not demo video, but the PPM can be converted to frames with FFmpeg.

---

## 2. Capturing Serial Log Alongside Video

The serial log is essential for technical credibility and post-production theorem overlays.

### QEMU

Append `-serial file:serial.log` to the command line. The log receives:
- Boot milestones
- Theorem verification lines (`[THEOREM] T1/TSS VERIFIED` … `T10/WPHB VERIFIED`)
- Benchmark markers
- Aether runtime proof
- Desktop readiness sentinels

### VirtualBox

Configure COM1 before first boot:

```powershell
VBoxManage modifyvm "SealOS" --uart1 0x3F8 4 --uartmode1 file C:\tmp\seal-os-serial.log
```

Or use the automated smoke script which sets this automatically:

```powershell
.\smoke-vbox.ps1 -Seconds 240
```

The serial log will be written to `vbox-smoke\serial.log`.

### Synchronizing Serial with Video

Both QEMU and VirtualBox serial logs are timestamp-free (monotonic boot ticks). Use the desktop-ready sentinel as the sync point:

```text
[BOOT] Seal OS desktop ready.
```

Align video frame "desktop appears" with this line in the serial log. All earlier boot events map backward from this anchor.

---

## 3. Recommended Resolution

Seal OS renders at **1024×768** natively. The framebuffer is set during UEFI GOP initialization and the compositor blits at this resolution.

| Purpose | Resolution | Notes |
|---------|-----------|-------|
| Native OS render | 1024×768 | Do not force other GOP modes; capping is automatic |
| Demo video export | 1920×1080 | Center-crop or pillarbox with dark border |
| Conference projector | 1920×1080 | Same as above; text remains crisp |
| Social media clips | 1080×1920 (vertical) | Crop to desktop center; may cut taskbar |

**QEMU tip:** If QEMU offers an oversized GOP mode (common on high-DPI hosts), Seal OS caps to a sane resolution automatically. If you need to force 1024×768 for pixel-perfect recording, verify the GOP mode list with `-vga std` and select the matching mode.

---

## 4. Slowing Down Boot for Narration

If the boot sequence is too fast to narrate live, you have three options:

### Option A: Slow-Mo in Post (Recommended)

Record at full speed. In post-production, apply 0.5× speed to the boot scroll segment (0:00–0:15). The theorem verification lines are text-heavy and read well at half speed.

### Option B: Pause in Debugger (Advanced)

Attach QEMU monitor (`-monitor stdio`) and issue:

```
(qemu) stop
(qemu) cont
```

This is only useful for still-frame capture, not smooth video.

### Option C: Serial Log Voiceover

Record narration over a black screen while the serial log scrolls as on-screen text. This decouples narration pacing from boot timing entirely.

---

## 5. Post-Production Tips

### Theorem Overlay Graphics

During the technical depth segment (2:15–2:45), overlay stylized theorem badges:

```
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ T1  │  │ T2  │  │ T3  │  │ T4  │  │ T5  │
│ TSS │  │ SCM │  │ GMC │  │AGCR │  │ HCS │
│ 🟢  │  │ 🟢  │  │ 🟢  │  │ 🟢  │  │ 🟢  │
└─────┘  └─────┘  └─────┘  └─────┘  └─────┘
```

Use the serial log to populate live counters (lookups, prefetch accuracy, epsilon) so overlays match the exact run.

### Color Palette

Seal OS uses a dark theme. Recommended overlay colors:
- Background: `#00101018` (matches theorem viewer)
- Active text: `#44CC66` (green)
- Label text: `#A0A0B0` (gray)
- Value text: `#89B4FA` (blue)
- Accent: `#CBA6F7` (purple)

### Audio

- **Music:** Ambient electronic, 60–90 BPM, no lyrics, -20 LUFS
- **Voiceover:** Record separately in a treated room or with a noise gate
- **System sounds:** Seal OS does not currently emit audio; add subtle UI click SFX in post if desired

### Export Settings

| Platform | Codec | Container | Bitrate |
|----------|-------|-----------|---------|
| YouTube / general | H.264 | MP4 | 8–12 Mbps |
| Conference loop | ProRes 422 LT | MOV | 150 Mbps |
| Social (Twitter/X) | H.264 | MP4 | 6 Mbps, <512 MB |

---

## 6. Recording Checklist

Before each take:

- [ ] Build fresh release image: `cargo +nightly build --release` + `seal-mkimage`
- [ ] Verify image: `seal-mkimage --verify`
- [ ] Pre-populate demo files (`trades.csv`, a few folders) in ramfs or disk
- [ ] Start serial log capture
- [ ] Set QEMU display to 1024×768 native (or capture window at 1:1 pixel ratio)
- [ ] Run one dry boot to confirm theorem gate and desktop timing
- [ ] Clear mouse from screen before boot
- [ ] Disable host notifications / screen savers

After each take:

- [ ] Check serial log contains all T1–T10 verified lines
- [ ] Check video has no host UI artifacts (taskbar, notifications)
- [ ] Rename files with take number: `demo_take_01.mp4`, `serial_take_01.log`

---

## 7. One-Command Demo Prep

From PowerShell:

```powershell
cd kernel\seal-os
cargo +nightly build --release
cd ..\seal-mkimage
cargo +stable run --release
cd ..\seal-os
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1
```

Then attach your screen recorder to the QEMU window and begin.
