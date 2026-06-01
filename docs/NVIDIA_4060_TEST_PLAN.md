# NVIDIA RTX 4060 Test Plan for Seal OS

> **Hardware under test:** NVIDIA GeForce RTX 4060 (Ada Lovelace, PCI `10DE:2882`)  
> **Seal OS component:** `drivers/gpu/nvidia.rs`, `drivers/gpu/firmware_scanner.rs`  
> **Honest scope:** We can probe, identify, and read safe registers. We **cannot** load GSP firmware (signed, vendor-only) or run compute kernels without it.

---

## 1. What Works vs. What Doesn't

| Capability | Status | Why |
|-----------|--------|-----|
| PCI enumeration & BAR discovery | ✅ Works | Standard config-space reads |
| Architecture detection (Ada) | ✅ Works | `NV_PMC_BOOT_0` mask `0x00014000` |
| VBIOS read & signature check | ✅ Works | ROM BAR mapping, `0x55AA` validation |
| VRAM size heuristic | ⚠️ Heuristic | `PFB_CFG0` bits are not documented by NVIDIA; may read zero |
| GSP firmware load | ❌ Blocked | Signed blob required, redistribution prohibited |
| Compute queue creation | ❌ Blocked | Requires GSP-RM to be running |
| Shader execution / CUDA | ❌ Blocked | Requires compute queue + PTX/cubin loader |
| Display mode-setting | ❌ Blocked | Requires full NVIDIA DRM driver stack |

**Bottom line:** Seal OS will detect your RTX 4060, report it accurately, and fall back to the CPU topology path. The GPU driver provides **honest telemetry**, not acceleration.

---

## 2. Test Levels

### Level 1 — Windows Identity Check (No Tools Required)

**Goal:** Confirm the PCI IDs Seal OS expects match your actual card.

**Steps:**

1. Open PowerShell as Administrator.
2. Run:
   ```powershell
   wmic path Win32_VideoController get name,PNPDeviceID
   ```
3. Verify output contains something like:
   ```
   Name                      PNPDeviceID
   NVIDIA GeForce RTX 4060   PCI\VEN_10DE&DEV_2882&SUBSYS_...
   ```
4. Confirm `DEV_2882` (or `28A0` / `28E0` for mobile) matches the constants in:
   - `kernel/seal-os/src/drivers/gpu/nvidia.rs`

**Expected result:** `VEN_10DE` and `DEV_2882` appear.

---

### Level 2 — Linux Host Probe (Validates Register Offsets)

**Goal:** Run the host-side probe tool to verify that Seal OS's MMIO register offsets read sensible values on real Ada hardware.

**Prerequisites:**
- Bare-metal Linux (Ubuntu Live USB, dual-boot, or native install)
- **Not WSL1** (no PCI sysfs). WSL2 may work if PCI devices are exposed.
- Root access (required for BAR0 mmap)

**Steps:**

```bash
cd Epsilon-Hollow/tools/nv-4060-probe
cargo run --release
```

**What it does:**
1. Walks `/sys/bus/pci/devices` for NVIDIA vendor ID `0x10DE`
2. Enables PCI memory decoding and ROM BAR
3. Reads `NV_PMC_BOOT_0` (offset `0x000000`) and checks architecture mask
4. Reads `NV_PMC_ENABLE` (offset `0x000200`) and checks display-engine bit
5. Reads `NV_PFB_CFG0` (offset `0x100000`) and reports framebuffer heuristic
6. Dumps VBIOS ROM and verifies `0x55AA` signature

**Expected output for RTX 4060:**
```
Found NVIDIA GPU: GeForce RTX 4060 [10DE:2882]
BAR0 physical address: 00000000F0000000
NV_PMC_BOOT_0  = 00014xxx  => arch = 'Ada (AD10x)'
NV_PMC_ENABLE  = 00001xxx  => display_engine = true
NV_PFB_CFG0    = xxxxxxxx  => fb_heuristic = ??? MiB
[PASS] VBIOS signature 0x55AA OK
[PASS] Architecture mask matches Ada Lovelace
OVERALL: PASS
```

**If it fails:**
- `BAR0 resource file not found` → run with `sudo`
- `ROM resource not available` → some kernels disable ROM BAR by default; tool will skip
- `Expected Ada architecture mask 0x00014000` → your card is not RTX 4060, or register offset is wrong

---

### Level 3 — QEMU PCI Passthrough (End-to-End Boot Test)

**Goal:** Boot Seal OS inside QEMU with the RTX 4060 passed through. Verify the firmware scanner detects it and the kernel logs show the correct probe sequence.

**Prerequisites:**
- Linux host with IOMMU enabled (`intel_iommu=on` or `amd_iommu=on`)
- RTX 4060 bound to `vfio-pci` driver (not `nouveau` or `nvidia`)
- QEMU 6.0+ with VFIO support

**Steps:**

1. **Bind GPU to vfio-pci:**
   ```bash
   # Find your RTX 4060 BDF (e.g., 0000:01:00.0)
   lspci -nn | grep -i nvidia

   # Unbind from current driver
   echo "0000:01:00.0" | sudo tee /sys/bus/pci/devices/0000:01:00.0/driver/unbind

   # Bind to vfio-pci
   echo "10de 2882" | sudo tee /sys/bus/pci/drivers/vfio-pci/new_id
   ```

2. **Boot Seal OS with passthrough:**
   ```bash
   cd Epsilon-Hollow
   ./scripts/run_qemu_gpu.sh --passthrough 0000:01:00.0 --headless
   ```

3. **Watch serial output** for:
   ```
   [FWSCAN] NVIDIA 10DE:2882 — compute=false sdma=false fallback=...
   [NVGPU] Detected GeForce RTX 4060 [10DE:2882] -- arch=Ada (AD10x) ...
   [NVGPU] WARN: 3D acceleration requires signed firmware (GSP) -- not available
   [TOPO-ACCEL] No GPU acceleration available — using CPU fallback
   ```

**Expected result:**
- Scanner identifies RTX 4060 correctly
- `fallback_reason = FirmwareBlobMissing` is logged
- Seal OS continues booting and uses `CpuFallbackAccelerator`
- Shell command `gpu_benchmark` runs and prints CPU timing numbers

**Safety note:** PCI passthrough removes the GPU from the host. Ensure you have another GPU (iGPU or second dGPU) for host display, or run this via SSH from another machine.

---

### Level 4 — NVIDIA-SMI Cross-Reference (Optional Sanity Check)

**Goal:** Compare Seal OS's register reads against NVIDIA's own driver reporting.

**Steps (on Linux host with proprietary driver installed):**

```bash
nvidia-smi --query-gpu=name,pci.bus_id,vbios_version,memory.total,compute_cap \
           --format=csv
```

Compare against Seal OS probe output:
| Field | `nvidia-smi` | Seal OS probe | Match? |
|-------|-------------|---------------|--------|
| Product name | `NVIDIA GeForce RTX 4060` | `GeForce RTX 4060` | ✅ |
| PCI ID | `10DE:2882` | `10DE:2882` | ✅ |
| VBIOS version | `95.07.46.00.0C` | First 16 bytes @ ROM+0x18 | ≈ |
| VRAM | `8192 MiB` | `PFB_CFG0` heuristic | ⚠️ heuristic |

---

## 3. Known Limitations & Honest Claims

### What Seal OS CAN claim today
- "We detect RTX 4060 via PCI config space and BAR0 MMIO."
- "We read `NV_PMC_BOOT_0` and correctly identify Ada Lovelace architecture."
- "We validate VBIOS ROM signature `0x55AA` and extract version string."
- "We gracefully fall back to CPU topology when GSP is unavailable."

### What Seal OS CANNOT claim yet
- "We offload topology kernels to RTX 4060." (needs GSP firmware)
- "We achieve 15× speedup on RTX 4060." (needs compiled shaders + GSP)
- "We display graphics via RTX 4060." (needs full DRM driver + GSP + mode-setting)

### Path to compute on RTX 4060
1. **Obtain GSP firmware** — Extract `gsp.bin` from NVIDIA proprietary driver (cannot redistribute).
2. **Load GSP into GPU WPR** — Write blob to GPU write-protected region, boot GSP-RM.
3. **Create compute context via RM API** — Use NVIDIA's Resource Manager message protocol.
4. **Submit PTX/cubin kernels** — Compile topology kernels to PTX, load via RM.

Steps 1–4 are documented in `kernel/seal-os/src/drivers/gpu/nvidia.rs` as stubs. They require reverse-engineering NVIDIA's closed GSP-RM ABI, which is outside the scope of a research kernel unless you have significant resources.

---

## 4. Quick Reference: RTX 4060 Register Offsets

| Register | Offset | Seal OS constant | Expected value (RTX 4060) |
|----------|--------|-----------------|---------------------------|
| `NV_PMC_BOOT_0` | `0x000000` | `NV_PMC_BOOT_0` | `0x00014xxx` (Ada) |
| `NV_PMC_ENABLE` | `0x000200` | `NV_PMC_ENABLE` | Bit 12 = 1 (display) |
| `NV_PFB_CFG0` | `0x100000` | `NV_PFB_CFG0` | Undocumented heuristic |
| ROM BAR config | `0x30` | `PCI_ROM_BAR_OFFSET` | Upper 31 bits = phys addr |
| VBIOS signature | ROM + `0x00` | — | `0x55AA` |

---

## 5. Test Checklist

- [ ] Level 1: Windows `wmic` shows `VEN_10DE` `DEV_2882`
- [ ] Level 2: Linux `nv-4060-probe` reports `PASS` for Ada mask
- [ ] Level 2: `nv-4060-probe` VBIOS signature = `0x55AA`
- [ ] Level 3: QEMU passthrough boots Seal OS without crash
- [ ] Level 3: Serial log shows `[NVGPU] Detected GeForce RTX 4060`
- [ ] Level 3: Shell `gpu_benchmark` completes with CPU timing numbers
- [ ] (Optional) Level 4: `nvidia-smi` PCI ID matches Seal OS probe

---

*Last updated: 2026-06-01*  
*Seal OS v0.4.5 — honest about what works and what doesn't*
