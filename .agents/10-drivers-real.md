# 10 — Real Drivers: Replace Simulations with Hardware Access

## Goal

Replace all `[Sim]`-labeled driver stubs with real hardware drivers or honest virtio emulations — so that every driver either talks to real hardware or clearly delegates to a tested virtio device under QEMU.

## Current State

| Driver | File | Status |
|--------|------|--------|
| PCI | `drivers/pci.rs` (79 lines) | **Real** — actual port I/O at 0xCF8/0xCFC |
| Serial | `drivers/serial.rs` (67 lines) | **Real** — COM1 port I/O |
| Interrupts | `drivers/interrupts.rs` (293 lines) | **Real** — IDT, PIC, keyboard/mouse IRQ |
| GPU | `drivers/gpu.rs` (66 lines) | PCI detection real; `blit()`/`fill_rect()` empty stubs |
| WiFi | `drivers/wifi.rs` (98 lines) | **All simulated** — hardcoded fake networks |
| Bluetooth | `drivers/bluetooth.rs` (90 lines) | **All simulated** — hardcoded fake devices |
| USB/xHCI | `drivers/usb/` (252 lines total) | Register structs correct; all runtime simulated |
| Network | `drivers/net/` (283 lines total) | **All simulated** — no NIC, no packets |

## Implementation Steps

1. **GPU: Implement virtio-gpu 2D operations**
   - Detect virtio-gpu via PCI (vendor 0x1AF4, device 0x1050)
   - Implement `VIRTIO_GPU_CMD_RESOURCE_CREATE_2D`, `TRANSFER_TO_HOST_2D`, `RESOURCE_FLUSH`
   - Wire `blit()` and `fill_rect()` to virtio-gpu commands
   - Fallback: if no virtio-gpu, use direct framebuffer (current behavior)

2. **USB: Implement real xHCI initialization**
   - Read capability registers from PCI BAR (structs already correct in `xhci.rs`)
   - Perform controller reset sequence (USBCMD.HCRST)
   - Initialize device context base address array
   - Set up command ring and event ring
   - Enable port power and detect connected devices
   - For connected devices: perform USB SET_ADDRESS, GET_DESCRIPTOR
   - Implement HID class driver for USB keyboards/mice (supplement PS/2)
   - Implement mass storage class for USB drives (wire to block device trait from plan 03)

3. **WiFi: Replace simulation with honest status**
   - Option A: Remove fake scan results, report "No WiFi hardware detected" on bare metal
   - Option B: If virtio-net is available, present it as a "wired connection" instead of fake WiFi
   - Remove all `[Sim]` prefixed hardcoded data
   - Keep the `WifiDriver` API for future real WiFi chipset support (ath9k, iwlwifi)

4. **Bluetooth: Replace simulation with honest status**
   - Remove fake device list (`[Sim] AirPods Pro`, etc.)
   - Report "No Bluetooth hardware detected" on bare metal
   - Keep `BluetoothDriver` API for future USB-BT adapter support
   - If USB subsystem detects a BT adapter (USB class 0xE0), initialize HCI transport

5. **GPU VRAM detection**
   - Read VRAM size from PCI BAR size or GPU-specific register
   - Report actual VRAM in GPU driver status instead of hardcoded 0

## Dependencies

- **01-kernel-safety** (IRQ handling for USB/GPU interrupts)
- **02-memory-allocator** (DMA buffers for virtio queues)
- **09-network-stack** (shares virtio infrastructure patterns)

## Acceptance Criteria

- [ ] No `[Sim]` prefixed strings in any driver output
- [ ] GPU `blit()` and `fill_rect()` do something (virtio-gpu or framebuffer fallback)
- [ ] USB xHCI: controller reset succeeds under QEMU with `-device qemu-xhci`
- [ ] USB: at least enumerate connected devices (vendor/product ID)
- [ ] WiFi/Bluetooth: honest "not detected" instead of fake device lists
- [ ] All existing functionality preserved (keyboard, mouse, serial still work)
- [ ] QEMU smoke test passes

## Files to Modify

- `kernel/seal-os/src/drivers/gpu.rs`
- `kernel/seal-os/src/drivers/wifi.rs`
- `kernel/seal-os/src/drivers/bluetooth.rs`
- `kernel/seal-os/src/drivers/usb/mod.rs`
- `kernel/seal-os/src/drivers/usb/xhci.rs`
- `kernel/seal-os/src/drivers/usb/hid.rs`
- `kernel/seal-os/src/drivers/usb/mass_storage.rs`

## Files to Create

- `kernel/seal-os/src/drivers/virtio_gpu.rs`
- `kernel/seal-os/src/drivers/virtio.rs` (shared virtio infrastructure: virtqueue, descriptor management)
