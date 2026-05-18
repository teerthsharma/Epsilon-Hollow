# 11 — Window Manager: Drag, Resize, Z-Order, Focus

## Goal

Polish the window manager into a fully interactive desktop environment with proper window drag, resize, z-order management, focus switching, and keyboard shortcuts.

## Current State

`kernel/seal-os/src/wm/` — 9 files:
- `compositor.rs` — composites windows onto framebuffer
- `window.rs` — window struct with position, size, buffer
- `desktop.rs` — renders background
- `cursor.rs` — mouse cursor rendering
- `taskbar.rs` — bottom taskbar
- `app_state.rs` — owns all app instances, dispatches events
- `event.rs` — input event types (key, mouse)
- `themes.rs` — color themes
- `mod.rs` — module declarations

The event loop in `main.rs:181-207` is real and working: polls keyboard/mouse events, dispatches to apps, recomposes framebuffer. Windows are created and rendered. Basic keyboard input works.

## Gap Analysis

- Window dragging: partially implemented or missing
- Window resizing: not implemented
- Z-order management: no bring-to-front on click
- Focus ring: no visual indicator of focused window
- Window close button: not functional
- Minimize/maximize: not implemented
- Alt-Tab window switching: not implemented
- Taskbar click to switch windows: may not be wired
- Window title bar rendering: basic or missing

## Implementation Steps

1. **Window drag**
   - Detect mouse down on title bar region (top 24px of window)
   - Track drag offset: `mouse_pos - window_pos` at drag start
   - On mouse move during drag: update `window.x = mouse_x - offset_x`, `window.y = mouse_y - offset_y`
   - On mouse up: end drag
   - Clamp window position to screen bounds

2. **Window resize**
   - Detect mouse down on window border (8px from any edge)
   - Track which edges are being dragged (N, S, E, W, NE, NW, SE, SW)
   - On mouse move: adjust window width/height and position accordingly
   - Enforce minimum window size (200×150)
   - Resize the window's internal buffer
   - Notify the app of resize so it can re-render

3. **Z-order management**
   - `Compositor` maintains window list in z-order (back to front)
   - On mouse click inside a window: move it to front of z-order
   - Compose in z-order: back windows first, front windows last
   - Hit testing: iterate windows front-to-back, first hit gets the event

4. **Focus management**
   - Only the frontmost window (or explicitly focused window) receives keyboard events
   - Visual focus indicator: highlighted title bar for focused window, dimmed for unfocused
   - Click on window → focus it (and bring to front)
   - Clicking desktop → unfocus all windows

5. **Window close button**
   - Render an "×" button in the title bar (top-right corner)
   - On click: remove window from compositor, notify app to clean up
   - Don't close the last window (or show confirmation for important apps)

6. **Minimize / Maximize**
   - Minimize button (−): hide window, show in taskbar
   - Maximize button (□): expand to full screen minus taskbar
   - Double-click title bar: toggle maximize
   - Store pre-maximize position/size for restore

7. **Alt-Tab window switching**
   - Alt held + Tab pressed: cycle focus through windows in MRU order
   - Optional: show a window switcher overlay with window thumbnails
   - Release Alt: switch to selected window

8. **Taskbar integration**
   - Each open window gets a button in the taskbar
   - Click taskbar button: focus/bring-to-front that window
   - If window is already focused: minimize it
   - Highlight the focused window's taskbar button

9. **Cursor shape changes**
   - Default arrow cursor
   - Resize cursors when hovering window borders (↔, ↕, ↗, ↘)
   - Text cursor (I-beam) when hovering text input areas
   - Grabbing cursor during window drag

## Dependencies

- **01-kernel-safety** (mouse event handling must not deadlock)

## Acceptance Criteria

- [ ] Windows can be dragged by title bar
- [ ] Windows can be resized by dragging edges/corners
- [ ] Clicking a window brings it to front
- [ ] Only the focused window receives keyboard input
- [ ] Close button removes window
- [ ] Minimize hides window, taskbar click restores
- [ ] Maximize fills screen, double-click restores
- [ ] Alt-Tab cycles through windows
- [ ] Taskbar shows all open windows
- [ ] QEMU smoke test: desktop renders, windows visible

## Files to Modify

- `kernel/seal-os/src/wm/compositor.rs` (z-order, hit testing)
- `kernel/seal-os/src/wm/window.rs` (resize, title bar, buttons)
- `kernel/seal-os/src/wm/app_state.rs` (event dispatch, focus management)
- `kernel/seal-os/src/wm/cursor.rs` (cursor shapes)
- `kernel/seal-os/src/wm/taskbar.rs` (window buttons, click handling)
- `kernel/seal-os/src/wm/desktop.rs` (click-to-unfocus)
- `kernel/seal-os/src/main.rs` (event loop: Alt-Tab handling)
