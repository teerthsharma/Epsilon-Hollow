# 12 — Applications: Complete Built-in Apps

## Goal

Make every built-in application fully functional — the terminal executes real commands, the IDE edits real files, the calculator computes correctly, the media player decodes actual frames, and the games are playable.

## Current State

`kernel/seal-os/src/apps/` — 15 files:

| App | File | Status (estimated) |
|-----|------|--------------------|
| Shell | `shell.rs` | Partial — some commands work, others stub |
| Terminal | `terminal.rs` | Working — renders text, handles key input |
| Seal IDE | `seal_ide.rs` | Partial — renders code, editing may be limited |
| Calculator | `calculator.rs` | Partial — basic ops, scientific functions may stub |
| Media Player | `media_player.rs` | Stub — no codec, no decode, UI shell only |
| File Manager | `file_manager.rs` | Partial — may list ManifoldFS files |
| Snake | `snake.rs` | Working or near-working (game logic is simple) |
| Breakout | `breakout.rs` | Working or near-working |
| Warp Racer | `warp_racer.rs` | Partial — game engine may be incomplete |
| Settings | `settings.rs` | Partial — UI shell |
| Theorem Viewer | `theorem_viewer.rs` | Working — displays theorem status |
| Game Engine | `game_engine.rs` | Shared game framework |
| Help | `help.rs` | Working — displays help text |
| Clipboard | `clipboard.rs` | Working — in-memory clipboard |

## Implementation Steps

### Terminal + Shell

1. **Implement missing shell commands**
   - `ls` → `ManifoldFS::list_dir()`
   - `cat` → `ManifoldFS::read_file()`
   - `touch` → `ManifoldFS::create_file()`
   - `rm` → `ManifoldFS::delete_file()`
   - `teleport <src> <dest>` → `ManifoldFS::teleport()`
   - `search <query>` → `ManifoldFS::similarity_search()`
   - `wifi scan`, `wifi connect <ssid>` → WiFi driver
   - `bluetooth scan`, `bluetooth pair <device>` → Bluetooth driver
   - `ps` → list scheduler tasks
   - `free` → heap usage from memory module
   - `clear` → clear terminal buffer

2. **Command history** — up/down arrows cycle through previous commands
3. **Tab completion** — complete file names from ManifoldFS

### Seal IDE

4. **Text editing** — insert/delete characters at cursor, arrow key navigation
5. **Syntax highlighting** — color Aether-Lang keywords (use lexer token types)
6. **File open/save** — load from ManifoldFS, save back to ManifoldFS
7. **Line numbers** — display line numbers in left gutter

### Calculator

8. **Expression parser** — infix expression evaluation (not just button presses)
9. **Scientific functions** — sin, cos, tan, log, exp, sqrt, pow
10. **History** — show previous calculations

### Media Player

11. **Audio framework** — PC speaker beep generation (square wave via PIT)
12. **Simple formats** — WAV playback (PCM only, no compression)
13. **UI** — play/pause/stop buttons, progress bar, file browser
14. **Note**: Full codec support (MP4, MKV, FLAC) is out of scope for bare-metal. Focus on WAV/PCM and honest "format not supported" messages for others.

### Games

15. **Snake** — verify: food spawning, collision detection, score display, game over
16. **Breakout** — verify: ball physics, brick destruction, paddle control, lives
17. **Warp Racer** — implement: track rendering, ship movement, obstacle avoidance, speed scaling

### File Manager

18. **Directory tree view** — navigate ManifoldFS hierarchy
19. **File operations** — create, delete, rename, teleport via UI
20. **Point cloud visualization** — show file's S² point cloud in a mini-viewer

### Settings

21. **Theme selection** — switch between color themes (from `themes.rs`)
22. **Display settings** — resolution info (read-only from framebuffer)
23. **System info** — kernel version, uptime, heap usage, active theorems

## Dependencies

- **03-manifold-fs** (file operations for shell/IDE/file manager)
- **11-window-manager** (apps need proper focus, resize, close)
- **10-drivers-real** (WiFi/BT commands in shell need honest driver responses)

## Acceptance Criteria

- [ ] `ls`, `cat`, `touch`, `rm`, `teleport` work in shell
- [ ] IDE can open, edit, and save an Aether-Lang file
- [ ] Calculator evaluates `sin(3.14159/2)` correctly
- [ ] Snake and Breakout are fully playable
- [ ] File Manager shows ManifoldFS directory tree
- [ ] Media Player plays a WAV file (or honestly reports unsupported format)
- [ ] Settings shows real system info
- [ ] All apps render correctly in their windows

## Files to Modify

- All files in `kernel/seal-os/src/apps/`
- `kernel/seal-os/src/wm/app_state.rs` (app lifecycle)
