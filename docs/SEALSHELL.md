# SealShell — The Human-Friendly Shell

SealShell is Seal OS's native shell. Commands are plain English — no cryptic flags, no man pages needed.

## Command Reference

| Command | Description |
|---|---|
| `open <folder>` | Enter a folder |
| `back` | Go up one folder |
| `home` | Go to root |
| `look` / `look <folder>` | List contents |
| `peek <file>` | Show file contents |
| `create <name>` | Create a new folder |
| `write <file> <content>` | Create or overwrite a file |
| `copy <file>` | Copy a file to clipboard |
| `paste` | Paste copied file into current folder |
| `move <file> <dest>` | Teleport file metadata to another folder without rewriting file bytes on the same filesystem |
| `rename <old> <new>` | Rename a file or folder |
| `delete <file>` | Remove a file |
| `search <query>` | Content-addressable search |
| `info <file>` | File details (size, Voronoi cell, payload) |
| `seal` | System info + theorem status |
| `tasks` | Show running processes |
| `race <size>` | Benchmark teleport vs copy |
| `stats` | Filesystem statistics |
| `history` | Show command history |
| `clear` | Clear terminal |
| `help` / `help <cmd>` | Show handbook or per-command help |
| `run <script.aether>` | Execute an Aether-Lang script |
| `aether` | Enter Aether-Lang interactive mode |
| `install <pkg>` | Install a package via ManifoldPkg; `<name>.eph` installs local bytes, bare names use the registry path |
| `remove <pkg>` | Remove a package |
| `packages` | List installed packages |
| `update` | Report registry refresh status |
| `wifi` / `wifi connect <ssid>` | WiFi status or connect |
| `bluetooth` | Bluetooth device scan |
| `theme <name>` | Change theme (dark, light, seal, matrix) |
| `set <key> <value>` | Change a setting |
| `prefetch` | Show aether-link prefetch stats |
| `snake` / `breakout` / `warp` | Launch built-in games |

## Seal-Native Commands

SealShell intentionally uses Seal-native verbs (`look`, `open`, `peek`, `move`, `search`, `tasks`, `seal`). It does not emulate a Unix shell; the command surface is shaped around ManifoldFS and theorem status.

## File Operations

All file operations go through ManifoldFS. Files are 64-point clouds on S^2. Moving a file between folders has a bounded metadata-rewiring core. The same-filesystem persistent path updates metadata and keeps raw bytes in place; cross-mount moves still use copy/delete fallback.
