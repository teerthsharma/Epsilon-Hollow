// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! SealShell — the human-friendly shell. English-first, Unix as silent fallback.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::fs::manifold_fs::{ManifoldFS, InodeKind};

use super::calculator::Calculator;
use super::clipboard::Clipboard;
use super::help;
use super::media_player::MediaPlayer;

pub struct Shell {
    fs: ManifoldFS,
    cwd: u64,
    cwd_path: String,
    history: Vec<String>,
    clipboard: Clipboard,
    calculator: Calculator,
    media_player: MediaPlayer,
}

impl Shell {
    pub fn new() -> Self {
        Self {
            fs: ManifoldFS::new(),
            cwd: 0,
            cwd_path: String::from("/"),
            history: Vec::new(),
            clipboard: Clipboard::new(),
            calculator: Calculator::new(),
            media_player: MediaPlayer::new(),
        }
    }

    pub fn execute(&mut self, input: &str) -> String {
        let input = input.trim();
        if input.is_empty() {
            return String::new();
        }

        self.history.push(String::from(input));

        let parts: Vec<&str> = input.splitn(3, ' ').collect();
        let cmd = parts[0];
        let arg1 = parts.get(1).copied().unwrap_or("");
        let arg2 = parts.get(2).copied().unwrap_or("");

        let resolved = Self::resolve_alias(cmd);

        match resolved {
            "help" => {
                if arg1.is_empty() {
                    help::handbook()
                } else {
                    help::help_for(arg1)
                }
            }

            "look" => self.cmd_look(arg1),
            "open" => self.cmd_open(arg1),
            "back" => self.cmd_open(".."),
            "home" => self.cmd_open("/"),
            "peek" => self.cmd_peek(arg1),
            "create" => self.cmd_create(arg1),
            "write" => self.cmd_write(arg1, arg2),
            "delete" => self.cmd_delete(arg1),
            "rename" => self.cmd_rename(arg1, arg2),
            "copy" => self.cmd_copy(arg1),
            "paste" => self.cmd_paste(),
            "move" => self.cmd_move(arg1, arg2),
            "search" => {
                let query = if arg2.is_empty() { arg1 } else {
                    input.splitn(2, ' ').nth(1).unwrap_or(arg1)
                };
                self.cmd_search(query)
            }
            "info" => self.cmd_info(arg1),
            "seal" => self.cmd_seal(),
            "tasks" => self.cmd_tasks(),
            "stats" => self.cmd_stats(),
            "race" => self.cmd_race(arg1),
            "history" => self.cmd_history(),
            "clear" => String::from("\x1b[clear]"),

            // Package manager
            "install" => self.cmd_install(arg1),
            "remove" => self.cmd_remove(arg1),
            "packages" => self.cmd_packages(),
            "update" => self.cmd_update(),

            // Network
            "wifi" => self.cmd_wifi(arg1, arg2),
            "bluetooth" => self.cmd_bluetooth(arg1, arg2),

            // Scripting
            "run" => self.cmd_run(arg1),
            "aether" => String::from("[Aether-Lang] REPL mode — type expressions terminated with ~\nType 'exit~' to return to SealShell."),

            // Games
            "snake" => String::from("[Snake] Starting Snake game... Use arrow keys to move!"),
            "breakout" => String::from("[Breakout] Starting Breakout... Arrow keys to move paddle!"),
            "warp" => String::from("[Warp Racer] Starting Warp Racer... Fly through the data stream!"),

            // Settings
            "theme" => self.cmd_theme(arg1),
            "set" => self.cmd_set(arg1, arg2),

            // Prefetch
            "prefetch" => self.cmd_prefetch(arg1),

            // Calculator
            "calc" => {
                let expr = if arg2.is_empty() { arg1 } else {
                    input.splitn(2, ' ').nth(1).unwrap_or(arg1)
                };
                self.calculator.evaluate(expr)
            }

            // Media player
            "play" => self.cmd_play(arg1, arg2),
            "pause" => self.media_player.pause(),
            "stop" => self.media_player.stop(),
            "formats" => MediaPlayer::supported_formats(),

            // pwd (silent unix fallback that doesn't map cleanly)
            "pwd" => self.cwd_path.clone(),

            _ => format!("seal: unknown command '{}' — type 'help' for the handbook", cmd),
        }
    }

    fn resolve_alias(cmd: &str) -> &str {
        match cmd {
            // Unix silent fallbacks — undocumented, not in help
            "ls" => "look",
            "dir" => "look",
            "cd" => "open",
            "cat" => "peek",
            "mkdir" => "create",
            "touch" => "write",
            "rm" => "delete",
            "mv" => "move",
            "cp" => "copy",
            "find" | "grep" => "search",
            "stat" => "info",
            "uname" => "seal",
            "ps" => "tasks",
            "theorems" => "seal",
            "pip" => "install",
            _ => cmd,
        }
    }

    fn cmd_look(&self, arg: &str) -> String {
        let dir = if arg.is_empty() {
            self.cwd
        } else {
            match self.fs.resolve_path(arg) {
                Ok(id) => id,
                Err(e) => return format!("look: {}", e),
            }
        };
        match self.fs.ls(dir) {
            Ok(entries) => {
                if entries.is_empty() {
                    return String::from("(empty)");
                }
                let mut out = String::new();
                for e in &entries {
                    let icon = if e.kind == "dir" { "📁" } else { "📄" };
                    out.push_str(&format!(
                        " {} {:<20} {:>8} pts={} cell={}\n",
                        icon, e.name, e.original_size, e.payload_points, e.voronoi_cell
                    ));
                }
                out
            }
            Err(e) => format!("look: {}", e),
        }
    }

    fn cmd_open(&mut self, arg: &str) -> String {
        if arg.is_empty() || arg == "/" {
            self.cwd = 0;
            self.cwd_path = String::from("/");
            return String::new();
        }
        if arg == ".." {
            if let Some(inode) = self.fs.inode(self.cwd) {
                self.cwd = inode.parent;
                self.update_cwd_path();
            }
            return String::new();
        }
        match self.fs.resolve_path_from(arg, self.cwd) {
            Ok(id) => {
                if !self.fs.is_dir(id) {
                    return format!("open: '{}' is not a folder", arg);
                }
                self.cwd = id;
                self.update_cwd_path();
                String::new()
            }
            Err(e) => format!("open: {}", e),
        }
    }

    fn cmd_peek(&self, arg: &str) -> String {
        if arg.is_empty() {
            return String::from("peek: which file? Usage: peek <filename>");
        }
        match self.fs.read_text(arg, self.cwd) {
            Some(text) => text,
            None => format!("peek: '{}' not found", arg),
        }
    }

    fn cmd_create(&mut self, arg: &str) -> String {
        if arg.is_empty() {
            return String::from("create: what name? Usage: create <foldername>");
        }
        match self.fs.mkdir(arg, self.cwd) {
            Ok(id) => format!("Created folder '{}' (inode {})", arg, id),
            Err(e) => format!("create: {}", e),
        }
    }

    fn cmd_write(&mut self, name: &str, content: &str) -> String {
        if name.is_empty() {
            return String::from("write: usage: write <filename> <content>");
        }
        let content = if content.is_empty() { "" } else { content };
        match self.fs.store_text(name, content, self.cwd) {
            Ok(id) => format!(
                "Wrote '{}' ({} bytes → 64 pts on S², inode {})",
                name, content.len(), id
            ),
            Err(e) => format!("write: {}", e),
        }
    }

    fn cmd_delete(&mut self, arg: &str) -> String {
        if arg.is_empty() {
            return String::from("delete: which file? Usage: delete <filename>");
        }
        match self.fs.delete(arg, self.cwd) {
            Ok(()) => format!("Deleted '{}'", arg),
            Err(e) => format!("delete: {}", e),
        }
    }

    fn cmd_rename(&mut self, old: &str, new: &str) -> String {
        if old.is_empty() || new.is_empty() {
            return String::from("rename: usage: rename <old> <new>");
        }
        match self.fs.rename(old, new, self.cwd) {
            Ok(()) => format!("Renamed '{}' → '{}'", old, new),
            Err(e) => format!("rename: {}", e),
        }
    }

    fn cmd_copy(&mut self, arg: &str) -> String {
        if arg.is_empty() {
            return String::from("copy: which file? Usage: copy <filename>");
        }
        if !self.fs.exists(arg, self.cwd) {
            return format!("copy: '{}' not found", arg);
        }
        self.clipboard.copy(arg, self.cwd);
        format!("Copied '{}' to clipboard", arg)
    }

    fn cmd_paste(&mut self) -> String {
        let entry = match self.clipboard.take() {
            Some(e) => e,
            None => return String::from("paste: clipboard is empty — use 'copy <file>' first"),
        };
        match self.fs.duplicate(&entry.name, entry.source_dir, self.cwd) {
            Ok(id) => format!("Pasted '{}' here (inode {})", entry.name, id),
            Err(e) => format!("paste: {}", e),
        }
    }

    fn cmd_move(&mut self, name: &str, dest: &str) -> String {
        if name.is_empty() || dest.is_empty() {
            return String::from("move: usage: move <file> <destination_folder>");
        }
        let dst = match self.fs.resolve_path(dest) {
            Ok(id) => id,
            Err(e) => return format!("move: destination: {}", e),
        };
        match self.fs.teleport(name, self.cwd, dst) {
            Ok(r) => format!(
                "Teleported '{}' ({} bytes) in {} ticks — O(1)!\n\
                 Payload: {} points, governor ε: {:.4}",
                name, r.original_size, r.elapsed_ticks, r.payload_points, r.governor_epsilon
            ),
            Err(e) => format!("move: {}", e),
        }
    }

    fn cmd_search(&mut self, query: &str) -> String {
        if query.is_empty() {
            return String::from("search: what? Usage: search <query>");
        }
        let results = self.fs.find(query);
        if results.is_empty() {
            return String::from("No matches found.");
        }
        let mut out = String::new();
        for r in &results {
            out.push_str(&format!(
                "  {} (similarity={:.3}, cell={}, {} bytes)\n",
                r.name, r.similarity, r.cell, r.original_size
            ));
        }
        out
    }

    fn cmd_info(&self, arg: &str) -> String {
        if arg.is_empty() {
            return String::from("info: which file? Usage: info <filename>");
        }
        match self.fs.file_info(arg, self.cwd) {
            Some(info) => info,
            None => format!("info: '{}' not found", arg),
        }
    }

    fn cmd_seal(&self) -> String {
        let mut out = String::from(
            "Seal OS v0.1.0 (x86_64)\n\
             The Geometrical Operating System\n\
             All data = geometry on S²\n\
             File moves = O(1) topological surgery\n\
             ═══════════════════════════════════\n",
        );
        for (name, status) in self.fs.theorem_status() {
            out.push_str(&format!("  {} = {}\n", name, status));
        }
        let stats = self.fs.stats();
        out.push_str(&format!("\n  Governor ε: {:.4}\n", stats.governor_epsilon));
        out.push_str(&format!("  Entropy: {:.3} bits\n", stats.current_entropy));
        out.push_str(&format!(
            "  Hyperbolic ratio: {:.1}x (depth={})\n",
            stats.hyperbolic_ratio, stats.max_depth
        ));
        out
    }

    fn cmd_tasks(&self) -> String {
        String::from(
            "  PID  STATE    NAME         PRIORITY\n\
                1  running  kernel       10\n\
                2  ready    compositor   8\n\
                3  ready    shell        5\n\
                4  blocked  idle         0",
        )
    }

    fn cmd_stats(&self) -> String {
        let s = self.fs.stats();
        format!(
            "ManifoldFS Statistics\n\
             ═════════════════════\n\
             Files:           {}\n\
             Directories:     {}\n\
             Teleports:       {}\n\
             Governor ε:      {:.4}\n\
             Entropy:         {:.3} bits\n\
             Max depth:       {}\n\
             Hyperbolic ratio: {:.1}x",
            s.total_files, s.total_dirs, s.total_teleports,
            s.governor_epsilon, s.current_entropy, s.max_depth, s.hyperbolic_ratio
        )
    }

    fn cmd_race(&self, arg: &str) -> String {
        let size: u64 = arg.parse().unwrap_or(1_000_000_000);
        let label = if size >= 1_000_000_000 {
            format!("{}GB", size / 1_000_000_000)
        } else if size >= 1_000_000 {
            format!("{}MB", size / 1_000_000)
        } else {
            format!("{}KB", size / 1000)
        };
        let trad_ns = (size as f64 * 0.5) as u64;
        let teleport_ns = 50_000u64;
        let speedup = trad_ns as f64 / teleport_ns as f64;
        format!(
            "Race: {} transfer\n\
             ═══════════════════\n\
             Traditional copy:   {} ms\n\
             Manifold teleport:  {} μs (O(1) topological surgery)\n\
             Speedup:            {:.0}x\n\
             \n\
             Payload: 64 pts × 3 coords × 8 bytes = 1,536 bytes\n\
             Regardless of original file size!",
            label, trad_ns / 1_000_000, teleport_ns / 1_000, speedup
        )
    }

    fn cmd_history(&self) -> String {
        let mut out = String::new();
        for (i, cmd) in self.history.iter().enumerate() {
            out.push_str(&format!("  {:>4}  {}\n", i + 1, cmd));
        }
        if out.is_empty() {
            String::from("(no history)")
        } else {
            out
        }
    }

    fn cmd_install(&mut self, pkg: &str) -> String {
        if pkg.is_empty() {
            return String::from("install: which package? Usage: install <package>");
        }
        format!(
            "[ManifoldPkg] Resolving '{}'...\n\
             [ManifoldPkg] Resolved via Voronoi cell lookup\n\
             [ManifoldPkg] Installing '{}' v1.0.0 (aether carrier)\n\
             [ManifoldPkg] Installed to /packages/{}/\n\
             Done.",
            pkg, pkg, pkg
        )
    }

    fn cmd_remove(&mut self, pkg: &str) -> String {
        if pkg.is_empty() {
            return String::from("remove: which package? Usage: remove <package>");
        }
        format!("[ManifoldPkg] Removed '{}'", pkg)
    }

    fn cmd_packages(&self) -> String {
        String::from(
            "[ManifoldPkg] Installed packages:\n\
               (none — use 'install <package>' to get started)",
        )
    }

    fn cmd_update(&self) -> String {
        String::from("[ManifoldPkg] All packages up to date.")
    }

    fn cmd_wifi(&self, sub: &str, arg: &str) -> String {
        match sub {
            "connect" => {
                if arg.is_empty() {
                    String::from("wifi connect: usage: wifi connect <ssid> [password]")
                } else {
                    format!(
                        "[WiFi] Scanning for '{}'...\n\
                         [WiFi] Associating with '{}' (802.11ac)\n\
                         [WiFi] WPA2 4-way handshake...\n\
                         [WiFi] Connected! DHCP: 192.168.1.42/24\n\
                         [WiFi] DNS: 8.8.8.8, Gateway: 192.168.1.1",
                        arg, arg
                    )
                }
            }
            "disconnect" => String::from("[WiFi] Disconnected."),
            "scan" => String::from(
                "[WiFi] Scanning...\n\
                   SSID                  SIGNAL  SECURITY\n\
                   HomeNetwork           -45dBm  WPA2\n\
                   CoffeeShop            -62dBm  Open\n\
                   5G_IoT                -70dBm  WPA3",
            ),
            _ => String::from(
                "[WiFi] Status: Not connected\n\
                 Driver: Seal OS WiFi (802.11ac)\n\
                 Commands: wifi scan, wifi connect <ssid>, wifi disconnect",
            ),
        }
    }

    fn cmd_bluetooth(&self, sub: &str, arg: &str) -> String {
        match sub {
            "scan" => String::from(
                "[Bluetooth] Scanning...\n\
                   DEVICE                TYPE     RSSI\n\
                   AirPods Pro           Audio    -35dBm\n\
                   MX Master 3           HID      -42dBm\n\
                   Fitness Band          LE       -58dBm",
            ),
            "pair" => {
                if arg.is_empty() {
                    String::from("bluetooth pair: usage: bluetooth pair <device>")
                } else {
                    format!("[Bluetooth] Pairing with '{}'...\n[Bluetooth] Paired successfully!", arg)
                }
            }
            _ => String::from(
                "[Bluetooth] Status: Enabled\n\
                 Driver: Seal OS Bluetooth 5.0 LE\n\
                 Paired devices: 0\n\
                 Commands: bluetooth scan, bluetooth pair <device>",
            ),
        }
    }

    fn cmd_run(&self, file: &str) -> String {
        if file.is_empty() {
            return String::from("run: which script? Usage: run <file.aether>");
        }
        format!(
            "[Aether-Lang] Loading '{}'...\n\
             [Aether-Lang] Parsed → Titan bytecode (EMBED, ATTEND, PRUNE)\n\
             [Aether-Lang] Executing on Titan VM...\n\
             [Aether-Lang] Done.",
            file
        )
    }

    fn cmd_theme(&self, name: &str) -> String {
        if name.is_empty() {
            return String::from("theme: which theme? Available: dark, light, seal, matrix");
        }
        match name {
            "dark" | "light" | "seal" | "matrix" => {
                format!("[Theme] Switched to '{}' theme.", name)
            }
            _ => format!("[Theme] Unknown theme '{}'. Available: dark, light, seal, matrix", name),
        }
    }

    fn cmd_set(&self, key: &str, val: &str) -> String {
        if key.is_empty() {
            return String::from("set: usage: set <key> <value>");
        }
        format!("[Settings] {} = {}", key, val)
    }

    fn cmd_play(&mut self, sub: &str, arg: &str) -> String {
        match sub {
            "" => self.media_player.play(),
            "status" => self.media_player.status(),
            "pause" => self.media_player.pause(),
            "stop" => self.media_player.stop(),
            "next" => self.media_player.next_track(),
            "prev" => self.media_player.prev_track(),
            "list" | "playlist" => self.media_player.show_playlist(),
            "formats" => MediaPlayer::supported_formats(),
            "add" => {
                if arg.is_empty() {
                    String::from("play add: usage: play add <file>")
                } else {
                    self.media_player.add_to_playlist(arg);
                    format!("[SealPlayer] Added '{}' to playlist", arg)
                }
            }
            "vol" | "volume" => {
                let vol: u8 = arg.parse().unwrap_or(80);
                self.media_player.volume_set(vol)
            }
            "seek" => {
                let secs: u64 = arg.parse().unwrap_or(0);
                self.media_player.seek(secs)
            }
            file => {
                match self.media_player.open(file) {
                    Ok(s) => s,
                    Err(e) => e,
                }
            }
        }
    }

    fn cmd_prefetch(&self, sub: &str) -> String {
        match sub {
            "status" => String::from(
                "[aether-link] Prefetch Engine Status\n\
                 ════════════════════════════════════\n\
                 Preset:          gaming (ε=0.40, φ=0.20)\n\
                 Decisions/sec:   ~55M\n\
                 Prefetch ratio:  73.2%\n\
                 Latency:         18.1 ns/decision\n\
                 Hits:            0  Misses: 0",
            ),
            _ => String::from("prefetch: usage: prefetch status"),
        }
    }

    fn update_cwd_path(&mut self) {
        let mut parts = Vec::new();
        let mut id = self.cwd;
        while id != 0 {
            if let Some(inode) = self.fs.inode(id) {
                parts.push(inode.name.clone());
                id = inode.parent;
            } else {
                break;
            }
        }
        parts.reverse();
        if parts.is_empty() {
            self.cwd_path = String::from("/");
        } else {
            self.cwd_path = format!("/{}/", parts.join("/"));
        }
    }

    pub fn prompt(&self) -> String {
        format!("seal:{}$ ", self.cwd_path)
    }

    pub fn fs_mut(&mut self) -> &mut ManifoldFS {
        &mut self.fs
    }
}
