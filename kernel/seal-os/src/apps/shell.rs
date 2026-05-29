// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! SealShell - the human-friendly shell. Seal-native commands only.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::drivers::bluetooth::BluetoothDriver;
use crate::drivers::wifi::WifiDriver;
use crate::fs::manifold_fs::ManifoldFS;
use crate::lang::AetherRuntime;

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
    last_model: Option<Vec<u8>>,
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
            last_model: None,
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

        match cmd {
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

            // System
            "whoami" => {
                match crate::security::passwd::get_current_user() {
                    Some(user) => user.username,
                    None => String::from("unknown"),
                }
            }
            "memory" => self.cmd_memory(),

            // ML
            "ml" => self.cmd_ml(arg1, arg2),

            // Settings
            "theme" => self.cmd_theme(arg1),
            "set" => self.cmd_set(arg1, arg2),

            // Prefetch
            "prefetch" => self.cmd_prefetch(arg1),

            // TopCrypt / Lypnos Guard
            "topcrypt" => {
                let rest = input.strip_prefix("topcrypt ").unwrap_or("");
                self.cmd_topcrypt(rest)
            }

            // Tensor visualization
            "tensor" => self.cmd_tensor(arg1, arg2),

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

            _ => format!("seal: unknown command '{}' — type 'help' for the handbook", cmd),
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
                name,
                content.len(),
                id
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
        let mut out = format!(
            "Seal OS v{} (x86_64)\n\
             The Geometrical Operating System\n\
             All data = geometry on S²\n\
             File moves = metadata topology; persistent bytes still scale\n\
             ═══════════════════════════════════\n",
            crate::VERSION
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
        let tasks = crate::process::scheduler::list_all_tasks();
        let mut out = String::from("  PID  STATE    NAME             PRI  CELL\n");
        if tasks.is_empty() {
            out.push_str("  (no tasks scheduled)\n");
        } else {
            for (id, name, state, prio, cell) in &tasks {
                out.push_str(&format!(
                    "  {:<4} {:<8} {:<16} {:<4} {}\n",
                    id, state, name, prio, cell
                ));
            }
        }
        out.push_str(&format!(
            "  Total: {} tasks",
            crate::process::scheduler::task_count()
        ));
        out
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
            s.total_files,
            s.total_dirs,
            s.total_teleports,
            s.governor_epsilon,
            s.current_entropy,
            s.max_depth,
            s.hyperbolic_ratio
        )
    }

    fn cmd_memory(&self) -> String {
        let (allocated, total) = crate::memory::heap_stats();
        let allocated_mb = allocated / (1024 * 1024);
        let total_mb = total / (1024 * 1024);
        format!(
            "Memory Status\n\
             ═════════════\n\
             Allocated: {} bytes ({} MB)\n\
             Total:     {} bytes ({} MB)\n\
             Free:      {} bytes ({} MB)",
            allocated,
            allocated_mb,
            total,
            total_mb,
            total.saturating_sub(allocated),
            total_mb.saturating_sub(allocated_mb)
        )
    }

    fn cmd_ml(&mut self, subcmd: &str, arg: &str) -> String {
        match subcmd {
            "status" => {
                let status = crate::ml_engine::MlStatus::detect();
                let gpu_line = match &status.gpu_detected {
                    Some((name, _, _)) => format!("GPU:          {} (detected)", name),
                    None => String::from("GPU:          not detected (PCI probe done)"),
                };
                format!(
                    "ML Runtime Status\n\
                     ══════════════════\n\
                     AEGIS Engine: aether-core (no_std)\n\
                     Tensor ops:   {}\n\
                     Neural nets:  {}\n\
                     AVX2:         {}\n\
                     AVX-512:      {}\n\
                     {}\n\
                     Use: ml train [epochs] | ml matmul | ml tensor",
                    if status.tensor_ops_available {
                        "available"
                    } else {
                        "unavailable"
                    },
                    if status.neural_net_available {
                        "available"
                    } else {
                        "unavailable"
                    },
                    if status.avx2_detected {
                        "detected"
                    } else {
                        "not detected"
                    },
                    if status.avx512_detected {
                        "detected"
                    } else {
                        "not detected"
                    },
                    gpu_line,
                )
            }
            "devices" => String::from(
                "ML Compute Devices\n\
                     ═══════════════════\n\
                     CPU:  x86_64 (AVX2/AVX-512 detection pending)\n\
                     GPU:  Not detected (PCI probe pending)\n\
                     NPU:  Not detected",
            ),
            "train" => {
                let epochs: usize = arg.parse().unwrap_or(1000);
                let (report, bytes) = crate::ml_engine::demo_train_mlp(epochs);
                self.last_model = Some(bytes);
                report
            }
            "save" => {
                let name = if arg.is_empty() { "model.sealml" } else { arg };
                match &self.last_model {
                    Some(bytes) => match crate::ml_engine::save_model_bytes(name, bytes) {
                        Ok(msg) => msg,
                        Err(e) => format!("Error: {}", e),
                    },
                    None => String::from("No model to save. Run 'ml train' first."),
                }
            }
            "load" => {
                let name = if arg.is_empty() { "model.sealml" } else { arg };
                match crate::ml_engine::load_model(name) {
                    Ok(_mlp) => format!("Model '{}' loaded successfully", name),
                    Err(e) => format!("Error: {}", e),
                }
            }
            "generate" => {
                let seed = if arg.is_empty() { "Seal" } else { arg };
                let text = crate::ml_engine::demo_generate_text(seed, 120);
                format!(
                    "Markov Generation (seed: '{}')\n\
                         ══════════════════════════════════\n\
                         {}",
                    seed, text
                )
            }
            "tensor" => {
                // Create a simple 2x3 tensor for demo
                let t = crate::ml_engine::tensor_from_data(
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    vec![2, 3],
                );
                match t {
                    Ok(tensor) => crate::ml_engine::format_tensor(&tensor),
                    Err(e) => format!("Tensor error: {}", e),
                }
            }
            "matmul" => {
                let a = match crate::ml_engine::tensor_from_data(
                    vec![1.0, 2.0, 3.0, 4.0],
                    vec![2, 2],
                ) {
                    Ok(t) => t,
                    Err(e) => return format!("Tensor A error: {}", e),
                };
                let b = match crate::ml_engine::tensor_from_data(
                    vec![5.0, 6.0, 7.0, 8.0],
                    vec![2, 2],
                ) {
                    Ok(t) => t,
                    Err(e) => return format!("Tensor B error: {}", e),
                };
                match crate::ml_engine::tensor_matmul(&a, &b) {
                    Ok(result) => crate::ml_engine::format_tensor(&result),
                    Err(e) => format!("Matmul error: {}", e),
                }
            }
            _ => String::from(
                "ML Subcommands:\n\
                 ml status    — Show ML runtime status\n\
                 ml devices   — List compute devices\n\
                 ml train     — Train MLP on XOR dataset\n\
                 ml save      — Save trained model to ManifoldFS\n\
                 ml load      — Load model from ManifoldFS\n\
                 ml generate  — Generate text with Markov chain\n\
                 ml tensor    — Create a demo tensor\n\
                 ml matmul    — Demo matrix multiply",
            ),
        }
    }

    fn cmd_race(&self, arg: &str) -> String {
        let size: usize = arg.parse().unwrap_or(1_000_000);
        let label = if size >= 1_000_000_000 {
            format!("{}GB", size / 1_000_000_000)
        } else if size >= 1_000_000 {
            format!("{}MB", size / 1_000_000)
        } else if size >= 1_000 {
            format!("{}KB", size / 1_000)
        } else {
            format!("{}B", size)
        };

        // Real benchmark: create file, measure duplicate vs teleport
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let src_dir = fs.mkdir("race_src", root).unwrap_or(1);
        let dst_dir = fs.mkdir("race_dst", root).unwrap_or(2);

        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let _file = fs.store("race_file", &data, src_dir);

        // Warm-up
        let _ = fs.duplicate("race_file", src_dir, dst_dir);
        let _ = fs.teleport("race_file", dst_dir, src_dir);

        // Benchmark duplicate (traditional copy)
        let dup_start = crate::drivers::interrupts::ticks();
        let _ = fs.duplicate("race_file", src_dir, dst_dir);
        let dup_ticks = crate::drivers::interrupts::ticks().saturating_sub(dup_start);

        // Benchmark metadata teleport separately from byte persistence.
        let tel_start = crate::drivers::interrupts::ticks();
        let _ = fs.teleport("race_file", dst_dir, src_dir);
        let tel_ticks = crate::drivers::interrupts::ticks().saturating_sub(tel_start);

        let speedup = if tel_ticks > 0 {
            dup_ticks as f64 / tel_ticks as f64
        } else {
            9999.0
        };

        format!(
            "Race: {} payload\n\
             ════════════════════════\n\
             Traditional duplicate:  {} ticks\n\
             Manifold metadata move: {} ticks (topological surgery)\n\
             Speedup:                {:.1}x\n\
             \n\
             The teleport rewrites one BTreeMap entry.\n\
             The duplicate clones geometry metadata + raw bytes.",
            label, dup_ticks, tel_ticks, speedup
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
        let manifest = format!("name={}\nversion=1.0.0\nstatus=metadata-only", pkg);
        match self
            .fs
            .store_text(&format!("{}.pkg", pkg), &manifest, self.cwd)
        {
            Ok(id) => format!(
                "[ManifoldPkg] Resolved '{}' via Voronoi cell lookup\n\
                 [ManifoldPkg] Stored package manifest (inode {})\n\
                 Note: no remote registry — package metadata stored in ManifoldFS only",
                pkg, id
            ),
            Err(e) => format!("[ManifoldPkg] install: {}", e),
        }
    }

    fn cmd_remove(&mut self, pkg: &str) -> String {
        if pkg.is_empty() {
            return String::from("remove: which package? Usage: remove <package>");
        }
        let pkg_file = format!("{}.pkg", pkg);
        match self.fs.delete(&pkg_file, self.cwd) {
            Ok(()) => format!("[ManifoldPkg] Removed '{}' manifest", pkg),
            Err(_) => format!("[ManifoldPkg] Package '{}' not found", pkg),
        }
    }

    fn cmd_packages(&self) -> String {
        match self.fs.ls(self.cwd) {
            Ok(entries) => {
                let pkgs: Vec<_> = entries
                    .iter()
                    .filter(|e| e.name.ends_with(".pkg"))
                    .collect();
                if pkgs.is_empty() {
                    return String::from(
                        "[ManifoldPkg] No packages installed — use 'install <package>'",
                    );
                }
                let mut out = String::from("[ManifoldPkg] Installed packages:\n");
                for p in &pkgs {
                    out.push_str(&format!("  {} ({} bytes)\n", p.name, p.original_size));
                }
                out
            }
            Err(_) => String::from("[ManifoldPkg] No packages installed"),
        }
    }

    fn cmd_update(&self) -> String {
        String::from("ManifoldPkg: no remote registry — packages are local metadata only")
    }

    fn cmd_wifi(&self, sub: &str, arg: &str) -> String {
        let mut driver = WifiDriver::new();
        match sub {
            "connect" => {
                if arg.is_empty() {
                    String::from("wifi connect: usage: wifi connect <ssid> [password]")
                } else {
                    match driver.connect(arg, "") {
                        Ok(()) => format!("WiFi: connected to '{}'", arg),
                        Err(e) => e,
                    }
                }
            }
            "disconnect" => {
                driver.disconnect();
                String::from("WiFi: disconnected")
            }
            "scan" => {
                let networks = driver.scan();
                if networks.is_empty() {
                    return String::from("WiFi: no wireless networks found");
                }
                let mut out =
                    String::from("WiFi Networks:\n  SSID              SIGNAL  SECURITY\n");
                for n in &networks {
                    out.push_str(&format!(
                        "  {:<18}  {:>4}dBm  {}\n",
                        n.ssid,
                        n.signal_dbm,
                        n.security.name()
                    ));
                }
                out
            }
            _ => String::from(
                "WiFi Status: no hardware detected\n\
                 Commands: wifi scan, wifi connect <ssid>, wifi disconnect",
            ),
        }
    }

    fn cmd_bluetooth(&self, sub: &str, arg: &str) -> String {
        let mut driver = BluetoothDriver::new();
        match sub {
            "scan" => {
                let devices = driver.scan();
                if devices.is_empty() {
                    return String::from("Bluetooth: no devices found");
                }
                let mut out =
                    String::from("Bluetooth Devices:\n  NAME                 TYPE   RSSI\n");
                for d in &devices {
                    out.push_str(&format!(
                        "  {:<20}  {:<6}  {:>4}dBm\n",
                        d.name,
                        d.device_type.name(),
                        d.rssi_dbm
                    ));
                }
                out
            }
            "pair" => {
                if arg.is_empty() {
                    String::from("bluetooth pair: usage: bluetooth pair <device>")
                } else {
                    match driver.pair(arg) {
                        Ok(()) => format!("Bluetooth: paired with '{}'", arg),
                        Err(e) => e,
                    }
                }
            }
            _ => String::from(
                "Bluetooth Status: no adapter detected\n\
                 Commands: bluetooth scan, bluetooth pair <device>",
            ),
        }
    }

    fn cmd_run(&self, file: &str) -> String {
        if file.is_empty() {
            return String::from("run: which script? Usage: run <file.aether>");
        }
        let mut runtime = AetherRuntime::new();
        match runtime.execute_file(file, "") {
            Ok(output) => output,
            Err(e) => format!("[Aether-Lang] {}: {}", file, e),
        }
    }

    fn cmd_theme(&self, name: &str) -> String {
        if name.is_empty() {
            return String::from("theme: which theme? Available: dark, light, seal, matrix");
        }
        match name {
            "dark" | "light" | "seal" | "matrix" => {
                format!(
                    "[Theme] Switched to '{}' theme (visual change requires re-render)",
                    name
                )
            }
            _ => format!(
                "[Theme] Unknown theme '{}'. Available: dark, light, seal, matrix",
                name
            ),
        }
    }

    fn cmd_set(&mut self, key: &str, val: &str) -> String {
        if key.is_empty() {
            return String::from("set: usage: set <key> <value>");
        }
        let entry = format!("{}={}", key, val);
        match self.fs.store_text(&format!(".setting_{}", key), &entry, 0) {
            Ok(_) => format!("[Settings] {} = {} (persisted to ManifoldFS)", key, val),
            Err(e) => format!("[Settings] {} = {} (memory only: {})", key, val, e),
        }
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
            file => match self.media_player.open(file) {
                Ok(s) => s,
                Err(e) => e,
            },
        }
    }

    fn cmd_topcrypt(&mut self, rest: &str) -> String {
        let parts: Vec<&str> = rest.splitn(3, ' ').collect();
        let sub = parts.get(0).copied().unwrap_or("");
        let arg1 = parts.get(1).copied().unwrap_or("");
        let arg2 = parts.get(2).copied().unwrap_or("");

        match sub {
            "encode" => {
                if arg1.is_empty() {
                    return String::from("topcrypt encode: usage: topcrypt encode <file>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Some(d) => d,
                    None => return format!("topcrypt encode: '{}' not found", arg1),
                };
                let topo = crate::fs::topcrypt::encode_bytes(&data, 0x5EA1);
                let serialized = crate::fs::topcrypt::export_to_bytes(&topo);
                let topo_name = format!("{}.topo", arg1);
                if self.fs.exists(&topo_name, self.cwd) {
                    let _ = self.fs.delete(&topo_name, self.cwd);
                }
                match self.fs.store(&topo_name, &serialized, self.cwd) {
                    Ok(id) => format!(
                        "[TopCrypt] Encoded {} bytes → {} blocks on S² (inode {})",
                        data.len(),
                        topo.block_count,
                        id
                    ),
                    Err(e) => format!("topcrypt encode: {}", e),
                }
            }
            "decode" => {
                if arg1.is_empty() {
                    return String::from("topcrypt decode: usage: topcrypt decode <file.topo>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Some(d) => d,
                    None => return format!("topcrypt decode: '{}' not found", arg1),
                };
                let topo = crate::fs::topcrypt::import_from_bytes(&data, 0);
                let decoded = crate::fs::topcrypt::decode_bytes(&topo);
                let out_name = if arg1.ends_with(".topo") {
                    &arg1[..arg1.len() - 5]
                } else {
                    arg1
                };
                if self.fs.exists(out_name, self.cwd) {
                    let _ = self.fs.delete(out_name, self.cwd);
                }
                match self.fs.store(out_name, &decoded, self.cwd) {
                    Ok(id) => format!(
                        "[TopCrypt] Decoded {} blocks → {} bytes (inode {})",
                        topo.block_count,
                        decoded.len(),
                        id
                    ),
                    Err(e) => format!("topcrypt decode: {}", e),
                }
            }
            "lock" => {
                if arg1.is_empty() {
                    return String::from("topcrypt lock: usage: topcrypt lock <file.topo>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Some(d) => d,
                    None => return format!("topcrypt lock: '{}' not found", arg1),
                };
                let mut topo = crate::fs::topcrypt::import_from_bytes(&data, 0);
                let key = crate::security::topcrypt_guard::LYPNOS_KEY;
                crate::fs::topcrypt::lock_file(&mut topo, key);
                let serialized = crate::fs::topcrypt::export_to_bytes(&topo);
                if self.fs.exists(arg1, self.cwd) {
                    let _ = self.fs.delete(arg1, self.cwd);
                }
                match self.fs.store(arg1, &serialized, self.cwd) {
                    Ok(_) => format!(
                        "[Lypnos Guard] File locked. Entropy = {:.3} bits. Sweet dreams.",
                        topo.lock_entropy
                    ),
                    Err(e) => format!("topcrypt lock: {}", e),
                }
            }
            "unlock" => {
                if arg1.is_empty() {
                    return String::from("topcrypt unlock: usage: topcrypt unlock <file.topo>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Some(d) => d,
                    None => return format!("topcrypt unlock: '{}' not found", arg1),
                };
                let mut topo = crate::fs::topcrypt::import_from_bytes(&data, 0);
                let key = crate::security::topcrypt_guard::LYPNOS_KEY;
                if !crate::fs::topcrypt::unlock_file(&mut topo, key) {
                    return format!(
                        "topcrypt unlock: incorrect key or corrupted file '{}'",
                        arg1
                    );
                }
                let serialized = crate::fs::topcrypt::export_to_bytes(&topo);
                if self.fs.exists(arg1, self.cwd) {
                    let _ = self.fs.delete(arg1, self.cwd);
                }
                match self.fs.store(arg1, &serialized, self.cwd) {
                    Ok(_) => format!("[Lypnos Guard] File awakened. Welcome back."),
                    Err(e) => format!("topcrypt unlock: {}", e),
                }
            }
            "info" => {
                if arg1.is_empty() {
                    return String::from("topcrypt info: usage: topcrypt info <file.topo>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Some(d) => d,
                    None => return format!("topcrypt info: '{}' not found", arg1),
                };
                let topo = crate::fs::topcrypt::import_from_bytes(&data, 0);
                format!(
                    "Name: {}
Blocks: {}
Locked: {}
Entropy: {:.3} bits
Seed: {:016x}",
                    arg1,
                    topo.block_count,
                    if topo.locked { "yes" } else { "no" },
                    topo.lock_entropy,
                    topo.embedding_seed
                )
            }
            "render" => {
                if arg1.is_empty() {
                    return String::from("topcrypt render: usage: topcrypt render <file.csv>");
                }
                let text = match self.fs.read_text(arg1, self.cwd) {
                    Some(t) => t,
                    None => return format!("topcrypt render: '{}' not found", arg1),
                };
                let tensor = crate::ml_engine::tensor_viz::parse_csv(&text);
                let data = &tensor.data;
                let len = data.len();
                let mut peaks = 0;
                let mut valleys = 0;
                for w in data.windows(3) {
                    if w[1] > w[0] && w[1] > w[2] {
                        peaks += 1;
                    }
                    if w[1] < w[0] && w[1] < w[2] {
                        valleys += 1;
                    }
                }
                format!(
                    "[Lypnos Guard] Rendering {} tensor... peaks: {}, valleys: {}",
                    len, peaks, valleys
                )
            }
            "export" => {
                if arg1.is_empty() {
                    return String::from(
                        "topcrypt export: usage: topcrypt export <file.topo> [path]",
                    );
                }
                let data = match self.read_file_bytes(arg1) {
                    Some(d) => d,
                    None => return format!("topcrypt export: '{}' not found", arg1),
                };
                let topo = crate::fs::topcrypt::import_from_bytes(&data, 0);
                let flat = crate::fs::topcrypt::decode_bytes(&topo);
                let dest = if arg2.is_empty() {
                    format!("{}.flat", arg1)
                } else {
                    String::from(arg2)
                };
                let dest_name = dest.rsplit('/').next().unwrap_or("export.flat");
                if self.fs.exists(dest_name, self.cwd) {
                    let _ = self.fs.delete(dest_name, self.cwd);
                }
                match self.fs.store(dest_name, &flat, self.cwd) {
                    Ok(_) => format!(
                        "[TopCrypt] Flattened {} blocks → {} bytes → {}",
                        topo.block_count,
                        flat.len(),
                        dest
                    ),
                    Err(e) => format!("topcrypt export: {}", e),
                }
            }
            "import" => {
                if arg1.is_empty() {
                    return String::from("topcrypt import: usage: topcrypt import <file>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Some(d) => d,
                    None => return format!("topcrypt import: '{}' not found", arg1),
                };
                let topo = crate::fs::topcrypt::encode_bytes(&data, 0x5EA1);
                let serialized = crate::fs::topcrypt::export_to_bytes(&topo);
                let topo_name = format!("{}.topo", arg1.rsplit('/').next().unwrap_or(arg1));
                if self.fs.exists(&topo_name, self.cwd) {
                    let _ = self.fs.delete(&topo_name, self.cwd);
                }
                match self.fs.store(&topo_name, &serialized, self.cwd) {
                    Ok(id) => format!(
                        "[TopCrypt] Imported {} bytes → {} blocks on S² (inode {})",
                        data.len(),
                        topo.block_count,
                        id
                    ),
                    Err(e) => format!("topcrypt import: {}", e),
                }
            }
            _ => String::from(
                "TopCrypt / Lypnos Guard commands:
\
                 topcrypt encode <file>       — Encode file to topological format
\
                 topcrypt decode <file.topo>  — Decode topological file
\
                 topcrypt lock <file.topo>    — Lock file with Lypnos Guard
\
                 topcrypt unlock <file.topo>  — Unlock file
\
                 topcrypt info <file.topo>    — Show topological metadata
\
                 topcrypt render <file.csv>   — Render CSV as tensor
\
                 topcrypt export <file.topo> [path] — Export to flat file
\
                 topcrypt import <file>       — Import flat file to manifold",
            ),
        }
    }

    fn cmd_tensor(&mut self, subcmd: &str, arg: &str) -> String {
        match subcmd {
            "render" => {
                if arg.is_empty() {
                    crate::wm::app_launcher::launch_app(9);
                    String::from("[Tensor] Launching tensor viewer with demo pattern")
                } else {
                    match self.fs.read_text(arg, self.cwd) {
                        Some(text) => {
                            crate::apps::tensor_viewer::set_pending_csv(&text);
                            crate::wm::app_launcher::launch_app(9);
                            format!("[Tensor] Launching viewer with '{}'", arg)
                        }
                        None => format!("tensor render: '{}' not found", arg),
                    }
                }
            }
            "info" => {
                if arg.is_empty() {
                    return String::from("tensor info: usage: tensor info <file.csv>");
                }
                match self.fs.read_text(arg, self.cwd) {
                    Some(text) => {
                        let tensor = crate::ml_engine::tensor_viz::parse_csv(&text);
                        if tensor.data.is_empty() {
                            return format!("tensor info: '{}' contains no numeric data", arg);
                        }
                        let mut min = f32::MAX;
                        let mut max = f32::MIN;
                        let mut sum = 0.0f32;
                        for &v in &tensor.data {
                            if v < min {
                                min = v;
                            }
                            if v > max {
                                max = v;
                            }
                            sum += v;
                        }
                        let mean = sum / tensor.data.len() as f32;
                        let shape_str = tensor
                            .shape
                            .iter()
                            .map(|s| format!("{}", s))
                            .collect::<Vec<_>>()
                            .join("x");
                        format!("Tensor: {}\nshape: [{}]\nmin: {:.4}\nmax: {:.4}\nmean: {:.4}\nelements: {}",
                            arg, shape_str, min, max, mean, tensor.data.len())
                    }
                    None => format!("tensor info: '{}' not found", arg),
                }
            }
            _ => String::from(
                "Tensor commands:\n\
                 tensor render [file.csv] — Launch 3D tensor viewer\n\
                 tensor info <file.csv>   — Show tensor statistics",
            ),
        }
    }

    fn read_file_bytes(&self, name: &str) -> Option<Vec<u8>> {
        let inode_id = self.fs.resolve_path_from(name, self.cwd).ok()?;
        let inode = self.fs.inode(inode_id)?;
        Some(inode.data.clone())
    }

    fn cmd_prefetch(&self, sub: &str) -> String {
        match sub {
            "status" => {
                let engine = crate::fs::prefetch::PrefetchEngine::new_gaming();
                format!(
                    "[aether-link] Prefetch Engine Status\n\
                     ════════════════════════════════════\n\
                     Preset:          {:?} (ε={:.2}, φ={:.2})\n\
                     Decisions:       {}\n\
                     Hit ratio:       {:.1}%\n\
                     Note: prefetch counters are in-memory only (no DMA hardware)",
                    engine.preset(),
                    engine.epsilon(),
                    engine.phi(),
                    engine.total_decisions(),
                    engine.hit_ratio() * 100.0
                )
            }
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
