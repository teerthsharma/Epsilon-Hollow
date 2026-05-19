// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! SealShell help system — per-command documentation.

use alloc::string::String;

pub fn handbook() -> String {
    String::from(
        "SealShell — The Human-Friendly Shell\n\
         =====================================\n\
         \n\
         Navigation:\n\
           open <folder>     Enter a folder\n\
           back              Go up one folder\n\
           home              Go to root\n\
         \n\
         Files:\n\
           look [folder]     List contents\n\
           peek <file>       Show file contents\n\
           write <f> <text>  Create/write a file\n\
           create <name>     Create a new folder\n\
           delete <file>     Remove a file\n\
           rename <old> <n>  Rename a file/folder\n\
           info <file>       File details\n\
         \n\
         Transfer:\n\
           copy <file>       Copy file to clipboard\n\
           paste             Paste here\n\
           move <f> <dest>   O(1) teleport a file\n\
           search <query>    Content-addressable search\n\
         \n\
         System:\n\
           seal              System info + theorems\n\
           tasks             Running processes\n\
           stats             Filesystem statistics\n\
           race [size]       Benchmark teleport vs copy\n\
           prefetch status   Aether-link I/O stats\n\
           memory            Memory usage statistics\n\
         \n\
         ML:\n\
           ml status         ML runtime status\n\
           ml devices        List compute devices\n\
           ml train          Train a model (coming soon)\n\
           ml infer          Run inference (coming soon)\n\
         \n\
         Packages:\n\
           install <pkg>     Install a package\n\
           remove <pkg>      Remove a package\n\
           packages          List installed packages\n\
           update            Update all packages\n\
         \n\
         Network:\n\
           wifi              WiFi status/networks\n\
           wifi connect <s>  Connect to WiFi\n\
           bluetooth         Bluetooth devices\n\
         \n\
         Scripting:\n\
           run <file.aether> Execute Aether-Lang script\n\
           aether            Interactive Aether-Lang REPL\n\
         \n\
         Games:\n\
           snake             Play Snake\n\
           breakout          Play Breakout\n\
           warp              Play Warp Racer\n\
         \n\
         Settings:\n\
           theme <name>      Change theme (dark/light/seal/matrix)\n\
           set <key> <val>   Change a setting\n\
         \n\
         Other:\n\
           history           Command history\n\
           clear             Clear terminal\n\
           help              This handbook\n\
           help <command>    Detailed help\n",
    )
}

pub fn help_for(cmd: &str) -> String {
    match cmd {
        "open" => String::from(
            "open <folder>\n  Enter a folder. Use folder names visible from 'look'.\n  Example: open docs",
        ),
        "back" => String::from("back\n  Go up one folder (to the parent directory)."),
        "home" => String::from("home\n  Go to the root directory (/)."),
        "look" => String::from(
            "look [folder]\n  List contents of current folder, or a specific folder.\n  Shows: type, name, size, payload points, Voronoi cell.",
        ),
        "peek" => String::from(
            "peek <file>\n  Display the contents of a text file.\n  Files are stored as 64-point clouds on S² and decoded on read.",
        ),
        "write" => String::from(
            "write <filename> <content>\n  Create a file with the given content.\n  Content is encoded as geometry on S² (64 points × 3 coords).\n  Example: write hello.txt Hello from Seal OS!",
        ),
        "create" => String::from(
            "create <name>\n  Create a new folder in the current directory.\n  Folders are Voronoi cells in the ManifoldFS topology.",
        ),
        "delete" => String::from(
            "delete <file>\n  Remove a file from the current directory.\n  The inode is freed and its Voronoi cell updated.",
        ),
        "rename" => String::from(
            "rename <old> <new>\n  Rename a file or folder in the current directory.",
        ),
        "copy" => String::from(
            "copy <file>\n  Copy a file reference to the clipboard.\n  Use 'paste' in another folder to duplicate it.",
        ),
        "paste" => String::from(
            "paste\n  Paste the copied file into the current folder.\n  Creates a new inode with the same content.",
        ),
        "move" => String::from(
            "move <file> <dest_folder>\n  O(1) teleport a file to another folder.\n  Uses topological surgery — no data copying!\n  Example: move data.txt /archive",
        ),
        "search" => String::from(
            "search <query>\n  Content-addressable search across ManifoldFS.\n  Encodes query as geometry and finds nearest files by\n  cosine similarity in the Voronoi index.",
        ),
        "info" => String::from(
            "info <file>\n  Show detailed file information:\n  size, payload points, Voronoi cell, cluster ID, permissions.",
        ),
        "seal" => String::from(
            "seal\n  Show Seal OS system information:\n  version, architecture, theorem status (T1-T5),\n  governor epsilon, entropy, hyperbolic ratio.",
        ),
        "tasks" => String::from(
            "tasks\n  Show running processes from the ManifoldScheduler.\n  Displays PID, state, name, priority, and Voronoi cell.",
        ),
        "stats" => String::from(
            "stats\n  Show ManifoldFS filesystem statistics:\n  files, directories, teleports, entropy, governor epsilon.",
        ),
        "race" => String::from(
            "race [size_in_bytes]\n  Benchmark traditional copy vs ManifoldFS teleport.\n  Default: 1GB. Shows speedup from O(1) topological surgery.\n  Example: race 10000000000  (10GB)",
        ),
        "install" => String::from(
            "install <package>\n  Install a package via ManifoldPkg.\n  Resolves dependencies via Voronoi cell lookup.\n  Example: install math-core",
        ),
        "remove" => String::from("remove <package>\n  Remove an installed package."),
        "packages" => String::from("packages\n  List all installed packages with version and carrier type."),
        "update" => String::from("update\n  Check for and apply package updates."),
        "wifi" => String::from(
            "wifi\n  Show WiFi status and available networks.\n  wifi connect <ssid> [password] — connect to a network.\n  wifi disconnect — disconnect from current network.",
        ),
        "bluetooth" => String::from(
            "bluetooth\n  Show Bluetooth status and paired devices.\n  bluetooth scan — scan for nearby devices.\n  bluetooth pair <device> — pair with a device.",
        ),
        "run" => String::from(
            "run <file.aether>\n  Execute an Aether-Lang script from ManifoldFS.\n  Scripts use Titan bytecode VM with topology opcodes.\n  Example: run hello.aether",
        ),
        "aether" => String::from(
            "aether\n  Launch the interactive Aether-Lang REPL.\n  Type Aether-Lang expressions, terminated with ~\n  Type 'exit~' to return to SealShell.",
        ),
        "snake" => String::from("snake\n  Play the classic Snake game.\n  Arrow keys to move. Eat apples to grow. Don't hit walls!"),
        "breakout" => String::from(
            "breakout\n  Play Breakout — bounce the ball to destroy bricks.\n  Arrow keys to move paddle. 3 lives.",
        ),
        "warp" => String::from(
            "warp\n  Play Warp Racer — fly through data blocks!\n  Showcases aether-link's prefetch predictions in real-time.\n  Green = prefetched, Red = cache miss.",
        ),
        "theme" => String::from(
            "theme <name>\n  Change the desktop theme.\n  Available: dark, light, seal, matrix\n  Example: theme matrix",
        ),
        "set" => String::from(
            "set <key> <value>\n  Change a system setting.\n  Example: set font-size 16",
        ),
        "history" => String::from("history\n  Show command history for this session."),
        "clear" => String::from("clear\n  Clear the terminal screen."),
        "memory" => String::from(
            "memory\n  Show memory usage statistics.\n  Includes: allocated bytes, total bytes, free bytes.",
        ),
        "ml" => String::from(
            "ml <subcommand>\n  ML runtime control.\n  Subcommands: status, devices, train, infer",
        ),
        _ => alloc::format!("No help available for '{}'.\nType 'help' for the full handbook.", cmd),
    }
}
