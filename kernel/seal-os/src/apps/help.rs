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
         Filters (read the pipeline, write the pipeline):\n\
           grep [-cinv] <t>  Lines holding <t>; -v inverts, -i folds case,\n\
                             -n numbers them, -c counts them\n\
           wc [-l|-w|-c]     Lines, words, bytes\n\
           head [-n <k>]     First <k> lines (10 by default)\n\
           tail [-n <k>]     Last <k> lines (10 by default)\n\
           sort [-r] [-u]    Sort lines; -r reverses, -u drops repeats\n\
           uniq [-c]         Collapse adjacent equal lines; -c counts each\n\
           tr <from> <to>    Replace one character with another\n\
           cat [file]        A file, or the pipeline, unchanged\n\
         \n\
         Pipelines and redirection (help pipes):\n\
           a | b | c         Run a, feed its output to b, then to c\n\
           b < file          Read file as the first stage's input\n\
           b > file          Write the last stage's output to file\n\
           b >> file         Append it to file instead\n\
           Example: peek notes.txt | grep -i seal | sort -u > hits.txt\n\
         \n\
         Variables (help variables):\n\
           NAME=value        Name a value; no spaces in it, no quoting here\n\
           $NAME  ${NAME}    Use it; an unset name expands to nothing\n\
           export NAME       Mark it for the environment; export NAME=v too\n\
           unset NAME        Forget it\n\
           env               List the exported ones\n\
           set               List all of them\n\
           Example: OUT=hits.txt then peek notes.txt | grep seal > $OUT\n\
         \n\
         Transfer:\n\
           copy <file>       Copy file to clipboard\n\
           paste             Paste here\n\
           move <f> <dest>   metadata teleport a file\n\
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
            "move <file> <dest_folder>\n  Metadata-teleport a file to another folder.\n  Uses topological surgery; current persistence still writes bytes.\n  Example: move data.txt /archive",
        ),
        "search" => String::from(
            "search <query>\n  Content-addressable search across ManifoldFS.\n  Encodes query as geometry and finds nearest files by\n  cosine similarity in the Voronoi index.",
        ),
        "grep" => String::from(
            "grep [-c] [-i] [-n] [-v] <text>\n  Keep the lines of the pipeline that hold <text>.\n  -v keeps the lines that do not, -i ignores case,\n  -n prefixes each line with its number, -c reports how many.\n  <text> is matched literally; there are no regular expressions.\n  Example: peek notes.txt | grep -in seal",
        ),
        "wc" => String::from(
            "wc [-l] [-w] [-c]\n  Count the pipeline: lines, words, then bytes.\n  A flag selects one count; with none, all three are printed.\n  Example: look | wc -l",
        ),
        "head" => String::from(
            "head [-n <lines>]\n  The first <lines> lines of the pipeline, 10 by default.\n  A count that is zero, negative or not a number is refused.\n  Example: peek log.txt | head -n 3",
        ),
        "tail" => String::from(
            "tail [-n <lines>]\n  The last <lines> lines of the pipeline, 10 by default.\n  A count that is zero, negative or not a number is refused.\n  Example: peek log.txt | tail -n 3",
        ),
        "sort" => String::from(
            "sort [-r] [-u]\n  Sort the pipeline's lines in byte order.\n  -r reverses the order, -u drops repeated lines.\n  Every line comes out terminated, even if the input's last was not.\n  Example: look | sort -u",
        ),
        "uniq" => String::from(
            "uniq [-c]\n  Collapse runs of identical neighbouring lines into one.\n  -c prefixes each with the length of its run.\n  Only neighbours: sort first to collapse every duplicate.\n  Example: peek words.txt | sort | uniq -c",
        ),
        "tr" => String::from(
            "tr <from> <to>\n  Replace every <from> character in the pipeline with <to>.\n  One character each — no ranges (a-z) and no classes.\n  Example: peek data.csv | tr , ;",
        ),
        "cat" => String::from(
            "cat [file]\n  With a file, print it, like 'peek'.\n  With none, pass the pipeline through unchanged, which is how\n  'cat < file | grep x' reads a file into a pipeline.",
        ),
        "pipes" | "pipe" | "redirect" | "redirection" => String::from(
            "Pipelines and redirection\n  a | b | c    Run a, hand its output to b, then b's to c.\n  b < file     The first stage reads file as its input.\n  b > file     The last stage's output replaces file.\n  b >> file    The last stage's output is added to the end of file.\n  '<' is only allowed on the first stage and '>' on the last.\n  A stage handing on more than 65536 bytes stops the line.\n  There is no quoting, so | < > cannot appear inside an argument.\n  Example: peek notes.txt | grep -i seal | sort -u > hits.txt",
        ),
        "info" => String::from(
            "info <file>\n  Show detailed file information:\n  size, payload points, Voronoi cell, cluster ID, permissions.",
        ),
        "seal" => String::from(
            "seal\n  Show Seal OS system information:\n  version, architecture, theorem status (T1-T10),\n  governor epsilon, entropy, hyperbolic ratio.",
        ),
        "tasks" => String::from(
            "tasks\n  Show running processes from the ManifoldScheduler.\n  Displays PID, state, name, priority, and Voronoi cell.",
        ),
        "stats" => String::from(
            "stats\n  Show ManifoldFS filesystem statistics:\n  files, directories, teleports, entropy, governor epsilon.",
        ),
        "race" => String::from(
            "race [size_in_bytes]\n  Benchmark traditional copy vs ManifoldFS metadata teleport.\n  Default: 1GB. Reports topology metadata cost separately from byte persistence.\n  Example: race 10000000000  (10GB)",
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
            "set\n  With no argument, list every shell variable, sorted by name.\n  set <key> <value>\n  Change a system setting.\n  Example: set font-size 16",
        ),
        "variables" | "variable" | "var" | "$" => String::from(
            "Shell variables\n  NAME=value   Name a value. NAME is a letter or '_' followed by\n               letters, digits or '_'; anything else is a command.\n  $NAME        Use it. ${NAME} does the same and can touch the text\n  ${NAME}      after it, as in ${F}.txt. An unset name expands to\n               nothing, and a '$' no name follows is a plain '$'.\n  A value may not contain a space. There is no quoting here, so\n  'A=1 look' cannot be told from a value with a space in it, and the\n  two readings do opposite things; it is refused instead.\n  The line is split on | < > first and expanded second, so a value\n  holding one of them is an argument and never an operator.\n  Expansion is one pass: what a value expands to is never re-read,\n  so no pair of variables can chase each other.\n  At most 64 variables, 4096 bytes of names and values together.\n  There is no $? — this shell has no exit status.\n  Example: OUT=hits.txt then peek notes.txt | grep seal > $OUT",
        ),
        "export" => String::from(
            "export <NAME>\nexport <NAME>=<value>\n  Mark a variable for the environment; 'env' lists the marked set.\n  Exporting a name that is not set yet creates it empty.\n  Reassigning a marked variable keeps the mark; 'unset' clears it.",
        ),
        "unset" => String::from(
            "unset <NAME>\n  Forget a variable, along with any 'export' mark it carried.\n  Unsetting a name that was never set is not an error.",
        ),
        "env" => String::from(
            "env\n  List the exported variables as NAME=value, sorted by name.\n  'set' with no argument lists every variable, exported or not.\n  Nothing is passed to a command's environment: Seal OS commands are\n  shell methods, not processes, so 'export' only marks the set that\n  'env' reports.",
        ),
        "history" => String::from("history\n  Show command history for this session."),
        "clear" => String::from("clear\n  Clear the terminal screen."),
        "memory" => String::from(
            "memory\n  Show memory usage statistics.\n  Includes: allocated bytes, total bytes, free bytes.",
        ),
        "ml" => String::from(
            "ml <subcommand>\n  ML runtime control.\n  Subcommands: status, devices, train, save, load, generate, tensor, matmul",
        ),
        _ => alloc::format!("No help available for '{}'.\nType 'help' for the full handbook.", cmd),
    }
}
