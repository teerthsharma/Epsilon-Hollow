// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Epsilon-Hollow Topological OS — CLI Shell
//!
//! Usage:
//!   cargo run -p epsilon-os -- --api-key MINIMAX_KEY
//!   MINIMAX_API_KEY=key cargo run -p epsilon-os

mod world;

use std::io::{self, BufRead, Write};
use world::World;

const BANNER: &str = r#"
 ___________              _ __                __  __      ____
|  ___|  _  \      _     (_) |               / / / /     / / /
| |__ | |_| |  ___(_) ___ _| | ___  _ __    / /_/ / ___ / / /____ _ __
|  __||  ___/ / __| |/ / || | |/ _ \| '_ \  |  _  |/ _ \ | / _ \ | '_ \
| |___| |     \__ \ |   <| | | (_) | | | | | | | | (_) | | (_) | | | | |
\____/\_|     |___/_|\___|_|_|\___/|_| |_| \_| |_/\___/_|\___/|_| |_|
                    Topological OS v0.1 — Manifold Kernel (Rust)
"#;

const HELP: &str = r#"Commands:
  /step <text>     Ingest observation into topological memory
  /query <text>    Query memory + LLM reasoning (Minimax 2.7)
  /dream <text>    Counterfactual rollout through latent dynamics
  /status          World model status
  /memory          Topological memory stats (Betti numbers, size)
  /theorems        Run T1-T10 theorem verification suite
  /history         Show ingestion history
  /reset           Reset memory
  /help            Show this help
  /exit            Quit
  <plain text>     Treated as /query"#;

fn parse_args() -> (String, usize, usize) {
    let args: Vec<String> = std::env::args().collect();
    let mut api_key = std::env::var("MINIMAX_API_KEY").unwrap_or_default();
    let mut dim = 128usize;
    let mut capacity = 100_000usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--api-key" => {
                i += 1;
                if i < args.len() {
                    api_key = args[i].clone();
                }
            }
            "--dim" => {
                i += 1;
                if i < args.len() {
                    dim = args[i].parse().unwrap_or(128);
                }
            }
            "--capacity" => {
                i += 1;
                if i < args.len() {
                    capacity = args[i].parse().unwrap_or(100_000);
                }
            }
            _ => {}
        }
        i += 1;
    }
    (api_key, dim, capacity)
}

fn boot(w: &World) {
    print!("{BANNER}");
    let status = w.status();
    println!(
        "  dim={}  capacity={}  API key: {}",
        status.dim,
        status.capacity,
        if status.api_key_set {
            "set"
        } else {
            "NOT SET (LLM disabled)"
        }
    );
    println!();

    println!("  Running theorem verification...");
    let t0 = std::time::Instant::now();
    let results = w.verify_theorems();
    let dt = t0.elapsed();
    let passed = results.iter().filter(|(_, ok)| *ok).count();
    let total = results.len();
    println!("  Theorems: {passed}/{total} passed in {:.2?}", dt);
    if passed < total {
        let failed: Vec<&str> = results
            .iter()
            .filter(|(_, ok)| !*ok)
            .map(|(n, _)| *n)
            .collect();
        println!("  Failed: {}", failed.join(", "));
    }
    println!();
    println!("  Type /help for commands. Ready.");
    println!();
}

fn json_print<T: serde::Serialize>(val: &T) {
    println!("{}", serde_json::to_string_pretty(val).unwrap_or_default());
}

fn repl(w: &mut World) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("epsilon> ");
        let _ = stdout.flush();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                println!("\nShutting down.");
                break;
            }
            Err(_) => break,
            _ => {}
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with("/exit") || line.starts_with("/quit") {
            println!("Goodbye.");
            break;
        } else if line.starts_with("/help") {
            println!("{HELP}");
        } else if let Some(text) = line.strip_prefix("/step ") {
            let text = text.trim();
            if text.is_empty() {
                println!("Usage: /step <text>");
                continue;
            }
            let result = w.update(text);
            json_print(&result);
        } else if let Some(text) = line.strip_prefix("/query ") {
            let text = text.trim();
            if text.is_empty() {
                println!("Usage: /query <text>");
                continue;
            }
            let result = w.query(text, 5);
            if let Some(ref resp) = result.llm_response {
                println!("\n{resp}\n");
                println!(
                    "  [{} memories, {:.2}ms]",
                    result.top_k.len(),
                    result.retrieval_ms
                );
            } else {
                json_print(&result);
            }
        } else if let Some(text) = line.strip_prefix("/dream ") {
            let text = text.trim();
            if text.is_empty() {
                println!("Usage: /dream <text>");
                continue;
            }
            let result = w.dream(text, 5);
            json_print(&result);
        } else if line.starts_with("/status") {
            json_print(&w.status());
        } else if line.starts_with("/memory") {
            json_print(&w.memory.stats());
        } else if line.starts_with("/theorems") {
            println!("Running T1-T10 verification suite...");
            let t0 = std::time::Instant::now();
            let results = w.verify_theorems();
            let dt = t0.elapsed();
            for (name, ok) in &results {
                let tag = if *ok { "PASS" } else { "FAIL" };
                println!("  [{tag}] {name}");
            }
            let passed = results.iter().filter(|(_, ok)| *ok).count();
            println!("\n  {}/{} passed in {:.2?}", passed, results.len(), dt);
        } else if line.starts_with("/history") {
            for (i, h) in w.history.iter().enumerate() {
                let display = if h.len() > 80 { &h[..80] } else { h.as_str() };
                println!("  [{i}] {display}");
            }
            println!("  Total: {}", w.history.len());
        } else if line.starts_with("/reset") {
            w.memory.reset();
            w.history.clear();
            println!("Memory reset.");
        } else if line.starts_with('/') {
            let cmd = line.split_whitespace().next().unwrap_or(line);
            println!("Unknown command: {cmd}. Type /help");
        } else {
            // Plain text → query
            let result = w.query(line, 5);
            if let Some(ref resp) = result.llm_response {
                println!("\n{resp}\n");
                println!(
                    "  [{} memories, {:.2}ms]",
                    result.top_k.len(),
                    result.retrieval_ms
                );
            } else {
                json_print(&result);
            }
        }
    }
}

fn main() {
    let (api_key, dim, capacity) = parse_args();
    let mut w = World::new(api_key, dim, capacity);
    boot(&w);
    repl(&mut w);
}
