// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

use clap::{Parser as ClapParser, Subcommand};
use std::path::PathBuf;

use repl_core::LangConfig;

static CONFIG: LangConfig = LangConfig {
    name: "Aether",
    prompt: "aether> ",
    extensions: &["aether", "ae"],
};

#[derive(ClapParser)]
#[command(name = "aether", version = "0.1.0")]
#[command(about = "Cross-platform Aether-Lang runtime")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive REPL
    Repl,
    /// Run an Aether-Lang script
    Run {
        #[arg(value_name = "FILE")]
        file: PathBuf,
        #[arg(long, default_value = "bio")]
        mode: String,
    },
    /// Check syntax of an Aether-Lang script
    Check {
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
}

fn main() {
    let builder = std::thread::Builder::new().stack_size(8 * 1024 * 1024);
    let handler = builder
        .spawn(|| {
            let cli = Cli::parse();
            match cli.command {
                Some(Commands::Repl) | None => repl_core::run_repl(&CONFIG),
                Some(Commands::Run { file, mode }) => repl_core::run_file(&CONFIG, &file, &mode),
                Some(Commands::Check { file }) => repl_core::check_file(&file),
            }
        })
        .unwrap();
    handler.join().unwrap();
}
