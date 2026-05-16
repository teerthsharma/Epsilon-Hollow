// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

use aether_lang::vm::{Compiler, TitanVM};
use aether_lang::{Interpreter, Parser};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::fs;
use std::path::Path;

pub struct LangConfig {
    pub name: &'static str,
    pub prompt: &'static str,
    pub extensions: &'static [&'static str],
}

pub fn run_repl(config: &LangConfig) {
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  \u{1f6e1}\u{fe0f} {} v0.1.0 - The Universal Programming Language",
        config.name.to_uppercase()
    );
    println!("  Cross-Platform REPL (Windows/Linux/macOS)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Type 'exit' or Ctrl+C to quit. End statements with ~");
    println!();

    let mut rl = match DefaultEditor::new() {
        Ok(editor) => editor,
        Err(e) => {
            eprintln!("Error initializing readline: {}", e);
            return;
        }
    };

    let mut interpreter = Interpreter::new();

    loop {
        let readline = rl.readline(config.prompt);
        match readline {
            Ok(line) => {
                let trimmed = line.trim();

                if trimmed == "exit" || trimmed == "quit" {
                    println!("Goodbye! \u{1f9ad}");
                    break;
                }

                if trimmed.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(&line);

                match execute_line(&mut interpreter, trimmed) {
                    Ok(result) => println!("{}", result),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("\nInterrupted. Type 'exit' to quit.");
            }
            Err(ReadlineError::Eof) => {
                println!("\nGoodbye! \u{1f9ad}");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }
}

fn execute_line(interpreter: &mut Interpreter, source: &str) -> Result<String, String> {
    let mut parser = Parser::new(source);
    let ast = parser
        .parse()
        .map_err(|e| format!("Parse error: {:?}", e))?;
    let value = interpreter
        .execute(&ast)
        .map_err(|e| format!("Runtime error: {}", e))?;
    Ok(format!("{:?}", value))
}

pub fn run_file(config: &LangConfig, path: &Path, mode: &str) {
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  \u{1f6e1}\u{fe0f} {} - Running: {}",
        config.name.to_uppercase(),
        path.display()
    );
    println!("  Mode: {}", mode);
    println!("═══════════════════════════════════════════════════════════════");

    let source = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            std::process::exit(1);
        }
    };

    if let Some(ext) = path.extension() {
        let s = ext.to_string_lossy();
        if !config.extensions.contains(&s.as_ref()) {
            let exts: Vec<_> = config.extensions.iter().map(|e| format!(".{}", e)).collect();
            println!(
                "Warning: File extension '.{}' is not standard ({})",
                s,
                exts.join(" or ")
            );
        }
    }

    let mut parser = Parser::new(&source);
    let ast = match parser.parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Parse error: {:?}", e);
            std::process::exit(1);
        }
    };

    if mode == "titan" {
        let compiler = Compiler::new();
        let code = compiler.compile(&ast);
        let mut vm = TitanVM::new();
        vm.load_code(code);
        match vm.run() {
            Ok(result) => {
                println!("{:?}", result);
                println!();
                println!("Titan Execution complete. \u{26a1}");
            }
            Err(e) => {
                eprintln!("Titan Runtime error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        let mut interpreter = Interpreter::new();
        match interpreter.execute(&ast) {
            Ok(result) => {
                println!("{:?}", result);
                println!();
                println!("Bio-Script Execution complete. \u{1f9ad}");
            }
            Err(e) => {
                eprintln!("Runtime error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

pub fn check_file(path: &Path) {
    println!("Checking: {}", path.display());

    let source = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            std::process::exit(1);
        }
    };

    let mut parser = Parser::new(&source);
    match parser.parse() {
        Ok(_) => println!("\u{2713} Syntax OK"),
        Err(e) => {
            eprintln!("\u{274c} Parse error: {:?}", e);
            std::process::exit(1);
        }
    }
}
