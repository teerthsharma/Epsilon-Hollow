// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Boot splash screen with seal ASCII art.

use crate::llm::LlmConfig;

const SEAL_ART: &str = r#"
                         .--"""""--.
                       .'            '.
                      /    .-----.     \
                     |   .'       '.    |
                     |  /   .---.   \   |
                     | ;   ( o o )   ;  |
                     | |    '-.-'    |  |
                     |  \   .---.   /   |
                     |   '.       .'    |
                      \    '-----'     /
                       '.    ___    .'
                     _.--'--'   '--'--._
                   .'    /         \    '.
                  ;     ;    🦭     ;     ;
                  |     |           |     |
                  ;     ;           ;     ;
                   '.    \         /    .'
                     '-.__'-------'__.-'
"#;

const BOX_TOP: &str = "    ╔═══════════════════════════════════════════════╗";
const BOX_BOT: &str = "    ╚═══════════════════════════════════════════════╝";

pub fn print_splash(config: &LlmConfig, dim: usize, capacity: usize, theorems_passed: usize) {
    print!("{SEAL_ART}");
    println!("{BOX_TOP}");
    println!(
        "    ║  EPSILON-HOLLOW v0.5.0                        ║"
    );
    println!(
        "    ║  Geometrical Scientific Rust OS                ║"
    );
    println!(
        "    ║                                               ║"
    );
    println!(
        "    ║  Theorems: T1-T10 verified ({}/10)    ║",
        if theorems_passed == 10 {
            "10/10 ✓".to_string()
        } else {
            format!("{theorems_passed}/10 !")
        }
    );
    println!(
        "    ║  Memory: dim={:<4} capacity={:<7}          ║",
        dim, capacity
    );
    println!(
        "    ║  LLM: {:<42}║",
        if config.is_configured() {
            format!("{} ({})", config.provider_name(), config.model)
        } else {
            "offline — core math active".into()
        }
    );
    println!("{BOX_BOT}");
    println!();
}
