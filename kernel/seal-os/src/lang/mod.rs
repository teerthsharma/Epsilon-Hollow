// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Aether-Lang runtime bridge — connects SealShell to the AEGIS interpreter.

pub mod stdlib;

use alloc::format;
use alloc::string::String;

pub struct AetherRuntime {
    interpreter: aether_lang::Interpreter,
}

impl AetherRuntime {
    pub fn new() -> Self {
        Self {
            interpreter: aether_lang::Interpreter::new(),
        }
    }

    pub fn execute_source(&mut self, source: &str) -> Result<String, String> {
        let mut parser = aether_lang::Parser::new(source);
        let program = parser.parse().map_err(|e| format!("parse error: {}", e.message))?;
        let value = self.interpreter.execute(&program).map_err(|e| format!("{:?}", e))?;
        Ok(format!("{:?}", value))
    }

    pub fn execute_file(&mut self, name: &str, source: &str) -> Result<String, String> {
        self.execute_source(source)
            .map_err(|e| format!("in '{}': {}", name, e))
    }

    pub fn is_ready(&self) -> bool {
        true
    }
}
