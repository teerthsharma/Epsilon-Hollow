// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Carriers — language runtime bridges for package execution.

use alloc::string::String;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CarrierType {
    Aether,
    Rust,
    Python,
    C,
    Js,
}

impl CarrierType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Aether => "aether",
            Self::Rust => "rust",
            Self::Python => "python",
            Self::C => "c",
            Self::Js => "js",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "aether" => Some(Self::Aether),
            "rust" => Some(Self::Rust),
            "python" | "pip" => Some(Self::Python),
            "c" => Some(Self::C),
            "js" | "javascript" => Some(Self::Js),
            _ => None,
        }
    }

    /// All carriers are metadata-only; no real runtime is available.
    pub fn is_available(&self) -> bool {
        // No actual runtimes — all carriers store metadata only
        matches!(self, Self::Aether | Self::Rust | Self::Python)
    }
}

pub trait Carrier {
    fn carrier_type(&self) -> CarrierType;
    fn execute(&self, entry: &str) -> Result<String, String>;
}

pub struct AetherCarrier;

impl Carrier for AetherCarrier {
    fn carrier_type(&self) -> CarrierType {
        CarrierType::Aether
    }

    fn execute(&self, entry: &str) -> Result<String, String> {
        Ok(alloc::format!(
            "No runtime — package metadata stored only (aether entry: '{}')",
            entry
        ))
    }
}

pub struct PipCarrier;

impl Carrier for PipCarrier {
    fn carrier_type(&self) -> CarrierType {
        CarrierType::Python
    }

    fn execute(&self, entry: &str) -> Result<String, String> {
        Ok(alloc::format!(
            "No runtime — package metadata stored only (pip entry: '{}')",
            entry
        ))
    }
}
