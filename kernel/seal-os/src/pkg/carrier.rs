// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Carriers — native package payload labels.

use alloc::string::String;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CarrierType {
    Aether,
    Rust,
    C,
    Js,
}

impl CarrierType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Aether => "aether",
            Self::Rust => "rust",
            Self::C => "c",
            Self::Js => "js",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "aether" => Some(Self::Aether),
            "rust" => Some(Self::Rust),
            "c" => Some(Self::C),
            "js" | "javascript" => Some(Self::Js),
            _ => None,
        }
    }

    /// True for carriers that are part of the native Seal package surface.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Aether | Self::Rust)
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
            "Aether package entry '{}' is installed; launch through the kernel Aether app host",
            entry
        ))
    }
}
