use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Wrapper that matches the TOML structure where everything is under [engine]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineManifestFile {
    pub engine: EngineManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineManifest {
    pub name: String,
    pub category: String,
    #[serde(default)]
    pub icon: String,
    #[serde(default)]
    pub color: String,
    #[serde(default)]
    pub description: String,
    pub entry: EngineEntry,
    #[serde(default)]
    pub inputs: HashMap<String, PortSpec>,
    #[serde(default)]
    pub outputs: HashMap<String, PortSpec>,
    #[serde(default)]
    pub metrics: HashMap<String, MetricSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub command: String,
    #[serde(default)]
    pub working_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortSpec {
    #[serde(rename = "type")]
    pub port_type: String,
    #[serde(default, alias = "desc")]
    pub description: Option<String>,
    #[serde(default)]
    pub default: Option<serde_json::Value>,
    #[serde(default)]
    pub range: Option<Vec<f64>>,
    #[serde(default)]
    pub shape: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSpec {
    pub source: String,
    pub pattern: String,
}
