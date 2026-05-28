use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub pipeline: serde_json::Value,
    pub params: HashMap<String, serde_json::Value>,
    pub results: Option<serde_json::Value>,
    pub timestamp: i64,
    pub duration_ms: u64,
    pub status: ExperimentStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExperimentStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

#[derive(Default)]
pub struct ExperimentHistory {
    pub experiments: Vec<Experiment>,
}

impl ExperimentHistory {
    pub fn list(&self) -> Vec<&Experiment> {
        self.experiments.iter().collect()
    }

    pub fn get(&self, id: &str) -> Option<&Experiment> {
        self.experiments.iter().find(|e| e.id == id)
    }

    pub fn save(&mut self, exp: Experiment) {
        self.experiments.retain(|e| e.id != exp.id);
        self.experiments.push(exp);
    }
}
