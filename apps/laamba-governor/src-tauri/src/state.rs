use crate::engine::registry::EngineRegistry;
use crate::timeline::history::ExperimentHistory;
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::Mutex;

#[derive(Default)]
pub struct AppState {
    pub engine_registry: Mutex<EngineRegistry>,
    pub experiments: Mutex<ExperimentHistory>,
    pub running_processes: Mutex<HashMap<String, tokio::process::Child>>,
    /// Vitals cache: path -> (mtime_secs, result)
    pub vitals_cache: Mutex<HashMap<String, (u64, Value)>>,
}
