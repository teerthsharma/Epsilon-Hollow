use super::manifest::{EngineManifest, EngineManifestFile};
use std::collections::HashMap;
use std::path::Path;

#[derive(Default)]
pub struct EngineRegistry {
    pub engines: HashMap<String, EngineManifest>,
}

impl EngineRegistry {
    pub fn scan_directory(&mut self, dir: &Path) -> Result<(), String> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    match toml::from_str::<EngineManifestFile>(&content) {
                        Ok(file) => {
                            let id = file.engine.name.to_lowercase().replace(' ', "-");
                            self.engines.insert(id, file.engine);
                        }
                        Err(e) => {
                            eprintln!("Failed to parse {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&EngineManifest> {
        self.engines.get(id)
    }

    pub fn list(&self) -> Vec<&EngineManifest> {
        self.engines.values().collect()
    }
}
