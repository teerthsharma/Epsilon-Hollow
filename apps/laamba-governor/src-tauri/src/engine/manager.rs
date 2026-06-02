use crate::engine::manifest::EngineManifest;
use serde_json::Value;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;

pub struct EngineManager;

impl EngineManager {
    pub async fn start(
        manifest: &EngineManifest,
        params: Value,
        path: Option<&str>,
    ) -> Result<tokio::process::Child, String> {
        let parts = shell_words::split(&manifest.entry.command)
            .map_err(|e| format!("Invalid command in engine manifest: {}", e))?;
        if parts.is_empty() {
            return Err("Empty command in engine manifest".to_string());
        }

        let is_native = parts[0] == "laamba-native";
        let mut cmd = if is_native {
            let mut cmd =
                tokio::process::Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
            cmd.arg("--laamba-native-engine");
            if parts.len() > 1 {
                cmd.args(&parts[1..]);
            }
            cmd
        } else {
            let mut cmd = tokio::process::Command::new(parts[0].clone());
            if parts.len() > 1 {
                cmd.args(&parts[1..]);
            }
            cmd
        };

        if let Some(wd) = &manifest.entry.working_dir {
            if is_native {
                // Native engines run in this binary; legacy manifest working dirs
                // must not send the self-hosted child into a missing source tree.
            } else {
                cmd.current_dir(wd);
            }
        }

        if let Some(p) = path {
            cmd.env("LAAMBA_DATASET_PATH", p);
        }

        match params {
            Value::Array(arr) => {
                for v in arr {
                    if let Some(s) = v.as_str() {
                        cmd.arg(s);
                    } else {
                        cmd.arg(v.to_string());
                    }
                }
            }
            Value::Null => {}
            other => {
                let json = other.to_string();
                cmd.stdin(Stdio::piped());
                let mut child = cmd
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true)
                    .spawn()
                    .map_err(|e| e.to_string())?;
                if let Some(mut stdin) = child.stdin.take() {
                    stdin
                        .write_all(json.as_bytes())
                        .await
                        .map_err(|e| format!("Failed to write to stdin: {}", e))?;
                }
                return Ok(child);
            }
        }

        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        cmd.spawn().map_err(|e| e.to_string())
    }
}
