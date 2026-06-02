use crate::engine::manifest::EngineManifest;
use crate::native;
use crate::pipeline::graph::{PipelineGraph, ValidationResult};
use crate::state::AppState;
use crate::timeline::history::Experiment;
use serde_json::{json, Value};
use std::path::PathBuf;
use tauri::State;

fn project_root() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    // Walk up from exe: target/debug/exe → target/debug → target → src-tauri → repo-root
    let mut root = exe.clone();
    for _ in 0..8 {
        if root.join("data").exists() && root.join("cli").exists() {
            return root;
        }
        match root.parent() {
            Some(p) => root = p.to_path_buf(),
            None => break,
        }
    }
    // Fallback: CWD parent (Tauri dev runs from src-tauri/)
    let cwd = std::env::current_dir().unwrap_or(exe);
    if cwd.join("data").exists() {
        return cwd;
    }
    if let Some(p) = cwd.parent() {
        if p.join("data").exists() {
            return p.to_path_buf();
        }
    }
    cwd
}

fn file_mtime(path: &std::path::Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

async fn run_cli(args: &[&str]) -> Result<Value, String> {
    native::command(args)
}

fn native_engine_id(command: &str) -> Option<&str> {
    let mut parts = command.split_whitespace();
    if parts.next()? == "laamba-native" {
        parts.next()
    } else {
        None
    }
}

// ── Engine commands ──

#[tauri::command]
pub async fn scan_engines(
    paths: Vec<String>,
    state: State<'_, AppState>,
) -> Result<Vec<EngineManifest>, String> {
    let mut registry = state.engine_registry.lock().await;
    let root = project_root();
    // Always scan the engines/ dir
    let _ = registry.scan_directory(&root.join("engines"));
    for p in paths {
        let _ = registry.scan_directory(std::path::Path::new(&p));
    }
    Ok(registry.list().into_iter().cloned().collect())
}

#[tauri::command]
pub async fn engine_start(
    id: String,
    params: Value,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let registry = state.engine_registry.lock().await;
    let manifest = registry.get(&id).ok_or("Engine not found")?;
    if let Some(engine_id) = native_engine_id(&manifest.entry.command) {
        return Ok(format!("native:{engine_id}"));
    }
    let child = crate::engine::manager::EngineManager::start(manifest, params, None).await?;
    let pid = format!("{}", child.id().unwrap_or(0));
    drop(registry);
    state
        .running_processes
        .lock()
        .await
        .insert(pid.clone(), child);
    Ok(pid)
}

#[tauri::command]
pub async fn engine_stop(pid: String, state: State<'_, AppState>) -> Result<(), String> {
    let mut procs = state.running_processes.lock().await;
    if let Some(mut child) = procs.remove(&pid) {
        let _ = child.kill().await;
    }
    Ok(())
}

#[tauri::command]
pub async fn engine_status(pid: String, state: State<'_, AppState>) -> Result<String, String> {
    let procs = state.running_processes.lock().await;
    if procs.contains_key(&pid) {
        Ok("running".to_string())
    } else {
        Ok("stopped".to_string())
    }
}

#[tauri::command]
pub async fn run_engine(
    id: String,
    path: String,
    params: Value,
    state: State<'_, AppState>,
) -> Result<Value, String> {
    let registry = state.engine_registry.lock().await;
    let manifest = registry.get(&id).ok_or("Engine not found")?;
    if let Some(engine_id) = native_engine_id(&manifest.entry.command) {
        return Ok(native::engine_command(engine_id, Some(&path), params));
    }
    let child = crate::engine::manager::EngineManager::start(manifest, params, Some(&path)).await?;
    let pid = format!("{}", child.id().unwrap_or(0));
    drop(registry);
    state
        .running_processes
        .lock()
        .await
        .insert(pid.clone(), child);
    Ok(json!({ "pid": pid, "status": "started" }))
}

// ── Dataset commands ──

#[tauri::command]
pub async fn scan_datasets(_paths: Vec<String>) -> Result<Value, String> {
    let root = project_root();
    let data_dir = root.join("data");
    let index_path = data_dir.join("index.json");

    if index_path.exists() {
        let content = std::fs::read_to_string(&index_path).map_err(|e| e.to_string())?;
        let mut parsed: Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
        // Resolve relative paths to absolute (files live in data/)
        if let Some(datasets) = parsed.get_mut("datasets").and_then(|d| d.as_array_mut()) {
            for ds in datasets.iter_mut() {
                if let Some(p) = ds.get("path").and_then(|p| p.as_str()) {
                    let abs = if std::path::Path::new(p).is_absolute() {
                        p.to_string()
                    } else {
                        data_dir.join(p).to_string_lossy().to_string()
                    };
                    ds["path"] = serde_json::Value::String(abs);
                }
            }
        }
        return Ok(parsed);
    }

    // Fallback: list CSV files
    let mut datasets = Vec::new();
    if data_dir.exists() {
        for entry in std::fs::read_dir(&data_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("csv") {
                datasets.push(serde_json::json!({
                    "name": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
                    "path": path.to_string_lossy(),
                    "type": "point_cloud",
                    "format": "csv",
                }));
            }
        }
    }
    Ok(serde_json::json!({"datasets": datasets}))
}

#[tauri::command]
pub async fn dataset_preview(path: String, state: State<'_, AppState>) -> Result<Value, String> {
    // Check vitals cache first
    let mtime = file_mtime(std::path::Path::new(&path));
    {
        let cache = state.vitals_cache.lock().await;
        if let Some((cached_mtime, value)) = cache.get(&path) {
            if mtime == Some(*cached_mtime) {
                return Ok(value.clone());
            }
        }
    }

    let result = run_cli(&["preview", &path]).await?;

    // Store in cache
    if let Some(mt) = mtime {
        let mut cache = state.vitals_cache.lock().await;
        cache.insert(path, (mt, result.clone()));
    }

    Ok(result)
}

// ── Analysis commands ──

#[tauri::command]
pub async fn run_analysis(path: String) -> Result<Value, String> {
    run_cli(&["analyze", &path]).await
}

#[tauri::command]
pub async fn run_battle(path: String) -> Result<Value, String> {
    run_cli(&["battle", &path]).await
}

#[tauri::command]
pub async fn run_rank(path: String) -> Result<Value, String> {
    run_cli(&["rank", &path]).await
}

#[tauri::command]
pub async fn run_regress(path: String, target: Option<i32>) -> Result<Value, String> {
    let mut args = vec!["regress", &path];
    let target_str;
    if let Some(t) = target {
        target_str = format!("--target={}", t);
        args.push(&target_str);
    }
    run_cli(&args).await
}

#[tauri::command]
pub async fn run_classify(path: String, target: Option<i32>) -> Result<Value, String> {
    let mut args = vec!["classify", &path];
    let target_str;
    if let Some(t) = target {
        target_str = format!("--target={}", t);
        args.push(&target_str);
    }
    run_cli(&args).await
}

#[tauri::command]
pub async fn create_engine(name: String, task: String, topology: String) -> Result<Value, String> {
    run_cli(&[
        "create_engine",
        &name,
        &format!("--task={}", task),
        &format!("--topology={}", topology),
    ])
    .await
}

#[tauri::command]
pub async fn run_formula(path: String, source: String) -> Result<Value, String> {
    // Write source to a temp file so CLI can read it
    let root = project_root();
    let tmp = root.join(".formula_tmp.txt");
    std::fs::write(&tmp, source).map_err(|e| e.to_string())?;
    let result = run_cli(&[
        "formula",
        &path,
        &format!("--source={}", tmp.to_string_lossy()),
    ])
    .await;
    let _ = std::fs::remove_file(&tmp);
    result
}

#[tauri::command]
pub async fn formula_build(name: String, source: String) -> Result<Value, String> {
    let root = project_root();
    let tmp = root.join(".formula_tmp.txt");
    std::fs::write(&tmp, source).map_err(|e| e.to_string())?;
    let result = run_cli(&[
        "formula_build",
        &name,
        &format!("--source={}", tmp.to_string_lossy()),
    ])
    .await;
    let _ = std::fs::remove_file(&tmp);
    result
}

// ── Pipeline commands ──

#[tauri::command]
pub fn pipeline_validate(graph: PipelineGraph) -> ValidationResult {
    graph.validate()
}

#[tauri::command]
pub async fn pipeline_run(
    graph: PipelineGraph,
    state: State<'_, AppState>,
) -> Result<Value, String> {
    let registry = state.engine_registry.lock().await;
    let result = crate::pipeline::executor::PipelineExecutor::execute(&graph, &registry).await?;
    Ok(serde_json::to_value(result).unwrap_or_default())
}

// ── Template commands ──

#[tauri::command]
pub async fn scan_templates() -> Result<Value, String> {
    let root = project_root();
    let templates_dir = root.join("templates");
    let mut templates = Vec::new();
    if templates_dir.exists() {
        for entry in std::fs::read_dir(&templates_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
                let parsed: Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
                templates.push(serde_json::json!({
                    "name": path.file_stem().and_then(|s| s.to_str()).unwrap_or(""),
                    "path": path.to_string_lossy(),
                    "content": parsed,
                }));
            }
        }
    }
    Ok(serde_json::json!({ "templates": templates }))
}

// ── Experiment commands ──

#[tauri::command]
pub async fn experiments_list(state: State<'_, AppState>) -> Result<Vec<Experiment>, String> {
    let hist = state.experiments.lock().await;
    Ok(hist.list().into_iter().cloned().collect())
}

#[tauri::command]
pub async fn experiment_get(
    id: String,
    state: State<'_, AppState>,
) -> Result<Option<Experiment>, String> {
    let hist = state.experiments.lock().await;
    Ok(hist.get(&id).cloned())
}

#[tauri::command]
pub async fn experiment_save(
    exp: Experiment,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let mut hist = state.experiments.lock().await;
    hist.save(exp);
    Ok("saved".to_string())
}
