pub mod engine;
pub mod ipc;
pub mod native;
pub mod pipeline;
pub mod sample;
pub mod state;
pub mod timeline;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(state::AppState::default())
        .invoke_handler(tauri::generate_handler![
            ipc::commands::scan_engines,
            ipc::commands::engine_start,
            ipc::commands::engine_stop,
            ipc::commands::engine_status,
            ipc::commands::run_engine,
            ipc::commands::pipeline_validate,
            ipc::commands::pipeline_run,
            ipc::commands::scan_datasets,
            ipc::commands::dataset_preview,
            ipc::commands::experiments_list,
            ipc::commands::experiment_get,
            ipc::commands::experiment_save,
            ipc::commands::run_analysis,
            ipc::commands::run_battle,
            ipc::commands::run_rank,
            ipc::commands::run_regress,
            ipc::commands::run_classify,
            ipc::commands::create_engine,
            ipc::commands::run_formula,
            ipc::commands::formula_build,
            ipc::commands::scan_templates,
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state = handle.state::<state::AppState>();
                let mut registry = state.engine_registry.lock().await;
                let engines_dir = handle
                    .path()
                    .app_data_dir()
                    .unwrap_or_default()
                    .join("engines");
                let _ = registry.scan_directory(&engines_dir);
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
