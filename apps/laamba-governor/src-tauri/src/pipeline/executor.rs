use super::graph::{PipelineGraph, PipelineNode};
use crate::engine::registry::EngineRegistry;
use crate::native;
use std::collections::HashMap;

pub struct PipelineExecutor;

impl PipelineExecutor {
    pub async fn execute(
        graph: &PipelineGraph,
        registry: &EngineRegistry,
    ) -> Result<HashMap<String, serde_json::Value>, String> {
        let order = graph.topological_order()?;
        let mut outputs: HashMap<String, serde_json::Value> = HashMap::new();

        for node in order {
            let result = Self::run_node(node, &outputs, registry).await?;
            outputs.insert(node.id.clone(), result);
        }

        Ok(outputs)
    }

    async fn run_node(
        node: &PipelineNode,
        upstream: &HashMap<String, serde_json::Value>,
        registry: &EngineRegistry,
    ) -> Result<serde_json::Value, String> {
        match node.node_type.as_str() {
            "datasource" | "data_source" | "dataset" => {
                let path = node
                    .params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .or_else(|| {
                        upstream
                            .values()
                            .find(|v| v.get("path").is_some())
                            .and_then(|v| v.get("path").and_then(|p| p.as_str()))
                    })
                    .ok_or("DataSource missing path")?;
                Ok(serde_json::json!({ "path": path, "type": "dataset" }))
            }
            "euclidean" | "spherical" | "hyperbolic_poincare" | "grassmannian" | "product" => {
                let ds = upstream
                    .values()
                    .find(|v| {
                        v.get("type") == Some(&serde_json::Value::String("dataset".to_string()))
                    })
                    .or_else(|| upstream.values().next())
                    .ok_or("No upstream data for topology node")?;
                let path = ds
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or("Upstream missing path")?;
                Self::run_native(&["analyze", path, &format!("--topology={}", node.node_type)])
                    .await
            }
            "comparator" | "battle" | "battle_royale" => {
                let ds = upstream
                    .values()
                    .find(|v| {
                        v.get("type") == Some(&serde_json::Value::String("dataset".to_string()))
                    })
                    .or_else(|| upstream.values().next())
                    .ok_or("No upstream data for battle node")?;
                let path = ds
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or("Upstream missing path")?;
                Self::run_native(&["battle", path]).await
            }
            _ => {
                if let Some(engine_id) = &node.engine_id {
                    if let Some(manifest) = registry.get(engine_id) {
                        let params =
                            serde_json::Value::Object(node.params.clone().into_iter().collect());
                        let child =
                            crate::engine::manager::EngineManager::start(manifest, params, None)
                                .await?;
                        let pid = format!("{}", child.id().unwrap_or(0));
                        Ok(serde_json::json!({
                            "engine_id": engine_id,
                            "node_type": node.node_type,
                            "pid": pid,
                            "status": "started"
                        }))
                    } else {
                        Err(format!("Engine '{}' not found in registry", engine_id))
                    }
                } else {
                    // Pass-through: aggregate upstream outputs
                    let mut merged = serde_json::Map::new();
                    for (k, v) in upstream {
                        merged.insert(k.clone(), v.clone());
                    }
                    merged.insert(
                        "node_type".to_string(),
                        serde_json::Value::String(node.node_type.clone()),
                    );
                    Ok(serde_json::Value::Object(merged))
                }
            }
        }
    }
    async fn run_native(args: &[&str]) -> Result<serde_json::Value, String> {
        native::command(args)
    }
}
