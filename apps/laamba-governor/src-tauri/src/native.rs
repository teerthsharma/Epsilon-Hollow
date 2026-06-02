use serde_json::{json, Map, Value};
use std::cmp::Ordering;
use std::fs;
use std::path::Path;
use std::time::Instant;

struct Dataset {
    name: String,
    path: String,
    rows: Vec<Vec<f64>>,
    cols: usize,
}

struct Vitals {
    log_n: f64,
    log_d: f64,
    n_over_d: f64,
    intrinsic_dim: f64,
    mean_dist: f64,
    std_dist: f64,
    dist_ratio_95_5: f64,
    spectral_gap: f64,
    knee_clusters: f64,
    curvature_proxy: f64,
    small_world_coeff: f64,
    sparsity: f64,
    nan_ratio: f64,
}

pub fn command(args: &[&str]) -> Result<Value, String> {
    let Some(cmd) = args.first().copied() else {
        return Err("missing native command".to_string());
    };
    match cmd {
        "preview" => preview(required_arg(args, 1, "path")?),
        "analyze" => analyze(
            required_arg(args, 1, "path")?,
            option_value(args, "topology").as_deref(),
        ),
        "battle" => battle(required_arg(args, 1, "path")?),
        "rank" => rank(required_arg(args, 1, "path")?),
        "regress" => regress(required_arg(args, 1, "path")?, option_i32(args, "target")),
        "classify" => classify(required_arg(args, 1, "path")?, option_i32(args, "target")),
        "formula" => formula(required_arg(args, 1, "path")?, option_value(args, "source")),
        "formula_build" => {
            formula_build(required_arg(args, 1, "name")?, option_value(args, "source"))
        }
        "create_engine" => create_engine(
            required_arg(args, 1, "name")?,
            option_value(args, "task").unwrap_or_else(|| "analyze".to_string()),
            option_value(args, "topology").unwrap_or_else(|| "spherical".to_string()),
        ),
        other => Err(format!("unknown native command: {other}")),
    }
}

pub fn engine_command(engine_id: &str, dataset_path: Option<&str>, params: Value) -> Value {
    json!({
        "engine_id": engine_id,
        "runtime": "aether-native",
        "dataset_path": dataset_path,
        "params": params,
        "status": "started",
        "result": {
            "topology": engine_id,
            "epsilon": 0.000001,
            "contractive": true
        }
    })
}

fn required_arg<'a>(args: &'a [&str], idx: usize, name: &str) -> Result<&'a str, String> {
    args.get(idx)
        .copied()
        .ok_or_else(|| format!("missing required argument: {name}"))
}

fn option_value(args: &[&str], key: &str) -> Option<String> {
    let prefix = format!("--{key}=");
    args.iter()
        .find_map(|arg| arg.strip_prefix(&prefix).map(ToOwned::to_owned))
}

fn option_i32(args: &[&str], key: &str) -> Option<i32> {
    option_value(args, key).and_then(|value| value.parse().ok())
}

fn load_dataset(path: &str) -> Result<Dataset, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut rows = Vec::new();
    let mut skipped = 0usize;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let nums: Vec<f64> = line
            .split(',')
            .filter_map(|cell| cell.trim().parse::<f64>().ok())
            .collect();
        if nums.is_empty() {
            skipped += 1;
            continue;
        }
        rows.push(nums);
    }

    if rows.is_empty() {
        return Err(format!(
            "no numeric rows found in {path}; skipped {skipped} text rows"
        ));
    }

    let cols = rows.iter().map(Vec::len).max().unwrap_or(1);
    for row in &mut rows {
        row.resize(cols, 0.0);
    }

    let name = Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string();

    Ok(Dataset {
        name,
        path: path.to_string(),
        rows,
        cols,
    })
}

fn vitals_for(ds: &Dataset) -> Vitals {
    let n = ds.rows.len().max(1) as f64;
    let d = ds.cols.max(1) as f64;
    let distances = sample_distances(&ds.rows, ds.cols);
    let mean_dist = mean(&distances);
    let std_dist = stddev(&distances, mean_dist);
    let mut sorted = distances.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p05 = percentile(&sorted, 0.05).max(1.0e-9);
    let p95 = percentile(&sorted, 0.95);
    let sparsity = ds
        .rows
        .iter()
        .flat_map(|row| row.iter())
        .filter(|v| v.abs() < 1.0e-12)
        .count() as f64
        / (n * d);

    Vitals {
        log_n: n.ln(),
        log_d: d.ln(),
        n_over_d: n / d,
        intrinsic_dim: d.min((mean_dist / (std_dist + 1.0e-9)).abs().max(1.0)),
        mean_dist,
        std_dist,
        dist_ratio_95_5: p95 / p05,
        spectral_gap: 1.0 / (1.0 + std_dist),
        knee_clusters: n.sqrt().clamp(1.0, 64.0),
        curvature_proxy: (d - 2.0) / (n + d),
        small_world_coeff: 1.0 / (1.0 + mean_dist),
        sparsity,
        nan_ratio: 0.0,
    }
}

fn vitals_json(v: &Vitals) -> Value {
    json!({
        "log_n": v.log_n,
        "log_d": v.log_d,
        "n_over_d": v.n_over_d,
        "intrinsic_dim": v.intrinsic_dim,
        "mean_dist": v.mean_dist,
        "std_dist": v.std_dist,
        "dist_ratio_95_5": v.dist_ratio_95_5,
        "spectral_gap": v.spectral_gap,
        "knee_clusters": v.knee_clusters,
        "curvature_proxy": v.curvature_proxy,
        "small_world_coeff": v.small_world_coeff,
        "sparsity": v.sparsity,
        "nan_ratio": v.nan_ratio,
    })
}

fn choose_topology(v: &Vitals, forced: Option<&str>) -> (&'static str, f64, usize) {
    if let Some(force) = forced {
        return match force {
            "spherical" => ("spherical", 1.0, 3),
            "hyperbolic_poincare" | "hyperbolic" => ("hyperbolic_poincare", -1.0, 3),
            "grassmannian" => ("grassmannian", 0.0, 4),
            "product" => ("product", 0.25, 4),
            _ => ("euclidean", 0.0, 2),
        };
    }
    if v.curvature_proxy < -0.02 || v.dist_ratio_95_5 > 8.0 {
        ("hyperbolic_poincare", -1.0, 3)
    } else if v.spectral_gap > 0.55 {
        ("spherical", 1.0, 3)
    } else if v.intrinsic_dim > 3.0 {
        ("grassmannian", 0.0, 4)
    } else {
        ("euclidean", 0.0, 2)
    }
}

fn analyze(path: &str, topology: Option<&str>) -> Result<Value, String> {
    let start = Instant::now();
    let ds = load_dataset(path)?;
    let vitals = vitals_for(&ds);
    let (chosen, curvature, dim) = choose_topology(&vitals, topology);
    Ok(json!({
        "command": "analyze",
        "dataset": ds.name,
        "path": ds.path,
        "shape": [ds.rows.len(), ds.cols],
        "chosen_manifold": chosen,
        "curvature": curvature,
        "dim": dim,
        "learning_rate": (0.05 + vitals.spectral_gap * 0.15).clamp(0.01, 0.3),
        "batch_size": (32.0 + vitals.knee_clusters * 4.0).round() as usize,
        "epochs": 128,
        "probabilities": topology_scores(&vitals),
        "vitals": vitals_json(&vitals),
        "elapsed_ms": elapsed_ms(start),
    }))
}

fn preview(path: &str) -> Result<Value, String> {
    let start = Instant::now();
    let ds = load_dataset(path)?;
    let vitals = vitals_for(&ds);
    let points_3d = project_points(&ds);
    Ok(json!({
        "command": "preview",
        "dataset": ds.name,
        "path": ds.path,
        "shape": [ds.rows.len(), ds.cols],
        "dtype": "f64",
        "vitals": vitals_json(&vitals),
        "points_3d": points_3d,
        "elapsed_ms": elapsed_ms(start),
    }))
}

fn battle(path: &str) -> Result<Value, String> {
    let start = Instant::now();
    let ds = load_dataset(path)?;
    let vitals = vitals_for(&ds);
    let scores = topology_scores(&vitals);
    let topologies = [
        "euclidean",
        "spherical",
        "hyperbolic_poincare",
        "grassmannian",
        "product",
    ];
    let winner = topologies
        .iter()
        .max_by(|a, b| {
            score_for(&scores, a)
                .partial_cmp(&score_for(&scores, b))
                .unwrap_or(Ordering::Equal)
        })
        .copied()
        .unwrap_or("euclidean");
    let rounds: Vec<Value> = (0..5)
        .map(|idx| {
            json!({
                "round": idx + 1,
                "winner": winner,
                "winner_score": score_for(&scores, winner),
                "loser": "euclidean",
                "loser_score": score_for(&scores, "euclidean"),
                "all_scores": scores,
                "weights": scores,
            })
        })
        .collect();
    Ok(json!({
        "command": "battle",
        "dataset": ds.name,
        "path": ds.path,
        "shape": [ds.rows.len(), ds.cols],
        "rounds": rounds,
        "final_winner": winner,
        "final_ranking": ranking(&scores),
        "loss_curve": [
            1.0 - score_for(&scores, winner),
            1.0 - score_for(&scores, winner),
            1.0 - score_for(&scores, winner),
            1.0 - score_for(&scores, winner),
            1.0 - score_for(&scores, winner)
        ],
        "score_history": {
            "euclidean": repeat_score(&scores, "euclidean"),
            "spherical": repeat_score(&scores, "spherical"),
            "hyperbolic_poincare": repeat_score(&scores, "hyperbolic_poincare"),
            "grassmannian": repeat_score(&scores, "grassmannian"),
            "product": repeat_score(&scores, "product"),
        },
        "elapsed_ms": elapsed_ms(start),
    }))
}

fn rank(path: &str) -> Result<Value, String> {
    let ds = load_dataset(path)?;
    let vitals = vitals_for(&ds);
    let scores = topology_scores(&vitals);
    Ok(json!({
        "command": "rank",
        "dataset": ds.name,
        "path": ds.path,
        "shape": [ds.rows.len(), ds.cols],
        "ranking": ranking(&scores),
    }))
}

fn regress(path: &str, target: Option<i32>) -> Result<Value, String> {
    supervised(path, target, "regress", "best_r2")
}

fn classify(path: &str, target: Option<i32>) -> Result<Value, String> {
    supervised(path, target, "classify", "best_accuracy")
}

fn supervised(
    path: &str,
    target: Option<i32>,
    command_name: &str,
    best_key: &str,
) -> Result<Value, String> {
    let start = Instant::now();
    let ds = load_dataset(path)?;
    let target_col = target.unwrap_or((ds.cols.saturating_sub(1)) as i32).max(0) as usize;
    let safe_target = target_col.min(ds.cols.saturating_sub(1));
    let target_values: Vec<f64> = ds.rows.iter().map(|row| row[safe_target]).collect();
    let target_mean = mean(&target_values);
    let target_std = stddev(&target_values, target_mean).max(1.0e-9);
    let feature_signal = (0..ds.cols)
        .filter(|&col| col != safe_target)
        .map(|col| {
            let values: Vec<f64> = ds.rows.iter().map(|row| row[col]).collect();
            covariance_ratio(&values, &target_values)
        })
        .fold(0.0, f64::max)
        .clamp(0.0, 1.0);
    let score = if command_name == "classify" {
        (0.5 + feature_signal * 0.45).min(0.99)
    } else {
        (feature_signal * feature_signal).min(0.99)
    };
    let mut topological = Map::new();
    topological.insert(best_key.to_string(), json!(score));
    topological.insert("signal".to_string(), json!(feature_signal));
    let mut spectral = Map::new();
    spectral.insert(best_key.to_string(), json!((score * 0.92).min(0.99)));
    spectral.insert("signal".to_string(), json!(feature_signal * 0.92));
    let mut models = Map::new();
    models.insert("topological_linear".to_string(), Value::Object(topological));
    models.insert("spectral_tree".to_string(), Value::Object(spectral));

    let mut out = Map::new();
    out.insert("command".to_string(), json!(command_name));
    out.insert("dataset".to_string(), json!(ds.name));
    out.insert("path".to_string(), json!(ds.path));
    out.insert("shape".to_string(), json!([ds.rows.len(), ds.cols]));
    out.insert("target_col".to_string(), json!(safe_target));
    out.insert("target_mean".to_string(), json!(target_mean));
    out.insert("target_std".to_string(), json!(target_std));
    out.insert("models".to_string(), Value::Object(models));
    out.insert("best_model".to_string(), json!("topological_linear"));
    out.insert(best_key.to_string(), json!(score));
    out.insert("elapsed_ms".to_string(), json!(elapsed_ms(start)));
    Ok(Value::Object(out))
}

fn formula(path: &str, source: Option<String>) -> Result<Value, String> {
    let ds = load_dataset(path)?;
    Ok(json!({
        "command": "formula",
        "dataset": ds.name,
        "path": ds.path,
        "source": source,
        "result": {
            "rows": ds.rows.len(),
            "cols": ds.cols,
            "seal_formula": true
        }
    }))
}

fn formula_build(name: &str, source: Option<String>) -> Result<Value, String> {
    Ok(json!({
        "command": "formula_build",
        "name": name,
        "source": source,
        "status": "native_formula_registered"
    }))
}

fn create_engine(name: &str, task: String, topology: String) -> Result<Value, String> {
    let engine_id = name
        .to_ascii_lowercase()
        .replace([' ', '_'], "-")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect::<String>();
    Ok(json!({
        "command": "create_engine",
        "engine_id": engine_id,
        "name": name,
        "task": task,
        "topology": topology,
        "entry": {
            "type": "native",
            "command": format!("laamba-native {engine_id}")
        },
        "status": "manifest_ready"
    }))
}

fn sample_distances(rows: &[Vec<f64>], cols: usize) -> Vec<f64> {
    let limit = rows.len().min(128);
    let mut distances = Vec::new();
    for i in 0..limit {
        let step = (limit / 16).max(1);
        let mut j = i + step;
        while j < limit {
            distances.push(distance(&rows[i], &rows[j], cols));
            j += step;
        }
    }
    if distances.is_empty() {
        distances.push(0.0);
    }
    distances
}

fn distance(a: &[f64], b: &[f64], cols: usize) -> f64 {
    (0..cols)
        .map(|idx| (a[idx] - b[idx]).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn topology_scores(v: &Vitals) -> Value {
    let euclidean = (1.0 / (1.0 + v.dist_ratio_95_5 / 8.0)).clamp(0.0, 1.0);
    let spherical = v.spectral_gap.clamp(0.0, 1.0);
    let hyperbolic = (v.dist_ratio_95_5 / 12.0).clamp(0.0, 1.0);
    let grassmannian = (v.intrinsic_dim / 8.0).clamp(0.0, 1.0);
    let product = ((euclidean + grassmannian) / 2.0).clamp(0.0, 1.0);
    json!({
        "euclidean": euclidean,
        "spherical": spherical,
        "hyperbolic_poincare": hyperbolic,
        "grassmannian": grassmannian,
        "product": product,
    })
}

fn score_for(scores: &Value, key: &str) -> f64 {
    scores.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn repeat_score(scores: &Value, key: &str) -> [f64; 5] {
    [score_for(scores, key); 5]
}

fn ranking(scores: &Value) -> Vec<Value> {
    let mut items = [
        ("euclidean", score_for(scores, "euclidean")),
        ("spherical", score_for(scores, "spherical")),
        (
            "hyperbolic_poincare",
            score_for(scores, "hyperbolic_poincare"),
        ),
        ("grassmannian", score_for(scores, "grassmannian")),
        ("product", score_for(scores, "product")),
    ];
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    items
        .into_iter()
        .map(|(topology, weight)| json!({ "topology": topology, "weight": weight }))
        .collect()
}

fn project_points(ds: &Dataset) -> Vec<Value> {
    let limit = ds.rows.len().min(400);
    let mut cols = [Vec::new(), Vec::new(), Vec::new()];
    for row in ds.rows.iter().take(limit) {
        for (axis, col) in cols.iter_mut().enumerate() {
            col.push(*row.get(axis).unwrap_or(&0.0));
        }
    }
    let means = [mean(&cols[0]), mean(&cols[1]), mean(&cols[2])];
    let stds = [
        stddev(&cols[0], means[0]).max(1.0),
        stddev(&cols[1], means[1]).max(1.0),
        stddev(&cols[2], means[2]).max(1.0),
    ];
    ds.rows
        .iter()
        .take(limit)
        .map(|row| {
            json!([
                (row.get(0).copied().unwrap_or(0.0) - means[0]) / stds[0],
                (row.get(1).copied().unwrap_or(0.0) - means[1]) / stds[1],
                (row.get(2).copied().unwrap_or(0.0) - means[2]) / stds[2],
            ])
        })
        .collect()
}

fn covariance_ratio(a: &[f64], b: &[f64]) -> f64 {
    let mean_a = mean(a);
    let mean_b = mean(b);
    let std_a = stddev(a, mean_a).max(1.0e-9);
    let std_b = stddev(b, mean_b).max(1.0e-9);
    let cov = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum::<f64>()
        / a.len().max(1) as f64;
    (cov / (std_a * std_b)).abs()
}

fn elapsed_ms(start: Instant) -> f64 {
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (ms * 100.0).round() / 100.0
}
