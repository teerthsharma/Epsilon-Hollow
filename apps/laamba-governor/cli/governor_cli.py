#!/usr/bin/env python3
"""
GOVERNOR CLI — bridge between Tauri Rust backend and Python topology engine.
Outputs single-line JSON to stdout. Tauri reads it line-by-line.

Usage:
    python cli/governor_cli.py analyze data/ring.csv
    python cli/governor_cli.py battle data/ring.csv
    python cli/governor_cli.py vitals data/ring.csv
    python cli/governor_cli.py rank data/ring.csv
    python cli/governor_cli.py engine faraday data/ring.csv
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
import traceback

import numpy as np

# Add project root to path so we can import the existing Python modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from topological_governor_ml import TopologicalGovernor, VitalsExtractor, MANIFOLD_CATALOG
from governor_orchestrator import GovernorOrchestrator, OrchestratorConfig


def load_csv(path: str) -> np.ndarray:
    """Load a CSV file into a numpy array. Handles headers and non-numeric gracefully."""
    try:
        # Try loadtxt first (fastest for pure numeric)
        data = np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError:
        # Likely has header row — try skipping first row
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        except ValueError as e2:
            # Try genfromtxt which handles mixed types better
            try:
                data = np.genfromtxt(path, delimiter=",", skip_header=1, ndmin=2)
                # Remove NaN rows (header leftovers or empty rows)
                data = data[~np.isnan(data).any(axis=1)]
                if data.size == 0:
                    raise ValueError("No numeric data found after header removal")
            except Exception:
                emit({"error": f"Failed to load CSV: {str(e2)}. Check for text headers or non-numeric values.", "path": path})
                sys.exit(1)
    except Exception as e:
        emit({"error": f"Failed to load CSV: {str(e)}", "path": path})
        sys.exit(1)

    rows, cols = data.shape
    if rows < 1 or cols < 1:
        emit({"error": f"CSV must have at least 1 row and 1 column, got shape ({rows}, {cols})", "path": path})
        sys.exit(1)

    return data


def emit(obj: dict) -> None:
    """Print a single-line JSON object to stdout."""
    print(json.dumps(obj, default=_json_default), flush=True)


def _json_default(o):
    """JSON serializer for numpy types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def parse_params(args: list[str]) -> dict[str, str]:
    """Parse --key=value or --flag arguments into a dict."""
    params: dict[str, str] = {}
    for arg in args:
        if arg.startswith("--"):
            body = arg[2:]
            if "=" in body:
                key, val = body.split("=", 1)
                params[key] = val
            else:
                params[body] = "true"
    return params


def cmd_vitals(path: str) -> None:
    """Extract topological vitals from dataset."""
    data = load_csv(path)
    name = os.path.basename(path)

    extractor = VitalsExtractor()
    t0 = time.perf_counter()
    feats = extractor(data)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    vitals = {
        "log_n": float(feats[0]),
        "log_d": float(feats[1]),
        "n_over_d": float(feats[2]),
        "intrinsic_dim": float(feats[3]),
        "mean_dist": float(feats[4]),
        "std_dist": float(feats[5]),
        "dist_ratio_95_5": float(feats[6]),
        "spectral_gap": float(feats[7]),
        "knee_clusters": float(feats[8]),
        "curvature_proxy": float(feats[9]),
        "small_world_coeff": float(feats[10]),
        "sparsity": float(feats[11]),
        "nan_ratio": float(feats[12]),
    }

    emit({
        "command": "vitals",
        "dataset": name,
        "path": path,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "vitals": vitals,
        "elapsed_ms": round(elapsed_ms, 2),
    })


def cmd_analyze(path: str, force_topology: str | None = None) -> None:
    """Full analysis: vitals + topology prediction."""
    data = load_csv(path)
    name = os.path.basename(path)

    # Extract vitals
    extractor = VitalsExtractor()
    feats = extractor(data)

    vitals = {
        "log_n": float(feats[0]),
        "log_d": float(feats[1]),
        "n_over_d": float(feats[2]),
        "intrinsic_dim": float(feats[3]),
        "mean_dist": float(feats[4]),
        "std_dist": float(feats[5]),
        "dist_ratio_95_5": float(feats[6]),
        "spectral_gap": float(feats[7]),
        "knee_clusters": float(feats[8]),
        "curvature_proxy": float(feats[9]),
        "small_world_coeff": float(feats[10]),
        "sparsity": float(feats[11]),
        "nan_ratio": float(feats[12]),
    }

    # Run governor prediction
    governor = TopologicalGovernor()
    t0 = time.perf_counter()
    cfg = governor.predict_topology(data, temperature=0.8, force=force_topology)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs = cfg.extra.get("governor_probs", {})

    emit({
        "command": "analyze",
        "dataset": name,
        "path": path,
        "shape": list(data.shape),
        "chosen_manifold": cfg.name,
        "curvature": cfg.curvature,
        "dim": cfg.dim,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "probabilities": probs,
        "vitals": vitals,
        "elapsed_ms": round(elapsed_ms, 2),
    })


def cmd_battle(path: str) -> None:
    """Run battle royale across all topologies."""
    data = load_csv(path)
    name = os.path.basename(path)

    topologies = ["euclidean", "spherical", "hyperbolic_poincare", "grassmannian", "product"]

    cfg = OrchestratorConfig(
        topologies=topologies,
        task="cluster",
        auto_transfer=False,
        learning_rate=0.2,
    )
    orchestrator = GovernorOrchestrator(cfg)

    # Run multiple rounds
    rounds = []
    t0 = time.perf_counter()
    for i in range(5):
        record = orchestrator.run_battle(data)
        rounds.append({
            "round": i + 1,
            "winner": record["winner"],
            "winner_score": float(record["winner_score"]),
            "loser": record["loser"],
            "loser_score": float(record["loser_score"]),
            "all_scores": {k: float(v) for k, v in record["all_scores"].items()},
            "weights": {k: float(v) for k, v in record["weights"].items()},
        })
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Compute final rankings
    final_weights = rounds[-1]["weights"]
    final_scores = rounds[-1]["all_scores"]
    ranking = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)

    # Build a synthetic loss curve from the rounds
    loss_curve = []
    for r in rounds:
        loss_curve.append(1.0 - r["winner_score"])

    # Also build per-topology score history
    score_history = {t: [] for t in topologies}
    for r in rounds:
        for t in topologies:
            score_history[t].append(r["all_scores"].get(t, 0.0))

    emit({
        "command": "battle",
        "dataset": name,
        "path": path,
        "shape": list(data.shape),
        "rounds": rounds,
        "final_winner": ranking[0][0],
        "final_ranking": [{"topology": k, "weight": v} for k, v in ranking],
        "loss_curve": loss_curve,
        "score_history": score_history,
        "elapsed_ms": round(elapsed_ms, 2),
    })


def cmd_preview(path: str) -> None:
    """Return vitals + real 3D PCA projection of the actual dataset."""
    data = load_csv(path)
    name = os.path.basename(path)

    # Vitals
    extractor = VitalsExtractor()
    t0 = time.perf_counter()
    feats = extractor(data)
    vitals = {
        "log_n": float(feats[0]),
        "log_d": float(feats[1]),
        "n_over_d": float(feats[2]),
        "intrinsic_dim": float(feats[3]),
        "mean_dist": float(feats[4]),
        "std_dist": float(feats[5]),
        "dist_ratio_95_5": float(feats[6]),
        "spectral_gap": float(feats[7]),
        "knee_clusters": float(feats[8]),
        "curvature_proxy": float(feats[9]),
        "small_world_coeff": float(feats[10]),
        "sparsity": float(feats[11]),
        "nan_ratio": float(feats[12]),
    }

    # Real 3D PCA projection — NOT synthetic RNG
    n_show = min(len(data), 400)
    idx = np.random.choice(len(data), n_show, replace=False) if len(data) > n_show else np.arange(len(data))
    sample = data[idx]

    # PCA to 3D
    if sample.shape[1] >= 3:
        cov = np.cov(sample.T)
        vals, vecs = np.linalg.eigh(cov)
        pts_3d = sample @ vecs[:, -3:]
    elif sample.shape[1] == 2:
        pts_3d = np.column_stack([sample, np.zeros(len(sample))])
    else:
        pts_3d = np.column_stack([sample[:, :1], np.zeros((len(sample), 2))])

    # Normalize to unit scale for viewer
    std = pts_3d.std(axis=0)
    std[std == 0] = 1.0
    pts_3d = (pts_3d - pts_3d.mean(axis=0)) / std

    elapsed_ms = (time.perf_counter() - t0) * 1000

    emit({
        "command": "preview",
        "dataset": name,
        "path": path,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "vitals": vitals,
        "points_3d": pts_3d.tolist(),
        "elapsed_ms": round(elapsed_ms, 2),
    })


def cmd_regress(path: str, target_col: int | None = None) -> None:
    """Real regression analysis on CSV."""
    data = load_csv(path)
    name = os.path.basename(path)

    # Split features / target
    if target_col is None:
        target_col = data.shape[1] - 1
    y = data[:, target_col]
    X = np.delete(data, target_col, axis=1)

    if X.shape[1] == 0:
        emit({"error": "No feature columns after removing target", "path": path})
        return

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    models = {
        "ridge": Ridge(alpha=1.0),
        "forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=2),
    }

    results = {}
    t0 = time.perf_counter()
    for mname, model in models.items():
        try:
            scores = cross_val_score(model, Xs, y, cv=min(3, len(X)), scoring="r2")
            results[mname] = {"r2_mean": float(np.mean(scores)), "r2_std": float(np.std(scores))}
        except Exception as e:
            results[mname] = {"error": str(e)}
    elapsed_ms = (time.perf_counter() - t0) * 1000

    best = max(results.items(), key=lambda kv: kv[1].get("r2_mean", -1e9))
    emit({
        "command": "regress",
        "dataset": name,
        "path": path,
        "shape": list(data.shape),
        "target_col": target_col,
        "models": results,
        "best_model": best[0],
        "best_r2": best[1].get("r2_mean"),
        "elapsed_ms": round(elapsed_ms, 2),
    })


def cmd_classify(path: str, target_col: int | None = None) -> None:
    """Real classification analysis on CSV."""
    data = load_csv(path)
    name = os.path.basename(path)

    if target_col is None:
        target_col = data.shape[1] - 1
    y = data[:, target_col].astype(int)
    X = np.delete(data, target_col, axis=1)

    if X.shape[1] == 0:
        emit({"error": "No feature columns after removing target", "path": path})
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    models = {
        "logistic": LogisticRegression(max_iter=500, C=1.0),
        "forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=2),
    }

    results = {}
    t0 = time.perf_counter()
    for mname, model in models.items():
        try:
            scores = cross_val_score(model, Xs, y, cv=min(3, len(X)), scoring="accuracy")
            results[mname] = {"accuracy_mean": float(np.mean(scores)), "accuracy_std": float(np.std(scores))}
        except Exception as e:
            results[mname] = {"error": str(e)}
    elapsed_ms = (time.perf_counter() - t0) * 1000

    best = max(results.items(), key=lambda kv: kv[1].get("accuracy_mean", -1e9))
    emit({
        "command": "classify",
        "dataset": name,
        "path": path,
        "shape": list(data.shape),
        "target_col": target_col,
        "models": results,
        "best_model": best[0],
        "best_accuracy": best[1].get("accuracy_mean"),
        "elapsed_ms": round(elapsed_ms, 2),
    })


def cmd_create_engine(name: str, task: str, topology: str) -> None:
    """Generate a new engine manifest + compute script."""
    import textwrap
    engine_id = name.lower().replace(" ", "-").replace("_", "-")
    engines_dir = os.path.join(PROJECT_ROOT, "engines")
    os.makedirs(engines_dir, exist_ok=True)

    toml_path = os.path.join(engines_dir, f"{engine_id}.toml")
    py_path = os.path.join(engines_dir, f"{engine_id}_run.py")

    toml = textwrap.dedent(f"""\
        [engine]
        name = "{name}"
        version = "0.1.0"
        category = "Custom"
        description = "Custom {task} engine using {topology} topology"

        [engine.entry]
        type = "python"
        command = "python engines/{engine_id}_run.py"

        [engine.inputs]
        data = {{ type = "csv", desc = "Input dataset" }}
        target = {{ type = "int", default = -1, desc = "Target column index" }}

        [engine.outputs]
        metrics = {{ type = "json", desc = "Model performance metrics" }}
        embedding = {{ type = "tensor", desc = "Topology embedding" }}
    """)

    py = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import json, sys, os, numpy as np
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from governor_orchestrator import ComputeKernel, TopologyRunner
        from topological_governor_ml import TopologicalGovernor

        def main(path, target_col=-1):
            data = np.loadtxt(path, delimiter=",", ndmin=2)
            target_col = int(target_col)
            if target_col < 0:
                target_col = data.shape[1] - 1
            y = data[:, target_col]
            X = np.delete(data, target_col, axis=1)

            gov = TopologicalGovernor()
            cfg = gov.predict_topology(X)

            kernel = ComputeKernel()
            runner = TopologyRunner(name="{name}", topology_type="{topology}", kernel=kernel)
            res = runner.run(X, "{task}")

            print(json.dumps({{
                "engine": "{name}",
                "topology": "{topology}",
                "task": "{task}",
                "chosen_manifold": cfg.name,
                "score": float(res.score),
                "latency_ms": float(res.latency_ms),
            }}))

        if __name__ == "__main__":
            path = sys.argv[1] if len(sys.argv) > 1 else "data/blob.csv"
            target = sys.argv[2].split("=")[1] if len(sys.argv) > 2 and "=" in sys.argv[2] else -1
            main(path, target)
    """)

    with open(toml_path, "w") as f:
        f.write(toml)
    with open(py_path, "w") as f:
        f.write(py)

    emit({
        "command": "create_engine",
        "engine_id": engine_id,
        "name": name,
        "task": task,
        "topology": topology,
        "manifest": toml_path,
        "script": py_path,
    })


def cmd_formula(path: str, source: str) -> None:
    """Execute a formula on a dataset."""
    from formula_engine import run_formula
    name = os.path.basename(path)
    t0 = time.perf_counter()
    try:
        result = run_formula(source, path)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        emit({
            "command": "formula",
            "dataset": name,
            "path": path,
            "result": result,
            "elapsed_ms": round(elapsed_ms, 2),
        })
    except Exception as e:
        emit({"error": f"Formula execution failed: {str(e)}", "dataset": name, "path": path})


def cmd_formula_build(name: str, source: str) -> None:
    """Build a standalone engine from formula source."""
    from formula_engine import generate_engine
    from pathlib import Path
    engines_dir = Path(PROJECT_ROOT) / "engines"
    engines_dir.mkdir(exist_ok=True)
    info = generate_engine(source, name, engines_dir)
    emit({
        "command": "formula_build",
        "engine_id": info["engine_id"],
        "name": name,
        "manifest": info["manifest"],
        "script": info["script"],
    })


def cmd_rank(path: str) -> None:
    """Rank all manifolds by probability."""
    data = load_csv(path)
    name = os.path.basename(path)

    governor = TopologicalGovernor()
    t0 = time.perf_counter()
    ranked = governor.rank_manifolds(data)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ranking = [{"topology": topo, "probability": float(prob)} for topo, prob in ranked]

    emit({
        "command": "rank",
        "dataset": name,
        "path": path,
        "shape": list(data.shape),
        "ranking": ranking,
        "elapsed_ms": round(elapsed_ms, 2),
    })


def cmd_engine(engine_id: str, path: str, params: dict) -> None:
    """Load engine manifest and dispatch to external engine command."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', engine_id):
        emit({"error": f"Invalid engine_id: {engine_id}"})
        sys.exit(1)
    manifest_path = os.path.join(PROJECT_ROOT, "engines", f"{engine_id}.toml")
    manifest_path = os.path.normpath(manifest_path)
    engines_dir = os.path.normpath(os.path.join(PROJECT_ROOT, "engines"))
    if not os.path.commonpath([manifest_path, engines_dir]) == engines_dir:
        emit({"error": f"Engine manifest outside engines dir: {engine_id}"})
        sys.exit(1)
    if not os.path.exists(manifest_path):
        emit({"error": f"Engine manifest not found: {manifest_path}", "engine_id": engine_id})
        sys.exit(1)

    try:
        with open(manifest_path, "rb") as f:
            manifest = tomllib.load(f)
    except Exception as e:
        emit({"error": f"Failed to parse manifest: {str(e)}", "engine_id": engine_id})
        sys.exit(1)

    entry = manifest.get("engine", {}).get("entry", {})
    command = entry.get("command")
    if not command:
        emit({"error": f"No entry.command in manifest: {manifest_path}", "engine_id": engine_id})
        sys.exit(1)

    working_dir = entry.get("working_dir", PROJECT_ROOT)
    working_dir = os.path.expanduser(working_dir)
    if not os.path.isabs(working_dir):
        working_dir = os.path.join(PROJECT_ROOT, working_dir)
    if not os.path.isdir(working_dir):
        working_dir = PROJECT_ROOT

    # Build command list
    cmd_parts = shlex.split(command)
    env = os.environ.copy()
    env["GOVERNOR_DATASET"] = path
    env["GOVERNOR_ENGINE_ID"] = engine_id

    # Append dataset path and any params as arguments
    cmd_parts.append(path)
    for k, v in params.items():
        cmd_parts.append(f"--{k}={v}")

    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            cwd=working_dir,
            env=env,
            timeout=60,
        )
    except Exception as e:
        emit({"error": f"Failed to spawn engine: {str(e)}", "engine_id": engine_id, "command": command})
        sys.exit(1)

    if result.returncode != 0:
        emit({
            "error": f"Engine command failed (exit {result.returncode})",
            "engine_id": engine_id,
            "command": command,
            "stderr": result.stderr.strip()[:500],
        })
        sys.exit(1)

    stdout = result.stdout.strip()
    if not stdout:
        emit({
            "error": "Engine produced no stdout",
            "engine_id": engine_id,
            "command": command,
        })
        sys.exit(1)

    # Attempt to parse stdout as JSON; fall back to raw wrapper
    try:
        output = json.loads(stdout)
    except json.JSONDecodeError:
        output = {"raw_stdout": stdout}

    emit({
        "command": "engine",
        "engine_id": engine_id,
        "dataset": os.path.basename(path),
        "path": path,
        "output": output,
    })


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        emit({
            "help": "GOVERNOR CLI",
            "usage": "python cli/governor_cli.py <command> <path> [options]",
            "commands": {
                "vitals": "Extract topological vitals",
                "preview": "Vitals + real 3D PCA projection",
                "analyze": "Full analysis + topology prediction",
                "battle": "Run battle royale across topologies",
                "rank": "Rank manifolds by probability",
                "engine": "Dispatch to external engine via manifest",
            },
            "options": ["--topology=<name>", "--key=value"],
        })
        sys.exit(0)

    if len(sys.argv) < 3:
        emit({
            "error": "Usage: governor_cli.py <command> <path> [options]",
            "commands": ["analyze", "battle", "vitals", "preview", "rank", "regress", "classify", "create_engine", "formula", "formula_build", "engine"],
        })
        sys.exit(1)

    command = sys.argv[1].lower()

    # Commands that do NOT require a file path
    if command == "create_engine":
        if len(sys.argv) < 3:
            emit({"error": "Usage: governor_cli.py create_engine <name> --task=<task> --topology=<topology>"})
            sys.exit(1)
        engine_name = sys.argv[2]
        parsed_params = parse_params(sys.argv[3:])
        task = parsed_params.pop("task", "cluster")
        topology = parsed_params.pop("topology", "euclidean")
        try:
            cmd_create_engine(engine_name, task, topology)
        except Exception as e:
            emit({"error": str(e)})
            sys.exit(1)
        sys.exit(0)

    if command == "formula_build":
        if len(sys.argv) < 4:
            emit({"error": "Usage: governor_cli.py formula_build <name> --source=<formula_file>"})
            sys.exit(1)
        engine_name = sys.argv[2]
        parsed_params = parse_params(sys.argv[3:])
        source_path = parsed_params.pop("source", None)
        if not source_path:
            emit({"error": "Missing --source=<formula_file>"})
            sys.exit(1)
        if not os.path.isabs(source_path):
            source_path = os.path.join(PROJECT_ROOT, source_path)
        try:
            source = open(source_path, "r", encoding="utf-8").read()
            cmd_formula_build(engine_name, source)
        except Exception as e:
            emit({"error": str(e)})
            sys.exit(1)
        sys.exit(0)

    if command == "engine":
        if len(sys.argv) < 4:
            emit({"error": "Usage: governor_cli.py engine <engine_id> <path> [options]"})
            sys.exit(1)
        engine_id = sys.argv[2]
        path = sys.argv[3]
        parsed_params = parse_params(sys.argv[4:])
        force_topology = None
    else:
        path = sys.argv[2]
        parsed_params = parse_params(sys.argv[3:])
        force_topology = parsed_params.pop("topology", None)

    # Resolve relative paths from project root
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)

    if not os.path.exists(path):
        emit({"error": f"File not found: {path}"})
        sys.exit(1)

    try:
        if command == "analyze":
            cmd_analyze(path, force_topology)
        elif command == "battle":
            cmd_battle(path)
        elif command == "vitals":
            cmd_vitals(path)
        elif command == "preview":
            cmd_preview(path)
        elif command == "rank":
            cmd_rank(path)
        elif command == "regress":
            target_col = int(parsed_params.pop("target", -1))
            cmd_regress(path, target_col)
        elif command == "classify":
            target_col = int(parsed_params.pop("target", -1))
            cmd_classify(path, target_col)
        elif command == "formula":
            source_path = parsed_params.pop("source", None)
            if not source_path:
                emit({"error": "Missing --source=<formula_file>"})
                sys.exit(1)
            if not os.path.isabs(source_path):
                source_path = os.path.join(PROJECT_ROOT, source_path)
            source = open(source_path, "r", encoding="utf-8").read()
            cmd_formula(path, source)
        elif command == "engine":
            cmd_engine(engine_id, path, parsed_params)
        else:
            emit({"error": f"Unknown command: {command}", "commands": ["analyze", "battle", "vitals", "preview", "rank", "regress", "classify", "create_engine", "formula", "formula_build", "engine"]})
            sys.exit(1)
    except Exception as e:
        emit({"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
