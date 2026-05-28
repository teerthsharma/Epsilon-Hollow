#!/usr/bin/env python3
"""Built-in engine wrapper — runs topology analysis using app's own code."""
import json, sys, os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from topological_governor_ml import TopologicalGovernor, VitalsExtractor
from governor_orchestrator import ComputeKernel, TopologyRunner

def main(path, engine_id):
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    name = os.path.basename(path)

    gov = TopologicalGovernor()
    cfg = gov.predict_topology(data)

    kernel = ComputeKernel()

    topo_map = {
        "omnitopos": "spherical",
        "topoflow": "hyperbolic_poincare",
        "lambda-topo": "grassmannian",
        "topo-asm": "euclidean",
        "epsilon-hollow": "hyperbolic_poincare",
        "hollow-asm": "spherical",
        "pgtable-asm": "euclidean",
        "vec-simd": "euclidean",
        "topobridge-q": "spherical",
        "neuralwhisper": "product",
    }
    topo = topo_map.get(engine_id, "euclidean")

    runner = TopologyRunner(name=engine_id, topology_type=topo, kernel=kernel)
    res = runner.run(data, "cluster")

    extractor = VitalsExtractor()
    vitals = extractor(data)

    result = {
        "engine": engine_id,
        "dataset": name,
        "shape": list(data.shape),
        "topology": topo,
        "chosen_manifold": cfg.name,
        "score": float(res.score),
        "latency_ms": float(res.latency_ms),
        "vitals": {
            "intrinsic_dim": float(vitals[3]),
            "spectral_gap": float(vitals[7]),
            "curvature_proxy": float(vitals[9]),
        },
        "status": "ok",
        "note": "Running built-in topology analysis (external repo not cloned).",
    }
    print(json.dumps(result))

if __name__ == "__main__":
    path = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
    engine_id = sys.argv[1] if len(sys.argv) > 2 else "builtin"
    main(path, engine_id)
