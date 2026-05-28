#!/usr/bin/env python3
"""Faraday engine wrapper — runs real faraday code on CSV data."""
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "repos", "faraday"))

try:
    from faraday.barcode import field_to_pointcloud, topological_fingerprint
except ImportError as e:
    print(json.dumps({"engine": "Faraday", "status": "error", "error": str(e)}))
    sys.exit(1)

def main(path):
    data = np.loadtxt(path, delimiter=",", ndmin=2)

    try:
        if data.ndim == 2 and data.shape[1] >= 2:
            grid_size = int(np.sqrt(len(data))) or len(data)
            field = np.zeros((grid_size, grid_size), dtype=complex)
            for i in range(min(len(data), grid_size * grid_size)):
                x, y = i % grid_size, i // grid_size
                field[y, x] = complex(data[i, 0], data[i, 1] if data.shape[1] > 1 else 0)

            pc = field_to_pointcloud(field, threshold=0.1)
            fp = topological_fingerprint(field, threshold=0.1)

            result = {
                "engine": "Faraday",
                "dataset": os.path.basename(path),
                "shape": list(data.shape),
                "point_cloud_shape": list(pc.shape) if hasattr(pc, 'shape') else None,
                "fingerprint": {
                    "betti_0": int(fp.get("betti_0", 0)) if isinstance(fp, dict) else None,
                    "betti_1": int(fp.get("betti_1", 0)) if isinstance(fp, dict) else None,
                    "h0_bars": int(fp.get("h0_bars", 0)) if isinstance(fp, dict) else None,
                    "h1_bars": int(fp.get("h1_bars", 0)) if isinstance(fp, dict) else None,
                    "field_max": float(fp.get("field_max", 0)) if isinstance(fp, dict) else None,
                    "confinement_ratio": float(fp.get("confinement_ratio", 0)) if isinstance(fp, dict) else None,
                },
                "status": "ok",
            }
        else:
            result = {"engine": "Faraday", "dataset": os.path.basename(path), "shape": list(data.shape), "status": "unsupported_shape"}
    except Exception as e:
        result = {"engine": "Faraday", "dataset": os.path.basename(path), "shape": list(data.shape), "status": "error", "error": str(e)}

    print(json.dumps(result))

if __name__ == "__main__":
    main(sys.argv[1])
