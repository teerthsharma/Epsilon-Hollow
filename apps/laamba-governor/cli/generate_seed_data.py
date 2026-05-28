#!/usr/bin/env python3
"""Generate seed datasets for Laamba Governor."""

import json
import os
import sys
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def generate_ring(n=800, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(t) + rng.normal(0, noise, n)
    y = np.sin(t) + rng.normal(0, noise, n)
    return np.column_stack([x, y])


def generate_tree(n=800, dims=8, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, dims))
    data[:, 0] = rng.exponential(1.0, n)
    return data


def generate_blob(n=800, dims=8, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dims))


def generate_swiss_roll(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
    height = 21 * rng.uniform(0, 1, n)
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)
    return np.column_stack([x, y, z])


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    datasets = {
        "ring.csv": {
            "generator": generate_ring,
            "description": "800 points on a 2D ring (cos/sin + noise) — spherical topology should win",
            "expected_topology": "spherical",
            "shape": [800, 2],
        },
        "tree.csv": {
            "generator": generate_tree,
            "description": "800 points with exponential first dim, 8 dims — hyperbolic should win",
            "expected_topology": "hyperbolic_poincare",
            "shape": [800, 8],
        },
        "blob.csv": {
            "generator": generate_blob,
            "description": "800 points, 8 dims, standard normal — euclidean should win",
            "expected_topology": "euclidean",
            "shape": [800, 8],
        },
        "swiss_roll.csv": {
            "generator": generate_swiss_roll,
            "description": "1000 points, 3D swiss roll",
            "expected_topology": "mixed_curvature",
            "shape": [1000, 3],
        },
    }

    index = []
    for filename, meta in datasets.items():
        data = meta["generator"]()
        path = os.path.join(DATA_DIR, filename)
        np.savetxt(path, data, delimiter=",", fmt="%.8f")
        actual_shape = list(data.shape)
        entry = {
            "name": filename,
            "path": filename,
            "description": meta["description"],
            "expected_topology": meta["expected_topology"],
            "shape": actual_shape,
            "rows": actual_shape[0],
            "cols": actual_shape[1],
            "type": "point_cloud",
            "format": "csv",
        }
        index.append(entry)
        print(f"Generated {filename}: shape={actual_shape}")

    index_path = os.path.join(DATA_DIR, "index.json")
    with open(index_path, "w") as f:
        json.dump({"datasets": index}, f, indent=2)
    print(f"Index written to {index_path}")


if __name__ == "__main__":
    main()
