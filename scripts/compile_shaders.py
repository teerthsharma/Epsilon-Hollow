#!/usr/bin/env python3
import sys
import os

def generate_stubs(out_dir):
    stubs = [
        "voronoi_assign.bin",
        "jl_project.bin",
        "spectral_step.bin",
        "s2_distance.bin"
    ]
    for stub in stubs:
        path = os.path.join(out_dir, stub)
        print(f"Generating stub: {path}")
        with open(path, "wb") as f:
            f.write(b"\0")

if __name__ == "__main__":
    if "--stubs" in sys.argv:
        out_dir = sys.argv[3]
        generate_stubs(out_dir)
        sys.exit(0)
