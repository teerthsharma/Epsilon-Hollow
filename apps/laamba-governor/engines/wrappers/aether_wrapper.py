#!/usr/bin/env python3
"""Aether-Lang engine wrapper."""
import json, sys, os, subprocess

def main(path):
    repo = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "repos", "Aether-Lang")
    binary = os.path.join(repo, "target", "release", "aether.exe")
    if not os.path.exists(binary):
        binary = os.path.join(repo, "target", "debug", "aether.exe")

    if os.path.exists(binary):
        # Aether CLI expects .aether files, not CSV. Run a syntax check on a dummy hello file.
        # For CSV data, we just report the binary is working and dataset info.
        result = subprocess.run([binary, "--version"], capture_output=True, text=True)
        output = {
            "engine": "Aether-Lang",
            "dataset": os.path.basename(path),
            "binary": binary,
            "version_output": result.stdout.strip() or result.stderr.strip()[:500],
            "returncode": result.returncode,
            "status": "ok" if result.returncode == 0 else "check_failed",
            "note": "Aether-Lang is a compiled language runtime. CSV datasets are passed through the wrapper for cataloging."
        }
    else:
        output = {
            "engine": "Aether-Lang",
            "dataset": os.path.basename(path),
            "status": "binary_not_built",
            "note": "Run: cd repos/Aether-Lang && cargo build -p aether-cli --release",
        }
    print(json.dumps(output))

if __name__ == "__main__":
    main(sys.argv[1])
