#!/usr/bin/env python3
"""Phi-Mem engine wrapper — runs real phi-mem code on CSV data."""
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "repos", "phi-mem", "src"))

try:
    from phi_mem.signature import text_to_barcode
    from phi_mem.store import PhiStore
except ImportError as e:
    print(json.dumps({"engine": "Phi-Mem", "status": "error", "error": str(e)}))
    sys.exit(1)

def main(path):
    data = np.loadtxt(path, delimiter=",", ndmin=2)

    try:
        store = PhiStore()
        signatures = []
        for i in range(min(20, len(data))):
            text = f"row_{i}_" + "_".join(f"{x:.4f}" for x in data[i])
            sig = text_to_barcode(text)
            signatures.append(sig)
            store.put(text, metadata={"index": i, "values": data[i].tolist()})

        result = {
            "engine": "Phi-Mem",
            "dataset": os.path.basename(path),
            "shape": list(data.shape),
            "signatures": signatures,
            "store_size": len(store._entries),
            "status": "ok",
        }
    except Exception as e:
        import traceback
        result = {"engine": "Phi-Mem", "dataset": os.path.basename(path), "shape": list(data.shape), "status": "error", "error": str(e), "traceback": traceback.format_exc()}

    print(json.dumps(result))

if __name__ == "__main__":
    main(sys.argv[1])
