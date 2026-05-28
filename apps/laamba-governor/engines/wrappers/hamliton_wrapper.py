#!/usr/bin/env python3
"""Hamilton engine wrapper — runs real hamliton code on CSV data."""
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "repos", "hamliton", "src"))

try:
    from hamliton.hilbert import hilbert_norm, tensor_product_signatures, n_body_overlap_matrix, multi_state_normalize
    from hamliton.gauge import gauge_invariant_trace, su2_casimir_invariant
except ImportError as e:
    print(json.dumps({"engine": "Hamilton", "status": "error", "error": str(e)}))
    sys.exit(1)

def main(path):
    data = np.loadtxt(path, delimiter=",", ndmin=2)

    try:
        if data.ndim == 2 and data.shape[1] >= 2:
            # Normalize each row to a fixed latent dimension (pad or truncate to 16)
            latent_dim = 8
            signatures = []
            for i in range(min(3, len(data))):
                row = data[i].flatten()
                if len(row) < latent_dim:
                    padded = np.zeros(latent_dim, dtype=np.float64)
                    padded[:len(row)] = row
                else:
                    padded = row[:latent_dim]
                signatures.append(padded)

            # Compute Hilbert norms for each signature
            norms = [float(hilbert_norm(sig)) for sig in signatures]

            # Compute tensor product of signatures (N-body coupling)
            coupling = tensor_product_signatures(signatures)

            # Compute gauge invariant trace on the coupling tensor
            trace = float(gauge_invariant_trace(coupling, latent_dim=latent_dim))

            # Compute SU(2) Casimir invariant
            casimir = float(su2_casimir_invariant(coupling, latent_dim=latent_dim))

            # Compute overlap matrix
            states = np.vstack(signatures)
            overlaps = n_body_overlap_matrix(states).tolist()

            result = {
                "engine": "Hamilton",
                "dataset": os.path.basename(path),
                "shape": list(data.shape),
                "n_signatures": len(signatures),
                "latent_dim": latent_dim,
                "hilbert_norms": norms,
                "tensor_shape": list(coupling.shape) if hasattr(coupling, 'shape') else None,
                "gauge_trace": trace,
                "su2_casimir": casimir,
                "overlap_matrix": overlaps,
                "status": "ok",
            }
        else:
            result = {"engine": "Hamilton", "dataset": os.path.basename(path), "shape": list(data.shape), "status": "unsupported_shape"}
    except Exception as e:
        import traceback
        result = {"engine": "Hamilton", "dataset": os.path.basename(path), "shape": list(data.shape), "status": "error", "error": str(e), "traceback": traceback.format_exc()}

    print(json.dumps(result))

if __name__ == "__main__":
    main(sys.argv[1])
