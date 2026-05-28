#!/usr/bin/env python3
"""
FORMULA ENGINE — Math DSL for topology algorithms.

User writes formulas like:
    emb = pca(data, dim=3)
    labels = kmeans(emb, k=5)
    score = silhouette(emb, labels)
    L = laplacian(emb, k=15)
    eig = eigvals(L)
    dists = pairwise_dist(emb)
    K = curvature_proxy(emb)

FormulaEngine compiles to real numpy/sklearn compute.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# 0.  TOPOLOGY PRIMITIVES  —  real math, no fake
# ---------------------------------------------------------------------------

class TopoOps:
    """Library of topological operations available to formulas."""

    @staticmethod
    def pca(x: NDArray, dim: int = 3) -> NDArray:
        """PCA projection to `dim` dimensions."""
        d = min(dim, x.shape[1])
        if d <= 0:
            return x
        cov = np.cov(x.T)
        vals, vecs = np.linalg.eigh(cov)
        return x @ vecs[:, -d:]

    @staticmethod
    def kmeans(x: NDArray, k: int = 5) -> NDArray:
        """K-means labels."""
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=min(k, len(x)), n_init=3, random_state=42)
        return km.fit_predict(x)

    @staticmethod
    def dbscan(x: NDArray, eps: float = 0.5, min_samples: int = 5) -> NDArray:
        """DBSCAN labels."""
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)

    @staticmethod
    def silhouette(x: NDArray, labels: NDArray) -> float:
        """Silhouette score. Higher = better clusters."""
        from sklearn.metrics import silhouette_score
        if len(set(labels)) < 2:
            return 0.0
        return float(silhouette_score(x, labels))

    @staticmethod
    def pairwise_dist(x: NDArray) -> NDArray:
        """Pairwise squared Euclidean distances."""
        return np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)

    @staticmethod
    def normalize(x: NDArray) -> NDArray:
        """L2 normalize rows."""
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

    @staticmethod
    def sphere_proj(x: NDArray) -> NDArray:
        """Project rows to unit sphere S^n."""
        return TopoOps.normalize(x)

    @staticmethod
    def poincare_proj(x: NDArray) -> NDArray:
        """Project to Poincare disk via tanh scaling."""
        r = np.linalg.norm(x, axis=1, keepdims=True)
        return np.tanh(r) * x / (r + 1e-9)

    @staticmethod
    def hyperboloid_proj(x: NDArray) -> NDArray:
        """Map to hyperboloid: (sqrt(1+||x||^2), x)."""
        norm_sq = np.sum(x ** 2, axis=1, keepdims=True)
        return np.concatenate([np.sqrt(1 + norm_sq), x], axis=1)

    @staticmethod
    def grassmann_proj(x: NDArray, rank: int = 2) -> NDArray:
        """Project to Grassmannian via top `rank` singular vectors."""
        u, s, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
        r = min(rank, len(s))
        return x @ vt[:r].T

    @staticmethod
    def laplacian(x: NDArray, k: int = 15) -> NDArray:
        """Unnormalized graph Laplacian from k-NN graph."""
        n = min(len(x), 2048)
        idx = np.random.choice(len(x), n, replace=False) if len(x) > n else np.arange(len(x))
        S = x[idx]
        D2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(axis=2)
        knn = np.argsort(D2, axis=1)[:, 1 : k + 1]
        W = np.zeros((n, n))
        for i, neigh in enumerate(knn):
            W[i, neigh] = 1.0
            W[neigh, i] = 1.0
        deg = W.sum(axis=1)
        return np.diag(deg) - W

    @staticmethod
    def eigvals(m: NDArray) -> NDArray:
        """Eigenvalues of symmetric matrix."""
        return np.linalg.eigvalsh(m)

    @staticmethod
    def betti(x: NDArray, eps: float = 0.5) -> Dict[str, int]:
        """Betti numbers from Vietoris-Rips complex at threshold eps."""
        n = min(len(x), 512)
        idx = np.random.choice(len(x), n, replace=False) if len(x) > n else np.arange(len(x))
        S = x[idx]
        D = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(axis=2))
        A = (D < eps).astype(int)
        np.fill_diagonal(A, 0)
        # Union-Find for B0
        parent = np.arange(n)
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j]:
                    union(i, j)
        b0 = len(set(find(i) for i in range(n)))
        # Crude B1: count independent cycles (E - V + C)
        edges = A.sum() // 2
        b1 = max(0, edges - n + b0)
        return {"b0": int(b0), "b1": int(b1), "betti_0": int(b0), "betti_1": int(b1)}

    @staticmethod
    def ripser(x: NDArray, maxdim: int = 1) -> Dict[str, Any]:
        """Persistent homology via ripser if available, else fallback to crude betti."""
        try:
            import ripser as rp
            result = rp.ripser(x, maxdim=maxdim)
            dgms = result["dgms"]
            out = {}
            for i, dgm in enumerate(dgms):
                finite = dgm[dgm[:, 1] < np.inf]
                out[f"betti_{i}"] = len(finite)
                out[f"dgm_{i}"] = finite.tolist()
            return out
        except Exception:
            return TopoOps.betti(x, eps=0.5)

    @staticmethod
    def curvature_proxy(x: NDArray, k: int = 20) -> float:
        """Local PCA eigenvalue ratio variance."""
        n = len(x)
        ratios = []
        for i in np.random.choice(n, min(n, 512), replace=False):
            dists = np.sum((x - x[i]) ** 2, axis=1)
            neigh = np.argsort(dists)[1 : k + 1]
            loc = x[neigh] - x[i]
            cov = loc.T @ loc / k
            vals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            ratios.append(vals[1] / (vals[0] + 1e-9))
        return float(np.std(ratios))

    @staticmethod
    def geodesic_sphere(a: NDArray, b: NDArray) -> float:
        """Geodesic distance on unit sphere: arccos(a·b)."""
        a = TopoOps.normalize(a.reshape(1, -1)).flatten()
        b = TopoOps.normalize(b.reshape(1, -1)).flatten()
        dot = np.clip(a @ b, -1.0, 1.0)
        return float(np.arccos(abs(dot)))

    @staticmethod
    def geodesic_poincare(a: NDArray, b: NDArray) -> float:
        """Poincare disk distance: arccosh(1 + 2||a-b||^2 / ((1-||a||^2)(1-||b||^2)))."""
        a = a.flatten()
        b = b.flatten()
        num = 2 * np.sum((a - b) ** 2)
        den = (1 - np.sum(a ** 2)) * (1 - np.sum(b ** 2)) + 1e-9
        return float(np.arccosh(1 + num / den))

    @staticmethod
    def exp_map_sphere(x: NDArray, v: NDArray) -> NDArray:
        """Exponential map on sphere: x*cos(||v||) + v/||v||*sin(||v||)."""
        x = x.flatten()
        v = v.flatten()
        nv = np.linalg.norm(v) + 1e-9
        return (x * np.cos(nv) + (v / nv) * np.sin(nv)).reshape(1, -1)

    @staticmethod
    def log_map_sphere(x: NDArray, y: NDArray) -> NDArray:
        """Log map on sphere: (y - x*cos(d)) * d/sin(d)."""
        x = x.flatten()
        y = y.flatten()
        dot = np.clip(x @ y, -1.0, 1.0)
        d = np.arccos(abs(dot)) + 1e-9
        return ((y - x * np.cos(d)) * d / np.sin(d)).reshape(1, -1)

    @staticmethod
    def concat(a: NDArray, b: NDArray, axis: int = 1) -> NDArray:
        return np.concatenate([a, b], axis=axis)

    @staticmethod
    def slice_(x: NDArray, start: int = 0, end: int | None = None) -> NDArray:
        return x[:, start:end]

    @staticmethod
    def mean(x: NDArray, axis: int | None = None) -> NDArray:
        return np.mean(x, axis=axis, keepdims=True) if axis is not None else np.mean(x)

    @staticmethod
    def std(x: NDArray, axis: int | None = None) -> NDArray:
        return np.std(x, axis=axis, keepdims=True) if axis is not None else np.std(x)

    @staticmethod
    def var(x: NDArray, axis: int | None = None) -> NDArray:
        return np.var(x, axis=axis, keepdims=True) if axis is not None else np.var(x)

    @staticmethod
    def dot(a: NDArray, b: NDArray) -> NDArray:
        return a @ b.T if a.ndim == 2 and b.ndim == 2 else a @ b

    @staticmethod
    def transpose(x: NDArray) -> NDArray:
        return x.T

    @staticmethod
    def inv(x: NDArray) -> NDArray:
        return np.linalg.inv(x)

    @staticmethod
    def svd(x: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        return u, s, vt

    @staticmethod
    def qr(x: NDArray) -> Tuple[NDArray, NDArray]:
        return np.linalg.qr(x)


# ---------------------------------------------------------------------------
# 1.  FORMULA PARSER
# ---------------------------------------------------------------------------

@dataclass
class FormulaLine:
    var: str
    func: str
    args: List[str]
    kwargs: Dict[str, Any]


class FormulaParser:
    """Parse simple assignment lines:
        emb = pca(data, dim=3)
        score = silhouette(emb, labels)
    """

    BUILTINS: Dict[str, Callable] = {
        name: getattr(TopoOps, name)
        for name in dir(TopoOps)
        if not name.startswith("_") and callable(getattr(TopoOps, name))
    }

    def parse(self, source: str) -> List[FormulaLine]:
        lines: List[FormulaLine] = []
        for raw in source.strip().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise SyntaxError(f"Invalid line (no assignment): {line}")
            lhs, rhs = line.split("=", 1)
            var = lhs.strip()
            if not var.isidentifier():
                raise SyntaxError(f"Invalid variable name: {var}")
            func, args, kwargs = self._parse_call(rhs.strip())
            lines.append(FormulaLine(var, func, args, kwargs))
        return lines

    def _parse_call(self, s: str) -> Tuple[str, List[str], Dict[str, Any]]:
        # s looks like: pca(data, dim=3)
        if "(" not in s or not s.endswith(")"):
            raise SyntaxError(f"Expected function call, got: {s}")
        func_name = s[: s.index("(")].strip()
        inner = s[s.index("(") + 1 : -1].strip()
        args: List[str] = []
        kwargs: Dict[str, Any] = {}
        if inner:
            for part in self._split_args(inner):
                if "=" in part and not part.startswith("="):
                    k, v = part.split("=", 1)
                    kwargs[k.strip()] = self._literal(v.strip())
                else:
                    args.append(part.strip())
        return func_name, args, kwargs

    def _split_args(self, s: str) -> List[str]:
        parts = []
        depth = 0
        current = ""
        for ch in s:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current)
        return parts

    def _literal(self, s: str) -> Any:
        s = s.strip()
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s

    def validate(self, lines: List[FormulaLine]) -> None:
        for line in lines:
            if line.func not in self.BUILTINS:
                raise NameError(f"Unknown function: {line.func}")
            sig = inspect.signature(self.BUILTINS[line.func])
            # Check for too many positional args
            params = list(sig.parameters.values())
            n_pos = sum(1 for p in params if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
            if len(line.args) > len(params):
                raise TypeError(f"{line.func} takes at most {len(params)} args, got {len(line.args)}")


# ---------------------------------------------------------------------------
# 2.  FORMULA COMPILER  —  generate Python script
# ---------------------------------------------------------------------------

class FormulaCompiler:
    """Compile parsed formulas to a standalone Python script."""

    def compile(self, lines: List[FormulaLine], engine_name: str = "FormulaEngine") -> str:
        body = []
        for line in lines:
            arg_strs = line.args
            kwarg_strs = [f"{k}={repr(v)}" for k, v in line.kwargs.items()]
            all_args = ", ".join(arg_strs + kwarg_strs)
            body.append(f"    {line.var} = TopoOps.{line.func}({all_args})")

        code = textwrap.dedent(f'''\
            #!/usr/bin/env python3
            # Auto-generated formula engine: {engine_name}
            import json, sys, os, numpy as np
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from formula_engine import TopoOps

            def main(path):
                data = np.loadtxt(path, delimiter=",", ndmin=2)
            {chr(10).join(body)}
                # Output all computed variables as JSON
                result = {{}}
                for k, v in locals().items():
                    if k in ("data", "path", "np", "json", "sys", "os", "TopoOps"):
                        continue
                    if isinstance(v, np.ndarray):
                        result[k] = v.tolist()
                    elif isinstance(v, (int, float, str, list, dict)):
                        result[k] = v
                    else:
                        result[k] = str(v)
                print(json.dumps(result, default=str))

            if __name__ == "__main__":
                main(sys.argv[1])
        ''')
        return code

    def execute(self, lines: List[FormulaLine], data: NDArray) -> Dict[str, Any]:
        """Execute formulas in-memory on given data."""
        env = {"data": data, "np": np, "TopoOps": TopoOps}
        for line in lines:
            func = getattr(TopoOps, line.func)
            # Resolve args from env or as literals
            resolved_args = []
            for a in line.args:
                if a in env:
                    resolved_args.append(env[a])
                else:
                    try:
                        resolved_args.append(ast.literal_eval(a))
                    except (ValueError, SyntaxError):
                        raise NameError(f"Undefined variable: {a}")
            resolved_kwargs = {k: (env[v] if isinstance(v, str) and v in env else v) for k, v in line.kwargs.items()}
            result = func(*resolved_args, **resolved_kwargs)
            env[line.var] = result
        # Collect outputs
        out = {}
        for k, v in env.items():
            if k in ("data", "np", "TopoOps"):
                continue
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (int, float, str, list, dict)):
                out[k] = v
            else:
                out[k] = str(v)
        return out


# ---------------------------------------------------------------------------
# 3.  FORMULA ENGINE CLI WRAPPER
# ---------------------------------------------------------------------------

def run_formula(source: str, data_path: str) -> Dict[str, Any]:
    """Parse, validate, and execute a formula on a CSV."""
    data = np.loadtxt(data_path, delimiter=",", ndmin=2)
    parser = FormulaParser()
    lines = parser.parse(source)
    parser.validate(lines)
    compiler = FormulaCompiler()
    return compiler.execute(lines, data)


def generate_engine(source: str, name: str, out_dir: Path) -> Dict[str, str]:
    """Generate a standalone engine from formula source."""
    parser = FormulaParser()
    lines = parser.parse(source)
    parser.validate(lines)
    compiler = FormulaCompiler()
    code = compiler.compile(lines, name)

    engine_id = name.lower().replace(" ", "-").replace("_", "-")
    toml_path = out_dir / f"{engine_id}.toml"
    py_path = out_dir / f"{engine_id}_formula.py"

    toml = textwrap.dedent(f"""\
        [engine]
        name = "{name}"
        version = "0.1.0"
        category = "Formula"
        description = "Formula engine: {name}"

        [engine.entry]
        type = "python"
        command = "python engines/{engine_id}_formula.py"

        [engine.inputs]
        data = {{ type = "csv", desc = "Input dataset" }}

        [engine.outputs]
        result = {{ type = "json", desc = "Formula outputs" }}
    """)

    toml_path.write_text(toml, encoding="utf-8")
    py_path.write_text(code, encoding="utf-8")

    return {
        "engine_id": engine_id,
        "manifest": str(toml_path),
        "script": str(py_path),
    }


# ---------------------------------------------------------------------------
# 4.  DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = """
    # Project data to 3D sphere
    emb = sphere_proj(data)
    # Cluster on sphere
    labels = kmeans(emb, k=5)
    # Measure cluster quality
    score = silhouette(emb, labels)
    # Topology of embedding
    L = laplacian(emb, k=15)
    eig = eigvals(L)
    # Persistent homology
    hom = ripser(emb, maxdim=1)
    # Curvature estimate
    K = curvature_proxy(emb)
    """
    print("=" * 60)
    print("FORMULA ENGINE DEMO")
    print("=" * 60)
    parser = FormulaParser()
    lines = parser.parse(sample)
    parser.validate(lines)
    print(f"Parsed {len(lines)} lines:")
    for line in lines:
        print(f"  {line.var} = {line.func}({', '.join(line.args)}{', ' if line.args and line.kwargs else ''}{', '.join(f'{k}={v}' for k,v in line.kwargs.items())})")

    # Run on ring.csv
    import os
    path = os.path.join(os.path.dirname(__file__), "data", "ring.csv")
    result = run_formula(sample, path)
    print("\nResults:")
    for k, v in result.items():
        preview = str(v)[:80] + "..." if len(str(v)) > 80 else str(v)
        print(f"  {k}: {preview}")

    # Generate engine
    info = generate_engine(sample, "SphereCluster", Path("engines"))
    print(f"\nEngine generated: {info['manifest']}")
