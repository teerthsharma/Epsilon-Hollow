#!/usr/bin/env python3
"""
GOVERNOR STUDIO — Backend Server
Serves the FL-Studio-for-Topologies UI and runs the compute graph.

Run:
    cd governor-studio
    python main.py
    # open http://localhost:8080

Dependencies:
    pip install fastapi uvicorn numpy
"""

from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 0.  IMPORT GOVERNOR LOGIC FROM PARENT DIR
# ---------------------------------------------------------------------------
PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))

from topological_governor_ml import TopologicalGovernor, ManifoldConfig, VitalsExtractor
from governor_orchestrator import (
    ComputeKernel,
    TopologyRunner,
    TopologyResult,
    TransferBus,
    OutcomeComparator,
    GovernorOrchestrator,
    OrchestratorConfig,
    TopologyFusion,
)

# ---------------------------------------------------------------------------
# 1.  FASTAPI APP
# ---------------------------------------------------------------------------

app = FastAPI(title="Governor Studio")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# global state (demo — in production use Redis or DB)
_SESSION: Dict[str, Any] = {
    "datasets": {},
    "last_results": [],
}


# ---------------------------------------------------------------------------
# 2.  DATA MODELS
# ---------------------------------------------------------------------------

class NodeSpec(BaseModel):
    id: int
    type: str
    x: float
    y: float
    params: Dict[str, Any]
    inputs: List[str]
    outputs: List[str]


class ConnSpec(BaseModel):
    from_: int
    fromOutput: int
    to: int
    toInput: int

    class Config:
        populate_by_name = True


class GraphRequest(BaseModel):
    nodes: List[NodeSpec]
    connections: List[ConnSpec]


# ---------------------------------------------------------------------------
# 3.  UTILS
# ---------------------------------------------------------------------------

def make_dataset(params: Dict[str, Any]) -> np.ndarray:
    """Generate synthetic data from DataSource params."""
    shape = params.get("shape", "blob")
    n = int(params.get("samples", 500))
    d = int(params.get("dims", 8))
    rng = np.random.default_rng(42)

    if shape == "ring":
        t = np.linspace(0, 2 * np.pi, n)
        return np.stack([np.cos(t), np.sin(t)], axis=1) + rng.randn(n, 2) * 0.03
    if shape == "tree":
        x = rng.randn(n, d)
        x[:, 0] = rng.exponential(1.0, n)
        return x.astype(np.float32)
    if shape == "grid":
        side = int(np.sqrt(n))
        xs, ys = np.meshgrid(np.linspace(-1, 1, side), np.linspace(-1, 1, side))
        return np.column_stack([xs.ravel()[:n], ys.ravel()[:n]])
    if shape == "swiss_roll":
        t = 1.5 * np.pi * (1 + 2 * rng.random(n))
        x = t * np.cos(t)
        z = t * np.sin(t)
        y = 21 * rng.random(n)
        return np.column_stack([x, y, z]).astype(np.float32)
    # default blob
    return rng.randn(n, d).astype(np.float32)


def topological_sort(nodes: List[NodeSpec], conns: List[ConnSpec]) -> List[int]:
    """Topologically sort node IDs by connection graph."""
    ids = [n.id for n in nodes]
    adj: Dict[int, List[int]] = {i: [] for i in ids}
    indeg = {i: 0 for i in ids}
    node_map = {n.id: n for n in nodes}

    for c in conns:
        if c.from_ in adj and c.to in indeg:
            adj[c.from_].append(c.to)
            indeg[c.to] += 1

    queue = [i for i in ids if indeg[i] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    # append any remaining (cycles or isolated)
    for i in ids:
        if i not in order:
            order.append(i)
    return order


# ---------------------------------------------------------------------------
# 4.  GRAPH EXECUTOR
# ---------------------------------------------------------------------------

class GraphExecutor:
    """Walks the node graph, executes each node, passes data along edges."""

    def __init__(self, nodes: List[NodeSpec], conns: List[ConnSpec]):
        self.nodes = {n.id: n for n in nodes}
        self.conns = conns
        self.kernel = ComputeKernel()
        self.bus = TransferBus()
        self.outputs: Dict[int, np.ndarray] = {}
        self.meta: Dict[int, Dict] = {}

    def run(self) -> Dict[str, Any]:
        order = topological_sort(list(self.nodes.values()), self.conns)
        results_log = []

        for nid in order:
            node = self.nodes[nid]
            out, meta = self._exec_node(node)
            self.outputs[nid] = out
            self.meta[nid] = meta
            self.bus.write(str(nid), "out", out)
            if meta.get("score") is not None:
                results_log.append({
                    "node": nid,
                    "type": node.type,
                    "score": meta["score"],
                    "latency_ms": meta.get("latency_ms", 0),
                })

        # if comparator present, compute winner
        winner = None
        for nid in order:
            if self.nodes[nid].type == "comparator":
                winner = self._run_comparator(nid)

        return {
            "execution_order": order,
            "node_outputs": {k: f"array{v.shape}" for k, v in self.outputs.items()},
            "results": results_log,
            "winner": winner,
        }

    def _exec_node(self, node: NodeSpec) -> tuple:
        t0 = __import__("time").perf_counter()
        t = node.type

        # gather inputs from connected upstream nodes
        inputs = []
        for c in self.conns:
            if c.to == node.id and c.toInput == 0:
                upstream = self.outputs.get(c.from_)
                if upstream is not None:
                    inputs.append(upstream)

        data = inputs[0] if inputs else None

        if t == "datasource":
            arr = make_dataset(node.params)
            return arr, {"shape": arr.shape}

        if t in ("euclidean", "spherical", "hyperbolic", "grassmannian", "product"):
            if data is None:
                data = np.random.randn(100, 8).astype(np.float32)
            runner = TopologyRunner(name=t, topology_type=t, kernel=self.kernel)
            task = node.params.get("task", "cluster")
            res = runner.run(data, task)
            return res.output, {
                "score": res.score,
                "latency_ms": res.latency_ms,
                "topology_type": res.topology_type,
            }

        if t == "fusion":
            # gather all upstream topology outputs
            mats = []
            for c in self.conns:
                if c.to == node.id:
                    up = self.outputs.get(c.from_)
                    if up is not None:
                        mats.append(up)
            if len(mats) < 2:
                return (mats[0] if mats else np.zeros((10, 8))), {"score": 0.5}
            fusion = TopologyFusion([f"n{i}" for i in range(len(mats))], embed_dim=min(m.shape[1] for m in mats))
            fused = fusion.fuse({f"n{i}": m for i, m in enumerate(mats)})
            return fused, {"score": float(np.var(fused)), "fused_from": len(mats)}

        if t == "transferbus":
            # pass-through with optional blend from previous winner
            if data is None:
                return np.zeros((10, 8)), {}
            prev = self.bus.read("winner", "out")
            if prev is not None and prev.shape == data.shape:
                blend = 0.8 * data + 0.2 * prev
                return blend, {"blended": True}
            return data, {"blended": False}

        if t == "viz":
            if data is None:
                data = np.random.randn(100, 2)
            # reduce to 2D for viz
            if data.shape[1] > 2:
                data = ComputeKernel.pca_project(data, 2)
            return data, {"viz_shape": data.shape}

        if t == "export":
            if data is None:
                data = np.zeros((10, 8))
            fmt = node.params.get("format", "npz")
            fname = f"/tmp/governor_export_{node.id}.{fmt}"
            if fmt == "npz":
                np.savez(fname, data=data)
            else:
                np.savetxt(fname, data, delimiter=",")
            return data, {"exported_to": fname}

        # comparator handled separately after all upstreams run
        return (data if data is not None else np.zeros((10, 8))), {}

    def _run_comparator(self, nid: int) -> Dict[str, Any]:
        """Find all upstream topology results and compare."""
        upstream_results = []
        for c in self.conns:
            if c.to == nid:
                up_meta = self.meta.get(c.from_)
                up_out = self.outputs.get(c.from_)
                up_node = self.nodes.get(c.from_)
                if up_meta and up_out is not None and up_node:
                    upstream_results.append(TopologyResult(
                        name=up_node.type,
                        output=up_out,
                        score=up_meta.get("score", 0.0),
                        latency_ms=up_meta.get("latency_ms", 0.0),
                        topology_type=up_node.type,
                    ))

        if len(upstream_results) < 2:
            return {"error": "Comparator needs 2+ upstream topologies"}

        comp = OutcomeComparator()
        ranked = comp.compare(upstream_results)
        winner = ranked[0]
        self.bus.write("winner", "out", winner.output)

        return {
            "winner": winner.name,
            "winner_score": winner.score,
            "winner_latency_ms": winner.latency_ms,
            "ranking": [{"name": r.name, "score": r.score, "latency_ms": r.latency_ms} for r in ranked],
        }


# ---------------------------------------------------------------------------
# 5.  API ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/run_graph")
def run_graph(req: GraphRequest):
    try:
        exe = GraphExecutor(req.nodes, req.connections)
        result = exe.run()
        _SESSION["last_results"] = result.get("results", [])
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


@app.post("/api/run_battle")
def run_battle(req: GraphRequest):
    """Dedicated battle endpoint: expects exactly N topologies feeding 1 comparator."""
    try:
        exe = GraphExecutor(req.nodes, req.connections)
        result = exe.run()
        return JSONResponse(content={
            "winner": result.get("winner"),
            "all_results": result.get("results"),
        })
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


@app.get("/api/topologies")
def list_topologies():
    return {
        "topologies": ["euclidean", "spherical", "hyperbolic", "grassmannian", "product"],
        "sources": ["datasource"],
        "flow": ["comparator", "fusion", "transferbus", "viz", "export"],
    }


# ---------------------------------------------------------------------------
# 6.  MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    host = "127.0.0.1"
    port = 8080
    url = f"http://{host}:{port}"
    print(f"\n{'='*60}")
    print(f"  GOVERNOR STUDIO")
    print(f"  Open: {url}")
    print(f"{'='*60}\n")
    webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="warning")
