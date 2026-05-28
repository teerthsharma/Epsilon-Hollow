import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStore } from "../store";
import { Code, Play, Save, Wrench, BookOpen, X, Loader2, ChevronRight } from "lucide-react";

const BUILTINS = [
  { name: "pca", sig: "pca(x, dim=3)", desc: "PCA projection" },
  { name: "kmeans", sig: "kmeans(x, k=5)", desc: "K-means clustering" },
  { name: "dbscan", sig: "dbscan(x, eps=0.5, min_samples=5)", desc: "DBSCAN clustering" },
  { name: "silhouette", sig: "silhouette(x, labels)", desc: "Silhouette score" },
  { name: "normalize", sig: "normalize(x)", desc: "L2 normalize rows" },
  { name: "sphere_proj", sig: "sphere_proj(x)", desc: "Project to unit sphere" },
  { name: "poincare_proj", sig: "poincare_proj(x)", desc: "Project to Poincare disk" },
  { name: "hyperboloid_proj", sig: "hyperboloid_proj(x)", desc: "Map to hyperboloid" },
  { name: "grassmann_proj", sig: "grassmann_proj(x, rank=2)", desc: "Grassmannian projection" },
  { name: "pairwise_dist", sig: "pairwise_dist(x)", desc: "Pairwise distances" },
  { name: "laplacian", sig: "laplacian(x, k=15)", desc: "Graph Laplacian" },
  { name: "eigvals", sig: "eigvals(m)", desc: "Eigenvalues of symmetric matrix" },
  { name: "betti", sig: "betti(x, eps=0.5)", desc: "Betti numbers from VR complex" },
  { name: "ripser", sig: "ripser(x, maxdim=1)", desc: "Persistent homology" },
  { name: "curvature_proxy", sig: "curvature_proxy(x)", desc: "Local curvature estimate" },
  { name: "geodesic_sphere", sig: "geodesic_sphere(a, b)", desc: "Sphere geodesic distance" },
  { name: "geodesic_poincare", sig: "geodesic_poincare(a, b)", desc: "Poincare geodesic distance" },
  { name: "exp_map_sphere", sig: "exp_map_sphere(x, v)", desc: "Sphere exponential map" },
  { name: "log_map_sphere", sig: "log_map_sphere(x, y)", desc: "Sphere log map" },
  { name: "concat", sig: "concat(a, b, axis=1)", desc: "Concatenate arrays" },
  { name: "slice_", sig: "slice_(x, start=0, end=None)", desc: "Column slice" },
  { name: "mean", sig: "mean(x, axis=None)", desc: "Mean" },
  { name: "std", sig: "std(x, axis=None)", desc: "Standard deviation" },
  { name: "dot", sig: "dot(a, b)", desc: "Matrix/vector product" },
  { name: "svd", sig: "svd(x)", desc: "Singular value decomposition" },
  { name: "qr", sig: "qr(x)", desc: "QR decomposition" },
];

const DEFAULT_FORMULA = `# Welcome to Formula Engine
# Write math formulas to transform data.
# 'data' is your input CSV. Define variables with =

# Project to 3D sphere
emb = sphere_proj(data)

# Cluster on the manifold
labels = kmeans(emb, k=5)

# Measure topology quality
score = silhouette(emb, labels)

# Graph Laplacian spectrum
L = laplacian(emb, k=15)
eig = eigvals(L)

# Persistent homology
hom = ripser(emb, maxdim=1)

# Curvature estimate
K = curvature_proxy(emb)
`;

export default function FormulaEditor({ onClose }: { onClose: () => void }) {
  const { selectedDataset, addLog } = useStore();
  const [source, setSource] = useState(DEFAULT_FORMULA);
  const [running, setRunning] = useState(false);
  const [building, setBuilding] = useState(false);
  const [engineName, setEngineName] = useState("");
  const [result, setResult] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDocs, setShowDocs] = useState(false);

  const handleRun = async () => {
    if (!selectedDataset) {
      addLog("Select a dataset first");
      return;
    }
    setRunning(true);
    setError(null);
    setResult(null);
    addLog("Running formula...");
    try {
      const res: any = await invoke("run_formula", { path: selectedDataset.path, source });
      if (res.error) {
        setError(res.error);
        addLog(`Formula error: ${res.error}`);
      } else {
        setResult(res.result);
        addLog(`Formula complete (${res.elapsed_ms}ms)`);
      }
    } catch (e: any) {
      setError(String(e));
      addLog(`Formula failed: ${e}`);
    } finally {
      setRunning(false);
    }
  };

  const handleBuild = async () => {
    if (!engineName.trim()) {
      addLog("Engine name required");
      return;
    }
    setBuilding(true);
    addLog(`Building formula engine: ${engineName}...`);
    try {
      const res: any = await invoke("formula_build", { name: engineName.trim(), source });
      addLog(`Formula engine created: ${res.engine_id}`);
      onClose();
    } catch (e: any) {
      addLog(`Build failed: ${e}`);
    } finally {
      setBuilding(false);
    }
  };

  const insertSnippet = (sig: string) => {
    setSource((s) => s + `\n${sig}`);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md">
      <div className="bg-gov-panel border border-gov-border rounded-xl w-[900px] h-[600px] flex flex-col shadow-2xl shadow-gov-accent/10">
        {/* Header */}
        <div className="h-10 border-b border-gov-border flex items-center px-4 gap-3 shrink-0">
          <Code size={14} className="text-gov-accent" />
          <span className="text-sm font-bold text-gov-accent">Formula Engine</span>
          <span className="text-[10px] text-gov-dim">— write math, build topology</span>
          <div className="flex-1" />
          <button onClick={() => setShowDocs(!showDocs)} className="text-gov-dim hover:text-gov-accent text-xs flex items-center gap-1">
            <BookOpen size={12} /> {showDocs ? "Hide" : "Docs"}
          </button>
          <button onClick={onClose} className="text-gov-dim hover:text-gov-error">
            <X size={14} />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 flex min-h-0">
          {/* Editor */}
          <div className="flex-1 flex flex-col min-w-0">
            <div className="flex-1 flex">
              <textarea
                value={source}
                onChange={(e) => setSource(e.target.value)}
                className="flex-1 bg-gov-bg text-gov-text font-mono text-xs p-3 resize-none outline-none border-r border-gov-border"
                spellCheck={false}
              />
              {/* Docs sidebar */}
              {showDocs && (
                <div className="w-64 bg-gov-bg border-l border-gov-border overflow-auto p-2 space-y-1 shrink-0">
                  <div className="text-[10px] text-gov-accent uppercase font-bold mb-2">Functions</div>
                  {BUILTINS.map((b) => (
                    <button
                      key={b.name}
                      onClick={() => insertSnippet(b.sig)}
                      className="w-full text-left p-1.5 rounded hover:bg-white/5 text-[10px] group"
                    >
                      <div className="font-mono text-gov-accent group-hover:text-white">{b.name}</div>
                      <div className="text-gov-dim">{b.desc}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Toolbar */}
            <div className="h-10 border-t border-gov-border flex items-center px-3 gap-2 shrink-0">
              <button
                onClick={handleRun}
                disabled={running || !selectedDataset}
                className="flex items-center gap-1 px-3 py-1 rounded text-xs font-bold bg-gov-accent text-black hover:brightness-110 disabled:opacity-50"
              >
                {running ? <Loader2 size={10} className="animate-spin" /> : <Play size={10} />}
                RUN
              </button>
              <input
                value={engineName}
                onChange={(e) => setEngineName(e.target.value)}
                placeholder="EngineName"
                className="bg-gov-bg border border-gov-border rounded px-2 py-1 text-xs text-gov-text outline-none focus:border-gov-accent w-32"
              />
              <button
                onClick={handleBuild}
                disabled={building || !engineName.trim()}
                className="flex items-center gap-1 px-3 py-1 rounded text-xs font-bold border border-gov-accent text-gov-accent hover:bg-gov-accent/10 disabled:opacity-50"
              >
                {building ? <Loader2 size={10} className="animate-spin" /> : <Save size={10} />}
                BUILD ENGINE
              </button>
              <div className="flex-1" />
              {selectedDataset && (
                <span className="text-[10px] text-gov-dim">{selectedDataset.name}</span>
              )}
            </div>
          </div>

          {/* Results */}
          <div className="w-64 bg-gov-bg border-l border-gov-border overflow-auto p-2 shrink-0">
            <div className="text-[10px] text-gov-accent uppercase font-bold mb-2">Results</div>
            {error && (
              <div className="text-[10px] text-gov-error bg-gov-error/10 rounded p-2 border border-gov-error/30">
                {error}
              </div>
            )}
            {result && (
              <div className="space-y-2">
                {Object.entries(result).map(([k, v]) => (
                  <div key={k} className="text-[10px] border border-gov-border rounded p-1.5">
                    <div className="font-mono text-gov-accent mb-0.5">{k}</div>
                    <div className="text-gov-dim truncate">
                      {Array.isArray(v)
                        ? `Array[${v.length}]`
                        : typeof v === "object"
                        ? JSON.stringify(v).slice(0, 60)
                        : String(v).slice(0, 60)}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {!result && !error && (
              <div className="text-gov-dim text-[10px] text-center mt-8">
                Run formula to see results
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
