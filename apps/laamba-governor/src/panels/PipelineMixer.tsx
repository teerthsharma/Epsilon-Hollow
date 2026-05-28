import { useCallback, useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  type Connection,
  type Edge,
  type Node,
} from "reactflow";
import "reactflow/dist/style.css";
import { Swords, Loader2, FolderOpen } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { useStore } from "../store";

const ORIGINAL_STYLES: Record<string, any> = {
  src: { background: "#FF8C00", color: "#000", border: "none", width: 130, fontSize: 11, fontWeight: 700 },
  euc: { background: "#00E5FF", color: "#000", border: "none", width: 130, fontSize: 11, fontWeight: 700 },
  sph: { background: "#AA00FF", color: "#fff", border: "none", width: 130, fontSize: 11, fontWeight: 700 },
  hyp: { background: "#FF6B35", color: "#000", border: "none", width: 130, fontSize: 11, fontWeight: 700 },
  gra: { background: "#00E676", color: "#000", border: "none", width: 130, fontSize: 11, fontWeight: 700 },
  pro: { background: "#FFD600", color: "#000", border: "none", width: 130, fontSize: 11, fontWeight: 700 },
  cmp: { background: "#FFEA00", color: "#000", border: "none", width: 130, fontSize: 11, fontWeight: 700 },
  fus: { background: "#AA00FF", color: "#fff", border: "none", width: 130, fontSize: 11, fontWeight: 700 },
  viz: { background: "#FF1744", color: "#fff", border: "none", width: 130, fontSize: 11, fontWeight: 700 },
};

const initialNodes: Node[] = [
  { id: "src", type: "default", position: { x: 30, y: 120 }, data: { label: "Data Source" }, style: ORIGINAL_STYLES.src },
  { id: "euc", type: "default", position: { x: 220, y: 30 }, data: { label: "Euclidean" }, style: ORIGINAL_STYLES.euc },
  { id: "sph", type: "default", position: { x: 220, y: 90 }, data: { label: "Spherical" }, style: ORIGINAL_STYLES.sph },
  { id: "hyp", type: "default", position: { x: 220, y: 150 }, data: { label: "Hyperbolic" }, style: ORIGINAL_STYLES.hyp },
  { id: "gra", type: "default", position: { x: 220, y: 210 }, data: { label: "Grassmannian" }, style: ORIGINAL_STYLES.gra },
  { id: "pro", type: "default", position: { x: 220, y: 270 }, data: { label: "Product" }, style: ORIGINAL_STYLES.pro },
  { id: "cmp", type: "default", position: { x: 430, y: 120 }, data: { label: "Comparator" }, style: ORIGINAL_STYLES.cmp },
  { id: "fus", type: "default", position: { x: 620, y: 90 }, data: { label: "Fusion" }, style: ORIGINAL_STYLES.fus },
  { id: "viz", type: "default", position: { x: 620, y: 170 }, data: { label: "Visualize" }, style: ORIGINAL_STYLES.viz },
];

const initialEdges: Edge[] = [
  { id: "e-src-euc", source: "src", target: "euc", animated: false, style: { stroke: "#444" } },
  { id: "e-src-sph", source: "src", target: "sph", animated: false, style: { stroke: "#444" } },
  { id: "e-src-hyp", source: "src", target: "hyp", animated: false, style: { stroke: "#444" } },
  { id: "e-src-gra", source: "src", target: "gra", animated: false, style: { stroke: "#444" } },
  { id: "e-src-pro", source: "src", target: "pro", animated: false, style: { stroke: "#444" } },
  { id: "e-euc-cmp", source: "euc", target: "cmp", animated: false, style: { stroke: "#444" } },
  { id: "e-sph-cmp", source: "sph", target: "cmp", animated: false, style: { stroke: "#444" } },
  { id: "e-hyp-cmp", source: "hyp", target: "cmp", animated: false, style: { stroke: "#444" } },
  { id: "e-gra-cmp", source: "gra", target: "cmp", animated: false, style: { stroke: "#444" } },
  { id: "e-pro-cmp", source: "pro", target: "cmp", animated: false, style: { stroke: "#444" } },
  { id: "e-cmp-fus", source: "cmp", target: "fus", animated: false, style: { stroke: "#444" } },
  { id: "e-cmp-viz", source: "cmp", target: "viz", animated: false, style: { stroke: "#444" } },
];

const TOPO_MAP: Record<string, string> = {
  euc: "euclidean",
  sph: "spherical",
  hyp: "hyperbolic_poincare",
  gra: "grassmannian",
  pro: "product",
};

export default function PipelineMixer() {
  const { selectedDataset, setBattleResult, addLog, addExperiment, updateExperiment, isRunning, setRunning } = useStore();
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [templates, setTemplates] = useState<{ name: string; content: any }[]>([]);

  // Sync source node label with selected dataset
  useEffect(() => {
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === "src") {
          return {
            ...n,
            data: { ...n.data, label: selectedDataset ? selectedDataset.name : "Data Source" },
          };
        }
        return n;
      })
    );
  }, [selectedDataset, setNodes]);

  useEffect(() => {
    (async () => {
      try {
        const result: any = await invoke("scan_templates");
        setTemplates(result.templates || []);
      } catch (e: any) {
        addLog(`Template scan failed: ${e}`);
      }
    })();
  }, []);

  const onConnect = useCallback(
    (params: Connection) =>
      setEdges((eds) => addEdge({ ...params, animated: false, style: { stroke: "#444" } }, eds)),
    [setEdges]
  );

  const resetNodeStyles = () => {
    setNodes((nds) =>
      nds.map((n) => {
        const orig = ORIGINAL_STYLES[n.id];
        if (orig) {
          return { ...n, style: { ...orig }, data: { ...n.data, status: "idle" } };
        }
        return n;
      })
    );
    setEdges((eds) => eds.map((e) => ({ ...e, animated: false, style: { ...e.style, stroke: "#444" } })));
  };

  const setNodesRunning = () => {
    setNodes((nds) =>
      nds.map((n) => {
        if (["euc", "sph", "hyp", "gra", "pro"].includes(n.id)) {
          return {
            ...n,
            style: { ...n.style, border: "2px solid #fff", boxShadow: "0 0 16px #fff", opacity: 1 },
            data: { ...n.data, status: "running" },
          };
        }
        if (n.id === "cmp") {
          return { ...n, style: { ...n.style, border: "2px solid #00E5FF" }, data: { ...n.data, status: "running" } };
        }
        return n;
      })
    );
    setEdges((eds) =>
      eds.map((e) =>
        e.source === "src" ? { ...e, animated: true, style: { ...e.style, stroke: "#00E5FF" } } : e
      )
    );
  };

  const setNodesResult = (winner: string) => {
    setNodes((nds) =>
      nds.map((n) => {
        const topoName = TOPO_MAP[n.id];
        if (topoName === winner) {
          return {
            ...n,
            style: {
              ...ORIGINAL_STYLES[n.id],
              background: "#00FF00",
              color: "#000",
              border: "3px solid #fff",
              boxShadow: "0 0 24px #00FF00",
              width: 140,
              fontSize: 12,
              fontWeight: 900,
            },
            data: { ...n.data, status: "winner" },
          };
        }
        if (["euc", "sph", "hyp", "gra", "pro"].includes(n.id)) {
          return {
            ...n,
            style: {
              ...ORIGINAL_STYLES[n.id],
              opacity: 0.35,
              border: "1px solid #333",
            },
            data: { ...n.data, status: "loser" },
          };
        }
        if (n.id === "cmp") {
          return {
            ...n,
            style: { ...ORIGINAL_STYLES.cmp, border: "2px solid #00FF00", boxShadow: "0 0 12px #00FF00" },
            data: { ...n.data, status: "winner" },
          };
        }
        return n;
      })
    );
    setEdges((eds) =>
      eds.map((e) => {
        const winnerId = Object.keys(TOPO_MAP).find((k) => TOPO_MAP[k] === winner);
        if (e.source === winnerId && e.target === "cmp") {
          return { ...e, animated: true, style: { ...e.style, stroke: "#00FF00", strokeWidth: 3 } };
        }
        if (e.source === "src" && e.target === winnerId) {
          return { ...e, animated: true, style: { ...e.style, stroke: "#00FF00", strokeWidth: 2 } };
        }
        return { ...e, animated: false, style: { ...e.style, stroke: "#222" } };
      })
    );
  };

  const loadTemplate = (tpl: any) => {
    try {
      const content = tpl.content;
      if (content.nodes) {
        setNodes(
          content.nodes.map((n: any) => ({
            ...n,
            style: n.style || initialNodes.find((in_) => in_.id === n.id)?.style,
          }))
        );
      }
      if (content.edges) {
        setEdges(
          content.edges.map((e: any) => ({
            ...e,
            animated: false,
            style: { stroke: "#444" },
          }))
        );
      }
      addLog(`Loaded template: ${tpl.name}`);
    } catch (e: any) {
      addLog(`Failed to load template: ${e}`);
    }
  };

  const runBattle = async () => {
    if (!selectedDataset) {
      addLog("Select a dataset in Sample Bay first");
      return;
    }
    setRunning(true);
    resetNodeStyles();
    setNodesRunning();

    const expId = `battle-${Date.now()}`;
    addExperiment({
      id: expId,
      name: `Battle Royale: ${selectedDataset.name}`,
      command: "battle",
      dataset: selectedDataset.name,
      status: "running",
      startedAt: new Date().toISOString(),
    });
    addLog(`Battle Royale started on ${selectedDataset.name}...`);

    try {
      const result: any = await invoke("run_battle", { path: selectedDataset.path });
      setBattleResult(result);
      updateExperiment(expId, {
        status: "completed",
        completedAt: new Date().toISOString(),
        result,
        elapsed_ms: result.elapsed_ms,
      });
      addLog(`Battle complete: winner=${result.final_winner} (${result.elapsed_ms}ms)`);
      setNodesResult(result.final_winner);
    } catch (e: any) {
      updateExperiment(expId, { status: "failed", completedAt: new Date().toISOString() });
      addLog(`Battle failed: ${e}`);
      resetNodeStyles();
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="panel-header flex items-center justify-between gap-2">
        <span>Pipeline Mixer</span>
        <div className="flex items-center gap-1">
          {templates.length > 0 && (
            <select
              onChange={(e) => {
                const tpl = templates.find((t) => t.name === e.target.value);
                if (tpl) loadTemplate(tpl);
                e.target.value = "";
              }}
              className="bg-gov-bg border border-gov-border rounded text-[10px] px-1 py-0.5 text-gov-dim outline-none focus:border-gov-accent"
              defaultValue=""
            >
              <option value="" disabled>Load template...</option>
              {templates.map((t) => (
                <option key={t.name} value={t.name}>{t.name}</option>
              ))}
            </select>
          )}
          <button
            onClick={runBattle}
            disabled={isRunning || !selectedDataset}
            className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-gov-accent text-black hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isRunning ? (
              <><Loader2 size={10} className="animate-spin" /> RUNNING...</>
            ) : (
              <><Swords size={10} /> BATTLE</>
            )}
          </button>
        </div>
      </div>
      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
          style={{ background: "#0d0d0d" }}
        >
          <Background color="#2a2a2a" gap={20} size={1} />
          <MiniMap nodeStrokeWidth={3} zoomable pannable style={{ background: "#141414" }} />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
}
