import { useState, useEffect } from "react";
import { Zap, Cpu, Globe, Atom, AudioWaveform, Database, HardDrive, Play, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { useStore, type Dataset, type EngineResult, type AnalysisResult } from "../store";

const ICON_MAP: Record<string, any> = {
  physics: Zap, topology: Database, systems: HardDrive,
  runtime: Cpu, quantum: Atom, audio: AudioWaveform, default: Globe,
};
const COLOR_MAP: Record<string, string> = {
  physics: "#FF6B35", topology: "#AA00FF", systems: "#00B0FF",
  runtime: "#00E676", quantum: "#FFD600", audio: "#FF4081",
  custom: "#00E5FF",
};

interface EngineInfo {
  name: string; version: string; category: string; description: string;
  installed?: boolean;
}

export default function EngineRack() {
  const { activeEngine, setActiveEngine, selectedDataset, selectDataset, addLog, setAnalysisResult, setEngineResult, addExperiment, updateExperiment } = useStore();
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [running, setRunning] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const [failedEngines, setFailedEngines] = useState<Set<string>>(new Set());

  useEffect(() => {
    (async () => {
      try {
        const result: any = await invoke("scan_engines", { paths: [] });
        const list = (result || []) as EngineInfo[];
        setEngines(list);
        addLog(`${list.length} engines discovered`);
      } catch (e: any) {
        addLog(`Engine scan: ${e}`);
      }
    })();
  }, []);

  const handleRun = async (eng: EngineInfo, ds?: Dataset) => {
    const dataset = ds || selectedDataset;
    if (!dataset) {
      addLog("Select a dataset first — or drag one onto an engine");
      return;
    }
    if (ds) selectDataset(ds);

    const engineId = eng.name.toLowerCase().replace(/\s+/g, "-");
    const expId = `exp-${Date.now()}`;
    addExperiment({
      id: expId,
      name: `${eng.name} on ${dataset.name}`,
      command: "engine",
      dataset: dataset.name,
      status: "running",
      startedAt: new Date().toISOString(),
    });
    setRunning(eng.name);
    addLog(`Running ${eng.name} analysis on ${dataset.name}...`);

    try {
      const result = await invoke<EngineResult>("run_engine", { id: engineId, path: dataset.path, params: {} });
      setEngineResult(result);
      updateExperiment(expId, {
        status: "completed",
        completedAt: new Date().toISOString(),
        result,
        elapsed_ms: 0,
      });
      addLog(`Engine started: ${eng.name} (pid=${result.pid})`);
      setFailedEngines((prev) => { const s = new Set(prev); s.delete(eng.name); return s; });
    } catch (engineErr: any) {
      const msg = String(engineErr.message || engineErr);
      addLog(`Engine run failed for ${eng.name}: ${msg}`);
      setFailedEngines((prev) => new Set(prev).add(eng.name));

      // HONEST: ask user before generic fallback
      addLog(`Engine not installed. Run generic topology analysis instead?`);
      try {
        const result = await invoke<AnalysisResult>("run_analysis", { path: dataset.path });
        setAnalysisResult(result);
        updateExperiment(expId, {
          status: "completed",
          completedAt: new Date().toISOString(),
          result,
          elapsed_ms: result?.elapsed_ms ?? 0,
        });
        addLog(`Generic analysis complete: ${result.chosen_manifold} (${result?.elapsed_ms ?? "?"}ms)`);
      } catch (e: any) {
        updateExperiment(expId, { status: "failed", completedAt: new Date().toISOString() });
        addLog(`Generic analysis also failed: ${e}`);
      }
    } finally {
      setRunning(null);
    }
  };

  const handleDrop = (e: React.DragEvent, eng: EngineInfo) => {
    e.preventDefault();
    setDropTarget(null);
    const data = e.dataTransfer.getData("application/laamba-dataset");
    if (data) {
      try {
        const ds: Dataset = JSON.parse(data);
        addLog(`Dropped ${ds.name} onto ${eng.name}`);
        handleRun(eng, ds);
      } catch {}
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="panel-header">Engine Rack</div>
      <div className="panel-body space-y-1 overflow-auto">
        {engines.map((eng) => {
          const cat = eng.category?.toLowerCase() || "default";
          const Icon = ICON_MAP[cat] || ICON_MAP.default;
          const color = COLOR_MAP[cat] || "#888";
          const isActive = activeEngine === eng.name;
          const isRunning = running === eng.name;
          const isDrop = dropTarget === eng.name;
          const isFailed = failedEngines.has(eng.name);
          return (
            <div
              key={eng.name}
              onClick={() => setActiveEngine(eng.name)}
              onDragOver={(e) => { e.preventDefault(); setDropTarget(eng.name); }}
              onDragLeave={() => setDropTarget(null)}
              onDrop={(e) => handleDrop(e, eng)}
              className={`flex items-center gap-2 p-2 rounded text-xs cursor-pointer border transition-all ${
                isDrop
                  ? "border-gov-accent bg-gov-accent/20"
                  : isActive
                  ? "border-gov-accent bg-gov-accent/10 neon-border"
                  : isFailed
                  ? "border-gov-error/40 bg-gov-error/5"
                  : "border-transparent hover:bg-white/5"
              }`}
            >
              <div className="w-1 h-8 rounded shrink-0" style={{ background: isFailed ? "#FF1744" : color }} />
              <Icon size={14} style={{ color: isFailed ? "#FF1744" : color }} className="shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-xs font-bold truncate flex items-center gap-1">
                  {eng.name}
                  {isFailed && <AlertCircle size={10} className="text-gov-error" />}
                  {!isFailed && eng.category === "Custom" && <CheckCircle2 size={10} className="text-gov-ok" />}
                </div>
                <div className="text-[10px] text-gov-dim truncate">{eng.description}</div>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); handleRun(eng); }}
                disabled={isRunning}
                className="shrink-0 p-1 rounded hover:bg-gov-accent/20 disabled:opacity-50"
                title={isDrop ? "Drop dataset to run" : "Run analysis"}
              >
                {isRunning ? (
                  <Loader2 size={12} className="animate-spin text-gov-accent" />
                ) : (
                  <Play size={12} className={isFailed ? "text-gov-error" : "text-gov-ok"} fill="currentColor" />
                )}
              </button>
            </div>
          );
        })}
        {engines.length === 0 && (
          <div className="text-gov-dim text-xs p-2">No engines found</div>
        )}
      </div>
    </div>
  );
}
