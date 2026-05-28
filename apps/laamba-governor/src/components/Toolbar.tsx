import { Play, Square, BarChart3, Loader2, Search, TrendingUp, Tag, Wrench, Code } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { useStore } from "../store";
import { useState } from "react";
import FormulaEditor from "../panels/FormulaEditor";

export default function Toolbar() {
  const {
    selectedDataset,
    isRunning,
    setRunning,
    addLog,
    setBattleResult,
    setAnalysisResult,
    setRankResult,
    setRegressResult,
    setClassifyResult,
    addExperiment,
    updateExperiment,
  } = useStore();

  const [showFormula, setShowFormula] = useState(false);

  const runCmd = async (
    cmd: string,
    invokeCmd: string,
    args: any,
    setResult: (r: any) => void,
    logPrefix: string,
    resultKey: string
  ) => {
    if (!selectedDataset) {
      addLog(`No dataset selected — click one in Sample Bay`);
      return;
    }
    setRunning(true);
    const expId = `${cmd}-${Date.now()}`;
    addExperiment({
      id: expId,
      name: `${logPrefix}: ${selectedDataset.name}`,
      command: cmd,
      dataset: selectedDataset.name,
      status: "running",
      startedAt: new Date().toISOString(),
    });
    addLog(`${logPrefix} started on ${selectedDataset.name}...`);
    try {
      const result: any = await invoke(invokeCmd, args);
      setResult(result);
      updateExperiment(expId, {
        status: "completed",
        completedAt: new Date().toISOString(),
        result,
        elapsed_ms: result.elapsed_ms,
      });
      const summary = result[resultKey] ?? result.final_winner ?? result.chosen_manifold ?? "done";
      addLog(`${logPrefix} complete: ${summary} (${result.elapsed_ms}ms)`);
    } catch (e: any) {
      updateExperiment(expId, { status: "failed", completedAt: new Date().toISOString() });
      addLog(`${logPrefix} failed: ${e}`);
    } finally {
      setRunning(false);
    }
  };

  const handlePlay = () => runCmd("battle", "run_battle", { path: selectedDataset?.path }, setBattleResult, "Pipeline", "final_winner");
  const handleAnalyze = () => runCmd("analyze", "run_analysis", { path: selectedDataset?.path }, setAnalysisResult, "Analyze", "chosen_manifold");
  const handleRank = () => runCmd("rank", "run_rank", { path: selectedDataset?.path }, setRankResult, "Rank", "ranking");
  const handleRegress = () => runCmd("regress", "run_regress", { path: selectedDataset?.path, target: -1 }, setRegressResult, "Regress", "best_model");
  const handleClassify = () => runCmd("classify", "run_classify", { path: selectedDataset?.path, target: -1 }, setClassifyResult, "Classify", "best_model");

  const handleStop = () => {
    addLog("Stop requested (current run will complete)");
  };

  return (
    <div className="h-10 bg-gov-panel border-b border-gov-border flex items-center px-3 gap-2 shrink-0">
      <span className="text-gov-accent font-bold text-sm tracking-widest mr-1">LAAMBA GOVERNOR</span>
      <span className="text-[9px] text-gov-dim/40 mr-3 self-end mb-1">by seal</span>
      <button
        onClick={handlePlay}
        disabled={isRunning}
        className="flex items-center gap-1 bg-gov-accent text-black px-3 py-1 rounded text-xs font-bold hover:brightness-110 disabled:opacity-50"
      >
        {isRunning ? (
          <><Loader2 size={12} className="animate-spin" /> RUNNING</>
        ) : (
          <><Play size={12} fill="black" /> PLAY</>
        )}
      </button>
      <button
        onClick={handleAnalyze}
        disabled={isRunning || !selectedDataset}
        className="flex items-center gap-1 bg-gov-panel border border-gov-border px-3 py-1 rounded text-xs hover:border-gov-accent disabled:opacity-40"
      >
        <Search size={12} /> ANALYZE
      </button>
      <button
        onClick={handleRegress}
        disabled={isRunning || !selectedDataset}
        className="flex items-center gap-1 bg-gov-panel border border-gov-border px-3 py-1 rounded text-xs hover:border-gov-accent disabled:opacity-40"
      >
        <TrendingUp size={12} /> REGRESS
      </button>
      <button
        onClick={handleClassify}
        disabled={isRunning || !selectedDataset}
        className="flex items-center gap-1 bg-gov-panel border border-gov-border px-3 py-1 rounded text-xs hover:border-gov-accent disabled:opacity-40"
      >
        <Tag size={12} /> CLASSIFY
      </button>
      <button
        onClick={handleStop}
        className="flex items-center gap-1 bg-gov-panel border border-gov-border px-3 py-1 rounded text-xs hover:border-gov-accent"
      >
        <Square size={12} /> STOP
      </button>
      <button
        onClick={handleRank}
        disabled={isRunning || !selectedDataset}
        className="flex items-center gap-1 bg-gov-panel border border-gov-border px-3 py-1 rounded text-xs hover:border-gov-accent disabled:opacity-40"
      >
        <BarChart3 size={12} /> RANK
      </button>

      <div className="flex-1" />

      <button
        onClick={() => setShowFormula(true)}
        className="flex items-center gap-1 bg-gov-panel border border-gov-border px-3 py-1 rounded text-xs hover:border-gov-accent text-gov-accent"
      >
        <Code size={12} /> FORMULA
      </button>
      <button
        onClick={() => setShowFormula(true)}
        className="flex items-center gap-1 bg-gov-panel border border-gov-border px-3 py-1 rounded text-xs hover:border-gov-accent"
      >
        <Wrench size={12} /> BUILD ENGINE
      </button>

      {showFormula && <FormulaEditor onClose={() => setShowFormula(false)} />}

      {selectedDataset && (
        <div className="text-[10px] text-gov-dim font-mono">
          {selectedDataset.name} [{selectedDataset.shape?.join("x")}]
        </div>
      )}
      <div className={`w-2 h-2 rounded-full ${isRunning ? "bg-gov-accent animate-pulse" : "bg-gov-ok"}`} />
    </div>
  );
}
