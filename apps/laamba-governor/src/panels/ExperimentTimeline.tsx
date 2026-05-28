import { GitCommit, Trash2, BarChart3 } from "lucide-react";
import { useStore } from "../store";

export default function ExperimentTimeline() {
  const { experiments, analysisResult, battleResult, regressResult, classifyResult, setAnalysisResult, setBattleResult, setRegressResult, setClassifyResult } = useStore();

  const handleSelect = (exp: any) => {
    if (!exp.result) return;
    if (exp.command === "battle") {
      setBattleResult(exp.result);
    } else if (exp.command === "regress") {
      setRegressResult(exp.result);
    } else if (exp.command === "classify") {
      setClassifyResult(exp.result);
    } else {
      setAnalysisResult(exp.result);
    }
  };

  const statusColor = (s: string) => {
    if (s === "completed") return "bg-gov-ok/20 text-gov-ok";
    if (s === "running") return "bg-gov-accent/20 text-gov-accent";
    return "bg-gov-error/20 text-gov-error";
  };

  return (
    <div className="flex flex-col h-full">
      <div className="panel-header flex items-center justify-between">
        <span>Experiment Timeline</span>
        <span className="text-[10px] text-gov-dim">{experiments.length} runs</span>
      </div>
      <div className="panel-body space-y-1 overflow-auto">
        {experiments.length === 0 && (
          <div className="text-gov-dim text-xs p-2 text-center">
            No experiments yet. Run an analysis or battle.
          </div>
        )}
        {experiments.map((exp) => (
          <div
            key={exp.id}
            onClick={() => handleSelect(exp)}
            className={`flex items-center gap-2 p-2 rounded text-xs cursor-pointer border border-transparent hover:bg-white/5 transition-all ${
              exp.result ? "hover:border-gov-accent/30" : ""
            }`}
          >
            <GitCommit
              size={12}
              className={
                exp.status === "running"
                  ? "text-gov-accent animate-pulse shrink-0"
                  : exp.status === "completed"
                  ? "text-gov-ok shrink-0"
                  : "text-gov-error shrink-0"
              }
            />
            <div className="flex-1 min-w-0">
              <div className="truncate font-medium">{exp.name}</div>
              <div className="text-gov-dim text-[10px]">
                {exp.dataset} · {exp.elapsed_ms ? `${exp.elapsed_ms}ms` : "..."}
              </div>
              {exp.result && "chosen_manifold" in exp.result && (
                <div className="text-[9px] text-gov-accent">
                  {(exp.result as any).chosen_manifold}
                </div>
              )}
              {exp.result && "final_winner" in exp.result && (
                <div className="text-[9px] text-gov-accent">
                  winner: {(exp.result as any).final_winner}
                </div>
              )}
              {exp.result && "best_model" in exp.result && (
                <div className="text-[9px] text-gov-accent">
                  {(exp.result as any).best_model} · R²={(exp.result as any).best_r2?.toFixed(3) ?? "?"}
                </div>
              )}
              {exp.result && "best_accuracy" in exp.result && (
                <div className="text-[9px] text-gov-accent">
                  {(exp.result as any).best_model} · Acc={(exp.result as any).best_accuracy?.toFixed(3) ?? "?"}
                </div>
              )}
            </div>
            <div className={`text-[10px] px-1.5 py-0.5 rounded uppercase font-bold shrink-0 ${statusColor(exp.status)}`}>
              {exp.status}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
