import { useStore } from "../store";
import { Cpu, TrendingUp, Hash, Gauge, BarChart3, Tag } from "lucide-react";

export default function ParameterRoll() {
  const { vitalsResult, analysisResult, battleResult, regressResult, classifyResult, selectedDataset } = useStore();

  const vitals = vitalsResult?.vitals || analysisResult?.vitals;
  const analysis = analysisResult;
  const battle = battleResult;
  const regress = regressResult;
  const classify = classifyResult;

  if (!vitals && !analysis && !battle && !regress && !classify) {
    return (
      <div className="flex flex-col h-full">
        <div className="panel-header">Parameter Roll</div>
        <div className="panel-body text-gov-dim text-xs flex items-center justify-center">
          Select a dataset and run analysis
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="panel-header flex items-center justify-between">
        <span>Parameter Roll</span>
        {selectedDataset && (
          <span className="text-[10px] text-gov-accent">{selectedDataset.name}</span>
        )}
      </div>
      <div className="panel-body space-y-2 overflow-auto">
        {/* Analysis config */}
        {analysis && (
          <div className="space-y-1">
            <div className="text-[10px] text-gov-accent uppercase tracking-wider font-bold flex items-center gap-1">
              <Cpu size={10} /> Governor Config
            </div>
            <Row label="Manifold" value={analysis.chosen_manifold} accent />
            <Row label="Curvature" value={analysis.curvature.toFixed(4)} />
            <Row label="Dim" value={analysis.dim} />
            <Row label="LR" value={analysis.learning_rate.toExponential(2)} />
            <Row label="Batch" value={analysis.batch_size} />
            <Row label="Epochs" value={analysis.epochs} />
          </div>
        )}

        {/* Battle ranking */}
        {battle && (
          <div className="space-y-1">
            <div className="text-[10px] text-gov-accent uppercase tracking-wider font-bold flex items-center gap-1 mt-2">
              <TrendingUp size={10} /> Battle Ranking
            </div>
            {battle.final_ranking.map((r, i) => (
              <div key={r.topology} className="flex items-center gap-2 text-[11px]">
                <span className={`w-3 text-right font-mono ${i === 0 ? "text-gov-ok font-bold" : "text-gov-dim"}`}>
                  {i + 1}
                </span>
                <div className="flex-1 h-2 bg-gov-bg rounded overflow-hidden">
                  <div
                    className="h-full rounded"
                    style={{
                      width: `${r.weight * 100}%`,
                      background: i === 0 ? "#00E5FF" : "#444",
                    }}
                  />
                </div>
                <span className="text-[10px] font-mono w-24 truncate">{r.topology}</span>
                <span className="text-[10px] font-mono text-gov-dim w-10 text-right">
                  {(r.weight * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Regression results */}
        {regress && (
          <div className="space-y-1">
            <div className="text-[10px] text-gov-accent uppercase tracking-wider font-bold flex items-center gap-1 mt-2">
              <BarChart3 size={10} /> Regression
            </div>
            {Object.entries(regress.models).map(([mname, m]) => (
              <div key={mname} className="flex justify-between text-[11px] px-1">
                <span className="text-gov-dim">{mname}</span>
                <span className={`font-mono ${mname === regress.best_model ? "text-gov-ok font-bold" : "text-gov-text"}`}>
                  {m.r2_mean !== undefined ? `R²=${m.r2_mean.toFixed(3)}` : `err`}
                </span>
              </div>
            ))}
            <Row label="Best" value={regress.best_model} accent />
            <Row label="Target col" value={regress.target_col} />
          </div>
        )}

        {/* Classification results */}
        {classify && (
          <div className="space-y-1">
            <div className="text-[10px] text-gov-accent uppercase tracking-wider font-bold flex items-center gap-1 mt-2">
              <Tag size={10} /> Classification
            </div>
            {Object.entries(classify.models).map(([mname, m]) => (
              <div key={mname} className="flex justify-between text-[11px] px-1">
                <span className="text-gov-dim">{mname}</span>
                <span className={`font-mono ${mname === classify.best_model ? "text-gov-ok font-bold" : "text-gov-text"}`}>
                  {m.accuracy_mean !== undefined ? `Acc=${m.accuracy_mean.toFixed(3)}` : `err`}
                </span>
              </div>
            ))}
            <Row label="Best" value={classify.best_model} accent />
            <Row label="Target col" value={classify.target_col} />
          </div>
        )}

        {/* Vitals */}
        {vitals && (
          <div className="space-y-1">
            <div className="text-[10px] text-gov-accent uppercase tracking-wider font-bold flex items-center gap-1 mt-2">
              <Gauge size={10} /> Topological Vitals
            </div>
            <Row label="Intrinsic Dim" value={vitals.intrinsic_dim.toFixed(2)} />
            <Row label="Spectral Gap" value={vitals.spectral_gap.toFixed(4)} />
            <Row label="Curvature" value={vitals.curvature_proxy.toFixed(4)} />
            <Row label="Clusters" value={vitals.knee_clusters.toFixed(0)} />
            <Row label="Small-World" value={vitals.small_world_coeff.toFixed(3)} />
            <Row label="Mean Dist" value={vitals.mean_dist.toFixed(4)} />
            <Row label="Sparsity" value={vitals.sparsity.toFixed(4)} />
            <Row label="n/d Ratio" value={vitals.n_over_d.toFixed(1)} />
          </div>
        )}
      </div>
    </div>
  );
}

function Row({ label, value, accent }: { label: string; value: any; accent?: boolean }) {
  return (
    <div className="flex justify-between items-center text-[11px] px-1">
      <span className="text-gov-dim">{label}</span>
      <span className={`font-mono ${accent ? "text-gov-accent font-bold" : "text-gov-text"}`}>
        {value}
      </span>
    </div>
  );
}
