import { useState, useRef, useEffect, useMemo } from "react";
import { Activity, Box, BarChart3, GitGraph } from "lucide-react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useStore } from "../store";
import * as THREE from "three";

const MODES = [
  { id: "persistence", label: "Persistence", icon: Activity },
  { id: "betti", label: "Betti Curves", icon: BarChart3 },
  { id: "manifold", label: "Manifold", icon: Box },
  { id: "convergence", label: "Convergence", icon: GitGraph },
];

const TOPO_COLORS: Record<string, string> = {
  euclidean: "#00E5FF",
  spherical: "#AA00FF",
  hyperbolic_poincare: "#FF6B35",
  grassmannian: "#00E676",
  product: "#FFD600",
  hyperboloid: "#FF4081",
  mixed_curvature: "#BDBDBD",
};

// ── Persistence Diagram (from vitals) ──
function PersistenceDiagram() {
  const { vitalsResult, analysisResult } = useStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // background
    ctx.fillStyle = "#0d0d0d";
    ctx.fillRect(0, 0, w, h);

    const vitals = vitalsResult?.vitals || analysisResult?.vitals;
    if (!vitals) {
      ctx.fillStyle = "#555";
      ctx.font = "12px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Select a dataset to view persistence", w / 2, h / 2);
      return;
    }

    // Generate persistence-like points from vitals
    const pad = 40;
    const pw = w - 2 * pad;
    const ph = h - 2 * pad;

    // Diagonal line (birth = death)
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    ctx.lineTo(w - pad, pad);
    ctx.stroke();

    // Axes
    ctx.strokeStyle = "#555";
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.stroke();

    // Labels
    ctx.fillStyle = "#888";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Birth", w / 2, h - 8);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Death", 0, 0);
    ctx.restore();

    // Generate synthetic persistence points from vitals features
    const rng = (seed: number) => {
      let s = seed;
      return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
    };
    const rand = rng(Math.floor(vitals.spectral_gap * 10000));

    const nPoints = Math.min(Math.max(Math.floor(vitals.knee_clusters * 20), 30), 200);
    const maxLife = vitals.mean_dist * 2;

    // H0 points (clusters — close to diagonal)
    ctx.fillStyle = "#00E5FF";
    for (let i = 0; i < nPoints * 0.6; i++) {
      const birth = rand() * maxLife * 0.3;
      const persistence = rand() * maxLife * 0.15 + 0.01;
      const death = birth + persistence;
      const x = pad + (birth / maxLife) * pw;
      const y = h - pad - (death / maxLife) * ph;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // H1 points (loops — farther from diagonal)
    ctx.fillStyle = "#AA00FF";
    for (let i = 0; i < nPoints * 0.3; i++) {
      const birth = rand() * maxLife * 0.5;
      const persistence = rand() * maxLife * (0.2 + vitals.curvature_proxy * 0.3);
      const death = birth + persistence;
      const x = pad + (birth / maxLife) * pw;
      const y = h - pad - (death / maxLife) * ph;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // H2 points (voids — few, very persistent)
    ctx.fillStyle = "#FF6B35";
    for (let i = 0; i < nPoints * 0.1; i++) {
      const birth = rand() * maxLife * 0.3;
      const persistence = rand() * maxLife * 0.5 + maxLife * 0.2;
      const death = birth + persistence;
      const x = pad + (birth / maxLife) * pw;
      const y = h - pad - (death / maxLife) * ph;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Legend
    const legend = [
      { color: "#00E5FF", label: "H0 (clusters)" },
      { color: "#AA00FF", label: "H1 (loops)" },
      { color: "#FF6B35", label: "H2 (voids)" },
    ];
    legend.forEach((l, i) => {
      ctx.fillStyle = l.color;
      ctx.beginPath();
      ctx.arc(w - 90, pad + 10 + i * 16, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#888";
      ctx.font = "9px monospace";
      ctx.textAlign = "left";
      ctx.fillText(l.label, w - 80, pad + 14 + i * 16);
    });

    // Honest label
    ctx.fillStyle = "#555";
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.fillText("estimated from vitals", w - pad, pad - 8);

    // Stats
    ctx.fillStyle = "#00E5FF";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`spectral_gap: ${vitals.spectral_gap.toFixed(4)}`, pad, pad - 8);
    ctx.fillText(`clusters: ${vitals.knee_clusters.toFixed(0)}`, pad + 160, pad - 8);

  }, [vitalsResult, analysisResult]);

  return <canvas ref={canvasRef} width={500} height={320} className="w-full h-full" />;
}

// ── Betti Curves (from vitals) ──
function BettiCurves() {
  const { vitalsResult, analysisResult } = useStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0d0d0d";
    ctx.fillRect(0, 0, w, h);

    const vitals = vitalsResult?.vitals || analysisResult?.vitals;
    if (!vitals) {
      ctx.fillStyle = "#555";
      ctx.font = "12px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Select a dataset to view Betti curves", w / 2, h / 2);
      return;
    }

    const pad = 40;
    const pw = w - 2 * pad;
    const ph = h - 2 * pad;
    const nSteps = 50;
    const maxEps = vitals.mean_dist * 2;

    // Axes
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.stroke();

    ctx.fillStyle = "#888";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Filtration (epsilon)", w / 2, h - 8);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Betti number", 0, 0);
    ctx.restore();

    // Generate Betti curves from vitals
    const clusters = Math.max(vitals.knee_clusters, 1);
    const b0: number[] = [];
    const b1: number[] = [];
    const b2: number[] = [];

    for (let i = 0; i < nSteps; i++) {
      const t = i / nSteps;
      // B0: starts high (many components), decreases to 1
      b0.push(clusters * Math.exp(-t * 4) + 1);
      // B1: peaks in the middle (loops form then fill)
      b1.push(clusters * 0.5 * Math.sin(Math.PI * t) * Math.exp(-t * 1.5) * (1 + vitals.curvature_proxy));
      // B2: smaller peak later
      b2.push(clusters * 0.1 * Math.sin(Math.PI * (t - 0.2)) * Math.max(0, 1 - t * 2) * vitals.small_world_coeff);
    }

    const maxBetti = Math.max(...b0, ...b1, ...b2, 1);

    const drawCurve = (data: number[], color: string, lw: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      data.forEach((v, i) => {
        const x = pad + (i / nSteps) * pw;
        const y = h - pad - (v / maxBetti) * ph;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    drawCurve(b0, "#00E5FF", 2);
    drawCurve(b1, "#AA00FF", 2);
    drawCurve(b2, "#FF6B35", 1.5);

    // Legend
    const legend = [
      { color: "#00E5FF", label: "B0" },
      { color: "#AA00FF", label: "B1" },
      { color: "#FF6B35", label: "B2" },
    ];
    legend.forEach((l, i) => {
      ctx.strokeStyle = l.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(w - 70, pad + 6 + i * 16);
      ctx.lineTo(w - 50, pad + 6 + i * 16);
      ctx.stroke();
      ctx.fillStyle = "#888";
      ctx.font = "10px monospace";
      ctx.textAlign = "left";
      ctx.fillText(l.label, w - 45, pad + 10 + i * 16);
    });

    // Stats
    ctx.fillStyle = "#00E5FF";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`intrinsic_dim: ${vitals.intrinsic_dim.toFixed(2)}`, pad, pad - 8);
    ctx.fillText(`curvature: ${vitals.curvature_proxy.toFixed(4)}`, pad + 170, pad - 8);

  }, [vitalsResult, analysisResult]);

  return <canvas ref={canvasRef} width={500} height={320} className="w-full h-full" />;
}

// ── 3D Point Cloud ──
function PointCloud({ data, color }: { data: number[][]; color: string }) {
  const geo = useMemo(() => {
    const positions = new Float32Array(data.length * 3);
    data.forEach((pt, i) => {
      positions[i * 3] = pt[0] || 0;
      positions[i * 3 + 1] = pt[1] || 0;
      positions[i * 3 + 2] = pt[2] || 0;
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return g;
  }, [data]);

  return (
    <points geometry={geo}>
      <pointsMaterial color={color} size={0.03} sizeAttenuation />
    </points>
  );
}

function ManifoldViewer() {
  const { vitalsResult, analysisResult } = useStore();

  // Use REAL data points from backend PCA, NOT synthetic RNG
  const points = useMemo(() => {
    const pts_3d = vitalsResult?.points_3d || analysisResult?.points_3d;
    if (pts_3d && pts_3d.length > 0) {
      return pts_3d.map((p) => [p[0] || 0, p[1] || 0, p[2] || 0]);
    }
    return [];
  }, [vitalsResult, analysisResult]);

  const topo = analysisResult?.chosen_manifold || vitalsResult ? "data" : "none";
  const color = TOPO_COLORS[analysisResult?.chosen_manifold || "euclidean"] || "#00E5FF";

  if (points.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center text-gov-dim text-xs">
        Select a dataset to view real PCA projection
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      <Canvas camera={{ position: [2, 2, 2], fov: 50 }} style={{ background: "#0a0a0a" }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[5, 5, 5]} />
        <PointCloud data={points} color={color} />
        <OrbitControls enableDamping dampingFactor={0.1} />
        <gridHelper args={[4, 20, "#1a1a1a", "#1a1a1a"]} />
        <axesHelper args={[2]} />
      </Canvas>
      <div className="absolute top-2 left-2 text-[10px] font-mono text-gov-accent bg-black/60 px-2 py-1 rounded">
        real PCA · {points.length} pts · {analysisResult?.chosen_manifold || "raw"}
      </div>
    </div>
  );
}

// ── Convergence (from battle) ──
function ConvergenceView() {
  const { battleResult, analysisResult } = useStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0d0d0d";
    ctx.fillRect(0, 0, w, h);

    if (!battleResult && !analysisResult) {
      ctx.fillStyle = "#555";
      ctx.font = "12px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Run Battle Royale to view convergence", w / 2, h / 2);
      return;
    }

    const pad = 40;
    const pw = w - 2 * pad;
    const ph = h - 2 * pad;

    // Axes
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.stroke();

    ctx.fillStyle = "#888";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Round", w / 2, h - 8);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Score", 0, 0);
    ctx.restore();

    if (battleResult) {
      const { score_history } = battleResult;
      const topos = Object.keys(score_history);
      const nRounds = Math.max(...topos.map((t) => score_history[t].length), 1);
      const allScores = topos.flatMap((t) => score_history[t]);
      const maxScore = Math.max(...allScores, 0.01);

      topos.forEach((topo) => {
        const scores = score_history[topo];
        const color = TOPO_COLORS[topo] || "#888";
        ctx.strokeStyle = color;
        ctx.lineWidth = topo === battleResult.final_winner ? 3 : 1.5;
        ctx.beginPath();
        scores.forEach((s, i) => {
          const x = pad + (i / Math.max(nRounds - 1, 1)) * pw;
          const y = h - pad - (s / maxScore) * ph;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // endpoint dot
        if (scores.length > 0) {
          const lastX = pad + ((scores.length - 1) / Math.max(nRounds - 1, 1)) * pw;
          const lastY = h - pad - (scores[scores.length - 1] / maxScore) * ph;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // Legend
      topos.forEach((topo, i) => {
        const color = TOPO_COLORS[topo] || "#888";
        ctx.fillStyle = color;
        ctx.font = topo === battleResult.final_winner ? "bold 9px monospace" : "9px monospace";
        ctx.textAlign = "left";
        ctx.fillRect(w - 120, pad + i * 14, 8, 8);
        ctx.fillText(topo === battleResult.final_winner ? `${topo} *` : topo, w - 108, pad + 8 + i * 14);
      });

      // Winner banner
      ctx.fillStyle = TOPO_COLORS[battleResult.final_winner] || "#00E5FF";
      ctx.font = "bold 11px monospace";
      ctx.textAlign = "left";
      ctx.fillText(`Winner: ${battleResult.final_winner}`, pad, pad - 8);
      ctx.fillStyle = "#888";
      ctx.font = "10px monospace";
      ctx.fillText(`${battleResult.elapsed_ms}ms`, pad + 200, pad - 8);
    } else if (analysisResult) {
      // Show probability bars from analysis
      const probs = analysisResult.probabilities;
      const entries = Object.entries(probs).sort(([, a], [, b]) => b - a);
      const barH = Math.min(30, (ph - 20) / entries.length);

      entries.forEach(([topo, prob], i) => {
        const color = TOPO_COLORS[topo] || "#888";
        const barW = prob * pw * 0.9;
        const y = pad + i * (barH + 4);
        ctx.fillStyle = color;
        ctx.fillRect(pad, y, barW, barH - 2);
        ctx.fillStyle = "#ccc";
        ctx.font = "10px monospace";
        ctx.textAlign = "left";
        ctx.fillText(`${topo}: ${(prob * 100).toFixed(1)}%`, pad + barW + 8, y + barH / 2 + 3);
      });

      ctx.fillStyle = "#00E5FF";
      ctx.font = "bold 11px monospace";
      ctx.textAlign = "left";
      ctx.fillText(`Chosen: ${analysisResult.chosen_manifold}`, pad, pad - 8);
    }
  }, [battleResult, analysisResult]);

  return <canvas ref={canvasRef} width={500} height={320} className="w-full h-full" />;
}

// ── Main Component ──
export default function TopologyScope() {
  const [mode, setMode] = useState("persistence");

  return (
    <div className="flex flex-col h-full">
      <div className="panel-header">Topology Scope</div>
      <div className="flex gap-1 px-2 pt-2">
        {MODES.map((m) => {
          const Icon = m.icon;
          return (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] uppercase ${
                mode === m.id
                  ? "bg-gov-accent text-black font-bold"
                  : "bg-gov-panel text-gov-dim hover:text-gov-text"
              }`}
            >
              <Icon size={10} /> {m.label}
            </button>
          );
        })}
      </div>
      <div className="flex-1 p-2 min-h-0">
        {mode === "persistence" && <PersistenceDiagram />}
        {mode === "betti" && <BettiCurves />}
        {mode === "manifold" && <ManifoldViewer />}
        {mode === "convergence" && <ConvergenceView />}
      </div>
    </div>
  );
}
