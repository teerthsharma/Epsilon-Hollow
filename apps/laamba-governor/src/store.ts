import { create } from "zustand";

export interface Dataset {
  name: string;
  path: string;
  description?: string;
  expected_topology?: string;
  shape: number[];
  rows?: number;
  cols?: number;
  type: string;
  format: string;
}

export interface Vitals {
  log_n: number;
  log_d: number;
  n_over_d: number;
  intrinsic_dim: number;
  mean_dist: number;
  std_dist: number;
  dist_ratio_95_5: number;
  spectral_gap: number;
  knee_clusters: number;
  curvature_proxy: number;
  small_world_coeff: number;
  sparsity: number;
  nan_ratio: number;
}

export interface AnalysisResult {
  command: string;
  dataset: string;
  path: string;
  shape: number[];
  chosen_manifold: string;
  curvature: number;
  dim: number;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  probabilities: Record<string, number>;
  vitals: Vitals;
  points_3d?: number[][];
  elapsed_ms: number;
}

export interface BattleRound {
  round: number;
  winner: string;
  winner_score: number;
  loser: string;
  loser_score: number;
  all_scores: Record<string, number>;
  weights: Record<string, number>;
}

export interface BattleResult {
  command: string;
  dataset: string;
  path: string;
  shape: number[];
  rounds: BattleRound[];
  final_winner: string;
  final_ranking: { topology: string; weight: number }[];
  loss_curve: number[];
  score_history: Record<string, number[]>;
  elapsed_ms: number;
}

export interface RankResult {
  command: string;
  dataset: string;
  path: string;
  shape: number[];
  ranking: { topology: string; probability: number }[];
  elapsed_ms: number;
}

export interface RegressResult {
  command: string;
  dataset: string;
  path: string;
  shape: number[];
  target_col: number;
  models: Record<string, { r2_mean?: number; r2_std?: number; error?: string }>;
  best_model: string;
  best_r2: number | null;
  elapsed_ms: number;
}

export interface ClassifyResult {
  command: string;
  dataset: string;
  path: string;
  shape: number[];
  target_col: number;
  models: Record<string, { accuracy_mean?: number; accuracy_std?: number; error?: string }>;
  best_model: string;
  best_accuracy: number | null;
  elapsed_ms: number;
}

export interface VitalsResult {
  command: string;
  dataset: string;
  path: string;
  shape: number[];
  dtype: string;
  vitals: Vitals;
  points_3d?: number[][];
  elapsed_ms: number;
}

export interface EngineResult {
  pid?: string;
  status?: string;
  elapsed_ms?: number;
  chosen_manifold?: string;
  [key: string]: any;
}

export interface Experiment {
  id: string;
  name: string;
  command: string;
  dataset: string;
  status: "running" | "completed" | "failed";
  startedAt: string;
  completedAt?: string;
  result?: AnalysisResult | BattleResult | RankResult | EngineResult;
  elapsed_ms?: number;
}

export interface AppStore {
  // Datasets
  datasets: Dataset[];
  selectedDataset: Dataset | null;
  setDatasets: (ds: Dataset[]) => void;
  selectDataset: (ds: Dataset | null) => void;

  // Vitals preview
  vitalsResult: VitalsResult | null;
  setVitalsResult: (v: VitalsResult | null) => void;

  // Analysis
  analysisResult: AnalysisResult | null;
  setAnalysisResult: (r: AnalysisResult | null) => void;

  // Battle
  battleResult: BattleResult | null;
  setBattleResult: (r: BattleResult | null) => void;

  // Rank
  rankResult: RankResult | null;
  setRankResult: (r: RankResult | null) => void;

  // Regress
  regressResult: RegressResult | null;
  setRegressResult: (r: RegressResult | null) => void;

  // Classify
  classifyResult: ClassifyResult | null;
  setClassifyResult: (r: ClassifyResult | null) => void;

  // Engine result
  engineResult: EngineResult | null;
  setEngineResult: (r: EngineResult | null) => void;

  // Experiments
  experiments: Experiment[];
  addExperiment: (e: Experiment) => void;
  updateExperiment: (id: string, patch: Partial<Experiment>) => void;

  // Logs
  logs: string[];
  addLog: (msg: string) => void;
  clearLogs: () => void;

  // Engine
  activeEngine: string | null;
  setActiveEngine: (id: string | null) => void;

  // Running state
  isRunning: boolean;
  setRunning: (r: boolean) => void;
}

export const useStore = create<AppStore>((set) => ({
  datasets: [],
  selectedDataset: null,
  setDatasets: (ds) => set({ datasets: ds }),
  selectDataset: (ds) => set({ selectedDataset: ds }),

  vitalsResult: null,
  setVitalsResult: (v) => set({ vitalsResult: v }),

  analysisResult: null,
  setAnalysisResult: (r) => set({ analysisResult: r }),

  battleResult: null,
  setBattleResult: (r) => set({ battleResult: r }),

  rankResult: null,
  setRankResult: (r) => set({ rankResult: r }),

  regressResult: null,
  setRegressResult: (r) => set({ regressResult: r }),

  classifyResult: null,
  setClassifyResult: (r) => set({ classifyResult: r }),

  engineResult: null,
  setEngineResult: (r) => set({ engineResult: r }),

  experiments: [],
  addExperiment: (e) => set((s) => ({ experiments: [e, ...s.experiments] })),
  updateExperiment: (id, patch) =>
    set((s) => ({
      experiments: s.experiments.map((e) => (e.id === id ? { ...e, ...patch } : e)),
    })),

  logs: ["> LAAMBA GOVERNOR initialized", "> Ready"],
  addLog: (msg) => set((s) => ({ logs: [...s.logs, `> ${msg}`] })),
  clearLogs: () => set({ logs: ["> Console cleared"] }),

  activeEngine: null,
  setActiveEngine: (id) => set({ activeEngine: id }),

  isRunning: false,
  setRunning: (r) => set({ isRunning: r }),
}));
