import { useState, useCallback } from "react";
import { Database, Eye, Loader2, Plus, Upload } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { useStore, type Dataset } from "../store";

export default function SampleBay() {
  const { datasets, selectedDataset, selectDataset, setVitalsResult, addLog, setDatasets } = useStore();
  const [loading, setLoading] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleSelect = async (ds: Dataset) => {
    selectDataset(ds);
    setLoading(ds.name);
    addLog(`Loading vitals for ${ds.name}...`);
    try {
      const result: any = await invoke("dataset_preview", { path: ds.path });
      setVitalsResult(result);
      addLog(`Vitals extracted: ${ds.name} [${result.shape?.join("x")}] in ${result.elapsed_ms}ms`);
    } catch (e: any) {
      addLog(`Vitals failed for ${ds.name}: ${e}`);
    } finally {
      setLoading(null);
    }
  };

  const importFile = async (filePath: string) => {
    const name = filePath.split(/[/\\]/).pop() || filePath;
    const newDs: Dataset = {
      name,
      path: filePath,
      shape: [0, 0],
      type: "point_cloud",
      format: "csv",
    };

    // Add to datasets list
    setDatasets([...datasets, newDs]);
    addLog(`Imported: ${name}`);

    // Auto-load vitals
    selectDataset(newDs);
    setLoading(name);
    try {
      const result: any = await invoke("dataset_preview", { path: filePath });
      setVitalsResult(result);
      // Update the dataset with real shape
      const updated = {
        ...newDs,
        shape: result.shape || [0, 0],
        rows: result.shape?.[0],
        cols: result.shape?.[1],
      };
      setDatasets(datasets.map((d) => (d.name === name ? updated : d)).concat(
        datasets.some((d) => d.name === name) ? [] : [updated]
      ));
      addLog(`Vitals: ${name} [${result.shape?.join("x")}] ${result.elapsed_ms}ms`);
    } catch (e: any) {
      addLog(`Import vitals failed: ${e}`);
    } finally {
      setLoading(null);
    }
  };

  const handleImportClick = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: "CSV", extensions: ["csv"] }],
      });
      if (selected) {
        await importFile(selected as string);
      }
    } catch (e: any) {
      addLog(`File dialog error: ${e}`);
    }
  };

  // HTML5 drag-drop (Tauri passes file paths through dataTransfer)
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(false);

      // In Tauri, dropped files come through dataTransfer.files
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        for (let i = 0; i < files.length; i++) {
          const file = files[i];
          if (file.name.endsWith(".csv")) {
            // Tauri provides the path property on File objects
            const path = (file as any).path || file.name;
            await importFile(path);
          } else {
            addLog(`Skipped ${file.name} — only CSV supported`);
          }
        }
      }
    },
    [datasets]
  );

  return (
    <div
      className="flex flex-col h-full"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="panel-header flex items-center justify-between">
        <span>Sample Bay</span>
        <button
          onClick={handleImportClick}
          className="flex items-center gap-1 text-[10px] text-gov-dim hover:text-gov-accent"
          title="Import CSV"
        >
          <Plus size={10} />
        </button>
      </div>
      <div className="panel-body space-y-1 flex-1 overflow-auto">
        {datasets.length === 0 && !dragOver && (
          <div className="text-gov-dim text-xs flex items-center gap-2 p-2">
            <Loader2 size={12} className="animate-spin" /> Scanning...
          </div>
        )}
        {datasets.map((ds) => (
          <div
            key={ds.name}
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData("application/laamba-dataset", JSON.stringify(ds));
              e.dataTransfer.effectAllowed = "copy";
            }}
            onClick={() => handleSelect(ds)}
            className={`flex items-center gap-2 p-2 rounded text-xs cursor-pointer border transition-all ${
              selectedDataset?.name === ds.name
                ? "border-gov-accent bg-gov-accent/10"
                : "border-transparent hover:bg-white/5"
            }`}
          >
            <Database size={14} className="text-gov-data shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="truncate font-medium">{ds.name}</div>
              <div className="text-gov-dim text-[10px]">
                {ds.rows || ds.shape?.[0] || "?"}x{ds.cols || ds.shape?.[1] || "?"} · {ds.format}
              </div>
              {ds.expected_topology && (
                <div className="text-[9px] text-gov-accent/60">{ds.expected_topology}</div>
              )}
            </div>
            {loading === ds.name ? (
              <Loader2 size={12} className="animate-spin text-gov-accent shrink-0" />
            ) : (
              <Eye size={12} className="text-gov-dim hover:text-gov-accent shrink-0" />
            )}
          </div>
        ))}
      </div>

      {/* Drop zone */}
      <div
        className={`mx-2 mb-2 border-2 border-dashed rounded-lg flex flex-col items-center justify-center p-3 transition-all cursor-pointer ${
          dragOver
            ? "border-gov-accent bg-gov-accent/10 text-gov-accent"
            : "border-gov-border/50 text-gov-dim/50 hover:border-gov-dim hover:text-gov-dim"
        }`}
        onClick={handleImportClick}
      >
        <Upload size={16} className="mb-1" />
        <div className="text-[10px] text-center">
          {dragOver ? "Drop CSV here" : "Drop CSV or click to import"}
        </div>
      </div>
    </div>
  );
}
