import { useEffect, useRef } from "react";
import { Terminal, Trash2 } from "lucide-react";
import { useStore } from "../store";

export default function ConsolePanel() {
  const { logs, clearLogs } = useStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="h-32 bg-gov-panel border-t border-gov-border flex flex-col shrink-0">
      <div className="h-6 flex items-center px-3 gap-2 text-[10px] uppercase tracking-wider text-gov-dim border-b border-gov-border">
        <Terminal size={10} /> Console
        <span className="text-[9px] text-gov-dim/50">{logs.length} lines</span>
        <div className="flex-1" />
        <button
          onClick={clearLogs}
          aria-label="Clear logs"
          title={logs.length === 0 ? "No logs to clear" : "Clear logs"}
          disabled={logs.length === 0}
          className="hover:text-gov-accent disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-gov-accent rounded"
        >
          <Trash2 size={10} />
        </button>
      </div>
      <div ref={scrollRef} role="log" aria-live="polite" className="flex-1 overflow-auto p-2 font-mono text-[11px] leading-relaxed">
        {logs.length === 0 ? (
          <div className="text-gov-dim/50 italic p-2">System ready. Awaiting pipeline execution...</div>
        ) : (
          logs.map((log, i) => (
          <div
            key={i}
            className={
              log.includes("failed") || log.includes("error")
                ? "text-gov-error"
                : log.includes("complete") || log.includes("winner")
                ? "text-gov-ok"
                : "text-gov-accent/80"
            }
          >
            {log}
          </div>
        ))
        )}
      </div>
    </div>
  );
}
