import React, { useEffect, useRef } from "react";
import { Terminal, Trash2 } from "lucide-react";
import { usePick } from "../store";

// ⚡ Bolt: Extract list item into a React.memo component to achieve O(1) rendering for new additions and prevent O(N) re-renders across the entire list.
const LogEntry = React.memo(({ log }: { log: string }) => {
  return (
    <div
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
  );
});

export default function ConsolePanel() {
  const { logs, clearLogs } = usePick("logs", "clearLogs");
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
          title="Clear console logs"
          aria-label="Clear console logs"
          className="hover:text-gov-error focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gov-error rounded"
        >
          <Trash2 size={10} />
        </button>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-auto p-2 font-mono text-[11px] leading-relaxed">
        {logs.map((log, i) => (
          <LogEntry key={i} log={log} />
        ))}
      </div>
    </div>
  );
}
