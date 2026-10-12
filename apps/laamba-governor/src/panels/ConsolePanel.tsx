import { useEffect, useRef, memo } from "react";
import { Terminal, Trash2 } from "lucide-react";
import { usePick } from "../store";

// ⚡ Bolt: Extract list item into a memoized component to achieve O(1) rendering for append-only list additions
const LogEntry = memo(({ log }: { log: string }) => (
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
));
LogEntry.displayName = "LogEntry";

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
