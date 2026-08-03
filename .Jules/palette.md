## 2024-08-03 - Dynamic Empty States in Consoles
**Learning:** Empty logs or telemetry consoles shouldn't just be a blank screen. The absence of data can look like a broken app rather than a waiting state.
**Action:** Always implement a dynamic empty state (`logs.length === 0 ? "Awaiting telemetry..." : ...`) to clearly communicate that the system is functioning but has no data yet.
