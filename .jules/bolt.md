
## 2025-02-12 - Optimized ConsolePanel logging
**Learning:** Mapping append-only lists directly inside the parent causes O(N) re-renders across the entire list for every new item added.
**Action:** Extract log entry into a `React.memo` component (`LogEntry`) to achieve O(1) rendering for new log additions and reduce unnecessary diffing/re-renders for the existing logs.
