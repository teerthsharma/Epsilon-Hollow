
## 2024-10-09 - O(N) List Rendering in Append-Only React Lists
**Learning:** In append-only lists like `ConsolePanel` logs, mapping items directly in the parent component forces O(N) re-renders across the entire list for every single new addition, severely degrading performance during high-frequency log events.
**Action:** Extract list items into a `React.memo()` wrapped component (e.g., `LogEntry`) to achieve O(1) rendering for new additions, effectively skipping the re-evaluation of existing items.
