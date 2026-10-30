## 2024-05-18 - Optimized React append-only logging lists
**Learning:** In `ConsolePanel.tsx`, rendering an array of logs using `.map()` on primitive strings directly within the parent component causes the entire list to re-render for every single log addition, which is an O(N) operation per addition. Given that logging is frequent, this leads to significant main thread blocking.
**Action:** Extract list items into a `React.memo` component (e.g. `LogEntry`) to achieve O(1) rendering for new additions while skipping reconciliations for the existing entries.
