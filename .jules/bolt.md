
## 2024-05-24 - Zustand Full-Store Subscription Anti-Pattern
**Learning:** In the `laamba-governor` app, destructuring from the full Zustand store (e.g., `const { setDatasets, addLog } = useStore()`) in root components like `App.tsx` causes the entire component tree to re-render on *any* state change, such as high-frequency log updates. This is a severe performance bottleneck specific to this architecture where logs and telemetry stream frequently.
**Action:** Always use individual selectors (e.g., `const addLog = useStore(s => s.addLog)`) or `useShallow` when pulling from Zustand, especially in root or layout components, to prevent unnecessary re-renders of the entire app.
