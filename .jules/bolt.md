
## 2024-09-20 - Root Re-renders from Zustand Destructuring
**Learning:** Destructuring the entire Zustand store (e.g., `const { a, b } = useStore()`) in high-level components like App.tsx subscribes the component to every state change, causing the entire React tree to unnecessarily re-render on any unrelated state update (like rapidly changing telemetry logs).
**Action:** Always extract specific pieces of state using individual selectors (e.g., `const a = useStore(s => s.a)`) or `useShallow`, especially in root-level or high-frequency rendering components to maintain stable rendering performance.
