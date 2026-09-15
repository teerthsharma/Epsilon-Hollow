
## 2024-05-24 - Zustand Root Re-renders
**Learning:** Destructuring from `useStore()` in the root component subscribes it to the entire state object, causing the entire application tree to re-render on any unrelated state update (e.g. log additions).
**Action:** Always use individual selectors (e.g., `useStore(s => s.action)`) for Zustand stores, especially in high-level root components to restrict re-renders.
