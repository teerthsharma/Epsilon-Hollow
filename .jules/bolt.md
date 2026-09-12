
## 2024-10-18 - Full App Re-renders from Zustand Destructuring
**Learning:** Destructuring `useStore` at the root `App` component level (e.g., `const { setDatasets, addLog } = useStore();`) subscribes the component to ALL state changes, causing the entire React tree to unnecessarily re-render on every high-frequency log update.
**Action:** Always extract individual selectors (e.g., `const addLog = useStore((s) => s.addLog);`) or use `useShallow` for state that changes independently, especially at the root level.
