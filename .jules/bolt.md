
## 2024-10-24 - Avoid destructuring Zustand store in Root Components
**Learning:** Using `const { a, b } = useStore()` in root components like App.tsx is a major codebase-specific anti-pattern here because it subscribes the entire React component tree to all state changes (including high-frequency telemetry streams), causing unnecessary layout recalculations and full app re-renders.
**Action:** Always use individual selectors like `const a = useStore(s => s.a)` or `useShallow` for Zustand stores, especially in high-level components.
