
## 2024-05-18 - Zustand Full State Destructuring
**Learning:** Full state destructuring (e.g., `const { x, y } = useStore()`) is an anti-pattern in Zustand because it subscribes the component to the entire store, causing unnecessary re-renders on ANY state change.
**Action:** Always use individual selectors (e.g., `const x = useStore(s => s.x)`) or `useShallow` when selecting state from Zustand stores.
