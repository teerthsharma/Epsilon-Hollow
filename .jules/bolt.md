
## 2024-09-08 - Zustand Store Destructuring
**Learning:** Destructuring directly from `useStore()` (e.g. `const { a, b } = useStore()`) subscribes the component to the entire store, causing unnecessary re-renders on every single state change across the app.
**Action:** Always extract Zustand selectors individually (e.g. `const a = useStore(s => s.a)`) or use `useShallow` to prevent excessive re-renders, especially in root or high-level components.
