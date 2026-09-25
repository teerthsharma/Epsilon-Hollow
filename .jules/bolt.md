
## 2024-10-18 - Zustand Selector Optimization
**Learning:** Destructuring directly from `useStore` in React components subscribes the component to the entire store, causing unnecessary re-renders on ANY state change.
**Action:** Always extract individual selectors using `useStore((s) => s.property)` to limit re-renders to only when that specific property changes.
