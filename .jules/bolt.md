
## 2024-05-24 - Zustand Full-Store Subscriptions
**Learning:** The entire application uses `const { ... } = useStore()` which subscribes components to the full store. In high-frequency update scenarios, this causes the root `App` component to re-render, thrashing the entire React tree.
**Action:** Always use individual selectors (`useStore(s => s.x)`) when accessing the Zustand store in root components to prevent catastrophic re-renders.
