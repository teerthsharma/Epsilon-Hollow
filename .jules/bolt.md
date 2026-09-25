
## 2024-11-20 - Prevent Full App Re-renders via Zustand Selectors
**Learning:** Destructuring `useStore()` in a root component (like `App.tsx`) inadvertently subscribes the entire application tree to all state updates, causing massive layout thrashing even for isolated state changes (e.g. telemetry).
**Action:** Always use individual Zustand selectors (e.g., `useStore(s => s.property)`) in root or high-level layout components to ensure only necessary properties trigger re-renders.
