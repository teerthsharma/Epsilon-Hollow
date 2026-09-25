
## 2025-02-14 - Zustand Re-renders in Top-Level Component
**Learning:** Destructuring `useStore()` in the top-level component subscribes it to the entire Zustand store, causing unnecessary re-renders of the whole app tree on any state change.
**Action:** Always use individual Zustand selectors (e.g., `useStore(s => s.action)`) instead of destructuring from the full store.
