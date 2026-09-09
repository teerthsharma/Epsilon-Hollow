

## 2024-10-24 - Full App Re-render Bottleneck via Zustand
**Learning:** Destructuring `useStore` in `App.tsx` (the root component) subscribes it to the entire store. Any state change (like frequent log updates) triggers a full application re-render, creating a major performance bottleneck specific to this React architecture.
**Action:** Always use individual Zustand selectors (e.g., `useStore(s => s.state)`) in root components to prevent catastrophic re-render cascades.
