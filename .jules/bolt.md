## 2024-07-21 - Extract Array Mappings to useMemo
**Learning:** Even if child components are memoized with `React.memo()`, a parent component mapping over a large array directly in its render function executes that O(N) mapping on every parent re-render. If the parent re-renders frequently due to unrelated state (e.g., streaming telemetry), this causes severe performance overhead.
**Action:** Extract the array mapping logic and wrap it in a `useMemo` hook at the top level of the component to prevent unnecessary overhead during unrelated state changes.

## 2024-07-21 - Extract Array Mappings to useMemo
**Learning:** Even if child components are memoized with `React.memo()`, a parent component mapping over a large array directly in its render function executes that O(N) mapping on every parent re-render. If the parent re-renders frequently due to unrelated state (e.g., streaming telemetry), this causes severe performance overhead.
**Action:** Extract the array mapping logic and wrap it in a `useMemo` hook at the top level of the component to prevent unnecessary overhead during unrelated state changes.
