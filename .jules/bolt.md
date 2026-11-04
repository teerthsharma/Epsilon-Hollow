# Bolt's Journal

## 2024-11-04 - Initial Setup
**Learning:** Initializing the journal for the Bolt persona.
**Action:** Always read this file before starting and append critical learnings.

## 2024-11-04 - Optimize React Append-Only Lists
**Learning:** In React, appending items to a list managed by state and mapped in the parent component causes O(N) re-renders across all items on every addition, executing string searches or logic on previously rendered items.
**Action:** Always extract items in append-only lists (like logs) into a `React.memo` component to achieve O(1) rendering for new additions, skipping unnecessary checks on existing elements.
