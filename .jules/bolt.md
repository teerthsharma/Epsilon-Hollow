## 2024-05-18 - [Render optimization for streamed list items]
**Learning:** React arrays updated on every streamed chunk (like DSP Bus chat messages) cause O(N^2) rendering bottlenecks.
**Action:** Wrap individual list items in React.memo() to prevent full-list re-renders on streaming updates.
