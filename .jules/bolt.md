## 2024-06-13 - Memoizing Streaming Array Maps
**Learning:** Rendering streamed text arrays (like DSP Bus chat messages) causes an O(N^2) rendering bottleneck if the state array update triggers full-list re-renders on every streamed chunk.
**Action:** Always extract the individual message item into its own component and wrap it in `React.memo()` so only the *currently streaming* message re-renders, reducing re-renders significantly. Ensure the type (`Message`) is exported if necessary from hooks.
