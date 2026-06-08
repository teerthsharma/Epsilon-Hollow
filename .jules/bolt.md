## 2024-03-24 - Streaming React Arrays
**Learning:** When rendering streamed text arrays (like Chat messages receiving data chunk by chunk), updating a state array triggers full re-renders of the list. With hundreds of items, this creates an O(N^2) rendering bottleneck that can freeze the main thread.
**Action:** Always wrap individual list items in `React.memo()` when dealing with arrays that receive rapid, incremental updates from streams to prevent unnecessary full-list reconciliation.
