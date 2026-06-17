
## 2024-05-22 - Memoizing Chat Interface
**Learning:** During chat stream rendering, appending chunks to a message caused all messages in the `messages` array to re-render. Since `useApeiron` constantly updates the `messages` state during a response stream, this resulted in O(N) rendering operations per chunk, leading to noticeable UI sluggishness.
**Action:** Extract list items into a `MessageItem` component wrapped in `React.memo()`. This prevents unchanged messages from re-rendering during active streams, reducing the render complexity for new chunks from O(N) to O(1).
