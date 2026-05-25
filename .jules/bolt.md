## 2024-05-24 - [O(N^2) Render Bottleneck in Chat Streaming]
**Learning:** During token-by-token streaming in Next.js/React, rendering the entire message history inline causes an O(N) render cost per token, resulting in noticeable UI lag as the context depth increases.
**Action:** Extract list items into separate components wrapped in `React.memo()` to ensure only the actively streaming message re-renders, reducing time complexity per token from O(N) to O(1).
