## 2024-06-10 - [React Stream Rendering Bottleneck]
**Learning:** [When rendering continuously streaming text (like DSP bus payloads that update a state array rapidly), mapping inline JSX blocks inside the render function causes an O(N^2) bottleneck. Every new chunk triggers a re-render of the entire list instead of just the latest updated item, tanking the FPS.]
**Action:** [Always extract list items that receive frequent state updates into separate components and wrap them in React.memo() to ensure only the actively updating elements are re-rendered.]
