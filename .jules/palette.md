## 2024-06-01 - Dynamic Tooltips for Disabled States
**Learning:** Found that static tooltips like "Send message" on disabled submit buttons can be confusing, particularly when users don't know *why* the button is disabled. In React/Next.js, toggling the `title` attribute dynamically based on the same condition that disables the button (`!input.trim() ? "Enter a message to send" : "Send message"`) provides immediate contextual feedback for screen reader and mouse users alike.
**Action:** When creating form submit buttons, always tie the `title` or `aria-label` attribute dynamically to the validation state, and ensure visual disabled styling (`disabled:cursor-not-allowed`) is coupled with `focus-visible` styles for comprehensive accessibility.
## 2024-06-01 - Prevent ghost interactions during offline states
**Learning:** Found that leaving input fields active while the system is offline leads to ghost interactions. In React/Next.js applications, inputs and buttons should be disabled based on network connection state, providing clear dynamic feedback (like "System offline. Reconnecting...") via placeholder and aria-label attributes for accessibility.
**Action:** When implementing chat interfaces that rely on external connections, always tie input and submit button disabled states to the connection status (e.g. `tunnelStatus !== 'LOCKED'`). Also provide visual disabled styling (`disabled:opacity-50 disabled:cursor-not-allowed`).

## 2026-07-11 - Dynamic Empty States in Connection-Dependent UIs
**Learning:** In chat interfaces tied to real-time connections (like the Sanctuary DSP Bus), a statically blank message list fails to convey system readiness. Users cannot distinguish between "ready but empty" and "still connecting/offline".
**Action:** Always bind the empty state UI directly to the underlying connection variables (e.g., `tunnelStatus`) to clearly articulate "Establishing Link" vs "System Ready", eliminating ambiguity.

## 2026-07-13 - Independent Scrolling Refs and Dynamic ARIA Attributes for Thought Streams
**Learning:** Found that attaching the same `useRef` (e.g., `scrollRef`) to multiple independent scrolling DOM elements (like a chat history and a side thought stream) causes the ref to only point to the last rendered element. This breaks auto-scrolling for all but one container. Additionally, dynamic side streams (like the thought stream) must have appropriate ARIA attributes (`role="log"`, `aria-live="polite"`) for screen readers to announce new thoughts as they arrive.
**Action:** When implementing multiple independent scrolling areas (such as chat and telemetry streams), always create distinct `useRef` hooks for each container. Additionally, ensure all live-updating dynamic content areas have proper `role="log"` and `aria-live="polite"` attributes for accessibility.

## 2025-07-20 - Adding visually hidden screen-reader context to chat streams
**Learning:** In chat interfaces like `ChatInterface.tsx`, relying solely on flex alignment (e.g., `justify-start` vs `justify-end`) or background colors to differentiate the sender provides zero semantic context to screen readers, rendering the stream confusing.
**Action:** When mapping chat message components, always inject explicit `sr-only` text (like `<span className="sr-only">User:</span>`) alongside the visible content to strictly announce authorship for visually impaired users.
