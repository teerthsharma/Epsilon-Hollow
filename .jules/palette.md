## 2024-06-01 - Dynamic Tooltips for Disabled States
**Learning:** Found that static tooltips like "Send message" on disabled submit buttons can be confusing, particularly when users don't know *why* the button is disabled. In React/Next.js, toggling the `title` attribute dynamically based on the same condition that disables the button (`!input.trim() ? "Enter a message to send" : "Send message"`) provides immediate contextual feedback for screen reader and mouse users alike.
**Action:** When creating form submit buttons, always tie the `title` or `aria-label` attribute dynamically to the validation state, and ensure visual disabled styling (`disabled:cursor-not-allowed`) is coupled with `focus-visible` styles for comprehensive accessibility.
## 2024-06-01 - Prevent ghost interactions during offline states
**Learning:** Found that leaving input fields active while the system is offline leads to ghost interactions. In React/Next.js applications, inputs and buttons should be disabled based on network connection state, providing clear dynamic feedback (like "System offline. Reconnecting...") via placeholder and aria-label attributes for accessibility.
**Action:** When implementing chat interfaces that rely on external connections, always tie input and submit button disabled states to the connection status (e.g. `tunnelStatus !== 'LOCKED'`). Also provide visual disabled styling (`disabled:opacity-50 disabled:cursor-not-allowed`).

## 2026-07-11 - Dynamic Empty States in Connection-Dependent UIs
**Learning:** In chat interfaces tied to real-time connections (like the Sanctuary DSP Bus), a statically blank message list fails to convey system readiness. Users cannot distinguish between "ready but empty" and "still connecting/offline".
**Action:** Always bind the empty state UI directly to the underlying connection variables (e.g., `tunnelStatus`) to clearly articulate "Establishing Link" vs "System Ready", eliminating ambiguity.

## 2024-07-12 - Independent Scroll Refs for Multiple Containers
**Learning:** Found that attaching a single React `useRef` (e.g., `scrollRef`) to multiple distinct scrolling containers (like a thought stream and a chat stream) results in the ref only binding to the last rendered element in the DOM. This breaks auto-scrolling for the other containers and can cause unpredictable scroll jumping.
**Action:** Always create and assign completely independent `useRef` instances (e.g., `chatScrollRef` and `thoughtScrollRef`) when implementing auto-scroll behavior for multiple distinct UI areas.
