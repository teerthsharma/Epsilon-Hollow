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

## 2026-07-28 - Dynamic Suggestion Chips for Empty States
**Learning:** In conversational UIs, users often experience "blank canvas paralysis" when faced with a completely empty chat, even if the system states it is "Ready". Providing quick-start suggestion chips significantly improves the initial UX. However, in connection-dependent apps, these chips must correctly bind their `disabled` state and dynamic accessibility attributes (`aria-label`, `title`) to the connection status (e.g. `tunnelStatus !== 'LOCKED'`) to prevent users from attempting actions before the system is actually ready.
**Action:** When implementing empty states in chat interfaces, always include clickable suggestion chips for common initial actions, ensuring their interactivity and accessibility properties are strictly governed by the underlying connection state.

## 2024-07-28 - Dynamic Disabled States During Processing
**Learning:** Adding a visible loading state (like a spinner) and explicitly disabling inputs during async processing (e.g., `isLearning`) prevents double submissions and provides immediate feedback. In Next.js/React applications, when replacing a static icon with a conditionally rendered one, it is critical to verify the icon is imported to prevent runtime `ReferenceError` crashes.
**Action:** When conditionally replacing components (like swapping `Send` for `Cpu`), always search the file's imports to ensure the new component is available before finalizing the change.
