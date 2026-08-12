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

## 2025-01-01 - Copy Button Overlay Constraints
**Learning:** When adding absolute positioned interactive elements (like a Copy button) to message elements, it's critical to add sufficient padding (e.g., `pr-8`) to the main text container. Otherwise, long text lines will overlap and obscure the button, rendering it inaccessible and ugly.
**Action:** Always verify text flow and add appropriate padding when using `absolute` positioned interactive UI elements inside content containers.

## 2024-08-04 - Accessible Focus States for Toolbar Buttons
**Learning:** Found that icon-only buttons or primary action buttons in toolbars often lack clear keyboard focus indicators, making them difficult for keyboard navigators to use. Additionally, when these buttons are disabled, they simply show lower opacity but no `cursor-not-allowed` pointer, which can lead users to think their click simply didn't register.
**Action:** When implementing toolbar buttons, always include `focus-visible:ring-2 focus-visible:ring-gov-accent focus-visible:outline-none` to ensure keyboard navigation is clear and accessible. Furthermore, always pair `disabled:opacity-50` with `disabled:cursor-not-allowed` for unambiguous disabled visual feedback.

## 2024-08-12 - Live Regions for Log Containers
**Learning:** Found that dynamic log containers (like the ConsolePanel in laamba-governor) fail to announce new entries to screen reader users unless explicitly marked as a live region.
**Action:** When implementing dynamically updating side streams or text areas (e.g., console logs, telemetry), always add `role="log"` and `aria-live="polite"` to the scrollable container. This ensures assistive technologies correctly announce new content as it is added to the DOM without aggressively interrupting the user.
