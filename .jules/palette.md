## 2024-06-01 - Dynamic Tooltips for Disabled States
**Learning:** Found that static tooltips like "Send message" on disabled submit buttons can be confusing, particularly when users don't know *why* the button is disabled. In React/Next.js, toggling the `title` attribute dynamically based on the same condition that disables the button (`!input.trim() ? "Enter a message to send" : "Send message"`) provides immediate contextual feedback for screen reader and mouse users alike.
**Action:** When creating form submit buttons, always tie the `title` or `aria-label` attribute dynamically to the validation state, and ensure visual disabled styling (`disabled:cursor-not-allowed`) is coupled with `focus-visible` styles for comprehensive accessibility.
## 2024-06-01 - Prevent ghost interactions during offline states
**Learning:** Found that leaving input fields active while the system is offline leads to ghost interactions. In React/Next.js applications, inputs and buttons should be disabled based on network connection state, providing clear dynamic feedback (like "System offline. Reconnecting...") via placeholder and aria-label attributes for accessibility.
**Action:** When implementing chat interfaces that rely on external connections, always tie input and submit button disabled states to the connection status (e.g. `tunnelStatus !== 'LOCKED'`). Also provide visual disabled styling (`disabled:opacity-50 disabled:cursor-not-allowed`).

## 2024-07-08 - Dynamic Empty State
**Learning:** When creating a chat interface or data list, leaving the screen completely blank during the empty state is poor UX, especially if the user is waiting for a background connection to establish.
**Action:** Always provide a dynamic empty state that uses existing connection or readiness variables (e.g., `tunnelStatus`) to clearly communicate whether the system is loading, offline, or ready for input.
