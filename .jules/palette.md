## 2024-06-01 - Dynamic Tooltips for Disabled States
**Learning:** Found that static tooltips like "Send message" on disabled submit buttons can be confusing, particularly when users don't know *why* the button is disabled. In React/Next.js, toggling the `title` attribute dynamically based on the same condition that disables the button (`!input.trim() ? "Enter a message to send" : "Send message"`) provides immediate contextual feedback for screen reader and mouse users alike.
**Action:** When creating form submit buttons, always tie the `title` or `aria-label` attribute dynamically to the validation state, and ensure visual disabled styling (`disabled:cursor-not-allowed`) is coupled with `focus-visible` styles for comprehensive accessibility.

## 2024-10-27 - Input Disabled States Linked to Status
**Learning:** Users can easily perform "ghost interactions" when network connections are lost. Disabling form inputs and using dynamic titles/aria-labels based on network status (like `tunnelStatus`) provides immediate, vital accessibility context about *why* the form is unresponsive.
**Action:** Always link primary interactive elements (inputs, buttons) to backend connection status when available, applying `disabled:cursor-not-allowed` and dynamic labels to explain the offline state.
