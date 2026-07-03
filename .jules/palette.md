## 2024-06-01 - Dynamic Tooltips for Disabled States
**Learning:** Found that static tooltips like "Send message" on disabled submit buttons can be confusing, particularly when users don't know *why* the button is disabled. In React/Next.js, toggling the `title` attribute dynamically based on the same condition that disables the button (`!input.trim() ? "Enter a message to send" : "Send message"`) provides immediate contextual feedback for screen reader and mouse users alike.
**Action:** When creating form submit buttons, always tie the `title` or `aria-label` attribute dynamically to the validation state, and ensure visual disabled styling (`disabled:cursor-not-allowed`) is coupled with `focus-visible` styles for comprehensive accessibility.

## 2024-07-03 - Contextual Disabled States for Streaming Disconnects
**Learning:** Found that when a streaming backend (like DSP Bus) disconnects, leaving the chat input active leads to ghost interactions where users try to send messages into the void. Dynamically disabling the input and button based on a `tunnelStatus` and explaining *why* (e.g., "System Offline - Reconnecting...") is critical for preventing user frustration.
**Action:** Always link form input disabled states to the connection tunnel status, and use dynamic placeholders/titles to explain offline states, preventing ghost interactions.
