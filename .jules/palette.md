## Palette's Journal


## 2024-05-16 - Prevent Ghost Interactions During Offline States
**Learning:** In components connected to real-time streams (like useApeiron), users can attempt interactions when the system is offline if inputs aren't explicitly disabled, leading to ghost interactions and silent failures.
**Action:** Always link user input and button disabled states to connection status (e.g., tunnelStatus) and provide clear visual/textual feedback (like dynamic placeholders and disabled styles) to explain why the element is disabled.
