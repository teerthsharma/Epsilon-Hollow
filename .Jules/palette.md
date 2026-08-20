
## 2024-10-24 - Missing Accessibility on Icon-Only Panel Utility Buttons
**Learning:** Utility buttons used within panel headers (e.g., Console, Sample Bay) frequently use raw Lucide icons without accessible names or keyboard focus indicators, rendering them invisible to screen readers and difficult to use via keyboard.
**Action:** Always apply `title` (for mouse hover), `aria-label` (for screen readers), and Tailwind focus rings (`focus-visible:ring-2 focus-visible:ring-gov-accent focus-visible:outline-none rounded`) to all icon-only utility buttons in panel headers.
