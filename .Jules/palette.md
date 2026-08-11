
## 2024-07-31 - Explicit ARIA Live Feedback for Transient States
**Learning:** Visual-only transient states (like a temporary checkmark on a copy button) are invisible to screen readers unless explicitly announced. Just changing the aria-label isn't enough because the focus might not be announced dynamically upon change.
**Action:** Always add an `aria-live="polite"` visually hidden region (like `<span className="sr-only">`) to icon-only buttons that trigger transient success states (e.g., "Copied!") so assistive technologies announce the confirmation immediately.
