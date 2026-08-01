## 2024-10-24 - Screen Reader Announcements for Transient Visual States
**Learning:** Visual-only transient states (like a temporary checkmark icon on a copy button after clicking) are invisible to screen readers unless explicitly announced. Simply changing the `aria-label` or relying on visual changes is insufficient for accessibility.
**Action:** Always add an `aria-live="polite"` visually hidden region (e.g., `<span className="sr-only">`) inside icon-only buttons that trigger transient success states to ensure assistive technologies reliably announce the confirmation to the user.
