

## 2024-05-18 - Transient states in icon buttons
**Learning:** Visual-only transient states (like a temporary checkmark) are invisible to screen readers. Simply changing the aria-label is insufficient for assistive tech to announce the change reliably.
**Action:** Always add an aria-live="polite" visually hidden region inside icon-only buttons that trigger transient success states to ensure reliable announcements.
