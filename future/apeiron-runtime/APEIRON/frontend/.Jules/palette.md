

## 2024-08-02 - Accessible Transient Success States
**Learning:** Visual-only transient states (like a temporary checkmark replacing a copy icon) are completely invisible to screen readers unless explicitly announced. Merely changing an icon or even the aria-label dynamically may not trigger a read-out.
**Action:** Always add an `aria-live="polite"` visually hidden region (e.g., `<span className="sr-only">`) inside buttons that trigger transient success states to ensure assistive technologies reliably announce the confirmation (e.g., "Copied to clipboard").
