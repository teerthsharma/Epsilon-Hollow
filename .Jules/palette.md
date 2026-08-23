
## 2024-11-23 - Accessible Destructive Icon Buttons
**Learning:** The `laamba-governor` app standardizes on using `gov-error` instead of `gov-accent` for standard keyboard focus indicators on destructive actions (like clearing the console).
**Action:** Always add `focus-visible:ring-2 focus-visible:ring-gov-error focus-visible:outline-none rounded`, along with `title` and `aria-label`, when improving a11y for destructive icon-only utility buttons.
