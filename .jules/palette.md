

## 2024-05-18 - Icon-Only Utility Buttons Accessibility
**Learning:** Utility buttons using icon libraries (like `lucide-react`) are completely invisible to screen readers without ARIA labels, and lack keyboard navigability without explicit focus states.
**Action:** Always add `aria-label`, `title` (for mouse hover), and `focus-visible:ring-*` classes when implementing icon-only buttons.
