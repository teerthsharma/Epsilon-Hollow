

## 2024-11-09 - Interactive Icon Focus Pattern
**Learning:** Found an accessibility issue pattern in laamba-governor where utility icon buttons in panel headers lack keyboard focus rings and ARIA labels. Destructive/close actions (like Console clear or Formula close) were using generic accent hover colors instead of explicit error colors.
**Action:** When adding utility icon buttons to panel headers, always include `aria-label`, `title`, and explicitly map `focus-visible:ring-*` colors to match their contextual action type (e.g., `gov-error` for clear/close actions).
