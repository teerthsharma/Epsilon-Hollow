
## 2024-10-24 - Accessible Icon-Only Buttons
**Learning:** In the laamba-governor UI, icon-only utility buttons require explicit `aria-label` for screen readers, `title` for mouse hover context, and Tailwind `focus-visible` classes (like `focus-visible:ring-gov-error` for destructive actions) to support keyboard navigation.
**Action:** Always verify `aria-label`, `title`, and `focus-visible` styling exist on buttons that contain only icons.
