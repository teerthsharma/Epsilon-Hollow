
## 2024-05-24 - Missing Accessibility on Icon-Only Buttons
**Learning:** The application extensively uses `lucide-react` icon-only buttons without `aria-label`, `title`, or `focus-visible` indicators, rendering them completely inaccessible to screen readers and difficult to use via keyboard navigation.
**Action:** Add `aria-label` for screen readers, `title` for mouse hover context, and Tailwind `focus-visible` ring utilities (like `focus-visible:ring-2 focus-visible:ring-gov-error focus-visible:outline-none rounded`) to all interactive icon buttons to meet basic accessibility standards.
