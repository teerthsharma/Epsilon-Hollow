
## 2024-08-14 - Icon-only Button Accessibility
**Learning:** Icon-only utility buttons (like the clear console trash icon) lacked proper screen reader announcements and semantic focus states, making them invisible to keyboard and assistive tech users.
**Action:** Always add explicit `title` (for mouse hover) and `aria-label` (for screen readers) alongside `focus-visible:ring-*` classes on icon-only buttons to ensure universal accessibility.
