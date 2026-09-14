
## 2024-09-14 - Accessible Icon-only Utility Buttons
**Learning:** Icon-only utility buttons in UI panels lack text labels, making them inaccessible to screen readers and difficult to discover for mouse users, while missing focus outlines hinder keyboard navigation.
**Action:** Always add `aria-label` for screen readers, `title` for mouse users, and explicit `focus-visible:ring-*` styles to icon-only buttons to ensure full accessibility and discoverability.
