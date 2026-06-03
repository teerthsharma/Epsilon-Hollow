## 2024-06-03 - [Clarifying Disabled States]
**Learning:** Found a reusable UX pattern for this design system: Interactive elements (like send buttons) lack clear context when disabled and miss robust focus visibility for keyboard navigation.
**Action:** Always apply dynamic `title` attributes explaining the disabled reason, paired with `disabled:cursor-not-allowed` and `focus-visible:ring` styles.
