
## 2024-05-15 - Semantic Focus States for Destructive Actions
**Learning:** In LAAMBA Governor, icon-only utility buttons for destructive actions (like clearing logs) were incorrectly using the default `gov-accent` for hover states. This fails to communicate danger visually and lacks keyboard accessibility.
**Action:** Always substitute `gov-accent` with `gov-error` for both hover and `focus-visible` styling on destructive or reset actions to provide clear, accessible semantic feedback.
