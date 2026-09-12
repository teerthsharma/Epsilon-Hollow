
## 2024-10-24 - Interactive utility buttons accessibility
**Learning:** In the `laamba-governor` interface, icon-only utility buttons lack labels and keyboard navigation support by default. Destructive actions need appropriate color coding (error instead of accent).
**Action:** Always add explicit `title` for mouse users, `aria-label` for screen readers, and clear `focus-visible` styling (`focus-visible:ring-2 focus-visible:outline-none`). For destructive/close actions, use `gov-error` instead of `gov-accent`.
