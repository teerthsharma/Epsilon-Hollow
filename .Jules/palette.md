## 2024-08-16 - Dynamic Logging Accessibility and Empty States
**Learning:** Visual chat and logging interfaces lack context for screen readers when new content appears. In addition, an empty container with no feedback feels broken or stuck.
**Action:** Always add `role="log"` and `aria-live="polite"` to dynamically updating telemetry or chat containers. Provide an explicit empty state string (like "Console is empty") when the log array is empty so users have clear confirmation of readiness.
