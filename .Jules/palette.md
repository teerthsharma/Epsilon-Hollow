
## 2024-08-05 - Dynamic Console Logs Accessibility
**Learning:** Dynamically updating UI streams (like consoles or telemetry logs) are invisible to screen readers unless explicitly marked.
**Action:** Always add `role="log"` and `aria-live="polite"` to dynamic text areas so new content is announced smoothly.
