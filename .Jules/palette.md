
## 2024-08-06 - Dynamic Updating Streams Accessibility
**Learning:** Telemetry and console streams are dynamically updating and visually apparent, but assistive technologies will remain silent unless explicitly informed. Changing the `aria-label` alone on a container is not sufficient.
**Action:** When creating a dynamic text container, such as a log stream, add `role="log"` and `aria-live="polite"` so screen readers can automatically and correctly announce the new content to the user as it streams in, without interrupting their current tasks.
