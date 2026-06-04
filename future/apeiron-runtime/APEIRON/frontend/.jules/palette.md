## 2025-06-04 - Dynamic Title for Disabled Button
**Learning:** Adding a dynamic `title` to a disabled button provides necessary context for users with screen readers or those hovering, explaining *why* the button is disabled (e.g., 'Enter a message to send' vs 'Send message'). Paired with visual cues like `disabled:cursor-not-allowed` and keyboard accessibility styles (`focus-visible:ring-2`), it creates a much better holistic user experience.
**Action:** Always pair disabled states with a clear, explaining title/aria-label and visual cues like cursor-not-allowed.
