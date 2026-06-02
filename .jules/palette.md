## 2024-06-02 - [Button Disabled State Clarification]
**Learning:** While "disabled" is good, explaining *why* it's disabled via dynamic `aria-label` and `title` attributes provides critical context for screen readers and mouse users alike, especially in chat interfaces where the requirement (typing text) is implicit.
**Action:** When creating form submissions or interactive buttons, pair disabled states with dynamic aria-labels ("Cannot send empty message") and visual cues like `disabled:cursor-not-allowed` to ensure comprehensive accessibility.
