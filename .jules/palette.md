## 2024-06-11 - Dynamic Accessibility States for Disabled Buttons
**Learning:** Disabled icon buttons without dynamic titles/aria-labels leave screen reader users and sighted users confused about why an action is unavailable. Simple `disabled={true}` is insufficient.
**Action:** Always pair `disabled` state with dynamic `aria-label` and `title` explaining the condition (e.g., "Cannot send empty message"), and reinforce visually with `disabled:cursor-not-allowed` and explicit `focus-visible` ring styles.
