## 2024-06-13 - Dynamic Disabled Button ARIA/Titles
**Learning:** For interactive chat components, users with screen readers or who use hover often miss why a send button is disabled if it only uses static "Send" labels.
**Action:** Pair dynamic `aria-label` and `title` attributes that explicitly explain the disabled state (e.g., "Type a message to send") with `disabled:cursor-not-allowed` and keyboard focus rings (`focus-visible:ring-2`) on form inputs for better comprehensive accessibility.
