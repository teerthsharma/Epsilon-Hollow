## 2024-06-17 - Connection State Feedback
**Learning:** In applications with intermittent or required backend connections (like the DSP Bus), failing to disable inputs during an 'OFFLINE' state can lead to user confusion when messages fail to send or disappear.
**Action:** Always link input/button `disabled` states to the connection/tunnel status, and update the placeholder and `title` to provide clear, actionable feedback (e.g., "Reconnecting to kernel...") to prevent ghost interactions.

## 2024-06-17 - Connection State Feedback
**Learning:** In applications with intermittent or required backend connections (like the DSP Bus), failing to disable inputs during an 'OFFLINE' state can lead to user confusion when messages fail to send or disappear.
**Action:** Always link input/button `disabled` states to the connection/tunnel status, and update the placeholder and `title` to provide clear, actionable feedback (e.g., "Reconnecting to kernel...") to prevent ghost interactions.
