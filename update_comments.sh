cat << 'INNER_EOF' > /tmp/merge_diff
<<<<<<< SEARCH
const LogEntry = memo(({ log }: { log: string }) => {
  return (
    <div
      className={
=======
// ⚡ Bolt Optimization:
// Extracts log entries into a memoized component to achieve O(1) rendering for new additions.
// Prevents React from needlessly re-evaluating string checks (e.g., \`log.includes\`)
// and reconciling the VDOM for the entire list every time a new log is added.
const LogEntry = memo(({ log }: { log: string }) => {
  return (
    <div
      className={
>>>>>>> REPLACE
INNER_EOF
