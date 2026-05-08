#!/usr/bin/env sh
# Fail if any tracked text file under kernel/, infrastructure/, or scripts/
# contains a UTF-8 BOM (0xEF 0xBB 0xBF) anywhere in the file.
#
# Exits 1 on first-found BOM (after scanning all files), 0 otherwise.

set -eu

ROOTS="kernel infrastructure scripts"
FOUND=0

# Use git to enumerate tracked files only, restricted to text-likely extensions.
# Fall back to find if not in a git repo.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  FILES=$(git ls-files -- $ROOTS 2>/dev/null || true)
else
  FILES=$(find $ROOTS -type f 2>/dev/null || true)
fi

[ -z "$FILES" ] && exit 0

# Iterate over each file, search for the BOM byte sequence anywhere.
# grep -l -P with binary-safe pattern; portable fallback uses perl.
for f in $FILES; do
  [ -f "$f" ] || continue
  # Skip obvious binaries by extension.
  case "$f" in
    *.png|*.jpg|*.jpeg|*.gif|*.ico|*.pdf|*.zip|*.gz|*.tar|*.bin|*.exe|*.dll|*.so|*.dylib|*.wasm|*.o|*.a|*.rlib)
      continue ;;
  esac
  # Find byte offset of first BOM occurrence, if any.
  offset=$(perl -e '
    local $/;
    open(my $fh, "<:raw", $ARGV[0]) or exit 0;
    my $data = <$fh>;
    close $fh;
    my $idx = index($data, "\xEF\xBB\xBF");
    print $idx if $idx >= 0;
  ' "$f")
  if [ -n "$offset" ]; then
    echo "BOM found: $f (byte offset $offset)" >&2
    FOUND=1
  fi
done

if [ "$FOUND" -ne 0 ]; then
  echo "ERROR: UTF-8 BOM detected in tracked files. Strip BOMs and recommit." >&2
  exit 1
fi

exit 0
