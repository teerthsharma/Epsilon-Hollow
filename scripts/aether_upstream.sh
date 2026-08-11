#!/usr/bin/env bash
# Report what changed in upstream Aether-Lang since the last reviewed commit.
# kernel/aether/Aether-Lang is a fork with no mergeable base -- see kernel/aether/UPSTREAM.
# This never touches the fork; porting is manual.
set -euo pipefail
cd "$(dirname "$0")/.."

UP=kernel/aether/UPSTREAM
val() { awk -F'[ ]*=[ ]*' "/^$1 /{print \$2}" "$UP"; }
URL=$(val url); BRANCH=$(val branch); BASE=$(val reviewed)
CACHE=${AETHER_UPSTREAM_CACHE:-target/aether-upstream}

[ -d "$CACHE" ] || git clone -q --bare --branch "$BRANCH" "$URL" "$CACHE"
git -C "$CACHE" fetch -q origin "+$BRANCH:$BRANCH"
HEAD_SHA=$(git -C "$CACHE" rev-parse "$BRANCH")

case "${1:-check}" in
  check)
    echo "upstream  $URL ($BRANCH)"
    echo "reviewed  $BASE"
    echo "head      $HEAD_SHA"
    if [ "$BASE" = "$HEAD_SHA" ]; then echo; echo "up to date"; exit 0; fi
    echo; git -C "$CACHE" log --oneline --no-decorate "$BASE..$BRANCH"
    echo; git -C "$CACHE" diff --stat "$BASE..$BRANCH"
    ;;
  diff) shift; git -C "$CACHE" diff "$BASE..$BRANCH" -- "$@" ;;
  show) git -C "$CACHE" show "${2:?usage: $0 show <sha>}" ;;
  bump)
    [ "$BASE" = "$HEAD_SHA" ] && { echo "already at $HEAD_SHA"; exit 0; }
    sed -i "s/^reviewed = .*/reviewed = $HEAD_SHA/" "$UP"
    echo "reviewed: $BASE -> $HEAD_SHA"
    ;;
  *) echo "usage: $0 [check | diff [path...] | show <sha> | bump]" >&2; exit 2 ;;
esac
