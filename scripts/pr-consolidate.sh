#!/usr/bin/env bash
# Consolidate duplicate bot PRs: one merge per cluster, close the rest as superseded.
#
#   scripts/pr-consolidate.sh            dry run, prints the plan
#   scripts/pr-consolidate.sh --apply    merges and closes
#
# A PR is eligible only if every file it changes lives under a frontend path in
# SAFE_PATHS. Old bot branches share no recent history with main and carry the
# untracked agent scaffolding back in; this guard refuses them.
set -euo pipefail

REPO=teerthsharma/Epsilon-Hollow
APPLY=${1:-}
SAFE_PATHS='^(apps/laamba-governor/|future/apeiron-runtime/APEIRON/frontend/|\.jules/)'

cluster() {
  local t=${1,,}
  case $t in
    *liquidstream*)                                  echo liquidstream ;;
    *zustand*|*useshallow*|*selector*)               echo zustand ;;
    *console*|*"clear logs"*)                        echo console ;;
    *copy*)                                          echo copy-button ;;
    *"layout thrash"*|*auto-scroll*)                 echo layout ;;
    *offline*|*ghost*|*neuro-link*|*disabled*)       echo offline-state ;;
    *"empty state"*)                                 echo empty-state ;;
    *"screen reader"*|*sr-only*|*authorship*)        echo sr-authorship ;;
    *icon*)                                          echo icon-buttons ;;
    *matmul*)                                        echo matmul ;;
    *chat*|*memo*|*message*|*keyboard*|*button*)     echo chat-misc ;;
    *)                                               echo other ;;
  esac
}

mapfile -t PRS < <(gh api "repos/$REPO/pulls?state=open&per_page=100" --paginate \
  --jq '.[] | select(.draft|not) | "\(.number)\t\(.title)"' | sort -rn)

# GitHub reports null while it recomputes mergeability, which it does after every merge.
mergeable() {
  local m
  for _ in 1 2 3 4 5; do
    m=$(gh api "repos/$REPO/pulls/$1" --jq .mergeable)
    [[ $m != null ]] && { echo "$m"; return; }
    sleep 3
  done
  echo false
}

declare -A KEEP
printf '%-6s %-15s %-8s %s\n' PR CLUSTER ACTION TITLE
for row in "${PRS[@]}"; do
  n=${row%%$'\t'*}; title=${row#*$'\t'}; c=$(cluster "$title")
  outside=$(gh api "repos/$REPO/pulls/$n/files" --paginate --jq '.[].filename' | grep -Evc "$SAFE_PATHS" || true)
  if [[ $c == other || $outside -gt 0 ]]; then
    action="SKIP(out=$outside)"
  elif [[ -z ${KEEP[$c]:-} && $(mergeable "$n") == true ]]; then
    KEEP[$c]=$n; action=MERGE
    [[ $APPLY == --apply ]] && gh pr merge "$n" -R "$REPO" --squash --delete-branch
  elif [[ -n ${KEEP[$c]:-} ]]; then
    action="CLOSE>#${KEEP[$c]}"
    [[ $APPLY == --apply ]] && gh pr close "$n" -R "$REPO" --delete-branch \
      -c "Superseded by #${KEEP[$c]}, which lands the same change; closed by scripts/pr-consolidate.sh."
  else
    action="CONFLICT"   # newest in cluster not mergeable yet; an older one may still be picked
  fi
  printf '%-6s %-15s %-8s %s\n' "$n" "$c" "$action" "${title:0:60}"
done
