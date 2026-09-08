#!/usr/bin/env bash
# Post the benchmark report as a single pull-request comment, editing
# the previous one in place so repeated pushes do not stack comments.
set -euo pipefail

BODY_FILE="${1:?usage: sticky_comment.sh <markdown-file>}"
: "${GITHUB_REPOSITORY:?}" "${PR_NUMBER:?}"

# Must stay in sync with MARKER in tools/bench_regression.py.
MARKER='<!-- emberregex-bench-report -->'

filter="map(select(.body | startswith(\"$MARKER\"))) | .[0].id // empty"
existing="$(
  gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments" \
    --jq "$filter" | head -n 1
)"

payload="$(jq -n --rawfile body "$BODY_FILE" '{body: $body}')"

if [ -n "$existing" ]; then
  printf '%s' "$payload" | gh api -X PATCH --silent --input - \
    "repos/$GITHUB_REPOSITORY/issues/comments/$existing"
  echo "Updated existing comment $existing."
else
  printf '%s' "$payload" | gh api -X POST --silent --input - \
    "repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments"
  echo "Posted a new comment."
fi
