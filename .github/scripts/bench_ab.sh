#!/usr/bin/env bash
# Run each benchmark suite against two source trees, writing one JSON
# file per (side, suite, round) for tools/bench_regression.py.
#
# Three properties this script exists to guarantee:
#
#   * A discarded warmup pass per side. A freshly built binary runs slow
#     for its first few executions on macOS (code-signing page
#     validation) and converges downward; measuring that first pass would
#     make whichever side built last look slower.
#   * Paired rounds. Base and head run back to back within a round, so
#     interference that lasts longer than a pass moves both sides
#     together and cancels when the comparison divides them.
#   * Alternating order (base,head then head,base then base,head ...).
#     Always running the same side first turns any drift across the job
#     into a systematic bias against the side that always runs second.
set -euo pipefail

BASE_TREE="${BASE_TREE:?base checkout path}"
HEAD_TREE="${HEAD_TREE:?head checkout path}"
RUNS="${RUNS:?output directory}"
ROUNDS="${ROUNDS:-5}"
SUITES="${SUITES:-bench bench_set}"

mkdir -p "$RUNS"

run_one() {
  local side="$1" tree="$2" suite="$3" label="$4"
  local raw="$RUNS/$side.$suite.$label.txt"

  echo "::group::$side / $suite / pass $label"
  # Absolute paths for both -I and the source file: `pixi run` does not
  # guarantee a working directory, and the base tree must resolve
  # `emberregex` from its own checkout rather than from head's.
  if ! pixi run mojo -I "$tree" "$tree/bench/$suite.mojo" >"$raw" 2>&1; then
    echo "::endgroup::"
    echo "::error::$side/$suite pass $label failed to build or run"
    cat "$raw"
    return 1
  fi
  tail -n 5 "$raw"
  echo "::endgroup::"

  if [ "$label" != "warmup" ]; then
    # Always the head tree's copy of the script: the base revision
    # predates it on the pull request that introduces it.
    python3 "$HEAD_TREE/tools/bench_regression.py" parse \
      --input "$raw" --suite "$suite" --round "$label" \
      --out "$RUNS/$side.$suite.$label.json"
  fi
}

for suite in $SUITES; do
  if [ ! -f "$BASE_TREE/bench/$suite.mojo" ] ||
     [ ! -f "$HEAD_TREE/bench/$suite.mojo" ]; then
    echo "::notice::skipping suite '$suite' — absent on one side"
    continue
  fi

  run_one base "$BASE_TREE" "$suite" warmup
  run_one head "$HEAD_TREE" "$suite" warmup

  for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then
      run_one base "$BASE_TREE" "$suite" "$r"
      run_one head "$HEAD_TREE" "$suite" "$r"
    else
      run_one head "$HEAD_TREE" "$suite" "$r"
      run_one base "$BASE_TREE" "$suite" "$r"
    fi
  done
done
