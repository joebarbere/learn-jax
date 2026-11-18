#!/usr/bin/env bash
# Render every lesson with the reference solutions into notebooks (outputs included) for the book.
#   tools/book/render.sh light|dark [module-glob]        (make site runs both)
# GPU work: each lesson runs under a lock so it never overlaps another GPU job. Writes site/build/<theme>/*.ipynb.
set -uo pipefail
cd "$(dirname "$0")/../.."
theme=${1:?light or dark}; glob=${2:-*}
out=site/build/$theme; mkdir -p "$out"
lock=${GPU_LOCK:-/tmp/learn-jax-gpu.lock}
for f in modules/$glob/lesson.py; do
  m=$(basename "$(dirname "$f")")
  # up to date unless the lesson or any other .py in its folder (events.py, models.py, ...) is newer
  [[ -s "$out/$m.ipynb" && -z $(find "$(dirname "$f")" -name '*.py' -newer "$out/$m.ipynb" -print -quit) ]] &&
    { echo "$m: up to date"; continue; }
  s=$(date +%s)
  flock "$lock" env LEARNJAX_SOLUTIONS=1 LEARNJAX_BOOK=1 MATPLOTLIBRC=/course/tools/book/$theme.matplotlibrc \
    containers/run.sh marimo export ipynb "$f" -o "/course/$out/$m.ipynb" --include-outputs -f >/dev/null 2>"$out/$m.log"
  echo "$m: exit $? in $(( $(date +%s) - s )) s"
done
