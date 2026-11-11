#!/usr/bin/env bash
# Regenerate every figure, table and quoted number from scratch.
# usage: ./run_all.sh [-P N] [DIR ...]
#   -P N   number of runs in parallel (default: all cores); every run is single-threaded
#   DIR    only these directories (default: all, in the order below)
# Needs the built solver (code_diagrammatics/build/dp5drive, see README.md) and the conda environment.
# For each directory: run.sh (run cards -> output/), postprocess.py (output/ -> data/), then the notebooks
# (data/ -> figure / printed numbers). Finished runs are skipped on a rerun. quoted_numbers reads the data/ of
# fig_sigma, fig_impss, fig_zeno and tab_plaqerr, so it runs last.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P="$(getconf _NPROCESSORS_ONLN)"; DIRS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -P) P="$2"; shift 2;;
    -h|--help) sed -n 2,9p "${BASH_SOURCE[0]}"; exit 0;;
    *) DIRS+=("${1%/}"); shift;;
  esac
done
[ ${#DIRS[@]} -gt 0 ] || DIRS=(fig_site fig_plaq tab_plaqerr fig_zeno fig_impss fig_sigma quoted_numbers)
PY="${PYTHON:-python3}"
[ -x "${DP5DRIVE:-$R/code_diagrammatics/build/dp5drive}" ] || {
  echo "dp5drive not found: build it first (README.md, 'Build the solver')" >&2; exit 2; }
command -v jupyter >/dev/null || { echo "jupyter not found: activate the conda environment (README.md)" >&2; exit 2; }
for d in "${DIRS[@]}"; do
  echo "=== $d: runs ($(date))"
  "$R/$d/run.sh" -P "$P"
  echo "=== $d: postprocess"
  (cd "$R/$d" && "$PY" postprocess.py)
  for nb in "$R/$d"/*.ipynb; do
    echo "=== $d: notebook $(basename "$nb")"
    jupyter nbconvert --execute --to notebook --inplace "$nb"
  done
done
echo "=== done ($(date)). Figures: <dir>/<dir>.pdf; quoted numbers: the tables printed in quoted_numbers/*.ipynb"
