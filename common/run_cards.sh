#!/usr/bin/env bash
# Run the cards of one figure/table directory: output/<run_id>/ for every row of runcards/index.csv.
# usage: common/run_cards.sh DIR [-P N] [--driver-only|--exact-only] [--dry-run] [--force] [RUN_ID_GLOB ...]
#   DP5DRIVE (default <release>/code_diagrammatics/build/dp5drive), PYTHON (default python3), NPROC (default all cores)
#   RUN_TIMING=1: wrap every run in /usr/bin/time -v (-> output/<id>/time.txt: wall, peak RSS)
# A run is finished iff output/<id>/exit_code is 0, 5 or 7 (driver) / 0 (exact); finished runs are skipped unless
# --force. Longest runs (cost_s) start first. Every run is single-threaded.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$(cd "$1" && pwd)"; shift
P="${NPROC:-$(getconf _NPROCESSORS_ONLN)}"; ONLY=""; DRY=0; FORCE=0; GLOBS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -P) P="$2"; shift 2;;
    --driver-only) ONLY=driver; shift;;
    --exact-only) ONLY=exact; shift;;
    --dry-run) DRY=1; shift;;
    --force) FORCE=1; shift;;
    -h|--help) sed -n 2,6p "${BASH_SOURCE[0]}"; exit 0;;
    *) GLOBS+=("$1"); shift;;
  esac
done
export DP5DRIVE="${DP5DRIVE:-$R/code_diagrammatics/build/dp5drive}"
export PYTHON="${PYTHON:-python3}"
export RUN_EXACT="$R/code_exact/run_exact.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DIR FORCE
IDX="$DIR/runcards/index.csv"
[ -f "$IDX" ] || { echo "no $IDX" >&2; exit 2; }

# select rows: kind filter, globs; sort by cost_s descending
LIST="$(tail -n +2 "$IDX" | awk -F, -v only="$ONLY" '(only=="" || $2==only){print $8","$1","$2}' | sort -t, -k1,1nr)"
SEL=()
while IFS=, read -r cost id kind; do
  [ -n "$id" ] || continue
  if [ ${#GLOBS[@]} -gt 0 ]; then
    ok=0; for g in "${GLOBS[@]}"; do [[ "$id" == $g ]] && ok=1; done; [ $ok = 1 ] || continue
  fi
  SEL+=("$id,$kind")
done <<< "$LIST"
n=${#SEL[@]}
need_drv=0; need_ex=0
for s in "${SEL[@]}"; do case "${s#*,}" in driver) need_drv=1;; exact) need_ex=1;; esac; done
if [ $need_drv = 1 ] && [ ! -x "$DP5DRIVE" ]; then echo "dp5drive not found: $DP5DRIVE (build code_diagrammatics or set DP5DRIVE)" >&2; exit 2; fi
if [ $need_ex = 1 ] && ! command -v "$PYTHON" >/dev/null; then echo "python not found: $PYTHON" >&2; exit 2; fi
echo "run_cards: $n runs in $DIR, -P $P"
[ $DRY = 1 ] && { printf '%s\n' "${SEL[@]}"; exit 0; }

one() {
  local k="$1" n="$2" id="${3%,*}" kind="${3#*,}"
  local o="$DIR/output/$id" rc="$DIR/runcards/$id"
  if [ "$FORCE" = 0 ] && [ -f "$o/exit_code" ]; then
    local c; c="$(cat "$o/exit_code")"
    if { [ "$kind" = driver ] && [[ "$c" =~ ^(0|5|7)$ ]]; } || { [ "$kind" = exact ] && [ "$c" = 0 ]; }; then
      echo "[$k/$n] $id exit $c (finished, skipped)"; return 0
    fi
  fi
  rm -rf "$o"; mkdir -p "$o"
  local t0 c=0; t0=$(date +%s%N)
  local TW=()
  if [ "${RUN_TIMING:-0}" = 1 ]; then TW=(/usr/bin/time -v -o "$o/time.txt"); fi
  if [ "$kind" = driver ]; then
    "${TW[@]}" "$DP5DRIVE" --runcard "$rc/rc.h5" --output "$o/out.h5" > "$o/run.log" 2>&1 || c=$?
  else
    "${TW[@]}" "$PYTHON" -I "$RUN_EXACT" "$rc/card.json" "$o/" > "$o/run.log" 2>&1 || c=$?
  fi
  echo "$c" > "$o/exit_code"
  local ms=$(( ($(date +%s%N) - t0) / 1000000 ))
  printf '[%d/%d] %s exit %d %d.%03ds\n' "$k" "$n" "$id" "$c" $((ms / 1000)) $((ms % 1000))
}
export -f one
k=0
for s in "${SEL[@]}"; do k=$((k+1)); echo "$k $n $s"; done | xargs -P "$P" -n 3 bash -c 'one "$@"' _ || true
# summary
fin=0; bad=0
for s in "${SEL[@]}"; do
  id="${s%,*}"; kind="${s#*,}"; c="$(cat "$DIR/output/$id/exit_code" 2>/dev/null || echo none)"
  if { [ "$kind" = driver ] && [[ "$c" =~ ^(0|5|7)$ ]]; } || { [ "$kind" = exact ] && [ "$c" = 0 ]; }; then fin=$((fin+1)); else bad=$((bad+1)); echo "FAILED $id exit $c"; fi
done
echo "run_cards: $fin finished, $bad failed"
[ $bad = 0 ]
