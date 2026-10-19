#!/usr/bin/env bash
# Zero-copy benchmark matrix for the s3dlio Python API (compares installed wheels).
#
# This script STARTS NOTHING. Start a local S3-compatible server yourself first, listening on
# ENDPOINT and accepting SigV4 requests signed with the key pair below (s3dlio always signs, so a
# server must be configured with credentials; the defaults are benchkey / benchsecret). For a
# throughput measurement, use a server whose per-byte cost is low enough not to be the bottleneck.
# Stop the server and delete its data when done.
#
# Usage:
#   scripts/bench_zero_copy.sh [LABEL::WHEEL_SPEC ...]
#
#   A wheel spec is anything `uv --with` accepts: "s3dlio==0.9.128" or a path to a .whl.
#   The optional LABEL:: names the wheel in the results (default: derived from the spec).
#   Default wheels: PyPI 0.9.128, plus the wheel named by S3DLIO_BENCH_WHEEL when it is set.
#   Example:
#     scripts/bench_zero_copy.sh 0.9.128::s3dlio==0.9.128 f2::/path/to/s3dlio-0.10.0-cp313-....whl
#   With several wheels the order is rotated per cell to avoid ordering bias.
#
# Matrix: workload x payload x size (64K, 1M, 16M, 256M) x concurrency (1, 8, 32; 1 and 4 for
# 256M). count per cell = CELL_GIB GiB / size (so a cell moves ~CELL_GIB GiB per pass),
# clamped to [concurrency, MAX_COUNT]. Each cell does one warm-up and REPEAT timed passes.
# Objects for the GET workloads are written once per size by an untimed `prepare` step using
# the first wheel; PUT workloads write under a per-wheel prefix.
#
# Environment overrides (all optional):
#   ENDPOINT   http://127.0.0.1:9300      AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  benchkey / benchsecret
#   BUCKET     s3dlio-bench               PREFIX  bench          REPEAT  3
#   CELL_GIB   4 (integer)                MAX_COUNT 32768        PYTHON_VERSION 3.13
#   SIZES      "64K 1M 16M 256M"          CONCURRENCY (override for the small sizes) "1 8 32"
#   CONCURRENCY_LARGE "1 4" (256M)        ONLY    egrep on "workload/payload/borrow" cell names
#   OUT_DIR    $BENCH_ROOT/run-<timestamp>  (must stay under BENCH_ROOT, default $TMPDIR/s3dlio-bench)
#   CELL_TIMEOUT 3600 (seconds per invocation)
# Quick smoke, one size:  SIZES=1M CELL_GIB=1 REPEAT=1 ONLY='^get' scripts/bench_zero_copy.sh
#
# Output: $OUT_DIR/results.jsonl (one JSON object per cell and wheel), $OUT_DIR/run.log, and a
# comparison table printed at the end (also $OUT_DIR/summary.txt). Rates are decimal GB/s;
# extraMiB = peak RSS (ru_maxrss) minus RSS before the timed passes.

set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_TOOL="$HERE/bench_zero_copy.py"
BENCH_ROOT="${BENCH_ROOT:-${TMPDIR:-/tmp}/s3dlio-bench}"

ENDPOINT="${ENDPOINT:-http://127.0.0.1:9300}"
BUCKET="${BUCKET:-s3dlio-bench}"
PREFIX="${PREFIX:-bench}"
REPEAT="${REPEAT:-3}"
CELL_GIB="${CELL_GIB:-4}"
MAX_COUNT="${MAX_COUNT:-32768}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
SIZES="${SIZES:-64K 1M 16M 256M}"
CONCURRENCY="${CONCURRENCY:-1 8 32}"
CONCURRENCY_LARGE="${CONCURRENCY_LARGE:-1 4}"
ONLY="${ONLY:-.}"
CELL_TIMEOUT="${CELL_TIMEOUT:-3600}"
OUT_DIR="${OUT_DIR:-$BENCH_ROOT/run-$(date +%Y%m%d-%H%M%S)}"

DEFAULT_WHEELS=("0.9.128::s3dlio==0.9.128")
if [ -n "${S3DLIO_BENCH_WHEEL:-}" ]; then DEFAULT_WHEELS+=("local::${S3DLIO_BENCH_WHEEL}"); fi

case "$OUT_DIR" in
  "$BENCH_ROOT"/*) ;;
  *) echo "OUT_DIR must be under $BENCH_ROOT (got $OUT_DIR)" >&2; exit 2 ;;
esac

export AWS_ENDPOINT_URL="$ENDPOINT"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-benchkey}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-benchsecret}"
export AWS_REGION=us-east-1
export S3DLIO_LOAD_DOTENV=0

# --- wheel list: LABELS[i] / SPECS[i]
LABELS=()
SPECS=()
if [ "$#" -eq 0 ]; then set -- "${DEFAULT_WHEELS[@]}"; fi
for w in "$@"; do
  if [[ "$w" == *::* ]]; then
    LABELS+=("${w%%::*}")
    SPECS+=("${w#*::}")
  else
    SPECS+=("$w")
    base="$(basename "$w")"
    case "$base" in
      *.whl) LABELS+=("$(echo "$base" | sed -e 's/^s3dlio-\([^-]*\)-.*/\1/')") ;;
      *) LABELS+=("$(echo "$base" | sed -e 's/^s3dlio//' -e 's/^[=<>~! ]*//')") ;;
    esac
  fi
done
NW=${#SPECS[@]}

mkdir -p "$OUT_DIR"
RESULTS="$OUT_DIR/results.jsonl"
LOG="$OUT_DIR/run.log"
: >>"$LOG"

# --- run the python tool under a given wheel: bench <wheel-index> <args...>
bench() {
  local i="$1"
  shift
  timeout "$CELL_TIMEOUT" uv run --no-project --isolated --python "$PYTHON_VERSION" \
    --with "${SPECS[$i]}" --with numpy \
    python "$PY_TOOL" --endpoint "$ENDPOINT" --bucket "$BUCKET" --label "${LABELS[$i]}" "$@" 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

count_for() { # size_bytes concurrency -> CELL_GIB GiB / size, clamped to [concurrency, MAX_COUNT]
  local n=$((CELL_GIB * 1073741824 / $1))
  [ "$n" -gt "$MAX_COUNT" ] && n=$MAX_COUNT
  [ "$n" -lt "$2" ] && n=$2
  echo "$n"
}

size_bytes() { # K/M/G suffix -> bytes
  local s="$1" n
  n="${s%[KMGkmg]}"
  case "$s" in
    *[Kk]) echo $((n * 1024)) ;;
    *[Mm]) echo $((n * 1024 * 1024)) ;;
    *[Gg]) echo $((n * 1024 * 1024 * 1024)) ;;
    *) echo "$s" ;;
  esac
}

# --- cells: "workload payload borrow sequential(0/1)"; payload/borrow "-" when not applicable
CELLS=(
  "put_bytes bytes - 1"
  "put_bytes bytearray - 1"
  "put_bytes memoryview - 1"
  "put_bytes numpy - 1"
  "put_bytes_async bytes default 0"
  "put_bytes_async bytearray default 0"
  "put_bytes_async bytearray true 0"
  "put_bytes_async memoryview default 0"
  "put_bytes_async memoryview true 0"
  "put_bytes_async numpy default 0"
  "put_bytes_async numpy true 0"
  "put_many bytes - 0"
  "put_many bytearray - 0"
  "put_many numpy - 0"
  "get - - 1"
  "get_async - - 0"
  "get_many - - 0"
  "get_range - - 0"
  "get_into_async - - 0"
)

echo "== s3dlio zero-copy benchmark: endpoint=$ENDPOINT bucket=$BUCKET out=$OUT_DIR" | tee -a "$LOG"
for ((i = 0; i < NW; i++)); do echo "   wheel[$i] ${LABELS[$i]} = ${SPECS[$i]}" | tee -a "$LOG"; done

# --- bucket
bench 0 --workload create_bucket || { echo "create_bucket failed; is the server up at $ENDPOINT?" >&2; exit 1; }

cell_no=0
for sz in $SIZES; do
  bytes="$(size_bytes "$sz")"
  if [ "$bytes" -ge $((256 * 1024 * 1024)) ]; then conc_list="$CONCURRENCY_LARGE"; else conc_list="$CONCURRENCY"; fi
  maxc=1
  for c in $conc_list; do [ "$c" -gt "$maxc" ] && maxc=$c; done
  nobj="$(count_for "$bytes" "$maxc")"
  pfx="$PREFIX/$sz"

  # untimed: objects for the GET workloads (first wheel; the data is identical for all wheels)
  if printf '%s\n' get/-/- get_async/-/- get_many/-/- get_range/-/- get_into_async/-/- | grep -Eq "$ONLY"; then
    bench 0 --workload prepare --prefix "$pfx" --size "$bytes" --count "$nobj" --concurrency 8 \
      || echo "prepare failed for $sz" | tee -a "$LOG"
  fi

  for cell in "${CELLS[@]}"; do
    read -r wl payload borrow seq <<<"$cell"
    name="$wl/$payload/$borrow"
    echo "$name" | grep -Eq "$ONLY" || continue
    if [ "$seq" = 1 ]; then cl="1"; else cl="$conc_list"; fi
    for c in $cl; do
      cnt="$(count_for "$bytes" "$c")"
      [ "$cnt" -gt "$nobj" ] && cnt="$nobj" # never read more objects than were prepared
      extra=()
      [ "$payload" != "-" ] && extra+=(--payload "$payload")
      [ "$borrow" != "-" ] && extra+=(--borrow "$borrow")
      # alternate (rotate) the wheel order per cell
      for ((k = 0; k < NW; k++)); do
        i=$(((cell_no + k) % NW))
        bench "$i" --workload "$wl" "${extra[@]}" --prefix "$pfx" --put-prefix "$pfx/put-${LABELS[$i]}" \
          --size "$bytes" --count "$cnt" --concurrency "$c" --repeat "$REPEAT" --out "$RESULTS" \
          || echo "cell failed: $name size=$sz c=$c wheel=${LABELS[$i]}" | tee -a "$LOG"
      done
      cell_no=$((cell_no + 1))
    done
  done
done

echo
echo "== summary ($RESULTS)"
if [ -s "$RESULTS" ]; then
  uv run --no-project --isolated --python "$PYTHON_VERSION" python "$PY_TOOL" --summarize "$RESULTS" | tee "$OUT_DIR/summary.txt"
else
  echo "no results were written" >&2
  exit 1
fi
