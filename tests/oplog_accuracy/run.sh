#!/bin/bash
# s3dlio op-log TRACE-ACCURACY harness: one entry point.
#   ./run.sh [--wheel PATH] [--python 3.13] [--out DIR] [--cases a,b,c] [--all-forms] [--no-cleanup]
#            [--delbatch-bin PATH] [--check-only RUN_DIR]
# Required environment: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL (S3-compatible endpoint, designed and
# validated against Google Cloud Storage), OPLOG_HARNESS_BUCKET1, OPLOG_HARNESS_BUCKET2 (two different, pre-existing,
# scratch buckets).  Optional: OPLOG_HARNESS_HOSTFORM_HOST (default: host of AWS_ENDPOINT_URL), AWS_REGION.
# Creates a fresh prefix s3dlio-harness-<date>-<rand>/ in both buckets, sets up unique objects with gcloud, runs every
# case in a fresh bounded process (SIGTERM only, never SIGKILL), snapshots native state, cleans up, verifies cleanup,
# then prints the per-call table (check.py).  Python only via `uv run --no-project --isolated`.
# --wheel defaults to the newest target/wheels/*cp313*.whl of this repository.  Credentials are never written anywhere.
set -u
H="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$H/../.." && pwd)"
WHEEL=""
PYV=3.13
OUT=""
CHECK_ONLY=""
EXTRA=()
DELBATCH=""
while [ $# -gt 0 ]; do
  case "$1" in
    --wheel) WHEEL="$2"; shift 2;;
    --python) PYV="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --cases) EXTRA+=(--cases "$2"); shift 2;;
    --all-forms) EXTRA+=(--all-forms); shift;;
    --no-cleanup) EXTRA+=(--no-cleanup); shift;;
    --delbatch-bin) DELBATCH="$2"; shift 2;;
    --check-only) CHECK_ONLY="$2"; shift 2;;
    -h|--help) sed -n 2,12p "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done

# --check-only re-evaluates saved evidence: no network, no credentials, no wheel needed beyond an interpreter.
if [ -n "$CHECK_ONLY" ]; then
  uv run --no-project --isolated --python "$PYV" python -I "$H/check.py" "$CHECK_ONLY" --known "$H/known.json"; exit $?
fi

# fail fast, before anything is created anywhere
MISSING=()
for v in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL OPLOG_HARNESS_BUCKET1 OPLOG_HARNESS_BUCKET2; do
  [ -n "${!v:-}" ] || MISSING+=("$v")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "required environment variable(s) not set: ${MISSING[*]}  (see README.md); nothing was created" >&2
  exit 2
fi
if [ "$OPLOG_HARNESS_BUCKET1" = "$OPLOG_HARNESS_BUCKET2" ]; then
  echo "OPLOG_HARNESS_BUCKET1 and OPLOG_HARNESS_BUCKET2 must be two different buckets" >&2; exit 2
fi
export AWS_REGION="${AWS_REGION:-us-east-1}"

if [ -z "$WHEEL" ]; then
  WHEEL="$(ls -t "$REPO"/target/wheels/*cp313*.whl 2>/dev/null | sed -n 1p)"
  [ -n "$WHEEL" ] || { echo "no --wheel given and no $REPO/target/wheels/*cp313*.whl found (build one with maturin)" >&2; exit 2; }
fi
[ -f "$WHEEL" ] || { echo "wheel not found: $WHEEL" >&2; exit 2; }
mkdir -p "$H/wheels" "$H/runs"
# keep the exact wheel that was tested next to the results. -f: a same-named stale copy from an earlier run
# (every build of one version has the same file name) must not be tested instead of this wheel.
[ "$WHEEL" -ef "$H/wheels/$(basename "$WHEEL")" ] || cp -f "$WHEEL" "$H/wheels/" || { echo "cannot copy the wheel to $H/wheels/" >&2; exit 2; }
W="$H/wheels/$(basename "$WHEEL")"
[ -f "$W" ] || W="$WHEEL"
UV=(uv run --no-project --isolated --python "$PYV" --with "$W" python -I)

[ -n "$OUT" ] || OUT="$H/runs/$(date +%Y%m%d-%H%M%S)"
S0=$(date +%s)
"${UV[@]}" "$H/orchestrate.py" --wheel "$W" --out "$OUT" --python "$PYV" --delbatch-bin "$DELBATCH" "${EXTRA[@]}"
RC=$?
"${UV[@]}" "$H/check.py" "$OUT" --known "$H/known.json"
CRC=$?
# no process may be left behind (bracket pattern: do not match this shell)
LEFT=$(pgrep -af "[o]plog_accuracy/(cases|orchestrate)" | grep -v "$$" || true)
if [ -n "$LEFT" ]; then echo "LEFTOVER PROCESSES:"; echo "$LEFT"; else echo "processes: none left running"; fi
echo "run dir: $OUT   (report.txt, results.json, run.json, requests.jsonl, cases/*)   wall $(( $(date +%s) - S0 )) s"
exit $(( RC != 0 ? RC : CRC ))
