#!/usr/bin/env bash
# scripts/direct_bounce_bench.sh - measure the direct:// bounce-buffer strategies on THE RIG.
#
# THE RIG: 4 x 8 GiB memory-backed null_blk devices in an MD RAID-0 (256 KiB chunk), XFS on the
# array, a FRESH mkfs before every single run. Arms run in alternating order (forward on odd
# repetitions, reversed on even ones), N repetitions each, medians reported. A raw fio reference
# (direct=1, time_based, bs 64M and 1G, QD 1-2) is run on the same rig with runtime and io_bytes
# checked. The rig is torn down at the end (also on error).
#
# Needs root (mdadm, mkfs, mount) and a QUIET host (no builds or agents running): the numbers are
# meaningless otherwise. Everything is written under OUT (default /mnt/scratch/direct-bounce-bench-<date>).
#
# Usage:
#   sudo scripts/direct_bounce_bench.sh              # the full matrix (several hours)
#   sudo QUICK=1 scripts/direct_bounce_bench.sh      # one size, conc 1/4, 1 rep: a smoke run
#
# Environment overrides (space-separated lists):
#   WORKLOADS="0 50 70 100"   read percentage of each workload (0 = 100% write, 100 = 100% read)
#   ARMS="A B C4 C16 C64 aligned"
#   CONCS="1 2 4 8"           SIZES_MIB="256 1024"      REPS=3
#   RAMP=5 WINDOW=15          BUSY=wait|fallback        PAIRS=1
#   OUT=<dir>                 MD_NAME=bounce-bench      PREFIX=bounceb
#   TESTBED=~/Code/storage-testbed   FORCE=1 (skip the free-memory check)
#
# Afterwards: scripts/direct_bounce_bench_summary.py <OUT>  (also run automatically at the end).

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: run with sudo (mdadm, mkfs.xfs and mount need root)" >&2
    exit 1
fi
RUN_USER="${SUDO_USER:-root}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTBED="${TESTBED:-/home/${RUN_USER}/Code/storage-testbed}"
MD_NAME="${MD_NAME:-bounce-bench}"
PREFIX="${PREFIX:-bounceb}"
OUT="${OUT:-/mnt/scratch/direct-bounce-bench-$(date +%Y%m%d-%H%M)}"
MNT="${MNT:-/mnt/scratch/direct-bounce-rig}"
RAMP="${RAMP:-5}"
WINDOW="${WINDOW:-15}"
BUSY="${BUSY:-wait}"
PAIRS="${PAIRS:-1}"

if [[ -n "${QUICK:-}" ]]; then
    WORKLOADS="${WORKLOADS:-0 100 50}"; ARMS="${ARMS:-A B C16 C64 aligned}"
    CONCS="${CONCS:-1 4}"; SIZES_MIB="${SIZES_MIB:-256}"; REPS="${REPS:-1}"
else
    WORKLOADS="${WORKLOADS:-0 100 70 50}"; ARMS="${ARMS:-A B C4 C16 C64 aligned}"
    CONCS="${CONCS:-1 2 4 8}"; SIZES_MIB="${SIZES_MIB:-256 1024}"; REPS="${REPS:-3}"
fi

mkdir -p "$OUT"
BIN="$REPO/target/release/examples/direct_bounce_bench"
LOG="$OUT/driver.log"
exec > >(tee -a "$LOG") 2>&1

echo "== direct:// bounce bench $(date -Is)"
echo "   repo      : $REPO ($(git -C "$REPO" rev-parse --short HEAD))"
echo "   out       : $OUT"
echo "   workloads : read% = $WORKLOADS   arms: $ARMS"
echo "   conc      : $CONCS   sizes MiB: $SIZES_MIB   reps: $REPS   ramp/window: $RAMP/$WINDOW s"

# --- preconditions -------------------------------------------------------------------------
avail_gib=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if [[ ${avail_gib} -lt 44 && -z "${FORCE:-}" ]]; then
    echo "ERROR: only ${avail_gib} GiB available; the rig holds 32 GiB in RAM and the 1 GiB x conc 8" >&2
    echo "       cases need more (stop other work, or FORCE=1 to run anyway)" >&2
    exit 1
fi
busy_procs=$(pgrep -c -x 'cargo|rustc|cc1|ld' || true)
if [[ ${busy_procs:-0} -gt 0 && -z "${FORCE:-}" ]]; then
    echo "ERROR: ${busy_procs} build process(es) running: the host is not quiet (FORCE=1 to override)" >&2
    exit 1
fi
for t in mdadm mkfs.xfs fio; do command -v "$t" >/dev/null || { echo "ERROR: $t missing" >&2; exit 1; }; done
[[ -x "$TESTBED/scripts/mdraid/create-and-assemble.sh" ]] || { echo "ERROR: $TESTBED/scripts/mdraid/create-and-assemble.sh not found" >&2; exit 1; }

# --- build the bench as the invoking user ---------------------------------------------------
echo "== building the bench binary"
sudo -u "$RUN_USER" -H bash -c "cd '$REPO' && CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=4 nice -n 10 cargo build --release --example direct_bounce_bench"
[[ -x "$BIN" ]] || { echo "ERROR: $BIN not built" >&2; exit 1; }
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
echo "   bench binary sha256 $BIN_SHA"

# --- the rig ---------------------------------------------------------------------------------
MD_DEV=""
teardown() {
    set +e
    mountpoint -q "$MNT" && umount "$MNT"
    "$TESTBED/scripts/mdraid/teardown.sh" --md-name "$MD_NAME" --prefix "$PREFIX" --count 4 >/dev/null 2>&1
    echo "== rig torn down"
}
trap teardown EXIT

echo "== building THE RIG: 4 x 8192 MiB null_blk, MD RAID-0, chunk 256 KiB"
MD_DEV=$("$TESTBED/scripts/mdraid/create-and-assemble.sh" --level 0 --count 4 --size-mb 8192 --chunk-kb 256 \
    --md-name "$MD_NAME" --prefix "$PREFIX")
echo "   array: $MD_DEV"
mkdir -p "$MNT"

fresh_fs() {
    mountpoint -q "$MNT" && umount "$MNT"
    mkfs.xfs -f -q "$MD_DEV"
    mount "$MD_DEV" "$MNT"
    mkdir -p "$MNT/bench"
    chown -R "$RUN_USER" "$MNT"
    # sanity: the filesystem must support O_DIRECT (not tmpfs) - the bench would silently fall back
    # to buffered I/O otherwise, so check the counters afterwards (puts/fallbacks in the JSON).
}

# --- raw reference: fio on the same rig -----------------------------------------------------
echo "== fio reference (direct=1, time_based, runtime ${WINDOW}s after ${RAMP}s ramp)"
for rw in write read; do
    for bs in 64M 1G; do
        for qd in 1 2; do
            fresh_fs
            name="fio-$rw-$bs-qd$qd"
            fio --name="$name" --filename="$MNT/bench/fio.bin" --rw="$rw" --bs="$bs" --iodepth="$qd" \
                --ioengine=libaio --direct=1 --size=8G --time_based --runtime="$WINDOW" --ramp_time="$RAMP" \
                --numjobs=1 --output-format=json --output="$OUT/$name.json" >/dev/null
            echo "   $name done"
        done
    done
done

# --- the matrix ---------------------------------------------------------------------------------
RESULTS="$OUT/results.jsonl"
: > "$RESULTS"
run_one() { # arm size conc readpct rep
    local arm=$1 size=$2 conc=$3 rp=$4 rep=$5
    fresh_fs
    sync; echo 3 > /proc/sys/vm/drop_caches
    local out
    if out=$(sudo -u "$RUN_USER" -H nice -n 0 "$BIN" --dir "$MNT/bench" --arm "$arm" --conc "$conc" \
            --size-mib "$size" --read-pct "$rp" --ramp-secs "$RAMP" --window-secs "$WINDOW" \
            --busy "$BUSY" --pairs "$PAIRS" --rep "$rep"); then
        echo "$out" >> "$RESULTS"
        echo "   ok   arm=$arm size=$size conc=$conc read%=$rp rep=$rep"
    else
        echo "   FAIL arm=$arm size=$size conc=$conc read%=$rp rep=$rep (no result recorded)" | tee -a "$OUT/failures.txt"
    fi
}

for rp in $WORKLOADS; do
  for size in $SIZES_MIB; do
    for conc in $CONCS; do
      for rep in $(seq 1 "$REPS"); do
        arms=($ARMS)
        if (( rep % 2 == 0 )); then
            rev=(); for ((i=${#arms[@]}-1; i>=0; i--)); do rev+=("${arms[i]}"); done; arms=("${rev[@]}")
        fi
        for arm in "${arms[@]}"; do
            [[ "$arm" == "aligned" && "$rp" == "100" ]] && continue   # no PUT in a pure read mix
            run_one "$arm" "$size" "$conc" "$rp" "$rep"
        done
      done
    done
  done
done

echo "== summary"
sudo -u "$RUN_USER" -H uv run --no-project python "$REPO/scripts/direct_bounce_bench_summary.py" "$OUT" | tee "$OUT/summary.md"
echo "== done: $OUT"
