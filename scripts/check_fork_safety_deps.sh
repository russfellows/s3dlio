#!/usr/bin/env bash
# CI guard for fork safety (#297): tokio must NOT be built with its `parking_lot` feature, and the
# only crate allowed to bring parking_lot_core (whose process-global table of parked threads is
# corrupted by fork) into a native build is dashmap.
#
# Why: with tokio's `parking_lot` feature every idle runtime worker parks in parking_lot's global
# table, which keeps a pointer into each parked thread's stack. A fork child inherits that table
# without the threads; its own parking_lot use then follows dangling pointers (random SIGABRT,
# measured at 10-37% of children). A dependency can switch the feature back on through feature
# unification, which is why this is checked on the RESOLVED graph, for every feature set we ship.
#
# Usage: scripts/check_fork_safety_deps.sh   (needs cargo; run from anywhere in the repo)
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0
# name|cargo args
sets=(
  "default|"
  "python wheel|--features extension-module,backend-azure"
  "full-backends|--features full-backends,extension-module"
  "arrow-backend|--no-default-features --features arrow-backend"
)
for entry in "${sets[@]}"; do
  name="${entry%%|*}"; args="${entry#*|}"
  # shellcheck disable=SC2086
  feats="$(cargo tree -e features -i tokio $args 2>&1)" || { echo "cargo tree failed for [$name]:"; echo "$feats"; exit 2; }
  if grep -E 'tokio feature "(parking_lot|full)"' <<<"$feats" >/dev/null; then
    echo "FAIL [$name]: tokio is built with a feature that enables parking_lot (first matches):"
    grep -E 'tokio feature "(parking_lot|full)"' <<<"$feats" | sed -n '1,3p'
    fail=1
  else
    echo "ok   [$name]: tokio has no parking_lot feature"
  fi
  case "$name" in arrow-backend) continue ;; esac   # object_store (arrow backend) itself uses parking_lot
  # shellcheck disable=SC2086
  users="$(cargo tree -e normal -i parking_lot_core --depth 1 $args 2>&1 | sed -n '2,$p' | sed -E 's/^[^a-zA-Z]*//; s/ v.*//' | sort -u | tr '\n' ' ')"
  if [ "${users// /}" != "dashmap" ]; then
    echo "FAIL [$name]: parking_lot_core is used by: ${users:-<error>} (allowed: dashmap only)"
    fail=1
  else
    echo "ok   [$name]: parking_lot_core is used by dashmap only"
  fi
done
exit $fail
