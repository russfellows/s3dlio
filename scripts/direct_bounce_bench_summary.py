"""Summarise a scripts/direct_bounce_bench.sh run: medians over the repetitions per cell, the fio
reference, and sanity checks (runtime / io_bytes verified, no buffered fallbacks).

Usage: uv run --no-project python scripts/direct_bounce_bench_summary.py <OUT dir>
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def fio_rows(out):
    rows = []
    for p in sorted(out.glob("fio-*.json")):
        d = json.loads(p.read_text())
        job = d["jobs"][0]
        side = job["write"] if "-write-" in p.name else job["read"]
        runtime_ms, io_bytes = side["runtime"], side["io_bytes"]
        asked_ms = float(job.get("job options", {}).get("runtime", 0)) * 1000
        ok = io_bytes > 0 and runtime_ms >= 0.9 * asked_ms
        rows.append(
            (
                p.stem,
                io_bytes / 2**30 / (runtime_ms / 1000) if runtime_ms else 0.0,
                runtime_ms / 1000,
                io_bytes / 2**30,
                "ok" if ok else "INVALID (runtime/io_bytes)",
            )
        )
    return rows


# Result fields that distinguish one configuration from another (they must never be pooled):
# buffer pairs per node, the busy policy and the source (aligned / unaligned). A field a result
# does not carry reads as "-" so older result files still group.
CONFIG_FIELDS = ("pairs", "busy", "source")


def config_of(r):
    return tuple(str(r.get(f, r.get("pool", {}).get(f, "-"))) for f in CONFIG_FIELDS)


def arm_order(arm):
    """Sort key for an arm name: A, B, then C<size> numerically, then anything else (aligned...)
    alphabetically. Arms come from the data, so a C32 (or any other C<size>) is never dropped."""
    if arm in ("A", "B"):
        return (0, arm, 0)
    if arm.startswith("C") and arm[1:].isdigit():
        return (1, "C", int(arm[1:]))
    return (2, arm, 0)


def med(xs):
    return statistics.median(xs) if xs else 0.0


def main():
    out = Path(sys.argv[1])
    cells = defaultdict(list)
    bad = []
    for line in (out / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["read_pct"], r["size_mib"], r["conc"], r["arm"], *config_of(r))
        cells[key].append(r)
        if r["pool"]["put_fallbacks"] or r["pool"]["get_fallbacks"]:
            bad.append(("pool fallbacks (busy policy)", key))
        # Rows from before the implicit counters existed carry none of these keys.
        implicit = [
            f"{name}={r['pool'][name]}"
            for name in (
                "implicit_put",
                "implicit_get",
                "implicit_bytes",
                "pool_unavailable",
                "alignment_mismatches",
            )
            if r["pool"].get(name)
        ]
        if implicit:
            bad.append(
                (
                    "IMPLICIT private bounce allocations ("
                    + ", ".join(implicit)
                    + "): "
                    "the pool did not serve this row, it is NOT a pooled measurement",
                    key,
                )
            )
    print("# direct:// bounce bench summary\n")
    print(f"source: `{out}`  (medians over repetitions; N shown)\n")
    print("## fio reference (raw device path, direct=1)\n")
    print("| job | GiB/s | runtime s | io GiB | check |")
    print("|---|---|---|---|---|")
    for name, gibps, rt, io, chk in fio_rows(out):
        print(f"| {name} | {gibps:.2f} | {rt:.1f} | {io:.1f} | {chk} |")
    for rp in sorted({k[0] for k in cells}):
        label = {0: "100% write", 100: "100% read"}.get(
            rp, f"{rp}% read / {100 - rp}% write"
        )
        for size in sorted({k[1] for k in cells if k[0] == rp}):
            print(f"\n## {label}, {size} MiB objects\n")
            print(
                "| conc | arm | pairs | busy | source | n | put GiB/s | get GiB/s | total GiB/s | "
                "CPU s | CPU s/GiB | minor faults | peak RSS MiB | pool wait ms (put/get) | "
                "window s |"
            )
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            in_cell = [k for k in cells if k[0] == rp and k[1] == size]
            for conc in sorted({k[2] for k in in_cell}):
                # arms and configurations come from the data, one row per distinct key
                for key in sorted(
                    (k for k in in_cell if k[2] == conc),
                    key=lambda k: (arm_order(k[3]), k[4:]),
                ):
                    rs = cells[key]
                    arm, (pairs, busy, source) = key[3], key[4:]
                    tot = med([r["total_gibps"] for r in rs])
                    cpu = med([r["cpu_total_s"] for r in rs])
                    moved = med([r["total_gibps"] * r["window_secs"] for r in rs])
                    print(
                        f"| {conc} | {arm} | {pairs} | {busy} | {source} | {len(rs)} | "
                        f"{med([r['put_gibps'] for r in rs]):.2f} | "
                        f"{med([r['get_gibps'] for r in rs]):.2f} | {tot:.2f} | {cpu:.1f} | "
                        f"{(cpu / moved) if moved else 0:.2f} | {int(med([r['minflt'] for r in rs]))} | "
                        f"{int(med([r['peak_rss_mib'] for r in rs]))} | "
                        f"{int(med([r['pool']['put_wait_ms'] for r in rs]))}/"
                        f"{int(med([r['pool']['get_wait_ms'] for r in rs]))} | "
                        f"{med([r['window_secs'] for r in rs]):.1f} |"
                    )
    if bad:
        print("\n## WARNINGS\n")
        for what, key in bad:
            print(f"- {what}: {key}")
    fails = out / "failures.txt"
    if fails.exists():
        print("\n## FAILED RUNS\n")
        print(fails.read_text())


if __name__ == "__main__":
    main()
