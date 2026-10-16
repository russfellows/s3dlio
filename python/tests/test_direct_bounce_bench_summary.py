"""scripts/direct_bounce_bench_summary.py derives arms from the data (a C32 arm is not dropped) and
keeps runs that differ in pairs / busy policy / source apart instead of pooling them."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "direct_bounce_bench_summary.py"
)


def result(arm, pairs=1, busy="wait", source="unaligned", gibps=1.0):
    return {
        "read_pct": 0,
        "size_mib": 64,
        "conc": 4,
        "arm": arm,
        "pairs": pairs,
        "busy": busy,
        "source": source,
        "total_gibps": gibps,
        "put_gibps": gibps,
        "get_gibps": 0.0,
        "cpu_total_s": 1.0,
        "window_secs": 15.0,
        "minflt": 10,
        "peak_rss_mib": 100,
        "pool": {
            "put_fallbacks": 0,
            "get_fallbacks": 0,
            "put_wait_ms": 0,
            "get_wait_ms": 0,
        },
    }


def summarize(tmp_path, rows):
    (tmp_path / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line.startswith("| 4 |")]


def test_unknown_arm_c32_is_not_dropped(tmp_path):
    rows = summarize(
        tmp_path, [result(a) for a in ("A", "C16", "C32", "C64", "aligned")]
    )
    arms = [line.split("|")[2].strip() for line in rows]
    assert arms == ["A", "C16", "C32", "C64", "aligned"], arms


def test_pairs_busy_and_source_are_not_pooled(tmp_path):
    rows = summarize(
        tmp_path,
        [
            result("C32", pairs=1, gibps=1.0),
            result("C32", pairs=2, gibps=2.0),
            result("C32", pairs=1, busy="fallback", gibps=3.0),
            result("C32", pairs=1, source="aligned", gibps=4.0),
            result("C32", pairs=1, gibps=1.0),  # a repetition of the first cell
        ],
    )
    assert len(rows) == 4, rows
    counts = sorted(int(line.split("|")[6].strip()) for line in rows)
    assert counts == [1, 1, 1, 2], (
        f"repetitions must group, configurations must not: {rows}"
    )


def test_script_is_importable_without_running():
    ns = runpy.run_path(str(SCRIPT), run_name="not_main")
    assert ns["arm_order"]("C32") < ns["arm_order"]("C64") < ns["arm_order"]("aligned")
