#!/usr/bin/env python3
"""Zero-copy throughput and memory benchmark for the s3dlio Python API.

Measures one workload per process invocation (so peak RSS belongs to that workload only)
against an S3-compatible server, using ONLY the public s3dlio Python API, so the same
file runs against any installed s3dlio wheel (for example 0.9.128 from PyPI and 0.10.0).
APIs that a wheel lacks are detected with ``hasattr`` / signature checks and the workload is
reported as skipped, with a note, instead of failing.

Run it through ``scripts/bench_zero_copy.sh`` (it uses ``uv run --no-project --isolated`` per
wheel); a direct example for one cell:

    uv run --no-project --isolated --python 3.13 --with s3dlio==0.9.128 --with numpy \\
        python scripts/bench_zero_copy.py --endpoint http://127.0.0.1:9300 --bucket zc \\
        --prefix p/1M --workload get_async --size 1M --count 512 --concurrency 8 \\
        --repeat 3 --out results.jsonl

Workloads (``--workload``):

  create_bucket     create ``--bucket`` (s3dlio.create_bucket, else the s3-cli); untimed
  prepare           write ``--count`` objects of ``--size`` under ``<prefix>/obj/`` (untimed);
                    GET workloads read these
  put_bytes         s3dlio.put_bytes, sequential (``--concurrency`` is ignored: 1)
  put_bytes_async   s3dlio.put_bytes_async, ``--concurrency`` workers under asyncio.gather;
                    ``--borrow true`` passes borrow=True (zero-copy; the shared payload is
                    never mutated), ``--borrow default`` leaves the keyword out
  put_many          s3dlio.put_many(list of (uri, payload), max_in_flight=concurrency)
  get               s3dlio.get, sequential; the BytesView is consumed via len(memoryview)
  get_async         s3dlio.get_async, ``--concurrency`` workers under asyncio.gather
  get_many          s3dlio.get_many(uris, max_in_flight=concurrency)
  get_range         s3dlio.get_range_async, three ranges per object, ``--concurrency`` workers
  get_into_async    s3dlio.get_into_async into one preallocated numpy array per in-flight
                    request (0.10.0 and later; skipped when the API is missing)

``--payload`` (PUT workloads): bytes | bytearray | memoryview | numpy.  One payload object is
shared by every request and is never modified, so concurrent reads are safe even with
borrow=True.  GET results are never copied: only the length of ``memoryview(result)`` is read.

Memory: ``baseline_rss_kib`` is the current RSS after the payload / destination buffers exist
and are touched (so they are not counted as extra); ``peak_rss_kib`` is
``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (KiB on Linux) after all repeats;
``extra_rss_kib = peak - baseline``.  A hidden payload copy shows up as roughly another copy of
the in-flight data (concurrency x size; for put_many / get_many, count x size).

Rates use decimal GB (1e9 bytes).  The JSON line is appended to ``--out``.

``--summarize FILE.jsonl`` prints a per-cell comparison table across labels (wheels) instead of
running a workload.
"""

import argparse
import asyncio
import gc
import itertools
import json
import os
import resource
import statistics
import subprocess
import sys
import time

WORKLOADS = (
    "create_bucket",
    "prepare",
    "put_bytes",
    "put_bytes_async",
    "put_many",
    "get",
    "get_async",
    "get_many",
    "get_range",
    "get_into_async",
)
PAYLOADS = ("bytes", "bytearray", "memoryview", "numpy")
MAX_RANGE_BYTES = 4 << 20
S3_CLI = "/usr/local/bin/s3-cli"


def parse_size(text):
    """Parse a byte count with an optional K/M/G suffix (powers of 1024)."""
    t = str(text).strip().upper().rstrip("B")
    t = t.removesuffix("I")
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}.get(t[-1:] if t else "", 1)
    if t and t[-1] in "KMG":
        t = t[:-1]
    return int(float(t) * mult)


def fmt_size(n):
    for unit, div in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if n >= div and n % div == 0:
            return f"{n // div}{unit}"
    return str(n)


def current_rss_kib():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def peak_rss_kib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def make_payload(kind, size):
    """Build one shared, touched payload of ``size`` random-ish bytes without transient copies.

    Returns (payload, keepalive).  The payload is never mutated afterwards.
    """
    if kind == "bytes":
        return os.urandom(size), None
    ba = bytearray(size)
    chunk = os.urandom(min(size, 1 << 20)) if size else b""
    view = memoryview(ba)
    for off in range(0, size, len(chunk) or 1):
        n = min(len(chunk), size - off)
        view[off : off + n] = chunk[:n]
    view.release()
    if kind == "bytearray":
        return ba, None
    if kind == "memoryview":
        return memoryview(ba), ba
    if kind == "numpy":
        import numpy as np

        return np.frombuffer(ba, dtype=np.uint8), ba
    raise SystemExit(f"unknown payload type {kind!r}")


class Skip(Exception):
    """The workload cannot run against this wheel; the message is the note."""


def uri_of(args, rel):
    return f"s3://{args.bucket}/{rel}"


def obj_uris(args, n):
    return [uri_of(args, f"{args.prefix}/obj/{i:07d}") for i in range(n)]


def put_uris(args, n):
    return [uri_of(args, f"{args.put_prefix}/{i:07d}") for i in range(n)]


def ranges_for(size):
    length = max(1, min(size // 8, MAX_RANGE_BYTES))
    offs = sorted({0, max(0, size // 2 - length // 2), max(0, size - length)})
    return [(o, min(length, size - o)) for o in offs]


async def pool(n, concurrency, one):
    """Run ``one(i)`` for i in range(n) with ``concurrency`` workers under asyncio.gather."""
    counter = itertools.count()

    async def worker():
        total = 0
        while True:
            i = next(counter)
            if i >= n:
                return total
            total += await one(i)

    return sum(await asyncio.gather(*[worker() for _ in range(max(1, concurrency))]))


def check_len(got, want, what):
    if got != want:
        raise RuntimeError(f"{what}: got {got} bytes, expected {want}")
    return got


def build(s3dlio, args, runner):
    """Return ``run(n) -> bytes_moved`` for the workload; may raise Skip."""
    w = args.workload
    size, conc = args.size, max(1, args.concurrency)
    has = lambda name: hasattr(s3dlio, name)

    if w in ("put_bytes", "put_bytes_async", "put_many", "prepare"):
        payload, args._keepalive = make_payload(args.payload, size)
        args._touched = payload

    if w == "prepare":
        use_many = has("put_many")

        def run(n):
            uris = obj_uris(args, n)
            if use_many:
                s3dlio.put_many(
                    [(u, payload) for u in uris], max_in_flight=max(conc, 8)
                )
            else:
                for u in uris:
                    s3dlio.put_bytes(u, payload)
            return n * size

        return run

    if w == "put_bytes":
        args.concurrency = 1

        def run(n):
            for u in put_uris(args, n):
                s3dlio.put_bytes(u, payload)
            return n * size

        return run

    if w == "put_bytes_async":
        if not has("put_bytes_async"):
            raise Skip("s3dlio.put_bytes_async is missing")
        kw = {}
        if args.borrow == "true":
            try:
                sig = str(s3dlio.put_bytes_async.__text_signature__)
            except AttributeError:
                sig = ""
            if "borrow" not in sig:
                raise Skip("put_bytes_async has no borrow= keyword in this wheel")
            kw["borrow"] = True

        def run(n):
            uris = put_uris(args, n)

            async def one(i):
                await s3dlio.put_bytes_async(uris[i], payload, **kw)
                return size

            return runner.run(pool(n, conc, one))

        return run

    if w == "put_many":
        if not has("put_many"):
            raise Skip("s3dlio.put_many is missing")

        def run(n):
            items = [(u, payload) for u in put_uris(args, n)]
            s3dlio.put_many(items, max_in_flight=conc)
            return n * size

        return run

    if w == "get":
        args.concurrency = 1

        def run(n):
            total = 0
            for u in obj_uris(args, n):
                bv = s3dlio.get(u)
                total += check_len(len(memoryview(bv)), size, u)
            return total

        return run

    if w == "get_async":
        if not has("get_async"):
            raise Skip("s3dlio.get_async is missing")

        def run(n):
            uris = obj_uris(args, n)

            async def one(i):
                bv = await s3dlio.get_async(uris[i])
                return check_len(len(memoryview(bv)), size, uris[i])

            return runner.run(pool(n, conc, one))

        return run

    if w == "get_many":
        if not has("get_many"):
            raise Skip("s3dlio.get_many is missing")

        def run(n):
            res = s3dlio.get_many(obj_uris(args, n), max_in_flight=conc)
            total = 0
            for u, bv in res:
                total += check_len(len(memoryview(bv)), size, u)
            del res
            return total

        return run

    if w == "get_range":
        fn = getattr(s3dlio, "get_range_async", None)
        if fn is None:
            raise Skip("s3dlio.get_range_async is missing")
        rngs = ranges_for(size)

        def run(n):
            uris = obj_uris(args, n)

            async def one(i):
                total = 0
                for off, length in rngs:
                    bv = await fn(uris[i], off, length)
                    total += check_len(len(memoryview(bv)), length, uris[i])
                return total

            return runner.run(pool(n, conc, one))

        return run

    if w == "get_into_async":
        if not has("get_into_async"):
            raise Skip("s3dlio.get_into_async is missing (added in 0.10.0)")
        import numpy as np

        bufs = [np.empty(size, dtype=np.uint8) for _ in range(conc)]
        for b in bufs:
            b.fill(0)  # first-touch now so the destinations are part of the baseline
        args._keepalive = bufs
        free = list(range(conc))

        def run(n):
            uris = obj_uris(args, n)
            slots = free[:]

            async def one(i):
                slot = slots.pop()  # one buffer per in-flight request; never shared
                try:
                    got = await s3dlio.get_into_async(
                        uris[i], bufs[slot], expected_len=size
                    )
                finally:
                    slots.append(slot)
                return check_len(got if isinstance(got, int) else size, size, uris[i])

            return runner.run(pool(n, conc, one))

        return run

    raise SystemExit(f"unknown workload {w!r}")


def do_create_bucket(s3dlio, args):
    try:
        if hasattr(s3dlio, "create_bucket"):
            s3dlio.create_bucket(args.bucket)
            return "created via s3dlio.create_bucket"
    except Exception as e:
        msg = str(e)
        if "exist" in msg.lower() or "owned" in msg.lower():
            return f"bucket already exists ({msg[:80]})"
        raise
    cmd = [
        S3_CLI,
        "--endpoint-url",
        args.endpoint,
        "create-bucket",
        f"s3://{args.bucket}",
    ]
    subprocess.run(cmd, check=True)
    return "created via s3-cli"


def base_record(args, version):
    """The identity fields of a result row (the same for ok, skipped and failed cells)."""
    puts = args.workload.startswith("put") or args.workload == "prepare"
    return {
        "version": version,
        "label": args.label or version,
        "workload": args.workload,
        "payload": args.payload if puts else "-",
        "borrow": args.borrow if args.workload == "put_bytes_async" else "-",
        "size": args.size,
        "count": args.count,
        "concurrency": args.concurrency,
    }


def summarize_run(args, version, times, nbytes, nobj, base_kib, note):
    gbps = [nbytes / t / 1e9 for t in times]
    ops = [nobj / t for t in times]
    peak = peak_rss_kib()
    return {
        **base_record(args, version),
        "repeat": len(times),
        "bytes_per_repeat": nbytes,
        "wall_s": times,
        "gbps_median": statistics.median(gbps),
        "gbps_min": min(gbps),
        "gbps_max": max(gbps),
        "objs_median": statistics.median(ops),
        "objs_min": min(ops),
        "objs_max": max(ops),
        "baseline_rss_kib": base_kib,
        "peak_rss_kib": peak,
        "extra_rss_kib": max(0, peak - base_kib),
        "note": note,
    }


def tag_of(args):
    parts = [args.workload]
    if args.workload.startswith("put") or args.workload == "prepare":
        parts.append(args.payload)
    if args.workload == "put_bytes_async" and args.borrow == "true":
        parts.append("borrow")
    return "/".join(parts)


def run_workload(args):
    os.environ["AWS_ENDPOINT_URL"] = args.endpoint
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "benchkey")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "benchsecret")
    os.environ.setdefault("AWS_REGION", "us-east-1")
    os.environ.setdefault("S3DLIO_LOAD_DOTENV", "0")
    import s3dlio

    version = getattr(s3dlio, "__version__", "unknown")
    args._keepalive = None
    args.put_prefix = args.put_prefix or f"{args.prefix}/put"

    if args.workload == "create_bucket":
        print(f"s3dlio {version}: {do_create_bucket(s3dlio, args)}")
        return 0

    record = None
    try:
        with asyncio.Runner() as runner:
            run = build(s3dlio, args, runner)
            if args.workload != "prepare":
                # one-object init pass: starts the runtime, connection pools and store, so the
                # baseline below excludes that constant (a few MiB) from "extra" memory
                run(1)
            gc.collect()
            base_kib = current_rss_kib()
            if args.workload == "prepare":
                t0 = time.perf_counter()
                run(args.count)
                print(
                    f"s3dlio {version} prepare size={fmt_size(args.size)} count={args.count}: "
                    f"{time.perf_counter() - t0:.2f}s (untimed step)"
                )
                return 0
            # warm-up: connections, thread pools, first-touch
            run(min(args.count, max(2 * max(1, args.concurrency), 8)))
            times, moved = [], 0
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                moved = run(args.count)
                times.append(time.perf_counter() - t0)
        record = summarize_run(args, version, times, moved, args.count, base_kib, "")
    except Skip as s:
        record = {**base_record(args, version), "skipped": str(s)}
        print(f"s3dlio {version} {tag_of(args)}: SKIPPED - {s}")
    except Exception as e:  # noqa: BLE001 - a failing cell is recorded, not fatal
        record = {**base_record(args, version), "error": f"{type(e).__name__}: {e}"}
        print(
            f"s3dlio {version} {tag_of(args)}: ERROR - {record['error']}",
            file=sys.stderr,
        )
    else:
        print(
            f"s3dlio {version} [{record['label']}] {tag_of(args)} size={fmt_size(args.size)} "
            f"n={record['count']} c={record['concurrency']}: "
            f"{record['gbps_median']:.3f} GB/s (min {record['gbps_min']:.3f} max {record['gbps_max']:.3f}) "
            f"{record['objs_median']:.0f} obj/s  rss base={record['baseline_rss_kib'] / 1024:.0f} MiB "
            f"peak={record['peak_rss_kib'] / 1024:.0f} MiB extra={record['extra_rss_kib'] / 1024:.0f} MiB "
            f"({record['extra_rss_kib'] * 1024 / max(1, record['concurrency'] * args.size):.2f}x in-flight)"
        )
    if args.out and record is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(record) + "\n")
    return 1 if record is not None and "error" in record else 0


def summarize_file(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    labels = []
    cells = {}
    for r in rows:
        lab = r.get("label") or r.get("version")
        if lab not in labels:
            labels.append(lab)
        key = (
            r["workload"],
            r.get("payload", "-"),
            r.get("borrow", "-"),
            r["size"],
            r["concurrency"],
        )
        cells.setdefault(key, {})[lab] = (
            r  # a later run of the same cell replaces an earlier one
        )

    def order(k):
        w = WORKLOADS.index(k[0]) if k[0] in WORKLOADS else 99
        return (k[3], w, k[1], k[2], k[4])

    head = f"{'workload':<16}{'payload':<11}{'borrow':<8}{'size':>6}{'c':>4}"
    for lab in labels:
        head += f" | {lab[:14] + ' GB/s':>19}{'extraMiB':>9}"
    if len(labels) > 1:
        head += f" | {labels[-1][:8] + '/' + labels[0][:8]:>17}{'extra x':>9}"
    print(head)
    print("-" * len(head))
    for key in sorted(cells, key=order):
        wl, pay, bor, size, conc = key
        line = f"{wl:<16}{pay:<11}{bor:<8}{fmt_size(size):>6}{conc:>4}"
        for lab in labels:
            r = cells[key].get(lab)
            if r is None:
                line += f" | {'-':>19}{'-':>9}"
            elif "skipped" in r:
                line += f" | {'skipped':>19}{'':>9}"
            elif "error" in r:
                line += f" | {'ERROR':>19}{'':>9}"
            else:
                line += f" | {r['gbps_median']:>19.3f}{r['extra_rss_kib'] / 1024:>9.0f}"
        if len(labels) > 1:
            a, b = cells[key].get(labels[0]), cells[key].get(labels[-1])
            ok = lambda r: r is not None and "gbps_median" in r
            if ok(a) and ok(b):
                ratio = (
                    b["gbps_median"] / a["gbps_median"]
                    if a["gbps_median"]
                    else float("nan")
                )
                ex = (b["extra_rss_kib"] + 1024) / (
                    a["extra_rss_kib"] + 1024
                )  # +1 MiB: damp noise
                line += f" | {ratio:>17.2f}{ex:>9.2f}"
        print(line)
    return 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--endpoint",
        default=os.environ.get("AWS_ENDPOINT_URL", "http://127.0.0.1:9300"),
    )
    p.add_argument("--bucket", default="s3dlio-bench")
    p.add_argument(
        "--prefix", default="bench", help="GET objects live under <prefix>/obj/"
    )
    p.add_argument(
        "--put-prefix", default=None, help="PUT destination (default <prefix>/put)"
    )
    p.add_argument("--workload", choices=WORKLOADS)
    p.add_argument("--payload", choices=PAYLOADS, default="bytes")
    p.add_argument("--borrow", choices=("default", "true"), default="default")
    p.add_argument(
        "--size", type=parse_size, default=parse_size("1M"), help="bytes, K/M/G suffix"
    )
    p.add_argument("--count", type=int, default=64, help="objects per repeat")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument(
        "--label", default=None, help="name of this wheel in results (default: version)"
    )
    p.add_argument("--out", default=None, help="JSON lines file to append to")
    p.add_argument(
        "--summarize", metavar="JSONL", help="print a comparison table and exit"
    )
    args = p.parse_args()
    if args.summarize:
        return summarize_file(args.summarize)
    if not args.workload:
        p.error("--workload is required (or use --summarize)")
    if args.count < 1 or args.repeat < 1:
        p.error("--count and --repeat must be >= 1")
    return run_workload(args)


if __name__ == "__main__":
    sys.exit(main())
