"""
Multipart write throughput comparison: s3dlio vs s3torchconnector

This is a head-to-head benchmark for the one operation where users choose
between these libraries: streaming write of large objects to S3.

Both libraries are called with comparable settings (same object size, same
part size, same loopback endpoint) so the comparison is fair.

s3torchconnector uses AWS CRT (Mountpoint) underneath, which uses a fully
async non-blocking write path. s3dlio uses a bounded-semaphore backpressure
model (blocking semaphore acquire on the Python caller thread).  The goal is
to confirm s3dlio is competitive.

Run against real MinIO (from s3dlio/.env):
  cd ~/Documents/Code/s3dlio && source .venv/bin/activate
  set -a && source .env && set +a
  AWS_CA_BUNDLE=/home/eval/Documents/Code/mlp-storage/.certs/public.crt \\
  S3DLIO_TEST_BUCKET=perf-s3dlio \\
  python python/tests/bench_multipart_vs_s3torchconnector.py
"""

import os
import sys
import time
import threading

os.environ.setdefault("RUST_LOG", "s3dlio=warn,aws_sdk_s3=warn")

try:
    import s3dlio
except ImportError:
    print("s3dlio not installed — run: uv pip install target/wheels/s3dlio-*.whl")
    sys.exit(1)

try:
    from s3torchconnector import S3ClientConfig
    from s3torchconnector._s3client import S3Client as TorchS3Client
    HAS_TORCH_CONNECTOR = True
except ImportError:
    HAS_TORCH_CONNECTOR = False
    print("s3torchconnector not installed — skipping that leg of the benchmark")

# ── config ──────────────────────────────────────────────────────────────────

BUCKET   = os.environ.get("S3DLIO_TEST_BUCKET") or os.environ.get("S3_BUCKET") or "perf-s3dlio"
ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "")
REGION   = os.environ.get("AWS_REGION", "us-east-1")

PART_SIZE    = 32 << 20    # 32 MiB — large enough to keep network busy
TOTAL_SIZE   = 5120 << 20  # 5 GiB per run (10× original 512 MiB)
MAX_IN_FLIGHT = None        # None = auto-scale (max(32, ceil(512 MiB / part_size)))

# ── utilities ────────────────────────────────────────────────────────────────

def _gbps(total_bytes, elapsed_s):
    return (total_bytes / elapsed_s) / (1 << 30)

def _separator(title=""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("=" * width)

def _check_server():
    if not ENDPOINT:
        print("AWS_ENDPOINT_URL not set — exiting")
        sys.exit(1)
    try:
        s3dlio.put_bytes(f"s3://{BUCKET}/_bench_probe", b"ok")
    except Exception as e:
        print(f"Cannot reach S3 server at {ENDPOINT}: {e}")
        sys.exit(1)

# ── benchmark legs ───────────────────────────────────────────────────────────

def _bench_s3dlio_single(label, part_size, max_in_flight, total_size):
    """One s3dlio MultipartUploadWriter run. Returns GB/s."""
    key = f"bench/s3dlio-single-{int(time.time())}.bin"
    uri = f"s3://{BUCKET}/{key}"
    chunk = bytes([0xAA]) * part_size

    w = s3dlio.MultipartUploadWriter.from_uri(uri, part_size=part_size, max_in_flight=max_in_flight)
    t0 = time.perf_counter()
    written = 0
    while written < total_size:
        w.write(chunk)
        written += len(chunk)
    w.close()
    elapsed = time.perf_counter() - t0

    try:
        s3dlio.delete(uri)
    except Exception:
        pass
    return _gbps(total_size, elapsed), elapsed


def _bench_s3dlio_parallel(n_writers, part_size, max_in_flight, total_size):
    """N concurrent s3dlio writers. Returns aggregate GB/s."""
    chunk = bytes([0xBB]) * part_size
    results = [None] * n_writers
    errors = []

    def worker(idx):
        try:
            key = f"bench/s3dlio-par{idx}-{int(time.time())}.bin"
            uri = f"s3://{BUCKET}/{key}"
            w = s3dlio.MultipartUploadWriter.from_uri(uri, part_size=part_size, max_in_flight=max_in_flight)
            t0 = time.perf_counter()
            written = 0
            while written < total_size:
                w.write(chunk)
                written += len(chunk)
            w.close()
            results[idx] = time.perf_counter() - t0
            try:
                s3dlio.delete(uri)
            except Exception:
                pass
        except Exception as e:
            errors.append(f"worker {idx}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_writers)]
    wall_t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    wall_elapsed = time.perf_counter() - wall_t0

    if errors:
        raise RuntimeError("Parallel workers failed:\n" + "\n".join(errors))

    agg_bytes = n_writers * total_size
    return _gbps(agg_bytes, wall_elapsed), wall_elapsed


def _bench_s3torchconnector_single(part_size, total_size):
    """One s3torchconnector S3Writer run. Returns GB/s."""
    if not HAS_TORCH_CONNECTOR:
        return None, None

    cfg = S3ClientConfig(
        throughput_target_gbps=10.0,
        part_size=part_size,
        force_path_style=True,   # required for s3-ultra / MinIO
    )
    client = TorchS3Client(region=REGION, endpoint=ENDPOINT, s3client_config=cfg)
    key = f"bench/s3torch-single-{int(time.time())}.bin"
    chunk = bytes([0xCC]) * part_size

    writer = client.put_object(BUCKET, key)
    t0 = time.perf_counter()
    written = 0
    while written < total_size:
        writer.write(chunk)
        written += len(chunk)
    writer.close()
    elapsed = time.perf_counter() - t0

    # cleanup
    try:
        client.delete_object(BUCKET, key)
    except Exception:
        pass
    return _gbps(total_size, elapsed), elapsed


def _bench_s3torchconnector_parallel(n_writers, part_size, total_size):
    """N concurrent s3torchconnector writers. Returns aggregate GB/s."""
    if not HAS_TORCH_CONNECTOR:
        return None, None

    cfg = S3ClientConfig(
        throughput_target_gbps=10.0,
        part_size=part_size,
        force_path_style=True,
    )
    chunk = bytes([0xDD]) * part_size
    results = [None] * n_writers
    errors = []

    def worker(idx):
        try:
            # Each thread needs its own client (s3torchconnector is per-pid/per-thread)
            c = TorchS3Client(region=REGION, endpoint=ENDPOINT, s3client_config=cfg)
            key = f"bench/s3torch-par{idx}-{int(time.time())}.bin"
            wtr = c.put_object(BUCKET, key)
            t0 = time.perf_counter()
            written = 0
            while written < total_size:
                wtr.write(chunk)
                written += len(chunk)
            wtr.close()
            results[idx] = time.perf_counter() - t0
            try:
                c.delete_object(BUCKET, key)
            except Exception:
                pass
        except Exception as e:
            errors.append(f"worker {idx}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_writers)]
    wall_t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    wall_elapsed = time.perf_counter() - wall_t0

    if errors:
        raise RuntimeError("Parallel workers failed:\n" + "\n".join(errors))

    agg_bytes = n_writers * total_size
    return _gbps(agg_bytes, wall_elapsed), wall_elapsed


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    _check_server()

    _separator("Setup")
    print(f"  s3dlio            {s3dlio.__version__}")
    if HAS_TORCH_CONNECTOR:
        import s3torchconnector
        print(f"  s3torchconnector  {s3torchconnector.__version__}")
    else:
        print("  s3torchconnector  (not installed)")
    print(f"  bucket            s3://{BUCKET}")
    print(f"  endpoint          {ENDPOINT}")
    print(f"  part_size         {PART_SIZE >> 20} MiB")
    print(f"  object_size       {TOTAL_SIZE >> 20} MiB per writer")
    print(f"  max_in_flight     {MAX_IN_FLIGHT if MAX_IN_FLIGHT is not None else 'auto'}  (s3dlio only)")
    print()

    rows = []  # (label, gbps, elapsed)

    # ── single-writer ─────────────────────────────────────────────────────
    _separator("Single writer")

    gbps, elapsed = _bench_s3dlio_single("s3dlio NP=1", PART_SIZE, MAX_IN_FLIGHT, TOTAL_SIZE)
    print(f"  s3dlio NP=1          {gbps:.3f} GB/s  ({elapsed:.2f}s for {TOTAL_SIZE>>20} MiB)")
    rows.append(("s3dlio NP=1", gbps, elapsed))

    if HAS_TORCH_CONNECTOR:
        gbps, elapsed = _bench_s3torchconnector_single(PART_SIZE, TOTAL_SIZE)
        if gbps is not None:
            print(f"  s3torchconnector NP=1  {gbps:.3f} GB/s  ({elapsed:.2f}s for {TOTAL_SIZE>>20} MiB)")
            rows.append(("s3torchconnector NP=1", gbps, elapsed))
    print()

    # ── parallel writers (NP=4) ───────────────────────────────────────────
    _separator("4 concurrent writers")

    gbps, elapsed = _bench_s3dlio_parallel(4, PART_SIZE, MAX_IN_FLIGHT, TOTAL_SIZE)
    total_mib = 4 * TOTAL_SIZE >> 20
    print(f"  s3dlio NP=4          {gbps:.3f} GB/s  ({elapsed:.2f}s for {total_mib} MiB)")
    rows.append(("s3dlio NP=4", gbps, elapsed))

    if HAS_TORCH_CONNECTOR:
        gbps, elapsed = _bench_s3torchconnector_parallel(4, PART_SIZE, TOTAL_SIZE)
        if gbps is not None:
            print(f"  s3torchconnector NP=4  {gbps:.3f} GB/s  ({elapsed:.2f}s for {total_mib} MiB)")
            rows.append(("s3torchconnector NP=4", gbps, elapsed))
    print()

    # ── summary table ─────────────────────────────────────────────────────
    _separator("Summary")
    print(f"  {'Library':<28}  {'GB/s':>8}  {'Elapsed':>10}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*10}")
    for label, gbps, elapsed in rows:
        print(f"  {label:<28}  {gbps:>8.3f}  {elapsed:>9.2f}s")
    print()

    # ── conclusion ────────────────────────────────────────────────────────
    s3dlio_np1  = next((r for r in rows if r[0] == "s3dlio NP=1"), None)
    torch_np1   = next((r for r in rows if r[0] == "s3torchconnector NP=1"), None)

    if s3dlio_np1 and torch_np1:
        ratio = s3dlio_np1[1] / torch_np1[1]
        if ratio >= 1.0:
            verdict = f"s3dlio is {ratio:.2f}x FASTER than s3torchconnector (NP=1)"
        else:
            verdict = f"s3dlio is {1/ratio:.2f}x SLOWER than s3torchconnector (NP=1)"
        print(f"  {verdict}")

        if s3dlio_np1[1] < 0.20:
            print("  WARNING: s3dlio NP=1 throughput < 0.20 GB/s — possible regression")
        else:
            print("  s3dlio NP=1 throughput ≥ 0.20 GB/s  ✓")
    print()
    _separator()


if __name__ == "__main__":
    main()
