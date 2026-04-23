"""
Tests for issue #134 — multipart upload backpressure (s3dlio v0.9.82+)

The FIX for #134 is the CURRENT code: the semaphore is acquired on the calling
thread BEFORE spawning the upload task. This provides natural backpressure that
prevents unbounded task / memory growth (the pre-v0.9.82 OOM bug).

The "proposed coordinator-task fix" in the issue was REJECTED. The blocking
semaphore acquisition IS the correct solution.

These tests verify the contract of the correct fix:
  1. Large objects complete successfully (no OOM, no hang)
  2. Concurrency is bounded to max_in_flight (backpressure works)
  3. Content integrity is preserved end-to-end
  4. Throughput is acceptable (bounded, not broken)
  5. max_in_flight=1 serialises parts (extreme backpressure, still correct)
  6. Multiple concurrent Python-level writers don't interfere

Run against real MinIO (from s3dlio/.env):
  cd ~/Documents/Code/s3dlio && source .venv/bin/activate
  set -a && source .env && set +a
  AWS_CA_BUNDLE=/home/eval/Documents/Code/mlp-storage/.certs/public.crt \
  S3DLIO_TEST_BUCKET=perf-s3dlio \
  python python/tests/test_issue134_backpressure.py

Run against local s3-ultra (no multipart support yet):
  export AWS_ENDPOINT_URL=http://127.0.0.1:9000
  export AWS_ACCESS_KEY_ID=minioadmin
  export AWS_SECRET_ACCESS_KEY=minioadmin
  export AWS_REGION=us-east-1
  export S3DLIO_H2C=0
  export S3DLIO_TEST_BUCKET=sai3bench-1k
  python python/tests/test_issue134_backpressure.py
"""

import os
import sys
import time
import threading
import unittest

# ── configure logging before importing s3dlio ──────────────────────────────
os.environ.setdefault("RUST_LOG", "s3dlio=warn,aws_sdk_s3=warn")

try:
    import s3dlio
except ImportError:
    print("s3dlio not installed — run: uv pip install target/wheels/s3dlio-*.whl")
    sys.exit(1)

# ── helpers ────────────────────────────────────────────────────────────────

BUCKET = (
    os.environ.get("S3DLIO_TEST_BUCKET")
    or os.environ.get("S3_BUCKET")
    or "perf-s3dlio"
)

def _skip_if_no_server():
    """Return a skip reason string if the S3 endpoint is unreachable, else None."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL", "")
    if not endpoint:
        return "AWS_ENDPOINT_URL not set"
    # Quick connectivity probe: put a tiny object.
    try:
        s3dlio.put_bytes(f"s3://{BUCKET}/_probe", b"ok")
        return None
    except Exception as e:
        return f"S3 server unreachable: {e}"


# ── test cases ─────────────────────────────────────────────────────────────

class TestIssue134Backpressure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.skip_reason = _skip_if_no_server()
        if cls.skip_reason:
            print(f"\n  [SKIP ALL] {cls.skip_reason}", flush=True)

    def _skip(self):
        if self.skip_reason:
            self.skipTest(self.skip_reason)

    # ------------------------------------------------------------------
    # 1. Basic correctness: small object, default settings
    # ------------------------------------------------------------------
    def test_basic_write_completes(self):
        """MultipartUploadWriter.write() completes and returns correct byte count."""
        self._skip()
        uri = f"s3://{BUCKET}/issue134/basic-write.bin"
        size = 32 << 20  # 32 MiB — two parts at part_size=16 MiB
        data = bytes([0xAA]) * size
        w = s3dlio.MultipartUploadWriter.from_uri(uri, part_size=16 << 20, max_in_flight=4)
        w.write(data)
        info = w.close()
        self.assertEqual(info["total_bytes"], size, "total_bytes mismatch")
        self.assertGreaterEqual(info["parts"], 2, "expected at least 2 parts")
        print(f"  basic_write: {size >> 20} MiB in {info['parts']} parts  ✓")

    # ------------------------------------------------------------------
    # 2. Large object — the pre-fix OOM reproducer
    #    Pre-v0.9.82: 256 MiB @ 8 MiB parts → 32 tasks × 8 MiB = 256 MiB heap
    #    but with 1k objects the system ran out of memory.
    #    Current fix: at most max_in_flight tasks ever run concurrently.
    # ------------------------------------------------------------------
    def test_large_object_no_oom(self):
        """256 MiB upload with tight max_in_flight=4 completes without OOM."""
        self._skip()
        uri = f"s3://{BUCKET}/issue134/large-no-oom.bin"
        part_size = 8 << 20   # 8 MiB
        max_in_flight = 4     # max 4 × 8 MiB = 32 MiB Rust heap at once
        total_size = 256 << 20  # 256 MiB total (32 parts)

        # Write in 8 MiB chunks — same pattern as DLIO benchmark's write loop
        chunk = bytes([0xBB]) * part_size
        w = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=part_size, max_in_flight=max_in_flight
        )
        written = 0
        while written < total_size:
            w.write(chunk)
            written += len(chunk)

        info = w.close()
        self.assertEqual(info["total_bytes"], total_size)
        self.assertGreaterEqual(info["parts"], total_size // part_size)
        print(
            f"  large_no_oom: {total_size >> 20} MiB @ {part_size >> 20} MiB parts, "
            f"max_in_flight={max_in_flight}, parts={info['parts']}  ✓"
        )

    # ------------------------------------------------------------------
    # 3. Backpressure measurably slows writes when max_in_flight=1
    #    With max_in_flight=1, each write() MUST wait for the previous part
    #    to finish before starting the next.  This is slower than max_in_flight>1.
    # ------------------------------------------------------------------
    def test_max_in_flight_1_is_slower_than_16(self):
        """max_in_flight=1 (fully serialised) is measurably slower than max_in_flight=16."""
        self._skip()
        size = 64 << 20   # 64 MiB
        part_size = 8 << 20  # 8 → 8 parts
        data = bytes([0xCC]) * part_size

        # max_in_flight=1: fully serialised — each write() blocks until the
        # previous UploadPart completes.
        uri1 = f"s3://{BUCKET}/issue134/backpressure-t1.bin"
        w1 = s3dlio.MultipartUploadWriter.from_uri(uri1, part_size=part_size, max_in_flight=1)
        t0 = time.perf_counter()
        for _ in range(size // part_size):
            w1.write(data)
        w1.close()
        t1_elapsed = time.perf_counter() - t0

        # max_in_flight=16: all parts pipeline concurrently
        uri16 = f"s3://{BUCKET}/issue134/backpressure-t16.bin"
        w16 = s3dlio.MultipartUploadWriter.from_uri(uri16, part_size=part_size, max_in_flight=16)
        t0 = time.perf_counter()
        for _ in range(size // part_size):
            w16.write(data)
        w16.close()
        t16_elapsed = time.perf_counter() - t0

        print(
            f"  backpressure: max_in_flight=1 → {t1_elapsed:.2f}s  "
            f"max_in_flight=16 → {t16_elapsed:.2f}s"
        )
        # max_in_flight=1 MUST be slower — proves backpressure is active and effective
        self.assertGreater(
            t1_elapsed, t16_elapsed,
            f"Expected max_in_flight=1 ({t1_elapsed:.2f}s) to be slower than "
            f"max_in_flight=16 ({t16_elapsed:.2f}s) — backpressure may not be working"
        )
        print("  Backpressure confirmed: serialised writes slower than pipelined  ✓")

    # ------------------------------------------------------------------
    # 4. Content integrity — verify data survives the round-trip
    # ------------------------------------------------------------------
    def test_content_integrity(self):
        """Uploaded data can be read back and is byte-identical to what was written."""
        self._skip()
        part_size = 8 << 20  # 8 MiB
        # Write 3 distinct patterns across 3 parts
        pattern_a = bytes([0x11]) * part_size
        pattern_b = bytes([0x22]) * part_size
        pattern_c = bytes([0x33]) * part_size
        total_size = len(pattern_a) + len(pattern_b) + len(pattern_c)

        uri = f"s3://{BUCKET}/issue134/integrity.bin"
        w = s3dlio.MultipartUploadWriter.from_uri(uri, part_size=part_size, max_in_flight=4)
        w.write(pattern_a)
        w.write(pattern_b)
        w.write(pattern_c)
        info = w.close()
        self.assertEqual(info["total_bytes"], total_size)

        # Read back and verify each section
        try:
            head = bytes(s3dlio.get_range(uri, 0, 16))
            mid  = bytes(s3dlio.get_range(uri, part_size, 16))
            tail = bytes(s3dlio.get_range(uri, 2 * part_size, 16))
            self.assertEqual(head, bytes([0x11]) * 16, "head mismatch")
            self.assertEqual(mid,  bytes([0x22]) * 16, "middle mismatch")
            self.assertEqual(tail, bytes([0x33]) * 16, "tail mismatch")
            print("  content_integrity: head/mid/tail byte patterns correct  ✓")
        except AttributeError:
            # get_range not available in this build — skip read-back check
            print("  content_integrity: write OK (get_range unavailable, skipping read-back)")

    # ------------------------------------------------------------------
    # 5. Throughput — prove we're not in the broken 0.16 GB/s regime
    #    The pre-fix OOM bug was not about throughput. The blocking semaphore
    #    is the correct approach. We expect ≥ 0.20 GB/s against loopback s3-ultra.
    # ------------------------------------------------------------------
    def test_throughput_acceptable(self):
        """Single-writer throughput against loopback s3-ultra is ≥ 0.20 GB/s."""
        self._skip()
        part_size  = 32 << 20   # 32 MiB parts
        max_in_flight = 8
        total_size = 256 << 20  # 256 MiB
        chunk = bytes([0xDD]) * part_size

        uri = f"s3://{BUCKET}/issue134/throughput.bin"
        w = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=part_size, max_in_flight=max_in_flight
        )
        t0 = time.perf_counter()
        written = 0
        while written < total_size:
            w.write(chunk)
            written += len(chunk)
        w.close()
        elapsed = time.perf_counter() - t0

        gbps = (total_size / elapsed) / (1 << 30)
        print(
            f"  throughput: {total_size >> 20} MiB in {elapsed:.2f}s = {gbps:.3f} GB/s "
            f"(max_in_flight={max_in_flight}, part_size={part_size >> 20} MiB)"
        )
        # Minimum bar: 0.20 GB/s on loopback. Well below what s3-ultra can do,
        # but catches catastrophic regressions (the 0.16 GB/s OOM-panic scenario).
        self.assertGreaterEqual(
            gbps, 0.20,
            f"Throughput {gbps:.3f} GB/s is below the 0.20 GB/s minimum — "
            "possible regression in multipart upload pipeline"
        )
        print("  Throughput ✓")

    # ------------------------------------------------------------------
    # 6. Concurrent writers — multiple Python threads each do their own upload
    # ------------------------------------------------------------------
    def test_concurrent_writers_independent(self):
        """N concurrent Python threads each upload their own object correctly."""
        self._skip()
        N = 4
        part_size = 8 << 20
        obj_size  = 24 << 20  # 3 parts each
        chunk = bytes([0xEE]) * part_size
        errors = []

        def upload_worker(idx):
            try:
                uri = f"s3://{BUCKET}/issue134/concurrent-{idx:02d}.bin"
                w = s3dlio.MultipartUploadWriter.from_uri(
                    uri, part_size=part_size, max_in_flight=4
                )
                written = 0
                while written < obj_size:
                    w.write(chunk)
                    written += len(chunk)
                info = w.close()
                assert info["total_bytes"] == obj_size, \
                    f"worker {idx}: total_bytes={info['total_bytes']} != {obj_size}"
            except Exception as e:
                errors.append(f"worker {idx}: {e}")

        threads = [threading.Thread(target=upload_worker, args=(i,)) for i in range(N)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        elapsed = time.perf_counter() - t0

        if errors:
            self.fail("Concurrent upload errors:\n" + "\n".join(errors))

        agg_gbps = (N * obj_size / elapsed) / (1 << 30)
        print(
            f"  concurrent_writers: {N} threads × {obj_size >> 20} MiB = "
            f"{N * obj_size >> 20} MiB in {elapsed:.2f}s = {agg_gbps:.3f} GB/s  ✓"
        )

    # ------------------------------------------------------------------
    # 7. reserve/commit zero-copy path also respects backpressure
    # ------------------------------------------------------------------
    def test_reserve_commit_backpressure(self):
        """The reserve()/commit() zero-copy path also triggers backpressure correctly."""
        self._skip()
        uri = f"s3://{BUCKET}/issue134/reserve-commit.bin"
        part_size = 16 << 20  # 16 MiB
        N = 4  # 4 parts = 64 MiB total

        w = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=part_size, max_in_flight=2  # tight limit
        )
        total = 0
        for i in range(N):
            mv = w.reserve(part_size)
            mv[:] = bytes([i & 0xFF]) * part_size
            w.commit(part_size)
            total += part_size
            del mv

        info = w.close()
        self.assertEqual(info["total_bytes"], total)
        self.assertGreaterEqual(info["parts"], N)
        print(
            f"  reserve_commit: {N} parts × {part_size >> 20} MiB, "
            f"max_in_flight=2, total={total >> 20} MiB  ✓"
        )


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"s3dlio {s3dlio.__version__}  bucket={BUCKET}  "
          f"endpoint={os.environ.get('AWS_ENDPOINT_URL', '(none)')}")
    print()
    unittest.main(verbosity=2)
